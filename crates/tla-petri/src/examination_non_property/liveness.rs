// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::super::examination_plan::ExecutionPlan;
use super::common::checkpoint_cannot_compute;
use crate::examinations::global_properties_bmc;
use crate::examinations::global_properties_pdr;
use crate::examinations::liveness::check_liveness;
use crate::examinations::mu_calculus::{ctl_to_mu, LocalMuSolver};
use crate::examinations::quasi_liveness::QuasiLivenessObserver;
use crate::explorer::ExplorationConfig;
use crate::output::Verdict;
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::reduction::ReducedNet;
use crate::resolved_predicate::ResolvedPredicate;
use crate::scc::{bottom_sccs, tarjan_scc};
use crate::stubborn::PorStrategy;
use std::time::{Duration, Instant};
use tla_mc_core::CtlFormula as GenericCtlFormula;

/// Fraction of the remaining deadline reserved for the exact SCC graph-build
/// fallback. The deep mu-calculus phase keeps the rest — the bulk.
///
/// Replaces the old flat split (`LIVENESS_MU_PHASE_CAP` = 5s mu cap +
/// `LIVENESS_SCC_FALLBACK_RESERVE` = 10s SCC reserve), which starved the deep
/// engine on long deadlines: on a 60s budget the old split gave the
/// mu-calculus only 5s and reserved 10s for an SCC reachability-graph build
/// that, on the large nets that fall through to it, cannot plausibly complete
/// anyway. Scaling both budgets to the deadline lets the symbolic mu engine —
/// which decides liveness WITHOUT materialising the full reachability graph —
/// take the majority of a generous budget while still leaving the exact
/// fallback a proportional slice.
///
/// Budget-only / verdict-preserving: changing the time split never changes a
/// verdict. A mu True/False is sound at any budget; the SCC fallback is exact
/// only when its graph build completes, and it still runs afterwards with the
/// real wall-clock remaining (the reserve is merely a floor for the case where
/// the mu phase consumes its whole slice without deciding). In practice the
/// deep mu solver also aborts early on its memory-budgeted node cap for the
/// large nets where SCC cannot finish, returning the unused time to SCC, so the
/// split only bites on medium nets — where keeping ~30% for SCC (below) avoids
/// starving an exact graph build that would otherwise have completed.
const LIVENESS_SCC_FALLBACK_FRACTION: f64 = 0.3;

/// Floor on the SCC reserve so even a modest deadline leaves the exact fallback
/// a usable slice when the mu phase consumes its whole budget without deciding.
const LIVENESS_SCC_FALLBACK_RESERVE_FLOOR: Duration = Duration::from_secs(5);

/// Wall-clock cap for the native-PDR reachable-deadlock FALSE shortcut (LV-3).
///
/// Small so a slow PDR call cannot starve the deep mu-calculus engine (LV-2);
/// an inconclusive/expired PDR returns `None` and falls through, so the bound
/// is strictly verdict-preserving.
const LIVENESS_DEADLOCK_PDR_PHASE_CAP: Duration = Duration::from_secs(3);

const LIVENESS_MU_MIN_BUDGET: Duration = Duration::from_millis(250);

/// Fraction of the mu-phase budget reserved for the Phase-A cheap FALSE scan
/// (see [`liveness_via_mu_calculus`]). Kept small so the Phase-B deep proof
/// keeps the bulk of the budget: a deciding group that needs nearly the whole
/// phase to resolve is not starved by the scan.
const LIVENESS_MU_FALSE_SCAN_FRACTION: f64 = 0.2;

/// Wall-clock cap for the strong per-transition dead-transition LP sweep used
/// by [`quasi_liveness_verdict_with_groups`].
///
/// Each per-transition LP (with up to 100 trap re-solves) does not poll a
/// deadline internally, so the sweep polls only *between* transitions. On a
/// transition-heavy net the sweep can therefore run long before the next poll;
/// capping it at this slice reserves the remaining global budget for the
/// exhaustive BFS fallback, mirroring the deadlock siphon-LP soft cap
/// (`DEADLOCK_SIPHON_SOFT_CAP`). Abandoning the sweep yields no verdict and
/// falls through to BMC/BFS, so the bound is strictly verdict-preserving.
const QUASI_LIVENESS_LP_PHASE_CAP: Duration = Duration::from_secs(5);

/// Wall-clock cap for the structural net-class liveness certificate chain
/// (`structural_live`). The state-machine / marked-graph certificates are
/// O(arcs); the cap bounds the free-choice Commoner minimal-siphon
/// enumeration, which is worst-case exponential. An expired deadline makes
/// the certificate decline (`None`) and fall through to the exact engines,
/// so the bound is strictly verdict-preserving.
const LIVENESS_STRUCTURAL_PHASE_CAP: Duration = Duration::from_secs(5);

/// Soft deadline for one `structural_live` certificate attempt: the sooner of
/// [`LIVENESS_STRUCTURAL_PHASE_CAP`] from now and the global deadline.
/// Mirrors [`quasi_liveness_lp_soft_deadline`].
fn structural_live_soft_deadline(global: Option<Instant>) -> Option<Instant> {
    let now = Instant::now();
    Some(match global {
        // B4 scheduling hygiene: reserve `LIVENESS_SCC_FALLBACK_RESERVE_FLOOR`
        // for the SCC/BMC fallback so a structural certificate attempt cannot
        // spin its full flat cap into the fallback's budget near the deadline.
        // No-op at the contest budget (`remaining` ≫ cap + reserve). Soundness-
        // neutral: `structural_live` only PROVES liveness and otherwise falls
        // through, so this changes WHEN it yields, never the verdict.
        Some(global) => {
            let remaining = global.saturating_duration_since(now);
            now + LIVENESS_STRUCTURAL_PHASE_CAP
                .min(remaining.saturating_sub(LIVENESS_SCC_FALLBACK_RESERVE_FLOOR))
        }
        None => now + LIVENESS_STRUCTURAL_PHASE_CAP,
    })
}

/// Triage/benchmark kill-switch for the QuasiLiveness random-walk witness lane.
///
/// `true` iff `TY_MCC_DISABLE_QUASILIVENESS_WALK` is set to `1`/`on`/`true`. The
/// walk is a strict under-approximation that only ever EXPANDS the exhaustive-BFS
/// observer seed with directly-observed reachable enablements (sound TRUE
/// witnesses). The final verdict still comes from the identical exhaustive BFS
/// over the (now larger) seed, so disabling the walk is always verdict-preserving
/// — it merely removes a pre-pass that could let BFS cover fewer transitions.
/// Mirrors `one_safe_walk_disabled()` and the other `TY_MCC_DISABLE_*` switches.
fn quasi_liveness_walk_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_QUASILIVENESS_WALK")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
}

pub(crate) fn quasi_liveness_verdict(net: &PetriNet, config: &ExplorationConfig) -> Verdict {
    // P/T entry point: no colored grouping. Every transition is its own
    // group (singleton), so the colored-question and the P/T-question
    // coincide. Existing call sites (tests, P/T dispatch) keep working
    // unchanged.
    quasi_liveness_verdict_with_groups(net, config, &[])
}

/// Quasi-liveness verdict with optional **colored** transition groups.
///
/// For P/T inputs, pass `&[]` — the function behaves exactly like the
/// classic `quasi_liveness_verdict(net, config)` (every transition is
/// its own group).
///
/// For colored inputs, pass `aliases.colored_transition_groups_as_usize()`.
/// The MCC question on colored nets is *colored* quasi-liveness: every
/// **colored** transition must have at least one binding that can fire,
/// NOT every unfolded P/T transition. Without grouping, a colored
/// transition with a guard like `[i ineq k]` (Peterson-COL,
/// Lamport-COL) flips to F because diagonal bindings (i=k) are
/// structurally dead in the unfolding even though the colored
/// transition is quasi-live via off-diagonal bindings. Twelve broad-
/// measurement wrong answers (b89f4fd2) trace to this conflation.
pub(crate) fn quasi_liveness_verdict_with_groups(
    net: &PetriNet,
    config: &ExplorationConfig,
    colored_transition_groups: &[Vec<usize>],
) -> Verdict {
    let has_colored_groups = !colored_transition_groups.is_empty();

    // --- Pre-reduction structural analysis on the ORIGINAL net ---
    // L4-liveness implies quasi-liveness (L1). Run on the original net before
    // reduction because agglomeration can destroy the free-choice property.
    // Only the TRUE direction is sound: structural non-liveness does NOT
    // imply non-quasi-liveness (a net can be quasi-live without being L4-live).
    if let Some(true) =
        crate::structural::structural_live(net, structural_live_soft_deadline(config.deadline()))
    {
        eprintln!("QuasiLiveness: structurally live (original net, exact net-class certificate)");
        return Verdict::True;
    }

    // LP dead-transition on the original net.
    //
    // - P/T input: any LP-dead transition → net is not quasi-live → F.
    // - Colored input: skip — guarded colored transitions can have
    //   structurally-dead bindings while the colored transition itself
    //   is quasi-live via off-diagonal bindings (the LP cannot
    //   distinguish "binding is dead" from "colored transition is
    //   dead"). Defer to BFS, which CAN ask the colored question.
    if !has_colored_groups {
        if let Some(false) = crate::structural::lp_dead_transition(
            net,
            quasi_liveness_lp_soft_deadline(config.deadline()),
        ) {
            eprintln!("QuasiLiveness: LP-proved dead transition on original net");
            return Verdict::False;
        }
    }

    // --- Identity net (no reduction) ---
    // Structural reductions are currently unsound for QuasiLiveness: on nets
    // like FMS-PT-*, agglomeration/source-place elimination suppresses real
    // firing behavior, flipping the verdict from TRUE to FALSE. Use the
    // identity net until a sound reduction contract is validated.
    let reduced = ReducedNet::identity(net);
    let config = config.refitted_for_net(&reduced.net);

    // --- Post-reduction structural analysis on the REDUCED net ---
    // Structural liveness is stronger than quasi-liveness. On classified
    // state-machine / marked-graph / free-choice nets, the exact certificate
    // can therefore short-circuit the examination before exploration.
    if let Some(true) = crate::structural::structural_live(
        &reduced.net,
        structural_live_soft_deadline(config.deadline()),
    ) {
        eprintln!("QuasiLiveness: structurally live (exact net-class certificate)");
        return Verdict::True;
    }

    // LP dead-transition check on the reduced (or original, when colored)
    // net. Same reasoning as above: skip for colored input.
    if !has_colored_groups {
        if let Some(false) = crate::structural::lp_dead_transition(
            &reduced.net,
            quasi_liveness_lp_soft_deadline(config.deadline()),
        ) {
            eprintln!("QuasiLiveness: LP-proved dead transition (upper bound insufficient)");
            return Verdict::False;
        }
    }

    // --- Strong per-transition dead-transition LP (state equation + traps) ---
    //
    // The weak `lp_dead_transition` above maximises each input place in
    // ISOLATION over the bare state equation (no trap tightening), so it only
    // sees a transition as dead when *some single* input place can never reach
    // its arc weight. It is blind to the JOINT infeasibility "all input places
    // simultaneously >= their weights" — the case where each place can reach
    // its weight individually but never together.
    //
    // `lp_first_dead_transition` feeds the joint enabling conjunction
    // `AND_p weight(p,t) <= M[p]` to the state-equation + initially-marked-trap
    // polytope. That polytope is a SUPERSET of the reachable markings, so
    // LP-infeasibility of "t enabled" proves t is never enabled in ANY
    // reachable marking => t can never fire => the net is NOT quasi-live. We
    // emit FALSE on the first such transition.
    //
    // SOUNDNESS: only the LP-INFEASIBLE direction is consumed (a sound FALSE).
    // An inconclusive transition (LP feasible, size guard, or deadline) makes
    // the sweep return `None`, which is NEVER read as "quasi-live" — we fall
    // through to BMC/BFS unchanged, exactly like the `None` case of the weak
    // shortcut. A witnessed enabling marking (BMC/BFS) is the only way a
    // transition is marked quasi-live, and the net is TRUE only when ALL
    // transitions are witnessed (`all_groups_covered`).
    //
    // COLORED GATING: skipped for colored input for the same reason the weak
    // LP shortcut is — a single unfolded binding can be structurally dead
    // while its colored transition is quasi-live via off-diagonal bindings.
    //
    // The reduced net here is the identity net, so the sweep runs once on the
    // original transition set; it is wall-capped (see below) so a
    // transition-heavy net cannot starve the BFS fallback.
    if !has_colored_groups {
        let soft_deadline = quasi_liveness_lp_soft_deadline(config.deadline());
        if let Some(dead) =
            crate::lp_state_equation::lp_first_dead_transition(&reduced.net, soft_deadline)
        {
            eprintln!(
                "QuasiLiveness: LP-proved transition {} never enabled in any \
                 reachable marking (joint enabling conjunction + traps) → NOT quasi-live",
                dead.0
            );
            return Verdict::False;
        }
    }

    // Decision-Diagram exact fast-path (off by default — gated by
    // `dd-backend`). Placed AFTER the cheap structural/LP shortcuts and
    // BEFORE BMC/BFS. On a small bounded net the DD backend builds the
    // EXACT reachable-marking set and decides quasi-liveness directly:
    //
    //   quasi-live  ⟺  for EVERY transition group, some member is fireable
    //                   from some reachable marking,
    //                   i.e. EF(IsFireable(group_members)) holds for all groups.
    //
    // For P/T input each transition is its own singleton group; for colored
    // input `colored_transition_groups` carries the per-colored-transition
    // binding sets, so the colored question (some binding fireable) is asked
    // exactly via the OR semantics of `IsFireable`. Verdict TRUE iff every
    // group query is true, FALSE otherwise.
    //
    // The reduced net here is the identity net (QuasiLiveness uses no
    // structural reduction), so the original-net transition indices that
    // `colored_transition_groups` reference are valid. Soundness: the
    // reachable set is exact (build_sound_dd_spec); on ANY DD failure
    // (decline/timeout/panic) no verdict is emitted and we fall through to
    // BMC/BFS unchanged.
    #[cfg(feature = "dd-backend")]
    if let Some(verdict) =
        super::dd_fastpath::try_dd_quasi_liveness(net, colored_transition_groups, config.deadline())
    {
        eprintln!("QuasiLiveness: resolved exactly by DD reachable-set fast-path");
        return verdict;
    }

    // SMT-based per-transition BMC on the reduced net.
    let mut bmc_resolved =
        global_properties_bmc::run_quasi_liveness_bmc(&reduced.net, config.deadline());
    if all_groups_covered(&bmc_resolved, colored_transition_groups) {
        eprintln!("QuasiLiveness: all transitions resolved by BMC");
        return Verdict::True;
    }

    // --- Random-walk quasi-liveness witness PRE-PASS (strictly additive) ---
    //
    // A random walk that fires ONLY enabled transitions from the initial marking
    // visits only reachable markings BY CONSTRUCTION, so ANY transition observed
    // enabled there is provably quasi-live — a sound TRUE witness. We OR those
    // observed flags into the BMC seed, EXPANDING it. The identical exhaustive
    // BFS below still keeps the final word (it can only need to cover FEWER
    // transitions), so the walk can never remove or fake a verdict.
    //
    // SOUNDNESS: only the TRUE direction is consumed. A transition the walk never
    // sees enabled stays unresolved (it is NOT marked quasi-live) and falls
    // through to BFS. The walk never emits FALSE / never concludes the net is
    // (non-)quasi-live on its own.
    //
    // INDEX ALIGNMENT: `reduced` is the identity net (QuasiLiveness applies no
    // structural reduction — see `ReducedNet::identity(net)` above), so the walk
    // result vector aligns 1:1 with `bmc_resolved` and the (original-net)
    // `colored_transition_groups` indices.
    //
    // BUDGET: ADDITIVE / leftover-only. `under_approx_lane_deadline` takes at most
    // `min(remaining / 4, 8s)` and reserves the BFS-fallback tail, so the walk
    // can NEVER starve the exhaustive BFS. A skip/already-expired sentinel means
    // there is no leftover slice — fall through without walking.
    if !quasi_liveness_walk_disabled() {
        let walk_deadline = crate::examinations::reachability::under_approx_lane_deadline(
            &reduced.net,
            config.deadline(),
        );
        let walk_skip = walk_deadline.is_some_and(|d| Instant::now() >= d);
        if !walk_skip {
            let observed =
                crate::examinations::reachability_walk::run_random_walk_quasi_liveness_witness(
                    &reduced.net,
                    walk_deadline,
                );
            for (slot, &seen) in bmc_resolved.iter_mut().zip(observed.iter()) {
                *slot = *slot || seen;
            }
            if all_groups_covered(&bmc_resolved, colored_transition_groups) {
                eprintln!(
                    "QuasiLiveness: all transitions witnessed quasi-live by random walk \
                     (+BMC seed)"
                );
                return Verdict::True;
            }
        }
    }

    // GPU explicit-BFS tier (probe-then-GPU, mirroring the deadlock lane).
    // A bounded CPU probe first: transitions it observes enabled are sound
    // TRUE witnesses (merged into the seed), and a COMPLETED probe is the
    // exhaustive answer. Only a tripped cap escalates to the device, where
    // each unresolved group's ¬IsFireable(members) rides the engine's
    // invariant mechanism: a published witness marking enables some pending
    // group (classified host-side with `net.is_enabled`, the same oracle the
    // CPU observer uses) and the search reruns for the remainder; a clean
    // completion proves the pending groups can NEVER fire ⇒ FALSE.
    // Fail-closed: any GPU decline falls through to the CPU BFS unchanged.
    #[cfg(feature = "gpu")]
    if crate::gpu_state_space::gpu_lane_enabled(&reduced.net) {
        if let Some(cap) = crate::gpu_state_space::cpu_probe_cap(config.max_states()) {
            let probe_config = ExplorationConfig::new(cap)
                .with_deadline(config.deadline())
                .with_examination(config.examination());
            let mut observer = QuasiLivenessObserver::new_seeded(&bmc_resolved);
            let result = ExecutionPlan::observer(PorStrategy::None).run_observer(
                &reduced.net,
                &probe_config,
                &mut observer,
            );
            if all_groups_covered(observer.fired_slice(), colored_transition_groups) {
                return Verdict::True;
            }
            if result.completed {
                return Verdict::False;
            }
            // Sound TRUE witnesses from the probe (observed enabled at
            // reachable markings) shrink the device's query set.
            for (slot, &seen) in bmc_resolved.iter_mut().zip(observer.fired_slice()) {
                *slot = *slot || seen;
            }
            eprintln!(
                "[mcc] QuasiLiveness: bounded CPU probe tripped (cap {cap}); \
                 escalating to the GPU lane"
            );
        }
        if let Some(verdict) = quasi_liveness_gpu(
            &reduced.net,
            &bmc_resolved,
            colored_transition_groups,
            config.max_states(),
        ) {
            return verdict;
        }
    }

    let plan = ExecutionPlan::observer(PorStrategy::None);
    // Seed the observer with the BMC + random-walk witness results so BFS only
    // needs to discover the remaining unresolved transitions.
    let mut observer = QuasiLivenessObserver::new_seeded(&bmc_resolved);
    let result = match plan.run_checkpointable_observer(&reduced.net, &config, &mut observer) {
        Ok(result) => result,
        Err(error) => return checkpoint_cannot_compute("QuasiLiveness", &error),
    };

    if all_groups_covered(observer.fired_slice(), colored_transition_groups) {
        Verdict::True
    } else if result.completed {
        Verdict::False
    } else {
        Verdict::CannotCompute
    }
}

/// GPU quasi-liveness: decide the groups not yet witnessed quasi-live.
/// `Some(True)` = every group covered (witness markings enable them),
/// `Some(False)` = the EXHAUSTIVE exploration proved some pending group can
/// never fire, `None` = decline (CPU BFS decides).
///
/// Group semantics mirror [`all_groups_covered`]: the colored groups, plus a
/// singleton group for every transition not in any group (all singletons for
/// P/T input). `IsFireable(members)` carries the group's OR semantics
/// natively. The `IsFireable`-parity gate (summed vs per-arc weights)
/// declines nets with parallel input arcs from one place.
#[cfg(feature = "gpu")]
fn quasi_liveness_gpu(
    net: &PetriNet,
    resolved: &[bool],
    colored_transition_groups: &[Vec<usize>],
    max_states: usize,
) -> Option<Verdict> {
    use crate::gpu_state_space::{reachability_explore_gpu, GpuReachabilityOutcome};
    use crate::petri_net::TransitionIdx;
    use crate::resolved_predicate::ResolvedPredicate;

    let has_parallel_input_arcs = net.transitions.iter().any(|t| {
        let mut seen = std::collections::HashSet::new();
        t.inputs.iter().any(|arc| !seen.insert(arc.place.0))
    });
    if has_parallel_input_arcs {
        return None;
    }

    // Effective group list per `all_groups_covered`: colored groups + a
    // singleton per uncovered transition (all singletons for P/T).
    let mut groups: Vec<Vec<usize>> = colored_transition_groups.to_vec();
    let mut in_group = vec![false; net.num_transitions()];
    for group in colored_transition_groups {
        for &idx in group {
            if let Some(slot) = in_group.get_mut(idx) {
                *slot = true;
            }
        }
    }
    for (idx, &covered) in in_group.iter().enumerate() {
        if !covered {
            groups.push(vec![idx]);
        }
    }

    let mut group_covered: Vec<bool> = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .any(|&t| resolved.get(t).copied().unwrap_or(false))
        })
        .collect();

    loop {
        let pending: Vec<usize> = group_covered
            .iter()
            .enumerate()
            .filter(|(_, &covered)| !covered)
            .map(|(i, _)| i)
            .collect();
        if pending.is_empty() {
            eprintln!("[mcc] QuasiLiveness GPU lane: all groups witnessed quasi-live");
            return Some(Verdict::True);
        }
        let invariants: Vec<ResolvedPredicate> = pending
            .iter()
            .map(|&gi| {
                ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(
                    groups[gi]
                        .iter()
                        .map(|&t| TransitionIdx(u32::try_from(t).unwrap_or(u32::MAX)))
                        .collect(),
                )))
            })
            .collect();
        if invariants.iter().any(|inv| {
            if let ResolvedPredicate::Not(inner) = inv {
                if let ResolvedPredicate::IsFireable(ts) = inner.as_ref() {
                    return ts.iter().any(|t| (t.0 as usize) >= net.num_transitions());
                }
            }
            false
        }) {
            return None;
        }
        let invariant_refs: Vec<&ResolvedPredicate> = invariants.iter().collect();

        match reachability_explore_gpu(net, max_states, &invariant_refs)? {
            GpuReachabilityOutcome::Exhausted => {
                eprintln!(
                    "[mcc] QuasiLiveness GPU lane: {} group(s) proven never fireable \
                     by exhaustive completion",
                    pending.len(),
                );
                return Some(Verdict::False);
            }
            GpuReachabilityOutcome::Witness(marking) => {
                let mut resolved_any = false;
                for &gi in &pending {
                    if groups[gi].iter().any(|&t| {
                        u32::try_from(t)
                            .is_ok_and(|t32| net.is_enabled(&marking, TransitionIdx(t32)))
                    }) {
                        group_covered[gi] = true;
                        resolved_any = true;
                    }
                }
                if !resolved_any {
                    eprintln!(
                        "[mcc] QuasiLiveness GPU lane declined: witness enables no pending \
                         group (engine fault)"
                    );
                    return None;
                }
            }
        }
    }
}

/// Soft deadline for the strong per-transition dead-transition LP sweep.
///
/// Reserves the tail of the global budget for the exhaustive BFS fallback by
/// capping the sweep at [`QUASI_LIVENESS_LP_PHASE_CAP`] (or the global
/// deadline, whichever is sooner). Mirrors the deadlock siphon-LP soft cap: an
/// abandoned sweep yields no verdict and falls through, so the bound is
/// verdict-preserving.
fn quasi_liveness_lp_soft_deadline(global: Option<Instant>) -> Option<Instant> {
    let now = Instant::now();
    Some(match global {
        // B4: reserve the SCC/BMC tail (see `structural_live_soft_deadline`).
        // No-op at the contest budget; verdict-preserving (abandoned sweep falls
        // through).
        Some(global) => {
            let remaining = global.saturating_duration_since(now);
            now + QUASI_LIVENESS_LP_PHASE_CAP
                .min(remaining.saturating_sub(LIVENESS_SCC_FALLBACK_RESERVE_FLOOR))
        }
        None => now + QUASI_LIVENESS_LP_PHASE_CAP,
    })
}

/// Whether every quasi-liveness group has at least one fired transition.
///
/// `fired[t]` is the per-P/T-transition fired flag. `groups` is the
/// colored-transition grouping (one entry per colored transition,
/// listing its unfolded binding indices). When `groups` is empty (P/T
/// input), this reduces to "every P/T transition fired" — the classic
/// `observer.all_fired()` semantics. When `groups` is non-empty, every
/// group must have at least one fired member (the colored transition
/// has at least one fireable binding).
fn all_groups_covered(fired: &[bool], groups: &[Vec<usize>]) -> bool {
    if groups.is_empty() {
        return fired.iter().all(|&f| f);
    }
    let mut in_group = vec![false; fired.len()];
    for group in groups {
        for &idx in group {
            if idx < in_group.len() {
                in_group[idx] = true;
            }
        }
        if !group
            .iter()
            .any(|&idx| fired.get(idx).copied().unwrap_or(false))
        {
            return false;
        }
    }
    for (idx, &flag) in fired.iter().enumerate() {
        if !in_group[idx] && !flag {
            return false;
        }
    }
    true
}

pub(crate) fn liveness_verdict(net: &PetriNet, config: &ExplorationConfig) -> Verdict {
    // P/T entry point: no colored grouping. Every transition is its own
    // group (singleton), so the colored-question and the P/T-question
    // coincide. Existing call sites keep working unchanged.
    liveness_verdict_with_groups(net, config, &[])
}

/// L4-Liveness verdict with optional **colored** transition groups.
///
/// For P/T inputs, pass `&[]` — the function behaves exactly like the
/// classic `liveness_verdict(net, config)` (every transition is its own
/// group).
///
/// For colored inputs, pass `aliases.colored_transition_groups()`. The
/// MCC question on colored nets is *colored* L4-liveness: every
/// **colored** transition must have AT LEAST ONE binding that is L4-
/// live, NOT every unfolded P/T binding. Without grouping, the per-
/// binding shortcuts (`lp_dead_transition`, `structural_not_live_t_
/// semiflows`, `run_liveness_bmc`, the per-binding mu-calculus
/// iteration, and the per-binding SCC check) all incorrectly disprove
/// liveness when a *single* binding (e.g. a diagonal `[i ineq k]`
/// combination) is structurally dead — even though the colored
/// transition is live via off-diagonal bindings. The two wrong answers
/// at the 13-exam measurement (`GlobalResAllocation-COL-03` from the
/// LP-dead shortcut, `TokenRing-COL-005` from the per-binding mu-
/// calculus iteration) both trace to this conflation.
pub(crate) fn liveness_verdict_with_groups(
    net: &PetriNet,
    config: &ExplorationConfig,
    colored_transition_groups: &[Vec<usize>],
) -> Verdict {
    let has_colored_groups = !colored_transition_groups.is_empty();

    // --- Pre-reduction structural analysis on the ORIGINAL net ---
    // `structural_live → Some(true)` is sound for colored: structural
    // liveness of the unfolded P/T net implies every binding (and thus
    // every colored transition) is live. The `Some(false)` direction
    // is per-binding and unsound for colored, so we gate it.
    match crate::structural::structural_live(net, structural_live_soft_deadline(config.deadline()))
    {
        Some(true) => {
            eprintln!("Liveness: structurally live (original net, exact net-class certificate)");
            return Verdict::True;
        }
        Some(false) if !has_colored_groups => {
            eprintln!(
                "Liveness: structurally non-live (original net, exact net-class certificate)"
            );
            return Verdict::False;
        }
        _ => {}
    }

    if !has_colored_groups {
        if let Some(false) = crate::structural::structural_not_live_t_semiflows(net) {
            eprintln!("Liveness: uncovered transition on original net (T-semiflow + bounded)");
            return Verdict::False;
        }

            if let Some(false) = crate::structural::lp_dead_transition(
            net,
            structural_live_soft_deadline(config.deadline()),
        ) {
            eprintln!("Liveness: LP-proved dead transition on original net");
            return Verdict::False;
        }
        }

    // --- Stutter-sensitive reduced net ---
    // Agglomeration is unsound for Liveness (#1503), but dead-transition,
    // constant-place, and isolated-place removal are safe.
    //
    // NOTE: when colored groups are present, we cannot apply the
    // reduction here — reduction reindexes transitions, which would
    // invalidate the `colored_transition_groups` indices passed in.
    let reduced = if has_colored_groups {
        ReducedNet::identity(net)
    } else {
        crate::reduction::reduce_iterative_structural_with_mode(
            net,
            &[],
            crate::reduction::ReductionMode::StutterSensitiveLTL,
            config.deadline(),
        )
        .unwrap_or_else(|_| ReducedNet::identity(net))
    };
    // --- Dead-transition removal is itself a non-liveness witness (P/T) ---
    //
    // Under `StutterSensitiveLTL` the ONLY transition-removing rule that
    // fires is dead-transition removal (`allows_dead_transition_removal`
    // is the sole transition-removal predicate true for this mode; all of
    // agglomeration / duplicate / self-loop / dominated / sink / Rule R/S
    // require `Reachability` or next-free modes). So any transition the
    // reduction deleted is a *structurally dead* transition — one whose
    // input place has no producer and starts below the arc weight (or sits
    // in a permanently-empty trap). Such a transition can NEVER be enabled
    // in ANY reachable marking, hence `AG EF enabled(t)` is FALSE, hence
    // the original net is NOT L4-live and the correct verdict is FALSE.
    //
    // The downstream liveness decision (mu-calculus AG EF, BMC, SCC) runs
    // only over the SURVIVING transitions in `reduced.net`, so it never
    // sees the deleted dead transition and can wrongly report a definite
    // TRUE. The pre-reduction `lp_dead_transition` shortcut normally
    // catches this on small nets, but it FAILS OPEN on large nets
    // (`np + nt > MAX_LP_VARIABLES`, finding #12) while the reduction still
    // deletes the dead transition — exactly the unsound gap. We therefore
    // read the dead-transition fact from the reduction's OWN report (a
    // purely structural, uncapped analysis) rather than the capped LP, and
    // emit the exact FALSE here.
    //
    // Gating on `report.dead_transitions` (not merely "some transition was
    // removed") keeps this precise: only genuinely never-enabled
    // transitions appear there. The colored path uses an identity
    // reduction (no removals), so its `dead_transitions` is always empty
    // and the colored verdict is untouched.
    if !has_colored_groups && !reduced.report.dead_transitions.is_empty() {
        eprintln!(
            "Liveness: reduction removed {} structurally-dead transition(s) \
             (never enabled in any reachable marking) → NOT live",
            reduced.report.dead_transitions.len()
        );
        return Verdict::False;
    }

    let config = config
        .refitted_for_net(&reduced.net)
        .with_workers(1)
        .with_storage_mode(crate::explorer::StorageMode::Memory);

    // --- Post-reduction structural analysis on the REDUCED net ---
    // Same colored-soundness gating as above.
    match crate::structural::structural_live(
        &reduced.net,
        structural_live_soft_deadline(config.deadline()),
    ) {
        Some(true) => {
            eprintln!("Liveness: structurally live (exact net-class certificate)");
            return Verdict::True;
        }
        Some(false) if !has_colored_groups => {
            eprintln!("Liveness: structurally non-live (exact net-class certificate)");
            return Verdict::False;
        }
        _ => {}
    }

    if !has_colored_groups {
        // T-semiflow coverage: per-binding FALSE shortcut — unsound for colored.
        if let Some(false) = crate::structural::structural_not_live_t_semiflows(&reduced.net) {
            eprintln!("Liveness: uncovered transition (T-semiflow + bounded)");
            return Verdict::False;
        }

        // LP dead-transition: per-binding FALSE shortcut — unsound for colored.
        if let Some(false) = crate::structural::lp_dead_transition(
            &reduced.net,
            structural_live_soft_deadline(config.deadline()),
        ) {
            eprintln!("Liveness: LP-proved dead transition (upper bound insufficient)");
            return Verdict::False;
        }
    }

    // Deadlock implies non-liveness: SOUND for colored inputs too.
    // A reachable marking with zero enabled transitions disables every
    // colored binding, hence every colored transition — system not live
    // regardless of grouping.
    if let Some(true) = global_properties_bmc::run_deadlock_bmc(&reduced.net, config.deadline()) {
        eprintln!("Liveness: reachable deadlock found (BMC) → NOT live");
        return Verdict::False;
    }

    // Native PDR reachable-deadlock FALSE shortcut (LV-3): a reachable marking
    // with no enabled transition makes EVERY transition permanently disabled
    // from there on, so (for nt > 0) the net is NOT L4-live → FALSE. SOUND for
    // colored inputs too: a zero-enabled marking disables every colored binding.
    //
    // `run_deadlock_bmc` above needs an external `ay` solver — it returns
    // `None` the instant `find_ay()` fails (the local / no-ay configuration) —
    // so without this lane the reachable-deadlock FALSE shortcut never fires
    // off-cluster. PDR is NATIVE (no ay) and decides many deadlocking nets
    // symbolically.
    //
    // STRICT verdict-preservation: only the `Some(true)` (reachable deadlock
    // found) direction is consumed. `Some(false)` (deadlock-free) does NOT
    // imply liveness and `None` (inconclusive) both fall through to the
    // mu-calculus / SCC path unchanged. The `num_transitions() > 0` guard
    // suppresses the `nt == 0` early-return inside `run_deadlock_pdr` (which
    // reports the empty net as a trivial deadlock) so a transition-free net —
    // vacuously L4-live, the value the mu-calculus path returns — is never
    // flipped to FALSE. The deadline is capped so a slow PDR cannot starve the
    // deep mu engine.
    if reduced.net.num_transitions() > 0 {
        let pdr_deadline = liveness_deadlock_pdr_deadline(config.deadline());
        if let Some(true) = global_properties_pdr::run_deadlock_pdr(&reduced.net, pdr_deadline) {
            eprintln!("Liveness: reachable deadlock found (native PDR) → NOT live");
            return Verdict::False;
        }
    }

    if !has_colored_groups {
        // Per-binding liveness BMC: unsound for colored (the binding
        // can be permanently disabled while another binding of the
        // same colored transition remains live).
        if let Some(false) =
            global_properties_bmc::run_liveness_bmc(&reduced.net, config.deadline())
        {
            eprintln!("Liveness: transition permanently disabled (BMC + k-induction) → NOT live");
            return Verdict::False;
        }
    }

    // --- Decision-Diagram exact fast-path (off by default — gated by
    // `dd-backend`). Placed AFTER the cheap structural / LP / deadlock-PDR /
    // liveness-BMC shortcuts and BEFORE the budget-limited local mu-calculus
    // and the full SCC graph build, which are exactly the lanes that DECLINE
    // (OOM / node-cap abort / graph-build timeout) on the bounded-but-large
    // nets where the symbolic DD reachable set still converges — the ~30%
    // CANNOT_COMPUTE category this lane targets.
    //
    // L4-liveness is the nested CTL property `AG(EF(IsFireable(t)))` for every
    // transition (P/T) or colored group (some binding fireable). On a bounded
    // net the DD backend builds the EXACT reachable-marking set and decides it
    // directly via the oracle-verified nested-CTL symbolic evaluator:
    //
    //   live  ⟺  AND_groups AG(EF(IsFireable(group_members)))  at the initial
    //            marking over the exact reachable set.
    //
    // The reachable set is exact (`build_sound_dd_spec`), and the spec
    // preserves transition order, so the group indices match. SOUNDNESS: this
    // runs on the ORIGINAL `net`, not `reduced.net`. For the colored path
    // `reduced` is an identity reduction (no reindexing), so original and
    // reduced indices coincide and the passed `colored_transition_groups` are
    // valid. For the P/T path the StutterSensitiveLTL reduction can only
    // remove structurally-dead transitions, and any such removal already
    // returned FALSE above (`reduced.report.dead_transitions` guard), so on
    // reaching here the P/T transition set is unchanged from the original and
    // running over the original net is exact. On ANY DD failure
    // (decline / timeout / panic / above-cap) no verdict is emitted and we
    // fall through to the mu-calculus / SCC pipeline unchanged. The DD budget
    // is clamped (see `dd_fastpath::dd_budget`) so a non-converging fixpoint
    // cannot starve the mu / SCC fallback under a finite deadline.
    //
    // `TY_DISABLE_DD_LIVENESS=1` skips the lane (escape hatch + differential-
    // measurement toggle); unset/0 keeps it ON (the default at HEAD, where
    // `dd-backend` is a default feature). Skipping only ever falls through to
    // the existing pipeline, so the toggle is strictly verdict-preserving.
    #[cfg(feature = "dd-backend")]
    if !matches!(
        std::env::var("TY_DISABLE_DD_LIVENESS").ok().as_deref(),
        Some("1") | Some("true")
    ) {
        if let Some(verdict) =
            super::dd_fastpath::try_dd_liveness(net, colored_transition_groups, config.deadline())
        {
            eprintln!("Liveness: resolved exactly by DD reachable-set nested-CTL fast-path");
            return verdict;
        }
    }

    // --- GPU retained-graph nested-CTL tier (feature `gpu`) ---
    //
    // The explicit twin of the DD lane above: L4-liveness is
    // `AND_groups AG(EF(IsFireable(group)))`, evaluated here by device
    // fixpoints over the RETAINED exhaustive reachable set (the same engine
    // the deep-CTL pipeline uses, differentially validated against the
    // exhaustive CtlChecker). Runs on the ORIGINAL `net` with the same
    // index-validity argument as the DD lane: the colored path's reduction
    // is an identity, and any P/T dead-transition removal already returned
    // FALSE above. EXACT both ways on success (the retained set is the full
    // reachable set). Fail-closed: any decline (inadmissible atom, capacity,
    // deadline, CUDA-less host) falls through to the mu-calculus / SCC
    // pipeline unchanged.
    #[cfg(feature = "gpu")]
    if crate::gpu_state_space::gpu_lane_enabled(net) {
        use crate::examinations::ctl::resolve::ResolvedCtl;
        use crate::petri_net::TransitionIdx;
        use crate::resolved_predicate::ResolvedPredicate;

        let groups_valid = colored_transition_groups
            .iter()
            .flatten()
            .all(|&t| t < net.num_transitions());
        if groups_valid {
            let mut groups: Vec<Vec<usize>> = colored_transition_groups.to_vec();
            let mut in_group = vec![false; net.num_transitions()];
            for group in colored_transition_groups {
                for &idx in group {
                    in_group[idx] = true;
                }
            }
            for (idx, &covered) in in_group.iter().enumerate() {
                if !covered {
                    groups.push(vec![idx]);
                }
            }
            let formulas: Vec<ResolvedCtl> = groups
                .iter()
                .map(|group| {
                    ResolvedCtl::AG(Box::new(ResolvedCtl::EF(Box::new(ResolvedCtl::Atom(
                        ResolvedPredicate::IsFireable(
                            group
                                .iter()
                                .map(|&t| TransitionIdx(u32::try_from(t).unwrap_or(u32::MAX)))
                                .collect(),
                        ),
                    )))))
                })
                .collect();
            if let Some(verdicts) = crate::gpu_state_space::ctl_check_gpu(
                net,
                &formulas,
                config.max_states(),
                config.deadline(),
            ) {
                let live = verdicts.iter().all(|&v| v);
                eprintln!(
                    "Liveness: resolved exactly by the GPU retained-set nested-CTL lane \
                     ({} group(s))",
                    groups.len(),
                );
                return if live { Verdict::True } else { Verdict::False };
            }
        }
    }

    // --- Primary verdict path: mu-calculus AG(EF(enabled(group))) per (colored) group ---
    //
    // For P/T (empty `colored_transition_groups`): one singleton group
    // per transition — exact pre-fix semantics.
    //
    // For colored: one group per colored transition, with `IsFireable
    // (group)` evaluating to true at marking M iff ANY binding in the
    // group is enabled at M. So the formula `AG(EF(IsFireable(group)))`
    // asks "from every reachable marking, can we reach a marking where
    // at least one binding of this colored transition is enabled?",
    // which is the textbook colored-L4-liveness question.
    if let Some(mu_config) = liveness_mu_config(&config) {
        match liveness_via_mu_calculus(&reduced.net, &mu_config, colored_transition_groups) {
            Verdict::True => return Verdict::True,
            Verdict::False => return Verdict::False,
            Verdict::CannotCompute => {
                // Fall through to SCC path below.
            }
        }
    } else {
        eprintln!("Liveness: skipping mu-calculus phase to preserve exact SCC fallback budget");
    }

    let live_transition_count = reduced.transition_unmap.len();

    let plan = ExecutionPlan::graph();
    let graph = plan.run_graph(&reduced.net, &config);

    if !graph.completed {
        Verdict::CannotCompute
    } else {
        let sccs = tarjan_scc(&graph);
        let bottoms = bottom_sccs(&graph.adj, &sccs);
        let live = if has_colored_groups {
            check_colored_liveness(
                &graph,
                &sccs,
                &bottoms,
                live_transition_count,
                colored_transition_groups,
            )
        } else {
            check_liveness(&graph, &sccs, &bottoms, live_transition_count)
        };
        if live {
            Verdict::True
        } else {
            Verdict::False
        }
    }
}

/// Reserve for the exact SCC graph-build fallback: a fraction of the remaining
/// budget, floored so a modest deadline still leaves the fallback a usable
/// slice. The deep mu-calculus phase keeps `remaining - reserve` (the bulk).
fn liveness_scc_fallback_reserve(remaining: Duration) -> Duration {
    remaining
        .mul_f64(LIVENESS_SCC_FALLBACK_FRACTION)
        .max(LIVENESS_SCC_FALLBACK_RESERVE_FLOOR)
}

/// Bounded deadline for the native-PDR reachable-deadlock FALSE shortcut: the
/// sooner of [`LIVENESS_DEADLOCK_PDR_PHASE_CAP`] from now and the global
/// deadline. Always imposes the cap (even with no global deadline) so the
/// shortcut is a quick attempt that leaves the bulk of the budget to the deep
/// mu engine. Bounding only makes PDR more likely to be inconclusive (and fall
/// through), so it is verdict-preserving.
fn liveness_deadlock_pdr_deadline(global: Option<Instant>) -> Option<Instant> {
    let cap = Instant::now() + LIVENESS_DEADLOCK_PDR_PHASE_CAP;
    Some(match global {
        Some(global) => cap.min(global),
        None => cap,
    })
}

fn liveness_mu_config(config: &ExplorationConfig) -> Option<ExplorationConfig> {
    let Some(global_deadline) = config.deadline() else {
        return Some(config.clone());
    };

    let now = Instant::now();
    let remaining = global_deadline.saturating_duration_since(now);
    let scc_reserve = liveness_scc_fallback_reserve(remaining);
    if remaining <= scc_reserve.saturating_add(LIVENESS_MU_MIN_BUDGET) {
        return None;
    }

    // The mu phase gets the bulk: everything except the proportional SCC
    // reserve. No fixed cap — on a generous deadline the deep engine should
    // take the majority, since the SCC fallback that the reserve protects
    // cannot plausibly finish the graph build on the large nets that reach it.
    let phase_budget = remaining.checked_sub(scc_reserve).unwrap();
    if phase_budget < LIVENESS_MU_MIN_BUDGET {
        return None;
    }

    Some(config.clone().with_deadline(Some(now + phase_budget)))
}

/// Colored-aware bottom-SCC liveness check.
///
/// A colored Petri net is L4-live iff in every bottom SCC of the
/// reachability graph, every colored transition has at least one
/// binding that fires somewhere inside the SCC, AND every singleton
/// (non-grouped) P/T transition fires somewhere inside the SCC.
///
/// This is the colored generalisation of `check_liveness`: when
/// `colored_transition_groups` is empty the two checks coincide
/// (every transition is its own singleton group).
fn check_colored_liveness(
    graph: &crate::explorer::ReachabilityGraph,
    sccs: &[Vec<u32>],
    bottom_indices: &[usize],
    num_transitions: usize,
    colored_transition_groups: &[Vec<usize>],
) -> bool {
    use rustc_hash::FxHashSet;

    if num_transitions == 0 {
        return true;
    }

    // Index: per P/T transition, is it inside a multi-binding colored group?
    let mut in_group = vec![false; num_transitions];
    for group in colored_transition_groups {
        for &idx in group {
            if idx < in_group.len() {
                in_group[idx] = true;
            }
        }
    }

    for &scc_idx in bottom_indices {
        let scc = &sccs[scc_idx];
        let scc_states: FxHashSet<u32> = scc.iter().copied().collect();
        let mut fireable: FxHashSet<u32> = FxHashSet::default();
        for &state in scc {
            for &(succ, trans) in &graph.adj[state as usize] {
                if scc_states.contains(&succ) {
                    fireable.insert(trans);
                }
            }
        }

        // Every singleton (non-group) transition must fire in this SCC.
        for tidx in 0..num_transitions {
            if !in_group[tidx] && !fireable.contains(&(tidx as u32)) {
                return false;
            }
        }
        // Every multi-binding colored group must have ≥1 binding fire.
        for group in colored_transition_groups {
            if !group.iter().any(|&idx| fireable.contains(&(idx as u32))) {
                return false;
            }
        }
    }

    true
}

/// L4-Liveness via mu-calculus.
///
/// For each transition `t` in `net`, evaluates the CTL formula
/// `AG(EF(IsFireable(t)))` — "from every reachable marking, there
/// exists a firing sequence that eventually enables `t`". Translates
/// through [`ctl_to_mu`] and dispatches to [`LocalMuSolver`].
/// Aggregates the per-transition verdicts into the system-liveness
/// answer with short-circuit on the first non-live transition.
///
/// Returns:
/// - `Verdict::True` if every transition is L4-live.
/// - `Verdict::False` on the first transition that is not L4-live
///   (with the transition index logged to stderr as a witness).
/// - `Verdict::CannotCompute` if the solver hits a resource cap
///   (deadline / node cap / state cap) on any transition before any
///   transition is proven non-live. Soundness: never returns
///   `Verdict::True` when any transition's verdict is unknown.
///
/// ## Soundness
///
/// - The atomic predicate `IsFireable(vec![t])` evaluates to true at
///   a marking `M` iff `net.is_enabled(M, t)`, i.e. every input arc's
///   place has at least the arc weight. This is the exact MCC notion
///   of "transition `t` is enabled at `M`". The check is overflow-
///   safe by virtue of unsigned `u64` arithmetic in
///   [`PetriNet::is_enabled`](crate::petri_net::PetriNet::is_enabled)
///   and colored-aware because callers pass the unfolded P/T net.
/// - `AG(EF(p))` translated via `ctl_to_mu` becomes
///   `νZ. (μY. p ∨ ◇Y) ∧ □Z`. Pure CTL nesting — no strict mu/nu
///   alternation, so the alternation-free solver soundly resolves it
///   (see `mu_calculus` module docstring).
/// - Aggregation: a system-True verdict requires every transition's
///   verdict to be True. A single False (or CannotCompute) is enough
///   to deny True. Short-circuit on False is sound; we never aggregate
///   "True" from a strict subset of transitions.
/// - Reduction interaction: this function operates on a net that has
///   already been reduced under a liveness-preserving mode (the
///   caller in `liveness_verdict` passes the
///   `ReductionMode::StutterSensitiveLTL`-reduced net, which only
///   strips dead transitions, constant places, and isolated places —
///   all liveness-safe). The structural shortcuts in
///   `liveness_verdict` already rule out the case where any of those
///   dead transitions exist on the original net, so the surviving
///   transitions in the reduced net are exactly the transitions whose
///   liveness must be verified.
pub(crate) fn liveness_via_mu_calculus(
    net: &PetriNet,
    config: &ExplorationConfig,
    colored_transition_groups: &[Vec<usize>],
) -> Verdict {
    let num_transitions = net.num_transitions();
    if num_transitions == 0 {
        // Vacuously live: no transitions means there is nothing that
        // could be dead.
        return Verdict::True;
    }

    // Build the iteration list of binding-sets. For P/T (empty
    // `colored_transition_groups`), each P/T transition is its own
    // singleton group — exact pre-fix semantics. For colored, the
    // multi-binding groups carry the colored aggregation, and the
    // P/T transitions NOT in any multi-binding group remain
    // singletons (their colored parent has exactly one binding).
    let groups: Vec<Vec<TransitionIdx>> = if colored_transition_groups.is_empty() {
        (0..num_transitions)
            .map(|tidx| vec![TransitionIdx(tidx as u32)])
            .collect()
    } else {
        let mut in_group = vec![false; num_transitions];
        for group in colored_transition_groups {
            for &idx in group {
                if idx < in_group.len() {
                    in_group[idx] = true;
                }
            }
        }
        let mut out: Vec<Vec<TransitionIdx>> = colored_transition_groups
            .iter()
            .map(|g| g.iter().map(|&idx| TransitionIdx(idx as u32)).collect())
            .collect();
        for tidx in 0..num_transitions {
            if !in_group[tidx] {
                out.push(vec![TransitionIdx(tidx as u32)]);
            }
        }
        out
    };

    let total_groups = groups.len();
    let mu_deadline = config.deadline();

    // ONE reused solver across every group's `AG(EF(enabled(group)))`. The
    // reachable state space + cached successors are a function of the NET only
    // (not the per-group formula), so building them once and re-exploring each
    // group's own EDG over the warm cache replaces the previous O(G·|R|) per-group
    // rebuild (G ≈ |T| singleton groups on P/T; BridgeAndVehicles-PT ~970 groups)
    // with O(|R|) + O(Σ_g EDG_g). Verdict-identical: the successor relation is
    // deterministic and formula-independent, `solve` resets the per-formula EDG,
    // and the mu-fixpoint verdict is invariant to state-interning order. The
    // memory-budgeted node cap still bounds the shared state space to a
    // memory-proportional size (OOM-safe; a cap/deadline hit is `Err(abort)`).
    let mut solver =
        LocalMuSolver::new(net, config).with_node_cap(LocalMuSolver::memory_budgeted_node_cap(net));
    let solve_group =
        |solver: &mut LocalMuSolver, group: &[TransitionIdx], deadline: Option<Instant>| {
            solver.set_deadline(deadline);
            solver.solve(&ctl_to_mu(&ag_ef_enabled(group)))
        };
    let report_not_live = |gidx: usize, group: &[TransitionIdx]| {
        eprintln!(
            "Liveness (mu-calculus): colored group {} (bindings {:?}) is not L4-live",
            gidx,
            group.iter().map(|t| t.0).collect::<Vec<_>>()
        );
    };

    // Groups proven live in Phase A are skipped by Phase B.
    let mut resolved_live = vec![false; total_groups];
    let mut cannot_compute_witness: Option<usize> = None;

    // ── Phase A: cheap FALSE scan ───────────────────────────────────────────
    // Give EVERY group a small bounded slice to surface a quick FALSE in any
    // group. This is the de-serialization win: previously a single hard early
    // group could consume the entire mu phase, so a later group's decisive
    // FALSE was never reached — the system fell to CANNOT_COMPUTE even though a
    // sound FALSE existed. A FALSE from any group denies L4-liveness for the
    // whole net, so returning it here is sound. The scan is capped at
    // `LIVENESS_MU_FALSE_SCAN_FRACTION` of the budget so Phase B keeps the rest.
    if let Some(deadline) = mu_deadline {
        let scan_start = Instant::now();
        let phase_a_deadline = scan_start
            + deadline
                .saturating_duration_since(scan_start)
                .mul_f64(LIVENESS_MU_FALSE_SCAN_FRACTION);
        for (gidx, group) in groups.iter().enumerate() {
            // Wall-cap Phase A: unscanned groups are handled by Phase B.
            let now = Instant::now();
            if now >= phase_a_deadline {
                break;
            }
            let remaining_groups = (total_groups - gidx).max(1) as u32;
            let slice = phase_a_deadline.saturating_duration_since(now) / remaining_groups;
            match solve_group(&mut solver, group, Some(now + slice)) {
                Ok(true) => resolved_live[gidx] = true,
                Ok(false) => {
                    report_not_live(gidx, group);
                    return Verdict::False;
                }
                // Aborts (node cap or slice deadline) defer to Phase B.
                Err(_) => {}
            }
        }
    }

    // ── Phase B: deep proof of the still-unresolved groups ──────────────────
    // Each unresolved group gets the FULL remaining budget (deadline =
    // `mu_deadline`), reproducing the original sequential behaviour so a
    // deciding group that needs nearly the whole phase to resolve is NOT
    // starved by the scan. The top-of-loop wall-cap stops the loop at
    // `mu_deadline`: each solver does un-pollable setup before its first
    // deadline check, so a net with hundreds of binding groups
    // (BridgeAndVehicles-PT-V20P10N20: ~970 groups) would otherwise overrun the
    // wall clock churning ~0-budget solvers (75 s observed). Unscanned groups
    // stay unknown — exactly the CANNOT_COMPUTE the old single-group bail gave.
    for (gidx, group) in groups.iter().enumerate() {
        if resolved_live[gidx] {
            continue;
        }
        if let Some(deadline) = mu_deadline {
            if Instant::now() >= deadline {
                if cannot_compute_witness.is_none() {
                    cannot_compute_witness = Some(gidx);
                }
                break;
            }
        }
        match solve_group(&mut solver, group, mu_deadline) {
            Ok(true) => {
                // This (colored) transition is live; continue checking the rest.
            }
            Ok(false) => {
                report_not_live(gidx, group);
                return Verdict::False;
            }
            Err(abort) => {
                // Record but keep scanning: a later group may still yield FALSE
                // and resolve the system verdict — a sound short-circuit even
                // when this group's verdict is unknown.
                eprintln!(
                    "Liveness (mu-calculus): colored group {} aborted ({})",
                    gidx, abort
                );
                if cannot_compute_witness.is_none() {
                    cannot_compute_witness = Some(gidx);
                }
            }
        }
    }

    match cannot_compute_witness {
        Some(_) => Verdict::CannotCompute,
        None => Verdict::True,
    }
}

/// Build `AG(EF(IsFireable(group)))`.
///
/// `IsFireable(group)` evaluates true at a marking `M` iff ANY
/// binding in `group` is enabled at `M` (the disjunctive semantics
/// is already baked into [`crate::resolved_predicate::crate::resolved_predicate::ResolvedPredicate::IsFireable`]). When
/// `group` is a singleton, this reduces to the classic per-
/// transition encoding `AG(EF(IsFireable([t])))`.
fn ag_ef_enabled(group: &[TransitionIdx]) -> GenericCtlFormula<ResolvedPredicate> {
    let enabled = ResolvedPredicate::IsFireable(group.to_vec());
    GenericCtlFormula::AG(Box::new(GenericCtlFormula::EF(Box::new(
        GenericCtlFormula::Atom(enabled),
    ))))
}

/// Real-device tests for the QuasiLiveness GPU tier (skipped on CUDA-less
/// hosts). These call [`quasi_liveness_gpu`] directly so the structural/LP
/// short-circuits in the full verdict path cannot mask the device path.
#[cfg(all(test, feature = "gpu"))]
mod gpu_quasi_liveness_tests {
    use super::*;
    use crate::petri_net::{Arc, PlaceInfo, TransitionInfo};

    fn cuda_available() -> bool {
        if tla_gpu::probe().is_err() {
            eprintln!("skipping GPU quasi-liveness test: no usable CUDA device");
            return false;
        }
        true
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: crate::petri_net::PlaceIdx(place),
            weight,
        }
    }

    #[test]
    fn all_transitions_witnessed_quasi_live() {
        if !cuda_available() {
            return;
        }
        // Toggle: both transitions fire somewhere => TRUE via witnesses.
        let net = PetriNet {
            name: Some("toggle".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        };
        let resolved = vec![false, false];
        assert_eq!(
            quasi_liveness_gpu(&net, &resolved, &[], 1000),
            Some(Verdict::True)
        );
    }

    #[test]
    fn dead_transition_proven_by_exhaustion() {
        if !cuda_available() {
            return;
        }
        // `tdead` consumes a place no transition ever marks => the pending
        // group survives the exhaustive sweep => FALSE.
        let net = PetriNet {
            name: Some("dead".into()),
            places: vec![place("p0"), place("pdead")],
            transitions: vec![
                trans("tlive", vec![arc(0, 1)], vec![arc(0, 1)]),
                trans("tdead", vec![arc(1, 1)], vec![arc(1, 1)]),
            ],
            initial_marking: vec![1, 0],
        };
        let resolved = vec![false, false];
        assert_eq!(
            quasi_liveness_gpu(&net, &resolved, &[], 1000),
            Some(Verdict::False)
        );
    }

    #[test]
    fn colored_group_or_semantics_cover_dead_binding() {
        if !cuda_available() {
            return;
        }
        // Group {tlive, tdead}: SOME member fireable => the colored group is
        // quasi-live even though `tdead` alone is not => TRUE.
        let net = PetriNet {
            name: Some("group".into()),
            places: vec![place("p0"), place("pdead")],
            transitions: vec![
                trans("tlive", vec![arc(0, 1)], vec![arc(0, 1)]),
                trans("tdead", vec![arc(1, 1)], vec![arc(1, 1)]),
            ],
            initial_marking: vec![1, 0],
        };
        let resolved = vec![false, false];
        let groups = vec![vec![0usize, 1usize]];
        assert_eq!(
            quasi_liveness_gpu(&net, &resolved, &groups, 1000),
            Some(Verdict::True)
        );
    }
}

#[cfg(test)]
mod mu_tests {
    //! Differential and unit tests for the mu-calculus Liveness path.
    //!
    //! The encoding is `AG(EF(IsFireable(t)))` for each transition `t`,
    //! aggregated conjunctively over all transitions. These tests
    //! exercise the encoding and the aggregation logic on small hand-
    //! crafted nets, then differentially compare against the
    //! independently-implemented SCC verdict on the same net to confirm
    //! the two paths agree.
    //!
    //! See the parent module's `liveness_via_mu_calculus` for the
    //! encoding semantics.
    use super::*;
    use crate::examinations::liveness::check_liveness;
    use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionInfo};
    use crate::scc::{bottom_sccs, tarjan_scc};

    fn config() -> ExplorationConfig {
        ExplorationConfig::new(10_000)
    }

    #[test]
    fn test_liveness_mu_config_reserves_scc_fallback_budget() {
        let now = Instant::now();
        let global = now + Duration::from_secs(60);
        let config = ExplorationConfig::new(10_000).with_deadline(Some(global));

        let mu_config = liveness_mu_config(&config).expect("mu phase should have budget");
        let mu_deadline = mu_config
            .deadline()
            .expect("mu phase should be deadline-limited");

        // The mu phase must leave at least the proportional SCC reserve.
        let reserve = liveness_scc_fallback_reserve(Duration::from_secs(60));
        // `global - reserve` is Instant - Duration; checked_sub would yield an
        // Option<Instant> and break the comparison expression.
        #[allow(clippy::unchecked_time_subtraction)]
        {
            assert!(mu_deadline <= global - reserve + Duration::from_millis(50));
        }
        // ...and it must get the BULK of a generous budget — far more than the
        // old flat 5s cap (here ≈ 45s of a 60s deadline).
        assert!(mu_deadline >= now + Duration::from_secs(30));
    }

    #[test]
    fn test_liveness_scc_fallback_reserve_scales_and_floors() {
        // Large budget: a flat fraction of the remaining deadline.
        assert_eq!(
            liveness_scc_fallback_reserve(Duration::from_secs(60)),
            Duration::from_secs(60).mul_f64(LIVENESS_SCC_FALLBACK_FRACTION)
        );
        // Small budget: floored so the exact fallback keeps a usable slice.
        assert_eq!(
            liveness_scc_fallback_reserve(Duration::from_secs(4)),
            LIVENESS_SCC_FALLBACK_RESERVE_FLOOR
        );
    }

    #[test]
    fn test_liveness_mu_config_skips_when_only_scc_fallback_remains() {
        // A tiny budget where the floor reserve plus the mu minimum exceeds the
        // whole remaining deadline must skip the mu phase entirely (all budget
        // goes to the exact SCC fallback).
        let global =
            Instant::now() + LIVENESS_SCC_FALLBACK_RESERVE_FLOOR + LIVENESS_MU_MIN_BUDGET / 2;
        let config = ExplorationConfig::new(10_000).with_deadline(Some(global));

        assert!(liveness_mu_config(&config).is_none());
    }

    /// 2-state alternator: place p0 starts with one token; t0 moves it
    /// to p1, t1 moves it back. Both transitions are L4-live (every
    /// reachable marking can reach a state where each is enabled).
    fn alternator_net() -> PetriNet {
        PetriNet {
            name: Some("alternator".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p0".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".to_string(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t1".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Net with a structurally dead transition. p0=1 initially; t0
    /// moves p0 → p1 (one-shot); t_dead needs p2 ≥ 1 but p2 is never
    /// produced. t_dead is NEVER firable from any reachable marking,
    /// so the net is not live.
    fn dead_transition_net() -> PetriNet {
        PetriNet {
            name: Some("dead-transition".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p0".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p2".to_string(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t_dead".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                    outputs: vec![],
                },
            ],
            initial_marking: vec![1, 0, 0],
        }
    }

    /// Net where t_lazy is enabled in the initial marking but firing
    /// t_cycle keeps the alternator running indefinitely without
    /// firing t_lazy. Since t_lazy stays enabled in every reachable
    /// marking, EF(enabled(t_lazy)) holds at every state, so AG EF
    /// enabled(t_lazy) is True. t_cycle is also live because it's
    /// always enabled. Net is L4-live.
    ///
    /// Structure: place p with 1 token; t_cycle consumes-and-produces
    /// p (self-loop, so p stays = 1), t_lazy consumes-and-produces p
    /// (self-loop too, no observable progress). Both transitions are
    /// always enabled, so each AG(EF(enabled(t))) is trivially True.
    fn lazy_transition_net() -> PetriNet {
        PetriNet {
            name: Some("lazy-transition".to_string()),
            places: vec![PlaceInfo {
                id: "p".to_string(),
                name: None,
            }],
            transitions: vec![
                TransitionInfo {
                    id: "t_cycle".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t_lazy".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1],
        }
    }

    /// Single transition deadlocking net: p starts empty, t needs p
    /// ≥ 1. Initial marking is itself a deadlock. The single
    /// transition is dead → not L4-live.
    fn immediate_dead_net() -> PetriNet {
        PetriNet {
            name: Some("immediate-dead".to_string()),
            places: vec![PlaceInfo {
                id: "p".to_string(),
                name: None,
            }],
            transitions: vec![TransitionInfo {
                id: "t".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![],
            }],
            initial_marking: vec![0],
        }
    }

    /// Compute the SCC-based liveness verdict directly. Used as the
    /// independent oracle for differential tests.
    fn scc_liveness_verdict(net: &PetriNet) -> Verdict {
        let cfg = config();
        let plan = ExecutionPlan::graph();
        let graph = plan.run_graph(net, &cfg);
        if !graph.completed {
            return Verdict::CannotCompute;
        }
        let sccs = tarjan_scc(&graph);
        let bottoms = bottom_sccs(&graph.adj, &sccs);
        let live = check_liveness(&graph, &sccs, &bottoms, net.num_transitions());
        if live {
            Verdict::True
        } else {
            Verdict::False
        }
    }

    #[test]
    fn test_liveness_simple_live_net() {
        let net = alternator_net();
        let verdict = liveness_via_mu_calculus(&net, &config(), &[]);
        assert_eq!(
            verdict,
            Verdict::True,
            "every transition of the 2-state alternator is L4-live"
        );
    }

    #[test]
    fn test_liveness_dead_transition_returns_false() {
        let net = dead_transition_net();
        let verdict = liveness_via_mu_calculus(&net, &config(), &[]);
        assert_eq!(
            verdict,
            Verdict::False,
            "structurally dead transition disproves L4-liveness"
        );
    }

    #[test]
    fn test_liveness_lazy_transition_is_live() {
        let net = lazy_transition_net();
        let verdict = liveness_via_mu_calculus(&net, &config(), &[]);
        assert_eq!(
            verdict,
            Verdict::True,
            "lazy transitions that stay perpetually enabled satisfy AG EF enabled"
        );
    }

    #[test]
    fn test_liveness_immediate_dead_net() {
        let net = immediate_dead_net();
        let verdict = liveness_via_mu_calculus(&net, &config(), &[]);
        assert_eq!(
            verdict,
            Verdict::False,
            "transition unfireable at the initial-and-only state is not live"
        );
    }

    #[test]
    fn test_liveness_empty_transition_set_is_vacuously_live() {
        // A net with zero transitions is vacuously live (universal
        // quantification over an empty set).
        let net = PetriNet {
            name: Some("empty".to_string()),
            places: vec![PlaceInfo {
                id: "p".to_string(),
                name: None,
            }],
            transitions: vec![],
            initial_marking: vec![0],
        };
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &[]),
            Verdict::True
        );
    }

    /// Differential test: the mu-calculus path and the SCC path must
    /// agree on every net we have an SCC verdict for. This is the
    /// soundness pillar — disagreement indicates either a translation
    /// bug or an SCC bug.
    #[test]
    fn test_liveness_mu_matches_scc_on_alternator() {
        let net = alternator_net();
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &[]),
            scc_liveness_verdict(&net),
            "mu-calculus and SCC verdicts must agree on alternator_net"
        );
    }

    #[test]
    fn test_liveness_mu_matches_scc_on_dead_transition() {
        let net = dead_transition_net();
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &[]),
            scc_liveness_verdict(&net),
            "mu-calculus and SCC verdicts must agree on dead_transition_net"
        );
    }

    #[test]
    fn test_liveness_mu_matches_scc_on_lazy_transition() {
        let net = lazy_transition_net();
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &[]),
            scc_liveness_verdict(&net),
            "mu-calculus and SCC verdicts must agree on lazy_transition_net"
        );
    }

    #[test]
    fn test_liveness_mu_matches_scc_on_immediate_dead() {
        let net = immediate_dead_net();
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &[]),
            scc_liveness_verdict(&net),
            "mu-calculus and SCC verdicts must agree on immediate_dead_net"
        );
    }

    /// Regression test for the 13-exam wrong-answer pair
    /// (`GlobalResAllocation-COL-03`, `TokenRing-COL-005`):
    /// a colored transition with one dead binding and one live binding
    /// must be classified L4-live (because at least one binding is
    /// live), NOT non-live. Without the colored grouping fix, the
    /// per-binding mu-calculus iteration sees the dead binding and
    /// short-circuits to FALSE — exactly the bug this commit fixes.
    ///
    /// Net structure (mirrors a colored transition unfolded into two
    /// bindings: t0 = live, t1 = dead-by-construction):
    ///   p_live = 1; p_dead = 0
    ///   t0: p_live --[1]--> p_live   (self-loop, always live)
    ///   t1: p_dead --[1]--> p_dead   (self-loop on a place that is
    ///                                  never marked → dead binding)
    ///
    /// Under P/T semantics (no group), t1 is dead → False (correct).
    /// Under colored semantics with group = [t0, t1], the colored
    /// transition has at least one live binding (t0) → True.
    #[test]
    fn test_liveness_colored_group_one_dead_one_live_binding_is_live() {
        let net = PetriNet {
            name: Some("colored-mixed-bindings".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p_live".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p_dead".to_string(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0_live".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t1_dead".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        };

        // No grouping → per-binding semantics → t1 is dead → False.
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &[]),
            Verdict::False,
            "P/T (no grouping): a dead binding disproves liveness"
        );

        // With grouping [t0, t1] → colored semantics → colored
        // transition has at least one live binding (t0) → True.
        let groups: Vec<Vec<usize>> = vec![vec![0, 1]];
        assert_eq!(
            liveness_via_mu_calculus(&net, &config(), &groups),
            Verdict::True,
            "colored (grouped): a colored transition with at least one \
             live binding is L4-live"
        );

        // End-to-end: the dispatcher must also return True for the
        // colored input, exercising both the structural-shortcut
        // guards and the mu-calculus path.
        assert_eq!(
            liveness_verdict_with_groups(&net, &config(), &groups),
            Verdict::True,
            "liveness_verdict_with_groups must respect colored grouping \
             end-to-end (the bug that caused the 13-exam wrong answers)"
        );
    }

    /// Sister regression for the LP-dead-transition shortcut path.
    /// `GlobalResAllocation-COL-03` was wrong because the per-binding
    /// `lp_dead_transition` shortcut fired BEFORE the mu-calculus path.
    /// Net: one structurally LP-dead binding (input place never
    /// produced) AND one always-live binding, grouped as one colored
    /// transition. Pre-fix this returned False from the LP shortcut;
    /// post-fix returns True via the mu-calculus colored path.
    #[test]
    fn test_liveness_colored_lp_dead_binding_shortcut_gated() {
        let net = PetriNet {
            name: Some("colored-lp-dead-mixed".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p_live".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p_unreachable".to_string(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0_live".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t1_lp_dead".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![],
                },
            ],
            initial_marking: vec![1, 0],
        };

        // With grouping [t0, t1] the LP shortcut must be SKIPPED
        // (because the dead binding does not disprove the colored
        // transition) and the mu-calculus path must conclude True.
        let groups: Vec<Vec<usize>> = vec![vec![0, 1]];
        assert_eq!(
            liveness_verdict_with_groups(&net, &config(), &groups),
            Verdict::True,
            "the LP-dead-transition shortcut must be gated by \
             colored grouping (GlobalResAllocation-COL-03 root cause)"
        );
    }

    /// Soundness regression: place-swap canonicalization must NOT be applied
    /// to QuasiLiveness exploration.
    ///
    /// QuasiLiveness asks, per transition `t`, whether `enabled(t)` holds at
    /// some reachable marking. `enabled(t)` for a *specific* transition is
    /// NOT σ-invariant under place-swap symmetry: a permutation σ in the
    /// place orbit maps transition `t` to a *different* transition σ(t), so a
    /// marking that enables `t` is orbit-equivalent to one that enables σ(t),
    /// not `t` itself. The canonicalizing BFS only ever visits canonical
    /// orbit representatives and records the transition index enabled *there*
    /// (`successors.rs` records the raw fired index but iterates only the
    /// transitions enabled at the canonical marking). A transition enabled
    /// solely at a non-canonical orbit member is therefore never recorded,
    /// producing a false-negative ("not quasi-live") verdict — a wrong MCC
    /// answer worth −16 points.
    ///
    /// Counterexample net (`p0`, `p1` are a symmetric place orbit fed by
    /// source `s`):
    ///   m0 = [p0=0, p1=0, s=1]
    ///   t_load0: s → p0      t_load1: s → p1
    ///   t_use0:  p0 →        t_use1:  p1 →
    /// Every transition fires on the real reachability graph
    /// ([0,0,1]→[1,0,0]→[0,0,0] fires t_load0,t_use0; the p1 branch fires
    /// t_load1,t_use1), so the net IS quasi-live. With ascending orbit
    /// canonicalization (`[1,0,_]` and `[0,1,_]` both collapse to `[0,1,_]`),
    /// the explorer only ever visits the `p1`-high representative, so
    /// `t_use0` is never observed firing and the verdict flips to False.
    ///
    /// This test runs the real canonicalizing observer path under
    /// `Examination::QuasiLiveness`. Post-fix (`canonicalization_is_sound`
    /// returns false for QuasiLiveness) canon is disabled regardless of
    /// `TY_AUTO_SYMMETRY`, so every transition is observed firing.
    fn quasi_liveness_orbit_counterexample_net() -> PetriNet {
        PetriNet {
            name: Some("quasi-live-orbit-counterexample".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p0".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "s".to_string(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t_load0".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t_load1".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t_use0".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![],
                },
                TransitionInfo {
                    id: "t_use1".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![],
                },
            ],
            initial_marking: vec![0, 0, 1],
        }
    }

    #[test]
    fn test_quasi_liveness_canonicalization_records_all_orbit_transitions() {
        use crate::examination::Examination;
        use crate::explorer::explore_observer;
        use crate::explorer::symmetry::PetriCanonicalizer;

        let net = quasi_liveness_orbit_counterexample_net();

        // Sanity: the {p0, p1} symmetry must actually be discovered, otherwise
        // the test would vacuously pass without exercising the canon path.
        assert!(
            !PetriCanonicalizer::build(&net).is_empty(),
            "expected the p0<->p1 place orbit to be discovered",
        );

        let config = config().with_examination(Some(Examination::QuasiLiveness));
        let mut observer = QuasiLivenessObserver::new(net.transitions.len());
        // The observer stops exploration early once every transition has been
        // observed firing (`is_done()`), so `result.completed` is expectedly
        // false on success — `all_fired()` is the meaningful check.
        let _result = explore_observer(&net, &config, &mut observer);

        assert!(
            observer.all_fired(),
            "every transition must be observed firing; place-swap \
             canonicalization is unsound for QuasiLiveness because \
             enabled(t) is not σ-invariant (missed transitions: {:?})",
            net.transitions
                .iter()
                .zip(observer.fired_slice())
                .filter(|(_, &fired)| !fired)
                .map(|(t, _)| t.id.as_str())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_quasi_liveness_orbit_counterexample_verdict_is_true() {
        // End-to-end verdict on the same net. The net IS quasi-live, so the
        // verdict must be True. (Pre-fix, if BMC failed to resolve every
        // transition and the canonicalizing observer ran, this returned
        // False — a wrong answer.)
        let net = quasi_liveness_orbit_counterexample_net();
        assert_eq!(
            quasi_liveness_verdict(&net, &config()),
            Verdict::True,
            "the counterexample net is quasi-live",
        );
    }

    /// Net that is L4-live for its single live transition EXCEPT for two
    /// structurally-dead transitions hidden inside a permanently-empty trap.
    ///
    /// Structure:
    ///   - `pLive` (init 1) ──► `tLive` ──► `pLive`     (self-loop, always live)
    ///   - `pA` (init 0) ──► `tAB` ──► `pB`             (token-conserving cycle …
    ///   - `pB` (init 0) ──► `tBA` ──► `pA`             … over the empty trap {pA,pB})
    ///
    /// `{pA, pB}` is a zero-initialized **trap**: every producer of a trap
    /// place is also a consumer inside the trap, so the trap can never
    /// receive a token from outside and stays empty forever. Hence `tAB`
    /// and `tBA` are NEVER enabled in any reachable marking — the net is
    /// NOT L4-live and the correct Liveness verdict is `False`.
    ///
    /// The net is intentionally non-free-choice (so `structural_live`
    /// declines) and chosen so the *downstream* liveness check on the
    /// dead-transition-reduced net wrongly reports `True` (only `tLive`
    /// survives). On large nets the pre-reduction `lp_dead_transition`
    /// shortcut fails open (finding #12), exposing the unsound `True`.
    fn live_except_dead_trap_net() -> PetriNet {
        PetriNet {
            name: Some("live-except-dead-trap".to_string()),
            places: vec![
                PlaceInfo {
                    id: "pA".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "pB".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "pLive".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "tAB".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "tBA".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "tLive".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![0, 0, 1],
        }
    }

    /// Soundness regression for the dead-transition-removal liveness gap.
    ///
    /// The `StutterSensitiveLTL` reduction physically DELETES the two
    /// structurally-dead transitions (`tAB`, `tBA`) and reports them in
    /// `report.dead_transitions`. The downstream liveness decision then
    /// runs only over the surviving `tLive`, which IS live — so without the
    /// gate the post-reduction path returns a WRONG definite `True`. This
    /// test pins:
    ///   1. the reduction really removes the dead transitions (the trigger),
    ///   2. the downstream verdict on the reduced net is the WRONG `True`
    ///      (what the OLD code emitted whenever the pre-reduction LP
    ///      shortcut failed open — finding #12 on large nets), and
    ///   3. the FIXED `liveness_verdict` emits the CORRECT `False`.
    #[test]
    fn test_liveness_dead_transition_removal_forces_false() {
        let net = live_except_dead_trap_net();

        // (1) The reduction removes the structurally-dead transitions and
        // records them in its own (uncapped, structural) report.
        let reduced = crate::reduction::reduce_iterative_structural_with_mode(
            &net,
            &[],
            crate::reduction::ReductionMode::StutterSensitiveLTL,
            None,
        )
        .expect("StutterSensitiveLTL reduction must succeed");
        assert!(
            !reduced.report.dead_transitions.is_empty(),
            "reduction must remove the trap-dead transitions tAB/tBA \
             (got dead_transitions = {:?})",
            reduced.report.dead_transitions
        );
        assert!(
            reduced.net.num_transitions() < net.num_transitions(),
            "the dead transitions must be physically removed from the reduced net"
        );

        // (2) OLD-CODE WITNESS: the downstream liveness check, run only over
        // the surviving transition(s) of the reduced net, reports the WRONG
        // `True` — every survivor (tLive) is live. This is exactly the
        // verdict the pre-fix code emitted whenever the pre-reduction LP
        // shortcut declined (large nets, finding #12).
        assert_eq!(
            scc_liveness_verdict(&reduced.net),
            Verdict::True,
            "downstream SCC liveness over survivors alone is the wrong True \
             that motivates the gate"
        );
        assert_eq!(
            liveness_via_mu_calculus(&reduced.net, &config(), &[]),
            Verdict::True,
            "downstream mu-calculus over survivors alone is the wrong True"
        );

        // (3) NEW-CODE: the full verdict is the CORRECT `False` — the
        // original net is not L4-live because tAB/tBA can never fire.
        assert_eq!(
            liveness_verdict(&net, &config()),
            Verdict::False,
            "a structurally-dead transition makes the net NOT L4-live; the \
             dead-transition-removal gate must force the correct False"
        );
    }

    /// The dead-transition gate must NOT fire on a genuinely-live net: when
    /// the reduction removes nothing as dead, the verdict path stays exactly
    /// as before and a live net still yields `True`.
    #[test]
    fn test_liveness_no_dead_removal_keeps_live_true() {
        let net = alternator_net();
        let reduced = crate::reduction::reduce_iterative_structural_with_mode(
            &net,
            &[],
            crate::reduction::ReductionMode::StutterSensitiveLTL,
            None,
        )
        .expect("reduction must succeed");
        assert!(
            reduced.report.dead_transitions.is_empty(),
            "the live alternator has no dead transitions to remove"
        );
        assert_eq!(
            liveness_verdict(&net, &config()),
            Verdict::True,
            "a genuinely live net must stay True — the gate must not fire"
        );
    }

    // ----------------------------------------------------------------------
    // Strong per-transition dead-transition LP (QuasiLiveness FALSE shortcut)
    // ----------------------------------------------------------------------

    /// Exhaustive BFS oracle for quasi-liveness. Builds the full reachable
    /// state space and observes every fired transition.
    ///
    /// - `True`  — every transition fired at least once (net is quasi-live).
    /// - `False` — exploration COMPLETED and some transition never fired.
    /// - `CannotCompute` — exploration was truncated (does not apply to the
    ///   tiny nets used here).
    ///
    /// This is the independent ground truth the structural LP shortcut must
    /// never contradict.
    fn exhaustive_quasi_liveness_verdict(net: &PetriNet) -> Verdict {
        use crate::explorer::explore_observer;
        let cfg = config();
        let mut observer = QuasiLivenessObserver::new(net.transitions.len());
        let result = explore_observer(net, &cfg, &mut observer);
        if observer.all_fired() {
            Verdict::True
        } else if result.completed {
            Verdict::False
        } else {
            Verdict::CannotCompute
        }
    }

    /// Net whose dead transition is invisible to the WEAK per-place LP but
    /// caught by the strong JOINT enabling LP.
    ///
    /// `p1 + p2 = 1` is a P-invariant: a single token alternates between the
    /// two places via `ta: p1 -> p2` and `tb: p2 -> p1`. `t_join` requires
    /// BOTH `p1 >= 1` and `p2 >= 1` simultaneously — impossible, since the two
    /// places can never be marked at the same time. So `t_join` is structurally
    /// dead and the net is NOT quasi-live.
    ///
    /// The weak `lp_dead_transition` maximises each input place in isolation:
    /// `max M[p1] = 1 >= 1` and `max M[p2] = 1 >= 1`, so it sees NEITHER place
    /// as individually starved and declines. Only the joint conjunction
    /// `p1 >= 1 AND p2 >= 1` (which forces `p1 + p2 >= 2`, contradicting the
    /// state-equation bound `p1 + p2 <= 1`) is infeasible. This is exactly the
    /// gap the strong test closes.
    fn jointly_dead_transition_net() -> PetriNet {
        PetriNet {
            name: Some("jointly-dead".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p1".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p2".to_string(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "ta".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "tb".to_string(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t_join".to_string(),
                    name: None,
                    inputs: vec![
                        Arc {
                            place: PlaceIdx(0),
                            weight: 1,
                        },
                        Arc {
                            place: PlaceIdx(1),
                            weight: 1,
                        },
                    ],
                    outputs: vec![],
                },
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// The strong joint+trap LP must prove `t_join` dead where the weak
    /// per-place LP cannot. This is the discriminating witness for the new
    /// shortcut.
    #[test]
    fn test_lp_first_dead_transition_catches_joint_infeasibility() {
        let net = jointly_dead_transition_net();

        // The WEAK per-place test must DECLINE (its blind spot).
        assert_eq!(
            crate::structural::lp_dead_transition(&net, None),
            None,
            "the weak per-place LP cannot see the joint p1&p2 infeasibility"
        );

        // The STRONG joint+trap test must identify t_join (index 2) as dead.
        assert_eq!(
            crate::lp_state_equation::lp_first_dead_transition(&net, None),
            Some(TransitionIdx(2)),
            "the strong joint LP must prove t_join never enabled",
        );
    }

    /// Strong test still subsumes the weak test: a place that is never
    /// produced (the classic structural dead transition) is also caught.
    #[test]
    fn test_lp_first_dead_transition_subsumes_weak() {
        let net = dead_transition_net();
        // t_dead (index 1) needs p2 >= 1, but p2 is never produced.
        assert_eq!(
            crate::lp_state_equation::lp_first_dead_transition(&net, None),
            Some(TransitionIdx(1)),
            "the strong LP must also catch every weakly-dead transition",
        );
    }

    /// SOUNDNESS: the strong sweep must NEVER flag a transition on a genuinely
    /// quasi-live (firing) net — that would be a false FALSE. Checked across
    /// every firing fixture in this module.
    #[test]
    fn test_lp_first_dead_transition_none_on_firing_nets() {
        for net in [
            alternator_net(),
            lazy_transition_net(),
            quasi_liveness_orbit_counterexample_net(),
        ] {
            assert_eq!(
                crate::lp_state_equation::lp_first_dead_transition(&net, None),
                None,
                "no transition of a quasi-live net may be proven dead (net {:?})",
                net.name,
            );
        }
    }

    /// Cross-check: the full structural verdict must agree with the exhaustive
    /// BFS oracle on the jointly-dead net (both FALSE) and on every firing net
    /// (both TRUE). Disagreement would mean the LP shortcut emitted a wrong
    /// answer — the catastrophic case the task guards against.
    #[test]
    fn test_quasi_liveness_structural_matches_exhaustive() {
        // FALSE case: the strong LP decides it (BFS would too).
        let dead = jointly_dead_transition_net();
        let structural = quasi_liveness_verdict(&dead, &config());
        let exhaustive = exhaustive_quasi_liveness_verdict(&dead);
        assert_eq!(
            structural,
            Verdict::False,
            "jointly-dead net is not quasi-live",
        );
        assert_eq!(
            structural, exhaustive,
            "structural and exhaustive verdicts must agree on the jointly-dead net",
        );

        // TRUE cases: the LP must NOT short-circuit; BFS/seeding decides TRUE.
        for net in [
            alternator_net(),
            lazy_transition_net(),
            quasi_liveness_orbit_counterexample_net(),
        ] {
            let structural = quasi_liveness_verdict(&net, &config());
            let exhaustive = exhaustive_quasi_liveness_verdict(&net);
            assert_eq!(
                structural,
                Verdict::True,
                "firing net {:?} is quasi-live",
                net.name,
            );
            assert_eq!(
                structural, exhaustive,
                "structural and exhaustive verdicts must agree on firing net {:?}",
                net.name,
            );
        }
    }

    /// The wall-cap must never turn a determinate sweep into a wrong verdict:
    /// an already-expired soft deadline makes the sweep decline (`None`), so
    /// the caller falls through to the exact engine rather than guessing.
    #[test]
    fn test_lp_first_dead_transition_expired_deadline_declines() {
        let net = jointly_dead_transition_net();
        // Intentional past Instant; checked_sub would change the type to Option<Instant>.
        #[allow(clippy::unchecked_time_subtraction)]
        let past = Instant::now() - Duration::from_secs(1);
        assert_eq!(
            crate::lp_state_equation::lp_first_dead_transition(&net, Some(past)),
            None,
            "an expired deadline must yield an inconclusive (None) sweep, never FALSE",
        );
    }
}
