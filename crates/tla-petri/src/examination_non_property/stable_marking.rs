// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::super::examination_plan::ExecutionPlan;
use super::common::checkpoint_cannot_compute;
use crate::examinations::global_properties_bmc;
use crate::examinations::global_properties_pdr;
use crate::examinations::query_support::visible_transitions_for_support;
use crate::examinations::stable_marking::StableMarkingObserver;
use crate::explorer::{
    explore_observer, CheckpointableObserver, ExplorationConfig, ExplorationObserver,
    ParallelExplorationObserver, ParallelExplorationSummary,
};
use crate::output::Verdict;
use crate::petri_net::{PetriNet, QuerySupport, TransitionIdx};
use crate::reduction::{analyze, ReducedNet};
use crate::stubborn::PorStrategy;
use serde::{Deserialize, Serialize};

/// Wall-clock cap for the structural stable-place prover
/// (`invariant::structural_stable_place`).
///
/// The prover runs `compute_p_invariants` (Farkas elimination), which — like
/// the OneSafe structural short-circuit (see `ONE_SAFE_STRUCTURAL_PHASE_CAP`)
/// — does NOT poll a deadline and can spin several seconds on very high-arity
/// nets. Bounding it on a worker thread keeps the rest of the StableMarking
/// pipeline (LP pinning + BFS) reachable. Strictly verdict-preserving: the
/// prover only ever yields a TRUE witness, so abandon/panic/`None` all fall
/// through exactly like an inconclusive result.
const STABLE_STRUCTURAL_PHASE_CAP: std::time::Duration = std::time::Duration::from_secs(8);

/// Run [`crate::invariant::structural_stable_place`] under a wall-clock cap.
///
/// With a finite `deadline`, the prover runs on a detached worker thread and is
/// abandoned if it exceeds `min(remaining, STABLE_STRUCTURAL_PHASE_CAP)` (the
/// thread leak is acceptable — the ty-mcc process exits at the global deadline,
/// mirroring the deadlock-phase wall caps). A panic in the Farkas core (latent
/// i128 overflow on extreme nets) is caught and treated as "no witness". With
/// no deadline (public-API path), the prover runs inline with a panic guard.
fn structural_stable_place_capped(
    net: &PetriNet,
    deadline: Option<std::time::Instant>,
) -> Option<usize> {
    let run_inline = || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::invariant::structural_stable_place(net)
        }))
        .ok()
        .flatten()
    };

    let Some(deadline) = deadline else {
        return run_inline();
    };
    let cap = deadline
        .saturating_duration_since(std::time::Instant::now())
        .min(STABLE_STRUCTURAL_PHASE_CAP);
    if cap.is_zero() {
        return None;
    }

    let net = net.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Option<usize>>(1);
    let spawned = std::thread::Builder::new()
        .name("ty-stable-structural".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::invariant::structural_stable_place(&net)
            }))
            .ok()
            .flatten();
            let _ = tx.send(result);
        });
    if spawned.is_err() {
        return None;
    }
    // Ok(witness) yields the witness; timeout (abandon the worker) or worker
    // panic/disconnect yields `None` (the `Option` default).
    rx.recv_timeout(cap).unwrap_or_default()
}

/// StableMarking PDR is fail-closed: `global_properties_pdr::run_stable_marking_pdr`
/// only returns a definite verdict when its IC3 invariant proof completes, so a
/// timeout or unknown leaves the BFS fallback in charge. Defaulting on lets
/// IC3 resolve identity-net properties before the explicit-state explorer
/// burns its time budget. Set `TY_MCC_ENABLE_STABLE_PDR=0` (or `false`, `off`)
/// to force-disable, e.g. when bisecting an unsoundness regression like the
/// historical Sudoku-PT-BN01 incident.
fn stable_pdr_enabled() -> bool {
    match std::env::var("TY_MCC_ENABLE_STABLE_PDR")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
    {
        Some(value) => !matches!(value.as_str(), "0" | "false" | "no" | "off"),
        None => true,
    }
}

/// Observer for StableMarking on colored models.
///
/// Checks group-level stability: a colored place group is "stable" if the
/// sum of tokens across all unfolded instances stays constant.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColoredStableMarkingObserver {
    groups: Vec<Vec<usize>>,
    initial_sums: Vec<u64>,
    unstable: Vec<bool>,
    unstable_count: usize,
}

impl ExplorationObserver for ColoredStableMarkingObserver {
    fn on_new_state(&mut self, marking: &[u64]) -> bool {
        for (gi, group) in self.groups.iter().enumerate() {
            if !self.unstable[gi] {
                let sum: u64 = group.iter().map(|&p| marking[p]).sum();
                if sum != self.initial_sums[gi] {
                    self.unstable[gi] = true;
                    self.unstable_count += 1;
                    if self.unstable_count == self.groups.len() {
                        return false; // all groups unstable
                    }
                }
            }
        }
        true
    }

    fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
        true
    }

    fn on_deadlock(&mut self, _marking: &[u64]) {}

    fn is_done(&self) -> bool {
        self.unstable_count == self.groups.len()
    }

    fn canonicalization_safe(&self) -> bool {
        // Like the colored OneSafe observer, this sums token counts over a fixed
        // set of unfolded place indices per colored place. Place-swap
        // canonicalization permutes indices within orbits and corrupts those
        // group sums when an orbit crosses colored-group boundaries. StableMarking
        // is already refused at the examination level (`StableMarking => false`),
        // but we override here too as defense-in-depth so a future guard change
        // cannot silently reintroduce the wrong-verdict bug.
        false
    }
}

struct ColoredStableMarkingSummary {
    groups: Vec<Vec<usize>>,
    initial_sums: Vec<u64>,
    unstable: Vec<bool>,
}

impl ParallelExplorationSummary for ColoredStableMarkingSummary {
    fn on_new_state(&mut self, marking: &[u64]) {
        for (gi, group) in self.groups.iter().enumerate() {
            if !self.unstable[gi] {
                let sum: u64 = group.iter().map(|&p| marking[p]).sum();
                if sum != self.initial_sums[gi] {
                    self.unstable[gi] = true;
                }
            }
        }
    }

    fn on_transition_fire(&mut self, _trans: TransitionIdx) {}
    fn on_deadlock(&mut self, _marking: &[u64]) {}

    fn stop_requested(&self) -> bool {
        self.unstable.iter().all(|&u| u)
    }
}

impl ParallelExplorationObserver for ColoredStableMarkingObserver {
    type Summary = ColoredStableMarkingSummary;

    fn new_summary(&self) -> Self::Summary {
        ColoredStableMarkingSummary {
            groups: self.groups.clone(),
            initial_sums: self.initial_sums.clone(),
            unstable: vec![false; self.groups.len()],
        }
    }

    fn merge_summary(&mut self, summary: Self::Summary) {
        for (gi, unstable) in summary.unstable.into_iter().enumerate() {
            if unstable && !self.unstable[gi] {
                self.unstable[gi] = true;
                self.unstable_count += 1;
            }
        }
    }
}

impl CheckpointableObserver for ColoredStableMarkingObserver {
    type Snapshot = Self;

    const CHECKPOINT_KIND: &'static str = "ColoredStableMarkingObserver";

    fn snapshot(&self) -> Self::Snapshot {
        self.clone()
    }

    fn restore_from_snapshot(&mut self, snapshot: Self::Snapshot) {
        *self = snapshot;
    }
}

/// Wall-clock cap for the random-walk StableMarking FALSE-witness lane.
///
/// Mirrors `DEADLOCK_WALK_PHASE_CAP` in the deadlock pipeline: the lane is a
/// FALSE-only under-approximation that runs AFTER the BMC/PDR seeding and BEFORE
/// the explicit BFS observer. On flat, no-symmetry P/T nets where BFS explores
/// only a shallow depth before the deadline (PolyORBLF-PT-S04J06T06,
/// DLCround-PT-07b), a deep random walk perturbs far more places and can prove
/// every place non-constant.
const STABLE_WALK_PHASE_CAP: std::time::Duration = std::time::Duration::from_secs(8);

/// Budget reserved for the explicit BFS observer; the walk lane never eats into
/// this tail. Mirrors `DEADLOCK_BFS_FALLBACK_RESERVE`.
const STABLE_WALK_BFS_FALLBACK_RESERVE: std::time::Duration = std::time::Duration::from_secs(10);

/// Fraction of the leftover (post-reserve) budget the walk lane may consume.
const STABLE_WALK_DEADLINE_FRACTION: u32 = 4;

/// Compute the random-walk StableMarking-witness phase deadline.
///
/// Additive / leftover-only, mirroring `deadlock_walk_deadline_at`:
/// - `None` when there is no global deadline (the walk runs to its own internal
///   walk/step budget without external interference).
/// - `Some(now)` (already expired, i.e. "skip") when the remaining budget is at
///   or below [`STABLE_WALK_BFS_FALLBACK_RESERVE`] — the full remaining budget
///   is reserved for the BFS observer.
/// - `Some(phase_deadline)` capped at `min(STABLE_WALK_PHASE_CAP, (remaining −
///   STABLE_WALK_BFS_FALLBACK_RESERVE) / STABLE_WALK_DEADLINE_FRACTION)`
///   otherwise. Subtracting the reserve BEFORE the fraction guarantees the walk
///   can never eat into the BFS tail.
fn stable_walk_deadline_at(
    global_deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> Option<std::time::Instant> {
    let global_deadline = global_deadline?;
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= STABLE_WALK_BFS_FALLBACK_RESERVE {
        return Some(now);
    }
    let leftover = remaining
        .checked_sub(STABLE_WALK_BFS_FALLBACK_RESERVE)
        .unwrap();
    let phase_budget = STABLE_WALK_PHASE_CAP.min(leftover / STABLE_WALK_DEADLINE_FRACTION);
    Some(now + phase_budget)
}

fn stable_walk_deadline(global_deadline: Option<std::time::Instant>) -> Option<std::time::Instant> {
    stable_walk_deadline_at(global_deadline, std::time::Instant::now())
}

/// `true` iff the walk phase deadline says "skip" (already expired).
fn stable_walk_skip(phase_deadline: Option<std::time::Instant>) -> bool {
    phase_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
}

pub(crate) fn stable_marking_verdict(
    net: &PetriNet,
    config: &ExplorationConfig,
    colored_groups: &[Vec<usize>],
) -> Verdict {
    // Check for structurally stable places on the ORIGINAL (pre-reduction)
    // net. A place is truly stable only if every transition in the original
    // net has zero net effect on it. We must NOT use reduced.report here
    // because iterative reduction (agglomerations) can remove transitions
    // that affect a place, making it appear constant in the reduced net
    // while it is NOT constant in the original net.
    let pre_reduction = analyze(net);
    if !colored_groups.is_empty() {
        // Colored: a colored place group is structurally stable if every
        // transition has zero net effect on the group sum.
        let has_stable_group = colored_groups.iter().any(|group| {
            net.transitions.iter().all(|t| {
                let flow_in: i64 = t
                    .inputs
                    .iter()
                    .filter(|a| group.contains(&(a.place.0 as usize)))
                    .map(|a| a.weight as i64)
                    .sum();
                let flow_out: i64 = t
                    .outputs
                    .iter()
                    .filter(|a| group.contains(&(a.place.0 as usize)))
                    .map(|a| a.weight as i64)
                    .sum();
                flow_in == flow_out
            })
        });
        if has_stable_group {
            return Verdict::True;
        }
    } else {
        let has_structurally_stable =
            !pre_reduction.constant_places.is_empty() || !pre_reduction.isolated_places.is_empty();
        if has_structurally_stable {
            return Verdict::True;
        }
    }

    // For colored models, skip reduction and BFS on original net with
    // group-level stability check (group-level token accounting through
    // reduction is complex and potentially unsound).
    if !colored_groups.is_empty() {
        let initial_group_sums: Vec<u64> = colored_groups
            .iter()
            .map(|g| g.iter().map(|&p| net.initial_marking[p]).sum())
            .collect();
        let n_groups = colored_groups.len();
        let mut observer = ColoredStableMarkingObserver {
            groups: colored_groups.to_vec(),
            initial_sums: initial_group_sums,
            unstable: vec![false; n_groups],
            unstable_count: 0,
        };
        let result = if config.checkpoint().is_some() {
            match crate::explorer::explore_checkpointable_observer(net, config, &mut observer) {
                Ok(result) => result,
                Err(error) => return checkpoint_cannot_compute("StableMarking", &error),
            }
        } else {
            explore_observer(net, config, &mut observer)
        };
        return if observer.unstable_count == colored_groups.len() {
            Verdict::False
        } else if result.completed {
            Verdict::True
        } else {
            Verdict::CannotCompute
        };
    }

    // P-invariant constancy pinning: a sound, UNCAPPED structural TRUE shortcut.
    // `p_invariant_constant_place` proves — via semi-positive P-invariants and
    // their structural place bounds, with no LP solve, no size gate and no time
    // slice — that some place is constant (= its initial marking) in every
    // reachable marking, which is exactly a stably-marked place ⇒ StableMarking
    // TRUE. It generalises the per-transition zero-incidence-row test above to
    // *coupled* places pinned by a multi-place invariant (the FlexibleBarrier
    // class the gated 2-second `lp_pinned_place` sweep below times out on). It
    // can only ever certify constancy; an inconclusive net returns `None` and we
    // fall through to the LP / BMC / PDR / BFS engines unchanged, so no existing
    // verdict is affected. Runs on the ORIGINAL net for the same soundness reason
    // as the structural test (reduction can make a non-constant place look
    // constant).
    if let Some(place) = structural_stable_place_capped(net, config.deadline()) {
        eprintln!(
            "StableMarking: place {place} structurally pinned to its initial marking \
             (constant in every reachable marking) → stable"
        );
        return Verdict::True;
    }

    // LP state-equation pinning: a sound structural TRUE shortcut that extends
    // the per-transition zero-incidence-row constant-place test above. A place
    // pinned to its initial marking across the state-equation polytope (which
    // over-approximates the reachable set and is tightened with initially-marked
    // traps) is constant in every reachable marking, i.e. a stable place, so
    // StableMarking is TRUE. This catches reachability-constant-but-coupled
    // places (pinned by a P-invariant, or by a transition the state equation
    // forces never to fire) that the structural zero-row test misses. It can
    // only ever yield TRUE; an inconclusive relaxation falls through to the BMC
    // / PDR / BFS engines unchanged, so no existing verdict is affected. Runs on
    // the ORIGINAL net for the same soundness reason as the structural test.
    //
    // Bound the sweep to a short slice of the budget so it can never starve the
    // BFS fallback: it is a best-effort shortcut that stops at whichever comes
    // first (the examination deadline or the slice) and falls through if no pin
    // is found in time.
    let pin_deadline = {
        let slice = std::time::Instant::now() + std::time::Duration::from_secs(2);
        Some(config.deadline().map_or(slice, |limit| limit.min(slice)))
    };
    if crate::lp_state_equation::lp_pinned_place(net, pin_deadline).is_some() {
        return Verdict::True;
    }

    // INTEGER state-equation pinning: the integer-tightened dual of the rational
    // LP pin above. Where the LP RELAXATION admits a fractional firing vector
    // that perturbs a place (so `lp_pinned_place` declines), the INTEGER program
    // (state equation + trap + siphon cuts) may prove BOTH `M[p] ≥ M0[p]+1` and
    // `M[p] ≤ M0[p]-1` integer-infeasible — a genuine constancy proof that the
    // relaxation gap hid. A pinned place is constant in every reachable marking,
    // i.e. stably marked, so StableMarking is TRUE. Strictly ADDITIVE: it returns
    // a place only on a real integer-infeasibility proof and DECLINES (falls
    // through unchanged) on inconclusive / oversized / overflow, so it can never
    // change an existing verdict. Runs on the ORIGINAL net (reduction can make a
    // non-constant place look constant) and is bounded by the same `pin_deadline`
    // slice plus a short per-place solver timeout, so it cannot starve the BFS.
    if crate::symbolic::int_state_equation::integer_pinned_place(
        net,
        std::time::Duration::from_millis(400),
        pin_deadline,
    )
    .is_some()
    {
        eprintln!(
            "StableMarking: place integer-state-equation pinned to its initial marking \
             (constant in every reachable marking) → stable"
        );
        return Verdict::True;
    }

    // Decision-Diagram exact fast-path (off by default — gated by
    // `dd-backend`). Placed AFTER the cheap structural/LP TRUE shortcuts
    // (zero-incidence / P-invariant / LP-pinned) and BEFORE the BMC / PDR /
    // BFS engines. This is the additive lane the SM diagnosis identified: on
    // the DD-eligible (bounded, structured) nets where every structural lane
    // declines and BFS cannot enumerate the reachable set in budget (the
    // airplane_ld / cloud_deployment profile), the DD backend builds the EXACT
    // reachable-marking set and decides per-place constancy directly:
    //
    //   place p stable  ⟺  EF( m[p] != init[p] ) is NOT reachable
    //   StableMarking   =  TRUE iff SOME place is stable, FALSE otherwise.
    //
    // P/T only: on colored models StableMarking is a per-group-SUM constancy
    // question (handled by the colored observer above), not the per-individual
    // -place formulation this lane computes — so we only reach here when
    // `colored_groups` is empty.
    //
    // Soundness: `build_sound_dd_spec` gates the encoded value range to a
    // superset of every place's reachable projection, so a converged BDD
    // reachable set is EXACT and each per-place EF verdict is ground truth
    // (equivalent to a completed StableMarking BFS). Fail-closed: on ANY DD
    // failure (decline / timeout / OOM / panic) NO verdict is emitted and we
    // fall through to BMC / PDR / BFS unchanged, so no existing verdict can
    // change. The lane is budget-bounded by `dd_budget`, which reserves the
    // BFS fallback slice, so it cannot starve the existing engines (the CTL
    // DD-lane starvation lesson).
    #[cfg(feature = "dd-backend")]
    if let Some(verdict) = super::dd_fastpath::try_dd_stable_marking(net, config.deadline()) {
        eprintln!("StableMarking: resolved exactly by DD reachable-set fast-path");
        return verdict;
    }

    // --- Identity net (no reduction) ---
    // Structural reductions are currently unsound for StableMarking: on nets
    // like FMS-PT-*, agglomeration suppresses dynamic behavior, making places
    // appear constant when they are not (flipping FALSE to TRUE). Use the
    // identity net until a sound reduction contract is validated.
    let reduced = ReducedNet::identity(net);
    let config = config.refitted_for_net(&reduced.net);

    // SMT-based BMC + k-induction for stability on the reduced net.
    // Even when inconclusive, partial per-place instability seeds the observer.
    let mut bmc_unstable =
        match global_properties_bmc::run_stable_marking_bmc(&reduced.net, config.deadline()) {
            Some(result) => {
                if let Some(verdict) = result.verdict {
                    return if verdict {
                        Verdict::True
                    } else {
                        Verdict::False
                    };
                }
                result.unstable
            }
            None => vec![false; reduced.net.num_places()],
        };

    // PDR/IC3 for StableMarking is fail-closed: only definite IC3 verdicts
    // resolve the property, so unknowns/timeouts always fall through to BFS.
    // We default on to harvest invariants before the explicit-state explorer
    // burns its time budget; `TY_MCC_ENABLE_STABLE_PDR=0` force-disables.
    if stable_pdr_enabled() {
        if let Some(pdr_result) =
            global_properties_pdr::run_stable_marking_pdr(&reduced.net, config.deadline())
        {
            if let Some(verdict) = pdr_result.verdict {
                return if verdict {
                    Verdict::True
                } else {
                    Verdict::False
                };
            }
            // Merge PDR instability results with BMC results.
            for (i, &pdr_unstable) in pdr_result.unstable.iter().enumerate() {
                if pdr_unstable {
                    bmc_unstable[i] = true;
                }
            }
        }
    }

    // Random-walk FALSE-witness lane (P/T only). Runs AFTER the BMC/PDR seeding
    // and BEFORE the explicit BFS observer. MCC StableMarking is FALSE iff NO
    // place is constant across all reachable markings; observing any reachable
    // marking where `m[p] != initial[p]` PROVES place p is non-constant. If a
    // deep random walk proves EVERY place non-constant, StableMarking = FALSE.
    //
    // Soundness: this is a strict under-approximation that emits ONLY FALSE, and
    // ONLY when every place has been directly observed to differ from its initial
    // value in a reachable marking (the walk fires only enabled transitions from
    // the initial marking). It NEVER claims TRUE/stable; a miss or timeout returns
    // None and we fall through to the BFS observer unchanged. It runs on the
    // ORIGINAL `net` (never a reduced one) because structural reduction is unsound
    // for StableMarking. Colored models took the group-SUM path above and returned
    // before reaching here, so this lane is P/T-only by construction.
    //
    // Budget: ADDITIVE / leftover-only. `stable_walk_deadline` takes at most
    // `(remaining - STABLE_WALK_BFS_FALLBACK_RESERVE) / 4` (capped at
    // STABLE_WALK_PHASE_CAP) and reserves the full BFS-fallback tail, so it can
    // NEVER starve the exhaustive BFS observer.
    //
    // The seed `bmc_unstable` is the per-place "already proven non-constant" set
    // from the BMC/PDR phase; `reduced` is the identity net here, so its place
    // indexing matches the original `net`.
    let walk_phase_deadline = stable_walk_deadline(config.deadline());
    if !stable_walk_skip(walk_phase_deadline)
        && crate::examinations::reachability_walk::run_random_walk_stable_marking(
            net,
            &bmc_unstable,
            walk_phase_deadline,
        )
        .is_some()
    {
        eprintln!(
            "StableMarking: random walk proved every place non-constant \
             (reachable witness per place) → unstable"
        );
        return Verdict::False;
    }

    // POR: StableMarking is a safety property over candidate-stable places.
    // After BMC seeding, only places that BMC did NOT prove unstable remain as
    // candidates — and only transitions that touch one of those places can
    // produce a counterexample marking. Transitions touching exclusively
    // already-unstable places are invisible to the remaining query and can be
    // re-ordered freely. SafetyPreserving stubborn sets preserve reachability
    // of the per-place safety violation `m[p] != initial[p]` for every visible
    // place p, which is exactly what the observer's per-place check needs.
    //
    // When ALL places are still candidates (BMC produced no seeds), visibility
    // degenerates to "any place" and visible_transitions_for_support returns
    // None — falling back to PorStrategy::None preserves the prior behavior.
    // GPU explicit-BFS tier (probe-then-GPU, mirroring the sibling lanes),
    // on the RAW net: per-place minima and maxima over the exhaustive
    // reachable set decide constancy directly — some place with min == max
    // (⇒ pinned to its reachable initial value) ⇒ TRUE; every place varying
    // ⇒ FALSE. Exact both ways; the original-net answer is authoritative
    // regardless of the reduction the CPU path uses. The bounded CPU probe
    // (reduced net, same observer as the fallback) answers small nets with
    // the device untouched. Fail-closed: any GPU decline falls through to
    // the CPU BFS unchanged.
    #[cfg(feature = "gpu")]
    if crate::gpu_state_space::gpu_lane_enabled(net) {
        if let Some(cap) = crate::gpu_state_space::cpu_probe_cap(config.max_states()) {
            let probe_config = ExplorationConfig::new(cap)
                .with_deadline(config.deadline())
                .with_examination(config.examination());
            let mut observer =
                StableMarkingObserver::new_seeded(&reduced.net.initial_marking, &bmc_unstable);
            let result = ExecutionPlan::observer(PorStrategy::None).run_observer(
                &reduced.net,
                &probe_config,
                &mut observer,
            );
            if observer.all_unstable() {
                return Verdict::False;
            }
            if result.completed {
                return Verdict::True;
            }
            eprintln!(
                "[mcc] StableMarking: bounded CPU probe tripped (cap {cap}); \
                 escalating to the GPU lane"
            );
        }
        if let Some(stable) = crate::gpu_state_space::stable_marking_gpu(net, config.max_states()) {
            return if stable {
                Verdict::True
            } else {
                Verdict::False
            };
        }
    }

    let por_strategy =
        stable_marking_por_strategy(&reduced.net, &bmc_unstable).unwrap_or(PorStrategy::None);
    let plan = ExecutionPlan::observer(por_strategy);
    // Seed the observer with BMC results so BFS starts with known-unstable
    // places already eliminated.
    let mut observer =
        StableMarkingObserver::new_seeded(&reduced.net.initial_marking, &bmc_unstable);
    let result = match plan.run_checkpointable_observer(&reduced.net, &config, &mut observer) {
        Ok(result) => result,
        Err(error) => return checkpoint_cannot_compute("StableMarking", &error),
    };

    if observer.all_unstable() {
        Verdict::False
    } else if result.completed {
        Verdict::True
    } else {
        Verdict::CannotCompute
    }
}

/// Build a SafetyPreserving POR strategy for StableMarking on the reduced net.
///
/// The visibility set is constructed from the places that BMC did not already
/// prove unstable: those are the candidate stable places the BFS observer is
/// still trying to falsify. A transition is visible iff it can change the
/// token count of any candidate place. Transitions touching only places that
/// are already known to be unstable are invisible — their reordering cannot
/// affect the remaining safety witnesses the observer searches for.
///
/// Returns `None` when no useful reduction is possible (no candidate places,
/// or every transition is visible).
fn stable_marking_por_strategy(net: &PetriNet, bmc_unstable: &[bool]) -> Option<PorStrategy> {
    let num_places = net.num_places();
    let num_transitions = net.num_transitions();
    if num_places == 0 || num_transitions == 0 {
        return None;
    }

    let mut support = QuerySupport::new(num_places, num_transitions);
    let mut candidate_count = 0usize;
    for (idx, &unstable) in bmc_unstable.iter().enumerate().take(num_places) {
        if !unstable {
            support.places[idx] = true;
            candidate_count += 1;
        }
    }
    if candidate_count == 0 {
        // Every place is already known unstable — observer will terminate
        // on the first state. POR adds no value.
        return None;
    }

    let visible = visible_transitions_for_support(net, &support)?;
    Some(PorStrategy::SafetyPreserving { visible })
}
