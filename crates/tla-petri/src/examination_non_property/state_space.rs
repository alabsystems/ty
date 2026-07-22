// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::super::examination_plan::ExecutionPlan;
use super::common::checkpoint_cannot_compute;
use crate::examinations::state_space::{StateSpaceObserver, StateSpaceStats};
use crate::explorer::ExplorationConfig;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, TransitionInfo};
use crate::stubborn::PorStrategy;
#[cfg_attr(not(test), allow(unused_imports))]
use tla_bignum::{BigUint, ToPrimitive, Zero};

/// Absolute ceiling on the DD phase budget under a finite deadline.
///
/// With a deadline the budget is `remaining − reserve` so the DD lane gets
/// the *bulk* of the wall clock (the symbolic engine is the only exact path
/// for the 5–50s band that the old hard 5s cap killed). This ceiling only
/// bounds a pathologically large (or overflowing) deadline; it is sized far
/// above any realistic MCC per-examination confinement so for normal runs
/// the deadline term is the binding one. Raising the wall-clock budget can
/// only let an *exact* symbolic computation finish or leave the result
/// unchanged — it never alters a value (the fail-closed `BudgetExceeded` /
/// `CountInexact` declines and the preserved BFS reserve are untouched), so
/// this is soundness-neutral.
// StateSpace DD runs either converge FAST (a well-ordered structured net
// saturates in well under a second — e.g. Philosophers-PT-000010's 59049
// states in 0.12s) or do not converge at all (a wide net has no good order
// and never finishes). So — unlike UpperBounds, which has a genuine 5–50s
// converging band that wants the bulk of the deadline — giving the StateSpace
// DD phase the whole wall clock is counterproductive: it lets a futile
// non-converging run burn ~50s and STARVE the exact BFS fallback that would
// otherwise have enumerated a BFS-sized net (regression observed on
// BridgeAndVehicles-PT-V20P10N20: 9.07M states, BFS-solvable in the full
// budget, lost when DD ate the budget first). Cap the DD phase modestly so it
// still captures every fast DD win (the huge *structured* nets BFS cannot
// hold, which converge quickly) while leaving BFS the bulk of the deadline.
// Soundness-neutral: a smaller DD budget can only change which exact engine
// answers, never a value.
#[cfg(feature = "dd-backend")]
const STATE_SPACE_DD_MAX_BUDGET: std::time::Duration = std::time::Duration::from_secs(12);
/// DD phase budget when **no** wall-clock deadline is supplied (local runs
/// without `--timeout` / `BK_TIME_CONFINEMENT`, and the convenience
/// wrappers). Kept small so a deadline-less invocation still falls back to
/// BFS promptly instead of spinning the detached DD worker. Production MCC
/// always supplies a deadline and takes the scaled branch.
#[cfg(feature = "dd-backend")]
const STATE_SPACE_DD_NO_DEADLINE_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);
#[cfg(feature = "dd-backend")]
const STATE_SPACE_BFS_FALLBACK_RESERVE: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) fn state_space_stats(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> Option<StateSpaceStats> {
    state_space_stats_with_nupn(net, config, None)
}

/// [`state_space_stats`] with the model's NUPN structure (when the PNML
/// carries one) for seeding the DD lane's variable order. PERFORMANCE-ONLY:
/// the seed competes under `tla_dd`'s span guard and any permutation is
/// answer-preserving, so all four metrics are identical with or without it.
pub(crate) fn state_space_stats_with_nupn(
    net: &PetriNet,
    config: &ExplorationConfig,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Option<StateSpaceStats> {
    // Tier-1 structural CONSERVATION FAST-PATH — attempted FIRST, before the
    // budget-consuming pre-reduction / DD / MDD lanes.
    //
    // The structural recognizer is pure arithmetic (no marking enumeration, no
    // Farkas, no LP): gate (1) is an O(arcs) scan, gate (2) is two BFS over the
    // place digraph, and the closed forms are a handful of memoized BigUint
    // binomials. On the WIDE unit-state-machine nets it targets (large
    // NeighborGrid / Diffusion2D instances) it decides in tens of milliseconds.
    //
    // It MUST run before `reduce_dead_transitions_only`: that pre-reduction runs
    // a per-transition joint-enabling + trap-LP oracle wall-capped at the
    // deadline, which on a wide net (e.g. 19404 transitions) can consume the
    // ENTIRE budget and starve every downstream lane — including this one when
    // it ran later. Running the recognizer up front lets it win those cells.
    //
    // SOUNDNESS-NEUTRAL ordering: the recognizer computes the EXACT four metrics
    // of the ORIGINAL net by closed form (the pre-reduction is itself
    // metric-preserving, so it would not have changed the answer — it only
    // shrinks the transition set for the explicit lanes). On any decline it
    // falls through to the IDENTICAL existing pipeline (pre-reduction → inner
    // lanes) unchanged, so no net the old path decided changes value. Gated by
    // the same `TY_MCC_DISABLE_TIER1_STATESPACE` kill-switch.
    if !tier1_state_space_disabled() {
        if let Some(stats) = tier1_structural_state_space_stats(net, config) {
            return Some(stats);
        }
    }

    // Universally-sound pre-reduction: remove transitions proven never enabled
    // in any reachable marking (structural detector ∪ joint-enabling + trap LP
    // oracle), wall-capped under the deadline. This is the ONLY structural
    // reduction that is verdict-preserving for StateSpace: a dead transition
    // fires on no run, so the reduced net has the IDENTICAL reachable marking
    // set (counts), the IDENTICAL edge set (a dead transition contributes zero
    // outgoing edges from every reachable marking), and — because places are
    // preserved exactly (transition-only removal) — the IDENTICAL
    // `max_token_in_place` and `max_token_sum`. All four metrics therefore map
    // straight through with no reconstruction. The smaller transition set
    // shrinks per-state successor work (and can split a structurally-bridged
    // net into independent components whose product the decomposition path
    // counts exactly), so a net that timed out on the raw transition set may
    // now complete. On no dead transition this is the original net unchanged.
    let dead_reduced = crate::reduction::reduce_dead_transitions_only(net, config.deadline());
    if !dead_reduced.report.dead_transitions.is_empty() {
        eprintln!(
            "StateSpace: pre-reduction removed {} dead transition(s) never enabled in \
             any reachable marking (metric-preserving)",
            dead_reduced.report.dead_transitions.len()
        );
        // NUPN stays valid across this reduction: it removes transitions
        // only, so the place set and indices are preserved exactly.
        return state_space_stats_inner(&dead_reduced.net, config, nupn);
    }
    state_space_stats_inner(net, config, nupn)
}

fn state_space_stats_inner(
    net: &PetriNet,
    config: &ExplorationConfig,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Option<StateSpaceStats> {
    #[cfg(not(feature = "dd-backend"))]
    let _ = nupn;
    // Soundness guard (#1483): StateSpace requires state/edge counts of the
    // ORIGINAL net. Structural reduction changes the reachability graph, so
    // reduced-net counts are wrong. We explore the original net directly.
    //
    // When the original net is too large to explore completely, we return
    // None (CANNOT_COMPUTE = 0 pts) instead of reporting wrong counts
    // from the reduced net (-8 pts per wrong value).
    //
    // Decision-Diagram authoritative path (off by default — gated by
    // `dd-backend`): for very small bounded nets we try a BDD-based
    // forward reachability engine first under a hard 5-second phase cap.
    // Under finite deadlines, that cap is clipped so explicit BFS keeps a
    // reserve; when only the reserve remains, DD is skipped. It now computes
    // ALL FOUR `StateSpace` metrics (states, edges, max_token_in_place,
    // max_token_sum), so on success we return its result directly and skip
    // BFS entirely. Soundness invariant: every per-metric extraction is
    // differentially tested vs the BFS observer in
    // `tla_dd::tests::test_*_matches_bfs`.
    //
    // On ANY DD failure (timeout, precondition violation, panic) the
    // explicit BFS engine runs unchanged. Soundness floor: a slow or
    // missing DD never produces a wrong answer.
    #[cfg(feature = "dd-backend")]
    let mut dd_metrics_opt: Option<tla_dd::DdStateSpaceMetrics> = None;
    #[cfg(feature = "dd-backend")]
    if let Some(dd_budget) = state_space_dd_budget(config.deadline(), std::time::Instant::now())
        .filter(|_| !dd_state_space_disabled())
    {
        if let Some(dd_metrics) = try_dd_full_metrics_timed_seeded(net, dd_budget, nupn) {
            // The DD `state_count`/`edge_count` are `u64` and now widen
            // LOSSLESSLY into the `u128` `StateSpaceStats` (the `usize`
            // narrowing is deferred — and fail-closed via `states_wide` — to the
            // final `StateSpaceReport` boundary), so a DD bundle is always
            // adoptable here and we return it directly.
            if let Some(stats) = dd_metrics_to_stats(&dd_metrics) {
                return Some(stats);
            }
            // Defensive fallback (currently unreachable: `dd_metrics_to_stats`
            // never declines a `u64`→`u128` widening): if a future DD metric
            // ever could not be adopted, keep the bundle so the MDD lane below
            // can be CROSS-VALIDATED against it (gate case a) before adoption.
            dd_metrics_opt = Some(dd_metrics);
        }
    }

    // GPU explicit-BFS lane (device-resident frontier + fingerprint table via
    // the shared `tla-gpu` CUDA engine). RESEQUENCED BEFORE THE MDD PHASE
    // (audit 2026-07-11): the lane is device-fast (bounded CPU probe ≈ ≤1 s,
    // device BFS ≈ seconds, both capacity-capped) while the MDD phase is now
    // deadline-SCALED — running the cheap exact lane first means a
    // GPU-winnable net resolves in seconds instead of waiting behind a long
    // non-converging MDD phase, and a giant net declines fast on the
    // capacity caps. Guarded to only start with more than the BFS reserve
    // remaining. Auto-escalation mirrors the TLA+ engine's probe-then-GPU
    // tier: a bounded CPU probe answers small spaces exactly with the device
    // never touched; only a tripped cap escalates to the GPU. Fail-closed:
    // probe / emission / capacity / engine errors all decline onward
    // unchanged. Kill-switch `TY_MCC_DISABLE_GPU_STATESPACE`;
    // `TY_MCC_GPU_STATESPACE_FORCE` skips the probe (testing lever).
    #[cfg(feature = "gpu")]
    if crate::gpu_state_space::gpu_lane_enabled(net)
        && config.deadline().is_none_or(|d| {
            d.saturating_duration_since(std::time::Instant::now())
                > STATE_SPACE_BFS_FALLBACK_RESERVE
        })
    {
        if let Some(cap) = crate::gpu_state_space::cpu_probe_cap(config.max_states()) {
            let probe_config = ExplorationConfig::new(cap)
                .with_deadline(config.deadline())
                .with_examination(config.examination());
            if let Some(stats) = state_space_stats_bfs(net, &probe_config) {
                return Some(stats);
            }
            eprintln!(
                "[mcc] StateSpace: bounded CPU probe tripped (cap {cap}); escalating to the GPU lane"
            );
        }
        if let Some(stats) = crate::gpu_state_space::state_space_stats_gpu(net, config.max_states())
        {
            return Some(stats);
        }
    }

    // MDD lane (per-place multi-valued DD; `tla-mdd`). Runs AFTER the BDD lane
    // and BEFORE BFS, behind a CROSS-VALIDATED soundness gate. The BDD lane is
    // large-final-BDD-bound; the MDD (one level per place, bound+1 edges) is
    // far more compact on counter / conserved nets, so it computes |R| + the
    // three other metrics where the BDD lane declined or timed out.
    //
    // Soundness contract (`try_mdd_full_metrics_gated`):
    //   (a) if the BDD lane ALSO produced metrics, the MDD MUST match them
    //       exactly — a mismatch DECLINES the MDD (impossible given the
    //       cross-check battery, but gated anyway, fail-closed);
    //   (b) if the BDD lane DECLINED (the target case), the MDD answer is
    //       adopted because it is exact-by-construction (proven by the
    //       crate's BFS/BDD cross-check battery), plus a debug-only sample
    //       cross-check.
    // Kill-switch `TY_MCC_DISABLE_MDD_STATESPACE` (default ON) skips the lane.
    // Any MDD decline/overflow/budget falls through to BFS unchanged.
    #[cfg(feature = "dd-backend")]
    if let Some(stats) = try_mdd_full_metrics_gated(net, config, dd_metrics_opt.as_ref(), nupn) {
        return Some(stats);
    }

    // Tier-1 structural reduction-equation lane (the full-simplex block
    // recognizer composed with the shipping independent-component product).
    // Runs AFTER the MDD lane and BEFORE the disconnected-component BFS path,
    // behind the kill-switch `TY_MCC_DISABLE_TIER1_STATESPACE` (default ON, i.e.
    // the lane RUNS unless the var is set — mirroring the MDD kill-switch).
    //
    // SOUNDNESS: a component recognized as a strongly-connected ordinary
    // single-simplex net has its EXACT four metrics computed by the certified
    // closed forms (states = multichoose(d, n), edges = Σ_t multichoose(d,
    // n−Σpre_t); see ty_algebraic_geometry/PetriFiberCount.lean +
    // PetriEdges.lean). A component NOT recognized falls back to the shipping
    // per-component BFS. If a component neither recognizes NOR BFS-completes,
    // the whole lane DECLINES (returns None) and the existing lanes run — exact,
    // fail-closed. Behavior-preserving on every net the recognizer declines.
    if !tier1_state_space_disabled() {
        if let Some(stats) = tier1_structural_state_space_stats(net, config) {
            return Some(stats);
        }
    }

    if let Some(stats) = disconnected_component_state_space_stats(net, config) {
        return Some(stats);
    }

    state_space_stats_bfs(net, config)
}

/// Kill-switch for the Tier-1 structural StateSpace lane. The lane is ON by
/// default; set `TY_MCC_DISABLE_TIER1_STATESPACE` to a truthy value
/// (`1`/`on`/`true`/`yes`) to disable it and fall back to the
/// disconnected-component / BFS behavior unchanged.
///
/// SOUNDNESS-NEUTRAL either way: disabling the lane can only make a net decline
/// to the (exact) disconnected-component / BFS path instead of being decided by
/// the (exact) closed form. It never changes a published value. Mirrors
/// [`mdd_state_space_disabled`] but is NOT feature-gated (the Tier-1 lane is
/// pure structural arithmetic, no `dd-backend` dependency).
fn tier1_state_space_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_TIER1_STATESPACE").is_ok_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// Tier-1 structural reduction-equation StateSpace counter (FIRST increment:
/// the full-simplex block recognizer composed with the shipping
/// independent-component product).
///
/// Decomposes the net into independent components, attempts the full-simplex
/// recognizer per component (closed-form exact count when recognized), falls
/// back to per-component BFS otherwise, and composes the per-component metrics
/// with [`combine_component_state_space_stats`]. Returns `None` (whole lane
/// declines, fail-closed) if any component neither recognizes nor BFS-completes,
/// or if any structural step is uncertain.
///
/// SOUNDNESS: see [`recognize_full_simplex_component`] for the exact gate that
/// makes the closed form provably correct (machine-checked leaves in
/// `ty_algebraic_geometry/PetriFiberCount.lean` + `PetriEdges.lean`). The
/// recognizer is CONSERVATIVE: any doubt ⇒ decline that component ⇒ BFS
/// fallback ⇒ (if BFS also fails) `None`.
pub(crate) fn tier1_structural_state_space_stats(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> Option<StateSpaceStats> {
    // CONSTANT-READ-ARC R-REDUCTION (run FIRST, before decomposition).
    //
    // Strip every place that is a provably-redundant CONSTANT READ self-loop:
    // its token count is invariant on the whole reachable set and it never
    // gates any transition's enabledness (see [`strip_constant_read_places`]
    // for the exact, conservative gate — `pre(t,p) == post(t,p)` for every
    // transition AND `init(p) >= max_t pre(t,p)`). Such places hold `init(p)`
    // forever and form a constant read self-loop that can be deleted with |R|
    // UNCHANGED (machine-checked: the singleton-`{p}` P-invariant pins
    // `m[p] = init[p]`, so each fiber of the forget-`p` projection is a
    // singleton — `ty_algebraic_geometry/PetriRedundant.lean`
    // `card_reachable_eq_card_reduced`; the read-guard-always-satisfied
    // hypothesis is the read-arc specialization discharged in
    // `PetriReadArcRedundant.lean`).
    //
    // PURPOSE: nets like BART couple per-train unit state machines ONLY through
    // such constant-guard resource places (DistStation / NewDistTable /
    // StopTable). Removing them decouples the trains, so `independent_components`
    // below splits the net into the disjoint strongly-connected unit-state-machine
    // blocks that `recognize_full_simplex_component` counts exactly, and
    // `combine_component_state_space_stats` multiplies into the product simplex
    // count (BART-PT-NNN: 132^N).
    //
    // SOUNDNESS-NEUTRAL on a decline: the reduced net's |R| and edge set equal
    // the original's (the stripped places are constant — they never gate or get
    // consumed), so the closed-form product is the ORIGINAL `states`/`edges`. If
    // nothing strips, `net` is the original unchanged. If after stripping any
    // component still fails the recognizer's unit-move / SC gates, that component
    // falls to per-component BFS, and if BFS also fails the whole lane DECLINES —
    // fail-closed, never a partial/guessed count.
    //
    // BUT `max_token_in_place` and `max_token_sum` DO depend on the stripped
    // places: each held `init(p)` on every reachable marking. So the strip also
    // returns those constant contributions, and we FOLD THEM BACK below:
    //   max_token_in_place(orig) = max(reduced max_in_place, max_p init(p_stripped))
    //   max_token_sum(orig)      = reduced max_sum + Σ_p init(p_stripped)
    // (uniform shift: every reachable marking carries exactly Σ init(stripped)
    // extra tokens, so the max total rises by that constant.)
    let stripped = strip_constant_read_places(net);
    let (net, stripped_max_in_place, stripped_token_sum): (&PetriNet, u64, u64) = match &stripped {
        Some(r) => (&r.net, r.max_stripped_token_in_place, r.stripped_token_sum),
        None => (net, 0, 0),
    };
    let components = independent_components(net)?;
    // `independent_components` returns the single trivial component for a
    // connected net.

    // STRICT-ADDITIVITY GATE — the invariant the Tier-1 lane must uphold: a
    // non-success is a TRUE no-op. It MUST NOT lose a cell the existing pipeline
    // would have decided.
    //
    // The per-component BFS fallback below is only sound to run when the net
    // ACTUALLY DECOMPOSED into >1 independent component. In that case the
    // product (each recognized component's closed form × a SMALL per-component
    // BFS for the unrecognized ones) is strictly cheaper than a whole-net BFS,
    // and that is a genuine decomposition win the downstream lanes cannot match
    // (`disconnected_component_state_space_stats` would BFS *every* component,
    // forgoing the closed forms).
    //
    // For a SINGLE trivial component (the connected net itself) the recognizer
    // is the ONLY structural value Tier-1 can add. If it declines, running a
    // per-component BFS here would be a REDUNDANT WHOLE-NET BFS that
    //   (i) bypasses the richer downstream lanes (dead-transition reduction, the
    //       DD / MDD symbolic engines) when run from the EARLY call site (before
    //       those lanes), and
    //   (ii) can DECLINE (state cap / deadline) where a downstream lane would
    //        have succeeded — turning a decidable net into CANNOT_COMPUTE.
    // This was the SharedMemory-PT-000005 regression: a single 41-place
    // component the recognizer declines was decided by Tier-1's own whole-net
    // BFS instead of falling through to the identical downstream BFS — fragile
    // (any deadline / size shift flips it to CC, losing the cell). So: when
    // there is only ONE component and the recognizer declines, DECLINE the whole
    // Tier-1 lane (return None) and let the existing pipeline decide it.
    //
    // SOUNDNESS-NEUTRAL: declining here can only ROUTE the net to the existing
    // (exact) lanes — it never changes a value, and it strictly cannot lose a
    // cell the old path decided (the old path is exactly what then runs).
    let single_trivial_component = components.len() == 1;

    let mut component_stats = Vec::with_capacity(components.len());
    for component in &components {
        let component_net = build_component_net(net, component)?;
        let stat = match recognize_full_simplex_component(&component_net) {
            // Recognized: the certified closed form is exact for this component.
            Some(stat) => stat,
            // Not recognized. Only fall back to per-component BFS when the net
            // genuinely DECOMPOSED (>1 component); for a single trivial
            // component DECLINE the whole lane so the existing pipeline decides
            // it unchanged (the STRICT-ADDITIVITY GATE above). When >1
            // component, a per-component BFS that cannot complete declines the
            // whole lane (fail-closed, true no-op).
            None => {
                if single_trivial_component {
                    return None;
                }
                state_space_stats_bfs(&component_net, config)?
            }
        };
        component_stats.push(stat);
    }

    let mut combined = combine_component_state_space_stats(&component_stats)?;
    // Fold the stripped constant places' contributions back into the two
    // marking-magnitude metrics (states/edges are already exact — the stripped
    // places contribute a factor/term of identity to those).
    combined.max_token_in_place = combined.max_token_in_place.max(stripped_max_in_place);
    combined.max_token_sum = combined.max_token_sum.checked_add(stripped_token_sum)?;
    Some(combined)
}

/// Output of [`strip_constant_read_places`]: the reduced net plus the constant
/// marking contributions the stripped places carried (needed to reconstruct the
/// ORIGINAL net's `max_token_in_place` / `max_token_sum` — see the fold in
/// [`tier1_structural_state_space_stats`]).
struct ConstReadReduction {
    net: PetriNet,
    /// `max` over stripped places `p` of `init(p)` (each held `init(p)` on every
    /// reachable marking). `0` when no stripped place out-tops the reduced net's
    /// own per-place max (the fold takes the overall max).
    max_stripped_token_in_place: u64,
    /// `Σ` over stripped places `p` of `init(p)` — the uniform token shift every
    /// reachable marking carries from the constant places.
    stripped_token_sum: u64,
}

/// CONSTANT-READ-ARC R-REDUCTION. Returns `Some(ConstReadReduction)` with every
/// provably-redundant constant read self-loop place removed (and its now-empty
/// arcs dropped from each transition) plus the stripped places' constant marking
/// contributions, or `None` when NO place qualifies (the caller then keeps the
/// original net unchanged — behavior-preserving).
///
/// ## The gate (CONSERVATIVE — must match the Lean hypotheses EXACTLY)
///
/// A place `p` is a CONSTANT READ place iff BOTH hold:
///
///   (a) for EVERY transition `t`, the total input weight from `p` equals the
///       total output weight to `p`: `pre(t, p) == post(t, p)`. Then `p` is a
///       pure self-loop on every transition — never net-consumed or
///       net-produced — so its token count is INVARIANT under every firing.
///
///   (b) `init(p) >= max_t pre(t, p)`: the read guard `m[p] >= pre(t, p)` is
///       ALWAYS satisfied (because `m[p]` stays `= init(p)` by (a)), so `p`
///       never disables any transition.
///
/// Under (a)+(b), on every reachable marking `m[p] = init(p)` and `p` gates no
/// transition's enabledness, so deleting `p` (and its balanced self-loop arcs)
/// preserves the reachable set up to the forget-`p` bijection: `|R|` is
/// UNCHANGED. This is the read-arc specialization of TINA's rule R, licensed by
/// `ty_algebraic_geometry/PetriRedundant.lean` (the singleton-fiber count
/// equality `card_reachable_eq_card_reduced`, with the singleton-`{p}`
/// place-invariant `w = 1_{p}` whose conserved value is `init(p)`) and the
/// read-guard-always-satisfied hypothesis being discharged in
/// `PetriReadArcRedundant.lean`. The runtime gate below checks EXACTLY those
/// hypotheses.
///
/// SOUNDNESS NOTES (the conservative edges):
///   * The pre/post totals are summed with `checked_add` per place; ANY overflow
///     ⇒ that place does NOT qualify (treated as non-constant). No place is ever
///     stripped on uncertain arithmetic.
///   * A place with NO incident arcs trivially satisfies (a) with `max_t pre = 0`
///     and (b) `init(p) >= 0`, so it is constant and stripped. That is correct:
///     an isolated place's token count never changes and never gates anything, so
///     it contributes a factor of exactly `1` to `|R|` — dropping it preserves
///     the count. (It would otherwise force the recognizer to decline a `d == 0`
///     or non-strongly-connected isolated vertex.)
///   * The reduced net's place indices are RENUMBERED densely; the initial
///     marking and every surviving arc are remapped. The returned net is a
///     standalone `PetriNet` — the original is never mutated.
///   * Each stripped place `p` holds `init(p)` on every reachable marking, so
///     the result also returns `max_p init(p)` and `Σ_p init(p)` over stripped
///     places — the caller folds these into the original's `max_token_in_place`
///     / `max_token_sum` (those two metrics are NOT invariant under the strip,
///     unlike `states` / `edges`).
fn strip_constant_read_places(net: &PetriNet) -> Option<ConstReadReduction> {
    let num_places = net.num_places();
    if num_places == 0 || net.initial_marking.len() != num_places {
        return None;
    }

    // For each place: track whether it is still a constant-read candidate, and
    // the running max of pre(t, p) over transitions (for gate (b)).
    let mut is_constant = vec![true; num_places];
    let mut max_pre = vec![0u64; num_places];

    for t in &net.transitions {
        // Sum input / output weights per place for THIS transition (parallel
        // arcs to the same place add — matching `canonicalize_parallel_arcs`
        // and the dense `(pre, post)` semantics).
        let mut pre: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
        let mut post: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
        for arc in &t.inputs {
            let p = arc.place.0 as usize;
            if p >= num_places {
                return None; // malformed index ⇒ decline the whole reduction
            }
            let slot = pre.entry(p).or_insert(0);
            *slot = slot.checked_add(arc.weight)?;
        }
        for arc in &t.outputs {
            let p = arc.place.0 as usize;
            if p >= num_places {
                return None;
            }
            let slot = post.entry(p).or_insert(0);
            *slot = slot.checked_add(arc.weight)?;
        }
        // Gate (a): for every place touched by this transition, pre == post.
        // A place in `pre` but not `post` (or vice versa) has pre != post.
        for (&p, &pre_w) in &pre {
            let post_w = post.get(&p).copied().unwrap_or(0);
            if pre_w != post_w {
                is_constant[p] = false;
            }
            if pre_w > max_pre[p] {
                max_pre[p] = pre_w;
            }
        }
        for (&p, &post_w) in &post {
            if !pre.contains_key(&p) && post_w != 0 {
                // produced but not consumed ⇒ net-produced ⇒ pre != post.
                is_constant[p] = false;
            }
        }
    }

    // Gate (b): init(p) >= max_t pre(t, p). Apply to the survivors of gate (a).
    let mut strip = vec![false; num_places];
    let mut any = false;
    let mut all = true;
    for p in 0..num_places {
        if is_constant[p] && net.initial_marking[p] >= max_pre[p] {
            strip[p] = true;
            any = true;
        } else {
            all = false;
        }
    }

    if !any {
        return None; // nothing to strip ⇒ pipeline unchanged
    }
    if all {
        // Stripping EVERY place would leave an empty net (no places). That is
        // not a useful reduction for the recognizer (which needs `d >= 1`), and
        // a net whose every place is constant is the trivial single-marking net
        // the existing lanes already decide. Decline so the original flows on.
        return None;
    }

    // Constant contributions of the stripped places (each holds init(p) on every
    // reachable marking): max over stripped init for `max_token_in_place`, sum of
    // stripped init for the uniform `max_token_sum` shift. `checked_add` ⇒ decline
    // the whole reduction on the (astronomically unlikely) overflow.
    let mut max_stripped_token_in_place = 0u64;
    let mut stripped_token_sum = 0u64;
    for p in 0..num_places {
        if strip[p] {
            let v = net.initial_marking[p];
            if v > max_stripped_token_in_place {
                max_stripped_token_in_place = v;
            }
            stripped_token_sum = stripped_token_sum.checked_add(v)?;
        }
    }

    // Build the reduced net: surviving places renumbered densely, the initial
    // marking projected, and each transition's arcs filtered to survivors and
    // remapped to the new indices.
    let mut old_to_new = vec![usize::MAX; num_places];
    let mut survivors = Vec::with_capacity(num_places);
    for (p, drop) in strip.iter().enumerate() {
        if !drop {
            old_to_new[p] = survivors.len();
            survivors.push(p);
        }
    }

    let places = survivors.iter().map(|&p| net.places[p].clone()).collect();
    let initial_marking = survivors.iter().map(|&p| net.initial_marking[p]).collect();

    let filter_remap = |arcs: &[Arc]| -> Vec<Arc> {
        arcs.iter()
            .filter_map(|arc| {
                let new = old_to_new[arc.place.0 as usize];
                (new != usize::MAX).then_some(Arc {
                    // `new < num_places <= u32::MAX-bounded place count`; the
                    // index fits u32 because the original did.
                    place: PlaceIdx(new as u32),
                    weight: arc.weight,
                })
            })
            .collect()
    };

    let transitions = net
        .transitions
        .iter()
        .map(|t| TransitionInfo {
            id: t.id.clone(),
            name: t.name.clone(),
            inputs: filter_remap(&t.inputs),
            outputs: filter_remap(&t.outputs),
        })
        .collect();

    Some(ConstReadReduction {
        net: PetriNet {
            name: net
                .name
                .as_ref()
                .map(|name| format!("{name}::const-read-reduced")),
            places,
            transitions,
            initial_marking,
        },
        max_stripped_token_in_place,
        stripped_token_sum,
    })
}

/// The FULL-SIMPLEX RECOGNIZER (the whitelist) for a single (already
/// component-local) net. Returns `Some(StateSpaceStats)` with the four metrics
/// in EXACT closed form when the component is provably a strongly-connected,
/// ordinary, single-simplex net; `None` (DECLINE) on any doubt.
///
/// ## Soundness gate (every condition must hold, else DECLINE)
///
/// 1. **Unit one-in/one-out transitions.** Every transition has EXACTLY one
///    input arc and EXACTLY one output arc, both weight 1 — a single-token move
///    `pre(t) → post(t)`. This is EXACTLY the `PetriStateMachineComplete.lean`
///    `StateMachine` model (`pre` = a single source place of weight 1, `post` =
///    a single target place of weight 1). It PROVES total tokens are conserved
///    (each transition's net effect is `−1 + 1 = 0`), so the all-ones vector is
///    a P-invariant with conserved value `n = Σ` initial marking. A
///    multi-input/multi-output (join/fork/sync) transition can carve the
///    reachable set into a STRICT subset of the simplex even when every other
///    gate passes (verified counterexample), so we DECLINE on any such
///    transition. (Strong connectivity of the bipartite place/transition graph
///    is NOT sufficient on its own — the unit-move restriction is necessary.)
///
///    This gate runs FIRST. When it passes, the covering weight-1 conservation
///    law `Σ p = n` is **SYNTHESIZED** from the unit-move structure (`n = Σ`
///    initial, `checked_add`, decline on overflow) rather than searched for via
///    the general Farkas [`crate::invariant::compute_p_invariants`] — which
///    truncates at `MAX_ROWS` on wide nets and would spuriously DECLINE a
///    perfectly good state machine. Self-loop transitions (input place ==
///    output place) are valid unit moves (they conserve tokens and add a
///    harmless self-edge to the digraph), so they are NOT declined.
/// 2. **Strongly-connected place digraph.** Build the directed graph `G` on the
///    component's places with an edge `pre(t) → post(t)` per transition; require
///    `G` strongly connected. With (1), strong connectivity means every token
///    can be routed to every place, so every simplex lattice point is reachable
///    — the reachable set is EXACTLY the full simplex. (Single-vertex `G` is
///    strongly connected by convention.)
///
/// Under gates (1)+(2), `PetriStateMachineComplete.sm_reachable_eq_simplex`
/// (a strongly-connected ordinary state machine ⇒ `ReachableSet` = the full
/// simplex `{m : Σ m = n}`) guarantees the count is EXACTLY
/// `multichoose(#places, n)`. The per-place maximum `= n` (a single place can
/// hold all `n` tokens) is then a THEOREM of `sm_reachable_eq_simplex`, so the
/// old Farkas-dependent `structural_place_bound` self-check is no longer needed
/// (and is removed — it was the other Farkas truncation point). The recognizer
/// therefore has NO dependency on `compute_p_invariants` at all, so it SCALES to
/// wide nets (large `NeighborGrid` / `Diffusion2D` instances) the general Farkas
/// computation would have truncated.
///
/// When recognized: `states = multichoose(d, n)` where `d = #places`;
/// `edges = Σ_t (if Σpre_t ≤ n then multichoose(d, n−Σpre_t) else 0)` over the
/// component's transitions (the `PetriEdges.block_edges_eq_sum` closed form,
/// with `pre_t` the transition's input arc weights as a `Fin d → ℕ` vector);
/// `max_token_in_place = n`; `max_token_sum = n`.
fn recognize_full_simplex_component(net: &PetriNet) -> Option<StateSpaceStats> {
    let d = net.num_places();
    if d == 0 {
        return None;
    }

    // n = Σ initial tokens over the component, as u64 (overflow ⇒ decline).
    // This SYNTHESIZES the conserved total of the covering weight-1 P-invariant
    // `Σ p = n` directly — once gate (1) confirms the net is an ordinary unit
    // state machine, conservation is a structural fact (each transition is
    // −1 +1 = 0), so no Farkas search is needed.
    let n: u64 = net
        .initial_marking
        .iter()
        .try_fold(0u64, |acc, &t| acc.checked_add(t))?;

    // --- Gate (1): every transition is a unit one-in/one-out token move. ---
    // Runs FIRST. This IS the `StateMachine` model: pre = single source weight 1,
    // post = single target weight 1. Decline ANY transition that is not a unit
    // move: 0 inputs, 0 outputs, ≥2 of either, or any weight ≠ 1. Self-loops
    // (src == dst) are VALID (they conserve and add a harmless self-edge to the
    // digraph), so they are NOT declined. (Also collect the place digraph edges
    // for gate (2) in the same pass.)
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(net.num_transitions());
    for t in &net.transitions {
        if t.inputs.len() != 1 || t.outputs.len() != 1 {
            return None;
        }
        let i = &t.inputs[0];
        let o = &t.outputs[0];
        if i.weight != 1 || o.weight != 1 {
            return None;
        }
        let src = i.place.0 as usize;
        let dst = o.place.0 as usize;
        if src >= d || dst >= d {
            return None;
        }
        edges.push((src, dst));
    }

    // --- Gate (2): the place digraph G must be strongly connected. ---
    if !place_digraph_strongly_connected(d, &edges) {
        return None;
    }

    // Recognized: gate (1) + gate (2) ⇒ `sm_reachable_eq_simplex` licenses
    // R = full simplex {m : Σ m = n}, so `max_token_in_place = n` is a THEOREM
    // (no structural self-check needed). Compute the certified closed forms in
    // BigUint.
    //
    // states = multichoose(d, n) = C(d + n − 1, n).
    let states = multichoose(d as u64, n);

    // edges = Σ_t (if Σpre_t ≤ n then multichoose(d, n − Σpre_t) else 0).
    // Each transition has a single input arc of weight 1 (gate 2), so Σpre_t = 1
    // for every transition; but we compute Σpre_t generically from the input
    // arcs to mirror `block_edges_eq_sum` EXACTLY (and stay correct if gate (2)
    // is ever relaxed in a future increment).
    //
    // MEMOIZE `multichoose(d, n − pre_sum)` keyed by `pre_sum`: on a wide net
    // (e.g. NeighborGrid-PT-d5n4m1t35: 196608 transitions, each `pre_sum = 1`,
    // `multichoose(1024, 1023)` a ~600-digit BigUint binomial) recomputing the
    // SAME binomial once per transition dominated the wall clock (>120s). Under
    // gate (1) every `pre_sum` is 1, so the cache collapses the whole sum to a
    // SINGLE binomial scaled by the transition count. VALUE-PRESERVING: the sum
    // is bit-identical to the per-transition version (`a + a + … = count · a`),
    // only the redundant recomputation is removed.
    let mut multichoose_cache: std::collections::HashMap<u64, BigUint> =
        std::collections::HashMap::new();
    let mut edge_total = BigUint::zero();
    for t in &net.transitions {
        let pre_sum: u64 = t
            .inputs
            .iter()
            .try_fold(0u64, |acc, arc| acc.checked_add(arc.weight))?;
        if pre_sum <= n {
            let term = multichoose_cache
                .entry(pre_sum)
                .or_insert_with(|| multichoose(d as u64, n - pre_sum));
            edge_total += &*term;
        }
        // else: contributes 0 (the `else 0` branch of block_edges_eq_sum).
    }

    Some(StateSpaceStats {
        states,
        edges: edge_total,
        max_token_in_place: n,
        max_token_sum: n,
    })
}

/// `multichoose(d, n) = C(d + n − 1, n)` — the number of `f : Fin d → ℕ` with
/// `Σ f = n` (stars-and-bars; the size-`n` simplex lattice-point count in
/// dimension `d`). Computed in `BigUint` so an astronomically-large count is
/// EXACT (no `u128` cap). `d == 0` is the empty index set: `multichoose(0, 0) =
/// 1` (the empty sum) and `multichoose(0, n) = 0` for `n > 0` (no solution).
///
/// Certified by `ty_algebraic_geometry/PetriFiberCount.lean`
/// `simplex_lattice_count : #{f : Fin d → ℕ | Σ f = n} = Nat.multichoose d n`.
fn multichoose(d: u64, n: u64) -> BigUint {
    if d == 0 {
        // C(n − 1, n): 1 when n == 0 (empty sum), 0 otherwise.
        return if n == 0 {
            BigUint::from(1u32)
        } else {
            BigUint::zero()
        };
    }
    // C(d + n − 1, n) in exact arbitrary precision.
    let top = BigUint::from(d) + BigUint::from(n) - BigUint::from(1u32);
    tla_bignum::integer::binomial::<BigUint>(top, BigUint::from(n))
}

/// True iff the directed graph on `d` vertices with the given `edges`
/// (`(src, dst)` per transition) is STRONGLY CONNECTED — every vertex reachable
/// from every other via directed edges. A single vertex (`d == 1`) is strongly
/// connected by convention.
///
/// Uses the standard two-BFS test: pick vertex `0`; the digraph is strongly
/// connected iff every vertex is reachable from `0` in the forward graph AND
/// every vertex is reachable from `0` in the reverse graph (equivalently, `0`
/// is reachable from every vertex). Self-loops are harmless (no-op edges).
fn place_digraph_strongly_connected(d: usize, edges: &[(usize, usize)]) -> bool {
    if d <= 1 {
        return true; // single (or empty, unreachable here) vertex
    }
    let mut fwd: Vec<Vec<usize>> = vec![Vec::new(); d];
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); d];
    for &(s, t) in edges {
        fwd[s].push(t);
        rev[t].push(s);
    }
    reaches_all(d, &fwd) && reaches_all(d, &rev)
}

/// BFS from vertex `0`: returns true iff every vertex in `0..d` is reachable.
fn reaches_all(d: usize, adj: &[Vec<usize>]) -> bool {
    let mut seen = vec![false; d];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1usize;
    while let Some(v) = stack.pop() {
        for &w in &adj[v] {
            if !seen[w] {
                seen[w] = true;
                count += 1;
                stack.push(w);
            }
        }
    }
    count == d
}

/// Convert authoritative DD metrics into [`StateSpaceStats`].
///
/// The DD engine reports `state_count` as a `u64`; `StateSpaceStats::states` is
/// now a `u128`, so the count widens LOSSLESSLY (`u64 ⊆ u128`) and there is no
/// narrowing hazard at this boundary — the `usize` narrowing the previous
/// version guarded against is now performed (and fail-closed via `states_wide`)
/// only at the final `StateSpaceReport` boundary in `examination.rs`. The other
/// three metrics are already `u64` and map straight through.
#[cfg(feature = "dd-backend")]
fn dd_metrics_to_stats(dd_metrics: &tla_dd::DdStateSpaceMetrics) -> Option<StateSpaceStats> {
    // The BDD lane's `state_count`/`edge_count` are `u64`; they widen LOSSLESSLY
    // into the `BigUint` carriers (`u64 ⊆ BigUint`), so a BDD bundle is always
    // adoptable and the count value is identical to before.
    Some(StateSpaceStats {
        states: BigUint::from(dd_metrics.state_count),
        edges: BigUint::from(dd_metrics.edge_count),
        max_token_in_place: dd_metrics.max_token_in_place,
        max_token_sum: dd_metrics.max_token_sum,
    })
}

fn state_space_stats_bfs(net: &PetriNet, config: &ExplorationConfig) -> Option<StateSpaceStats> {
    let plan = ExecutionPlan::observer(PorStrategy::None);
    let config = config.refitted_for_net(net);
    let mut observer = StateSpaceObserver::new(&net.initial_marking);
    let result = match plan.run_checkpointable_observer(net, &config, &mut observer) {
        Ok(result) => result,
        Err(error) => {
            let _ = checkpoint_cannot_compute("StateSpace", &error);
            return None;
        }
    };
    if !result.completed {
        return None;
    }

    Some(observer.stats())
}

fn disconnected_component_state_space_stats(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> Option<StateSpaceStats> {
    let components = independent_components(net)?;
    if components.len() <= 1 {
        return None;
    }

    let mut component_stats = Vec::with_capacity(components.len());
    for component in &components {
        let component_net = build_component_net(net, component)?;
        component_stats.push(state_space_stats_bfs(&component_net, config)?);
    }

    combine_component_state_space_stats(&component_stats)
}

#[derive(Debug, Clone)]
struct IndependentComponent {
    places: Vec<usize>,
    transitions: Vec<usize>,
}

fn independent_components(net: &PetriNet) -> Option<Vec<IndependentComponent>> {
    let num_places = net.num_places();
    if num_places == 0 {
        return None;
    }

    let mut dsu = DisjointSet::new(num_places);
    let mut transition_places: Vec<Vec<usize>> = Vec::with_capacity(net.num_transitions());
    for transition in &net.transitions {
        let mut touched = Vec::with_capacity(transition.inputs.len() + transition.outputs.len());
        for arc in transition.inputs.iter().chain(&transition.outputs) {
            let place = arc.place.0 as usize;
            if place >= num_places {
                return None;
            }
            if !touched.contains(&place) {
                touched.push(place);
            }
        }

        let (&first, rest) = touched.split_first()?;
        for &place in rest {
            dsu.union(first, place);
        }
        transition_places.push(touched);
    }

    let mut place_component = vec![usize::MAX; num_places];
    let mut components: Vec<IndependentComponent> = Vec::new();
    for (place, slot) in place_component.iter_mut().enumerate() {
        let root = dsu.find(place);
        let index = components
            .iter()
            .position(|component| dsu.find(component.places[0]) == root)
            .unwrap_or_else(|| {
                components.push(IndependentComponent {
                    places: Vec::new(),
                    transitions: Vec::new(),
                });
                components.len() - 1
            });
        *slot = index;
        components[index].places.push(place);
    }

    for (transition, touched) in transition_places.iter().enumerate() {
        let component = place_component[touched[0]];
        if touched
            .iter()
            .any(|&place| place_component[place] != component)
        {
            return None;
        }
        components[component].transitions.push(transition);
    }

    Some(components)
}

fn build_component_net(net: &PetriNet, component: &IndependentComponent) -> Option<PetriNet> {
    let mut place_to_local = vec![usize::MAX; net.num_places()];
    for (local, &original) in component.places.iter().enumerate() {
        place_to_local[original] = local;
    }

    let places = component
        .places
        .iter()
        .map(|&place| net.places[place].clone())
        .collect();
    let initial_marking = component
        .places
        .iter()
        .map(|&place| net.initial_marking[place])
        .collect();

    let mut transitions = Vec::with_capacity(component.transitions.len());
    for &transition in &component.transitions {
        let original = &net.transitions[transition];
        transitions.push(TransitionInfo {
            id: original.id.clone(),
            name: original.name.clone(),
            inputs: remap_arcs(&original.inputs, &place_to_local)?,
            outputs: remap_arcs(&original.outputs, &place_to_local)?,
        });
    }

    Some(PetriNet {
        name: net
            .name
            .as_ref()
            .map(|name| format!("{name}::state-space-component")),
        places,
        transitions,
        initial_marking,
    })
}

fn remap_arcs(arcs: &[Arc], place_to_local: &[usize]) -> Option<Vec<Arc>> {
    arcs.iter()
        .map(|arc| {
            let local = place_to_local[arc.place.0 as usize];
            (local != usize::MAX).then_some(Arc {
                place: PlaceIdx(local.try_into().ok()?),
                weight: arc.weight,
            })
        })
        .collect()
}

fn combine_component_state_space_stats(stats: &[StateSpaceStats]) -> Option<StateSpaceStats> {
    // |R| of the product net = ∏_i |R_i|, EXACT as a `BigUint`: the product of
    // independent-component counts no longer fails closed past `u128` — a
    // decomposable net whose product is, say, 1e47 or 1e238 is now REPORTED at
    // full precision (the representational unblock). The count VALUE is
    // identical to the old `u128` path for any product that fit `u128`.
    let states = stats
        .iter()
        .fold(BigUint::from(1u32), |acc, stat| acc * &stat.states);

    // Edge count of the product net = Σ_i (edges_i · ∏_{j≠i} |R_j|). Exact
    // bignum throughout. `∏_{j≠i} |R_j|` is computed as `states / |R_i|`, which
    // is exact integer division (the total product is divisible by each
    // component count by construction) — skipping any empty component.
    let mut edges = BigUint::zero();
    let mut max_token_in_place = 0u64;
    let mut max_token_sum = 0u64;
    for stat in stats {
        if stat.states.is_zero() {
            continue;
        }
        let other_states = &states / &stat.states;
        edges += &stat.edges * other_states;
        max_token_in_place = max_token_in_place.max(stat.max_token_in_place);
        max_token_sum = max_token_sum.checked_add(stat.max_token_sum)?;
    }

    Some(StateSpaceStats {
        states,
        edges,
        max_token_in_place,
        max_token_sum,
    })
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent == value {
            value
        } else {
            let root = self.find(parent);
            self.parent[value] = root;
            root
        }
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root != right_root {
            self.parent[right_root] = left_root;
        }
    }
}

/// Test-only hook: build a [`PetriNet`] from raw parts (the `PetriNet` type is
/// `#[non_exhaustive]`, so an out-of-crate integration test cannot construct it
/// directly), then run the Tier-1 structural recognizer on it and return its
/// four metrics. This lets `tests/tier1_crosscheck_bfs.rs` differentially
/// validate the recognizer against a lightweight inline BFS oracle (which the
/// test computes itself, mirroring `tla-mdd/tests/crosscheck_bfs.rs`) without
/// reaching into crate-private internals.
///
/// `transitions` is `(inputs, outputs)` per transition, each arc a
/// `(place_index, weight)`. The result is `Some((states, edges,
/// max_token_in_place, max_token_sum))` or `None` when the Tier-1 lane declines.
/// `max_states` caps the per-component BFS FALLBACK inside the lane (so a
/// recognized component is decided by the closed form regardless).
///
/// `doc(hidden)` so it carries no advertised public-API surface (it is not real
/// API — only the crate's own integration tests consume it); unconditionally
/// `pub` so `cargo test --test tier1_crosscheck_bfs` reaches it without feature
/// plumbing. Soundness-neutral: a pure read-out of the Tier-1 lane's output.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn tier1_crosscheck_hook(
    num_places: usize,
    initial_marking: Vec<u64>,
    transitions: Vec<(Vec<(u32, u64)>, Vec<(u32, u64)>)>,
    max_states: usize,
) -> Option<(BigUint, BigUint, u64, u64)> {
    let mk_arcs = |arcs: &[(u32, u64)]| -> Vec<Arc> {
        arcs.iter()
            .map(|&(place, weight)| Arc {
                place: PlaceIdx(place),
                weight,
            })
            .collect()
    };
    let net = PetriNet {
        name: Some("tier1-crosscheck".into()),
        places: (0..num_places)
            .map(|i| crate::petri_net::PlaceInfo {
                id: format!("p{i}"),
                name: None,
            })
            .collect(),
        transitions: transitions
            .iter()
            .enumerate()
            .map(|(i, (inputs, outputs))| TransitionInfo {
                id: format!("t{i}"),
                name: None,
                inputs: mk_arcs(inputs),
                outputs: mk_arcs(outputs),
            })
            .collect(),
        initial_marking,
    };
    let config = ExplorationConfig::new(max_states);
    tier1_structural_state_space_stats(&net, &config)
        .map(|s| (s.states, s.edges, s.max_token_in_place, s.max_token_sum))
}

#[cfg(test)]
mod disconnected_component_tests {
    use super::*;
    use crate::petri_net::PlaceInfo;

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
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

    #[test]
    fn disconnected_components_publish_exact_product_when_monolithic_cap_is_too_small() {
        let net = PetriNet {
            name: Some("independent-counters".into()),
            places: vec![place("a"), place("b")],
            transitions: vec![
                trans("dec_a", vec![arc(0, 1)], Vec::new()),
                trans("dec_b", vec![arc(1, 1)], Vec::new()),
            ],
            initial_marking: vec![2, 1],
        };

        let stats = state_space_stats(&net, &ExplorationConfig::new(3))
            .expect("component product should finish each component exactly");

        assert_eq!(stats.states, BigUint::from(6u32));
        assert_eq!(stats.edges, BigUint::from(7u32));
        assert_eq!(stats.max_token_in_place, 2);
        assert_eq!(stats.max_token_sum, 3);
    }

    /// \>u128 disconnected-product: 60 INDEPENDENT counter components, each a
    /// place with 7 initial tokens drained by a single `dec` transition (8
    /// reachable markings, 7 edges per component). The product is
    /// `|R| = 8^60 = 2^180 ≈ 1.5e54`, FAR beyond `u128::MAX (≈3.4e38)`. Each
    /// component is decided by explicit BFS (tiny), and the combine path now
    /// carries the exact `BigUint` product instead of failing closed on the
    /// `u128` cap — the representational unblock. Cross-checked below against the
    /// closed-form 8^60, and the per-component product computed separately.
    #[test]
    fn disconnected_product_above_u128_reports_exact_bignum() {
        let comps = 60usize;
        let tokens = 7u64; // |R_i| = tokens + 1 = 8; edges_i = tokens = 7.
        let mut places = Vec::with_capacity(comps);
        let mut transitions = Vec::with_capacity(comps);
        let mut initial_marking = Vec::with_capacity(comps);
        for i in 0..comps {
            places.push(place(&format!("p{i}")));
            initial_marking.push(tokens);
            transitions.push(trans(
                &format!("dec{i}"),
                vec![arc(i as u32, 1)],
                Vec::new(),
            ));
        }
        let net = PetriNet {
            name: Some("independent-counters-60".into()),
            places,
            transitions,
            initial_marking,
        };

        // A small BFS cap so the MONOLITHIC exploration cannot complete (8^60
        // states); the disconnected-component path is the only way to a count.
        let stats = state_space_stats(&net, &ExplorationConfig::new(1000))
            .expect("disconnected-component product decides each component exactly");

        // Closed form: |R| = 8^60, edges = Σ_i (7 · 8^59) = 60 · 7 · 8^59.
        let r_each = BigUint::from(8u32);
        let expected_states = r_each.pow(60);
        assert_eq!(stats.states, expected_states, "|R| = 8^60 exactly");
        assert!(
            stats.states > BigUint::from(u128::MAX),
            "the product is genuinely > u128::MAX",
        );
        let expected_edges =
            BigUint::from(60u32) * BigUint::from(7u32) * BigUint::from(8u32).pow(59);
        assert_eq!(stats.edges, expected_edges, "edges = 60 · 7 · 8^59");
        assert_eq!(stats.max_token_in_place, 7);
        assert_eq!(stats.max_token_sum, 7 * 60, "sum of per-component max sums");

        // Cross-check |R| against ∏ of the SEPARATELY-computed per-component
        // counts (each component is a tokens-drain counter ⇒ |R_i| = 8).
        let mut product = BigUint::from(1u32);
        for _ in 0..comps {
            product *= BigUint::from(8u32);
        }
        assert_eq!(stats.states, product, "matches ∏ of per-component counts");
    }

    /// Tier-1 full-simplex recognizer on a STRONGLY-CONNECTED single-simplex
    /// net too large for BFS: a directed token ring of `d = 80` places with all
    /// `n = 80` tokens initially on place 0, with one unit transition
    /// `p_i → p_{(i+1) mod d}` per place. This is ordinary, the only covering
    /// P-invariant is `Σ p = 80` (all weights 1), the place digraph is strongly
    /// connected, and each place's structural bound is 80 — so the recognizer
    /// fires and the reachable set is EXACTLY the full simplex `{x : Σx = 80}`.
    ///
    /// `|R| = multichoose(80, 80) = C(159, 80) ≈ 4.6e46 > u128::MAX (≈3.4e38)` —
    /// FAR beyond what BFS (or the u128 carriers) could ever enumerate, so we cap
    /// the monolithic BFS so ONLY the Tier-1 closed form can answer.
    /// `edges = Σ_{t} multichoose(80, 80 − 1) = 80 · C(158, 79)` (every
    /// transition has `Σpre_t = 1 ≤ 80`). `max_token_in_place = max_token_sum =
    /// 80`.
    #[test]
    fn tier1_full_simplex_directed_ring_above_u128_reports_closed_form() {
        let d = 80usize;
        let n = 80u64;
        let mut places = Vec::with_capacity(d);
        let mut transitions = Vec::with_capacity(d);
        let mut initial_marking = vec![0u64; d];
        initial_marking[0] = n; // all tokens on place 0
        for i in 0..d {
            places.push(place(&format!("p{i}")));
            // unit move p_i -> p_{(i+1) mod d}
            let next = (i + 1) % d;
            transitions.push(trans(
                &format!("t{i}"),
                vec![arc(i as u32, 1)],
                vec![arc(next as u32, 1)],
            ));
        }
        let net = PetriNet {
            name: Some("directed-ring-80".into()),
            places,
            transitions,
            initial_marking,
        };

        // Cap monolithic BFS far below |R| so it CANNOT complete — only the
        // Tier-1 recognizer can produce a count.
        let stats = state_space_stats(&net, &ExplorationConfig::new(1000))
            .expect("Tier-1 full-simplex recognizer decides the directed ring exactly");

        // Closed forms: states = C(159, 80); edges = 80 · C(158, 79).
        let expected_states =
            tla_bignum::integer::binomial::<BigUint>(BigUint::from(159u32), BigUint::from(80u32));
        assert_eq!(
            stats.states, expected_states,
            "|R| = multichoose(80,80) = C(159,80)"
        );
        assert!(
            stats.states > BigUint::from(u128::MAX),
            "the simplex count is genuinely > u128::MAX (≈4.6e46)",
        );
        let expected_edges = BigUint::from(80u32)
            * tla_bignum::integer::binomial::<BigUint>(BigUint::from(158u32), BigUint::from(79u32));
        assert_eq!(
            stats.edges, expected_edges,
            "edges = 80 · multichoose(80,79) = 80 · C(158,79)"
        );
        assert_eq!(
            stats.max_token_in_place, 80,
            "a single place can hold all 80 tokens"
        );
        assert_eq!(stats.max_token_sum, 80, "conserved total");
    }

    /// The recognizer must DECLINE a net with a multi-input (join) transition
    /// even when it conserves tokens and carries a single covering weight-1
    /// P-invariant `Σp = n` — because such a net's reachable set can be a STRICT
    /// subset of the simplex (the verified counterexample). With BFS capped too
    /// small, the whole StateSpace lane must fail closed (return `None`) rather
    /// than emit the WRONG closed-form count.
    #[test]
    fn tier1_declines_multi_input_join_even_when_conserving() {
        // 4 places, n = 2 (one token each on p0,p1). Sync transitions:
        //   t : {p0,p1} -> {p2,p3}   (2-in/2-out, conserves count)
        //   t': {p2,p3} -> {p0,p1}
        // Σp = 2 is a covering weight-1 P-invariant, but |R| = {(1,1,0,0),
        // (0,0,1,1)} = 2, NOT the full simplex C(5,2) = 10. The recognizer must
        // DECLINE (gate 2: transitions are not one-in/one-out).
        let net = PetriNet {
            name: Some("join-sync".into()),
            places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
            transitions: vec![
                trans("t", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1), arc(3, 1)]),
                trans("t2", vec![arc(2, 1), arc(3, 1)], vec![arc(0, 1), arc(1, 1)]),
            ],
            initial_marking: vec![1, 1, 0, 0],
        };
        // The recognizer itself must DECLINE this component.
        let recognized = recognize_full_simplex_component(&net);
        assert!(
            recognized.is_none(),
            "recognizer must DECLINE a join/sync net (reachable set is a strict subset)",
        );
        // BFS can still decide it exactly (tiny): the full lane returns the
        // correct |R| = 2 via BFS fallback, never the wrong simplex count of 10.
        let stats = state_space_stats(&net, &ExplorationConfig::new(10_000))
            .expect("BFS decides the tiny net exactly");
        assert_eq!(
            stats.states,
            BigUint::from(2u32),
            "BFS |R| = 2, not the simplex 10"
        );
        assert_ne!(
            stats.states,
            BigUint::from(10u32),
            "must NOT be the (wrong) full-simplex count",
        );
    }

    /// Tier-1 recognizer agreement with BFS on a SMALL strongly-connected
    /// directed ring where BFS CAN enumerate (d=3, n=2 ⇒ |R| = C(4,2) = 6).
    /// Pins the closed form to the explicit oracle on a net both lanes decide.
    #[test]
    fn tier1_recognizer_matches_bfs_on_small_directed_ring() {
        let net = PetriNet {
            name: Some("directed-ring-3".into()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
                trans("t2", vec![arc(2, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![2, 0, 0],
        };
        let config = ExplorationConfig::new(10_000);
        let recognized =
            recognize_full_simplex_component(&net).expect("small directed ring is recognized");
        let bfs = state_space_stats_bfs(&net, &config).expect("BFS completes on tiny net");
        assert_eq!(recognized.states, bfs.states, "recognizer |R| vs BFS");
        assert_eq!(recognized.edges, bfs.edges, "recognizer edges vs BFS");
        assert_eq!(
            recognized.max_token_in_place, bfs.max_token_in_place,
            "recognizer max_in_place vs BFS",
        );
        assert_eq!(
            recognized.max_token_sum, bfs.max_token_sum,
            "recognizer max_sum vs BFS",
        );
        // Pin the closed form: |R| = C(4,2) = 6.
        assert_eq!(bfs.states, BigUint::from(6u32), "C(4,2) = 6");
    }

    /// CONSTANT-READ-ARC R-REDUCTION end-to-end (the BART shape in miniature):
    /// two independent unit-state-machine RINGS (each `d=3`, `n=1` ⇒ |R_i| = 3)
    /// coupled ONLY through a shared CONSTANT READ resource place `r` that every
    /// ring transition reads-and-returns (`pre(t,r) = post(t,r) = 1`, `init(r) =
    /// 1 >= 1`). WITHOUT the reduction the net is one connected component and the
    /// recognizer declines (the resource place makes transitions 2-in/2-out, and
    /// the digraph is not a unit state machine). WITH the reduction, `r` is
    /// stripped, the net decouples into two SC unit rings, and the closed-form
    /// product is |R| = 3 * 3 = 9 — the EXACT count of the original net (the
    /// resource place is constant, so it multiplies |R| by 1). BFS is capped
    /// below 9 so ONLY the reduced closed form can answer.
    #[test]
    fn const_read_reduction_decouples_resource_coupled_rings() {
        // places: p0,p1,p2 (ring A), p3,p4,p5 (ring B), p6 = resource r.
        // Each ring move reads-and-returns r: t: {p_i, r} -> {p_next, r}.
        let r = 6u32;
        let mut transitions = Vec::new();
        for base in [0u32, 3] {
            for i in 0..3u32 {
                let p = base + i;
                let next = base + (i + 1) % 3;
                transitions.push(trans(
                    &format!("t_{p}"),
                    vec![arc(p, 1), arc(r, 1)],
                    vec![arc(next, 1), arc(r, 1)],
                ));
            }
        }
        let net = PetriNet {
            name: Some("resource-coupled-rings".into()),
            places: vec![
                place("p0"),
                place("p1"),
                place("p2"),
                place("p3"),
                place("p4"),
                place("p5"),
                place("r"),
            ],
            // one token on each ring's place 0, one on the resource.
            initial_marking: vec![1, 0, 0, 1, 0, 0, 1],
            transitions,
        };

        // The raw net is a SINGLE connected component (r couples everything) and
        // the recognizer declines it directly (2-in/2-out transitions).
        assert!(
            recognize_full_simplex_component(&net).is_none(),
            "raw resource-coupled net is not a unit state machine",
        );
        // The strip removes exactly the resource place.
        let reduced = strip_constant_read_places(&net).expect("resource place must strip");
        assert_eq!(
            reduced.net.num_places(),
            6,
            "only the resource place r is removed"
        );
        assert_eq!(
            reduced.max_stripped_token_in_place, 1,
            "resource held 1 token"
        );
        assert_eq!(
            reduced.stripped_token_sum, 1,
            "one stripped place with 1 token"
        );
        // BFS capped below |R|=9 so only the reduced closed form can answer.
        let stats = state_space_stats(&net, &ExplorationConfig::new(4))
            .expect("const-read reduction decouples into two SC rings -> product simplex");
        assert_eq!(stats.states, BigUint::from(9u32), "|R| = 3 * 3 = 9");
        // max_token_in_place = max(reduced 1, stripped resource 1) = 1.
        assert_eq!(stats.max_token_in_place, 1);
        // max_token_sum = 2 (one token per ring) + 1 (constant resource) = 3.
        assert_eq!(
            stats.max_token_sum, 3,
            "two ring tokens + the constant resource"
        );
    }

    /// Larger >u128 version: 60 independent `d=3,n=1` unit rings coupled by a
    /// single shared constant-read resource place. |R| = 3^60 ≈ 4.2e28 (just
    /// above the BFS cap; the per-component closed form decides it). Confirms the
    /// reduction scales the way BART-PT-NNN does.
    #[test]
    fn const_read_reduction_scales_to_many_rings() {
        let rings = 60usize;
        let r = (rings * 3) as u32;
        let mut places: Vec<_> = (0..rings * 3).map(|i| place(&format!("p{i}"))).collect();
        places.push(place("r"));
        let mut initial_marking = vec![0u64; rings * 3 + 1];
        for k in 0..rings {
            initial_marking[k * 3] = 1; // one token on each ring's place 0
        }
        initial_marking[rings * 3] = 1; // resource
        let mut transitions = Vec::new();
        for k in 0..rings {
            let base = (k * 3) as u32;
            for i in 0..3u32 {
                let p = base + i;
                let next = base + (i + 1) % 3;
                transitions.push(trans(
                    &format!("t_{p}"),
                    vec![arc(p, 1), arc(r, 1)],
                    vec![arc(next, 1), arc(r, 1)],
                ));
            }
        }
        let net = PetriNet {
            name: Some("resource-coupled-rings-60".into()),
            places,
            transitions,
            initial_marking,
        };
        let stats = state_space_stats(&net, &ExplorationConfig::new(1000))
            .expect("60-ring const-read reduction -> 3^60 product simplex");
        let expected = BigUint::from(3u32).pow(60);
        assert_eq!(stats.states, expected, "|R| = 3^60 ≈ 4.2e28");
        // max_token_sum = 60 ring tokens + 1 constant resource = 61.
        assert_eq!(
            stats.max_token_sum, 61,
            "60 ring tokens + the constant resource"
        );
        assert_eq!(
            stats.max_token_in_place, 1,
            "every place holds at most 1 token"
        );
    }

    /// DECLINE when the constant-read condition FAILS via `pre != post` on some
    /// transition (a NET-CONSUMING resource — not a balanced self-loop). The
    /// strip must NOT remove the place, and with BFS capped too small the whole
    /// lane must fail closed rather than emit a wrong decoupled count.
    #[test]
    fn const_read_declines_when_pre_ne_post_net_consuming() {
        // r is consumed by t but only partially returned: pre(t,r)=2, post(t,r)=1.
        // This is NOT a constant read place (net effect -1 on r each firing).
        let r = 2u32;
        let net = PetriNet {
            name: Some("net-consuming-resource".into()),
            places: vec![place("p0"), place("p1"), place("r")],
            transitions: vec![
                // ring move p0->p1 that drains r by 1 (pre 2, post 1).
                trans("t0", vec![arc(0, 1), arc(r, 2)], vec![arc(1, 1), arc(r, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0, 5],
        };
        // The place r must NOT be stripped (pre != post on t0).
        assert!(
            strip_constant_read_places(&net).is_none(),
            "a net-consuming resource (pre != post) must NOT be stripped",
        );
        // The net IS BFS-decidable (r drains, bounded), so the lane returns the
        // CORRECT count — but never via a wrong decoupling. Pin against an
        // independent BFS over the full net.
        let bfs = state_space_stats_bfs(&net, &ExplorationConfig::new(100_000))
            .expect("BFS decides the small net");
        let lane = state_space_stats(&net, &ExplorationConfig::new(100_000))
            .expect("lane decides via BFS");
        assert_eq!(lane.states, bfs.states, "lane |R| matches full-net BFS");
        assert_eq!(lane.edges, bfs.edges, "lane edges matches full-net BFS");
    }

    /// DECLINE when the constant-read condition FAILS via `init(p) < max_t
    /// pre(t,p)` — the read guard is NOT always satisfied even though the place
    /// is a balanced self-loop (`pre == post`). Stripping it would wrongly drop a
    /// place that can DISABLE transitions, inflating |R|. The strip must refuse.
    #[test]
    fn const_read_declines_when_init_below_pre_guard() {
        // r is a balanced self-loop on t (pre=post=3) BUT init(r)=2 < 3, so t is
        // initially DISABLED and the marking of the ring is constrained. r is not
        // a *constant* read place under our gate (b).
        let r = 2u32;
        let net = PetriNet {
            name: Some("under-marked-guard".into()),
            places: vec![place("p0"), place("p1"), place("r")],
            transitions: vec![
                // p0->p1 guarded by reading 3 from r (balanced) but init(r)=2.
                trans("t0", vec![arc(0, 1), arc(r, 3)], vec![arc(1, 1), arc(r, 3)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0, 2],
        };
        assert!(
            strip_constant_read_places(&net).is_none(),
            "init(r) < max pre(t,r) ⇒ guard not always satisfied ⇒ must NOT strip",
        );
        // Cross-check the lane equals full-net BFS (here t0 is permanently
        // disabled: |R| = 2, the ring cannot move). Never the decoupled count.
        let bfs = state_space_stats_bfs(&net, &ExplorationConfig::new(100_000))
            .expect("BFS decides the tiny net");
        let lane = state_space_stats(&net, &ExplorationConfig::new(100_000))
            .expect("lane decides via BFS");
        assert_eq!(lane.states, bfs.states, "lane |R| matches full-net BFS");
    }

    /// A constant-read place with `init == max pre` (the boundary of gate (b))
    /// IS stripped (the guard is always exactly satisfiable, never disabling).
    #[test]
    fn const_read_strips_at_init_equals_pre_boundary() {
        let r = 2u32;
        let net = PetriNet {
            name: Some("boundary-guard".into()),
            places: vec![place("p0"), place("p1"), place("r")],
            transitions: vec![
                trans("t0", vec![arc(0, 1), arc(r, 3)], vec![arc(1, 1), arc(r, 3)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0, 3], // init(r) == max pre == 3
        };
        let reduced = strip_constant_read_places(&net).expect("init == pre boundary must strip");
        assert_eq!(
            reduced.net.num_places(),
            2,
            "the resource place is stripped"
        );
        // The reduced net is a 2-place ring with 1 token ⇒ |R| = 2, identical to
        // the original (r is constant). Cross-check against full-net BFS.
        let bfs = state_space_stats_bfs(&net, &ExplorationConfig::new(100_000))
            .expect("BFS decides the tiny net");
        let lane = state_space_stats(&net, &ExplorationConfig::new(100_000)).expect("lane decides");
        assert_eq!(lane.states, bfs.states, "constant resource ⇒ |R| unchanged");
        assert_eq!(lane.states, BigUint::from(2u32), "2-place ring, 1 token");
    }

    #[test]
    fn connected_net_still_fails_closed_when_monolithic_exploration_is_incomplete() {
        // The net must be unable to complete on every path so the
        // fail-closed contract is genuinely exercised. We make it
        // *unbounded*: a source transition with no input perpetually
        // refills `a`, and `move` shuttles into `b`, so the reachable set
        // is infinite. The sound DD path declines because the LP upper
        // bound is infinite (no finite per-place bound to encode) —
        // independent of the encoding cap. The net is a single connected
        // component, so the disconnected-component path also declines. With
        // BFS capped at 3 every path must fail closed: no decomposed/partial
        // counts.
        //
        // (The production DD gate admits the unary range `bound <= 16` and
        // the binary band `16 < bound <= 2^20`; a *finite* high-bound net is
        // now decided exactly by the binary path. This net declines only
        // because it is genuinely UNBOUNDED — the LP bound is infinite, so
        // no finite per-place field can encode it — independent of the
        // encoding cap. See
        // `dd_backend_branch_admits_high_bound_net_via_binary_band`.)
        let net = PetriNet {
            name: Some("connected-unbounded".into()),
            places: vec![place("a"), place("b")],
            transitions: vec![
                trans("gen", Vec::new(), vec![arc(0, 1)]), // source ⇒ a unbounded
                trans("move", vec![arc(0, 1)], vec![arc(1, 1)]),
            ],
            initial_marking: vec![0, 0],
        };

        assert!(
            state_space_stats(&net, &ExplorationConfig::new(3)).is_none(),
            "connected nets must not publish decomposed counts when BFS is incomplete",
        );
    }

    /// REGRESSION (SharedMemory-PT-000005 shape): a SINGLE connected component
    /// that the full-simplex recognizer DECLINES (it has a join/sync transition,
    /// so it is not a unit state machine) must make the Tier-1 lane a TRUE NO-OP
    /// — it must DECLINE (`tier1_structural_state_space_stats` returns `None`) so
    /// the EXISTING downstream pipeline decides the net unchanged, rather than
    /// running Tier-1's own redundant whole-net BFS (which, from the EARLY call
    /// site, bypasses the dead-reduction / DD / MDD lanes and can decline where a
    /// downstream lane would have succeeded — losing the cell to CANNOT_COMPUTE).
    ///
    /// Before the strict-additivity gate, Tier-1 ran a per-component BFS even for
    /// a single trivial component; on SharedMemory-PT-000005 that whole-net BFS
    /// was the only thing deciding the net, and any deadline/size shift flipped
    /// it to CC even though the OFF (no-Tier-1) pipeline decided it correctly.
    /// This test pins the invariant: on a single unrecognized component the
    /// Tier-1 lane returns `None`, and the lane's published answer is IDENTICAL
    /// whether Tier-1 is consulted or not (ON == OFF).
    #[test]
    fn tier1_single_unrecognized_component_is_true_no_op() {
        // A SINGLE connected component with a join/sync transition (2-in/2-out):
        //   t : {p0,p1} -> {p2,p3}      (not a unit one-in/one-out move)
        //   back: {p2,p3} -> {p0,p1}    (keeps the place digraph connected)
        // Σp = 2 is conserved but |R| = {(1,1,0,0),(0,0,1,1)} = 2 ≠ the simplex
        // C(5,2)=10, so the recognizer MUST decline — and, being one component,
        // the whole Tier-1 lane must decline too (no per-component BFS).
        let net = PetriNet {
            name: Some("sharedmemory-000005-shape".into()),
            places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
            transitions: vec![
                trans(
                    "sync",
                    vec![arc(0, 1), arc(1, 1)],
                    vec![arc(2, 1), arc(3, 1)],
                ),
                trans(
                    "back",
                    vec![arc(2, 1), arc(3, 1)],
                    vec![arc(0, 1), arc(1, 1)],
                ),
            ],
            initial_marking: vec![1, 1, 0, 0],
        };

        // It is a SINGLE component (one connected net).
        let components = independent_components(&net).expect("components computed");
        assert_eq!(
            components.len(),
            1,
            "the net is a single connected component"
        );

        // The Tier-1 lane must be a TRUE NO-OP: a single unrecognized component
        // declines the whole lane (returns None), NOT a Tier-1-internal BFS
        // result. This is the load-bearing regression assertion — even with a
        // generous BFS cap the lane must DECLINE rather than answer from its own
        // BFS, so the decline genuinely falls through to the existing pipeline.
        let config = ExplorationConfig::new(10_000);
        assert!(
            tier1_structural_state_space_stats(&net, &config).is_none(),
            "Tier-1 must DECLINE a single unrecognized component (true no-op), not \
             answer from its own redundant whole-net BFS",
        );

        // The full lane (Tier-1 ON) still decides the net via the downstream
        // pipeline, IDENTICAL to the answer a forced-decline BFS gives. |R| = 2
        // (the two synced markings), NOT the wrong simplex count of 10.
        let on = state_space_stats(&net, &ExplorationConfig::new(10_000))
            .expect("downstream pipeline decides the tiny net (ON)");
        assert_eq!(
            on.states,
            BigUint::from(2u32),
            "|R| = 2 (synced pair), not 10"
        );
        // Sanity: directly via BFS the same value (the lane just routes here).
        let bfs = state_space_stats_bfs(&net, &ExplorationConfig::new(10_000))
            .expect("BFS decides the tiny net");
        assert_eq!(on.states, bfs.states, "ON answer == downstream BFS answer");
        assert_eq!(on.edges, bfs.edges, "ON edges == downstream BFS edges");
        assert_eq!(on.max_token_in_place, bfs.max_token_in_place);
        assert_eq!(on.max_token_sum, bfs.max_token_sum);
    }
}

/// Shared symbolic-phase budget policy for the StateSpace lanes.
///
/// Returns `no_deadline_budget` when no wall-clock deadline is supplied;
/// `None` when only the [`STATE_SPACE_BFS_FALLBACK_RESERVE`] (or less) remains
/// under a finite deadline; otherwise the time above the reserve clamped to
/// `max_budget`. The BDD (`state_space_dd_budget`) and MDD
/// (`state_space_mdd_budget`) lanes share this arithmetic but pass their OWN
/// (independently-tunable) ceilings, so a per-lane ceiling change stays local.
///
/// Soundness-neutral: the budget only gates *which* exact engine answers
/// within the deadline, never a published value.
#[cfg(feature = "dd-backend")]
fn state_space_phase_budget(
    global_deadline: Option<std::time::Instant>,
    now: std::time::Instant,
    max_budget: std::time::Duration,
    no_deadline_budget: std::time::Duration,
) -> Option<std::time::Duration> {
    let Some(global_deadline) = global_deadline else {
        return Some(no_deadline_budget);
    };

    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= STATE_SPACE_BFS_FALLBACK_RESERVE {
        return None;
    }

    // Give the phase everything above the BFS reserve (the bulk of the wall
    // clock), bounded only by the absolute ceiling. This replaces the old
    // `.min(5s)`, which clipped the phase to 5s and threw away ~all of a long
    // deadline (e.g. 45s of a 60s deadline).
    let budget = max_budget.min(
        remaining
            .checked_sub(STATE_SPACE_BFS_FALLBACK_RESERVE)
            .unwrap(),
    );
    (!budget.is_zero()).then_some(budget)
}

#[cfg(feature = "dd-backend")]
fn state_space_dd_budget(
    global_deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> Option<std::time::Duration> {
    state_space_phase_budget(
        global_deadline,
        now,
        STATE_SPACE_DD_MAX_BUDGET,
        STATE_SPACE_DD_NO_DEADLINE_BUDGET,
    )
}

/// Per-branch fall-through messages for [`run_state_space_dd_worker`].
///
/// Each StateSpace symbolic lane (BDD count, BDD full-metric, MDD) logs a
/// distinct, lane-specific stderr line on each decline path (spawn failure,
/// worker decline, timeout, panic). The control flow is identical; only the
/// wording differs (lane label, "fell through" vs "declined", `Display` vs
/// `Debug` of the error), so the messages are passed in while the channel /
/// spawn / `recv_timeout` scaffolding is shared.
#[cfg(feature = "dd-backend")]
struct DdWorkerLabels<D> {
    /// Full literal logged when the OS refuses the thread spawn.
    spawn_failed: &'static str,
    /// Lane label interpolated into the timeout line
    /// (`"StateSpace: {label} exceeded {n}s budget — using BFS"`).
    timeout_label: &'static str,
    /// Full literal logged when the worker drops the sender without a value
    /// (panic / disconnect).
    panicked: &'static str,
    /// Formats the full line logged when the work closure returns `Err`
    /// (`FnOnce(E) -> String`, constrained at the [`run_state_space_dd_worker`]
    /// call site so the error type stays inferable from `work`). Takes the
    /// error by value — it is owned once received — which keeps the closure
    /// free of higher-ranked-lifetime inference.
    on_decline: D,
}

/// Run `work` (the lane-specific symbolic computation) on a detached worker
/// thread with a hard wall-clock `budget`, returning `Some(value)` only when
/// the worker finishes within `budget + 1.5s` slack and `work` returns `Ok`.
///
/// Shared spawn / channel / `recv_timeout` / decline-dispatch scaffolding for
/// the three StateSpace symbolic lanes ([`try_dd_reachable_count_timed`],
/// [`try_dd_full_metrics_timed_seeded`], [`run_mdd_metrics_timed`]). The
/// lane-specific parts stay in the caller: `work` performs the spec build,
/// deadline install, and dispatch; [`DdWorkerLabels`] carries each lane's
/// stderr wording. The worker is detached (never joined) on timeout — it drops
/// its BDD/MDD manager on the way out, so the budget is a soft cap with no
/// resource leak, keeping the soundness floor that a slow symbolic phase never
/// delays the explicit BFS fallback.
///
/// The `+1.5s` slack lets the worker surface its own clean `BudgetExceeded`
/// decline (so it stops itself) rather than detaching a still-running thread.
#[cfg(feature = "dd-backend")]
fn run_state_space_dd_worker<T, E, W, D>(
    name: &'static str,
    budget: std::time::Duration,
    work: W,
    labels: DdWorkerLabels<D>,
) -> Option<T>
where
    T: Send + 'static,
    E: Send + 'static,
    W: FnOnce() -> Result<T, E> + Send + 'static,
    D: FnOnce(E) -> String,
{
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(name.into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let _ = tx.send(work());
        });
    let Ok(_thread) = handle else {
        // OS refused the spawn — fall through to BFS rather than running the
        // symbolic phase inline (the budget would not be enforceable).
        eprintln!("{}", labels.spawn_failed);
        return None;
    };
    match rx.recv_timeout(budget + std::time::Duration::from_millis(1500)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(err)) => {
            eprintln!("{}", (labels.on_decline)(err));
            None
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "StateSpace: {} exceeded {}s budget — using BFS",
                labels.timeout_label,
                budget.as_secs(),
            );
            None
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // Worker panicked (or otherwise dropped the sender without
            // sending). Treat as a decline; BFS remains authoritative.
            eprintln!("{}", labels.panicked);
            None
        }
    }
}

/// Decision-Diagram MVP entry point with a hard time budget.
///
/// Spawns the DD computation on a worker thread and joins for at most
/// `budget`. Returns `Some(|R|)` only when:
///   - the net passes the MVP preconditions (small, bounded, low per-place tokens),
///   - the BDD fixpoint converges within `budget`,
///   - and the OxiDD computation returns successfully.
///
/// Returns `None` on timeout, precondition violation, or DD failure so
/// the explicit BFS engine remains the unique source of truth. The
/// worker thread is detached on timeout — the BDD manager it holds is
/// dropped on its way out, so no resources leak even on a budget-blown
/// run. This keeps the soundness floor: a slow DD never delays BFS.
///
/// The gate enforced inside `try_dd_reachable_count` is intentionally
/// narrower than the gate inside `tla-dd` itself so we never even
/// allocate a BDD manager for a net that would obviously fall through.
#[cfg(feature = "dd-backend")]
fn try_dd_reachable_count_timed(net: &PetriNet, budget: std::time::Duration) -> Option<u64> {
    // Build the DD spec on this thread (cheap, deterministic, no BDD work);
    // only the OxiDD fixpoint runs on the worker.
    let spec = build_dd_spec_for_net(net)?;
    run_state_space_dd_worker(
        "tla-dd-mvp",
        budget,
        move || {
            // Native ROBDD reachable-state count (oxidd engine removed). The worker's
            // detach-on-timeout bounds the caller; for a DD-eligible (bounded) net the
            // count converges. Infallible — tla-bdd always returns a count here.
            Ok::<u64, std::convert::Infallible>(
                crate::examinations::mdd_common::reachable_count_via_bdd(&spec),
            )
        },
        DdWorkerLabels {
            spawn_failed: "StateSpace: DD MVP thread spawn failed — using BFS",
            timeout_label: "DD MVP",
            panicked: "StateSpace: DD MVP worker panicked — using BFS",
            on_decline: |err| format!("StateSpace: DD MVP fell through ({err}) — using BFS"),
        },
    )
}

/// Decision-Diagram authoritative-path entry point — runs the full
/// MCC `StateSpace` metric extraction (`|R|`, edges,
/// `max_token_in_place`, `max_token_sum`) on a worker thread with a
/// hard time budget.
///
/// Promotes the DD result to authoritative when all four metrics
/// converge inside `budget`. On ANY failure (precondition violation,
/// timeout, panic, MVP iteration cap) the caller falls through to
/// explicit BFS — soundness floor preserved.
///
/// Mirrors the budget-and-thread plumbing of
/// [`try_dd_reachable_count_timed`] so a regression in one path
/// surfaces immediately in the other.
#[cfg(feature = "dd-backend")]
fn try_dd_full_metrics_timed(
    net: &PetriNet,
    budget: std::time::Duration,
) -> Option<tla_dd::DdStateSpaceMetrics> {
    try_dd_full_metrics_timed_seeded(net, budget, None)
}

/// [`try_dd_full_metrics_timed`] with an optional NUPN structure seeding
/// the DD variable order (via `dd_spec::nupn_order_seed` +
/// `tla_dd::dispatch_reachable_state_space_metrics_seeded`).
/// PERFORMANCE-ONLY: the metrics are permutation-invariant, so the seed can
/// only change whether the fixpoint converges inside `budget`, never a
/// value.
#[cfg(feature = "dd-backend")]
fn try_dd_full_metrics_timed_seeded(
    net: &PetriNet,
    budget: std::time::Duration,
    // NUPN seed was an oxidd variable-order perf hint; tla-bdd builds its own
    // order, so this is currently unused (kept for the signature).
    _nupn: Option<&crate::nupn::NupnStructure>,
) -> Option<tla_dd::DdStateSpaceMetrics> {
    let spec = build_dd_spec_for_net(net)?;
    run_state_space_dd_worker(
        "tla-bdd-statespace",
        budget,
        move || {
            // Native ROBDD full StateSpace metrics (oxidd engine removed). The
            // NUPN/P-invariant variable-order seed was an oxidd perf hint and is
            // dropped — tla-bdd builds its own order. The worker's detach-on-timeout
            // bounds the caller. Infallible — tla-bdd always returns metrics for a
            // DD-eligible net.
            Ok::<tla_dd::DdStateSpaceMetrics, std::convert::Infallible>(
                crate::examinations::mdd_common::state_space_metrics_via_bdd(&spec),
            )
        },
        DdWorkerLabels {
            spawn_failed: "StateSpace: DD full-metric thread spawn failed — using BFS",
            timeout_label: "DD full-metric",
            panicked: "StateSpace: DD full-metric worker panicked — using BFS",
            on_decline: |err| {
                format!("StateSpace: DD full-metric fell through ({err}) — using BFS")
            },
        },
    )
}

/// Translate a [`PetriNet`] into a [`tla_dd::DdNetSpec`] under the MVP
/// precondition gate. Returns `None` when any precondition fails so the
/// caller never spawns a worker thread for a net the DD MVP cannot
/// handle.
///
/// Kept as a free function (not a method on `PetriNet`) because it lives
/// behind the `dd-backend` feature flag and must be inert in default
/// builds.
#[cfg(feature = "dd-backend")]
fn build_dd_spec_for_net(net: &PetriNet) -> Option<tla_dd::DdNetSpec> {
    // Delegate to the shared, sound spec builder so StateSpace and
    // UpperBounds cannot drift apart on the soundness gate. The builder
    // bounds each place by its LP upper bound (a sound over-approximation
    // of the per-place reachable maximum), so the encoded value range is a
    // superset of every place's reachable projection. The DD reachable set
    // is therefore EXACT, and so are all four StateSpace metrics read off
    // it. Any net the gate cannot prove sound returns `None`, and the
    // caller falls back to explicit BFS. See `examinations::dd_spec`.
    //
    // This supersedes the previous crude gate (require sub-conservative,
    // total-initial ≤ 16, uniform per-place bound), which rejected many
    // bounded nets unnecessarily and only ever used the total token count
    // as the per-place bound.
    let (spec, bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;

    // StateSpace-only soundness gate: the state/edge metrics are computed by
    // exact model counting (`tla_dd`'s `Saturating<u128>` sat_count), which
    // is exact only while the current-side variable count stays `<= 127`
    // (the terminal weight is `2^vars`, which must fit `u128`). Beyond that
    // the DD count would saturate and the engine would *decline* anyway — so
    // we decline up front here to avoid burning the DD budget on a net that
    // can only fall back.
    //
    // The quantity to bound is the *actual* number of current-side BDD
    // variables `sat_count` runs over (`num_current_vars` inside `tla-dd`),
    // which depends on each place's *production encoding*: a place with
    // `bound <= 16` is unary (`bound + 1` variables), but a place with
    // `16 < bound <= 2^20` is binary/log-encoded (`ceil(log2(bound + 1))`
    // variables). We therefore sum the engine's own per-place encoded width
    // via `tla_dd::encoded_current_side_vars` — the single source of truth
    // that mirrors `tla-dd`'s `place_var_counts` (asserted equal to the real
    // `num_current_vars` by a debug_assert in `DdReachability::new_inner`).
    //
    // The previous gate summed `bound + 1` for EVERY place (the unary width
    // even for binary-encoded places), which over-declined: a single place
    // with bound 130 — only `ceil(log2(131)) = 8` binary variables — alone
    // tripped the cap at `131 > 127`, blocking the symbolic StateSpace count
    // on a net whose true current-side var count is tiny. Summing the real
    // encoded width admits these nets soundly (the `<= 127` invariant still
    // guarantees `2^vars` fits `u128`, so `sat_count` stays exact); a place
    // above the binary cap, or a net whose real var count still exceeds 127,
    // declines exactly as before — fail-closed, never a wrong count.
    //
    // NOTE: this var-count gate is intentionally *not* applied to the
    // UpperBounds DD fast-path: UpperBounds reads maxima via BDD emptiness
    // checks, never model counting, so it has no such limit.
    const MAX_CURRENT_SIDE_VARS: u64 = 127;
    let mut current_side_vars: u64 = 0;
    for &b in &bounds {
        // A bound above the binary cap would be declined by `build_sound_dd_spec`
        // already; `None` here is belt-and-braces (fail-closed → BFS).
        let w = tla_dd::encoded_current_side_vars(b)?;
        current_side_vars = current_side_vars.saturating_add(w);
    }
    if current_side_vars > MAX_CURRENT_SIDE_VARS {
        return None;
    }
    Some(spec)
}

/// The MDD StateSpace lane's OWN admission gate — DECOUPLED from the BDD lane's
/// `MAX_CURRENT_SIDE_VARS = 127` model-counting cap.
///
/// The 127-var cap exists because the BDD lane's `sat_count` weights a terminal
/// by `2^vars`, which must fit `u128`; it is a property of the BIT-BLASTED BDD
/// encoding. The MDD spends ONE LEVEL PER PLACE with `bound+1` edges — no
/// bit-blasting, no `2^vars` terminal weight — so it does NOT inherit that
/// limit. A net the BDD gate rejects on the var cap (Anderson / TokenRing /
/// Kanban / Philosophers / FMS), but whose reachable set the MDD can represent,
/// should therefore RUN on the MDD.
///
/// The MDD's resource model is instead:
///   - the SAME sound per-place LP bounds + structural gates as the BDD lane
///     (via [`crate::examinations::dd_spec::build_sound_dd_spec`]), so the
///     encoded value range is a superset of every place's reachable projection
///     and the MDD reachable set — and all four metrics — are EXACT;
///   - a bound on total edge-width (`Σ (bound[p]+1)`) so a single huge-bound
///     place cannot make per-node child vectors blow memory (the MDD's node
///     arena holds a `bound+1`-entry vector per node at a level). The deeper
///     node-count + wall-clock budget inside `tla-mdd` itself
///     (`MAX_INTERIOR_NODES`, the deadline) catches the rest — fail-closed:
///     any MDD decline / budget overrun falls through to BFS.
///
/// Returns the sound spec, or `None` (fall through to the BDD-or-BFS path
/// unchanged) when a gate fails. SOUNDNESS: identical to the BDD spec gate
/// minus the BDD-only var cap; the count-representability ceiling is enforced
/// downstream by the MDD's `u128` fail-closed count
/// (`count_markings_u128` declines past `u128::MAX`).
#[cfg(feature = "dd-backend")]
fn build_mdd_spec_for_net(net: &PetriNet) -> Option<tla_dd::DdNetSpec> {
    let (spec, bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;

    // Total edge-width gate: bound `Σ (bound[p] + 1)` so the per-node child
    // vectors stay affordable. Generous (the MDD is compact in NODES even when
    // edge-width is large), but it fail-closes a pathological single-place
    // bound near `2^20` that would allocate a million-entry vector per node.
    // Picked well above every structured MCC family the lane targets (a
    // Philosophers / Kanban / TokenRing place bound is tiny) while still
    // capping the per-node-vector blowup the interior-NODE-count budget does
    // not see.
    const MAX_TOTAL_EDGE_WIDTH: u128 = 1 << 22; // ~4.2M edges across all levels
    let mut total_edge_width: u128 = 0;
    for &b in &bounds {
        total_edge_width = total_edge_width.saturating_add(b as u128 + 1);
        if total_edge_width > MAX_TOTAL_EDGE_WIDTH {
            return None;
        }
    }
    Some(spec)
}

/// MDD-lane phase budget when no wall-clock deadline is supplied. Small (like
/// the DD no-deadline budget) so a deadline-less invocation falls back to BFS
/// promptly. Production MCC always supplies a deadline.
#[cfg(feature = "dd-backend")]
const STATE_SPACE_MDD_NO_DEADLINE_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);
/// Floor ceiling on the MDD phase under a finite deadline — the pre-scaling
/// fixed cap, kept as the LOWER bound of [`state_space_mdd_budget`] so no
/// configuration ever gives the MDD less time than the historical policy.
///
/// HISTORY (v8 diagnosis, 2026-07-10): this used to be the whole ceiling —
/// `min(12s, remaining − 10s)` — inherited from the BDD lane's
/// "converges fast-or-never" tuning. That predates the MDD engine's GC +
/// sifting + apply-cache (July wave), which made LONG saturation runs
/// productive; the measured effect was that 13/16 StateSpace sample cells
/// burned the whole 240 s budget in explicit BFS while the MDD — the engine
/// class the field's StateSpace leaders (tedd/ITS-Tools) win with at
/// 35–3400 s — was clipped to 12 s at ANY deadline. The scaled policy in
/// [`state_space_mdd_budget`] fixes that.
#[cfg(feature = "dd-backend")]
const STATE_SPACE_MDD_MAX_BUDGET: std::time::Duration = std::time::Duration::from_secs(12);

/// Minimum explicit-BFS reserve under the scaled MDD budget policy. The BFS
/// tail keeps at least this much (or 25% of the remaining budget, whichever is
/// larger), so BFS-winnable nets — which in the measured samples finish in
/// seconds — are never starved by a non-converging MDD phase.
#[cfg(feature = "dd-backend")]
const STATE_SPACE_MDD_SCALED_BFS_RESERVE_FLOOR: std::time::Duration =
    std::time::Duration::from_secs(30);

/// Kill-switch for the MDD StateSpace lane. The lane is ON by default; set
/// `TY_MCC_DISABLE_MDD_STATESPACE` to a truthy value (`1`/`on`/`true`/`yes`) to
/// disable it and fall back to the BDD-or-BFS behavior unchanged.
///
/// SOUNDNESS-NEUTRAL either way: disabling the lane can only make a net decline
/// to BFS (exact) instead of being decided by the MDD (also exact). It never
/// changes a published value.
#[cfg(feature = "dd-backend")]
fn mdd_state_space_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_MDD_STATESPACE").is_ok_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// The legacy oxidd BDD StateSpace lane is **RETIRED by default**: the native
/// MDD lane subsumes its coverage and is formally proven sound, so the BDD lane
/// no longer runs unless explicitly re-enabled with `TY_MCC_ENABLE_DD_STATESPACE`.
///
/// Evidence for the retirement (SOUNDNESS-FIRST):
/// - **Formal proof:** `tla_mdd::MddNet::verify_saturation_inductive_fixpoint`
///   discharges `init ∈ R ∧ ∀t. image_t(R) ⊆ R` (⇒ `R ⊇ reachable`, no marking
///   missed), proven structurally over 4096 random nets + the regime battery.
/// - **Subsumption A/B** (`scripts/mdd_bdd_statespace_subsumption_ab.sh`): on a
///   148-model family-spanning corpus spread, disabling the BDD lane gave
///   IDENTICAL exact-unit coverage (0 regression, 0 wrong) — the MDD lane (+ BFS)
///   decides everything the BDD lane did.
///
/// SOUNDNESS-NEUTRAL either way (both lanes are exact; the gate only chooses
/// which engine decides). The enable flag is retained for rollback / further
/// A/B until the `oxidd` dependency is fully dropped.
#[cfg(feature = "dd-backend")]
fn dd_state_space_disabled() -> bool {
    !std::env::var("TY_MCC_ENABLE_DD_STATESPACE").is_ok_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// MDD phase budget under the global deadline — DEADLINE-SCALED (v8 diagnosis,
/// 2026-07-10): the MDD gets everything above a BFS reserve that grows with
/// the budget (`max(30 s, remaining/4)`), and never LESS than the historical
/// fixed policy `min(12 s, remaining − 10 s)`. Examples: 240 s cell → ~180 s
/// MDD / 60 s BFS (was 12 s / 228 s); contest 3600 s → 2700 s MDD (was 12 s);
/// 22 s remaining → 12 s (identical to the old policy).
///
/// `None` ⇒ skip the MDD (only the BFS reserve remains, or the deadline
/// expired). Soundness-neutral (budget only gates *which* exact engine runs —
/// both are exact and the MDD adoption stays behind the fail-closed
/// cross-validation gate in `try_mdd_full_metrics_gated`).
#[cfg(feature = "dd-backend")]
fn state_space_mdd_budget(
    global_deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> Option<std::time::Duration> {
    let Some(global_deadline) = global_deadline else {
        return Some(STATE_SPACE_MDD_NO_DEADLINE_BUDGET);
    };
    let remaining = global_deadline.saturating_duration_since(now);
    if remaining <= STATE_SPACE_BFS_FALLBACK_RESERVE {
        return None;
    }
    // Legacy floor: the old fixed-ceiling policy — no configuration gets a
    // smaller MDD phase than before the scaling change.
    let legacy =
        STATE_SPACE_MDD_MAX_BUDGET.min(remaining.saturating_sub(STATE_SPACE_BFS_FALLBACK_RESERVE));
    // Deadline-scaled: everything above the growing BFS reserve.
    let reserve = STATE_SPACE_MDD_SCALED_BFS_RESERVE_FLOOR.max(remaining / 4);
    let scaled = remaining.saturating_sub(reserve);
    let budget = legacy.max(scaled);
    (!budget.is_zero()).then_some(budget)
}

/// Adapter: a [`tla_dd::DdNetSpec`] → [`tla_mdd::MddNet`], field-for-field.
///
/// `MddNet` mirrors `DdNetSpec` exactly (`bounds`, `initial_marking`, and a
/// transition list with per-place `pre`/`post`), so the MDD lane consumes the
/// IDENTICAL net the BDD lane does — built by the same `build_dd_spec_for_net`
/// soundness gate. Sharing the spec is what makes the two lanes' answers
/// directly cross-validatable.
#[cfg(feature = "dd-backend")]
fn dd_spec_to_mdd_net(spec: &tla_dd::DdNetSpec) -> tla_mdd::MddNet {
    tla_mdd::MddNet {
        bounds: spec.bounds.clone(),
        initial_marking: spec.initial_marking.clone(),
        transitions: spec
            .transitions
            .iter()
            .map(|t| tla_mdd::MddTransition {
                pre: t.pre.clone(),
                post: t.post.clone(),
            })
            .collect(),
    }
}

/// Convert an MDD metric bundle into [`StateSpaceStats`].
///
/// `StateSpaceStats::states` is now `u128`, and the MDD bundle carries the
/// exact reachable-set size as a `u128` (`state_count_u128`), so the state
/// count maps straight through — including reachable sets larger than
/// `u64::MAX` (high-bound counter / Philosophers nets ≈ 1e23) that the BDD lane
/// and the explicit BFS observer can never report. The MDD engine has ALREADY
/// fail-closed (declined with `CountError::CountOverflow`) for any net whose
/// `|R| > u128::MAX` (e.g. FMS ≈ 1e47), so a bundle reaching here is always
/// representable. The other three metrics are `u64` and map straight through.
#[cfg(feature = "dd-backend")]
fn mdd_metrics_to_stats(metrics: &tla_mdd::MddStateSpaceMetrics) -> Option<StateSpaceStats> {
    // The MDD bundle now carries the EXACT |R| / edges as `BigUint`
    // (`state_count_big` / `edge_count_big`), so a reachable set whose count
    // exceeds `u128` (e.g. FMS ≈1e47, Kanban/Philosophers ≈1e238) maps straight
    // through and is REPORTED instead of declining on the carrier. The MDD
    // engine never declines on count magnitude any more (only on resource caps),
    // so a bundle reaching here is always representable.
    Some(StateSpaceStats {
        states: metrics.state_count_big.clone(),
        edges: metrics.edge_count_big.clone(),
        max_token_in_place: metrics.max_token_in_place,
        max_token_sum: metrics.max_token_sum,
    })
}

/// Run the MDD StateSpace lane on a worker thread with a hard budget, then
/// apply the CROSS-VALIDATED soundness gate before returning a verdict.
///
/// Returns `Some(stats)` only when the MDD bundle is SOUND to adopt:
///   - **case (a)** — `dd_metrics` is `Some` (the BDD lane produced metrics):
///     every MDD metric MUST equal the BDD metric exactly. A mismatch is
///     impossible given the crate's BFS/BDD cross-check battery, but we gate
///     anyway: on any mismatch we DECLINE (return `None`), keeping the
///     fail-closed floor.
///   - **case (b)** — `dd_metrics` is `None` (the BDD lane DECLINED, the target
///     case): the MDD answer is adopted because it is exact-by-construction
///     (the four metrics are pinned to the BFS oracle, 0 disagreements), after
///     a debug-only sample cross-check (`debug_assert`) of `|R|` against an
///     independent explicit BFS on the spec.
///
/// On the kill-switch, an over-large net the spec gate rejects, an MDD decline
/// (overflow / node budget / deadline / panic), or a failed gate, returns
/// `None` so the caller falls through to BFS UNCHANGED.
#[cfg(feature = "dd-backend")]
fn try_mdd_full_metrics_gated(
    net: &PetriNet,
    config: &ExplorationConfig,
    dd_metrics: Option<&tla_dd::DdStateSpaceMetrics>,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Option<StateSpaceStats> {
    if mdd_state_space_disabled() {
        return None;
    }
    let budget = state_space_mdd_budget(config.deadline(), std::time::Instant::now())?;
    // The MDD lane's OWN admission gate (sound per-place LP bounds + structural
    // gates + an edge-width cap), DECOUPLED from the BDD lane's `2^vars` model-
    // counting var cap. This is gap (b): nets the BDD gate rejects on the
    // 127-var cap (Anderson / TokenRing / Kanban / Philosophers / FMS) but
    // whose `|R|` the MDD can represent now RUN on the MDD. Fail-closed: a
    // declined spec, an MDD budget overrun, or a `> u128` count all fall
    // through to BFS.
    let spec = build_mdd_spec_for_net(net)?;

    // Variable ordering — the MDD scale lever the BDD lane already pulls but the
    // MDD lane historically did not. Node-level saturation's peak node count is
    // acutely sensitive to the place→level order; an arbitrary PNML order blows
    // the interior-node budget on nets a good order keeps compact. Apply the
    // span-guarded FORCE place order (identity when it cannot strictly improve
    // the transition span, so a net whose PNML order is already good is never
    // made worse). SOUND: `permute_spec` is an isomorphic relabeling of places,
    // and every StateSpace metric (state/edge count, max-token-in-place,
    // max-token-sum) is a function of the reachable-marking SET, hence invariant
    // under a place bijection (see `tla_dd::order` soundness note). So the metrics
    // — and the BDD-parity gate + BFS cross-check below — are unchanged; only the
    // MDD size (feasibility) improves. Seed the FORCE search with the NUPN
    // unit-hierarchy block order when the model carries one (span-guarded, so the
    // seed only helps or no-ops).
    let seed =
        nupn.and_then(|n| crate::examinations::dd_spec::nupn_order_seed(n, net.num_places()));
    let order = tla_dd::force_place_order_seeded(&spec, seed.as_deref());
    let spec = tla_dd::permute_spec(&spec, &order);

    let metrics = run_mdd_metrics_timed(&spec, budget)?;

    // --- Gate case (a): BDD metrics present ⇒ MUST match exactly. ---
    // The BDD `state_count` is `u64`; compare against the MDD's WIDENED
    // (`u128`) count so the equality is exact across the type boundary (a BDD
    // count always fits `u64 ⊆ u128`).
    if let Some(dd) = dd_metrics {
        if metrics.state_count_u128 != dd.state_count as u128
            || metrics.edge_count != dd.edge_count as u128
            || metrics.max_token_in_place != dd.max_token_in_place
            || metrics.max_token_sum != dd.max_token_sum
        {
            eprintln!(
                "StateSpace: MDD metrics disagreed with BDD metrics \
                 (MDD R={} E={} mip={} ms={}; BDD R={} E={} mip={} ms={}) — \
                 DECLINING MDD (fail-closed), falling back to BFS",
                metrics.state_count_u128,
                metrics.edge_count,
                metrics.max_token_in_place,
                metrics.max_token_sum,
                dd.state_count,
                dd.edge_count,
                dd.max_token_in_place,
                dd.max_token_sum,
            );
            return None;
        }
        // Matched the BDD lane exactly: sound to adopt.
        return mdd_metrics_to_stats(&metrics);
    }

    // --- Gate case (b): BDD declined ⇒ adopt the exact MDD answer. ---
    // Debug-only independent sample cross-check of |R| (release builds rely on
    // the soaked BFS/BDD cross-check battery; the value is exact-by-construction
    // and any narrowing is handled by `mdd_metrics_to_stats`).
    //
    // The explicit BFS oracle is `u64` AND enumerates every marking, so it is
    // only a usable cross-check when `|R|` fits `u64` (i.e. the MDD's narrowed
    // `state_count` is `Some`). For the WIDE case — the whole point of the u128
    // widening, where `|R| > u64::MAX` (high-bound counter / Philosophers nets)
    // — explicit BFS cannot enumerate the set at all, so the soundness gate is
    // the soaked saturation-vs-relprod-vs-BFS differential battery in `tla-mdd`,
    // not a per-run BFS replay. We skip the debug replay in that case.
    if let Some(state_count_u64) = metrics.state_count {
        debug_assert_eq!(
            state_count_u64,
            tla_dd::bfs_reachable_set_count(&spec),
            "MDD |R| disagreed with explicit BFS on the same spec (debug cross-check)",
        );
    }
    eprintln!(
        "StateSpace: MDD lane decided the net the BDD lane declined \
         (|R|={}, edges={}, max_in_place={}, max_sum={})",
        metrics.state_count_big,
        metrics.edge_count_big,
        metrics.max_token_in_place,
        metrics.max_token_sum,
    );
    mdd_metrics_to_stats(&metrics)
}

/// Run [`tla_mdd::MddNet::state_space_metrics`] on a detached worker thread
/// with a hard time budget. Mirrors the DD lane's thread/budget plumbing so a
/// regression in one path surfaces in the other. Returns `None` on decline,
/// timeout, spawn failure, or panic (fail-closed → BFS).
#[cfg(feature = "dd-backend")]
fn run_mdd_metrics_timed(
    spec: &tla_dd::DdNetSpec,
    budget: std::time::Duration,
) -> Option<tla_mdd::MddStateSpaceMetrics> {
    let mdd_net = dd_spec_to_mdd_net(spec);
    run_state_space_dd_worker(
        "tla-mdd-statespace",
        budget,
        move || {
            // The MDD engine takes an optional wall-clock deadline directly and
            // declines (fail-closed) rather than overrun it.
            let deadline = std::time::Instant::now() + budget;
            mdd_net.state_space_metrics(Some(deadline))
        },
        DdWorkerLabels {
            spawn_failed: "StateSpace: MDD thread spawn failed — using BFS",
            timeout_label: "MDD lane",
            panicked: "StateSpace: MDD lane worker panicked — using BFS",
            on_decline: |err| format!("StateSpace: MDD lane declined ({err:?}) — using BFS"),
        },
    )
}

#[cfg(all(test, feature = "dd-backend"))]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceInfo, TransitionInfo};
    use std::time::{Duration, Instant};

    /// The deadline-scaled MDD budget (v8 diagnosis 2026-07-10): scales with
    /// the remaining wall clock, keeps the growing BFS reserve, and never
    /// gives the MDD less than the historical `min(12 s, remaining − 10 s)`.
    #[test]
    fn mdd_budget_scales_with_deadline_and_keeps_bfs_reserve() {
        let now = Instant::now();
        let at = |secs: u64| Some(now + Duration::from_secs(secs));

        // 240 s sweep cell: reserve = max(30, 60) = 60 → MDD 180 s (was 12 s).
        assert_eq!(
            state_space_mdd_budget(at(240), now),
            Some(Duration::from_secs(180))
        );
        // Contest 3600 s: reserve = 900 → MDD 2700 s (was 12 s).
        assert_eq!(
            state_space_mdd_budget(at(3600), now),
            Some(Duration::from_secs(2700))
        );
        // 45 s: scaled = 45 − 30 = 15 beats legacy min(12, 35) = 12.
        assert_eq!(
            state_space_mdd_budget(at(45), now),
            Some(Duration::from_secs(15))
        );
        // 22 s: scaled saturates to 0 (reserve 30 > 22); legacy floor keeps
        // the OLD policy exactly: min(12, 22−10) = 12.
        assert_eq!(
            state_space_mdd_budget(at(22), now),
            Some(Duration::from_secs(12))
        );
        // At/below the hard BFS fallback reserve: skip the MDD entirely.
        assert_eq!(state_space_mdd_budget(at(10), now), None);
        assert_eq!(state_space_mdd_budget(at(3), now), None);
        // No deadline: the small prompt-fallback budget, unchanged.
        assert_eq!(
            state_space_mdd_budget(None, now),
            Some(STATE_SPACE_MDD_NO_DEADLINE_BUDGET)
        );
    }

    /// Monotone in `remaining`: more wall clock can never shrink the MDD
    /// phase (no cliff where a larger budget yields a smaller phase).
    #[test]
    fn mdd_budget_is_monotone_in_remaining() {
        let now = Instant::now();
        let mut prev = Duration::ZERO;
        for secs in (11..=4000).step_by(7) {
            let b = state_space_mdd_budget(Some(now + Duration::from_secs(secs)), now)
                .unwrap_or(Duration::ZERO);
            assert!(
                b >= prev,
                "budget shrank from {prev:?} to {b:?} at remaining={secs}s"
            );
            prev = b;
        }
    }

    /// DD metrics map to `StateSpaceStats` losslessly: the DD `state_count`
    /// (`u64`) widens into the `u128` `states` field with no narrowing hazard
    /// (`u64 ⊆ u128`), so even a `u64::MAX` count round-trips EXACTLY rather
    /// than declining. The `usize` narrowing the old version guarded against is
    /// now performed — fail-closed via `states_wide` — only at the final
    /// `StateSpaceReport` boundary, tested by
    /// [`state_space_report_carries_wide_state_count_without_truncation`].
    #[test]
    fn dd_metrics_widen_state_count_losslessly() {
        // In-range metrics must map straight through (the production path).
        let ok = dd_metrics_to_stats(&tla_dd::DdStateSpaceMetrics {
            state_count: 42,
            edge_count: 7,
            max_token_in_place: 3,
            max_token_sum: 9,
            iterations: 1,
        })
        .expect("in-range DD metrics must publish");
        assert_eq!(ok.states, BigUint::from(42u32));
        assert_eq!(ok.edges, BigUint::from(7u32));
        assert_eq!(ok.max_token_in_place, 3);
        assert_eq!(ok.max_token_sum, 9);

        // A `u64::MAX` count widens EXACTLY into the `BigUint` `states` field
        // (no decline, no truncation): the wider carrier removes the hazard at
        // this boundary entirely.
        let wide = dd_metrics_to_stats(&tla_dd::DdStateSpaceMetrics {
            state_count: u64::MAX,
            edge_count: 0,
            max_token_in_place: 0,
            max_token_sum: 0,
            iterations: 0,
        })
        .expect("u64 count always fits the BigUint states field");
        assert_eq!(
            wide.states,
            BigUint::from(u64::MAX),
            "u64::MAX widens exactly"
        );
    }

    /// Report boundary, BEYOND u128: a count that exceeds `u128::MAX` must NOT
    /// be truncated — the EXACT value is carried in the `BigUint` field and the
    /// emitted `STATE_SPACE STATES` / `TRANSITIONS` rows reproduce the full
    /// decimal. This is the representational unblock at the emission boundary.
    #[test]
    fn state_space_report_emits_full_bignum_decimal_above_u128() {
        use crate::examination::{ExaminationRecord, ExaminationValue, StateSpaceReport};
        // 2^200 ≈ 1.6e60, FAR beyond u128::MAX (≈3.4e38). Edges: a distinct big
        // value (3^130 ≈ 2.5e62) to confirm the TRANSITIONS row is also bignum.
        let states_big = BigUint::from(2u32).pow(200);
        let edges_big = BigUint::from(3u32).pow(130);
        let report = StateSpaceReport::from_big(states_big.clone(), edges_big.clone(), 3, 9);
        // The exact-count accessors reproduce the full bignum; the narrowed
        // back-compat fields saturate (markers).
        assert_eq!(
            report.states_exact(),
            &states_big,
            "states must round-trip exact"
        );
        assert_eq!(
            report.edges_exact(),
            &edges_big,
            "edges must round-trip exact"
        );
        assert_eq!(
            report.states,
            usize::MAX,
            "narrowed usize saturates (marker)"
        );
        assert_eq!(report.edges, u128::MAX, "narrowed u128 saturates (marker)");
        let record = ExaminationRecord::with_techniques(
            "StateSpace".to_string(),
            ExaminationValue::StateSpace(Some(report)),
            crate::output::Techniques::default(),
        );
        let lines = record.to_mcc_line();
        assert!(
            lines.contains(&states_big.to_string()),
            "emitted STATE_SPACE STATES must carry the exact 2^200, got:\n{lines}",
        );
        assert!(
            lines.contains(&edges_big.to_string()),
            "emitted STATE_SPACE TRANSITIONS must carry the exact 3^130, got:\n{lines}",
        );
    }

    #[test]
    fn dd_budget_preserves_bfs_reserve_under_deadline() {
        let now = Instant::now();

        assert_eq!(
            state_space_dd_budget(None, now),
            Some(STATE_SPACE_DD_NO_DEADLINE_BUDGET),
            "no-deadline runs keep the conservative no-deadline budget",
        );
        // StateSpace DD converges fast-or-never, so the budget is capped at the
        // modest `STATE_SPACE_DD_MAX_BUDGET` (12s) rather than handed the bulk of
        // the wall clock: a 600s deadline still gives DD only the 12s cap, leaving
        // the remaining ~588s to the exact BFS workhorse (the fix for the
        // BridgeAndVehicles DD-starves-BFS regression). The cap binds here.
        assert_eq!(
            state_space_dd_budget(
                Some(now + STATE_SPACE_BFS_FALLBACK_RESERVE + Duration::from_secs(600)),
                now,
            ),
            Some(STATE_SPACE_DD_MAX_BUDGET),
            "a long deadline is capped to the modest DD ceiling, leaving BFS the bulk",
        );
        // The absolute ceiling still bounds a pathologically large deadline.
        assert_eq!(
            state_space_dd_budget(
                Some(
                    now + STATE_SPACE_BFS_FALLBACK_RESERVE
                        + STATE_SPACE_DD_MAX_BUDGET
                        + Duration::from_secs(100),
                ),
                now,
            ),
            Some(STATE_SPACE_DD_MAX_BUDGET),
            "an over-large deadline is clamped to the absolute ceiling",
        );
        assert_eq!(
            state_space_dd_budget(
                Some(now + STATE_SPACE_BFS_FALLBACK_RESERVE + Duration::from_millis(250)),
                now,
            ),
            Some(Duration::from_millis(250)),
            "tight bounded runs cap DD to the time above the BFS reserve",
        );
        assert_eq!(
            state_space_dd_budget(Some(now + STATE_SPACE_BFS_FALLBACK_RESERVE), now),
            None,
            "DD is skipped when only the BFS reserve remains",
        );
        // `now - Duration` is an intentional past Instant; checked_sub would change
        // the value type (Option<Instant>) and the expected None result.
        #[allow(clippy::unchecked_time_subtraction)]
        {
            assert_eq!(
                state_space_dd_budget(Some(now - Duration::from_millis(1)), now),
                None,
                "expired deadlines skip DD and fall through to BFS",
            );
        }
    }

    /// Build the 2-place swap net (identical to `tla_dd`'s smoke test)
    /// as a real `PetriNet`, then call `try_dd_reachable_count_timed`
    /// with a generous budget and assert |R| = 2. Exercises the
    /// feature-gated DD branch end-to-end including the budget plumbing.
    #[test]
    fn dd_backend_branch_handles_two_place_swap() {
        let net = PetriNet {
            name: Some("swap".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t10".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        };
        let count = try_dd_reachable_count_timed(&net, std::time::Duration::from_secs(30))
            .expect("DD MVP must handle the swap net within the budget");
        assert_eq!(count, 2, "swap net has two reachable markings");
    }

    /// Second case: 10-state synthetic net (3 places, 2 transitions:
    /// producer/consumer chain with initial marking (2,0,0)). Asserts
    /// the timed branch returns exactly |R|=6 to match the reference
    /// BFS in `tla_dd::tests::test_dd_reachability_matches_bfs_on_small_net`.
    /// Closes the loop: the petri-side gate, the threaded budget, and
    /// the DD engine all agree on the same value.
    #[test]
    fn dd_backend_branch_matches_bfs_count_on_chain_net() {
        let net = PetriNet {
            name: Some("chain".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p2".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t1".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(2),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![2, 0, 0],
        };
        let count = try_dd_reachable_count_timed(&net, std::time::Duration::from_secs(30))
            .expect("chain net is inside the MVP gate");
        assert_eq!(
            count, 6,
            "producer-consumer chain has six reachable markings"
        );
    }

    /// Binary band ADMISSION on the StateSpace path: a 2-place conserved
    /// shuttle of 17 tokens (`p0 + p1 = 17`) has per-place LP bound 17 > 16
    /// (the unary cap), which the binary (log-encoded) field represents
    /// exactly. Since `MAX_PER_PLACE_BOUND` was raised to
    /// `tla_dd::MAX_BINARY_PLACE_BOUND`, the StateSpace gate now ADMITS the
    /// net (the binary `apply` non-termination was fixed by the isolated
    /// big-stack worker + node-ceiling/deadline guards), so a DD spec is
    /// built and the count path decides exactly: the reachable markings are
    /// {(17,0),(16,1),...,(0,17)} ⇒ 18 markings. The node ceiling + deadline
    /// keep this memory- and time-bounded; a net that overflowed either
    /// would still DECLINE (fall back to BFS), never OOM, never crash.
    #[test]
    fn dd_backend_branch_admits_high_bound_net_via_binary_band() {
        let net = PetriNet {
            name: Some("shuttle17".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t10".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![17, 0],
        };
        // Bound 17 <= MAX_PER_PLACE_BOUND (now 2^20) ⇒ the gate ADMITS the
        // net (binary band) and builds a DD spec. The conserved sum bounds
        // each place at 17, well inside the var-count gate.
        assert!(
            build_dd_spec_for_net(&net).is_some(),
            "17-token net (bound 17 <= binary cap 2^20) must now be DD-eligible",
        );
        // And the count path decides exactly, memory- and time-bounded.
        let count = try_dd_reachable_count_timed(&net, std::time::Duration::from_secs(30))
            .expect("17-token conserved shuttle is decided by the binary DD path");
        assert_eq!(
            count, 18,
            "conserved shuttle p0+p1=17 has 18 reachable markings (17,0)..(0,17)",
        );
    }

    /// Regression for the count-gate var-count fix: a conserved shuttle with
    /// 130 tokens. The LP bound for each of the two places is 130, so the
    /// OLD gate — which summed the *unary* width `Σ(bound + 1) = 131 + 131 =
    /// 262` — DECLINED (262 > 127), falling back to BFS even though the net
    /// is trivially decided symbolically. Under the binary encoding each
    /// place is `ceil(log2(131)) = 8` bits, so the REAL current-side var
    /// count is `8 + 8 = 16 <= 127` and the count stays exact. The fixed gate
    /// sums `tla_dd::encoded_current_side_vars` (the real encoded width) and
    /// therefore ADMITS the net; the DD count path decides it exactly:
    /// reachable markings {(130,0),(129,1),...,(0,130)} ⇒ 131 markings.
    ///
    /// This is the case the increment unlocks — a moderately-bounded place
    /// whose unary width alone exceeds 127 but whose binary width is tiny.
    #[test]
    fn dd_count_gate_admits_net_whose_unary_width_exceeds_127_but_binary_does_not() {
        let net = PetriNet {
            name: Some("shuttle130".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t10".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![130, 0],
        };
        // The OLD unary-width sum would be 131 + 131 = 262 > 127 (decline).
        // Assert the new gate's quantity (the real encoded width) is <= 127.
        let (_, bounds) =
            crate::examinations::dd_spec::build_sound_dd_spec(&net).expect("net is DD-eligible");
        let unary_width: u64 = bounds.iter().map(|&b| b + 1).sum();
        assert!(
            unary_width > 127,
            "fixture must exceed the OLD unary gate (got {unary_width})",
        );
        let real_width: u64 = bounds
            .iter()
            .map(|&b| tla_dd::encoded_current_side_vars(b).expect("within cap"))
            .sum();
        assert!(
            real_width <= 127,
            "real binary width must be inside the gate (got {real_width})",
        );

        // The fixed gate must ADMIT the net (old gate declined it).
        assert!(
            build_dd_spec_for_net(&net).is_some(),
            "130-token shuttle (binary width 16 <= 127) must be DD-eligible now",
        );
        // And the count path decides it exactly.
        let count = try_dd_reachable_count_timed(&net, std::time::Duration::from_secs(30))
            .expect("130-token conserved shuttle is decided by the binary DD count path");
        assert_eq!(
            count, 131,
            "conserved shuttle p0+p1=130 has 131 reachable markings (130,0)..(0,130)",
        );
    }

    /// Authoritative-path test: on an amenable net the DD full-metric
    /// helper must produce all four metrics, matching the explicit
    /// BFS values we already pin in `dd_backend_branch_matches_bfs_count_on_chain_net`
    /// for `|R|`. State space:
    ///   - markings: {(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)} ⇒ |R|=6
    ///   - edges: from each marking, sum over enabled transitions.
    ///     (2,0,0): t0 enabled → (1,1,0). 1 edge.
    ///     (1,1,0): t0,t1 enabled → (0,2,0),(1,0,1). 2 edges.
    ///     (1,0,1): t0 enabled → (0,1,1). 1 edge.
    ///     (0,2,0): t1 enabled → (0,1,1). 1 edge.
    ///     (0,1,1): t1 enabled → (0,0,2). 1 edge.
    ///     (0,0,2): no transition enabled.
    ///     Total edges = 6.
    ///   - max_token_in_place = 2 (initial).
    ///   - max_token_sum = 2 (conservation: every reachable marking sums
    ///     to 2).
    ///
    /// Pins the authoritative DD path end-to-end: spec construction,
    /// budget plumbing, full-metric extraction, all four MCC fields.
    #[test]
    fn dd_backend_full_metrics_authoritative_path_on_chain_net() {
        let net = PetriNet {
            name: Some("chain-full-metrics".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p2".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t0".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t1".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(2),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![2, 0, 0],
        };
        let metrics = try_dd_full_metrics_timed(&net, std::time::Duration::from_secs(30))
            .expect("chain net is inside the DD full-metric gate");
        assert_eq!(metrics.state_count, 6, "|R| from authoritative DD path");
        assert_eq!(metrics.edge_count, 6, "edges from authoritative DD path");
        assert_eq!(
            metrics.max_token_in_place, 2,
            "max_token_in_place from authoritative DD path",
        );
        assert_eq!(
            metrics.max_token_sum, 2,
            "max_token_sum from authoritative DD path",
        );
    }

    /// Differential: on the swap net the DD authoritative path must
    /// agree on all four metrics with explicit reasoning.
    /// Reachable markings: {(1,0),(0,1)}. Edges: each marking enables
    /// exactly one transition (t01 from (1,0), t10 from (0,1)) ⇒ edges=2.
    /// max_token_in_place=1, max_token_sum=1.
    #[test]
    fn dd_backend_full_metrics_authoritative_path_on_swap_net() {
        let net = PetriNet {
            name: Some("swap-full-metrics".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t10".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        };
        let metrics = try_dd_full_metrics_timed(&net, std::time::Duration::from_secs(30))
            .expect("swap net is inside the DD full-metric gate");
        assert_eq!(metrics.state_count, 2);
        assert_eq!(metrics.edge_count, 2);
        assert_eq!(metrics.max_token_in_place, 1);
        assert_eq!(metrics.max_token_sum, 1);
    }

    /// Regression for the sound-LP-bounds upgrade (roadmap item 2): a
    /// **non-conservative but bounded** net. The old crude gate rejected
    /// any net with a transition whose total output weight exceeded its
    /// total input weight (`post_sum > pre_sum`), so this net would have
    /// fallen through to BFS. The LP-based gate proves each place is
    /// finitely bounded (p0 ≤ 1, p1 ≤ 2), so the DD path now fires and
    /// must produce metrics matching the explicit BFS exactly.
    ///
    /// Net: p0 --t(x2)--> p1, initial (1, 0). t consumes 1 from p0 and
    /// produces 2 into p1, then is dead (p0 never refilled).
    ///   - Reachable: {(1,0), (0,2)} ⇒ |R| = 2.
    ///   - Edges: (1,0) →t (0,2); (0,2) has no enabled transition ⇒ 1.
    ///   - max_token_in_place = 2 (the `2` in p1 of (0,2)).
    ///   - max_token_sum = 2.
    #[test]
    fn dd_backend_admits_non_conservative_bounded_net_and_matches_bfs() {
        let net = PetriNet {
            name: Some("doubler".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![TransitionInfo {
                id: "t".into(),
                name: None,
                inputs: vec![Arc {
                    place: crate::petri_net::PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: crate::petri_net::PlaceIdx(1),
                    weight: 2,
                }],
            }],
            initial_marking: vec![1, 0],
        };

        // The net is non-conservative: the old gate rejected it.
        let post_sum: u64 = net.transitions[0].outputs.iter().map(|a| a.weight).sum();
        let pre_sum: u64 = net.transitions[0].inputs.iter().map(|a| a.weight).sum();
        assert!(
            post_sum > pre_sum,
            "test net must be non-conservative to exercise the upgrade",
        );

        // The sound LP gate now admits it.
        assert!(
            build_dd_spec_for_net(&net).is_some(),
            "sound LP gate must admit a finitely-bounded non-conservative net",
        );

        // DD metrics must match the explicit BFS exactly (differential).
        let config = ExplorationConfig::new(10_000);
        let bfs = state_space_stats_bfs(&net, &config).expect("BFS completes on tiny net");
        let dd = try_dd_full_metrics_timed(&net, std::time::Duration::from_secs(30))
            .expect("DD full-metric path now fires on this net");
        assert_eq!(BigUint::from(dd.state_count), bfs.states, "|R| DD vs BFS");
        assert_eq!(BigUint::from(dd.edge_count), bfs.edges, "edges DD vs BFS");
        assert_eq!(
            dd.max_token_in_place, bfs.max_token_in_place,
            "max_token_in_place DD vs BFS",
        );
        assert_eq!(
            dd.max_token_sum, bfs.max_token_sum,
            "max_token_sum DD vs BFS",
        );

        // Pin the hand-computed ground truth too.
        assert_eq!(bfs.states, BigUint::from(2u32));
        assert_eq!(bfs.edges, BigUint::from(1u32));
        assert_eq!(bfs.max_token_in_place, 2);
        assert_eq!(bfs.max_token_sum, 2);
    }

    /// Build a 1-token shuttle on `n` places (a conserved ring fragment).
    fn shuttle_net(initial_p0: u64, bound: u64) -> PetriNet {
        PetriNet {
            name: Some("mdd-shuttle".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t10".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: crate::petri_net::PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![initial_p0, bound - initial_p0],
        }
    }

    /// The adapter must reproduce the net field-for-field: the MDD metrics
    /// computed off the adapted spec must match the explicit BFS observer on
    /// the original net, for ALL FOUR metrics. This is the end-to-end win
    /// example: a conserved shuttle of 17 tokens (`p0+p1=17`, 18 markings) the
    /// MDD decides exactly off a tiny diagram.
    #[test]
    fn mdd_lane_matches_bfs_on_conserved_shuttle_all_four_metrics() {
        let net = shuttle_net(17, 17);
        let config = ExplorationConfig::new(10_000);
        let bfs = state_space_stats_bfs(&net, &config).expect("BFS completes on tiny net");

        let spec = build_dd_spec_for_net(&net).expect("conserved shuttle is DD-eligible");
        let metrics = run_mdd_metrics_timed(&spec, std::time::Duration::from_secs(30))
            .expect("MDD lane decides the conserved shuttle");
        assert_eq!(metrics.state_count_big, bfs.states, "|R| MDD vs BFS");
        assert_eq!(metrics.edge_count_big, bfs.edges, "edges MDD vs BFS");
        assert_eq!(
            metrics.max_token_in_place, bfs.max_token_in_place,
            "max_token_in_place MDD vs BFS",
        );
        assert_eq!(
            metrics.max_token_sum, bfs.max_token_sum,
            "max_token_sum MDD vs BFS",
        );

        // Pin the ground truth: 18 markings (17,0)..(0,17); each marking
        // (except endpoints fire one transition, interior fire two) — let BFS
        // be the authority, but pin |R| and the conserved sum.
        assert_eq!(bfs.states, BigUint::from(18u32));
        assert_eq!(bfs.max_token_in_place, 17);
        assert_eq!(bfs.max_token_sum, 17, "p0+p1 conserved at 17");
    }

    /// Gate case (b): the BDD lane DECLINED (`dd_metrics = None`) ⇒ the MDD
    /// answer is adopted, exact, matching BFS on all four metrics.
    #[test]
    fn mdd_gate_adopts_when_bdd_declined() {
        let net = shuttle_net(13, 17);
        let config = ExplorationConfig::new(10_000);
        let bfs = state_space_stats_bfs(&net, &config).expect("BFS completes");

        let stats = try_mdd_full_metrics_gated(&net, &config, None, None)
            .expect("MDD adopts (BDD declined)");
        assert_eq!(stats.states, bfs.states);
        assert_eq!(stats.edges, bfs.edges);
        assert_eq!(stats.max_token_in_place, bfs.max_token_in_place);
        assert_eq!(stats.max_token_sum, bfs.max_token_sum);
    }

    /// Gate case (a): the BDD lane produced metrics that MATCH the MDD ⇒ adopt.
    #[test]
    fn mdd_gate_adopts_when_bdd_metrics_match() {
        let net = shuttle_net(5, 17);
        let config = ExplorationConfig::new(10_000);
        let bfs = state_space_stats_bfs(&net, &config).expect("BFS completes");

        // Synthesize a BDD metric bundle equal to the (exact) ground truth.
        let dd = tla_dd::DdStateSpaceMetrics {
            state_count: bfs.states.to_u64().expect("fits u64"),
            edge_count: bfs.edges.to_u64().expect("fits u64"),
            max_token_in_place: bfs.max_token_in_place,
            max_token_sum: bfs.max_token_sum,
            iterations: 1,
        };
        let stats = try_mdd_full_metrics_gated(&net, &config, Some(&dd), None)
            .expect("MDD adopts (matches BDD)");
        assert_eq!(stats.states, bfs.states);
        assert_eq!(stats.edges, bfs.edges);
        assert_eq!(stats.max_token_in_place, bfs.max_token_in_place);
        assert_eq!(stats.max_token_sum, bfs.max_token_sum);
    }

    /// Gate case (a) DECLINE arm: if the (hypothetical) BDD metrics DISAGREE
    /// with the MDD, the gate DECLINES (returns `None`) — fail-closed, never a
    /// wrong adoption. This can't happen given the cross-check battery, but the
    /// gate must still refuse to adopt a mismatched answer.
    #[test]
    fn mdd_gate_declines_on_bdd_mismatch() {
        let net = shuttle_net(5, 17);
        let config = ExplorationConfig::new(10_000);

        // A deliberately WRONG BDD bundle (state_count off by one).
        let bogus = tla_dd::DdStateSpaceMetrics {
            state_count: 99_999,
            edge_count: 0,
            max_token_in_place: 0,
            max_token_sum: 0,
            iterations: 1,
        };
        assert!(
            try_mdd_full_metrics_gated(&net, &config, Some(&bogus), None).is_none(),
            "gate must DECLINE when MDD disagrees with the BDD bundle",
        );
    }

    /// GAP (b): the MDD-OWN admission gate is DECOUPLED from the BDD's 127
    /// current-side-var cap.
    ///
    /// We build a net whose LP per-place upper bound forces BINARY encoding
    /// (bound > 16 ⇒ `ceil(log2(bound+1))` BDD vars) on enough places that the
    /// summed current-side var count exceeds 127 — so `build_dd_spec_for_net`
    /// DECLINES on its `MAX_CURRENT_SIDE_VARS` cap. The SAME net has a tiny MDD
    /// edge-width (`Σ (bound+1)`), so `build_mdd_spec_for_net` ADMITS it. This
    /// is the structural decoupling that lets the MDD run on the var-cap-
    /// rejected families (Anderson / TokenRing / Kanban / Philosophers).
    ///
    /// Construction: 30 places, each fed by a dedicated transition that moves a
    /// weight-30 batch from a shared 1-bounded source, so each place's LP upper
    /// bound is 30 (binary: `ceil(log2(31)) = 5` vars). 30 × 5 = 150 > 127 ⇒
    /// BDD gate rejects; MDD edge-width = 30 × 31 + source ≈ 932 ≪ 2^22 ⇒ MDD
    /// gate admits.
    #[test]
    fn mdd_gate_decoupled_from_bdd_var_cap() {
        let n = 30usize;
        let batch = 30u64;
        let mut places = vec![PlaceInfo {
            id: "src".into(),
            name: None,
        }];
        let mut transitions = Vec::new();
        let mut initial_marking = vec![batch]; // source holds one batch
        for i in 0..n {
            let p = i + 1;
            places.push(PlaceInfo {
                id: format!("p{p}"),
                name: None,
            });
            initial_marking.push(0);
            // src --(weight batch)--> p_i : gives p_i an LP bound of `batch`.
            transitions.push(TransitionInfo {
                id: format!("fill{p}"),
                name: None,
                inputs: vec![Arc {
                    place: crate::petri_net::PlaceIdx(0),
                    weight: batch,
                }],
                outputs: vec![Arc {
                    place: crate::petri_net::PlaceIdx(p as u32),
                    weight: batch,
                }],
            });
        }
        let net = PetriNet {
            name: Some("var-cap-decouple".into()),
            places,
            transitions,
            initial_marking,
        };

        // The BDD spec gate must REJECT this net on its var cap (binary width
        // sum > 127), while the MDD-own gate ADMITS it.
        assert!(
            build_dd_spec_for_net(&net).is_none(),
            "BDD gate must reject: summed binary current-side vars exceed the 127 cap"
        );
        assert!(
            build_mdd_spec_for_net(&net).is_some(),
            "MDD-own gate must ADMIT the var-cap-rejected net (its edge-width is tiny) — \
             this is the gap-(b) decoupling"
        );
    }

    /// Kill-switch: with `TY_MCC_DISABLE_MDD_STATESPACE` set, the lane is inert
    /// and returns `None` regardless of how amenable the net is.
    #[test]
    fn mdd_kill_switch_disables_lane() {
        // Serialize env mutation within this test; other tests do not read it.
        let net = shuttle_net(5, 17);
        let config = ExplorationConfig::new(10_000);
        // Single-threaded within this test; we restore immediately.
        crate::env_guard::set_var("TY_MCC_DISABLE_MDD_STATESPACE", "1");
        let disabled = try_mdd_full_metrics_gated(&net, &config, None, None);
        crate::env_guard::remove_var("TY_MCC_DISABLE_MDD_STATESPACE");
        assert!(disabled.is_none(), "kill-switch must disable the MDD lane");
    }

    /// The adapter maps a `DdNetSpec` field-for-field into an `MddNet`.
    #[test]
    fn adapter_mirrors_dd_spec_field_for_field() {
        let net = shuttle_net(7, 17);
        let spec = build_dd_spec_for_net(&net).expect("DD-eligible");
        let mdd = dd_spec_to_mdd_net(&spec);
        assert_eq!(mdd.bounds, spec.bounds);
        assert_eq!(mdd.initial_marking, spec.initial_marking);
        assert_eq!(mdd.transitions.len(), spec.transitions.len());
        for (m, d) in mdd.transitions.iter().zip(&spec.transitions) {
            assert_eq!(m.pre, d.pre);
            assert_eq!(m.post, d.post);
        }
    }
}
