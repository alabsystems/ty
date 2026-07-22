// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! UpperBounds execution pipeline: reduction, slicing, and exploration orchestration.

use std::time::Instant;

use crate::circulation_loop::reduce_query_local_circulation_loops_fixpoint;
use crate::error::PnmlError;
use crate::explorer::{explore_observer, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::petri_net::{PetriNet, PlaceIdx};
use crate::property_xml::Property;
use crate::query_slice::{build_query_local_slice, QuerySlice};
#[cfg(test)]
use crate::reduction::apply_query_guarded_prefire;
use crate::reduction::{
    apply_final_place_gcd_scaling, reduce_iterative_structural_query_with_protected,
    reduce_query_guarded, ParallelExpandingObserver, ReducedNet,
};
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};
use crate::stubborn::PorStrategy;

use super::model::{
    assemble_upper_bounds_results, monotone_query_initial_bound,
    prepare_upper_bounds_properties_with_aliases, structural_query_bound, BoundTracker,
    PreparedUpperBoundProperty,
};
use super::observer::UpperBoundsObserver;

use crate::examinations::query_support::{
    relevance_cone_on_reduced_net, upper_bounds_support, visible_transitions_for_support,
};

/// Optional ground-truth pair for the UpperBounds soundness fallback.
///
/// The default Safety-Net-B fallback uses `ReducedNet::identity(net)` as its
/// "ground truth" — that's correct when the only reduction layer applied
/// before reaching this pipeline is the local structural-reduction layer. The
/// COL pipeline runs colored-relevance reduction + per-property unfolding
/// BEFORE calling here, so the per-property `net` argument is itself a
/// **reduced** net. In that case the caller passes the truly unmodified
/// unfolded net via this struct so the fallback re-resolves trackers from
/// `aliases` and explores the genuine state space.
///
/// When the BFS on the per-property reduced net produced a value that
/// disagrees with the ground-truth net's reachable maximum (either
/// over- or under-counted), the ground-truth-net value is authoritative.
pub(crate) struct GroundTruthNet<'a> {
    pub net: &'a PetriNet,
    pub aliases: &'a PropertyAliases,
}

/// Build an `ExplorationConfig` with POR for UpperBounds.
///
/// Visible transitions = transitions whose input or output arcs touch any
/// query place (after reduction mapping). When query places span the entire
/// net, `compute_stubborn_set` detects all-visible and returns `None`,
/// so the overhead is a single boolean scan per state.
fn upper_bounds_por_config(
    reduced: &ReducedNet,
    trackers: &[BoundTracker],
    config: &ExplorationConfig,
) -> ExplorationConfig {
    let base = config.clone();

    let unresolved_place_sets: Vec<Vec<PlaceIdx>> = trackers
        .iter()
        .filter(|tracker| !tracker.is_structurally_resolved())
        .map(|tracker| tracker.place_indices.clone())
        .collect();

    match upper_bounds_support(reduced, &unresolved_place_sets)
        .and_then(|support| visible_transitions_for_support(&reduced.net, &support))
    {
        Some(visible) => base.with_por(PorStrategy::SafetyPreserving { visible }),
        None => base,
    }
}

#[cfg(test)]
pub(super) fn apply_upper_bounds_prefire(
    reduced: &mut ReducedNet,
    trackers: &[BoundTracker],
) -> Result<bool, PnmlError> {
    let unresolved_place_sets: Vec<Vec<PlaceIdx>> = trackers
        .iter()
        .filter(|tracker| !tracker.is_structurally_resolved())
        .map(|tracker| tracker.place_indices.clone())
        .collect();
    if unresolved_place_sets.is_empty() {
        return Ok(false);
    }

    match upper_bounds_support(reduced, &unresolved_place_sets) {
        Some(support) => apply_query_guarded_prefire(reduced, &support.places),
        None => Ok(false),
    }
}

pub(super) fn build_upper_bounds_slice(
    reduced: &ReducedNet,
    trackers: &[BoundTracker],
) -> Option<(QuerySlice, Vec<usize>, Vec<BoundTracker>)> {
    let unresolved_slots: Vec<usize> = trackers
        .iter()
        .enumerate()
        .filter_map(|(slot, tracker)| (!tracker.is_structurally_resolved()).then_some(slot))
        .collect();
    if unresolved_slots.is_empty() {
        return None;
    }

    let unresolved_place_sets: Vec<Vec<PlaceIdx>> = unresolved_slots
        .iter()
        .map(|&slot| trackers[slot].place_indices.clone())
        .collect();
    let support = upper_bounds_support(reduced, &unresolved_place_sets)?;
    let protected_seed_places = support.places.clone();
    let cone = relevance_cone_on_reduced_net(&reduced.net, support);
    let slice = build_query_local_slice(&reduced.net, &cone)?;
    // Rule H: contract circulation loops inside the query-local slice.
    // UpperBounds is place-only, so this is always safe.
    let slice =
        reduce_query_local_circulation_loops_fixpoint(slice.clone(), &protected_seed_places)
            .unwrap_or(slice);
    let original_to_slice = slice.compose_place_map(&reduced.place_map);

    let remapped_trackers = unresolved_slots
        .iter()
        .map(|&slot| {
            let mut tracker = trackers[slot].clone();
            tracker.place_indices = tracker
                .place_indices
                .iter()
                .map(|place| original_to_slice[place.0 as usize])
                .collect::<Option<Vec<_>>>()?;
            Some(tracker)
        })
        .collect::<Option<Vec<_>>>()?;

    Some((slice, unresolved_slots, remapped_trackers))
}

pub(super) fn explore_upper_bounds_on_reduced_net(
    reduced: &ReducedNet,
    trackers: Vec<BoundTracker>,
    config: &ExplorationConfig,
) -> Result<(crate::explorer::ExplorationResult, Vec<BoundTracker>), PnmlError> {
    let por_config = upper_bounds_por_config(reduced, &trackers, config);
    let mut observer = UpperBoundsObserver::new(trackers);
    let result = {
        let mut expanding = ParallelExpandingObserver::new(reduced, &mut observer);
        let result = explore_observer(&reduced.net, &por_config, &mut expanding);
        if let Some(error) = expanding.take_error() {
            return Err(error);
        }
        result
    };
    Ok((result, observer.into_trackers()))
}

fn explore_upper_bounds_on_slice(
    slice: &QuerySlice,
    trackers: Vec<BoundTracker>,
    config: &ExplorationConfig,
) -> (crate::explorer::ExplorationResult, Vec<BoundTracker>) {
    let reduced = ReducedNet::identity(&slice.net);
    let por_config = upper_bounds_por_config(&reduced, &trackers, config);
    let mut observer = UpperBoundsObserver::new(trackers);
    let result = explore_observer(&slice.net, &por_config, &mut observer);
    (result, observer.into_trackers())
}

/// Decide whether `tracker.max_bound` — as observed by the BFS — is a *confirmed
/// reachable* value on `net`, i.e. a genuine reachability witness rather than a
/// possibly-inflated reduced-net artifact.
///
/// **Why this matters (soundness):** [`confirm_observed_maxima_with_trap_lp`]
/// ratifies `max_bound` as the EXACT bound by proving only the *ceiling*
/// (`max_bound + 1` is unreachable). A ceiling alone does not establish that
/// `max_bound` itself is reachable. When the BFS that produced `max_bound` ran
/// on a *reduced* net, a reduction rule (GCD/arc-weight scaling, self-loop
/// stripping, lateral fusion, P-invariant reconstruction, …) can make the
/// reduced-net observation EXCEED the true original-net maximum for the query.
/// Ratifying such an over-count yields a too-high — i.e. WRONG — UpperBounds
/// answer. MCC UpperBounds requires the exact maximum; a too-high bound is a
/// wrong answer, never merely imprecise.
///
/// We return `true` only when `max_bound` is *provably* reachable without
/// trusting the correctness of any individual reduction rule:
///
/// * **Trivial witness:** `max_bound == Σ initial_marking[query]`. The initial
///   marking is always a reachable state, so its query sum is always witnessed.
/// * **Faithful passthrough:** every query place survives the reduction as a
///   pure 1:1 mapping with unit scale and is not the target of any value
///   transformation (constant fold, P-invariant reconstruction, or lateral
///   fusion). In that regime the reduced marking's value for the place *is*
///   literally the original place's value, so the observed expanded sum is a
///   real reachable original-net marking by construction — independent of
///   whether any reduction rule is itself UpperBounds-preserving.
///
/// Any query place that the reduction transformed (scaled, removed and
/// reconstructed/folded, or fused) makes the observation *unwitnessed*: the
/// expand-to-original map could have synthesised a value that no reachable
/// original marking attains, so we must not certify `max_bound` as exact off a
/// ceiling proof alone. This is strictly soundness-preserving: declining a
/// witness can only cost coverage (the tracker stays unresolved and the
/// safety-net / exact path searches for the true reachable max), never
/// correctness.
pub(super) fn reduced_observation_witnesses_max(
    reduced: &ReducedNet,
    net: &PetriNet,
    tracker: &BoundTracker,
) -> bool {
    // Trivial witness: the observed maximum is exactly the initial-marking
    // query sum, which is always reachable.
    //
    // SOUNDNESS: `initial_sum` is in *original-net* token units, but
    // `tracker.max_bound` was observed on `reduced.net` — and
    // `apply_final_place_gcd_scaling` may have divided a query place's tokens by
    // its arc-weight GCD (`place_scales[orig] > 1`), so the reduced observation
    // is in *scaled* units. Comparing the two across different units can make a
    // GCD-scaled query coincidentally satisfy `max_bound == initial_sum` and
    // wrongly ratify an inflated/deflated bound as exact. The trivial witness is
    // only valid when every query place is unit-scale (same coordinate system as
    // the original initial marking); otherwise fall through to the per-place
    // passthrough check below (which rejects scaled places at the
    // `place_scales != 1` guard), leaving the tracker for the exact search.
    let initial_sum: u64 = tracker
        .place_indices
        .iter()
        .map(|p| net.initial_marking[p.0 as usize])
        .sum();
    let query_unit_scale = tracker
        .place_indices
        .iter()
        .all(|&PlaceIdx(orig)| reduced.place_scales[orig as usize] == 1);
    if query_unit_scale && tracker.max_bound == initial_sum {
        return true;
    }

    // Otherwise every query place must be a faithful 1:1 passthrough: unit
    // scale, mapped to a surviving reduced place, and untouched by any
    // value-transforming reduction. Then the reduced marking carries the
    // place's true value and the observed expanded sum is a real reachable
    // original-net marking.
    tracker.place_indices.iter().all(|&PlaceIdx(orig)| {
        let orig = orig as usize;
        // Unit scale and a live 1:1 mapping.
        if reduced.place_scales[orig] != 1 || reduced.place_map[orig].is_none() {
            return false;
        }
        // Not a folded constant.
        if reduced
            .constant_values
            .iter()
            .any(|&(PlaceIdx(p), _)| p as usize == orig)
        {
            return false;
        }
        // Not a P-invariant reconstruction target.
        if reduced
            .reconstructions
            .iter()
            .any(|recon| recon.place.0 as usize == orig)
        {
            return false;
        }
        // Not a lateral-fusion duplicate (value derived from a canonical).
        if reduced
            .report
            .lateral_fusions
            .iter()
            .any(|merge| merge.duplicate.0 as usize == orig)
        {
            return false;
        }
        true
    })
}

/// Ratify each unresolved tracker's observed maximum as the EXACT bound when
/// trap-aware LP proves the *ceiling* (`max_bound + 1` is unreachable) **and**
/// the observed `max_bound` is a confirmed reachable witness.
///
/// `witnessed(i)` reports whether `trackers[i].max_bound` is reachability-faithful
/// (see [`reduced_observation_witnesses_max`] for the reduced-net case;
/// identity-net observations are always witnessed by construction). The ceiling
/// alone only proves `true_max <= max_bound`; ratifying `max_bound` as *exact*
/// additionally requires `true_max >= max_bound`, which the witness supplies.
/// Without the witness an inflated reduced-net `max_bound` (where the reduction
/// over-counted the query) would be wrongly published as a too-high bound — so
/// we leave such trackers unresolved for the safety-net / exact downward search.
pub(super) fn confirm_observed_maxima_with_trap_lp_witnessed(
    net: &PetriNet,
    trackers: &mut [BoundTracker],
    witnessed: impl Fn(usize) -> bool,
) -> usize {
    let mut confirmed = 0usize;
    for (i, tracker) in trackers.iter_mut().enumerate() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        // Soundness gate: never ratify an unwitnessed (possibly inflated)
        // observation as exact off a ceiling proof alone.
        if !witnessed(i) {
            continue;
        }
        let Some(threshold) = tracker.max_bound.checked_add(1) else {
            continue;
        };
        let impossible_stronger_witness = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(threshold),
            ResolvedIntExpr::TokensCount(tracker.place_indices.clone()),
        );
        if crate::lp_state_equation::lp_unreachable_with_traps(net, &impossible_stronger_witness) {
            tracker.lp_bound = Some(
                tracker
                    .lp_bound
                    .map_or(tracker.max_bound, |bound| bound.min(tracker.max_bound)),
            );
            confirmed += 1;
        }
    }
    confirmed
}

/// Trap-LP ratification for observations that are reachability-faithful by
/// construction (identity-net BFS markings are genuine original-net markings).
///
/// Every tracker is treated as witnessed. Use this only when `max_bound` was
/// observed on the un-reduced (identity) net; for reduced-net observations use
/// [`confirm_observed_maxima_with_trap_lp_witnessed`] with a per-tracker witness
/// predicate so a reduction over-count is never ratified as exact.
pub(super) fn confirm_observed_maxima_with_trap_lp(
    net: &PetriNet,
    trackers: &mut [BoundTracker],
) -> usize {
    confirm_observed_maxima_with_trap_lp_witnessed(net, trackers, |_| true)
}

/// Singleton residual retry (LP-only): for each unresolved tracker, refine the
/// LP cap downward by trap-aware LP dichotomy. Each iteration calls
/// `lp_unreachable_with_traps` to check whether `tokens >= current_lp` is
/// unreachable; if so, `lp_bound` drops by one. Trap LP adds polyhedral
/// trap-set cuts on top of the state equation, so it can prove tighter
/// unreachability than the bare `lp_upper_bound` from initial seeding.
///
/// Sound by construction: trap-LP unreachability is a polynomial-time
/// over-approximation of true unreachability, so any threshold it rules out is
/// genuinely unreachable, and the LP cap is a sound upper bound on the true
/// reachable maximum. The refinement only ratifies the tracker when the new
/// `lp_bound` matches the observed `max_bound` — i.e. the BFS lower bound and
/// LP upper bound coincide. If trap-LP would push `lp_bound` BELOW `max_bound`
/// (the BFS observation is inconsistent with the trap-LP cap, indicating an
/// unsound upstream reduction inflated max_bound), the refinement stops at
/// `lp_bound == max_bound` and we do not adopt a lower cap, because emitting a
/// value > `lp_bound` would be wrong.
///
/// Used to free deadline before identity-net BFS in Safety-Net A/B/C paths:
/// in per-property COL calls the single-worker identity BFS dominates wall
/// time, so pre-resolving via LP avoids burning the rest of the budget on
/// state-space exploration that won't terminate.
///
/// Returns the number of trackers newly resolved.
fn singleton_residual_retry_lp_only(net: &PetriNet, trackers: &mut [BoundTracker]) -> usize {
    let unresolved_count = trackers
        .iter()
        .filter(|t| !t.is_structurally_resolved())
        .count();
    if unresolved_count == 0 || unresolved_count > 4 {
        // Bound the per-tracker work: only fire when residuals are few.
        return 0;
    }

    let mut newly_resolved = 0usize;
    for tracker in trackers.iter_mut() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        // Walk lp_bound downward via trap-LP; bound iterations to avoid
        // pathological loops on degenerate nets. Stop at lp == max_bound so
        // we never push lp_bound BELOW the observed BFS max — that would
        // ratify an inflated max_bound as exact (Lamport-PT-2 regression:
        // reduced-net BFS observes 2, trap-LP could prove 2 unreachable,
        // but emitting `max_bound=2` after dropping `lp_bound` to 1 would
        // publish a wrong answer of 2).
        let mut iter_budget = 16usize;
        while iter_budget > 0 {
            iter_budget -= 1;
            let Some(current_lp) = tracker.lp_bound else {
                break;
            };
            if current_lp <= tracker.max_bound {
                break;
            }
            let witness = ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(current_lp),
                ResolvedIntExpr::TokensCount(tracker.place_indices.clone()),
            );
            if crate::lp_state_equation::lp_unreachable_with_traps(net, &witness) {
                tracker.lp_bound = Some(current_lp - 1);
            } else {
                break;
            }
        }
        if tracker.is_structurally_resolved() {
            newly_resolved += 1;
        }
    }
    newly_resolved
}

/// `true` iff `TY_MCC_DISABLE_UPPER_BOUNDS_WALK` is set to `1`/`on`/`true`. The
/// walk is a strict under-approximation that only ever RAISES a query's achievable
/// lower bound (`BoundTracker::max_bound`) toward the existing sound upper bound;
/// it never lowers an upper bound and never publishes a value on its own. Disabling
/// it is therefore always answer-preserving — it merely removes a cheap pinning
/// shortcut, falling through to the unchanged reduce+BFS / DD / trap-LP pipeline.
/// Mirrors `one_safe_walk_disabled()` and the other `TY_MCC_DISABLE_*` switches.
fn upper_bounds_walk_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_UPPER_BOUNDS_WALK")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
}

/// Random-walk achievable-maximum lane for UpperBounds (LOWER-bound only).
///
/// Runs EARLY on the RAW `net` (never a reduced net — a reduction can remap or
/// rescale token counts, so a place index no longer means the same thing), AFTER
/// `seed_upper_bounds_exactness` has populated the sound structural/LP/monotone
/// UPPER bounds and BEFORE the heavy reduce+BFS / DD path. It fires only enabled
/// transitions from the initial marking and, for every reachable marking it
/// visits, records each query's observed token SUM. It then raises each tracker's
/// achievable lower bound `max_bound = max(max_bound, observed)`.
///
/// Soundness: the observed sum is a value the query genuinely attains in some
/// reachable marking (`observed <= true_max`), so raising `max_bound` to it can
/// never exceed the true maximum. The EXISTING `is_structurally_resolved()` then
/// pins a query EXACTLY when its raised lower bound meets the sound upper bound
/// (`max_bound >= effective_bound` ⇒ lower == upper ⇒ the true maximum). The lane
/// NEVER lowers an upper bound and NEVER reports the observed value as the answer
/// on its own; a query whose observed max stays below its upper bound stays
/// unresolved and falls through to the unchanged pipeline (no regression).
///
/// Budget: ADDITIVE / leftover-only. `under_approx_lane_deadline` grants at most
/// `min(remaining / 4, 8s)` and reserves the exhaustive-BFS tail, so the lane can
/// NEVER starve the downstream lanes. A skip/already-expired sentinel means there
/// is no leftover slice — fall through without walking.
fn run_upper_bounds_walk_pass(
    net: &PetriNet,
    trackers: &mut [BoundTracker],
    deadline: Option<Instant>,
) -> usize {
    if upper_bounds_walk_disabled() {
        return 0;
    }
    // Only walk for queries that are still unresolved AND have a known upper
    // bound to meet (an unbounded query can never be pinned by a lower bound, so
    // walking it cannot help). We still pass through resolved/unbounded slots so
    // index alignment with `trackers` is trivial; their target is `None` (resolved
    // ones are skipped on write-back) so they never gate the early stop.
    if trackers.iter().all(|t| t.is_structurally_resolved()) {
        return 0;
    }

    let walk_deadline =
        crate::examinations::reachability::under_approx_lane_deadline(net, deadline);
    if walk_deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        // No leftover slice — fall through without walking.
        return 0;
    }

    let queries: Vec<Vec<usize>> = trackers
        .iter()
        .map(|t| t.place_indices.iter().map(|p| p.0 as usize).collect())
        .collect();
    // Target = the effective UPPER bound for already-unresolved trackers; for
    // resolved ones use `Some(0)` so they are trivially "pinned" (observed >= 0
    // always) and therefore drop OUT of the early-stop gate — leaving the gate to
    // depend only on the genuinely-unresolved queries. An unresolved tracker with
    // no cap at all (`effective_bound() == None`) keeps a `None` target: it can
    // never be pinned by a lower bound, so it never lets the walk stop early (it
    // simply tracks its observed max for the whole budget). The walk uses targets
    // only to decide when it can stop early; it never emits the target as a value.
    let targets: Vec<Option<u64>> = trackers
        .iter()
        .map(|t| {
            if t.is_structurally_resolved() {
                Some(0)
            } else {
                t.effective_bound()
            }
        })
        .collect();

    let observed = crate::examinations::reachability_walk::run_random_walk_upper_bounds(
        net,
        &queries,
        &targets,
        walk_deadline,
    );

    let mut newly_resolved = 0usize;
    for (tracker, &obs) in trackers.iter_mut().zip(observed.iter()) {
        if tracker.is_structurally_resolved() {
            continue;
        }
        // Raise the achievable LOWER bound only (monotone, never lowers the cap).
        tracker.max_bound = tracker.max_bound.max(obs);
        if tracker.is_structurally_resolved() {
            newly_resolved += 1;
        }
    }
    newly_resolved
}

fn seed_upper_bounds_exactness(net: &PetriNet, trackers: &mut [BoundTracker]) -> (usize, usize) {
    let invariants = crate::invariant::compute_p_invariants(net);
    let mut exact_proof_count = 0usize;
    for tracker in trackers.iter_mut() {
        tracker.structural_bound = structural_query_bound(&invariants, &tracker.place_indices);
        tracker.max_bound = tracker
            .place_indices
            .iter()
            .map(|p| net.initial_marking[p.0 as usize])
            .sum();
        tracker.monotone_bound = monotone_query_initial_bound(net, &tracker.place_indices);
        if tracker.is_structurally_resolved() {
            exact_proof_count += 1;
        }
    }

    let mut lp_count = 0usize;
    for tracker in trackers.iter_mut() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        if let Some(bound) = crate::lp_state_equation::lp_upper_bound(net, &tracker.place_indices) {
            tracker.lp_bound = Some(bound);
            lp_count += 1;
            if tracker.is_structurally_resolved() {
                exact_proof_count += 1;
            }
        }
    }

    (exact_proof_count, lp_count)
}

/// A2 (coverage lever): tighten each unresolved tracker's `lp_bound` *ceiling*
/// using the LP state-equation bound computed on the **reduced** net. The
/// reduced net is smaller — so the LP may return a finite bound where it was
/// too large / declined on the original net — and the state-equation polytope is
/// often tighter after structural reduction, so more trackers reach
/// `max_bound >= effective_bound()` and resolve exactly.
///
/// SOUNDNESS: `lp_bound` is a CEILING (`true_max <= lp_bound`), and resolution
/// emits `max_bound` as exact once `max_bound >= min(structural, lp, monotone)`
/// (see `BoundTracker::is_structurally_resolved`). A reduced-net ceiling is a
/// valid ORIGINAL-net ceiling only when every query place is a faithful 1:1
/// unit-scale passthrough — the same predicate the witness check
/// (`reduced_observation_witnesses_max`) uses: then the reduced net's value of
/// each query place equals the original's across all reachable markings, so the
/// reduced query max equals the original true max and the reduced LP bound
/// (`>= reduced max`) is `>= true_max`. For any non-faithful query place we skip
/// and leave the original ceiling untouched. We only ever LOWER `lp_bound` to a
/// still-valid ceiling (clamped at `max_bound`, which a valid ceiling already
/// dominates), so no wrong answer can be introduced — at worst coverage is
/// unchanged. Returns the number of trackers whose ceiling tightened.
pub(super) fn tighten_lp_bounds_on_reduced_net(
    reduced: &ReducedNet,
    trackers: &mut [BoundTracker],
) -> usize {
    let mut tightened = 0usize;
    for tracker in trackers.iter_mut() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        // Faithful unit-scale passthrough for every query place (mirrors the
        // per-place passthrough arm of `reduced_observation_witnesses_max`).
        let faithful = tracker.place_indices.iter().all(|&PlaceIdx(orig)| {
            let orig = orig as usize;
            reduced.place_scales[orig] == 1
                && reduced.place_map[orig].is_some()
                && !reduced
                    .constant_values
                    .iter()
                    .any(|&(PlaceIdx(p), _)| p as usize == orig)
                && !reduced
                    .reconstructions
                    .iter()
                    .any(|recon| recon.place.0 as usize == orig)
                && !reduced
                    .report
                    .lateral_fusions
                    .iter()
                    .any(|merge| merge.duplicate.0 as usize == orig)
        });
        if !faithful {
            continue;
        }
        // Remap the query indices into the reduced net (faithful ⇒ all map).
        let remapped: Vec<PlaceIdx> = tracker
            .place_indices
            .iter()
            .filter_map(|&PlaceIdx(orig)| reduced.place_map[orig as usize])
            .collect();
        if remapped.len() != tracker.place_indices.len() {
            continue;
        }
        let Some(reduced_bound) = crate::lp_state_equation::lp_upper_bound(&reduced.net, &remapped)
        else {
            continue;
        };
        // Lower the ceiling to the tighter of the two valid ceilings; clamp at
        // the observed reachable lower bound (a valid ceiling already dominates
        // it — the clamp is belt-and-braces against a surprising LP result).
        let candidate = match tracker.lp_bound {
            Some(cur) => cur.min(reduced_bound),
            None => reduced_bound,
        }
        .max(tracker.max_bound);
        if Some(candidate) != tracker.lp_bound {
            tracker.lp_bound = Some(candidate);
            tightened += 1;
        }
    }
    tightened
}

/// Historical fixed per-call QF_LIA budget for the integer state-equation
/// dichotomy. Retained as the no-deadline default and as the affordability
/// threshold: [`int_tighten_per_call_timeout`] runs the pass only when the phase
/// can give each dichotomy call at least this long, so a tight cell declines
/// (keeping the rational LP cap) instead of spreading a meaningless sliver of
/// budget across the residual queries.
const INT_TIGHTEN_PER_CALL_FLOOR: std::time::Duration = std::time::Duration::from_millis(400);

/// Deadline-aware per-call QF_LIA timeout for the integer state-equation
/// dichotomy, or `None` to decline the pass this phase.
///
/// The integer tightening is a pre-pass, so it may claim at most an eighth of the
/// remaining phase (leaving the rest for the heavier reduce/BFS/DD/trap-LP lanes),
/// split across the residual queries and their bounded dichotomy calls
/// ([`MAX_INT_TIGHTEN_CALLS`](crate::symbolic::int_state_equation::MAX_INT_TIGHTEN_CALLS)).
/// A long (contest) cell therefore affords far more than the historical fixed
/// budget per hard bound; a phase too tight to give each call at least
/// [`INT_TIGHTEN_PER_CALL_FLOOR`] declines (`None`) rather than doing futile
/// sub-budget work. With no deadline it is the historical floor.
///
/// Answer-preserving: the downstream tightening is cap-only and witnessed, so
/// more solve time can only resolve more queries exactly, never change a verdict.
fn int_tighten_per_call_timeout(
    deadline: Option<Instant>,
    residual_queries: usize,
) -> Option<std::time::Duration> {
    let Some(deadline) = deadline else {
        return Some(INT_TIGHTEN_PER_CALL_FLOOR);
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    let splits = (residual_queries.max(1) as u32)
        .saturating_mul(crate::symbolic::int_state_equation::MAX_INT_TIGHTEN_CALLS)
        .max(1);
    let per_call = (remaining / 8) / splits;
    (per_call >= INT_TIGHTEN_PER_CALL_FLOOR).then_some(per_call)
}

/// `true` iff `TY_MCC_DISABLE_INT_UPPER_BOUND` is set to `1`/`on`/`true`. Disables
/// the integer state-equation bound tightening. The pass only ever lowers a
/// `lp_bound` to a SOUND integer bound (`≤` the rational LP cap, `≥` the
/// achievable witness), so disabling it is answer-preserving: it simply leaves
/// the looser rational cap in place and falls through to the unchanged
/// reduce+BFS / DD pipeline.
fn int_upper_bound_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_INT_UPPER_BOUND")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
}

/// Tighten each unresolved tracker's rational `lp_bound` to the INTEGER
/// state-equation bound where the integer program is strictly tighter.
///
/// For every still-unresolved tracker that has a finite `lp_bound`, this runs the
/// bounded integer dichotomy
/// [`crate::symbolic::int_state_equation::integer_state_equation_upper_bound`]
/// with the tracker's current `max_bound` (a genuinely achievable witness — the
/// initial-marking sum at this point) as the SOUND lower floor and `lp_bound` as
/// the SOUND ceiling. The integer bound returned is `≤ lp_bound` and `≥ max_bound`,
/// so replacing `lp_bound` with it:
///   - never raises the cap (monotone tightening),
///   - never drops below an achievable value (so a completed BFS would still
///     observe a maximum `≤` the new cap),
///
/// hence is SOUND and verdict-preserving. When the integer bound coincides with
/// `max_bound`, the tracker becomes exactly resolved (lower == upper) without any
/// BFS — the integer-tightening analogue of the trap-LP ratification above.
///
/// Returns the number of trackers newly resolved by the tightening.
fn tighten_upper_bounds_with_integer(
    net: &PetriNet,
    trackers: &mut [BoundTracker],
    deadline: Option<Instant>,
) -> usize {
    if int_upper_bound_disabled() {
        return 0;
    }
    // Reach as far as the rational LP relaxation this refines (the pass only ever
    // tightens that cap); the deadline-aware per-call budget below — not a fixed
    // net-size cutoff — is what keeps the pass a bounded pre-pass.
    if net.num_places() + net.num_transitions() > crate::lp_state_equation::MAX_LP_VARIABLES {
        return 0;
    }

    // Residual queries the pass will actually attempt: unresolved trackers whose
    // finite LP cap still sits strictly above a witnessed value. The per-call
    // QF_LIA budget is split across these so the whole pass spends a bounded share
    // of the remaining phase regardless of residual count; a phase too tight to
    // afford a meaningful solve declines the pass entirely.
    let residual_queries = trackers
        .iter()
        .filter(|t| !t.is_structurally_resolved() && t.lp_bound.is_some_and(|lp| lp > t.max_bound))
        .count();
    let Some(per_call_timeout) = int_tighten_per_call_timeout(deadline, residual_queries) else {
        return 0;
    };

    let mut newly_resolved = 0usize;
    for tracker in trackers.iter_mut() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        let Some(lp_bound) = tracker.lp_bound else {
            continue; // no finite cap to tighten (unbounded query)
        };
        // `max_bound` is a sound achievable lower bound; only tighten when the LP
        // cap is strictly above it (otherwise nothing to gain).
        if lp_bound <= tracker.max_bound {
            continue;
        }
        if let Some(int_bound) =
            crate::symbolic::int_state_equation::integer_state_equation_upper_bound(
                net,
                &tracker.place_indices,
                lp_bound,
                tracker.max_bound,
                per_call_timeout,
            )
        {
            // SOUND: int_bound ∈ [max_bound, lp_bound]. `min` is belt-and-braces.
            let tightened = int_bound.min(lp_bound);
            if tightened < lp_bound {
                tracker.lp_bound = Some(tightened);
            }
            if tracker.is_structurally_resolved() {
                newly_resolved += 1;
            }
        }
    }
    newly_resolved
}

/// Check UpperBounds properties with fail-closed unresolved-place handling.
///
/// Returns `Some(bound)` for valid properties after completed exploration and
/// `None` for invalid properties or incomplete exploration.
#[cfg(test)]
pub(crate) fn check_upper_bounds_properties(
    net: &PetriNet,
    properties: &[Property],
    config: &ExplorationConfig,
) -> Vec<(String, Option<u64>)> {
    let aliases = PropertyAliases::identity(net);
    check_upper_bounds_properties_with_aliases(net, properties, &aliases, config)
}

#[cfg(test)]
pub(crate) fn check_upper_bounds_properties_with_aliases(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
) -> Vec<(String, Option<u64>)> {
    check_upper_bounds_properties_core(net, properties, aliases, config, None, None, None)
}

/// Like [`check_upper_bounds_properties_with_aliases`] but with the model's
/// NUPN structure (when the PNML carries one) for seeding the DD fast-path
/// variable order. PERFORMANCE-ONLY: the seed never changes a bound (see
/// `dd_fastpath::try_dd_fast_path`); `nupn = None` is identical to the
/// plain entry point.
pub(crate) fn check_upper_bounds_properties_with_aliases_and_nupn(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Vec<(String, Option<u64>)> {
    check_upper_bounds_properties_core(net, properties, aliases, config, None, nupn, None)
}

/// Like [`check_upper_bounds_properties_with_aliases`] but with an explicit
/// "ground truth" net/aliases pair for the Safety-Net-B fallback.
///
/// When `ground_truth` is `Some(_)`, the caller (typically the COL pipeline)
/// is telling us "the `net` argument has already been reduced by some upstream
/// layer that I don't trust to preserve UpperBounds; use this pair as the
/// real, un-pre-reduced net for the fallback BFS". The fallback re-resolves
/// each property's `PlaceBound` names against `ground_truth.aliases` so the
/// BFS runs against the ground-truth net's true place indices.
///
/// When `ground_truth` is `Some(_)`, the fallback fires unconditionally
/// after the reduced-net BFS (not gated on LP-gap or multi-place
/// over-count): the caller has already self-attested that the upstream
/// reduction may have produced spurious markings, so we always cross-check.
pub(crate) fn check_upper_bounds_properties_with_aliases_and_ground_truth(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    ground_truth: Option<GroundTruthNet<'_>>,
    colored: Option<&crate::hlpnml::ColoredNet>,
) -> Vec<(String, Option<u64>)> {
    // No NUPN here: the COL ground-truth callers run on unfolded nets whose
    // place indexing need not match any NUPN annotation, and the DD
    // fast-path is skipped under ground-truth anyway.
    check_upper_bounds_properties_core(
        net,
        properties,
        aliases,
        config,
        ground_truth,
        None,
        colored,
    )
}

/// Kill-switch for the legacy oxidd BDD UpperBounds fast-path. Default-ON; set
/// `TY_MCC_DISABLE_DD_UPPERBOUNDS` truthy to skip it (the native MDD lane + the
/// LP/BFS pipeline then resolve those trackers). SOUNDNESS-NEUTRAL — the lane
/// only ratifies exact reachable maxima; the A/B mechanism for retiring it.
#[cfg(feature = "dd-backend")]
fn dd_upper_bounds_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_DD_UPPERBOUNDS").is_ok_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// Kill-switch for the native MDD UpperBounds fast-path. Default-ON; set
/// `TY_MCC_DISABLE_MDD_UPPERBOUNDS` truthy to skip it. SOUNDNESS-NEUTRAL: the
/// lane only ratifies the EXACT reachable maximum on otherwise-unresolved
/// trackers (additive coverage), so disabling it can only make a tracker fall
/// through to LP+BFS — never changes a published bound.
#[cfg(feature = "dd-backend")]
fn mdd_upper_bounds_disabled() -> bool {
    std::env::var("TY_MCC_DISABLE_MDD_UPPERBOUNDS").is_ok_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// Shared pipeline core behind every `check_upper_bounds_properties_*`
/// entry point. `nupn` only seeds the DD fast-path's variable order
/// (performance-only; values are unaffected by construction).
fn check_upper_bounds_properties_core(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    ground_truth: Option<GroundTruthNet<'_>>,
    nupn: Option<&crate::nupn::NupnStructure>,
    colored: Option<&crate::hlpnml::ColoredNet>,
) -> Vec<(String, Option<u64>)> {
    #[cfg(not(feature = "dd-backend"))]
    let _ = nupn;
    let has_ground_truth = ground_truth.is_some();
    let cannot_compute_results = |error: &PnmlError| {
        eprintln!("UpperBounds: CANNOT_COMPUTE ({error})");
        properties
            .iter()
            .map(|property| (property.id.clone(), None))
            .collect()
    };
    let cannot_compute_ground_truth_results = |reason: &str| {
        eprintln!("UpperBounds: COL ground-truth fallback unusable ({reason}) — CANNOT_COMPUTE");
        properties
            .iter()
            .map(|property| (property.id.clone(), None))
            .collect()
    };

    let (prepared, mut trackers) =
        prepare_upper_bounds_properties_with_aliases(properties, aliases);
    if trackers.is_empty() {
        return prepared
            .into_iter()
            .map(|property| match property {
                PreparedUpperBoundProperty::Invalid { id } => (id, None),
                PreparedUpperBoundProperty::Valid { .. } => unreachable!(),
            })
            .collect();
    }

    let (mut exact_proof_count, lp_count) = seed_upper_bounds_exactness(net, &mut trackers);
    if lp_count > 0 {
        eprintln!(
            "UpperBounds: LP tightened bounds on {lp_count}/{} properties",
            trackers.len(),
        );
    }

    // Pre-BFS trap-LP ratification of the initial-marking witness.
    //
    // At this point every tracker's `max_bound` is the *initial-marking* token
    // sum — a genuinely reachable witness. `confirm_observed_maxima_with_trap_lp`
    // lowers `lp_bound` to meet `max_bound` only when the strictly stronger
    // state-equation + trap LP proves `max_bound + 1` unreachable. When the bare
    // `lp_upper_bound` cap (used during seeding) is too loose but the trap cuts
    // close the gap, this ratifies the witness as the EXACT maximum without any
    // BFS. Sound & verdict-preserving: it only ever lowers a cap to a witnessed
    // value, so a completed BFS would observe the same maximum; it never changes
    // an emitted value, only converts a would-be BFS resolution into a static
    // one. Runs on the primary (un-reduced) `net`, so — unlike the post-BFS
    // reduced-net call — `max_bound` cannot be a reduction artifact.
    let pre_bfs_trap_lp = confirm_observed_maxima_with_trap_lp(net, &mut trackers);
    if pre_bfs_trap_lp > 0 {
        exact_proof_count = trackers
            .iter()
            .filter(|t| t.is_structurally_resolved())
            .count();
        eprintln!(
            "UpperBounds: pre-BFS trap-LP ratified the initial-marking witness on \
             {pre_bfs_trap_lp}/{} properties",
            trackers.len(),
        );
    }

    // Integer state-equation bound tightening (additive, cap-only). The rational
    // `lp_upper_bound` from seeding can leave a relaxation gap — a rational cap
    // strictly above the true integer maximum. Where the INTEGER state equation
    // (with trap + siphon cuts) proves a tighter ceiling, this lowers `lp_bound`
    // to that sound integer value. It never raises a cap and never descends below
    // the achievable `max_bound` witness, so it is verdict-preserving; when the
    // integer cap meets `max_bound` the query is pinned EXACTLY without any BFS
    // (the integer analogue of the trap-LP ratification above). Runs on the
    // primary un-reduced `net`, so `max_bound` is a genuine reachable witness.
    let int_tightened = tighten_upper_bounds_with_integer(net, &mut trackers, config.deadline());
    if int_tightened > 0 {
        exact_proof_count = trackers
            .iter()
            .filter(|t| t.is_structurally_resolved())
            .count();
        eprintln!(
            "UpperBounds: integer state-equation tightening pinned {int_tightened}/{} properties",
            trackers.len(),
        );
    }

    // EARLY achievable-maximum walk lane (additive, lower-bound only). Runs on
    // the RAW `net` over its own leftover budget slice, raising each query's
    // achievable lower bound (`max_bound`) toward the already-seeded structural/LP
    // UPPER bound. The existing `is_structurally_resolved()` pins a query EXACTLY
    // when the raised lower bound meets the upper bound; misses fall through to the
    // unchanged reduce+BFS / DD / trap-LP pipeline below (see
    // `run_upper_bounds_walk_pass` for the full soundness argument). Skipped under
    // ground-truth (the COL fallback re-prepares trackers against a different net,
    // so a pin computed against `net` here would not be reused — and the COL
    // ground-truth path is authoritative on its own net).
    if !has_ground_truth {
        let walk_resolved = run_upper_bounds_walk_pass(net, &mut trackers, config.deadline());
        if walk_resolved > 0 {
            exact_proof_count = trackers
                .iter()
                .filter(|t| t.is_structurally_resolved())
                .count();
            eprintln!(
                "UpperBounds: achievable-maximum walk pinned {walk_resolved}/{} properties",
                trackers.len(),
            );
        }
    }

    // If all properties are resolved by exact static proofs, skip BFS.
    if exact_proof_count == trackers.len() && !has_ground_truth {
        eprintln!(
            "UpperBounds: all {} properties resolved by static exactness proofs",
            trackers.len(),
        );
        return assemble_upper_bounds_results(&prepared, &trackers, true);
    }

    if exact_proof_count > 0 {
        eprintln!(
            "UpperBounds: {exact_proof_count}/{} properties resolved by static exactness proofs, \
             exploring state space for remaining",
            trackers.len(),
        );
    }

    // Decision-Diagram authoritative fast-path (off by default — gated
    // by `dd-backend`). For small bounded nets we try a BDD-based
    // forward reachability fixpoint on the original net under a hard
    // 5-second time budget. The DD path computes the *exact*
    // `max_{M ∈ R} Σ_p coeff[p] · m[p]` for every unresolved tracker
    // via [`tla_dd::dispatch_upper_bounds_for_queries`] on a single
    // shared reachable BDD, then validates each value against the
    // existing LP/structural cap before accepting it. Soundness floor:
    // every DD result is bounded above by the LP cap and below by the
    // initial-marking sum; a value outside that envelope is treated as
    // a DD failure and we fall through to BFS unchanged.
    //
    // On success the DD path returns authoritative `(id, Some(bound))`
    // tuples bypassing the reduce-and-BFS pipeline entirely. The
    // pipeline's reduction-related safety nets (A LP-gap, B multi-place
    // over-count, C COL ground-truth) are unnecessary because the DD
    // computes against the un-reduced net by construction.
    #[cfg(feature = "dd-backend")]
    if !has_ground_truth && !dd_upper_bounds_disabled() {
        if let Some(dd_result) = super::dd_fastpath::try_dd_fast_path(net, &trackers, nupn) {
            let mut all_resolved = true;
            let mut dd_count = 0usize;
            for (i, dd_value) in dd_result.bounds.iter().enumerate() {
                match dd_value {
                    Some(value) => {
                        trackers[i].max_bound = *value;
                        // Force `is_structurally_resolved()` to recognise
                        // the DD-supplied bound as exact even if the LP
                        // bound was looser. Setting `structural_bound` to
                        // the DD value is sound: the DD computed the true
                        // reachable maximum, so any tighter structural
                        // cap that happens to coincide with it is also
                        // sound; a looser one is irrelevant because we
                        // already have the exact value.
                        trackers[i].structural_bound = Some(*value);
                        dd_count += 1;
                    }
                    None => {
                        // Either tracker was already resolved on entry
                        // (then DD declined to recompute) or DD failed for
                        // this tracker only. In either case if the tracker
                        // is not now structurally resolved, we cannot
                        // short-circuit BFS.
                        if !trackers[i].is_structurally_resolved() {
                            all_resolved = false;
                        }
                    }
                }
            }
            eprintln!(
                "UpperBounds: DD fast-path resolved {dd_count}/{} unresolved trackers",
                trackers
                    .iter()
                    .filter(|t| !t.is_structurally_resolved())
                    .count()
                    + dd_count,
            );
            if all_resolved {
                return assemble_upper_bounds_results(&prepared, &trackers, true);
            }
        }
    }

    // Native MDD UpperBounds fast-path — the MDD twin of the BDD lane above.
    // Runs on any trackers still unresolved (e.g. nets the BDD lane's bit-blast
    // gate declined but the per-place MDD admits): it computes the EXACT
    // `max Σ coeff·m` over the reachable set, so it can only ADD coverage, never
    // change a value. Default-ON; `TY_MCC_DISABLE_MDD_UPPERBOUNDS` skips it.
    #[cfg(feature = "dd-backend")]
    if !has_ground_truth
        && !mdd_upper_bounds_disabled()
        && trackers.iter().any(|t| !t.is_structurally_resolved())
    {
        if let Some(mdd_result) =
            super::mdd_fastpath::try_mdd_fast_path(net, &trackers, config.deadline(), colored, nupn)
        {
            let mut all_resolved = true;
            let mut mdd_count = 0usize;
            for (i, mdd_value) in mdd_result.bounds.iter().enumerate() {
                match mdd_value {
                    Some(value) => {
                        // EXACT reachable maximum — sound to ratify (see the BDD
                        // twin's identical reasoning).
                        trackers[i].max_bound = *value;
                        trackers[i].structural_bound = Some(*value);
                        mdd_count += 1;
                    }
                    None => {
                        if !trackers[i].is_structurally_resolved() {
                            all_resolved = false;
                        }
                    }
                }
            }
            eprintln!("UpperBounds: MDD fast-path resolved {mdd_count} additional tracker(s)");
            if all_resolved {
                return assemble_upper_bounds_results(&prepared, &trackers, true);
            }
        }
    }

    // GPU exhaustive-reachability fast-path — the explicit twin of the DD/MDD
    // lanes above, for SINGLE-PLACE trackers. The device BFS enumerates the
    // full reachable set once (raw net, no reduction) and reports per-place
    // token maxima; each is an EXACT reachable maximum with an achieving
    // witness marking, so ratifying both the witness (`max_bound`) and the
    // ceiling (`structural_bound`) is sound — identical reasoning to the DD
    // twin. Multi-place sum queries are left to the pipeline: per-place
    // maxima need not be co-achieved in one marking, so their sum is only a
    // ceiling, not a value. Fail-closed: probe/emission/capacity/engine
    // errors (including the `config.max_states()` distinct-marking cap)
    // decline and the reduce+BFS pipeline below runs unchanged.
    #[cfg(feature = "gpu")]
    if !has_ground_truth
        && trackers
            .iter()
            .any(|t| !t.is_structurally_resolved() && t.place_indices.len() == 1)
        && crate::gpu_state_space::gpu_lane_enabled(net)
    {
        if let Some(maxima) =
            crate::gpu_state_space::place_maxima_gpu(net, config.max_states(), "UpperBounds")
        {
            let mut gpu_count = 0usize;
            for tracker in &mut trackers {
                if tracker.is_structurally_resolved() {
                    continue;
                }
                if let [place] = tracker.place_indices.as_slice() {
                    let value = maxima[place.0 as usize];
                    tracker.max_bound = value;
                    tracker.structural_bound = Some(value);
                    gpu_count += 1;
                }
            }
            eprintln!(
                "UpperBounds: GPU exhaustive fast-path resolved {gpu_count} additional tracker(s)"
            );
            if trackers.iter().all(|t| t.is_structurally_resolved()) {
                return assemble_upper_bounds_results(&prepared, &trackers, true);
            }
        }
    }

    // Explore the reduced net with an expanding observer so that the
    // UpperBoundsObserver receives expanded (original-index) markings.
    // Constant/isolated places get their fixed values in the expanded
    // marking, so bounds are correctly computed over the full net.
    let unresolved_place_sets: Vec<Vec<PlaceIdx>> = trackers
        .iter()
        .filter(|tracker| !tracker.is_structurally_resolved())
        .map(|tracker| tracker.place_indices.clone())
        .collect();
    let initial_protected = if unresolved_place_sets.is_empty() {
        vec![false; net.num_places()]
    } else {
        upper_bounds_support(&ReducedNet::identity(net), &unresolved_place_sets)
            .map(|support| support.places)
            .unwrap_or_else(|| vec![true; net.num_places()])
    };
    let reduced = match reduce_iterative_structural_query_with_protected(net, &initial_protected) {
        Ok(reduced) => reduced,
        Err(error) => return cannot_compute_results(&error),
    };
    let mut reduced = match reduce_query_guarded(reduced, |r| {
        let unresolved_place_sets: Vec<Vec<PlaceIdx>> = trackers
            .iter()
            .filter(|tracker| !tracker.is_structurally_resolved())
            .map(|tracker| tracker.place_indices.clone())
            .collect();
        if unresolved_place_sets.is_empty() {
            return None;
        }
        let support = upper_bounds_support(r, &unresolved_place_sets)?;
        Some(support.places)
    }) {
        Ok(reduced) => reduced,
        Err(error) => return cannot_compute_results(&error),
    };
    if let Err(error) = apply_final_place_gcd_scaling(&mut reduced) {
        return cannot_compute_results(&error);
    }
    // A2 (coverage): tighten unresolved trackers' LP ceiling using the smaller,
    // often-tighter reduced-net LP — gated to faithful unit-scale passthrough
    // queries so the reduced ceiling is a valid original-net ceiling (cannot
    // introduce a wrong answer; see `tighten_lp_bounds_on_reduced_net`).
    let lp_tightened = tighten_lp_bounds_on_reduced_net(&reduced, &mut trackers);
    if lp_tightened > 0 {
        eprintln!("UpperBounds: reduced-net LP tightened the ceiling on {lp_tightened} tracker(s)");
    }
    let config = config.refitted_for_net(&reduced.net);
    // `used_slice` records whether the query-local slice path ran. The slice
    // applies an *additional* reduction layer (relevance cone + circulation-loop
    // contraction) on top of `reduced` that `reduced`'s `place_map`/`place_scales`
    // do not describe. A slice-only reduction could inflate the observed query
    // sum invisibly to `reduced_observation_witnesses_max`, so when the slice was
    // used we only trust the trivial (initial-marking) witness below.
    let mut used_slice = false;
    let (completed, mut trackers) = if let Some((slice, unresolved_slots, slice_trackers)) =
        build_upper_bounds_slice(&reduced, &trackers)
    {
        used_slice = true;
        let (result, slice_trackers) =
            explore_upper_bounds_on_slice(&slice, slice_trackers, &config);
        for (slot, tracker) in unresolved_slots.into_iter().zip(slice_trackers) {
            trackers[slot].max_bound = tracker.max_bound;
        }
        (result.completed, trackers)
    } else {
        match explore_upper_bounds_on_reduced_net(&reduced, trackers, &config) {
            Ok((result, trackers)) => (result.completed, trackers),
            Err(error) => return cannot_compute_results(&error),
        }
    };
    // SOUNDNESS (FINDING #11): the observed `max_bound` came from BFS on the
    // *reduced* net (and possibly a further slice reduction). Trap-LP can prove
    // only the ceiling (`max_bound + 1` unreachable); it must NOT ratify
    // `max_bound` as the exact bound unless `max_bound` is also a *confirmed
    // reachable* value. A reduction rule can over-count the reduced-net maximum
    // above the true original-net maximum, and ratifying that inflated value
    // would publish a too-high (wrong) UpperBounds answer. Gate ratification on
    // a per-tracker reachability witness; unwitnessed trackers stay unresolved
    // for the safety-net / exact path below to resolve at the true maximum.
    let witness_flags: Vec<bool> = trackers
        .iter()
        .map(|tracker| {
            if used_slice {
                // Only the trivial (initial-marking) witness is sound when an
                // opaque slice reduction layer sits between `reduced` and the
                // observation. As in `reduced_observation_witnesses_max`, the
                // trivial witness compares the original-net `initial_sum` against
                // a reduced-net observation, so it is sound only when every query
                // place is unit-scale — a GCD-scaled query (`place_scales > 1`)
                // would compare different unit systems and could ratify a wrong
                // bound. Withhold the witness in that case; the exact search below
                // resolves the tracker at its true maximum.
                let initial_sum: u64 = tracker
                    .place_indices
                    .iter()
                    .map(|p| net.initial_marking[p.0 as usize])
                    .sum();
                let query_unit_scale = tracker
                    .place_indices
                    .iter()
                    .all(|&PlaceIdx(orig)| reduced.place_scales[orig as usize] == 1);
                query_unit_scale && tracker.max_bound == initial_sum
            } else {
                reduced_observation_witnesses_max(&reduced, net, tracker)
            }
        })
        .collect();
    let trap_lp_count =
        confirm_observed_maxima_with_trap_lp_witnessed(net, &mut trackers, |i| witness_flags[i]);
    if trap_lp_count > 0 {
        eprintln!(
            "UpperBounds: trap-LP ratified witnessed observed maxima on {trap_lp_count}/{} \
             properties",
            trackers.len(),
        );
    }

    // Safety net A (LP-gap, under-count): if BFS completed on the reduced net
    // but some trackers have max_bound < lp_bound, the reduction may have
    // pruned reachable markings needed to achieve the true maximum. Fall back
    // to identity-net BFS for these formulas. (#1501: IBM5964 UpperBounds-14
    // returns 2 instead of 3 because structural reduction removes a producer
    // transition.)
    //
    // Safety net B (reduction over-count for multi-place trackers): even
    // when the reduced-net BFS terminates "structurally resolved" by hitting
    // the LP bound (`max_bound == lp_bound`), the reduction may have
    // introduced spurious reachable markings whose expanded sums inflate a
    // joint-place tracker above its true reachable maximum. Lamport-PT-2
    // UpperBounds-00 (`bound(P-await_13_0, _1, _2)`) is the reproducer:
    // consensus max=1 (per-process mutex), TY's reduced-net BFS observes
    // sum=2 (=LP bound, ratified by `is_done`), but on the identity net BFS
    // confirms max=1. Cross-check **every** multi-place tracker on the
    // identity net when reduction touched the net — identity-net BFS is
    // ground truth, so its observed maximum is sound by construction.
    // Single-place trackers cannot be inflated this way (a single place's
    // reduced value tracks the original through `place_scales` /
    // `reconstructions`, both of which are reachability-faithful for the
    // single place), so we skip them and keep the reduced-net path's
    // performance.
    //
    // Safety net C (COL-layer ground-truth fallback): the caller has
    // provided a `ground_truth` net/aliases pair attesting that the `net`
    // argument was already reduced by an upstream layer (currently the COL
    // `colored_relevance::reduce` backward-closure pass, which is not
    // UpperBounds-preserving in general). Always cross-check against the
    // ground-truth net in this case — re-resolve every property's
    // `PlaceBound` names through `ground_truth.aliases` and BFS against
    // `ground_truth.net`. This catches **both** over-counts (e.g. LamportFastMutEx-COL-2
    // UpperBounds-02 reduced reports 2 but ground truth is 1) and
    // under-counts (e.g. LamportFastMutEx-COL-2 UpperBounds-06 reduced
    // reports 1 but ground truth is 2), neither of which the PT-layer
    // Safety-Net-A/B would detect when the per-property `net` itself has
    // already lost the relevant transitions or accumulated spurious
    // markings.
    let reduction_touched_net =
        reduced
            .place_map
            .iter()
            .enumerate()
            .any(|(orig, mapped)| match mapped {
                Some(p) => p.0 as usize != orig,
                None => true,
            })
            || reduced.net.num_transitions() != net.num_transitions();
    let has_lp_gap = trackers.iter().any(|t| {
        t.lp_bound
            .is_some_and(|lp| t.max_bound < lp && !t.is_structurally_resolved())
    });
    let needs_multi_place_verify = reduction_touched_net
        && trackers.iter().any(|t| {
            if t.place_indices.len() <= 1 {
                return false;
            }
            let initial_sum: u64 = t
                .place_indices
                .iter()
                .map(|p| net.initial_marking[p.0 as usize])
                .sum();
            // Verify whenever a multi-place tracker observed a value above
            // its initial-marking sum: that's the regime in which a
            // spurious reduced-net marking could matter.
            t.max_bound > initial_sum
        });
    // Branch 1: ground-truth fallback path (Safety-Net-C). Re-prepare
    // trackers from scratch against `ground_truth.net`/`ground_truth.aliases`
    // and run BFS on that net — bypassing the per-property `net` entirely
    // for the fallback. Identity-net BFS is authoritative when it
    // completes; otherwise CANNOT_COMPUTE.
    if let Some(gt) = ground_truth {
        let (gt_prepared, mut gt_trackers) =
            prepare_upper_bounds_properties_with_aliases(properties, gt.aliases);
        if gt_trackers.is_empty() {
            return cannot_compute_ground_truth_results("no valid trackers on ground-truth net");
        }
        if let Some((id, place)) = gt_trackers.iter().find_map(|tracker| {
            tracker
                .place_indices
                .iter()
                .find(|place| place.0 as usize >= gt.net.num_places())
                .map(|place| (tracker.id.as_str(), *place))
        }) {
            return cannot_compute_ground_truth_results(&format!(
                "{id} maps to out-of-range ground-truth place {}",
                place.0,
            ));
        }
        let (gt_exact_proof_count, gt_lp_count) =
            seed_upper_bounds_exactness(gt.net, &mut gt_trackers);
        if gt_lp_count > 0 {
            eprintln!(
                "UpperBounds: COL ground-truth LP tightened bounds on {gt_lp_count}/{} properties",
                gt_trackers.len(),
            );
        }
        if gt_exact_proof_count == gt_trackers.len() {
            eprintln!(
                "UpperBounds: COL ground-truth static proofs resolved all {} properties",
                gt_trackers.len(),
            );
            return assemble_upper_bounds_results(&gt_prepared, &gt_trackers, true);
        }
        // Singleton residual retry first: trap-LP dichotomy can resolve
        // residuals without spending any BFS budget, and it leaves the full
        // deadline available for the identity-net BFS that follows.
        let parallel_config = config.refitted_for_net(gt.net);
        let pre_bfs_resolved = singleton_residual_retry_lp_only(gt.net, &mut gt_trackers);
        if pre_bfs_resolved > 0 {
            eprintln!(
                "UpperBounds: trap-LP dichotomy pre-resolved {pre_bfs_resolved} \
                 ground-truth tracker(s) before BFS",
            );
        }
        if gt_trackers.iter().all(|t| t.is_structurally_resolved()) {
            return assemble_upper_bounds_results(&gt_prepared, &gt_trackers, true);
        }
        // Use the full configured worker pool for the ground-truth BFS. The
        // historical `with_workers(1)` was a conservative choice — but the
        // single-worker BFS is the dominant bottleneck for per-property COL
        // calls (15/16 residual scenarios), and parallel exploration on the
        // small per-property unfolded net is sound (observer only ratchets
        // max_bound upward, never reductions).
        let identity_config = parallel_config.clone();
        let identity = ReducedNet::identity(gt.net);
        match explore_upper_bounds_on_reduced_net(&identity, gt_trackers, &identity_config) {
            Ok((identity_result, mut identity_trackers)) => {
                eprintln!(
                    "UpperBounds: COL ground-truth fallback on un-relevance-reduced net \
                     (completed={})",
                    identity_result.completed,
                );
                // Trap-LP ratification picks up cases where the freshly
                // observed max_bound is the true reachable max.
                let _ratified =
                    confirm_observed_maxima_with_trap_lp(gt.net, &mut identity_trackers);
                let completed_or_resolved = identity_result.completed
                    || identity_trackers
                        .iter()
                        .all(|t| t.is_structurally_resolved());
                return assemble_upper_bounds_results(
                    &gt_prepared,
                    &identity_trackers,
                    completed_or_resolved,
                );
            }
            Err(error) => {
                eprintln!(
                    "UpperBounds: COL ground-truth fallback failed ({error}), \
                     refusing per-property reduced-net exactness",
                );
                return cannot_compute_ground_truth_results("fallback exploration failed");
            }
        }
    }

    if has_lp_gap || needs_multi_place_verify {
        // CRITICAL: do NOT trap-LP-pre-resolve trackers here. The reduced-net
        // `trackers[*].max_bound` is suspect in this branch (that's exactly
        // why we're entering it). Trap-LP refinement that ratifies a tracker
        // with `lp_bound = max_bound` would short-circuit the identity-net
        // cross-check below and publish a value that's actually a reduction
        // artifact (Lamport-PT-2 UpperBounds-00 regression: reduced-net
        // observes 2, identity-net observes 1; if we ratified 2 here we'd
        // emit a wrong answer of 2).
        //
        // Use the full configured worker pool — the historical
        // `with_workers(1)` single-thread fallback was the dominant
        // bottleneck for residual singletons in per-property COL calls
        // (15/16 patterns). Parallel BFS on the identity net is sound
        // (observer only ratchets upward).
        let identity_config = config.refitted_for_net(net);
        let identity = ReducedNet::identity(net);
        // Re-seed trackers' `max_bound` from the initial marking so the
        // identity-net observer cannot inherit a spurious reduced-net value.
        // The observer only takes maxima, never reductions, so without this
        // reset a reduced-net over-estimate of e.g. 2 would survive an
        // identity-net BFS whose true observed max is 1.
        let identity_seed_trackers: Vec<BoundTracker> = trackers
            .iter()
            .map(|t| {
                let mut copy = t.clone();
                copy.max_bound = t
                    .place_indices
                    .iter()
                    .map(|p| net.initial_marking[p.0 as usize])
                    .sum();
                copy
            })
            .collect();
        match explore_upper_bounds_on_reduced_net(
            &identity,
            identity_seed_trackers.clone(),
            &identity_config,
        ) {
            Ok((identity_result, mut identity_trackers)) => {
                let reason = if has_lp_gap && needs_multi_place_verify {
                    "LP-gap + multi-place reduction guard"
                } else if has_lp_gap {
                    "LP-gap"
                } else {
                    "multi-place reduction guard"
                };
                eprintln!(
                    "UpperBounds: {reason} fallback on identity net (completed={})",
                    identity_result.completed,
                );
                if identity_result.completed {
                    // Ground truth available — identity-net values are
                    // authoritative for every tracker (sound for both
                    // over- and under-estimates).
                    return assemble_upper_bounds_results(&prepared, &identity_trackers, true);
                }
                // Identity-net BFS did not complete: the reduced-net
                // observation is not trustworthy (this whole branch fired
                // because we suspect it). Try a last-mile trap-LP dichotomy
                // on the un-reduced net using observed maxima as the lower
                // bound — this can ratify residuals without spending any
                // additional BFS budget. Sound because trap-LP can only
                // tighten the upper cap, never raise it.
                let _ratified = confirm_observed_maxima_with_trap_lp(net, &mut identity_trackers);
                let resolved = singleton_residual_retry_lp_only(net, &mut identity_trackers);
                if resolved > 0 {
                    eprintln!(
                        "UpperBounds: trap-LP residual dichotomy resolved {resolved} \
                         tracker(s) after identity-net timeout",
                    );
                }
                let completed_or_resolved = identity_trackers
                    .iter()
                    .all(|t| t.is_structurally_resolved());
                // Fall back to identity-net's observed maxima (sound lower
                // bounds on the true reachable max). When `completed_or_resolved`
                // is true, every tracker has been ratified by its effective
                // cap, so the assembly emits exact bounds. Otherwise emit
                // CANNOT_COMPUTE via `completed=false`. A reduced-net value
                // that happens to coincide with the LP bound would otherwise
                // make `is_structurally_resolved()` return true and ratify a
                // spurious answer as exact — so we deliberately disregard the
                // reduced-net max here.
                return assemble_upper_bounds_results(
                    &prepared,
                    &identity_trackers,
                    completed_or_resolved,
                );
            }
            Err(error) => {
                eprintln!(
                    "UpperBounds: identity-net fallback failed ({error}), \
                     refusing suspect reduced-net exactness",
                );
                return assemble_upper_bounds_results(&prepared, &identity_seed_trackers, false);
            }
        }
    }

    // Final exit: regular pipeline did not enter any safety net. If BFS did
    // not complete and at least one tracker is still unresolved, fire the
    // trap-LP dichotomy on the per-property `net`. This catches residuals
    // that the per-property reduced-net BFS missed (typically when the
    // reduced net has a single straggler whose LP cap is tight enough for
    // trap-LP to push down to the observed max).
    let any_unresolved = trackers.iter().any(|t| !t.is_structurally_resolved());
    if !completed && any_unresolved {
        let resolved = singleton_residual_retry_lp_only(net, &mut trackers);
        if resolved > 0 {
            eprintln!(
                "UpperBounds: trap-LP residual dichotomy resolved {resolved} tracker(s) \
                 after incomplete primary BFS",
            );
        }
    }
    let completed_or_resolved = completed || trackers.iter().all(|t| t.is_structurally_resolved());
    assemble_upper_bounds_results(&prepared, &trackers, completed_or_resolved)
}

#[cfg(test)]
mod int_tighten_budget_tests {
    use super::{int_tighten_per_call_timeout, INT_TIGHTEN_PER_CALL_FLOOR};
    use std::time::{Duration, Instant};

    #[test]
    fn no_deadline_uses_historical_floor() {
        // Budget-unaware callers keep the exact legacy per-call budget.
        assert_eq!(
            int_tighten_per_call_timeout(None, 1),
            Some(INT_TIGHTEN_PER_CALL_FLOOR),
        );
        assert_eq!(
            int_tighten_per_call_timeout(None, 64),
            Some(INT_TIGHTEN_PER_CALL_FLOOR),
        );
    }

    #[test]
    fn long_phase_scales_per_call_above_the_floor() {
        // A contest-scale cell affords far more than the historical 400ms/call.
        let deadline = Instant::now() + Duration::from_secs(3600);
        let per_call = int_tighten_per_call_timeout(Some(deadline), 1)
            .expect("a 3600s phase with one residual affords the pass");
        assert!(
            per_call > INT_TIGHTEN_PER_CALL_FLOOR,
            "expected a scaled budget above the floor, got {per_call:?}",
        );
        // ~ (3600s / 8) / (1 * 12) = 37.5s; assert a generous band, not the exact value.
        assert!(
            per_call >= Duration::from_secs(30) && per_call <= Duration::from_secs(45),
            "per_call out of expected band: {per_call:?}",
        );
    }

    #[test]
    fn more_residuals_split_the_budget_smaller() {
        let one = int_tighten_per_call_timeout(Some(Instant::now() + Duration::from_secs(3600)), 1)
            .unwrap();
        let many =
            int_tighten_per_call_timeout(Some(Instant::now() + Duration::from_secs(3600)), 8)
                .unwrap();
        assert!(
            many < one,
            "more residuals must share the reserve: {many:?} !< {one:?}"
        );
    }

    #[test]
    fn tight_phase_declines_rather_than_spending_a_sliver() {
        // Below the affordability threshold (~38.4s * residual): decline (None),
        // keeping the rational LP cap instead of doing futile sub-floor solves.
        assert_eq!(
            int_tighten_per_call_timeout(Some(Instant::now() + Duration::from_secs(10)), 1),
            None,
        );
        // Spreading even a long phase across too many residuals also declines.
        assert_eq!(
            int_tighten_per_call_timeout(Some(Instant::now() + Duration::from_secs(3600)), 1000),
            None,
        );
    }

    #[test]
    fn expired_phase_declines() {
        // remaining ~ 0 => per_call ~ 0 < floor => decline.
        assert_eq!(int_tighten_per_call_timeout(Some(Instant::now()), 1), None);
    }
}
