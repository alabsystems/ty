// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Reduction, slicing, and reduced-net exploration helpers for reachability.

use crate::circulation_loop::reduce_query_local_circulation_loops_fixpoint;
use crate::error::PnmlError;
use crate::explorer::{
    explore_checkpointable_observer, explore_observer, CheckpointableObserver, ExplorationConfig,
    ExplorationResult, ParallelExplorationObserver,
};
use crate::petri_net::PetriNet;
use crate::query_slice::{build_query_local_slice, QuerySlice};
use crate::reduction::{
    apply_final_place_gcd_scaling, reduce_irrelevant,
    reduce_iterative_structural_query_with_protected, reduce_query_guarded,
    ParallelExpandingObserver, ReducedNet,
};
use crate::resolved_predicate::{predicate_reduction_safe, remap_predicate_scaled};

use super::observer::ReachabilityObserver;
use super::types::PropertyTracker;

use crate::examinations::query_support::{
    reachability_support, relevance_cone_on_reduced_net, QuerySupport,
};
use crate::examinations::reachability_por::reachability_por_config;

#[derive(Debug, thiserror::Error)]
pub(in crate::examinations) enum ReachabilityExploreError {
    #[error(transparent)]
    Pnml(#[from] PnmlError),
    #[error("checkpoint error: {0}")]
    Checkpoint(#[from] std::io::Error),
}

fn explore_with_optional_checkpoint<O>(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut O,
) -> Result<ExplorationResult, std::io::Error>
where
    O: ParallelExplorationObserver + CheckpointableObserver + Send,
{
    if config.checkpoint().is_some() {
        explore_checkpointable_observer(net, config, observer)
    } else {
        Ok(explore_observer(net, config, observer))
    }
}

pub(in crate::examinations) fn reduce_reachability_queries(
    net: &PetriNet,
    trackers: &[PropertyTracker],
) -> Result<ReducedNet, PnmlError> {
    let initial_protected = reachability_support(&ReducedNet::identity(net), trackers)
        .map(|support| protected_places_for_prefire(net, &support))
        .unwrap_or_else(|| vec![true; net.num_places()]);
    let reduced = reduce_iterative_structural_query_with_protected(net, &initial_protected)?;
    let mut reduced = reduce_query_guarded(reduced, |r| {
        let support = reachability_support(r, trackers)?;
        Some(protected_places_for_prefire(&r.net, &support))
    })?;

    // Rule I: remove places and transitions provably irrelevant to
    // the query. The closure of the query support identifies the
    // connected component(s) needed; everything else is pruned.
    if let Some(support) = reachability_support(&reduced, trackers) {
        let closure =
            crate::examinations::query_support::closure_on_reduced_net(&reduced.net, support);
        if let Some(step) = reduce_irrelevant(&reduced.net, &closure) {
            let before = reduced.net.num_places() + reduced.net.num_transitions();
            reduced = reduced.compose(&step)?;
            let after = reduced.net.num_places() + reduced.net.num_transitions();
            if after < before {
                eprintln!(
                    "Rule I pruned {} places+transitions ({} → {})",
                    before - after,
                    before,
                    after,
                );
            }
        }
    }

    apply_final_place_gcd_scaling(&mut reduced)?;
    Ok(reduced)
}

pub(in crate::examinations) fn build_reachability_slice(
    reduced: &ReducedNet,
    trackers: &[PropertyTracker],
) -> Option<(QuerySlice, Vec<PropertyTracker>)> {
    let support = reachability_support(reduced, trackers)?;
    let has_transition_support = support.transitions.iter().any(|&t| t);
    let protected_seed_places = support.places.clone();
    let cone = relevance_cone_on_reduced_net(&reduced.net, support);
    let slice = build_query_local_slice(&reduced.net, &cone)?;
    // Rule H: contract circulation loops inside the query-local slice.
    // Only for place-based queries (no fireability transition references).
    let slice = if !has_transition_support {
        reduce_query_local_circulation_loops_fixpoint(slice.clone(), &protected_seed_places)
            .unwrap_or(slice)
    } else {
        slice
    };
    let place_map = slice.compose_place_map(&reduced.place_map);
    let trans_map = slice.compose_transition_map(&reduced.transition_map);

    // The slice is built atop `reduced.net`, whose surviving places may have
    // been divided by `reduced.place_scales` during GCD scaling (a place `p`
    // now holds `m_orig / place_scales[p]`). The non-slice path restores the
    // original coordinates by multiplying via `ParallelExpandingObserver`, but
    // the slice evaluates predicates directly on the scaled marking. So the
    // remap must be scale-aware: `IntLe` cardinality comparisons are rewritten
    // to be exactly equivalent against the scaled slice marking. If any
    // predicate cannot be transformed exactly (mixed scales in one token sum, or
    // a scaled tokens-vs-tokens comparison), `remap_predicate_scaled` returns
    // `None`, which propagates here to decline the slice entirely — routing all
    // trackers to the sound non-slice reduced-net path. `place_scales` is indexed
    // by original place index, matching the predicate's place indices and the
    // source index of `place_map`.
    let place_scales = &reduced.place_scales;
    let remapped_trackers = trackers
        .iter()
        .cloned()
        .map(|mut tracker| {
            if tracker.verdict.is_none() {
                tracker.predicate = remap_predicate_scaled(
                    &tracker.predicate,
                    &place_map,
                    &trans_map,
                    place_scales,
                )?;
            }
            Some(tracker)
        })
        .collect::<Option<Vec<_>>>()?;

    Some((slice, remapped_trackers))
}

pub(crate) fn protected_places_for_prefire(net: &PetriNet, support: &QuerySupport) -> Vec<bool> {
    let mut protected = support.places.clone();
    for (idx, targeted) in support.transitions.iter().enumerate() {
        if !*targeted {
            continue;
        }
        for arc in net.transitions[idx]
            .inputs
            .iter()
            .chain(net.transitions[idx].outputs.iter())
        {
            protected[arc.place.0 as usize] = true;
        }
    }
    protected
}

pub(in crate::examinations) fn explore_reachability_on_reduced_net(
    net: &PetriNet,
    reduced: &ReducedNet,
    trackers: Vec<PropertyTracker>,
    config: &ExplorationConfig,
) -> Result<(ExplorationResult, Vec<PropertyTracker>), ReachabilityExploreError> {
    let por_config = reachability_por_config(reduced, &trackers, config);
    let mut observer = ReachabilityObserver::from_trackers(net, trackers);
    let result = {
        let mut expanding = ParallelExpandingObserver::new(reduced, &mut observer);
        let result = explore_with_optional_checkpoint(&reduced.net, &por_config, &mut expanding)?;
        if let Some(error) = expanding.take_error() {
            return Err(error.into());
        }
        result
    };
    Ok((result, observer.into_trackers()))
}

pub(in crate::examinations) fn explore_reachability_on_slice(
    slice: &QuerySlice,
    trackers: Vec<PropertyTracker>,
    config: &ExplorationConfig,
) -> Result<(ExplorationResult, Vec<PropertyTracker>), ReachabilityExploreError> {
    let reduced = ReducedNet::identity(&slice.net);
    let por_config = reachability_por_config(&reduced, &trackers, config);
    let mut observer = ReachabilityObserver::from_trackers(&slice.net, trackers);
    let result = explore_with_optional_checkpoint(&slice.net, &por_config, &mut observer)?;
    Ok((result, observer.into_trackers()))
}

/// Check if all predicates are safe after reduction (referenced entities survived).
pub(in crate::examinations) fn all_predicates_reduction_safe(
    net: &PetriNet,
    reduced: &ReducedNet,
    trackers: &[PropertyTracker],
) -> bool {
    trackers.iter().all(|tracker| {
        tracker.verdict.is_some()
            || predicate_reduction_safe(&tracker.predicate, net, &reduced.transition_map)
    })
}

/// A verdict-preserving reduced net for the symbolic *seeding* lanes
/// (PDR / CHC / LP), with every unresolved tracker's predicate rewritten onto
/// the reduced (GCD-scaled) coordinates.
///
/// # Soundness
///
/// Built from the *same* [`reduce_reachability_queries`] +
/// [`all_predicates_reduction_safe`] gate that the exhaustive Phase-3 BFS
/// (`run_reduced_reachability_fallback`) already trusts to produce the
/// authoritative original-net EF/AG verdict. Two facts compose:
///
/// 1. PDR/CHC/LP are sound EF/AG proof systems on *whatever* net they run on, so
///    a definite verdict on the reduced net equals the reduced-net exhaustive
///    BFS verdict for the same (remapped) predicate.
/// 2. The reduced-net EF/AG verdict equals the original-net EF/AG verdict —
///    exactly the `ReductionMode::Reachability` guarantee Phase 3 relies on.
///
/// Therefore symbolic-on-reduced == BFS-on-reduced == BFS-on-original, and the
/// *boolean* verdict maps back 1:1 to the original tracker by `id`/position. No
/// marking/witness ever crosses the boundary, so no `expand_marking_into` is
/// needed; predicate remapping (`remap_predicate_scaled`) fully accounts for the
/// GCD scaling of surviving places.
pub(in crate::examinations) struct SymbolicSeedReduction {
    reduced: ReducedNet,
    /// Trackers whose predicates have been rewritten onto reduced/scaled
    /// coordinates. Same length and ordering as the original trackers, with
    /// `id` preserved. The `verdict`/`resolved_by` fields are a build-time
    /// snapshot; [`Self::worker_trackers`] refreshes them before each lane so
    /// trackers resolved by an earlier phase are skipped.
    remapped: Vec<PropertyTracker>,
}

impl SymbolicSeedReduction {
    /// The reduced net a seeding lane should run on.
    pub(in crate::examinations) fn net(&self) -> &PetriNet {
        &self.reduced.net
    }

    /// Clone the remapped trackers and sync their verdict state from the live
    /// `current` trackers, so a lane skips anything an earlier phase resolved.
    /// Predicates stay in reduced coordinates; `id` order is preserved 1:1.
    pub(in crate::examinations) fn worker_trackers(
        &self,
        current: &[PropertyTracker],
    ) -> Vec<PropertyTracker> {
        debug_assert_eq!(self.remapped.len(), current.len());
        self.remapped
            .iter()
            .zip(current.iter())
            .map(|(remapped, live)| {
                debug_assert_eq!(remapped.id, live.id);
                let mut tracker = remapped.clone();
                tracker.verdict = live.verdict;
                tracker.resolved_by = live.resolved_by;
                tracker.flushed = live.flushed;
                tracker
            })
            .collect()
    }
}

/// Build a verdict-preserving reduced net for the symbolic seeding lanes, or
/// `None` to decline (lanes stay on the original net) whenever the reduction
/// would not be provably verdict-preserving for every unresolved tracker, does
/// not actually shrink the net, or cannot be remapped exactly.
///
/// Declining is always sound: the lanes are first-writer-wins and only ever ADD
/// definite verdicts, so falling through to the original net (or, ultimately,
/// the exhaustive BFS) can never produce a wrong answer.
pub(in crate::examinations) fn build_symbolic_seed_reduction(
    net: &PetriNet,
    trackers: &[PropertyTracker],
) -> Option<SymbolicSeedReduction> {
    // Nothing to seed if every formula is already resolved.
    if trackers.iter().all(|t| t.verdict.is_some()) {
        return None;
    }

    let reduced = reduce_reachability_queries(net, trackers).ok()?;

    // Only worth the clone + remap if the reduced net is strictly smaller; an
    // identity-sized reduction just adds overhead with no speedup.
    let orig_size = net.num_places() + net.num_transitions();
    let reduced_size = reduced.net.num_places() + reduced.net.num_transitions();
    if reduced_size >= orig_size {
        return None;
    }

    // Verdict-preservation gate: every entity referenced by an unresolved
    // predicate must survive the reduction (the identical guard Phase-3 BFS
    // checks before trusting the reduced net for the authoritative verdict).
    if !all_predicates_reduction_safe(net, &reduced, trackers) {
        return None;
    }

    // Rewrite every unresolved predicate onto reduced/scaled coordinates,
    // scale-aware. All-or-nothing: if any cannot be represented exactly
    // (removed entity, or a scaled tokens-vs-tokens comparison), decline so no
    // lane ever runs a partially-remapped tracker set — those trackers stay on
    // the sound original-net lanes.
    let remapped = trackers
        .iter()
        .cloned()
        .map(|mut tracker| {
            if tracker.verdict.is_none() {
                tracker.predicate = remap_predicate_scaled(
                    &tracker.predicate,
                    &reduced.place_map,
                    &reduced.transition_map,
                    &reduced.place_scales,
                )?;
            }
            Some(tracker)
        })
        .collect::<Option<Vec<_>>>()?;

    Some(SymbolicSeedReduction { reduced, remapped })
}
