// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::time::Instant;

use crate::error::PnmlError;
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};

use super::super::gcd_scale::apply_final_place_gcd_scaling;
use super::super::{ReducedNet, ReductionMode};
use super::prefire::apply_query_guarded_prefire;
use super::structural::{
    reduce_with_mode, reduce_with_protected_sinks, StructuralReductionSemantics,
};

/// Apply query-guarded prefire and structural reduction in a fixpoint loop.
///
/// After each prefire pass, re-runs structural reduction to catch newly-dead
/// transitions, isolated places, and cascade simplifications exposed by the
/// marking changes. Repeats until neither prefire nor structural reduction
/// makes progress.
///
/// The `protected_places_for` callback receives the current [`ReducedNet`] and
/// returns a protected-places mask for prefire, or `None` to skip prefire
/// entirely (e.g., when all queries are already resolved).
///
/// Does **not** apply GCD scaling. Callers should apply
/// [`apply_final_place_gcd_scaling`](super::super::gcd_scale::apply_final_place_gcd_scaling)
/// once after this function returns.
pub(crate) fn reduce_query_guarded<F>(
    reduced: ReducedNet,
    protected_places_for: F,
) -> Result<ReducedNet, PnmlError>
where
    F: Fn(&ReducedNet) -> Option<Vec<bool>>,
{
    let mut current = reduced;
    // Prefire chains have length at most num_transitions (each transition
    // prefires at most once before its inputs are exhausted). Cycle
    // transitions can oscillate tokens indefinitely without enabling new
    // structural reductions, so we cap consecutive prefire-without-reduction
    // rounds to prevent infinite loops.
    let mut prefire_only_rounds: usize = 0;
    loop {
        let Some(protected) = protected_places_for(&current) else {
            return Ok(current);
        };

        let prefired = apply_query_guarded_prefire(&mut current, &protected)?;

        let step = reduce_iterative_structural_query_with_protected(&current.net, &protected)?;
        let reduced_more = step.report.has_reductions();

        if reduced_more {
            current = current.compose(&step)?;
            prefire_only_rounds = 0;
        }

        if !prefired && !reduced_more {
            return Ok(current);
        }

        if prefired && !reduced_more {
            prefire_only_rounds += 1;
            if prefire_only_rounds > current.net.num_transitions() {
                return Ok(current);
            }
        }
    }
}

/// Apply reductions until a full pass finds no new reducible structure.
///
/// Self-loop-touching places from earlier rounds are propagated as
/// LP-redundancy protection to prevent soundness bugs where
/// `restore_self_loops` cannot remap arcs for places removed as redundant.
///
/// **Quarantined (#1503):** unsound for GlobalProperties exams (QuasiLiveness,
/// Liveness, StableMarking) — agglomeration suppresses real firing behavior.
/// Retained for re-enablement once a sound reduction contract is validated.
#[allow(dead_code)]
pub(crate) fn reduce_iterative_structural(net: &PetriNet) -> Result<ReducedNet, PnmlError> {
    reduce_iterative_structural_with_protected(net, &[])
}

#[allow(dead_code)]
pub(crate) fn reduce_iterative_structural_with_protected(
    net: &PetriNet,
    base_protected: &[bool],
) -> Result<ReducedNet, PnmlError> {
    reduce_iterative_structural_with_semantics(
        net,
        base_protected,
        StructuralReductionSemantics::ExactMarking,
        None,
    )
}

/// Structural reduction that keeps only rules sound for OneSafe.
///
/// Source-place elimination, agglomeration, sink-transition removal,
/// redundant-place removal, and non-decreasing place removal can hide
/// original-net token magnitudes, so this lane keeps the deadlock-safe
/// enabling contract while excluding those magnitude-changing rules.
pub(crate) fn reduce_iterative_structural_one_safe(
    net: &PetriNet,
) -> Result<ReducedNet, PnmlError> {
    reduce_iterative_structural_with_semantics(
        net,
        &[],
        StructuralReductionSemantics::OneSafe,
        None,
    )
}

pub(crate) fn reduce_iterative_structural_query_with_protected(
    net: &PetriNet,
    base_protected: &[bool],
) -> Result<ReducedNet, PnmlError> {
    reduce_iterative_structural_with_semantics(
        net,
        base_protected,
        StructuralReductionSemantics::QueryRelevantOnly,
        None,
    )
}

/// Deadlock-safe structural reduction with query-protected places.
///
/// Skips Rule K (self-loop arc removal) and Rule N (never-disabling arcs)
/// which change transition enabling conditions. Used in tests to verify
/// that CTL/LTL wrong answers come from structural reduction (not slicing).
#[cfg(test)]
pub(crate) fn reduce_iterative_structural_deadlock_safe_with_protected(
    net: &PetriNet,
    base_protected: &[bool],
) -> Result<ReducedNet, PnmlError> {
    reduce_iterative_structural_with_semantics(
        net,
        base_protected,
        StructuralReductionSemantics::DeadlockSafe,
        None,
    )
}

/// Test-only dead/constant/isolated structural reduction candidate.
///
/// Keep this lane quarantined while production CTL/LTL stays on
/// `ReducedNet::identity(net)`. IBM5964 parity coverage protects against the
/// candidate silently widening again, but that benchmark alone is not a proof
/// of general CTL/LTL safety.
#[cfg(test)]
pub(crate) fn reduce_iterative_temporal_projection_candidate(
    net: &PetriNet,
) -> Result<ReducedNet, PnmlError> {
    reduce_iterative_structural_with_semantics(
        net,
        &[],
        StructuralReductionSemantics::TemporalProjectionCandidate,
        None,
    )
}

/// Reduction fixpoint with an optional wall-clock `deadline`.
///
/// This loop runs as MCC pre-exploration preprocessing. On a large net a single
/// round can be expensive and the loop can otherwise burn the entire budget
/// before exploration begins (#4). At the top of each round we poll the
/// `deadline`: once `Instant::now() >= deadline` we stop and return the
/// reductions accumulated so far. A partially-reduced net is always sound — the
/// fixpoint composes verdict-preserving steps, so stopping early yields a net
/// that is at worst less reduced than the full fixpoint, never one that changes
/// any answer. `deadline == None` preserves the original run-to-fixpoint
/// behaviour.
fn reduce_iterative_structural_with_semantics(
    net: &PetriNet,
    base_protected: &[bool],
    semantics: StructuralReductionSemantics,
    deadline: Option<Instant>,
) -> Result<ReducedNet, PnmlError> {
    assert!(
        base_protected.is_empty() || base_protected.len() == net.num_places(),
        "protected place mask must match net place count"
    );

    let mut current = net.clone();
    let mut combined = ReducedNet::identity(net);
    let mut first_round = true;

    loop {
        // Fail-fast on budget exhaustion: returning the accumulated `combined`
        // (a partially-reduced but verdict-preserving net) is sound.
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Ok(combined);
        }

        let np_current = current.num_places();
        let mut round_protected = vec![false; np_current];
        for (orig_place, &protected) in base_protected.iter().enumerate() {
            if !protected {
                continue;
            }
            if let Some(current_place) = combined.place_map[orig_place] {
                round_protected[current_place.0 as usize] = true;
            }
        }

        // Map accumulated self-loop places to current-net coordinates.
        // After round N, combined.report.self_loop_transitions contains
        // all self-loops from rounds 1..N in original-net indices.
        // We map their arc places through combined.place_map to get
        // current-net indices, then pass them as extra protection.
        for &TransitionIdx(t) in &combined.report.self_loop_transitions {
            let orig_trans = &net.transitions[t as usize];
            for arc in orig_trans.inputs.iter().chain(orig_trans.outputs.iter()) {
                if let Some(cur_p) = combined.place_map[arc.place.0 as usize] {
                    round_protected[cur_p.0 as usize] = true;
                }
            }
        }

        let mut protected_sink_transitions = Vec::new();
        if matches!(semantics, StructuralReductionSemantics::QueryRelevantOnly) {
            for &token_place in &combined.report.token_eliminated_places {
                for transition in &net.transitions {
                    if transition.inputs.iter().any(|arc| arc.place == token_place) {
                        for arc in &transition.outputs {
                            if let Some(cur_p) = combined.place_map[arc.place.0 as usize] {
                                round_protected[cur_p.0 as usize] = true;
                            }
                        }
                    }
                }
            }
            let mut representative_counts = vec![0usize; current.num_places()];
            for PlaceIdx(p) in combined.place_map.iter().flatten() {
                representative_counts[*p as usize] += 1;
            }
            for (place, count) in representative_counts.into_iter().enumerate() {
                if count > 1 {
                    round_protected[place] = true;
                }
            }

            if !first_round {
                protected_sink_transitions = current
                    .transitions
                    .iter()
                    .zip(&combined.transition_unmap)
                    .map(
                        |(current_transition, &TransitionIdx(original_transition))| {
                            current_transition.outputs.is_empty()
                                && !net.transitions[original_transition as usize]
                                    .outputs
                                    .is_empty()
                        },
                    )
                    .collect();
            }
        }
        if matches!(semantics, StructuralReductionSemantics::ExactMarking)
            && !first_round
            && combined.report.self_loop_transitions.is_empty()
        {
            let mut stable_removed = vec![false; net.num_places()];
            for &PlaceIdx(p) in combined
                .report
                .constant_places
                .iter()
                .chain(&combined.report.isolated_places)
            {
                stable_removed[p as usize] = true;
            }
            protected_sink_transitions = current
                .transitions
                .iter()
                .zip(&combined.transition_unmap)
                .map(
                    |(current_transition, &TransitionIdx(original_transition))| {
                        current_transition.outputs.is_empty()
                            && net.transitions[original_transition as usize]
                                .outputs
                                .iter()
                                .any(|arc| !stable_removed[arc.place.0 as usize])
                    },
                )
                .collect();
        }

        let step = reduce_with_protected_sinks(
            &current,
            &round_protected,
            &protected_sink_transitions,
            semantics,
        );
        if !step.report.has_reductions() {
            return Ok(combined);
        }

        current = step.net.clone();
        combined = combined.compose(&step)?;
        first_round = false;
    }
}

/// Apply structural reductions and then normalize surviving place scales once.
#[allow(dead_code)]
pub(crate) fn reduce_iterative(net: &PetriNet) -> Result<ReducedNet, PnmlError> {
    let mut combined = reduce_iterative_structural(net)?;
    apply_final_place_gcd_scaling(&mut combined)?;
    Ok(combined)
}

/// Deadlock-safe reduction: skips Rule K (self-loop arcs) and Rule N
/// (never-disabling arcs). Removing self-loop input arcs makes transitions
/// easier to fire, potentially eliminating deadlocks. Removing never-disabling
/// output arcs can starve downstream transitions, also affecting deadlocks.
///
/// **Quarantined (#1506):** currently unsound for deadlock analysis on nets
/// like IOTPpurchase-PT (agglomeration/source-place elimination introduces
/// spurious deadlocks). Retained as infrastructure for future deadlock-safe
/// reduction contract validation.
#[allow(dead_code)]
pub(crate) fn reduce_iterative_deadlock_safe(net: &PetriNet) -> Result<ReducedNet, PnmlError> {
    let mut combined = reduce_iterative_structural_with_semantics(
        net,
        &[],
        StructuralReductionSemantics::DeadlockSafe,
        None,
    )?;
    apply_final_place_gcd_scaling(&mut combined)?;
    Ok(combined)
}

/// Apply query-aware structural reductions gated by [`ReductionMode`].
///
/// The mode's `allows_*` methods determine which reduction rules are included
/// in each fixpoint iteration. This is the primary entrypoint for CTL/LTL
/// examinations that need temporal-logic-aware reduction selection.
///
/// The `base_protected` mask protects query-relevant places from removal.
///
/// `deadline` bounds this preprocessing fixpoint (#4): this loop runs before
/// MCC exploration and a single round on a large net can be expensive, so
/// without a bound it can burn the entire budget. At the top of each round we
/// poll `Instant::now() >= deadline` and, on expiry, return the reductions
/// accumulated so far. The fixpoint composes only verdict-preserving steps, so
/// the partially-reduced net is sound — at worst less reduced than the full
/// fixpoint, never wrong. `deadline == None` preserves the original
/// run-to-fixpoint behaviour.
pub(crate) fn reduce_iterative_structural_with_mode(
    net: &PetriNet,
    base_protected: &[bool],
    mode: ReductionMode,
    deadline: Option<Instant>,
) -> Result<ReducedNet, PnmlError> {
    assert!(
        base_protected.is_empty() || base_protected.len() == net.num_places(),
        "protected place mask must match net place count"
    );

    let mut current = net.clone();
    let mut combined = ReducedNet::identity(net);

    loop {
        // Fail-fast on budget exhaustion: returning the accumulated `combined`
        // (a partially-reduced but verdict-preserving net) is sound.
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Ok(combined);
        }

        let np_current = current.num_places();
        let mut round_protected = vec![false; np_current];
        for (orig_place, &protected) in base_protected.iter().enumerate() {
            if !protected {
                continue;
            }
            if let Some(current_place) = combined.place_map[orig_place] {
                round_protected[current_place.0 as usize] = true;
            }
        }

        // Map accumulated self-loop places to current-net coordinates.
        for &TransitionIdx(t) in &combined.report.self_loop_transitions {
            let orig_trans = &net.transitions[t as usize];
            for arc in orig_trans.inputs.iter().chain(orig_trans.outputs.iter()) {
                if let Some(cur_p) = combined.place_map[arc.place.0 as usize] {
                    round_protected[cur_p.0 as usize] = true;
                }
            }
        }

        // Invariant (#4303): cycle survivors from prior rounds must be
        // protected in the current round. Rule H drops all cycle transitions,
        // so the survivor becomes structurally isolated and would otherwise
        // be deleted by isolated-place or LP-redundancy removal in the next
        // iteration — stranding every `place_map[absorbed] = Some(survivor)`
        // redirect at a dead reduced-net index. Protecting the survivor keeps
        // the aggregate-token-count slot alive for `expand_marking` and for
        // any downstream query that references a place in the merged cycle.
        for cycle in &combined.report.token_cycle_merges {
            if let Some(cur_p) = combined.place_map[cycle.survivor.0 as usize] {
                round_protected[cur_p.0 as usize] = true;
            }
        }

        let step = reduce_with_mode(&current, &round_protected, mode);
        if !step.report.has_reductions() {
            return Ok(combined);
        }

        current = step.net.clone();
        combined = combined.compose(&step)?;
    }
}
