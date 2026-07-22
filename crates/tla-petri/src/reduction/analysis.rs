// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::{BTreeMap, VecDeque};

use crate::petri_net::{Arc, PetriNet, PlaceIdx, TransitionIdx};

use super::model::{NeverDisablingArc, NeverDisablingProof, ReductionMode, ReductionReport};
use super::redundant::find_redundant_places;

// Re-export from sibling modules so external callers don't need to know the split.
pub(super) use super::analysis_agglomeration::{build_place_connectivity, find_agglomerations};
pub(crate) use super::analysis_invariant::{
    find_lateral_fusions, find_never_disabling_arcs, find_non_decreasing_places,
    find_parallel_places, find_token_eliminated_places,
};
#[allow(unused_imports)]
pub(super) use super::analysis_structural_rules::{
    find_simple_chain_places, find_sink_transitions, find_trap_dead_transitions,
};
pub(super) use super::analysis_transitions::{
    find_dominated_transitions, find_duplicate_transitions, find_self_loop_arcs,
    find_self_loop_transitions,
};

/// Per-rule progress counters for the combined EFMNOP fixpoint analysis.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EfmnopProgress {
    pub rule_e_workqueue_dead: usize,
    pub rule_o_dead: usize,
    pub rule_o_orphan: usize,
    pub rule_e_parallel: usize,
    pub rule_f_non_decreasing: usize,
    pub rule_f_source: usize,
    pub rule_m_duplicate: usize,
    pub rule_m_dominated: usize,
    pub rule_m_self_loop: usize,
    pub rule_n_self_loop_arcs: usize,
    pub rule_n_never_disabling: usize,
    pub rule_n_fixpoint_lower_bound: usize,
    pub rule_p_pre_agglomeration: usize,
    pub rule_p_post_agglomeration: usize,
}

/// Result of the combined EFMNOP structural fixpoint analysis.
#[derive(Debug, Clone, Default)]
pub(crate) struct EfmnopAnalysis {
    pub report: ReductionReport,
    pub iterations: usize,
    pub dead_removed_by_cascade: usize,
    pub per_rule_progress: EfmnopProgress,
}

#[derive(Debug, Clone)]
pub(crate) struct EfmnopFixpointState {
    can_inc: Vec<bool>,
    can_dec: Vec<bool>,
    #[cfg_attr(not(test), allow(dead_code))]
    lower: Vec<u64>,
    fired_transitions: Vec<bool>,
}

impl EfmnopFixpointState {
    fn dead_transitions(&self) -> Vec<TransitionIdx> {
        self.fired_transitions
            .iter()
            .enumerate()
            .filter(|(_, fired)| !**fired)
            .map(|(idx, _)| TransitionIdx(idx as u32))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn can_inc(&self, place: PlaceIdx) -> bool {
        self.can_inc[place.0 as usize]
    }

    #[cfg(test)]
    pub(crate) fn can_dec(&self, place: PlaceIdx) -> bool {
        self.can_dec[place.0 as usize]
    }

    #[cfg(test)]
    pub(crate) fn lower_bound(&self, place: PlaceIdx) -> u64 {
        self.lower[place.0 as usize]
    }

    #[cfg(test)]
    pub(crate) fn fired(&self, transition: TransitionIdx) -> bool {
        self.fired_transitions[transition.0 as usize]
    }
}

#[derive(Debug, Clone, Default)]
struct EfmnopTransitionSummary {
    inputs: Vec<(PlaceIdx, u64)>,
    outputs: Vec<(PlaceIdx, u64)>,
}

fn aggregate_arc_weights(arcs: &[Arc]) -> Vec<(PlaceIdx, u64)> {
    // Petri semantics treat duplicate same-place arcs as additive weights.
    let mut weights = BTreeMap::<u32, u64>::new();
    for arc in arcs {
        let entry = weights.entry(arc.place.0).or_default();
        *entry = entry.saturating_add(arc.weight);
    }
    weights
        .into_iter()
        .map(|(place, weight)| (PlaceIdx(place), weight))
        .collect()
}

fn sort_dedup_transitions(transitions: &mut Vec<TransitionIdx>) {
    transitions.sort_by_key(|transition| transition.0);
    transitions.dedup();
}

fn transition_mask(num_transitions: usize, transitions: &[TransitionIdx]) -> Vec<bool> {
    let mut mask = vec![false; num_transitions];
    for &TransitionIdx(transition) in transitions {
        mask[transition as usize] = true;
    }
    mask
}

fn push_never_disabling_arc_unique(arcs: &mut Vec<NeverDisablingArc>, arc: NeverDisablingArc) {
    if arcs
        .iter()
        .any(|existing| existing.transition == arc.transition && existing.place == arc.place)
    {
        return;
    }
    arcs.push(arc);
    arcs.sort_by_key(|arc| (arc.transition.0, arc.place.0));
}

fn summarized_transitions(net: &PetriNet) -> Vec<EfmnopTransitionSummary> {
    net.transitions
        .iter()
        .map(|transition| EfmnopTransitionSummary {
            inputs: aggregate_arc_weights(&transition.inputs),
            outputs: aggregate_arc_weights(&transition.outputs),
        })
        .collect()
}

fn find_fixpoint_lower_bound_arcs(
    net: &PetriNet,
    state: &EfmnopFixpointState,
    dead_transitions: &[TransitionIdx],
    self_loop_transitions: &[TransitionIdx],
    protected_places: &[bool],
) -> Vec<NeverDisablingArc> {
    let summaries = summarized_transitions(net);
    let is_dead = transition_mask(net.num_transitions(), dead_transitions);
    let is_self_loop = transition_mask(net.num_transitions(), self_loop_transitions);
    let mut results = Vec::new();

    for place_idx in 0..net.num_places() {
        if protected_places.get(place_idx).copied().unwrap_or(false) {
            continue;
        }

        let place = PlaceIdx(place_idx as u32);
        let lower_bound = state.lower[place_idx];
        if lower_bound == 0 {
            continue;
        }

        let lower_bound_is_stable =
            summaries
                .iter()
                .enumerate()
                .all(|(transition_idx, summary)| {
                    if is_dead[transition_idx] || is_self_loop[transition_idx] {
                        return true;
                    }
                    weight_for(&summary.outputs, place) >= weight_for(&summary.inputs, place)
                });
        if !lower_bound_is_stable {
            continue;
        }

        for (transition_idx, summary) in summaries.iter().enumerate() {
            if is_dead[transition_idx] || is_self_loop[transition_idx] {
                continue;
            }
            let weight = weight_for(&summary.inputs, place);
            if weight == 0 || weight > lower_bound {
                continue;
            }
            results.push(NeverDisablingArc {
                transition: TransitionIdx(transition_idx as u32),
                place,
                weight,
                proof: NeverDisablingProof::FixpointLowerBound { lower_bound },
            });
        }
    }

    results.sort_by_key(|arc| (arc.transition.0, arc.place.0));
    results
}

fn consumers_by_place(
    net: &PetriNet,
    summaries: &[EfmnopTransitionSummary],
) -> Vec<Vec<TransitionIdx>> {
    let mut consumers = vec![Vec::new(); net.num_places()];
    for (transition_idx, summary) in summaries.iter().enumerate() {
        for &(place, _) in &summary.inputs {
            consumers[place.0 as usize].push(TransitionIdx(transition_idx as u32));
        }
    }
    consumers
}

fn weight_for(arcs: &[(PlaceIdx, u64)], place: PlaceIdx) -> u64 {
    arcs.iter()
        .find_map(|&(arc_place, weight)| (arc_place == place).then_some(weight))
        .unwrap_or(0)
}

fn initially_enabled(summary: &EfmnopTransitionSummary, initial_marking: &[u64]) -> bool {
    summary
        .inputs
        .iter()
        .all(|&(place, weight)| initial_marking[place.0 as usize] >= weight)
}

fn enabled_under_fixpoint(
    summary: &EfmnopTransitionSummary,
    initial_marking: &[u64],
    state: &EfmnopFixpointState,
) -> bool {
    summary.inputs.iter().all(|&(place, weight)| {
        initial_marking[place.0 as usize] >= weight || state.can_inc[place.0 as usize]
    })
}

fn enqueue_consumers(
    work: &mut VecDeque<TransitionIdx>,
    consumers: &[Vec<TransitionIdx>],
    fired_transitions: &[bool],
    place: PlaceIdx,
) {
    for &consumer in &consumers[place.0 as usize] {
        if !fired_transitions[consumer.0 as usize] {
            work.push_back(consumer);
        }
    }
}

fn mark_can_inc(
    state: &mut EfmnopFixpointState,
    work: &mut VecDeque<TransitionIdx>,
    consumers: &[Vec<TransitionIdx>],
    place: PlaceIdx,
) {
    let place_idx = place.0 as usize;
    if state.can_inc[place_idx] {
        return;
    }
    state.can_inc[place_idx] = true;
    enqueue_consumers(work, consumers, &state.fired_transitions, place);
}

fn mark_can_dec(
    state: &mut EfmnopFixpointState,
    work: &mut VecDeque<TransitionIdx>,
    consumers: &[Vec<TransitionIdx>],
    place: PlaceIdx,
    consumed_weight: u64,
) {
    let place_idx = place.0 as usize;
    let new_lower = state.lower[place_idx].saturating_sub(consumed_weight);
    let lower_changed = new_lower < state.lower[place_idx];
    if lower_changed {
        state.lower[place_idx] = new_lower;
    }
    if state.can_dec[place_idx] && !lower_changed {
        return;
    }
    state.can_dec[place_idx] = true;
    enqueue_consumers(work, consumers, &state.fired_transitions, place);
}

fn process_efmnop_enabled_transition(
    transition_idx: TransitionIdx,
    summaries: &[EfmnopTransitionSummary],
    consumers: &[Vec<TransitionIdx>],
    state: &mut EfmnopFixpointState,
    work: &mut VecDeque<TransitionIdx>,
) {
    let idx = transition_idx.0 as usize;
    if state.fired_transitions[idx] {
        return;
    }
    state.fired_transitions[idx] = true;

    let summary = &summaries[idx];
    for &(place, weight) in &summary.inputs {
        mark_can_dec(state, work, consumers, place, weight);
    }
    for &(place, output_weight) in &summary.outputs {
        let input_weight = weight_for(&summary.inputs, place);
        if output_weight > input_weight {
            mark_can_inc(state, work, consumers, place);
        }
    }
}

/// Compute the first EFMNOP dataflow slice: Tapaal-style transition work queue
/// over CAN_INC/CAN_DEC bits and conservative per-place lower bounds.
///
/// This state is deliberately used only as a dead-transition seed for now.
/// `fired_transitions` is an over-approximation of transitions that may become
/// enabled, so `!fired` is safe to classify as dead; lower-bound Rule N proof
/// emission remains a later slice.
#[must_use]
pub(crate) fn compute_efmnop_fixpoint_state(net: &PetriNet) -> EfmnopFixpointState {
    let summaries = summarized_transitions(net);
    let consumers = consumers_by_place(net, &summaries);
    let mut state = EfmnopFixpointState {
        can_inc: vec![false; net.num_places()],
        can_dec: vec![false; net.num_places()],
        lower: net.initial_marking.clone(),
        fired_transitions: vec![false; net.num_transitions()],
    };
    let mut work = VecDeque::new();

    for (transition_idx, summary) in summaries.iter().enumerate() {
        if initially_enabled(summary, &net.initial_marking) {
            process_efmnop_enabled_transition(
                TransitionIdx(transition_idx as u32),
                &summaries,
                &consumers,
                &mut state,
                &mut work,
            );
        }
    }

    while let Some(transition) = work.pop_front() {
        if state.fired_transitions[transition.0 as usize] {
            continue;
        }
        if enabled_under_fixpoint(
            &summaries[transition.0 as usize],
            &net.initial_marking,
            &state,
        ) {
            process_efmnop_enabled_transition(
                transition, &summaries, &consumers, &mut state, &mut work,
            );
        }
    }

    state
}

/// Run the combined EFMNOP structural analysis to a full fixpoint.
///
/// Each round evaluates the enabled rule families against the reductions found
/// in previous rounds. Newly discovered reductions are merged into the report,
/// and the process repeats until a full pass adds nothing new.
#[must_use]
pub(crate) fn analyze_efmnop_fixpoint(
    net: &PetriNet,
    protected_places: &[bool],
    mode: ReductionMode,
) -> EfmnopAnalysis {
    assert!(
        protected_places.is_empty() || protected_places.len() == net.num_places(),
        "protected place mask must match net place count"
    );

    let dead_removed_by_cascade = if mode.allows_dead_transition_removal() {
        let mut has_producer = vec![false; net.num_places()];
        for transition in &net.transitions {
            for arc in &transition.outputs {
                has_producer[arc.place.0 as usize] = true;
            }
        }

        let initial_dead = net
            .transitions
            .iter()
            .filter(|transition| {
                transition.inputs.iter().any(|arc| {
                    let place = arc.place.0 as usize;
                    !has_producer[place] && net.initial_marking[place] < arc.weight
                })
            })
            .count();

        find_dead_transitions(net)
            .len()
            .saturating_sub(initial_dead)
    } else {
        0
    };

    let mut report = ReductionReport::default();
    let mut per_rule_progress = EfmnopProgress::default();
    let mut iterations = 0;
    let workqueue_state = compute_efmnop_fixpoint_state(net);
    let workqueue_dead_transitions = workqueue_state.dead_transitions();
    let workqueue_dead_mask = transition_mask(net.num_transitions(), &workqueue_dead_transitions);

    loop {
        let snapshot = report.clone();
        let num_places = net.num_places();
        let num_transitions = net.num_transitions();
        let mut self_loop_protected = vec![false; num_places];

        for (place_idx, &protected) in protected_places.iter().enumerate() {
            self_loop_protected[place_idx] = protected;
        }
        for &transition_idx in &snapshot.self_loop_transitions {
            let transition = &net.transitions[transition_idx.0 as usize];
            for arc in transition.inputs.iter().chain(transition.outputs.iter()) {
                self_loop_protected[arc.place.0 as usize] = true;
            }
        }
        for merge in &snapshot.parallel_places {
            self_loop_protected[merge.canonical.0 as usize] = true;
        }

        let mut already_removed = vec![false; num_places];
        for &place in &snapshot.constant_places {
            if !self_loop_protected[place.0 as usize] {
                already_removed[place.0 as usize] = true;
            }
        }
        for &place in &snapshot.source_places {
            already_removed[place.0 as usize] = true;
        }
        for &place in &snapshot.isolated_places {
            if !self_loop_protected[place.0 as usize] {
                already_removed[place.0 as usize] = true;
            }
        }
        for agglomeration in &snapshot.pre_agglomerations {
            if !self_loop_protected[agglomeration.place.0 as usize] {
                already_removed[agglomeration.place.0 as usize] = true;
            }
        }
        for agglomeration in &snapshot.post_agglomerations {
            if !self_loop_protected[agglomeration.place.0 as usize] {
                already_removed[agglomeration.place.0 as usize] = true;
            }
        }
        for merge in &snapshot.parallel_places {
            already_removed[merge.duplicate.0 as usize] = true;
        }
        for &place in &snapshot.non_decreasing_places {
            already_removed[place.0 as usize] = true;
        }
        for &place in &snapshot.redundant_places {
            already_removed[place.0 as usize] = true;
        }

        let mut changed = false;

        if mode.allows_dead_transition_removal() {
            let mut rule_o_dead_transitions = find_dead_transitions(net);
            let trap_rule_o_dead_transitions =
                find_trap_dead_transitions(net, &rule_o_dead_transitions);
            rule_o_dead_transitions.extend(trap_rule_o_dead_transitions);
            sort_dedup_transitions(&mut rule_o_dead_transitions);
            let rule_o_dead_mask = transition_mask(num_transitions, &rule_o_dead_transitions);

            let mut dead_transitions = rule_o_dead_transitions;
            dead_transitions.extend(workqueue_dead_transitions.iter().copied());
            sort_dedup_transitions(&mut dead_transitions);

            let trap_combined_dead_transitions = find_trap_dead_transitions(net, &dead_transitions);
            dead_transitions.extend(trap_combined_dead_transitions);
            sort_dedup_transitions(&mut dead_transitions);

            for transition in dead_transitions {
                if !report.dead_transitions.contains(&transition) {
                    report.dead_transitions.push(transition);
                    let transition_idx = transition.0 as usize;
                    if workqueue_dead_mask[transition_idx] && !rule_o_dead_mask[transition_idx] {
                        per_rule_progress.rule_e_workqueue_dead += 1;
                    } else {
                        per_rule_progress.rule_o_dead += 1;
                    }
                    changed = true;
                }
            }
        }

        if mode.allows_constant_place_removal() {
            for place in find_constant_places(net) {
                if !report.constant_places.contains(&place) {
                    report.constant_places.push(place);
                    changed = true;
                }
            }
        }

        if mode.allows_isolated_place_removal() {
            for place in find_isolated_places(net) {
                if !report.isolated_places.contains(&place) {
                    report.isolated_places.push(place);
                    per_rule_progress.rule_o_orphan += 1;
                    changed = true;
                }
            }
            for place in find_cascade_isolated_places(net, &snapshot.dead_transitions) {
                if !report.isolated_places.contains(&place) {
                    report.isolated_places.push(place);
                    per_rule_progress.rule_o_orphan += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_source_place_removal() {
            for place in find_source_places(net, &snapshot.dead_transitions, protected_places) {
                if !report.source_places.contains(&place) {
                    report.source_places.push(place);
                    per_rule_progress.rule_f_source += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_parallel_place_merge() {
            for merge in find_parallel_places(net, &snapshot.dead_transitions, &self_loop_protected)
            {
                if !report.parallel_places.contains(&merge) {
                    report.parallel_places.push(merge);
                    per_rule_progress.rule_e_parallel += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_non_decreasing_place_removal() {
            for place in find_non_decreasing_places(
                net,
                &snapshot.dead_transitions,
                &self_loop_protected,
                &snapshot.source_places,
            ) {
                if !report.non_decreasing_places.contains(&place) {
                    report.non_decreasing_places.push(place);
                    per_rule_progress.rule_f_non_decreasing += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_duplicate_transition_removal() {
            for class in find_duplicate_transitions(net, &snapshot.dead_transitions) {
                if !report.duplicate_transitions.contains(&class) {
                    per_rule_progress.rule_m_duplicate += class.duplicates.len();
                    report.duplicate_transitions.push(class);
                    changed = true;
                }
            }
        }

        if mode.allows_self_loop_transition_removal() {
            for transition in find_self_loop_transitions(net, &snapshot.dead_transitions) {
                if !report.self_loop_transitions.contains(&transition) {
                    report.self_loop_transitions.push(transition);
                    per_rule_progress.rule_m_self_loop += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_dominated_transition_removal() {
            for transition in find_dominated_transitions(net, &snapshot.dead_transitions) {
                if !report.dominated_transitions.contains(&transition) {
                    report.dominated_transitions.push(transition);
                    per_rule_progress.rule_m_dominated += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_self_loop_arc_removal() {
            for arc in find_self_loop_arcs(
                net,
                &snapshot.dead_transitions,
                &snapshot.self_loop_transitions,
                protected_places,
            ) {
                if !report.self_loop_arcs.contains(&arc) {
                    report.self_loop_arcs.push(arc);
                    per_rule_progress.rule_n_self_loop_arcs += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_never_disabling_arc_removal() {
            for arc in find_never_disabling_arcs(
                net,
                &snapshot.dead_transitions,
                &snapshot.self_loop_transitions,
                protected_places,
            ) {
                if !report.never_disabling_arcs.contains(&arc) {
                    report.never_disabling_arcs.push(arc);
                    per_rule_progress.rule_n_never_disabling += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_agglomeration() {
            let (pre_agglomerations, post_agglomerations) =
                find_agglomerations(net, &snapshot.dead_transitions);

            for agglomeration in pre_agglomerations {
                let exists = report.pre_agglomerations.iter().any(|existing| {
                    existing.transition == agglomeration.transition
                        && existing.place == agglomeration.place
                });
                if !exists {
                    report.pre_agglomerations.push(agglomeration);
                    per_rule_progress.rule_p_pre_agglomeration += 1;
                    changed = true;
                }
            }

            for agglomeration in post_agglomerations {
                let exists = report.post_agglomerations.iter().any(|existing| {
                    existing.transition == agglomeration.transition
                        && existing.place == agglomeration.place
                });
                if !exists {
                    report.post_agglomerations.push(agglomeration);
                    per_rule_progress.rule_p_post_agglomeration += 1;
                    changed = true;
                }
            }
        }

        if mode.allows_redundant_place_removal() {
            for implied in find_redundant_places(
                net,
                &already_removed,
                &snapshot.dead_transitions,
                &self_loop_protected,
            ) {
                let place = PlaceIdx(implied.place as u32);
                if !report.redundant_places.contains(&place) {
                    report.redundant_places.push(place);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }

        iterations += 1;
    }

    if mode.allows_never_disabling_arc_removal() {
        for arc in find_fixpoint_lower_bound_arcs(
            net,
            &workqueue_state,
            &report.dead_transitions,
            &report.self_loop_transitions,
            protected_places,
        ) {
            if !report.never_disabling_arcs.iter().any(|existing| {
                existing.transition == arc.transition && existing.place == arc.place
            }) {
                push_never_disabling_arc_unique(&mut report.never_disabling_arcs, arc);
                per_rule_progress.rule_n_fixpoint_lower_bound += 1;
            }
        }
    }

    EfmnopAnalysis {
        report,
        iterations,
        dead_removed_by_cascade,
        per_rule_progress,
    }
}

/// Analyze a Petri net and report which structures can be reduced.
///
/// Uses iterative fixpoint analysis for dead transitions: removing a dead
/// transition can make other transitions dead (by removing the only producer
/// for their input places). Places that become isolated after dead transition
/// removal are also detected and reported.
#[must_use]
pub(crate) fn analyze(net: &PetriNet) -> ReductionReport {
    let constant = find_constant_places(net);
    let mut dead = find_dead_transitions(net);

    // Extend with trap-based dead transitions: transitions consuming from
    // zero-marked place sets that are closed under token production from
    // live transitions. This catches dead transitions that cascading
    // analysis misses (e.g., zero-marked cycles with no external producer).
    let trap_dead = find_trap_dead_transitions(net, &dead);
    for t in trap_dead {
        if !dead.contains(&t) {
            dead.push(t);
        }
    }

    let duplicate_transitions = find_duplicate_transitions(net, &dead);
    let self_loop_transitions = find_self_loop_transitions(net, &dead);
    let non_decreasing_places = find_non_decreasing_places(net, &dead, &[], &[]);
    let parallel_places = find_parallel_places(net, &dead, &[]);
    // Berthelot lateral fusion (R_lat): affine-coupled, enabling-preserving
    // place removal — the affine generalization of parallel-place merge.
    // Reported here with no places pre-removed and none protected, matching the
    // other detector calls in this static analyzer.
    let none_removed = vec![false; net.num_places()];
    let lateral_fusions = find_lateral_fusions(net, &none_removed, &dead, &[]);
    let mut isolated = find_isolated_places(net);

    // Places that become isolated after dead transition removal.
    let cascade_isolated = find_cascade_isolated_places(net, &dead);
    isolated.extend(cascade_isolated);

    // Agglomeration detection (connectivity + candidates + conflict resolution).
    let (pre_agg, post_agg) = find_agglomerations(net, &dead);

    let dominated = find_dominated_transitions(net, &dead);

    ReductionReport {
        constant_places: constant,
        source_places: Vec::new(),
        dead_transitions: dead,
        isolated_places: isolated,
        pre_agglomerations: pre_agg,
        post_agglomerations: post_agg,
        duplicate_transitions,
        self_loop_transitions,
        dominated_transitions: dominated,
        self_loop_arcs: Vec::new(),
        never_disabling_arcs: Vec::new(),
        token_eliminated_places: Vec::new(),
        redundant_places: Vec::new(),
        parallel_places,
        lateral_fusions,
        non_decreasing_places,
        sink_transitions: Vec::new(),
        token_cycle_merges: Vec::new(),
        rule_r_agglomerations: Vec::new(),
        rule_s_agglomerations: Vec::new(),
    }
}

/// Find producer-only places that are safe to elide because no live transition
/// ever consumes from them and they are not query-protected.
pub(crate) fn find_source_places(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
    protected_places: &[bool],
) -> Vec<PlaceIdx> {
    let connectivity = build_place_connectivity(net, dead_transitions);
    let mut source_places = Vec::new();

    for place_idx in 0..net.num_places() {
        if protected_places.get(place_idx).copied().unwrap_or(false) {
            continue;
        }
        if !connectivity.consumers[place_idx].is_empty() {
            continue;
        }
        if connectivity.producers[place_idx].is_empty() {
            continue;
        }
        source_places.push(PlaceIdx(place_idx as u32));
    }

    source_places
}

/// A place is constant if every transition has zero net effect on it.
///
/// For each transition, we compute `sum(output_arcs_to_p) - sum(input_arcs_from_p)`.
/// If this delta is zero for ALL transitions, the place's token count can never change.
fn find_constant_places(net: &PetriNet) -> Vec<PlaceIdx> {
    let num_places = net.num_places();
    // Track whether each place has ever been affected with a nonzero delta.
    let mut is_constant = vec![true; num_places];
    // Also track whether the place is connected to any transition at all.
    let mut is_connected = vec![false; num_places];
    // Reusable delta buffer — avoids O(P) allocation per transition.
    let mut delta = vec![0i64; num_places];
    // Track which places were touched so we can reset only those entries.
    let mut touched = Vec::with_capacity(16);

    for t in &net.transitions {
        touched.clear();
        for arc in &t.inputs {
            let p = arc.place.0 as usize;
            delta[p] -= arc.weight as i64;
            is_connected[p] = true;
            touched.push(p);
        }
        for arc in &t.outputs {
            let p = arc.place.0 as usize;
            delta[p] += arc.weight as i64;
            is_connected[p] = true;
            touched.push(p);
        }
        for &p in &touched {
            if delta[p] != 0 {
                is_constant[p] = false;
            }
            delta[p] = 0;
        }
    }

    // Only report connected constant places. Isolated places are reported separately.
    is_constant
        .iter()
        .zip(is_connected.iter())
        .enumerate()
        .filter(|(_, (&c, &conn))| c && conn)
        .map(|(i, _)| PlaceIdx(i as u32))
        .collect()
}

/// Find all dead transitions via iterative fixpoint analysis.
///
/// A transition is dead if it can never fire from any reachable marking.
/// This uses cascading analysis: when a dead transition is removed, its
/// output places lose a producer. If a place then has no remaining producer
/// and insufficient initial marking, transitions consuming from it also
/// become dead. The analysis iterates until no new dead transitions are found.
///
/// Example cascade: `p0(0) → t0 → p1(0) → t1 → p2`. If p0 has no producer
/// and initial marking 0, t0 is dead. Removing t0 means p1 has no producer,
/// so t1 is also dead. Single-pass analysis would only catch t0.
fn find_dead_transitions(net: &PetriNet) -> Vec<TransitionIdx> {
    let num_places = net.num_places();
    let num_transitions = net.num_transitions();
    let summaries = summarized_transitions(net);
    let mut is_dead = vec![false; num_transitions];

    loop {
        // Recompute producer status ignoring dead transitions.
        let mut has_producer = vec![false; num_places];
        for (tidx, t) in net.transitions.iter().enumerate() {
            if is_dead[tidx] {
                continue;
            }
            for arc in &t.outputs {
                has_producer[arc.place.0 as usize] = true;
            }
        }

        // Find newly dead transitions.
        let mut changed = false;
        for (tidx, summary) in summaries.iter().enumerate() {
            if is_dead[tidx] {
                continue;
            }
            let dead = summary.inputs.iter().any(|&(place, weight)| {
                let place_idx = place.0 as usize;
                !has_producer[place_idx] && net.initial_marking[place_idx] < weight
            });
            if dead {
                is_dead[tidx] = true;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    is_dead
        .iter()
        .enumerate()
        .filter(|(_, &d)| d)
        .map(|(i, _)| TransitionIdx(i as u32))
        .collect()
}

/// Places with no incoming or outgoing arcs. These contribute nothing to
/// the reachable state space (their token count is always the initial value).
fn find_isolated_places(net: &PetriNet) -> Vec<PlaceIdx> {
    find_isolated_places_ignoring(net, &[])
}

/// Places with no arcs to any alive transition.
///
/// A place is "cascade-isolated" if all transitions it connects to are dead.
/// Unlike structurally isolated places (zero arcs), these places HAVE arcs
/// but only to dead transitions. Their token count is invariant because no
/// alive transition can consume from or produce into them.
fn find_cascade_isolated_places(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
) -> Vec<PlaceIdx> {
    if dead_transitions.is_empty() {
        return Vec::new();
    }

    // Only return places that were NOT already structurally isolated
    // (those are reported separately to avoid double-counting).
    let structurally_isolated = find_isolated_places(net);
    find_isolated_places_ignoring(net, dead_transitions)
        .into_iter()
        .filter(|p| !structurally_isolated.contains(p))
        .collect()
}

/// Places with no arcs to any alive (non-ignored) transition.
fn find_isolated_places_ignoring(
    net: &PetriNet,
    ignored_transitions: &[TransitionIdx],
) -> Vec<PlaceIdx> {
    let num_places = net.num_places();
    let num_transitions = net.num_transitions();

    let mut is_ignored = vec![false; num_transitions];
    for &TransitionIdx(t) in ignored_transitions {
        is_ignored[t as usize] = true;
    }

    let mut connected = vec![false; num_places];
    for (tidx, t) in net.transitions.iter().enumerate() {
        if is_ignored[tidx] {
            continue;
        }
        for arc in &t.inputs {
            connected[arc.place.0 as usize] = true;
        }
        for arc in &t.outputs {
            connected[arc.place.0 as usize] = true;
        }
    }

    connected
        .iter()
        .enumerate()
        .filter(|(_, &c)| !c)
        .map(|(i, _)| PlaceIdx(i as u32))
        .collect()
}
