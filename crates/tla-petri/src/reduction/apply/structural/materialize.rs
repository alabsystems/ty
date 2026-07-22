// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::{BTreeMap, BTreeSet};

use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

use super::super::super::ReducedNet;
use super::planning::StructuralPlan;

/// Remap arcs through `place_map`, combining weights for duplicate places.
///
/// Arcs to removed places (where `place_map` is `None`) are dropped.
/// When multiple arcs reference the same surviving place, their weights
/// are summed. This handles the merge step of agglomeration and Rule H
/// (token-cycle absorption) correctly. NOTE: Rule B parallel duplicates
/// must NOT be redirected here — their arcs are dropped via
/// `arc_place_map` (duplicate -> `None`), because the canonical's own
/// identical arc already carries the exact constraint; summing would
/// double the weight against an un-doubled initial marking.
fn remap_and_combine_arcs(arcs: &[Arc], place_map: &[Option<PlaceIdx>]) -> Vec<Arc> {
    let mut combined: BTreeMap<u32, u64> = BTreeMap::new();
    for arc in arcs {
        if let Some(new_p) = place_map[arc.place.0 as usize] {
            *combined.entry(new_p.0).or_default() += arc.weight;
        }
    }
    combined
        .into_iter()
        .map(|(p, w)| Arc {
            place: PlaceIdx(p),
            weight: w,
        })
        .collect()
}

/// Subtract `weight_to_strip` from arcs on `place`, preserving any unmatched
/// residual weight on the same place.
fn strip_arc_weight(arcs: &mut Vec<Arc>, place: PlaceIdx, mut weight_to_strip: u64) {
    if weight_to_strip == 0 {
        return;
    }

    for arc in arcs.iter_mut().filter(|arc| arc.place == place) {
        if weight_to_strip == 0 {
            break;
        }
        let stripped = arc.weight.min(weight_to_strip);
        arc.weight -= stripped;
        weight_to_strip -= stripped;
    }

    arcs.retain(|arc| arc.weight > 0);
    debug_assert_eq!(
        weight_to_strip, 0,
        "Rule K strip weight must be fully accounted for by original transition arcs"
    );
}

/// Turn a [`StructuralPlan`] into a [`ReducedNet`] by building place/transition
/// mappings, splicing agglomeration arcs, and stripping self-loop weights.
pub(super) fn build_reduced_net(net: &PetriNet, plan: StructuralPlan) -> ReducedNet {
    let StructuralPlan {
        report,
        place_removed,
        place_agglomerated,
        redundant_set,
        lateral_fused_set,
        reconstructions,
    } = plan;

    let num_places = net.num_places();
    let num_transitions = net.num_transitions();

    // Build place mappings.
    let mut place_map: Vec<Option<PlaceIdx>> = vec![None; num_places];
    let mut place_unmap: Vec<PlaceIdx> = Vec::new();
    let mut new_pidx = 0u32;
    for (orig, &removed) in place_removed.iter().enumerate() {
        if !removed {
            place_map[orig] = Some(PlaceIdx(new_pidx));
            place_unmap.push(PlaceIdx(orig as u32));
            new_pidx += 1;
        }
    }
    for merge in &report.parallel_places {
        if let Some(canonical) = place_map[merge.canonical.0 as usize] {
            place_map[merge.duplicate.0 as usize] = Some(canonical);
        }
    }
    // Rule H: redirect every absorbed cycle place to the cycle's survivor.
    // Survivors themselves remain in place_unmap (they survive as regular
    // reduced-net places). `place_removed[survivor]` must be false; planning
    // enforces this by protecting the survivor from other place-removal rules.
    for cycle in &report.token_cycle_merges {
        if let Some(survivor_new) = place_map[cycle.survivor.0 as usize] {
            for &PlaceIdx(absorbed) in &cycle.absorbed {
                place_map[absorbed as usize] = Some(survivor_new);
            }
        }
    }

    // Rule B (textbook): a parallel duplicate's arcs are DROPPED, not
    // redirected-and-summed. `find_parallel_places` guarantees the duplicate
    // and canonical have IDENTICAL per-transition arc signatures AND equal
    // initial markings, so m(duplicate) == m(canonical) at every reachable
    // marking: the canonical's own (identical) arc already carries the exact
    // enabling constraint and firing effect. Redirecting the duplicate's arcs
    // onto the canonical would sum the weights (w -> 2w) against an un-summed
    // initial marking, distorting enabledness at every marking with
    // m0 > 0 — a traced wrong-deadlock/wrong-reachability source.
    //
    // `place_map` keeps the duplicate -> canonical alias for query-atom
    // rewriting and `expand_marking` (m(dup) = m(canon), which is exact);
    // only arc remapping uses `arc_place_map`, where the duplicate maps to
    // `None` (drop). Rule H redirects (intentional summing semantics) are
    // preserved: a duplicate's entry is only cleared when it still points at
    // its Rule B canonical.
    let mut arc_place_map = place_map.clone();
    for merge in &report.parallel_places {
        if let Some(canonical_new) = place_map[merge.canonical.0 as usize] {
            if arc_place_map[merge.duplicate.0 as usize] == Some(canonical_new) {
                arc_place_map[merge.duplicate.0 as usize] = None;
            }
        }
    }

    // Build set of transitions to remove.
    let mut trans_removed = vec![false; num_transitions];
    for &TransitionIdx(t) in &report.dead_transitions {
        trans_removed[t as usize] = true;
    }
    for agg in &report.pre_agglomerations {
        trans_removed[agg.transition.0 as usize] = true;
    }
    for agg in &report.post_agglomerations {
        trans_removed[agg.transition.0 as usize] = true;
    }
    for class in &report.duplicate_transitions {
        for duplicate in &class.duplicates {
            trans_removed[duplicate.0 as usize] = true;
        }
    }
    for &TransitionIdx(t) in &report.self_loop_transitions {
        trans_removed[t as usize] = true;
    }
    for &TransitionIdx(t) in &report.dominated_transitions {
        trans_removed[t as usize] = true;
    }
    for &TransitionIdx(t) in &report.sink_transitions {
        trans_removed[t as usize] = true;
    }
    // Rule H: cycle transitions are dropped (each reduces to a self-loop
    // with zero net effect on the merged survivor place).
    for cycle in &report.token_cycle_merges {
        for &TransitionIdx(t) in &cycle.transitions {
            trans_removed[t as usize] = true;
        }
    }
    // Rule R: fuseable producers are always removed. Consumers are removed
    // only when `remove_place` (the intermediate place disappears too).
    for agg in &report.rule_r_agglomerations {
        for &(TransitionIdx(t), _) in &agg.fuseable_producers {
            trans_removed[t as usize] = true;
        }
        if agg.remove_place {
            for &TransitionIdx(t) in &agg.consumers {
                trans_removed[t as usize] = true;
            }
        }
    }
    // Rule S: all producers, all consumers removed (place also removed via
    // `place_removed` mask from planning).
    for agg in &report.rule_s_agglomerations {
        for &TransitionIdx(t) in &agg.producers {
            trans_removed[t as usize] = true;
        }
        for &TransitionIdx(t) in &agg.consumers {
            trans_removed[t as usize] = true;
        }
    }
    // Transitions blocked by a constant/isolated place with insufficient tokens.
    //
    // The `m0 < weight` dead-consumer argument is only valid for places whose
    // marking is genuinely FROZEN at its initial value (constant / isolated /
    // cascade drops). Places that are removed but REMAPPED to a surviving
    // place (`place_map[p].is_some()`: Rule B parallel duplicates, Rule H
    // absorbed cycle places) are replenished by their producers, so they must
    // never be treated as frozen — their constraint lives on in the surviving
    // place's own arcs. The same holds for LP-redundant places (`redundant_set`)
    // and lateral-fused duplicates (`lateral_fused_set`): both are removed but
    // reconstructed from a surviving place and are LP-PROVEN to never gate any
    // live transition, so they must not poison the dead-consumer guard. This
    // mirrors the `constant_values` filter below.
    for (tidx, t) in net.transitions.iter().enumerate() {
        if trans_removed[tidx] {
            continue;
        }
        let blocked_by_constant = t.inputs.iter().any(|arc| {
            let p = arc.place.0 as usize;
            place_removed[p]
                && place_map[p].is_none()
                && !place_agglomerated[p]
                && !redundant_set[p]
                && !lateral_fused_set[p]
                && net.initial_marking[p] < arc.weight
        });
        if blocked_by_constant {
            trans_removed[tidx] = true;
        }
    }

    // Build extra arcs from agglomerations.
    let mut extra_inputs: Vec<Vec<Arc>> = vec![Vec::new(); num_transitions];
    let mut extra_outputs: Vec<Vec<Arc>> = vec![Vec::new(); num_transitions];

    for agg in &report.pre_agglomerations {
        let source = &net.transitions[agg.transition.0 as usize];
        for &succ in &agg.successors {
            extra_inputs[succ.0 as usize].extend(source.inputs.iter().cloned());
        }
    }
    for agg in &report.post_agglomerations {
        let sink = &net.transitions[agg.transition.0 as usize];
        for &pred in &agg.predecessors {
            extra_outputs[pred.0 as usize].extend(sink.outputs.iter().cloned());
        }
    }

    // Build transition mappings.
    let mut transition_map: Vec<Option<TransitionIdx>> = vec![None; num_transitions];
    let mut transition_unmap: Vec<TransitionIdx> = Vec::new();
    let mut new_tidx = 0u32;
    for (orig, &removed) in trans_removed.iter().enumerate() {
        if !removed {
            transition_map[orig] = Some(TransitionIdx(new_tidx));
            transition_unmap.push(TransitionIdx(orig as u32));
            new_tidx += 1;
        }
    }
    for class in &report.duplicate_transitions {
        let Some(canonical) = transition_map[class.canonical.0 as usize] else {
            continue;
        };
        for duplicate in &class.duplicates {
            transition_map[duplicate.0 as usize] = Some(canonical);
        }
    }

    // Build reduced net.
    let new_places: Vec<PlaceInfo> = place_unmap
        .iter()
        .map(|&PlaceIdx(orig)| net.places[orig as usize].clone())
        .collect();

    let mut new_initial: Vec<u64> = place_unmap
        .iter()
        .map(|&PlaceIdx(orig)| net.initial_marking[orig as usize])
        .collect();

    // Rule H: add absorbed cycle place tokens into the survivor's initial
    // marking. The total cycle token count is invariant across all firings,
    // so accumulating it on the survivor preserves the aggregate that
    // reachability queries can observe.
    for cycle in &report.token_cycle_merges {
        let Some(PlaceIdx(survivor_new)) = place_map[cycle.survivor.0 as usize] else {
            continue;
        };
        let mut added: u64 = 0;
        for &PlaceIdx(absorbed) in &cycle.absorbed {
            added = added.saturating_add(net.initial_marking[absorbed as usize]);
        }
        if added > 0 {
            new_initial[survivor_new as usize] =
                new_initial[survivor_new as usize].saturating_add(added);
        }
    }

    // Build exact self-loop strip weights from the original transition arcs (Rule K).
    let mut self_loop_strip_weights: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for self_loop_arc in &report.self_loop_arcs {
        *self_loop_strip_weights
            .entry((self_loop_arc.transition.0, self_loop_arc.place.0))
            .or_default() += self_loop_arc.weight;
    }
    let self_loop_places: BTreeSet<(u32, u32)> = report
        .self_loop_arcs
        .iter()
        .map(|self_loop_arc| (self_loop_arc.transition.0, self_loop_arc.place.0))
        .collect();
    let mut new_transitions: Vec<TransitionInfo> = transition_unmap
        .iter()
        .map(|&TransitionIdx(orig)| {
            let t = &net.transitions[orig as usize];
            let orig_idx = orig as usize;

            let mut all_inputs: Vec<Arc> = t.inputs.clone();
            if !self_loop_places.is_empty() {
                for &(transition, place) in self_loop_places
                    .iter()
                    .filter(|(transition, _)| *transition == orig)
                {
                    let weight = self_loop_strip_weights[&(transition, place)];
                    strip_arc_weight(&mut all_inputs, PlaceIdx(place), weight);
                }
            }
            all_inputs.extend_from_slice(&extra_inputs[orig_idx]);
            let new_inputs = remap_and_combine_arcs(&all_inputs, &arc_place_map);

            let mut all_outputs: Vec<Arc> = t.outputs.clone();
            if !self_loop_places.is_empty() {
                for &(transition, place) in self_loop_places
                    .iter()
                    .filter(|(transition, _)| *transition == orig)
                {
                    let weight = self_loop_strip_weights[&(transition, place)];
                    strip_arc_weight(&mut all_outputs, PlaceIdx(place), weight);
                }
            }
            all_outputs.extend_from_slice(&extra_outputs[orig_idx]);
            let new_outputs = remap_and_combine_arcs(&all_outputs, &arc_place_map);

            TransitionInfo {
                id: t.id.clone(),
                name: t.name.clone(),
                inputs: new_inputs,
                outputs: new_outputs,
            }
        })
        .collect();

    // Rule R synthesis: for every (producer, consumer) pair per agglomeration,
    // emit one new transition whose pre-set is the producer's pre-set and
    // whose post-set is (producer.outputs − arc_on_place) ∪ consumer.outputs.
    // Reasoning: producer writes `max_consumer_weight` tokens into `place`,
    // consumer reads exactly `max_consumer_weight` tokens (Phase-1 invariant)
    // and then writes its own outputs. Fusing skips the intermediate `place`.
    for agg in &report.rule_r_agglomerations {
        for &(producer_tidx, _producer_w) in &agg.fuseable_producers {
            let producer = &net.transitions[producer_tidx.0 as usize];

            // Producer outputs minus the arc on `place` (matching
            // max_consumer_weight, which is the whole producer arc in Phase-1).
            let mut producer_outputs_minus_p: Vec<Arc> = producer.outputs.to_vec();
            strip_arc_weight(
                &mut producer_outputs_minus_p,
                agg.place,
                agg.max_consumer_weight,
            );

            for &consumer_tidx in &agg.consumers {
                let consumer = &net.transitions[consumer_tidx.0 as usize];

                // Synthesized inputs: producer's inputs (remapped, combined).
                let new_inputs = remap_and_combine_arcs(&producer.inputs, &arc_place_map);

                // Synthesized outputs: producer.outputs (minus arc on place)
                // unioned with consumer.outputs.
                let mut fused_outputs: Vec<Arc> = producer_outputs_minus_p.clone();
                fused_outputs.extend_from_slice(&consumer.outputs);
                let new_outputs = remap_and_combine_arcs(&fused_outputs, &arc_place_map);

                let fused_name = match (&producer.name, &consumer.name) {
                    (Some(p), Some(c)) => Some(format!("{p}__{c}")),
                    (Some(p), None) => Some(p.clone()),
                    (None, Some(c)) => Some(c.clone()),
                    (None, None) => None,
                };
                new_transitions.push(TransitionInfo {
                    id: format!("__rule_r_{}_{}", producer_tidx.0, consumer_tidx.0),
                    name: fused_name,
                    inputs: new_inputs,
                    outputs: new_outputs,
                });
                // Provenance: attribute the synthesized transition to the
                // producer. This is a Phase-1 approximation — it means
                // compose() can translate inner-reduction references to the
                // synthesized transition through `transition_unmap` without
                // OOB, and any downstream duplicate-transition class gets
                // attributed to the producer's original index. Phase-2
                // introduces `TransitionProvenance::RuleR { producer, consumer }`
                // to distinguish genuine original transitions from fused ones.
                transition_unmap.push(producer_tidx);
            }
        }
    }

    // Rule S synthesis: for every (producer × consumer) pair per agglomeration,
    // emit one new transition. Because Phase-1 requires `producer.post == {place}`
    // with weight `w` and `consumer.pre == {place}` with weight `w` exactly, the
    // fused inputs are simply `producer.inputs` and the fused outputs are
    // `consumer.outputs`. No `strip_arc_weight` is needed: the producer's only
    // post-arc is to `place` (which is being removed), and the consumer's only
    // pre-arc is from `place`, so no residual arcs survive the fusion.
    for agg in &report.rule_s_agglomerations {
        for &producer_tidx in &agg.producers {
            let producer = &net.transitions[producer_tidx.0 as usize];
            for &consumer_tidx in &agg.consumers {
                let consumer = &net.transitions[consumer_tidx.0 as usize];
                let new_inputs = remap_and_combine_arcs(&producer.inputs, &arc_place_map);
                let new_outputs = remap_and_combine_arcs(&consumer.outputs, &arc_place_map);
                let fused_name = match (&producer.name, &consumer.name) {
                    (Some(p), Some(c)) => Some(format!("{p}__{c}")),
                    (Some(p), None) => Some(p.clone()),
                    (None, Some(c)) => Some(c.clone()),
                    (None, None) => None,
                };
                new_transitions.push(TransitionInfo {
                    id: format!("__rule_s_{}_{}", producer_tidx.0, consumer_tidx.0),
                    name: fused_name,
                    inputs: new_inputs,
                    outputs: new_outputs,
                });
                // Provenance: attribute to producer (Phase-1 approximation,
                // parallel to Rule R). Phase-2 will use
                // `TransitionProvenance::RuleS { producer, consumer }`.
                transition_unmap.push(producer_tidx);
            }
        }
    }

    // Record expansion values for genuinely-frozen removed places. EXCLUDES
    // P-invariant-reconstructed (`redundant_set`) and lateral-fused
    // (`lateral_fused_set`) places: both are reconstructed from a surviving
    // place during `expand_marking` (the lateral-fusion pass overwrites
    // `full[duplicate] = ratio*full[canonical] + offset`), so freezing them at
    // `m0` here would be wrong.
    let constant_values: Vec<(PlaceIdx, u64)> = place_removed
        .iter()
        .enumerate()
        .filter(|(p, &removed)| {
            removed && place_map[*p].is_none() && !redundant_set[*p] && !lateral_fused_set[*p]
        })
        .map(|(p, _)| (PlaceIdx(p as u32), net.initial_marking[p]))
        .collect();

    let reduced_net = PetriNet {
        name: net.name.clone(),
        places: new_places,
        transitions: new_transitions,
        initial_marking: new_initial,
    };

    ReducedNet {
        net: reduced_net,
        place_map,
        place_unmap,
        place_scales: vec![1; num_places],
        transition_map,
        transition_unmap,
        constant_values,
        reconstructions,
        report,
    }
}
