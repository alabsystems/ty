// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::BTreeMap;

use crate::invariant::compute_p_invariants;
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx, TransitionInfo};

use super::analysis_invariant::compute_invariant_lower_bounds;
use super::model::{DuplicateTransitionClass, SelfLoopArc};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TransitionShape {
    inputs: Vec<(u32, u64)>,
    outputs: Vec<(u32, u64)>,
}

fn transition_shape(net: &PetriNet, transition: TransitionIdx) -> TransitionShape {
    let mut inputs = net.transitions[transition.0 as usize]
        .inputs
        .iter()
        .map(|arc| (arc.place.0, arc.weight))
        .collect::<Vec<_>>();
    inputs.sort_unstable();

    let mut outputs = net.transitions[transition.0 as usize]
        .outputs
        .iter()
        .map(|arc| (arc.place.0, arc.weight))
        .collect::<Vec<_>>();
    outputs.sort_unstable();

    TransitionShape { inputs, outputs }
}

pub(super) fn find_duplicate_transitions(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
) -> Vec<DuplicateTransitionClass> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(transition) in dead_transitions {
        is_dead[transition as usize] = true;
    }

    let mut groups: BTreeMap<TransitionShape, Vec<TransitionIdx>> = BTreeMap::new();
    for (index, _) in net.transitions.iter().enumerate() {
        if is_dead[index] {
            continue;
        }
        let transition = TransitionIdx(index as u32);
        groups
            .entry(transition_shape(net, transition))
            .or_default()
            .push(transition);
    }

    let mut duplicate_transitions = groups
        .into_values()
        .filter(|group| group.len() > 1)
        .map(|group| DuplicateTransitionClass {
            canonical: group[0],
            duplicates: group[1..].to_vec(),
        })
        .collect::<Vec<_>>();
    duplicate_transitions.sort_by_key(|class| class.canonical.0);
    duplicate_transitions
}

/// A self-loop transition has zero net effect on every place.
/// Firing it consumes and produces exactly the same tokens — it is a no-op.
/// This is the dual of `find_constant_places` (which finds places with
/// zero net effect from all transitions).
pub(super) fn find_self_loop_transitions(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
) -> Vec<TransitionIdx> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }

    let mut result = Vec::new();
    for (tidx, t) in net.transitions.iter().enumerate() {
        if is_dead[tidx] {
            continue;
        }

        let mut delta: BTreeMap<u32, i64> = BTreeMap::new();
        for arc in &t.inputs {
            *delta.entry(arc.place.0).or_default() -= arc.weight as i64;
        }
        for arc in &t.outputs {
            *delta.entry(arc.place.0).or_default() += arc.weight as i64;
        }

        if delta.values().all(|&d| d == 0) {
            result.push(TransitionIdx(tidx as u32));
        }
    }
    result
}

/// Transition net effect: sorted (place, delta) pairs where delta != 0.
///
/// Two transitions with identical `TransitionDelta` values have the same
/// effect on the marking when fired, even if they consume different amounts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TransitionDelta(Vec<(u32, i64)>);

fn transition_delta(t: &TransitionInfo) -> TransitionDelta {
    let mut delta: BTreeMap<u32, i64> = BTreeMap::new();
    for arc in &t.inputs {
        *delta.entry(arc.place.0).or_default() -= arc.weight as i64;
    }
    for arc in &t.outputs {
        *delta.entry(arc.place.0).or_default() += arc.weight as i64;
    }
    TransitionDelta(delta.into_iter().filter(|&(_, d)| d != 0).collect())
}

/// Input multiset: sorted (place, weight) pairs for a transition's input arcs.
fn input_multiset(t: &TransitionInfo) -> Vec<(u32, u64)> {
    let mut inputs: BTreeMap<u32, u64> = BTreeMap::new();
    for arc in &t.inputs {
        *inputs.entry(arc.place.0).or_default() += arc.weight;
    }
    inputs.into_iter().collect()
}

/// Returns true if `heavy` is strictly dominated by `light`: for every input
/// place, `heavy` needs at least as many tokens as `light`, and for at least
/// one place `heavy` needs strictly more.
fn is_strictly_dominated(heavy: &[(u32, u64)], light: &[(u32, u64)]) -> bool {
    let mut hi = 0;
    let mut li = 0;
    let mut has_strict = false;

    // Both are sorted by place index. Walk them in lockstep.
    while hi < heavy.len() && li < light.len() {
        let (hp, hw) = heavy[hi];
        let (lp, lw) = light[li];

        if hp == lp {
            if hw < lw {
                return false; // heavy is lighter here — not dominated
            }
            if hw > lw {
                has_strict = true;
            }
            hi += 1;
            li += 1;
        } else if hp < lp {
            // heavy needs tokens from a place light doesn't — strictly heavier
            has_strict = true;
            hi += 1;
        } else {
            // light needs tokens from a place heavy doesn't — heavy is lighter
            return false;
        }
    }

    // Remaining entries in heavy are extra requirements (strictly heavier).
    if hi < heavy.len() {
        has_strict = true;
    }
    // Remaining entries in light are requirements heavy doesn't have.
    if li < light.len() {
        return false;
    }

    has_strict
}

/// Find transitions dominated by another live transition with identical net
/// effect but strictly weaker enabling condition (Tapaal Rule L).
///
/// Transition `t_heavy` is dominated by `t_light` if:
/// 1. They have the same net effect on every place.
/// 2. For every input place p, `in(t_heavy, p) >= in(t_light, p)`.
/// 3. There exists at least one place where `in(t_heavy, p) > in(t_light, p)`.
///
/// Whenever `t_heavy` is enabled, `t_light` is also enabled and produces the
/// same successor marking, so `t_heavy` is redundant for reachability and all
/// branching-time properties (the transition systems are bisimilar).
pub(super) fn find_dominated_transitions(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
) -> Vec<TransitionIdx> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }

    // Group live transitions by their net effect.
    let mut groups: BTreeMap<TransitionDelta, Vec<usize>> = BTreeMap::new();
    for (tidx, t) in net.transitions.iter().enumerate() {
        if is_dead[tidx] {
            continue;
        }
        groups.entry(transition_delta(t)).or_default().push(tidx);
    }

    let mut dominated = Vec::new();
    for group in groups.values() {
        if group.len() < 2 {
            continue;
        }
        // Pre-compute input multisets for this group.
        let inputs: Vec<Vec<(u32, u64)>> = group
            .iter()
            .map(|&tidx| input_multiset(&net.transitions[tidx]))
            .collect();

        // For each transition in the group, check if it's dominated by any other.
        let mut is_dominated = vec![false; group.len()];
        for i in 0..group.len() {
            if is_dominated[i] {
                continue;
            }
            for j in 0..group.len() {
                if i == j || is_dominated[j] {
                    continue;
                }
                // Check if j is dominated by i (i.e., i is lighter).
                if is_strictly_dominated(&inputs[j], &inputs[i]) {
                    is_dominated[j] = true;
                }
            }
        }

        for (idx, &dom) in is_dominated.iter().enumerate() {
            if dom {
                dominated.push(TransitionIdx(group[idx] as u32));
            }
        }
    }

    dominated.sort_by_key(|t| t.0);
    dominated
}

/// Find individual self-loop arc pairs on non-self-loop transitions (Tapaal Rule K).
///
/// A self-loop arc pair exists when a transition both consumes from and produces
/// to the same place with equal weight. The net effect on that place is zero,
/// but the pair still acts as a TEST ARC: it gates enabling (`m[p] >= w`).
/// Removing it without proof that the place always carries `>= w` tokens
/// over-approximates behavior — the transition would fire from markings where
/// the original is disabled, inflating reachability (wrong UpperBounds /
/// reachability verdicts on Szymanski-PT-a02, CANInsertWithFailure-PT-005,
/// SimpleLoadBal-PT-05).
///
/// The pair is therefore only removed with a never-disabling proof (same
/// obligation as Rule N), discharged by either:
/// - a P-invariant token lower bound `>= w` on the place, or
/// - the place being non-decreasing (no live transition has a net-negative
///   effect on it) with initial marking `>= w`, so `m[p] >= m0[p] >= w` in
///   every reachable marking.
///
/// With the proof, the arc pair never disables the transition and has zero
/// marking effect, so removal preserves the reachability graph exactly.
///
/// Excludes:
/// - Dead transitions (already removed by Rule E/P)
/// - Full self-loop transitions (Rule G handles those — every arc is a self-loop)
/// - Arcs to query-protected places (preserved for property evaluation)
pub(super) fn find_self_loop_arcs(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
    self_loop_transitions: &[TransitionIdx],
    protected_places: &[bool],
) -> Vec<SelfLoopArc> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }
    let mut is_full_self_loop = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in self_loop_transitions {
        is_full_self_loop[t as usize] = true;
    }

    // A place can decrease if any live transition has a net-negative effect on
    // it. Non-decreasing places satisfy m[p] >= m0[p] in every reachable
    // marking, which discharges the never-disabling obligation when
    // m0[p] >= weight.
    let mut can_decrease = vec![false; net.num_places()];
    for (tidx, t) in net.transitions.iter().enumerate() {
        if is_dead[tidx] {
            continue;
        }
        let mut delta: BTreeMap<u32, i128> = BTreeMap::new();
        for arc in &t.inputs {
            *delta.entry(arc.place.0).or_default() -= i128::from(arc.weight);
        }
        for arc in &t.outputs {
            *delta.entry(arc.place.0).or_default() += i128::from(arc.weight);
        }
        for (place, net_effect) in delta {
            if net_effect < 0 {
                can_decrease[place as usize] = true;
            }
        }
    }

    let mut candidates = Vec::new();

    for (tidx, t) in net.transitions.iter().enumerate() {
        if is_dead[tidx] || is_full_self_loop[tidx] {
            continue;
        }

        // Aggregate weights per place so semantically equivalent arc encodings
        // (single weight-2 arc vs two weight-1 arcs) yield the same Rule K result.
        let mut inputs: BTreeMap<u32, u64> = BTreeMap::new();
        for arc in &t.inputs {
            *inputs.entry(arc.place.0).or_default() += arc.weight;
        }

        let mut outputs: BTreeMap<u32, u64> = BTreeMap::new();
        for arc in &t.outputs {
            *outputs.entry(arc.place.0).or_default() += arc.weight;
        }

        for (place, output_weight) in outputs {
            if protected_places
                .get(place as usize)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if let Some(&input_weight) = inputs.get(&place) {
                if input_weight == output_weight {
                    candidates.push(SelfLoopArc {
                        transition: TransitionIdx(tidx as u32),
                        place: PlaceIdx(place),
                        weight: output_weight,
                    });
                }
            }
        }
    }

    let non_decreasing_proof = |arc: &SelfLoopArc| {
        !can_decrease[arc.place.0 as usize]
            && net.initial_marking[arc.place.0 as usize] >= arc.weight
    };

    // P-invariant lower bounds are only needed for candidates the cheap
    // non-decreasing proof cannot discharge.
    if candidates.iter().all(non_decreasing_proof) {
        return candidates;
    }
    let invariants = compute_p_invariants(net);
    let lower_bounds = compute_invariant_lower_bounds(net, &invariants);

    candidates.retain(|arc| {
        non_decreasing_proof(arc)
            || lower_bounds[arc.place.0 as usize]
                .as_ref()
                .is_some_and(|proof| proof.lower_bound() >= arc.weight)
    });
    candidates
}
