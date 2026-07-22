// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::BTreeMap;

use crate::invariant::{compute_p_invariants, structural_place_bound, PInvariant};
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};

use super::model::{
    LateralFusionMerge, NeverDisablingArc, NeverDisablingProof, ParallelPlaceMerge,
};
use super::redundant::{is_lp_redundant, MAX_LP_VARIABLES};

fn div_ceil_u64(numerator: u64, denominator: u64) -> u64 {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    quotient + u64::from(remainder != 0)
}

pub(super) fn compute_invariant_lower_bounds(
    net: &PetriNet,
    invariants: &[PInvariant],
) -> Vec<Option<NeverDisablingProof>> {
    let upper_bounds = (0..net.num_places())
        .map(|place_idx| structural_place_bound(invariants, place_idx))
        .collect::<Vec<_>>();
    let mut proofs: Vec<Option<NeverDisablingProof>> = vec![None; net.num_places()];

    for (invariant_idx, invariant) in invariants.iter().enumerate() {
        for (place_idx, place_weight) in invariant.support() {
            let mut other_support_upper_sum = 0u64;
            let mut missing_bound = false;
            for (other_idx, other_weight) in invariant.support() {
                if other_idx == place_idx {
                    continue;
                }
                let Some(upper_bound) = upper_bounds[other_idx] else {
                    missing_bound = true;
                    break;
                };
                let Some(term) = other_weight.checked_mul(upper_bound) else {
                    missing_bound = true;
                    break;
                };
                let Some(sum) = other_support_upper_sum.checked_add(term) else {
                    missing_bound = true;
                    break;
                };
                other_support_upper_sum = sum;
            }
            if missing_bound {
                continue;
            }

            let residual = invariant
                .token_count
                .saturating_sub(other_support_upper_sum);
            let lower_bound = div_ceil_u64(residual, place_weight);
            if lower_bound == 0 {
                continue;
            }

            let candidate = NeverDisablingProof::PInvariant {
                invariant_idx,
                lower_bound,
            };
            let should_replace = match proofs[place_idx].as_ref() {
                Some(existing) => candidate.lower_bound() > existing.lower_bound(),
                None => true,
            };
            if should_replace {
                proofs[place_idx] = Some(candidate);
            }
        }
    }

    proofs
}

/// Find input arcs whose place has a proven structural token lower bound, so
/// the arc can never disable the transition (Tapaal Rule N).
pub(crate) fn find_never_disabling_arcs(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
    self_loop_transitions: &[TransitionIdx],
    protected_places: &[bool],
) -> Vec<NeverDisablingArc> {
    let invariants = compute_p_invariants(net);
    if invariants.is_empty() {
        return Vec::new();
    }

    let lower_bounds = compute_invariant_lower_bounds(net, &invariants);

    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }
    let mut is_self_loop = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in self_loop_transitions {
        is_self_loop[t as usize] = true;
    }

    let mut results = Vec::new();
    for (tidx, transition) in net.transitions.iter().enumerate() {
        if is_dead[tidx] || is_self_loop[tidx] {
            continue;
        }

        let mut required_inputs: BTreeMap<u32, u64> = BTreeMap::new();
        for arc in &transition.inputs {
            *required_inputs.entry(arc.place.0).or_default() += arc.weight;
        }

        for (place, weight) in required_inputs {
            let place_idx = place as usize;
            if protected_places.get(place_idx).copied().unwrap_or(false) {
                continue;
            }
            let Some(proof) = lower_bounds[place_idx].clone() else {
                continue;
            };
            if proof.lower_bound() >= weight {
                results.push(NeverDisablingArc {
                    transition: TransitionIdx(tidx as u32),
                    place: PlaceIdx(place),
                    weight,
                    proof,
                });
            }
        }
    }

    results.sort_by_key(|arc| (arc.transition.0, arc.place.0));
    results
}

/// Find query-unobserved places whose every live consumer already has a Rule N
/// proof, so the place can be elided in query-relevant reductions.
pub(crate) fn find_token_eliminated_places(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
    self_loop_transitions: &[TransitionIdx],
    protected_places: &[bool],
    parallel_places: &[ParallelPlaceMerge],
    source_places: &[PlaceIdx],
    non_decreasing_places: &[PlaceIdx],
    never_disabling_arcs: &[NeverDisablingArc],
) -> Vec<PlaceIdx> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }
    let mut is_self_loop = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in self_loop_transitions {
        is_self_loop[t as usize] = true;
    }
    let mut is_source = vec![false; net.num_places()];
    for &PlaceIdx(p) in source_places {
        is_source[p as usize] = true;
    }
    let mut is_non_decreasing = vec![false; net.num_places()];
    for &PlaceIdx(p) in non_decreasing_places {
        is_non_decreasing[p as usize] = true;
    }
    let mut is_parallel_participant = vec![false; net.num_places()];
    for merge in parallel_places {
        is_parallel_participant[merge.canonical.0 as usize] = true;
        is_parallel_participant[merge.duplicate.0 as usize] = true;
    }

    let proof_map: BTreeMap<(u32, u32), u64> = never_disabling_arcs
        .iter()
        .map(|arc| ((arc.transition.0, arc.place.0), arc.proof.lower_bound()))
        .collect();

    let mut result = Vec::new();
    for place_idx in 0..net.num_places() {
        if protected_places.get(place_idx).copied().unwrap_or(false) {
            continue;
        }
        if is_source[place_idx] || is_non_decreasing[place_idx] {
            continue;
        }
        // Rule B carries an exact aliasing contract for parallel places. Keep
        // query-only token elimination off both sides of a merge so the
        // canonical/duplicate mapping cannot be degraded into a placeholder or
        // dropped entirely by asymmetric lower-bound proofs.
        if is_parallel_participant[place_idx] {
            continue;
        }

        let mut has_live_consumer = false;
        let mut all_consumers_proved = true;
        for (tidx, transition) in net.transitions.iter().enumerate() {
            if is_dead[tidx] || is_self_loop[tidx] {
                continue;
            }
            let consumed: u64 = transition
                .inputs
                .iter()
                .filter(|arc| arc.place.0 as usize == place_idx)
                .map(|arc| arc.weight)
                .sum();
            if consumed == 0 {
                continue;
            }
            has_live_consumer = true;
            let proof_lower_bound = proof_map
                .get(&(tidx as u32, place_idx as u32))
                .copied()
                .unwrap_or(0);
            if proof_lower_bound < consumed {
                all_consumers_proved = false;
                break;
            }
        }

        if has_live_consumer && all_consumers_proved {
            result.push(PlaceIdx(place_idx as u32));
        }
    }

    result
}

/// Find non-decreasing places that never constrain any transition (Tapaal Rule F).
///
/// A place `p` qualifies when:
/// 1. Every alive transition has net effect >= 0 on `p` (non-decreasing).
/// 2. The initial marking covers the maximum consumption from any single transition.
/// 3. `p` is not query-protected.
/// 4. `p` has at least one consumer (otherwise it is a source place, not Rule F).
/// 5. `p` is not already identified as a source place.
pub(crate) fn find_non_decreasing_places(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
    protected_places: &[bool],
    source_places: &[PlaceIdx],
) -> Vec<PlaceIdx> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }
    let mut is_source = vec![false; net.num_places()];
    for &PlaceIdx(p) in source_places {
        is_source[p as usize] = true;
    }

    let mut result = Vec::new();

    for place_idx in 0..net.num_places() {
        if protected_places.get(place_idx).copied().unwrap_or(false) {
            continue;
        }
        if is_source[place_idx] {
            continue;
        }

        let mut has_consumer = false;
        let mut non_decreasing = true;
        let mut max_consume: u64 = 0;

        for (tidx, t) in net.transitions.iter().enumerate() {
            if is_dead[tidx] {
                continue;
            }
            let consumes: u64 = t
                .inputs
                .iter()
                .filter(|arc| arc.place.0 as usize == place_idx)
                .map(|arc| arc.weight)
                .sum();
            let produces: u64 = t
                .outputs
                .iter()
                .filter(|arc| arc.place.0 as usize == place_idx)
                .map(|arc| arc.weight)
                .sum();

            if consumes > 0 {
                has_consumer = true;
                max_consume = max_consume.max(consumes);
            }
            if consumes > produces {
                non_decreasing = false;
                break;
            }
        }

        if non_decreasing && has_consumer && net.initial_marking[place_idx] >= max_consume {
            result.push(PlaceIdx(place_idx as u32));
        }
    }

    result
}

/// Find parallel places with identical connectivity and initial marking (Tapaal Rule B).
///
/// Two places are parallel (k=1 strict case) when they have identical
/// input/output arc patterns to all alive transitions and identical initial markings.
/// The duplicate can be removed since its marking always equals the canonical's.
pub(crate) fn find_parallel_places(
    net: &PetriNet,
    dead_transitions: &[TransitionIdx],
    protected_places: &[bool],
) -> Vec<ParallelPlaceMerge> {
    let mut is_dead = vec![false; net.num_transitions()];
    for &TransitionIdx(t) in dead_transitions {
        is_dead[t as usize] = true;
    }

    // Compute a signature for each place: sorted (transition_idx, direction_tag, weight).
    // direction_tag: 0 = input (place consumed by transition), 1 = output (produced by transition).
    let mut signatures: BTreeMap<Vec<(u32, u8, u64)>, Vec<usize>> = BTreeMap::new();

    for place_idx in 0..net.num_places() {
        if protected_places.get(place_idx).copied().unwrap_or(false) {
            continue;
        }

        let mut sig: Vec<(u32, u8, u64)> = Vec::new();
        let mut connected = false;

        for (tidx, t) in net.transitions.iter().enumerate() {
            if is_dead[tidx] {
                continue;
            }
            let consumes: u64 = t
                .inputs
                .iter()
                .filter(|arc| arc.place.0 as usize == place_idx)
                .map(|arc| arc.weight)
                .sum();
            let produces: u64 = t
                .outputs
                .iter()
                .filter(|arc| arc.place.0 as usize == place_idx)
                .map(|arc| arc.weight)
                .sum();

            if consumes > 0 {
                sig.push((tidx as u32, 0, consumes));
                connected = true;
            }
            if produces > 0 {
                sig.push((tidx as u32, 1, produces));
                connected = true;
            }
        }

        if !connected {
            continue; // isolated places handled elsewhere
        }

        signatures.entry(sig).or_default().push(place_idx);
    }

    let mut result = Vec::new();

    for places in signatures.values() {
        if places.len() < 2 {
            continue;
        }

        // Group by initial marking for strict k=1 match.
        let mut by_marking: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for &p in places {
            by_marking
                .entry(net.initial_marking[p])
                .or_default()
                .push(p);
        }

        for group in by_marking.values() {
            if group.len() < 2 {
                continue;
            }
            let canonical = PlaceIdx(group[0] as u32);
            for &duplicate_idx in &group[1..] {
                result.push(ParallelPlaceMerge {
                    canonical,
                    duplicate: PlaceIdx(duplicate_idx as u32),
                });
            }
        }
    }

    result.sort_by_key(|m| m.duplicate.0);
    result
}

/// Find Berthelot lateral place fusions (`R_lat`) — the affine generalization
/// of parallel-place merge (Rule B).
///
/// `R_lat` removes a place `d` (the *duplicate*) whose marking is an exact
/// **non-negative affine** function of a surviving place `c` (the *canonical*)
/// at every reachable marking:
///
/// ```text
///   m(d) = ratio * m(c) + offset          (ratio, offset >= 0 integers)
/// ```
///
/// matching the reconstruction in `ReducedNet::expand_marking_into`
/// (`reduced_net.rs:484`). The duplicate's arcs are dropped; its value is
/// recovered from the canonical's during marking expansion.
///
/// # Where the coupling comes from
///
/// `compute_p_invariants` returns **semi-positive** invariants (`y >= 0`, all
/// support weights strictly positive). A *single* support-2 such invariant
/// `y_c*m_c + y_d*m_d = K0` gives `m_d = (K0 - y_c*m_c)/y_d`, whose `m_c`
/// coefficient `-y_c/y_d` is **negative** (a capacity-complement, e.g.
/// `m_c + m_d = C`). A negative ratio is not representable by the non-negative
/// `LateralFusionMerge`, so a lone such invariant is **fail-closed**.
///
/// Positive-ratio couplings are extracted by **eliminating a shared place
/// between two support-2 invariants** that both cover a common place `a` (the
/// pivot):
///
/// ```text
///   I1:  y_a1 * m_a + y_d * m_d = C1
///   I2:  y_a2 * m_a + y_c * m_c = C2
/// ```
///
/// Eliminate `m_a` (multiply I1 by `y_a2`, I2 by `y_a1`, subtract):
///
/// ```text
///   y_a2*y_d * m_d - y_a1*y_c * m_c = y_a2*C1 - y_a1*C2
///   =>  m_d = (y_a1*y_c * m_c + (y_a2*C1 - y_a1*C2)) / (y_a2*y_d)
///   =>  ratio  = (y_a1*y_c) / (y_a2*y_d)        (must be a non-neg integer)
///       offset = (y_a2*C1 - y_a1*C2) / (y_a2*y_d)   (must be a non-neg integer)
/// ```
///
/// This is the family Rule B misses: the capacity-complement pair
/// `m_a + m_d = C1`, `m_a + m_c = C2` (`y_*=1`) yields `m_d = m_c + (C1 - C2)`
/// — a ratio-1, positive-offset coupling whose two places have DIFFERENT
/// initial markings and DIFFERENT arc signatures, so Rule B never fires on
/// them.
///
/// # Soundness (class A, 0-wrong)
///
/// 1. **Exact reconstruction.** Both invariants hold at EVERY reachable
///    marking, so the eliminated relation `m_d = ratio*m_c + offset` holds at
///    every reachable marking — `expand_marking` reproduces `m_d` with no
///    over-approximation.
/// 2. **Enabling preserved.** `is_lp_redundant` (the same Colom-Silva LP that
///    backs Rule B's redundant-place check) proves the duplicate never gates
///    any live transition. Dropping its arcs leaves the enabled-transition set
///    unchanged at every reachable marking, so the reduced reachability graph
///    is in bijection with the original's — exactly Rule B's class-A guarantee.
///    The same modes are therefore admissible (see `allows_lateral_fusion`).
///
/// If `(ratio, offset)` are not exact non-negative integers, or the LP cannot
/// prove never-constrains, NO fusion is emitted (fail-closed). All arithmetic
/// is checked; any overflow declines the candidate.
///
/// `already_removed` excludes places already claimed by earlier rules (their
/// arcs will be gone — a certificate must not lean on them, mirroring
/// `find_redundant_places`); a place there can still serve as a *pivot* (its
/// state-equation row remains a true fact). `protected_places` excludes
/// query/self-loop-protected places from being a *duplicate* candidate.
pub(crate) fn find_lateral_fusions(
    net: &PetriNet,
    already_removed: &[bool],
    dead_transitions: &[TransitionIdx],
    protected_places: &[bool],
) -> Vec<LateralFusionMerge> {
    let np = net.num_places();
    let nt = net.num_transitions();

    // Same size guard as the redundant-place LP path: the never-constrains
    // certificate solves an LP with `np + nt` variables per candidate.
    if np + nt > MAX_LP_VARIABLES {
        return Vec::new();
    }

    let dead_set: Vec<bool> = {
        let mut d = vec![false; nt];
        for &TransitionIdx(t) in dead_transitions {
            d[t as usize] = true;
        }
        d
    };

    let invariants = compute_p_invariants(net);
    if invariants.is_empty() {
        return Vec::new();
    }

    // Only support-2 invariants participate (single-pivot affine elimination).
    let pairs: Vec<&PInvariant> = invariants.iter().filter(|i| i.support_len() == 2).collect();
    if pairs.len() < 2 {
        return Vec::new();
    }

    // `removed` grows as we accept fusions so each LP certificate is proven
    // without leaning on a place already scheduled for removal (sequentially
    // exact, exactly as `find_redundant_places` does).
    let mut removed = vec![false; np];
    for (p, &r) in already_removed.iter().enumerate() {
        removed[p] = r;
    }

    let mut result: Vec<LateralFusionMerge> = Vec::new();

    // For every ordered pair of distinct support-2 invariants (I1, I2) that
    // share a pivot place `a`, eliminate `a` and try to remove I1's other place
    // `d`, reconstructing it from I2's other place `c`.
    for (i, &inv1) in pairs.iter().enumerate() {
        let s1: Vec<(usize, u64)> = inv1.support().collect();
        for (j, &inv2) in pairs.iter().enumerate() {
            if i == j {
                continue;
            }
            let s2: Vec<(usize, u64)> = inv2.support().collect();

            // Find a shared pivot place `a` and the two "other" places.
            for &(a1, ya1) in &s1 {
                // a1 is the pivot in inv1; find it in inv2.
                let Some(&(_, ya2)) = s2.iter().find(|&&(p, _)| p == a1) else {
                    continue;
                };
                // `d` is inv1's non-pivot place; `c` is inv2's non-pivot place.
                let Some(&(d, yd)) = s1.iter().find(|&&(p, _)| p != a1) else {
                    continue;
                };
                let Some(&(c, yc)) = s2.iter().find(|&&(p, _)| p != a1) else {
                    continue;
                };

                // Degenerate / not-useful eliminations.
                if d == c || d == a1 || c == a1 {
                    continue;
                }
                // The canonical must survive; the duplicate must be removable.
                if removed[d] || removed[c] {
                    continue;
                }
                if protected_places.get(d).copied().unwrap_or(false) {
                    continue;
                }

                // ratio  = (ya1 * yc) / (ya2 * yd)
                // offset = (ya2 * C1 - ya1 * C2) / (ya2 * yd)
                let c1 = inv1.token_count as i128;
                let c2 = inv2.token_count as i128;
                let (ya1, ya2, yc, yd) = (ya1 as i128, ya2 as i128, yc as i128, yd as i128);

                let denom = ya2 * yd; // > 0 (semi-positive supports)
                if denom == 0 {
                    continue;
                }
                let ratio_num = ya1 * yc;
                let offset_num = ya2 * c1 - ya1 * c2;

                // Both must be EXACT NON-NEGATIVE integers (fail-closed otherwise).
                if ratio_num < 0 || ratio_num % denom != 0 {
                    continue;
                }
                if offset_num < 0 || offset_num % denom != 0 {
                    continue;
                }
                let ratio = ratio_num / denom;
                let offset = offset_num / denom;
                let (Ok(ratio), Ok(offset)) = (u64::try_from(ratio), u64::try_from(offset)) else {
                    continue;
                };

                // Cross-check the derived coupling against the initial marking:
                // m0(d) MUST equal ratio*m0(c) + offset, or the elimination was
                // unsound for this net (defensive; the algebra guarantees it).
                let m0c = net.initial_marking[c] as u128;
                let m0d = net.initial_marking[d] as u128;
                let predicted = (ratio as u128)
                    .checked_mul(m0c)
                    .and_then(|p| p.checked_add(offset as u128));
                if predicted != Some(m0d) {
                    continue; // fail-closed: coupling inconsistent with m0
                }

                // Enabling-irrelevance: the duplicate must NEVER gate a live
                // transition (same LP as Rule B's redundant-place certificate).
                if !is_lp_redundant(net, d, &dead_set, &removed) {
                    continue; // LP cannot prove never-constrains — fail-closed
                }

                removed[d] = true;
                result.push(LateralFusionMerge {
                    canonical: PlaceIdx(c as u32),
                    duplicate: PlaceIdx(d as u32),
                    offset,
                    ratio,
                });
                break; // `d` consumed; move to the next invariant pair
            }
        }
    }

    result.sort_by_key(|m| m.duplicate.0);
    result.dedup_by_key(|m| m.duplicate.0);
    result
}
