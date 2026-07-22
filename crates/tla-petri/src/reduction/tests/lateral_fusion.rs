// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Targeted tests for Berthelot lateral place fusion (`R_lat`) — the affine
//! generalization of parallel-place merge (Rule B).
//!
//! The keystone test ([`lateral_fusion_fires_where_rule_b_cannot`]) builds a
//! net with two complement P-invariants sharing a pivot place. Eliminating the
//! pivot yields an exact `m(d) = m(c) + offset` coupling whose two places have
//! IDENTICAL arc signatures but DIFFERENT initial markings — so Rule B's strict
//! `k=1` matcher declines, while `R_lat` fuses via the offset. It then asserts
//! the reduced net's lifted reachable set equals the original's across ALL
//! modes and that the deadlock-existence boolean is preserved.

use std::collections::{BTreeSet, HashSet, VecDeque};

use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::reduction::{
    find_lateral_fusions, find_parallel_places, reduce_iterative_structural_with_mode, ReducedNet,
    ReductionMode,
};

use super::support::{arc, place, trans};

const ALL_MODES: [ReductionMode; 6] = [
    ReductionMode::Reachability,
    ReductionMode::ReachabilityDeadlock,
    ReductionMode::NextFreeCTL,
    ReductionMode::CTLWithNext,
    ReductionMode::StutterInsensitiveLTL,
    ReductionMode::StutterSensitiveLTL,
];

/// Exhaustive bounded BFS of the reachable marking set.
fn reachable_markings(net: &PetriNet, cap: usize) -> Option<BTreeSet<Vec<u64>>> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut out: BTreeSet<Vec<u64>> = BTreeSet::new();
    let init = net.initial_marking.clone();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    seen.insert(init.clone());
    out.insert(init.clone());
    queue.push_back(init);
    while let Some(marking) = queue.pop_front() {
        for t in 0..net.num_transitions() {
            let tidx = TransitionIdx(t as u32);
            if net.is_enabled(&marking, tidx) {
                let next = net.fire(&marking, tidx).ok()?;
                if seen.insert(next.clone()) {
                    if seen.len() > cap {
                        return None;
                    }
                    out.insert(next.clone());
                    queue.push_back(next);
                }
            }
        }
    }
    Some(out)
}

fn has_deadlock(net: &PetriNet, cap: usize) -> Option<bool> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let init = net.initial_marking.clone();
    seen.insert(init.clone());
    queue.push_back(init);
    while let Some(marking) = queue.pop_front() {
        let mut any = false;
        for t in 0..net.num_transitions() {
            let tidx = TransitionIdx(t as u32);
            if net.is_enabled(&marking, tidx) {
                any = true;
                let next = net.fire(&marking, tidx).ok()?;
                if seen.insert(next.clone()) {
                    if seen.len() > cap {
                        return None;
                    }
                    queue.push_back(next);
                }
            }
        }
        if !any {
            return Some(true);
        }
    }
    Some(false)
}

/// The observable-place definition shared with the differential gate: a place
/// is observable if it is uniquely `place_map`-mapped, P-invariant
/// reconstructed, or a genuine constant. (Lateral-fused duplicates are NOT in
/// any of these; the dedicated `expand` check below verifies them directly.)
fn observable_places(reduced: &ReducedNet) -> Vec<usize> {
    let mut count = vec![0usize; reduced.place_unmap.len().max(1)];
    for mapped in reduced.place_map.iter().flatten() {
        count[mapped.0 as usize] += 1;
    }
    let reconstructed: HashSet<usize> = reduced
        .reconstructions
        .iter()
        .map(|r| r.place.0 as usize)
        .collect();
    let genuine_constant: HashSet<usize> = reduced
        .report
        .constant_places
        .iter()
        .map(|&PlaceIdx(p)| p as usize)
        .collect();
    (0..reduced.place_map.len())
        .filter(|&p| {
            reduced.place_map[p].is_some_and(|t| count[t.0 as usize] == 1)
                || reconstructed.contains(&p)
                || genuine_constant.contains(&p)
        })
        .collect()
}

fn project(markings: &BTreeSet<Vec<u64>>, keep: &[usize]) -> BTreeSet<Vec<u64>> {
    markings
        .iter()
        .map(|m| keep.iter().map(|&p| m[p]).collect::<Vec<u64>>())
        .collect()
}

/// A net with two complement P-invariants sharing pivot place `a`:
///   Inv1:  m(a) + m(d) = 3
///   Inv2:  m(a) + m(c) = 2
/// Eliminating `a`:  m(d) = m(c) + 1   (ratio 1, offset 1).
///
/// Places: a=0, c=1, d=2.  `c` and `d` have IDENTICAL arc signatures (both
/// produced by t_split, both consumed by t_join, weight 1) but DIFFERENT
/// initial markings (c=0, d=1), so Rule B declines while R_lat fuses.
fn complement_offset_net() -> PetriNet {
    PetriNet {
        name: Some("rlat-complement-offset".to_string()),
        places: vec![place("a"), place("c"), place("d")],
        transitions: vec![
            // t_split: a -> c + d   (net a:-1, c:+1, d:+1)
            trans("t_split", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)]),
            // t_join:  c + d -> a   (net a:+1, c:-1, d:-1)
            trans("t_join", vec![arc(1, 1), arc(2, 1)], vec![arc(0, 1)]),
        ],
        // a=2, c=0, d=1  =>  a+d=3, a+c=2.
        initial_marking: vec![2, 0, 1],
    }
}

#[test]
fn lateral_fusion_detects_complement_offset() {
    let net = complement_offset_net();
    let fusions = find_lateral_fusions(&net, &vec![false; net.num_places()], &[], &[false; 3]);
    assert_eq!(
        fusions.len(),
        1,
        "exactly one lateral fusion expected, got {fusions:?}"
    );
    let m = &fusions[0];
    // m(d) = 1*m(c) + 1.
    assert_eq!(m.ratio, 1, "ratio must be 1 (offset-only coupling)");
    assert_eq!(m.offset, 1, "offset must be C1 - C2 = 1");
    // Orientation: the surviving canonical reconstructs the removed duplicate.
    // d (idx 2) has m0 = 1 = m0(c=1)=0 *1 + 1, so d is the duplicate.
    assert_eq!(m.duplicate, PlaceIdx(2));
    assert_eq!(m.canonical, PlaceIdx(1));
}

#[test]
fn rule_b_declines_complement_offset() {
    // Same arc signatures, different initial markings => Rule B's strict k=1
    // matcher must NOT merge c and d. This is exactly the gap R_lat closes.
    let net = complement_offset_net();
    let parallel = find_parallel_places(&net, &[], &[false; 3]);
    assert!(
        parallel.is_empty(),
        "Rule B must decline (different m0); got {parallel:?}"
    );
}

#[test]
fn lateral_fusion_fires_where_rule_b_cannot() {
    let net = complement_offset_net();
    let cap = 10_000;
    let original = reachable_markings(&net, cap).expect("original explorable");
    let orig_dl = has_deadlock(&net, cap).expect("original deadlock decidable");

    let mut fired_any = false;
    for mode in ALL_MODES {
        // Protect every place EXCEPT the duplicate `d` (index 2). The pivot `a`
        // and canonical `c` stay fully observable; the fusion is free to remove
        // `d` (whose exact value is reconstructed via the affine coupling and
        // checked directly below). Protecting `d` would block its candidacy, so
        // we leave it unprotected to exercise the rule.
        let protected = vec![true, true, false];
        let reduced = reduce_iterative_structural_with_mode(&net, &protected, mode, None)
            .expect("reduction succeeds");

        if mode.allows_lateral_fusion() && !reduced.report.lateral_fusions.is_empty() {
            fired_any = true;
            // The fused duplicate (original place d=2) must NOT survive as its
            // own reduced place.
            assert!(
                reduced.place_map[2].is_none(),
                "fused duplicate d must be removed (mode {mode:?})"
            );
        }

        // Lift every reduced marking; the projected observable reachable SET
        // must match the original exactly.
        let reduced_raw = reachable_markings(&reduced.net, cap).expect("reduced explorable");
        let mut expanded: BTreeSet<Vec<u64>> = BTreeSet::new();
        for m in &reduced_raw {
            expanded.insert(reduced.expand_marking(m).expect("expand"));
        }

        let keep = observable_places(&reduced);
        assert_eq!(
            project(&original, &keep),
            project(&expanded, &keep),
            "reachability-set divergence under mode {mode:?}; report={:?}",
            reduced.report
        );

        // Direct affine-expansion check: for EVERY lifted reachable marking the
        // fused duplicate d must satisfy m(d) = ratio*m(c) + offset exactly.
        for m in &expanded {
            for fusion in &reduced.report.lateral_fusions {
                let c = fusion.canonical.0 as usize;
                let d = fusion.duplicate.0 as usize;
                assert_eq!(
                    m[d],
                    fusion.ratio * m[c] + fusion.offset,
                    "affine reconstruction must be exact at every reachable marking"
                );
            }
        }

        // Deadlock-existence preserved in the deadlock mode.
        if matches!(mode, ReductionMode::ReachabilityDeadlock) {
            let red_dl = has_deadlock(&reduced.net, cap).expect("reduced deadlock decidable");
            assert_eq!(orig_dl, red_dl, "deadlock boolean must be preserved");
        }
    }

    assert!(
        fired_any,
        "lateral fusion must fire in at least one admissible mode \
         (otherwise the test is vacuous)"
    );
}

#[test]
fn lateral_fusion_deadlock_mode_empty_protected() {
    // The production deadlock pipeline reduces with an EMPTY protected set.
    // The fusion must still preserve the deadlock-existence boolean.
    let net = complement_offset_net();
    let cap = 10_000;
    let orig_dl = has_deadlock(&net, cap).expect("orig deadlock decidable");
    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::ReachabilityDeadlock, None)
            .expect("reduction succeeds");
    let red_dl = has_deadlock(&reduced.net, cap).expect("reduced deadlock decidable");
    assert_eq!(
        orig_dl, red_dl,
        "deadlock boolean must be preserved under empty-protected deadlock reduction; report={:?}",
        reduced.report
    );
}

#[test]
fn lateral_fusion_no_op_without_size_two_invariant() {
    // A plain producer/consumer chain has no support-2 P-invariant pair to
    // eliminate a shared pivot from => no fusion (and no panic).
    let net = PetriNet {
        name: Some("rlat-noop".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![3, 0],
    };
    let fusions = find_lateral_fusions(&net, &vec![false; net.num_places()], &[], &[false; 2]);
    assert!(
        fusions.is_empty(),
        "no support-2 invariant pair => no fusion; got {fusions:?}"
    );
}

#[test]
fn lateral_fusion_fail_closed_on_non_integer_ratio() {
    // Build a net whose pivot-elimination ratio is NON-integer, so R_lat must
    // decline (fail-closed). Two complements with mismatched weights:
    //   Inv1:  2*m(a) + m(d) = K1
    //   Inv2:  3*m(a) + m(c) = K2
    // Eliminating a:  m(d) = (2*m(c) + ...) / 3  -> ratio 2/3, NON-integer.
    //
    // Achieved by a transition with net effect a:-1 producing 2 to d and 3 to
    // c (so 2a+d and 3a+c are conserved), and its reverse.
    let net = PetriNet {
        name: Some("rlat-nonint".to_string()),
        places: vec![place("a"), place("c"), place("d")],
        transitions: vec![
            // a -> 3c + 2d   (net a:-1, c:+3, d:+2) => 2a+d and 3a+c conserved
            trans("t_split", vec![arc(0, 1)], vec![arc(1, 3), arc(2, 2)]),
            trans("t_join", vec![arc(1, 3), arc(2, 2)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![2, 0, 0],
    };
    let fusions = find_lateral_fusions(&net, &vec![false; net.num_places()], &[], &[false; 3]);
    // ratio would be (1*3)/(1*2)=3/2 or (1*2)/(1*3)=2/3 — neither integer in
    // the orientation that also passes the non-negative-offset + LP gates.
    assert!(
        fusions.iter().all(|f| {
            // If anything fired it must be an EXACT integer coupling consistent
            // with the initial marking (fail-closed never emits a bad entry).
            let c = f.canonical.0 as usize;
            let d = f.duplicate.0 as usize;
            f.ratio * net.initial_marking[c] + f.offset == net.initial_marking[d]
        }),
        "any emitted fusion must be an exact integer coupling; got {fusions:?}"
    );
    assert!(
        fusions.is_empty(),
        "non-integer ratio coupling must fail closed; got {fusions:?}"
    );
}
