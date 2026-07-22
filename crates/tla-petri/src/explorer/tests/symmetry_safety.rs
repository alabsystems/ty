// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use std::collections::BTreeMap;

type ArcSignature = (u32, u64);
type TransitionSignature = (Vec<ArcSignature>, Vec<ArcSignature>);

fn arc(place: u32, weight: u64) -> Arc {
    Arc {
        place: PlaceIdx(place),
        weight,
    }
}

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.to_string(),
        name: None,
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

fn transition_multiset(
    net: &PetriNet,
    place_swap: Option<(u32, u32)>,
) -> BTreeMap<TransitionSignature, usize> {
    let mut signatures = BTreeMap::new();
    for t in &net.transitions {
        let signature = (
            normalized_arc_signature(&t.inputs, place_swap),
            normalized_arc_signature(&t.outputs, place_swap),
        );
        *signatures.entry(signature).or_insert(0) += 1;
    }
    signatures
}

fn normalized_arc_signature(arcs: &[Arc], place_swap: Option<(u32, u32)>) -> Vec<ArcSignature> {
    let mut signature: Vec<_> = arcs
        .iter()
        .map(|arc| (swap_place(arc.place.0, place_swap), arc.weight))
        .collect();
    signature.sort_unstable();
    signature
}

fn swap_place(place: u32, place_swap: Option<(u32, u32)>) -> u32 {
    match place_swap {
        Some((left, right)) if place == left => right,
        Some((left, right)) if place == right => left,
        _ => place,
    }
}

fn assert_discovered_groups_are_pairwise_swap_automorphisms(net: &PetriNet) {
    let identity = transition_multiset(net, None);
    for group in crate::explorer::symmetry::discover_place_symmetry(net) {
        for (idx, &left) in group.iter().enumerate() {
            for &right in &group[(idx + 1)..] {
                assert_eq!(
                    net.initial_marking[left as usize], net.initial_marking[right as usize],
                    "symmetry group contains places with different initial markings: {group:?}",
                );
                assert_eq!(
                    identity,
                    transition_multiset(net, Some((left, right))),
                    "symmetry group is not backed by a place-swap transition automorphism: {group:?}",
                );
            }
        }
    }
}

#[test]
fn discovered_symmetry_groups_must_be_swap_automorphisms() {
    let net = PetriNet {
        name: Some("weighted-false-positive".into()),
        places: vec![place("p0"), place("p1"), place("sink")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(1, 2)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![2, 2, 0],
    };

    assert!(
        crate::explorer::symmetry::discover_place_symmetry(&net).is_empty(),
        "equal-color or degree heuristics must not group places with different arc weights",
    );
    assert_discovered_groups_are_pairwise_swap_automorphisms(&net);
}

#[test]
fn known_symmetric_places_still_collapse() {
    let net = PetriNet {
        name: Some("known-symmetric".into()),
        places: vec![place("p0"), place("p1"), place("sink")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    };

    assert_eq!(
        crate::explorer::symmetry::discover_place_symmetry(&net),
        vec![vec![0, 1]],
    );
    assert_discovered_groups_are_pairwise_swap_automorphisms(&net);
}

/// When the place-symmetry group fits within the canonicalizer's BFS
/// budget, `closure_is_complete()` must return `true` so that
/// soundness-critical consumers (e.g. orbit-multiplication) may multiply
/// by the cached `permutations.len()` and trust the result.
#[test]
fn closure_complete_when_within_budget() {
    use crate::explorer::symmetry::{PetriCanonicalizer, PETRI_CANONICALIZER_CLOSURE_BUDGET};

    // Two interchangeable places (Sym = S_2, |Sym| = 2 ≤ 500).
    let net = PetriNet {
        name: Some("two-symmetric".into()),
        places: vec![place("p0"), place("p1"), place("sink")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    };

    let canon = PetriCanonicalizer::build(&net);
    assert!(
        canon.closure_is_complete(),
        "BFS closure of |Sym|=2 must fit comfortably under the budget of {}",
        PETRI_CANONICALIZER_CLOSURE_BUDGET,
    );
    assert!(
        !canon.generators().is_empty(),
        "expected at least one non-trivial generator for a 2-place symmetric net",
    );
}

/// When |Sym(G)| exceeds [`PETRI_CANONICALIZER_CLOSURE_BUDGET`] the BFS
/// closure is forcibly truncated and the cached permutation set is no
/// longer a subgroup. `closure_is_complete()` MUST return `false` so
/// callers that rely on the set being closed under composition (orbit
/// multiplication, exact-count recovery) refuse the truncated value.
#[test]
fn closure_flag_false_when_truncated() {
    use crate::explorer::symmetry::{PetriCanonicalizer, PETRI_CANONICALIZER_CLOSURE_BUDGET};

    // Seven mutually-interchangeable places fanning into a common sink via
    // one transition each. The discovered group is S_7 with |S_7| = 5040,
    // which is well above the 500-permutation budget.
    let num_sym = 7usize;
    let mut places = Vec::with_capacity(num_sym + 1);
    for i in 0..num_sym {
        places.push(place(&format!("p{i}")));
    }
    places.push(place("sink"));
    let sink_idx = num_sym as u32;

    let mut transitions = Vec::with_capacity(num_sym);
    for i in 0..num_sym {
        transitions.push(trans(
            &format!("t{i}"),
            vec![arc(i as u32, 1)],
            vec![arc(sink_idx, 1)],
        ));
    }
    let mut initial_marking = vec![1u64; num_sym];
    initial_marking.push(0);
    let net = PetriNet {
        name: Some("seven-symmetric".into()),
        places,
        transitions,
        initial_marking,
    };

    let canon = PetriCanonicalizer::build(&net);
    assert!(
        !canon.closure_is_complete(),
        "expected closure flag = false when |Sym(G)| = {}! exceeds budget {}",
        num_sym,
        PETRI_CANONICALIZER_CLOSURE_BUDGET,
    );
    assert!(
        !canon.generators().is_empty(),
        "generators must remain exposed even when closure is truncated, so \
         soundness-critical callers can enumerate the full group on demand",
    );
}

/// Deterministic-pseudorandom check that every place orbit published by
/// `discover_place_symmetry` on small generated nets is a true Petri-net
/// place-swap automorphism (H1 + H2 of the soundness proof). 200 nets
/// covers the parameter ranges (place count 2-6, transition count 1-6,
/// arc weight 1-2, marking 0-3) at which the Nauty/Bliss path is most
/// likely to discover non-trivial orbits.
#[test]
fn nauty_published_groups_are_true_automorphisms_random() {
    // Deterministic xorshift64 PRNG so failures reproduce without an extra
    // dev-dependency on `rand`.
    let mut state: u64 = 0xdead_beef_cafe_babe;
    let mut next_u64 = || -> u64 {
        let mut x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        x
    };
    let mut bounded = |bound: u64| -> u64 { next_u64() % bound };

    for trial in 0..200 {
        let num_places = (bounded(5) + 2) as usize; // 2..=6
        let num_trans = (bounded(6) + 1) as usize; // 1..=6
        let places: Vec<_> = (0..num_places).map(|i| place(&format!("p{i}"))).collect();
        let initial_marking: Vec<u64> = (0..num_places).map(|_| bounded(4)).collect();
        let mut transitions = Vec::with_capacity(num_trans);
        for ti in 0..num_trans {
            let num_inputs = (bounded(3) + 1) as usize; // 1..=3
            let num_outputs = (bounded(3) + 1) as usize; // 1..=3
            let inputs: Vec<_> = (0..num_inputs)
                .map(|_| arc(bounded(num_places as u64) as u32, bounded(2) + 1))
                .collect();
            let outputs: Vec<_> = (0..num_outputs)
                .map(|_| arc(bounded(num_places as u64) as u32, bounded(2) + 1))
                .collect();
            transitions.push(trans(&format!("t{ti}"), inputs, outputs));
        }
        let net = PetriNet {
            name: Some(format!("random-{trial}")),
            places,
            transitions,
            initial_marking,
        };
        assert_discovered_groups_are_pairwise_swap_automorphisms(&net);
    }
}
