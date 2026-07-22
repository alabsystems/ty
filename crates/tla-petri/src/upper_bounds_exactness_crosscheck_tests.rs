// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-check the UpperBounds exactness pipeline (static P-invariant / LP /
//! monotone seeding, the pre-BFS trap-LP ratification, and trap-LP residual
//! refinement) against a BRUTE-FORCE reachable maximum on small bounded nets.
//!
//! The pipeline emits `tracker.max_bound` — always a *witnessed* reachable
//! token-sum — and only tags it exact when a sound cap (P-invariant / LP /
//! monotone / trap-LP) provably coincides with it. This suite asserts that the
//! emitted value equals the true reachable maximum, i.e. the exactness shortcut
//! never over- or under-states a bound. Bare LP-max is therefore only ever a
//! pruning cap, never the answer.

use crate::examinations::upper_bounds::check_upper_bounds_properties;
use crate::explorer::ExplorationConfig;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{Formula, Property};

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.to_string(),
        name: None,
    }
}

fn arc(p: u32, weight: u64) -> Arc {
    Arc {
        place: PlaceIdx(p),
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

/// Brute-force the exact maximum weighted token-sum over the query places across
/// the reachable set. Returns `None` if the reachable set exceeds `cap` markings
/// (net is too large / unbounded for this oracle).
fn brute_force_max(net: &PetriNet, query: &[usize], cap: usize) -> Option<u64> {
    use std::collections::{HashSet, VecDeque};

    let sum_of = |m: &[u64]| -> u64 { query.iter().map(|&p| m[p]).sum() };

    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let start = net.initial_marking.clone();
    let mut best = sum_of(&start);
    seen.insert(start.clone());
    queue.push_back(start);

    while let Some(m) = queue.pop_front() {
        for t in &net.transitions {
            let enabled = t.inputs.iter().all(|a| m[a.place.0 as usize] >= a.weight);
            if !enabled {
                continue;
            }
            let mut next = m.clone();
            for a in &t.inputs {
                next[a.place.0 as usize] -= a.weight;
            }
            for a in &t.outputs {
                next[a.place.0 as usize] += a.weight;
            }
            if seen.insert(next.clone()) {
                best = best.max(sum_of(&next));
                if seen.len() > cap {
                    return None; // too large / unbounded for the oracle
                }
                queue.push_back(next);
            }
        }
    }
    Some(best)
}

/// Run the full UpperBounds pipeline for a single `bound(query)` query.
fn pipeline_bound(net: &PetriNet, query_ids: &[&str]) -> Option<u64> {
    let property = Property {
        id: "X-UpperBounds-00".to_string(),
        formula: Formula::PlaceBound(query_ids.iter().map(|s| s.to_string()).collect()),
    };
    let config = ExplorationConfig::new(10_000_000);
    let results = check_upper_bounds_properties(net, std::slice::from_ref(&property), &config);
    results[0].1
}

fn assert_pipeline_matches_bruteforce(net: &PetriNet, query_ids: &[&str], query_idx: &[usize]) {
    let truth = brute_force_max(net, query_idx, 1_000_000)
        .expect("oracle net must be small/bounded enough to brute-force");
    let pipeline = pipeline_bound(net, query_ids);
    assert_eq!(
        pipeline,
        Some(truth),
        "UpperBounds pipeline for query {query_ids:?} returned {pipeline:?}, \
         exhaustive reachable maximum is {truth}",
    );
}

/// Conserving shuttle p0(3) -> p1: every single- and multi-place query bound is
/// pinned by the P-invariant p0 + p1 = 3 and witnessed by BFS.
#[test]
fn conserving_net_bounds_are_exact() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![3, 0],
    };
    assert_pipeline_matches_bruteforce(&net, &["p0"], &[0]);
    assert_pipeline_matches_bruteforce(&net, &["p1"], &[1]);
    assert_pipeline_matches_bruteforce(&net, &["p0", "p1"], &[0, 1]);
}

/// Place p0 is pinned at its initial value 1 by a transition the state equation
/// (and trap LP) forces never to fire (its other input p2 has no producer). The
/// query sum {p0} is NOT monotone non-increasing in isolation — t_dead also
/// consumes p0 — so the value comes from the LP/structural cap meeting the
/// witnessed initial marking, exactly as exhaustive BFS reports.
#[test]
fn dead_transition_pinned_place_bound_is_exact() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p2")],
        transitions: vec![
            trans("t_live", vec![arc(0, 1)], vec![arc(0, 1)]),
            trans("t_dead", vec![arc(0, 1), arc(1, 1)], vec![]),
        ],
        initial_marking: vec![1, 0],
    };
    assert_pipeline_matches_bruteforce(&net, &["p0"], &[0]);
}

/// A net whose siphon `{p0,p1}` is kept marked by an initially-marked trap: the
/// bare state-equation LP admits `p0 = p1 = 0`, but the trap forces the sum ≥ 1.
/// The per-place upper bounds are still exact; this exercises the trap-aware LP
/// path inside the pipeline. Cross-checked against the exhaustive maximum.
#[test]
fn dual_kill_trap_net_bounds_are_exact() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1), arc(1, 1)], vec![arc(0, 1)]),
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(1, 1)]),
            trans("t2", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 1],
    };
    assert_pipeline_matches_bruteforce(&net, &["p0"], &[0]);
    assert_pipeline_matches_bruteforce(&net, &["p1"], &[1]);
    assert_pipeline_matches_bruteforce(&net, &["p0", "p1"], &[0, 1]);
}

/// Weighted net p0(4) -2-> t0 -1-> p1: the bare LP rounds the fractional optimum
/// soundly (ceil) and BFS witnesses the integer maximum; the emitted bound must
/// equal the exhaustive reachable maximum, never the looser LP relaxation.
#[test]
fn weighted_net_bounds_are_exact() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1)])],
        initial_marking: vec![4, 0],
    };
    assert_pipeline_matches_bruteforce(&net, &["p0"], &[0]);
    assert_pipeline_matches_bruteforce(&net, &["p1"], &[1]);
    assert_pipeline_matches_bruteforce(&net, &["p0", "p1"], &[0, 1]);
}
