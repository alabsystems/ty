// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-check harness for the universally-sound pre-reductions.
//!
//! For a battery of small bounded nets this module asserts, with **zero
//! tolerated disagreements**, that the verdict computed on the *reduced* net
//! equals the verdict computed by an **independent exhaustive oracle** on the
//! *original* net, for every examination affected by dead-transition removal:
//!
//! 1. The dead-transition reduction itself is a state-graph **isomorphism**
//!    (modulo the never-taken dead transitions): same reachable-marking count,
//!    same edge count, same per-place token range, and every removed transition
//!    was genuinely never enabled in the original. This is the master soundness
//!    premise — an isomorphism that preserves every property atom preserves
//!    **every** examination verdict.
//! 2. End-to-end, the production examination entry points (which now apply the
//!    pre-reduction) agree with the independent oracle on StateSpace (all four
//!    metrics), QuasiLiveness, and Deadlock.
//!
//! The oracle is a deliberately naive textbook BFS so a bug in the production
//! exploration/reduction path cannot be masked by shared code.

use std::collections::{BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use super::universally_sound::{proven_dead_transitions, reduce_dead_transitions_only};
use crate::examination::{deadlock_verdict, quasi_liveness_verdict, state_space_stats};
use crate::explorer::ExplorationConfig;
use crate::output::Verdict;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

/// Independent exhaustive ground truth for a bounded net.
struct Oracle {
    states: usize,
    edges: u64,
    max_token_in_place: u64,
    max_token_sum: u64,
    /// Per original transition: enabled at some reachable marking.
    fireable: Vec<bool>,
    /// Some reachable marking enables no transition.
    has_deadlock: bool,
}

/// Textbook BFS over the reachable marking set. Asserts boundedness via a hard
/// state cap so a buggy (unbounded) battery net fails loudly instead of hanging.
fn oracle(net: &PetriNet) -> Oracle {
    const CAP: usize = 100_000;
    let nt = net.num_transitions();
    let mut seen: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut fireable = vec![false; nt];
    let mut edges = 0u64;
    let mut max_in_place = 0u64;
    let mut max_sum = 0u64;
    let mut has_deadlock = false;

    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());
    while let Some(marking) = queue.pop_front() {
        assert!(seen.len() <= CAP, "oracle: battery net is not bounded");
        max_in_place = max_in_place.max(marking.iter().copied().max().unwrap_or(0));
        max_sum = max_sum.max(marking.iter().sum());
        let mut any_enabled = false;
        for t in 0..nt {
            let ti = TransitionIdx(t as u32);
            if net.is_enabled(&marking, ti) {
                any_enabled = true;
                fireable[t] = true;
                edges += 1;
                let succ = net.fire(&marking, ti).expect("fire (test)");
                if !seen.contains(&succ) {
                    seen.insert(succ.clone());
                    queue.push_back(succ);
                }
            }
        }
        if !any_enabled {
            has_deadlock = true;
        }
    }

    Oracle {
        states: seen.len(),
        edges,
        max_token_in_place: max_in_place,
        max_token_sum: max_sum,
        fireable,
        has_deadlock,
    }
}

fn config() -> ExplorationConfig {
    // Generous deadline so the deadline-aware examination entries still fall
    // through to exact exploration on these tiny nets, while the LP/BMC phases
    // remain bounded.
    ExplorationConfig::new(1_000_000).with_deadline(Some(Instant::now() + Duration::from_secs(60)))
}

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.to_string(),
        name: None,
    }
}

fn arc(place: u32, weight: u64) -> Arc {
    Arc {
        place: PlaceIdx(place),
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

// ---------------------------------------------------------------------------
// Battery of small bounded nets.
// ---------------------------------------------------------------------------

/// Fully live 2-state alternator. No dead transition; reduction is a no-op.
fn net_live_alternator() -> PetriNet {
    PetriNet {
        name: Some("live-alternator".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t01", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t10", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    }
}

/// One structurally-dead transition: `t_dead` consumes from `p2`, which has no
/// producer and starts empty. The cheap structural detector finds it.
fn net_structural_dead() -> PetriNet {
    PetriNet {
        name: Some("structural-dead".into()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            trans("t_dead", vec![arc(2, 1)], vec![]),
        ],
        initial_marking: vec![1, 0, 0],
    }
}

/// LP-ONLY dead transition: `{p0, p1}` is a conserved mutex (`p0 + p1 = 1`), so
/// `t_join`, which needs `p0 >= 1 AND p1 >= 1`, is never enabled — but each
/// place individually reaches 1 and has a producer, so the structural detector
/// misses it; only the joint-enabling state-equation LP proves it dead.
fn net_lp_joint_dead() -> PetriNet {
    PetriNet {
        name: Some("lp-joint-dead".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t01", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t10", vec![arc(1, 1)], vec![arc(0, 1)]),
            // Self-loop probe needing both tokens at once (net effect zero).
            trans(
                "t_join",
                vec![arc(0, 1), arc(1, 1)],
                vec![arc(0, 1), arc(1, 1)],
            ),
        ],
        initial_marking: vec![1, 0],
    }
}

/// Bounded counter with a complement place (`busy + free = 2`) — gives the
/// production pipeline a redundant/parallel-place opportunity while staying a
/// fully live net.
fn net_mutex_complement() -> PetriNet {
    PetriNet {
        name: Some("mutex-complement".into()),
        places: vec![place("free"), place("busy")],
        transitions: vec![
            trans("acquire", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("release", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![2, 0],
    }
}

/// Net that deadlocks: a one-shot chain that drains to a dead marking, plus a
/// transition that is dead because its input trap stays empty.
fn net_deadlocking_with_dead() -> PetriNet {
    PetriNet {
        name: Some("deadlocking-with-dead".into()),
        places: vec![place("a"), place("b"), place("c")],
        transitions: vec![
            // a -> b -> (sink): terminates in marking (0, 0, ...).
            trans("ab", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("b_sink", vec![arc(1, 1)], vec![]),
            // c has no producer and starts empty: dead.
            trans("c_dead", vec![arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    }
}

/// Feed-forward / cone-shaped net: a producer fans into two independent
/// branches. Live; exercises the metric path on a wider net.
fn net_feed_forward() -> PetriNet {
    PetriNet {
        name: Some("feed-forward".into()),
        places: vec![place("src"), place("l"), place("r")],
        transitions: vec![
            trans("to_l", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("to_r", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("recycle", vec![arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    }
}

fn battery() -> Vec<PetriNet> {
    vec![
        net_live_alternator(),
        net_structural_dead(),
        net_lp_joint_dead(),
        net_mutex_complement(),
        net_deadlocking_with_dead(),
        net_feed_forward(),
    ]
}

// ---------------------------------------------------------------------------
// (1) Dead-transition reduction is a verdict-preserving state-graph isomorphism.
// ---------------------------------------------------------------------------

fn assert_dead_removal_isomorphic(net: &PetriNet) {
    let name = net.name.clone().unwrap_or_default();
    let reduced = reduce_dead_transitions_only(net, None);
    let orig = oracle(net);
    let red = oracle(&reduced.net);

    assert_eq!(
        orig.states, red.states,
        "{name}: reachable-marking count changed by dead-transition removal",
    );
    assert_eq!(orig.edges, red.edges, "{name}: edge count changed");
    assert_eq!(
        orig.max_token_in_place, red.max_token_in_place,
        "{name}: max_token_in_place changed",
    );
    assert_eq!(
        orig.max_token_sum, red.max_token_sum,
        "{name}: max_token_sum changed",
    );

    // Every removed transition must be genuinely never enabled in the original
    // (no false positive in the dead set).
    for &TransitionIdx(t) in &reduced.report.dead_transitions {
        assert!(
            !orig.fireable[t as usize],
            "{name}: removed transition t{t} WAS fireable in the original net — UNSOUND",
        );
    }

    // Every surviving transition's reachable-fireability is preserved, mapped
    // back through the reduced→original transition unmap.
    for (red_idx, &TransitionIdx(orig_t)) in reduced.transition_unmap.iter().enumerate() {
        assert_eq!(
            orig.fireable[orig_t as usize], red.fireable[red_idx],
            "{name}: surviving transition t{orig_t} fireability changed under reduction",
        );
    }
}

#[test]
fn dead_transition_removal_is_state_graph_isomorphic_on_battery() {
    for net in battery() {
        assert_dead_removal_isomorphic(&net);
    }
}

#[test]
fn dead_transition_pass_finds_structural_and_lp_dead() {
    // Structural detector alone catches the no-producer transition.
    let n = net_structural_dead();
    let dead = proven_dead_transitions(&n, None);
    assert_eq!(
        dead,
        vec![TransitionIdx(2)],
        "structural dead transition t_dead must be found",
    );

    // The joint-enabling LP is REQUIRED for the conserved-mutex probe: it has a
    // producer for each input place, so only the state-equation LP proves it
    // dead.
    let n = net_lp_joint_dead();
    let dead = proven_dead_transitions(&n, None);
    assert!(
        dead.contains(&TransitionIdx(2)),
        "LP-only dead transition t_join must be found (joint enabling infeasible), got {dead:?}",
    );
    // And it is genuinely dead per the oracle.
    assert!(
        !oracle(&n).fireable[2],
        "t_join must be unfireable in the real reachable set",
    );

    // A fully live net yields no dead transitions (no over-reduction).
    assert!(
        proven_dead_transitions(&net_live_alternator(), None).is_empty(),
        "live net must have no dead transitions",
    );
}

// ---------------------------------------------------------------------------
// (2) End-to-end: production verdict (with reduction) == independent oracle.
//     ZERO disagreements tolerated across the whole battery.
// ---------------------------------------------------------------------------

#[test]
fn statespace_production_matches_oracle_on_battery() {
    for net in battery() {
        let name = net.name.clone().unwrap_or_default();
        let oracle = oracle(&net);
        let stats = state_space_stats(&net, &config())
            .unwrap_or_else(|| panic!("{name}: StateSpace must complete on tiny net"));
        assert_eq!(
            stats.states,
            tla_bignum::BigUint::from(oracle.states as u64),
            "{name}: StateSpace states",
        );
        assert_eq!(
            stats.edges,
            tla_bignum::BigUint::from(oracle.edges),
            "{name}: StateSpace edges",
        );
        assert_eq!(
            stats.max_token_in_place, oracle.max_token_in_place,
            "{name}: StateSpace max_token_in_place",
        );
        assert_eq!(
            stats.max_token_sum, oracle.max_token_sum,
            "{name}: StateSpace max_token_sum",
        );
    }
}

#[test]
fn quasi_liveness_production_matches_oracle_on_battery() {
    for net in battery() {
        let name = net.name.clone().unwrap_or_default();
        let oracle = oracle(&net);
        let expected = if oracle.fireable.iter().all(|&f| f) {
            Verdict::True
        } else {
            Verdict::False
        };
        let got = quasi_liveness_verdict(&net, &config());
        assert_eq!(
            got, expected,
            "{name}: QuasiLiveness production verdict disagrees with exhaustive oracle",
        );
    }
}

#[test]
fn deadlock_production_matches_oracle_on_battery() {
    for net in battery() {
        let name = net.name.clone().unwrap_or_default();
        let oracle = oracle(&net);
        let expected = if oracle.has_deadlock {
            Verdict::True
        } else {
            Verdict::False
        };
        let got = deadlock_verdict(&net, &config());
        assert_eq!(
            got, expected,
            "{name}: Deadlock production verdict disagrees with exhaustive oracle",
        );
    }
}
