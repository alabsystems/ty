// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ReductionMode::ReachabilityDeadlock` gate enforcement + deadlock-existence
//! preservation for the rules NOT already covered by `deadlock_preservation.rs`.
//!
//! # The contract
//!
//! `ReachabilityDeadlock` answers a single boolean: "∃ a reachable marking with
//! NO enabled transition?". A rule is admissible iff it leaves the
//! reachable-deadlock-EXISTENCE verdict unchanged (`types.rs:46`). The
//! admissible set (gates returning TRUE) is exactly:
//!   dead, constant, isolated, duplicate-transition, dominated-transition,
//!   parallel-place (Rule B), never-disabling-arc (Rule N), LP-redundant,
//!   source-place (Rule C), and non-decreasing-place (Rule F). Rules C and F
//!   never change the enabled-transition set at any reachable marking (a source
//!   place gates nothing; a non-decreasing place with m0 >= max_consume is
//!   always-satisfied), so the deadlock-existence boolean is identical — proven
//!   by the random-net differential gate (`reduction_differential_proptest.rs`).
//!
//! The FORBIDDEN set (gates returning FALSE) is the catastrophic class for
//! deadlock:
//!   - agglomeration / Rule R / Rule S / token-cycle (Rule H) fuse a
//!     producer-then-consumer pair and DELETE the transient intermediate
//!     marking — can hide or manufacture a deadlock;
//!   - self-loop transition removal and sink-transition removal can delete the
//!     ONLY enabled transition at some marking — manufacturing a deadlock (or,
//!     for a draining sink, ERASING one); both are rejected by the differential
//!     gate;
//!   - self-loop-arc removal (Rule K) strips an input requirement — can ERASE
//!     a deadlock.
//!
//! # What these tests pin
//!
//! 1. GATE ENFORCEMENT (structural): on nets that WOULD trigger each forbidden
//!    rule under `Reachability`, the `ReachabilityDeadlock` report must contain
//!    ZERO entries for that rule. A future edit that flips a gate to `true`
//!    (re-introducing the exact failure class) fails here.
//! 2. DEADLOCK-EXISTENCE preservation (differential BFS) for the admissible
//!    Rule N and dominated-transition rules, which `deadlock_preservation.rs`
//!    does not exercise.

use std::collections::{HashSet, VecDeque};

use crate::petri_net::{PetriNet, TransitionIdx};
use crate::reduction::{reduce_iterative_structural_with_mode, ReducedNet, ReductionMode};

use super::support::{arc, place, trans};

/// Bounded exhaustive BFS deadlock-existence check.
fn has_deadlock_bfs(net: &PetriNet, max_states: usize) -> Option<bool> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let init = net.initial_marking.clone();
    seen.insert(init.clone());
    queue.push_back(init);

    while let Some(marking) = queue.pop_front() {
        let mut any_enabled = false;
        for t in 0..net.num_transitions() {
            let tidx = TransitionIdx(t as u32);
            if net.is_enabled(&marking, tidx) {
                any_enabled = true;
                let next = net.fire(&marking, tidx).expect("fire (test)");
                if seen.insert(next.clone()) {
                    if seen.len() > max_states {
                        return None;
                    }
                    queue.push_back(next);
                }
            }
        }
        if !any_enabled {
            return Some(true);
        }
    }
    Some(false)
}

fn reduce_dl(net: &PetriNet) -> ReducedNet {
    reduce_iterative_structural_with_mode(net, &[], ReductionMode::ReachabilityDeadlock, None)
        .expect("ReachabilityDeadlock reduction must not fail")
}

fn reduce_reach(net: &PetriNet) -> ReducedNet {
    reduce_iterative_structural_with_mode(net, &[], ReductionMode::Reachability, None)
        .expect("Reachability reduction must not fail")
}

// ===========================================================================
// GATE ENFORCEMENT: forbidden rules must stay OFF for ReachabilityDeadlock.
//
// Each test uses a net the rule fires on under Reachability (positive control),
// then asserts the SAME net produces zero entries for that rule under
// ReachabilityDeadlock.
// ===========================================================================

/// Agglomeration (pre/post) is FORBIDDEN: fusing a producer-then-consumer pair
/// deletes the transient intermediate marking, which can hide/manufacture a
/// deadlock (`types.rs:37-45`).
#[test]
fn test_deadlock_mode_forbids_agglomeration() {
    // p_mid (m0=0) single producer t_src, single consumer t_use -> pre-agg
    // candidate under Reachability.
    let net = PetriNet {
        name: Some("dl-no-agg".into()),
        places: vec![place("p_in"), place("p_mid"), place("p_out")],
        transitions: vec![
            trans("t_src", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t_use", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![2, 0, 0],
    };

    // Positive control: agglomeration fires under Reachability.
    let reach = reduce_reach(&net);
    assert!(
        !reach.report.pre_agglomerations.is_empty()
            || !reach.report.post_agglomerations.is_empty()
            || !reach.report.rule_s_agglomerations.is_empty()
            || !reach.report.rule_r_agglomerations.is_empty(),
        "expected some agglomeration under Reachability (positive control)"
    );

    // The gate: NO agglomeration under ReachabilityDeadlock.
    let dl = reduce_dl(&net);
    assert!(
        dl.report.pre_agglomerations.is_empty()
            && dl.report.post_agglomerations.is_empty()
            && dl.report.rule_r_agglomerations.is_empty()
            && dl.report.rule_s_agglomerations.is_empty(),
        "ReachabilityDeadlock must NOT agglomerate (deadlock-manufacturing class). \
         pre={:?} post={:?} ruleR={:?} ruleS={:?}",
        dl.report.pre_agglomerations,
        dl.report.post_agglomerations,
        dl.report.rule_r_agglomerations,
        dl.report.rule_s_agglomerations,
    );
}

/// Token-cycle merge (Rule H) is FORBIDDEN: deletes cycle transitions that are
/// real firings and collapses individual markings.
#[test]
fn test_deadlock_mode_forbids_token_cycle_merge() {
    let net = PetriNet {
        name: Some("dl-no-ruleh".into()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t2", vec![arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };

    let reach = reduce_reach(&net);
    assert!(
        !reach.report.token_cycle_merges.is_empty(),
        "expected Rule H under Reachability (positive control)"
    );

    let dl = reduce_dl(&net);
    assert!(
        dl.report.token_cycle_merges.is_empty(),
        "ReachabilityDeadlock must NOT merge token cycles (Rule H deletes real \
         firings): {:?}",
        dl.report.token_cycle_merges
    );
}

/// Self-loop ARC removal (Rule K) is FORBIDDEN: stripping a test arc relaxes an
/// input requirement and can ERASE a deadlock (`types.rs:43`).
#[test]
fn test_deadlock_mode_forbids_self_loop_arc_removal() {
    // guard is non-decreasing with m0 >= weight, so Rule K WOULD strip the test
    // arc under Reachability (proven never-disabling).
    let net = PetriNet {
        name: Some("dl-no-rulek".into()),
        places: vec![place("guard"), place("work"), place("done")],
        transitions: vec![trans(
            "t",
            vec![arc(0, 1), arc(1, 1)],
            vec![arc(0, 1), arc(2, 1)],
        )],
        initial_marking: vec![2, 3, 0],
    };

    let reach = reduce_reach(&net);
    assert!(
        !reach.report.self_loop_arcs.is_empty(),
        "expected Rule K under Reachability (positive control)"
    );

    let dl = reduce_dl(&net);
    assert!(
        dl.report.self_loop_arcs.is_empty(),
        "ReachabilityDeadlock must NOT strip self-loop arcs (Rule K erases \
         deadlocks): {:?}",
        dl.report.self_loop_arcs
    );
}

/// Self-loop TRANSITION removal is FORBIDDEN: a self-loop transition may be the
/// ONLY enabled transition at some marking, so removing it manufactures a
/// deadlock.
#[test]
fn test_deadlock_mode_forbids_self_loop_transition_removal() {
    // t_loop is a pure self-loop on p (in 1 + out 1, net zero). t_move drains p.
    let net = PetriNet {
        name: Some("dl-no-selfloop-trans".into()),
        places: vec![place("p"), place("sink")],
        transitions: vec![
            trans("t_loop", vec![arc(0, 1)], vec![arc(0, 1)]),
            trans("t_move", vec![arc(0, 1)], vec![arc(1, 1)]),
        ],
        initial_marking: vec![2, 0],
    };

    let reach = reduce_reach(&net);
    assert!(
        !reach.report.self_loop_transitions.is_empty(),
        "expected self-loop transition removal under Reachability (positive control)"
    );

    let dl = reduce_dl(&net);
    assert!(
        dl.report.self_loop_transitions.is_empty(),
        "ReachabilityDeadlock must NOT remove self-loop transitions (can be the \
         only enabled transition at a marking): {:?}",
        dl.report.self_loop_transitions
    );
}

/// Sink-transition removal is FORBIDDEN: a sink may be the only enabled
/// transition at a marking; removing it manufactures a deadlock.
#[test]
fn test_deadlock_mode_forbids_sink_transition_removal() {
    let net = PetriNet {
        name: Some("dl-no-sink".into()),
        places: vec![place("p"), place("q")],
        transitions: vec![
            // t_fill: q -> p (keeps p replenished).
            trans("t_fill", vec![arc(1, 1)], vec![arc(0, 1)]),
            // t_sink: p -> (nothing). Pure consumer = sink candidate.
            trans("t_sink", vec![arc(0, 1)], vec![]),
        ],
        initial_marking: vec![1, 1],
    };

    let reach = reduce_reach(&net);
    assert!(
        !reach.report.sink_transitions.is_empty(),
        "expected sink-transition removal under Reachability (positive control)"
    );

    let dl = reduce_dl(&net);
    assert!(
        dl.report.sink_transitions.is_empty(),
        "ReachabilityDeadlock must NOT remove sink transitions: {:?}",
        dl.report.sink_transitions
    );
}

/// Source-place removal (Rule C) and non-decreasing-place removal (Rule F) are
/// ADMISSIBLE for deadlock (broadened, proven by the random-net differential
/// gate): a source place is consumed by no transition (gates nothing), and a
/// non-decreasing place with `m0 >= max_consume` is always-satisfied (never the
/// binding constraint), so neither changes the enabled-transition set at any
/// reachable marking — the deadlock-existence boolean is identical.
///
/// This net fires `t` until the finite supply `p_in` drains, then deadlocks
/// (p_in=0 disables the sole transition): a REAL reachable deadlock. The
/// reduction removes the source `p_acc` and/or the non-decreasing `p_mono`, and
/// the deadlock must survive.
#[test]
fn test_deadlock_mode_admits_source_and_non_decreasing_preserving_deadlock() {
    // p_acc is produced-only (source place). p_mono is non-decreasing with
    // sufficient m0 (Rule F candidate). p_in is a finite supply.
    let net = PetriNet {
        name: Some("dl-source-f".into()),
        places: vec![place("p_in"), place("p_acc"), place("p_mono")],
        transitions: vec![
            // produces into p_acc (never consumed) => source place.
            // reads+writes p_mono with equal weight (net >= 0) and consumes p_in.
            trans("t", vec![arc(0, 1), arc(2, 1)], vec![arc(1, 1), arc(2, 1)]),
        ],
        initial_marking: vec![3, 0, 2],
    };

    let reach = reduce_reach(&net);
    // At least one of source / non-decreasing should fire under Reachability.
    assert!(
        !reach.report.source_places.is_empty() || !reach.report.non_decreasing_places.is_empty(),
        "expected source/non-decreasing removal under Reachability (positive control): \
         source={:?} nondec={:?}",
        reach.report.source_places,
        reach.report.non_decreasing_places
    );

    // Ground truth: the original net reaches a deadlock (p_in drains to 0).
    let original_dl = has_deadlock_bfs(&net, 200_000).expect("original explorable");
    assert!(original_dl, "ground truth: this net deadlocks");

    let dl = reduce_dl(&net);
    // The broadened admission must actually fire one of the two rules here.
    assert!(
        !dl.report.source_places.is_empty() || !dl.report.non_decreasing_places.is_empty(),
        "ReachabilityDeadlock now ADMITS source (Rule C) / non-decreasing (Rule F) \
         removal: at least one must fire on this net. source={:?} nondec={:?}",
        dl.report.source_places,
        dl.report.non_decreasing_places
    );
    // And it must preserve the deadlock-existence verdict.
    let reduced_dl = has_deadlock_bfs(&dl.net, 200_000).expect("reduced explorable");
    assert_eq!(
        original_dl, reduced_dl,
        "source (Rule C) / non-decreasing (Rule F) removal must preserve \
         deadlock-existence under ReachabilityDeadlock"
    );
}

// ===========================================================================
// ADMISSIBLE rules: deadlock-existence preservation under differential BFS.
// (Rule B / blocked-by-constant / LP-redundancy are covered separately in
// deadlock_preservation.rs; here we cover Rule N and dominated transitions.)
// ===========================================================================

/// Rule N (never-disabling arc removal) is ADMISSIBLE: it removes an input arc
/// whose place has a proven structural token lower bound >= the arc weight, so
/// the arc never changes enabling at any reachable marking — the
/// deadlock-existence set is identical.
///
/// Net: a "resource" place `r` with a P-invariant `r + busy = 3` (every
/// acquire moves 1 r->busy, every release moves it back). A worker transition
/// `work` consumes 1 from `r` as a guard but `lower(r) = 3 - 1 = 2 >= 1`, so
/// that input arc is never-disabling (Rule N strips it). The net is bounded
/// (total tokens conserved) AND deadlock-free, so the reduced net must remain
/// deadlock-free.
#[test]
fn test_deadlock_mode_rule_n_preserves_deadlock_existence() {
    // Places: r (resource), busy, gate. Invariant r + busy = 3.
    //   acquire: r -> busy
    //   release: busy -> r
    //   work:    r (guard, weight 1) + gate -> r + gate   [net zero on r+gate;
    //            r input arc is the Rule N target]
    // lower(r): from r + busy = 3 and busy <= 3, the structural lower bound on
    // r is computed by the invariant machinery; with m0=(r:3,busy:0) and busy
    // bounded by 3, the proof yields lower(r) >= 0. To get a strict positive
    // lower bound we keep at least 2 in r by capping busy at 1.
    // Places: r (resource), busy, cap, g0, g1 (a gate token oscillating g0<->g1).
    // Invariant r + busy = 3, busy capped at 1 by `cap` so r in {2,3}, giving
    // a structural lower bound lower(r) >= 2.
    //
    //   acquire: r + cap -> busy
    //   release: busy -> r + cap
    //   work:    r(guard,1) + g0 -> r + g1   (the r input arc is the Rule N
    //            target: never-disabling since r>=2; the transition has a REAL
    //            net effect g0->g1, so it is NOT a full self-loop transition)
    //   reset:   g1 -> g0
    let net = PetriNet {
        name: Some("dl-rule-n".into()),
        places: vec![
            place("r"),
            place("busy"),
            place("cap"),
            place("g0"),
            place("g1"),
        ],
        transitions: vec![
            trans("acquire", vec![arc(0, 1), arc(2, 1)], vec![arc(1, 1)]),
            trans("release", vec![arc(1, 1)], vec![arc(0, 1), arc(2, 1)]),
            // work: reads r as a guard (self-loop on r) but really moves g0->g1.
            trans(
                "work",
                vec![arc(0, 1), arc(3, 1)],
                vec![arc(0, 1), arc(4, 1)],
            ),
            trans("reset", vec![arc(4, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![3, 0, 1, 1, 0],
    };

    let original_dl = has_deadlock_bfs(&net, 200_000).expect("original explorable");
    let dl = reduce_dl(&net);
    let reduced_dl = has_deadlock_bfs(&dl.net, 200_000).expect("reduced explorable");

    assert_eq!(
        original_dl, reduced_dl,
        "the admissible ReachabilityDeadlock rule set (incl. Rule N) must \
         preserve deadlock-existence on a bounded resource net"
    );
}

/// Dominated-transition removal (Rule L) is ADMISSIBLE: a dominated transition
/// is strictly harder to enable than its dominator and has the same net
/// effect, so whenever it is enabled the dominator is too. Removing it cannot
/// change which markings are dead. This net has a dominated/dominator pair; the
/// reduced net must be deadlock-equivalent.
#[test]
fn test_deadlock_mode_dominated_transition_preserves_deadlock_existence() {
    // t_light: a -> c (needs 1 of a). t_heavy: a + b -> c (needs 1 of a AND 1
    // of b, same net delta on c is +1; but deltas differ on a/b). To make them
    // share net effect AND have t_heavy strictly dominated, both must move the
    // SAME net effect. Use: t_light consumes {a:1}, t_heavy consumes {a:1,b:1},
    // both produce {a:1} (so net effect is -b only differs)... instead use the
    // canonical shape: identical delta, t_heavy needs a strict superset.
    //
    // t_light: a(1)->x(1)  [delta a:-1, x:+1]
    // t_heavy: a(1)+b(0?) ... must match delta exactly. Make both delta {a:-1,x:+1}
    // with t_heavy additionally a self-loop test arc on b (consume+produce b),
    // so its delta is identical but it ALSO requires b>=1 to fire => dominated.
    let net = PetriNet {
        name: Some("dl-dominated".into()),
        places: vec![place("a"), place("b"), place("x")],
        transitions: vec![
            // t_light: a -> x.
            trans("t_light", vec![arc(0, 1)], vec![arc(2, 1)]),
            // t_heavy: a -> x, plus a self-loop test arc on b (same net delta,
            // strictly stronger precondition) => dominated by t_light.
            trans(
                "t_heavy",
                vec![arc(0, 1), arc(1, 1)],
                vec![arc(2, 1), arc(1, 1)],
            ),
        ],
        initial_marking: vec![2, 0, 0],
    };

    // Positive control: dominated transition detected under Reachability.
    let reach = reduce_reach(&net);
    assert!(
        !reach.report.dominated_transitions.is_empty(),
        "expected a dominated transition (positive control): {:?}",
        reach.report.dominated_transitions
    );

    let original_dl = has_deadlock_bfs(&net, 200_000).expect("original explorable");
    let dl = reduce_dl(&net);
    let reduced_dl = has_deadlock_bfs(&dl.net, 200_000).expect("reduced explorable");
    assert_eq!(
        original_dl, reduced_dl,
        "dominated-transition removal must preserve deadlock-existence"
    );
}

/// Duplicate-transition removal is ADMISSIBLE: two structurally identical
/// transitions have the same enabling and effect, so collapsing them does not
/// change the enabled set at any marking. Deadlock-existence is preserved.
#[test]
fn test_deadlock_mode_duplicate_transition_preserves_deadlock_existence() {
    let net = PetriNet {
        name: Some("dl-duplicate".into()),
        places: vec![place("p"), place("q")],
        transitions: vec![
            trans("t_a", vec![arc(0, 1)], vec![arc(1, 1)]),
            // t_b is identical to t_a (duplicate).
            trans("t_b", vec![arc(0, 1)], vec![arc(1, 1)]),
            // t_back recycles so the net is not trivially terminating.
            trans("t_back", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    };

    let reach = reduce_reach(&net);
    assert!(
        !reach.report.duplicate_transitions.is_empty(),
        "expected duplicate transitions (positive control)"
    );

    let original_dl = has_deadlock_bfs(&net, 200_000).expect("original explorable");
    let dl = reduce_dl(&net);
    let reduced_dl = has_deadlock_bfs(&dl.net, 200_000).expect("reduced explorable");
    assert_eq!(
        original_dl, reduced_dl,
        "duplicate-transition removal must preserve deadlock-existence"
    );
}
