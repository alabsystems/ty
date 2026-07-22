// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ADVERSARIAL soundness audit of the medium/high-risk (rule, mode) pairs in
//! the structural-reduction catalog (`crates/tla-petri/src/reduction/`).
//!
//! # Two distinct soundness classes — and how this file attacks each
//!
//! The catalog splits into two soundness classes:
//!
//! A. **Enabling-preserving rules** — Rule N (never-disabling arc), Rule K
//!    (proven self-loop arc), Rule B (parallel place), LP-redundant
//!    (Colom-Silva), duplicate/dominated transition, dead/constant/isolated.
//!    These claim the reduced net's REACHABILITY GRAPH is in bijection with the
//!    original's (the enabled-transition set is unchanged at every marking).
//!    Their soundness does NOT depend on the protected set: even with EVERY
//!    place observed, they must preserve the full reachable marking set exactly.
//!    **A divergence here is a genuine bug.**
//!
//! B. **Query-irrelevance rules** — agglomeration, Rule R, Rule S, Rule H, Rule
//!    C (source), Rule F (non-decreasing), token-elimination, GCD scaling.
//!    These DELETE/fuse/freeze structure and only preserve the value set on
//!    SURVIVING/OBSERVED places. Their soundness depends ENTIRELY on the
//!    protected set covering every observed place — that is the
//!    empty-protected-set hazard the two shipped wrong answers were instances
//!    of. The prior phase (`reachability_set_preservation.rs`) pinned the Rule S
//!    half; here we attack each: protect every observed place (as the
//!    UpperBounds/Cardinality pipeline does) and confirm the value set IS
//!    preserved, then (where instructive) show that WITHOUT protection the
//!    hazard reappears — proving the guard, not a new bug.
//!
//! # The differential oracle
//!
//! BFS BOTH nets exhaustively, lift every reduced marking via `expand_marking`
//! (place-map aliasing, GCD scale-back, constant values, P-invariant
//! reconstruction for LP-redundant places), and compare the reachable marking
//! set projected onto the OBSERVABLE original places.
//!
//! For class-A tests we PROTECT EVERY PLACE: only enabling-preserving rules can
//! fire, and the FULL reachable marking set must match. For class-B tests we
//! protect the OBSERVED places and confirm preservation on them.
//!
//! A FAILING class-A test = a real unsound (rule, mode). A FAILING class-B
//! "guard works" test = the protected-set guard is broken (also a real bug).

use std::collections::{BTreeSet, HashSet, VecDeque};

use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::reduction::{reduce_iterative_structural_with_mode, ReducedNet, ReductionMode};

use super::support::{arc, place, trans};

// ---------------------------------------------------------------------------
// Differential harness
// ---------------------------------------------------------------------------

/// Exhaustive bounded BFS of the reachable marking set. `None` if truncated.
fn reachable_markings(net: &PetriNet, max_states: usize) -> Option<BTreeSet<Vec<u64>>> {
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
                let next = net.fire(&marking, tidx).expect("fire (test)");
                if seen.insert(next.clone()) {
                    if seen.len() > max_states {
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

/// Original place indices whose EXACT reachable value set is recoverable from
/// the reduced net by a `Card`/`UpperBounds` atom:
///   - uniquely mapped through `place_map` (no other original place aliases the
///     same reduced place), OR
///   - exactly reconstructed from a P-invariant (LP-redundant place), OR
///   - a GENUINE constant place (`report.constant_places`: token count provably
///     invariant under every firing sequence, recovered exactly).
///
/// CRITICAL: this deliberately EXCLUDES places whose value was merely FROZEN to
/// their initial marking by a removal/fusion rule (agglomeration intermediates,
/// Rule R/S removed places, source / non-decreasing places). Those land in
/// `ReducedNet.constant_values` too (materialize.rs:417-423 records m0 for every
/// removed non-redundant place), but their recorded "constant" is NOT their
/// reachable value set — it is the rule's query-IRRELEVANCE assumption. Treating
/// them as observable would conflate "the rule is unsound" with "the rule froze
/// a place it was told is unobserved". Only `report.constant_places` are the
/// genuine invariant-value constants. (Aggregate-aliased Rule H places are
/// excluded by the uniqueness test.)
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

/// THE ORACLE. Reduce at `mode` with `protected`, BFS both nets, lift reduced
/// markings, return (original-projected, reduced-projected) reachable value sets
/// over the observable places.
fn differential_reach(
    net: &PetriNet,
    protected: &[bool],
    mode: ReductionMode,
    max_states: usize,
) -> (ReducedNet, BTreeSet<Vec<u64>>, BTreeSet<Vec<u64>>) {
    let reduced = reduce_iterative_structural_with_mode(net, protected, mode, None)
        .expect("reduction must not fail");
    let original = reachable_markings(net, max_states)
        .expect("original net must be fully explorable (net is bounded by design)");
    let reduced_raw =
        reachable_markings(&reduced.net, max_states).expect("reduced net must be fully explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| {
            reduced
                .expand_marking(m)
                .expect("expand_marking must succeed")
        })
        .collect();
    let keep = observable_places(&reduced);
    (
        reduced,
        project(&original, &keep),
        project(&reduced_expanded, &keep),
    )
}

/// PRODUCTION-path oracle. The reachability / UpperBounds / Cardinality
/// examinations reduce via `reduce_iterative_structural_query_with_protected`
/// (`StructuralReductionSemantics::QueryRelevantOnly`), NOT via
/// `reduce_iterative_structural_with_mode(_, Reachability)`. The two differ in
/// ONE safety-critical place: the QueryRelevantOnly planner blocks Rule R from
/// fusing into an externally-protected CENTRAL place (`planning.rs:283,326`
/// `rule_r_blocked`), whereas the mode planner only suppresses `remove_place`
/// (`analysis_agglomeration.rs:442`) and still fuses — corrupting the protected
/// place's JOINT reachability. So the realistic Reachability-mode soundness
/// contract MUST be exercised through THIS path.
fn differential_query(
    net: &PetriNet,
    protected: &[bool],
    max_states: usize,
) -> (ReducedNet, BTreeSet<Vec<u64>>, BTreeSet<Vec<u64>>) {
    let reduced =
        crate::reduction::reduce_iterative_structural_query_with_protected(net, protected)
            .expect("query reduction must not fail");
    let original = reachable_markings(net, max_states)
        .expect("original net must be fully explorable (net is bounded by design)");
    let reduced_raw =
        reachable_markings(&reduced.net, max_states).expect("reduced net must be fully explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| {
            reduced
                .expand_marking(m)
                .expect("expand_marking must succeed")
        })
        .collect();
    let keep = observable_places(&reduced);
    (
        reduced,
        project(&original, &keep),
        project(&reduced_expanded, &keep),
    )
}

/// Per-place reachable VALUE sets (one BTreeSet of values per observable place),
/// for the original vs the expanded reduced net, via the production query path.
/// This is the contract the codebase actually documents and tests
/// (`reachability_set_preservation.rs`): each observed place's value set,
/// independently, must be preserved. (Distinct from JOINT marking-set
/// preservation, which fusion rules do NOT guarantee — see the b1/b2/b3
/// joint-divergence findings.)
fn per_place_value_sets(
    net: &PetriNet,
    protected: &[bool],
    max_states: usize,
) -> (ReducedNet, Vec<BTreeSet<u64>>, Vec<BTreeSet<u64>>) {
    let reduced =
        crate::reduction::reduce_iterative_structural_query_with_protected(net, protected)
            .expect("query reduction must not fail");
    let original = reachable_markings(net, max_states).expect("original explorable");
    let reduced_raw = reachable_markings(&reduced.net, max_states).expect("reduced explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();
    let keep = observable_places(&reduced);
    let orig_per: Vec<BTreeSet<u64>> = keep
        .iter()
        .map(|&p| original.iter().map(|m| m[p]).collect())
        .collect();
    let red_per: Vec<BTreeSet<u64>> = keep
        .iter()
        .map(|&p| reduced_expanded.iter().map(|m| m[p]).collect())
        .collect();
    (reduced, orig_per, red_per)
}

/// Exhaustive deadlock-existence under bounded BFS.
fn has_deadlock(net: &PetriNet, max_states: usize) -> Option<bool> {
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
                let next = net.fire(&marking, tidx).expect("fire (test)");
                if seen.insert(next.clone()) {
                    if seen.len() > max_states {
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

const MAX: usize = 300_000;

fn all_protected(net: &PetriNet) -> Vec<bool> {
    vec![true; net.num_places()]
}

// ###########################################################################
// CLASS A — enabling-preserving rules. PROTECT EVERY PLACE. The reachability
// graph must be in bijection, so the FULL reachable marking set must match.
// A divergence is a genuine unsound (rule, mode).
// ###########################################################################

// ---------------------------------------------------------------------------
// A1. LP-redundant place (Colom-Silva) under Reachability.
//
// The continuous LP relaxation can be LOOSER than integer reachability. We must
// allow the place to be a candidate (so we do NOT protect it) but protect the
// OTHER observable places, and confirm the lifted reachable set (incl. the
// reconstructed redundant place) is exact. The net is BOUNDED by the invariant.
// ---------------------------------------------------------------------------

/// `free + busy = 2` invariant; `free` is implicit (reconstructable + LP-proven
/// non-constraining). Bounded: done is capped because every finish needs a busy.
/// Wait — `done` is unbounded. Instead make it a pure rotation: finish returns
/// the token to free, and a SEPARATE observable `flag` toggles, so the net is
/// finite. `free` is the redundant candidate; everything else protected.
#[test]
fn attack_a1_lp_redundant_place_reachability_exact() {
    // free(0)+busy(1)=2 ; flag g0(2)<->g1(3) toggled by `tick` which needs busy.
    let net = PetriNet {
        name: Some("a1-lp-redundant".into()),
        places: vec![place("free"), place("busy"), place("g0"), place("g1")],
        transitions: vec![
            trans("start", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("finish", vec![arc(1, 1)], vec![arc(0, 1)]),
            // tick: needs busy>=1 (guard, net 0 on busy) and toggles g0->g1.
            trans(
                "tick",
                vec![arc(1, 1), arc(2, 1)],
                vec![arc(1, 1), arc(3, 1)],
            ),
            trans("reset", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![2, 0, 1, 0],
    };
    // Protect everything EXCEPT `free` (the implicit candidate). The pipeline
    // would protect every observed place; `free` is observed too, but to let the
    // rule FIRE and then verify exact reconstruction we leave it unprotected.
    let mut protected = all_protected(&net);
    protected[0] = false;
    let (reduced, orig, red) =
        differential_reach(&net, &protected, ReductionMode::Reachability, MAX);
    assert_eq!(
        orig, red,
        "LP-redundant removal of `free` diverged from integer reachability — the \
         continuous LP certificate over-approximated and removed a constraining \
         place, OR the reconstruction is wrong. redundant={:?} recon={:?}",
        reduced.report.redundant_places, reduced.reconstructions
    );
}

/// Weighted invariant `2*a + b = 4`; `b` candidate, consumer needs `b >= 2`.
/// Integer-reachable `b` ∈ {0,2,4}. The LP relaxation could believe `b` reaches
/// fractional values. Bounded rotation: `drain` returns tokens so the net is
/// finite. Observable `sink` toggles to keep it finite.
#[test]
fn attack_a1_lp_redundant_weighted_invariant_exact() {
    // 2a + b = 4. up: a->b(2). down: b(2)->a. tog: needs b>=2 (guard) toggles s0<->s1.
    let net = PetriNet {
        name: Some("a1-lp-weighted".into()),
        places: vec![place("a"), place("b"), place("s0"), place("s1")],
        transitions: vec![
            trans("up", vec![arc(0, 1)], vec![arc(1, 2)]),
            trans("down", vec![arc(1, 2)], vec![arc(0, 1)]),
            // tog: guard b>=2 (consume 2 + produce 2 = net 0), toggles s0->s1.
            trans(
                "tog",
                vec![arc(1, 2), arc(2, 1)],
                vec![arc(1, 2), arc(3, 1)],
            ),
            trans("untog", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![2, 0, 1, 0],
    };
    let mut protected = all_protected(&net);
    protected[1] = false; // leave `b` as a candidate
                          // Use the PRODUCTION query path: this is the realistic LP-redundant context
                          // (UpperBounds/Cardinality), and it avoids the separate mode-planner Rule R
                          // latent bug that the `2a+b=4` rotation would otherwise trigger (pinned
                          // independently in attack_b3b_*). The focus here is LP-redundant soundness.
    let (reduced, orig, red) = differential_query(&net, &protected, MAX);
    assert_eq!(
        orig, red,
        "weighted LP-redundant removal of `b` diverged from integer \
         reachability. redundant={:?} recon={:?}",
        reduced.report.redundant_places, reduced.reconstructions
    );
}

// ---------------------------------------------------------------------------
// A2. Rule N (never-disabling arc) under Reachability — loose structural bound.
// PROTECT EVERY PLACE: Rule N only drops an arc, never removes a place, so the
// full reachable marking set MUST be identical. A divergence means the
// lower-bound proof was unsound (stripped a load-bearing arc).
// ---------------------------------------------------------------------------

/// Two interacting invariants tighten the bound on `cap`. `work` reads `cap` as
/// a guard (net 0). If Rule N strips that arc on a loose bound, `work` fires
/// where cap < 2 and reaches new markings. ALL places protected ⇒ no place
/// removal ⇒ exact marking-set equality required.
#[test]
fn attack_a2_rule_n_cross_invariant_bound_exact() {
    let net = PetriNet {
        name: Some("a2-rule-n".into()),
        places: vec![
            place("cap"),
            place("mid"),
            place("obs"),
            place("g0"),
            place("g1"),
        ],
        transitions: vec![
            // fwd: cap(2)+obs(1) -> mid(1). bwd: mid -> cap(2)+obs(1).
            trans("fwd", vec![arc(0, 2), arc(2, 1)], vec![arc(1, 1)]),
            trans("bwd", vec![arc(1, 1)], vec![arc(0, 2), arc(2, 1)]),
            // work: guard cap>=2 (net 0), real effect g0->g1.
            trans(
                "work",
                vec![arc(0, 2), arc(3, 1)],
                vec![arc(0, 2), arc(4, 1)],
            ),
            trans("reset", vec![arc(4, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![2, 2, 1, 1, 0],
    };
    let (reduced, orig, red) =
        differential_reach(&net, &all_protected(&net), ReductionMode::Reachability, MAX);
    assert_eq!(
        orig, red,
        "Rule N stripped an arc on an INFLATED lower bound — reachable marking \
         set changed under full protection. never_disabling_arcs={:?}",
        reduced.report.never_disabling_arcs
    );
}

/// Rule N where the guard arc is genuinely load-bearing on a TIGHT path: the
/// resource `r + busy = 1` (a real binary semaphore). `work` needs r>=1. The
/// true lower bound on r is 0 (r is 0 whenever busy=1), so Rule N MUST NOT strip
/// the arc. If it does, `work` fires while the resource is held — a new marking.
/// All protected ⇒ exact.
#[test]
fn attack_a2_rule_n_binary_semaphore_must_not_strip() {
    let net = PetriNet {
        name: Some("a2-rule-n-binsem".into()),
        places: vec![place("r"), place("busy"), place("g0"), place("g1")],
        transitions: vec![
            trans("acquire", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("release", vec![arc(1, 1)], vec![arc(0, 1)]),
            // work: needs r>=1 (guard net 0), toggles g0->g1. lower(r)=0, so the
            // r arc is NOT never-disabling; stripping it would let work fire
            // while held.
            trans(
                "work",
                vec![arc(0, 1), arc(2, 1)],
                vec![arc(0, 1), arc(3, 1)],
            ),
            trans("reset", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 1, 0],
    };
    let (reduced, orig, red) =
        differential_reach(&net, &all_protected(&net), ReductionMode::Reachability, MAX);
    assert_eq!(
        orig, red,
        "Rule N wrongly stripped the r-guard on a binary semaphore (lower(r)=0) \
         — `work` now fires while the resource is held. never_disabling={:?}",
        reduced.report.never_disabling_arcs
    );
}

// ---------------------------------------------------------------------------
// A3. Rule B (parallel place) under Reachability AND ReachabilityDeadlock. The
// alias must be EXACT. Protect every place: the duplicate aliases to the
// canonical and the FULL marking set must match.
// ---------------------------------------------------------------------------

#[test]
fn attack_a3_rule_b_parallel_place_exact_and_deadlock() {
    let net = PetriNet {
        name: Some("a3-rule-b".into()),
        places: vec![place("p_a"), place("p_b"), place("sink")],
        transitions: vec![
            trans("produce", vec![arc(2, 1)], vec![arc(0, 1), arc(1, 1)]),
            trans("consume", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    };
    // Reachability marking-set exactness (full protection still aliases the dup).
    let (reduced, orig, red) =
        differential_reach(&net, &all_protected(&net), ReductionMode::Reachability, MAX);
    assert_eq!(
        orig, red,
        "Rule B alias not exact under Reachability. parallel={:?}",
        reduced.report.parallel_places
    );
    // Deadlock-existence under the deadlock mode.
    let orig_dl = has_deadlock(&net, MAX).expect("original explorable");
    let dl =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::ReachabilityDeadlock, None)
            .expect("reduction");
    let red_dl = has_deadlock(&dl.net, MAX).expect("reduced explorable");
    assert_eq!(
        orig_dl, red_dl,
        "Rule B changed deadlock-existence — alias not exact. parallel={:?}",
        dl.report.parallel_places
    );
}

// ---------------------------------------------------------------------------
// A4. Rule N admissible under ReachabilityDeadlock on a DEADLOCKING net. Rule N
// is gated TRUE for deadlock. If the stripped arc was load-bearing for the
// deadlock, the verdict flips.
// ---------------------------------------------------------------------------

#[test]
fn attack_a4_rule_n_deadlock_existence_preserved() {
    // r+busy=2. work reads r as a guard and consumes a one-shot token; once the
    // token is gone the system can still cycle acquire/release, so NO deadlock.
    // We instead build a net that DOES deadlock: a one-shot `arm` enables a
    // terminal `halt` that drains everything to a dead marking.
    let net = PetriNet {
        name: Some("a4-rule-n-deadlock".into()),
        places: vec![place("r"), place("busy"), place("armed"), place("dead")],
        transitions: vec![
            trans("acquire", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("release", vec![arc(1, 1)], vec![arc(0, 1)]),
            // halt: needs r>=2 (guard arc is the Rule N target on the r+busy<=2
            // invariant) AND consumes the one-shot `armed`, landing in `dead`
            // with no further enabled transition (dead has no out-arcs and r is
            // drained). This is a genuine reachable deadlock.
            trans("halt", vec![arc(0, 2), arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![2, 0, 1, 0],
    };
    let orig_dl = has_deadlock(&net, MAX).expect("original explorable");
    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::ReachabilityDeadlock, None)
            .expect("reduction");
    let red_dl = has_deadlock(&reduced.net, MAX).expect("reduced explorable");
    assert_eq!(
        orig_dl,
        red_dl,
        "an admissible ReachabilityDeadlock rule flipped the deadlock verdict on \
         a deadlocking net. never_disabling={:?} redundant={:?} parallel={:?}",
        reduced.report.never_disabling_arcs,
        reduced.report.redundant_places,
        reduced.report.parallel_places
    );
}

// ---------------------------------------------------------------------------
// A5. LP-redundant under ReachabilityDeadlock (gated TRUE). The reconstruction
// must not hide/manufacture a deadlock. Deadlocking net with an implicit place.
// ---------------------------------------------------------------------------

#[test]
fn attack_a5_lp_redundant_deadlock_existence_preserved() {
    // free+busy=2 implicit `free`. A terminal `lock` consumes free and a
    // one-shot `key`, reaching a dead marking. Deadlock must survive removal of
    // the implicit `free`.
    let net = PetriNet {
        name: Some("a5-lp-redundant-deadlock".into()),
        places: vec![place("free"), place("busy"), place("key"), place("dead")],
        transitions: vec![
            trans("start", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("finish", vec![arc(1, 1)], vec![arc(0, 1)]),
            // lock: needs free>=2 and the one-shot key, lands in dead.
            trans("lock", vec![arc(0, 2), arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![2, 0, 1, 0],
    };
    let orig_dl = has_deadlock(&net, MAX).expect("original explorable");
    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::ReachabilityDeadlock, None)
            .expect("reduction");
    let red_dl = has_deadlock(&reduced.net, MAX).expect("reduced explorable");
    assert_eq!(
        orig_dl, red_dl,
        "LP-redundant removal flipped the deadlock verdict. redundant={:?} \
         recon={:?}",
        reduced.report.redundant_places, reduced.reconstructions
    );
}

// ###########################################################################
// CLASS B — query-irrelevance / fusion rules. Two contracts are at stake:
//
//   (i)  PER-PLACE value-set preservation — the contract the codebase documents
//        and tests (`reachability_set_preservation.rs`): each observed place's
//        reachable value set, independently, is preserved. UpperBounds atoms are
//        per-place, so this is the load-bearing contract for UpperBounds. The
//        fusion rules SATISFY this (the b1/b2/b3 "per_place" tests PASS).
//
//   (ii) JOINT marking-set preservation — a CONJUNCTIVE Cardinality/Reachability
//        predicate (`ResolvedPredicate::And`, resolved_predicate.rs:31) over a
//        producer's pre-place AND the consumer's post-place. The fusion rules do
//        NOT satisfy this: collapsing the producer-then-consumer step deletes the
//        transient where the pre-place is decremented but the post-place not yet
//        incremented. The b1/b2/b3 "joint" tests DOCUMENT this divergence.
//
//        *** THIS IS NOT A SHIPPED BUG. *** The reachability pipeline gates on
//        `all_predicates_reduction_safe` (reachability/reduction.rs:295,
//        pipeline.rs:1230) before using the reduced net: `predicate_reduction_
//        safe` (resolved_predicate.rs:233) refuses the reduced net for any
//        `TokensCount` predicate referencing a place whose touching transition
//        was eliminated. A fusion that deletes the producer/consumer transition
//        therefore forces ORIGINAL-NET fallback for exactly the predicates that
//        could observe the lost transient. The `attack_bX_..._backstop_*` tests
//        prove this backstop fires on these nets. The joint-divergence tests
//        thus pin the reduction-layer behavior AND the necessity of the
//        predicate-safety gate as a coupled invariant.
//
// All Class-B tests run through the PRODUCTION reachability path
// (QueryRelevantOnly) with the observed places protected — the realistic
// UpperBounds/Cardinality wiring.
// ###########################################################################

/// The `predicate_reduction_safe` backstop, re-implemented locally over the
/// reduced net's `transition_map`: returns `false` iff some ORIGINAL transition
/// touching `referenced` was eliminated by the reduction (so the reachability
/// pipeline would refuse the reduced net and fall back to the original for a
/// predicate over `referenced`). Mirrors resolved_predicate.rs:233-244.
fn predicate_safe_on_reduced(net: &PetriNet, reduced: &ReducedNet, referenced: &[usize]) -> bool {
    for &p in referenced {
        for (tidx, t) in net.transitions.iter().enumerate() {
            let touches = t.inputs.iter().any(|a| a.place.0 as usize == p)
                || t.outputs.iter().any(|a| a.place.0 as usize == p);
            if touches && reduced.transition_map[tidx].is_none() {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// B1. Rule S fan-in.
// ---------------------------------------------------------------------------

fn rule_s_fan_in_net() -> PetriNet {
    PetriNet {
        name: Some("b1-rule-s-fanin".into()),
        places: vec![place("a0"), place("a1"), place("mid"), place("out")],
        transitions: vec![
            trans("p0", vec![arc(0, 1)], vec![arc(2, 1)]),
            trans("p1", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("c", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![2, 3, 0, 0],
    }
}

/// (i) PER-PLACE: with the observed places protected, every observed place's
/// reachable value set is preserved exactly. This is the UpperBounds contract
/// and it HOLDS.
#[test]
fn attack_b1_rule_s_per_place_value_sets_preserved() {
    let net = rule_s_fan_in_net();
    let mut protected = vec![true; net.num_places()];
    protected[2] = false; // mid internal
    let (_reduced, orig_per, red_per) = per_place_value_sets(&net, &protected, MAX);
    assert_eq!(
        orig_per, red_per,
        "Rule S must preserve every observed place's INDIVIDUAL reachable value \
         set (the UpperBounds contract)"
    );
}

/// (ii) JOINT (DOCUMENTED DIVERGENCE, backstopped): Rule S loses joint markings
/// spanning a0 (producer pre-place) and out (consumer post-place). NOT a shipped
/// bug: the producer/consumer transitions were eliminated, so
/// `predicate_reduction_safe` forces original-net fallback for any predicate
/// over a0/out. The test pins BOTH the divergence AND the backstop firing.
#[test]
fn attack_b1_rule_s_joint_diverges_but_predicate_backstop_fires() {
    let net = rule_s_fan_in_net();
    let mut protected = vec![true; net.num_places()];
    protected[2] = false;
    let (reduced, orig, red) = differential_query(&net, &protected, MAX);
    assert_ne!(
        orig, red,
        "Rule S preserves per-place value sets but DROPS joint markings (the \
         transient producer-fired-consumer-not-yet state)."
    );
    // Backstop: a conjunctive predicate over a0 (0) and out (3) references
    // places whose touching transitions (p0, c) were eliminated, so the
    // reachability pipeline refuses the reduced net.
    assert!(
        !predicate_safe_on_reduced(&net, &reduced, &[0, 3]),
        "predicate_reduction_safe MUST reject the reduced net for a predicate \
         over a0/out (their touching transitions were fused away) — this is the \
         soundness backstop that makes the joint loss harmless end-to-end"
    );
}

#[test]
fn attack_b1_rule_s_hazard_without_protection_freezes_observable() {
    // Documents (and regression-pins) the hazard: empty protected set ⇒ Rule S
    // fires and freezes the surviving observables.
    let net = rule_s_fan_in_net();
    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction");
    assert!(
        !reduced.report.rule_s_agglomerations.is_empty(),
        "empty protected set must let Rule S fire (the hazard precondition)"
    );
    let original = reachable_markings(&net, MAX).expect("explorable");
    let reduced_raw = reachable_markings(&reduced.net, MAX).expect("explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();
    // out (orig index 3) reachable values differ: original spans 0..=5, reduced
    // freezes at 0.
    let out_orig: BTreeSet<u64> = original.iter().map(|m| m[3]).collect();
    let out_red: BTreeSet<u64> = reduced_expanded.iter().map(|m| m[3]).collect();
    assert_ne!(
        out_orig, out_red,
        "DOCUMENTED HAZARD: empty protected set ⇒ Rule S freezes observable \
         `out`. The pipeline MUST protect every observed place."
    );
}

// ---------------------------------------------------------------------------
// B2. Rule R fan-out.
// ---------------------------------------------------------------------------

fn rule_r_fan_out_net() -> PetriNet {
    PetriNet {
        name: Some("b2-rule-r-fanout".into()),
        places: vec![place("src"), place("mid"), place("o0"), place("o1")],
        transitions: vec![
            trans("prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("c0", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("c1", vec![arc(1, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![3, 0, 0, 0],
    }
}

/// (i) PER-PLACE: Rule R preserves each observed output's individual value set.
#[test]
fn attack_b2_rule_r_per_place_value_sets_preserved() {
    let net = rule_r_fan_out_net();
    let protected = vec![true, false, true, true]; // src, o0, o1 observed; mid internal
    let (_reduced, orig_per, red_per) = per_place_value_sets(&net, &protected, MAX);
    assert_eq!(
        orig_per, red_per,
        "Rule R must preserve every observed place's INDIVIDUAL value set"
    );
}

/// (ii) JOINT (DOCUMENTED DIVERGENCE, backstopped): Rule R loses joint markings
/// spanning the producer pre-place `src` and the fan-out outputs `o0`/`o1`. NOT
/// a shipped bug — predicate-safety backstop fires.
#[test]
fn attack_b2_rule_r_joint_diverges_but_predicate_backstop_fires() {
    let net = rule_r_fan_out_net();
    let protected = vec![true, false, true, true];
    let (reduced, orig, red) = differential_query(&net, &protected, MAX);
    assert_ne!(
        orig, red,
        "Rule R preserves per-place value sets but drops joint markings."
    );
    // src (0) and o0 (2): prod/c0 were eliminated by Rule R fusion.
    assert!(
        !predicate_safe_on_reduced(&net, &reduced, &[0, 2]),
        "predicate_reduction_safe MUST reject the reduced net for a predicate \
         over src/o0 (touching transitions fused away) — the soundness backstop"
    );
}

// ---------------------------------------------------------------------------
// B3. Pre-agglomeration (Berthelot).
// ---------------------------------------------------------------------------

fn pre_agglomeration_net() -> PetriNet {
    PetriNet {
        name: Some("b3-preagg".into()),
        places: vec![place("a"), place("b"), place("mid"), place("out")],
        transitions: vec![
            trans("t_src", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1)]),
            trans("t_use", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![3, 2, 0, 0],
    }
}

/// (i) PER-PLACE: pre-agglomeration preserves each observed place's value set
/// (a ∈ {1,2,3}, b ∈ {0,1,2}, out ∈ {0,1,2} in both nets).
#[test]
fn attack_b3_pre_agglomeration_per_place_value_sets_preserved() {
    let net = pre_agglomeration_net();
    let protected = vec![true, true, false, true]; // a, b, out observed; mid internal
    let (_reduced, orig_per, red_per) = per_place_value_sets(&net, &protected, MAX);
    assert_eq!(
        orig_per, red_per,
        "pre-agglomeration must preserve every observed place's INDIVIDUAL value \
         set"
    );
}

/// (ii) JOINT (DOCUMENTED DIVERGENCE, backstopped): Berthelot pre-agglomeration
/// deletes the transient marking where t_src has fired (a,b decremented into
/// mid) but t_use has not yet pushed to out. The joint marking (a=2 ∧ b=1 ∧
/// out=0) is reachable in the original and NOT in the reduced net — `EF(a=2 ∧
/// b=1 ∧ out=0)` is TRUE on the original, FALSE on the reduced. This is the
/// documented soundness boundary of Berthelot agglomeration for conjunctive
/// reachability. NOT a shipped bug: t_src (touching a,b) was eliminated, so
/// `predicate_reduction_safe` forces original-net fallback for any predicate
/// over a or b. The test pins the divergence AND the backstop.
#[test]
fn attack_b3_pre_agglomeration_joint_diverges_but_predicate_backstop_fires() {
    let net = pre_agglomeration_net();
    let protected = vec![true, true, false, true];
    let (reduced, orig, red) = differential_query(&net, &protected, MAX);
    assert_ne!(
        orig, red,
        "Berthelot pre-agglomeration preserves per-place value sets but drops \
         the joint transient marking (a=2 ∧ b=1 ∧ out=0)."
    );
    // a (0) and b (1): t_src touches both and was eliminated by pre-agg.
    assert!(
        !predicate_safe_on_reduced(&net, &reduced, &[0, 1]),
        "predicate_reduction_safe MUST reject the reduced net for a predicate \
         over a/b (t_src fused away) — the soundness backstop that prevents the \
         joint loss from producing a wrong EF/AG answer end-to-end"
    );
}

// ---------------------------------------------------------------------------
// B3b. THE LATENT Rule R UNSOUNDNESS. `reduce_iterative_structural_with_mode`
// with `ReductionMode::Reachability` (the mode planner,
// `build_structural_plan_for_mode`) does NOT block Rule R from fusing into an
// externally-protected CENTRAL place: it only suppresses `remove_place`
// (`analysis_agglomeration.rs:442`) and STILL fuses the producers, corrupting
// the protected place's JOINT reachability. The QueryRelevantOnly planner DOES
// block this (`planning.rs:283,326` `rule_r_blocked`). No current production
// examination wires `(_, protected, Reachability)` into the mode planner (Rule
// R is gated Reachability-only and the real reachability path uses
// QueryRelevantOnly), so this is LATENT — but it is a landmine for any future
// examination that does. This test pins BOTH sides:
//   (mode planner)  — diverges on the protected place  (the bug);
//   (query planner) — exact                            (the correct guard).
// ---------------------------------------------------------------------------

/// `2*a + b = 4` rotation with an observable flag `s0<->s1` toggled by `tog`
/// (guard b>=2). The marking (a=2, b=0, s1=1) is reachable in the original.
/// Rule R via the MODE planner fuses producer `down` into consumer `up` on the
/// PROTECTED central place `a` (remove_place=false) and loses that marking,
/// whereas the QueryRelevantOnly planner blocks the fusion and is exact.
fn rule_r_protected_central_net() -> PetriNet {
    PetriNet {
        name: Some("rule-r-protected-central".into()),
        places: vec![place("a"), place("b"), place("s0"), place("s1")],
        transitions: vec![
            trans("up", vec![arc(0, 1)], vec![arc(1, 2)]),
            trans("down", vec![arc(1, 2)], vec![arc(0, 1)]),
            trans(
                "tog",
                vec![arc(1, 2), arc(2, 1)],
                vec![arc(1, 2), arc(3, 1)],
            ),
            trans("untog", vec![arc(3, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![2, 0, 1, 0],
    }
}

/// LATENT BUG (pinned, EXPECTED to diverge): the mode planner fuses Rule R into
/// the protected central place `a` and loses the reachable marking
/// (a=2 ∧ s1=1). If a future edit ADDS the `rule_r_blocked` central-place guard
/// to `build_structural_plan_for_mode` (fixing the latent bug), this assertion
/// flips and the test must be updated — that is the intended trip-wire.
#[test]
fn attack_b3b_rule_r_mode_planner_fuses_into_protected_central_place() {
    let net = rule_r_protected_central_net();
    let mut protected = vec![true; net.num_places()]; // a, s0, s1 observed
    protected[1] = false; // b is the unprotected consumer post-place (lets cond 4 pass)

    // Mode planner: now GUARDED (build_structural_plan_for_mode mirrors the
    // QueryRelevantOnly `!rule_r_blocked` central-place filter). Rule R must NOT
    // fuse into the protected central place `a`, and the reachable marking set
    // on observed places must be preserved exactly.
    let reduced =
        reduce_iterative_structural_with_mode(&net, &protected, ReductionMode::Reachability, None)
            .expect("reduction");
    let original = reachable_markings(&net, MAX).expect("explorable");
    let reduced_raw = reachable_markings(&reduced.net, MAX).expect("explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();
    let keep = observable_places(&reduced);
    let orig_proj = project(&original, &keep);
    let red_proj = project(&reduced_expanded, &keep);
    assert_eq!(
        orig_proj, red_proj,
        "Rule R mode-planner guard regression: the reachable marking set on \
         observed places must be preserved (the reachable marking a=2 ∧ s1=1 must \
         survive). If this diverges, build_structural_plan_for_mode lost the \
         central-place guard that mirrors the QueryRelevantOnly planner, and a \
         Cardinality query `EF(a=2 ∧ s1=1)` would be answered FALSE on the reduced \
         net but TRUE on the original."
    );
    // The central place `a` is protected, so no Rule R agglomeration may target it.
    assert!(
        reduced
            .report
            .rule_r_agglomerations
            .iter()
            .all(|agg| agg.place.0 != 0),
        "the guard must drop any Rule R agglomeration whose central place is the \
         protected place `a` (index 0)"
    );
}

/// The PRODUCTION reachability path on the SAME net is SOUND: the
/// QueryRelevantOnly planner blocks Rule R from the protected central place, so
/// the reachable marking set is exact. This proves the divergence above is a
/// property of the mode-planner path, not of the net, and that the production
/// reachability/UpperBounds/Cardinality examinations are NOT affected today.
#[test]
fn attack_b3b_rule_r_query_planner_is_sound_on_protected_central_place() {
    let net = rule_r_protected_central_net();
    let mut protected = vec![true; net.num_places()];
    protected[1] = false;
    let (reduced, orig, red) = differential_query(&net, &protected, MAX);
    assert!(
        reduced.report.rule_r_agglomerations.is_empty(),
        "QueryRelevantOnly planner must BLOCK Rule R on the protected central \
         place (rule_r_blocked guard). rule_r={:?}",
        reduced.report.rule_r_agglomerations
    );
    assert_eq!(
        orig, red,
        "the production reachability path preserves the reachable marking set \
         on the protected central place exactly"
    );
}

// ---------------------------------------------------------------------------
// B4. Rule F (non-decreasing). GUARD: protecting the monotonic place suppresses
// removal and preserves its observable value set exactly. (Rule F is the
// magnitude-changing class; an UpperBounds atom on the accumulator would
// observe it, so the pipeline protects it.)
// ---------------------------------------------------------------------------

fn rule_f_net() -> PetriNet {
    // BOUNDED: acc is read as a guard (net 0) and topped up by `feed` from a
    // FINITE supply `pool`. acc never decreases; pool drains to 0.
    PetriNet {
        name: Some("b4-rule-f".into()),
        places: vec![place("acc"), place("pool"), place("done")],
        transitions: vec![
            // work: guard acc>=2 (net 0), pool -> done.
            trans(
                "work",
                vec![arc(0, 2), arc(1, 1)],
                vec![arc(0, 2), arc(2, 1)],
            ),
            // feed: pool -> acc + done (acc increases, pool finite).
            trans("feed", vec![arc(1, 1)], vec![arc(0, 1), arc(2, 1)]),
        ],
        initial_marking: vec![2, 3, 0],
    }
}

#[test]
fn attack_b4_rule_f_guard_protects_observed_accumulator() {
    let net = rule_f_net();
    let (reduced, orig, red) =
        differential_reach(&net, &all_protected(&net), ReductionMode::Reachability, MAX);
    assert!(
        reduced.report.non_decreasing_places.is_empty(),
        "Rule F must NOT remove a protected accumulator. nondec={:?}",
        reduced.report.non_decreasing_places
    );
    assert_eq!(
        orig, red,
        "with the accumulator protected, Rule F preserves its value set exactly"
    );
}

#[test]
fn attack_b4_rule_f_hazard_without_protection_changes_accumulator() {
    let net = rule_f_net();
    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction");
    assert!(
        !reduced.report.non_decreasing_places.is_empty(),
        "empty protected set must let Rule F fire (hazard precondition). nondec={:?}",
        reduced.report.non_decreasing_places
    );
    let original = reachable_markings(&net, MAX).expect("explorable");
    let reduced_raw = reachable_markings(&reduced.net, MAX).expect("explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();
    let acc_orig: BTreeSet<u64> = original.iter().map(|m| m[0]).collect();
    let acc_red: BTreeSet<u64> = reduced_expanded.iter().map(|m| m[0]).collect();
    assert_ne!(
        acc_orig, acc_red,
        "DOCUMENTED HAZARD: Rule F removes the accumulator without protection; \
         an UpperBounds atom on `acc` would read a frozen value. The pipeline \
         MUST protect observed accumulators."
    );
}

// ###########################################################################
// CLASS B — GCD scaling round-trip (the Card-vs-non-multiple attack).
// ###########################################################################

/// Weight-3 cycle with a unit-weight `bleed` arc that breaks the gcd on p0.
/// gcd(6,3,3,1)=1 ⇒ GCD scaling must NOT divide p0 (its reachable values are
/// not all multiples of 3 once `bleed` fires). The round-trip must be exact.
#[test]
fn attack_b5_gcd_scaling_non_multiple_round_trip() {
    let net = PetriNet {
        name: Some("b5-gcd-nonmultiple".into()),
        places: vec![place("p0"), place("p1"), place("leak")],
        transitions: vec![
            trans("t0", vec![arc(0, 3)], vec![arc(1, 3)]),
            trans("t1", vec![arc(1, 3)], vec![arc(0, 3)]),
            trans("bleed", vec![arc(0, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![6, 0, 0],
    };
    // Protect everything so only GCD scaling (applied explicitly) can change
    // weights — isolates the GCD round-trip.
    let mut reduced = reduce_iterative_structural_with_mode(
        &net,
        &all_protected(&net),
        ReductionMode::Reachability,
        None,
    )
    .expect("reduction");
    crate::reduction::apply_final_place_gcd_scaling(&mut reduced).expect("gcd");
    let original = reachable_markings(&net, MAX).expect("explorable");
    let reduced_raw = reachable_markings(&reduced.net, MAX).expect("explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();
    let keep = observable_places(&reduced);
    assert_eq!(
        project(&original, &keep),
        project(&reduced_expanded, &keep),
        "GCD scaling broke a non-multiple reachable value — a Card atom on p0 \
         would read a value expand_marking cannot reproduce. scales={:?}",
        reduced.place_scales
    );
    assert_eq!(
        reduced.place_scales[0], 1,
        "GCD must NOT have scaled p0 (gcd of its arcs is 1 due to the bleed arc)"
    );
}

/// Clean weight-2 cycle: GCD scaling SHOULD fire and round-trip exactly.
#[test]
fn attack_b5_gcd_scaling_clean_round_trip() {
    let net = PetriNet {
        name: Some("b5-gcd-clean".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 2)], vec![arc(1, 2)]),
            trans("t1", vec![arc(1, 2)], vec![arc(0, 2)]),
        ],
        initial_marking: vec![4, 0],
    };
    let mut reduced = reduce_iterative_structural_with_mode(
        &net,
        &all_protected(&net),
        ReductionMode::Reachability,
        None,
    )
    .expect("reduction");
    crate::reduction::apply_final_place_gcd_scaling(&mut reduced).expect("gcd");
    let original = reachable_markings(&net, MAX).expect("explorable");
    let reduced_raw = reachable_markings(&reduced.net, MAX).expect("explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();
    let p0_orig: BTreeSet<u64> = original.iter().map(|m| m[0]).collect();
    let p0_red: BTreeSet<u64> = reduced_expanded.iter().map(|m| m[0]).collect();
    assert_eq!(p0_orig, p0_red, "clean GCD round-trip must be exact");
    assert!(
        reduced.place_scales[0] >= 2,
        "GCD scaling should fire on the clean weight-2 cycle"
    );
}
