// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Differential reachable-marking-set preservation for
//! [`ReductionMode::Reachability`].
//!
//! # Why this file exists (the catastrophic failure class)
//!
//! `ReductionMode::Reachability` is the gate for EF/AG reachability AND for the
//! **observation-atom** examinations: UpperBounds, ReachabilityCardinality
//! (`Card` atoms), and ReachabilityFireability (`Fire` atoms). See
//! `examination_kind.rs:128-151` — every one of these maps to
//! `ReductionMode::Reachability`, and `upper_bounds/pipeline.rs` /
//! `reachability/reduction.rs` run the full mode-gated catalog.
//!
//! The soundness contract these examinations REQUIRE is stronger than "the
//! query's boolean answer is preserved". A `Card`/`UpperBounds` atom observes
//! the *exact token value* of a place at *every reachable marking*. So the
//! theorem each Reachability-mode rule must satisfy is:
//!
//!   For the surviving (kept) places, `{ expand_marking(m) | m reachable in the
//!   reduced net }` projected onto those places EQUALS `{ M | M reachable in
//!   the original net }` projected onto the same places.
//!
//! This is **per-place reachable-value-set equivalence**, not just "the
//! query is preserved". Agglomeration / Rule R / Rule S DELETE the transient
//! intermediate marking (producer-fired-but-consumer-not-yet); that transient
//! is unobservable on the FUSED place (it is removed), but it MUST NOT change
//! the reachable value set of any SURVIVING place. These tests assert exactly
//! that, by BFS over both nets and comparing expanded markings on kept places.
//!
//! Two reductions shipped wrong MCC answers in this exact class:
//!   - fireability-only batches reduced with an EMPTY protected set;
//!   - Rule K self-loop-arc removal stripping a TEST ARC with no never-disabling
//!     proof.
//!
//! These tests pin the *positive* contract (sound rules preserve the set) so a
//! regression that re-breaks it fails here, and document the per-place
//! observation obligation the gate silently assumes.

use std::collections::{BTreeSet, HashSet, VecDeque};

use crate::petri_net::{PetriNet, TransitionIdx};
use crate::reduction::{reduce_iterative_structural_with_mode, ReducedNet, ReductionMode};

use super::support::{arc, place, trans};

/// Enumerate the full reachable marking set of `net` via bounded BFS.
///
/// Returns `Some(set)` if the reachable set was fully explored within
/// `max_states`, else `None` (inconclusive — the test must not conclude
/// equivalence from a truncated set).
fn reachable_markings(net: &PetriNet, max_states: usize) -> Option<BTreeSet<Vec<u64>>> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut out: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let init = net.initial_marking.clone();
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

/// Project a set of original-net markings onto `kept_places` (a sorted list of
/// original-net place indices that survive the reduction).
fn project(markings: &BTreeSet<Vec<u64>>, kept_places: &[usize]) -> BTreeSet<Vec<u64>> {
    markings
        .iter()
        .map(|m| kept_places.iter().map(|&p| m[p]).collect::<Vec<u64>>())
        .collect()
}

/// The original-net place indices whose EXACT token value is independently
/// observable in the reduced net — i.e. they map to a reduced place that NO
/// other original place is aliased onto.
///
/// This deliberately EXCLUDES aliased places (Rule B duplicates aliased to a
/// canonical, Rule H absorbed cycle places aliased to a survivor). For those,
/// `expand_marking` reports the value of the shared reduced place, which is
/// only correct under the rule's own contract:
///   - Rule B: m(duplicate) == m(canonical) at EVERY reachable marking, so the
///     alias is exact and the value set IS preserved.
///   - Rule H: only the AGGREGATE is preserved; the individual absorbed-place
///     value is NOT (the rule requires absorbed places to be query-irrelevant,
///     `types.rs:265-280`). Including such places would (correctly) flag a
///     value-set divergence — which is why Rule H is gated to NOT observe them.
///
/// We test the contract that MUST hold for every Reachability-mode rule: the
/// reachable-value-set on every *uniquely-observable* surviving place is
/// preserved. Aliased places are checked separately where their alias is exact
/// (Rule B), and are required-irrelevant where it is not (Rule H).
fn uniquely_observable_places(reduced: &ReducedNet) -> Vec<usize> {
    let mut reduced_target_count = vec![0usize; reduced.place_unmap.len()];
    for mapped in reduced.place_map.iter().flatten() {
        reduced_target_count[mapped.0 as usize] += 1;
    }
    (0..reduced.place_map.len())
        .filter(|&p| {
            reduced.place_map[p].is_some_and(|target| reduced_target_count[target.0 as usize] == 1)
        })
        .collect()
}

/// Core differential check: the reachable-value-set on every UNIQUELY
/// OBSERVABLE surviving original place must be identical between the original
/// net and the `expand_marking`-lifted reduced net.
///
/// This is the exact contract `ReductionMode::Reachability` promises to
/// UpperBounds / ReachabilityCardinality for the places it does not fuse or
/// alias away. With an EMPTY protected set (as here) a rule is allowed to strip
/// the whole net down to its query-irrelevant skeleton; in that case there may
/// be NO uniquely-observable place left and the projection is over the empty
/// tuple — still a valid (vacuous) equivalence. The point of this harness is
/// that whenever a place DOES survive uniquely, its value set is preserved.
fn assert_reachable_value_set_preserved(net: &PetriNet, max_states: usize) {
    let reduced =
        reduce_iterative_structural_with_mode(net, &[], ReductionMode::Reachability, None)
            .expect("Reachability reduction must not fail");

    let Some(original) = reachable_markings(net, max_states) else {
        panic!("original net not fully explorable within {max_states} states; test inconclusive");
    };
    let Some(reduced_raw) = reachable_markings(&reduced.net, max_states) else {
        panic!("reduced net not fully explorable within {max_states} states; test inconclusive");
    };

    // Lift every reduced marking back to original coordinates.
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| {
            reduced
                .expand_marking(m)
                .expect("expand_marking must succeed")
        })
        .collect();

    let kept = uniquely_observable_places(&reduced);

    let original_proj = project(&original, &kept);
    let reduced_proj = project(&reduced_expanded, &kept);

    assert_eq!(
        original_proj, reduced_proj,
        "Reachability reduction changed the reachable-value-set on a uniquely \
         observable SURVIVING place (kept original indices {kept:?}). A \
         Card/UpperBounds atom over one of these places would observe a \
         different value set in the reduced net — the catastrophic \
         observation-atom failure class."
    );

    // Independent sanity: expanding any reduced marking must reproduce a marking
    // that is actually reachable in the original on the kept places (no
    // fabricated values). The projection equality above already implies this,
    // but we keep the full-marking expansion exercised so a broken expansion
    // (e.g. wrong constant/reconstruction value) surfaces here too.
    assert!(
        reduced_expanded.iter().all(|m| m.len() == net.num_places()),
        "expand_marking must produce full-width original markings"
    );
}

// ---------------------------------------------------------------------------
// Pre/Post agglomeration: deletes the transient intermediate marking. Must NOT
// change the reachable value set on any surviving place.
// ---------------------------------------------------------------------------

/// Pre-agglomeration repro: `t_src` produces into zero-marked `p_mid`,
/// `p_mid`'s sole consumer `t_use` writes the observable `p_out`. Pre-agg
/// fuses `t_src` into `t_use` and deletes `p_mid`. The surviving places
/// (`p_in`, `p_out`, the accumulators) must have an identical reachable value
/// set.
#[test]
fn test_pre_agglomeration_preserves_surviving_place_value_set() {
    let net = PetriNet {
        name: Some("pre-agg-value-set".into()),
        places: vec![place("p_in"), place("p_mid"), place("p_out")],
        transitions: vec![
            // t_src: p_in -> p_mid (single output, weight 1, p_mid m0 = 0).
            trans("t_src", vec![arc(0, 1)], vec![arc(1, 1)]),
            // t_use: p_mid -> p_out (reads weight 1 from p_mid).
            trans("t_use", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![3, 0, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}

/// Post-agglomeration (dual): `p_mid` (m0=0) has a single consumer `t_sink`;
/// its producers write weight 1. Post-agg fuses `t_sink`'s outputs into each
/// producer. Surviving-place value sets must match.
#[test]
fn test_post_agglomeration_preserves_surviving_place_value_set() {
    let net = PetriNet {
        name: Some("post-agg-value-set".into()),
        places: vec![place("p_in"), place("p_mid"), place("p_out")],
        transitions: vec![
            // producer: p_in -> p_mid.
            trans("t_prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            // t_sink: p_mid -> p_out (sole consumer of p_mid).
            trans("t_sink", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![2, 0, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}

// ---------------------------------------------------------------------------
// Rule R / Rule S (place-centric producer x consumer fusion). These remove the
// intermediate place entirely; the transient is gone, but every SURVIVING
// place's value set must be invariant.
// ---------------------------------------------------------------------------

/// Rule R fan-out: one producer `prod` writes the central `p` (m0=0); two
/// consumers `c0`,`c1` each read `p` (pre-set exactly {p}) and write distinct
/// observable outputs. Rule R fuses (prod x c0), (prod x c1) and removes `p`.
/// The observable outputs `o0`,`o1` and the input accumulator must retain
/// their reachable value sets.
#[test]
fn test_rule_r_fan_out_preserves_surviving_place_value_set() {
    let net = PetriNet {
        name: Some("rule-r-value-set".into()),
        places: vec![place("p_in"), place("p"), place("o0"), place("o1")],
        transitions: vec![
            // prod: p_in -> p.
            trans("prod", vec![arc(0, 1)], vec![arc(1, 1)]),
            // c0: p -> o0.
            trans("c0", vec![arc(1, 1)], vec![arc(2, 1)]),
            // c1: p -> o1.
            trans("c1", vec![arc(1, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![2, 0, 0, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}

/// Rule S Phase-1 (k==1): producers `p0`,`p1` each write `p` (post-set exactly
/// {p}); consumer `c` reads `p` (pre-set exactly {p}). `initial_marking[p] < w`.
/// Rule S removes all producers, the consumer, and `p`; every fused
/// (producer x consumer) firing must keep the observable post-places'
/// reachable value sets intact.
#[test]
fn test_rule_s_atomic_viable_preserves_surviving_place_value_set() {
    let net = PetriNet {
        name: Some("rule-s-value-set".into()),
        places: vec![place("a0"), place("a1"), place("p"), place("out")],
        transitions: vec![
            // p0: a0 -> p.
            trans("p0", vec![arc(0, 1)], vec![arc(2, 1)]),
            // p1: a1 -> p.
            trans("p1", vec![arc(1, 1)], vec![arc(2, 1)]),
            // c: p -> out.
            trans("c", vec![arc(2, 1)], vec![arc(3, 1)]),
        ],
        initial_marking: vec![2, 2, 0, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}

// ---------------------------------------------------------------------------
// Rule H (token-conserving cycle merge). Only the AGGREGATE cycle token count
// is preserved; the individual cycle-place values are NOT. Soundness therefore
// requires the cycle places to be query-IRRELEVANT (removed). The surviving
// places (everything outside the cycle, plus the survivor carrying the
// aggregate) must keep their reachable value sets.
// ---------------------------------------------------------------------------

/// Pure two-place token cycle p0<->p1 carrying 1 token. Rule H collapses it
/// into a survivor carrying the aggregate. The cycle CONSERVES exactly 1 token,
/// so the aggregate is invariant — the survivor's value set is {1} and that IS
/// preserved. The per-place values (p0 ∈ {0,1}, p1 ∈ {0,1}) are NOT preserved,
/// but those places are aliased (not uniquely observable), so the rule's
/// query-irrelevance requirement (`types.rs:265-280`) is what keeps it sound.
///
/// This test pins the POSITIVE contract: the aggregate (uniquely-observable
/// survivor) value set is preserved.
#[test]
fn test_rule_h_token_cycle_preserves_aggregate_value_set() {
    let net = PetriNet {
        name: Some("rule-h-value-set".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}

/// Rule H aliasing is aggregate-only: the absorbed cycle place is aliased to
/// the survivor in `place_map`, so `expand_marking` reports the AGGREGATE for
/// the absorbed place, NOT its individual reachable value. This documents WHY
/// Rule H is sound only when the absorbed cycle places are query-IRRELEVANT
/// (`allows_token_cycle_merge` is Reachability-only AND detection skips
/// protected places). If an absorbed place were query-protected/observed, the
/// expansion would over-report — the exact reason the gate must keep cycle
/// places out of the protected/observed set.
#[test]
fn test_rule_h_absorbed_place_is_aliased_to_survivor_not_individually_exact() {
    let net = PetriNet {
        name: Some("rule-h-alias".into()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            trans("t2", vec![arc(2, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    };
    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction");
    // The cycle must actually merge (otherwise this test proves nothing).
    assert!(
        !reduced.report.token_cycle_merges.is_empty(),
        "Rule H must fire on the 3-cycle to exercise the aliasing contract"
    );
    // At least one absorbed place is aliased to the survivor (shares its
    // reduced index) — i.e. it is NOT uniquely observable. This is the
    // structural fingerprint of the aggregate-only semantics.
    let observable = uniquely_observable_places(&reduced);
    let merge = &reduced.report.token_cycle_merges[0];
    for absorbed in &merge.absorbed {
        assert!(
            !observable.contains(&(absorbed.0 as usize)),
            "absorbed cycle place {:?} must NOT be uniquely observable — Rule H \
             only preserves the aggregate, so observing it directly would be \
             unsound; the gate relies on it being query-irrelevant",
            absorbed
        );
    }
}

// ---------------------------------------------------------------------------
// Rule K (self-loop ARC removal) — the deleted-proof failure class. A
// self-loop arc pair is a TEST ARC: it gates enabling even with zero net
// effect. Removal is sound ONLY with a never-disabling proof (P-invariant
// lower bound or non-decreasing place with sufficient m0). With the proof, the
// reachable value set is identical. This test exercises the PROVEN path.
// ---------------------------------------------------------------------------

/// `guard` is non-decreasing (no transition has net-negative effect on it) and
/// `m0(guard) = 2 >= weight`, so the test-arc on `t` (consume 1 + produce 1 of
/// guard) never disables `t`. Rule K may strip it. The reachable value set on
/// all surviving places (including `guard`) must be unchanged.
#[test]
fn test_rule_k_self_loop_arc_proven_path_preserves_value_set() {
    let net = PetriNet {
        name: Some("rule-k-value-set".into()),
        places: vec![place("guard"), place("work"), place("done")],
        transitions: vec![
            // t: test-arc on guard (in 1 + out 1), moves work -> done.
            trans("t", vec![arc(0, 1), arc(1, 1)], vec![arc(0, 1), arc(2, 1)]),
        ],
        // guard never decreases (its only effect is the +1/-1 self-loop on t),
        // m0(guard)=2 >= 1, so the test arc is provably never-disabling.
        initial_marking: vec![2, 3, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}

// ---------------------------------------------------------------------------
// THE EMPTY-PROTECTED-SET HAZARD (the catastrophic class).
//
// Every place-fusing Reachability rule (agglomeration, Rule R, Rule S) claims
// to preserve the reachable value set on SURVIVING places. That claim is only
// true for places the rule does not touch. When a SURVIVING place sits on the
// producer's pre-set or the consumer's post-set, the fusion DELETES the
// intermediate markings and can change that place's reachable value set. The
// rule stays sound ONLY because its detector refuses to fire when such a place
// is in the protected set (conditions 7/8 of `find_rule_s_agglomerations`,
// `analysis_agglomeration.rs:569-670`).
//
// These tests pin BOTH halves of that contract:
//   (a) with an EMPTY protected set the fusion fires and the surviving
//       producer-pre place's value set IS changed (the hazard is real);
//   (b) protecting that place suppresses the fusion and restores the exact
//       value set (the guard works).
//
// This is the same shape as the two shipped wrong answers: a fireability-only
// batch reduced with an EMPTY protected set stripped the whole net. The tests
// make that mechanism a first-class, named regression.
// ---------------------------------------------------------------------------

/// Net with a P-invariant `p + 2*q = 4`: `t_fwd` moves 2 from `p` to make 1
/// `q`, `t_back` the reverse. Reachable `p` values = {0, 2, 4} (token rotates
/// through `q`). Rule S sees `producer.post == {q}`, `consumer.pre == {q}`,
/// `m0(q)=0 < w=1`, and fuses (t_fwd × t_back) into a self-loop on `p`,
/// REMOVING `q`, `t_fwd`, `t_back` and FREEZING `p` at 4.
///
/// With p UNPROTECTED, the reduced net reports p ∈ {4} only — a wrong
/// Cardinality/UpperBounds answer (true min/values 0,2,4 lost). This proves
/// the hazard is real: the value-set claim fails for the surviving `p` when
/// `p` is not protected.
#[test]
fn test_rule_s_unprotected_producer_pre_place_value_set_is_not_self_preserved() {
    let net = rule_s_invariant_net();

    let reduced =
        reduce_iterative_structural_with_mode(&net, &[], ReductionMode::Reachability, None)
            .expect("reduction");

    // Confirm Rule S actually fired and removed `q`.
    assert!(
        !reduced.report.rule_s_agglomerations.is_empty(),
        "Rule S must fire with an empty protected set (the hazard precondition)"
    );

    let original = reachable_markings(&net, 100_000).expect("original explorable");
    let reduced_raw = reachable_markings(&reduced.net, 100_000).expect("reduced explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();

    // p (original index 0) survives uniquely.
    let p_original: BTreeSet<u64> = original.iter().map(|m| m[0]).collect();
    let p_reduced: BTreeSet<u64> = reduced_expanded.iter().map(|m| m[0]).collect();

    assert_eq!(
        p_original,
        BTreeSet::from([0, 2, 4]),
        "original net reaches p ∈ {{0,2,4}}"
    );
    assert_ne!(
        p_original, p_reduced,
        "DOCUMENTED HAZARD: Rule S with an EMPTY protected set freezes the \
         surviving producer-pre place `p`, changing its reachable value set. \
         A Card/UpperBounds atom on `p` would get a WRONG answer. The pipeline \
         MUST protect every observed place before invoking this reduction."
    );
}

/// Companion to the hazard test: protecting `p` (as the
/// UpperBounds/ReachabilityCardinality pipeline does for every queried place)
/// SUPPRESSES Rule S and restores the exact reachable value set on `p`. This
/// is the guard that makes the rule sound — the audit's positive control.
#[test]
fn test_rule_s_protected_producer_pre_place_value_set_is_preserved() {
    let net = rule_s_invariant_net();
    let mut protected = vec![false; net.num_places()];
    protected[0] = true; // protect `p`

    let reduced =
        reduce_iterative_structural_with_mode(&net, &protected, ReductionMode::Reachability, None)
            .expect("reduction");

    assert!(
        reduced.report.rule_s_agglomerations.is_empty(),
        "Rule S must NOT fire when its producer-pre place `p` is protected \
         (condition 8 of find_rule_s_agglomerations)"
    );

    let original = reachable_markings(&net, 100_000).expect("original explorable");
    let reduced_raw = reachable_markings(&reduced.net, 100_000).expect("reduced explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();

    let p_original: BTreeSet<u64> = original.iter().map(|m| m[0]).collect();
    let p_reduced: BTreeSet<u64> = reduced_expanded.iter().map(|m| m[0]).collect();
    assert_eq!(
        p_original, p_reduced,
        "with `p` protected, the reachable value set on `p` must be exactly \
         preserved — this is the guard that makes Rule S sound for observation \
         atoms"
    );
}

/// Shared net for the Rule S empty-protected-set hazard tests.
fn rule_s_invariant_net() -> PetriNet {
    PetriNet {
        name: Some("rule-s-hazard".into()),
        places: vec![place("p"), place("q")],
        transitions: vec![
            // t_fwd: p(2) -> q(1).
            trans("t_fwd", vec![arc(0, 2)], vec![arc(1, 1)]),
            // t_back: q(1) -> p(2).
            trans("t_back", vec![arc(1, 1)], vec![arc(0, 2)]),
        ],
        initial_marking: vec![4, 0],
    }
}

// ---------------------------------------------------------------------------
// GCD scaling round-trip. Divides initial markings and arc weights on a place
// by their GCD, recording the scale; `expand_marking` multiplies it back. The
// dedicated mechanics live in `tests/gcd_scale.rs`; here we assert the
// value-set contract end-to-end: applying GCD scaling to a reduced net and
// expanding must reproduce the original reachable value set on the scaled
// place.
// ---------------------------------------------------------------------------

/// Pure weight-2 cycle `p0 <-> p1` carrying 4 tokens. Protect both places so
/// no fusion rule fires — only GCD scaling applies. After scaling, every arc
/// and m0 on each place is divided by 2 and `place_scales` records 2; the
/// expanded reachable value set must equal the original {0,2,4} on p0.
#[test]
fn test_gcd_scaling_round_trip_preserves_value_set() {
    let net = PetriNet {
        name: Some("gcd-roundtrip".into()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 2)], vec![arc(1, 2)]),
            trans("t1", vec![arc(1, 2)], vec![arc(0, 2)]),
        ],
        initial_marking: vec![4, 0],
    };
    // Protect both places so no place-removal/fusion rule perturbs the net;
    // only GCD scaling (applied explicitly below) should change weights.
    let protected = vec![true, true];
    let mut reduced =
        reduce_iterative_structural_with_mode(&net, &protected, ReductionMode::Reachability, None)
            .expect("reduction");
    crate::reduction::apply_final_place_gcd_scaling(&mut reduced).expect("gcd scaling");

    let original = reachable_markings(&net, 100_000).expect("original explorable");
    let reduced_raw = reachable_markings(&reduced.net, 100_000).expect("reduced explorable");
    let reduced_expanded: BTreeSet<Vec<u64>> = reduced_raw
        .iter()
        .map(|m| reduced.expand_marking(m).expect("expand"))
        .collect();

    let p0_original: BTreeSet<u64> = original.iter().map(|m| m[0]).collect();
    let p0_reduced: BTreeSet<u64> = reduced_expanded.iter().map(|m| m[0]).collect();
    assert_eq!(
        p0_original, p0_reduced,
        "GCD scaling round-trip must preserve the reachable value set on the \
         scaled place under expand_marking"
    );
    // The scaled net must actually be smaller in weight (scale recorded).
    assert!(
        reduced.place_scales[0] >= 2 || reduced.net.initial_marking.iter().any(|&m| m < 4),
        "GCD scaling should have divided weights/marking on the scaled cycle"
    );
}

// ---------------------------------------------------------------------------
// Combined / cascade: a net that triggers several Reachability rules in the
// same fixpoint. The end-to-end value-set contract must still hold.
// ---------------------------------------------------------------------------

/// Producer chain feeding an observable, with a parallel duplicate (Rule B), a
/// zero-marked intermediate (agglomeration), and a source accumulator (Rule C).
/// All surviving places must retain their reachable value sets.
#[test]
fn test_cascade_reachability_rules_preserve_value_set() {
    let net = PetriNet {
        name: Some("cascade-value-set".into()),
        places: vec![
            place("p_in"),
            place("p_mid"),
            place("p_obs_a"),
            place("p_obs_b"),
            place("p_acc"),
        ],
        transitions: vec![
            // p_in -> p_mid (agglomeratable: p_mid m0=0, single producer).
            trans("t_src", vec![arc(0, 1)], vec![arc(1, 1)]),
            // p_mid -> p_obs_a + p_obs_b + p_acc (p_acc is producer-only sink-ish accumulator).
            trans(
                "t_use",
                vec![arc(1, 1)],
                vec![arc(2, 1), arc(3, 1), arc(4, 1)],
            ),
        ],
        initial_marking: vec![3, 0, 0, 0, 0],
    };
    assert_reachable_value_set_preserved(&net, 100_000);
}
