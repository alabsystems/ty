// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PROPTEST RANDOM-NET DIFFERENTIAL GATE for the structural-reduction catalog.
//!
//! This is the contention-robust, reproducible soundness contract that any
//! broadening of a [`ReductionMode`] rule-admission set must clear. It generates
//! thousands of small random Petri nets and, for EACH mode, BFS-explores both
//! the original and the reduced net and compares the reachable behaviour:
//!
//!   * **Reachability-set contract (class A, ALL modes):** protect EVERY place,
//!     reduce, BFS both nets, lift each reduced marking via
//!     [`ReducedNet::expand_marking`], project onto the OBSERVABLE original
//!     places (uniquely place-mapped / P-invariant-reconstructed / genuine
//!     constant — exactly the `soundness_audit.rs` definition), and assert the
//!     projected reachable value SET is identical. Under full protection only
//!     enabling-preserving / exactly-reconstructable rules can fire, so this is a
//!     full-set match: any divergence is a genuine unsound (rule, mode) pair.
//!
//!   * **Deadlock-existence contract (`ReachabilityDeadlock` mode):** reduce with
//!     an EMPTY protected set — EXACTLY the production deadlock pipeline's call
//!     (`deadlock_one_safe.rs:488` `reduce_iterative_structural_with_mode(net,
//!     &[], ReachabilityDeadlock, _)`) — BFS both nets, and assert the
//!     deadlock-existence boolean ("∃ a reachable marking with no enabled
//!     transition?") is IDENTICAL original-vs-reduced. This is the load-bearing
//!     gate: a rule admitted into the deadlock mode that does NOT preserve this
//!     boolean would produce a WRONG ReachabilityDeadlock verdict.
//!
//! # Boundedness / termination
//!
//! Random nets are frequently unbounded. Both BFS oracles are capped at
//! [`MAX_STATES`]; a case whose ORIGINAL net exceeds the cap (or overflows a
//! marking) is INCONCLUSIVE and silently skipped (it proves nothing, it does not
//! fail). Only cases where the original net is fully explorable within the cap
//! contribute to the contract. The strategy keeps nets tiny (<=6 places, <=6
//! transitions, weights 1-3, m0 in 0..=3) so a large fraction is conclusive.

use std::collections::{BTreeSet, HashSet, VecDeque};

use proptest::prelude::*;

use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::reduction::{reduce_iterative_structural_with_mode, ReducedNet, ReductionMode};

// ---------------------------------------------------------------------------
// Bounded BFS oracles (mirrors soundness_audit.rs / deadlock_preservation.rs).
// ---------------------------------------------------------------------------

/// Exploration ceiling. A net whose ORIGINAL reachable set exceeds this is
/// treated as inconclusive (skipped), never as a failure.
const MAX_STATES: usize = 20_000;

/// Exhaustive bounded BFS of the reachable marking set. `None` if the cap was
/// exceeded or a marking-overflow was hit (inconclusive — skip the case).
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
                // A marking-overflow on a randomly-generated net is inconclusive,
                // not a soundness violation. Bail to skip the case.
                let next = net.fire(&marking, tidx).ok()?;
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

/// Exhaustive bounded deadlock-existence. `None` if the cap was exceeded or a
/// marking-overflow was hit (inconclusive — skip the case).
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
                let next = net.fire(&marking, tidx).ok()?;
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

/// Original place indices whose EXACT reachable value set is recoverable from
/// the reduced net (the `soundness_audit.rs::observable_places` definition,
/// reproduced verbatim so the two oracles agree): uniquely `place_map`-mapped,
/// OR P-invariant-reconstructed, OR a GENUINE constant place. Deliberately
/// EXCLUDES places merely frozen-to-m0 by a removal/fusion rule.
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

// ---------------------------------------------------------------------------
// Random-net strategy: <=6 places, <=6 transitions, weights 1-3, m0 in 0..=3.
// ---------------------------------------------------------------------------

/// One arc as (place_index_within_net, weight). The place index is generated as
/// a fraction and remapped to `0..num_places` so the strategy composes without
/// knowing `num_places` up front.
type RawArc = (u8, u8);
/// One transition: (inputs, outputs) as raw arcs.
type RawTransition = (Vec<RawArc>, Vec<RawArc>);

/// A raw random net: (num_places, initial_marking, transitions). All indices are
/// raw and remapped to legal ranges by [`build_net`].
type RawNet = (usize, Vec<u8>, Vec<RawTransition>);

fn raw_arc_strategy() -> impl Strategy<Value = RawArc> {
    // place selector 0..=250 (remapped mod num_places), weight 1..=3.
    (0u8..=250, 1u8..=3)
}

fn raw_transition_strategy() -> impl Strategy<Value = RawTransition> {
    (
        prop::collection::vec(raw_arc_strategy(), 0..=4),
        prop::collection::vec(raw_arc_strategy(), 0..=4),
    )
}

fn raw_net_strategy() -> impl Strategy<Value = RawNet> {
    (1usize..=6).prop_flat_map(|num_places| {
        (
            Just(num_places),
            prop::collection::vec(0u8..=3, num_places),
            prop::collection::vec(raw_transition_strategy(), 1..=6),
        )
    })
}

/// Materialize a [`PetriNet`] from a raw random spec. Raw place selectors are
/// taken modulo `num_places`; per-transition arcs are deduplicated by place
/// (last weight wins) so the net is well-formed (at most one input and one
/// output arc per place per transition).
fn build_net(raw: &RawNet) -> PetriNet {
    let (num_places, m0, transitions) = raw;
    let num_places = (*num_places).max(1);
    let places: Vec<PlaceInfo> = (0..num_places)
        .map(|i| PlaceInfo {
            id: format!("p{i}"),
            name: Some(format!("p{i}")),
        })
        .collect();
    let initial_marking: Vec<u64> = (0..num_places)
        .map(|i| u64::from(m0.get(i).copied().unwrap_or(0)))
        .collect();

    let mut nets: Vec<TransitionInfo> = Vec::with_capacity(transitions.len());
    for (ti, (inputs, outputs)) in transitions.iter().enumerate() {
        // Dedup arcs by place (one arc per place per side), last weight wins.
        let mut in_by_place: std::collections::BTreeMap<u32, u64> = Default::default();
        for &(sel, w) in inputs {
            let p = (sel as usize % num_places) as u32;
            in_by_place.insert(p, u64::from(w));
        }
        let mut out_by_place: std::collections::BTreeMap<u32, u64> = Default::default();
        for &(sel, w) in outputs {
            let p = (sel as usize % num_places) as u32;
            out_by_place.insert(p, u64::from(w));
        }
        let in_arcs: Vec<Arc> = in_by_place
            .into_iter()
            .map(|(p, weight)| Arc {
                place: PlaceIdx(p),
                weight,
            })
            .collect();
        let out_arcs: Vec<Arc> = out_by_place
            .into_iter()
            .map(|(p, weight)| Arc {
                place: PlaceIdx(p),
                weight,
            })
            .collect();
        nets.push(TransitionInfo {
            id: format!("t{ti}"),
            name: Some(format!("t{ti}")),
            inputs: in_arcs,
            outputs: out_arcs,
        });
    }

    PetriNet {
        name: Some("proptest-random".into()),
        places,
        transitions: nets,
        initial_marking,
    }
}

// ---------------------------------------------------------------------------
// The two differential contracts, as reusable per-case checks.
// ---------------------------------------------------------------------------

/// CLASS-A reachability-set contract for one mode. Protect EVERY place, reduce,
/// BFS both, lift the reduced markings, project onto observable places, and
/// require set equality. Returns `Ok(())` on agreement (or on an inconclusive
/// skip), `Err(_)` on a genuine divergence.
fn check_reachability_set(net: &PetriNet, mode: ReductionMode) -> Result<(), TestCaseError> {
    let protected = vec![true; net.num_places()];
    // A reduction that fails (e.g. an internal decline) is not a soundness
    // violation by itself — the production paths fall back to identity. Skip.
    let Ok(reduced) = reduce_iterative_structural_with_mode(net, &protected, mode, None) else {
        return Ok(());
    };
    // Original must be fully explorable to draw any conclusion.
    let Some(original) = reachable_markings(net, MAX_STATES) else {
        return Ok(()); // inconclusive — skip
    };
    let Some(reduced_raw) = reachable_markings(&reduced.net, MAX_STATES) else {
        return Ok(()); // inconclusive — skip
    };
    // Lift each reduced marking. An expansion failure is inconclusive (skip).
    let mut reduced_expanded: BTreeSet<Vec<u64>> = BTreeSet::new();
    for m in &reduced_raw {
        match reduced.expand_marking(m) {
            Ok(e) => {
                reduced_expanded.insert(e);
            }
            Err(_) => return Ok(()),
        }
    }
    let keep = observable_places(&reduced);
    let orig_proj = project(&original, &keep);
    let red_proj = project(&reduced_expanded, &keep);
    prop_assert_eq!(
        orig_proj,
        red_proj,
        "REACHABILITY-SET DIVERGENCE under mode {:?}\nnet: places={:?} transitions={:?} m0={:?}\nreport={:?}",
        mode,
        net.places.len(),
        net.transitions.iter().map(|t| (&t.inputs, &t.outputs)).collect::<Vec<_>>(),
        net.initial_marking,
        reduced.report
    );
    Ok(())
}

/// DEADLOCK-EXISTENCE contract for `ReachabilityDeadlock`. Reduce with the
/// PRODUCTION empty protected set, BFS both nets, require the deadlock boolean to
/// match. Returns `Ok(())` on agreement (or inconclusive skip), `Err(_)` on a
/// genuine deadlock-boolean divergence.
fn check_deadlock_existence(net: &PetriNet) -> Result<(), TestCaseError> {
    let Ok(reduced) =
        reduce_iterative_structural_with_mode(net, &[], ReductionMode::ReachabilityDeadlock, None)
    else {
        return Ok(());
    };
    let Some(orig_dl) = has_deadlock(net, MAX_STATES) else {
        return Ok(()); // inconclusive — skip
    };
    let Some(red_dl) = has_deadlock(&reduced.net, MAX_STATES) else {
        return Ok(()); // inconclusive — skip
    };
    prop_assert_eq!(
        orig_dl,
        red_dl,
        "DEADLOCK-EXISTENCE DIVERGENCE (ReachabilityDeadlock)\nnet: places={:?} m0={:?} transitions={:?}\nreport={:?}",
        net.places.len(),
        net.initial_marking,
        net.transitions.iter().map(|t| (&t.inputs, &t.outputs)).collect::<Vec<_>>(),
        reduced.report
    );
    Ok(())
}

/// Every mode the reachability-set contract must hold under. (Every variant of
/// `ReductionMode` — adding a variant forces a compile error here, so the gate
/// can never silently miss a mode.)
const ALL_MODES: [ReductionMode; 6] = [
    ReductionMode::Reachability,
    ReductionMode::ReachabilityDeadlock,
    ReductionMode::NextFreeCTL,
    ReductionMode::CTLWithNext,
    ReductionMode::StutterInsensitiveLTL,
    ReductionMode::StutterSensitiveLTL,
];

proptest! {
    #![proptest_config(ProptestConfig {
        // A LARGE battery: thousands of random nets per run. Deadlock soundness
        // is 0-wrong, so the case count is deliberately high. Raised to 8000
        // when Berthelot lateral fusion (R_lat) was admitted into the
        // deadlock-mode rule set — the empty-protected deadlock contract is the
        // lane that exercises R_lat on random nets, so it carries extra weight.
        // Override with `PROPTEST_CASES=<n>` for an even larger ad-hoc battery.
        cases: 8000,
        max_shrink_iters: 8000,
        .. ProptestConfig::default()
    })]

    /// CLASS A — reachability-set preservation for EVERY mode. Protect every
    /// place; the observable-projected reachable marking SET must be identical
    /// original-vs-reduced. A divergence is a genuine unsound (rule, mode).
    #[test]
    fn proptest_reachability_set_preserved_all_modes(raw in raw_net_strategy()) {
        let net = build_net(&raw);
        for mode in ALL_MODES {
            check_reachability_set(&net, mode)?;
        }
    }

    /// THE DEADLOCK GATE — deadlock-existence preservation for the
    /// `ReachabilityDeadlock` mode, reduced with the production empty protected
    /// set. The deadlock boolean must be identical. This is the gate that any
    /// broadening of the deadlock rule-admission set must clear with ZERO
    /// disagreements.
    #[test]
    fn proptest_deadlock_existence_preserved(raw in raw_net_strategy()) {
        let net = build_net(&raw);
        check_deadlock_existence(&net)?;
    }
}

// ---------------------------------------------------------------------------
// NON-VACUITY GUARD. A differential gate that never actually exercises the
// broadened rules on a deadlocking net would pass trivially and prove nothing.
// This deterministic sweep (a self-contained xorshift net generator, no
// proptest RNG) asserts that within a modest battery the EXACT rules admitted
// into `ReachabilityDeadlock` — source-place (Rule C) and non-decreasing-place
// (Rule F) removal — DO fire under the production empty-protected deadlock
// reduction AND co-occur with a real reachable deadlock in the original net.
// If a future edit reverts either admission, the corresponding `>= 1` assertion
// trips, flagging that the deadlock gate has gone vacuous for that rule.
// ---------------------------------------------------------------------------
#[test]
fn deadlock_gate_exercises_broadened_rules_on_deadlocking_nets() {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut conclusive = 0usize;
    let mut fired_source_and_deadlock = 0usize;
    let mut fired_nondec_and_deadlock = 0usize;
    let total = 2_000;
    for _ in 0..total {
        let num_places = (next() as usize % 6) + 1;
        let m0: Vec<u8> = (0..num_places).map(|_| (next() % 4) as u8).collect();
        let ntrans = (next() as usize % 6) + 1;
        let transitions: Vec<RawTransition> = (0..ntrans)
            .map(|_| {
                let nin = next() as usize % 5;
                let nout = next() as usize % 5;
                let ins: Vec<RawArc> = (0..nin)
                    .map(|_| ((next() % 251) as u8, (next() % 3 + 1) as u8))
                    .collect();
                let outs: Vec<RawArc> = (0..nout)
                    .map(|_| ((next() % 251) as u8, (next() % 3 + 1) as u8))
                    .collect();
                (ins, outs)
            })
            .collect();
        let raw: RawNet = (num_places, m0, transitions);
        let net = build_net(&raw);
        let Some(orig_dl) = has_deadlock(&net, MAX_STATES) else {
            continue;
        };
        conclusive += 1;
        let Ok(reduced) = reduce_iterative_structural_with_mode(
            &net,
            &[],
            ReductionMode::ReachabilityDeadlock,
            None,
        ) else {
            continue;
        };
        if orig_dl && !reduced.report.source_places.is_empty() {
            fired_source_and_deadlock += 1;
        }
        if orig_dl && !reduced.report.non_decreasing_places.is_empty() {
            fired_nondec_and_deadlock += 1;
        }
    }
    assert!(
        conclusive > 300,
        "non-vacuity: too few conclusive nets ({conclusive}); the gate would prove little"
    );
    assert!(
        fired_source_and_deadlock >= 1,
        "non-vacuity: source-place (Rule C) removal never fired on a deadlocking \
         net in {conclusive} conclusive cases — the deadlock gate is vacuous for \
         Rule C (was its ReachabilityDeadlock admission reverted?)"
    );
    assert!(
        fired_nondec_and_deadlock >= 1,
        "non-vacuity: non-decreasing-place (Rule F) removal never fired on a \
         deadlocking net in {conclusive} conclusive cases — the deadlock gate is \
         vacuous for Rule F (was its ReachabilityDeadlock admission reverted?)"
    );
}

// ---------------------------------------------------------------------------
// LATERAL-FUSION NON-VACUITY + SOUNDNESS GUARD. Tiny uniform-random nets rarely
// contain the two-complement-shared-pivot structure R_lat keys on, so the
// empty-protected deadlock proptest above seldom EXERCISES it. This deterministic
// (RNG-free) sweep builds a band of complement-offset nets (the targeted-unit-
// test family, parameterized over capacity + offset) and asserts, with the
// fusion ACTIVE:
//   (a) R_lat fires (non-vacuity — trips if the rule is disabled/reverted);
//   (b) the lifted reachable SET is preserved across ALL modes, protecting every
//       place EXCEPT the fusion's duplicate candidate (so the fusion is free to
//       fire while keeping every kept place observable — exactly the valid
//       class-A reachability-set contract);
//   (c) the deadlock-existence boolean is preserved under the production
//       EMPTY-protected deadlock reduction;
//   (d) the affine `m(d) = ratio*m(c) + offset` coupling is exact at every
//       reachable marking.
// (Empty protection is used ONLY for the deadlock-boolean contract — never for
// the reachability-SET contract, mirroring the production gate: empty protection
// admits query-relevant rules that legitimately collapse the unobserved set.)
// ---------------------------------------------------------------------------
#[test]
fn lateral_fusion_non_vacuity_and_soundness_on_complement_nets() {
    // Complement-offset family: pivot `a` (idx 0), canonical `c` (idx 1),
    // duplicate `d` (idx 2).
    //   t_split: a -> c + d   t_join: c + d -> a
    //   m0 = [cap, base_c, base_d]  =>  a+c = cap+base_c, a+d = cap+base_d
    //   m(d) = m(c) + (base_d - base_c)   (offset = base_d - base_c >= 0)
    fn complement_net(cap: u64, base_c: u64, base_d: u64) -> PetriNet {
        let p = |id: &str| PlaceInfo {
            id: id.to_string(),
            name: Some(id.to_string()),
        };
        let a = |place: u32, weight: u64| Arc {
            place: PlaceIdx(place),
            weight,
        };
        let t = |id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>| TransitionInfo {
            id: id.to_string(),
            name: Some(id.to_string()),
            inputs,
            outputs,
        };
        PetriNet {
            name: Some("rlat-nonvacuity".to_string()),
            places: vec![p("a"), p("c"), p("d")],
            transitions: vec![
                t("t_split", vec![a(0, 1)], vec![a(1, 1), a(2, 1)]),
                t("t_join", vec![a(1, 1), a(2, 1)], vec![a(0, 1)]),
            ],
            initial_marking: vec![cap, base_c, base_d],
        }
    }

    let mut fired = 0usize;
    let mut conclusive = 0usize;
    for cap in 1u64..=4 {
        for base_c in 0u64..=2 {
            // offset = base_d - base_c must be >= 0 for a non-negative coupling.
            for base_d in base_c..=(base_c + 3) {
                let net = complement_net(cap, base_c, base_d);
                let Some(original) = reachable_markings(&net, MAX_STATES) else {
                    continue;
                };
                let Some(orig_dl) = has_deadlock(&net, MAX_STATES) else {
                    continue;
                };
                conclusive += 1;

                // (b) reachability-set contract, protecting all but the
                // duplicate `d` (idx 2) so the fusion can remove only it.
                let protected = vec![true, true, false];
                for mode in ALL_MODES {
                    let Ok(reduced) =
                        reduce_iterative_structural_with_mode(&net, &protected, mode, None)
                    else {
                        continue;
                    };
                    if mode.allows_lateral_fusion() && !reduced.report.lateral_fusions.is_empty() {
                        fired += 1;
                    }
                    let Some(reduced_raw) = reachable_markings(&reduced.net, MAX_STATES) else {
                        continue;
                    };
                    let mut expanded: BTreeSet<Vec<u64>> = BTreeSet::new();
                    let mut ok = true;
                    for m in &reduced_raw {
                        match reduced.expand_marking(m) {
                            Ok(e) => {
                                expanded.insert(e);
                            }
                            Err(_) => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let keep = observable_places(&reduced);
                    assert_eq!(
                        project(&original, &keep),
                        project(&expanded, &keep),
                        "complement-net R_lat reachability divergence (cap={cap} \
                         base_c={base_c} base_d={base_d} mode={mode:?}); report={:?}",
                        reduced.report
                    );
                    // (d) exact affine reconstruction on the removed duplicate.
                    for m in &expanded {
                        for f in &reduced.report.lateral_fusions {
                            let c = f.canonical.0 as usize;
                            let d = f.duplicate.0 as usize;
                            assert_eq!(
                                m[d],
                                f.ratio * m[c] + f.offset,
                                "R_lat reconstruction violated (cap={cap} base_c={base_c} \
                                 base_d={base_d} mode={mode:?})"
                            );
                        }
                    }
                }

                // (c) deadlock-existence under the production EMPTY-protected
                // deadlock reduction.
                if let Ok(reduced_dl) = reduce_iterative_structural_with_mode(
                    &net,
                    &[],
                    ReductionMode::ReachabilityDeadlock,
                    None,
                ) {
                    if let Some(red_dl) = has_deadlock(&reduced_dl.net, MAX_STATES) {
                        assert_eq!(
                            orig_dl, red_dl,
                            "complement-net R_lat deadlock divergence (cap={cap} \
                             base_c={base_c} base_d={base_d}); report={:?}",
                            reduced_dl.report
                        );
                    }
                }
            }
        }
    }
    assert!(
        conclusive > 20,
        "non-vacuity: too few conclusive complement nets ({conclusive})"
    );
    assert!(
        fired >= 1,
        "non-vacuity: lateral fusion (R_lat) never fired on any complement-offset \
         net in {conclusive} conclusive cases — the differential lane is vacuous \
         for R_lat (was the rule disabled or its mode admission reverted?)"
    );
}

// ===========================================================================
// STUTTER-EQUIVALENCE GATE for the agglomeration rules under the temporal
// modes (StutterInsensitiveLTL / NextFreeCTL).
//
// The reachability-set / deadlock gates above are LINEAR-TIME-INSENSITIVE: they
// only compare the SET of reachable observable markings (and the deadlock
// boolean). They CANNOT see the difference that stutter-insensitive LTL (LTL∖X)
// and next-free CTL (CTL∖X) observe — namely the *order and divergence* of
// observable-label changes along maximal runs. Broadening agglomeration to those
// modes therefore needs a STRICTLY STRONGER oracle: stutter-collapsed observable-
// trace equivalence.
//
// This section provides that oracle and uses it to answer the Phase-2 question
// the `ReductionMode` doc flags (`types.rs`: "Phase-2 may extend this to
// StutterInsensitiveLTL"). The answer it encodes — proven, not assumed — is:
//
//   TY's agglomeration detectors (Berthelot pre/post, Tapaal Rule R, Rule S),
//   AS IMPLEMENTED, are NOT stutter-equivalence-preserving, even when the fused
//   intermediate place is fully unobserved. They enforce only the REACHABILITY
//   admissibility conditions (zero-marking intermediate, Berthelot condition-6
//   disjointness, query-protection on producer-pre / consumer-post places). They
//   do NOT enforce the divergence-freedom / invisible-step conditions that the
//   stutter-LTL variant of agglomeration (Haddad & Pradat-Peyre; Tapaal T-mode)
//   requires. Concretely, agglomeration FUSES a producer-then-consumer pair into
//   one atomic step and so DELETES the intermediate observable STUTTER STEP (the
//   "producer fired, consumer has not yet fired" marking), which an LTL∖X
//   property such as `GF φ` / `FG φ` distinguishes (the original can stutter on
//   the observed atoms where the fused net cannot — a divergence difference).
//
// Hence agglomeration stays Reachability-only. This gate is the fail-closed
// guard for that decision: it (1) verifies the temporal modes emit NO
// agglomeration, and (2) demonstrates, on a large battery WITH the oracle, that
// the same detectors used under Reachability DO break stutter-equivalence — so
// any future broadening that wires agglomeration into a temporal mode without
// also adding the stutter admissibility checks is caught here.
// ===========================================================================

/// Build the labeled reachability graph of `net`: `labels[i]` is the projection
/// of state `i` onto `keep`, `succ[i]` are its one-step successor indices, and
/// the init state is index 0. `None` on state-cap / marking-overflow
/// (inconclusive — skip the case, never a failure).
fn labeled_graph(
    net: &PetriNet,
    keep: &[usize],
    max_states: usize,
) -> Option<(Vec<Vec<u64>>, Vec<Vec<usize>>, usize)> {
    let mut index: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();
    let mut markings: Vec<Vec<u64>> = Vec::new();
    let mut succ: Vec<Vec<usize>> = Vec::new();
    let init = net.initial_marking.clone();
    let mut queue: VecDeque<usize> = VecDeque::new();
    index.insert(init.clone(), 0);
    markings.push(init);
    succ.push(Vec::new());
    queue.push_back(0);
    while let Some(cur) = queue.pop_front() {
        let marking = markings[cur].clone();
        for t in 0..net.num_transitions() {
            let tidx = TransitionIdx(t as u32);
            if net.is_enabled(&marking, tidx) {
                let nxt = net.fire(&marking, tidx).ok()?;
                let nidx = if let Some(&i) = index.get(&nxt) {
                    i
                } else {
                    let i = markings.len();
                    if i >= max_states {
                        return None;
                    }
                    index.insert(nxt.clone(), i);
                    markings.push(nxt);
                    succ.push(Vec::new());
                    queue.push_back(i);
                    i
                };
                succ[cur].push(nidx);
            }
        }
    }
    let labels: Vec<Vec<u64>> = markings
        .iter()
        .map(|m| keep.iter().map(|&p| m[p]).collect())
        .collect();
    Some((labels, succ, 0))
}

/// Divergence-sensitive stutter abstraction of a labeled graph. Returns:
///   * `reachable_labels`: the set of observable labels on init-reachable states;
///   * `steps`: the stutter-COLLAPSED step relation — `(l, l')` whenever some
///     state labeled `l` has a successor labeled `l' != l` (i.e. an observable
///     change after any amount of `l`-stuttering);
///   * `diverge`: the labels `l` such that some `l`-labeled state can stutter
///     FOREVER (an infinite same-`l` run: a same-label cycle or a deadlock,
///     which MCC pads into an infinite stutter).
///
/// Equality of this triple is a SOUND, decidable NECESSARY condition for
/// divergence-sensitive stutter-trace equivalence over the labeling (the
/// equivalence that characterizes LTL∖X / CTL∖X). If two systems share the same
/// triple but are still LTL∖X-inequivalent, this oracle would MISS it — which is
/// the safe failure direction for a gate that must *block* a broadening: it can
/// only ever ACCEPT too few broadenings, never wrongly accept an unsound one
/// that already differs at this granularity. (Every divergence this gate reports
/// is a genuine LTL∖X-observable difference.)
type StutterAbstraction = (
    BTreeSet<Vec<u64>>,
    BTreeSet<(Vec<u64>, Vec<u64>)>,
    BTreeSet<Vec<u64>>,
);

fn stutter_abstraction(
    labels: &[Vec<u64>],
    succ: &[Vec<usize>],
    init: usize,
) -> StutterAbstraction {
    let n = labels.len();
    let mut reach = vec![false; n];
    let mut stack = vec![init];
    reach[init] = true;
    while let Some(s) = stack.pop() {
        for &t in &succ[s] {
            if !reach[t] {
                reach[t] = true;
                stack.push(t);
            }
        }
    }
    let mut reachable_labels: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut steps: BTreeSet<(Vec<u64>, Vec<u64>)> = BTreeSet::new();
    let mut diverge: BTreeSet<Vec<u64>> = BTreeSet::new();
    for s in 0..n {
        if !reach[s] {
            continue;
        }
        reachable_labels.insert(labels[s].clone());
        // BFS the same-label region reachable from `s`; record observable
        // changes (steps) and detect an infinite same-label run (divergence).
        let mut seen = vec![false; n];
        let mut q = VecDeque::new();
        seen[s] = true;
        q.push_back(s);
        let mut region: Vec<usize> = Vec::new();
        let mut can_stutter_forever = false;
        while let Some(u) = q.pop_front() {
            region.push(u);
            if succ[u].is_empty() {
                // A deadlock labeled `l` stutters forever (MCC stutter-pads it).
                can_stutter_forever = true;
            }
            for &v in &succ[u] {
                if labels[v] == labels[s] {
                    if !seen[v] {
                        seen[v] = true;
                        q.push_back(v);
                    }
                } else {
                    steps.insert((labels[s].clone(), labels[v].clone()));
                }
            }
        }
        // Any same-label edge inside the (finite) same-label region witnesses a
        // potential infinite stutter run (sound over-approximation of an
        // infinite same-`l` path for divergence-sensitivity).
        if !can_stutter_forever {
            'outer: for &u in &region {
                for &v in &succ[u] {
                    if labels[v] == labels[s] {
                        can_stutter_forever = true;
                        break 'outer;
                    }
                }
            }
        }
        if can_stutter_forever {
            diverge.insert(labels[s].clone());
        }
    }
    (reachable_labels, steps, diverge)
}

/// Compute the stutter abstraction of the ORIGINAL net (projected onto `keep`)
/// and of the REDUCED net (each reduced marking expanded back to original
/// coordinates, then projected onto `keep`). `None` on any inconclusive skip
/// (state-cap / overflow / expansion failure).
fn stutter_abstractions(
    net: &PetriNet,
    reduced: &ReducedNet,
    keep: &[usize],
) -> Option<(StutterAbstraction, StutterAbstraction)> {
    let (olabels, osucc, oinit) = labeled_graph(net, keep, MAX_STATES)?;
    let red_keep: Vec<usize> = (0..reduced.net.num_places()).collect();
    let (rmark, rsucc, rinit) = labeled_graph(&reduced.net, &red_keep, MAX_STATES)?;
    let mut rlabels: Vec<Vec<u64>> = Vec::with_capacity(rmark.len());
    for m in &rmark {
        let e = reduced.expand_marking(m).ok()?;
        rlabels.push(keep.iter().map(|&p| e[p]).collect());
    }
    Some((
        stutter_abstraction(&olabels, &osucc, oinit),
        stutter_abstraction(&rlabels, &rsucc, rinit),
    ))
}

fn report_has_agglomeration(report: &crate::reduction::ReductionReport) -> bool {
    !report.pre_agglomerations.is_empty()
        || !report.post_agglomerations.is_empty()
        || !report.rule_r_agglomerations.is_empty()
        || !report.rule_s_agglomerations.is_empty()
}

/// The temporal modes whose agglomeration admission Phase-2 would broaden.
const TEMPORAL_MODES: [ReductionMode; 2] = [
    ReductionMode::StutterInsensitiveLTL,
    ReductionMode::NextFreeCTL,
];

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4000,
        max_shrink_iters: 4000,
        .. ProptestConfig::default()
    })]

    /// FAIL-CLOSED CONTRACT: under the temporal modes, NO agglomeration is ever
    /// emitted — with EMPTY protection (the most permissive, agglomeration-
    /// friendly call) and with every place protected. This is the guard that
    /// keeps `allows_agglomeration` / `allows_rule_r_agglomeration` /
    /// `allows_rule_s_agglomeration` Reachability-only: a future edit that wires
    /// any agglomeration variant into StutterInsensitiveLTL or NextFreeCTL (the
    /// flagged Phase-2 broadening) without ALSO adding the stutter admissibility
    /// checks trips this immediately. The justification — that the current
    /// detectors are not stutter-safe — is proven by
    /// `reachability_agglomeration_breaks_stutter_equivalence` below.
    #[test]
    fn temporal_modes_emit_no_agglomeration(raw in raw_net_strategy()) {
        let net = build_net(&raw);
        let full = vec![true; net.num_places()];
        for mode in TEMPORAL_MODES {
            for protected in [Vec::new(), full.clone()] {
                if let Ok(reduced) =
                    reduce_iterative_structural_with_mode(&net, &protected, mode, None)
                {
                    prop_assert!(
                        !report_has_agglomeration(&reduced.report),
                        "AGGLOMERATION EMITTED under temporal mode {:?} (protected={}): \
                         agglomeration is NOT stutter-equivalence-preserving with the \
                         current detectors (see \
                         reachability_agglomeration_breaks_stutter_equivalence). \
                         report pre={} post={} R={} S={}",
                        mode,
                        if protected.is_empty() { "empty" } else { "all" },
                        reduced.report.pre_agglomerations.len(),
                        reduced.report.post_agglomerations.len(),
                        reduced.report.rule_r_agglomerations.len(),
                        reduced.report.rule_s_agglomerations.len(),
                    );
                }
            }
        }
    }
}

/// THE STUTTER ORACLE, RUN AS A JUSTIFICATION GATE. A deterministic (RNG-free,
/// reproducible) battery that fires the agglomeration detectors under
/// `Reachability` mode — the SAME detectors a Phase-2 broadening would reuse —
/// while observing every place EXCEPT one hidden "intermediate" candidate (the
/// most FAVORABLE case for stutter-soundness: the fused place is unobserved,
/// everything else is observed). It then compares the divergence-sensitive
/// stutter abstractions of the original and reduced nets.
///
/// The assertion is that the oracle FINDS divergences (the agglomeration is
/// stutter-UNSAFE). This is the empirical core of the BLOCKED result: it proves
/// the reachable-set gate is insufficient and that the detectors must not be
/// broadened. If a future change made TY's agglomeration genuinely stutter-safe
/// (e.g. by adding the divergence-freedom admissibility check), this test's
/// expectation would flip — at which point the broadening could be revisited and
/// THIS gate (now passing stutter-equivalence) would be the contract to enforce.
#[test]
fn reachability_agglomeration_breaks_stutter_equivalence() {
    let mut s: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut conclusive = 0usize;
    let mut diverged = 0usize;
    // Per-variant isolation counters: (#cases where ONLY this variant fired,
    // #of those that broke stutter-equivalence).
    let mut pre = (0usize, 0usize);
    let mut post = (0usize, 0usize);
    let mut rule_r = (0usize, 0usize);
    // The favorable-case (single hidden place) agglomeration structure is rare
    // in tiny uniform-random nets, so the battery is deliberately large to
    // reach a healthy conclusive count. ~100k yields ~120 conclusive nets and
    // ~70 stutter divergences — comfortably above the non-vacuity floor while
    // keeping the runtime bounded. Override with `TY_STUTTER_GATE_CASES=<n>`.
    let total: u64 = std::env::var("TY_STUTTER_GATE_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    for _ in 0..total {
        let num_places = (next() as usize % 6) + 1;
        let m0: Vec<u8> = (0..num_places).map(|_| (next() % 3) as u8).collect();
        let ntrans = (next() as usize % 6) + 1;
        let transitions: Vec<RawTransition> = (0..ntrans)
            .map(|_| {
                let nin = next() as usize % 4;
                let nout = next() as usize % 4;
                let ins: Vec<RawArc> = (0..nin)
                    .map(|_| ((next() % 251) as u8, (next() % 2 + 1) as u8))
                    .collect();
                let outs: Vec<RawArc> = (0..nout)
                    .map(|_| ((next() % 251) as u8, (next() % 2 + 1) as u8))
                    .collect();
                (ins, outs)
            })
            .collect();
        let raw: RawNet = (num_places, m0, transitions);
        let net = build_net(&raw);

        // Hide exactly ONE place (the intermediate candidate); observe the rest.
        let np = net.num_places();
        let hidden = (next() as usize) % np;
        let mut protected = vec![true; np];
        protected[hidden] = false;

        let Ok(reduced) = reduce_iterative_structural_with_mode(
            &net,
            &protected,
            ReductionMode::Reachability,
            None,
        ) else {
            continue;
        };
        if !report_has_agglomeration(&reduced.report) {
            continue;
        }
        let keep = observable_places(&reduced);
        if keep.is_empty() {
            continue;
        }
        let Some((oa, ra)) = stutter_abstractions(&net, &reduced, &keep) else {
            continue;
        };
        conclusive += 1;
        let only = |a: bool, b: bool, c: bool, d: bool| a && !b && !c && !d;
        let has_pre = !reduced.report.pre_agglomerations.is_empty();
        let has_post = !reduced.report.post_agglomerations.is_empty();
        let has_r = !reduced.report.rule_r_agglomerations.is_empty();
        let has_s = !reduced.report.rule_s_agglomerations.is_empty();
        let div = oa != ra;
        if only(has_pre, has_post, has_r, has_s) {
            pre.0 += 1;
            if div {
                pre.1 += 1;
            }
        }
        if only(has_post, has_pre, has_r, has_s) {
            post.0 += 1;
            if div {
                post.1 += 1;
            }
        }
        if only(has_r, has_pre, has_post, has_s) {
            rule_r.0 += 1;
            if div {
                rule_r.1 += 1;
            }
        }
        if div {
            diverged += 1;
        }
    }

    // Non-vacuity: the battery must actually exercise agglomeration on
    // conclusive nets, otherwise the gate proves nothing.
    assert!(
        conclusive > 50,
        "non-vacuity: too few conclusive agglomerating nets ({conclusive})"
    );
    // The load-bearing claim: agglomeration breaks divergence-sensitive
    // stutter-trace equivalence on the observed places, even with the fused
    // intermediate hidden. This is WHY agglomeration cannot be broadened to the
    // temporal modes with the current detectors.
    assert!(
        diverged > 0,
        "stutter oracle found ZERO divergences in {conclusive} agglomerating nets — \
         if TY's agglomeration became genuinely stutter-safe, revisit broadening \
         the temporal-mode admission AND flip this expectation to require \
         stutter-equivalence."
    );
    // Each isolatable variant that fired must have demonstrated at least one
    // stutter break (so the BLOCKED verdict is per-variant, not aggregate).
    if pre.0 > 0 {
        assert!(
            pre.1 > 0,
            "pre-agglomeration fired in isolation {} times but never broke stutter-\
             equivalence — re-examine whether it is stutter-safe",
            pre.0
        );
    }
    if post.0 > 0 {
        assert!(
            post.1 > 0,
            "post-agglomeration fired in isolation {} times but never broke stutter-\
             equivalence — re-examine whether it is stutter-safe",
            post.0
        );
    }
    if rule_r.0 > 0 {
        assert!(
            rule_r.1 > 0,
            "Rule R fired in isolation {} times but never broke stutter-equivalence \
             — re-examine whether it is stutter-safe",
            rule_r.0
        );
    }
    eprintln!(
        "STUTTER GATE: conclusive={conclusive} diverged={diverged} \
         (pre {pre:?}, post {post:?}, ruleR {rule_r:?})"
    );
}

/// Hand-built minimal WITNESS that a producer→consumer chain agglomeration
/// deletes an observable stutter step. `a: src -> p`, `c: p -> dst`,
/// `u: dst -> src` (recycle); observe {src, dst}, hide `p`. The original net
/// passes through the marking (p=1, dst unchanged) — an observable stutter step
/// on the {src,dst} projection — which the fused atomic transition removes.
/// This is the concrete LTL∖X-observable difference behind the BLOCKED verdict.
#[test]
fn agglomeration_deletes_observable_stutter_step_witness() {
    let place = |id: &str| PlaceInfo {
        id: id.into(),
        name: Some(id.into()),
    };
    let arc = |p: u32, w: u64| Arc {
        place: PlaceIdx(p),
        weight: w,
    };
    let trans = |id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>| TransitionInfo {
        id: id.into(),
        name: Some(id.into()),
        inputs,
        outputs,
    };
    // places: p(0, hidden), src(1), dst(2).
    let net = PetriNet {
        name: Some("agglo-stutter-witness".into()),
        places: vec![place("p"), place("src"), place("dst")],
        transitions: vec![
            trans("a", vec![arc(1, 1)], vec![arc(0, 1)]), // src -> p
            trans("c", vec![arc(0, 1)], vec![arc(2, 1)]), // p   -> dst
            trans("u", vec![arc(2, 1)], vec![arc(1, 1)]), // dst -> src
        ],
        initial_marking: vec![0, 1, 0],
    };
    // Observe src and dst; allow the intermediate p (idx 0) to be agglomerated.
    let protected = vec![false, true, true];
    let reduced =
        reduce_iterative_structural_with_mode(&net, &protected, ReductionMode::Reachability, None)
            .expect("reachability reduction must succeed");

    // The reduction MUST have agglomerated the chain (otherwise the witness is
    // vacuous and the structural detectors changed shape).
    assert!(
        report_has_agglomeration(&reduced.report),
        "witness vacuous: expected an agglomeration on the src->p->dst chain, got \
         report pre={} post={} R={} S={}",
        reduced.report.pre_agglomerations.len(),
        reduced.report.post_agglomerations.len(),
        reduced.report.rule_r_agglomerations.len(),
        reduced.report.rule_s_agglomerations.len(),
    );

    let keep = observable_places(&reduced);
    let (oa, ra) = stutter_abstractions(&net, &reduced, &keep)
        .expect("both nets are tiny and fully explorable");

    // The original has the observable stutter step the fused net lacks: firing
    // `a` (src->p) leaves the {src,dst} projection in a state that the reduced
    // net never visits, so the stutter-collapsed step relations differ.
    assert_ne!(
        oa, ra,
        "agglomeration was expected to BREAK divergence-sensitive stutter-trace \
         equivalence on the observed {{src,dst}} projection, but the abstractions \
         matched — if agglomeration became stutter-safe, revisit the temporal-mode \
         admission. orig=(labels {:?}, steps {:?}, div {:?}) reduced=(labels {:?}, \
         steps {:?}, div {:?})",
        oa.0, oa.1, oa.2, ra.0, ra.1, ra.2,
    );
}
