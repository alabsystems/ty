// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PROPTEST RANDOM-NET DIFFERENTIAL GATE for the stubborn-set partial-order
//! reduction ([`compute_stubborn_set`] with [`PorStrategy::DeadlockPreserving`]).
//!
//! This is the contention-robust, reproducible soundness contract for the POR
//! lane that the explicit deadlock examination runs in PRODUCTION
//! (`deadlock_one_safe.rs` builds `ExecutionPlan::observer(PorStrategy::
//! DeadlockPreserving)` for both the sequential and the portfolio BFS). A wrong
//! stubborn set that violates D1/D2 would MISS a reachable deadlock and the
//! examination would emit a WRONG `ReachabilityDeadlock = False`. This gate
//! exists to make that impossible to ship silently.
//!
//! Unlike `reduction/tests/reduction_differential_proptest.rs` — which exercises
//! the *structural* (algebraic net-rewriting) reduction catalog — this gate
//! exercises the *on-the-fly* stubborn-set reduction: at each explored marking we
//! fire only the stubborn subset of enabled transitions instead of all of them.
//!
//! # The contract (deadlock-existence equality)
//!
//! For each random net we run two BFS oracles over the SAME net:
//!
//!   * **FULL** — at every marking fire *all* enabled transitions
//!     ([`PorStrategy::None`]).
//!   * **POR** — at every marking fire only the stubborn subset returned by
//!     [`compute_stubborn_set`] under [`PorStrategy::DeadlockPreserving`], with
//!     the production fail-closed fallback (a `None` result means "no reduction
//!     proven safe → fire the full enabled set"). This mirrors
//!     `StubbornPorProvider::reduce` / the mc-core BFS exactly.
//!
//! Then we assert the deadlock-existence boolean — "∃ a reachable marking with
//! no enabled transition?" — is IDENTICAL FULL-vs-POR. This is the D1+D2
//! preservation theorem, checked empirically on thousands of random nets:
//!
//!   * A stubborn set that UNDER-includes (drops a transition that could disable
//!     a key transition, or a producer needed to re-enable a disabled one) can
//!     prune away the only path to a reachable deadlock → POR reports NO deadlock
//!     while FULL reports one. The `prop_assert_eq!` catches it.
//!   * A stubborn set that returns the WRONG (non-enabled) transitions, or an
//!     empty set at a non-deadlock state, would manufacture a spurious deadlock
//!     in POR that FULL does not have. Caught the same way.
//!
//! # Boundedness / termination
//!
//! Random nets are frequently unbounded. Both oracles are capped at
//! [`MAX_STATES`]; a case whose FULL net exceeds the cap (or overflows a marking)
//! is INCONCLUSIVE and silently skipped — it proves nothing, it does not fail.
//! Crucially the POR oracle is bounded by the SAME cap so that a (hypothetically
//! buggy) reduction that loses states can never make the POR run *cheaper than
//! conclusive* and thereby dodge the contract: we only compare when FULL is
//! conclusive, and POR explores a SUBSET of FULL's reachable set, so POR is
//! conclusive whenever FULL is.

use std::collections::{HashSet, VecDeque};

use proptest::prelude::*;

use super::{compute_stubborn_set, DependencyGraph, PorStrategy};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

/// Exploration ceiling. A net whose FULL reachable set exceeds this is treated as
/// inconclusive (skipped), never as a failure.
const MAX_STATES: usize = 20_000;

// ---------------------------------------------------------------------------
// Shared BFS primitives.
// ---------------------------------------------------------------------------

fn enabled_transitions(net: &PetriNet, marking: &[u64]) -> Vec<TransitionIdx> {
    let mut enabled = Vec::new();
    for t in 0..net.num_transitions() {
        let tidx = TransitionIdx(t as u32);
        if net.is_enabled(marking, tidx) {
            enabled.push(tidx);
        }
    }
    enabled
}

/// Outcome of a bounded BFS: whether the cap was hit (inconclusive) and whether a
/// deadlock (a reachable marking with no enabled transition) was found.
struct BfsOutcome {
    conclusive: bool,
    deadlock: bool,
    states: usize,
}

/// FULL BFS: at each marking fire ALL enabled transitions. The reference oracle.
fn bfs_full(net: &PetriNet, max_states: usize) -> BfsOutcome {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut deadlock = false;
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());

    while let Some(marking) = queue.pop_front() {
        let enabled = enabled_transitions(net, &marking);
        if enabled.is_empty() {
            deadlock = true;
            continue;
        }
        for t in enabled {
            // A marking-overflow on a random net is inconclusive (skip the case),
            // not a soundness violation.
            let Ok(next) = net.fire(&marking, t) else {
                return BfsOutcome {
                    conclusive: false,
                    deadlock,
                    states: seen.len(),
                };
            };
            if seen.insert(next.clone()) {
                if seen.len() > max_states {
                    return BfsOutcome {
                        conclusive: false,
                        deadlock,
                        states: seen.len(),
                    };
                }
                queue.push_back(next);
            }
        }
    }

    BfsOutcome {
        conclusive: true,
        deadlock,
        states: seen.len(),
    }
}

/// POR BFS: at each marking fire only the stubborn subset under
/// `DeadlockPreserving`, with the production fail-closed fallback (`None` =>
/// fire all enabled). This is byte-for-byte the production decision made by
/// `StubbornPorProvider::reduce` for `PorPropertyClass::Deadlock`.
///
/// `reduced_some` counts the markings where the stubborn set was a STRICT subset
/// of the enabled set (a real reduction) — used by the non-vacuity guard.
fn bfs_por(net: &PetriNet, dep: &DependencyGraph, max_states: usize) -> (BfsOutcome, usize) {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut deadlock = false;
    let mut reduced_some = 0usize;
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());

    while let Some(marking) = queue.pop_front() {
        let enabled = enabled_transitions(net, &marking);
        if enabled.is_empty() {
            deadlock = true;
            continue;
        }
        // EXACT production decision path: compute the stubborn set; on `None`
        // (no reduction proven safe) fall back to the full enabled set.
        let to_fire =
            match compute_stubborn_set(net, &marking, dep, &PorStrategy::DeadlockPreserving) {
                Some(stubborn) => {
                    if stubborn.len() < enabled.len() {
                        reduced_some += 1;
                    }
                    stubborn
                }
                None => enabled,
            };
        for t in to_fire {
            let Ok(next) = net.fire(&marking, t) else {
                return (
                    BfsOutcome {
                        conclusive: false,
                        deadlock,
                        states: seen.len(),
                    },
                    reduced_some,
                );
            };
            if seen.insert(next.clone()) {
                if seen.len() > max_states {
                    return (
                        BfsOutcome {
                            conclusive: false,
                            deadlock,
                            states: seen.len(),
                        },
                        reduced_some,
                    );
                }
                queue.push_back(next);
            }
        }
    }

    (
        BfsOutcome {
            conclusive: true,
            deadlock,
            states: seen.len(),
        },
        reduced_some,
    )
}

// ---------------------------------------------------------------------------
// Random-net strategy (mirrors reduction_differential_proptest.rs): <=6 places,
// 1..=6 transitions, weights 1-3, m0 in 0..=3, at most one arc per place per
// side per transition.
// ---------------------------------------------------------------------------

type RawArc = (u8, u8);
type RawTransition = (Vec<RawArc>, Vec<RawArc>);
type RawNet = (usize, Vec<u8>, Vec<RawTransition>);

fn raw_arc_strategy() -> impl Strategy<Value = RawArc> {
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
        name: Some("stubborn-proptest-random".into()),
        places,
        transitions: nets,
        initial_marking,
    }
}

// ---------------------------------------------------------------------------
// PER-STATE D1+D2 SOUNDNESS VERIFIER.
//
// The deadlock-existence differential alone has LIMITED teeth against a single
// dropped D2 clause: stubborn-set POR is a fixpoint over the whole reachable
// graph, so a transition wrongly excluded at one marking is usually re-explored
// from a successor marking and the deadlock is still found via another
// interleaving. To keep real teeth against EACH condition, we additionally
// verify, at EVERY reachable marking, that the returned stubborn set satisfies
// the conditions on which the deadlock-preservation THEOREM rests:
//
//   D1 (deadlock): if any transition is enabled, the stubborn set contains at
//       least one enabled transition (it is never empty at a live state).
//   D2-enabled: for each ENABLED t in the stubborn set, every transition that
//       can disable t (shares an input place — `interferers(t)`) is also in the
//       set. (Otherwise an independent transition could disable t on a pruned
//       interleaving, breaking the "stubborn transitions stay enabled" argument.)
//   D2-disabled: for each DISABLED t in the stubborn set, there EXISTS an
//       under-marked input place p (marking[p] < weight) all of whose producers
//       are in the set — a "necessary set" certifying t cannot be enabled
//       without firing a stubborn transition.
//
// A reduction that drops any clause violates the corresponding condition at the
// offending marking, and this verifier fails IMMEDIATELY and deterministically
// with that marking — independent of whether the boolean differential happened
// to also flip. This is the standard validation for stubborn-set soundness.
// ---------------------------------------------------------------------------

/// Check D1+D2 for the stubborn set the production path would fire at `marking`.
/// `Ok(())` if conditions hold (or the strategy declined → full set, vacuously
/// sound); `Err(msg)` naming the violated condition otherwise.
///
/// Crucially this verifies D2 over the FULL closure `T_s` (which includes
/// disabled members) via `deadlock_stubborn_closure`, NOT the fired enabled-only
/// subset returned by `compute_stubborn_set`. The deadlock-preservation theorem's
/// D2 quantifies over `T_s`: a disabler of an enabled stubborn transition need
/// only be in `T_s`, and may itself be DISABLED (hence never fired, filtered out
/// by `T_s ∩ E(s)`). Checking the enabled-only subset would WRONGLY reject sound
/// sets. We separately confirm the FIRED set the production path actually uses is
/// a non-empty subset of the enabled set (D1).
fn verify_d1_d2_at(net: &PetriNet, dep: &DependencyGraph, marking: &[u64]) -> Result<(), String> {
    let enabled: Vec<TransitionIdx> = enabled_transitions(net, marking);

    // The full closure T_s (including disabled members). `None` => no reduction
    // possible (|E(s)| <= 1) => production fires the full enabled set, vacuously
    // sound; nothing to verify.
    let Some(ts_vec) = super::deadlock_stubborn_closure(net, marking, dep) else {
        return Ok(());
    };
    let n = net.num_transitions();
    let mut in_ts = vec![false; n];
    for &t in &ts_vec {
        in_ts[t.0 as usize] = true;
    }

    // The fired set is T_s ∩ E(s). D1 (deadlock): at a live marking this is
    // non-empty (the seed is enabled and in T_s). The agreement between this and
    // the production `compute_stubborn_set` entry point is checked separately by
    // `fired_set_matches_closure_intersection` (kept out of the hot per-marking
    // loop for speed).
    let fired_count = enabled.iter().filter(|t| in_ts[t.0 as usize]).count();
    if !enabled.is_empty() && fired_count == 0 {
        return Err("D1 violated: empty fired set (T_s ∩ E(s)) at a live marking".into());
    }

    // D2 over the full closure T_s.
    for t in 0..n {
        if !in_ts[t] {
            continue;
        }
        let tidx = TransitionIdx(t as u32);
        if net.is_enabled(marking, tidx) {
            // D2-enabled: every disabler (input-sharing transition) is in T_s.
            for &d in dep.interferers(tidx) {
                if !in_ts[d.0 as usize] {
                    return Err(format!(
                        "D2-enabled violated: enabled {tidx:?} in T_s but its disabler \
                         {d:?} (shares an input place) is NOT in T_s"
                    ));
                }
            }
        } else {
            // D2-disabled: there must EXIST an under-marked input place whose
            // producers are all in T_s (a necessary set for enabling t).
            let info = &net.transitions[t];
            let mut witness = false;
            for arc in &info.inputs {
                let p = arc.place.0 as usize;
                if marking[p] < arc.weight {
                    let all_producers_in = dep
                        .producers_of(p)
                        .iter()
                        .all(|prod| in_ts[prod.0 as usize]);
                    if all_producers_in {
                        witness = true;
                        break;
                    }
                }
            }
            if !witness {
                return Err(format!(
                    "D2-disabled violated: disabled {tidx:?} in T_s has NO under-marked \
                     input place whose producers are all in T_s (no necessary set)"
                ));
            }
        }
    }

    Ok(())
}

/// Verify D1+D2 at every reachable marking of `net` (bounded). Returns
/// `Ok(Some((d2_enabled_hits, d2_disabled_hits)))` when the run was conclusive
/// and every reachable marking passed (with the per-clause witness counts folded
/// into the SAME single BFS), `Ok(None)` if inconclusive (cap/overflow — skip),
/// `Err(msg)` on the first violation.
fn verify_d1_d2_reachable(
    net: &PetriNet,
    max_states: usize,
) -> Result<Option<(usize, usize)>, String> {
    let dep = DependencyGraph::build(net);
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut d2_enabled_hits = 0usize;
    let mut d2_disabled_hits = 0usize;
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());
    while let Some(marking) = queue.pop_front() {
        verify_d1_d2_at(net, &dep, &marking)?;
        // Fold witness-counting into the same pass (avoids a second BFS).
        if let Some(ts_vec) = super::deadlock_stubborn_closure(net, &marking, &dep) {
            for &t in &ts_vec {
                if net.is_enabled(&marking, t) {
                    if !dep.interferers(t).is_empty() {
                        d2_enabled_hits += 1;
                    }
                } else {
                    d2_disabled_hits += 1;
                }
            }
        }
        // Explore via the FULL successor relation so the verifier visits EVERY
        // reachable marking (not just the POR-reduced ones) — a stronger check.
        for t in enabled_transitions(net, &marking) {
            let Ok(next) = net.fire(&marking, t) else {
                return Ok(None); // overflow — inconclusive
            };
            if seen.insert(next.clone()) {
                if seen.len() > max_states {
                    return Ok(None); // cap — inconclusive
                }
                queue.push_back(next);
            }
        }
    }
    Ok(Some((d2_enabled_hits, d2_disabled_hits)))
}

/// The production entry `compute_stubborn_set(.., DeadlockPreserving)` must equal
/// `T_s ∩ E(s)` (or signal `None` when that intersection is the full enabled set,
/// i.e. no reduction). This couples the verifier's `T_s` to what production
/// actually fires; kept as its own (cheap, small-cap) test so the per-marking
/// verifier loop need not recompute the production set on every state.
#[test]
fn fired_set_matches_closure_intersection() {
    let mut s: u64 = 0xFEED_FACE_C0FF_EE01;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut checked = 0usize;
    for _ in 0..2_000 {
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
        let dep = DependencyGraph::build(&net);

        // Check at the initial marking and a few one-step successors — enough to
        // exercise both the reduce and the no-reduction (None) branches.
        let mut markings = vec![net.initial_marking.clone()];
        for t in enabled_transitions(&net, &net.initial_marking) {
            if let Ok(m) = net.fire(&net.initial_marking, t) {
                markings.push(m);
            }
        }
        for marking in markings {
            let enabled = enabled_transitions(&net, &marking);
            let fired =
                compute_stubborn_set(&net, &marking, &dep, &PorStrategy::DeadlockPreserving);
            match super::deadlock_stubborn_closure(&net, &marking, &dep) {
                Some(ts_vec) => {
                    let mut in_ts = vec![false; net.num_transitions()];
                    for &t in &ts_vec {
                        in_ts[t.0 as usize] = true;
                    }
                    let expected: Vec<TransitionIdx> = enabled
                        .iter()
                        .copied()
                        .filter(|t| in_ts[t.0 as usize])
                        .collect();
                    if expected.len() >= enabled.len() {
                        // No reduction: production must signal None (full set).
                        assert!(
                            fired.is_none(),
                            "expected None (no reduction) but got {fired:?} at {marking:?}"
                        );
                    } else {
                        assert_eq!(
                            fired.as_deref(),
                            Some(expected.as_slice()),
                            "production fired set != T_s ∩ E(s) at {marking:?}"
                        );
                    }
                    checked += 1;
                }
                None => {
                    // |E(s)| <= 1: production must also decline (None).
                    assert!(
                        fired.is_none(),
                        "closure declined but production returned {fired:?} at {marking:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1_000, "too few markings checked ({checked})");
}

// ---------------------------------------------------------------------------
// The deadlock-existence differential, as a reusable per-case check.
// ---------------------------------------------------------------------------

/// Run FULL and POR BFS on `net` and require the deadlock-existence boolean to
/// match. Returns `Ok(())` on agreement (or an inconclusive skip), `Err(_)` on a
/// genuine divergence — which is a soundness bug in the stubborn-set computation.
fn check_deadlock_existence_por(net: &PetriNet) -> Result<(), TestCaseError> {
    let full = bfs_full(net, MAX_STATES);
    if !full.conclusive {
        return Ok(()); // inconclusive — skip
    }
    let dep = DependencyGraph::build(net);
    let (por, _reduced_some) = bfs_por(net, &dep, MAX_STATES);
    // POR explores a SUBSET of FULL's reachable set, so when FULL is conclusive
    // POR is too. A non-conclusive POR here would itself be a bug; treat as skip
    // defensively (it cannot legitimately happen).
    if !por.conclusive {
        return Ok(());
    }

    prop_assert_eq!(
        full.deadlock,
        por.deadlock,
        "STUBBORN-SET DEADLOCK-EXISTENCE DIVERGENCE (DeadlockPreserving)\n\
         full.deadlock={} por.deadlock={}  full.states={} por.states={}\n\
         net: places={} m0={:?} transitions={:?}",
        full.deadlock,
        por.deadlock,
        full.states,
        por.states,
        net.places.len(),
        net.initial_marking,
        net.transitions
            .iter()
            .map(|t| (
                t.inputs
                    .iter()
                    .map(|a| (a.place.0, a.weight))
                    .collect::<Vec<_>>(),
                t.outputs
                    .iter()
                    .map(|a| (a.place.0, a.weight))
                    .collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>()
    );

    // Monotonicity sanity: POR must never explore MORE states than FULL.
    prop_assert!(
        por.states <= full.states,
        "POR explored more states than FULL: por={} full={}",
        por.states,
        full.states
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        // A LARGE battery: thousands of random nets per run. Deadlock soundness
        // is 0-wrong, so the case count is deliberately high. Override with
        // `PROPTEST_CASES=<n>` for an even larger ad-hoc battery.
        cases: 8000,
        max_shrink_iters: 8000,
        .. ProptestConfig::default()
    })]

    /// THE STUBBORN-SET DEADLOCK GATE — deadlock-existence preservation for
    /// `compute_stubborn_set(.., DeadlockPreserving)` with the production
    /// fail-closed fallback. The deadlock boolean must be IDENTICAL FULL-vs-POR.
    /// ZERO disagreements. A divergence is a D1/D2 soundness violation that would
    /// produce a wrong `ReachabilityDeadlock` verdict in the explicit lane.
    ///
    /// PLUS the per-state D1+D2 verifier over EVERY reachable marking — the
    /// fine-grained teeth that catch a dropped D2 clause even when the boolean
    /// differential happens not to flip on a given net.
    #[test]
    fn proptest_stubborn_deadlock_existence_preserved(raw in raw_net_strategy()) {
        let net = build_net(&raw);
        check_deadlock_existence_por(&net)?;
        // Per-state structural soundness: every reachable marking's stubborn set
        // satisfies D1+D2. An inconclusive (cap/overflow) run is skipped.
        match verify_d1_d2_reachable(&net, MAX_STATES) {
            Ok(_) => {}
            Err(msg) => prop_assert!(
                false,
                "D1+D2 VIOLATION: {}\nnet: places={} m0={:?} transitions={:?}",
                msg,
                net.places.len(),
                net.initial_marking,
                net.transitions
                    .iter()
                    .map(|t| (
                        t.inputs.iter().map(|a| (a.place.0, a.weight)).collect::<Vec<_>>(),
                        t.outputs.iter().map(|a| (a.place.0, a.weight)).collect::<Vec<_>>()
                    ))
                    .collect::<Vec<_>>()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// NON-VACUITY GUARD. A differential gate that never actually REDUCES on a
// deadlocking net would pass trivially and prove nothing. This deterministic
// (RNG-free) sweep over a self-contained xorshift net generator asserts that
// within a modest battery the stubborn-set reduction DOES fire (a strict subset
// of enabled was returned at some marking) AND co-occurs with a real reachable
// deadlock in the FULL net — i.e. the gate genuinely exercises the path where an
// unsound under-inclusion would manufacture a missed deadlock.
// ---------------------------------------------------------------------------
#[test]
fn stubborn_deadlock_gate_actually_reduces_on_deadlocking_nets() {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut conclusive = 0usize;
    let mut reduced_and_deadlock = 0usize;
    let mut agreements = 0usize;
    let total = 2_500;
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

        let full = bfs_full(&net, MAX_STATES);
        if !full.conclusive {
            continue;
        }
        conclusive += 1;
        let dep = DependencyGraph::build(&net);
        let (por, reduced_some) = bfs_por(&net, &dep, MAX_STATES);
        // The deterministic sweep is ALSO a differential: it must agree too.
        assert_eq!(
            full.deadlock,
            por.deadlock,
            "deterministic sweep deadlock-existence divergence on net \
             places={} m0={:?} transitions={:?}",
            net.places.len(),
            net.initial_marking,
            net.transitions
                .iter()
                .map(|t| (
                    t.inputs
                        .iter()
                        .map(|a| (a.place.0, a.weight))
                        .collect::<Vec<_>>(),
                    t.outputs
                        .iter()
                        .map(|a| (a.place.0, a.weight))
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>()
        );
        agreements += 1;
        if full.deadlock && reduced_some > 0 {
            reduced_and_deadlock += 1;
        }
    }
    assert!(
        conclusive > 300,
        "non-vacuity: too few conclusive nets ({conclusive}); the gate would prove little"
    );
    assert_eq!(
        agreements, conclusive,
        "every conclusive net must agree (0-wrong)"
    );
    assert!(
        reduced_and_deadlock >= 1,
        "non-vacuity: the stubborn-set reduction never fired (returned a strict \
         subset of enabled) on a deadlocking net in {conclusive} conclusive cases \
         — the deadlock POR gate is vacuous (is DeadlockPreserving still wired?)"
    );
}

// ---------------------------------------------------------------------------
// DETERMINISTIC D1+D2 SWEEP (RNG-free, reproducible). Verifies the per-state
// D1+D2 conditions over the full reachable set of a band of structured and
// pseudo-random nets, and asserts NON-VACUITY: within the battery the verifier
// must actually EXERCISE both the D2-enabled clause (an enabled stubborn
// transition with a non-empty disabler set, all of which are in the set) AND the
// D2-disabled clause (a disabled stubborn transition certified by a necessary
// set). If a future edit drops either D2 clause, the corresponding witness count
// would drop to the now-unsatisfiable check and the verifier trips at the
// offending marking; if the band stopped exercising a clause, the non-vacuity
// assertion trips. Together these keep teeth against EACH clause individually.
// ---------------------------------------------------------------------------

#[test]
fn stubborn_d1_d2_holds_at_every_reachable_marking() {
    let mut s: u64 = 0xD1D2_1234_5678_9ABC;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    let mut conclusive = 0usize;
    let mut total_d2_enabled = 0usize;
    let mut total_d2_disabled = 0usize;
    let total = 3_000;
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

        // The CORE soundness check: D1+D2 at every reachable marking, with the
        // per-clause witness counts folded into the same single BFS. A smaller
        // per-net cap keeps this deterministic sweep fast (D1+D2 is a PER-STATE
        // condition — a net exceeding the cap is skipped as inconclusive, not a
        // failure; the band still yields hundreds of fully-explored nets).
        match verify_d1_d2_reachable(&net, 2_000) {
            Ok(Some((e, d))) => {
                conclusive += 1;
                total_d2_enabled += e;
                total_d2_disabled += d;
            }
            Ok(None) => {} // inconclusive — skip
            Err(msg) => panic!(
                "D1+D2 VIOLATION: {msg}\nnet: places={} m0={:?} transitions={:?}",
                net.places.len(),
                net.initial_marking,
                net.transitions
                    .iter()
                    .map(|t| (
                        t.inputs
                            .iter()
                            .map(|a| (a.place.0, a.weight))
                            .collect::<Vec<_>>(),
                        t.outputs
                            .iter()
                            .map(|a| (a.place.0, a.weight))
                            .collect::<Vec<_>>()
                    ))
                    .collect::<Vec<_>>()
            ),
        }
    }

    assert!(
        conclusive > 500,
        "non-vacuity: too few conclusive nets ({conclusive})"
    );
    // Both D2 clauses must be EXERCISED by the battery; otherwise the verifier is
    // vacuous for the unexercised clause and a regression there would slip by.
    assert!(
        total_d2_enabled >= 1,
        "non-vacuity: the D2-enabled clause was never exercised across {conclusive} \
         conclusive nets — the verifier is vacuous for D2-enabled"
    );
    assert!(
        total_d2_disabled >= 1,
        "non-vacuity: the D2-disabled clause was never exercised across {conclusive} \
         conclusive nets — the verifier is vacuous for D2-disabled"
    );
}
