// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PROPTEST RANDOM-NET DIFFERENTIAL GATE for the SAFETY-preserving stubborn-set
//! partial-order reduction ([`compute_stubborn_set`] with
//! [`PorStrategy::SafetyPreserving`]).
//!
//! This is the safety analogue of `stubborn_differential_proptest.rs` (the
//! deadlock gate). The safety POR is the lane the explicit
//! `ReachabilityCardinality` / `Fireability` / `OneSafe` / `UpperBounds`
//! examinations run in PRODUCTION: `reachability_por_config` /
//! `one_safe_por_config` derive a `visible` transition set from the query
//! support and build `ExecutionPlan::observer(PorStrategy::SafetyPreserving {
//! visible })`. An unsound stubborn set here would MISS a reachable
//! property-violating marking and emit a WRONG verdict (e.g.
//! `ReachabilityCardinality = False` when the cardinality bound is actually
//! reachable). This gate makes that impossible to ship silently.
//!
//! Before this file, the safety POR had only fixed hand-net crosschecks
//! (`stubborn_tests.rs::assert_safety_crosscheck`, `one_safe_por.rs`). This adds
//! the random-net battery + the per-state structural verifier that the deadlock
//! lane already has.
//!
//! # The contract (violation-reachability equality over OBSERVABLE places)
//!
//! The safety-preservation theorem for stubborn sets rests on D1 + D2 + the
//! VISIBILITY PROVISO: at every state, either ZERO or ALL visible transitions
//! are fired. Because every interleaving of the observable (visible) transitions
//! is thereby realized, any reachable marking whose property is a function of
//! the VISIBLE places is preserved by the reduction.
//!
//! So for each random net we:
//!   * choose a random non-empty, non-full set of VISIBLE PLACES;
//!   * derive the VISIBLE TRANSITIONS exactly as production does — a transition
//!     is visible iff it touches (input or output) a visible place
//!     (`visible_transitions_for_support` /
//!     `mark_transitions_touching_places`). This guarantees invisible
//!     transitions never change the property, which is what the theorem needs;
//!   * choose a random reachability VIOLATION PREDICATE φ over the visible
//!     places only (a token-count threshold on one visible place);
//!   * run FULL BFS (fire all enabled) and SafetyPreserving-POR BFS (fire the
//!     stubborn subset, with the production fail-closed `None => full` fallback)
//!     and require: "∃ a reachable marking with φ?" is IDENTICAL FULL-vs-POR.
//!
//! A stubborn set that UNDER-includes (drops the visibility proviso, or an
//! interferer / producer) can prune away the only path to a reachable
//! φ-violating marking → POR reports NO violation while FULL reports one. The
//! `prop_assert_eq!` catches it. The reverse direction (POR ⊆ FULL reachable
//! set) makes a spurious POR-only violation impossible, so equality is the right
//! relation.
//!
//! # Boundedness / termination
//!
//! Identical discipline to the deadlock gate: both oracles capped at
//! [`MAX_STATES`]; a case whose FULL net exceeds the cap (or overflows a
//! marking) is INCONCLUSIVE and silently skipped. POR explores a SUBSET of
//! FULL's reachable set, so POR is conclusive whenever FULL is — the comparison
//! is only made when FULL is conclusive.

use std::collections::{HashSet, VecDeque};

use proptest::prelude::*;

use super::{compute_stubborn_set, DependencyGraph, PorStrategy};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

/// Exploration ceiling. A net whose FULL reachable set exceeds this is treated
/// as inconclusive (skipped), never as a failure.
const MAX_STATES: usize = 20_000;

// ---------------------------------------------------------------------------
// Shared primitives.
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

/// A reachability violation predicate over VISIBLE places only: "place `p` holds
/// at least `threshold` tokens". `p` is guaranteed to be a visible place, so the
/// predicate is a function of the observable projection of the marking and the
/// safety-preservation theorem applies.
#[derive(Clone, Copy)]
struct ViolationPredicate {
    place: usize,
    threshold: u64,
}

impl ViolationPredicate {
    fn holds(&self, marking: &[u64]) -> bool {
        marking[self.place] >= self.threshold
    }
}

/// Derive the VISIBLE TRANSITIONS for a set of visible places, byte-for-byte the
/// production rule (`mark_transitions_touching_places`): a transition is visible
/// iff it has an input or output arc on a visible place.
fn visible_transitions_for_places(net: &PetriNet, visible_places: &[bool]) -> Vec<TransitionIdx> {
    let mut visible = Vec::new();
    for (idx, t) in net.transitions.iter().enumerate() {
        let touches = t
            .inputs
            .iter()
            .chain(t.outputs.iter())
            .any(|arc| visible_places[arc.place.0 as usize]);
        if touches {
            visible.push(TransitionIdx(idx as u32));
        }
    }
    visible
}

/// Outcome of a bounded BFS: whether the cap was hit (inconclusive) and whether a
/// marking satisfying the violation predicate was reached.
struct BfsOutcome {
    conclusive: bool,
    violation: bool,
    states: usize,
}

/// FULL BFS: at each marking fire ALL enabled transitions. The reference oracle.
fn bfs_full(net: &PetriNet, pred: ViolationPredicate, max_states: usize) -> BfsOutcome {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut violation = pred.holds(&net.initial_marking);
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());

    while let Some(marking) = queue.pop_front() {
        for t in enabled_transitions(net, &marking) {
            let Ok(next) = net.fire(&marking, t) else {
                return BfsOutcome {
                    conclusive: false,
                    violation,
                    states: seen.len(),
                };
            };
            if seen.insert(next.clone()) {
                if pred.holds(&next) {
                    violation = true;
                }
                if seen.len() > max_states {
                    return BfsOutcome {
                        conclusive: false,
                        violation,
                        states: seen.len(),
                    };
                }
                queue.push_back(next);
            }
        }
    }

    BfsOutcome {
        conclusive: true,
        violation,
        states: seen.len(),
    }
}

/// How to choose the fired set at each marking. The honest variant is exactly
/// the production path; the mutated variants are used ONLY by the teeth test to
/// prove the gate catches an unsound reduction.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Production path: `compute_stubborn_set(.., SafetyPreserving)` with the
    /// fail-closed `None => fire all enabled` fallback.
    Production,
    /// MUTANT: drop the visibility proviso entirely (use the deadlock closure,
    /// i.e. D1+D2 only). This is the unsound reduction the gate must catch.
    DropVisibility,
}

/// POR BFS in the requested `mode`. Returns the outcome plus a count of markings
/// where the fired set was a STRICT subset of enabled (a real reduction) — used
/// by the non-vacuity guard.
fn bfs_por(
    net: &PetriNet,
    dep: &DependencyGraph,
    visible: &[TransitionIdx],
    pred: ViolationPredicate,
    mode: Mode,
    max_states: usize,
) -> (BfsOutcome, usize) {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut violation = pred.holds(&net.initial_marking);
    let mut reduced_some = 0usize;
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());

    while let Some(marking) = queue.pop_front() {
        let enabled = enabled_transitions(net, &marking);
        if enabled.is_empty() {
            continue;
        }
        let to_fire = match mode {
            // EXACT production decision path.
            Mode::Production => {
                match compute_stubborn_set(
                    net,
                    &marking,
                    dep,
                    &PorStrategy::SafetyPreserving {
                        visible: visible.to_vec(),
                    },
                ) {
                    Some(stubborn) => {
                        if stubborn.len() < enabled.len() {
                            reduced_some += 1;
                        }
                        stubborn
                    }
                    None => enabled.clone(),
                }
            }
            // MUTANT: D1+D2 only (deadlock closure) — visibility proviso dropped.
            Mode::DropVisibility => {
                match compute_stubborn_set(net, &marking, dep, &PorStrategy::DeadlockPreserving) {
                    Some(stubborn) => {
                        if stubborn.len() < enabled.len() {
                            reduced_some += 1;
                        }
                        stubborn
                    }
                    None => enabled.clone(),
                }
            }
        };
        for t in to_fire {
            let Ok(next) = net.fire(&marking, t) else {
                return (
                    BfsOutcome {
                        conclusive: false,
                        violation,
                        states: seen.len(),
                    },
                    reduced_some,
                );
            };
            if seen.insert(next.clone()) {
                if pred.holds(&next) {
                    violation = true;
                }
                if seen.len() > max_states {
                    return (
                        BfsOutcome {
                            conclusive: false,
                            violation,
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
            violation,
            states: seen.len(),
        },
        reduced_some,
    )
}

// ---------------------------------------------------------------------------
// Random-net strategy (mirrors stubborn_differential_proptest.rs): <=6 places,
// 1..=6 transitions, weights 1-3, m0 in 0..=3, at most one arc per place per
// side per transition. Plus a random visible-place mask and a violation
// threshold.
// ---------------------------------------------------------------------------

type RawArc = (u8, u8);
type RawTransition = (Vec<RawArc>, Vec<RawArc>);
/// (num_places, m0, transitions, visible_place_mask, violation_threshold)
type RawCase = (usize, Vec<u8>, Vec<RawTransition>, Vec<bool>, u8);

fn raw_arc_strategy() -> impl Strategy<Value = RawArc> {
    (0u8..=250, 1u8..=3)
}

fn raw_transition_strategy() -> impl Strategy<Value = RawTransition> {
    (
        prop::collection::vec(raw_arc_strategy(), 0..=4),
        prop::collection::vec(raw_arc_strategy(), 0..=4),
    )
}

fn raw_case_strategy() -> impl Strategy<Value = RawCase> {
    (1usize..=6).prop_flat_map(|num_places| {
        (
            Just(num_places),
            prop::collection::vec(0u8..=3, num_places),
            prop::collection::vec(raw_transition_strategy(), 1..=6),
            prop::collection::vec(any::<bool>(), num_places),
            1u8..=4,
        )
    })
}

fn build_net(num_places: usize, m0: &[u8], transitions: &[RawTransition]) -> PetriNet {
    let num_places = num_places.max(1);
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
        name: Some("stubborn-safety-proptest-random".into()),
        places,
        transitions: nets,
        initial_marking,
    }
}

/// Resolve a raw case into a net, the production-derived visible-transition set,
/// and the violation predicate, OR `None` when the case does not yield a usable
/// safety-POR configuration (no/all visible places, no/all visible transitions —
/// production would decline POR there). Skipping those is sound: the gate proves
/// nothing on a case where POR is inert, and the differential is trivially true.
fn resolve_case(
    num_places: usize,
    m0: &[u8],
    transitions: &[RawTransition],
    visible_mask: &[bool],
    threshold: u8,
) -> Option<(PetriNet, Vec<TransitionIdx>, ViolationPredicate)> {
    let net = build_net(num_places, m0, transitions);
    let np = net.num_places();

    // Visible places: the mask, normalized to be non-empty and non-full so the
    // observable projection is a strict, non-trivial subset.
    let mut visible_places = vec![false; np];
    for (i, &b) in visible_mask.iter().take(np).enumerate() {
        visible_places[i] = b;
    }
    let vp_count = visible_places.iter().filter(|&&b| b).count();
    if vp_count == 0 || vp_count == np {
        return None;
    }

    let visible = visible_transitions_for_places(&net, &visible_places);
    // Production declines POR (returns None strategy) unless the visible set is a
    // strict non-empty subset of all transitions; mirror that.
    if visible.is_empty() || visible.len() >= net.num_transitions() {
        return None;
    }

    // Violation predicate over the FIRST visible place.
    let place = visible_places.iter().position(|&b| b).expect("vp_count>0");
    let pred = ViolationPredicate {
        place,
        threshold: u64::from(threshold),
    };
    Some((net, visible, pred))
}

// ---------------------------------------------------------------------------
// PER-STATE D1 + D2 + VISIBILITY SOUNDNESS VERIFIER.
//
// As in the deadlock gate, the boolean differential alone has limited teeth
// against a single dropped clause (a transition wrongly excluded at one marking
// is often re-explored from a successor). To keep real teeth against EACH
// condition, we verify at EVERY reachable marking that the safety closure `T_s`
// satisfies the conditions the SAFETY-preservation theorem rests on:
//
//   D1 (deadlock): if any transition is enabled, the FIRED set (T_s ∩ E(s)) is
//       non-empty.
//   D2-enabled: for each ENABLED t in T_s, every disabler (input-place sharer,
//       `interferers(t)`) is in T_s.
//   D2-disabled: for each DISABLED t in T_s, there EXISTS an under-marked input
//       place whose producers are all in T_s (a necessary set).
//   VISIBILITY: if T_s contains ANY visible transition, it contains ALL visible
//       transitions. THIS is the clause the deadlock lane lacks and that
//       upgrades D1+D2 to safety preservation.
//
// A reduction that drops the visibility proviso (or any D2 clause) violates the
// corresponding condition at the offending marking, and this verifier fails
// IMMEDIATELY and deterministically with that marking.
// ---------------------------------------------------------------------------

fn verify_at(
    net: &PetriNet,
    dep: &DependencyGraph,
    visible: &[TransitionIdx],
    marking: &[u64],
) -> Result<(), String> {
    let enabled: Vec<TransitionIdx> = enabled_transitions(net, marking);

    // The full safety closure T_s (including disabled members). `None` => no
    // reduction possible (all-visible or |E(s)| <= 1) => production fires the
    // full enabled set, vacuously sound; nothing to verify.
    let Some(ts_vec) = super::safety_stubborn_closure(net, marking, dep, visible) else {
        return Ok(());
    };
    let n = net.num_transitions();
    let mut in_ts = vec![false; n];
    for &t in &ts_vec {
        in_ts[t.0 as usize] = true;
    }

    // D1: the fired set T_s ∩ E(s) is non-empty at a live marking.
    let fired_count = enabled.iter().filter(|t| in_ts[t.0 as usize]).count();
    if !enabled.is_empty() && fired_count == 0 {
        return Err("D1 violated: empty fired set (T_s ∩ E(s)) at a live marking".into());
    }

    // VISIBILITY proviso: T_s contains a visible transition ⇒ T_s contains ALL
    // visible transitions.
    let any_visible_in_ts = visible.iter().any(|v| in_ts[v.0 as usize]);
    if any_visible_in_ts {
        for &v in visible {
            if !in_ts[v.0 as usize] {
                return Err(format!(
                    "VISIBILITY violated: T_s contains a visible transition but visible \
                     {v:?} is NOT in T_s (proviso requires all-or-none of visible)"
                ));
            }
        }
    }

    // D2 over the full closure T_s.
    for t in 0..n {
        if !in_ts[t] {
            continue;
        }
        let tidx = TransitionIdx(t as u32);
        if net.is_enabled(marking, tidx) {
            for &d in dep.interferers(tidx) {
                if !in_ts[d.0 as usize] {
                    return Err(format!(
                        "D2-enabled violated: enabled {tidx:?} in T_s but its disabler \
                         {d:?} (shares an input place) is NOT in T_s"
                    ));
                }
            }
        } else {
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

/// Per-clause witness counts folded into one BFS, for the non-vacuity guard.
#[derive(Default, Clone, Copy)]
struct VerifyHits {
    d2_enabled: usize,
    d2_disabled: usize,
    visibility_expanded: usize,
}

/// Verify D1+D2+visibility at every reachable marking (bounded, FULL successor
/// relation). `Ok(Some(hits))` when conclusive and every marking passed,
/// `Ok(None)` if inconclusive (cap/overflow), `Err(msg)` on the first violation.
fn verify_reachable(
    net: &PetriNet,
    visible: &[TransitionIdx],
    max_states: usize,
) -> Result<Option<VerifyHits>, String> {
    let dep = DependencyGraph::build(net);
    let mut is_visible = vec![false; net.num_transitions()];
    for &v in visible {
        is_visible[v.0 as usize] = true;
    }
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let mut hits = VerifyHits::default();
    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());
    while let Some(marking) = queue.pop_front() {
        verify_at(net, &dep, visible, &marking)?;
        if let Some(ts_vec) = super::safety_stubborn_closure(net, &marking, &dep, visible) {
            let mut in_ts = vec![false; net.num_transitions()];
            for &t in &ts_vec {
                in_ts[t.0 as usize] = true;
            }
            if visible.iter().any(|v| in_ts[v.0 as usize]) {
                hits.visibility_expanded += 1;
            }
            for &t in &ts_vec {
                if net.is_enabled(&marking, t) {
                    if !dep.interferers(t).is_empty() {
                        hits.d2_enabled += 1;
                    }
                } else {
                    hits.d2_disabled += 1;
                }
            }
        }
        for t in enabled_transitions(net, &marking) {
            let Ok(next) = net.fire(&marking, t) else {
                return Ok(None);
            };
            if seen.insert(next.clone()) {
                if seen.len() > max_states {
                    return Ok(None);
                }
                queue.push_back(next);
            }
        }
    }
    Ok(Some(hits))
}

// ---------------------------------------------------------------------------
// The violation-reachability differential, as a reusable per-case check.
// ---------------------------------------------------------------------------

/// Run FULL and SafetyPreserving-POR BFS on `net` and require the
/// violation-reachability boolean to match. `Ok(())` on agreement (or an
/// inconclusive skip), `Err(_)` on a genuine divergence — a safety-POR soundness
/// bug.
fn check_violation_reachability(
    net: &PetriNet,
    visible: &[TransitionIdx],
    pred: ViolationPredicate,
) -> Result<(), TestCaseError> {
    let full = bfs_full(net, pred, MAX_STATES);
    if !full.conclusive {
        return Ok(()); // inconclusive — skip
    }
    let dep = DependencyGraph::build(net);
    let (por, _reduced) = bfs_por(net, &dep, visible, pred, Mode::Production, MAX_STATES);
    if !por.conclusive {
        return Ok(());
    }

    prop_assert_eq!(
        full.violation,
        por.violation,
        "SAFETY STUBBORN-SET VIOLATION-REACHABILITY DIVERGENCE (SafetyPreserving)\n\
         full.violation={} por.violation={}  full.states={} por.states={}\n\
         visible={:?} pred(place={}, >= {})\n\
         net: places={} m0={:?} transitions={:?}",
        full.violation,
        por.violation,
        full.states,
        por.states,
        visible,
        pred.place,
        pred.threshold,
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
        cases: 8000,
        max_shrink_iters: 8000,
        .. ProptestConfig::default()
    })]

    /// THE SAFETY STUBBORN-SET GATE — violation-reachability preservation for
    /// `compute_stubborn_set(.., SafetyPreserving)` with the production
    /// fail-closed fallback. The reachable-violation boolean (over observable
    /// places) must be IDENTICAL FULL-vs-POR. ZERO disagreements. A divergence
    /// is a D1/D2/visibility soundness violation that would produce a wrong
    /// `ReachabilityCardinality` / `Fireability` / `OneSafe` verdict.
    ///
    /// PLUS the per-state D1+D2+visibility verifier over EVERY reachable
    /// marking — the fine-grained teeth that catch a dropped clause even when
    /// the boolean differential happens not to flip on a given net.
    #[test]
    fn proptest_safety_violation_reachability_preserved(case in raw_case_strategy()) {
        let (num_places, m0, transitions, visible_mask, threshold) = case;
        let Some((net, visible, pred)) =
            resolve_case(num_places, &m0, &transitions, &visible_mask, threshold)
        else {
            return Ok(()); // POR inert on this case — nothing to prove
        };
        check_violation_reachability(&net, &visible, pred)?;
        match verify_reachable(&net, &visible, MAX_STATES) {
            Ok(_) => {}
            Err(msg) => prop_assert!(
                false,
                "D1+D2+VISIBILITY VIOLATION: {}\nvisible={:?}\nnet: places={} m0={:?} transitions={:?}",
                msg,
                visible,
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
// Deterministic (RNG-free) helpers shared by the non-vacuity and teeth sweeps.
// ---------------------------------------------------------------------------

struct Xorshift(u64);
impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Generate a random raw case from an xorshift stream (deterministic).
fn gen_case(rng: &mut Xorshift) -> RawCase {
    let num_places = (rng.next() as usize % 6) + 1;
    let m0: Vec<u8> = (0..num_places).map(|_| (rng.next() % 4) as u8).collect();
    let ntrans = (rng.next() as usize % 6) + 1;
    let transitions: Vec<RawTransition> = (0..ntrans)
        .map(|_| {
            let nin = rng.next() as usize % 5;
            let nout = rng.next() as usize % 5;
            let ins: Vec<RawArc> = (0..nin)
                .map(|_| ((rng.next() % 251) as u8, (rng.next() % 3 + 1) as u8))
                .collect();
            let outs: Vec<RawArc> = (0..nout)
                .map(|_| ((rng.next() % 251) as u8, (rng.next() % 3 + 1) as u8))
                .collect();
            (ins, outs)
        })
        .collect();
    let visible_mask: Vec<bool> = (0..num_places).map(|_| rng.next() & 1 == 1).collect();
    let threshold = (rng.next() % 4 + 1) as u8;
    (num_places, m0, transitions, visible_mask, threshold)
}

// ---------------------------------------------------------------------------
// NON-VACUITY GUARD. A differential that never actually REDUCES, or never hits a
// reachable violation, proves nothing. This deterministic sweep asserts that
// within a modest battery the safety stubborn-set reduction DOES fire (strict
// subset of enabled at some marking) AND co-occurs with a reachable violation in
// the FULL net — the exact path where an unsound under-inclusion would
// manufacture a missed violation. It is ALSO a differential: every conclusive
// net must agree FULL-vs-POR (0-wrong).
// ---------------------------------------------------------------------------
#[test]
fn safety_gate_actually_reduces_on_violating_nets() {
    let mut rng = Xorshift(0xA11C_E555_AFE7_0001);
    let mut conclusive = 0usize;
    let mut agreements = 0usize;
    let mut reduced_and_violation = 0usize;
    let total = 6_000;
    for _ in 0..total {
        let (np, m0, transitions, mask, threshold) = gen_case(&mut rng);
        let Some((net, visible, pred)) = resolve_case(np, &m0, &transitions, &mask, threshold)
        else {
            continue;
        };
        let full = bfs_full(&net, pred, MAX_STATES);
        if !full.conclusive {
            continue;
        }
        conclusive += 1;
        let dep = DependencyGraph::build(&net);
        let (por, reduced_some) = bfs_por(&net, &dep, &visible, pred, Mode::Production, MAX_STATES);
        assert_eq!(
            full.violation,
            por.violation,
            "deterministic safety sweep violation-reachability divergence on net \
             places={} m0={:?} visible={:?} pred(place={},>= {}) transitions={:?}",
            net.places.len(),
            net.initial_marking,
            visible,
            pred.place,
            pred.threshold,
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
        if full.violation && reduced_some > 0 {
            reduced_and_violation += 1;
        }
    }
    assert!(
        conclusive > 300,
        "non-vacuity: too few conclusive safety-POR cases ({conclusive})"
    );
    assert_eq!(
        agreements, conclusive,
        "every conclusive net must agree (0-wrong)"
    );
    assert!(
        reduced_and_violation >= 1,
        "non-vacuity: the safety stubborn-set reduction never fired (returned a strict \
         subset of enabled) on a net with a reachable violation in {conclusive} conclusive \
         cases — the safety POR gate is vacuous (is SafetyPreserving still wired?)"
    );
}

// ---------------------------------------------------------------------------
// DETERMINISTIC D1+D2+VISIBILITY SWEEP (RNG-free). Verifies the per-state
// conditions over the full reachable set of a band of pseudo-random nets, and
// asserts NON-VACUITY: within the battery the verifier must EXERCISE the
// D2-enabled clause, the D2-disabled clause, AND the visibility-expansion clause
// (a state where a visible transition entered T_s, so the all-visible check was
// non-trivial). If a future edit drops a clause, the verifier trips at the
// offending marking; if the band stopped exercising a clause, the non-vacuity
// assertion trips.
// ---------------------------------------------------------------------------
#[test]
fn safety_d1_d2_visibility_holds_at_every_reachable_marking() {
    let mut rng = Xorshift(0x5AFE_D1D2_9ABC_DEF1);
    let mut conclusive = 0usize;
    let mut total = VerifyHits::default();
    let runs = 6_000;
    for _ in 0..runs {
        let (np, m0, transitions, mask, threshold) = gen_case(&mut rng);
        let Some((net, visible, _pred)) = resolve_case(np, &m0, &transitions, &mask, threshold)
        else {
            continue;
        };
        match verify_reachable(&net, &visible, 2_000) {
            Ok(Some(hits)) => {
                conclusive += 1;
                total.d2_enabled += hits.d2_enabled;
                total.d2_disabled += hits.d2_disabled;
                total.visibility_expanded += hits.visibility_expanded;
            }
            Ok(None) => {}
            Err(msg) => panic!(
                "D1+D2+VISIBILITY VIOLATION: {msg}\nvisible={:?}\nnet: places={} m0={:?} transitions={:?}",
                visible,
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
    assert!(
        conclusive > 300,
        "non-vacuity: too few conclusive nets ({conclusive})"
    );
    assert!(
        total.d2_enabled >= 1,
        "non-vacuity: the D2-enabled clause was never exercised ({conclusive} nets)"
    );
    assert!(
        total.d2_disabled >= 1,
        "non-vacuity: the D2-disabled clause was never exercised ({conclusive} nets)"
    );
    assert!(
        total.visibility_expanded >= 1,
        "non-vacuity: the VISIBILITY clause was never exercised — no reachable marking \
         pulled a visible transition into T_s across {conclusive} nets; the visibility \
         proviso check is vacuous"
    );
}

// ---------------------------------------------------------------------------
// TEETH VERIFICATION (mutation injection). Proves the gate is not vacuous by
// running the EXACT same differential against a MUTANT that drops the visibility
// proviso (Mode::DropVisibility — i.e. the D1+D2 deadlock closure, which is
// sound for deadlock but UNSOUND for safety). On at least one net in the battery
// the mutant MUST miss a reachable violation that FULL finds — i.e. the boolean
// differential FLIPS. If the mutant never diverged, the differential would have
// no teeth against the visibility proviso and this test fails loudly.
//
// This is the "drop the proviso => the gate catches a missed reachable
// violation" requirement. It does NOT change production: the mutant lives only
// here. The honest `Mode::Production` path is asserted to AGREE on every one of
// those same nets, confirming the production reduction is sound exactly where
// the mutant is not.
// ---------------------------------------------------------------------------
#[test]
fn dropping_visibility_proviso_is_caught_by_the_gate() {
    let mut rng = Xorshift(0x7EE7_1500_0DD0_0001);
    let mut conclusive = 0usize;
    let mut mutant_divergences = 0usize;
    let mut production_agreements = 0usize;
    let total = 20_000;
    for _ in 0..total {
        let (np, m0, transitions, mask, threshold) = gen_case(&mut rng);
        let Some((net, visible, pred)) = resolve_case(np, &m0, &transitions, &mask, threshold)
        else {
            continue;
        };
        let full = bfs_full(&net, pred, MAX_STATES);
        if !full.conclusive {
            continue;
        }
        conclusive += 1;
        let dep = DependencyGraph::build(&net);

        // Honest production path MUST agree (sound).
        let (prod, _) = bfs_por(&net, &dep, &visible, pred, Mode::Production, MAX_STATES);
        if prod.conclusive && prod.violation == full.violation {
            production_agreements += 1;
        } else if prod.conclusive {
            panic!(
                "PRODUCTION safety POR MISSED a reachable violation — REAL UNSOUNDNESS BUG.\n\
                 full.violation={} prod.violation={}\n\
                 visible={:?} pred(place={},>= {})\n\
                 net: places={} m0={:?} transitions={:?}",
                full.violation,
                prod.violation,
                visible,
                pred.place,
                pred.threshold,
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
        }

        // MUTANT (visibility proviso dropped) — record where it diverges from
        // FULL. Each such divergence is a missed reachable violation the gate
        // would catch on the production path if the proviso were removed.
        let (mutant, _) = bfs_por(&net, &dep, &visible, pred, Mode::DropVisibility, MAX_STATES);
        if mutant.conclusive && mutant.violation != full.violation {
            // The mutant can only UNDER-report (POR ⊆ FULL reachable set).
            assert!(
                full.violation && !mutant.violation,
                "mutant over-reported a violation FULL lacks — impossible (POR ⊆ FULL)"
            );
            mutant_divergences += 1;
        }
    }
    assert!(
        conclusive > 1_000,
        "teeth: too few conclusive cases ({conclusive}) to trust the mutation result"
    );
    assert_eq!(
        production_agreements, conclusive,
        "every conclusive net: production safety POR must agree with FULL (0-wrong)"
    );
    assert!(
        mutant_divergences >= 1,
        "TEETH FAILURE: dropping the visibility proviso never caused a missed reachable \
         violation across {conclusive} conclusive cases — the differential has no teeth \
         against the visibility proviso, so a regression that removed it would pass silently"
    );
}
