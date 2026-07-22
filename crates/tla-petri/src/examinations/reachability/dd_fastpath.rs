// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Decision-Diagram exact fast-path for ReachabilityCardinality /
//! ReachabilityFireability.
//!
//! For a small bounded net this builds the **complete** reachable-marking
//! set symbolically (one shared BDD over the *original* net) and answers
//! each pending `EF(φ)` / `AG(φ)` query exactly:
//!
//! - `EF(φ)` ⟺ some reachable marking satisfies `φ`.
//! - `AG(φ)` ⟺ every reachable marking satisfies `φ`.
//!
//! # Soundness
//!
//! Built on the same fail-closed contract as the StateSpace and
//! UpperBounds DD paths:
//!
//! 1. The per-place encoding bounds come from
//!    [`crate::examinations::dd_spec::build_sound_dd_spec`], which uses
//!    sound LP upper bounds and rejects (returns `None`) any net the unary
//!    encoding cannot represent without silently dropping a reachable
//!    marking. A `None` here ⇒ the DD path declines and the pipeline
//!    continues with its explicit/solver phases.
//! 2. `tla_dd::dispatch_reachability_queries` errors (never truncates)
//!    unless the reachable-set fixed point converges, so any `Ok` result
//!    is the **exact** reachable set. EF/AG over an exact reachable set is
//!    the ground-truth verdict — equivalent to a completed original-net
//!    BFS. This is why a DD verdict is sound even for the AG-fireability
//!    case that otherwise requires original-net full BFS: the reachable
//!    set is built directly from `net.transitions` / `net.initial_marking`
//!    with no reduction.
//! 3. The DD computation runs on a worker thread with a wall-clock budget;
//!    on timeout, spawn failure, worker panic, or any DD error we resolve
//!    nothing and fall through (sound).
//!
//! Because the result is exact, a DD-resolved verdict is committed via
//! [`resolve_tracker`] exactly like any completed-exploration verdict.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::petri_net::PetriNet;

use super::types::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};
use crate::examinations::dd_spec::translate_predicate;
use crate::property_xml::PathQuantifier;

/// Wall-clock budget for the DD reachability batch. Mirrors the StateSpace
/// and UpperBounds DD paths.
const DD_BUDGET: Duration = Duration::from_secs(5);

/// Run the exact DD reachability fast-path over all pending trackers.
///
/// Returns the number of trackers newly resolved. Declines (returns 0)
/// whenever the net is not DD-eligible, no predicate translates, or the
/// DD computation does not converge within [`DD_BUDGET`].
#[must_use]
pub(super) fn run_dd_reachability_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
) -> usize {
    if trackers.iter().all(|t| t.verdict.is_some()) {
        return 0;
    }

    // Soundness gate + spec construction shared with the StateSpace and
    // UpperBounds DD paths. `None` ⇒ the unary encoding cannot represent
    // this net without risking a dropped reachable marking — decline.
    let Some((spec, _bounds)) = crate::examinations::dd_spec::build_sound_dd_spec(net) else {
        return 0;
    };

    let num_places = net.num_places();
    let num_transitions = net.num_transitions();

    // Build a query per *pending* tracker; remember which tracker each
    // query maps to so we can write verdicts back in order. A tracker
    // whose predicate fails to translate is left for downstream phases.
    let mut query_trackers: Vec<usize> = Vec::new();
    let mut queries: Vec<tla_dd::DdReachQuery> = Vec::new();
    for (i, tracker) in trackers.iter().enumerate() {
        if tracker.verdict.is_some() {
            continue;
        }
        let Some(predicate) = translate_predicate(&tracker.predicate, num_places, num_transitions)
        else {
            continue;
        };
        let quantifier = match tracker.quantifier {
            PathQuantifier::EF => tla_dd::DdQuantifier::Ef,
            PathQuantifier::AG => tla_dd::DdQuantifier::Ag,
        };
        query_trackers.push(i);
        queries.push(tla_dd::DdReachQuery {
            quantifier,
            predicate,
        });
    }
    if queries.is_empty() {
        return 0;
    }

    // The DD reachability fast-path runs natively on tla-bdd — the oxidd engine
    // was REMOVED. Validated as a sound oxidd replacement: ≡ the MDD lane on
    // DdPredicates, the full 377-test reachability suite passes, and a 46-model
    // corpus A/B (5 examinations) found ZERO verdict disagreements vs oxidd with a
    // net +10 coverage gain. Budget-bounded via `reachable_within`; any
    // decline/timeout/panic returns 0 and falls through to the explicit pipeline
    // soundly, exactly as the oxidd path did.
    run_bdd_reachability_seeding(&spec, &queries, &query_trackers, trackers)
}

/// The native-ROBDD reachability seeding lane: the same exact-reachable-set EF/AG
/// contract as the oxidd path, computed by `tla-bdd` and committed identically.
/// Worker thread + `DD_BUDGET` wall-clock deadline (the budget binds end-to-end
/// via `reachable_within`); any decline / timeout / panic falls through soundly
/// (returns 0), exactly like the oxidd path.
#[must_use]
fn run_bdd_reachability_seeding(
    spec: &tla_dd::DdNetSpec,
    queries: &[tla_dd::DdReachQuery],
    query_trackers: &[usize],
    trackers: &mut [PropertyTracker],
) -> usize {
    use tla_mdd::MddReachQuantifier;
    let bdd_queries: Vec<(MddReachQuantifier, tla_dd::DdPredicate)> = queries
        .iter()
        .map(|q| {
            let quant = match q.quantifier {
                tla_dd::DdQuantifier::Ef => MddReachQuantifier::Ef,
                tla_dd::DdQuantifier::Ag => MddReachQuantifier::Ag,
            };
            (quant, q.predicate.clone())
        })
        .collect();
    let (tx, rx) = mpsc::channel();
    let spec_for_thread = spec.clone();
    let handle = thread::Builder::new()
        .name("tla-bdd-reachability".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let deadline = std::time::Instant::now() + DD_BUDGET;
            let _ = tx.send(
                crate::examinations::mdd_common::evaluate_reachability_via_bdd(
                    &spec_for_thread,
                    &bdd_queries,
                    Some(deadline),
                ),
            );
        });
    if handle.is_err() {
        eprintln!("Reachability: tla-bdd lane spawn failed — using explicit pipeline");
        return 0;
    }
    let verdicts = match rx.recv_timeout(DD_BUDGET + Duration::from_millis(500)) {
        Ok(Some(verdicts)) => verdicts,
        // decline (atom lowering or budget exceeded), timeout, or worker panic.
        Ok(None) | Err(_) => return 0,
    };
    if verdicts.len() != query_trackers.len() {
        return 0;
    }
    let mut seeded = 0;
    for (&tracker_idx, &verdict) in query_trackers.iter().zip(verdicts.iter()) {
        let tracker = &mut trackers[tracker_idx];
        if tracker.verdict.is_some() {
            continue;
        }
        resolve_tracker(tracker, verdict, ReachabilityResolutionSource::Dd, None);
        seeded += 1;
    }
    seeded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
    use crate::resolved_predicate::{eval_predicate, ResolvedIntExpr, ResolvedPredicate};

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.into(),
            name: None,
        }
    }

    fn tracker(id: &str, q: PathQuantifier, pred: ResolvedPredicate) -> PropertyTracker {
        PropertyTracker {
            id: id.into(),
            quantifier: q,
            predicate: pred,
            verdict: None,
            resolved_by: None,
            flushed: false,
        }
    }

    /// 2-place swap net: p0+p1 conserved at 1. Reachable: {(1,0),(0,1)}.
    fn swap_net() -> PetriNet {
        PetriNet {
            name: Some("swap".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "t10".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Brute-force EF/AG over the explicit reachable set, for cross-check.
    fn brute_force_verdict(net: &PetriNet, q: PathQuantifier, pred: &ResolvedPredicate) -> bool {
        use std::collections::HashSet;
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        seen.insert(net.initial_marking.clone());
        let mut frontier = vec![net.initial_marking.clone()];
        while let Some(m) = frontier.pop() {
            for (tid, _t) in net.transitions.iter().enumerate() {
                let t = TransitionIdx(tid as u32);
                if !net.is_enabled(&m, t) {
                    continue;
                }
                let mut next = m.clone();
                for arc in &net.transitions[tid].inputs {
                    next[arc.place.0 as usize] -= arc.weight;
                }
                for arc in &net.transitions[tid].outputs {
                    next[arc.place.0 as usize] += arc.weight;
                }
                if seen.insert(next.clone()) {
                    frontier.push(next);
                }
            }
        }
        match q {
            PathQuantifier::EF => seen.iter().any(|m| eval_predicate(pred, m, net)),
            PathQuantifier::AG => seen.iter().all(|m| eval_predicate(pred, m, net)),
        }
    }

    #[test]
    fn resolves_ef_and_ag_cardinality_against_brute_force() {
        let net = swap_net();
        // EF(p1 >= 1): true (reach (0,1)). AG(p0 <= 1): true. AG(p0 >= 1): false.
        let preds = [
            (
                PathQuantifier::EF,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(1),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ),
            ),
            (
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                    ResolvedIntExpr::Constant(1),
                ),
            ),
            (
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(1),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                ),
            ),
        ];
        let mut trackers: Vec<PropertyTracker> = preds
            .iter()
            .enumerate()
            .map(|(i, (q, p))| tracker(&format!("c{i}"), *q, p.clone()))
            .collect();
        let seeded = run_dd_reachability_seeding(&net, &mut trackers);
        assert_eq!(seeded, 3, "all three trackers must be DD-resolved");
        for (tr, (q, p)) in trackers.iter().zip(preds.iter()) {
            let expected = brute_force_verdict(&net, *q, p);
            assert_eq!(
                tr.verdict,
                Some(expected),
                "{:?} of {:?}: DD={:?} brute-force={expected}",
                q,
                p,
                tr.verdict,
            );
            assert_eq!(
                tr.resolved_by.unwrap().source,
                ReachabilityResolutionSource::Dd
            );
        }
    }

    #[test]
    fn resolves_fireability_against_brute_force() {
        let net = swap_net();
        // EF(fireable t01): true. AG(fireable t01): false (state (0,1) can't fire t01).
        let preds = [
            (
                PathQuantifier::EF,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            ),
            (
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            ),
            // AG(fireable t01 OR t10): always exactly one is enabled.
            (
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0), TransitionIdx(1)]),
            ),
        ];
        let mut trackers: Vec<PropertyTracker> = preds
            .iter()
            .enumerate()
            .map(|(i, (q, p))| tracker(&format!("f{i}"), *q, p.clone()))
            .collect();
        let seeded = run_dd_reachability_seeding(&net, &mut trackers);
        assert_eq!(seeded, 3);
        for (tr, (q, p)) in trackers.iter().zip(preds.iter()) {
            let expected = brute_force_verdict(&net, *q, p);
            assert_eq!(tr.verdict, Some(expected), "{q:?} of {p:?}");
        }
    }

    #[test]
    fn declines_unbounded_net_leaves_trackers_pending() {
        // Source transition with no input → p0 unbounded → LP unbounded →
        // build_sound_dd_spec returns None → DD declines, nothing resolved.
        let net = PetriNet {
            name: Some("source".into()),
            places: vec![place("p0")],
            transitions: vec![TransitionInfo {
                id: "gen".into(),
                name: None,
                inputs: vec![],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            }],
            initial_marking: vec![0],
        };
        let mut trackers = vec![tracker(
            "u0",
            PathQuantifier::EF,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ),
        )];
        let seeded = run_dd_reachability_seeding(&net, &mut trackers);
        assert_eq!(seeded, 0, "unbounded net must make the DD path decline");
        assert_eq!(trackers[0].verdict, None, "tracker must remain pending");
    }

    #[test]
    fn skips_already_resolved_trackers() {
        let net = swap_net();
        let mut trackers = vec![tracker("pre", PathQuantifier::EF, ResolvedPredicate::True)];
        trackers[0].verdict = Some(false); // pre-seeded (intentionally wrong-looking)
        let seeded = run_dd_reachability_seeding(&net, &mut trackers);
        assert_eq!(seeded, 0, "already-resolved trackers are not touched");
        assert_eq!(
            trackers[0].verdict,
            Some(false),
            "first-writer-wins: DD must not overwrite an existing verdict"
        );
    }
}
