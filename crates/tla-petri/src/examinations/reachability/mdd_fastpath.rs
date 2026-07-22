// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! MDD exact fast-path for ReachabilityCardinality / ReachabilityFireability —
//! the MDD twin of [`super::dd_fastpath`].
//!
//! The BDD fast-path ([`super::dd_fastpath::run_dd_reachability_seeding`])
//! bit-blasts each place into `ceil(log2(bound+1))` Boolean variables. On the
//! counter / conserved / high-bound net families the per-place bits interleave
//! badly and the reachable-set BDD blows up (the BDD lane then times out on its
//! 5s budget and DECLINES). The MDD spends ONE level per place — no bit-blasting
//! — and converges on exactly those families (it is the same backend that made
//! the StateSpace examination effective there).
//!
//! This lane builds the **complete** reachable-marking set of the *original*
//! net symbolically (`MddNet::build_reachable_saturation`, cross-checked EQUAL
//! to BFS by `tla-mdd`'s `crosscheck_bfs` battery) and answers each pending
//! `EF(φ)` / `AG(φ)` query EXACTLY against it:
//!
//! - `EF(φ)` ⟺ `R ∩ charset(φ) ≠ ∅`        — some reachable marking satisfies φ.
//! - `AG(φ)` ⟺ `R \ charset(φ) = ∅`         — no reachable marking violates φ.
//!
//! # Soundness (EXACT both directions, fail-closed)
//!
//! 1. **Admission.** [`crate::examinations::mdd_common::build_mdd_spec_for_net`]
//!    gates the net through [`crate::examinations::dd_spec::build_sound_dd_spec`]
//!    (sound per-place LP upper bounds + structural gates) plus an edge-width
//!    cap. The encoded value range is therefore a *superset* of every place's
//!    reachable projection, so no reachable marking can be dropped and both the
//!    reachable set `R` and every atom's characteristic set `charset(φ)` are
//!    EXACT. A `None` here ⇒ the lane declines and the pipeline continues with
//!    its explicit/solver phases. NOTE: unlike the BDD twin this gate drops the
//!    BDD-only 127-variable / bit-blast ceiling (the MDD has no such limit) but
//!    keeps the same sound LP bounds + edge-width cap.
//! 2. **Exact `R`.** `build_reachable_saturation` returns `Err` (never a
//!    partial / truncated set) on overflow, node cap, or deadline. So any `Ok`
//!    `R` is the exact reachable set — EF/AG membership in it is the
//!    ground-truth verdict, equivalent to a completed original-net BFS. This is
//!    STRICTLY STRONGER than the over-approximate integer/LP closers (whose
//!    UNSAT proves only the universal side, one-way).
//! 3. **Fail-closed.** The build runs on a worker thread with the big DD stack +
//!    a wall-clock budget; on timeout, spawn failure, worker panic, atom-lowering
//!    decline, or any MDD error we resolve NOTHING and the trackers fall through
//!    unchanged to the existing lanes. A wrong verdict is impossible — the lane
//!    can only withhold.
//!
//! Because the result is exact, an MDD-resolved verdict is committed via
//! [`resolve_tracker`] with the SAME [`ReachabilityResolutionSource::Dd`] source
//! the BDD twin uses (both are exact-reachable-set lanes; first-writer-wins).

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::petri_net::PetriNet;

use super::types::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};
use crate::examinations::dd_spec::translate_predicate;
use crate::examinations::mdd_common::{
    build_mdd_spec_for_net, dd_spec_to_ordered_mdd_net, lower_dd_predicate_to_mdd,
    permute_dd_predicate,
};
use crate::property_xml::PathQuantifier;

/// Wall-clock budget for the MDD reachability batch when no caller deadline is
/// supplied. Mirrors the BDD twin's `DD_BUDGET` so the two lanes have the same
/// soft cap. Production MCC always supplies a deadline upstream; this is the
/// floor when one is absent.
const MDD_BUDGET: Duration = Duration::from_secs(5);

/// Kill-switch for the MDD reachability lane. The lane is ON by default (it is a
/// count-R-once lever — the proven-good MDD ROI shape, like the StateSpace MDD
/// lane — NOT a per-formula fixpoint like the MDD CTL lane, which is opt-in).
/// Set `TY_MCC_ENABLE_MDD_REACHABILITY` to a FALSY value (`0`/`off`/`false`/`no`)
/// to disable it and fall back to the BDD-or-explicit behavior unchanged.
///
/// SOUNDNESS-NEUTRAL either way: disabling the lane can only make a net decline
/// to the (also exact) explicit/solver phases instead of being decided by the
/// MDD. It never changes a published verdict, only which lane decides.
///
/// Parsed like the sibling `TY_MCC_ENABLE_*` flags: a flag that is set but
/// explicitly falsy disables; unset or any non-falsy value keeps the default
/// (ON).
fn mdd_reachability_disabled() -> bool {
    std::env::var("TY_MCC_ENABLE_MDD_REACHABILITY").is_ok_and(|v| {
        let v = v.trim();
        v == "0"
            || v.eq_ignore_ascii_case("off")
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("no")
    })
}

/// Run the exact MDD reachability fast-path over all pending trackers.
///
/// Returns the number of trackers newly resolved. Declines (returns 0) whenever
/// the lane is disabled, the net is not MDD-eligible, no predicate translates,
/// or the MDD computation does not converge within the budget. NEVER resolves a
/// tracker outside the exact-`R` + admit-gate path.
#[must_use]
pub(super) fn run_mdd_reachability_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<std::time::Instant>,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> usize {
    if mdd_reachability_disabled() {
        return 0;
    }
    if trackers.iter().all(|t| t.verdict.is_some()) {
        return 0;
    }

    // Admission gate (sound LP bounds + structural gates + edge-width cap; NO
    // BDD 127-var ceiling). `None` ⇒ the encoding cannot represent this net
    // without risking a dropped reachable marking — decline.
    let Some(spec) = build_mdd_spec_for_net(net) else {
        return 0;
    };
    // Build the MDD in a saturation-friendly place order (the scale lever the
    // BDD lane already pulls). Seed the FORCE search with the NUPN unit-hierarchy
    // block order when the model carries one — nested units of mutually exclusive
    // places are exactly the DD locality a good order wants (SmartHome and most
    // MCC P/T nets carry a NUPN). Span-guarded: the seed only extends the
    // candidate set, so it can never make the order worse than the unseeded FORCE
    // choice. `inv` maps each original place to its level so the query predicates
    // below lower into the same coordinate. SOUND: an isomorphic relabeling —
    // verdicts unchanged, only the MDD size shrinks (see `dd_spec_to_ordered_mdd_net`).
    let seed =
        nupn.and_then(|n| crate::examinations::dd_spec::nupn_order_seed(n, net.num_places()));
    let (mdd_net, inv) = dd_spec_to_ordered_mdd_net(&spec, seed.as_deref());

    let num_places = net.num_places();
    let num_transitions = net.num_transitions();

    // Build one query per *pending* tracker; remember which tracker each query
    // maps to so we can write verdicts back in order. A tracker whose predicate
    // fails to translate is left for downstream phases.
    let mut query_trackers: Vec<usize> = Vec::new();
    let mut queries: Vec<(tla_mdd::MddReachQuantifier, tla_dd::DdPredicate)> = Vec::new();
    for (i, tracker) in trackers.iter().enumerate() {
        if tracker.verdict.is_some() {
            continue;
        }
        let Some(predicate) = translate_predicate(&tracker.predicate, num_places, num_transitions)
        else {
            continue;
        };
        // Lower the atom's place indices into the permuted level coordinates so
        // it agrees with the ordered net (fail-closed if any index is out of
        // range — same "leave the tracker for downstream phases" behavior as a
        // failed translate). Kept in lockstep with `query_trackers` below.
        let Some(predicate) = permute_dd_predicate(&predicate, &inv) else {
            continue;
        };
        let quantifier = match tracker.quantifier {
            PathQuantifier::EF => tla_mdd::MddReachQuantifier::Ef,
            PathQuantifier::AG => tla_mdd::MddReachQuantifier::Ag,
        };
        query_trackers.push(i);
        queries.push((quantifier, predicate));
    }
    if queries.is_empty() {
        return 0;
    }

    // Compute the worker budget: the caller's remaining deadline if any, else
    // the no-deadline floor. A non-positive remaining budget ⇒ skip (decline)
    // rather than spawn a fresh up-to-5s computation past the budget.
    let budget = match deadline {
        Some(d) => {
            let remaining = d.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return 0;
            }
            remaining
        }
        None => MDD_BUDGET,
    };

    // Run on a worker thread with the big DD stack + the deadline installed
    // INSIDE the worker (mirrors the MDD CTL harness). The saturation engine
    // takes the deadline directly and declines (fail-closed) rather than
    // overrun it; the worker boundary turns any panic into a clean DECLINE.
    let (tx, rx) = mpsc::channel();
    let mdd_net_for_thread = mdd_net.clone();
    let queries_for_thread = queries.clone();
    let inner_deadline = Some(deadline.unwrap_or_else(|| std::time::Instant::now() + MDD_BUDGET));
    let handle = thread::Builder::new()
        .name("tla-mdd-reachability".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let r = tla_mdd::evaluate_reachability_at_initial(
                &mdd_net_for_thread,
                &queries_for_thread,
                inner_deadline,
                lower_dd_predicate_to_mdd,
            );
            let _ = tx.send(r);
        });
    if handle.is_err() {
        eprintln!("Reachability: MDD fast-path thread spawn failed — using explicit pipeline");
        return 0;
    }

    // Wait at most until the budget (+ a small grace), else treat as a timeout
    // decline. The worker drops its store on the way out, so the budget is a
    // soft cap with no resource leak.
    let verdicts = match rx.recv_timeout(budget + Duration::from_millis(1500)) {
        Ok(Ok(verdicts)) => verdicts,
        Ok(Err(err)) => {
            eprintln!("Reachability: MDD fast-path declined ({err:?}) — using explicit pipeline");
            return 0;
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("Reachability: MDD fast-path exceeded budget — using explicit pipeline");
            return 0;
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!("Reachability: MDD fast-path worker panicked — using explicit pipeline");
            return 0;
        }
    };

    if verdicts.len() != query_trackers.len() {
        eprintln!(
            "Reachability: MDD fast-path returned {} verdicts for {} queries — \
             treating as failure and using explicit pipeline",
            verdicts.len(),
            query_trackers.len(),
        );
        return 0;
    }

    // The reachable set is exact (converged), so each EF/AG verdict is ground
    // truth: commit it like a completed exploration (first-writer-wins).
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

    /// High-bound conserved shuttle: N tokens shuttle between p0 and p1. The BDD
    /// twin must bit-blast `ceil(log2(N+1))` bits per place; the MDD spends one
    /// level per place. Reachable: {(N,0),(N-1,1),...,(0,N)} — N+1 markings, all
    /// with p0+p1 == N.
    fn shuttle_net(n: u64) -> PetriNet {
        PetriNet {
            name: Some("shuttle".into()),
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
            initial_marking: vec![n, 0],
        }
    }

    /// 2-counter net (two independent drains): p0->p1, p2->p3, each fires once.
    /// Reachable: the 2x2 product of {p0,p1 in {(1,0),(0,1)}} x {p2,p3 ...}.
    fn two_counter_net() -> PetriNet {
        PetriNet {
            name: Some("two-counter".into()),
            places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
            transitions: vec![
                TransitionInfo {
                    id: "a".into(),
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
                    id: "b".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(2),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(3),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0, 1, 0],
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

    /// EF & AG x cardinality, cross-checked against explicit-BFS ground truth on
    /// the conserved swap net (both a TRUE and a FALSE case for each quantifier).
    #[test]
    fn resolves_ef_and_ag_cardinality_against_brute_force() {
        let net = swap_net();
        // EF(p1 >= 1): true (reach (0,1)). EF(p0 >= 2): false (bound 1).
        // AG(p0 <= 1): true. AG(p0 >= 1): false ((0,1) has p0=0).
        let preds = [
            (
                PathQuantifier::EF,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(1),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ),
            ),
            (
                PathQuantifier::EF,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(2),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
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
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        assert_eq!(seeded, 4, "all four trackers must be MDD-resolved");
        let mut saw_true = false;
        let mut saw_false = false;
        for (tr, (q, p)) in trackers.iter().zip(preds.iter()) {
            let expected = brute_force_verdict(&net, *q, p);
            assert_eq!(
                tr.verdict,
                Some(expected),
                "{:?} of {:?}: MDD={:?} brute-force={expected}",
                q,
                p,
                tr.verdict,
            );
            assert_eq!(
                tr.resolved_by.unwrap().source,
                ReachabilityResolutionSource::Dd
            );
            saw_true |= expected;
            saw_false |= !expected;
        }
        assert!(saw_true && saw_false, "battery must be non-vacuous");
    }

    /// EF & AG x fireability, cross-checked against explicit-BFS ground truth.
    #[test]
    fn resolves_fireability_against_brute_force() {
        let net = swap_net();
        let preds = [
            // EF(fireable t01): true.
            (
                PathQuantifier::EF,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            ),
            // AG(fireable t01): false (state (0,1) can't fire t01).
            (
                PathQuantifier::AG,
                ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
            ),
            // AG(fireable t01 OR t10): always exactly one is enabled ⇒ true.
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
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        assert_eq!(seeded, 3);
        for (tr, (q, p)) in trackers.iter().zip(preds.iter()) {
            let expected = brute_force_verdict(&net, *q, p);
            assert_eq!(tr.verdict, Some(expected), "{q:?} of {p:?}");
        }
    }

    /// High-bound conserved net (the BDD twin bit-blasts; the MDD survives):
    /// EF/AG cardinality must still match explicit-BFS ground truth exactly.
    #[test]
    fn resolves_high_bound_conserved_against_brute_force() {
        let net = shuttle_net(40); // bound 40 > 16 ⇒ binary band for the BDD twin
        let preds = [
            // EF(p1 >= 40): true (the all-shuttled marking (0,40) is reachable).
            (
                PathQuantifier::EF,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(40),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ),
            ),
            // EF(p1 >= 41): false (conserved sum is 40).
            (
                PathQuantifier::EF,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(41),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ),
            ),
            // AG(p0 + p1 <= 40): true (conserved). Encoded as 40 >= p0+p1.
            (
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
                    ResolvedIntExpr::Constant(40),
                ),
            ),
            // AG(p0 >= 1): false ((0,40) has p0=0).
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
            .map(|(i, (q, p))| tracker(&format!("h{i}"), *q, p.clone()))
            .collect();
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        assert_eq!(seeded, 4);
        for (tr, (q, p)) in trackers.iter().zip(preds.iter()) {
            let expected = brute_force_verdict(&net, *q, p);
            assert_eq!(tr.verdict, Some(expected), "{q:?} of {p:?}");
        }
    }

    /// Multi-component counter net: EF/AG over the product reachable set.
    #[test]
    fn resolves_two_counter_against_brute_force() {
        let net = two_counter_net();
        let preds = [
            // EF(p1 + p3 >= 2): true (both drains fire ⇒ (0,1,0,1)).
            (
                PathQuantifier::EF,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::Constant(2),
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1), PlaceIdx(3)]),
                ),
            ),
            // AG(p1 + p3 <= 1): false (the fully-drained marking has 2).
            (
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(1), PlaceIdx(3)]),
                    ResolvedIntExpr::Constant(1),
                ),
            ),
            // AG(p0 + p1 <= 1): true (first component is conserved at 1).
            (
                PathQuantifier::AG,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
                    ResolvedIntExpr::Constant(1),
                ),
            ),
        ];
        let mut trackers: Vec<PropertyTracker> = preds
            .iter()
            .enumerate()
            .map(|(i, (q, p))| tracker(&format!("tc{i}"), *q, p.clone()))
            .collect();
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        assert_eq!(seeded, 3);
        for (tr, (q, p)) in trackers.iter().zip(preds.iter()) {
            let expected = brute_force_verdict(&net, *q, p);
            assert_eq!(tr.verdict, Some(expected), "{q:?} of {p:?}");
        }
    }

    /// Cross-validation: where the BDD twin also decides, the MDD verdict MUST
    /// equal the BDD verdict (the two exact lanes must never disagree).
    #[cfg(feature = "dd-backend")]
    #[test]
    fn agrees_with_bdd_twin_where_both_decide() {
        let nets = [swap_net(), shuttle_net(8), two_counter_net()];
        for net in &nets {
            let preds: Vec<(PathQuantifier, ResolvedPredicate)> = vec![
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
                        ResolvedIntExpr::Constant(0),
                    ),
                ),
                (
                    PathQuantifier::EF,
                    ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
                ),
                (
                    PathQuantifier::AG,
                    ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]),
                ),
            ];
            let mut mdd_trackers: Vec<PropertyTracker> = preds
                .iter()
                .enumerate()
                .map(|(i, (q, p))| tracker(&format!("m{i}"), *q, p.clone()))
                .collect();
            let mut bdd_trackers = mdd_trackers.clone();
            let mdd_seeded = run_mdd_reachability_seeding(net, &mut mdd_trackers, None, None);
            let bdd_seeded =
                super::super::dd_fastpath::run_dd_reachability_seeding(net, &mut bdd_trackers);
            assert_eq!(
                mdd_seeded, bdd_seeded,
                "both exact lanes must resolve the same trackers on this net",
            );
            for (m, b) in mdd_trackers.iter().zip(bdd_trackers.iter()) {
                assert_eq!(
                    m.verdict, b.verdict,
                    "MDD and BDD exact lanes disagree on {}: MDD={:?} BDD={:?}",
                    m.id, m.verdict, b.verdict,
                );
            }
        }
    }

    #[test]
    fn declines_unbounded_net_leaves_trackers_pending() {
        // Source transition with no input → p0 unbounded → LP unbounded →
        // build_sound_dd_spec returns None → MDD declines, nothing resolved.
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
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        assert_eq!(seeded, 0, "unbounded net must make the MDD path decline");
        assert_eq!(trackers[0].verdict, None, "tracker must remain pending");
    }

    #[test]
    fn skips_already_resolved_trackers() {
        let net = swap_net();
        let mut trackers = vec![tracker("pre", PathQuantifier::EF, ResolvedPredicate::True)];
        trackers[0].verdict = Some(false); // pre-seeded (intentionally wrong-looking)
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        assert_eq!(seeded, 0, "already-resolved trackers are not touched");
        assert_eq!(
            trackers[0].verdict,
            Some(false),
            "first-writer-wins: MDD must not overwrite an existing verdict"
        );
    }

    #[test]
    fn kill_switch_disables_lane() {
        let net = swap_net();
        crate::env_guard::set_var("TY_MCC_ENABLE_MDD_REACHABILITY", "0");
        let mut trackers = vec![tracker(
            "k0",
            PathQuantifier::EF,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ),
        )];
        let seeded = run_mdd_reachability_seeding(&net, &mut trackers, None, None);
        crate::env_guard::remove_var("TY_MCC_ENABLE_MDD_REACHABILITY");
        assert_eq!(seeded, 0, "falsy kill-switch must disable the lane");
        assert_eq!(trackers[0].verdict, None, "disabled lane resolves nothing");
    }

    #[test]
    fn default_on_when_flag_unset() {
        // No env var set ⇒ lane is ON (default). (Defensive: remove first in
        // case a sibling test leaked it; the kill-switch test removes it too.)
        crate::env_guard::remove_var("TY_MCC_ENABLE_MDD_REACHABILITY");
        assert!(
            !mdd_reachability_disabled(),
            "lane must default to ON when the flag is unset",
        );
    }
}
