// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the Liu-Smolka local EDG CTL solver.
//!
//! Covers the full 8-operator CTL surface:
//! - EF reachability (positive & negative).
//! - AG safety (positive & negative).
//! - EX / AX one-step (positive & negative, plus deadlock vacuity).
//! - EU until (positive).
//! - EG greatest fixpoint via Dalsgaard certain-zero closure
//!   (safety-path witness, no-safe-path, deadlock holds-when-psi-true,
//!   deadlock violates-when-psi-false).
//! - AF via duality `Not (EG (Not psi))`
//!   (eventually-psi, some-path-misses).
//! - AF/EG duality cross-check on a battery of psi values.
//! - Node-cap abort path.
//! - Differential agreement with the full-graph
//!   `tla-mc-core::ctl::CtlEngine` on a representative small net for
//!   every operator including EG and AF.

use super::super::local_edg::{solve_local_edg, EdgAbort, LocalEdgSolver};
use super::super::resolve::resolve_ctl_with_aliases;
use super::support::{atom_pred, simple_net, suffix_cycle_net, suffix_exit_net};

use crate::examinations::ctl::checker::CtlChecker;
use crate::explorer::{explore_full, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::{CtlFormula, IntExpr, StatePredicate};

// ---- small helpers -----------------------------------------------------

fn config() -> ExplorationConfig {
    ExplorationConfig::new(10_000)
}

fn ef(inner: CtlFormula) -> CtlFormula {
    CtlFormula::EF(Box::new(inner))
}

fn ag(inner: CtlFormula) -> CtlFormula {
    CtlFormula::AG(Box::new(inner))
}

fn ex(inner: CtlFormula) -> CtlFormula {
    CtlFormula::EX(Box::new(inner))
}

fn ax(inner: CtlFormula) -> CtlFormula {
    CtlFormula::AX(Box::new(inner))
}

fn eu(phi: CtlFormula, psi: CtlFormula) -> CtlFormula {
    CtlFormula::EU(Box::new(phi), Box::new(psi))
}

fn eg(inner: CtlFormula) -> CtlFormula {
    CtlFormula::EG(Box::new(inner))
}

fn af(inner: CtlFormula) -> CtlFormula {
    CtlFormula::AF(Box::new(inner))
}

fn not(inner: CtlFormula) -> CtlFormula {
    CtlFormula::Not(Box::new(inner))
}

fn atom_eq(place: &str, value: u64) -> CtlFormula {
    // Express "tokens(place) == value" as IntLe both ways.
    CtlFormula::And(vec![
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(value),
            IntExpr::TokensCount(vec![place.to_string()]),
        )),
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::TokensCount(vec![place.to_string()]),
            IntExpr::Constant(value),
        )),
    ])
}

fn atom_ge(place: &str, value: u64) -> CtlFormula {
    CtlFormula::Atom(atom_pred(place, value))
}

/// One-state Petri net whose initial marking IS a deadlock (the only
/// transition needs `p >= 1` and the initial marking is `p = 0`).
fn deadlock_net() -> PetriNet {
    PetriNet {
        name: Some(String::from("deadlock-single")),
        places: vec![PlaceInfo {
            id: String::from("p"),
            name: None,
        }],
        transitions: vec![TransitionInfo {
            id: String::from("t"),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![0],
    }
}

// ---- the tests ---------------------------------------------------------

#[test]
fn test_local_edg_ef_reachable() {
    // simple_net: p0 starts with 2 tokens, t0 moves p0→p1.
    // EF (p1 >= 1) — reachable after one step.
    let net = simple_net();
    let formula = ef(atom_ge("p1", 1));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);

    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("EF on reachable target should not abort");
    assert!(verdict, "EF(p1 >= 1) should be TRUE: p1 reachable via t0");
}

#[test]
fn test_local_edg_ef_unreachable() {
    // simple_net never produces p1 >= 99 — t0 only ever moves one token.
    let net = simple_net();
    let formula = ef(atom_ge("p1", 99));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);

    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("EF on unreachable target should resolve to FALSE");
    assert!(!verdict, "EF(p1 >= 99) should be FALSE: never reached");
}

#[test]
fn test_local_edg_ag_safety() {
    // simple_net: at every reachable marking, p0 + p1 == 2 (token
    // conservation: t0 simply moves one). Test AG(p0 + p1 >= 0) which
    // is a tautology.
    // We express the trivially true predicate "0 <= 0" so it doesn't
    // need place sums.
    let net = simple_net();
    let formula = ag(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    )));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);

    let verdict =
        solve_local_edg(&net, &resolved, &config()).expect("AG safety on small net should resolve");
    assert!(verdict, "AG(0 <= 0) is trivially TRUE everywhere");
}

#[test]
fn test_local_edg_ag_violation() {
    // simple_net initial p0=2, after one t0 step p1=1. The predicate
    // "p1 < 1" is true initially (p1=0) but FALSE after t0 fires.
    // Therefore AG(p1 == 0) is FALSE with reachable counterexample.
    let net = simple_net();
    let formula = ag(atom_eq("p1", 0));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);

    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("AG violation on small net should resolve");
    assert!(
        !verdict,
        "AG(p1 == 0) should be FALSE: p1 becomes 1 after t0"
    );
}

#[test]
fn test_local_edg_ex_reachable() {
    let net = simple_net();
    // After firing t0 once, p1 == 1. EX(p1 == 1) at the initial state
    // should be TRUE.
    let formula = ex(atom_ge("p1", 1));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict =
        solve_local_edg(&net, &resolved, &config()).expect("EX should resolve on small net");
    assert!(verdict);
}

#[test]
fn test_local_edg_ax_violation() {
    let net = simple_net();
    // Initial p0=2,p1=0. Only successor has p1=1, so AX(p1 == 0) is
    // FALSE — every successor violates it.
    let formula = ax(atom_eq("p1", 0));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict =
        solve_local_edg(&net, &resolved, &config()).expect("AX should resolve on small net");
    assert!(!verdict);
}

#[test]
fn test_local_edg_ax_deadlock_vacuous_true() {
    // A net with no enabled transitions at the initial marking: AX
    // should be vacuously true (no successor to violate).
    let net = deadlock_net();
    let formula = ax(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::Constant(0),
    ))); // never true predicate
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("AX at deadlock should resolve vacuously");
    assert!(
        verdict,
        "AX at deadlock should be TRUE regardless of inner predicate"
    );
}

#[test]
fn test_local_edg_eu_reachable() {
    // suffix_exit_net: start -> mid -> bad. Test E[!bad U bad] at start.
    // Path: start (!bad) -> mid (!bad) -> bad (bad). EU should hold.
    let net = suffix_exit_net();
    let phi = CtlFormula::Not(Box::new(atom_ge("bad", 1)));
    let psi = atom_ge("bad", 1);
    let formula = eu(phi, psi);
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config()).expect("EU should resolve");
    assert!(verdict, "E[!bad U bad] holds in suffix_exit_net");
}

#[test]
fn test_local_edg_eg_safety_path() {
    // suffix_cycle_net: start -> {loop, bad}; loop self-loops via
    // stay_loop and can also exit to bad via exit_bad; bad is a deadlock.
    // The predicate `not bad` holds at start (bad=0) and at loop (bad=0)
    // but fails at bad (bad=1). The path start -> loop -> loop -> ...
    // keeps `not bad` true forever via the self-loop, so EG (not bad)
    // is TRUE at start. The full-graph engine agrees (verified below
    // in the differential test).
    let net = suffix_cycle_net();
    let formula = eg(not(atom_ge("bad", 1)));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("EG (not bad) should resolve on suffix_cycle_net");
    assert!(
        verdict,
        "EG (not bad) must be TRUE: start -> loop -> loop -> ... keeps !bad"
    );
}

#[test]
fn test_local_edg_eg_no_safe_path() {
    // suffix_exit_net: start -> mid -> bad, no cycles. EG (start >= 1)
    // is FALSE everywhere reachable beyond the initial state -- once
    // we step to `mid`, start=0 and there is no way back. Therefore
    // at the initial state EG must be FALSE: the only successor is
    // `mid`, and `mid` has start=0 (so EG (start>=1) at mid is FALSE),
    // so no successor witness keeps the path inside the predicate.
    let net = suffix_exit_net();
    let formula = eg(atom_ge("start", 1));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("EG on linear path should resolve to FALSE");
    assert!(
        !verdict,
        "EG (start >= 1) must be FALSE: linear path loses start"
    );
}

#[test]
fn test_local_edg_eg_deadlock_holds_when_psi_true() {
    // deadlock_net's only state is a deadlock with p = 0. Under MCC
    // maximal-path semantics (matching the full-graph engine in
    // `tla-mc-core::ctl::CtlEngine::gfp_eg`), a deadlock satisfying
    // psi remains in the greatest fixpoint: there is no successor to
    // violate psi.
    let net = deadlock_net();
    // psi is `0 <= 0` (tautology) so it holds at the deadlock.
    let formula = eg(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(0),
        IntExpr::Constant(0),
    )));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("EG at deadlock with psi-true should resolve");
    assert!(
        verdict,
        "EG psi at deadlock where psi holds must be TRUE (maximal-path)"
    );
}

#[test]
fn test_local_edg_eg_deadlock_violates_when_psi_false() {
    // Same deadlock net, but psi is the never-true predicate `1 <= 0`.
    // EG demotes any state where psi fails directly to FALSE -- deadlock
    // or not.
    let net = deadlock_net();
    let formula = eg(CtlFormula::Atom(StatePredicate::IntLe(
        IntExpr::Constant(1),
        IntExpr::Constant(0),
    )));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("EG at deadlock with psi-false should resolve");
    assert!(!verdict, "EG psi at deadlock where psi fails must be FALSE");
}

#[test]
fn test_local_edg_af_eventually_psi() {
    // suffix_exit_net is acyclic: start -> mid -> bad (deadlock).
    // Every maximal path reaches `bad`, so AF (bad >= 1) is TRUE at
    // the initial state.
    let net = suffix_exit_net();
    let formula = af(atom_ge("bad", 1));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("AF (bad) should resolve on suffix_exit_net");
    assert!(verdict, "AF (bad) must be TRUE: every path reaches bad");
}

#[test]
fn test_local_edg_af_some_path_misses() {
    // suffix_cycle_net has a stay_loop self-loop in the loop state,
    // so the path start -> loop -> loop -> ... never reaches `bad`.
    // Therefore AF (bad >= 1) is FALSE at the initial state.
    let net = suffix_cycle_net();
    let formula = af(atom_ge("bad", 1));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);
    let verdict = solve_local_edg(&net, &resolved, &config())
        .expect("AF (bad) should resolve on suffix_cycle_net");
    assert!(
        !verdict,
        "AF (bad) must be FALSE: stay_loop path avoids bad forever"
    );
}

#[test]
fn test_local_edg_af_eg_duality() {
    // For every psi in a small battery, verify AF psi ≡ Not (EG (Not psi))
    // as produced by the EDG engine. This is the desugaring the engine
    // uses internally; the test checks that the duality is preserved
    // end-to-end (no off-by-one in the desugar walk, no operator
    // misclassification).
    let net = suffix_cycle_net();
    let aliases = PropertyAliases::identity(&net);

    let psi_battery: Vec<CtlFormula> = vec![
        atom_ge("bad", 1),
        atom_ge("loop", 1),
        atom_ge("start", 1),
        not(atom_ge("bad", 1)),
        CtlFormula::Or(vec![atom_ge("loop", 1), atom_ge("bad", 1)]),
    ];

    for (idx, psi) in psi_battery.into_iter().enumerate() {
        let af_form = af(psi.clone());
        let neg_eg_neg = not(eg(not(psi.clone())));

        let af_resolved = resolve_ctl_with_aliases(&af_form, &aliases);
        let dual_resolved = resolve_ctl_with_aliases(&neg_eg_neg, &aliases);

        let af_verdict = solve_local_edg(&net, &af_resolved, &config()).expect("AF should resolve");
        let dual_verdict = solve_local_edg(&net, &dual_resolved, &config())
            .expect("Not(EG(Not psi)) should resolve");

        assert_eq!(
            af_verdict, dual_verdict,
            "psi #{idx}: AF ({af_verdict}) != Not EG Not ({dual_verdict})"
        );
    }
}

#[test]
fn test_local_edg_node_cap_returns_cannot_compute() {
    // Construct a small but cycling net and pin the EDG node cap very
    // low so the solver must abort. AG over a non-trivial fixpoint
    // forces creation of multiple (state, AG ...) nodes.
    let net = suffix_cycle_net();
    let formula = ag(atom_ge("loop", 1));
    let aliases = PropertyAliases::identity(&net);
    let resolved = resolve_ctl_with_aliases(&formula, &aliases);

    let mut solver = LocalEdgSolver::new(&net, &config()).with_node_cap(1);
    let result = solver.solve_root(&resolved);
    assert!(
        matches!(
            result,
            Err(EdgAbort::NodeCapReached) | Err(EdgAbort::UnsupportedOperator)
        ),
        "expected NodeCapReached, got {result:?}"
    );
}

#[test]
fn test_local_edg_matches_full_graph_engine_on_small_net() {
    // Differential test against the full-graph CTL engine on a
    // representative net with both branching and cycles.
    //
    // suffix_cycle_net states:
    //   start (init)
    //   loop  (via to_loop)
    //   bad   (via to_bad_from_start or exit_bad)
    // Transitions allow loop to self-loop (stay_loop).
    //
    // We compare verdicts on a battery of operators covering the full
    // 8-operator CTL surface, including EG and AF added by this
    // commit. The full-graph engine is the soundness oracle.
    let net = suffix_cycle_net();
    let aliases = PropertyAliases::identity(&net);

    let formulas: Vec<CtlFormula> = vec![
        ef(atom_ge("bad", 1)),
        ef(atom_ge("loop", 1)),
        ef(atom_ge("start", 5)), // unreachable
        ag(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(0),
        ))),
        ag(CtlFormula::Not(Box::new(atom_ge("bad", 1)))),
        ex(atom_ge("loop", 1)),
        ex(atom_ge("bad", 1)),
        ax(atom_ge("bad", 1)),
        ax(CtlFormula::Or(vec![atom_ge("loop", 1), atom_ge("bad", 1)])),
        eu(
            CtlFormula::Not(Box::new(atom_ge("bad", 1))),
            atom_ge("bad", 1),
        ),
        // EG: greatest-fixpoint via Dalsgaard certain-zero closure.
        eg(not(atom_ge("bad", 1))), // True: loop self-loop keeps !bad.
        eg(atom_ge("bad", 1)),      // False at start (bad=0).
        eg(CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(0),
            IntExpr::Constant(0),
        ))), // True: tautology, true everywhere.
        // AF: rewritten internally as Not(EG(Not(_))).
        af(atom_ge("bad", 1)), // False: stay_loop avoids bad.
        af(CtlFormula::Or(vec![
            // True: every path ends in loop or bad.
            atom_ge("loop", 1),
            atom_ge("bad", 1),
        ])),
        // Nested: AG (EF bad) — every reachable state can still reach bad.
        ag(ef(atom_ge("bad", 1))),
    ];

    // Build the full-graph oracle once.
    let full = explore_full(&net, &config().refitted_for_full_graph(&net));
    assert!(full.graph.completed, "test net must explore fully");
    // The Petri-net checker expects the markings to already be in the original-net
    // place space; explore_full produces them directly when no reduction/slice is
    // applied — so no expansion is needed here.
    let checker = CtlChecker::new(&full, &net);

    for (idx, formula) in formulas.into_iter().enumerate() {
        let resolved = resolve_ctl_with_aliases(&formula, &aliases);
        let oracle = checker.eval(&resolved)[0];
        let edg = solve_local_edg(&net, &resolved, &config())
            .unwrap_or_else(|e| panic!("formula #{idx}: EDG aborted unexpectedly: {e:?}"));
        assert_eq!(
            edg, oracle,
            "differential mismatch on formula #{idx}: EDG={edg} oracle={oracle}"
        );
    }
}

#[test]
fn test_local_edg_pipeline_recovers_verdict_pre_fix_returned_cannot_compute() {
    // End-to-end: a tiny formula that the recursive checker can also
    // solve, but the key property is that the pipeline now routes via
    // EDG first. We assert that the pipeline emits a definite verdict
    // (no CANNOT_COMPUTE) on a net+formula combination known to need
    // the local fallback path.
    use super::super::check_ctl_properties;
    use super::super::pipeline::with_ctl_local_fallback_for_test;
    use super::support::{ctl_local_fallback_formula, ctl_local_fallback_net, make_ctl_prop};
    use crate::output::Verdict;

    let net = ctl_local_fallback_net();
    let props = vec![make_ctl_prop("edg-fallback", ctl_local_fallback_formula())];
    // Force tiny max_states so the full-graph explorer aborts and we
    // hit the local-fallback branch where EDG runs.
    let config = ExplorationConfig::new(3);
    let results =
        with_ctl_local_fallback_for_test(true, || check_ctl_properties(&net, &props, &config));
    assert_eq!(results.len(), 1);
    let (id, verdict) = &results[0];
    assert_eq!(id, "edg-fallback");
    assert!(
        matches!(verdict, Verdict::True | Verdict::False),
        "expected definite verdict from EDG/local fallback, got {verdict:?}"
    );
}
