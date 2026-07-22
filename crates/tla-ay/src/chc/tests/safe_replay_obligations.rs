// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Consumer-boundary pin for `ay_chc::engines::chc_safe_replay_obligations`,
//! ty's Safe-certificate replay entry point.
//!
//! An empty-model Safe from ay's acyclic-exhaustive BMC lane is a complete
//! bounded-search proof with NO per-predicate invariants. The plain
//! `InvariantModel::replay_obligations` exporter hard-errors on that shape
//! ("missing invariant interpretation ..."), which historically demoted
//! genuinely-proved SAFEs downstream. `chc_safe_replay_obligations`
//! independently re-validates exactly that class via the same deterministic
//! exhaustive re-run the discharge gate trusts and returns an EMPTY
//! obligation set; every other shape defers to the fail-closed exporter.
//!
//! These tests pin that contract against the ay revision ty resolves, so a
//! future ay bump cannot silently regress the Safe replay boundary.

use ay_chc::{
    engines, ChcExpr, ChcProblem, ChcSort, ChcVar, ClauseBody, ClauseHead, HornClause,
    InvariantModel,
};

/// `x = 0 -> P(x)`, `P(x) /\ y = x + 1 -> Q(y)`, plus a query clause on `Q`.
/// Acyclic two-predicate DAG over Int — the acyclic-exhaustive class.
fn acyclic_two_pred_problem(query: ChcExpr) -> ChcProblem {
    let mut problem = ChcProblem::new();
    let p = problem.declare_predicate("p", vec![ChcSort::Int]);
    let q = problem.declare_predicate("q", vec![ChcSort::Int]);
    let x = ChcVar::new("x", ChcSort::Int);
    let y = ChcVar::new("y", ChcSort::Int);
    problem.add_clause(HornClause::new(
        ClauseBody::new(
            vec![],
            Some(ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(0))),
        ),
        ClauseHead::Predicate(p, vec![ChcExpr::var(x.clone())]),
    ));
    problem.add_clause(HornClause::new(
        ClauseBody::new(
            vec![(p, vec![ChcExpr::var(x.clone())])],
            Some(ChcExpr::eq(
                ChcExpr::var(y.clone()),
                ChcExpr::add(ChcExpr::var(x), ChcExpr::int(1)),
            )),
        ),
        ClauseHead::Predicate(q, vec![ChcExpr::var(y.clone())]),
    ));
    problem.add_clause(HornClause::query(ClauseBody::new(
        vec![(q, vec![ChcExpr::var(y)])],
        Some(query),
    )));
    problem
}

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn empty_model_acyclic_exhaustive_safe_exports_empty_obligation_set() {
    // Query `y < 0` is unreachable (y is exactly 1): Safe by exhaustion,
    // proof carries no invariants. The helper must re-validate and return
    // an empty set instead of the "missing invariant interpretation" error.
    let y = ChcVar::new("y", ChcSort::Int);
    let problem = acyclic_two_pred_problem(ChcExpr::lt(ChcExpr::var(y), ChcExpr::int(0)));

    let obligations = engines::chc_safe_replay_obligations(&problem, &InvariantModel::new())
        .expect("empty-model acyclic-exhaustive Safe must export obligations");
    assert!(
        obligations.is_empty(),
        "exhaustion proofs have no invariant obligations; got {}",
        obligations.len()
    );
}

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn empty_model_on_reachable_error_stays_fail_closed() {
    // Query `y > 0` IS reachable (y = 1): the exhaustive re-run refuses to
    // confirm, so the standard exporter's fail-closed error is preserved —
    // the shortcut must never manufacture a certificate for a non-proof.
    let y = ChcVar::new("y", ChcSort::Int);
    let problem = acyclic_two_pred_problem(ChcExpr::gt(ChcExpr::var(y), ChcExpr::int(0)));

    let error = engines::chc_safe_replay_obligations(&problem, &InvariantModel::new())
        .expect_err("reachable-error problem must not receive the empty-obligation shortcut");
    assert!(
        error
            .to_string()
            .contains("missing invariant interpretation"),
        "fail-closed exporter error must be preserved; got: {error}"
    );
}
