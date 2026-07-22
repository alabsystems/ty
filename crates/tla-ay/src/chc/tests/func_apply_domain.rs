// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Domain-membership side-conditions for `f[i]` with non-literal indices
//! (Finding 14).
//!
//! The symbolic-index encoding of `f[i]` is an ITE chain over the finite
//! domain keys whose innermost else-branch is the LAST key's element. With
//! no guard that silently totalizes the partial TLA+ function: an
//! out-of-domain index evaluates to `f[last_key]` instead of being a TLC
//! error — producing false Safe verdicts (and bogus counterexamples).
//!
//! Contract under test (mirrors `lower_div_mod`; see
//! `translate_func_apply_value` in `translation.rs`):
//! - literal in-domain index: translates, NO side-condition;
//! - literal OUT-of-domain index: hard decline (TLC always errors);
//! - non-literal current-state index in Next/Safety: translates, records the
//!   membership side-condition `i ∈ {domain keys}` (a disjunction of
//!   per-key equalities) that is conjoined into the safety query;
//! - non-literal index in Init, or a primed index: decline;
//! - an `Unsafe` answer on the augmented query is only surfaced when the
//!   concrete replay confirms the ORIGINAL property is violated along a run
//!   where the index stays in-domain at every state.

use std::collections::HashMap;

use ay_chc::{ChcExpr, ChcSort, ChcVar};
use tla_core::ast::Expr;
use tla_core::Spanned;

use super::helpers::*;
use super::*;
use crate::chc::replay::TraceReplayInputs;
use crate::chc::result::PdrState;
use crate::error::AYError;
use crate::TlaSort;

fn int_var(name: &str) -> ChcExpr {
    ChcExpr::var(ChcVar::new(name, ChcSort::Int))
}

/// `f : [1..2 -> Int]` alongside an Int counter `x`.
fn translator_with_f_and_x() -> ChcTranslator {
    ChcTranslator::new(&[
        (
            "f",
            TlaSort::Function {
                domain_keys: vec!["1".to_string(), "2".to_string()],
                range: Box::new(TlaSort::Int),
            },
        ),
        ("x", TlaSort::Int),
    ])
    .unwrap()
}

/// The recorded membership obligation for an index expression `idx`:
/// `idx = 1 \/ idx = 2`.
fn membership_side_condition(idx: ChcExpr) -> ChcExpr {
    ChcExpr::or(
        ChcExpr::eq(idx.clone(), ChcExpr::Int(1)),
        ChcExpr::eq(idx, ChcExpr::Int(2)),
    )
}

// ---------------------------------------------------------------------------
// Translation-level contract
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_literal_in_domain_index_translates_without_side_condition() {
    let mut trans = translator_with_f_and_x();
    let safety = gt_expr(func_apply_expr(var_expr("f"), int_expr(1)), int_expr(0));
    trans.add_safety(&safety).unwrap();
    assert!(
        trans.side_conditions.is_empty(),
        "a literal in-domain index must take the fast path (no side-conditions), got {:?}",
        trans.side_conditions
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_literal_out_of_domain_index_declines() {
    let mut trans = translator_with_f_and_x();
    let safety = gt_expr(func_apply_expr(var_expr("f"), int_expr(5)), int_expr(0));
    let err = trans.add_safety(&safety).unwrap_err();
    match err {
        AYError::UnsupportedOp(msg) => assert!(
            msg.contains("outside the function's finite domain"),
            "out-of-domain literal index must hard-decline with cause, got: {msg}"
        ),
        other => panic!("expected UnsupportedOp decline, got {other:?}"),
    }

    // A literal of the wrong kind (bool index on an int domain) is equally
    // out-of-domain and must also decline — never totalize.
    let mut trans = translator_with_f_and_x();
    let safety = gt_expr(func_apply_expr(var_expr("f"), bool_expr(true)), int_expr(0));
    let err = trans.add_safety(&safety).unwrap_err();
    assert!(
        matches!(err, AYError::UnsupportedOp(_)),
        "bool literal index on int domain must decline, got {err:?}"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symbolic_index_in_safety_records_membership_side_condition() {
    let mut trans = translator_with_f_and_x();
    let safety = gt_expr(func_apply_expr(var_expr("f"), var_expr("x")), int_expr(0));
    trans.add_safety(&safety).unwrap();
    assert_eq!(
        trans.side_conditions,
        vec![membership_side_condition(int_var("x"))],
        "a non-literal current-state index must record `x = 1 \\/ x = 2`"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symbolic_index_in_next_records_membership_side_condition() {
    let mut trans = translator_with_f_and_x();
    let next = and_expr(
        Spanned::dummy(Expr::Unchanged(Box::new(var_expr("f")))),
        eq_expr(
            prime_expr("x"),
            func_apply_expr(var_expr("f"), var_expr("x")),
        ),
    );
    trans.add_next(&next).unwrap();
    assert_eq!(
        trans.side_conditions,
        vec![membership_side_condition(int_var("x"))],
        "an unprimed index in Next must record the membership obligation"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symbolic_index_side_conditions_are_deduplicated() {
    let mut trans = translator_with_f_and_x();
    let safety = and_expr(
        gt_expr(func_apply_expr(var_expr("f"), var_expr("x")), int_expr(0)),
        lt_expr(func_apply_expr(var_expr("f"), var_expr("x")), int_expr(9)),
    );
    trans.add_safety(&safety).unwrap();
    assert_eq!(
        trans.side_conditions.len(),
        1,
        "identical index obligations must record a single side-condition, got {:?}",
        trans.side_conditions
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_symbolic_index_in_init_declines() {
    let mut trans = translator_with_f_and_x();
    let init = gt_expr(func_apply_expr(var_expr("f"), var_expr("x")), int_expr(0));
    let err = trans.add_init(&init).unwrap_err();
    match err {
        AYError::UnsupportedOp(msg) => assert!(
            msg.contains("Init"),
            "Init decline must name the context, got: {msg}"
        ),
        other => panic!("non-constant index in Init must decline, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_primed_index_in_next_declines() {
    let mut trans = translator_with_f_and_x();
    let next = and_expr(
        Spanned::dummy(Expr::Unchanged(Box::new(var_expr("f")))),
        and_expr(
            eq_expr(prime_expr("x"), var_expr("x")),
            gt_expr(func_apply_expr(var_expr("f"), prime_expr("x")), int_expr(0)),
        ),
    );
    let err = trans.add_next(&next).unwrap_err();
    match err {
        AYError::UnsupportedOp(msg) => assert!(
            msg.contains("primed"),
            "primed-index decline must name the cause, got: {msg}"
        ),
        other => panic!("primed index must decline, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_membership_side_condition_augments_the_query_clause() {
    let mut trans = translator_with_f_and_x();
    let domain = range_expr(int_expr(1), int_expr(2));
    trans
        .add_init(&and_expr(
            eq_expr(
                var_expr("f"),
                func_def_expr(vec![bound_var("i", domain)], int_expr(3)),
            ),
            eq_expr(var_expr("x"), int_expr(1)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            Spanned::dummy(Expr::Unchanged(Box::new(var_expr("f")))),
            eq_expr(prime_expr("x"), add_expr(var_expr("x"), int_expr(1))),
        ))
        .unwrap();
    trans
        .add_safety(&gt_expr(
            func_apply_expr(var_expr("f"), var_expr("x")),
            int_expr(0),
        ))
        .unwrap();

    assert_eq!(
        trans.side_conditions,
        vec![membership_side_condition(int_var("x"))]
    );
    let problem = trans.into_problem();
    let clauses = problem.clauses();
    assert_eq!(
        clauses.len(),
        3,
        "finalize must add exactly one query clause"
    );
    let query = format!("{:?}", clauses[2]);
    assert!(
        query.contains("Or") && query.contains('x'),
        "query clause must conjoin the membership side-condition, got {query}"
    );
}

// ---------------------------------------------------------------------------
// Counterexample replay (deterministic, no solver involved)
// ---------------------------------------------------------------------------

fn state(pairs: &[(&str, i64)]) -> PdrState {
    let assignments: HashMap<String, i64> = pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect();
    PdrState { assignments }
}

/// Init: x = 1; Next: x' = x + 1; Safety: x < 3;
/// membership side-condition: x = 1 \/ x = 2.
fn replay_inputs() -> TraceReplayInputs {
    let x = || int_var("x");
    let xp = || ChcExpr::var(ChcVar::new("x", ChcSort::Int).primed());
    TraceReplayInputs {
        init_constraints: vec![ChcExpr::eq(x(), ChcExpr::Int(1))],
        next_constraints: vec![ChcExpr::eq(xp(), ChcExpr::add(x(), ChcExpr::Int(1)))],
        safety_constraints: vec![ChcExpr::lt(x(), ChcExpr::Int(3))],
        side_conditions: vec![membership_side_condition(x())],
        state_var_names: vec!["x".to_string()],
        canonical_arg_names: vec!["__p0_a0".to_string()],
    }
}

#[test]
fn test_replay_rejects_run_leaving_the_domain() {
    // The run 1 -> 2 -> 3 violates the original safety (x < 3) only at a
    // state where the membership side-condition x ∈ {1,2} fails: the CEX may
    // witness a would-be TLC out-of-domain error, so it must NOT be
    // confirmed as a genuine violation.
    let trace = vec![state(&[("x", 1)]), state(&[("x", 2)]), state(&[("x", 3)])];
    assert!(!replay_inputs().cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_confirms_in_domain_violation() {
    // With the domain widened to {1,2,3} the same run stays in-domain at
    // every state and genuinely violates x < 3 — confirmable. This isolates
    // the membership check as the rejection cause above.
    let x = || int_var("x");
    let widened = TraceReplayInputs {
        side_conditions: vec![ChcExpr::or(
            membership_side_condition(x()),
            ChcExpr::eq(x(), ChcExpr::Int(3)),
        )],
        ..replay_inputs()
    };
    let trace = vec![state(&[("x", 1)]), state(&[("x", 2)]), state(&[("x", 3)])];
    assert!(widened.cex_witnesses_original_violation(&trace));
}

// ---------------------------------------------------------------------------
// End-to-end PDR solves
// ---------------------------------------------------------------------------

/// OodApply2 (Finding 14 trigger): `f = [i \in 1..2 |-> 3]`, x increments
/// from 1, `Inv == f[x+0] > 0`. TLC errors at x = 3 (out-of-domain
/// application). The old encoding totalized `f[3]` to `f[2] = 3 > 0` and
/// proved a false Safe. With the membership obligation the augmented
/// property is falsifiable (x reaches 3), so Safe is impossible; and any
/// counterexample leaves the domain, so the replay cannot confirm a genuine
/// violation and Unsafe must be downgraded: the only sound outcome is
/// Unknown.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_pdr_out_of_domain_apply_only_unknown() {
    let mut trans = translator_with_f_and_x();
    let domain = range_expr(int_expr(1), int_expr(2));
    trans
        .add_init(&and_expr(
            eq_expr(
                var_expr("f"),
                func_def_expr(vec![bound_var("i", domain)], int_expr(3)),
            ),
            eq_expr(var_expr("x"), int_expr(1)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            Spanned::dummy(Expr::Unchanged(Box::new(var_expr("f")))),
            eq_expr(prime_expr("x"), add_expr(var_expr("x"), int_expr(1))),
        ))
        .unwrap();
    trans
        .add_safety(&gt_expr(
            func_apply_expr(var_expr("f"), add_expr(var_expr("x"), int_expr(0))),
            int_expr(0),
        ))
        .unwrap();
    let result = trans.solve_pdr(pdr_test_config()).unwrap();
    match result {
        PdrCheckResult::Unknown { .. } => {}
        PdrCheckResult::Safe { invariant } => {
            panic!("false Safe where TLC raises an out-of-domain error: {invariant}")
        }
        PdrCheckResult::Unsafe { trace } => {
            panic!("out-of-domain CEX must be downgraded to Unknown, got Unsafe: {trace:?}")
        }
    }
}

/// In-domain-provable control: x toggles 1 <-> 2 (x' = 3 - x), so the
/// membership obligation holds invariantly and `f[x] > 0` genuinely holds.
/// Sound outcomes are Safe (if PDR converges) or Unknown — never Unsafe.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_pdr_in_domain_symbolic_index_never_unsafe() {
    let mut trans = translator_with_f_and_x();
    let domain = range_expr(int_expr(1), int_expr(2));
    trans
        .add_init(&and_expr(
            eq_expr(
                var_expr("f"),
                func_def_expr(vec![bound_var("i", domain)], int_expr(3)),
            ),
            eq_expr(var_expr("x"), int_expr(1)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            Spanned::dummy(Expr::Unchanged(Box::new(var_expr("f")))),
            eq_expr(prime_expr("x"), sub_expr(int_expr(3), var_expr("x"))),
        ))
        .unwrap();
    trans
        .add_safety(&gt_expr(
            func_apply_expr(var_expr("f"), var_expr("x")),
            int_expr(0),
        ))
        .unwrap();
    let result = trans.solve_pdr(pdr_test_config()).unwrap();
    match result {
        PdrCheckResult::Safe { .. } | PdrCheckResult::Unknown { .. } => {}
        PdrCheckResult::Unsafe { trace } => {
            panic!("false Unsafe on an in-domain symbolic-index spec: {trace:?}")
        }
    }
}
