// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Divisor-positivity side-conditions for `\div`/`%` with non-literal
//! divisors in the CHC/PDR lane.
//!
//! Contract under test (see `lower_div_mod` in `translation.rs` and
//! `finalize_query_clauses` in `builder.rs`):
//! - literal positive divisor: translates, NO side-condition;
//! - non-literal current-state divisor in Next/Safety: translates, records a
//!   `divisor > 0` side-condition that is conjoined into the safety query;
//! - literal non-positive divisor: hard decline (TLC always errors);
//! - non-literal divisor in Init, or a primed divisor: decline (a
//!   reachable-state side-condition cannot cover TLC's candidate-assignment
//!   enumeration there);
//! - an `Unsafe` answer on the augmented query is only surfaced when the
//!   concrete replay confirms the ORIGINAL property is violated — otherwise
//!   it is downgraded to `Unknown` (never a false `Unsafe`, and side-condition
//!   failures never produce `Safe`).

use std::collections::HashMap;

use ay_chc::{ChcExpr, ChcSort, ChcVar};

use super::helpers::*;
use super::*;
use crate::chc::replay::TraceReplayInputs;
use crate::chc::result::PdrState;
use crate::error::AYError;
use crate::TlaSort;

fn int_var(name: &str) -> ChcExpr {
    ChcExpr::var(ChcVar::new(name, ChcSort::Int))
}

// ---------------------------------------------------------------------------
// Translation-level contract
// ---------------------------------------------------------------------------

#[test]
fn test_literal_positive_divisor_translates_without_side_conditions() {
    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    let next = and_expr(
        eq_expr(prime_expr("q"), div_expr(var_expr("x"), int_expr(3))),
        eq_expr(prime_expr("x"), mod_expr(var_expr("x"), int_expr(7))),
    );
    trans.add_next(&next).unwrap();
    assert!(
        trans.side_conditions.is_empty(),
        "literal positive divisors must take the fast path (no side-conditions), got {:?}",
        trans.side_conditions
    );
}

#[test]
fn test_variable_divisor_in_next_records_side_condition() {
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    let next = and_expr(
        eq_expr(prime_expr("w"), var_expr("w")),
        eq_expr(
            prime_expr("q"),
            mod_expr(add_expr(var_expr("q"), int_expr(1)), var_expr("w")),
        ),
    );
    trans.add_next(&next).unwrap();
    assert_eq!(
        trans.side_conditions,
        vec![ChcExpr::gt(int_var("w"), ChcExpr::Int(0))],
        "a non-literal current-state divisor must record `divisor > 0`"
    );
}

#[test]
fn test_variable_divisor_side_conditions_are_deduplicated() {
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    let next = and_expr(
        eq_expr(prime_expr("w"), div_expr(var_expr("w"), var_expr("w"))),
        eq_expr(prime_expr("q"), mod_expr(var_expr("q"), var_expr("w"))),
    );
    trans.add_next(&next).unwrap();
    assert_eq!(
        trans.side_conditions.len(),
        1,
        "identical divisors must record a single side-condition, got {:?}",
        trans.side_conditions
    );
}

#[test]
fn test_variable_divisor_in_safety_records_side_condition() {
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    let safety = ge_expr(mod_expr(var_expr("q"), var_expr("w")), int_expr(0));
    trans.add_safety(&safety).unwrap();
    assert_eq!(
        trans.side_conditions,
        vec![ChcExpr::gt(int_var("w"), ChcExpr::Int(0))]
    );
}

#[test]
fn test_literal_nonpositive_divisor_declines() {
    for divisor in [0, -2] {
        let mut trans = ChcTranslator::new(&[("q", TlaSort::Int)]).unwrap();
        let next = eq_expr(prime_expr("q"), div_expr(var_expr("q"), int_expr(divisor)));
        let err = trans.add_next(&next).unwrap_err();
        assert!(
            matches!(err, AYError::UnsupportedOp(_)),
            "\\div by literal {divisor} must hard-decline, got {err:?}"
        );

        let mut trans = ChcTranslator::new(&[("q", TlaSort::Int)]).unwrap();
        let next = eq_expr(prime_expr("q"), mod_expr(var_expr("q"), int_expr(divisor)));
        let err = trans.add_next(&next).unwrap_err();
        assert!(
            matches!(err, AYError::UnsupportedOp(_)),
            "% by literal {divisor} must hard-decline, got {err:?}"
        );
    }
}

#[test]
fn test_variable_divisor_in_init_declines() {
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    let init = and_expr(
        eq_expr(var_expr("w"), int_expr(3)),
        eq_expr(var_expr("q"), mod_expr(int_expr(5), var_expr("w"))),
    );
    let err = trans.add_init(&init).unwrap_err();
    assert!(
        matches!(err, AYError::UnsupportedOp(_)),
        "non-literal divisor in Init must decline, got {err:?}"
    );
}

#[test]
fn test_primed_divisor_in_next_declines() {
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    let next = and_expr(
        eq_expr(prime_expr("w"), var_expr("w")),
        eq_expr(prime_expr("q"), mod_expr(var_expr("q"), prime_expr("w"))),
    );
    let err = trans.add_next(&next).unwrap_err();
    assert!(
        matches!(err, AYError::UnsupportedOp(_)),
        "primed divisor must decline, got {err:?}"
    );
}

#[test]
fn test_side_conditions_augment_the_query_clause() {
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    trans
        .add_init(&and_expr(
            eq_expr(var_expr("w"), int_expr(3)),
            eq_expr(var_expr("q"), int_expr(0)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            eq_expr(prime_expr("w"), var_expr("w")),
            eq_expr(
                prime_expr("q"),
                mod_expr(add_expr(var_expr("q"), int_expr(1)), var_expr("w")),
            ),
        ))
        .unwrap();
    trans
        .add_safety(&lt_expr(var_expr("q"), int_expr(3)))
        .unwrap();

    // Query materialization is deferred until finalize/into_problem.
    assert_eq!(
        trans.problem.clauses().len(),
        2,
        "query clause must be deferred until finalize"
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
        query.contains("Gt") && query.contains('w'),
        "query clause must conjoin the `w > 0` side-condition, got {query}"
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

/// Init: w = 3 ∧ q = 0; Next: w' = w ∧ q' = q + 1; Safety: q < 2;
/// side-condition: w > 0.
fn replay_inputs() -> TraceReplayInputs {
    let w = || int_var("w");
    let q = || int_var("q");
    let wp = || ChcExpr::var(ChcVar::new("w", ChcSort::Int).primed());
    let qp = || ChcExpr::var(ChcVar::new("q", ChcSort::Int).primed());
    TraceReplayInputs {
        init_constraints: vec![ChcExpr::and(
            ChcExpr::eq(w(), ChcExpr::Int(3)),
            ChcExpr::eq(q(), ChcExpr::Int(0)),
        )],
        next_constraints: vec![ChcExpr::and(
            ChcExpr::eq(wp(), w()),
            ChcExpr::eq(qp(), ChcExpr::add(q(), ChcExpr::Int(1))),
        )],
        safety_constraints: vec![ChcExpr::lt(q(), ChcExpr::Int(2))],
        side_conditions: vec![ChcExpr::gt(w(), ChcExpr::Int(0))],
        state_var_names: vec!["w".to_string(), "q".to_string()],
        canonical_arg_names: vec!["__p0_a0".to_string(), "__p0_a1".to_string()],
    }
}

#[test]
fn test_replay_confirms_ay_shaped_trace() {
    // AY PDR traces name the initial step with canonical `__p{pred}_a{k}`
    // arguments and the transition steps with `x` (pre) / `x'` (post)
    // clause-local names (plus unconstrained canonical leftovers). The
    // normalizer must reassemble the genuine violating run from that shape.
    let trace = vec![
        state(&[("__p0_a0", 3), ("__p0_a1", 0)]),
        state(&[
            ("__p0_a0", 0),
            ("__p0_a1", 0),
            ("w", 3),
            ("q", 0),
            ("w'", 3),
            ("q'", 1),
        ]),
        state(&[
            ("__p0_a0", 0),
            ("__p0_a1", 0),
            ("w", 3),
            ("q", 1),
            ("w'", 3),
            ("q'", 2),
        ]),
    ];
    assert!(replay_inputs().cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_rejects_run_that_only_violates_a_side_condition() {
    // Init: w = 0 ∧ q = 0 (holds!); Next: w' = w ∧ q' = q + 1 (holds!);
    // final state violates the original safety q < 2 — but the divisor
    // side-condition `w > 0` fails along the run, so this trace would let a
    // division error masquerade as a violation. It must NOT be confirmed.
    let w = || int_var("w");
    let q = || int_var("q");
    let wp = || ChcExpr::var(ChcVar::new("w", ChcSort::Int).primed());
    let qp = || ChcExpr::var(ChcVar::new("q", ChcSort::Int).primed());
    let inputs = TraceReplayInputs {
        init_constraints: vec![ChcExpr::and(
            ChcExpr::eq(w(), ChcExpr::Int(0)),
            ChcExpr::eq(q(), ChcExpr::Int(0)),
        )],
        next_constraints: vec![ChcExpr::and(
            ChcExpr::eq(wp(), w()),
            ChcExpr::eq(qp(), ChcExpr::add(q(), ChcExpr::Int(1))),
        )],
        safety_constraints: vec![ChcExpr::lt(q(), ChcExpr::Int(2))],
        side_conditions: vec![ChcExpr::gt(w(), ChcExpr::Int(0))],
        state_var_names: vec!["w".to_string(), "q".to_string()],
        canonical_arg_names: vec!["__p0_a0".to_string(), "__p0_a1".to_string()],
    };
    let trace = vec![
        state(&[("w", 0), ("q", 0)]),
        state(&[("w", 0), ("q", 1)]),
        state(&[("w", 0), ("q", 2)]),
    ];
    assert!(!inputs.cex_witnesses_original_violation(&trace));

    // Sanity: with the side-condition satisfied instead (w > -1), the same
    // run IS a confirmable genuine violation — isolating the rejection cause.
    let confirmable = TraceReplayInputs {
        side_conditions: vec![ChcExpr::gt(w(), ChcExpr::Int(-1))],
        ..inputs
    };
    let trace = vec![
        state(&[("w", 0), ("q", 0)]),
        state(&[("w", 0), ("q", 1)]),
        state(&[("w", 0), ("q", 2)]),
    ];
    assert!(confirmable.cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_confirms_genuine_violation() {
    let trace = vec![
        state(&[("w", 3), ("q", 0)]),
        state(&[("w", 3), ("q", 1)]),
        state(&[("w", 3), ("q", 2)]),
    ];
    assert!(replay_inputs().cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_rejects_side_condition_violation() {
    // Same shape, but the divisor side-condition fails mid-trace: the CEX
    // may witness a would-be TLC division error, so it must NOT be confirmed.
    let trace = vec![
        state(&[("w", 3), ("q", 0)]),
        state(&[("w", 0), ("q", 1)]),
        state(&[("w", 0), ("q", 2)]),
    ];
    assert!(!replay_inputs().cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_rejects_trace_that_satisfies_original_safety() {
    // Final state satisfies the original property (q < 2): the CEX can only
    // have hit the augmented obligation some other way — not confirmable.
    let trace = vec![state(&[("w", 3), ("q", 0)]), state(&[("w", 3), ("q", 1)])];
    assert!(!replay_inputs().cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_rejects_non_transition() {
    // q jumps by 2: not a real step of the translated Next relation.
    let trace = vec![state(&[("w", 3), ("q", 0)]), state(&[("w", 3), ("q", 2)])];
    assert!(!replay_inputs().cex_witnesses_original_violation(&trace));
}

#[test]
fn test_replay_rejects_missing_assignment() {
    let trace = vec![state(&[("w", 3)])];
    assert!(!replay_inputs().cex_witnesses_original_violation(&trace));
}

// ---------------------------------------------------------------------------
// End-to-end PDR solves
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_pdr_variable_divisor_provably_positive_never_unsafe() {
    // Init: w = 3 ∧ q = 0; Next: w' = w ∧ q' = (q+1) % w; Safety: 0 <= q <= 3.
    // The divisor w is invariantly 3, so every division is well-defined and
    // the property holds: the sound results are Safe (if the solver can
    // reason about the variable-divisor mod term) or Unknown — never Unsafe.
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    trans
        .add_init(&and_expr(
            eq_expr(var_expr("w"), int_expr(3)),
            eq_expr(var_expr("q"), int_expr(0)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            eq_expr(prime_expr("w"), var_expr("w")),
            eq_expr(
                prime_expr("q"),
                mod_expr(add_expr(var_expr("q"), int_expr(1)), var_expr("w")),
            ),
        ))
        .unwrap();
    trans
        .add_safety(&and_expr(
            ge_expr(var_expr("q"), int_expr(0)),
            le_expr(var_expr("q"), int_expr(3)),
        ))
        .unwrap();
    let result = trans.solve_pdr(pdr_test_config()).unwrap();
    match result {
        PdrCheckResult::Safe { .. } | PdrCheckResult::Unknown { .. } => {}
        PdrCheckResult::Unsafe { trace } => {
            panic!("false Unsafe on a safe variable-divisor spec: {trace:?}")
        }
    }
}

#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_pdr_variable_divisor_reaches_zero_never_safe() {
    // Init: w = 0 ∧ q = 0; Next: w' = w ∧ q' = (q+1) % w; Safety: q >= 0.
    // TLC would ERROR on the very first step (divisor 0). The divisor
    // side-condition `w > 0` fails in the initial state, so the augmented
    // property is falsifiable: PDR must NOT return Safe. And because any
    // counterexample trace has w = 0, the replay cannot confirm a genuine
    // violation of the original property, so Unsafe must be downgraded:
    // the only acceptable outcome is Unknown.
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    trans
        .add_init(&and_expr(
            eq_expr(var_expr("w"), int_expr(0)),
            eq_expr(var_expr("q"), int_expr(0)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            eq_expr(prime_expr("w"), var_expr("w")),
            eq_expr(
                prime_expr("q"),
                mod_expr(add_expr(var_expr("q"), int_expr(1)), var_expr("w")),
            ),
        ))
        .unwrap();
    trans
        .add_safety(&ge_expr(var_expr("q"), int_expr(0)))
        .unwrap();
    let result = trans.solve_pdr(pdr_test_config()).unwrap();
    match result {
        PdrCheckResult::Unknown { .. } => {}
        PdrCheckResult::Safe { invariant } => {
            panic!("false Safe where TLC would raise a division error: {invariant}")
        }
        PdrCheckResult::Unsafe { trace } => {
            panic!("division-error CEX must be downgraded to Unknown, got Unsafe: {trace:?}")
        }
    }
}

#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_pdr_variable_divisor_genuine_violation_not_masked() {
    // Init: w = 3 ∧ q = 0; Next: w' = w ∧ q' = q + 1;
    // Safety: q < 2 ∧ (q % w) >= 0.
    // The divisor stays positive, every division is well-defined, and the
    // original property is genuinely violated at q = 2. The replay can
    // confirm such a trace, so Unsafe may be surfaced; Safe would be wrong.
    let mut trans = ChcTranslator::new(&[("w", TlaSort::Int), ("q", TlaSort::Int)]).unwrap();
    trans
        .add_init(&and_expr(
            eq_expr(var_expr("w"), int_expr(3)),
            eq_expr(var_expr("q"), int_expr(0)),
        ))
        .unwrap();
    trans
        .add_next(&and_expr(
            eq_expr(prime_expr("w"), var_expr("w")),
            eq_expr(prime_expr("q"), add_expr(var_expr("q"), int_expr(1))),
        ))
        .unwrap();
    trans
        .add_safety(&and_expr(
            lt_expr(var_expr("q"), int_expr(2)),
            ge_expr(mod_expr(var_expr("q"), var_expr("w")), int_expr(0)),
        ))
        .unwrap();
    let result = trans.solve_pdr(pdr_test_config()).unwrap();
    match result {
        PdrCheckResult::Unsafe { .. } | PdrCheckResult::Unknown { .. } => {}
        PdrCheckResult::Safe { invariant } => {
            panic!("false Safe on a genuinely violated spec: {invariant}")
        }
    }
}
