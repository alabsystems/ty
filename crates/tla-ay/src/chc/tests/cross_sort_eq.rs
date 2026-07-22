// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-sort equality must fail closed (Finding 13).
//!
//! Per-variable sorts are heuristically inferred (one scalar sort per
//! variable from the first matching Init/TypeOK pattern, default Int), so a
//! variable can range over values the inferred sort does not cover — e.g.
//! `Init == x = N \/ x = 1` with `N` a model value (interned as String)
//! leaves `x` sorted Int. Constant-folding the cross-sort equality `x = N`
//! to FALSE would erase the reachable `x = N` states and let PDR prove a
//! false Safe for `Inv == x # N`. Contract under test (`scalar_eq` in
//! `translation.rs`): cross-sort equality DECLINES with `UnsupportedOp`
//! (the PDR lane then returns Unknown); same-sort equality still translates.

use super::helpers::*;
use super::*;
use crate::error::AYError;
use crate::TlaSort;

fn assert_cross_sort_decline(err: AYError) {
    match err {
        AYError::UnsupportedOp(msg) => {
            assert!(
                msg.contains("different inferred sorts"),
                "decline must explain the cross-sort cause, got: {msg}"
            );
        }
        other => panic!("cross-sort equality must decline with UnsupportedOp, got {other:?}"),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cross_sort_eq_in_init_declines() {
    // Init == x = N \/ x = 1, with x inferred Int and N a model value
    // (undeclared identifier, interned as a String atom).
    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    let init = or_expr(
        eq_expr(var_expr("x"), ident_expr("N")),
        eq_expr(var_expr("x"), int_expr(1)),
    );
    let err = trans.add_init(&init).unwrap_err();
    assert_cross_sort_decline(err);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cross_sort_neq_in_safety_declines() {
    // Inv == x # N: folding `x = N` to false would make the invariant
    // vacuously true — a false Safe. Must decline instead.
    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    let err = trans
        .add_safety(&ne_expr(var_expr("x"), ident_expr("N")))
        .unwrap_err();
    assert_cross_sort_decline(err);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cross_sort_string_literal_eq_declines() {
    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    let err = trans
        .add_safety(&eq_expr(var_expr("x"), string_expr("a")))
        .unwrap_err();
    assert_cross_sort_decline(err);

    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    let err = trans
        .add_safety(&eq_expr(var_expr("x"), bool_expr(true)))
        .unwrap_err();
    assert_cross_sort_decline(err);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cross_sort_set_enum_membership_declines() {
    // x \in {N}: the per-element disjunct is a scalar equality across sorts;
    // folding it to false would misread membership. Must decline.
    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    let err = trans
        .add_safety(&in_expr(
            var_expr("x"),
            set_enum_expr(vec![ident_expr("N")]),
        ))
        .unwrap_err();
    assert_cross_sort_decline(err);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_same_sort_model_value_eq_still_translates() {
    // Positive control: the decline is sort-mismatch-based, not blanket.
    // A String-sorted variable compares fine against a model value, and an
    // Int-sorted variable against an int literal.
    let mut trans = ChcTranslator::new(&[("y", TlaSort::String)]).unwrap();
    trans
        .add_safety(&ne_expr(var_expr("y"), ident_expr("N")))
        .expect("same-sort (String) model-value equality must translate");

    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    trans
        .add_safety(&ne_expr(var_expr("x"), int_expr(1)))
        .expect("same-sort (Int) equality must translate");
}

/// Full Finding-13 trigger shape: `Init == x = N \/ x = 1`,
/// `Next == x' = x`, `Inv == x # N`. TLC finds the violating state `x = N`;
/// PDR must never report Safe. Today the translation declines at `add_init`
/// (which the PDR lane maps to Unknown); if translation ever starts
/// accepting it, the solve itself must still not return Safe.
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_model_value_violation_is_never_reported_safe() {
    let mut trans = ChcTranslator::new(&[("x", TlaSort::Int)]).unwrap();
    let init = or_expr(
        eq_expr(var_expr("x"), ident_expr("N")),
        eq_expr(var_expr("x"), int_expr(1)),
    );
    if trans.add_init(&init).is_err() {
        return; // decline: sound (PDR lane reports Unknown)
    }
    if trans
        .add_next(&eq_expr(prime_expr("x"), var_expr("x")))
        .is_err()
    {
        return;
    }
    if trans
        .add_safety(&ne_expr(var_expr("x"), ident_expr("N")))
        .is_err()
    {
        return;
    }
    match trans.solve_pdr(pdr_test_config()).unwrap() {
        PdrCheckResult::Unsafe { .. } | PdrCheckResult::Unknown { .. } => {}
        PdrCheckResult::Safe { invariant } => {
            panic!("false Safe: reachable x = N violates x # N, got invariant {invariant}")
        }
    }
}
