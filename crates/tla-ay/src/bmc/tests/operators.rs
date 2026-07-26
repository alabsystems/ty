// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

/// Test Equiv operator (logical equivalence)
/// Init == a = TRUE /\ b = TRUE
/// Inv == a <=> b (should hold since both are TRUE)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_equiv_operator() {
    let k = 0;
    let mut trans = BmcTranslator::new(k).unwrap();

    trans.declare_var("a", TlaSort::Bool).unwrap();
    trans.declare_var("b", TlaSort::Bool).unwrap();

    // Init: a = TRUE /\ b = TRUE
    let init = spanned(Expr::And(
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "a".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Bool(true))),
        ))),
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "b".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Bool(true))),
        ))),
    ));
    let init_term = trans.translate_init(&init).unwrap();
    trans.assert(init_term);

    // Safety: a <=> b (equivalence)
    let safety = spanned(Expr::Equiv(
        Box::new(spanned(Expr::Ident(
            "a".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "b".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    let not_safety = trans.translate_not_safety_all_steps(&safety, k).unwrap();
    trans.assert(not_safety);

    // Should be UNSAT - a <=> b holds when both are TRUE
    match trans.check_sat() {
        SolveResult::Unsat(_) => {} // Expected
        SolveResult::Sat => panic!("expected UNSAT (a <=> b holds), got SAT"),
        SolveResult::Unknown => panic!("solver returned unknown"),
        _ => panic!("unexpected SolveResult variant"),
    }
}

/// Test BOOLEAN set membership
/// Init == x \in BOOLEAN
/// Inv == x = TRUE \/ x = FALSE (always holds for boolean var)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_boolean_membership() {
    let k = 0;
    let mut trans = BmcTranslator::new(k).unwrap();

    trans.declare_var("x", TlaSort::Bool).unwrap();

    // Init: x \in BOOLEAN (trivially true for Bool sort)
    let init = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "BOOLEAN".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    let init_term = trans.translate_init(&init).unwrap();
    trans.assert(init_term);

    // Safety: x = TRUE \/ x = FALSE
    let safety = spanned(Expr::Or(
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Bool(true))),
        ))),
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Bool(false))),
        ))),
    ));
    let not_safety = trans.translate_not_safety_all_steps(&safety, k).unwrap();
    trans.assert(not_safety);

    // Should be UNSAT - x is always TRUE or FALSE
    match trans.check_sat() {
        SolveResult::Unsat(_) => {} // Expected
        SolveResult::Sat => panic!("expected UNSAT (x is always TRUE or FALSE), got SAT"),
        SolveResult::Unknown => panic!("solver returned unknown"),
        _ => panic!("unexpected SolveResult variant"),
    }
}

/// Test Implies operator
/// Init == a = TRUE /\ b = FALSE
/// Inv == a => b (should fail since TRUE => FALSE is FALSE)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_implies_operator() {
    let k = 0;
    let mut trans = BmcTranslator::new(k).unwrap();

    trans.declare_var("a", TlaSort::Bool).unwrap();
    trans.declare_var("b", TlaSort::Bool).unwrap();

    // Init: a = TRUE /\ b = FALSE
    let init = spanned(Expr::And(
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "a".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Bool(true))),
        ))),
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "b".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Bool(false))),
        ))),
    ));
    let init_term = trans.translate_init(&init).unwrap();
    trans.assert(init_term);

    // Safety: a => b (implication)
    let safety = spanned(Expr::Implies(
        Box::new(spanned(Expr::Ident(
            "a".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "b".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    let not_safety = trans.translate_not_safety_all_steps(&safety, k).unwrap();
    trans.assert(not_safety);

    // Should be SAT - TRUE => FALSE is FALSE, so invariant fails
    match trans.check_sat() {
        SolveResult::Sat => {} // Expected - violation found
        SolveResult::Unsat(_) => panic!("expected SAT (a => b fails), got UNSAT"),
        SolveResult::Unknown => panic!("solver returned unknown"),
        _ => panic!("unexpected SolveResult variant"),
    }
}

/// Test Range membership (x \in lo..hi)
/// Init == x \in 1..5
/// Inv == x >= 1 /\ x <= 5 (should hold)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_range_membership() {
    let k = 0;
    let mut trans = BmcTranslator::new(k).unwrap();

    trans.declare_var("x", TlaSort::Int).unwrap();

    // Init: x \in 1..5
    let init = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Range(
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ))),
    ));
    let init_term = trans.translate_init(&init).unwrap();
    trans.assert(init_term);

    // Safety: x >= 1 /\ x <= 5
    let safety = spanned(Expr::And(
        Box::new(spanned(Expr::Geq(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
        ))),
        Box::new(spanned(Expr::Leq(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ))),
    ));
    let not_safety = trans.translate_not_safety_all_steps(&safety, k).unwrap();
    trans.assert(not_safety);

    // Should be UNSAT - x \in 1..5 implies x >= 1 /\ x <= 5
    match trans.check_sat() {
        SolveResult::Unsat(_) => {} // Expected
        SolveResult::Sat => panic!("expected UNSAT (range membership correct), got SAT"),
        SolveResult::Unknown => panic!("solver returned unknown"),
        _ => panic!("unexpected SolveResult variant"),
    }
}

/// Test Neq (not equal) operator
/// Init == x = 5 /\ y = 5
/// Inv == x # y (should fail since x = y)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_neq_operator() {
    let k = 0;
    let mut trans = BmcTranslator::new(k).unwrap();

    trans.declare_var("x", TlaSort::Int).unwrap();
    trans.declare_var("y", TlaSort::Int).unwrap();

    // Init: x = 5 /\ y = 5
    let init = spanned(Expr::And(
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ))),
        Box::new(spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "y".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ))),
    ));
    let init_term = trans.translate_init(&init).unwrap();
    trans.assert(init_term);

    // Safety: x # y (not equal)
    let safety = spanned(Expr::Neq(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "y".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    let not_safety = trans.translate_not_safety_all_steps(&safety, k).unwrap();
    trans.assert(not_safety);

    // Should be SAT - x = y so x # y fails
    match trans.check_sat() {
        SolveResult::Sat => {} // Expected - violation found
        SolveResult::Unsat(_) => panic!("expected SAT (x # y fails), got UNSAT"),
        SolveResult::Unknown => panic!("solver returned unknown"),
        _ => panic!("unexpected SolveResult variant"),
    }
}

/// Test Neg (negation) operator
/// Init == x = -5
/// Inv == x < 0 (should hold)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_neg_operator() {
    let k = 0;
    let mut trans = BmcTranslator::new(k).unwrap();

    trans.declare_var("x", TlaSort::Int).unwrap();

    // Init: x = -5
    let init = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Neg(Box::new(spanned(Expr::Int(
            BigInt::from(5),
        )))))),
    ));
    let init_term = trans.translate_init(&init).unwrap();
    trans.assert(init_term);

    // Safety: x < 0
    let safety = spanned(Expr::Lt(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(BigInt::from(0)))),
    ));
    let not_safety = trans.translate_not_safety_all_steps(&safety, k).unwrap();
    trans.assert(not_safety);

    // Should be UNSAT - x = -5 < 0
    match trans.check_sat() {
        SolveResult::Unsat(_) => {} // Expected
        SolveResult::Sat => panic!("expected UNSAT (-5 < 0 holds), got SAT"),
        SolveResult::Unknown => panic!("solver returned unknown"),
        _ => panic!("unexpected SolveResult variant"),
    }
}

/// String interning shares AY's Int carrier, but never TLA+'s value kind.
/// The first intern id is deliberately used as the integer operand so this
/// catches translation-before-kind-check aliasing in both orientations.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_scalar_equality_rejects_string_int_carrier_alias() {
    const FIRST_STRING_ID: i64 = -1_000_000_007;

    for reverse in [false, true] {
        let mut trans = BmcTranslator::new(0).unwrap();
        let string = spanned(Expr::String("collision".to_string()));
        let integer = spanned(Expr::Int(BigInt::from(FIRST_STRING_ID)));
        let (left, right) = if reverse {
            (integer, string)
        } else {
            (string, integer)
        };
        let equality = spanned(Expr::Eq(Box::new(left), Box::new(right)));
        let term = trans.translate_init(&equality).unwrap();
        trans.assert(term);
        assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_scalar_disequality_and_expression_kinds_are_exact() {
    let mut trans = BmcTranslator::new(0).unwrap();
    trans.declare_var("s", TlaSort::String).unwrap();

    let neq = spanned(Expr::Neq(
        Box::new(spanned(Expr::Ident(
            "s".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Add(
            Box::new(spanned(Expr::Int(BigInt::from(-1_000_000_007_i64)))),
            Box::new(spanned(Expr::Int(BigInt::from(0)))),
        ))),
    ));
    let term = trans.translate_init(&neq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));

    let mut trans = BmcTranslator::new(0).unwrap();
    let bool_vs_int = spanned(Expr::Eq(
        Box::new(spanned(Expr::And(
            Box::new(spanned(Expr::Bool(true))),
            Box::new(spanned(Expr::Bool(true))),
        ))),
        Box::new(spanned(Expr::Int(BigInt::from(1)))),
    ));
    let false_term = trans.translate_init(&bool_vs_int).unwrap();
    trans.assert(false_term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}
