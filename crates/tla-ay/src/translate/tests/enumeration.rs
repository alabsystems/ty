// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

// Test AYTranslator enumeration with blocking clauses
// This verifies the translator correctly handles incremental solving
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_translator_enumeration_with_blocking() {
    use std::collections::BTreeSet;

    let mut trans = AYTranslator::new();

    // Declare x and y
    trans.declare_var("x", TlaSort::Int).unwrap();
    trans.declare_var("y", TlaSort::Int).unwrap();

    // Assert: x \in 1..2 /\ y \in 1..2
    // This should have 4 solutions: (1,1), (1,2), (2,1), (2,2)
    let x_in_range = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Range(
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
            Box::new(spanned(Expr::Int(BigInt::from(2)))),
        ))),
    ));
    let y_in_range = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "y".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Range(
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
            Box::new(spanned(Expr::Int(BigInt::from(2)))),
        ))),
    ));
    let init = spanned(Expr::And(Box::new(x_in_range), Box::new(y_in_range)));

    let init_term = trans.translate_bool(&init).unwrap();
    trans.assert(init_term);

    let mut solutions = BTreeSet::new();

    for i in 0..10 {
        let (result, summary) = trans.try_check_sat_with_decision_profile_summary().unwrap();
        match result {
            SolveResult::Sat => {
                assert!(summary.accepted_for_consumer);
                assert!(summary.model_validated);
                let model = trans.try_get_model().unwrap();
                let x_val = model.int_val("x").unwrap();
                let y_val = model.int_val("y").unwrap();
                eprintln!("[test] Solution {i}: x={x_val}, y={y_val}");
                assert!((bi(1)..=bi(2)).contains(x_val), "x out of range: {x_val}");
                assert!((bi(1)..=bi(2)).contains(y_val), "y out of range: {y_val}");
                assert!(
                    solutions.insert((x_val.clone(), y_val.clone())),
                    "duplicate model: ({x_val}, {y_val})"
                );

                trans
                    .assert_model_blocking_clause(&model, ["x", "y"])
                    .unwrap();
            }
            SolveResult::Unsat(_) => {
                eprintln!("[test] UNSAT after {i} solutions");
                break;
            }
            SolveResult::Unknown => {
                panic!("Unexpected Unknown result");
            }
            _ => panic!("unexpected SolveResult variant"),
        }
    }

    let expected: BTreeSet<(BigInt, BigInt)> = [
        (bi(1), bi(1)),
        (bi(1), bi(2)),
        (bi(2), bi(1)),
        (bi(2), bi(2)),
    ]
    .into_iter()
    .collect();
    eprintln!(
        "[test] Total solutions: {}, expected: {}",
        solutions.len(),
        expected.len()
    );
    assert_eq!(solutions, expected);
}

// Regression test for ay#949: incremental solving with arithmetic equality must
// enumerate all solutions when using blocking clauses.
//
// See: ay issue #949
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ay_949_arithmetic_enumeration_bug() {
    use std::collections::BTreeSet;

    let mut trans = AYTranslator::new();

    trans.declare_var("x", TlaSort::Int).unwrap();
    trans.declare_var("y", TlaSort::Int).unwrap();

    // Assert: x \in 1..10 /\ y \in 1..10 /\ x + y = 5
    let x_in_range = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Range(
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
            Box::new(spanned(Expr::Int(BigInt::from(10)))),
        ))),
    ));
    let y_in_range = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "y".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Range(
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
            Box::new(spanned(Expr::Int(BigInt::from(10)))),
        ))),
    ));
    let sum_eq_5 = spanned(Expr::Eq(
        Box::new(spanned(Expr::Add(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Ident(
                "y".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
        ))),
        Box::new(spanned(Expr::Int(BigInt::from(5)))),
    ));

    let init = spanned(Expr::And(
        Box::new(x_in_range),
        Box::new(spanned(Expr::And(Box::new(y_in_range), Box::new(sum_eq_5)))),
    ));

    let init_term = trans.translate_bool(&init).unwrap();
    trans.assert(init_term);

    let mut solutions = BTreeSet::new();

    for _ in 0..10 {
        let (result, summary) = trans.try_check_sat_with_decision_profile_summary().unwrap();
        match result {
            SolveResult::Sat => {
                assert!(summary.accepted_for_consumer);
                assert!(summary.model_validated);
                let model = trans.try_get_model().unwrap();
                let x_val = model.int_val("x").unwrap();
                let y_val = model.int_val("y").unwrap();
                assert!((bi(1)..=bi(10)).contains(x_val), "x out of range: {x_val}");
                assert!((bi(1)..=bi(10)).contains(y_val), "y out of range: {y_val}");
                assert_eq!(
                    x_val + y_val,
                    bi(5),
                    "model violates x + y = 5: ({x_val}, {y_val})"
                );
                assert!(
                    solutions.insert((x_val.clone(), y_val.clone())),
                    "duplicate model: ({x_val}, {y_val})"
                );

                trans
                    .assert_model_blocking_clause(&model, ["x", "y"])
                    .unwrap();
            }
            SolveResult::Unsat(_) => break,
            SolveResult::Unknown => panic!("Unexpected Unknown"),
            _ => panic!("unexpected SolveResult variant"),
        }
    }

    // x + y = 5 with x,y in 1..10 has 4 solutions:
    // (1,4), (2,3), (3,2), (4,1)
    let expected: BTreeSet<(BigInt, BigInt)> = [
        (bi(1), bi(4)),
        (bi(2), bi(3)),
        (bi(3), bi(2)),
        (bi(4), bi(1)),
    ]
    .into_iter()
    .collect();
    assert_eq!(solutions, expected, "ay#949 regression");
}
