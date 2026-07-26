// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BMC record and tuple encoding via per-field/per-element SMT variables.
//!
//! Part of #3787: Validates record construction, field access, EXCEPT,
//! tuple construction, indexing, and UNCHANGED operations in the BMC translator.

use super::*;
use ay_dpll::api::SolveResult;
use tla_core::ast::{ExceptPathElement, ExceptSpec, RecordFieldName};

/// Helper: create a BMC translator with array support.
fn bmc_array(k: usize) -> BmcTranslator {
    BmcTranslator::new_with_arrays(k).unwrap()
}

/// Helper: create an Ident expression with INVALID NameId.
fn ident(name: &str) -> Spanned<Expr> {
    spanned(Expr::Ident(
        name.to_string(),
        tla_core::name_intern::NameId::INVALID,
    ))
}

/// Helper: create an integer literal expression.
fn int(n: i64) -> Spanned<Expr> {
    spanned(Expr::Int(BigInt::from(n)))
}

/// Helper: create a spanned string.
fn sstr(s: &str) -> Spanned<String> {
    Spanned::dummy(s.to_string())
}

/// Helper: assert `term == int_val` in solver (avoids double borrow).
fn assert_term_eq_int(bmc: &mut BmcTranslator, term: ay_dpll::api::Term, val: i64) {
    let c = bmc.solver.int_const(val);
    let eq = bmc.solver.try_eq(term, c).unwrap();
    bmc.assert(eq);
}

// ================================================================
// Record declaration
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_record_var_int_fields() {
    let mut bmc = bmc_array(2);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();
    assert!(bmc.record_vars.contains_key("r"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_record_var_mixed_fields() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var(
        "r",
        vec![
            ("flag".to_string(), TlaSort::Bool),
            ("count".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_record_var_allows_set_field() {
    let mut bmc = bmc_array(0);
    let result = bmc.declare_record_var(
        "r",
        vec![(
            "s".to_string(),
            TlaSort::Set {
                element_sort: Box::new(TlaSort::Int),
            },
        )],
    );
    assert!(result.is_ok(), "Set fields should be allowed in records");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_record_var_rejects_unsupported_compound_field() {
    let mut bmc = bmc_array(0);
    // Sequence fields are still unsupported
    let result = bmc.declare_record_var(
        "r",
        vec![(
            "s".to_string(),
            TlaSort::Sequence {
                element_sort: Box::new(TlaSort::Int),
                max_len: 5,
            },
        )],
    );
    assert!(result.is_err(), "Sequence fields should still be rejected");
}

fn rejecting_nested_record_bmc(bound_k: usize) -> BmcTranslator {
    let mut bmc = bmc_array(bound_k);
    let result = bmc.declare_record_var(
        "r",
        vec![
            ("prefix".to_string(), TlaSort::Int),
            (
                "inner".to_string(),
                TlaSort::Record {
                    field_sorts: vec![("x".to_string(), TlaSort::Int)],
                },
            ),
        ],
    );
    let error = result.expect_err("nested records must fail closed until recursively encoded");
    assert!(
        error
            .to_string()
            .contains("nested records require a recursive encoding"),
        "unexpected rejection: {error}"
    );
    assert!(!bmc.record_vars.contains_key("r"));
    assert!(
        !bmc.record_vars.contains_key("r__f_inner"),
        "declaration must reject before registering a partial nested carrier"
    );
    bmc
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_nested_record_rejection_leaves_translator_usable_for_flat_records() {
    let mut bmc = rejecting_nested_record_bmc(0);
    bmc.declare_record_var(
        "r",
        vec![
            ("prefix".to_string(), TlaSort::Int),
            ("flag".to_string(), TlaSort::Bool),
            (
                "items".to_string(),
                TlaSort::Set {
                    element_sort: Box::new(TlaSort::Int),
                },
            ),
        ],
    )
    .expect("supported flat scalar/set record must still declare after rejection");
    assert!(bmc.record_vars.contains_key("r"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_duplicate_record_fields_reject_before_registration() {
    let mut bmc = bmc_array(0);
    let error = bmc
        .declare_record_var(
            "r",
            vec![
                ("same".to_string(), TlaSort::Int),
                ("same".to_string(), TlaSort::Bool),
            ],
        )
        .expect_err("duplicate record fields must fail closed");
    assert!(error.to_string().contains("duplicate field 'same'"));
    assert!(!bmc.record_vars.contains_key("r"));

    bmc.declare_record_var("r", vec![("same".to_string(), TlaSort::Int)])
        .expect("duplicate rejection must leave the translator reusable");
    assert!(bmc.record_vars.contains_key("r"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_redeclaration_requires_canonical_equivalent_shape() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var(
        "r",
        vec![
            ("count".to_string(), TlaSort::Int),
            ("flag".to_string(), TlaSort::Bool),
        ],
    )
    .unwrap();
    let original_count = bmc.get_record_field_at_step("r", "count", 0).unwrap();
    let original_flag = bmc.get_record_field_at_step("r", "flag", 1).unwrap();

    // Record field order is not semantically significant.
    bmc.declare_record_var(
        "r",
        vec![
            ("flag".to_string(), TlaSort::Bool),
            ("count".to_string(), TlaSort::Int),
        ],
    )
    .expect("canonical-equivalent re-declaration must be idempotent");
    assert_eq!(
        bmc.get_record_field_at_step("r", "count", 0).unwrap(),
        original_count
    );
    assert_eq!(
        bmc.get_record_field_at_step("r", "flag", 1).unwrap(),
        original_flag
    );

    let wrong_sort = bmc
        .declare_record_var(
            "r",
            vec![
                ("count".to_string(), TlaSort::Bool),
                ("flag".to_string(), TlaSort::Bool),
            ],
        )
        .expect_err("re-declaring a field with another sort must fail closed");
    assert!(matches!(wrong_sort, AYError::TypeMismatch { .. }));

    let wrong_name = bmc
        .declare_record_var(
            "r",
            vec![
                ("count".to_string(), TlaSort::Int),
                ("other".to_string(), TlaSort::Bool),
            ],
        )
        .expect_err("re-declaring a different field set must fail closed");
    assert!(matches!(wrong_name, AYError::TypeMismatch { .. }));

    // Rejected shapes must leave the original declaration intact.
    assert_eq!(bmc.record_vars.len(), 1);
    assert_eq!(
        bmc.get_record_field_at_step("r", "count", 0).unwrap(),
        original_count
    );
    assert_eq!(
        bmc.get_record_field_at_step("r", "flag", 1).unwrap(),
        original_flag
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_nested_record_rejects_before_equality_can_compare_sentinels() {
    let mut bmc = rejecting_nested_record_bmc(0);
    let equality = spanned(Expr::Eq(Box::new(ident("r")), Box::new(ident("r"))));
    assert!(
        bmc.translate_bool(&equality).is_err(),
        "a rejected nested record must not translate equality as TRUE"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_nested_record_rejects_before_unchanged_can_ignore_nested_state() {
    let mut bmc = rejecting_nested_record_bmc(1);
    let unchanged = spanned(Expr::Unchanged(Box::new(ident("r"))));
    assert!(
        bmc.translate_bool(&unchanged).is_err(),
        "a rejected nested record must not translate UNCHANGED as a sentinel tautology"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_nested_record_rejects_before_except_can_skip_nested_copy() {
    let mut bmc = rejecting_nested_record_bmc(1);
    let except = spanned(Expr::Except(
        Box::new(ident("r")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr(
                "inner",
            )))],
            value: spanned(Expr::Record(vec![(sstr("x"), int(1))])),
        }],
    ));
    let transition = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(ident("r"))))),
        Box::new(except),
    ));
    assert!(
        bmc.translate_bool(&transition).is_err(),
        "a rejected nested record must not translate EXCEPT through sentinel fields"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_var_delegates_record_sort() {
    let mut bmc = bmc_array(1);
    bmc.declare_var(
        "r",
        TlaSort::Record {
            field_sorts: vec![
                ("x".to_string(), TlaSort::Int),
                ("y".to_string(), TlaSort::Bool),
            ],
        },
    )
    .unwrap();
    assert!(bmc.record_vars.contains_key("r"));
    assert!(!bmc.vars.contains_key("r"));
}

// ================================================================
// Record field access
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_field_access_int() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    // r.a = 42
    let access = spanned(Expr::RecordAccess(
        Box::new(ident("r")),
        RecordFieldName::new(sstr("a")),
    ));
    let eq = spanned(Expr::Eq(Box::new(access), Box::new(int(42))));
    let term = bmc.translate_init(&eq).unwrap();
    bmc.assert(term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_field_access_contradicts_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var("r", vec![("x".to_string(), TlaSort::Int)])
        .unwrap();

    // r.x = 10 AND r.x = 20 -> UNSAT
    let rx = spanned(Expr::RecordAccess(
        Box::new(ident("r")),
        RecordFieldName::new(sstr("x")),
    ));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(rx.clone()), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(rx), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_different_fields_independent() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    // r.a = 10 AND r.b = 20 -> SAT (different fields)
    let ra = spanned(Expr::RecordAccess(
        Box::new(ident("r")),
        RecordFieldName::new(sstr("a")),
    ));
    let rb = spanned(Expr::RecordAccess(
        Box::new(ident("r")),
        RecordFieldName::new(sstr("b")),
    ));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ra), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(rb), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// ================================================================
// Record literal equality: r' = [a |-> 1, b |-> 2]
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_literal_eq() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    // r' = [a |-> 1, b |-> 2]
    let record_literal = spanned(Expr::Record(vec![
        (sstr("a"), Spanned::dummy(Expr::Int(BigInt::from(1)))),
        (sstr("b"), Spanned::dummy(Expr::Int(BigInt::from(2)))),
    ]));
    let r_prime = spanned(Expr::Prime(Box::new(ident("r"))));
    let eq = spanned(Expr::Eq(Box::new(r_prime), Box::new(record_literal)));

    bmc.current_step = 0;
    let term = bmc.translate_bool(&eq).unwrap();
    bmc.assert(term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify r'.a = 1 and r'.b = 2
    let a_step1 = bmc.get_record_field_at_step("r", "a", 1).unwrap();
    assert_term_eq_int(&mut bmc, a_step1, 1);
    let b_step1 = bmc.get_record_field_at_step("r", "b", 1).unwrap();
    assert_term_eq_int(&mut bmc, b_step1, 2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// Record equality includes exact DOMAIN equality. A literal missing a declared
/// field, adding an undeclared field, or repeating a field is FALSE; its negation
/// is TRUE. This polarity check prevents a partial field conjunction from
/// becoming a stronger predicate when negated in a proof obligation.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_literal_domain_mismatch_false_in_both_polarities() {
    fn check(declared: Vec<(String, TlaSort)>, literal: Vec<(&str, i64)>) {
        let equality = |literal_on_left: bool| {
            let literal = spanned(Expr::Record(
                literal
                    .iter()
                    .map(|(name, value)| (sstr(name), int(*value)))
                    .collect(),
            ));
            if literal_on_left {
                spanned(Expr::Eq(Box::new(literal), Box::new(ident("r"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("r")), Box::new(literal)))
            }
        };

        for literal_on_left in [false, true] {
            let mut positive = bmc_array(0);
            positive.declare_record_var("r", declared.clone()).unwrap();
            let term = positive.translate_init(&equality(literal_on_left)).unwrap();
            positive.assert(term);
            assert!(
                matches!(positive.check_sat(), SolveResult::Unsat(_)),
                "different record DOMAINs must make equality FALSE"
            );

            let mut negative = bmc_array(0);
            negative.declare_record_var("r", declared.clone()).unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality(literal_on_left))));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(
                matches!(negative.check_sat(), SolveResult::Sat),
                "the negation of a different-DOMAIN equality must be TRUE"
            );
        }
    }

    check(
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
        vec![("a", 0)],
    );
    check(
        vec![("a".to_string(), TlaSort::Int)],
        vec![("a", 0), ("b", 1)],
    );
    check(
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
        vec![("a", 0), ("a", 1)],
    );
}

/// A record literal's field value sorts are part of its shape. This is
/// especially important for Int/String, which use the same underlying SMT sort
/// in BMC but remain distinct TLA+ value kinds.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_literal_sort_mismatch_is_symmetric_false() {
    fn check(declared_sort: TlaSort, literal_value: Spanned<Expr>) {
        let equality = |literal_on_left: bool| {
            let literal = spanned(Expr::Record(vec![(sstr("a"), literal_value.clone())]));
            if literal_on_left {
                spanned(Expr::Eq(Box::new(literal), Box::new(ident("r"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("r")), Box::new(literal)))
            }
        };

        for literal_on_left in [false, true] {
            let mut positive = bmc_array(0);
            positive
                .declare_record_var("r", vec![("a".to_string(), declared_sort.clone())])
                .unwrap();
            let term = positive.translate_init(&equality(literal_on_left)).unwrap();
            positive.assert(term);
            assert!(
                matches!(positive.check_sat(), SolveResult::Unsat(_)),
                "different record field sorts must make equality FALSE"
            );

            let mut negative = bmc_array(0);
            negative
                .declare_record_var("r", vec![("a".to_string(), declared_sort.clone())])
                .unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality(literal_on_left))));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(
                matches!(negative.check_sat(), SolveResult::Sat),
                "negating a different-sort record equality must be TRUE"
            );
        }
    }

    check(TlaSort::Int, spanned(Expr::Bool(true)));
    check(TlaSort::Bool, int(1));
    check(TlaSort::Int, spanned(Expr::String("one".to_string())));
    check(TlaSort::String, int(1));
}

/// Exact mixed-sort shapes still translate pointwise. Field declaration and
/// literal order are intentionally different because record order is not part
/// of equality.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_literal_exact_mixed_shape_compares_fields() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var(
        "r",
        vec![
            ("flag".to_string(), TlaSort::Bool),
            ("count".to_string(), TlaSort::Int),
            ("name".to_string(), TlaSort::String),
        ],
    )
    .unwrap();

    let literal = spanned(Expr::Record(vec![
        (sstr("name"), spanned(Expr::String("ok".to_string()))),
        (sstr("count"), int(7)),
        (sstr("flag"), spanned(Expr::Bool(true))),
    ]));
    let equality = spanned(Expr::Eq(Box::new(literal), Box::new(ident("r"))));
    let term = bmc.translate_init(&equality).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    let flag = bmc.get_record_field_at_step("r", "flag", 0).unwrap();
    let not_flag = bmc.solver.try_not(flag).unwrap();
    bmc.assert(not_flag);
    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

/// Record-variable equality must be symmetric and exact in field names, sorts,
/// and arity. Every mismatch is the TLA value FALSE in either orientation, and
/// negating either orientation is satisfiable.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_var_shape_mismatch_is_symmetric_false() {
    fn check(left_fields: Vec<(String, TlaSort)>, right_fields: Vec<(String, TlaSort)>) {
        for (lhs, rhs) in [("left", "right"), ("right", "left")] {
            let equality = || spanned(Expr::Eq(Box::new(ident(lhs)), Box::new(ident(rhs))));

            let mut positive = bmc_array(0);
            positive
                .declare_record_var("left", left_fields.clone())
                .unwrap();
            positive
                .declare_record_var("right", right_fields.clone())
                .unwrap();
            let term = positive.translate_init(&equality()).unwrap();
            positive.assert(term);
            assert!(
                matches!(positive.check_sat(), SolveResult::Unsat(_)),
                "shape-mismatched equality {lhs} = {rhs} must be FALSE"
            );

            let mut negative = bmc_array(0);
            negative
                .declare_record_var("left", left_fields.clone())
                .unwrap();
            negative
                .declare_record_var("right", right_fields.clone())
                .unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality())));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(
                matches!(negative.check_sat(), SolveResult::Sat),
                "negated shape-mismatched equality {lhs} # {rhs} must be TRUE"
            );
        }
    }

    check(
        vec![("a".to_string(), TlaSort::Int)],
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    );
    check(
        vec![("a".to_string(), TlaSort::Int)],
        vec![("b".to_string(), TlaSort::Int)],
    );
    check(
        vec![("a".to_string(), TlaSort::Int)],
        vec![("a".to_string(), TlaSort::Bool)],
    );
}

/// Field declaration order is not part of a record's DOMAIN. Equal shapes in a
/// different order still compare pointwise and remain satisfiable.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_var_equality_ignores_declaration_order() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var(
        "left",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Bool),
            ("c".to_string(), TlaSort::String),
        ],
    )
    .unwrap();
    bmc.declare_record_var(
        "right",
        vec![
            ("c".to_string(), TlaSort::String),
            ("b".to_string(), TlaSort::Bool),
            ("a".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    let equality = spanned(Expr::Eq(Box::new(ident("left")), Box::new(ident("right"))));
    let term = bmc.translate_init(&equality).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    let left_a = bmc.get_record_field_at_step("left", "a", 0).unwrap();
    let right_a = bmc.get_record_field_at_step("right", "a", 0).unwrap();
    assert_term_eq_int(&mut bmc, left_a, 0);
    assert_term_eq_int(&mut bmc, right_a, 1);
    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

/// Differing Set element metadata does not prove unequal TLA values: two empty
/// sets compare equal regardless of the inferred element sorts. Decline rather
/// than replacing the equality with FALSE.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_set_element_sort_mismatch_fails_closed() {
    let set_int = TlaSort::Set {
        element_sort: Box::new(TlaSort::Int),
    };
    let set_string = TlaSort::Set {
        element_sort: Box::new(TlaSort::String),
    };

    for (lhs, rhs) in [("left", "right"), ("right", "left")] {
        let mut bmc = bmc_array(0);
        bmc.declare_record_var("left", vec![("s".to_string(), set_int.clone())])
            .unwrap();
        bmc.declare_record_var("right", vec![("s".to_string(), set_string.clone())])
            .unwrap();
        let equality = spanned(Expr::Eq(Box::new(ident(lhs)), Box::new(ident(rhs))));
        let error = bmc
            .translate_init(&equality)
            .expect_err("differing Set element metadata must decline");
        assert!(error.to_string().contains("differing set element sorts"));
    }
}

// ================================================================
// Record EXCEPT: r' = [r EXCEPT !.a = 99]
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_single_field() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    // Init: r.a = 1, r.b = 2
    let init_a = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::RecordAccess(
                Box::new(ident("r")),
                RecordFieldName::new(sstr("a")),
            ))),
            Box::new(int(1)),
        )))
        .unwrap();
    bmc.assert(init_a);

    let init_b = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::RecordAccess(
                Box::new(ident("r")),
                RecordFieldName::new(sstr("b")),
            ))),
            Box::new(int(2)),
        )))
        .unwrap();
    bmc.assert(init_b);

    // Next: r' = [r EXCEPT !.a = 99]
    let except_expr = spanned(Expr::Except(
        Box::new(ident("r")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr("a")))],
            value: Spanned::dummy(Expr::Int(BigInt::from(99))),
        }],
    ));
    let r_prime = spanned(Expr::Prime(Box::new(ident("r"))));
    let next = spanned(Expr::Eq(Box::new(r_prime), Box::new(except_expr)));

    bmc.current_step = 0;
    let next_term = bmc.translate_bool(&next).unwrap();
    bmc.assert(next_term);

    // Check: r'.a = 99 (overridden), r'.b = 2 (copied)
    let a_prime = bmc.get_record_field_at_step("r", "a", 1).unwrap();
    assert_term_eq_int(&mut bmc, a_prime, 99);
    let b_prime = bmc.get_record_field_at_step("r", "b", 1).unwrap();
    assert_term_eq_int(&mut bmc, b_prime, 2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_wrong_value_unsat() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    // Init: r.a = 1, r.b = 2
    let init_a = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::RecordAccess(
                Box::new(ident("r")),
                RecordFieldName::new(sstr("a")),
            ))),
            Box::new(int(1)),
        )))
        .unwrap();
    bmc.assert(init_a);

    let init_b = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::RecordAccess(
                Box::new(ident("r")),
                RecordFieldName::new(sstr("b")),
            ))),
            Box::new(int(2)),
        )))
        .unwrap();
    bmc.assert(init_b);

    // Next: r' = [r EXCEPT !.a = 99]
    let except_expr = spanned(Expr::Except(
        Box::new(ident("r")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr("a")))],
            value: Spanned::dummy(Expr::Int(BigInt::from(99))),
        }],
    ));
    let r_prime = spanned(Expr::Prime(Box::new(ident("r"))));
    let next = spanned(Expr::Eq(Box::new(r_prime), Box::new(except_expr)));

    bmc.current_step = 0;
    let next_term = bmc.translate_bool(&next).unwrap();
    bmc.assert(next_term);

    // Contradiction: r'.b should be 2 (copied), but we claim it's 999
    let b_prime = bmc.get_record_field_at_step("r", "b", 1).unwrap();
    assert_term_eq_int(&mut bmc, b_prime, 999);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

/// Record EXCEPT preserves the source DOMAIN and per-field value kinds. A
/// differently shaped target therefore compares FALSE in either equality
/// orientation, including when Int/String would share the same SMT carrier.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_shape_mismatch_is_symmetric_false() {
    fn check(
        target_fields: Vec<(String, TlaSort)>,
        source_fields: Vec<(String, TlaSort)>,
        override_value: Spanned<Expr>,
    ) {
        let equality = |except_on_left: bool| {
            let except = spanned(Expr::Except(
                Box::new(ident("source")),
                vec![ExceptSpec {
                    path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr("a")))],
                    value: override_value.clone(),
                }],
            ));
            if except_on_left {
                spanned(Expr::Eq(Box::new(except), Box::new(ident("target"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("target")), Box::new(except)))
            }
        };

        for except_on_left in [false, true] {
            let mut positive = bmc_array(0);
            positive
                .declare_record_var("target", target_fields.clone())
                .unwrap();
            positive
                .declare_record_var("source", source_fields.clone())
                .unwrap();
            let term = positive.translate_init(&equality(except_on_left)).unwrap();
            positive.assert(term);
            assert!(
                matches!(positive.check_sat(), SolveResult::Unsat(_)),
                "shape-mismatched record EXCEPT equality must be FALSE"
            );

            let mut negative = bmc_array(0);
            negative
                .declare_record_var("target", target_fields.clone())
                .unwrap();
            negative
                .declare_record_var("source", source_fields.clone())
                .unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality(except_on_left))));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(
                matches!(negative.check_sat(), SolveResult::Sat),
                "negated shape-mismatched record EXCEPT equality must be TRUE"
            );
        }
    }

    // Both arity directions used to be asymmetric: one constrained only the
    // shared prefix, while the other failed on the missing target field.
    check(
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
        vec![("a".to_string(), TlaSort::Int)],
        int(7),
    );
    check(
        vec![("a".to_string(), TlaSort::Int)],
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
        int(7),
    );

    // Int and String both use SMT Int in BMC, but remain distinct TLA values.
    check(
        vec![("a".to_string(), TlaSort::String)],
        vec![("a".to_string(), TlaSort::Int)],
        int(7),
    );
    check(
        vec![("a".to_string(), TlaSort::Int)],
        vec![("a".to_string(), TlaSort::String)],
        spanned(Expr::String("seven".to_string())),
    );
}

/// Updating a field with a different TLA value kind changes the EXCEPT result's
/// shape. It must compare FALSE before either the Int or String term is emitted.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_override_sort_mismatch_is_symmetric_false() {
    fn check(field_sort: TlaSort, override_value: Spanned<Expr>) {
        let equality = |except_on_left: bool| {
            let except = spanned(Expr::Except(
                Box::new(ident("source")),
                vec![ExceptSpec {
                    path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr("a")))],
                    value: override_value.clone(),
                }],
            ));
            if except_on_left {
                spanned(Expr::Eq(Box::new(except), Box::new(ident("target"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("target")), Box::new(except)))
            }
        };

        for except_on_left in [false, true] {
            let fields = vec![("a".to_string(), field_sort.clone())];
            let mut positive = bmc_array(0);
            positive
                .declare_record_var("target", fields.clone())
                .unwrap();
            positive
                .declare_record_var("source", fields.clone())
                .unwrap();
            let term = positive.translate_init(&equality(except_on_left)).unwrap();
            positive.assert(term);
            assert!(matches!(positive.check_sat(), SolveResult::Unsat(_)));

            let mut negative = bmc_array(0);
            negative
                .declare_record_var("target", fields.clone())
                .unwrap();
            negative.declare_record_var("source", fields).unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality(except_on_left))));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(matches!(negative.check_sat(), SolveResult::Sat));
        }
    }

    check(TlaSort::String, int(1));
    check(TlaSort::Int, spanned(Expr::String("one".to_string())));
}

/// Canonical-equivalent source/target shapes remain supported even when their
/// declaration orders differ.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_same_shape_different_order_supported() {
    for except_on_left in [false, true] {
        let mut bmc = bmc_array(0);
        bmc.declare_record_var(
            "target",
            vec![
                ("flag".to_string(), TlaSort::Bool),
                ("count".to_string(), TlaSort::Int),
            ],
        )
        .unwrap();
        bmc.declare_record_var(
            "source",
            vec![
                ("count".to_string(), TlaSort::Int),
                ("flag".to_string(), TlaSort::Bool),
            ],
        )
        .unwrap();

        let except = spanned(Expr::Except(
            Box::new(ident("source")),
            vec![ExceptSpec {
                path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr(
                    "count",
                )))],
                value: int(9),
            }],
        ));
        let equality = if except_on_left {
            spanned(Expr::Eq(Box::new(except), Box::new(ident("target"))))
        } else {
            spanned(Expr::Eq(Box::new(ident("target")), Box::new(except)))
        };
        let term = bmc.translate_init(&equality).unwrap();
        bmc.assert(term);

        let target_count = bmc.get_record_field_at_step("target", "count", 0).unwrap();
        assert_term_eq_int(&mut bmc, target_count, 9);
        assert!(matches!(bmc.check_sat(), SolveResult::Sat));
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_sort_changing_override_matches_target_shape() {
    for except_on_left in [false, true] {
        let mut bmc = bmc_array(0);
        bmc.declare_record_var("target", vec![("a".to_string(), TlaSort::String)])
            .unwrap();
        bmc.declare_record_var("source", vec![("a".to_string(), TlaSort::Int)])
            .unwrap();

        let except = spanned(Expr::Except(
            Box::new(ident("source")),
            vec![ExceptSpec {
                path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr("a")))],
                value: spanned(Expr::String("updated".to_string())),
            }],
        ));
        let equality = if except_on_left {
            spanned(Expr::Eq(Box::new(except), Box::new(ident("target"))))
        } else {
            spanned(Expr::Eq(Box::new(ident("target")), Box::new(except)))
        };
        let term = bmc.translate_init(&equality).unwrap();
        bmc.assert(term);

        let updated_id = bmc.bmc_intern_string("updated");
        let target_a = bmc.get_record_field_at_step("target", "a", 0).unwrap();
        assert_term_eq_int(&mut bmc, target_a, updated_id);
        assert!(matches!(bmc.check_sat(), SolveResult::Sat));
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_except_unsupported_path_fails_closed() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var("r", vec![("a".to_string(), TlaSort::Int)])
        .unwrap();

    for path in [
        vec![ExceptPathElement::Index(int(1))],
        vec![
            ExceptPathElement::Field(RecordFieldName::new(sstr("a"))),
            ExceptPathElement::Index(int(1)),
        ],
    ] {
        let except = spanned(Expr::Except(
            Box::new(ident("r")),
            vec![ExceptSpec {
                path,
                value: int(9),
            }],
        ));
        let equality = spanned(Expr::Eq(
            Box::new(spanned(Expr::Prime(Box::new(ident("r"))))),
            Box::new(except),
        ));
        let error = bmc
            .translate_bool(&equality)
            .expect_err("unsupported record EXCEPT paths must not become no-ops");
        assert!(error.to_string().contains("exactly one direct .field"));
    }

    // The rejected path forms must not poison a later supported translation.
    let valid_except = spanned(Expr::Except(
        Box::new(ident("r")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Field(RecordFieldName::new(sstr("a")))],
            value: int(9),
        }],
    ));
    let valid_equality = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(ident("r"))))),
        Box::new(valid_except),
    ));
    let term = bmc.translate_bool(&valid_equality).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// ================================================================
// Record UNCHANGED
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_unchanged() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var(
        "r",
        vec![
            ("a".to_string(), TlaSort::Int),
            ("b".to_string(), TlaSort::Int),
        ],
    )
    .unwrap();

    // Init: r.a = 5, r.b = 10
    let init_a = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::RecordAccess(
                Box::new(ident("r")),
                RecordFieldName::new(sstr("a")),
            ))),
            Box::new(int(5)),
        )))
        .unwrap();
    bmc.assert(init_a);

    let init_b = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::RecordAccess(
                Box::new(ident("r")),
                RecordFieldName::new(sstr("b")),
            ))),
            Box::new(int(10)),
        )))
        .unwrap();
    bmc.assert(init_b);

    // Next: UNCHANGED r
    bmc.current_step = 0;
    let unchanged = bmc
        .translate_bool(&spanned(Expr::Unchanged(Box::new(ident("r")))))
        .unwrap();
    bmc.assert(unchanged);

    // Check: r' values are preserved
    let a1 = bmc.get_record_field_at_step("r", "a", 1).unwrap();
    assert_term_eq_int(&mut bmc, a1, 5);
    let b1 = bmc.get_record_field_at_step("r", "b", 1).unwrap();
    assert_term_eq_int(&mut bmc, b1, 10);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// ================================================================
// Tuple declaration
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_tuple_var_int_elements() {
    let mut bmc = bmc_array(2);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();
    assert!(bmc.tuple_vars.contains_key("t"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_tuple_var_mixed_elements() {
    let mut bmc = bmc_array(1);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Bool, TlaSort::Int])
        .unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_tuple_var_rejects_compound_element() {
    let mut bmc = bmc_array(0);
    let result = bmc.declare_tuple_var(
        "t",
        vec![TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        }],
    );
    assert!(result.is_err());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_redeclaration_requires_exact_shape() {
    let mut bmc = bmc_array(1);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Bool, TlaSort::String])
        .unwrap();
    let original_first = bmc.get_tuple_element_at_step("t", 1, 0).unwrap();
    let original_last = bmc.get_tuple_element_at_step("t", 3, 1).unwrap();

    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Bool, TlaSort::String])
        .expect("same-shape tuple re-declaration must be idempotent");
    assert_eq!(
        bmc.get_tuple_element_at_step("t", 1, 0).unwrap(),
        original_first
    );
    assert_eq!(
        bmc.get_tuple_element_at_step("t", 3, 1).unwrap(),
        original_last
    );

    for wrong_shape in [
        vec![TlaSort::Int, TlaSort::Bool],
        vec![TlaSort::String, TlaSort::Bool, TlaSort::Int],
    ] {
        let error = bmc
            .declare_tuple_var("t", wrong_shape)
            .expect_err("tuple arity/order/sort changes must fail closed");
        assert!(matches!(error, AYError::TypeMismatch { .. }));
    }

    assert_eq!(bmc.tuple_vars.len(), 1);
    assert_eq!(
        bmc.get_tuple_element_at_step("t", 1, 0).unwrap(),
        original_first
    );
    assert_eq!(
        bmc.get_tuple_element_at_step("t", 3, 1).unwrap(),
        original_last
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_var_delegates_tuple_sort() {
    let mut bmc = bmc_array(1);
    bmc.declare_var(
        "t",
        TlaSort::Tuple {
            element_sorts: vec![TlaSort::Int, TlaSort::Bool],
        },
    )
    .unwrap();
    assert!(bmc.tuple_vars.contains_key("t"));
    assert!(!bmc.vars.contains_key("t"));
}

// ================================================================
// Tuple indexing
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_index_int() {
    let mut bmc = bmc_array(0);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();

    // t[1] = 42
    let t1 = spanned(Expr::FuncApply(Box::new(ident("t")), Box::new(int(1))));
    let eq = spanned(Expr::Eq(Box::new(t1), Box::new(int(42))));
    let term = bmc.translate_init(&eq).unwrap();
    bmc.assert(term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_index_contradicts_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();

    // t[1] = 10 AND t[1] = 20 -> UNSAT
    let t1 = spanned(Expr::FuncApply(Box::new(ident("t")), Box::new(int(1))));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(t1.clone()), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(t1), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_different_indices_independent() {
    let mut bmc = bmc_array(0);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();

    // t[1] = 10 AND t[2] = 20 -> SAT (different elements)
    let t1 = spanned(Expr::FuncApply(Box::new(ident("t")), Box::new(int(1))));
    let t2 = spanned(Expr::FuncApply(Box::new(ident("t")), Box::new(int(2))));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(t1), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(t2), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// ================================================================
// Tuple literal equality: t' = <<1, 2>>
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_literal_eq() {
    let mut bmc = bmc_array(1);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();

    // t' = <<1, 2>>
    let tuple_literal = spanned(Expr::Tuple(vec![int(1), int(2)]));
    let t_prime = spanned(Expr::Prime(Box::new(ident("t"))));
    let eq = spanned(Expr::Eq(Box::new(t_prime), Box::new(tuple_literal)));

    bmc.current_step = 0;
    let term = bmc.translate_bool(&eq).unwrap();
    bmc.assert(term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify t'[1] = 1 and t'[2] = 2
    let e1 = bmc.get_tuple_element_at_step("t", 1, 1).unwrap();
    assert_term_eq_int(&mut bmc, e1, 1);
    let e2 = bmc.get_tuple_element_at_step("t", 2, 1).unwrap();
    assert_term_eq_int(&mut bmc, e2, 2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// Tuple literal equality is exact in arity and per-index value kind, and is
/// symmetric in its two operands.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_literal_shape_mismatch_is_symmetric_false() {
    fn check(element_sorts: Vec<TlaSort>, elements: Vec<Spanned<Expr>>) {
        let equality = |literal_on_left: bool| {
            let literal = spanned(Expr::Tuple(elements.clone()));
            if literal_on_left {
                spanned(Expr::Eq(Box::new(literal), Box::new(ident("t"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("t")), Box::new(literal)))
            }
        };

        for literal_on_left in [false, true] {
            let mut positive = bmc_array(0);
            positive
                .declare_tuple_var("t", element_sorts.clone())
                .unwrap();
            let term = positive.translate_init(&equality(literal_on_left)).unwrap();
            positive.assert(term);
            assert!(matches!(positive.check_sat(), SolveResult::Unsat(_)));

            let mut negative = bmc_array(0);
            negative
                .declare_tuple_var("t", element_sorts.clone())
                .unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality(literal_on_left))));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(matches!(negative.check_sat(), SolveResult::Sat));
        }
    }

    check(vec![TlaSort::Int, TlaSort::Int], vec![int(1)]);
    check(vec![TlaSort::Int], vec![int(1), int(2)]);
    check(vec![TlaSort::String], vec![int(1)]);
    check(
        vec![TlaSort::Int],
        vec![spanned(Expr::String("one".to_string()))],
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_literal_exact_mixed_shape_supported() {
    for literal_on_left in [false, true] {
        let mut bmc = bmc_array(0);
        bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Bool, TlaSort::String])
            .unwrap();
        let literal = spanned(Expr::Tuple(vec![
            int(1),
            spanned(Expr::Bool(true)),
            spanned(Expr::String("one".to_string())),
        ]));
        let equality = if literal_on_left {
            spanned(Expr::Eq(Box::new(literal), Box::new(ident("t"))))
        } else {
            spanned(Expr::Eq(Box::new(ident("t")), Box::new(literal)))
        };
        let term = bmc.translate_init(&equality).unwrap();
        bmc.assert(term);
        assert!(matches!(bmc.check_sat(), SolveResult::Sat));
    }
}

/// Tuple-variable equality must compare complete ordered shapes before any
/// pointwise constraints are emitted.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_var_shape_mismatch_is_symmetric_false() {
    fn check(left_sorts: Vec<TlaSort>, right_sorts: Vec<TlaSort>) {
        for (lhs, rhs) in [("left", "right"), ("right", "left")] {
            let equality = || spanned(Expr::Eq(Box::new(ident(lhs)), Box::new(ident(rhs))));

            let mut positive = bmc_array(0);
            positive
                .declare_tuple_var("left", left_sorts.clone())
                .unwrap();
            positive
                .declare_tuple_var("right", right_sorts.clone())
                .unwrap();
            let term = positive.translate_init(&equality()).unwrap();
            positive.assert(term);
            assert!(matches!(positive.check_sat(), SolveResult::Unsat(_)));

            let mut negative = bmc_array(0);
            negative
                .declare_tuple_var("left", left_sorts.clone())
                .unwrap();
            negative
                .declare_tuple_var("right", right_sorts.clone())
                .unwrap();
            let not_equality = spanned(Expr::Not(Box::new(equality())));
            let term = negative.translate_init(&not_equality).unwrap();
            negative.assert(term);
            assert!(matches!(negative.check_sat(), SolveResult::Sat));
        }
    }

    check(vec![TlaSort::Int], vec![TlaSort::Int, TlaSort::Int]);
    check(vec![TlaSort::Int], vec![TlaSort::String]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_var_exact_mixed_shape_supported() {
    let mut bmc = bmc_array(0);
    let shape = vec![TlaSort::Int, TlaSort::Bool, TlaSort::String];
    bmc.declare_tuple_var("left", shape.clone()).unwrap();
    bmc.declare_tuple_var("right", shape).unwrap();

    let equality = spanned(Expr::Eq(Box::new(ident("left")), Box::new(ident("right"))));
    let term = bmc.translate_init(&equality).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    let left_first = bmc.get_tuple_element_at_step("left", 1, 0).unwrap();
    let right_first = bmc.get_tuple_element_at_step("right", 1, 0).unwrap();
    assert_term_eq_int(&mut bmc, left_first, 0);
    assert_term_eq_int(&mut bmc, right_first, 1);
    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// ================================================================
// Tuple UNCHANGED
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_unchanged() {
    let mut bmc = bmc_array(1);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();

    // Init: t[1] = 5, t[2] = 10
    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::FuncApply(
                Box::new(ident("t")),
                Box::new(int(1)),
            ))),
            Box::new(int(5)),
        )))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::FuncApply(
                Box::new(ident("t")),
                Box::new(int(2)),
            ))),
            Box::new(int(10)),
        )))
        .unwrap();
    bmc.assert(eq2);

    // UNCHANGED t
    bmc.current_step = 0;
    let unchanged = bmc
        .translate_bool(&spanned(Expr::Unchanged(Box::new(ident("t")))))
        .unwrap();
    bmc.assert(unchanged);

    // t'[1] must be 5, t'[2] must be 10
    let e1 = bmc.get_tuple_element_at_step("t", 1, 1).unwrap();
    assert_term_eq_int(&mut bmc, e1, 5);
    let e2 = bmc.get_tuple_element_at_step("t", 2, 1).unwrap();
    assert_term_eq_int(&mut bmc, e2, 10);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// ================================================================
// Tuple index out of bounds
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_index_out_of_bounds() {
    let mut bmc = bmc_array(0);
    bmc.declare_tuple_var("t", vec![TlaSort::Int, TlaSort::Int])
        .unwrap();

    // t[3] should fail (only 2 elements)
    let result = bmc.get_tuple_element_at_step("t", 3, 0);
    assert!(result.is_err());

    // t[0] should also fail (1-indexed)
    let result = bmc.get_tuple_element_at_step("t", 0, 0);
    assert!(result.is_err());
}

// ================================================================
// Record field not found
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_field_not_found() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var("r", vec![("a".to_string(), TlaSort::Int)])
        .unwrap();

    // r.nonexistent should fail
    let result = bmc.get_record_field_at_step("r", "nonexistent", 0);
    assert!(result.is_err());
}

// ================================================================
// Primed record field access: r'.a
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_record_primed_field_access() {
    let mut bmc = bmc_array(1);
    bmc.declare_record_var("r", vec![("a".to_string(), TlaSort::Int)])
        .unwrap();

    // r'.a = 42  (access field on primed record variable)
    let r_prime_a = spanned(Expr::RecordAccess(
        Box::new(spanned(Expr::Prime(Box::new(ident("r"))))),
        RecordFieldName::new(sstr("a")),
    ));
    let eq = spanned(Expr::Eq(Box::new(r_prime_a), Box::new(int(42))));

    bmc.current_step = 0;
    let term = bmc.translate_bool(&eq).unwrap();
    bmc.assert(term);

    // Verify: r__f_a__1 should be 42
    let a1 = bmc.get_record_field_at_step("r", "a", 1).unwrap();
    assert_term_eq_int(&mut bmc, a1, 42);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// ================================================================
// Primed tuple indexing: t'[1]
// ================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_tuple_primed_index() {
    let mut bmc = bmc_array(1);
    bmc.declare_tuple_var("t", vec![TlaSort::Int]).unwrap();

    // t'[1] = 7  (index on primed tuple variable)
    let t_prime_1 = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Prime(Box::new(ident("t"))))),
        Box::new(int(1)),
    ));
    let eq = spanned(Expr::Eq(Box::new(t_prime_1), Box::new(int(7))));

    bmc.current_step = 0;
    let term = bmc.translate_bool(&eq).unwrap();
    bmc.assert(term);

    // Verify: t__e_1__1 should be 7
    let e1_1 = bmc.get_tuple_element_at_step("t", 1, 1).unwrap();
    assert_term_eq_int(&mut bmc, e1_1, 7);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}
