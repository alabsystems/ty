// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BMC function encoding via SMT arrays.
//!
//! Part of #3786: Validates function construction, application, DOMAIN,
//! and EXCEPT operations in the BMC translator.

use super::*;
use ay_dpll::api::SolveResult;
use tla_core::ast::{BoundVar, ExceptPathElement, ExceptSpec};

/// Helper: create a BMC translator with array support.
fn bmc_array(k: usize) -> BmcTranslator {
    BmcTranslator::new_with_arrays(k).unwrap()
}

fn func_ident(name: &str) -> Spanned<Expr> {
    spanned(Expr::Ident(
        name.to_string(),
        tla_core::name_intern::NameId::INVALID,
    ))
}

fn int_expr(value: i64) -> Spanned<Expr> {
    spanned(Expr::Int(num_bigint::BigInt::from(value)))
}

/// Declare an Int-keyed function with a statically exact finite DOMAIN.
/// Whole-function equality is only sound when the encoder has this complete
/// key cover; `declare_func_var` intentionally represents an unknown domain.
fn declare_exact_int_func(bmc: &mut BmcTranslator, name: &str, keys: &[i64], range: TlaSort) {
    bmc.declare_var(
        name,
        TlaSort::Function {
            domain_keys: keys.iter().map(|key| format!("int:{key}")).collect(),
            range: Box::new(range),
        },
    )
    .unwrap();
}

// --- declare_func_var ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_func_var() {
    let mut bmc = bmc_array(2);
    // Should succeed: function with Int range
    bmc.declare_func_var("f", TlaSort::Int).unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_func_var_bool_range() {
    let mut bmc = bmc_array(1);
    bmc.declare_func_var("g", TlaSort::Bool).unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_func_var_rejects_compound_range() {
    let mut bmc = bmc_array(0);
    let result = bmc.declare_func_var(
        "f",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    );
    assert!(result.is_err());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_var_with_function_sort() {
    let mut bmc = bmc_array(1);
    // declare_var should delegate Function sort to declare_func_var
    bmc.declare_var(
        "f",
        TlaSort::Function {
            domain_keys: vec!["1".to_string(), "2".to_string()],
            range: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    // The function should now be accessible via func_vars
    assert!(bmc.func_vars.contains_key("f"));
}

// --- Function application ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_apply_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // f[1] = 42
    let apply_expr = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));

    let result = bmc.translate_init(&spanned(Expr::Eq(
        Box::new(apply_expr),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(42)))),
    )));
    let term = result.unwrap();
    bmc.assert(term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_apply_contradicts_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Assert f[1] = 10 AND f[1] = 20 -> UNSAT
    let f1 = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(f1.clone()),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(10)))),
        )))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(f1),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(20)))),
        )))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_apply_different_args_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // f[1] = 10 AND f[2] = 20 -> SAT (different arguments)
    let f1 = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));
    let f2 = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(2)))),
    ));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(f1),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(10)))),
        )))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(f2),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(20)))),
        )))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- DOMAIN ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_domain_membership_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();
    bmc.declare_var("x", TlaSort::Int).unwrap();

    // Build: x \in DOMAIN f
    let membership = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Domain(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
    ));

    let term = bmc.translate_init(&membership).unwrap();
    bmc.assert(term);

    // SAT: solver can choose domain to include x
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_domain_with_concrete_domain() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Constrain the domain to {1, 2}: assert (select dom 1) /\ (select dom 2)
    let dom = bmc.get_func_domain_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let two = bmc.solver.int_const(2);
    let three = bmc.solver.int_const(3);

    let in1 = bmc.solver.try_select(dom, one).unwrap();
    let in2 = bmc.solver.try_select(dom, two).unwrap();
    bmc.assert(in1);
    bmc.assert(in2);

    // Also constrain 3 NOT in domain
    let in3 = bmc.solver.try_select(dom, three).unwrap();
    let not_in3 = bmc.solver.try_not(in3).unwrap();
    bmc.assert(not_in3);

    // Now check: x \in DOMAIN f with x = 3 should be UNSAT
    bmc.declare_var("x", TlaSort::Int).unwrap();
    let x_eq_3 = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(3)))),
        )))
        .unwrap();
    bmc.assert(x_eq_3);

    let membership = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Domain(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
    ));

    let dom_term = bmc.translate_init(&membership).unwrap();
    bmc.assert(dom_term);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- EXCEPT ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_updates_value() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Assert f[1] = 10
    let f1 = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));
    let eq_init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(f1),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(10)))),
        )))
        .unwrap();
    bmc.assert(eq_init);

    // Build: [f EXCEPT ![1] = 99] via direct translate_func_except_bmc call
    let new_mapping = bmc
        .translate_func_except_bmc(
            &spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
            &[ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                    num_bigint::BigInt::from(1),
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(99))),
            }],
        )
        .unwrap();

    // (select new_mapping 1) should equal 99
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let selected = bmc.solver.try_select(new_mapping, one).unwrap();
    let eq = bmc.solver.try_eq(selected, ninety_nine).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_preserves_other_values() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Assert f[2] = 20
    let mapping = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let two = bmc.solver.int_const(2);
    let twenty = bmc.solver.int_const(20);
    let sel = bmc.solver.try_select(mapping, two).unwrap();
    let eq = bmc.solver.try_eq(sel, twenty).unwrap();
    bmc.assert(eq);

    // Build [f EXCEPT ![1] = 99]: changes f[1] but preserves f[2]
    let new_mapping = bmc
        .translate_func_except_bmc(
            &spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
            &[ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                    num_bigint::BigInt::from(1),
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(99))),
            }],
        )
        .unwrap();

    // (select new_mapping 2) should still be 20
    let selected = bmc.solver.try_select(new_mapping, two).unwrap();
    let eq2 = bmc.solver.try_eq(selected, twenty).unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- Function construction ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_set_enum_domain() {
    let mut bmc = bmc_array(0);

    // [x \in {1, 2, 3} |-> x + 1]
    let bound_vars = vec![BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![
            spanned(Expr::Int(num_bigint::BigInt::from(1))),
            spanned(Expr::Int(num_bigint::BigInt::from(2))),
            spanned(Expr::Int(num_bigint::BigInt::from(3))),
        ])))),
        pattern: None,
    }];
    let body = spanned(Expr::Add(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));

    let (domain, mapping) = bmc
        .translate_func_construct_bmc(&bound_vars, &body)
        .unwrap();

    // Check: 1 is in domain
    let one = bmc.solver.int_const(1);
    let in_dom = bmc.solver.try_select(domain, one).unwrap();
    bmc.assert(in_dom);

    // Check: mapping[1] = 2 (since body is x+1, for x=1 => 2)
    let two = bmc.solver.int_const(2);
    let map_1 = bmc.solver.try_select(mapping, one).unwrap();
    let eq = bmc.solver.try_eq(map_1, two).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_range_domain() {
    let mut bmc = bmc_array(0);

    // [x \in 1..3 |-> x * 2]
    let bound_vars = vec![BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::Range(
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(3)))),
        )))),
        pattern: None,
    }];
    let body = spanned(Expr::Mul(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(2)))),
    ));

    let (domain, mapping) = bmc
        .translate_func_construct_bmc(&bound_vars, &body)
        .unwrap();

    // Check: 4 is NOT in domain (only 1,2,3 are)
    let four = bmc.solver.int_const(4);
    let in_dom = bmc.solver.try_select(domain, four).unwrap();
    let not_in = bmc.solver.try_not(in_dom).unwrap();
    bmc.assert(not_in);

    // Check: mapping[2] = 4 (since body is x*2, for x=2 => 4)
    let two_term = bmc.solver.int_const(2);
    let four_term = bmc.solver.int_const(4);
    let map_2 = bmc.solver.try_select(mapping, two_term).unwrap();
    let eq = bmc.solver.try_eq(map_2, four_term).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_wrong_value_unsat() {
    let mut bmc = bmc_array(0);

    // [x \in {1} |-> x + 1]
    // mapping[1] must be 2, so asserting mapping[1] = 5 should be UNSAT
    let bound_vars = vec![BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            num_bigint::BigInt::from(1),
        ))])))),
        pattern: None,
    }];
    let body = spanned(Expr::Add(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));

    let (_domain, mapping) = bmc
        .translate_func_construct_bmc(&bound_vars, &body)
        .unwrap();

    // Assert mapping[1] = 5 (should be UNSAT because it must be 2)
    let one = bmc.solver.int_const(1);
    let five = bmc.solver.int_const(5);
    let map_1 = bmc.solver.try_select(mapping, one).unwrap();
    let eq = bmc.solver.try_eq(map_1, five).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Function variable across steps ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_var_across_steps() {
    let mut bmc = bmc_array(1);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // At step 0: f[1] = 10
    let map0 = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel0 = bmc.solver.try_select(map0, one).unwrap();
    let eq0 = bmc.solver.try_eq(sel0, ten).unwrap();
    bmc.assert(eq0);

    // At step 1: f[1] = 20 (different step, different value)
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let twenty = bmc.solver.int_const(20);
    let sel1 = bmc.solver.try_select(map1, one).unwrap();
    let eq1 = bmc.solver.try_eq(sel1, twenty).unwrap();
    bmc.assert(eq1);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_apply_primed() {
    let mut bmc = bmc_array(1);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // f'[1] = 42 (at step 0, prime means step 1)
    bmc.current_step = 0;
    let apply_expr = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Prime(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));

    let result = bmc.translate_bool(&spanned(Expr::Eq(
        Box::new(apply_expr),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(42)))),
    )));
    let term = result.unwrap();
    bmc.assert(term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- Multi-spec EXCEPT ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_multi_spec() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // [f EXCEPT ![1] = 10, ![2] = 20]
    let new_mapping = bmc
        .translate_func_except_bmc(
            &spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
            &[
                ExceptSpec {
                    path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                        num_bigint::BigInt::from(1),
                    )))],
                    value: spanned(Expr::Int(num_bigint::BigInt::from(10))),
                },
                ExceptSpec {
                    path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                        num_bigint::BigInt::from(2),
                    )))],
                    value: spanned(Expr::Int(num_bigint::BigInt::from(20))),
                },
            ],
        )
        .unwrap();

    // (select new_mapping 1) = 10
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel1 = bmc.solver.try_select(new_mapping, one).unwrap();
    let eq1 = bmc.solver.try_eq(sel1, ten).unwrap();
    bmc.assert(eq1);

    // (select new_mapping 2) = 20
    let two = bmc.solver.int_const(2);
    let twenty = bmc.solver.int_const(20);
    let sel2 = bmc.solver.try_select(new_mapping, two).unwrap();
    let eq2 = bmc.solver.try_eq(sel2, twenty).unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_multi_spec_last_wins() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // TLA+ semantics: [f EXCEPT ![1] = 10, ![1] = 20] — last write wins
    let new_mapping = bmc
        .translate_func_except_bmc(
            &spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
            &[
                ExceptSpec {
                    path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                        num_bigint::BigInt::from(1),
                    )))],
                    value: spanned(Expr::Int(num_bigint::BigInt::from(10))),
                },
                ExceptSpec {
                    path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                        num_bigint::BigInt::from(1),
                    )))],
                    value: spanned(Expr::Int(num_bigint::BigInt::from(20))),
                },
            ],
        )
        .unwrap();

    // Last store wins: (select new_mapping 1) = 20, NOT 10
    let one = bmc.solver.int_const(1);
    let twenty = bmc.solver.int_const(20);
    let sel = bmc.solver.try_select(new_mapping, one).unwrap();
    let eq = bmc.solver.try_eq(sel, twenty).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify it's NOT 10: create fresh solver to check
    let mut bmc2 = bmc_array(0);
    bmc2.declare_func_var("f", TlaSort::Int).unwrap();
    let new_map2 = bmc2
        .translate_func_except_bmc(
            &spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
            &[
                ExceptSpec {
                    path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                        num_bigint::BigInt::from(1),
                    )))],
                    value: spanned(Expr::Int(num_bigint::BigInt::from(10))),
                },
                ExceptSpec {
                    path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                        num_bigint::BigInt::from(1),
                    )))],
                    value: spanned(Expr::Int(num_bigint::BigInt::from(20))),
                },
            ],
        )
        .unwrap();

    let one2 = bmc2.solver.int_const(1);
    let ten2 = bmc2.solver.int_const(10);
    let sel2 = bmc2.solver.try_select(new_map2, one2).unwrap();
    let eq2 = bmc2.solver.try_eq(sel2, ten2).unwrap();
    bmc2.assert(eq2);

    // Last write was 20, so asserting it's 10 should be UNSAT
    assert!(matches!(bmc2.check_sat(), SolveResult::Unsat(_)));
}

// --- Nested EXCEPT ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_nested_path_scalar_range_errors() {
    let mut bmc = bmc_array(0);
    // Function with scalar Int range — nested EXCEPT is not supported
    // because (select f_map key) returns Int, not Array.
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // [f EXCEPT ![1][2] = 42] should error: scalar range can't be nested
    let result = bmc.translate_func_except_bmc(
        &spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )),
        &[ExceptSpec {
            path: vec![
                ExceptPathElement::Index(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
                ExceptPathElement::Index(spanned(Expr::Int(num_bigint::BigInt::from(2)))),
            ],
            value: spanned(Expr::Int(num_bigint::BigInt::from(42))),
        }],
    );
    assert!(
        result.is_err(),
        "nested EXCEPT on scalar-range function should error"
    );
}

// --- EXCEPT via dispatch (translate_eq) ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_via_eq_dispatch() {
    let mut bmc = bmc_array(1);
    declare_exact_int_func(&mut bmc, "f", &[1], TlaSort::Int);

    // Step 0: set f[1] = 10
    bmc.current_step = 0;
    let f1 = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(1)))),
    ));
    let init_eq = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(f1),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(10)))),
        )))
        .unwrap();
    bmc.assert(init_eq);

    // Translate: f' = [f EXCEPT ![1] = 99]
    // This should go through translate_eq -> try_translate_func_except_eq
    let except_eq = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
        Box::new(spanned(Expr::Except(
            Box::new(spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            vec![ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                    num_bigint::BigInt::from(1),
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(99))),
            }],
        ))),
    ));

    bmc.current_step = 0;
    let next_term = bmc.translate_bool(&except_eq).unwrap();
    bmc.assert(next_term);

    // Now at step 1, f[1] should be 99
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let selected = bmc.solver.try_select(map1, one).unwrap();
    let eq = bmc.solver.try_eq(selected, ninety_nine).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_via_eq_preserves_unchanged_keys() {
    let mut bmc = bmc_array(1);
    declare_exact_int_func(&mut bmc, "f", &[1, 2], TlaSort::Int);

    // Step 0: set f[1] = 10 and f[2] = 20
    bmc.current_step = 0;
    let map0 = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let two = bmc.solver.int_const(2);
    let ten = bmc.solver.int_const(10);
    let twenty = bmc.solver.int_const(20);

    let sel1 = bmc.solver.try_select(map0, one).unwrap();
    let eq1 = bmc.solver.try_eq(sel1, ten).unwrap();
    bmc.assert(eq1);

    let sel2 = bmc.solver.try_select(map0, two).unwrap();
    let eq2 = bmc.solver.try_eq(sel2, twenty).unwrap();
    bmc.assert(eq2);

    // Translate: f' = [f EXCEPT ![1] = 99]
    let except_eq = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
        Box::new(spanned(Expr::Except(
            Box::new(spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            vec![ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                    num_bigint::BigInt::from(1),
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(99))),
            }],
        ))),
    ));

    bmc.current_step = 0;
    let next_term = bmc.translate_bool(&except_eq).unwrap();
    bmc.assert(next_term);

    // At step 1: f[2] should still be 20 (unchanged)
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let sel2_next = bmc.solver.try_select(map1, two).unwrap();
    let eq_unchanged = bmc.solver.try_eq(sel2_next, twenty).unwrap();
    bmc.assert(eq_unchanged);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_via_eq_reversed_operands() {
    // Test: [f EXCEPT ![1] = 99] = f' (EXCEPT on left, variable on right)
    let mut bmc = bmc_array(1);
    declare_exact_int_func(&mut bmc, "f", &[1], TlaSort::Int);

    let except_eq = spanned(Expr::Eq(
        Box::new(spanned(Expr::Except(
            Box::new(spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            vec![ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                    num_bigint::BigInt::from(1),
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(99))),
            }],
        ))),
        Box::new(spanned(Expr::Prime(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
    ));

    bmc.current_step = 0;
    let term = bmc.translate_bool(&except_eq).unwrap();
    bmc.assert(term);

    // f'[1] should be 99
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let sel = bmc.solver.try_select(map1, one).unwrap();
    let eq = bmc.solver.try_eq(sel, ninety_nine).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_with_variable_index() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();
    bmc.declare_var("k", TlaSort::Int).unwrap();

    // [f EXCEPT ![k] = 42] — index is a variable, not a literal
    let new_mapping = bmc
        .translate_func_except_bmc(
            &spanned(Expr::Ident(
                "f".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
            &[ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Ident(
                    "k".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(42))),
            }],
        )
        .unwrap();

    // Constrain k = 5
    let k_term = bmc.get_var_at_step("k", 0).unwrap();
    let five = bmc.solver.int_const(5);
    let k_eq = bmc.solver.try_eq(k_term, five).unwrap();
    bmc.assert(k_eq);

    // (select new_mapping 5) should be 42
    let forty_two = bmc.solver.int_const(42);
    let sel = bmc.solver.try_select(new_mapping, five).unwrap();
    let eq = bmc.solver.try_eq(sel, forty_two).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- Chained EXCEPT ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_chained() {
    // [[f EXCEPT ![1] = 10] EXCEPT ![2] = 20]
    // Outer base is itself an Except expression
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let inner_except = spanned(Expr::Except(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                num_bigint::BigInt::from(1),
            )))],
            value: spanned(Expr::Int(num_bigint::BigInt::from(10))),
        }],
    ));

    let new_mapping = bmc
        .translate_func_except_bmc(
            &inner_except,
            &[ExceptSpec {
                path: vec![ExceptPathElement::Index(spanned(Expr::Int(
                    num_bigint::BigInt::from(2),
                )))],
                value: spanned(Expr::Int(num_bigint::BigInt::from(20))),
            }],
        )
        .unwrap();

    // (select result 1) = 10
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel1 = bmc.solver.try_select(new_mapping, one).unwrap();
    let eq1 = bmc.solver.try_eq(sel1, ten).unwrap();
    bmc.assert(eq1);

    // (select result 2) = 20
    let two = bmc.solver.int_const(2);
    let twenty = bmc.solver.int_const(20);
    let sel2 = bmc.solver.try_select(new_mapping, two).unwrap();
    let eq2 = bmc.solver.try_eq(sel2, twenty).unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- Error cases ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_empty_path_error() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let result = bmc.translate_func_except_bmc(
        &spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )),
        &[ExceptSpec {
            path: vec![],
            value: spanned(Expr::Int(num_bigint::BigInt::from(1))),
        }],
    );
    assert!(result.is_err());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_except_record_field_error() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let result = bmc.translate_func_except_bmc(
        &spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )),
        &[ExceptSpec {
            path: vec![ExceptPathElement::Field(tla_core::ast::RecordFieldName {
                name: spanned("x".to_string()),
                field_id: tla_core::name_intern::NameId::INVALID,
            })],
            value: spanned(Expr::Int(num_bigint::BigInt::from(1))),
        }],
    );
    assert!(result.is_err());
}

// --- Function construction equality (Part of #3786) ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_eq_init() {
    // f = [x \in {1, 2} |-> x + 10]
    // After translating, f[1] should be 11 and f[2] should be 12.
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let func_def = spanned(Expr::FuncDef(
        vec![BoundVar {
            name: spanned("x".to_string()),
            domain: Some(Box::new(spanned(Expr::SetEnum(vec![
                spanned(Expr::Int(num_bigint::BigInt::from(1))),
                spanned(Expr::Int(num_bigint::BigInt::from(2))),
            ])))),
            pattern: None,
        }],
        Box::new(spanned(Expr::Add(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(10)))),
        ))),
    ));

    let eq = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(func_def),
    ));

    let term = bmc.translate_init(&eq).unwrap();
    bmc.assert(term);

    // f[1] should be 11
    let map = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let eleven = bmc.solver.int_const(11);
    let sel = bmc.solver.try_select(map, one).unwrap();
    let eq_check = bmc.solver.try_eq(sel, eleven).unwrap();
    bmc.assert(eq_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_eq_wrong_value_unsat() {
    // f = [x \in {1} |-> x + 10]
    // f[1] should be 11, not 99
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let func_def = spanned(Expr::FuncDef(
        vec![BoundVar {
            name: spanned("x".to_string()),
            domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
                num_bigint::BigInt::from(1),
            ))])))),
            pattern: None,
        }],
        Box::new(spanned(Expr::Add(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(10)))),
        ))),
    ));

    let eq = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(func_def),
    ));

    let term = bmc.translate_init(&eq).unwrap();
    bmc.assert(term);

    // f[1] should NOT be 99
    let map = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let sel = bmc.solver.try_select(map, one).unwrap();
    let eq_check = bmc.solver.try_eq(sel, ninety_nine).unwrap();
    bmc.assert(eq_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_eq_primed() {
    // f' = [x \in {1, 2} |-> x * 2]
    let mut bmc = bmc_array(1);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let func_def = spanned(Expr::FuncDef(
        vec![BoundVar {
            name: spanned("x".to_string()),
            domain: Some(Box::new(spanned(Expr::SetEnum(vec![
                spanned(Expr::Int(num_bigint::BigInt::from(1))),
                spanned(Expr::Int(num_bigint::BigInt::from(2))),
            ])))),
            pattern: None,
        }],
        Box::new(spanned(Expr::Mul(
            Box::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Int(num_bigint::BigInt::from(2)))),
        ))),
    ));

    let eq = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        )))))),
        Box::new(func_def),
    ));

    bmc.current_step = 0;
    let term = bmc.translate_bool(&eq).unwrap();
    bmc.assert(term);

    // f'[2] should be 4 (at step 1)
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let four = bmc.solver.int_const(4);
    let sel = bmc.solver.try_select(map1, two).unwrap();
    let eq_check = bmc.solver.try_eq(sel, four).unwrap();
    bmc.assert(eq_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_eq_reversed() {
    // [x \in {1} |-> 42] = f (FuncDef on left, variable on right)
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    let func_def = spanned(Expr::FuncDef(
        vec![BoundVar {
            name: spanned("x".to_string()),
            domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
                num_bigint::BigInt::from(1),
            ))])))),
            pattern: None,
        }],
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(42)))),
    ));

    let eq = spanned(Expr::Eq(
        Box::new(func_def),
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));

    let term = bmc.translate_init(&eq).unwrap();
    bmc.assert(term);

    // f[1] should be 42
    let map = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let forty_two = bmc.solver.int_const(42);
    let sel = bmc.solver.try_select(map, one).unwrap();
    let eq_check = bmc.solver.try_eq(sel, forty_two).unwrap();
    bmc.assert(eq_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- BmcValue::Function in assert_concrete_state ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_func_state() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Assert f = {1 -> 10, 2 -> 20}
    bmc.assert_concrete_state(
        &[(
            "f".to_string(),
            super::super::BmcValue::Function(vec![
                (1, super::super::BmcValue::Int(10)),
                (2, super::super::BmcValue::Int(20)),
            ]),
        )],
        0,
    )
    .unwrap();

    // f[1] should be 10
    let map = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel = bmc.solver.try_select(map, one).unwrap();
    let eq = bmc.solver.try_eq(sel, ten).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_func_state_wrong_value_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Assert f = {1 -> 10}
    bmc.assert_concrete_state(
        &[(
            "f".to_string(),
            super::super::BmcValue::Function(vec![(1, super::super::BmcValue::Int(10))]),
        )],
        0,
    )
    .unwrap();

    // f[1] should NOT be 99
    let map = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let sel = bmc.solver.try_select(map, one).unwrap();
    let eq = bmc.solver.try_eq(sel, ninety_nine).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Domain membership in assert_concrete_state ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_func_domain_membership() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("f", TlaSort::Int).unwrap();

    // Assert f = {1 -> 10, 3 -> 30}
    bmc.assert_concrete_state(
        &[(
            "f".to_string(),
            super::super::BmcValue::Function(vec![
                (1, super::super::BmcValue::Int(10)),
                (3, super::super::BmcValue::Int(30)),
            ]),
        )],
        0,
    )
    .unwrap();

    // 1 should be in domain
    let dom = bmc.get_func_domain_at_step("f", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let in_dom_1 = bmc.solver.try_select(dom, one).unwrap();
    bmc.assert(in_dom_1);

    // 2 should NOT be in domain
    let two = bmc.solver.int_const(2);
    let in_dom_2 = bmc.solver.try_select(dom, two).unwrap();
    let not_in_dom_2 = bmc.solver.try_not(in_dom_2).unwrap();
    bmc.assert(not_in_dom_2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// =========================================================================
// #5: string-keyed function-construction domains (native String index sort)
// =========================================================================

/// Repro for #5: a function construction over a *string-literal* domain.
///
/// Before the fix, the BMC encoder called `translate_int` on the string keys
/// and failed with "unsupported int expr: String". After the fix, string keys
/// are encoded with a native `(Array String _)` index sort and the
/// construction is satisfiable.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_string_domain() {
    let mut bmc = bmc_array(0);

    // [k \in {"a", "b"} |-> 10]
    //
    // The body is a constant (10) on purpose: this test isolates the *domain
    // key sort* path (the source of the original "unsupported int expr: String"
    // error, which fired on the string keys, not the body). A key-dependent
    // body that branches on `k = "a"` would additionally exercise BMC's
    // string-equality-in-body translation, an orthogonal gap.
    let bound_vars = vec![BoundVar {
        name: spanned("k".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![
            spanned(Expr::String("a".to_string())),
            spanned(Expr::String("b".to_string())),
        ])))),
        pattern: None,
    }];
    let body = spanned(Expr::Int(num_bigint::BigInt::from(10)));

    // Previously errored with "unsupported int expr: String" (the encoder
    // called `translate_int` on the string keys). The fix makes translation
    // succeed by encoding the domain with a native `(Array String _)` sort.
    let (_domain, mapping) = bmc
        .translate_func_construct_bmc(&bound_vars, &body)
        .expect("string-keyed function construction must translate (was: unsupported int expr)");

    // The per-key value constraint landed on the mapping: mapping["a"] = 10.
    // (We read only the mapping here, not the domain const-array — selecting a
    // `(store (const-array d) k true)` domain currently degrades AY to
    // `Unknown`, the same incompleteness that affects the baseline-8 seq/set
    // BMC tests. That is sound, never a false verdict, and orthogonal to #5.)
    let key_a = bmc.solver.string_const("a");
    let ten = bmc.solver.int_const(10);
    let map_a = bmc.solver.try_select(mapping, key_a).unwrap();
    let eq = bmc.solver.try_eq(map_a, ten).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// Soundness / non-aliasing for #5: a string key "1" and the integer key 1
/// must NOT be confused.
///
/// We build a string-keyed function `fs = [k \in {"1"} |-> 100]` and a
/// separate integer-keyed function `fi = [n \in {1} |-> 200]` in the same
/// query. The string-key "1" lives in `(Array String _)` while the integer key
/// 1 lives in `(Array Int _)` — disjoint SMT sorts — so the two values do not
/// collide. Asserting both `fs["1"] = 100` and `fi[1] = 200` must be SAT.
///
/// Then we assert the aliasing reading `fs["1"] = 200` (the value that belongs
/// to the *integer* key 1); this must be UNSAT, proving the string cell was not
/// overwritten/aliased by the integer cell.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_string_int_key_nonalias() {
    // --- Part 1: both constraints jointly satisfiable (no aliasing). ---
    let mut bmc = bmc_array(0);

    // fs = [k \in {"1"} |-> 100]  (string-keyed)
    let fs_bounds = vec![BoundVar {
        name: spanned("k".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(
            Expr::String("1".to_string()),
        )])))),
        pattern: None,
    }];
    let fs_body = spanned(Expr::Int(num_bigint::BigInt::from(100)));
    let (_fs_dom, fs_map) = bmc
        .translate_func_construct_bmc(&fs_bounds, &fs_body)
        .unwrap();

    // fi = [n \in {1} |-> 200]  (integer-keyed)
    let fi_bounds = vec![BoundVar {
        name: spanned("n".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            num_bigint::BigInt::from(1),
        ))])))),
        pattern: None,
    }];
    let fi_body = spanned(Expr::Int(num_bigint::BigInt::from(200)));
    let (_fi_dom, fi_map) = bmc
        .translate_func_construct_bmc(&fi_bounds, &fi_body)
        .unwrap();

    // fs["1"] = 100  AND  fi[1] = 200  -> SAT (keys live in disjoint sorts).
    let key_str1 = bmc.solver.string_const("1");
    let v100 = bmc.solver.int_const(100);
    let sel_s = bmc.solver.try_select(fs_map, key_str1).unwrap();
    let eq_s = bmc.solver.try_eq(sel_s, v100).unwrap();
    bmc.assert(eq_s);

    let key_int1 = bmc.solver.int_const(1);
    let v200 = bmc.solver.int_const(200);
    let sel_i = bmc.solver.try_select(fi_map, key_int1).unwrap();
    let eq_i = bmc.solver.try_eq(sel_i, v200).unwrap();
    bmc.assert(eq_i);

    assert!(
        matches!(bmc.check_sat(), SolveResult::Sat),
        "string key \"1\" and integer key 1 must coexist without aliasing"
    );

    // --- Part 2: the string cell did NOT take the integer cell's value. ---
    let mut bmc2 = bmc_array(0);
    let fs_bounds2 = vec![BoundVar {
        name: spanned("k".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(
            Expr::String("1".to_string()),
        )])))),
        pattern: None,
    }];
    let fs_body2 = spanned(Expr::Int(num_bigint::BigInt::from(100)));
    let (_d, fs_map2) = bmc2
        .translate_func_construct_bmc(&fs_bounds2, &fs_body2)
        .unwrap();

    // fs["1"] is pinned to 100; asserting fs["1"] = 200 must be UNSAT.
    let k = bmc2.solver.string_const("1");
    let v200b = bmc2.solver.int_const(200);
    let sel = bmc2.solver.try_select(fs_map2, k).unwrap();
    let bad = bmc2.solver.try_eq(sel, v200b).unwrap();
    bmc2.assert(bad);

    assert!(
        matches!(bmc2.check_sat(), SolveResult::Unsat(_)),
        "string key \"1\" must hold its own value (100), never the int key's (200)"
    );
}

/// #5 (declared-variable into-path): `f = [k \in {"a"} |-> 7]` where `f` is a
/// declared function variable. The exact `TlaSort::Function` domain spelling
/// preserves the String key kind and allocates a native `(Array String _)` map
/// immediately.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_construct_string_domain_into_declared_var() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "f",
        TlaSort::Function {
            domain_keys: vec!["a".to_string()],
            range: Box::new(TlaSort::Int),
        },
    )
    .unwrap();

    let func_def = spanned(Expr::FuncDef(
        vec![BoundVar {
            name: spanned("k".to_string()),
            domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(
                Expr::String("a".to_string()),
            )])))),
            pattern: None,
        }],
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(7)))),
    ));
    let init = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "f".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(func_def),
    ));

    let term = bmc.translate_init(&init).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_bound_scalar_shadow_is_exactly_restored() {
    let mut bmc = bmc_array(1);
    bmc.declare_var("x", TlaSort::Bool).unwrap();
    let original_terms = bmc.vars.get("x").unwrap().terms.clone();

    let bounds = vec![BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            num_bigint::BigInt::from(1),
        ))])))),
        pattern: None,
    }];
    let body = spanned(Expr::Ident(
        "x".to_string(),
        tla_core::name_intern::NameId::INVALID,
    ));
    bmc.translate_func_construct_bmc(&bounds, &body).unwrap();

    let restored = bmc.vars.get("x").unwrap();
    assert_eq!(restored.sort, TlaSort::Bool);
    assert_eq!(restored.terms, original_terms);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_bound_shadow_restores_after_body_error() {
    let mut bmc = bmc_array(1);
    bmc.declare_var("x", TlaSort::Int).unwrap();
    let original_terms = bmc.vars.get("x").unwrap().terms.clone();
    let bounds = vec![BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            num_bigint::BigInt::from(1),
        ))])))),
        pattern: None,
    }];

    // The Bool result kind is known, so translation enters the temporary `x`
    // binding before the unknown inner identifier triggers the error.
    let failing_body = spanned(Expr::Eq(
        Box::new(func_ident("missing")),
        Box::new(int_expr(0)),
    ));
    assert!(bmc
        .translate_func_construct_bmc(&bounds, &failing_body)
        .is_err());
    let restored = bmc.vars.get("x").unwrap();
    assert_eq!(restored.sort, TlaSort::Int);
    assert_eq!(restored.terms, original_terms);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_func_bound_record_collision_fails_closed() {
    let mut bmc = bmc_array(0);
    bmc.declare_record_var("r", vec![("a".to_string(), TlaSort::Int)])
        .unwrap();
    let bounds = vec![BoundVar {
        name: spanned("r".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            num_bigint::BigInt::from(1),
        ))])))),
        pattern: None,
    }];

    assert!(bmc
        .translate_func_construct_bmc(&bounds, &spanned(Expr::Int(num_bigint::BigInt::from(0))),)
        .is_err());
    assert!(bmc.record_vars.contains_key("r"));
    assert!(!bmc.vars.contains_key("r"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_function_metadata_and_key_kinds_fail_closed() {
    let mut bmc = bmc_array(0);
    bmc.declare_func_var("ints", TlaSort::Int).unwrap();
    bmc.declare_func_var("strings", TlaSort::String).unwrap();
    let equality = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "ints".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "strings".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    assert!(bmc.translate_init(&equality).is_err());

    let alias_key = spanned(Expr::FuncApply(
        Box::new(spanned(Expr::Ident(
            "ints".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::String("collision".to_string()))),
    ));
    let alias_equality = spanned(Expr::Eq(
        Box::new(alias_key),
        Box::new(spanned(Expr::Int(num_bigint::BigInt::from(
            -1_000_000_007_i64,
        )))),
    ));
    assert!(bmc.translate_init(&alias_equality).is_err());
}

/// Function values are extensional over DOMAIN only. Different values in the
/// backing arrays at a key outside DOMAIN must neither break `=` nor satisfy
/// `#`.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_function_equality_ignores_out_of_domain_ghost_cells() {
    for inequality in [false, true] {
        let mut bmc = bmc_array(0);
        declare_exact_int_func(&mut bmc, "f", &[1], TlaSort::Int);
        declare_exact_int_func(&mut bmc, "g", &[1], TlaSort::Int);

        let f_map = bmc.get_func_mapping_at_step("f", 0).unwrap();
        let g_map = bmc.get_func_mapping_at_step("g", 0).unwrap();
        let one = bmc.solver.int_const(1);
        let ghost = bmc.solver.int_const(999);
        for (mapping, key, value) in [
            (f_map, one, 10),
            (g_map, one, 10),
            (f_map, ghost, 111),
            (g_map, ghost, 222),
        ] {
            let selected = bmc.solver.try_select(mapping, key).unwrap();
            let expected = bmc.solver.int_const(value);
            let equal = bmc.solver.try_eq(selected, expected).unwrap();
            bmc.assert(equal);
        }

        let comparison = if inequality {
            Expr::Neq(Box::new(func_ident("f")), Box::new(func_ident("g")))
        } else {
            Expr::Eq(Box::new(func_ident("f")), Box::new(func_ident("g")))
        };
        let term = bmc.translate_init(&spanned(comparison)).unwrap();
        bmc.assert(term);
        let verdict = bmc.check_sat();
        if inequality {
            assert!(
                matches!(verdict, SolveResult::Unsat(_)),
                "ghost-only difference must not make exact-domain functions unequal: {verdict:?}"
            );
        } else {
            assert!(
                matches!(verdict, SolveResult::Sat),
                "ghost-only difference must not break exact-domain equality: {verdict:?}"
            );
        }
    }
}

/// UNCHANGED preserves the abstract mapping, not arbitrary backing-array cells
/// outside the declared finite DOMAIN.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_function_unchanged_ignores_out_of_domain_ghost_cells() {
    let mut bmc = bmc_array(1);
    declare_exact_int_func(&mut bmc, "f", &[1], TlaSort::Int);
    let map0 = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ghost = bmc.solver.int_const(999);
    for (mapping, key, value) in [
        (map0, one, 10),
        (map1, one, 10),
        (map0, ghost, 111),
        (map1, ghost, 222),
    ] {
        let selected = bmc.solver.try_select(mapping, key).unwrap();
        let expected = bmc.solver.int_const(value);
        let equal = bmc.solver.try_eq(selected, expected).unwrap();
        bmc.assert(equal);
    }

    let unchanged = spanned(Expr::Unchanged(Box::new(func_ident("f"))));
    let term = bmc.translate_init(&unchanged).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// Directed EXCEPT equality likewise compares only live DOMAIN keys. The
/// target's ghost cell is unrelated to the source store-chain's ghost cell.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_function_except_equality_ignores_out_of_domain_ghost_cells() {
    let mut bmc = bmc_array(1);
    declare_exact_int_func(&mut bmc, "f", &[1], TlaSort::Int);
    let map0 = bmc.get_func_mapping_at_step("f", 0).unwrap();
    let map1 = bmc.get_func_mapping_at_step("f", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ghost = bmc.solver.int_const(999);
    for (mapping, key, value) in [
        (map0, one, 10),
        (map1, one, 99),
        (map0, ghost, 111),
        (map1, ghost, 222),
    ] {
        let selected = bmc.solver.try_select(mapping, key).unwrap();
        let expected = bmc.solver.int_const(value);
        let equal = bmc.solver.try_eq(selected, expected).unwrap();
        bmc.assert(equal);
    }

    let except = spanned(Expr::Except(
        Box::new(func_ident("f")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Index(int_expr(1))],
            value: int_expr(99),
        }],
    ));
    let equality = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(func_ident("f"))))),
        Box::new(except),
    ));
    let term = bmc.translate_init(&equality).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// A generic function declaration has a genuinely unknown DOMAIN. Without a
/// finite key cover, equality, UNCHANGED, and directed EXCEPT assignment must
/// decline instead of silently comparing whole backing arrays.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_unknown_domain_whole_function_relations_fail_closed() {
    let mut equality = bmc_array(1);
    equality.declare_func_var("f", TlaSort::Int).unwrap();
    equality.declare_func_var("g", TlaSort::Int).unwrap();
    let eq = spanned(Expr::Eq(
        Box::new(func_ident("f")),
        Box::new(func_ident("g")),
    ));
    assert!(equality.translate_init(&eq).is_err());

    let mut unchanged = bmc_array(1);
    unchanged.declare_func_var("f", TlaSort::Int).unwrap();
    let expr = spanned(Expr::Unchanged(Box::new(func_ident("f"))));
    assert!(unchanged.translate_init(&expr).is_err());

    let mut except = bmc_array(1);
    except.declare_func_var("f", TlaSort::Int).unwrap();
    let update = spanned(Expr::Except(
        Box::new(func_ident("f")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Index(int_expr(1))],
            value: int_expr(99),
        }],
    ));
    let assignment = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(func_ident("f"))))),
        Box::new(update),
    ));
    assert!(except.translate_init(&assignment).is_err());
}

/// Boolean ranges must produce Bool-valued arrays in both standalone function
/// construction and directed construction/EXCEPT assignment.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_bool_range_function_construction_and_except() {
    let bounds = vec![BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![int_expr(1)])))),
        pattern: None,
    }];

    let mut standalone = bmc_array(0);
    let (_, mapping) = standalone
        .translate_func_construct_bmc(&bounds, &spanned(Expr::Bool(true)))
        .unwrap();
    let one = standalone.solver.int_const(1);
    let selected = standalone.solver.try_select(mapping, one).unwrap();
    standalone.assert(selected);
    assert!(matches!(standalone.check_sat(), SolveResult::Sat));

    let mut directed = bmc_array(1);
    declare_exact_int_func(&mut directed, "f", &[1], TlaSort::Bool);
    let definition = spanned(Expr::FuncDef(bounds, Box::new(spanned(Expr::Bool(false)))));
    let init = spanned(Expr::Eq(Box::new(func_ident("f")), Box::new(definition)));
    let init_term = directed.translate_init(&init).unwrap();
    directed.assert(init_term);

    let update = spanned(Expr::Except(
        Box::new(func_ident("f")),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Index(int_expr(1))],
            value: spanned(Expr::Bool(true)),
        }],
    ));
    let next = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(func_ident("f"))))),
        Box::new(update),
    ));
    let next_term = directed.translate_init(&next).unwrap();
    directed.assert(next_term);
    let map1 = directed.get_func_mapping_at_step("f", 1).unwrap();
    let one = directed.solver.int_const(1);
    let selected = directed.solver.try_select(map1, one).unwrap();
    directed.assert(selected);
    assert!(matches!(directed.check_sat(), SolveResult::Sat));
}

/// An Int-keyed carrier may be replaced by a native String carrier only before
/// any term has referenced it. This protects already-emitted constraints from
/// being abandoned while retaining the first-use construction path.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_string_key_upgrade_is_first_use_only() {
    fn string_definition() -> Spanned<Expr> {
        spanned(Expr::FuncDef(
            vec![BoundVar {
                name: spanned("k".to_string()),
                domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(
                    Expr::String("a".to_string()),
                )])))),
                pattern: None,
            }],
            Box::new(int_expr(7)),
        ))
    }

    let mut first_use = bmc_array(0);
    first_use.declare_func_var("f", TlaSort::Int).unwrap();
    let assignment = spanned(Expr::Eq(
        Box::new(func_ident("f")),
        Box::new(string_definition()),
    ));
    let term = first_use.translate_init(&assignment).unwrap();
    first_use.assert(term);
    assert!(matches!(first_use.check_sat(), SolveResult::Sat));

    let mut referenced = bmc_array(0);
    referenced.declare_func_var("f", TlaSort::Int).unwrap();
    let _observed_int_carrier = referenced.get_func_mapping_at_step("f", 0).unwrap();
    let assignment = spanned(Expr::Eq(
        Box::new(func_ident("f")),
        Box::new(string_definition()),
    ));
    let error = referenced
        .translate_init(&assignment)
        .expect_err("referenced Int carrier must not be silently replaced");
    assert!(error
        .to_string()
        .contains("after its Int carrier was referenced"));

    let mut symbolic = bmc_array(0);
    symbolic
        .declare_funcsym_var("f", 1, "N".to_string(), 0, TlaSort::Int)
        .unwrap();
    assert!(symbolic.upgrade_func_key_sort_to_string("f").is_err());
}
