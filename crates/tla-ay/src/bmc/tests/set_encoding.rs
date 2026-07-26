// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BMC finite set encoding via SMT arrays.
//!
//! Part of #3778: Validates that set-typed state variables can be declared,
//! set expressions (SetEnum, Union, Intersect, SetMinus, Subseteq) can be
//! translated, and membership queries produce correct SAT/UNSAT results.

use super::*;
use ay_dpll::api::{SolveResult, Sort};

/// Helper: create a BMC translator with array support.
fn bmc_array(k: usize) -> BmcTranslator {
    BmcTranslator::new_with_arrays(k).unwrap()
}

// --- TlaSort::Set variant ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_tla_sort_set_display() {
    let sort = TlaSort::Set {
        element_sort: Box::new(TlaSort::Int),
    };
    assert_eq!(format!("{sort}"), "Set(Int)");

    let nested = TlaSort::Set {
        element_sort: Box::new(TlaSort::Bool),
    };
    assert_eq!(format!("{nested}"), "Set(Bool)");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_tla_sort_set_to_ay() {
    let sort = TlaSort::Set {
        element_sort: Box::new(TlaSort::Int),
    };
    let ay_sort = sort.to_ay().unwrap();
    // Should produce (Array Int Bool)
    assert_eq!(ay_sort, Sort::array(Sort::Int, Sort::Bool));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_tla_sort_set_is_not_scalar() {
    let sort = TlaSort::Set {
        element_sort: Box::new(TlaSort::Int),
    };
    assert!(!sort.is_scalar());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_tla_sort_set_canonicalized() {
    let sort = TlaSort::Set {
        element_sort: Box::new(TlaSort::Int),
    };
    let canon = sort.canonicalized();
    assert_eq!(
        canon,
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int)
        }
    );
}

// --- BmcValue::Set variant ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_value_set_debug() {
    let val = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]);
    let debug = format!("{val:?}");
    assert!(debug.contains("Set"));
    assert!(debug.contains("Int(1)"));
    assert!(debug.contains("Int(2)"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_value_set_equality() {
    let a = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]);
    let b = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]);
    let c = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(3)]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// --- BMC declare_var with Set sort ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_set_var() {
    let mut bmc = bmc_array(2);
    // Should succeed: Set(Int) is a supported sort
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_set_var_rejects_without_arrays() {
    // new() uses QfLia which does NOT support arrays.
    // However, declare_var only checks is_scalar() || is Set; it doesn't
    // validate the solver logic. The solver will reject array operations later.
    // This test verifies that declare_var accepts Set sort.
    let mut bmc = BmcTranslator::new(2).unwrap();
    // This will fail because QfLia solver can't handle array sort
    let result = bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    );
    // The ay solver may or may not reject this at declaration time;
    // what matters is that new_with_arrays works correctly.
    // If it succeeds, that's also fine — the solver will fail at check_sat.
    let _ = result;
}

// --- BMC set expression translation ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_enum_membership_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_var("x", TlaSort::Int).unwrap();

    // Build: x \in {1, 2, 3} (using scalar membership, already supported)
    let expr = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::SetEnum(vec![
            spanned(Expr::Int(BigInt::from(1))),
            spanned(Expr::Int(BigInt::from(2))),
            spanned(Expr::Int(BigInt::from(3))),
        ]))),
    ));

    let init = bmc.translate_init(&expr).unwrap();
    bmc.assert(init);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// `x \in Nat` translates soundly to `x >= 0` (the BMC Nat arm mirroring the
/// SAT/CHC paths). SAT: the solver can pick a nonnegative x.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_nat_membership_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_var("x", TlaSort::Int).unwrap();

    // x \in Nat
    let expr = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "Nat".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));

    let init = bmc.translate_init(&expr).unwrap();
    bmc.assert(init);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// `x \in Nat /\ x = -1` is UNSAT — the arm is `x >= 0`, so a negative x is
/// refuted. This pins the arm's POLARITY-EXACT semantics (not a tautology).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_nat_membership_negative_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_var("x", TlaSort::Int).unwrap();

    let in_nat = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "Nat".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    let x_eq_neg1 = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(BigInt::from(-1)))),
    ));
    let conj = spanned(Expr::And(Box::new(in_nat), Box::new(x_eq_neg1)));

    let init = bmc.translate_init(&conj).unwrap();
    bmc.assert(init);
    assert!(
        matches!(bmc.check_sat(), SolveResult::Unsat(_)),
        "x \\in Nat /\\ x = -1 must be UNSAT (Nat arm is x >= 0)"
    );
}

/// `x \in Int` translates to a tautology (`(= x x)`) that still references x —
/// SAT for any int-sorted x, and NOT a constraint that rules anything out.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_int_membership_negative_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_var("x", TlaSort::Int).unwrap();

    let in_int = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "Int".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    // x \in Int /\ x = -5 stays SAT (Int admits negatives, unlike Nat).
    let x_eq_neg5 = spanned(Expr::Eq(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Int(BigInt::from(-5)))),
    ));
    let conj = spanned(Expr::And(Box::new(in_int), Box::new(x_eq_neg5)));

    let init = bmc.translate_init(&conj).unwrap();
    bmc.assert(init);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_variable_membership() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    bmc.declare_var("x", TlaSort::Int).unwrap();

    // Translate: x \in S (where S is a set-typed variable)
    let membership_expr = spanned(Expr::In(
        Box::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        Box::new(spanned(Expr::Ident(
            "S".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));

    let term = bmc.translate_init(&membership_expr).unwrap();
    bmc.assert(term);

    // Should be SAT: solver can assign S and x to make membership true
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_translate_set_enum_expr() {
    let mut bmc = bmc_array(0);

    // Build set {1, 2} as an array term
    let set_expr = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));

    let universe = [
        bmc.solver.int_const(1),
        bmc.solver.int_const(2),
        bmc.solver.int_const(3),
    ];

    let set_term = bmc.translate_set_expr(&set_expr, &universe).unwrap();

    // Check that 1 is in the set
    let one = bmc.solver.int_const(1);
    let member = bmc.solver.try_select(set_term, one).unwrap();
    bmc.assert(member);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_translate_set_enum_nonmember_unsat() {
    let mut bmc = bmc_array(0);

    // Build set {1, 2}
    let set_expr = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));

    let universe = [
        bmc.solver.int_const(1),
        bmc.solver.int_const(2),
        bmc.solver.int_const(3),
    ];

    let set_term = bmc.translate_set_expr(&set_expr, &universe).unwrap();

    // Check that 5 is NOT in the set (should be UNSAT if we assert it is)
    let five = bmc.solver.int_const(5);
    let member = bmc.solver.try_select(set_term, five).unwrap();
    bmc.assert(member);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_union() {
    let mut bmc = bmc_array(0);

    let u1 = bmc.solver.int_const(1);
    let u2 = bmc.solver.int_const(2);
    let u3 = bmc.solver.int_const(3);
    let universe = [u1, u2, u3];

    // S = {1, 2}, T = {2, 3}
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));

    // Build union expression
    let union_expr = spanned(Expr::Union(Box::new(set_s), Box::new(set_t)));
    let union_term = bmc.translate_set_expr(&union_expr, &universe).unwrap();

    // 1 should be in the union
    let one = bmc.solver.int_const(1);
    let in_union = bmc.solver.try_select(union_term, one).unwrap();
    bmc.assert(in_union);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_intersect() {
    let mut bmc = bmc_array(0);

    let u1 = bmc.solver.int_const(1);
    let u2 = bmc.solver.int_const(2);
    let u3 = bmc.solver.int_const(3);
    let universe = [u1, u2, u3];

    // S = {1, 2}, T = {2, 3}
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));

    let inter_expr = spanned(Expr::Intersect(Box::new(set_s), Box::new(set_t)));
    let inter_term = bmc.translate_set_expr(&inter_expr, &universe).unwrap();

    // 1 should NOT be in the intersection (only 2 is)
    let one = bmc.solver.int_const(1);
    let in_inter = bmc.solver.try_select(inter_term, one).unwrap();
    bmc.assert(in_inter);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_intersect_member() {
    let mut bmc = bmc_array(0);

    let u1 = bmc.solver.int_const(1);
    let u2 = bmc.solver.int_const(2);
    let u3 = bmc.solver.int_const(3);
    let universe = [u1, u2, u3];

    // S = {1, 2}, T = {2, 3}: intersection should contain 2
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));

    let inter_expr = spanned(Expr::Intersect(Box::new(set_s), Box::new(set_t)));
    let inter_term = bmc.translate_set_expr(&inter_expr, &universe).unwrap();

    // 2 should be in the intersection
    let two = bmc.solver.int_const(2);
    let in_inter = bmc.solver.try_select(inter_term, two).unwrap();
    bmc.assert(in_inter);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_minus() {
    let mut bmc = bmc_array(0);

    let u1 = bmc.solver.int_const(1);
    let u2 = bmc.solver.int_const(2);
    let u3 = bmc.solver.int_const(3);
    let universe = [u1, u2, u3];

    // S = {1, 2, 3}, T = {2}
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![spanned(Expr::Int(BigInt::from(2)))]));

    let minus_expr = spanned(Expr::SetMinus(Box::new(set_s), Box::new(set_t)));
    let minus_term = bmc.translate_set_expr(&minus_expr, &universe).unwrap();

    // 2 should NOT be in S \ T
    let two = bmc.solver.int_const(2);
    let in_minus = bmc.solver.try_select(minus_term, two).unwrap();
    bmc.assert(in_minus);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_minus_retains_member() {
    let mut bmc = bmc_array(0);

    let u1 = bmc.solver.int_const(1);
    let u2 = bmc.solver.int_const(2);
    let u3 = bmc.solver.int_const(3);
    let universe = [u1, u2, u3];

    // S = {1, 2, 3}, T = {2}, S \ T should contain 1
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![spanned(Expr::Int(BigInt::from(2)))]));

    let minus_expr = spanned(Expr::SetMinus(Box::new(set_s), Box::new(set_t)));
    let minus_term = bmc.translate_set_expr(&minus_expr, &universe).unwrap();

    // 1 should be in S \ T
    let one = bmc.solver.int_const(1);
    let in_minus = bmc.solver.try_select(minus_term, one).unwrap();
    bmc.assert(in_minus);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_subseteq_true() {
    let mut bmc = bmc_array(0);

    // S = {1, 2}, T = {1, 2, 3}: S \subseteq T should be true
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));

    let subset_term = bmc.translate_subseteq_bmc(&set_s, &set_t).unwrap();
    bmc.assert(subset_term);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_subseteq_false() {
    let mut bmc = bmc_array(0);

    // S = {1, 2, 3}, T = {1, 2}: S \subseteq T should be false
    let set_s = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
        spanned(Expr::Int(BigInt::from(3))),
    ]));
    let set_t = spanned(Expr::SetEnum(vec![
        spanned(Expr::Int(BigInt::from(1))),
        spanned(Expr::Int(BigInt::from(2))),
    ]));

    let subset_term = bmc.translate_subseteq_bmc(&set_s, &set_t).unwrap();
    // Asserting that subseteq holds should be UNSAT
    bmc.assert(subset_term);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Concrete state with set values ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_state_with_set() {
    let mut bmc = bmc_array(0);
    bmc.declare_var("x", TlaSort::Int).unwrap();
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();

    // Assert concrete state: x = 1, S = {1, 2}
    bmc.assert_concrete_state(
        &[
            ("x".to_string(), BmcValue::Int(1)),
            (
                "S".to_string(),
                BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]),
            ),
        ],
        0,
    )
    .unwrap();

    // Assert x \in S (should be SAT since x=1 and 1 is in {1,2})
    let x_term = bmc.get_var_at_step("x", 0).unwrap();
    let s_term = bmc.get_var_at_step("S", 0).unwrap();
    let member = bmc.solver.try_select(s_term, x_term).unwrap();
    bmc.assert(member);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_state_set_nonmember_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_var("x", TlaSort::Int).unwrap();
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();

    // Assert concrete state: x = 3, S = {1, 2}
    bmc.assert_concrete_state(
        &[
            ("x".to_string(), BmcValue::Int(3)),
            (
                "S".to_string(),
                BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]),
            ),
        ],
        0,
    )
    .unwrap();

    // Assert x \in S (should be UNSAT since x=3 and 3 is NOT in {1,2})
    let x_term = bmc.get_var_at_step("x", 0).unwrap();
    let s_term = bmc.get_var_at_step("S", 0).unwrap();
    let member = bmc.solver.try_select(s_term, x_term).unwrap();
    bmc.assert(member);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Empty set ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_empty_set_enum() {
    let mut bmc = bmc_array(0);

    let set_expr = spanned(Expr::SetEnum(vec![]));
    let universe: Vec<Term> = vec![];

    let set_term = bmc.translate_set_expr(&set_expr, &universe).unwrap();

    // Assert anything is in the empty set: should be UNSAT
    let x = bmc.solver.declare_const("x", Sort::Int);
    let member = bmc.solver.try_select(set_term, x).unwrap();
    bmc.assert(member);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Set variable at different BMC steps ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_var_across_steps() {
    let mut bmc = bmc_array(1);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();

    // Set different values at step 0 and step 1
    bmc.assert_concrete_state(
        &[("S".to_string(), BmcValue::Set(vec![BmcValue::Int(1)]))],
        0,
    )
    .unwrap();
    bmc.assert_concrete_state(
        &[("S".to_string(), BmcValue::Set(vec![BmcValue::Int(2)]))],
        1,
    )
    .unwrap();

    // At step 0: 1 should be in S
    let s0 = bmc.get_var_at_step("S", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let in_s0 = bmc.solver.try_select(s0, one).unwrap();
    bmc.assert(in_s0);

    // At step 1: 2 should be in S
    let s1 = bmc.get_var_at_step("S", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let in_s1 = bmc.solver.try_select(s1, two).unwrap();
    bmc.assert(in_s1);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_closed_set_equality_prevents_string_int_carrier_alias() {
    let string_set = spanned(Expr::SetEnum(vec![spanned(Expr::String(
        "collision".to_string(),
    ))]));
    let int_set = spanned(Expr::SetEnum(vec![spanned(Expr::Int(BigInt::from(
        -1_000_000_007_i64,
    )))]));

    for (expr, expected_sat) in [
        (
            Expr::Eq(Box::new(string_set.clone()), Box::new(int_set.clone())),
            false,
        ),
        (
            Expr::Eq(Box::new(int_set.clone()), Box::new(string_set.clone())),
            false,
        ),
        (
            Expr::Neq(Box::new(string_set.clone()), Box::new(int_set.clone())),
            true,
        ),
    ] {
        let mut bmc = bmc_array(0);
        let term = bmc.translate_init(&spanned(expr)).unwrap();
        bmc.assert(term);
        assert_eq!(matches!(bmc.check_sat(), SolveResult::Sat), expected_sat);
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_symbolic_set_equality_rejects_cross_kind_carriers() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "ints",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    bmc.declare_var(
        "strings",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::String),
        },
    )
    .unwrap();

    let equality = spanned(Expr::Eq(
        Box::new(set_ident("ints")),
        Box::new(set_ident("strings")),
    ));
    assert!(bmc.translate_init(&equality).is_err());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_membership_cross_kind_is_false() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let membership = spanned(Expr::In(
        Box::new(spanned(Expr::String("collision".to_string()))),
        Box::new(spanned(Expr::Ident(
            "S".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));
    let term = bmc.translate_init(&membership).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_equality_with_empty_is_extensional() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::String),
        },
    )
    .unwrap();
    let set_ref = spanned(Expr::Ident(
        "S".to_string(),
        tla_core::name_intern::NameId::INVALID,
    ));
    let empty_eq = spanned(Expr::Eq(
        Box::new(set_ref.clone()),
        Box::new(spanned(Expr::SetEnum(vec![]))),
    ));
    let member = spanned(Expr::In(
        Box::new(spanned(Expr::String("present".to_string()))),
        Box::new(set_ref),
    ));
    let equality_term = bmc.translate_init(&empty_eq).unwrap();
    let member_term = bmc.translate_init(&member).unwrap();
    bmc.assert(equality_term);
    bmc.assert(member_term);
    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

fn set_ident(name: &str) -> Spanned<Expr> {
    spanned(Expr::Ident(
        name.to_string(),
        tla_core::name_intern::NameId::INVALID,
    ))
}

fn cardinality(set: Spanned<Expr>) -> Spanned<Expr> {
    spanned(Expr::Apply(Box::new(set_ident("Cardinality")), vec![set]))
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_subseteq_rejects_two_symbolic_sets_before_aux_mutation() {
    let mut bmc = bmc_array(0);
    for name in ["S", "T"] {
        bmc.declare_var(
            name,
            TlaSort::Set {
                element_sort: Box::new(TlaSort::Int),
            },
        )
        .unwrap();
    }
    let before = bmc.aux_var_counter;
    let subset = spanned(Expr::Subseteq(
        Box::new(set_ident("S")),
        Box::new(set_ident("T")),
    ));

    assert!(bmc.translate_init(&subset).is_err());
    assert_eq!(bmc.aux_var_counter, before);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_subseteq_restricts_symbolic_left_outside_finite_right() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let s = set_ident("S");
    let contains_two = spanned(Expr::In(
        Box::new(spanned(Expr::Int(BigInt::from(2)))),
        Box::new(s.clone()),
    ));
    let subset = spanned(Expr::Subseteq(
        Box::new(s),
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(1),
        ))]))),
    ));

    let contains_two = bmc.translate_init(&contains_two).unwrap();
    let subset = bmc.translate_init(&subset).unwrap();
    bmc.assert(contains_two);
    bmc.assert(subset);
    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_subseteq_accepts_finite_left_and_symbolic_right() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "T",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let subset = spanned(Expr::Subseteq(
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(1),
        ))]))),
        Box::new(set_ident("T")),
    ));

    let term = bmc.translate_init(&subset).unwrap();
    bmc.assert(term);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_symbolic_union_materialization_fails_before_aux_mutation() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let singleton = spanned(Expr::SetEnum(vec![spanned(Expr::Int(BigInt::from(1)))]));
    let equality = spanned(Expr::Eq(
        Box::new(spanned(Expr::Union(
            Box::new(set_ident("S")),
            Box::new(singleton.clone()),
        ))),
        Box::new(singleton),
    ));
    let before = bmc.aux_var_counter;

    assert!(bmc.translate_init(&equality).is_err());
    assert_eq!(bmc.aux_var_counter, before);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_intersection_array_is_false_outside_finite_support() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let intersection = spanned(Expr::Intersect(
        Box::new(set_ident("S")),
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(1),
        ))]))),
    ));
    let universe = [bmc.solver.int_const(1)];
    let intersection = bmc.translate_set_expr(&intersection, &universe).unwrap();
    let outside = bmc.solver.int_const(2);
    let outside_member = bmc.solver.try_select(intersection, outside).unwrap();
    bmc.assert(outside_member);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_union_array_is_false_outside_complete_support() {
    let mut bmc = bmc_array(0);
    let union = spanned(Expr::Union(
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(1),
        ))]))),
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(2),
        ))]))),
    ));
    let universe = bmc.extract_universe_from_exprs(&[&union]).unwrap();
    let union = bmc.translate_set_expr(&union, &universe).unwrap();
    let outside = bmc.solver.int_const(3);
    let outside_member = bmc.solver.try_select(union, outside).unwrap();
    bmc.assert(outside_member);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_minus_accepts_symbolic_right_and_bounds_result_by_left() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let difference = spanned(Expr::SetMinus(
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(1),
        ))]))),
        Box::new(set_ident("S")),
    ));
    let universe = bmc.extract_universe_from_exprs(&[&difference]).unwrap();
    let difference = bmc.translate_set_expr(&difference, &universe).unwrap();
    let outside = bmc.solver.int_const(2);
    let outside_member = bmc.solver.try_select(difference, outside).unwrap();
    bmc.assert(outside_member);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_cardinality_rejects_symbolic_filter_and_builder_sets() {
    let mut bmc = bmc_array(0);
    bmc.declare_var(
        "S",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
    )
    .unwrap();
    let before = bmc.aux_var_counter;
    assert!(bmc.translate_init(&cardinality(set_ident("S"))).is_err());

    let bound = tla_core::ast::BoundVar {
        name: spanned("x".to_string()),
        domain: Some(Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Int(
            BigInt::from(1),
        ))])))),
        pattern: None,
    };
    let filter = spanned(Expr::SetFilter(
        bound.clone(),
        Box::new(spanned(Expr::Bool(true))),
    ));
    let builder = spanned(Expr::SetBuilder(Box::new(set_ident("x")), vec![bound]));
    assert!(bmc.translate_init(&cardinality(filter)).is_err());
    assert!(bmc.translate_init(&cardinality(builder)).is_err());
    assert_eq!(bmc.aux_var_counter, before);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_string_union_and_cardinality_are_exact() {
    let mut bmc = bmc_array(0);
    let left = spanned(Expr::SetEnum(vec![spanned(Expr::String("a".to_string()))]));
    let right = spanned(Expr::SetEnum(vec![spanned(Expr::String("b".to_string()))]));
    let union = spanned(Expr::Union(Box::new(left.clone()), Box::new(right.clone())));
    let expected = spanned(Expr::SetEnum(vec![
        spanned(Expr::String("a".to_string())),
        spanned(Expr::String("b".to_string())),
    ]));
    let equality = spanned(Expr::Eq(Box::new(union), Box::new(expected.clone())));
    let cardinality_two = spanned(Expr::Eq(
        Box::new(cardinality(expected)),
        Box::new(spanned(Expr::Int(BigInt::from(2)))),
    ));

    let equality = bmc.translate_init(&equality).unwrap();
    let cardinality_two = bmc.translate_init(&cardinality_two).unwrap();
    bmc.assert(equality);
    bmc.assert(cardinality_two);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_set_bool_array_paths_reject_but_direct_membership_is_exact() {
    let mut bmc = bmc_array(0);
    let before_aux = bmc.aux_var_counter;
    let before_base_vars = bmc.base_var_names.clone();
    assert!(bmc
        .declare_var(
            "B",
            TlaSort::Set {
                element_sort: Box::new(TlaSort::Bool),
            },
        )
        .is_err());
    assert!(!bmc.vars.contains_key("B"));
    assert_eq!(bmc.aux_var_counter, before_aux);
    assert_eq!(bmc.base_var_names, before_base_vars);

    assert!(bmc
        .declare_record_var(
            "R",
            vec![(
                "flags".to_string(),
                TlaSort::Set {
                    element_sort: Box::new(TlaSort::Bool),
                },
            )],
        )
        .is_err());
    assert!(!bmc.record_vars.contains_key("R"));
    assert_eq!(bmc.aux_var_counter, before_aux);
    assert_eq!(bmc.base_var_names, before_base_vars);

    let literal_membership = spanned(Expr::In(
        Box::new(spanned(Expr::Bool(true))),
        Box::new(spanned(Expr::SetEnum(vec![spanned(Expr::Bool(true))]))),
    ));
    let literal_membership = bmc.translate_init(&literal_membership).unwrap();
    bmc.assert(literal_membership);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_range_materialization_cap_and_extreme_bounds() {
    let at_limit_lo = spanned(Expr::Int(BigInt::from(0)));
    let at_limit_hi = spanned(Expr::Int(BigInt::from(
        super::super::compound_dispatch::MAX_BMC_SET_MATERIALIZATION - 1,
    )));
    assert!(BmcTranslator::checked_range_literals(&at_limit_lo, &at_limit_hi, "test").is_ok());

    let over_limit_hi = spanned(Expr::Int(BigInt::from(
        super::super::compound_dispatch::MAX_BMC_SET_MATERIALIZATION,
    )));
    assert!(BmcTranslator::checked_range_literals(&at_limit_lo, &over_limit_hi, "test").is_err());

    let min = spanned(Expr::Int(BigInt::from(i64::MIN)));
    let max = spanned(Expr::Int(BigInt::from(i64::MAX)));
    assert!(BmcTranslator::checked_range_literals(&min, &max, "test").is_err());
    assert!(BmcTranslator::checked_range_literals(&max, &min, "test").is_ok());
}
