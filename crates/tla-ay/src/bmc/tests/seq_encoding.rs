// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BMC sequence encoding via SMT arrays.
//!
//! Part of #3793: Validates sequence declaration, Len, Head, Tail,
//! Append, indexing, UNCHANGED, and sequence equality operations
//! in the BMC translator.

use super::*;
use ay_dpll::api::SolveResult;

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

/// Pin one representation cell without changing the sequence's logical length.
fn assert_seq_cell(bmc: &mut BmcTranslator, name: &str, step: usize, index: i64, value: i64) {
    let array = bmc.get_seq_array_at_step(name, step).unwrap();
    let index = bmc.solver.int_const(index);
    let value = bmc.solver.int_const(value);
    let selected = bmc.solver.try_select(array, index).unwrap();
    let equals = bmc.solver.try_eq(selected, value).unwrap();
    bmc.assert(equals);
}

// --- declare_seq_var ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_seq_var_int() {
    let mut bmc = bmc_array(2);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    assert!(bmc.seq_vars.contains_key("s"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_seq_var_bool() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Bool, 3).unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_seq_var_rejects_compound_element() {
    let mut bmc = bmc_array(0);
    let result = bmc.declare_seq_var(
        "s",
        TlaSort::Set {
            element_sort: Box::new(TlaSort::Int),
        },
        5,
    );
    assert!(result.is_err());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_var_delegates_sequence_sort() {
    let mut bmc = bmc_array(1);
    bmc.declare_var(
        "s",
        TlaSort::Sequence {
            element_sort: Box::new(TlaSort::Int),
            max_len: 4,
        },
    )
    .unwrap();
    // The sequence should be in seq_vars, not vars
    assert!(bmc.seq_vars.contains_key("s"));
    assert!(!bmc.vars.contains_key("s"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_declare_seq_var_idempotent() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    // Second declaration should be a no-op
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
}

// --- Len ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_len_constrained() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Assert Len(s) = 3
    let len_expr = spanned(Expr::Apply(Box::new(ident("Len")), vec![ident("s")]));
    let eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(len_expr), Box::new(int(3)))))
        .unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_len_exceeds_max_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 3).unwrap();

    // Assert Len(s) = 5 with max_len = 3 -> UNSAT
    let len_expr = spanned(Expr::Apply(Box::new(ident("Len")), vec![ident("s")]));
    let eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(len_expr), Box::new(int(5)))))
        .unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Head ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_head_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Assert Len(s) >= 1 (so Head is defined)
    let len_expr = spanned(Expr::Apply(Box::new(ident("Len")), vec![ident("s")]));
    let len_ge_1 = bmc
        .translate_init(&spanned(Expr::Geq(Box::new(len_expr), Box::new(int(1)))))
        .unwrap();
    bmc.assert(len_ge_1);

    // Assert Head(s) = 42
    let head_expr = spanned(Expr::Apply(Box::new(ident("Head")), vec![ident("s")]));
    let eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(head_expr), Box::new(int(42)))))
        .unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_head_contradicts_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Assert Head(s) = 10 AND Head(s) = 20 -> UNSAT
    let head = || spanned(Expr::Apply(Box::new(ident("Head")), vec![ident("s")]));
    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(head()), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(head()), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Sequence indexing: s[i] ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_indexing_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // s[1] = 10 AND s[2] = 20 -> SAT (different indices)
    let s1 = spanned(Expr::FuncApply(Box::new(ident("s")), Box::new(int(1))));
    let s2 = spanned(Expr::FuncApply(Box::new(ident("s")), Box::new(int(2))));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(s1), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(s2), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_indexing_same_index_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // s[1] = 10 AND s[1] = 20 -> UNSAT
    let s1_a = spanned(Expr::FuncApply(Box::new(ident("s")), Box::new(int(1))));
    let s1_b = spanned(Expr::FuncApply(Box::new(ident("s")), Box::new(int(1))));

    let eq1 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(s1_a), Box::new(int(10)))))
        .unwrap();
    bmc.assert(eq1);

    let eq2 = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(s1_b), Box::new(int(20)))))
        .unwrap();
    bmc.assert(eq2);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Sequence literal equality: s = <<e1, e2, ...>> ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_literal_eq_sat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // s = <<10, 20, 30>>
    let tuple = spanned(Expr::Tuple(vec![int(10), int(20), int(30)]));
    let eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(tuple))))
        .unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: Len(s) should be 3 in the model
    let len = bmc.get_seq_length_at_step("s", 0).unwrap();
    let three = bmc.solver.int_const(3);
    let len_check = bmc.solver.try_eq(len, three).unwrap();
    bmc.assert(len_check);
    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_literal_eq_head_matches() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // s = <<10, 20, 30>>
    let tuple = spanned(Expr::Tuple(vec![int(10), int(20), int(30)]));
    let eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(tuple))))
        .unwrap();
    bmc.assert(eq);

    // Head(s) = 10 should be SAT
    let head_expr = spanned(Expr::Apply(Box::new(ident("Head")), vec![ident("s")]));
    let head_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(head_expr), Box::new(int(10)))))
        .unwrap();
    bmc.assert(head_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_literal_eq_wrong_head_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // s = <<10, 20>>
    let tuple = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(tuple))))
        .unwrap();
    bmc.assert(eq);

    // Head(s) = 99 should be UNSAT (Head is 10)
    let head_expr = spanned(Expr::Apply(Box::new(ident("Head")), vec![ident("s")]));
    let head_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(head_expr), Box::new(int(99)))))
        .unwrap();
    bmc.assert(head_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- Append ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_append_step() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10>>
    let init_tuple = spanned(Expr::Tuple(vec![int(10)]));
    let init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(init_tuple),
        )))
        .unwrap();
    bmc.assert(init);

    // Index 3 is a ghost both before and after this one-element append. A TLA+
    // sequence transition must not require the two representations to agree.
    assert_seq_cell(&mut bmc, "s", 0, 3, 303);
    assert_seq_cell(&mut bmc, "s", 1, 3, 404);

    // Step 0->1: s' = Append(s, 20)
    bmc.current_step = 0;
    let primed_s = spanned(Expr::Prime(Box::new(ident("s"))));
    let append_expr = spanned(Expr::Apply(
        Box::new(ident("Append")),
        vec![ident("s"), int(20)],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_s),
            Box::new(append_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Verify: at step 1, Len(s) should be 2
    let len1 = bmc.get_seq_length_at_step("s", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let len_check = bmc.solver.try_eq(len1, two).unwrap();
    bmc.assert(len_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: at step 1, s[2] = 20
    let arr1 = bmc.get_seq_array_at_step("s", 1).unwrap();
    let two_idx = bmc.solver.int_const(2);
    let twenty = bmc.solver.int_const(20);
    let sel = bmc.solver.try_select(arr1, two_idx).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, twenty).unwrap();
    bmc.assert(sel_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- Tail ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_tail_step() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10, 20, 30>>
    let init_tuple = spanned(Expr::Tuple(vec![int(10), int(20), int(30)]));
    let init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(init_tuple),
        )))
        .unwrap();
    bmc.assert(init);

    // Tail has length two. Its index 3 must not be forced to the source's
    // capacity-only index 4 by the implementation's bounded shift witness.
    assert_seq_cell(&mut bmc, "s", 0, 4, 404);
    assert_seq_cell(&mut bmc, "s", 1, 3, 505);

    // Step 0->1: s' = Tail(s)
    bmc.current_step = 0;
    let primed_s = spanned(Expr::Prime(Box::new(ident("s"))));
    let tail_expr = spanned(Expr::Apply(Box::new(ident("Tail")), vec![ident("s")]));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(Box::new(primed_s), Box::new(tail_expr))))
        .unwrap();
    bmc.assert(next);

    // Verify: at step 1, Len(s) should be 2
    let len1 = bmc.get_seq_length_at_step("s", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let len_check = bmc.solver.try_eq(len1, two).unwrap();
    bmc.assert(len_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: at step 1, s[1] should be 20 (shifted from s[2] at step 0)
    let arr1 = bmc.get_seq_array_at_step("s", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let twenty = bmc.solver.int_const(20);
    let sel = bmc.solver.try_select(arr1, one).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, twenty).unwrap();
    bmc.assert(sel_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

/// Tail and SubSeq must allocate result arrays with the source sequence's Bool
/// element sort. An `(Array Int Int)` witness cannot store/select Bool cells.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_bool_sequence_tail_and_subseq_preserve_element_sort() {
    let bool_source = spanned(Expr::Tuple(vec![
        spanned(Expr::Bool(false)),
        spanned(Expr::Bool(true)),
        spanned(Expr::Bool(false)),
    ]));

    let mut tail = bmc_array(0);
    tail.declare_seq_var("s", TlaSort::Bool, 3).unwrap();
    let init = tail
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(bool_source.clone()),
        )))
        .unwrap();
    tail.assert(init);
    let (tail_array, tail_len) = tail
        .translate_seq_tail_bmc(&ident("s"))
        .expect("Tail(Bool sequence) must use a Bool-valued array");
    let two = tail.solver.int_const(2);
    let length_is_two = tail.solver.try_eq(tail_len, two).unwrap();
    tail.assert(length_is_two);
    let one = tail.solver.int_const(1);
    let first = tail.solver.try_select(tail_array, one).unwrap();
    tail.assert(first);
    let second = tail.solver.try_select(tail_array, two).unwrap();
    let second_is_false = tail.solver.try_not(second).unwrap();
    tail.assert(second_is_false);
    assert!(matches!(tail.check_sat(), SolveResult::Sat));

    let mut subseq = bmc_array(0);
    subseq.declare_seq_var("s", TlaSort::Bool, 3).unwrap();
    let init = subseq
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(bool_source),
        )))
        .unwrap();
    subseq.assert(init);
    let (subseq_array, subseq_len) = subseq
        .translate_seq_subseq_bmc(&ident("s"), &int(2), &int(3))
        .expect("SubSeq(Bool sequence) must use a Bool-valued array");
    let two = subseq.solver.int_const(2);
    let length_is_two = subseq.solver.try_eq(subseq_len, two).unwrap();
    subseq.assert(length_is_two);
    let one = subseq.solver.int_const(1);
    let first = subseq.solver.try_select(subseq_array, one).unwrap();
    subseq.assert(first);
    let second = subseq.solver.try_select(subseq_array, two).unwrap();
    let second_is_false = subseq.solver.try_not(second).unwrap();
    subseq.assert(second_is_false);
    assert!(matches!(subseq.check_sat(), SolveResult::Sat));
}

// --- UNCHANGED ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_unchanged() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10, 20>>
    let init_tuple = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(init_tuple),
        )))
        .unwrap();
    bmc.assert(init);

    // UNCHANGED preserves the abstract sequence, not unused array cells.
    assert_seq_cell(&mut bmc, "s", 0, 3, 303);
    assert_seq_cell(&mut bmc, "s", 1, 3, 404);

    // Step 0->1: UNCHANGED s
    bmc.current_step = 0;
    let unchanged = bmc
        .translate_bool(&spanned(Expr::Unchanged(Box::new(ident("s")))))
        .unwrap();
    bmc.assert(unchanged);

    // Verify: at step 1, s[1] = 10 and Len(s) = 2
    let arr1 = bmc.get_seq_array_at_step("s", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel = bmc.solver.try_select(arr1, one).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, ten).unwrap();
    bmc.assert(sel_eq);

    let len1 = bmc.get_seq_length_at_step("s", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let len_eq = bmc.solver.try_eq(len1, two).unwrap();
    bmc.assert(len_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_var_equality_ignores_ghost_cells() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 5).unwrap();

    let s_literal = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let s_init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(s_literal),
        )))
        .unwrap();
    bmc.assert(s_init);

    let t_len = bmc.get_seq_length_at_step("t", 0).unwrap();
    let two = bmc.solver.int_const(2);
    let t_len_eq = bmc.solver.try_eq(t_len, two).unwrap();
    bmc.assert(t_len_eq);
    assert_seq_cell(&mut bmc, "t", 0, 1, 10);

    // Equal logical values with deliberately different index-3 ghosts.
    assert_seq_cell(&mut bmc, "s", 0, 3, 303);
    assert_seq_cell(&mut bmc, "t", 0, 3, 404);
    let seq_eq = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(ident("t")),
        )))
        .unwrap();
    bmc.assert(seq_eq);

    let base_verdict = bmc.check_sat();
    assert!(
        matches!(base_verdict, SolveResult::Sat),
        "expected exact Sat for logical sequence equality with different ghosts, got \
         {base_verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );

    // Equality still constrains every live cell.
    bmc.push_scope().unwrap();
    assert_seq_cell(&mut bmc, "t", 0, 2, 99);
    let live_mismatch = bmc.check_sat();
    assert!(
        matches!(live_mismatch, SolveResult::Unsat(_)),
        "expected exact Unsat for a live sequence-equality mismatch, got \
         {live_mismatch:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );
    bmc.pop_scope().unwrap();
}

/// Int and String sequence elements share an SMT Int carrier but are distinct
/// TLA+ values. Mixed metadata must decline symmetrically, including under
/// negation; returning FALSE would also be wrong because the two empty
/// sequences are equal.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_var_mixed_element_sorts_fail_closed_symmetrically() {
    for (lhs, rhs) in [("ints", "strings"), ("strings", "ints")] {
        for negated in [false, true] {
            let mut bmc = bmc_array(0);
            bmc.declare_seq_var("ints", TlaSort::Int, 3).unwrap();
            bmc.declare_seq_var("strings", TlaSort::String, 3).unwrap();
            let equality = spanned(Expr::Eq(Box::new(ident(lhs)), Box::new(ident(rhs))));
            let expression = if negated {
                spanned(Expr::Not(Box::new(equality)))
            } else {
                equality
            };
            let error = bmc
                .translate_init(&expression)
                .expect_err("mixed sequence element sorts must fail closed");
            assert!(error
                .to_string()
                .contains("differing sequence element sorts"));
        }
    }
}

/// The empty tuple is the same empty sequence for every declared element sort,
/// while nonempty literals must match the carrier kind before String/Int terms
/// can alias. Valid String and Bool literals exercise both scalar carriers.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_literal_element_sort_gate_and_empty_controls() {
    for sort in [TlaSort::String, TlaSort::Bool] {
        for literal_on_left in [false, true] {
            let mut bmc = bmc_array(0);
            bmc.declare_seq_var("s", sort.clone(), 2).unwrap();
            let empty = spanned(Expr::Tuple(Vec::new()));
            let equality = if literal_on_left {
                spanned(Expr::Eq(Box::new(empty), Box::new(ident("s"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("s")), Box::new(empty)))
            };
            let term = bmc.translate_init(&equality).unwrap();
            bmc.assert(term);
            assert!(matches!(bmc.check_sat(), SolveResult::Sat));
        }
    }

    for (sort, literal) in [
        (TlaSort::Int, spanned(Expr::String("x".to_string()))),
        (TlaSort::String, int(7)),
    ] {
        for literal_on_left in [false, true] {
            let mut bmc = bmc_array(0);
            bmc.declare_seq_var("s", sort.clone(), 2).unwrap();
            let tuple = spanned(Expr::Tuple(vec![literal.clone()]));
            let equality = if literal_on_left {
                spanned(Expr::Eq(Box::new(tuple), Box::new(ident("s"))))
            } else {
                spanned(Expr::Eq(Box::new(ident("s")), Box::new(tuple)))
            };
            let error = bmc
                .translate_init(&equality)
                .expect_err("nonempty mixed-sort sequence literal must fail closed");
            assert!(error
                .to_string()
                .contains("differing sequence element sorts"));
        }
    }

    for (sort, literal) in [
        (TlaSort::String, spanned(Expr::String("valid".to_string()))),
        (TlaSort::Bool, spanned(Expr::Bool(true))),
    ] {
        let mut bmc = bmc_array(0);
        bmc.declare_seq_var("s", sort, 2).unwrap();
        let equality = spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(spanned(Expr::Tuple(vec![literal]))),
        ));
        let term = bmc.translate_init(&equality).unwrap();
        bmc.assert(term);
        assert!(matches!(bmc.check_sat(), SolveResult::Sat));
    }
}

/// Every sequence-producing operator must validate source/result carriers
/// before building array terms. This covers the paths whose shared SMT carriers
/// previously allowed an Int/String alias.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_operators_reject_mixed_element_sorts_before_translation() {
    fn prime(name: &str) -> Spanned<Expr> {
        spanned(Expr::Prime(Box::new(ident(name))))
    }

    let mut tail = bmc_array(1);
    tail.declare_seq_var("source", TlaSort::Int, 3).unwrap();
    tail.declare_seq_var("target", TlaSort::String, 3).unwrap();
    let tail_eq = spanned(Expr::Eq(
        Box::new(prime("target")),
        Box::new(spanned(Expr::Apply(
            Box::new(ident("Tail")),
            vec![ident("source")],
        ))),
    ));
    assert!(tail.translate_bool(&tail_eq).is_err());

    let mut subseq = bmc_array(1);
    subseq
        .declare_seq_var("source", TlaSort::String, 3)
        .unwrap();
    subseq.declare_seq_var("target", TlaSort::Int, 3).unwrap();
    let subseq_eq = spanned(Expr::Eq(
        Box::new(prime("target")),
        Box::new(spanned(Expr::Apply(
            Box::new(ident("SubSeq")),
            vec![ident("source"), int(1), int(1)],
        ))),
    ));
    assert!(subseq.translate_bool(&subseq_eq).is_err());

    let mut append = bmc_array(1);
    append.declare_seq_var("source", TlaSort::Int, 3).unwrap();
    append.declare_seq_var("target", TlaSort::Int, 3).unwrap();
    let append_eq = spanned(Expr::Eq(
        Box::new(prime("target")),
        Box::new(spanned(Expr::Apply(
            Box::new(ident("Append")),
            vec![ident("source"), spanned(Expr::String("x".to_string()))],
        ))),
    ));
    assert!(append.translate_bool(&append_eq).is_err());

    let mut concat = bmc_array(1);
    concat.declare_seq_var("left", TlaSort::Int, 3).unwrap();
    concat.declare_seq_var("right", TlaSort::String, 3).unwrap();
    concat.declare_seq_var("target", TlaSort::Int, 6).unwrap();
    let concat_eq = spanned(Expr::Eq(
        Box::new(prime("target")),
        Box::new(spanned(Expr::Apply(
            Box::new(ident("\\o")),
            vec![ident("left"), ident("right")],
        ))),
    ));
    assert!(concat.translate_bool(&concat_eq).is_err());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_string_append_remains_supported() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("source", TlaSort::String, 3).unwrap();
    bmc.declare_seq_var("target", TlaSort::String, 3).unwrap();

    let init = spanned(Expr::Eq(
        Box::new(ident("source")),
        Box::new(spanned(Expr::Tuple(vec![spanned(Expr::String(
            "first".to_string(),
        ))]))),
    ));
    let init_term = bmc.translate_init(&init).unwrap();
    bmc.assert(init_term);

    let next = spanned(Expr::Eq(
        Box::new(spanned(Expr::Prime(Box::new(ident("target"))))),
        Box::new(spanned(Expr::Apply(
            Box::new(ident("Append")),
            vec![ident("source"), spanned(Expr::String("second".to_string()))],
        ))),
    ));
    let next_term = bmc.translate_bool(&next).unwrap();
    bmc.assert(next_term);
    let verdict = bmc.check_sat();
    assert!(
        matches!(verdict, SolveResult::Sat),
        "valid String Append must remain exact Sat, got {verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason()
    );
}

// --- Sequence variable across steps ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_var_across_steps() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 3).unwrap();

    // At step 0: Len(s) = 1
    let len0 = bmc.get_seq_length_at_step("s", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let eq0 = bmc.solver.try_eq(len0, one).unwrap();
    bmc.assert(eq0);

    // At step 1: Len(s) = 2 (independent value)
    let len1 = bmc.get_seq_length_at_step("s", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let eq1 = bmc.solver.try_eq(len1, two).unwrap();
    bmc.assert(eq1);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

// --- BmcValue::Sequence ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_value_sequence_debug() {
    let val = BmcValue::Sequence(vec![BmcValue::Int(1), BmcValue::Int(2)]);
    let debug = format!("{val:?}");
    assert!(debug.contains("Sequence"));
    assert!(debug.contains("Int(1)"));
    assert!(debug.contains("Int(2)"));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_value_sequence_equality() {
    let a = BmcValue::Sequence(vec![BmcValue::Int(1), BmcValue::Int(2)]);
    let b = BmcValue::Sequence(vec![BmcValue::Int(1), BmcValue::Int(2)]);
    let c = BmcValue::Sequence(vec![BmcValue::Int(1), BmcValue::Int(3)]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// --- assert_concrete_state with Sequence ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_seq_state() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Assert concrete state: s = <<10, 20>>
    bmc.assert_concrete_state(
        &[(
            "s".to_string(),
            BmcValue::Sequence(vec![BmcValue::Int(10), BmcValue::Int(20)]),
        )],
        0,
    )
    .unwrap();

    // Verify: s[1] = 10
    let arr = bmc.get_seq_array_at_step("s", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel = bmc.solver.try_select(arr, one).unwrap();
    let eq = bmc.solver.try_eq(sel, ten).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: Len(s) = 2
    let len = bmc.get_seq_length_at_step("s", 0).unwrap();
    let two = bmc.solver.int_const(2);
    let len_eq = bmc.solver.try_eq(len, two).unwrap();
    bmc.assert(len_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_assert_concrete_seq_wrong_value_unsat() {
    let mut bmc = bmc_array(0);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();

    // Assert concrete state: s = <<10, 20>>
    bmc.assert_concrete_state(
        &[(
            "s".to_string(),
            BmcValue::Sequence(vec![BmcValue::Int(10), BmcValue::Int(20)]),
        )],
        0,
    )
    .unwrap();

    // Assert s[1] = 99 -> UNSAT (already constrained to 10)
    let arr = bmc.get_seq_array_at_step("s", 0).unwrap();
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let sel = bmc.solver.try_select(arr, one).unwrap();
    let eq = bmc.solver.try_eq(sel, ninety_nine).unwrap();
    bmc.assert(eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

// --- SubSeq ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_subseq_step() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10, 20, 30, 40>>
    let init_tuple = spanned(Expr::Tuple(vec![int(10), int(20), int(30), int(40)]));
    let init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(init_tuple),
        )))
        .unwrap();
    bmc.assert(init);

    // Step 0->1: t' = SubSeq(s, 2, 3) -> should be <<20, 30>>
    bmc.current_step = 0;
    let primed_t = spanned(Expr::Prime(Box::new(ident("t"))));
    let subseq_expr = spanned(Expr::Apply(
        Box::new(ident("SubSeq")),
        vec![ident("s"), int(2), int(3)],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_t),
            Box::new(subseq_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Verify: at step 1, Len(t) should be 2
    let len1 = bmc.get_seq_length_at_step("t", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let len_check = bmc.solver.try_eq(len1, two).unwrap();
    bmc.assert(len_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: at step 1, t[1] = 20 (s[2])
    let arr1 = bmc.get_seq_array_at_step("t", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let twenty = bmc.solver.int_const(20);
    let sel = bmc.solver.try_select(arr1, one).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, twenty).unwrap();
    bmc.assert(sel_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: at step 1, t[2] = 30 (s[3])
    let two_idx = bmc.solver.int_const(2);
    let thirty = bmc.solver.int_const(30);
    let sel2 = bmc.solver.try_select(arr1, two_idx).unwrap();
    let sel2_eq = bmc.solver.try_eq(sel2, thirty).unwrap();
    bmc.assert(sel2_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_subseq_single_element() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10, 20, 30>>
    let init_tuple = spanned(Expr::Tuple(vec![int(10), int(20), int(30)]));
    let init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(init_tuple),
        )))
        .unwrap();
    bmc.assert(init);

    // Step 0->1: t' = SubSeq(s, 2, 2) -> should be <<20>>
    bmc.current_step = 0;
    let primed_t = spanned(Expr::Prime(Box::new(ident("t"))));
    let subseq_expr = spanned(Expr::Apply(
        Box::new(ident("SubSeq")),
        vec![ident("s"), int(2), int(2)],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_t),
            Box::new(subseq_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Verify: at step 1, Len(t) should be 1
    let len1 = bmc.get_seq_length_at_step("t", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let len_check = bmc.solver.try_eq(len1, one).unwrap();
    bmc.assert(len_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: at step 1, t[1] = 20
    let arr1 = bmc.get_seq_array_at_step("t", 1).unwrap();
    let twenty = bmc.solver.int_const(20);
    let sel = bmc.solver.try_select(arr1, one).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, twenty).unwrap();
    bmc.assert(sel_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_subseq_wrong_element_unsat() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10, 20, 30>>
    let init_tuple = spanned(Expr::Tuple(vec![int(10), int(20), int(30)]));
    let init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(init_tuple),
        )))
        .unwrap();
    bmc.assert(init);

    // Step 0->1: t' = SubSeq(s, 2, 3)
    bmc.current_step = 0;
    let primed_t = spanned(Expr::Prime(Box::new(ident("t"))));
    let subseq_expr = spanned(Expr::Apply(
        Box::new(ident("SubSeq")),
        vec![ident("s"), int(2), int(3)],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_t),
            Box::new(subseq_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Assert t[1] = 99 at step 1 (should be 20) -> UNSAT
    let arr1 = bmc.get_seq_array_at_step("t", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ninety_nine = bmc.solver.int_const(99);
    let sel = bmc.solver.try_select(arr1, one).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, ninety_nine).unwrap();
    bmc.assert(sel_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_subseq_symbolic_bounds_ignore_ghost_cells() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 5).unwrap();
    bmc.declare_var("m", TlaSort::Int).unwrap();
    bmc.declare_var("n", TlaSort::Int).unwrap();

    let s_literal = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let s_init = bmc
        .translate_init(&spanned(Expr::Eq(
            Box::new(ident("s")),
            Box::new(s_literal),
        )))
        .unwrap();
    bmc.assert(s_init);
    for bound in ["m", "n"] {
        let bound_eq = bmc
            .translate_init(&spanned(Expr::Eq(Box::new(ident(bound)), Box::new(int(2)))))
            .unwrap();
        bmc.assert(bound_eq);
    }

    // With symbolic bounds the bounded witness also computes index 2 from
    // s[3]. Both cells are ghosts for the logical one-element result, so their
    // deliberately different values must not make the transition UNSAT.
    assert_seq_cell(&mut bmc, "s", 0, 3, 303);
    assert_seq_cell(&mut bmc, "t", 1, 2, 404);

    bmc.current_step = 0;
    let primed_t = spanned(Expr::Prime(Box::new(ident("t"))));
    let subseq_expr = spanned(Expr::Apply(
        Box::new(ident("SubSeq")),
        vec![ident("s"), ident("m"), ident("n")],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_t),
            Box::new(subseq_expr),
        )))
        .unwrap();
    bmc.assert(next);

    let base_verdict = bmc.check_sat();
    assert!(
        matches!(base_verdict, SolveResult::Sat),
        "expected exact Sat for symbolic SubSeq with different ghosts, got \
         {base_verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );

    // The single live result cell remains constrained to s[2] = 20.
    bmc.push_scope().unwrap();
    assert_seq_cell(&mut bmc, "t", 1, 1, 99);
    let live_mismatch = bmc.check_sat();
    assert!(
        matches!(live_mismatch, SolveResult::Unsat(_)),
        "expected exact Unsat for a live symbolic-SubSeq mismatch, got \
         {live_mismatch:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );
    bmc.pop_scope().unwrap();
}

// --- Concatenation: s \o t ---

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_concat_step() {
    let mut bmc = bmc_array(1);
    // Deliberate slack: both sources have three capacity-only ghost cells.
    // `s`'s ghosts at indices 3..=5 overlap `t`'s live destination slots in
    // `r`, but concat must ignore those ghosts and expose only four elements;
    // `r`'s remaining six cells are semantically inert.
    bmc.declare_seq_var("s", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 5).unwrap();
    bmc.declare_seq_var("r", TlaSort::Int, 10).unwrap();

    // Step 0: s = <<10, 20>>, t = <<30, 40>>
    let s_init = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let t_init = spanned(Expr::Tuple(vec![int(30), int(40)]));
    let s_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(s_init))))
        .unwrap();
    bmc.assert(s_eq);
    let t_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("t")), Box::new(t_init))))
        .unwrap();
    bmc.assert(t_eq);

    // Pin adversarial source ghosts pointwise. Whole-array equalities would
    // incorrectly make the representation-only cells part of sequence
    // semantics and trigger unnecessary extensional model reconstruction.
    let s_arr0 = bmc.get_seq_array_at_step("s", 0).unwrap();
    let t_arr0 = bmc.get_seq_array_at_step("t", 0).unwrap();
    for (arr, values) in [
        (s_arr0, [(3, 303), (4, 404), (5, 505)]),
        (t_arr0, [(3, 606), (4, 707), (5, 808)]),
    ] {
        for (idx, value) in values {
            let idx = bmc.solver.int_const(idx);
            let value = bmc.solver.int_const(value);
            let selected = bmc.solver.try_select(arr, idx).unwrap();
            let ghost_eq = bmc.solver.try_eq(selected, value).unwrap();
            bmc.assert(ghost_eq);
        }
    }

    // Step 0->1: r' = s \o t -> should be <<10, 20, 30, 40>>
    bmc.current_step = 0;
    let primed_r = spanned(Expr::Prime(Box::new(ident("r"))));
    let concat_expr = spanned(Expr::Apply(
        Box::new(ident("\\o")),
        vec![ident("s"), ident("t")],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_r),
            Box::new(concat_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Verify: at step 1, Len(r) should be 4
    let len1 = bmc.get_seq_length_at_step("r", 1).unwrap();
    let four = bmc.solver.int_const(4);
    let len_check = bmc.solver.try_eq(len1, four).unwrap();
    bmc.assert(len_check);

    let base_verdict = bmc.check_sat();
    assert!(
        matches!(base_verdict, SolveResult::Sat),
        "expected exact Sat for slack concat, got {base_verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );

    // Verify: at step 1, r[1] = 10, r[2] = 20, r[3] = 30, r[4] = 40
    let arr1 = bmc.get_seq_array_at_step("r", 1).unwrap();
    bmc.push_scope().unwrap();
    for (idx, expected) in [(1, 10), (2, 20), (3, 30), (4, 40)] {
        let i = bmc.solver.int_const(idx);
        let val = bmc.solver.int_const(expected);
        let sel = bmc.solver.try_select(arr1, i).unwrap();
        let sel_eq = bmc.solver.try_eq(sel, val).unwrap();
        bmc.assert(sel_eq);
    }

    let live_verdict = bmc.check_sat();
    assert!(
        matches!(live_verdict, SolveResult::Sat),
        "expected exact Sat for live concat values, got {live_verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );
    bmc.pop_scope().unwrap();

    // The join must not expose s[3]'s pinned ghost value. This scope omits the
    // positive r[3] assertion above, so UNSAT follows from concat itself rather
    // than from contradictory test assumptions. Unknown is never accepted.
    bmc.push_scope().unwrap();
    let three = bmc.solver.int_const(3);
    let stale_s_ghost = bmc.solver.int_const(303);
    let join_value = bmc.solver.try_select(arr1, three).unwrap();
    let stale_join = bmc.solver.try_eq(join_value, stale_s_ghost).unwrap();
    bmc.assert(stale_join);
    let stale_verdict = bmc.check_sat();
    assert!(
        matches!(stale_verdict, SolveResult::Unsat(_)),
        "expected exact Unsat for stale join ghost, got {stale_verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );
    bmc.pop_scope().unwrap();
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_concat_empty_left() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 0).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 2).unwrap();
    bmc.declare_seq_var("r", TlaSort::Int, 2).unwrap();

    // Step 0: s = <<>>, t = <<10, 20>>
    let s_init = spanned(Expr::Tuple(vec![]));
    let t_init = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let s_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(s_init))))
        .unwrap();
    bmc.assert(s_eq);
    let t_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("t")), Box::new(t_init))))
        .unwrap();
    bmc.assert(t_eq);

    // Step 0->1: r' = s \o t -> should be <<10, 20>>
    bmc.current_step = 0;
    let primed_r = spanned(Expr::Prime(Box::new(ident("r"))));
    let concat_expr = spanned(Expr::Apply(
        Box::new(ident("\\o")),
        vec![ident("s"), ident("t")],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_r),
            Box::new(concat_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Verify: at step 1, Len(r) should be 2
    let len1 = bmc.get_seq_length_at_step("r", 1).unwrap();
    let two = bmc.solver.int_const(2);
    let len_check = bmc.solver.try_eq(len1, two).unwrap();
    bmc.assert(len_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));

    // Verify: r[1] = 10 (from t)
    let arr1 = bmc.get_seq_array_at_step("r", 1).unwrap();
    let one = bmc.solver.int_const(1);
    let ten = bmc.solver.int_const(10);
    let sel = bmc.solver.try_select(arr1, one).unwrap();
    let sel_eq = bmc.solver.try_eq(sel, ten).unwrap();
    bmc.assert(sel_eq);

    assert!(matches!(bmc.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_concat_wrong_length_unsat() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 1).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 2).unwrap();
    bmc.declare_seq_var("r", TlaSort::Int, 5).unwrap();

    // Step 0: s = <<10>>, t = <<20, 30>>
    let s_init = spanned(Expr::Tuple(vec![int(10)]));
    let t_init = spanned(Expr::Tuple(vec![int(20), int(30)]));
    let s_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(s_init))))
        .unwrap();
    bmc.assert(s_eq);
    let t_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("t")), Box::new(t_init))))
        .unwrap();
    bmc.assert(t_eq);

    // Step 0->1: r' = s \o t
    bmc.current_step = 0;
    let primed_r = spanned(Expr::Prime(Box::new(ident("r"))));
    let concat_expr = spanned(Expr::Apply(
        Box::new(ident("\\o")),
        vec![ident("s"), ident("t")],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_r),
            Box::new(concat_expr),
        )))
        .unwrap();
    bmc.assert(next);

    // Assert Len(r) = 5 at step 1 (should be 3) -> UNSAT
    let len1 = bmc.get_seq_length_at_step("r", 1).unwrap();
    let five = bmc.solver.int_const(5);
    let len_check = bmc.solver.try_eq(len1, five).unwrap();
    bmc.assert(len_check);

    assert!(matches!(bmc.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bmc_seq_concat_result_capacity_overflow_unsat() {
    let mut bmc = bmc_array(1);
    bmc.declare_seq_var("s", TlaSort::Int, 2).unwrap();
    bmc.declare_seq_var("t", TlaSort::Int, 2).unwrap();
    // The logical concat has length four, but r can represent at most three.
    bmc.declare_seq_var("r", TlaSort::Int, 3).unwrap();

    let s_init = spanned(Expr::Tuple(vec![int(10), int(20)]));
    let t_init = spanned(Expr::Tuple(vec![int(30), int(40)]));
    let s_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("s")), Box::new(s_init))))
        .unwrap();
    bmc.assert(s_eq);
    let t_eq = bmc
        .translate_init(&spanned(Expr::Eq(Box::new(ident("t")), Box::new(t_init))))
        .unwrap();
    bmc.assert(t_eq);

    bmc.current_step = 0;
    let primed_r = spanned(Expr::Prime(Box::new(ident("r"))));
    let concat_expr = spanned(Expr::Apply(
        Box::new(ident("\\o")),
        vec![ident("s"), ident("t")],
    ));
    let next = bmc
        .translate_bool(&spanned(Expr::Eq(
            Box::new(primed_r),
            Box::new(concat_expr),
        )))
        .unwrap();
    bmc.assert(next);

    let verdict = bmc.check_sat();
    assert!(
        matches!(verdict, SolveResult::Unsat(_)),
        "expected exact Unsat for concat capacity overflow, got {verdict:?}; unknown reason: {:?}",
        bmc.last_unknown_reason(),
    );
}
