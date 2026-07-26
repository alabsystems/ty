// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BMC `FuncSet` (`[D -> R]`) finite-domain enumeration.
//!
//! `\E f \in [1..n -> R] : P(f)` is expanded by exhaustive concrete enumeration
//! of the function table; each function is substituted as a concrete value and
//! the body translated. These tests pin both the SAT (a satisfying function
//! exists) and UNSAT (no function satisfies the body) directions, plus the
//! permutation/distinctness idiom that the Einstein riddle relies on.

use super::*;
use tla_core::ast::BoundVar;
use tla_core::name_intern::NameId;

fn bound_var(name: &str, domain: Spanned<Expr>) -> BoundVar {
    BoundVar {
        name: spanned(name.to_string()),
        domain: Some(Box::new(domain)),
        pattern: None,
    }
}

fn ident(name: &str) -> Spanned<Expr> {
    spanned(Expr::Ident(name.to_string(), NameId::INVALID))
}

fn int(n: i64) -> Spanned<Expr> {
    spanned(Expr::Int(BigInt::from(n)))
}

/// `f[k]` for a bound function variable.
fn apply(func: &str, key: i64) -> Spanned<Expr> {
    spanned(Expr::FuncApply(Box::new(ident(func)), Box::new(int(key))))
}

fn func_set(lo: i64, hi: i64, range: Spanned<Expr>) -> Spanned<Expr> {
    spanned(Expr::FuncSet(
        Box::new(spanned(Expr::Range(Box::new(int(lo)), Box::new(int(hi))))),
        Box::new(range),
    ))
}

fn range_set(lo: i64, hi: i64) -> Spanned<Expr> {
    spanned(Expr::Range(Box::new(int(lo)), Box::new(int(hi))))
}

fn and(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    spanned(Expr::And(Box::new(a), Box::new(b)))
}

fn eq(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    spanned(Expr::Eq(Box::new(a), Box::new(b)))
}

/// `\E f \in [1..3 -> 1..3] : f[1] = 1 /\ f[2] = 2 /\ f[3] = 3` — SAT
/// (the identity function is in the set).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_exists_sat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let body = and(
        and(eq(apply("f", 1), int(1)), eq(apply("f", 2), int(2))),
        eq(apply("f", 3), int(3)),
    );
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 3, range_set(1, 3)))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

/// `\E f \in [1..2 -> {7}] : f[1] = 9` — UNSAT
/// (the only range value is 7, so no function maps to 9).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_exists_unsat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let body = eq(apply("f", 1), int(9));
    let range = spanned(Expr::SetEnum(vec![int(7)]));
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 2, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}

/// Direct membership introduces an internal existential function witness. Its
/// source-level spelling must be fresh: otherwise a user variable named like
/// the old `__fs_member_0` placeholder is captured and `x = witness` folds to
/// the tautology `witness = witness`.
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_direct_membership_witness_is_ast_hygienic() {
    let mut trans = BmcTranslator::new_with_arrays(0).unwrap();
    trans
        .declare_tuple_var("__fs_member_0", vec![TlaSort::Int])
        .unwrap();

    let user_cell = trans
        .get_tuple_element_at_step("__fs_member_0", 1, 0)
        .unwrap();
    let two = trans.solver.int_const(2);
    let user_is_two = trans.solver.try_eq(user_cell, two).unwrap();
    trans.assert(user_is_two);

    // <<2>> is not a member of [1..1 -> {1}]. The result must therefore be
    // UNSAT even though the adversarial user name matches the first historical
    // witness spelling.
    let singleton_one = spanned(Expr::SetEnum(vec![int(1)]));
    let membership = spanned(Expr::In(
        Box::new(ident("__fs_member_0")),
        Box::new(func_set(1, 1, singleton_one)),
    ));
    let term = trans.translate_init(&membership).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}

/// `\A f \in [1..2 -> 1..2] : f[1] >= 1` — SAT (every value is >= 1).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_forall_sat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let body = spanned(Expr::Geq(Box::new(apply("f", 1)), Box::new(int(1))));
    let expr = spanned(Expr::Forall(
        vec![bound_var("f", func_set(1, 2, range_set(1, 2)))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

/// Permutation/distinctness idiom (the Einstein core):
/// `\E f \in [1..3 -> {10,20,30}] : f[1] # f[2] /\ f[1] # f[3] /\ f[2] # f[3]`
/// — SAT (any permutation of the three distinct range values works).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_distinct_permutation_sat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let range = spanned(Expr::SetEnum(vec![int(10), int(20), int(30)]));
    let neq = |a: Spanned<Expr>, b: Spanned<Expr>| spanned(Expr::Neq(Box::new(a), Box::new(b)));
    let body = and(
        and(
            neq(apply("f", 1), apply("f", 2)),
            neq(apply("f", 1), apply("f", 3)),
        ),
        neq(apply("f", 2), apply("f", 3)),
    );
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 3, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

/// Distinctness over a range too small to satisfy it:
/// `\E f \in [1..3 -> {10,20}] : <all three distinct>` — UNSAT
/// (pigeonhole: 3 distinct values cannot come from a 2-element range).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_distinct_pigeonhole_unsat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let range = spanned(Expr::SetEnum(vec![int(10), int(20)]));
    let neq = |a: Spanned<Expr>, b: Spanned<Expr>| spanned(Expr::Neq(Box::new(a), Box::new(b)));
    let body = and(
        and(
            neq(apply("f", 1), apply("f", 2)),
            neq(apply("f", 1), apply("f", 3)),
        ),
        neq(apply("f", 2), apply("f", 3)),
    );
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 3, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}

/// String-range function set (the Einstein value domain):
/// `\E f \in [1..2 -> {"a","b"}] : f[1] = "a" /\ f[2] = "b"` — SAT.
/// Exercises the interned-string equality path through FuncSet enumeration.
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_string_range_sat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let str_lit = |s: &str| spanned(Expr::String(s.to_string()));
    let range = spanned(Expr::SetEnum(vec![str_lit("a"), str_lit("b")]));
    let body = and(
        eq(apply("f", 1), str_lit("a")),
        eq(apply("f", 2), str_lit("b")),
    );
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 2, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

// === Sequence-builder reduction (FunAsSeq lowering) ===

use tla_core::ast::{OpParam, OperatorDef};

fn op_call(name: &str, args: Vec<Spanned<Expr>>) -> Spanned<Expr> {
    spanned(Expr::Apply(Box::new(ident(name)), args))
}

/// FuncDef `[i \in 1..n |-> body]`.
fn func_def(var: &str, lo: i64, hi: i64, body: Spanned<Expr>) -> Spanned<Expr> {
    spanned(Expr::FuncDef(
        vec![bound_var(var, range_set(lo, hi))],
        Box::new(body),
    ))
}

/// The Einstein-shaped membership: a tuple state var `t` equals
/// `SubSeq([__i \in 1..n |-> f[__i]], 1, n)` for a bound function `f`. This is
/// exactly the FunAsSeq-lowered chain (MkSeq already inlined to the FuncDef).
/// With `t` pinned to a concrete tuple and `f` enumerated, the chain must
/// reduce to a literal tuple and the equality decide.
///
/// `\E f \in [1..2 -> {"a","b"}] : <<"a","b">> "=" SubSeq([i\in1..2|->f[i]],1,2)`
/// /\ f[1] = "a" — SAT (f = <<"a","b">> reduces the SubSeq to <<"a","b">>).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_subseq_funasseq_chain_sat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    trans
        .declare_var(
            "t",
            TlaSort::Tuple {
                element_sorts: vec![TlaSort::String, TlaSort::String],
            },
        )
        .unwrap();

    let str_lit = |s: &str| spanned(Expr::String(s.to_string()));
    let range = spanned(Expr::SetEnum(vec![str_lit("a"), str_lit("b")]));

    // SubSeq([__i \in 1..2 |-> f[__i]], 1, 2)
    let elem_body = spanned(Expr::FuncApply(
        Box::new(ident("f")),
        Box::new(ident("__i")),
    ));
    let mkseq = func_def("__i", 1, 2, elem_body);
    let subseq = op_call("SubSeq", vec![mkseq, int(1), int(2)]);

    let body = and(eq(ident("t"), subseq), eq(apply("f", 1), str_lit("a")));
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 2, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

/// Tuple indexing with a constant-foldable arithmetic index: `t[1 + 1]` must
/// resolve to element 2 (the `colors[i + 1]` neighbour-access idiom after a
/// quantifier substitutes `i := 1`). `t = <<10, 20>> /\ t[1 + 1] = 20` — SAT;
/// flipping the expected value makes it UNSAT.
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_tuple_index_const_fold() {
    for (rhs, want_sat) in [(20, true), (10, false)] {
        let mut trans = BmcTranslator::new(0).unwrap();
        trans
            .declare_var(
                "t",
                TlaSort::Tuple {
                    element_sorts: vec![TlaSort::Int, TlaSort::Int],
                },
            )
            .unwrap();
        // t = <<10, 20>>
        let t_eq = eq(ident("t"), spanned(Expr::Tuple(vec![int(10), int(20)])));
        // t[1 + 1] = rhs
        let idx = spanned(Expr::Add(Box::new(int(1)), Box::new(int(1))));
        let t_idx = spanned(Expr::FuncApply(Box::new(ident("t")), Box::new(idx)));
        let body = and(t_eq, eq(t_idx, int(rhs)));
        let term = trans.translate_init(&body).unwrap();
        trans.assert(term);
        let res = trans.check_sat();
        if want_sat {
            assert!(
                matches!(res, SolveResult::Sat),
                "expected SAT for rhs={rhs}"
            );
        } else {
            assert!(
                matches!(res, SolveResult::Unsat(_)),
                "expected UNSAT for rhs={rhs}"
            );
        }
    }
}

/// Same chain but with the body forcing `t[1] = "b"` while the SubSeq makes
/// `t[1] = f[1]` and `f[1] = "a"` — contradiction → UNSAT. Confirms the
/// reduction is faithful (not vacuously satisfiable).
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_subseq_funasseq_chain_unsat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    trans
        .declare_var(
            "t",
            TlaSort::Tuple {
                element_sorts: vec![TlaSort::String, TlaSort::String],
            },
        )
        .unwrap();

    let str_lit = |s: &str| spanned(Expr::String(s.to_string()));
    let range = spanned(Expr::SetEnum(vec![str_lit("a"), str_lit("b")]));

    let elem_body = spanned(Expr::FuncApply(
        Box::new(ident("f")),
        Box::new(ident("__i")),
    ));
    let mkseq = func_def("__i", 1, 2, elem_body);
    let subseq = op_call("SubSeq", vec![mkseq, int(1), int(2)]);

    // t = SubSeq(...) /\ f[1] = "a" /\ t[1] = "b"  (t[1]=f[1]="a" contradicts "b")
    let t_idx1 = spanned(Expr::FuncApply(Box::new(ident("t")), Box::new(int(1))));
    let body = and(
        and(eq(ident("t"), subseq), eq(apply("f", 1), str_lit("a"))),
        eq(t_idx1, str_lit("b")),
    );
    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 2, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}

/// `MkSeq(2, ctor)` with a LET-bound `ctor(__i) == f[__i]` reduces to
/// `<<f[1], f[2]>>`. `\E f \in [1..2 -> {7,8}] : <<f[1],f[2]>>[2] = 8` via the
/// MkSeq form — SAT. Exercises the `MkSeq` + LET-operator reduction arms.
#[cfg_attr(test, ntest::timeout(20000))]
#[test]
fn test_bmc_funcset_mkseq_let_ctor_sat() {
    let mut trans = BmcTranslator::new(0).unwrap();
    let range = spanned(Expr::SetEnum(vec![int(7), int(8)]));

    // LET ctor(__i) == f[__i] IN MkSeq(2, ctor)
    let ctor = OperatorDef {
        name: spanned("ctor".to_string()),
        params: vec![OpParam {
            name: spanned("__i".to_string()),
            arity: 0,
        }],
        body: spanned(Expr::FuncApply(
            Box::new(ident("f")),
            Box::new(ident("__i")),
        )),
        local: false,
        contains_prime: false,
        guards_depend_on_prime: false,
        has_primed_param: false,
        is_recursive: false,
        self_call_count: 0,
    };
    let mkseq = op_call("MkSeq", vec![int(2), ident("ctor")]);
    let let_seq = spanned(Expr::Let(vec![ctor], Box::new(mkseq)));
    // (LET ... IN MkSeq(2, ctor))[2] = 8
    let idx2 = spanned(Expr::FuncApply(Box::new(let_seq), Box::new(int(2))));
    let body = eq(idx2, int(8));

    let expr = spanned(Expr::Exists(
        vec![bound_var("f", func_set(1, 2, range))],
        Box::new(body),
    ));
    let term = trans.translate_init(&expr).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}
