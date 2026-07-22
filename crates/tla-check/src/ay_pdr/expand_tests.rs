// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Capture-avoidance regression tests for the CHC operator expander.
//!
//! These pin the 2026-07-05 AY-lane FALSE-SAFE fix: a module operator (or config CONSTANT) whose
//! name COLLIDES with an enclosing binder's bound variable must NEVER be inlined into the binder's
//! body (that would capture the bound occurrence and collapse e.g. a function to a constant, letting
//! the symbolic prover certify an otherwise-FALSE invariant).

use super::expand_operators_for_chc;
use crate::eval::EvalCtx;
use crate::test_support::parse_module;
use num_bigint::BigInt;
use tla_core::ast::Expr;

fn ctx_of(src: &str) -> EvalCtx {
    let module = parse_module(src);
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    ctx
}

fn body_of(ctx: &EvalCtx, name: &str) -> tla_core::Spanned<Expr> {
    ctx.get_op(name).expect("operator present").body.clone()
}

/// Does `e` contain an `Int(n)` literal anywhere? Used to detect a WRONGLY-inlined operator body
/// (`x == 99` folding a bound `x` into the literal `99`).
fn contains_int(e: &Expr, n: i64) -> bool {
    let want = BigInt::from(n);
    let mut found = false;
    struct V<'a>(&'a BigInt, &'a mut bool);
    impl tla_core::ExprVisitor for V<'_> {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            if let Expr::Int(m) = expr {
                if m == self.0 {
                    *self.1 = true;
                }
            }
            None
        }
    }
    tla_core::walk_expr(&mut V(&want, &mut found), e);
    found
}

/// Does `e` contain a bare `Ident(name)` anywhere?
fn contains_ident(e: &Expr, name: &str) -> bool {
    let mut found = false;
    struct V<'a>(&'a str, &'a mut bool);
    impl tla_core::ExprVisitor for V<'_> {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            if let Expr::Ident(n, _) = expr {
                if n == self.0 {
                    *self.1 = true;
                }
            }
            None
        }
    }
    tla_core::walk_expr(&mut V(name, &mut found), e);
    found
}

/// THE EXPLOIT (CapFD): operator `x == 99` collides with the `FuncDef` bound var `x`. The body
/// `IF x = 1 THEN "a" ELSE "b"` must keep its bound `x` — inlining `x ⇒ 99` would collapse the
/// function to the constant `"b"` (a false safe).
#[test]
fn funcdef_bound_var_shadows_colliding_operator() {
    let src = "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE pc\n\
               Init == pc = [x \\in 1..2 |-> IF x = 1 THEN \"a\" ELSE \"b\"]\n\
               Next == UNCHANGED pc\nSafety == pc[1] = \"b\"\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Init"), false);
    assert!(
        !contains_int(&expanded.node, 99),
        "operator `x==99` was captured into the FuncDef body (false-safe): {expanded:?}"
    );
    assert!(
        contains_ident(&expanded.node, "x"),
        "the bound `x` must survive as an Ident: {expanded:?}"
    );
}

/// `∀`/`∃` bound var collides with operator `x == 99` — the quantifier body's `x` must not inline.
#[test]
fn forall_and_exists_bound_var_shadow_colliding_operator() {
    for quant in ["\\A", "\\E"] {
        let src = format!(
            "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE y\n\
             Init == y = 0\nNext == UNCHANGED y\n\
             P == {quant} x \\in 1..2 : x = y\n====\n"
        );
        let ctx = ctx_of(&src);
        let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "P"), false);
        assert!(
            !contains_int(&expanded.node, 99),
            "{quant}: operator `x==99` captured into quantifier body: {expanded:?}"
        );
        assert!(
            contains_ident(&expanded.node, "x"),
            "{quant}: bound `x` must survive"
        );
    }
}

/// `CHOOSE` bound var collides with operator `x == 99` — the CHOOSE body's `x` must not inline.
#[test]
fn choose_bound_var_shadows_colliding_operator() {
    let src = "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE y\n\
               Init == y = 0\nNext == UNCHANGED y\n\
               P == y = CHOOSE x \\in 1..2 : x = 1\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "P"), false);
    assert!(
        !contains_int(&expanded.node, 99),
        "operator `x==99` captured into CHOOSE body: {expanded:?}"
    );
    assert!(
        contains_ident(&expanded.node, "x"),
        "bound `x` must survive"
    );
}

/// A `LET`-defined name collides with a GLOBAL operator `x == 99` — a reference to `x` inside the
/// LET body must resolve to the LET binding, so the global `99` must NOT be inlined there.
#[test]
fn let_defined_name_shadows_colliding_global_operator() {
    let src = "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE y\n\
               Init == y = 0\nNext == UNCHANGED y\n\
               P == LET x == 7 IN y = x\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "P"), false);
    assert!(
        !contains_int(&expanded.node, 99),
        "global operator `x==99` captured a LET-shadowed `x`: {expanded:?}"
    );
}

/// NESTED binders: the inner FuncDef `x` still shadows the operator even under an outer `∃ x`.
#[test]
fn nested_binders_keep_inner_shadow() {
    let src = "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE pc\n\
               Init == \\E x \\in 1..2 : pc = [x \\in 1..2 |-> IF x = 1 THEN \"a\" ELSE \"b\"]\n\
               Next == UNCHANGED pc\nSafety == pc[1] = \"b\"\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Init"), false);
    assert!(
        !contains_int(&expanded.node, 99),
        "operator `x==99` captured under nested binders: {expanded:?}"
    );
}

/// REGRESSION (no over-decline): a genuine operator reference OUTSIDE any binder STILL inlines. The
/// shadow check is a no-op when there is no collision — the legit AY-lane behavior is preserved.
#[test]
fn operator_reference_outside_binder_still_inlines() {
    let src = "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE y\n\
               Init == y = 0\nNext == UNCHANGED y\nInv == y = x\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Inv"), false);
    assert!(
        contains_int(&expanded.node, 99),
        "an unshadowed `x` reference must still inline to 99: {expanded:?}"
    );
}

/// REGRESSION (domains fold in the OUTER scope): an operator in a binder's DOMAIN — where the bound
/// var is not yet in scope — still inlines. Here the bound var is `i` (no collision) and the domain
/// `1..x` must expand to `1..99`.
#[test]
fn operator_in_binder_domain_still_inlines() {
    let src = "---- MODULE M ----\nEXTENDS Naturals\nx == 99\nVARIABLE f\n\
               Init == f = [i \\in 1..x |-> i]\nNext == UNCHANGED f\nSafety == TRUE\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Init"), false);
    assert!(
        contains_int(&expanded.node, 99),
        "operator `x` in the (outer-scope) domain must inline to 99: {expanded:?}"
    );
}

/// THE TELESCOPING-DOMAIN FALSE-SAFE (ExW): in `\E n \in 2..2, j \in 1..n : …`, the FIRST bound
/// var `n` is IN SCOPE inside the LATER domain `1..n`. An operator `n == 99` (or colliding config
/// CONSTANT) must NOT be inlined there — the bound `n` shadows it. Inlining it would WIDEN `1..n`
/// to `1..99` and let the prover certify an otherwise-FALSE `\E`.
#[test]
fn telescoping_later_domain_sees_earlier_bound_var_exists() {
    let src = "---- MODULE M ----\nEXTENDS Integers\nn == 99\nVARIABLE v\n\
               Init == v = 5\nNext == UNCHANGED v\n\
               Safety == \\E n \\in 2..2, j \\in 1..n : v = j\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Safety"), false);
    assert!(
        !contains_int(&expanded.node, 99),
        "operator `n==99` was captured into the LATER domain `1..n` (telescoping false-safe): {expanded:?}"
    );
    assert!(
        contains_ident(&expanded.node, "n"),
        "the earlier bound `n` must survive in `1..n`: {expanded:?}"
    );
}

/// The same telescoping hazard on a multi-arg `FuncDef` domain: `[a \in 1..2, b \in 1..a |-> b]`
/// with operator `a == 99` — `a` in the second domain `1..a` must not inline.
#[test]
fn telescoping_later_domain_sees_earlier_bound_var_funcdef() {
    let src = "---- MODULE M ----\nEXTENDS Integers\na == 99\nVARIABLE f\n\
               Init == f = [a \\in 1..2, b \\in 1..a |-> b]\nNext == UNCHANGED f\nSafety == TRUE\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Init"), false);
    assert!(
        !contains_int(&expanded.node, 99),
        "operator `a==99` was captured into the FuncDef's later domain `1..a`: {expanded:?}"
    );
    assert!(
        contains_ident(&expanded.node, "a"),
        "the earlier bound `a` must survive in `1..a`"
    );
}

/// REGRESSION (no over-decline): a LEGIT telescoping binder WITHOUT a name clash still inlines its
/// operator domain. `\A i \in 1..N, j \in 1..i : …` with `N == 3` (no collision with i/j): the FIRST
/// domain `1..N` MUST inline to `1..3`, while the telescoping `i` in `1..i` stays a bound Ident.
#[test]
fn legit_telescoping_binder_without_collision_still_inlines_operator_domain() {
    let src = "---- MODULE M ----\nEXTENDS Integers\nN == 3\nVARIABLE v\n\
               Init == v = 0\nNext == UNCHANGED v\n\
               Safety == \\A i \\in 1..N, j \\in 1..i : v <= j\n====\n";
    let ctx = ctx_of(src);
    let expanded = expand_operators_for_chc(&ctx, &body_of(&ctx, "Safety"), false);
    assert!(
        contains_int(&expanded.node, 3),
        "operator `N` in the first (outer-scope) domain must inline to 3: {expanded:?}"
    );
    assert!(
        contains_ident(&expanded.node, "i"),
        "the telescoping bound `i` must survive in `1..i`: {expanded:?}"
    );
}
