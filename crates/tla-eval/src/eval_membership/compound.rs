// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-compound-set lazy membership handlers.
//!
//! Extracted from the `eval_membership_lazy` dispatcher in
//! `eval_membership/mod.rs` as a thin-dispatcher decomposition. Each function
//! handles one compound-set expression form (Powerset/SUBSET, SetFilter,
//! Union, FuncSet `[S -> T]`, `Seq(S)`, RecordSet) and routes recursion back
//! through `super::eval_membership_lazy`. Behaviour is identical to the inlined
//! handlers; this is code motion only.

use super::super::{
    eval, eval_iter_set, push_bound_var_mut, EvalCtx, EvalError, EvalResult, Expr, FuncValue, Span,
    Spanned, Value,
};

pub(super) fn func_in_func_set(
    ctx: &EvalCtx,
    func: &FuncValue,
    domain_expr: &Spanned<Expr>,
    range_expr: &Spanned<Expr>,
) -> EvalResult<bool> {
    let dv = eval(ctx, domain_expr)?;

    // F3 (lever L2): when the domain evaluates to a materialized Set, compare
    // against the function's domain by reference — no to_sorted_set() clone
    // per invariant evaluation. Other shapes (Interval, lazy sets) keep the
    // existing materializing path, including the type error for non-sets.
    let domain_matches = match &dv {
        Value::Set(domain_set) => func.domain_eq_sorted_set(domain_set),
        _ => {
            // Handle both Set and Interval domains
            let domain = dv
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &dv, Some(domain_expr.span)))?;
            func.domain_eq_sorted_set(&domain)
        }
    };
    if !domain_matches {
        return Ok(false);
    }

    // Part of #3123: Always evaluate range to a Value. Lazy set types (SetCup,
    // RecordSet, Subset, FuncSet) are constructed symbolically in O(1) without
    // materializing elements, and their set_contains() methods support efficient
    // value-level membership testing. This evaluates the range expression ONCE
    // instead of re-traversing the AST per domain element via eval_membership_lazy,
    // matching TLC's approach where SetOfFcnsValue stores a pre-evaluated range
    // and calls range.member() per function value.
    let range_val = eval(ctx, range_expr)?;

    // F3 (lever L2): iterate the function's stored values directly instead of
    // binary-searching `func.mapping_get(d)` for every domain element. This is
    // sound ONLY because the domain-equality check above guarantees DOMAIN
    // func = dv exactly (1:1): every stored value corresponds to exactly one
    // domain element and vice versa, so checking `mapping_values()` checks
    // precisely the values the old per-domain-element loop checked (in the
    // same Value::cmp order — both domains are sorted).
    debug_assert_eq!(
        dv.set_len(),
        Some(num_bigint::BigInt::from(func.domain_len())),
        "func_in_func_set: domain equality must hold before iterating mapping_values"
    );
    for v in func.mapping_values() {
        // Value::set_contains handles Set, Interval, ModelValue (Nat/Int/Real),
        // SetCup, RecordSet, Subset, FuncSet, etc. Returns None only when an
        // evaluation context is needed (SetPred inside compound sets).
        let in_range = match range_val.set_contains(v) {
            Some(c) => c,
            None => crate::set_contains_with_ctx(ctx, v, &range_val, Some(range_expr.span))?,
        };

        if !in_range {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Handle SUBSET lazily: v \in SUBSET S <==> v is a set AND v \subseteq S
pub(super) fn powerset_membership(
    ctx: &EvalCtx,
    value: &Value,
    inner: &Spanned<Expr>,
) -> EvalResult<bool> {
    // Use eval_iter_set for SetPred-aware iteration (Part of #1828/#1830).
    // Without this, SetPred values would return false for SUBSET membership
    // because Value::iter_set() returns None for SetPred.
    match eval_iter_set(ctx, value, None) {
        Ok(iter) => {
            for elem in iter {
                if !super::eval_membership_lazy(ctx, &elem, inner)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Err(EvalError::TypeError {
            expected: "Set", ..
        }) => {
            // Value is not a set-like type — cannot be in SUBSET S
            Ok(false)
        }
        Err(e) => {
            // Propagate real evaluation errors (e.g. SetPred predicate failures)
            Err(e)
        }
    }
}

/// Handle SetFilter lazily: v \in {x \in S : P(x)} <==> v \in S /\ P(v)
pub(super) fn set_filter_membership(
    ctx: &EvalCtx,
    value: &Value,
    bound: &tla_core::ast::BoundVar,
    pred: &Spanned<Expr>,
    span: Span,
) -> EvalResult<bool> {
    let domain_expr = bound.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "SetFilter requires bounded variable".into(),
        span: Some(span),
    })?;

    if !super::eval_membership_lazy(ctx, value, domain_expr)? {
        return Ok(false);
    }

    let mut local_ctx = ctx.clone();
    let mark = local_ctx.mark_stack();
    push_bound_var_mut(&mut local_ctx, bound, value, Some(span))?;

    // TLC propagates eval errors here (SetPredValue.member → Assert.fail).
    // Do NOT silently convert NotInDomain/IndexOutOfBounds to false.
    let pv = eval(&local_ctx, pred)?;
    let include = pv
        .as_bool()
        .ok_or_else(|| EvalError::type_error("BOOLEAN", &pv, Some(pred.span)))?;

    local_ctx.pop_to_mark(&mark);
    Ok(include)
}

/// Handle Union lazily: v \in (A \cup B) <==> (v \in A) \/ (v \in B)
///
/// This is critical for efficient type checking in specs like MultiPaxos where
/// Messages = PrepareMsgs \cup PrepareReplyMsgs \cup AcceptMsgs \cup ...
/// Without lazy union, we'd compute the full Cartesian product of all message types.
pub(super) fn union_membership(
    ctx: &EvalCtx,
    value: &Value,
    left: &Spanned<Expr>,
    right: &Spanned<Expr>,
) -> EvalResult<bool> {
    // Part of #758: Union trees produced by stdlib expansions can be *very* deep.
    // Avoid recursion proportional to union depth by flattening via an explicit stack.
    let mut pending: Vec<&Spanned<Expr>> = Vec::new();
    pending.push(right);
    pending.push(left);

    while let Some(expr) = pending.pop() {
        // Resolve identifier aliases iteratively (same logic as above) so
        // `(A ∪ B)` can safely contain `Ident` chains as children.
        let mut expr = expr;
        loop {
            let Expr::Ident(name, _) = &expr.node else {
                break;
            };
            let resolved_name = ctx.resolve_op_name(name);
            let Some(def) = ctx.get_op(resolved_name) else {
                break;
            };
            if !def.params.is_empty() {
                break;
            }
            if !super::is_lazy_membership_expr(&def.body.node) {
                break;
            }
            // Part of #3123: Same eager-eval shortcut as outer loop.
            // For union branches that resolve to cached concrete sets,
            // use set_contains() directly.
            // Fix #3802: Skip eager eval for SetFilter bodies (see outer loop comment).
            if !matches!(def.body.node, Expr::SetFilter(_, _)) {
                if let Ok(set_val) = eval(ctx, expr) {
                    match set_val.set_contains(value) {
                        Some(true) => return Ok(true),
                        Some(false) => break, // Not in this branch, try next
                        None => {
                            // Indeterminate -- fall back to context-aware check.
                            if crate::set_contains_with_ctx(ctx, value, &set_val, Some(expr.span))?
                            {
                                return Ok(true);
                            }
                            break; // Not in this branch, try next
                        }
                    }
                }
            }
            expr = &def.body;
        }

        match &expr.node {
            Expr::Union(l, r) => {
                // Check left first (TLA+ semantics: short-circuit is OK).
                pending.push(r);
                pending.push(l);
            }
            _ => {
                if super::eval_membership_lazy(ctx, value, expr)? {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Handle [S -> T] lazily: v \in [S -> T] <==> v is a function with domain S and range in T
pub(super) fn funcset_membership(
    ctx: &EvalCtx,
    value: &Value,
    domain_expr: &Spanned<Expr>,
    range_expr: &Spanned<Expr>,
) -> EvalResult<bool> {
    // Performance optimization: evaluate the range expression ONCE and use
    // Value::set_contains() for each element, matching TLC's approach in
    // SetOfFcnsValue. This avoids re-traversing the AST per element via
    // eval_membership_lazy. Falls back to lazy evaluation when set_contains()
    // returns None (indeterminate, e.g. SetPred).
    match value {
        Value::Func(f) => func_in_func_set(ctx, f, domain_expr, range_expr),
        // Compact bag: identical semantics via the cached materialized Func.
        Value::Bag(b) => func_in_func_set(ctx, b.as_func_value(), domain_expr, range_expr),
        // IntFunc is an array-backed function with integer interval domain
        Value::IntFunc(f) => {
            // Check domain: function set domain must equal min..max
            let domain_val = eval(ctx, domain_expr)?;
            let actual_domain = domain_val
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &domain_val, Some(domain_expr.span)))?;
            if !actual_domain.equals_integer_interval(
                tla_value::IntIntervalFunc::min(f),
                tla_value::IntIntervalFunc::max(f),
            ) {
                return Ok(false);
            }
            // Evaluate range once, then check each value via set_contains
            let range_val = eval(ctx, range_expr)?;
            for val in f.values() {
                let in_range = match range_val.set_contains(val) {
                    Some(c) => c,
                    None => {
                        crate::set_contains_with_ctx(ctx, val, &range_val, Some(range_expr.span))?
                    }
                };
                if !in_range {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        // Tuples/Seqs are functions with domain 1..n
        Value::Tuple(elems) => {
            // Check domain: expected is 1..n
            let domain_val = eval(ctx, domain_expr)?;
            let actual_domain = domain_val
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &domain_val, Some(domain_expr.span)))?;
            if !actual_domain.equals_sequence_domain(elems.len()) {
                return Ok(false);
            }
            // Evaluate range once, then check each element via set_contains
            let range_val = eval(ctx, range_expr)?;
            for elem in elems.iter() {
                let in_range = match range_val.set_contains(elem) {
                    Some(c) => c,
                    None => {
                        crate::set_contains_with_ctx(ctx, elem, &range_val, Some(range_expr.span))?
                    }
                };
                if !in_range {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Value::Seq(seq) => {
            // Check domain: expected is 1..n
            let domain_val = eval(ctx, domain_expr)?;
            let actual_domain = domain_val
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &domain_val, Some(domain_expr.span)))?;
            if !actual_domain.equals_sequence_domain(seq.len()) {
                return Ok(false);
            }
            // Evaluate range once, then check each element via set_contains
            let range_val = eval(ctx, range_expr)?;
            for elem in seq.iter() {
                let in_range = match range_val.set_contains(elem) {
                    Some(c) => c,
                    None => {
                        crate::set_contains_with_ctx(ctx, elem, &range_val, Some(range_expr.span))?
                    }
                };
                if !in_range {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Handle Seq(S) lazily: v \in Seq(S) <==> v is a sequence AND all elements are in S
/// Seq(S) is represented as Apply(Ident("Seq"), [S])
pub(super) fn seq_membership(
    ctx: &EvalCtx,
    value: &Value,
    elem_set_expr: &Spanned<Expr>,
) -> EvalResult<bool> {
    // Check if value is a sequence/tuple and all elements are in S
    if let Some(elems) = value.as_seq_or_tuple_elements() {
        for elem in elems.iter() {
            if !super::eval_membership_lazy(ctx, elem, elem_set_expr)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    match value {
        // TLA+ treats functions 1..n -> T as sequences
        Value::Func(f) => {
            // Check if domain is 1..n for some n
            if !f.domain_is_sequence() {
                return Ok(false);
            }
            // Check all values are in S
            for v in f.mapping_values() {
                if !super::eval_membership_lazy(ctx, v, elem_set_expr)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        // Compact bag: identical semantics via the cached materialized Func.
        Value::Bag(b) => {
            let f = b.as_func_value();
            if !f.domain_is_sequence() {
                return Ok(false);
            }
            for v in f.mapping_values() {
                if !super::eval_membership_lazy(ctx, v, elem_set_expr)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        // IntFunc with domain 1..n is also a sequence
        Value::IntFunc(f) => {
            // Check if domain is 1..n for some n
            if tla_value::IntIntervalFunc::min(f) != 1 {
                return Ok(false);
            }
            // Check all values are in S
            for v in f.values() {
                if !super::eval_membership_lazy(ctx, v, elem_set_expr)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Handle RecordSet lazily: v \in [f1: S1, f2: S2, ...] <==> v is a record with
/// exactly those fields AND v.f1 \in S1 AND v.f2 \in S2 AND ...
pub(super) fn recordset_membership(
    ctx: &EvalCtx,
    value: &Value,
    fields: &[(Spanned<String>, Spanned<Expr>)],
) -> EvalResult<bool> {
    match value {
        Value::Record(rec) => {
            // Check that record has exactly the same fields
            if rec.len() != fields.len() {
                return Ok(false);
            }
            // Check each field
            for (field_name, field_set_expr) in fields {
                match rec.get(field_name.node.as_str()) {
                    Some(field_val) => {
                        if !super::eval_membership_lazy(ctx, field_val, field_set_expr)? {
                            return Ok(false);
                        }
                    }
                    None => return Ok(false), // Record doesn't have this field
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
