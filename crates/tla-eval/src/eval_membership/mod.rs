// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Membership testing and lazy evaluation helpers for TLA+ set operations.
// Extracted from core.rs as part of #1219 decomposition.
//
// This module is a thin dispatcher: `eval_membership_lazy` resolves zero-arg
// operator aliases, then delegates each compound-set form to a dedicated
// submodule handler.
//
// This module contains:
// - is_lazy_membership_expr / needs_lazy_membership: Detect expressions needing lazy handling
// - eval_membership_lazy: Lazy membership dispatcher for SUBSET, [S->T], Seq(S),
//   RecordSet, Union, SetFilter, SetBuilder
//
// Submodules:
// - compound: per-compound-set handlers + func_in_func_set ([S -> T] membership)
// - set_builder: SetBuilder membership (`{expr : x \in S, ...}`) + inverse helpers
// - set_pred: check_set_pred_membership for SetPred values

mod compound;
mod set_builder;
mod set_pred;

pub use set_pred::{check_set_pred_membership, restore_setpred_ctx};

use super::{
    eval, should_prefer_builtin_override, EvalCtx, EvalError, EvalResult, Expr, Spanned, Value,
};
use num_bigint::BigInt;
use num_traits::Zero;

/// Check if an expression requires lazy membership checking (FuncSet, Powerset, Seq, RecordSet, SetBuilder)
pub(super) fn is_lazy_membership_expr(expr: &Expr) -> bool {
    match expr {
        Expr::FuncSet(_, _) | Expr::Powerset(_) | Expr::RecordSet(_) | Expr::SetFilter(_, _) => {
            true
        }
        // SetBuilder: {expr : x \in S} -- check inverse membership lazily by finding
        // x \in S such that expr(x) = target, avoiding full materialization. Part of #3978.
        Expr::SetBuilder(_, _) => true,
        // Union: can check membership lazily as (x \in A) \/ (x \in B)
        Expr::Union(_, _) => true,
        Expr::Apply(op, args) => {
            // Check for Seq(S) pattern
            if let Expr::Ident(name, _) = &op.node {
                name == "Seq" && args.len() == 1
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if an expression (potentially through Ident resolution) requires lazy membership checking
pub(super) fn needs_lazy_membership(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    if is_lazy_membership_expr(&expr.node) {
        return true;
    }
    // Also check if expr is an Ident that resolves to a lazy membership expression
    if let Expr::Ident(name, _) = &expr.node {
        let resolved_name = ctx.resolve_op_name(name);
        if let Some(def) = ctx.get_op(resolved_name) {
            if def.params.is_empty() && is_lazy_membership_expr(&def.body.node) {
                return true;
            }
        }
    }
    false
}

/// Check if a value is a member of a set expression, with lazy handling for SUBSET, [S -> T], Seq(S), and RecordSet
/// This avoids eager enumeration of large/infinite sets
///
/// Part of #3073: Takes `&Value` instead of `Value` to avoid cloning in the union
/// iteration hot path. For MultiPaxos TypeOK (`\A m \in msgs: m \in Messages`),
/// this eliminates ~5 record clones per message per state across 5 union branches.
pub(super) fn eval_membership_lazy(
    ctx: &EvalCtx,
    value: &Value,
    set_expr: &Spanned<Expr>,
) -> EvalResult<bool> {
    // Follow zero-arg operator aliases to their underlying lazy-membership expressions.
    // Keep this iterative so long chains of aliases cannot overflow the stack.
    let mut set_expr = set_expr;
    loop {
        let Expr::Ident(name, _) = &set_expr.node else {
            break;
        };
        let resolved_name = ctx.resolve_op_name(name);
        let Some(def) = ctx.get_op(resolved_name) else {
            break;
        };
        if !def.params.is_empty() {
            break;
        }
        if !is_lazy_membership_expr(&def.body.node) {
            break;
        }
        // Part of #3123: Before following through to the lazy body, try evaluating
        // the Ident normally. The zero-arg operator cache will return the cached
        // result if available (e.g., `Message` evaluated once then reused). If the
        // result is a concrete finite set (Set, Interval, SetCup, etc.), use
        // set_contains() directly instead of walking the AST tree for each
        // membership check. This avoids re-constructing SetMap elements on every
        // membership test in specs like MCLamportMutex where
        // `Message = {AckMessage, RelMessage} \union {ReqMessage(c) : c \in Clock}`
        // is checked ~19.5M times.
        //
        // Fix #3802: Skip eager eval for SetFilter bodies. SetFilter with
        // non-reducible domains (e.g., SUBSET SUBSET S) produces SetPred values
        // that the zero-arg cache materializes via materialize_setpred_to_vec,
        // causing exponential enumeration. Fall through to the lazy SetFilter
        // handler below which checks membership via source-set + predicate.
        if !matches!(def.body.node, Expr::SetFilter(_, _)) {
            if let Ok(set_val) = eval(ctx, set_expr) {
                match set_val.set_contains(value) {
                    Some(c) => return Ok(c),
                    None => {
                        // Indeterminate (e.g. SetPred inside compound set) --
                        // fall back to context-aware check.
                        return super::set_contains_with_ctx(
                            ctx,
                            value,
                            &set_val,
                            Some(set_expr.span),
                        );
                    }
                }
            }
        }
        set_expr = &def.body;
    }

    // Handle SUBSET lazily: v \in SUBSET S <==> v is a set AND v \subseteq S
    if let Expr::Powerset(inner) = &set_expr.node {
        return compound::powerset_membership(ctx, value, inner);
    }

    // Handle SetFilter lazily: v \in {x \in S : P(x)} <==> v \in S /\ P(v)
    if let Expr::SetFilter(bound, pred) = &set_expr.node {
        return compound::set_filter_membership(ctx, value, bound, pred, set_expr.span);
    }

    // Handle SetBuilder lazily: v \in {expr(x) : x \in S} <==> \E x \in S : expr(x) = v
    // Part of #3978: Instead of materializing all mapped values, iterate domain elements
    // and short-circuit on the first match. This is O(|S|) in the worst case but
    // O(1) best case, vs always O(|S|) for eager materialization + linear scan.
    if let Expr::SetBuilder(body, bounds) = &set_expr.node {
        return set_builder::check_set_builder_membership(
            ctx,
            value,
            body,
            bounds,
            Some(set_expr.span),
        );
    }

    // Handle Union lazily: v \in (A \cup B) <==> (v \in A) \/ (v \in B)
    // This is critical for efficient type checking in specs like MultiPaxos where
    // Messages = PrepareMsgs \cup PrepareReplyMsgs \cup AcceptMsgs \cup ...
    // Without lazy union, we'd compute the full Cartesian product of all message types.
    if let Expr::Union(left, right) = &set_expr.node {
        return compound::union_membership(ctx, value, left, right);
    }

    // Handle [S -> T] lazily: v \in [S -> T] <==> v is a function with domain S and range in T
    if let Expr::FuncSet(domain_expr, range_expr) = &set_expr.node {
        return compound::funcset_membership(ctx, value, domain_expr, range_expr);
    }

    // Handle Seq(S) lazily: v \in Seq(S) <==> v is a sequence AND all elements are in S
    // Seq(S) is represented as Apply(Ident("Seq"), [S]).
    //
    // N6: apply the built-in Seq semantics ONLY when "Seq" actually resolves to
    // the built-in. If it is shadowed by a user-defined `Seq(_)` operator, a cfg
    // operator replacement (`CONSTANT Seq <- BoundedSeq`), or a bound closure,
    // fall through to the eager path below so that membership agrees with what
    // `eval_apply` would compute for the same application.
    if let Expr::Apply(op, args) = &set_expr.node {
        if let Expr::Ident(name, _) = &op.node {
            if name == "Seq" && args.len() == 1 && seq_apply_is_builtin(ctx) {
                return compound::seq_membership(ctx, value, &args[0]);
            }
        }
    }

    // Handle RecordSet lazily: v \in [f1: S1, f2: S2, ...] <==> v is a record with exactly those fields AND v.f1 \in S1 AND v.f2 \in S2 AND ...
    if let Expr::RecordSet(fields) = &set_expr.node {
        return compound::recordset_membership(ctx, value, fields);
    }

    // For other expressions, evaluate eagerly and check membership
    let set_val = eval(ctx, set_expr)?;

    // Handle ModelValue for infinite sets (Nat, Int, Real)
    if let Value::ModelValue(name) = &set_val {
        return match name.as_ref() {
            "Nat" => match value {
                Value::SmallInt(n) => Ok(*n >= 0),
                Value::Int(n) => Ok(**n >= BigInt::zero()),
                _ => Ok(false),
            },
            "Int" => Ok(matches!(value, Value::SmallInt(_) | Value::Int(_))),
            "Real" => Ok(matches!(value, Value::SmallInt(_) | Value::Int(_))), // Int ⊆ Real
            _ => Err(EvalError::type_error("Set", &set_val, Some(set_expr.span))),
        };
    }

    // Handle both Set and Interval using set_contains.
    // If set_contains returns None (e.g. SetPred inside SetCup/SetDiff/SetCap),
    // fall back to context-aware recursive decomposition.
    let contains = match set_val.set_contains(value) {
        Some(c) => c,
        None => super::set_contains_with_ctx(ctx, value, &set_val, Some(set_expr.span))?,
    };
    Ok(contains)
}

/// Whether `Seq(_)` in a membership test dispatches to the built-in `Seq`
/// operator (the set of all finite sequences over the argument), mirroring how
/// `eval_apply` resolves the same application. Returns `false` when `Seq` is
/// shadowed — by a closure bound in the environment, a cfg operator replacement
/// (`CONSTANT Seq <- BoundedSeq`), or a user-defined `Seq(_)` operator — so the
/// caller falls through to the eager path and membership agrees with
/// application (N6).
fn seq_apply_is_builtin(ctx: &EvalCtx) -> bool {
    // A closure named `Seq` bound in the environment shadows the built-in
    // (matches `eval_apply`'s first check).
    if matches!(ctx.lookup("Seq"), Some(Value::Closure(_))) {
        return false;
    }
    // A cfg operator replacement (`Seq <- Other`) redirects to a different op.
    if ctx.resolve_op_name("Seq") != "Seq" {
        return false;
    }
    // A user-defined `Seq(_)` operator shadows the built-in unless a builtin
    // override is explicitly preferred. `Seq` is never an override target, so a
    // user definition always wins and we fall through to the eager path.
    if let Some(def) = ctx.get_op("Seq") {
        return should_prefer_builtin_override("Seq", def, 1, ctx);
    }
    true
}
