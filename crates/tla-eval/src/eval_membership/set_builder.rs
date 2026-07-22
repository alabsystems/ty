// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! SetBuilder lazy membership: `v \in {expr : x \in S, y \in T, ...}`.
//!
//! Extracted from `eval_membership/mod.rs` as a thin-dispatcher decomposition.
//! `check_set_builder_membership` is the entry point called by the
//! `eval_membership_lazy` dispatcher; the inverse-pattern and recursive helpers
//! live alongside it. All recursion routes back through
//! `super::eval_membership_lazy`.

use super::super::{
    eval, eval_iter_set, push_bound_var_mut, EvalCtx, EvalError, EvalResult, Expr, Span, Spanned,
    Value,
};

/// Check membership in a SetBuilder expression: v \in {expr : x \in S, y \in T, ...}
///
/// This is equivalent to `\E x \in S, y \in T, ... : expr(x, y, ...) = v`.
///
/// Part of #3979: Uses inverse membership checking for invertible patterns
/// (tuple construction, record construction, identity mapping) to avoid
/// iterating domain elements entirely. Falls back to iterate-and-short-circuit
/// for general expressions.
///
/// Inverse patterns (O(k) where k = number of bound variables):
/// - Tuple: `{<<x, y>> : x \in S, y \in T}` — decompose target tuple, check components
/// - Record: `{[a |-> x, b |-> y] : x \in S, y \in T}` — decompose target record, check fields
/// - Identity: `{x : x \in S}` — check `target \in S` directly
///
/// Fallback (O(|S1| * |S2| * ...) worst case, short-circuits on first match):
/// - General expressions: iterate domain elements, evaluate body, compare to target
pub(super) fn check_set_builder_membership(
    ctx: &EvalCtx,
    target: &Value,
    body: &Spanned<Expr>,
    bounds: &[tla_core::ast::BoundVar],
    span: Option<Span>,
) -> EvalResult<bool> {
    // Try inverse membership checking for invertible patterns.
    // This avoids iterating the entire Cartesian product of domains.
    if let Some(result) = try_inverse_membership(ctx, target, body, bounds, span)? {
        return Ok(result);
    }

    // Fallback: iterate domain elements and short-circuit on first match.
    let mut local_ctx = ctx.clone();
    check_set_builder_membership_rec(&mut local_ctx, target, body, bounds, span)
}

/// Try inverse membership checking for known invertible SetBuilder patterns.
///
/// Returns `Some(bool)` if the pattern was recognized and checked, `None` if
/// the body expression is not invertible and the caller should fall back to
/// iterate-and-check.
///
/// Part of #3979.
fn try_inverse_membership(
    ctx: &EvalCtx,
    target: &Value,
    body: &Spanned<Expr>,
    bounds: &[tla_core::ast::BoundVar],
    span: Option<Span>,
) -> EvalResult<Option<bool>> {
    // Pattern 1: Identity mapping — {x : x \in S}
    // Single bound variable, body is just the bound variable.
    if bounds.len() == 1 {
        if let Expr::Ident(name, _) = &body.node {
            if name == &bounds[0].name.node {
                let domain = bounds[0]
                    .domain
                    .as_ref()
                    .ok_or_else(|| EvalError::Internal {
                        message: "SetBuilder requires bounded variables".into(),
                        span,
                    })?;
                return Ok(Some(super::eval_membership_lazy(ctx, target, domain)?));
            }
        }
    }

    // Pattern 2: Tuple construction with 1:1 variable-to-component mapping
    // {<<x, y, ...>> : x \in S, y \in T, ...}
    // Each tuple component is a single bound variable reference (distinct).
    if let Expr::Tuple(components) = &body.node {
        if let Some(result) = try_inverse_tuple_membership(ctx, target, components, bounds, span)? {
            return Ok(Some(result));
        }
    }

    // Pattern 3: Record construction with 1:1 variable-to-field mapping
    // {[f1 |-> x, f2 |-> y, ...] : x \in S, y \in T, ...}
    // Each field value is a single bound variable reference (distinct).
    if let Expr::Record(fields) = &body.node {
        if let Some(result) = try_inverse_record_membership(ctx, target, fields, bounds, span)? {
            return Ok(Some(result));
        }
    }

    Ok(None)
}

/// Inverse membership for tuple-valued SetBuilder:
/// `target \in {<<x, y, ...>> : x \in S, y \in T, ...}`
///
/// If the target is a tuple of the right length and each component in the body
/// is a distinct bound variable, decompose the target and check each component
/// against the corresponding domain. O(k) instead of O(|S|*|T|*...).
///
/// Returns `None` if the pattern doesn't match (components aren't simple variable refs,
/// or variables aren't 1:1 with bounds).
fn try_inverse_tuple_membership(
    ctx: &EvalCtx,
    target: &Value,
    components: &[Spanned<Expr>],
    bounds: &[tla_core::ast::BoundVar],
    span: Option<Span>,
) -> EvalResult<Option<bool>> {
    // Must have same number of components as bound variables
    if components.len() != bounds.len() {
        return Ok(None);
    }

    // Each component must be a simple Ident referencing a distinct bound variable
    let mut var_to_bound_idx: Vec<usize> = Vec::with_capacity(components.len());
    let mut used_bounds = vec![false; bounds.len()];

    for component in components.iter() {
        let Expr::Ident(name, _) = &component.node else {
            return Ok(None); // Component is not a simple variable ref
        };
        // Find which bound variable this references
        let Some(idx) = bounds.iter().position(|b| &b.name.node == name) else {
            return Ok(None); // References something that isn't a bound variable
        };
        if used_bounds[idx] {
            return Ok(None); // Same bound variable used twice (not 1:1)
        }
        used_bounds[idx] = true;
        var_to_bound_idx.push(idx);
    }

    // All bounds must be used
    if used_bounds.iter().any(|&u| !u) {
        return Ok(None);
    }

    // Pattern matched. Decompose target and check each component.
    let target_elems = match target.to_tuple_like_elements() {
        Some(elems) => elems,
        None => return Ok(Some(false)), // Target is not a tuple — can't be in the set
    };

    if target_elems.len() != components.len() {
        return Ok(Some(false)); // Wrong tuple length
    }

    // Check each target component against the corresponding bound variable's domain.
    for (comp_idx, &bound_idx) in var_to_bound_idx.iter().enumerate() {
        let domain = bounds[bound_idx]
            .domain
            .as_ref()
            .ok_or_else(|| EvalError::Internal {
                message: "SetBuilder requires bounded variables".into(),
                span,
            })?;
        if !super::eval_membership_lazy(ctx, &target_elems[comp_idx], domain)? {
            return Ok(Some(false));
        }
    }

    Ok(Some(true))
}

/// Inverse membership for record-valued SetBuilder:
/// `target \in {[f1 |-> x, f2 |-> y, ...] : x \in S, y \in T, ...}`
///
/// If the target is a record with the right fields and each field value in the body
/// is a distinct bound variable, decompose the target and check each field value
/// against the corresponding domain. O(k) instead of O(|S|*|T|*...).
///
/// Returns `None` if the pattern doesn't match.
fn try_inverse_record_membership(
    ctx: &EvalCtx,
    target: &Value,
    fields: &[(Spanned<String>, Spanned<Expr>)],
    bounds: &[tla_core::ast::BoundVar],
    span: Option<Span>,
) -> EvalResult<Option<bool>> {
    // Must have same number of fields as bound variables
    if fields.len() != bounds.len() {
        return Ok(None);
    }

    // Each field value must be a simple Ident referencing a distinct bound variable
    let mut field_to_bound_idx: Vec<usize> = Vec::with_capacity(fields.len());
    let mut used_bounds = vec![false; bounds.len()];

    for (_field_name, field_expr) in fields.iter() {
        let Expr::Ident(name, _) = &field_expr.node else {
            return Ok(None); // Field value is not a simple variable ref
        };
        let Some(idx) = bounds.iter().position(|b| &b.name.node == name) else {
            return Ok(None); // References something that isn't a bound variable
        };
        if used_bounds[idx] {
            return Ok(None); // Same bound variable used twice
        }
        used_bounds[idx] = true;
        field_to_bound_idx.push(idx);
    }

    // All bounds must be used
    if used_bounds.iter().any(|&u| !u) {
        return Ok(None);
    }

    // Pattern matched. Decompose target record and check each field.
    let Value::Record(rec) = target else {
        return Ok(Some(false)); // Target is not a record
    };

    if rec.len() != fields.len() {
        return Ok(Some(false)); // Wrong number of fields
    }

    // Check each field value against the corresponding bound variable's domain.
    for (field_idx, (field_name, _field_expr)) in fields.iter().enumerate() {
        let bound_idx = field_to_bound_idx[field_idx];
        let Some(field_val) = rec.get(field_name.node.as_str()) else {
            return Ok(Some(false)); // Record doesn't have this field
        };
        let domain = bounds[bound_idx]
            .domain
            .as_ref()
            .ok_or_else(|| EvalError::Internal {
                message: "SetBuilder requires bounded variables".into(),
                span,
            })?;
        if !super::eval_membership_lazy(ctx, field_val, domain)? {
            return Ok(Some(false));
        }
    }

    Ok(Some(true))
}

/// Recursive helper: iterate first bound variable's domain, bind each element,
/// and recurse on remaining bounds. When all bounds are bound, evaluate the
/// mapping expression and compare to target.
fn check_set_builder_membership_rec(
    ctx: &mut EvalCtx,
    target: &Value,
    body: &Spanned<Expr>,
    bounds: &[tla_core::ast::BoundVar],
    span: Option<Span>,
) -> EvalResult<bool> {
    if bounds.is_empty() {
        let mapped = eval(ctx, body)?;
        return Ok(mapped == *target);
    }

    let first = &bounds[0];
    let domain = first.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "SetBuilder requires bounded variables".into(),
        span,
    })?;

    let dv = eval(ctx, domain)?;
    let mark = ctx.mark_stack();

    for elem in eval_iter_set(ctx, &dv, Some(domain.span))? {
        push_bound_var_mut(ctx, first, &elem, span)?;
        if check_set_builder_membership_rec(ctx, target, body, &bounds[1..], span)? {
            ctx.pop_to_mark(&mark);
            return Ok(true); // Short-circuit: found a match
        }
        ctx.pop_to_mark(&mark);
    }

    Ok(false)
}
