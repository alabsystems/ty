// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Function application (`f[x]`) including the borrowed read fast path.
//!
//! Handles eager application for Func, IntFunc, Seq, Tuple, Record and
//! lazy application for LazyFunc with memoization and recursion depth tracking.

use super::super::build_lazy_func_ctx;
use super::lazy::bind_lazy_func_args;
use crate::core::EvalCtx;
use crate::eval_state_var_lookup::try_borrow_materialized_state_var;
use crate::helpers::eval;
use crate::value::Value;
use crate::MAX_RECURSION_DEPTH;
use tla_core::ast::{BoundPattern, Expr};
use tla_core::{Span, Spanned};
use tla_value::error::{EvalError, EvalResult};

pub(crate) fn try_apply_func_value_eager_borrowed<'a>(
    func_value: &'a Value,
    arg: &Value,
    arg_span: Option<Span>,
    span: Option<Span>,
) -> Option<EvalResult<&'a Value>> {
    match func_value {
        Value::Func(f) => Some(f.apply(arg).ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{arg}"),
            func_display: Some(format!("{func_value}")),
            span,
        })),
        Value::IntFunc(f) => Some(f.apply(arg).ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{arg}"),
            func_display: Some(format!("{func_value}")),
            span,
        })),
        // 1-based indexing. TLC TupleValue.java:144 reports "out of bounds" for
        // sequence/tuple indexing, NOT "not in domain" (FcnRcdValue.java).
        Value::Seq(s) => {
            let idx = match arg.as_i64() {
                Some(i) => i,
                None => return Some(Err(EvalError::type_error("Int", arg, arg_span))),
            };
            if idx < 1 || idx as usize > s.len() {
                return Some(Err(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: s.len(),
                    value_display: Some(format!("{func_value}")),
                    span,
                }));
            }
            Some(Ok(&s[(idx - 1) as usize]))
        }
        Value::Tuple(t) => {
            let idx = match arg.as_i64() {
                Some(i) => i,
                None => return Some(Err(EvalError::type_error("Int", arg, arg_span))),
            };
            if idx < 1 || idx as usize > t.len() {
                return Some(Err(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: t.len(),
                    value_display: Some(format!("{func_value}")),
                    span,
                }));
            }
            Some(Ok(&t[(idx - 1) as usize]))
        }
        Value::Record(r) => {
            let field = match arg.as_string() {
                Some(s) => s,
                None => return Some(Err(EvalError::type_error("STRING", arg, arg_span))),
            };
            Some(r.get(field).ok_or_else(|| EvalError::NoSuchField {
                field: field.into(),
                record_display: Some(format!("{func_value}")),
                span,
            }))
        }
        _ => None,
    }
}

/// Like `try_apply_func_value_eager_borrowed` but returns owned `Value`.
/// Used by `try_borrow_materialized_read` after `StateEnvRef` switched to
/// returning owned values.
fn try_apply_func_value_eager_owned(
    func_value: &Value,
    arg: &Value,
    arg_span: Option<Span>,
    span: Option<Span>,
) -> Option<EvalResult<Value>> {
    let result = try_apply_func_value_eager_owned_inner(func_value, arg, arg_span, span);
    if let Some(Ok(ref v)) = result {
        tla_value::churn_stats::churn_count_value(
            tla_value::churn_stats::ChurnSite::FuncApplyResult,
            tla_value::churn_stats::ChurnSite::FuncApplyResultHeap,
            v,
        );
    }
    result
}

fn try_apply_func_value_eager_owned_inner(
    func_value: &Value,
    arg: &Value,
    arg_span: Option<Span>,
    span: Option<Span>,
) -> Option<EvalResult<Value>> {
    match func_value {
        Value::Func(f) => Some(f.apply(arg).cloned().ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{arg}"),
            func_display: Some(format!("{func_value}")),
            span,
        })),
        // Compact bag: bag[elem] = SmallInt(count); same not-in-domain error
        // contract as Func.
        Value::Bag(b) => Some(b.apply(arg).ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{arg}"),
            func_display: Some(format!("{func_value}")),
            span,
        })),
        Value::IntFunc(f) => Some(f.apply(arg).cloned().ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{arg}"),
            func_display: Some(format!("{func_value}")),
            span,
        })),
        Value::Seq(s) => {
            let idx = match arg.as_i64() {
                Some(i) => i,
                None => return Some(Err(EvalError::type_error("Int", arg, arg_span))),
            };
            if idx < 1 || idx as usize > s.len() {
                return Some(Err(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: s.len(),
                    value_display: Some(format!("{func_value}")),
                    span,
                }));
            }
            Some(Ok(s[(idx - 1) as usize].clone()))
        }
        Value::Tuple(t) => {
            let idx = match arg.as_i64() {
                Some(i) => i,
                None => return Some(Err(EvalError::type_error("Int", arg, arg_span))),
            };
            if idx < 1 || idx as usize > t.len() {
                return Some(Err(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: t.len(),
                    value_display: Some(format!("{func_value}")),
                    span,
                }));
            }
            Some(Ok(t[(idx - 1) as usize].clone()))
        }
        Value::Record(r) => {
            let field = match arg.as_string() {
                Some(s) => s,
                None => return Some(Err(EvalError::type_error("STRING", arg, arg_span))),
            };
            Some(r.get(field).cloned().ok_or_else(|| EvalError::NoSuchField {
                field: field.into(),
                record_display: Some(format!("{func_value}")),
                span,
            }))
        }
        _ => None,
    }
}

pub(crate) fn try_borrow_materialized_read(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
) -> Option<EvalResult<Value>> {
    match &expr.node {
        Expr::StateVar(name, idx, name_id) => {
            try_borrow_materialized_state_var(ctx, name, *idx, *name_id).map(Ok)
        }
        Expr::RecordAccess(record_expr, field) => {
            let record_value = match try_borrow_materialized_read(ctx, record_expr)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let record = match record_value.as_record() {
                Some(record) => record,
                None => {
                    return Some(Err(EvalError::type_error(
                        "Record",
                        &record_value,
                        Some(record_expr.span),
                    )));
                }
            };
            Some(
                record
                    .get_by_id(field.field_id)
                    .cloned()
                    .ok_or_else(|| EvalError::NoSuchField {
                        field: field.name.node.clone(),
                        record_display: Some(format!("{record_value}")),
                        span: Some(expr.span),
                    }),
            )
        }
        Expr::FuncApply(func_expr, arg_expr) => {
            let func_value = match try_borrow_materialized_read(ctx, func_expr)? {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if let Some(result) =
                try_eval_ast_virtual_tuple_apply(ctx, &func_value, arg_expr, Some(expr.span), None)
            {
                return Some(result);
            }
            let arg = match eval(ctx, arg_expr) {
                Ok(arg) => arg,
                Err(err) => return Some(Err(err)),
            };
            try_apply_func_value_eager_owned(
                &func_value,
                &arg,
                Some(arg_expr.span),
                Some(expr.span),
            )
        }
        _ => None,
    }
}

/// Apply a function-like value to an argument for eager (non-lazy) value types.
///
/// Handles Func, IntFunc, Seq, Tuple, Record. Returns `None` for
/// LazyFunc and other types — caller must handle lazy evaluation and type errors.
///
/// Part of #3251: shared dispatch eliminates ~75 lines of duplication between
/// the AST path (`apply_resolved_func_value`) and TIR path
/// (`apply_resolved_tir_func_value`).
pub(crate) fn apply_func_value_eager(
    func_value: &Value,
    arg: &Value,
    arg_span: Option<Span>,
    span: Option<Span>,
) -> Option<EvalResult<Value>> {
    // Compact bag returns an owned SmallInt count — no borrowed fast path.
    if let Value::Bag(b) = func_value {
        return Some(b.apply(arg).ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{arg}"),
            func_display: Some(format!("{func_value}")),
            span,
        }));
    }
    try_apply_func_value_eager_borrowed(func_value, arg, arg_span, span).map(|result| {
        if let Ok(ref v) = result {
            tla_value::churn_stats::churn_count_value(
                tla_value::churn_stats::ChurnSite::FuncApplyResult,
                tla_value::churn_stats::ChurnSite::FuncApplyResultHeap,
                v,
            );
        }
        result.cloned()
    })
}

/// Evaluate `f[x]` once both operands have already been reduced to `Value`s.
pub(crate) fn apply_resolved_func_value(
    ctx: &EvalCtx,
    func_value: &Value,
    arg: Value,
    arg_span: Option<Span>,
    span: Option<Span>,
    func_type_span: Option<Span>,
) -> EvalResult<Value> {
    // Part of #3251: delegate eager value types to shared dispatch
    if let Some(result) = apply_func_value_eager(func_value, &arg, arg_span, span) {
        return result;
    }
    match func_value {
        Value::LazyFunc(f) => {
            if !f.in_domain(&arg) {
                return Err(EvalError::NotInDomain {
                    arg: format!("{arg}"),
                    func_display: Some(format!("{func_value}")),
                    span,
                });
            }

            if let Some(v) = f.memoized_value(&arg) {
                return Ok(v);
            }

            let new_depth = ctx.recursion_depth + 1;
            if new_depth > MAX_RECURSION_DEPTH {
                return Err(EvalError::Internal {
                    message: format!(
                        "Maximum recursion depth ({MAX_RECURSION_DEPTH}) exceeded in function evaluation. \
                         This may indicate infinite recursion or an overly deep recursive definition."
                    ),
                    span,
                });
            }

            // Part of #2955: Self-reference is now handled inside build_lazy_func_ctx
            let base_ctx = build_lazy_func_ctx(ctx, f, new_depth)?;

            // Part of #3395: single-var simple pattern tries recursive-local
            // rebind fast path before falling through to the shared binding
            // helper in lazy.rs (Part of #3423).
            let bounds = f.bounds();
            let pre = f.pre_interned_bounds();
            let bound_ctx = if bounds.len() == 1
                && (bounds[0].pattern.is_none()
                    || matches!(&bounds[0].pattern, Some(BoundPattern::Var(_))))
            {
                let (ref _name, id) = pre[0];
                let mut recursive_ctx = base_ctx;
                if let Some(self_name) = f.name() {
                    use tla_core::name_intern::intern_name;
                    let self_name_id = intern_name(self_name.as_ref());
                    if recursive_ctx.try_rebind_recursive_local_preinterned(
                        id,
                        self_name_id,
                        arg.clone(),
                    ) {
                        recursive_ctx
                    } else {
                        recursive_ctx.into_bind_by_id(id, arg.clone())
                    }
                } else {
                    recursive_ctx.into_bind_by_id(id, arg.clone())
                }
            } else {
                // Multi-var or complex single-var pattern: shared binding helper
                bind_lazy_func_args(base_ctx, bounds, pre, &arg, span)?
            };

            let value = eval(&bound_ctx, f.body())?;

            f.memoize_value(arg.clone(), value.clone());
            #[cfg(feature = "memory-stats")]
            crate::value::memory_stats::inc_memo_entry();

            Ok(value)
        }
        _ => Err(EvalError::type_error(
            "Function/Seq/Record",
            func_value,
            func_type_span,
        )),
    }
}

/// Virtual-tuple apply: `f[a, b]` / `f[<<a, b>>]` where the subscript is a
/// tuple literal and the base is a general `FuncValue` — look the key up
/// component-wise via [`tla_value::FuncValue::apply_tuple_elems`] without
/// materializing the `Value::Tuple` key (top-ranked churn site on btree:
/// ~5.05M AST apply-key tuple builds, each a malloc+free pair).
///
/// Returns `None` when the shape doesn't match (non-Func base or non-tuple
/// subscript) — the caller proceeds with the unchanged materializing path.
///
/// EXACTNESS: components are evaluated in the same order the tuple literal
/// would evaluate them (same errors), `apply_tuple_elems` is byte-identical
/// to `apply` on the materialized key, and a domain miss rebuilds the tuple
/// and routes through the ordinary path so the `NotInDomain` error (which
/// formats the key) is produced identically.
fn try_eval_ast_virtual_tuple_apply(
    ctx: &EvalCtx,
    func_value: &Value,
    arg_expr: &Spanned<Expr>,
    span: Option<Span>,
    func_type_span: Option<Span>,
) -> Option<EvalResult<Value>> {
    let Value::Func(fv) = func_value else {
        // Non-Func base with a tuple subscript still materializes (diagnostic).
        if matches!(&arg_expr.node, Expr::Tuple(elems) if !elems.is_empty()) {
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::TupleKeyApplyAstBuild,
            );
        }
        return None;
    };
    let Expr::Tuple(elems) = &arg_expr.node else {
        return None;
    };
    if elems.is_empty() {
        return None;
    }
    let mut vals: smallvec::SmallVec<[Value; 4]> =
        smallvec::SmallVec::with_capacity(elems.len());
    for elem in elems {
        match eval(ctx, elem) {
            Ok(v) => vals.push(v),
            Err(err) => return Some(Err(err)),
        }
    }
    if let Some(v) = fv.apply_tuple_elems(&vals) {
        // Match the ordinary path's clone-out churn accounting.
        tla_value::churn_stats::churn_count_value(
            tla_value::churn_stats::ChurnSite::FuncApplyResult,
            tla_value::churn_stats::ChurnSite::FuncApplyResultHeap,
            v,
        );
        return Some(Ok(v.clone()));
    }
    // Not in domain: materialize the key and take the ordinary path so the
    // NotInDomain error text/shape is identical. Count the build (this
    // construction bypasses eval_tuple's TupleBuild counter).
    tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::TupleBuild);
    tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::TupleKeyApplyAstBuild);
    let tup = Value::Tuple(vals.into_iter().collect());
    Some(apply_resolved_func_value(
        ctx,
        func_value,
        tup,
        Some(arg_expr.span),
        span,
        func_type_span,
    ))
}

pub(crate) fn eval_func_apply(
    ctx: &EvalCtx,
    func_expr: &Spanned<Expr>,
    arg_expr: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    // Borrowed-base fast path: `sv[arg]` where `sv` is a state variable.
    //
    // Churn elimination (top-ranked site on btree): the owned flow below goes
    // through `try_borrow_materialized_read` → `try_borrow_plain_state_var_slot`
    // → `StateEnvRef::get_value`, cloning the function value out of the state
    // slot (Arc bump + later drop) only to apply and discard it. Borrow the
    // slot's boxed Value in place instead. `try_borrow_plain_state_var_ref`
    // applies the exact same overlay guards + `record_state_read` bookkeeping
    // as the owned plain path, and returns `None` (falling through to the
    // unchanged owned flow) in every other case.
    if let Expr::StateVar(name, idx, name_id) = &func_expr.node {
        if let Some(func_value) =
            crate::eval_state_var_lookup::try_borrow_plain_state_var_ref(ctx, name, *idx, *name_id)
        {
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::StateVarReadElided,
            );
            if let Some(result) = try_eval_ast_virtual_tuple_apply(
                ctx,
                func_value,
                arg_expr,
                span,
                Some(func_expr.span),
            ) {
                return result;
            }
            let arg = eval(ctx, arg_expr)?;
            return apply_resolved_func_value(
                ctx,
                func_value,
                arg,
                Some(arg_expr.span),
                span,
                Some(func_expr.span),
            );
        }
    }

    if let Some(func_value) = try_borrow_materialized_read(ctx, func_expr) {
        let func_value = func_value?;
        if let Some(result) = try_eval_ast_virtual_tuple_apply(
            ctx,
            &func_value,
            arg_expr,
            span,
            Some(func_expr.span),
        ) {
            return result;
        }
        let arg = eval(ctx, arg_expr)?;
        return apply_resolved_func_value(
            ctx,
            &func_value,
            arg,
            Some(arg_expr.span),
            span,
            Some(func_expr.span),
        );
    }

    let fv = eval(ctx, func_expr)?;
    if let Some(result) =
        try_eval_ast_virtual_tuple_apply(ctx, &fv, arg_expr, span, Some(func_expr.span))
    {
        return result;
    }
    let arg = eval(ctx, arg_expr)?;
    apply_resolved_func_value(
        ctx,
        &fv,
        arg,
        Some(arg_expr.span),
        span,
        Some(func_expr.span),
    )
}
