// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TIR dispatch helpers for function-like and operator-like expressions.

use super::dispatch::eval_tir;
use super::StoredTirBody;
use crate::core::EvalCtx;
use crate::helpers::function_values::{
    apply_func_value_eager, apply_resolved_except_spec, canonicalize_except_result,
    collect_resolved_except_path, ResolvedExceptPath, ResolvedExceptPathElement,
    TirLazyExceptHandler,
};
use crate::{
    apply_closure_with_values, apply_user_op_with_values_resolved, eval_domain_value,
    resolved_operator_name_id,
};
use smallvec::SmallVec;
use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::{Span, Spanned};
use tla_tir::nodes::{TirExceptPathElement, TirExceptSpec, TirExpr, TirOperatorRef};
use tla_value::error::{EvalError, EvalResult};
use tla_value::Rp;
use tla_value::{FuncSetValue, Value};

// Kill-switch (fail-safe to the shallow one-level borrow): when set, the
// read-only operand borrow only elides `Name` and one-level `Name[arg]` reads.
// When unset (default), it recursively borrows arbitrarily deep read chains —
// `f[i][j]`, `f[i].field`, `r.a.b`, etc. — so the intermediate function-apply /
// record-access results are never cloned out only to be indexed and discarded.
feature_flag!(pub(super) no_deep_read_chain, "TY_NO_DEEP_READ_CHAIN");

pub(super) fn eval_tir_except_at(ctx: &EvalCtx, span: Option<tla_core::Span>) -> EvalResult<Value> {
    ctx.lookup("@").ok_or_else(|| EvalError::Internal {
        message: "TIR eval: EXCEPT @ used outside an EXCEPT update".to_string(),
        span,
    })
}

pub(super) fn eval_tir_op_ref(
    ctx: &EvalCtx,
    name: &str,
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    let body_span = span.unwrap_or_else(Span::dummy);
    let closure = ctx.create_closure(
        vec!["x".to_string(), "y".to_string()],
        Spanned {
            node: Expr::OpRef(name.to_string()),
            span: body_span,
        },
        ctx.local_ops().clone(),
    );
    Ok(Value::Closure(Rp::new(closure)))
}

pub(super) fn eval_tir_apply(
    ctx: &EvalCtx,
    op: &Spanned<TirExpr>,
    args: &[Spanned<TirExpr>],
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    if let TirExpr::Name(name_ref) = &op.node {
        if let Some(result) =
            try_apply_direct_tir_user_op(ctx, &name_ref.name, name_ref.name_id, args, span)
        {
            return result;
        }
    }

    let callable = eval_tir(ctx, op)?;
    // Keep the common <=4-argument case on the stack. This path sits above
    // both closure application and direct user-operator application, so a heap
    // Vec here would otherwise be allocated and freed for every TIR call.
    let arg_values: SmallVec<[Value; 4]> = args
        .iter()
        .map(|arg| eval_tir(ctx, arg))
        .collect::<EvalResult<_>>()?;
    match &callable {
        Value::Closure(closure) => {
            apply_closure_with_values(ctx, closure.as_ref(), &arg_values, span)
        }
        other => Err(EvalError::type_error("Closure", other, Some(op.span))),
    }
}

fn try_apply_direct_tir_user_op(
    ctx: &EvalCtx,
    operator_name: &str,
    operator_name_id: tla_core::NameId,
    args: &[Spanned<TirExpr>],
    span: Option<tla_core::Span>,
) -> Option<EvalResult<Value>> {
    if args.is_empty() {
        return None;
    }

    let resolved_name = ctx.resolve_op_name(operator_name);
    let def = ctx.get_op(resolved_name)?;
    if def.is_recursive
        || def.has_primed_param
        || crate::should_prefer_builtin_override(resolved_name, def.as_ref(), args.len(), ctx)
    {
        return None;
    }

    // TLA+ helper operators overwhelmingly have <=4 parameters. Match the
    // binding-side SmallVec fast path so cache hits do not first pay for a
    // temporary heap Vec of evaluated arguments.
    let arg_values: EvalResult<SmallVec<[Value; 4]>> =
        args.iter().map(|arg| eval_tir(ctx, arg)).collect();
    let resolved_name_id =
        resolved_operator_name_id(operator_name, operator_name_id, resolved_name);
    Some(arg_values.and_then(|values| {
        apply_user_op_with_values_resolved(ctx, resolved_name, resolved_name_id, def, &values, span)
    }))
}

pub(super) fn eval_tir_func_apply(
    ctx: &EvalCtx,
    func: &Spanned<TirExpr>,
    arg: &Spanned<TirExpr>,
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    // Borrowed-base fast path: `sv[arg]` where `sv` is a state variable.
    //
    // Churn elimination (top-ranked site on btree: 107M heap-backed state-var
    // read clones): the generic path below materializes the function value via
    // `eval_tir` → `TirNameKind::StateVar` → `StateEnvRef::get_value`, which
    // costs an Arc refcount bump plus a later drop — only to apply and discard
    // it. Borrow the slot's boxed Value in place instead. Mirrors the
    // `TirNameKind::StateVar` arm of `eval_tir_name` exactly (sparse next-state
    // overlay first in Next mode, then the current state array); falls back to
    // the generic path for inline scalar slots or when `state_env` is unset,
    // so error behavior and semantics are byte-identical.
    if let TirExpr::Name(name_ref) = &func.node {
        let borrowed = match &name_ref.kind {
            tla_tir::nodes::TirNameKind::StateVar { index } => {
                try_borrow_tir_state_var(ctx, *index as usize)
            }
            tla_tir::nodes::TirNameKind::Ident => try_borrow_tir_ident_value(ctx, name_ref),
        };
        if let Some(func_value) = borrowed {
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::StateVarReadElided,
            );
            return apply_tir_func_value_to_arg_expr(ctx, func_value, arg, func.span, span);
        }
    }

    // Deep-borrow chained bases (`f[i][j]`, `r.a[j]`): the base is itself a
    // read chain, so borrow it in place rather than materializing the
    // intermediate function/record owned only to index and drop it. The final
    // apply result is produced by the same `apply_tir_func_value_to_arg_expr`
    // tail (dense-2D fast path preserved), so semantics stay byte-identical.
    if !no_deep_read_chain()
        && matches!(
            &func.node,
            TirExpr::FuncApply { .. } | TirExpr::RecordAccess { .. }
        )
    {
        let base = eval_tir_operand_deep(ctx, func)?;
        return apply_tir_func_value_to_arg_expr(ctx, base.as_value(), arg, func.span, span);
    }

    let func_value = eval_tir(ctx, func)?;
    apply_tir_func_value_to_arg_expr(ctx, &func_value, arg, func.span, span)
}

/// Shared application tail for a resolved function value — used by both the
/// borrowed-base fast path and the generic owned path of
/// `eval_tir_func_apply`, so the dense-2-D acceleration below applies to
/// borrowed state-variable/constant bases too.
///
/// Fast path: `f[<<a, b>>]` where `f` is a dense 2-D integer function (e.g.
/// `childOf`/`valOf` over `Nodes \X Keys`). Evaluate the two subscript
/// components and index directly, skipping the `<<a, b>>` tuple allocation
/// and the binary search. Any miss (out of domain, or a non-integer
/// component) rebuilds the tuple and takes the normal path, so the observed
/// value/error is identical.
fn apply_tir_func_value_to_arg_expr(
    ctx: &EvalCtx,
    func_value: &Value,
    arg: &Spanned<TirExpr>,
    func_span: Span,
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    if let (Value::Func(fv), TirExpr::Tuple(elems)) = (func_value, &arg.node) {
        if !elems.is_empty() {
            // Virtual-tuple apply: evaluate the subscript components in tuple
            // order and look the key up component-wise (dense-2D O(1) index or
            // component-wise binary search) — no `Value::Tuple` allocation.
            // A domain miss rebuilds the key and takes the ordinary path, so
            // the NotInDomain error (which formats the key) is identical.
            let mut vals: SmallVec<[Value; 4]> = SmallVec::with_capacity(elems.len());
            for elem in elems {
                vals.push(eval_tir(ctx, elem)?);
            }
            if let Some(v) = fv.apply_tuple_elems(&vals) {
                tla_value::churn_stats::churn_count(
                    tla_value::churn_stats::ChurnSite::TupleKeyApplyTirDenseHit,
                );
                tla_value::churn_stats::churn_count_value(
                    tla_value::churn_stats::ChurnSite::FuncApplyResult,
                    tla_value::churn_stats::ChurnSite::FuncApplyResultHeap,
                    v,
                );
                return Ok(v.clone());
            }
            tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::TupleBuild);
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::TupleKeyApplyTirBuild,
            );
            let tup = Value::Tuple(vals.into_iter().collect());
            return apply_resolved_tir_func_value(func_value, tup, arg.span, span, Some(func_span));
        }
    }
    if let TirExpr::Tuple(elems) = &arg.node {
        if !elems.is_empty() {
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::TupleKeyApplyTirBuild,
            );
        }
    }
    let arg_value = eval_tir(ctx, arg)?;
    apply_resolved_tir_func_value(func_value, arg_value, arg.span, span, Some(func_span))
}

/// Borrow a state variable's heap-backed value for read-only consumption.
///
/// Mirrors the `TirNameKind::StateVar` arm of `eval_tir_name`: in Next mode a
/// bound sparse next-state slot wins; otherwise the current state array is
/// read. Returns `None` (caller falls back to the owned path) for inline
/// scalar slots, unbound-sparse-with-no-state-env, or missing `state_env`.
#[inline]
fn try_borrow_tir_state_var(ctx: &EvalCtx, idx: usize) -> Option<&Value> {
    use crate::cache::{current_state_lookup_mode, StateLookupMode};
    if current_state_lookup_mode(ctx) == StateLookupMode::Next {
        if let Some(sparse_env) = ctx.sparse_next_state_env {
            // SAFETY: index from TIR lowering bounded by VarRegistry.
            let slot = unsafe { sparse_env.get_unchecked(idx) };
            if let Some(value) = slot {
                return Some(value);
            }
            // None = unbound in witness, fall through to current state.
        }
    }
    let state_env = ctx.state_env?;
    debug_assert!(idx < state_env.env_len());
    // SAFETY: index from TIR lowering bounded by VarRegistry; the state array
    // outlives this evaluation (same contract as get_value).
    unsafe { state_env.get_heap_value_ref(idx) }
}

/// Borrow a heap-backed value for an `Ident`-kind TIR name — either a state
/// variable (state-var references inside operator bodies lower as `Ident`,
/// resolved at runtime via `var_idx_by_name_id`) or a precomputed constant.
///
/// Mirrors the front half of the `TirNameKind::Ident` arm of `eval_tir_name`:
/// returns `None` — falling back to the generic owned path — whenever any
/// earlier resolution tier could win (unresolvable NameId, a binding-chain
/// entry for the name, missing `state_env`) or the slot holds an inline
/// scalar. When it returns `Some`, it performs the same
/// `record_next_read`/`record_state_read` bookkeeping the owned arm performs,
/// so semantics and dep tracking are identical. The binding-chain probe is
/// read-only (`lookup`), so falling back never duplicates side effects.
///
/// The precomputed-constant tier fires only when no shadowing machinery is
/// active (no call-by-name / INSTANCE / eager substitutions, no LET overlay
/// or local ops, no operator replacements) — the same guards the AST
/// `eval_ident` PrecomputedConstant hint arm applies before returning the
/// constant. The map lives on `ctx.shared`, so the borrow is plain safe code.
#[inline]
fn try_borrow_tir_ident_value<'a>(
    ctx: &'a EvalCtx,
    name_ref: &tla_tir::nodes::TirNameRef,
) -> Option<&'a Value> {
    use crate::cache::{current_state_lookup_mode, StateLookupMode};
    use crate::{record_next_read, record_state_read};

    let lookup_id = if name_ref.name_id != tla_core::NameId::INVALID {
        name_ref.name_id
    } else {
        tla_core::name_intern::lookup_name_id(&name_ref.name)?
    };

    // Any chain binding for this name (Local/LET/Instance/Liveness) → the
    // generic path must resolve it; conservative read-only probe.
    if !ctx.bindings.is_empty() && ctx.bindings.lookup(lookup_id).is_some() {
        return None;
    }

    if let Some(idx) = crate::eval_ident::fast_var_idx_lookup(ctx, lookup_id) {
        if current_state_lookup_mode(ctx) == StateLookupMode::Next {
            if let Some(sparse_env) = ctx.sparse_next_state_env {
                // SAFETY: idx bounded by VarRegistry, sparse_env has same layout.
                let slot = unsafe { sparse_env.get_unchecked(idx.as_usize()) };
                if let Some(value) = slot {
                    record_next_read(ctx, idx, value);
                    return Some(value);
                }
                // None = unbound in witness, fall through to current state.
            }
        }

        let state_env = ctx.state_env?;
        debug_assert!(idx.as_usize() < state_env.env_len());
        // SAFETY: idx bounded by VarRegistry; the state array outlives this
        // evaluation (same contract as get_value).
        let v = unsafe { state_env.get_heap_value_ref(idx.as_usize())? };
        record_state_read(ctx, idx, v);
        return Some(v);
    }

    // Not a state variable — precomputed constant tier (guards mirror the
    // AST eval_ident PrecomputedConstant hint arm's shadowing checks).
    if ctx.call_by_name_subs.is_none()
        && ctx.instance_substitutions().is_none()
        && ctx.eager_subst_bindings.is_none()
        && ctx.let_def_overlay.is_empty()
        && ctx.local_ops.is_none()
        && ctx.shared.op_replacements.is_empty()
    {
        if let Some(v) = ctx.shared.precomputed_constants.get(&lookup_id) {
            return Some(v);
        }
    }
    None
}

/// A TIR operand that may be borrowed (state slot / precomputed constant /
/// Const node / borrowed function-apply result) or owned (everything else).
pub(super) enum TirOperand<'a> {
    Borrowed(&'a Value),
    Owned(Value),
}

impl TirOperand<'_> {
    #[inline(always)]
    pub(super) fn as_value(&self) -> &Value {
        match self {
            TirOperand::Borrowed(v) => v,
            TirOperand::Owned(v) => v,
        }
    }
}

/// Evaluate a TIR expression as a read-only operand, borrowing instead of
/// cloning when trivially possible.
///
/// Borrow-capable shapes (everything else falls back to plain `eval_tir`,
/// preserving evaluation order, error behavior, and dep tracking exactly):
/// - `Name` (state variable / precomputed constant) — via the same borrow
///   helpers as the function-apply fast path
/// - `Const` with a heap-backed value — the node owns the value
/// - `FuncApply` whose base is a borrowable `Name` — the apply RESULT is
///   borrowed from inside the function value (Func/IntFunc/Seq/Tuple/Record);
///   Bag/LazyFunc and error cases delegate to the owned resolver for
///   identical semantics
///
/// Churn elimination: read-only consumers (Eq/Neq comparison, set
/// membership) previously cloned state-var reads, constants, and apply
/// results out only to compare and drop them.
pub(super) fn eval_tir_operand<'a>(
    ctx: &'a EvalCtx,
    expr: &'a Spanned<TirExpr>,
) -> EvalResult<TirOperand<'a>> {
    if no_deep_read_chain() {
        eval_tir_operand_shallow(ctx, expr)
    } else {
        eval_tir_operand_deep(ctx, expr)
    }
}

/// Kill-switch path (`TY_NO_DEEP_READ_CHAIN`): borrow only `Name`, `Const`, and
/// a single-level `Name[arg]` function apply. Preserved verbatim as the
/// fail-safe reference for the deep recursive path.
fn eval_tir_operand_shallow<'a>(
    ctx: &'a EvalCtx,
    expr: &'a Spanned<TirExpr>,
) -> EvalResult<TirOperand<'a>> {
    let span = Some(expr.span);
    match &expr.node {
        TirExpr::Name(name_ref) => {
            let borrowed = match &name_ref.kind {
                tla_tir::nodes::TirNameKind::StateVar { index } => {
                    try_borrow_tir_state_var(ctx, *index as usize)
                }
                tla_tir::nodes::TirNameKind::Ident => try_borrow_tir_ident_value(ctx, name_ref),
            };
            if let Some(v) = borrowed {
                tla_value::churn_stats::churn_count(
                    tla_value::churn_stats::ChurnSite::StateVarReadElided,
                );
                return Ok(TirOperand::Borrowed(v));
            }
        }
        TirExpr::Const { value, .. } => {
            return Ok(TirOperand::Borrowed(value));
        }
        TirExpr::FuncApply { func, arg } => {
            if let TirExpr::Name(name_ref) = &func.node {
                let base = match &name_ref.kind {
                    tla_tir::nodes::TirNameKind::StateVar { index } => {
                        try_borrow_tir_state_var(ctx, *index as usize)
                    }
                    tla_tir::nodes::TirNameKind::Ident => try_borrow_tir_ident_value(ctx, name_ref),
                };
                if let Some(func_value) = base {
                    tla_value::churn_stats::churn_count(
                        tla_value::churn_stats::ChurnSite::StateVarReadElided,
                    );
                    if let TirExpr::Tuple(elems) = &arg.node {
                        if !elems.is_empty() {
                            tla_value::churn_stats::churn_count(
                                tla_value::churn_stats::ChurnSite::TupleKeyApplyTirBuild,
                            );
                            if matches!(func_value, Value::Func(fv) if !fv.dense_is_dim2()) {
                                tla_value::churn_stats::churn_count(
                                    tla_value::churn_stats::ChurnSite::TupleKeyApplyTirSparse,
                                );
                            }
                        }
                    }
                    let arg_value = eval_tir(ctx, arg)?;
                    if let Some(result) =
                        crate::helpers::function_values::try_apply_func_value_eager_borrowed(
                            func_value,
                            &arg_value,
                            Some(arg.span),
                            span,
                        )
                    {
                        return result.map(TirOperand::Borrowed);
                    }
                    // Bag / LazyFunc / type errors: owned resolver, identical
                    // semantics to the generic FuncApply path.
                    return apply_resolved_tir_func_value(
                        func_value,
                        arg_value,
                        arg.span,
                        span,
                        Some(func.span),
                    )
                    .map(TirOperand::Owned);
                }
            }
        }
        _ => {}
    }
    eval_tir(ctx, expr).map(TirOperand::Owned)
}

/// Default path: recursively borrow an arbitrarily deep read chain rooted at a
/// borrowable base (state variable / precomputed constant / `Const` node). Each
/// `FuncApply` and `RecordAccess` level borrows its result *from inside* the
/// borrowed base instead of cloning an owned intermediate out, so `f[i][j]`,
/// `f[i].field`, and `r.a.b` pay zero clones for the intermediates (and zero
/// for the final value too, when the consumer only reads it — Eq/Neq/`\in`).
///
/// Soundness: the borrow roots (state array, sparse next-state overlay,
/// precomputed-constant map, `Const` node) are all stable for the whole
/// evaluation, and `try_borrow_tir_ident_value` bails to the owned path
/// whenever a mutable binding could shadow the name — so no borrow outlives its
/// backing store. Every arm falls back to `eval_tir(..).map(Owned)`, and any
/// step that isn't eagerly borrowable (Bag / LazyFunc / type error) delegates
/// to the exact owned resolver, making the observed value/error byte-identical
/// to the generic owned path. Evaluation order (base before arg) matches
/// `eval_tir_func_apply` / `eval_tir_record_access`.
pub(super) fn eval_tir_operand_deep<'a>(
    ctx: &'a EvalCtx,
    expr: &'a Spanned<TirExpr>,
) -> EvalResult<TirOperand<'a>> {
    let span = Some(expr.span);
    match &expr.node {
        TirExpr::Name(name_ref) => {
            let borrowed = match &name_ref.kind {
                tla_tir::nodes::TirNameKind::StateVar { index } => {
                    try_borrow_tir_state_var(ctx, *index as usize)
                }
                tla_tir::nodes::TirNameKind::Ident => try_borrow_tir_ident_value(ctx, name_ref),
            };
            if let Some(v) = borrowed {
                tla_value::churn_stats::churn_count(
                    tla_value::churn_stats::ChurnSite::StateVarReadElided,
                );
                return Ok(TirOperand::Borrowed(v));
            }
        }
        TirExpr::Const { value, .. } => {
            return Ok(TirOperand::Borrowed(value));
        }
        TirExpr::FuncApply { func, arg } => {
            // Recursively borrow the base (Name / chained apply / record access).
            let base = eval_tir_operand_deep(ctx, func)?;
            if let TirExpr::Tuple(elems) = &arg.node {
                if !elems.is_empty() {
                    tla_value::churn_stats::churn_count(
                        tla_value::churn_stats::ChurnSite::TupleKeyApplyTirBuild,
                    );
                    if matches!(base.as_value(), Value::Func(fv) if !fv.dense_is_dim2()) {
                        tla_value::churn_stats::churn_count(
                            tla_value::churn_stats::ChurnSite::TupleKeyApplyTirSparse,
                        );
                    }
                }
            }
            let arg_value = eval_tir(ctx, arg)?;
            match base {
                TirOperand::Borrowed(func_value) => {
                    if let Some(result) =
                        crate::helpers::function_values::try_apply_func_value_eager_borrowed(
                            func_value,
                            &arg_value,
                            Some(arg.span),
                            span,
                        )
                    {
                        return result.map(TirOperand::Borrowed);
                    }
                    // Bag / LazyFunc / type errors: owned resolver, identical
                    // semantics to the generic FuncApply path.
                    return apply_resolved_tir_func_value(
                        func_value,
                        arg_value,
                        arg.span,
                        span,
                        Some(func.span),
                    )
                    .map(TirOperand::Owned);
                }
                TirOperand::Owned(func_value) => {
                    return apply_resolved_tir_func_value(
                        &func_value,
                        arg_value,
                        arg.span,
                        span,
                        Some(func.span),
                    )
                    .map(TirOperand::Owned);
                }
            }
        }
        TirExpr::RecordAccess { record, field } => {
            let base = eval_tir_operand_deep(ctx, record)?;
            match base {
                TirOperand::Borrowed(rv) => {
                    let rec = rv
                        .as_record()
                        .ok_or_else(|| EvalError::type_error("Record", rv, Some(record.span)))?;
                    let f =
                        rec.get_by_id(field.field_id)
                            .ok_or_else(|| EvalError::NoSuchField {
                                field: field.name.clone(),
                                record_display: Some(format!("{rv}")),
                                span,
                            })?;
                    return Ok(TirOperand::Borrowed(f));
                }
                TirOperand::Owned(rv) => {
                    let rec = rv
                        .as_record()
                        .ok_or_else(|| EvalError::type_error("Record", &rv, Some(record.span)))?;
                    let f = rec.get_by_id(field.field_id).cloned().ok_or_else(|| {
                        EvalError::NoSuchField {
                            field: field.name.clone(),
                            record_display: Some(format!("{rv}")),
                            span,
                        }
                    })?;
                    return Ok(TirOperand::Owned(f));
                }
            }
        }
        _ => {}
    }
    eval_tir(ctx, expr).map(TirOperand::Owned)
}

pub(super) fn eval_tir_func_set(
    ctx: &EvalCtx,
    domain: &Spanned<TirExpr>,
    range: &Spanned<TirExpr>,
) -> EvalResult<Value> {
    let domain_value = eval_tir(ctx, domain)?;
    let range_value = eval_tir(ctx, range)?;
    if !domain_value.is_set() {
        return Err(EvalError::type_error(
            "Set",
            &domain_value,
            Some(domain.span),
        ));
    }
    if !range_value.is_set() {
        return Err(EvalError::type_error("Set", &range_value, Some(range.span)));
    }
    Ok(Value::FuncSet(FuncSetValue::new(domain_value, range_value)))
}

pub(super) fn eval_tir_domain(
    ctx: &EvalCtx,
    inner: &Spanned<TirExpr>,
    _span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    let value = eval_tir(ctx, inner)?;
    eval_domain_value(&value).map_err(|err| match err {
        EvalError::TypeError { .. } => {
            EvalError::type_error("Function/Seq/Tuple/Record", &value, Some(inner.span))
        }
        other => other,
    })
}

pub(super) fn eval_tir_except(
    ctx: &EvalCtx,
    base: &Spanned<TirExpr>,
    specs: &[TirExceptSpec],
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    let mut result = eval_tir(ctx, base)?;
    // Resolve each spec's path once, apply it, and KEEP the resolved path
    // for the post-EXCEPT record canonicalization walk (mirrors the AST
    // `eval_except`; see `canonicalize_except_result`).
    let mut resolved_paths: smallvec::SmallVec<[ResolvedExceptPath; 4]> = smallvec::SmallVec::new();
    for spec in specs {
        let resolved = resolve_tir_except_path(ctx, &spec.path)?;
        let mut eval_new = |new_ctx: &EvalCtx| eval_tir(new_ctx, &spec.value);
        result = apply_resolved_except_spec(
            ctx,
            result,
            &resolved,
            &mut eval_new,
            &TirLazyExceptHandler,
            span,
        )?;
        resolved_paths.push(resolved);
    }
    canonicalize_except_result(&mut result, &resolved_paths);
    Ok(result)
}

/// Part of #3251: delegates eager value types to shared dispatch, handles
/// LazyFunc (memoized-only) and type errors locally.
fn apply_resolved_tir_func_value(
    func_value: &Value,
    arg: Value,
    arg_span: tla_core::Span,
    span: Option<tla_core::Span>,
    func_type_span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    if let Some(result) = apply_func_value_eager(func_value, &arg, Some(arg_span), span) {
        return result;
    }
    match func_value {
        Value::LazyFunc(lazy) => {
            if !lazy.in_domain(&arg) {
                return Err(EvalError::NotInDomain {
                    arg: format!("{arg}"),
                    func_display: Some(format!("{func_value}")),
                    span,
                });
            }
            if let Some(value) = lazy.memoized_value(&arg) {
                return Ok(value);
            }
            Err(EvalError::Internal {
                message: "TIR eval: LazyFunc application is not yet supported".to_string(),
                span,
            })
        }
        _ => Err(EvalError::type_error(
            "Function/Seq/Record",
            func_value,
            func_type_span,
        )),
    }
}

/// Part of #3251: pre-resolve TIR path elements for the shared
/// `apply_resolved_except_spec` dispatch.
fn resolve_tir_except_path(
    ctx: &EvalCtx,
    path: &[TirExceptPathElement],
) -> EvalResult<ResolvedExceptPath> {
    collect_resolved_except_path(path.iter().map(|p| match p {
        TirExceptPathElement::Index(expr) => {
            if matches!(&expr.node, TirExpr::Tuple(elems) if !elems.is_empty()) {
                tla_value::churn_stats::churn_count(
                    tla_value::churn_stats::ChurnSite::TupleKeyExceptTirBuild,
                );
            }
            Ok(ResolvedExceptPathElement::Index(eval_tir(ctx, expr)?))
        }
        TirExceptPathElement::Field(f) => Ok(ResolvedExceptPathElement::Field {
            name: f.name.clone(),
            field_id: f.field_id,
        }),
    }))
}

/// Evaluate a module-qualified operator reference (`M!Op(args)`).
///
/// The TIR lowerer currently inlines all module references at lowering time,
/// so this variant is never constructed. This stub resolves through the
/// existing module ref infrastructure if it is ever produced.
pub(super) fn eval_tir_operator_ref(
    ctx: &EvalCtx,
    op_ref: &TirOperatorRef,
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    // If no path segments, resolve as a plain operator reference
    if op_ref.path.is_empty() {
        if op_ref.args.is_empty() {
            // Route zero-arg OperatorRef through the cached evaluation path,
            // matching the fix in dispatch/name.rs. Previously bypassed the
            // zero-arg operator cache by evaluating TIR directly.
            let resolved_name = ctx.resolve_op_name(&op_ref.operator);
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    let shared_scope = ctx
                        .local_ops()
                        .as_ref()
                        .and_then(|local| local.get(resolved_name))
                        .is_none()
                        && ctx
                            .shared
                            .ops
                            .get(resolved_name)
                            .is_some_and(|shared_def| Arc::ptr_eq(shared_def, def));
                    return crate::eval_ident_zero_arg::eval_resolved_zero_arg_op(
                        ctx,
                        resolved_name,
                        def,
                        span,
                        shared_scope,
                    );
                }
            }
            // Fallback for unresolved operators
            return ctx
                .eval_op(&op_ref.operator)
                .map_err(|_| EvalError::UndefinedVar {
                    name: op_ref.operator.clone(),
                    span,
                });
        }
        // Parameterized operator: evaluate args then apply
        if let Some(result) = try_apply_direct_tir_user_op(
            ctx,
            &op_ref.operator,
            op_ref.operator_id,
            &op_ref.args,
            span,
        ) {
            return result;
        }
        let arg_values = op_ref
            .args
            .iter()
            .map(|arg| eval_tir(ctx, arg))
            .collect::<EvalResult<Vec<_>>>()?;
        let resolved_name = ctx.resolve_op_name(&op_ref.operator);
        if let Some(def) = ctx.get_op(resolved_name) {
            let params = def.params.iter().map(|p| p.name.node.clone()).collect();
            let mut closure = ctx.create_closure(params, def.body.clone(), ctx.local_ops().clone());
            // Part of #3392: attach TIR body so closure application stays in TIR.
            if let Some(tir_body) = super::try_resolve_operator_tir(resolved_name) {
                closure = closure.with_tir_body(Box::new(StoredTirBody::from_arc(tir_body)));
            }
            if def.is_recursive {
                closure = closure.with_name_if_missing(Arc::from(resolved_name));
            }
            return apply_closure_with_values(ctx, &closure, &arg_values, span);
        }
        return Err(EvalError::UndefinedVar {
            name: op_ref.operator.clone(),
            span,
        });
    }

    // Module-qualified operator reference: the lowerer should have inlined this.
    // If we reach here, it means the lowerer produced an OperatorRef that wasn't
    // resolved. Report a descriptive error.
    let path_str: String = op_ref
        .path
        .iter()
        .map(|seg| seg.name.as_str())
        .collect::<Vec<_>>()
        .join("!");
    Err(EvalError::Internal {
        message: format!(
            "TIR eval: module-qualified OperatorRef '{path_str}!{}' was not inlined by the lowerer",
            op_ref.operator
        ),
        span,
    })
}

/// Evaluate a `LAMBDA x, y : body` expression.
///
/// Part of #3163: Creates a `ClosureValue` with both the AST body (for the
/// `ClosureValue` constructor) and the TIR body (for TIR-native evaluation
/// at application time). The AST body was preserved during TIR lowering.
/// When the closure is later applied, closure dispatch detects
/// the `tir_body` slot and dispatches through `eval_tir` instead of AST `eval`.
/// Kill switch for lambda-body sharing (default ON). Set `TY_LAMBDA_SHARE=0` to
/// fall back to the historical deep-clone-every-construction path — used to
/// A/B-validate that body sharing preserves the exact verdict and state count.
fn lambda_body_sharing_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("TY_LAMBDA_SHARE").map_or(true, |v| v != "0"));
    *ENABLED
}

pub(super) fn eval_tir_lambda(
    ctx: &EvalCtx,
    params: &[String],
    tir_body: &Spanned<TirExpr>,
    ast_body: &tla_tir::nodes::PreservedAstBody,
    span: Option<tla_core::Span>,
) -> EvalResult<Value> {
    let _ = span;
    if !lambda_body_sharing_enabled() {
        // Historical path: deep-clone both the AST body (already an Arc, so this
        // re-copies its contents) and the TIR body on every construction.
        let closure = ctx
            .create_closure(
                params.to_vec(),
                (*ast_body.0).clone(),
                ctx.local_ops().clone(),
            )
            .with_tir_body(Box::new(StoredTirBody::new(tir_body.clone())));
        return Ok(Value::Closure(Rp::new(closure)));
    }

    // Share the already-`Arc`-wrapped AST body directly (no deep clone), and
    // reuse a single `Arc`-shared TIR body per lambda node across all of its
    // constructions (deep-cloned once, keyed by the stable TIR body pointer).
    // Both bodies are immutable and evaluation over them is purely functional,
    // so sharing is observationally identical to owning private deep copies.
    let tir_body_arc =
        crate::cache::lambda_tir_body_arc(tir_body as *const Spanned<TirExpr> as usize, || {
            Arc::new(tir_body.clone())
        });
    let closure = ctx
        .create_closure_arc(
            params.to_vec(),
            Arc::clone(&ast_body.0),
            ctx.local_ops().clone(),
        )
        .with_tir_body(Box::new(StoredTirBody::from_arc(tir_body_arc)));
    Ok(Value::Closure(Rp::new(closure)))
}
