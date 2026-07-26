// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Operator application dispatch (`eval_apply`) and closure invocation
//! (`apply_closure`, `apply_closure_with_values`).
//!
//! Extracted from `helpers/mod.rs` as part of #1669.

use super::super::{apply_builtin_binary_op, eval, EvalCtx, EvalError, EvalResult};
use super::builtin_dispatch::{eval_builtin, should_prefer_builtin_override};
use super::closures::{build_closure_ctx, create_closure_from_arg};
use super::function_values::apply_func_value_eager;
use super::param_cache::{
    get_closure_free_mask, get_closure_param_cache, get_param_cache, is_trivially_evaluable,
    nary_cache_enabled,
};
use crate::binding_chain::{BindingValue, LazyBinding};
use crate::cache::{
    current_state_lookup_mode, hash_args, nary_insert, nary_lookup_value, op_cache_entry_valid,
    propagate_cached_deps, CachedOpResult, NaryCacheHit, NaryOpCacheEntry, NaryOpCacheKey,
    OpDepGuard,
};
use crate::tir::{closure_tir_body_expr, eval_tir, record_closure_body_eval};
use crate::value::{ClosureValue, Value};
use num_traits::ToPrimitive;
use smallvec::SmallVec;
use std::sync::Arc;
use tla_core::ast::{Expr, OperatorDef, Substitution};
use tla_core::name_intern::{intern_name, resolve_name_id};
use tla_core::{NameId, Span, Spanned};
use tla_value::Rp;

/// Inline capacity for preinterned binding buffers. Operators with <= 4 parameters
/// (the vast majority — TLA+ operators rarely exceed 3-4 params) avoid heap
/// allocation entirely. Part of #3805: eliminates ~5.5% allocator overhead
/// measured by profiling (mi_malloc+mi_free self-time).
type PreinternedBuf = SmallVec<[(Arc<str>, BindingValue, NameId); 4]>;

enum UserOpCacheOutcome {
    Hit(Value),
    Miss(NaryOpCacheKey),
}

#[derive(Clone, Copy)]
struct UnaryIntProjection {
    index: i64,
    index_span: Span,
    body_span: Span,
}

#[derive(Clone, Copy)]
struct UnaryEmptyTupleIntProjection {
    empty_result: i64,
    index: i64,
    index_span: Span,
    apply_span: Span,
}

/// Recognize the exact unary projection shape `Op(value) == value[integer]`.
#[inline]
fn unary_int_projection(def: &OperatorDef) -> Option<UnaryIntProjection> {
    let [param] = def.params.as_slice() else {
        return None;
    };
    if param.arity != 0 {
        return None;
    }

    let Expr::FuncApply(func_expr, index_expr) = &def.body.node else {
        return None;
    };
    let Expr::Ident(func_name, _) = &func_expr.node else {
        return None;
    };
    if func_name != &param.name.node {
        return None;
    }
    let Expr::Int(index) = &index_expr.node else {
        return None;
    };
    let index = index.to_i64()?;

    Some(UnaryIntProjection {
        index,
        index_span: index_expr.span,
        body_span: def.body.span,
    })
}

/// Apply a recognized projection without building a binding frame or probing
/// the general n-ary cache. Returning `None` for a non-eager function value
/// preserves the existing LazyFunc/type-error path.
#[inline]
fn apply_unary_int_projection(
    projection: UnaryIntProjection,
    arg_value: &Value,
) -> Option<EvalResult<Value>> {
    let index_value = Value::SmallInt(projection.index);

    apply_func_value_eager(
        arg_value,
        &index_value,
        Some(projection.index_span),
        Some(projection.body_span),
    )
}

/// Recognize the exact unary guarded-projection shape
/// `Op(value) == IF value = <<>> THEN integer ELSE value[integer]`.
#[inline]
fn unary_empty_tuple_int_projection(def: &OperatorDef) -> Option<UnaryEmptyTupleIntProjection> {
    let [param] = def.params.as_slice() else {
        return None;
    };
    if param.arity != 0 {
        return None;
    }

    let Expr::If(cond, then_branch, else_branch) = &def.body.node else {
        return None;
    };
    let Expr::Eq(value_expr, empty_expr) = &cond.node else {
        return None;
    };
    let Expr::Ident(value_name, _) = &value_expr.node else {
        return None;
    };
    if value_name != &param.name.node
        || !matches!(&empty_expr.node, Expr::Tuple(elements) if elements.is_empty())
    {
        return None;
    }
    let Expr::Int(empty_result) = &then_branch.node else {
        return None;
    };
    let empty_result = empty_result.to_i64()?;

    let Expr::FuncApply(func_expr, index_expr) = &else_branch.node else {
        return None;
    };
    let Expr::Ident(func_name, _) = &func_expr.node else {
        return None;
    };
    if func_name != &param.name.node {
        return None;
    }
    let Expr::Int(index) = &index_expr.node else {
        return None;
    };
    let index = index.to_i64()?;

    Some(UnaryEmptyTupleIntProjection {
        empty_result,
        index,
        index_span: index_expr.span,
        apply_span: else_branch.span,
    })
}

/// Apply a recognized empty-tuple guarded projection without constructing the
/// empty tuple, building a binding frame, or probing the general n-ary cache.
/// Returning `None` for a non-eager function value preserves the existing
/// LazyFunc/type-error path.
#[inline]
fn apply_unary_empty_tuple_int_projection(
    projection: UnaryEmptyTupleIntProjection,
    arg_value: &Value,
) -> Option<EvalResult<Value>> {
    if arg_value.equals_empty_tuple() {
        return Some(Ok(Value::SmallInt(projection.empty_result)));
    }

    let index_value = Value::SmallInt(projection.index);
    apply_func_value_eager(
        arg_value,
        &index_value,
        Some(projection.index_span),
        Some(projection.apply_span),
    )
}

pub(crate) fn eval_apply(
    ctx: &EvalCtx,
    op_expr: &Spanned<Expr>,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    // Check if it's an identifier (operator name or closure variable)
    if let Expr::Ident(name, name_id) = &op_expr.node {
        // First check if this is a closure bound in the environment
        // Use lookup() to check local_stack first for O(1) enumeration bindings
        if let Some(Value::Closure(ref closure)) = ctx.lookup(name) {
            if closure.params().is_empty() && !args.is_empty() {
                // Zero-arg thunk closure applied to arguments: the thunk wraps
                // an operator-valued expression, not a value. This arises from
                // operator-valued INSTANCE substitutions — `CONSTANTS Leader(_)`
                // instantiated `WITH Leader <- Leader` materializes the LET def
                // `__ty_subst_Leader == Leader` (zero-arg), whose thunk body
                // names the outer module's unary operator. Resolve the operator
                // in the thunk's CAPTURED context and apply it to the argument
                // values evaluated in the CALLER's context (where the argument
                // expressions' bindings live). Previously this path was a
                // guaranteed arity-mismatch error, so redirecting it cannot
                // change any passing verdict (fail-closed recovery).
                return apply_operator_thunk(ctx, name, closure, args, span);
            }
            return apply_closure(ctx, closure, args, span);
        }
        // If it's a non-closure value in env, fall through to check ops

        // Apply operator replacement if configured (e.g., Seq <- BoundedSeq)
        let resolved_name = ctx.resolve_op_name(name);

        // Check for user-defined operators (allows shadowing stdlib)
        if let Some(def) = ctx.get_op(resolved_name) {
            let resolved_name_id = resolved_operator_name_id(name, *name_id, resolved_name);
            return apply_user_op_with_exprs(ctx, resolved_name, resolved_name_id, def, args, span);
        }

        // Check for built-in operators from stdlib (after user-defined)
        // Use resolved name for builtins too
        if let Some(result) = eval_builtin(ctx, resolved_name, args, span)? {
            return Ok(result);
        }

        // Undefined operator
        return Err(EvalError::UndefinedOp {
            name: resolved_name.to_string(),
            span,
        });
    }

    // If we get here with Apply, evaluate the operator expression
    // It might be a closure or other callable value
    let fv = eval(ctx, op_expr)?;
    if let Value::Closure(ref closure) = &fv {
        return apply_closure(ctx, closure, args, span);
    }

    Err(EvalError::Internal {
        message: format!("Cannot apply non-operator value: {fv:?}"),
        span,
    })
}

/// Apply a zero-arg thunk closure (deferred LET def / INSTANCE substitution)
/// to arguments.
///
/// The thunk body is an operator-valued expression (bare operator name, LAMBDA,
/// or built-in OpRef). TLC parity: the substitution RHS resolves in the OUTER
/// (definition-site) module context, while the argument expressions belong to
/// the application site and must be evaluated in the caller's context.
///
/// `create_closure_from_arg` performs the operator resolution and arity check
/// (`Leader <- Leader` with `CONSTANTS Leader(_)` must resolve to a 1-ary
/// operator); anything that is not operator-shaped remains a hard error.
#[cold]
#[inline(never)]
fn apply_operator_thunk(
    ctx: &EvalCtx,
    name: &str,
    thunk: &ClosureValue,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    // Evaluate argument expressions in the CALLER's context — they reference
    // application-site bindings (operator params, quantifier vars).
    let mut arg_values = Vec::with_capacity(args.len());
    for arg in args {
        arg_values.push(eval(ctx, arg)?);
    }
    // Resolve the operator-valued thunk body in the CAPTURED (definition-site)
    // context.
    let thunk_ctx = build_closure_ctx(ctx, thunk)?;
    let op_value = create_closure_from_arg(&thunk_ctx, thunk.body(), name, args.len(), span)?;
    match &op_value {
        Value::Closure(op_closure) => {
            apply_closure_with_values(&thunk_ctx, op_closure, &arg_values, span)
        }
        other => Err(EvalError::Internal {
            message: format!(
                "operator-valued thunk '{name}' resolved to non-operator value: {other:?}"
            ),
            span,
        }),
    }
}

fn user_op_cacheable(def: &Arc<OperatorDef>) -> bool {
    // TEMP DEBUG #4145: Selective cache disable to isolate which operator is broken.
    // Read the env var ONCE — this predicate runs on every operator application
    // and a per-call getenv(3) walk of the process environment cost ~1.3% of
    // total cycles on op-application-heavy specs (TLCSailfish1).
    static NARY_SKIP: std::sync::LazyLock<Option<String>> =
        std::sync::LazyLock::new(|| std::env::var("TY_DEBUG_NARY_SKIP").ok());
    if NARY_SKIP.as_deref() == Some(&def.name.node) {
        return false;
    }
    !def.is_recursive && !def.params.is_empty() && def.params.iter().all(|p| p.arity == 0)
}

/// Return the interned identity of the resolved operator without repeating a
/// global interner lookup when the source node already carries that identity.
///
/// `resolve_op_name` returns the original `&str` unchanged when no configured
/// operator replacement applies. Pointer identity therefore proves that the
/// pre-resolved `NameId` still names `resolved_name`. AST and TIR nodes guarantee
/// that every non-invalid `NameId` names the spelling stored on that node; the
/// debug assertion below catches malformed nodes that violate this invariant.
/// An invalid source id or a different returned string (including an operator
/// replacement) falls back to interning the resolved spelling, preserving the
/// previous cache-key identity.
#[inline]
pub(crate) fn resolved_operator_name_id(
    source_name: &str,
    source_name_id: NameId,
    resolved_name: &str,
) -> NameId {
    if source_name_id != NameId::INVALID && std::ptr::eq(source_name, resolved_name) {
        debug_assert_eq!(
            resolve_name_id(source_name_id).as_ref(),
            source_name,
            "a resolved source NameId must name its node spelling",
        );
        source_name_id
    } else {
        intern_name(resolved_name)
    }
}

/// Finish a [`NaryCacheHit`]: propagate state-hit dependencies and unwrap the
/// value. Persistent hits carry no dependencies by construction.
///
/// The hot path (`prepare_user_op_cache`) uses `nary_lookup_value`, which
/// fuses this finish step into the cache borrow so state hits propagate deps
/// by reference instead of deep-cloning `OpEvalDeps` out of the cache. Kept
/// (with its test) as the reference semantics for that fusion.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn finish_nary_cache_hit(ctx: &EvalCtx, hit: NaryCacheHit) -> Value {
    match hit {
        NaryCacheHit::Persistent(value) => value,
        NaryCacheHit::State(result) => {
            propagate_cached_deps(ctx, &result.deps);
            result.value
        }
    }
}

fn prepare_user_op_cache(
    ctx: &EvalCtx,
    resolved_name: &str,
    resolved_name_id: NameId,
    def: &Arc<OperatorDef>,
    arg_values: &[Value],
) -> Option<UserOpCacheOutcome> {
    if !user_op_cacheable(def) || !nary_cache_enabled() || arg_values.is_empty() {
        return None;
    }

    let key = operator_cache_key(ctx, resolved_name, resolved_name_id, def, arg_values);

    // Churn elimination: `nary_lookup_value` fuses the lookup with the
    // `finish_nary_cache_hit` step inside the cache borrow — state hits
    // propagate deps by reference and clone only the result Value out,
    // avoiding the OpEvalDeps deep clone (Box alloc per heap dep entry) that
    // cloning a `NaryCacheHit::State` payload out of the cache would pay.
    // Persistent hits are value-only by construction, exactly as in
    // `nary_lookup` + `finish_nary_cache_hit`.
    if let Some(value) = nary_lookup_value(&key, arg_values, op_cache_entry_valid, ctx) {
        tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::NaryCacheHit);
        return Some(UserOpCacheOutcome::Hit(value));
    }

    // Part of #3962: Read from EvalRuntimeState shadow instead of TLS.
    if crate::cache::lifecycle::in_enabled_scope_ctx(ctx) {
        if let Some(value) = crate::cache::op_result_cache::nary_constant_fallback(&key, arg_values)
        {
            return Some(UserOpCacheOutcome::Hit(value));
        }
    }

    Some(UserOpCacheOutcome::Miss(key))
}

/// Build the scope- and state-mode-discriminated operator key shared by the
/// ordinary n-ary cache and the recursive-result memo.
///
/// Recursive definitions can be reused through multiple INSTANCE/local-op
/// environments while retaining the same `Arc<OperatorDef>`. Keying those
/// results by definition address alone aliases semantically distinct calls.
/// Keeping this construction shared prevents the recursive lane from drifting
/// from the established n-ary cache discrimination contract.
fn operator_cache_key(
    ctx: &EvalCtx,
    resolved_name: &str,
    resolved_name_id: NameId,
    def: &Arc<OperatorDef>,
    arg_values: &[Value],
) -> NaryOpCacheKey {
    let is_instance_op = ctx
        .local_ops
        .as_ref()
        .and_then(|lo| lo.get(resolved_name))
        .is_some();
    NaryOpCacheKey {
        shared_id: ctx.shared.id,
        local_ops_id: crate::cache::scope_ids::resolve_local_ops_id_with_recursive(
            ctx.scope_ids.local_ops,
            &ctx.local_ops,
            ctx.scope_ids.local_ops_recursive,
        ),
        instance_subs_id: crate::cache::scope_ids::resolve_instance_subs_id(
            ctx.scope_ids.instance_substitutions,
            &ctx.instance_substitutions,
        ),
        op_name: resolved_name_id,
        def_loc: def.body.span.start,
        is_next_state: current_state_lookup_mode(ctx) == crate::cache::StateLookupMode::Next,
        args_hash: hash_args(arg_values),
        param_args_hash: if is_instance_op {
            ctx.stable.param_args_hash
        } else {
            0
        },
    }
}

fn eval_user_op_body_with_bindings(
    ctx: &EvalCtx,
    def: &Arc<OperatorDef>,
    preinterned: PreinternedBuf,
    cache_key: Option<NaryOpCacheKey>,
    cache_values: Option<&[Value]>,
) -> EvalResult<Value> {
    let mut new_ctx = ctx.bind_preinterned(preinterned);
    new_ctx.install_outermost_tlc_action_context(def);

    if let Some(key) = cache_key {
        let base_stack_len = ctx.binding_depth;
        let guard = OpDepGuard::from_ctx(ctx, base_stack_len);
        let result = eval(&new_ctx, &def.body)?;

        if let Some(mut deps) = guard.try_take_deps() {
            // Fix #3024: Strip internal locals to enable nary cache insertion.
            deps.strip_internal_locals(&ctx.bindings, base_stack_len);
            propagate_cached_deps(ctx, &deps);
            // Part of #4158: Taint deps for LazyFunc results that capture state.
            // FuncDef evaluation produces LazyFuncs via `build_lazy_func_from_ctx()`
            // which captures state arrays, but dep tracking sees no state reads
            // during construction. Without this, empty deps → persistent partition →
            // stale captured state in subsequent BFS states.
            if let Value::LazyFunc(ref f) = result {
                if f.captured_state().is_some() || f.captured_next_state().is_some() {
                    deps.instance_lazy_read = true;
                }
            }
            // Part of #3962: Read from EvalRuntimeState shadow instead of TLS.
            if !deps.inconsistent && !crate::cache::lifecycle::in_enabled_scope_ctx(ctx) {
                let args_arc = Arc::from(
                    cache_values.expect("user-op cache insert requires the argument values"),
                );
                nary_insert(
                    key,
                    NaryOpCacheEntry {
                        args: args_arc,
                        result: CachedOpResult {
                            value: result.clone(),
                            deps,
                        },
                    },
                );
            }
        }

        return Ok(result);
    }

    eval(&new_ctx, &def.body)
}

/// Kill switch for the linear/chain-recursive operator memo
/// (`TY_DISABLE_RECURSIVE_MEMO=1`). When set, recursive operators fall through to
/// normal (un-memoized) evaluation — used by the differential soundness gate.
#[inline]
fn recursive_memo_disabled() -> bool {
    static FLAG: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("TY_DISABLE_RECURSIVE_MEMO").map_or(false, |v| !v.is_empty() && v != "0")
    });
    *FLAG
}

/// Soft cap on the recursive-operator memo (matches the implied-fp cache bound);
/// cleared wholesale when exceeded so memory stays bounded on large runs.
const RECURSIVE_MEMO_CAP: usize = 131_072;

/// Try to evaluate a linear/chain-recursive operator (NOT a set-fold) via a
/// process-long memo keyed by the established n-ary context key plus exact
/// argument values.
///
/// Detects the *tail/linear recursion* idiom that `try_eval_recursive_fold`
/// rejects — e.g. `PublicKeyOf(ledger, h) == IF ... THEN ... ELSE
/// PublicKeyOf(ledger, ledger[h].block.previous)` — where the operator walks a
/// data structure toward a base case. Because the op-result (n-ary) cache skips
/// recursive operators, without this memo each such call re-walks the entire
/// chain, giving O(D²) work per state when an invariant/action calls it for many
/// elements (Nano's `CryptographicInvariant` does exactly this).
///
/// SOUNDNESS. A recursive operator is a pure function of its arguments **iff it
/// reads no state except through those arguments**. We do not assume this — we
/// PROVE it per call via the same dependency tracking the n-ary cache trusts:
/// the body is evaluated under an `OpDepGuard`, and the result is stored ONLY
/// when the tracked deps show no current-state read, no next-state read, no
/// TLC-level read, no INSTANCE-lazy taint, and no unstripped local dep. For such
/// results the value depends solely on the scope-discriminated, extensional,
/// exactly-keyed argument values, so a hit is byte-identical to recomputation in
/// the same evaluation scope.
/// State-dependent recursive ops fall through uncached (the value is still
/// returned correctly, with its deps propagated to the caller). Keys use exact
/// `Value` equality — no fingerprint, hence no collision risk. Returns `Ok(None)`
/// for shapes the memo does not handle (falls back to normal recursive eval).
fn try_eval_recursive_memoized(
    ctx: &EvalCtx,
    resolved_name: &str,
    resolved_name_id: NameId,
    def: &Arc<OperatorDef>,
    args: &[Spanned<Expr>],
) -> EvalResult<Option<Value>> {
    use super::recursion_stats::{rec_count, RecSite};

    // Only value-parameter operators can be keyed by argument values.
    if def.params.is_empty() || !def.params.iter().all(|p| p.arity == 0) {
        return Ok(None);
    }
    // Operators that (transitively) contain a prime or take a primed parameter can
    // observe next-state directly; keying on unprimed argument values would be
    // unsound, and primed-param operators must use call-by-name, not value-apply.
    // (The dep-tracking gate below also rejects next-state reads; this is belt.)
    if def.contains_prime || def.has_primed_param {
        return Ok(None);
    }

    // Count every eligible-shaped recursive application, even when the memo is
    // disabled, so TY_REC_STATS measures the true (un-memoized) recursion volume.
    rec_count(RecSite::Apply);

    if recursive_memo_disabled() {
        return Ok(None);
    }

    // Preserve the historical parameter-name interning and selective-laziness
    // contract. The normal operator path interns every parameter name before it
    // evaluates any argument, and evaluates only O(1) argument shapes eagerly;
    // all other arity-0 arguments are LazyBindings and may never be forced. A
    // memo probe needs every argument value, so it is eligible only when the
    // normal path would eagerly evaluate every argument anyway.
    let _cached_params = get_param_cache(def);
    if !args.iter().all(|arg| is_trivially_evaluable(&arg.node)) {
        return Ok(None);
    }

    // Evaluate in the same left-to-right order as normal eager parameter
    // binding. Errors are returned directly; falling through would re-evaluate
    // earlier arguments and could perturb first-seen string-token ordering.
    let mut arg_values: Vec<Value> = Vec::with_capacity(args.len());
    for arg in args {
        arg_values.push(eval(ctx, arg)?);
    }
    // Only extensionally-comparable argument values can key a cross-state cache
    // (identity-like LazyFunc/Closure values are rejected — same rule as the
    // recursive-fold cache).
    if !super::recursive_fold::fold_cache_args_safe(&arg_values) {
        // Still evaluate via the tracked body path (avoids re-evaluating args) but
        // do not cache.
        let (value, _pure) = eval_recursive_body_tracked(ctx, def, &arg_values)?;
        rec_count(RecSite::Compute);
        return Ok(Some(value));
    }

    let context_key = operator_cache_key(ctx, resolved_name, resolved_name_id, def, &arg_values);

    // Memo lookup. A hit is a proven state-independent value → nothing to
    // propagate to the caller's dep frame (a constant carries no dependency).
    let hit = crate::cache::small_caches::SMALL_CACHES.with(|sc| {
        sc.borrow()
            .recursive_result_cache
            .get(&(context_key.clone(), arg_values.clone()))
            .cloned()
    });
    if let Some(value) = hit {
        rec_count(RecSite::Hit);
        return Ok(Some(value));
    }

    // Miss: evaluate the body under dependency tracking.
    let (value, state_independent) = eval_recursive_body_tracked(ctx, def, &arg_values)?;
    rec_count(RecSite::Compute);

    if state_independent {
        rec_count(RecSite::Cached);
        crate::cache::small_caches::SMALL_CACHES.with(|sc| {
            let mut sc = sc.borrow_mut();
            if sc.recursive_result_cache.len() >= RECURSIVE_MEMO_CAP {
                sc.recursive_result_cache.clear();
            }
            sc.recursive_result_cache
                .insert((context_key, arg_values), value.clone());
        });
    } else {
        rec_count(RecSite::Impure);
    }
    Ok(Some(value))
}

/// Bind `arg_values` as the operator's parameters and evaluate its body under an
/// `OpDepGuard`, returning the value and whether the tracked deps prove the
/// result state-independent. Mirrors `eval_user_op_body_with_bindings`'s dep
/// discipline (caller-relative `base_stack_len`, strip internal locals, propagate
/// deps upward) so the classification matches the n-ary cache's "constant" notion.
fn eval_recursive_body_tracked(
    ctx: &EvalCtx,
    def: &Arc<OperatorDef>,
    arg_values: &[Value],
) -> EvalResult<(Value, bool)> {
    let cached_params = get_param_cache(def);
    let mut preinterned: PreinternedBuf = SmallVec::with_capacity(def.params.len());
    for (i, value) in arg_values.iter().enumerate() {
        let (ref interned, name_id) = cached_params[i];
        preinterned.push((
            Arc::clone(interned),
            BindingValue::eager(value.clone()),
            name_id,
        ));
    }
    let mut new_ctx = ctx.bind_preinterned(preinterned);
    new_ctx.install_outermost_tlc_action_context(def);

    let base_stack_len = ctx.binding_depth;
    let guard = OpDepGuard::from_ctx(ctx, base_stack_len);
    let result = eval(&new_ctx, &def.body)?;
    let mut deps = guard.try_take_deps().ok_or_else(|| EvalError::Internal {
        message: "recursive-memo dependency frame unexpectedly empty".into(),
        span: Some(def.body.span),
    })?;
    deps.strip_internal_locals(&ctx.bindings, base_stack_len);
    // A FuncDef can construct a LazyFunc without reading the captured arrays
    // during construction. Dependency tracking is therefore empty even though
    // replaying that value in another state would observe stale state. Match the
    // established persistent-cache admission rule and fail closed for every
    // state-environment-capturing value.
    if result.captures_state_environment() {
        deps.instance_lazy_read = true;
    }
    propagate_cached_deps(ctx, &deps);

    let state_independent = !deps.inconsistent
        && deps.state.is_empty()
        && deps.next.is_empty()
        && deps.local.is_empty()
        && deps.tlc_level.is_none()
        && !deps.instance_lazy_read;

    Ok((result, state_independent))
}

fn apply_user_op_with_exprs(
    ctx: &EvalCtx,
    resolved_name: &str,
    resolved_name_id: NameId,
    def: &Arc<OperatorDef>,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    if should_prefer_builtin_override(resolved_name, def, args.len(), ctx) {
        if let Some(result) = eval_builtin(ctx, resolved_name, args, span)? {
            return Ok(result);
        }
    }

    if def.params.len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: resolved_name.to_string(),
            expected: def.params.len(),
            got: args.len(),
            span,
        });
    }

    if def.is_recursive {
        if let Some(result) = super::recursive_fold::try_eval_recursive_fold(ctx, def, args, span)?
        {
            return Ok(result);
        }
        // Linear/chain recursion (e.g. NanoBlockchain's PublicKeyOf walking
        // block.previous) that the fold path rejects: memoize by argument values
        // so it is not re-walked O(D) times per state. See the function's
        // SOUNDNESS note — results are cached only when proven state-independent.
        if let Some(result) =
            try_eval_recursive_memoized(ctx, resolved_name, resolved_name_id, def, args)?
        {
            return Ok(result);
        }
    }

    if def.has_primed_param {
        let subs: Vec<Substitution> = def
            .params
            .iter()
            .zip(args.iter())
            .map(|(param, arg)| Substitution {
                from: param.name.clone(),
                to: arg.clone(),
            })
            .collect();
        let ctx_with_subs = ctx.with_call_by_name_subs(subs);
        return eval(&ctx_with_subs, &def.body);
    }

    // Preserve the historical eager parameter-name interning order. Parameter
    // names share the TLC string-token table, so moving this below argument
    // evaluation could change TLC's first-seen string ordering.
    let cached_params = get_param_cache(def);
    let mut cache_key = None;
    let mut forced_values = None;

    if user_op_cacheable(def) && nary_cache_enabled() {
        // Part of #3805: SmallVec avoids heap allocation for operators with <= 4 params.
        let mut arg_values: SmallVec<[Value; 4]> = SmallVec::with_capacity(def.params.len());
        let mut all_forced = true;
        for arg in args {
            match eval(ctx, arg) {
                Ok(v) => arg_values.push(v),
                Err(_) => {
                    all_forced = false;
                    break;
                }
            }
        }

        if all_forced {
            if let Some(projection) = unary_int_projection(def) {
                if let Some(result) = apply_unary_int_projection(projection, &arg_values[0]) {
                    return result;
                }
            }
            if let Some(projection) = unary_empty_tuple_int_projection(def) {
                if let Some(result) =
                    apply_unary_empty_tuple_int_projection(projection, &arg_values[0])
                {
                    return result;
                }
            }
            match prepare_user_op_cache(ctx, resolved_name, resolved_name_id, def, &arg_values) {
                Some(UserOpCacheOutcome::Hit(value)) => return Ok(value),
                Some(UserOpCacheOutcome::Miss(key)) => {
                    cache_key = Some(key);
                    forced_values = Some(arg_values);
                }
                None => {}
            }
        }
    }

    // Part of #3805: SmallVec avoids heap allocation for operators with <= 4 params.
    tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::OpApplyWithValues);
    let mut preinterned: PreinternedBuf = SmallVec::with_capacity(def.params.len());
    for (i, (param, arg)) in def.params.iter().zip(args.iter()).enumerate() {
        let bv = if param.arity > 0 {
            BindingValue::eager(create_closure_from_arg(
                ctx,
                arg,
                &param.name.node,
                param.arity,
                span,
            )?)
        } else if let Some(ref vals) = forced_values {
            tla_value::churn_stats::churn_count_value(
                tla_value::churn_stats::ChurnSite::OpApplyArgClone,
                tla_value::churn_stats::ChurnSite::OpApplyArgCloneHeap,
                &vals[i],
            );
            BindingValue::eager(vals[i].clone())
        } else if is_trivially_evaluable(&arg.node) {
            BindingValue::eager(eval(ctx, arg)?)
        } else {
            BindingValue::Lazy(Box::new(LazyBinding::new(
                arg as *const Spanned<tla_core::ast::Expr>,
                &ctx.bindings,
            )))
        };
        let (ref interned, name_id) = cached_params[i];
        preinterned.push((Arc::clone(interned), bv, name_id));
    }

    eval_user_op_body_with_bindings(ctx, def, preinterned, cache_key, forced_values.as_deref())
}

/// Apply a user operator when the caller only has its resolved spelling.
///
/// Keep non-AST/TIR callers on this conservative wrapper so cache-key identity
/// is always obtained from the interner.
pub(crate) fn apply_user_op_with_values(
    ctx: &EvalCtx,
    resolved_name: &str,
    def: &Arc<OperatorDef>,
    arg_values: &[Value],
    span: Option<Span>,
) -> EvalResult<Value> {
    apply_user_op_with_values_resolved(
        ctx,
        resolved_name,
        intern_name(resolved_name),
        def,
        arg_values,
        span,
    )
}

/// Apply a user operator whose resolved spelling and `NameId` are already
/// known. The TIR direct-apply path uses this to avoid re-entering the global
/// name interner on every n-ary cache lookup; the AST expression path threads
/// the same identity directly into cache preparation.
pub(crate) fn apply_user_op_with_values_resolved(
    ctx: &EvalCtx,
    resolved_name: &str,
    resolved_name_id: NameId,
    def: &Arc<OperatorDef>,
    arg_values: &[Value],
    span: Option<Span>,
) -> EvalResult<Value> {
    if def.params.len() != arg_values.len() {
        return Err(EvalError::ArityMismatch {
            op: resolved_name.to_string(),
            expected: def.params.len(),
            got: arg_values.len(),
            span,
        });
    }

    if def.has_primed_param {
        return Err(EvalError::Internal {
            message: format!(
                "direct user-op value apply reached primed-parameter operator '{resolved_name}'"
            ),
            span,
        });
    }

    let mut cache_key = None;
    if user_op_cacheable(def) && nary_cache_enabled() {
        if let Some(projection) = unary_int_projection(def) {
            // A first cache miss historically interns parameter spellings
            // before evaluating the body. Preserve that TLC string-token and
            // NameId ordering even though this path bypasses the n-ary cache.
            drop(get_param_cache(def));
            if let Some(result) = apply_unary_int_projection(projection, &arg_values[0]) {
                return result;
            }
        }
        if let Some(projection) = unary_empty_tuple_int_projection(def) {
            // As for the direct projection above, retain first-miss parameter
            // interning before replacing the complete operator body.
            drop(get_param_cache(def));
            if let Some(result) = apply_unary_empty_tuple_int_projection(projection, &arg_values[0])
            {
                return result;
            }
        }
        match prepare_user_op_cache(ctx, resolved_name, resolved_name_id, def, arg_values) {
            Some(UserOpCacheOutcome::Hit(value)) => return Ok(value),
            Some(UserOpCacheOutcome::Miss(key)) => cache_key = Some(key),
            None => {}
        }
    }

    let cached_params = get_param_cache(def);
    tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::OpApplyWithValues);
    // Part of #3805: SmallVec avoids heap allocation for operators with <= 4 params.
    let preinterned: PreinternedBuf = arg_values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let (ref interned, name_id) = cached_params[i];
            tla_value::churn_stats::churn_count_value(
                tla_value::churn_stats::ChurnSite::OpApplyArgClone,
                tla_value::churn_stats::ChurnSite::OpApplyArgCloneHeap,
                value,
            );
            (
                Arc::clone(interned),
                BindingValue::eager(value.clone()),
                name_id,
            )
        })
        .collect();

    let cache_values = if cache_key.is_some() {
        Some(arg_values)
    } else {
        None
    };
    eval_user_op_body_with_bindings(ctx, def, preinterned, cache_key, cache_values)
}

fn eval_closure_body(
    ctx: &EvalCtx,
    closure: &ClosureValue,
    preinterned: PreinternedBuf,
) -> EvalResult<Value> {
    let mut closure_ctx = build_closure_ctx(ctx, closure)?;
    if let Some(name) = closure.name() {
        closure_ctx.push_binding_preinterned(
            Arc::clone(name),
            Value::Closure(Rp::new(closure.clone())),
            intern_name(name.as_ref()),
        );
    }
    let ctx_with_bindings = closure_ctx.bind_preinterned(preinterned);
    if let Some(tir_body) = closure_tir_body_expr(closure) {
        record_closure_body_eval();
        // Part of #3392: fall back to AST body if TIR eval fails. This mirrors
        // eval_named_op's fallback pattern and handles TIR expressiveness gaps
        // (e.g., recursive LET functions that TIR lowering doesn't fully support).
        match eval_tir(&ctx_with_bindings, tir_body) {
            Ok(v) => return Ok(v),
            Err(_) => {} // TIR eval failed, fall back to AST body
        }
    }
    eval(&ctx_with_bindings, closure.body())
}

/// Apply a closure to arguments
pub(super) fn apply_closure(
    ctx: &EvalCtx,
    closure: &ClosureValue,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    if closure.params().len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("<closure#{}>", closure.id()),
            expected: closure.params().len(),
            got: args.len(),
            span,
        });
    }

    // Fast-path: closure wraps a built-in operator reference (OpRef).
    if closure.params().len() == 2 {
        if let Expr::OpRef(op) = &closure.body().node {
            let left = eval(ctx, &args[0])?;
            let right = eval(ctx, &args[1])?;
            return apply_builtin_binary_op(op, left, right, span);
        }
    }

    // Part of #3021: Use cached interned params to avoid Arc::from() allocation
    // and 2 DashMap lookups per parameter on every closure application.
    // Part of #3805: SmallVec avoids heap allocation for closures with <= 4 params.
    let cached_params = get_closure_param_cache(closure);
    // Cache the per-parameter "free in body" mask (keyed by closure id) instead
    // of recomputing free_vars(body) — a full-body AST walk + HashSet alloc — on
    // every closure application. SlidingPuzzles applies one LAMBDA ~20M times.
    let free_mask = get_closure_free_mask(closure);
    let mut preinterned: PreinternedBuf = SmallVec::with_capacity(closure.params().len());
    for (i, arg) in args.iter().enumerate() {
        let bv = if free_mask[i] {
            BindingValue::eager(eval(ctx, arg)?)
        } else {
            BindingValue::Lazy(Box::new(LazyBinding::new(
                arg as *const Spanned<Expr>,
                &ctx.bindings,
            )))
        };
        let (ref interned, name_id) = cached_params[i];
        preinterned.push((Arc::clone(interned), bv, name_id));
    }

    eval_closure_body(ctx, closure, preinterned)
}

/// Apply a closure to already-evaluated arguments.
pub(crate) fn apply_closure_with_values(
    ctx: &EvalCtx,
    closure: &ClosureValue,
    args: &[Value],
    span: Option<Span>,
) -> EvalResult<Value> {
    if closure.params().len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("<closure#{}>", closure.id()),
            expected: closure.params().len(),
            got: args.len(),
            span,
        });
    }

    // Fast-path: closure wraps a built-in operator reference (OpRef).
    if closure.params().len() == 2 {
        if let Expr::OpRef(op) = &closure.body().node {
            return apply_builtin_binary_op(op, args[0].clone(), args[1].clone(), span);
        }
    }

    // Part of #3021: Use cached interned params to avoid Arc::from() allocation
    // and 2 DashMap lookups per parameter on every closure application.
    // Part of #3805: SmallVec avoids heap allocation for closures with <= 4 params.
    let cached_params = get_closure_param_cache(closure);
    tla_value::churn_stats::churn_count(tla_value::churn_stats::ChurnSite::OpApplyWithValues);
    let preinterned: PreinternedBuf = args
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let (ref interned, name_id) = cached_params[i];
            tla_value::churn_stats::churn_count_value(
                tla_value::churn_stats::ChurnSite::OpApplyArgClone,
                tla_value::churn_stats::ChurnSite::OpApplyArgCloneHeap,
                value,
            );
            (
                Arc::clone(interned),
                BindingValue::eager(value.clone()),
                name_id,
            )
        })
        .collect();

    eval_closure_body(ctx, closure, preinterned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::OpEvalDeps;
    use crate::value::{BagValue, FuncValue, IntIntervalFunc};
    use crate::var_index::VarIndex;
    use tla_core::ast::OpParam;
    use tla_core::FileId;

    fn projection_def_with_param(
        param_name: &str,
        projected_name: &str,
        index: i64,
    ) -> Arc<OperatorDef> {
        let param_span = Span::new(FileId(17), 10, 11);
        let func_span = Span::new(FileId(17), 20, 21);
        let index_span = Span::new(FileId(17), 22, 23);
        let body_span = Span::new(FileId(17), 20, 24);
        Arc::new(OperatorDef {
            name: Spanned::new("Project".to_string(), Span::new(FileId(17), 0, 7)),
            params: vec![OpParam {
                name: Spanned::new(param_name.to_string(), param_span),
                arity: 0,
            }],
            body: Spanned::new(
                Expr::FuncApply(
                    Box::new(Spanned::new(
                        Expr::Ident(projected_name.to_string(), NameId::INVALID),
                        func_span,
                    )),
                    Box::new(Spanned::new(Expr::Int(index.into()), index_span)),
                ),
                body_span,
            ),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        })
    }

    fn projection_def(projected_name: &str, index: i64) -> Arc<OperatorDef> {
        projection_def_with_param("value", projected_name, index)
    }

    fn empty_tuple_projection_def_with_param(
        param_name: &str,
        condition_name: &str,
        projected_name: &str,
        empty_result: i64,
        index: i64,
    ) -> Arc<OperatorDef> {
        let param_span = Span::new(FileId(19), 10, 11);
        let condition_value_span = Span::new(FileId(19), 20, 21);
        let empty_tuple_span = Span::new(FileId(19), 24, 28);
        let condition_span = Span::new(FileId(19), 20, 28);
        let then_span = Span::new(FileId(19), 34, 35);
        let func_span = Span::new(FileId(19), 41, 42);
        let index_span = Span::new(FileId(19), 43, 44);
        let apply_span = Span::new(FileId(19), 41, 45);
        let body_span = Span::new(FileId(19), 17, 45);
        Arc::new(OperatorDef {
            name: Spanned::new(
                "EmptyDefaultProject".to_string(),
                Span::new(FileId(19), 0, 10),
            ),
            params: vec![OpParam {
                name: Spanned::new(param_name.to_string(), param_span),
                arity: 0,
            }],
            body: Spanned::new(
                Expr::If(
                    Box::new(Spanned::new(
                        Expr::Eq(
                            Box::new(Spanned::new(
                                Expr::Ident(condition_name.to_string(), NameId::INVALID),
                                condition_value_span,
                            )),
                            Box::new(Spanned::new(Expr::Tuple(vec![]), empty_tuple_span)),
                        ),
                        condition_span,
                    )),
                    Box::new(Spanned::new(Expr::Int(empty_result.into()), then_span)),
                    Box::new(Spanned::new(
                        Expr::FuncApply(
                            Box::new(Spanned::new(
                                Expr::Ident(projected_name.to_string(), NameId::INVALID),
                                func_span,
                            )),
                            Box::new(Spanned::new(Expr::Int(index.into()), index_span)),
                        ),
                        apply_span,
                    )),
                ),
                body_span,
            ),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        })
    }

    fn empty_tuple_projection_def(empty_result: i64, index: i64) -> Arc<OperatorDef> {
        empty_tuple_projection_def_with_param("value", "value", "value", empty_result, index)
    }

    #[test]
    fn unary_int_projection_fast_path_covers_ast_and_resolved_value_calls() {
        let def = projection_def("value", 2);
        let ctx = EvalCtx::new();
        let tuple_expr = Spanned::new(
            Expr::Tuple(vec![
                Spanned::dummy(Expr::Int(10.into())),
                Spanned::dummy(Expr::Int(20.into())),
            ]),
            Span::new(FileId(17), 30, 40),
        );

        assert_eq!(
            apply_user_op_with_exprs(
                &ctx,
                "Project",
                intern_name("Project"),
                &def,
                &[tuple_expr],
                None,
            )
            .expect("AST projection should evaluate"),
            Value::SmallInt(20),
        );
        assert_eq!(
            apply_user_op_with_values_resolved(
                &ctx,
                "Project",
                intern_name("Project"),
                &def,
                &[Value::Tuple(
                    vec![Value::SmallInt(30), Value::SmallInt(40)].into(),
                )],
                None,
            )
            .expect("resolved-value projection should evaluate"),
            Value::SmallInt(40),
        );
    }

    #[test]
    fn resolved_projection_preserves_parameter_string_token_order() {
        const PARAM: &str = "__ty_projection_token_order_param_20260721";
        const SENTINEL: &str = "__ty_projection_token_order_sentinel_20260721";
        let def = projection_def_with_param(PARAM, PARAM, 1);
        let ctx = EvalCtx::new();

        apply_user_op_with_values_resolved(
            &ctx,
            "Project",
            intern_name("Project"),
            &def,
            &[Value::Tuple(vec![Value::SmallInt(7)].into())],
            None,
        )
        .expect("resolved projection should evaluate");

        let sentinel = crate::value::intern_string(SENTINEL);
        let param = crate::value::intern_string(PARAM);
        assert!(
            crate::value::tlc_string_token(&param) < crate::value::tlc_string_token(&sentinel),
            "the projection shortcut must retain the historical first-miss parameter interning",
        );
    }

    #[test]
    fn unary_int_projection_preserves_function_apply_error_span() {
        let def = projection_def("value", 2);
        let projection = unary_int_projection(&def).expect("projection shape");
        let err =
            apply_unary_int_projection(projection, &Value::Tuple(vec![Value::SmallInt(10)].into()))
                .expect("tuple application is eager")
                .expect_err("index two must be out of bounds");

        assert!(matches!(
            err,
            EvalError::IndexOutOfBounds {
                index: 2,
                len: 1,
                span: Some(span),
                ..
            } if span == def.body.span
        ));
    }

    #[test]
    fn unary_int_projection_rejects_a_different_identifier() {
        let def = projection_def("outer_value", 1);
        assert!(unary_int_projection(&def).is_none());
    }

    #[test]
    fn unary_int_projection_unsupported_value_falls_through_to_generic_error() {
        let def = projection_def("value", 1);
        let ctx = EvalCtx::new();
        let err = apply_user_op_with_values_resolved(
            &ctx,
            "Project",
            intern_name("Project"),
            &def,
            &[Value::SmallInt(7)],
            None,
        )
        .expect_err("an integer is not function-like");

        assert!(matches!(
            err,
            EvalError::TypeError {
                span: Some(span),
                ..
            } if span == Span::new(FileId(17), 20, 21)
        ));
    }

    #[test]
    fn unary_empty_tuple_int_projection_covers_both_call_paths_and_branches() {
        let def = empty_tuple_projection_def(7, 1);
        let ctx = EvalCtx::new();

        assert_eq!(
            apply_user_op_with_exprs(
                &ctx,
                "EmptyDefaultProject",
                intern_name("EmptyDefaultProject"),
                &def,
                &[Spanned::dummy(Expr::Tuple(vec![]))],
                None,
            )
            .expect("AST empty branch should evaluate"),
            Value::SmallInt(7),
        );
        assert_eq!(
            apply_user_op_with_exprs(
                &ctx,
                "EmptyDefaultProject",
                intern_name("EmptyDefaultProject"),
                &def,
                &[Spanned::dummy(Expr::Tuple(vec![Spanned::dummy(
                    Expr::Int(11.into()),
                )]))],
                None,
            )
            .expect("AST projection branch should evaluate"),
            Value::SmallInt(11),
        );
        assert_eq!(
            apply_user_op_with_values_resolved(
                &ctx,
                "EmptyDefaultProject",
                intern_name("EmptyDefaultProject"),
                &def,
                &[Value::seq([])],
                None,
            )
            .expect("resolved empty branch should evaluate"),
            Value::SmallInt(7),
        );
        assert_eq!(
            apply_user_op_with_values_resolved(
                &ctx,
                "EmptyDefaultProject",
                intern_name("EmptyDefaultProject"),
                &def,
                &[Value::seq([Value::SmallInt(13)])],
                None,
            )
            .expect("resolved projection branch should evaluate"),
            Value::SmallInt(13),
        );
    }

    #[test]
    fn unary_empty_tuple_int_projection_preserves_extensional_empty_equality() {
        let def = empty_tuple_projection_def(9, 2);
        let projection = unary_empty_tuple_int_projection(&def).expect("guarded projection shape");
        let empty_values = [
            Value::tuple([]),
            Value::seq([]),
            // Pre-existing upstream breakage repaired during the churn merge:
            // the Rp migration left these two constructors as `Arc::new`,
            // which fails to compile the tla-eval test target.
            Value::Func(Rp::new(FuncValue::from_sorted_entries(Vec::new()))),
            Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 0, Vec::new()))),
            Value::record(std::iter::empty::<(&'static str, Value)>()),
            Value::Bag(BagValue::empty_arc()),
        ];

        for value in empty_values {
            assert_eq!(
                apply_unary_empty_tuple_int_projection(projection, &value)
                    .expect("empty function representation is handled eagerly")
                    .expect("empty branch should evaluate"),
                Value::SmallInt(9),
                "{value:?}",
            );
        }
    }

    #[test]
    fn unary_empty_tuple_int_projection_preserves_else_error_spans() {
        let def = empty_tuple_projection_def(0, 2);
        let projection = unary_empty_tuple_int_projection(&def).expect("guarded projection shape");
        let short_tuple_error =
            apply_unary_empty_tuple_int_projection(projection, &Value::tuple([Value::SmallInt(1)]))
                .expect("tuple application is eager")
                .expect_err("index two must be out of bounds");
        assert!(matches!(
            short_tuple_error,
            EvalError::IndexOutOfBounds {
                span: Some(span),
                ..
            } if span == Span::new(FileId(19), 41, 45)
        ));

        let record_error = apply_unary_empty_tuple_int_projection(
            projection,
            &Value::record([("two", Value::SmallInt(2))]),
        )
        .expect("record application is eager")
        .expect_err("integer index is invalid for a record");
        assert!(matches!(
            record_error,
            EvalError::TypeError {
                span: Some(span),
                ..
            } if span == Span::new(FileId(19), 43, 44)
        ));
    }

    #[test]
    fn unary_empty_tuple_int_projection_rejects_near_misses_and_large_literals() {
        assert!(
            unary_empty_tuple_int_projection(&empty_tuple_projection_def_with_param(
                "value", "outer", "value", 0, 2
            ))
            .is_none()
        );
        assert!(
            unary_empty_tuple_int_projection(&empty_tuple_projection_def_with_param(
                "value", "value", "outer", 0, 2
            ))
            .is_none()
        );

        let mut large_default = (*empty_tuple_projection_def(0, 2)).clone();
        let Expr::If(_, then_branch, _) = &mut large_default.body.node else {
            unreachable!("test helper creates IF")
        };
        then_branch.node = Expr::Int((i128::MAX).into());
        assert!(unary_empty_tuple_int_projection(&large_default).is_none());

        let mut large_index = (*empty_tuple_projection_def(0, 2)).clone();
        let Expr::If(_, _, else_branch) = &mut large_index.body.node else {
            unreachable!("test helper creates IF")
        };
        let Expr::FuncApply(_, index_expr) = &mut else_branch.node else {
            unreachable!("test helper creates function application")
        };
        index_expr.node = Expr::Int((i128::MAX).into());
        assert!(unary_empty_tuple_int_projection(&large_index).is_none());
    }

    #[test]
    fn unary_empty_tuple_int_projection_unsupported_value_falls_back() {
        let def = empty_tuple_projection_def(0, 2);
        let projection = unary_empty_tuple_int_projection(&def).expect("guarded projection shape");
        assert!(apply_unary_empty_tuple_int_projection(projection, &Value::SmallInt(7)).is_none());

        let err = apply_user_op_with_values_resolved(
            &EvalCtx::new(),
            "EmptyDefaultProject",
            intern_name("EmptyDefaultProject"),
            &def,
            &[Value::SmallInt(7)],
            None,
        )
        .expect_err("generic path should report scalar function application");
        assert!(matches!(
            err,
            EvalError::TypeError {
                span: Some(span),
                ..
            } if span == Span::new(FileId(19), 41, 42)
        ));
    }

    #[test]
    fn resolved_empty_tuple_projection_preserves_parameter_string_token_order() {
        const PARAM: &str = "__ty_empty_projection_token_order_param_20260721";
        const SENTINEL: &str = "__ty_empty_projection_token_order_sentinel_20260721";
        let def = empty_tuple_projection_def_with_param(PARAM, PARAM, PARAM, 0, 2);

        apply_user_op_with_values_resolved(
            &EvalCtx::new(),
            "EmptyDefaultProject",
            intern_name("EmptyDefaultProject"),
            &def,
            &[Value::tuple([])],
            None,
        )
        .expect("resolved empty branch should evaluate");

        let sentinel = crate::value::intern_string(SENTINEL);
        let param = crate::value::intern_string(PARAM);
        assert!(
            crate::value::tlc_string_token(&param) < crate::value::tlc_string_token(&sentinel),
            "the guarded projection shortcut must retain first-miss parameter interning",
        );
    }

    #[test]
    fn resolved_operator_name_id_reuses_only_demonstrably_matching_source_id() {
        let source = String::from("DirectCachedOperator");
        let pre_resolved = intern_name(&source);

        assert_eq!(
            resolved_operator_name_id(&source, pre_resolved, &source),
            pre_resolved,
            "the unchanged spelling must reuse its pre-resolved identity",
        );

        let same_spelling_different_storage = source.clone();
        assert!(!std::ptr::eq(
            source.as_str(),
            same_spelling_different_storage.as_str(),
        ));
        assert_eq!(
            resolved_operator_name_id(&source, pre_resolved, &same_spelling_different_storage),
            intern_name(&same_spelling_different_storage),
            "a different returned string must use the resolved spelling's identity",
        );

        assert_eq!(
            resolved_operator_name_id(&source, NameId::INVALID, &source),
            intern_name(&source),
            "unresolved source nodes must retain the interner fallback",
        );

        let replacement = "ReplacementCachedOperator";
        assert_eq!(
            resolved_operator_name_id(&source, pre_resolved, replacement),
            intern_name(replacement),
            "operator replacements must be keyed by the replacement identity",
        );
    }

    #[test]
    fn recursive_memo_preserves_parameter_string_token_order() {
        const PARAM: &str = "__ty_recursive_memo_param_token_20260723";
        const ARGUMENT: &str = "__ty_recursive_memo_argument_token_20260723";

        let def = Arc::new(OperatorDef {
            name: Spanned::dummy("RecursiveIdentity".to_string()),
            params: vec![OpParam {
                name: Spanned::dummy(PARAM.to_string()),
                arity: 0,
            }],
            body: Spanned::dummy(Expr::Ident(PARAM.to_string(), NameId::INVALID)),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: true,
            self_call_count: 0,
        });
        let argument = Spanned::dummy(Expr::String(ARGUMENT.to_string()));

        assert_eq!(
            apply_user_op_with_exprs(
                &EvalCtx::new(),
                "RecursiveIdentity",
                intern_name("RecursiveIdentity"),
                &def,
                &[argument],
                None,
            )
            .expect("recursive identity should evaluate"),
            Value::String(crate::value::intern_string(ARGUMENT)),
        );

        let sentinel = crate::value::intern_string("__ty_recursive_memo_sentinel_20260723");
        let param = crate::value::intern_string(PARAM);
        let argument = crate::value::intern_string(ARGUMENT);
        assert!(
            crate::value::tlc_string_token(&param) < crate::value::tlc_string_token(&argument),
            "recursive memo must intern parameter names before evaluating arguments",
        );
        assert!(
            crate::value::tlc_string_token(&argument) < crate::value::tlc_string_token(&sentinel),
            "the regression must observe first-seen ordering from this call",
        );
    }

    #[test]
    fn recursive_memo_key_discriminates_operator_scopes() {
        let base = EvalCtx::new();
        let def = Arc::new(OperatorDef {
            name: Spanned::dummy("ScopedRecursive".to_string()),
            params: vec![OpParam {
                name: Spanned::dummy("n".to_string()),
                arity: 0,
            }],
            body: Spanned::dummy(Expr::Ident("n".to_string(), NameId::INVALID)),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: true,
            self_call_count: 0,
        });
        let alternate = Arc::new(OperatorDef {
            name: Spanned::dummy("Alternate".to_string()),
            params: Vec::new(),
            body: Spanned::dummy(Expr::Int(1.into())),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        });

        let mut first_ops = tla_core::OpEnv::new();
        first_ops.insert("ScopedRecursive".into(), Arc::clone(&def));
        let first = base.with_local_ops(first_ops);

        let mut second_ops = tla_core::OpEnv::new();
        second_ops.insert("Alternate".into(), alternate);
        let second = base.with_local_ops(second_ops);

        let op_name = intern_name("ScopedRecursive");
        let args = [Value::int(7)];
        let first_key = operator_cache_key(&first, "ScopedRecursive", op_name, &def, &args);
        let second_key = operator_cache_key(&second, "ScopedRecursive", op_name, &def, &args);
        assert_ne!(
            first_key, second_key,
            "one recursive definition reused in distinct local-op scopes must not alias",
        );
    }

    #[test]
    fn nary_hit_finish_propagates_only_state_dependencies() {
        let ctx = EvalCtx::new();
        let mut state_deps = OpEvalDeps::default();
        state_deps.record_state(VarIndex::new(0), &Value::int(7));

        let outer = OpDepGuard::from_ctx(&ctx, 0);
        assert_eq!(
            finish_nary_cache_hit(
                &ctx,
                NaryCacheHit::State(CachedOpResult {
                    value: Value::int(42),
                    deps: state_deps,
                }),
            ),
            Value::int(42)
        );
        let propagated = outer.try_take_deps().expect("outer dep frame");
        assert!(propagated.state.contains_key(&VarIndex::new(0)));

        let outer = OpDepGuard::from_ctx(&ctx, 0);
        assert_eq!(
            finish_nary_cache_hit(&ctx, NaryCacheHit::Persistent(Value::int(99))),
            Value::int(99)
        );
        let propagated = outer.try_take_deps().expect("outer dep frame");
        assert!(
            propagated.state.is_empty()
                && propagated.next.is_empty()
                && propagated.local.is_empty()
                && propagated.tlc_level.is_none()
                && !propagated.instance_lazy_read,
            "persistent hits must not manufacture or propagate dependencies"
        );
    }
}
