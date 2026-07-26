// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! UNCHANGED expression evaluation.
//!
//! `UNCHANGED <<x, y>>` is equivalent to `x' = x /\ y' = y`.
//! Evaluates the inner expression in both current and next-state environments
//! and returns true iff the values are equal.
//!
//! Extracted from `eval_prime.rs` as part of #3426.

#[cfg(debug_assertions)]
use super::debug_unchanged;
use super::{
    eval, record_next_read, record_state_read, values_equal, EvalCtx, EvalError, EvalResult, Expr,
    Span, Spanned, StateLookupMode, Value,
};
use crate::cache::{bump_hoist_state_generation_ctx, StateLookupModeGuard};
use crate::value::intern_string;
use tla_core::name_intern::intern_name;

// A/B kill switch for the narrow shadow check below. When set, any local
// binding restores the historical direct-slot-path rejection.
feature_flag!(
    no_unchanged_unrelated_bindings,
    "TY_NO_UNCHANGED_UNRELATED_BINDINGS"
);

/// Direct slot comparison for UNCHANGED on a single state variable.
///
/// Part of #3073: When both state_env and next_state_env are array-backed and no
/// binding overlays are active, compare the current and next-state slots directly
/// without eval dispatch, context cloning, or Value cloning. This eliminates ~2
/// eval calls + 1 ctx clone + 2 Value clones per UNCHANGED variable.
///
/// Returns `None` when the fast path is not applicable (overlays active, non-StateVar
/// inner expression), so the caller falls through to the general evaluation path.
#[inline]
fn try_unchanged_statevar_fast(
    ctx: &EvalCtx,
    inner: &Spanned<Expr>,
    span: Option<Span>,
) -> Option<EvalResult<Value>> {
    let state_env = ctx.state_env?;
    let next_state_env = ctx.next_state_env?;

    // Substitution overlays can redirect a syntactic state-variable reference.
    // Ordinary local bindings only matter when they shadow the particular
    // variable being compared; that narrower check is performed per element
    // below. This keeps parameterized actions such as `Action(self)` on the
    // direct-slot path for `UNCHANGED <<x, y>>` when only `self` is bound.
    if (no_unchanged_unrelated_bindings() && !ctx.bindings.is_empty())
        || ctx.instance_substitutions().is_some()
        || ctx.eager_subst_bindings.is_some()
        || ctx.call_by_name_subs().is_some()
    {
        return None;
    }

    match &inner.node {
        Expr::StateVar(name, idx, name_id) => {
            if ctx.has_local_binding(name) {
                return None;
            }
            let resolved_idx = ctx.resolve_state_var_slot(name, *idx, *name_id);
            let idx = resolved_idx.as_usize();
            if !crate::is_dep_tracking_active(ctx) {
                // Part of #3579: Compare compact slots directly when dep tracking is
                // inactive so UNCHANGED avoids rebuilding owned Values on the hot path.
                let eq = unsafe { state_env.values_equal(next_state_env, idx) };
                return Some(Ok(Value::Bool(eq)));
            }
            // SAFETY: resolved_idx is validated against the current VarRegistry.
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::StateVarReadUnchanged,
            );
            let cur = unsafe { state_env.get_value(idx) };
            tla_value::churn_stats::churn_count(
                tla_value::churn_stats::ChurnSite::StateVarReadUnchanged,
            );
            let next = unsafe { next_state_env.get_value(idx) };
            // Record deps for zero-arg operator cache correctness.
            record_state_read(ctx, resolved_idx, &cur);
            record_next_read(ctx, resolved_idx, &next);
            Some(if !cur.is_set() && !next.is_set() {
                Ok(Value::Bool(cur == next))
            } else {
                values_equal(ctx, &cur, &next, span).map(Value::Bool)
            })
        }
        Expr::Tuple(elems) => {
            // UNCHANGED <<x, y, z>> where all elements are state variables:
            // compare each pair directly, short-circuit on first mismatch.
            for elem in elems {
                let Expr::StateVar(name, idx, name_id) = &elem.node else {
                    return None;
                };
                if ctx.has_local_binding(name) {
                    return None;
                }
                let resolved_idx = ctx.resolve_state_var_slot(name, *idx, *name_id);
                let idx = resolved_idx.as_usize();
                if !crate::is_dep_tracking_active(ctx) {
                    let eq = unsafe { state_env.values_equal(next_state_env, idx) };
                    if !eq {
                        return Some(Ok(Value::Bool(false)));
                    }
                    continue;
                }
                tla_value::churn_stats::churn_count(
                    tla_value::churn_stats::ChurnSite::StateVarReadUnchanged,
                );
                let cur = unsafe { state_env.get_value(idx) };
                tla_value::churn_stats::churn_count(
                    tla_value::churn_stats::ChurnSite::StateVarReadUnchanged,
                );
                let next = unsafe { next_state_env.get_value(idx) };
                record_state_read(ctx, resolved_idx, &cur);
                record_next_read(ctx, resolved_idx, &next);
                let eq = if !cur.is_set() && !next.is_set() {
                    cur == next
                } else {
                    match values_equal(ctx, &cur, &next, span) {
                        Ok(eq) => eq,
                        Err(e) => return Some(Err(e)),
                    }
                };
                if !eq {
                    return Some(Ok(Value::Bool(false)));
                }
            }
            Some(Ok(Value::Bool(true)))
        }
        _ => None,
    }
}

/// Evaluate an UNCHANGED expression (`Expr::Unchanged`).
///
/// `UNCHANGED <<x, y>>` is equivalent to `x' = x /\ y' = y`.
/// Evaluates the inner expression in both current and next-state environments
/// and returns true iff the values are equal.
///
/// Uses the same multi-tier strategy as `eval_prime`:
/// 1. Direct slot comparison for StateVar / Tuple-of-StateVar (fast path)
/// 2. Array-backed next-state when `next_state_env` is set
/// 3. HashMap-based with full/partial next_state fallback
pub(super) fn eval_unchanged(
    ctx: &EvalCtx,
    inner: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    // Part of #3073: Direct slot comparison fast path. Avoids eval dispatch,
    // context cloning, and Value cloning for the common BFS case where inner
    // is a StateVar or Tuple of StateVars with no binding overlays.
    if let Some(result) = try_unchanged_statevar_fast(ctx, inner, span) {
        return result;
    }

    // Evaluate `inner` in the current-state environment.
    let cur_v = eval(ctx, inner)?;
    // Prefer array-backed next-state binding when available.
    if let Some(next_state_env) = ctx.next_state_env {
        let mut next_ctx = ctx.clone();
        let s = next_ctx.stable_mut();
        s.next_state = None;
        s.next_state_env = None;
        s.state_env = Some(next_state_env);
        let _ = s;
        let next_v = {
            let _guard = StateLookupModeGuard::new(ctx, StateLookupMode::Next);
            let _gen_guard = bump_hoist_state_generation_ctx(ctx);
            eval(&next_ctx, inner)?
        };
        let eq = values_equal(ctx, &cur_v, &next_v, span)?;
        debug_eprintln!(
            debug_unchanged(),
            "[UNCHANGED] array path: cur={:?}, next={:?}, eq={}",
            cur_v,
            next_v,
            eq
        );
        return Ok(Value::Bool(eq));
    }

    let Some(next_state) = &ctx.next_state else {
        return Err(EvalError::UnchangedNotEvaluable { span });
    };

    // Evaluate `inner` in the next-state environment by rebinding all
    // next-state variables to their primed values as unprimed names.
    let mut next_ctx = ctx.clone();
    if next_ctx.state_env.is_some() {
        let is_full_next_state = next_state.len() == ctx.var_registry().len();
        debug_eprintln!(
            debug_unchanged(),
            "[UNCHANGED] hashmap path: state_env=Some, is_full_next_state={}, next_state.len()={}, var_registry.len()={}",
            is_full_next_state,
            next_state.len(),
            ctx.var_registry().len()
        );
        if is_full_next_state {
            next_ctx.stable_mut().state_env = None;
            // Part of #3964: Hoist Arc::make_mut outside loop.
            next_ctx.batch_insert_into_env(next_state.iter());
        } else {
            for (name, value) in next_state.iter() {
                if !next_ctx.has_local_binding(name.as_ref()) {
                    // Part of #188, #232: Intern for pointer equality and NameId for O(1) lookups
                    let interned = intern_string(name.as_ref());
                    let name_id = intern_name(interned.as_ref());
                    next_ctx.push_binding_preinterned(interned, value.clone(), name_id);
                }
            }
        }
    } else {
        debug_eprintln!(
            debug_unchanged(),
            "[UNCHANGED] hashmap path: state_env=None"
        );
        // Part of #3964: Hoist Arc::make_mut outside loop.
        next_ctx.batch_insert_into_env(next_state.iter());
    }
    let next_v = {
        let _guard = StateLookupModeGuard::new(ctx, StateLookupMode::Next);
        let _gen_guard = bump_hoist_state_generation_ctx(ctx);
        eval(&next_ctx, inner)?
    };

    let eq = values_equal(ctx, &cur_v, &next_v, span)?;
    debug_eprintln!(
        debug_unchanged(),
        "[UNCHANGED] hashmap result: cur={:?}, next={:?}, eq={}",
        cur_v,
        next_v,
        eq
    );
    Ok(Value::Bool(eq))
}

#[cfg(test)]
mod fast_path_tests {
    use super::*;

    fn state_var(name: &str) -> Spanned<Expr> {
        Spanned::dummy(Expr::StateVar(name.to_string(), 0, intern_name(name)))
    }

    #[test]
    fn unrelated_local_binding_keeps_direct_slot_path() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("x");
        let current = [Value::int(1)];
        ctx.bind_state_array(&current);
        let mut ctx = ctx.into_bind_local("self", Value::int(7));
        let next = [Value::int(2)];
        let _guard = ctx.bind_next_state_array_guard(&next);

        let result = try_unchanged_statevar_fast(&ctx, &state_var("x"), None)
            .expect("unrelated local must not disable the direct-slot path")
            .expect("direct-slot comparison must evaluate");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn same_name_local_binding_disables_direct_slot_path() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("x");
        let current = [Value::int(1)];
        ctx.bind_state_array(&current);
        let mut ctx = ctx.into_bind_local("x", Value::int(7));
        let next = [Value::int(2)];
        let _guard = ctx.bind_next_state_array_guard(&next);

        assert!(try_unchanged_statevar_fast(&ctx, &state_var("x"), None).is_none());
    }
}
