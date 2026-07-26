// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Batch evaluation helpers for inline liveness leaf recording during BFS.

use super::boolean_contract::expect_live_bool;
use super::live_expr_eval::LiveExprEvaluator;
use crate::error::EvalResult;
use crate::eval::{BindingChain, Env, EvalCtx};
use crate::state::State;
use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::Spanned;

mod action;
mod fallback;
mod state;

#[cfg(test)]
mod tests;

pub(crate) use action::eval_action_leaves_array;
pub(crate) use state::eval_state_leaves_with_array_successors;

fn build_state_env(ctx: &EvalCtx, state: &State) -> Arc<Env> {
    let mut env = ctx.env().clone();
    for (name, value) in state.vars() {
        // Part of #2144: skip state vars that shadow local bindings.
        if !ctx.has_local_binding(name.as_ref()) {
            env.insert(Arc::clone(name), value.clone());
        }
    }
    Arc::new(env)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InlineLeafEntryMode {
    Boundary,
    Inline,
}

fn eval_bool_expr_with_entry_mode(
    eval_ctx: &EvalCtx,
    expr: &Arc<Spanned<Expr>>,
    entry_mode: InlineLeafEntryMode,
) -> EvalResult<bool> {
    let value = match entry_mode {
        InlineLeafEntryMode::Boundary => crate::eval::eval_entry(eval_ctx, expr)?,
        InlineLeafEntryMode::Inline => crate::eval::eval_entry_inline(eval_ctx, expr)?,
    };
    expect_live_bool(&value, Some(expr.span))
}

fn eval_subscript_changed_uncached(
    ctx: &EvalCtx,
    s1: &State,
    s2: &State,
    subscript: &Arc<Spanned<Expr>>,
) -> EvalResult<bool> {
    // Part of #ewd998-refine-fix: `with_explicit_env` sets `state_env = None`, so
    // s1 and s2 collide at `state_identity = 0` in every per-state cache. Clearing
    // only SUBST_CACHE (the historical Fix #2780) leaks the CHOOSE caches across the
    // two states, so an INSTANCE refinement mapping's `CHOOSE` (e.g. EWD998Chan's
    // `tpos`) computed for s1 is served STALE to s2. Also clear the CHOOSE caches (via
    // `invalidate_state_identity_tracking`), matching the post-BFS
    // `eval_subscript_changed` path; the `eval_choose` root fix keeps the mapping's
    // zero-arg operators from being mis-cached as constants during ENABLED scope.
    let ctx1 = ctx.with_explicit_env((*build_state_env(ctx, s1)).clone());
    crate::eval::clear_subst_cache();
    crate::eval::invalidate_state_identity_tracking();
    let val1 = super::eval_live_entry(&ctx1, subscript)?;

    crate::eval::clear_subst_cache();
    crate::eval::invalidate_state_identity_tracking();

    let ctx2 = ctx.with_explicit_env((*build_state_env(ctx, s2)).clone());
    let val2 = super::eval_live_entry(&ctx2, subscript)?;

    Ok(val1 != val2)
}

struct CachedSuccessorEvaluator<'a> {
    cached_successors: &'a [State],
}

impl LiveExprEvaluator for CachedSuccessorEvaluator<'_> {
    fn eval_subscript_changed(
        &self,
        ctx: &EvalCtx,
        current: &State,
        next: &State,
        sub_expr: &Arc<Spanned<Expr>>,
        _tag: u32,
    ) -> EvalResult<bool> {
        eval_subscript_changed_uncached(ctx, current, next, sub_expr)
    }

    fn eval_enabled(
        &mut self,
        ctx: &EvalCtx,
        current_state: &State,
        action: &Arc<Spanned<Expr>>,
        bindings: &Option<BindingChain>,
        require_state_change: bool,
        subscript: &Option<Arc<Spanned<Expr>>>,
        tag: u32,
    ) -> EvalResult<bool> {
        super::checker::eval_enabled_cached(ctx, current_state.fingerprint(), tag, || {
            super::enabled_eval::eval_enabled_uncached(
                super::enabled_eval::EnabledEvalRequest {
                    ctx_current: ctx,
                    current_state,
                    action,
                    bindings: bindings.as_ref(),
                    require_state_change,
                    subscript: subscript.as_ref(),
                    cached_successors: self.cached_successors,
                    pred_cache_tag: None,
                    state_prepared: false,
                },
                |eval_ctx, s1, s2, sub_expr| {
                    eval_subscript_changed_uncached(eval_ctx, s1, s2, sub_expr)
                },
            )
        })
    }
}
