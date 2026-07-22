// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ENABLED evaluation helpers for the EA precompute pass (#2572, #2364).
//!
//! Extracted from `ea_precompute.rs` to stay under the 500-line file limit.
//! Contains the `EnabledInfo` struct, tree-walking collectors, and the
//! array-based + fallback ENABLED evaluation methods used by
//! `populate_node_check_masks`.

use super::live_expr::LiveExpr;
use super::LivenessChecker;
use crate::state::Fingerprint;

/// Information about an ENABLED sub-expression extracted from check expressions.
///
/// Used by `populate_node_check_masks` and the `eval_enabled_for_node`/
/// `eval_enabled_fallback` helper methods.
pub(super) struct EnabledInfo {
    pub(super) action: std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>,
    pub(super) bindings: Option<crate::eval::BindingChain>,
    pub(super) require_state_change: bool,
    pub(super) subscript: Option<std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>>,
    pub(super) tag: u32,
}

/// Walk a `LiveExpr` tree and collect all `ENABLED` sub-expressions.
pub(super) fn collect_enabled_nodes(expr: &LiveExpr, out: &mut Vec<EnabledInfo>) {
    match expr {
        LiveExpr::Enabled {
            action,
            bindings,
            require_state_change,
            subscript,
            tag,
        } => {
            out.push(EnabledInfo {
                action: std::sync::Arc::clone(action),
                bindings: bindings.clone(),
                require_state_change: *require_state_change,
                subscript: subscript.clone(),
                tag: *tag,
            });
        }
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            for p in parts {
                collect_enabled_nodes(p, out);
            }
        }
        LiveExpr::Not(inner) => collect_enabled_nodes(inner, out),
        _ => {}
    }
}

impl LivenessChecker {
    /// Dispatch ENABLED evaluation for a node, trying full-state then fingerprint paths.
    ///
    /// Unified entry point used by `populate_node_check_masks` to avoid duplicating
    /// the state-successors vs fingerprint-successors dispatch logic inline.
    pub(super) fn eval_enabled_for_node(
        &mut self,
        info: &EnabledInfo,
        from_state: &crate::state::State,
        from_fp: Fingerprint,
        registry: &tla_core::VarRegistry,
    ) -> crate::error::EvalResult<bool> {
        if self.graph.has_owned_state_cache() {
            let succ_fps = self
                .state_successor_fps
                .get(&from_fp)
                .cloned()
                .ok_or_else(|| {
                    Self::behavior_graph_invariant_error(format!(
                        "owned compact cache is missing successor adjacency for ENABLED source {from_fp}"
                    ))
                })?;
            self.eval_enabled_array_fast_from_fps(info, from_state, from_fp, &succ_fps, registry)
        } else if let Some(succs) = self.state_successors.get(&from_fp).cloned() {
            self.eval_enabled_array_fast_lazy_envs(info, from_state, from_fp, &succs, registry)
        } else if let Some(succ_fps) = self.state_successor_fps.get(&from_fp).cloned() {
            self.eval_enabled_array_fast_from_fps(info, from_state, from_fp, &succ_fps, registry)
        } else {
            Ok(false)
        }
    }

    fn eval_enabled_array_fast_lazy_envs(
        &mut self,
        info: &EnabledInfo,
        from_state: &crate::state::State,
        from_fp: Fingerprint,
        cached_succs: &[crate::state::State],
        registry: &tla_core::VarRegistry,
    ) -> crate::error::EvalResult<bool> {
        for succ in cached_succs {
            let succ_fp = succ.fingerprint();

            if info.require_state_change {
                if info.subscript.is_some() {
                    match super::subscript_cache::check_subscript_changed_cached(
                        from_fp, succ_fp, info.tag,
                    ) {
                        Some(false) => continue,
                        Some(true) => {}
                        None => {
                            return Self::eval_enabled_fallback(
                                &self.ctx,
                                info,
                                from_state,
                                cached_succs,
                            );
                        }
                    }
                } else if succ_fp == from_fp {
                    continue;
                }
            }

            let next_values = succ.to_values(registry);
            let cached_env = self.get_cached_env(succ);
            let prev_next = self.ctx.next_state_mut().take();
            let _next_guard = self.ctx.take_next_state_env_guard();
            *self.ctx.next_state_mut() = Some(cached_env);
            self.ctx.bind_next_state_array(&next_values);

            let eval_ctx = match info.bindings {
                Some(ref chain) => self.ctx.with_liveness_bindings(chain),
                None => self.ctx.clone(),
            };
            match crate::eval::eval_entry(&eval_ctx, &info.action) {
                Ok(value) => {
                    *self.ctx.next_state_mut() = prev_next;
                    if crate::liveness::boolean_contract::expect_live_bool(
                        &value,
                        Some(info.action.span),
                    )? {
                        return Ok(true);
                    }
                }
                Err(e) if crate::enumerate::is_disabled_action_error(&e) => {
                    *self.ctx.next_state_mut() = prev_next;
                }
                Err(e) => {
                    *self.ctx.next_state_mut() = prev_next;
                    return Err(e);
                }
            }
        }

        Ok(false)
    }

    /// Evaluate ENABLED using successor fingerprints backed by the shared graph cache.
    ///
    /// This preserves the fingerprint-only direct graph path from #3065 by
    /// resolving successor states lazily instead of rebuilding `Vec<State>`
    /// after exploration.
    pub(super) fn eval_enabled_array_fast_from_fps(
        &mut self,
        info: &EnabledInfo,
        from_state: &crate::state::State,
        from_fp: Fingerprint,
        cached_succ_fps: &[Fingerprint],
        registry: &tla_core::VarRegistry,
    ) -> crate::error::EvalResult<bool> {
        for &succ_fp in cached_succ_fps {
            if info.require_state_change {
                if info.subscript.is_some() {
                    match super::subscript_cache::check_subscript_changed_cached(
                        from_fp, succ_fp, info.tag,
                    ) {
                        Some(false) => continue,
                        Some(true) => {}
                        None => {
                            let successors = self.successor_states_for_enabled(from_fp)?;
                            return Self::eval_enabled_fallback(
                                &self.ctx,
                                info,
                                from_state,
                                &successors,
                            );
                        }
                    }
                } else if succ_fp == from_fp {
                    if self.succ_witnesses.is_some() {
                        let successors = self.successor_states_for_enabled(from_fp)?;
                        return Self::eval_enabled_fallback(
                            &self.ctx,
                            info,
                            from_state,
                            &successors,
                        );
                    }
                    continue;
                }
            }

            // Part of #3746: When the graph's state cache doesn't contain the
            // successor state data, fall back to the full-state ENABLED evaluation
            // path which resolves successors via successor_states_for_enabled()
            // (filter_map over get_state_by_fp — gracefully skips missing fps).
            // This can happen in parallel mode when populate_state_successor_fps_from_graph
            // records successor fingerprints whose concrete state data is not available
            // in the shared state cache.
            let next_values = {
                let Some(succ) = self.graph.get_state_by_fp(succ_fp) else {
                    if self.graph.has_owned_state_cache() {
                        return Err(Self::behavior_graph_invariant_error(format!(
                            "owned compact cache is missing ENABLED successor payload {succ_fp} for source {from_fp}"
                        )));
                    }
                    let successors = self.successor_states_for_enabled(from_fp)?;
                    return Self::eval_enabled_fallback(&self.ctx, info, from_state, &successors);
                };
                succ.to_values(registry)
            };

            let prev_next = self.ctx.next_state_mut().take();
            let _next_guard = self.ctx.take_next_state_env_guard();
            let cached_env = match self.get_cached_env_by_fp(succ_fp) {
                Ok(env) => env,
                Err(_) => {
                    // Same fallback: missing state data for env construction.
                    if self.graph.has_owned_state_cache() {
                        return Err(Self::behavior_graph_invariant_error(format!(
                            "owned compact cache could not construct ENABLED successor environment for {succ_fp}"
                        )));
                    }
                    let successors = self.successor_states_for_enabled(from_fp)?;
                    return Self::eval_enabled_fallback(&self.ctx, info, from_state, &successors);
                }
            };
            *self.ctx.next_state_mut() = Some(cached_env);
            self.ctx.bind_next_state_array(&next_values);

            let eval_ctx = match info.bindings {
                Some(ref chain) => self.ctx.with_liveness_bindings(chain),
                None => self.ctx.clone(),
            };
            match crate::eval::eval_entry(&eval_ctx, &info.action) {
                Ok(value) => {
                    *self.ctx.next_state_mut() = prev_next;
                    if crate::liveness::boolean_contract::expect_live_bool(
                        &value,
                        Some(info.action.span),
                    )? {
                        return Ok(true);
                    }
                }
                Err(e) if crate::enumerate::is_disabled_action_error(&e) => {
                    *self.ctx.next_state_mut() = prev_next;
                }
                Err(e) => {
                    *self.ctx.next_state_mut() = prev_next;
                    return Err(e);
                }
            }
        }

        Ok(false)
    }

    /// Fallback ENABLED evaluation using HashMap-based state binding.
    ///
    /// Used when the array-based fast path cannot determine the result. This
    /// delegates to `eval_enabled_uncached` which clones the EvalCtx and uses
    /// HashMap-based state binding. Note: this invalidates SUBST_CACHE entries
    /// from the caller's array binding epoch.
    pub(super) fn eval_enabled_fallback(
        ctx: &crate::eval::EvalCtx,
        info: &EnabledInfo,
        from_state: &crate::state::State,
        cached_succs: &[crate::state::State],
    ) -> crate::error::EvalResult<bool> {
        // Part of #2895: Apply liveness bindings via BindingChain.
        let eval_ctx = match info.bindings {
            Some(ref chain) => ctx.with_liveness_bindings(chain),
            None => ctx.clone(),
        };
        super::super::enabled_eval::eval_enabled_uncached(
            super::super::enabled_eval::EnabledEvalRequest {
                ctx_current: &eval_ctx,
                current_state: from_state,
                action: &info.action,
                bindings: info.bindings.as_ref(),
                require_state_change: info.require_state_change,
                subscript: info.subscript.as_ref(),
                cached_successors: cached_succs,
                pred_cache_tag: None,
                state_prepared: false,
            },
            |eval_ctx, s1, s2, sub_expr| {
                // Direct subscript evaluation for fallback (no cache dependency).
                // Fix #2780, Part of #3458: Clear eval-scope caches before evaluating
                // val1 via with_explicit_env (state_env=None, pointer 0). A prior closure
                // invocation's val2 evaluation may have left entries keyed on the same
                // pointer 0, causing stale hits. Upgraded from clear_subst_cache() to
                // clear_for_eval_scope_boundary() to also clear zero-arg, nary, and
                // INSTANCE-scoped LET caches that could be stale across re-binding.
                crate::eval::clear_for_eval_scope_boundary();
                let mut env1 = eval_ctx.env().clone();
                for (name, value) in s1.vars() {
                    // Part of #2144: skip state vars that shadow local bindings.
                    if !eval_ctx.has_local_binding(name.as_ref()) {
                        env1.insert(std::sync::Arc::clone(name), value.clone());
                    }
                }
                let ctx1 = eval_ctx.with_explicit_env(env1);
                let v1 = crate::eval::eval_entry(&ctx1, sub_expr)?;
                crate::eval::clear_for_eval_scope_boundary();
                let mut env2 = eval_ctx.env().clone();
                for (name, value) in s2.vars() {
                    // Part of #2144: skip state vars that shadow local bindings.
                    if !eval_ctx.has_local_binding(name.as_ref()) {
                        env2.insert(std::sync::Arc::clone(name), value.clone());
                    }
                }
                let ctx2 = eval_ctx.with_explicit_env(env2);
                let v2 = crate::eval::eval_entry(&ctx2, sub_expr)?;
                Ok(v1 != v2)
            },
        )
    }
}
