// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-node check-mask precompute for grouped liveness checking (#2572).
//!
//! Part of #3174: Bitmask-only mode. With tags < 64 enforced, all liveness
//! results are captured in inline bitmask maps during BFS recording. The
//! per-tag cross-property HashMap caches have been removed.
//!
//! Evaluates each used check once across the graph, storing results in
//! `NodeInfo.state_check_mask` / `NodeInfo.action_check_masks` for O(1) SCC lookups.

use super::check_mask::{ActionCheckMatrix, CheckMask};
use super::ea_precompute_cache;
use super::ea_precompute_enabled::{collect_enabled_nodes, EnabledInfo};
use super::ea_precompute_leaf_batch::{
    try_populate_action_masks_from_leaf_batches, ActionLeafBatchPlan,
};
use super::ea_precompute_profile::PopulateMasksProfile;
use super::live_expr::LiveExpr;
use super::{BehaviorGraph, BehaviorGraphNode, InlineCheckResults, LivenessChecker, TirProgram};
use crate::liveness::debug::liveness_profile;
use crate::state::Fingerprint;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

fn unique_state_error(context: &'static str, fp: Fingerprint) -> crate::error::EvalError {
    crate::error::EvalError::Internal {
        message: format!("populate_node_check_masks: missing {context} state for fp {fp}"),
        span: None,
    }
}

fn get_unique_state<'a>(
    unique_states: &'a FxHashMap<Fingerprint, crate::state::ArrayState>,
    fp: Fingerprint,
    context: &'static str,
) -> Result<&'a crate::state::ArrayState, crate::error::EvalError> {
    unique_states
        .get(&fp)
        .ok_or_else(|| unique_state_error(context, fp))
}

/// Part of #3746: Returns `Ok(false)` (instead of an error) when the state is
/// missing from the graph's state cache.  This can happen in parallel/
/// fingerprint-only mode where successor fingerprints are recorded but the
/// concrete state data is not available in the shared state cache.  Callers
/// must tolerate missing states by skipping the affected nodes.
///
/// When `TY_STRICT_LIVENESS=1`, returns an error instead of `Ok(false)` so
/// that missing states surface as hard failures for debugging.
fn cache_graph_state(
    unique_states: &mut FxHashMap<Fingerprint, crate::state::ArrayState>,
    graph: &BehaviorGraph,
    node: &BehaviorGraphNode,
    context: &'static str,
    registry: &tla_core::VarRegistry,
) -> Result<bool, crate::error::EvalError> {
    if let std::collections::hash_map::Entry::Vacant(entry) = unique_states.entry(node.state_fp) {
        // Store the compact ArrayState (O(1) Arc clone), NOT a reconstructed
        // im::OrdMap State — holding ~N reconstructed States here is what
        // regressed nbacg 224→669MB. Hot leaf eval binds the ArrayState.
        // fp-only path: shared cache holds ArrayState. Full-state path: the
        // graph's state_cache holds State → convert once (from_state).
        let arr = graph.get_array_state_by_fp(node.state_fp).or_else(|| {
            graph
                .get_state(node)
                .map(|s| crate::state::ArrayState::from_state(&s, registry))
        });
        if let Some(arr) = arr {
            entry.insert(arr);
        } else {
            if graph.has_owned_state_cache() || crate::liveness::debug::strict_liveness() {
                return Err(unique_state_error(context, node.state_fp));
            }
            return Ok(false);
        }
    }
    Ok(true)
}

/// Part of #3746: Optional state lookup that returns None for missing states
/// instead of an error.  Used for successor states that may be absent from
/// the shared state cache in parallel/fingerprint-only mode.
fn try_get_unique_state<'a>(
    unique_states: &'a FxHashMap<Fingerprint, crate::state::ArrayState>,
    fp: Fingerprint,
) -> Option<&'a crate::state::ArrayState> {
    unique_states.get(&fp)
}

fn resolve_state(
    arr: &crate::state::ArrayState,
    fp: Fingerprint,
    unique_states_full: &FxHashMap<Fingerprint, crate::state::State>,
    registry: &tla_core::VarRegistry,
) -> crate::state::State {
    match unique_states_full.get(&fp) {
        Some(state) => state.clone(),
        None => arr.to_state(registry),
    }
}

impl LivenessChecker {
    /// Populate per-node check bitmasks in the behavior graph (#2572, #3065).
    ///
    /// Part of #3174: Simplified — no per-tag cross-property caches. Uses
    /// bitmask fast path when structurally cached, falls back to direct
    /// eval otherwise.
    #[cfg(test)]
    pub(super) fn populate_node_check_masks(
        &mut self,
        check_action: &[LiveExpr],
        check_state: &[LiveExpr],
        action_used: &[bool],
        state_used: &[bool],
        max_fairness_tag: u32,
    ) -> Result<(), crate::error::EvalError> {
        self.populate_node_check_masks_with_inline_cache(
            check_action,
            check_state,
            action_used,
            state_used,
            max_fairness_tag,
            None,
            None,
        )
    }

    /// Part of #3174: Bitmask-only mode. Per-tag cross-property caches removed.
    pub(super) fn populate_node_check_masks_with_inline_cache(
        &mut self,
        check_action: &[LiveExpr],
        check_state: &[LiveExpr],
        action_used: &[bool],
        state_used: &[bool],
        max_fairness_tag: u32,
        inline_results: Option<InlineCheckResults<'_>>,
        tir: Option<&TirProgram<'_>>,
    ) -> Result<(), crate::error::EvalError> {
        // Clear subscript value cache from any previous property group.
        super::subscript_cache::clear_subscript_value_cache();

        let profile = liveness_profile();

        // Part of #3065, #3100, #3174: Fast reconstruction path. When all used
        // check expressions have only fairness-tagged leaves or property-scoped
        // inline tags, skip the expensive O(V) State clone pass and reconstruct
        // masks purely from cached bitmask results.
        if Self::all_checks_structurally_cached(
            check_state,
            check_action,
            state_used,
            action_used,
            max_fairness_tag,
            inline_results,
        ) {
            if profile {
                eprintln!(
                    "    populate_masks: FAST PATH — all checks reconstructable from inline bitmask cache",
                );
            }
            return self.populate_node_check_masks_from_cache(
                check_action,
                check_state,
                action_used,
                state_used,
                max_fairness_tag,
                inline_results,
            );
        }

        // Exact raw on-the-fly exploration (or cross-group cache reuse) already
        // populated every ENABLED result needed by the supported fairness
        // forms. Reconstruct masks directly into graph nodes before allocating
        // the fallback's state/transition maps.
        if super::ea_precompute_exact_raw::try_populate_action_masks_from_exact_raw_fps(
            self,
            check_action,
            check_state,
            action_used,
            state_used,
        )? {
            return Ok(());
        }

        // Eval fallback: snapshot topology + clone each unique state once.
        // This path is only reached for symmetry/VIEW specs where inline
        // recording doesn't cover all tags.
        struct NodeData {
            node: BehaviorGraphNode,
        }

        let mut pf = PopulateMasksProfile::new();

        // Registry to reconstruct `State` transiently (one at a time, dropped)
        // at eval sites on this fallback path; `unique_states` holds compact
        // ArrayStates so no per-node `State` is retained.
        let registry = self.ctx.var_registry().clone();

        let mut unique_states: FxHashMap<Fingerprint, crate::state::ArrayState> =
            FxHashMap::default();
        // Part of #3746: skip nodes whose state is missing from the graph's
        // shared state cache (can happen in parallel/fingerprint-only mode).
        // Missing-state nodes get empty bitmasks, which is conservative:
        // they will never satisfy AE/EA constraints.
        let mut node_data: Vec<NodeData> = Vec::new();
        let mut fp_to_mask: FxHashMap<(Fingerprint, Fingerprint), CheckMask> = FxHashMap::default();
        let mut seen_transitions: FxHashSet<(Fingerprint, Fingerprint)> = FxHashSet::default();
        let mut by_from: FxHashMap<Fingerprint, Vec<Fingerprint>> = FxHashMap::default();

        for node in self.graph.node_keys() {
            let info = match self.graph.try_get_node_info(&node)? {
                Some(info) => info,
                None => continue,
            };
            match cache_graph_state(
                &mut unique_states,
                &self.graph,
                &node,
                "graph node",
                &registry,
            ) {
                Ok(false) => continue, // state missing — skip this node
                Err(e) => return Err(e),
                Ok(true) => {}
            }
            by_from.entry(node.state_fp).or_default();
            for succ in info.successors() {
                // Silently skip successors with missing states;
                // they won't be in unique_states and downstream
                // bitmask evaluation will produce default (false) masks.
                cache_graph_state(
                    &mut unique_states,
                    &self.graph,
                    succ,
                    "successor",
                    &registry,
                )?;
                let from_fp = node.state_fp;
                let to_fp = succ.state_fp;
                if seen_transitions.insert((from_fp, to_fp)) {
                    by_from.entry(from_fp).or_default().push(to_fp);
                }
            }
            node_data.push(NodeData { node });
        }

        pf.unique_transition_count = seen_transitions.len();
        pf.by_from_group_count = by_from.len();
        drop(seen_transitions);

        // Empty-registry checkers (unit-test fixtures that register no variables)
        // cannot use the compact `ArrayState` form — `ArrayState::from_state`
        // with an empty registry drops every variable, so `to_state` would
        // reconstruct an empty `State` and ENABLED/check eval would see no vars
        // (a false HOLD). On that path the graph's LOCAL `state_cache` still
        // holds the concrete `State`s, so resolve eval-site States from there.
        // Production (non-empty registry) leaves this map empty and uses the
        // lossless `ArrayState::to_state` directly — no behavior change.
        let registry_is_empty = registry.is_empty();
        let unique_states_full: FxHashMap<Fingerprint, crate::state::State> = if registry_is_empty {
            unique_states
                .keys()
                .filter_map(|&fp| self.graph.get_state_by_fp(fp).map(|s| (fp, s)))
                .collect()
        } else {
            FxHashMap::default()
        };
        pf.node_count = node_data.len();
        pf.unique_state_count = unique_states.len();
        pf.state_check_start = Instant::now();

        // Evaluate state checks: for each unique fingerprint, compute bitmask.
        let mut state_mask: FxHashMap<Fingerprint, CheckMask> = FxHashMap::default();
        for (check_idx, check) in check_state.iter().enumerate() {
            if !state_used.get(check_idx).copied().unwrap_or(false) {
                continue;
            }
            for (&state_fp, arr) in &unique_states {
                // Try bitmask lookup first, fall back to eval. Reconstruct State
                // transiently (dropped after this eval) — not retained.
                let result =
                    Self::try_reconstruct_state_check_bitmask(inline_results, state_fp, check)
                        .map_or_else(
                            || {
                                self.eval_live_check_expr(
                                    check,
                                    &resolve_state(arr, state_fp, &unique_states_full, &registry),
                                    None,
                                    tir,
                                )
                            },
                            Ok,
                        )?;
                if result {
                    state_mask.entry(state_fp).or_default().set(check_idx);
                } else {
                    state_mask.entry(state_fp).or_default();
                }
            }
        }

        pf.state_used_count = state_used.iter().filter(|&&u| u).count();
        pf.state_total_count = state_used.len();
        pf.action_check_start = Instant::now();

        pf.subscript_start = Instant::now();
        // Always precompute subscripts on the eval fallback path.
        self.precompute_subscript_values(check_action, action_used, &unique_states)?;
        pf.eval_loop_start = Instant::now();

        // The leaf-batch fast path binds the compact `ArrayState` directly, so it
        // cannot run on the empty-registry path (the ArrayStates are empty);
        // fall through to the State-based HashMap path below.
        if !registry_is_empty
            && try_populate_action_masks_from_leaf_batches(
                self,
                check_action,
                action_used,
                &mut unique_states,
                &by_from,
                &mut fp_to_mask,
                &mut pf,
                tir,
            )?
        {
            if profile {
                pf.print();
            }

            // Release evaluation scratch before allocating each node's final
            // action-mask vector. The final write only needs the compact mask
            // maps plus the topology keys in node_data.
            drop(by_from);
            drop(unique_states);
            drop(unique_states_full);

            // Write results into NodeInfo.
            for nd in &node_data {
                self.graph.update_node_masks(
                    &nd.node,
                    |successors, state_check_mask, action_check_masks| {
                        *state_check_mask = state_mask
                            .get(&nd.node.state_fp)
                            .cloned()
                            .unwrap_or_default();
                        *action_check_masks = ActionCheckMatrix::from_masks(
                            check_action.len(),
                            successors.iter().map(|succ| {
                                fp_to_mask
                                    .get(&(nd.node.state_fp, succ.state_fp))
                                    .cloned()
                                    .unwrap_or_default()
                            }),
                        );
                        debug_assert_eq!(action_check_masks.len(), successors.len());
                    },
                )?;
            }

            return Ok(());
        }

        // Collect ENABLED info from check expressions.
        let mut enabled_infos_raw: Vec<EnabledInfo> = Vec::new();
        for (check_idx, check) in check_action.iter().enumerate() {
            if !action_used.get(check_idx).copied().unwrap_or(false) {
                continue;
            }
            collect_enabled_nodes(check, &mut enabled_infos_raw);
        }
        let enabled_infos: Vec<&EnabledInfo> = {
            let mut seen_tags: FxHashSet<u32> = FxHashSet::default();
            enabled_infos_raw
                .iter()
                .filter(|info| seen_tags.insert(info.tag))
                .collect()
        };
        let exact_raw_fp_plan = ActionLeafBatchPlan::from_checks(
            check_action,
            action_used,
            self,
            self.exact_raw_fp_leaf_fast_path_allowed,
        )
        .filter(ActionLeafBatchPlan::supports_fallback_fp_reconstruction);

        // Eval path: evaluate expressions directly without cross-property per-tag caches.
        {
            if self.succ_witnesses.is_some() || self.ctx.var_registry().is_empty() {
                // Pre-compute ENABLED (HashMap/witness path).
                if !enabled_infos.is_empty() {
                    // Part of #3962: Use ctx-aware variant to sync in_enabled_scope shadow.
                    let _enabled_guard = crate::eval::enter_enabled_scope_with_ctx(&self.ctx);
                    let mut seen_fps: FxHashSet<Fingerprint> = FxHashSet::default();
                    for nd in &node_data {
                        let state_fp = nd.node.state_fp;
                        if !seen_fps.insert(state_fp) {
                            continue;
                        }
                        for info in &enabled_infos {
                            let eval_ctx = match info.bindings {
                                Some(ref chain) => self.ctx.with_liveness_bindings(chain),
                                None => self.ctx.clone(),
                            };
                            super::eval_enabled_cached(&eval_ctx, state_fp, info.tag, || {
                                let from_state = resolve_state(
                                    get_unique_state(&unique_states, state_fp, "ENABLED source")?,
                                    state_fp,
                                    &unique_states_full,
                                    &registry,
                                );
                                self.eval_enabled(
                                    &eval_ctx,
                                    &from_state,
                                    &info.action,
                                    info.bindings.as_ref(),
                                    info.require_state_change,
                                    info.subscript.as_ref(),
                                )
                            })?;
                        }
                    }
                }
                for (from_fp, transitions) in &by_from {
                    for &to_fp in transitions {
                        let from_state = resolve_state(
                            get_unique_state(&unique_states, *from_fp, "action source")?,
                            *from_fp,
                            &unique_states_full,
                            &registry,
                        );
                        // Part of #3746: skip transitions whose destination state
                        // is missing from the cache (parallel mode).
                        let Some(to_arr) = try_get_unique_state(&unique_states, to_fp) else {
                            if self.graph.has_owned_state_cache() {
                                return Err(unique_state_error("action destination", to_fp));
                            }
                            fp_to_mask.insert((*from_fp, to_fp), CheckMask::new());
                            continue;
                        };
                        let to_state = resolve_state(to_arr, to_fp, &unique_states_full, &registry);

                        if let Some(mask) = exact_raw_fp_plan.as_ref().and_then(|plan| {
                            plan.try_check_mask_from_fps(check_action, action_used, *from_fp, to_fp)
                        }) {
                            pf.cache_hit_transitions += 1;
                            fp_to_mask.insert((*from_fp, to_fp), mask);
                            continue;
                        }
                        pf.fresh_eval_transitions += 1;

                        // Bind state variables for this transition (empty-registry path).
                        // eval_live_check_expr_inner assumes ctx is pre-configured.
                        let prev_env = self.ctx.env().clone();
                        for (name, value) in from_state.vars() {
                            self.ctx
                                .bind_mut(std::sync::Arc::clone(name), value.clone());
                        }
                        let cached_next_env = self.get_cached_env(&to_state);
                        let _next_state_guard = self.ctx.take_next_state_guard();
                        *self.ctx.next_state_mut() = Some(cached_next_env);

                        let mut mask = CheckMask::new();
                        for (check_idx, check) in check_action.iter().enumerate() {
                            if !action_used.get(check_idx).copied().unwrap_or(false) {
                                continue;
                            }
                            if self.eval_live_check_expr_inner(
                                check,
                                &from_state,
                                Some(&to_state),
                                tir,
                            )? {
                                mask.set(check_idx);
                            }
                        }

                        *self.ctx.env_mut() = prev_env;
                        fp_to_mask.insert((*from_fp, to_fp), mask);
                    }
                }
            } else {
                // Optimized path: merged ENABLED + action checks with array binding.
                let registry_cloned = self.ctx.var_registry().clone();

                for (from_fp, transitions) in &by_from {
                    let from_array =
                        get_unique_state(&unique_states, *from_fp, "optimized source")?;
                    // Reconstruct a transient State (dropped after this source
                    // state's transitions) for binding + the evaluator's
                    // fingerprint use; unique_states retains only ArrayState.
                    let from_state = from_array.to_state(&registry_cloned);
                    let current_values = from_state.to_values(&registry_cloned);
                    let _state_guard = self.ctx.bind_state_array_guard(&current_values);

                    // Phase A: ENABLED evaluation.
                    if !enabled_infos.is_empty() {
                        // Part of #3962: Use ctx-aware variant to sync in_enabled_scope shadow.
                        let _enabled_guard = crate::eval::enter_enabled_scope_with_ctx(&self.ctx);
                        for info in &enabled_infos {
                            let state_fp = *from_fp;
                            if super::is_enabled_cached(state_fp, info.tag) {
                                continue;
                            }
                            let result = self.eval_enabled_for_node(
                                info,
                                &from_state,
                                state_fp,
                                &registry_cloned,
                            )?;
                            super::set_enabled_cache(state_fp, info.tag, result);
                        }
                    }

                    // Phase B: Evaluate action checks.
                    for &to_fp in transitions {
                        // Part of #3746: skip transitions whose destination state
                        // is missing from the cache (parallel mode).
                        let Some(next_array) = try_get_unique_state(&unique_states, to_fp) else {
                            if self.graph.has_owned_state_cache() {
                                return Err(unique_state_error("action destination", to_fp));
                            }
                            fp_to_mask.insert((*from_fp, to_fp), CheckMask::new());
                            continue;
                        };
                        if let Some(mask) = exact_raw_fp_plan.as_ref().and_then(|plan| {
                            plan.try_check_mask_from_fps(check_action, action_used, *from_fp, to_fp)
                        }) {
                            pf.cache_hit_transitions += 1;
                            fp_to_mask.insert((*from_fp, to_fp), mask);
                            continue;
                        }
                        pf.fresh_eval_transitions += 1;
                        let next_state = next_array.to_state(&registry_cloned);
                        let cached_next_env = self.get_cached_env(&next_state);
                        let next_values = next_state.to_values(&registry_cloned);

                        let _next_state_guard = self.ctx.take_next_state_guard();
                        let _next_guard = self.ctx.take_next_state_env_guard();
                        *self.ctx.next_state_mut() = Some(cached_next_env);
                        self.ctx.bind_next_state_array(&next_values);

                        let mut mask = CheckMask::new();
                        for (check_idx, check) in check_action.iter().enumerate() {
                            if !action_used.get(check_idx).copied().unwrap_or(false) {
                                continue;
                            }
                            let result = self.eval_live_check_expr_inner(
                                check,
                                &from_state,
                                Some(&next_state),
                                tir,
                            )?;
                            if result {
                                mask.set(check_idx);
                            }
                        }

                        fp_to_mask.insert((*from_fp, to_fp), mask);
                    }
                }
            }
        }
        pf.enabled_info_count = enabled_infos.len();
        if profile {
            pf.print();
        }

        // enabled_infos borrows enabled_infos_raw, so release the references
        // first. None of the remaining evaluation scratch is needed while the
        // final per-node mask vectors are installed.
        drop(enabled_infos);
        drop(enabled_infos_raw);
        drop(by_from);
        drop(unique_states);
        drop(unique_states_full);

        // Write results into NodeInfo.
        for nd in &node_data {
            self.graph.update_node_masks(
                &nd.node,
                |successors, state_check_mask, action_check_masks| {
                    *state_check_mask = state_mask
                        .get(&nd.node.state_fp)
                        .cloned()
                        .unwrap_or_default();
                    *action_check_masks = ActionCheckMatrix::from_masks(
                        check_action.len(),
                        successors.iter().map(|succ| {
                            fp_to_mask
                                .get(&(nd.node.state_fp, succ.state_fp))
                                .cloned()
                                .unwrap_or_default()
                        }),
                    );
                    debug_assert_eq!(action_check_masks.len(), successors.len());
                },
            )?;
        }

        Ok(())
    }

    /// Try to resolve a state check result from inline bitmask results.
    ///
    /// Returns `Some(bool)` if the check can be fully reconstructed from
    /// bitmask data, `None` if eval fallback is needed.
    fn try_reconstruct_state_check(
        inline_results: Option<InlineCheckResults<'_>>,
        fp: Fingerprint,
        check: &LiveExpr,
    ) -> Option<bool> {
        let results = inline_results?;
        let sbits = results.state_bitmasks.get_bits(&fp)?;
        // u64 first-word path: only valid for tags < 64 (multiword = false).
        if Self::all_leaves_within_tag_range(check, 0, results.max_tag, false) {
            Some(ea_precompute_cache::reconstruct_check_from_tag_bits(
                check, sbits, 0,
            ))
        } else {
            None
        }
    }

    /// Multi-word state-check reconstruction (#4159 follow-up — now LIVE).
    ///
    /// When the bitmask backend retains the full multi-word `LiveBitmask`,
    /// reconstruct via `storage::reconstruct_check_from_bitmask`, which reads
    /// tags >= 64 correctly. When the backend is u64-only (cross-property
    /// caches / disk cold tier) this falls back to the first-word path, which
    /// is fail-closed at tag < 64. Returns `None` (→ eval fallback) when the
    /// fingerprint is absent or a leaf is outside the reconstructable range.
    fn try_reconstruct_state_check_bitmask(
        inline_results: Option<InlineCheckResults<'_>>,
        fp: Fingerprint,
        check: &LiveExpr,
    ) -> Option<bool> {
        let results = inline_results?;
        if results.multiword_capable() {
            // Presence of the fingerprint means all tags were evaluated; absence
            // means no data → eval fallback.
            let state_bm = results.state_bitmasks.get_bitmask_words(&fp)?;
            if Self::all_leaves_within_tag_range(check, 0, results.max_tag, true) {
                let empty_action = crate::storage::LiveBitmask::default();
                return Some(crate::storage::reconstruct_check_from_bitmask(
                    check,
                    &state_bm,
                    &empty_action,
                ));
            }
            return None;
        }
        // Backend cannot serve multi-word bits — use the sound u64 first-word path.
        Self::try_reconstruct_state_check(inline_results, fp, check)
    }
}
