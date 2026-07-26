// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Leaf batching helpers for EA check-mask precompute (#2399).

use super::check_mask::CheckMask;
use super::ea_precompute_profile::PopulateMasksProfile;
use super::live_expr::LiveExpr;
use super::{LivenessChecker, TirProgram};
use crate::error::EvalError;
use crate::liveness::inline_leaf_eval::{
    eval_action_leaves_array, eval_state_leaves_with_array_successors,
};
use crate::state::{ArrayState, Fingerprint};
use rustc_hash::{FxHashMap, FxHashSet};

pub(super) struct ActionLeafBatchPlan<'a> {
    pub(super) state_leaves: Vec<&'a LiveExpr>,
    pub(super) action_leaves: Vec<&'a LiveExpr>,
    action_fp_modes: Vec<ActionLeafFpMode>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActionLeafFpMode {
    /// Whole-Next ActionPred leaves are true on every real behavior-graph
    /// edge because those edges were produced by Next enumeration.
    KnownTrue(u32),
    /// The preceding ENABLED scan may already have evaluated this exact
    /// ActionPred on the same raw transition.
    ScanPred(u32),
    /// A full-state subscript changes exactly when the exact raw state
    /// fingerprint changes (with the same fp64 trust used by state dedup).
    FullStateChanged(u32),
    Unsupported,
}

impl<'a> ActionLeafBatchPlan<'a> {
    pub(super) fn from_checks(
        check_action: &'a [LiveExpr],
        action_used: &[bool],
        checker: &LivenessChecker,
        allow_exact_raw_fp: bool,
    ) -> Option<Self> {
        let mut state_leaves = Vec::new();
        let mut action_leaves = Vec::new();
        let mut seen_state_tags = FxHashSet::default();
        let mut seen_action_tags = FxHashSet::default();

        for (check_idx, check) in check_action.iter().enumerate() {
            if !action_used.get(check_idx).copied().unwrap_or(false) {
                continue;
            }
            if !collect_action_check_leaves(
                check,
                &mut state_leaves,
                &mut action_leaves,
                &mut seen_state_tags,
                &mut seen_action_tags,
            ) {
                return None;
            }
        }

        let use_exact_raw_fp = allow_exact_raw_fp
            && checker.graph.has_owned_state_cache()
            && checker.succ_witnesses.is_none()
            && checker.state_fp_to_canon_fp.is_none()
            && !checker.ctx.var_registry().is_empty()
            && !exact_raw_fp_leaf_fast_path_disabled();
        let action_fp_modes = action_leaves
            .iter()
            .map(|leaf| {
                if !use_exact_raw_fp {
                    return ActionLeafFpMode::Unsupported;
                }
                match leaf {
                    LiveExpr::ActionPred { tag, .. }
                        if crate::liveness::whole_next_action_tag(*tag) =>
                    {
                        ActionLeafFpMode::KnownTrue(*tag)
                    }
                    LiveExpr::ActionPred { tag, .. } => ActionLeafFpMode::ScanPred(*tag),
                    LiveExpr::StateChanged {
                        subscript: None,
                        tag,
                        ..
                    } => ActionLeafFpMode::FullStateChanged(*tag),
                    LiveExpr::StateChanged {
                        subscript: Some(subscript),
                        bindings,
                        tag,
                    } if crate::liveness::enabled_eval::subscript_covers_all_vars(
                        &checker.ctx,
                        bindings.as_ref(),
                        subscript,
                    ) =>
                    {
                        ActionLeafFpMode::FullStateChanged(*tag)
                    }
                    _ => ActionLeafFpMode::Unsupported,
                }
            })
            .collect();

        Some(Self {
            state_leaves,
            action_leaves,
            action_fp_modes,
        })
    }

    /// Reconstruct an action-leaf mask without binding evaluator environments
    /// when every leaf has an exact raw answer already available. Any missing
    /// ENABLED-scan result or unsupported subscript fails closed to the normal
    /// array evaluator for this transition.
    fn try_action_leaf_mask_from_fps(
        &self,
        current_fp: Fingerprint,
        next_fp: Fingerprint,
    ) -> Option<CheckMask> {
        let mut mask = CheckMask::new();
        for mode in &self.action_fp_modes {
            let (tag, result) = match *mode {
                ActionLeafFpMode::KnownTrue(tag) => (tag, true),
                ActionLeafFpMode::ScanPred(tag) => (
                    tag,
                    crate::liveness::checker::get_scan_pred_result(current_fp, next_fp, tag)?,
                ),
                ActionLeafFpMode::FullStateChanged(tag) => (tag, current_fp != next_fp),
                ActionLeafFpMode::Unsupported => return None,
            };
            if result {
                mask.set(tag as usize);
            }
        }
        Some(mask)
    }

    pub(super) fn try_state_leaf_mask_from_cache(
        &self,
        state_fp: Fingerprint,
    ) -> Option<CheckMask> {
        let mut mask = CheckMask::new();
        for leaf in &self.state_leaves {
            let LiveExpr::Enabled { tag, .. } = leaf else {
                return None;
            };
            if super::get_enabled_cached(state_fp, *tag)? {
                mask.set(*tag as usize);
            }
        }
        Some(mask)
    }

    /// Reconstruct the final check-index mask from exact raw fingerprints and
    /// the authoritative ENABLED results computed for this source state.
    pub(super) fn try_check_mask_from_fps(
        &self,
        check_action: &[LiveExpr],
        action_used: &[bool],
        current_fp: Fingerprint,
        next_fp: Fingerprint,
    ) -> Option<CheckMask> {
        let state_mask = self.try_state_leaf_mask_from_cache(current_fp)?;
        self.try_check_mask_from_fps_with_state_mask(
            check_action,
            action_used,
            &state_mask,
            current_fp,
            next_fp,
        )
    }

    /// Reconstruct one edge's final check mask when the caller already loaded
    /// the source-state leaf mask. Streaming graph writers use this to avoid
    /// repeating ENABLED-cache probes for every outgoing product edge.
    pub(super) fn try_check_mask_from_fps_with_state_mask(
        &self,
        check_action: &[LiveExpr],
        action_used: &[bool],
        state_mask: &CheckMask,
        current_fp: Fingerprint,
        next_fp: Fingerprint,
    ) -> Option<CheckMask> {
        let action_mask = self.try_action_leaf_mask_from_fps(current_fp, next_fp)?;
        let mut check_mask = CheckMask::new();
        for (check_idx, check) in check_action.iter().enumerate() {
            if action_used.get(check_idx).copied().unwrap_or(false)
                && reconstruct_check_from_masks(check, state_mask, &action_mask)
            {
                check_mask.set(check_idx);
            }
        }
        Some(check_mask)
    }

    /// The fallback precompute path does not populate per-transition scan
    /// results. Avoid probing every edge unless all action facts are available
    /// from whole-Next provenance or full-state fingerprints alone.
    pub(super) fn supports_fallback_fp_reconstruction(&self) -> bool {
        self.state_leaves
            .iter()
            .all(|leaf| matches!(leaf, LiveExpr::Enabled { .. }))
            && self.action_fp_modes.iter().all(|mode| {
                matches!(
                    mode,
                    ActionLeafFpMode::KnownTrue(_) | ActionLeafFpMode::FullStateChanged(_)
                )
            })
    }
}

fn exact_raw_fp_leaf_fast_path_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("TY_DISABLE_LIVENESS_FP_LEAF_FAST_PATH").is_ok_and(|value| value == "1")
    })
}

fn collect_action_check_leaves<'a>(
    expr: &'a LiveExpr,
    state_leaves: &mut Vec<&'a LiveExpr>,
    action_leaves: &mut Vec<&'a LiveExpr>,
    seen_state_tags: &mut FxHashSet<u32>,
    seen_action_tags: &mut FxHashSet<u32>,
) -> bool {
    match expr {
        LiveExpr::Bool(_) => true,
        LiveExpr::StatePred { tag, .. } | LiveExpr::Enabled { tag, .. } => {
            if seen_state_tags.insert(*tag) {
                state_leaves.push(expr);
            }
            true
        }
        LiveExpr::ActionPred { tag, .. } | LiveExpr::StateChanged { tag, .. } => {
            if seen_action_tags.insert(*tag) {
                action_leaves.push(expr);
            }
            true
        }
        LiveExpr::Not(inner) => collect_action_check_leaves(
            inner,
            state_leaves,
            action_leaves,
            seen_state_tags,
            seen_action_tags,
        ),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => parts.iter().all(|part| {
            collect_action_check_leaves(
                part,
                state_leaves,
                action_leaves,
                seen_state_tags,
                seen_action_tags,
            )
        }),
        LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => false,
    }
}

fn reconstruct_check_from_masks(
    expr: &LiveExpr,
    state_tags: &CheckMask,
    action_tags: &CheckMask,
) -> bool {
    match expr {
        LiveExpr::Bool(value) => *value,
        LiveExpr::StatePred { tag, .. } | LiveExpr::Enabled { tag, .. } => {
            state_tags.get(*tag as usize)
        }
        LiveExpr::ActionPred { tag, .. } | LiveExpr::StateChanged { tag, .. } => {
            action_tags.get(*tag as usize)
        }
        LiveExpr::Not(inner) => !reconstruct_check_from_masks(inner, state_tags, action_tags),
        LiveExpr::And(parts) => parts
            .iter()
            .all(|part| reconstruct_check_from_masks(part, state_tags, action_tags)),
        LiveExpr::Or(parts) => parts
            .iter()
            .any(|part| reconstruct_check_from_masks(part, state_tags, action_tags)),
        LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => false,
    }
}

pub(super) fn try_populate_action_masks_from_leaf_batches(
    checker: &mut LivenessChecker,
    check_action: &[LiveExpr],
    action_used: &[bool],
    unique_states: &mut FxHashMap<Fingerprint, ArrayState>,
    by_from: &FxHashMap<Fingerprint, Vec<Fingerprint>>,
    fp_to_mask: &mut FxHashMap<(Fingerprint, Fingerprint), CheckMask>,
    pf: &mut PopulateMasksProfile,
    tir: Option<&TirProgram<'_>>,
) -> Result<bool, EvalError> {
    if checker.succ_witnesses.is_some()
        // VIEW/symmetry may key graph nodes by a canonical fingerprint that
        // does not identify the concrete ArrayState bound below. The leaf
        // caches require concrete-state fingerprints, so retain the canonical
        // evaluator path until those two domains are passed separately.
        || checker.state_fp_to_canon_fp.is_some()
        || checker.ctx.var_registry().is_empty()
        // Explicit TIR evaluation is an observable evaluator selection. Its
        // action path currently requires concrete States, so retain the
        // existing fallback rather than silently switching evaluators.
        || tir.is_some()
    {
        return Ok(false);
    }

    let Some(leaf_plan) = ActionLeafBatchPlan::from_checks(
        check_action,
        action_used,
        checker,
        checker.exact_raw_fp_leaf_fast_path_allowed,
    ) else {
        return Ok(false);
    };

    // Owned compact exploration records every raw successor before tableau
    // pruning. Some of those states are therefore absent from unique_states,
    // which initially contains only behavior-graph nodes. Hydrate their
    // Arc-backed payloads before borrowing the map for state-leaf evaluation;
    // otherwise a legitimate pruned successor makes this path do partial work
    // and then repeat the entire pass through the State-based fallback.
    if checker.graph.has_owned_state_cache() && !leaf_plan.state_leaves.is_empty() {
        for from_fp in by_from.keys() {
            let Some(fps) = checker.state_successor_fps.get(from_fp) else {
                return Err(EvalError::Internal {
                    message: format!(
                        "owned compact cache is missing successor adjacency for leaf-batch source {from_fp}"
                    ),
                    span: None,
                });
            };
            for &succ_fp in fps.iter() {
                if !unique_states.contains_key(&succ_fp) {
                    let Some(array) = checker.graph.get_array_state_by_fp(succ_fp) else {
                        return Err(EvalError::Internal {
                            message: format!(
                                "owned compact cache is missing state leaf-batch successor payload {succ_fp} from source {from_fp}"
                            ),
                            span: None,
                        });
                    };
                    unique_states.insert(succ_fp, array);
                }
            }
        }
    }

    // Reference the compact ArrayStates directly (no per-node State→ArrayState
    // conversion or second map).
    let unique_arrays: Option<&FxHashMap<Fingerprint, ArrayState>> =
        (!leaf_plan.state_leaves.is_empty()).then_some(&*unique_states);
    pf.enabled_info_count = leaf_plan
        .state_leaves
        .iter()
        .filter(|leaf| matches!(leaf, LiveExpr::Enabled { .. }))
        .count();

    for (from_fp, transitions) in by_from {
        // `eval_action_leaves_array` uses eval_entry_inline for batch-safe
        // leaves and therefore relies on its caller to establish the current
        // state boundary. Clear once per source, preserving cache reuse across
        // all of that source's destination transitions. The paired ENABLED
        // scan scratchpad has the same one-source lifetime.
        if !leaf_plan.action_leaves.is_empty() {
            crate::eval::clear_for_state_boundary();
        }
        crate::liveness::clear_scan_pred_results();
        let state_leaf_mask = if leaf_plan.state_leaves.is_empty() {
            CheckMask::new()
        } else {
            let unique_arrays = unique_arrays
                .as_ref()
                .expect("state leaves require preconverted arrays");
            // Part of #3746: from_fp should always be in unique_arrays
            // (only nodes with cached states are in node_data), but guard
            // defensively against parallel-mode cache misses.
            let Some(from_array) = unique_arrays.get(from_fp) else {
                if checker.graph.has_owned_state_cache() {
                    return Err(EvalError::Internal {
                        message: format!(
                            "owned compact cache is missing state leaf-batch source payload {from_fp}"
                        ),
                        span: None,
                    });
                }
                return Ok(false);
            };
            // Part of #3735: shared/fingerprint-only caches can legitimately
            // lack concrete successor payloads, so those paths bail out to the
            // regular evaluator. Owned compact exploration has a stronger
            // contract: its preflight above resolves every raw successor or
            // fails closed.
            let state_successors: Vec<(&ArrayState, Fingerprint)> = if checker
                .graph
                .has_owned_state_cache()
            {
                let Some(fps) = checker.state_successor_fps.get(from_fp) else {
                    return Err(EvalError::Internal {
                        message: format!(
                            "owned compact cache is missing successor adjacency for leaf-batch source {from_fp}"
                        ),
                        span: None,
                    });
                };
                let mut v = Vec::with_capacity(fps.len());
                for succ_fp in fps.iter() {
                    match unique_arrays.get(succ_fp) {
                        Some(array) => v.push((array, *succ_fp)),
                        // The owned-cache preflight above resolves every
                        // pre-tableau successor or fails closed.
                        None => {
                            return Err(EvalError::Internal {
                                message: format!(
                                    "owned compact cache is missing state leaf-batch successor payload {succ_fp} from source {from_fp}"
                                ),
                                span: None,
                            });
                        }
                    }
                }
                v
            } else if let Some(fps) = checker.state_successor_fps.get(from_fp) {
                let mut v = Vec::with_capacity(fps.len());
                for succ_fp in fps.iter() {
                    match unique_arrays.get(succ_fp) {
                        Some(array) => v.push((array, *succ_fp)),
                        None => return Ok(false),
                    }
                }
                v
            } else if let Some(succs) = checker.state_successors.get(from_fp) {
                let mut v = Vec::with_capacity(succs.len());
                for succ in succs.iter() {
                    let succ_fp = succ.fingerprint();
                    match unique_arrays.get(&succ_fp) {
                        Some(array) => v.push((array, succ_fp)),
                        None => return Ok(false),
                    }
                }
                v
            } else {
                Vec::new()
            };
            let results = eval_state_leaves_with_array_successors(
                &mut checker.ctx,
                &leaf_plan.state_leaves,
                *from_fp,
                from_array,
                &state_successors,
            )?;
            let mut mask = CheckMask::new();
            for &(tag, result) in &results {
                if result {
                    mask.set(tag as usize);
                }
            }
            mask
        };

        // Part of #3746: from_fp should always be in unique_states (only nodes
        // with successfully cached states are included in node_data), but guard
        // against missing source states in parallel mode by skipping.
        let Some(from_array) = unique_states.get(from_fp) else {
            if checker.graph.has_owned_state_cache() {
                return Err(EvalError::Internal {
                    message: format!(
                        "owned compact cache is missing action leaf-batch source payload {from_fp}"
                    ),
                    span: None,
                });
            }
            continue;
        };
        for &to_fp in transitions {
            pf.fresh_eval_transitions += 1;
            // Part of #3746: skip transitions whose destination state is
            // missing from the cache (parallel/fingerprint-only mode).
            let Some(to_array) = unique_states.get(&to_fp) else {
                if checker.graph.has_owned_state_cache() {
                    return Err(EvalError::Internal {
                        message: format!(
                            "owned compact cache is missing action leaf-batch destination payload {to_fp} from source {from_fp}"
                        ),
                        span: None,
                    });
                }
                fp_to_mask.insert((*from_fp, to_fp), CheckMask::new());
                continue;
            };
            let action_leaf_mask = if leaf_plan.action_leaves.is_empty() {
                CheckMask::new()
            } else if let Some(mask) = leaf_plan.try_action_leaf_mask_from_fps(*from_fp, to_fp) {
                mask
            } else {
                let mut mask = CheckMask::new();
                // Evaluate directly against the compact environments. This is
                // the same array-native leaf evaluator used while recording
                // inline liveness results during BFS, and avoids rebuilding two
                // `State`/value arrays plus a next-state `Env` per transition.
                for (tag, result) in eval_action_leaves_array(
                    &mut checker.ctx,
                    &leaf_plan.action_leaves,
                    *from_fp,
                    from_array,
                    to_fp,
                    to_array,
                )? {
                    if result {
                        mask.set(tag as usize);
                    }
                }
                mask
            };

            let mut mask = CheckMask::new();
            for (check_idx, check) in check_action.iter().enumerate() {
                if !action_used.get(check_idx).copied().unwrap_or(false) {
                    continue;
                }
                if reconstruct_check_from_masks(check, &state_leaf_mask, &action_leaf_mask) {
                    mask.set(check_idx);
                }
            }

            fp_to_mask.insert((*from_fp, to_fp), mask);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::test_helpers::{make_checker_with_vars, spanned};
    use crate::state::State;
    use crate::Value;
    use std::sync::Arc;
    use tla_core::ast::Expr;

    fn owned_checker() -> LivenessChecker {
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        checker
    }

    fn transition_fixture() -> (State, State, FxHashMap<Fingerprint, Vec<Fingerprint>>) {
        let from = State::from_pairs([("x", Value::int(0))]);
        let to = State::from_pairs([("x", Value::int(1))]);
        let mut by_from = FxHashMap::default();
        by_from.insert(from.fingerprint(), vec![to.fingerprint()]);
        (from, to, by_from)
    }

    #[test]
    fn owned_action_leaf_batch_missing_source_payload_fails_closed() {
        let mut checker = owned_checker();
        let (from, _to, by_from) = transition_fixture();
        let action = LiveExpr::state_changed(None, 1);
        let mut unique_states = FxHashMap::default();
        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        let error = try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[action],
            &[true],
            &mut unique_states,
            &by_from,
            &mut masks,
            &mut profile,
            None,
        )
        .expect_err("owned action source loss must fail closed");

        assert!(error.to_string().contains(&format!(
            "missing action leaf-batch source payload {}",
            from.fingerprint()
        )));
    }

    #[test]
    fn canonical_fingerprint_map_retains_legacy_action_evaluator() {
        let mut checker = owned_checker();
        checker.state_fp_to_canon_fp = Some(Arc::new(FxHashMap::default()));
        let action = LiveExpr::state_changed(None, 1);
        let mut unique_states = FxHashMap::default();
        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        assert!(!try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[action],
            &[true],
            &mut unique_states,
            &FxHashMap::default(),
            &mut masks,
            &mut profile,
            None,
        )
        .expect("canonical fingerprint mode should select the legacy evaluator"));
    }

    #[test]
    fn leaf_batch_respects_disabled_exact_raw_evaluator_substitution() {
        super::super::clear_leaf_result_cache();
        crate::liveness::extend_whole_next_action_tags([11]);

        let mut checker = owned_checker();
        checker.set_exact_raw_fp_leaf_fast_path_allowed(false);
        let (from, to, by_from) = transition_fixture();
        let registry = checker.ctx.var_registry().clone();
        let mut unique_states = FxHashMap::default();
        unique_states.insert(from.fingerprint(), ArrayState::from_state(&from, &registry));
        unique_states.insert(to.fingerprint(), ArrayState::from_state(&to, &registry));
        let check = LiveExpr::action_pred(Arc::new(spanned(Expr::Bool(false))), 11);
        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        assert!(try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[check],
            &[true],
            &mut unique_states,
            &by_from,
            &mut masks,
            &mut profile,
            None,
        )
        .expect("disabled exact-raw substitution should retain leaf-batch evaluation"));
        assert!(!masks[&(from.fingerprint(), to.fingerprint())].get(0));

        super::super::clear_leaf_result_cache();
    }

    #[test]
    fn owned_action_leaf_batch_missing_destination_payload_fails_closed() {
        let mut checker = owned_checker();
        let (from, to, by_from) = transition_fixture();
        let registry = checker.ctx.var_registry().clone();
        let mut unique_states = FxHashMap::default();
        unique_states.insert(from.fingerprint(), ArrayState::from_state(&from, &registry));
        let action = LiveExpr::state_changed(None, 1);
        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        let error = try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[action],
            &[true],
            &mut unique_states,
            &by_from,
            &mut masks,
            &mut profile,
            None,
        )
        .expect_err("owned action destination loss must fail closed");

        assert!(error.to_string().contains(&format!(
            "missing action leaf-batch destination payload {}",
            to.fingerprint()
        )));
    }

    #[test]
    fn owned_action_leaf_batch_hydrates_pre_tableau_successor_payload() {
        crate::liveness::clear_enabled_cache();

        let mut checker = owned_checker();
        let from = State::from_pairs([("x", Value::int(0))]);
        let pruned = State::from_pairs([("x", Value::int(1))]);
        let from_fp = from.fingerprint();
        let pruned_fp = pruned.fingerprint();
        let registry = checker.ctx.var_registry().clone();
        let mut unique_states = FxHashMap::default();
        unique_states.insert(from_fp, ArrayState::from_state(&from, &registry));
        checker.graph.cache_owned_state(from_fp, &from);
        checker.graph.cache_owned_state(pruned_fp, &pruned);
        checker
            .state_successor_fps
            .insert(from_fp, Arc::new(vec![pruned_fp]));

        // The behavior graph retained only a self-loop; the changed-x ENABLED
        // witness exists solely in the complete pre-tableau successor list.
        let mut by_from = FxHashMap::default();
        by_from.insert(from_fp, vec![from_fp]);
        let enabled_changed_x = LiveExpr::enabled_with_bindings(
            Arc::new(spanned(Expr::Bool(true))),
            true,
            Some(Arc::new(spanned(Expr::Ident(
                "x".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )))),
            47,
            None,
        );

        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        assert!(try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[enabled_changed_x],
            &[true],
            &mut unique_states,
            &by_from,
            &mut masks,
            &mut profile,
            None,
        )
        .expect("owned pre-tableau successor should resolve from the compact cache"));

        assert!(unique_states.contains_key(&pruned_fp));
        assert!(masks[&(from_fp, from_fp)].get(0));
    }

    fn wf_check_for_subscript(subscript: Arc<tla_core::Spanned<Expr>>) -> LiveExpr {
        let action = Arc::new(spanned(Expr::Bool(true)));
        LiveExpr::or(vec![
            LiveExpr::not(LiveExpr::enabled_subscripted(
                Arc::clone(&action),
                Some(Arc::clone(&subscript)),
                10,
            )),
            LiveExpr::and(vec![
                LiveExpr::action_pred(action, 11),
                LiveExpr::state_changed(Some(subscript), 12),
            ]),
        ])
    }

    #[test]
    fn exact_raw_fp_plan_reconstructs_whole_next_full_state_check() {
        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
        crate::liveness::extend_whole_next_action_tags([11]);

        let checker = owned_checker();
        let from = State::from_pairs([("x", Value::int(0))]);
        let to = State::from_pairs([("x", Value::int(1))]);
        let from_fp = from.fingerprint();
        let to_fp = to.fingerprint();
        let check = wf_check_for_subscript(Arc::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))));
        let checks = vec![check];
        let plan = ActionLeafBatchPlan::from_checks(&checks, &[true], &checker, true)
            .expect("boolean action check plan");
        assert!(plan.supports_fallback_fp_reconstruction());

        super::super::set_enabled_cache(from_fp, 10, true);
        let changed = plan
            .try_check_mask_from_fps(&checks, &[true], from_fp, to_fp)
            .expect("full-state exact raw reconstruction");
        assert!(changed.get(0));

        let stutter = plan
            .try_check_mask_from_fps(&checks, &[true], from_fp, from_fp)
            .expect("stuttering exact raw reconstruction");
        assert!(!stutter.get(0));

        // The WF disjunction is true on every edge when ENABLED is false.
        super::super::set_enabled_cache(from_fp, 10, false);
        let disabled = plan
            .try_check_mask_from_fps(&checks, &[true], from_fp, from_fp)
            .expect("cached false ENABLED reconstruction");
        assert!(disabled.get(0));

        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
    }

    #[test]
    fn exact_raw_fp_plan_falls_back_for_partial_state_subscript() {
        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
        crate::liveness::extend_whole_next_action_tags([11]);

        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x", "y"]);
        checker.enable_owned_behavior_graph_state_cache();
        let from = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);
        let to = State::from_pairs([("x", Value::int(1)), ("y", Value::int(0))]);
        let check = wf_check_for_subscript(Arc::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))));
        let checks = vec![check];
        let plan = ActionLeafBatchPlan::from_checks(&checks, &[true], &checker, true)
            .expect("boolean action check plan");
        assert!(!plan.supports_fallback_fp_reconstruction());
        super::super::set_enabled_cache(from.fingerprint(), 10, true);

        assert!(plan
            .try_check_mask_from_fps(&checks, &[true], from.fingerprint(), to.fingerprint(),)
            .is_none());

        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
    }

    #[test]
    fn exact_raw_fp_plan_falls_back_for_canonical_or_observable_evaluator() {
        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
        crate::liveness::extend_whole_next_action_tags([11]);

        let mut checker = owned_checker();
        let from = State::from_pairs([("x", Value::int(0))]);
        let to = State::from_pairs([("x", Value::int(1))]);
        let check = wf_check_for_subscript(Arc::new(spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))));
        let checks = vec![check];
        super::super::set_enabled_cache(from.fingerprint(), 10, true);

        checker.state_fp_to_canon_fp = Some(Arc::new(FxHashMap::default()));
        let canonical_plan = ActionLeafBatchPlan::from_checks(&checks, &[true], &checker, true)
            .expect("boolean action check plan");
        assert!(!canonical_plan.supports_fallback_fp_reconstruction());
        assert!(canonical_plan
            .try_check_mask_from_fps(&checks, &[true], from.fingerprint(), to.fingerprint(),)
            .is_none());

        checker.state_fp_to_canon_fp = None;
        let observable_evaluator_plan =
            ActionLeafBatchPlan::from_checks(&checks, &[true], &checker, false)
                .expect("boolean action check plan");
        assert!(!observable_evaluator_plan.supports_fallback_fp_reconstruction());
        assert!(observable_evaluator_plan
            .try_check_mask_from_fps(&checks, &[true], from.fingerprint(), to.fingerprint(),)
            .is_none());

        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
    }
}
