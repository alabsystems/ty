// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Leaf batching helpers for EA check-mask precompute (#2399).

use super::check_mask::CheckMask;
use super::ea_precompute_profile::PopulateMasksProfile;
use super::live_expr::LiveExpr;
use super::{LivenessChecker, TirProgram};
use crate::error::EvalError;
use crate::liveness::inline_leaf_eval::eval_state_leaves_with_array_successors;
use crate::state::{ArrayState, Fingerprint};
use rustc_hash::{FxHashMap, FxHashSet};

pub(super) struct ActionLeafBatchPlan<'a> {
    pub(super) state_leaves: Vec<&'a LiveExpr>,
    pub(super) action_leaves: Vec<&'a LiveExpr>,
}

impl<'a> ActionLeafBatchPlan<'a> {
    pub(super) fn from_checks(check_action: &'a [LiveExpr], action_used: &[bool]) -> Option<Self> {
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

        Some(Self {
            state_leaves,
            action_leaves,
        })
    }
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
    unique_states: &FxHashMap<Fingerprint, ArrayState>,
    by_from: &FxHashMap<Fingerprint, Vec<(Fingerprint, usize, usize)>>,
    fp_to_mask: &mut FxHashMap<(Fingerprint, Fingerprint), CheckMask>,
    pf: &mut PopulateMasksProfile,
    tir: Option<&TirProgram<'_>>,
) -> Result<bool, EvalError> {
    if checker.succ_witnesses.is_some() || checker.ctx.var_registry().is_empty() {
        return Ok(false);
    }

    let Some(leaf_plan) = ActionLeafBatchPlan::from_checks(check_action, action_used) else {
        return Ok(false);
    };

    let registry = checker.ctx.var_registry().clone();
    // unique_states already holds compact ArrayStates — reference it directly
    // (no per-node State→ArrayState conversion or extra map).
    let unique_arrays: Option<&FxHashMap<Fingerprint, ArrayState>> =
        (!leaf_plan.state_leaves.is_empty()).then_some(unique_states);
    pf.enabled_info_count = leaf_plan
        .state_leaves
        .iter()
        .filter(|leaf| matches!(leaf, LiveExpr::Enabled { .. }))
        .count();

    for (from_fp, transitions) in by_from {
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
                return Ok(false);
            };
            // Part of #3735: When successor fps from state_successor_fps are
            // not found in unique_arrays, bail out to the regular eval path
            // instead of returning a hard error. This can happen when the
            // behavior graph contains successor edges to states whose
            // concrete data is not available in the shared state cache.
            // The regular eval fallback path handles missing successors
            // gracefully via successor_states_for_enabled() (mod.rs:337-358).
            let state_successors: Vec<(&ArrayState, Fingerprint)> =
                if checker.graph.has_owned_state_cache() {
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
                            // A pre-tableau-pruning successor is legitimately
                            // absent from unique_arrays (which contains only
                            // behavior nodes). Fall back to the regular owned
                            // resolver, which can distinguish that case from a
                            // truly missing compact payload.
                            None => return Ok(false),
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
        // ArrayState IS the value array — bind directly. Reconstruct a State
        // transiently only for the evaluator's fingerprint use (dropped after
        // this source state's transitions).
        let from_state = from_array.to_state(&registry);
        let current_values = from_state.to_values(&registry);
        let _state_guard = checker.ctx.bind_state_array_guard(&current_values);
        for &(to_fp, _, _) in transitions {
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
            let to_state = to_array.to_state(&registry);
            let cached_next_env = checker.get_cached_env(&to_state);
            let next_values = to_state.to_values(&registry);
            let _next_state_guard = checker.ctx.take_next_state_guard();
            let _next_guard = checker.ctx.take_next_state_env_guard();
            *checker.ctx.next_state_mut() = Some(cached_next_env);
            checker.ctx.bind_next_state_array(&next_values);

            let action_leaf_mask = if leaf_plan.action_leaves.is_empty() {
                CheckMask::new()
            } else {
                let mut mask = CheckMask::new();
                for leaf in &leaf_plan.action_leaves {
                    if let Some(tag) = leaf.tag() {
                        if checker.eval_live_check_expr_inner(
                            leaf,
                            &from_state,
                            Some(&to_state),
                            tir,
                        )? {
                            mask.set(tag as usize);
                        }
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
    use crate::liveness::test_helpers::make_checker_with_vars;
    use crate::state::State;
    use crate::Value;

    fn owned_checker() -> LivenessChecker {
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        checker
    }

    fn transition_fixture() -> (
        State,
        State,
        FxHashMap<Fingerprint, Vec<(Fingerprint, usize, usize)>>,
    ) {
        let from = State::from_pairs([("x", Value::int(0))]);
        let to = State::from_pairs([("x", Value::int(1))]);
        let mut by_from = FxHashMap::default();
        by_from.insert(from.fingerprint(), vec![(to.fingerprint(), 0, 0)]);
        (from, to, by_from)
    }

    #[test]
    fn owned_action_leaf_batch_missing_source_payload_fails_closed() {
        let mut checker = owned_checker();
        let (from, _to, by_from) = transition_fixture();
        let action = LiveExpr::state_changed(None, 1);
        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        let error = try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[action],
            &[true],
            &FxHashMap::default(),
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
    fn owned_action_leaf_batch_missing_destination_payload_fails_closed() {
        let mut checker = owned_checker();
        let (from, to, by_from) = transition_fixture();
        let registry = checker.ctx.var_registry().clone();
        let mut unique_states = FxHashMap::default();
        unique_states.insert(
            from.fingerprint(),
            ArrayState::from_state(&from, &registry),
        );
        let action = LiveExpr::state_changed(None, 1);
        let mut masks = FxHashMap::default();
        let mut profile = PopulateMasksProfile::new();

        let error = try_populate_action_masks_from_leaf_batches(
            &mut checker,
            &[action],
            &[true],
            &unique_states,
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
}
