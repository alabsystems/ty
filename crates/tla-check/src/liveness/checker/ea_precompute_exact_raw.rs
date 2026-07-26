// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Allocation-light exact-raw action-mask reconstruction.

use super::check_mask::{ActionCheckMatrix, CheckMask};
use super::ea_precompute_leaf_batch::ActionLeafBatchPlan;
use super::{LiveExpr, LivenessChecker};
use crate::error::{EvalError, EvalResult};
use crate::liveness::debug::liveness_profile;
use crate::liveness::inline_leaf_eval::eval_state_leaves_with_array_successors;

/// Populate edge-aligned action masks directly from exact raw fingerprints.
///
/// The normal fallback snapshots compact states and deduplicated transitions
/// into several large maps before reconstructing the same cached facts. Owned
/// exact-raw exploration needs none of that scratch when every used action
/// check consists only of authoritative ENABLED results, whole-Next action
/// provenance, or full-state change tests.
pub(super) fn try_populate_action_masks_from_exact_raw_fps(
    checker: &mut LivenessChecker,
    check_action: &[LiveExpr],
    check_state: &[LiveExpr],
    action_used: &[bool],
    state_used: &[bool],
) -> EvalResult<bool> {
    if !checker.exact_raw_fp_leaf_fast_path_allowed
        || !checker.graph.has_owned_state_cache()
        || checker.succ_witnesses.is_some()
        || checker.state_fp_to_canon_fp.is_some()
        || checker.ctx.var_registry().is_empty()
        || check_state
            .iter()
            .enumerate()
            .any(|(idx, _)| state_used.get(idx).copied().unwrap_or(false))
    {
        return Ok(false);
    }

    let Some(plan) = ActionLeafBatchPlan::from_checks(
        check_action,
        action_used,
        checker,
        checker.exact_raw_fp_leaf_fast_path_allowed,
    ) else {
        return Ok(false);
    };
    if !plan.supports_fallback_fp_reconstruction() {
        return Ok(false);
    }

    let started = std::time::Instant::now();
    let nodes = checker.graph.node_keys();
    if nodes.is_empty() {
        if liveness_profile() {
            eprintln!(
                "    populate_masks: EXACT-RAW FAST PATH: {:.3}s (nodes=0, edges=0)",
                started.elapsed().as_secs_f64(),
            );
        }
        return Ok(true);
    }

    // Validate topology before evaluating missing ENABLED facts or mutating a
    // node. Terminal nodes have no action masks and need no source adjacency.
    let mut has_edges = false;
    for node in &nodes {
        let info = checker
            .graph
            .try_get_node_info(node)?
            .ok_or_else(|| missing_node_error(*node))?;
        if !info.successors().is_empty() {
            has_edges = true;
            if !checker.state_successor_fps.contains_key(&node.state_fp) {
                return Err(exact_cache_error(format!(
                    "missing successor adjacency for leaf-batch source {}",
                    node.state_fp
                )));
            }
        }
    }

    // Owned exact adjacency is the authority that makes graph fingerprints
    // sufficient here. Validate all source/destination payloads before the
    // first NodeInfo write, matching the retained-graph release gate. A graph
    // with no edges has no action facts to validate.
    if has_edges
        && checker
            .exact_raw_state_graph_cache_estimated_bytes(checker.ctx.var_registry().len())
            .is_none()
    {
        return Err(exact_cache_error(
            "owned exact-raw adjacency has a missing endpoint payload",
        ));
    }

    // Whole-Next ENABLED with a full-state subscript is exactly the existence
    // of a non-stuttering raw successor. Derive those masks from fingerprints
    // and drop any BFS/group cache backing before allocating per-edge masks.
    // Arbitrary ENABLED actions keep the authoritative evaluator/cache path.
    let whole_next_enabled_tags = all_whole_next_enabled_fp_tags(checker, &plan);
    if whole_next_enabled_tags.is_some() {
        super::release_enabled_cache_storage();
    } else {
        if !try_populate_missing_enabled_cache(checker, &plan, &nodes)? {
            return Ok(false);
        }

        // The prefill is allowed to update authoritative leaf caches, but every
        // NodeInfo write remains transactional: verify all source facts first.
        for node in &nodes {
            let info = checker
                .graph
                .try_get_node_info(node)?
                .ok_or_else(|| missing_node_error(*node))?;
            if !info.successors().is_empty()
                && plan.try_state_leaf_mask_from_cache(node.state_fp).is_none()
            {
                return Ok(false);
            }
        }
    }

    let mut edge_count = 0usize;
    for node in &nodes {
        let source_fp = node.state_fp;
        let source_state_mask = match whole_next_enabled_tags.as_deref() {
            Some(tags) => Some(whole_next_enabled_state_mask(checker, source_fp, tags)),
            None => plan.try_state_leaf_mask_from_cache(source_fp),
        };
        let updated = checker.graph.update_node_masks(
            node,
            |successors, state_check_mask, action_check_masks| {
                edge_count = edge_count.saturating_add(successors.len());
                *state_check_mask = CheckMask::new();
                *action_check_masks = ActionCheckMatrix::from_masks(
                    check_action.len(),
                    successors.iter().map(|successor| {
                        plan.try_check_mask_from_fps_with_state_mask(
                            check_action,
                            action_used,
                            source_state_mask
                                .as_ref()
                                .expect("exact-raw preflight validated nonterminal source facts"),
                            source_fp,
                            successor.state_fp,
                        )
                        .expect("exact-raw action-mask preflight validated every source")
                    }),
                );
                debug_assert_eq!(action_check_masks.len(), successors.len());
            },
        )?;
        if updated.is_none() {
            return Err(missing_node_error(*node));
        }
    }

    if liveness_profile() {
        eprintln!(
            "    populate_masks: EXACT-RAW FAST PATH: {:.3}s (nodes={}, edges={})",
            started.elapsed().as_secs_f64(),
            nodes.len(),
            edge_count,
        );
    }
    Ok(true)
}

fn all_whole_next_enabled_fp_tags(
    checker: &LivenessChecker,
    plan: &ActionLeafBatchPlan<'_>,
) -> Option<Vec<u32>> {
    if plan.state_leaves.is_empty() {
        return None;
    }
    plan.state_leaves
        .iter()
        .map(|leaf| {
            supports_whole_next_enabled_fp(checker, leaf).then(|| {
                leaf.tag()
                    .expect("whole-Next ENABLED leaf must carry a fairness tag")
            })
        })
        .collect()
}

fn whole_next_enabled_state_mask(
    checker: &LivenessChecker,
    source_fp: crate::state::Fingerprint,
    tags: &[u32],
) -> CheckMask {
    let enabled = checker
        .state_successor_fps
        .get(&source_fp)
        .is_some_and(|successors| {
            successors
                .iter()
                .any(|&successor_fp| successor_fp != source_fp)
        });
    let mut mask = CheckMask::new();
    if enabled {
        for &tag in tags {
            mask.set(tag as usize);
        }
    }
    mask
}

fn try_populate_missing_enabled_cache(
    checker: &mut LivenessChecker,
    plan: &ActionLeafBatchPlan<'_>,
    nodes: &[super::BehaviorGraphNode],
) -> EvalResult<bool> {
    if plan.state_leaves.is_empty() {
        return Ok(true);
    }

    // The mask write needs every source result to remain resident until the
    // post-fill completeness pass. Avoid a known retain-half eviction cycle;
    // the generic fallback can compute and consume masks incrementally instead.
    let Some(max_state_count) = checker.graph.owned_state_cache_len() else {
        return Ok(false);
    };
    if !super::enabled_cache::can_retain_exact_raw_enabled_prefill(
        max_state_count,
        plan.state_leaves.len(),
    ) {
        return Ok(false);
    }

    // Whole-Next ENABLED with a full-state change requirement can be answered
    // from exact raw fingerprints alone. Resolve full-subscript coverage once
    // per leaf, not once per state.
    let whole_next_fp_modes: Vec<bool> = plan
        .state_leaves
        .iter()
        .map(|leaf| supports_whole_next_enabled_fp(checker, leaf))
        .collect();

    let started = std::time::Instant::now();
    let mut source_count = 0usize;
    let mut fact_count = 0usize;
    for node in nodes {
        let info = checker
            .graph
            .try_get_node_info(node)?
            .ok_or_else(|| missing_node_error(*node))?;
        if info.successors().is_empty()
            || plan.state_leaves.iter().all(|leaf| {
                leaf.tag()
                    .is_some_and(|tag| super::is_enabled_cached(node.state_fp, tag))
            })
        {
            continue;
        }

        let successor_fps = checker
            .state_successor_fps
            .get_owned(&node.state_fp)
            .ok_or_else(|| {
                exact_cache_error(format!(
                    "missing successor adjacency for ENABLED source {}",
                    node.state_fp
                ))
            })?;
        source_count = source_count.saturating_add(1);

        let mut batch_leaves = Vec::new();
        for (leaf_idx, &leaf) in plan.state_leaves.iter().enumerate() {
            let tag = leaf
                .tag()
                .expect("exact-raw state-leaf plan contains only tagged ENABLED leaves");
            if super::is_enabled_cached(node.state_fp, tag) {
                continue;
            }
            if whole_next_fp_modes[leaf_idx] {
                let enabled = successor_fps
                    .iter()
                    .any(|&successor_fp| successor_fp != node.state_fp);
                super::set_enabled_cache(node.state_fp, tag, enabled);
                fact_count = fact_count.saturating_add(1);
            } else {
                batch_leaves.push(leaf);
            }
        }

        if batch_leaves.is_empty() {
            continue;
        }

        let source = checker
            .graph
            .get_array_state_by_fp(node.state_fp)
            .ok_or_else(|| {
                exact_cache_error(format!(
                    "missing source payload for ENABLED source {}",
                    node.state_fp
                ))
            })?;
        let mut successor_arrays = Vec::with_capacity(successor_fps.len());
        for &successor_fp in successor_fps.iter() {
            let successor = checker
                .graph
                .get_array_state_by_fp(successor_fp)
                .ok_or_else(|| {
                    exact_cache_error(format!(
                        "missing successor payload {successor_fp} for ENABLED source {}",
                        node.state_fp
                    ))
                })?;
            successor_arrays.push((successor, successor_fp));
        }
        let successor_refs: Vec<_> = successor_arrays
            .iter()
            .map(|(state, fp)| (state, *fp))
            .collect();
        super::clear_scan_pred_results();
        let results = eval_state_leaves_with_array_successors(
            &mut checker.ctx,
            &batch_leaves,
            node.state_fp,
            &source,
            &successor_refs,
        );
        super::clear_scan_pred_results();
        for (tag, result) in results? {
            super::set_enabled_cache(node.state_fp, tag, result);
            fact_count = fact_count.saturating_add(1);
        }
    }

    if liveness_profile() && fact_count != 0 {
        eprintln!(
            "    populate_masks: exact-raw ENABLED prefill: {:.3}s (sources={}, facts={})",
            started.elapsed().as_secs_f64(),
            source_count,
            fact_count,
        );
    }
    Ok(true)
}

fn supports_whole_next_enabled_fp(checker: &LivenessChecker, leaf: &LiveExpr) -> bool {
    let LiveExpr::Enabled {
        bindings,
        require_state_change: true,
        subscript,
        tag,
        ..
    } = leaf
    else {
        return false;
    };
    super::whole_next_enabled_tag(*tag)
        && subscript.as_ref().is_none_or(|subscript| {
            crate::liveness::enabled_eval::subscript_covers_all_vars(
                &checker.ctx,
                bindings.as_ref(),
                subscript,
            )
        })
}

fn missing_node_error(node: super::BehaviorGraphNode) -> EvalError {
    EvalError::Internal {
        message: format!(
            "populate_node_check_masks exact-raw fast path: missing graph node {node}"
        ),
        span: None,
    }
}

fn exact_cache_error(message: impl std::fmt::Display) -> EvalError {
    EvalError::Internal {
        message: format!("populate_node_check_masks exact-raw fast path: {message}"),
        span: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::test_helpers::{make_checker_with_vars, spanned};
    use crate::state::State;
    use crate::Value;
    use std::sync::Arc;
    use tla_core::ast::Expr;

    fn owned_checker_with_duplicate_edges() -> (LivenessChecker, State, State) {
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        checker.set_exact_raw_fp_leaf_fast_path_allowed(true);

        let source = State::from_pairs([("x", Value::int(0))]);
        let changed = State::from_pairs([("x", Value::int(1))]);
        let source_for_next = source.clone();
        let changed_for_next = changed.clone();
        let mut next = move |state: &State| {
            if state == &source_for_next {
                Ok(vec![
                    changed_for_next.clone(),
                    source_for_next.clone(),
                    changed_for_next.clone(),
                ])
            } else {
                Ok(Vec::new())
            }
        };
        checker
            .explore_state_graph_direct(std::slice::from_ref(&source), &mut next)
            .expect("exact-raw duplicate-edge fixture");
        (checker, source, changed)
    }

    #[test]
    fn exact_raw_masks_preserve_duplicate_edge_order() {
        let (mut checker, source, changed) = owned_checker_with_duplicate_edges();
        let checks = vec![LiveExpr::state_changed(None, 7)];

        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("exact-raw mask reconstruction"));

        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert_eq!(
            info.successors()
                .iter()
                .map(|node| node.state_fp)
                .collect::<Vec<_>>(),
            vec![
                changed.fingerprint(),
                source.fingerprint(),
                changed.fingerprint()
            ]
        );
        assert_eq!(info.action_check_masks.len(), 3);
        assert!(info.action_check_masks.get(0).unwrap().get(0));
        assert!(!info.action_check_masks.get(1).unwrap().get(0));
        assert!(info.action_check_masks.get(2).unwrap().get(0));
        assert_eq!(
            info.action_check_masks.get(0).unwrap(),
            info.action_check_masks.get(2).unwrap()
        );
    }

    #[test]
    fn exact_raw_masks_derive_whole_next_fairness_without_enabled_cache() {
        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
        crate::liveness::extend_whole_next_action_tags([11]);
        crate::liveness::extend_whole_next_enabled_tags([10]);
        let (mut checker, source, changed) = owned_checker_with_duplicate_edges();
        super::super::set_enabled_cache(source.fingerprint(), 99, true);
        let action = Arc::new(spanned(Expr::Bool(true)));
        let checks = vec![LiveExpr::or(vec![
            LiveExpr::not(LiveExpr::enabled_subscripted(Arc::clone(&action), None, 10)),
            LiveExpr::and(vec![
                LiveExpr::action_pred(action, 11),
                LiveExpr::state_changed(None, 12),
            ]),
        ])];

        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("whole-Next exact-raw mask reconstruction"));

        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 10),
            None,
            "fingerprint-derived whole-Next masks must not retain ENABLED cache entries"
        );
        assert_eq!(
            super::super::get_enabled_cached(changed.fingerprint(), 10),
            None
        );
        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 99),
            None,
            "fingerprint-derived masks release stale ENABLED cache backing"
        );
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert_eq!(info.action_check_masks.len(), 3);
        assert!(info.action_check_masks.get(0).unwrap().get(0));
        assert!(!info.action_check_masks.get(1).unwrap().get(0));
        assert!(info.action_check_masks.get(2).unwrap().get(0));
        super::super::clear_enabled_cache();
        super::super::clear_leaf_result_cache();
    }

    #[test]
    fn exact_raw_whole_next_is_disabled_when_every_successor_stutters() {
        super::super::clear_enabled_cache();
        crate::liveness::extend_whole_next_enabled_tags([10]);
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        checker.set_exact_raw_fp_leaf_fast_path_allowed(true);
        let source = State::from_pairs([("x", Value::int(0))]);
        let source_for_next = source.clone();
        let mut next = move |state: &State| {
            if state == &source_for_next {
                Ok(vec![source_for_next.clone()])
            } else {
                Ok(Vec::new())
            }
        };
        checker
            .explore_state_graph_direct(std::slice::from_ref(&source), &mut next)
            .expect("exact-raw all-stutter fixture");
        let checks = vec![LiveExpr::not(LiveExpr::enabled_subscripted(
            Arc::new(spanned(Expr::Bool(true))),
            None,
            10,
        ))];

        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("all-stutter exact-raw mask reconstruction"));

        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 10),
            None
        );
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert_eq!(info.action_check_masks.len(), 1);
        assert!(info.action_check_masks.get(0).unwrap().get(0));
        super::super::clear_enabled_cache();
    }

    #[test]
    fn exact_raw_mixed_enabled_leaves_keep_authoritative_cache_path() {
        super::super::clear_enabled_cache();
        crate::liveness::extend_whole_next_enabled_tags([10]);
        let (mut checker, source, _) = owned_checker_with_duplicate_edges();
        let checks = vec![LiveExpr::and(vec![
            LiveExpr::enabled_subscripted(Arc::new(spanned(Expr::Bool(true))), None, 10),
            LiveExpr::not(LiveExpr::enabled(Arc::new(spanned(Expr::Bool(false))), 31)),
            LiveExpr::state_changed(None, 32),
        ])];

        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("mixed ENABLED exact-raw mask reconstruction"));

        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 10),
            Some(true)
        );
        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 31),
            Some(false)
        );
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert_eq!(info.action_check_masks.len(), 3);
        assert!(info.action_check_masks.get(0).unwrap().get(0));
        assert!(!info.action_check_masks.get(1).unwrap().get(0));
        assert!(info.action_check_masks.get(2).unwrap().get(0));
        super::super::clear_enabled_cache();
    }

    #[test]
    fn exact_raw_enabled_prefill_error_preserves_nodes_and_reuses_cache() {
        super::super::clear_enabled_cache();
        let (mut checker, source, _) = owned_checker_with_duplicate_edges();
        let checks = vec![LiveExpr::and(vec![
            LiveExpr::enabled(Arc::new(spanned(Expr::Bool(true))), 20),
            LiveExpr::enabled(Arc::new(spanned(Expr::Int(1.into()))), 21),
            LiveExpr::state_changed(None, 22),
        ])];
        let nodes = checker.graph.node_keys();
        for node in &nodes {
            checker
                .graph
                .update_node_info(node, |info| {
                    info.state_check_mask = CheckMask::from_indices(&[8]);
                    info.action_check_masks =
                        vec![CheckMask::from_indices(&[9]); info.successors.len()].into();
                })
                .expect("seed sentinel masks")
                .expect("graph node");
        }

        let error =
            try_populate_action_masks_from_exact_raw_fps(&mut checker, &checks, &[], &[true], &[])
                .expect_err("non-boolean ENABLED action must fail");
        assert!(error.to_string().contains("BOOLEAN"));
        for node in &nodes {
            let info = checker
                .graph()
                .get_node_info(node)
                .expect("sentinel graph node");
            assert!(info.state_check_mask.get(8));
            assert!(info.action_check_masks.iter().all(|mask| mask.get(9)));
        }
        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 20),
            Some(true)
        );
        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 21),
            None
        );

        super::super::set_enabled_cache(source.fingerprint(), 21, true);
        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("cached retry must succeed"));

        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert!(!info.state_check_mask.get(8));
        assert_eq!(info.action_check_masks.len(), 3);
        assert!(info.action_check_masks.get(0).unwrap().get(0));
        assert!(!info.action_check_masks.get(1).unwrap().get(0));
        assert!(info.action_check_masks.get(2).unwrap().get(0));
        super::super::clear_enabled_cache();
    }

    #[test]
    fn exact_raw_enabled_prefill_enumerates_action_beyond_next_edges() {
        super::super::clear_enabled_cache();
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        checker.set_exact_raw_fp_leaf_fast_path_allowed(true);
        let source = State::from_pairs([("x", Value::int(0))]);
        let source_for_next = source.clone();
        let mut next = move |state: &State| {
            if state == &source_for_next {
                Ok(vec![source_for_next.clone()])
            } else {
                Ok(Vec::new())
            }
        };
        checker
            .explore_state_graph_direct(std::slice::from_ref(&source), &mut next)
            .expect("exact-raw self-loop fixture");

        let x = spanned(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ));
        let x_prime = spanned(Expr::Prime(Box::new(x.clone())));
        let x_plus_one = spanned(Expr::Add(
            Box::new(x),
            Box::new(spanned(Expr::Int(1.into()))),
        ));
        let increment = Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(x_plus_one))));
        let checks = vec![LiveExpr::not(LiveExpr::enabled(increment, 30))];

        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("authoritative ENABLED prefill"));
        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 30),
            Some(true),
            "ENABLED(x' = x + 1) is true even when exact Next adjacency contains only x -> x"
        );
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert_eq!(info.action_check_masks.len(), 1);
        assert!(!info.action_check_masks.get(0).unwrap().get(0));
        super::super::clear_enabled_cache();
    }

    #[test]
    fn exact_raw_masks_prefill_missing_disabled_result() {
        let (mut checker, source, _) = owned_checker_with_duplicate_edges();
        super::super::clear_enabled_cache();
        let enabled = LiveExpr::enabled(Arc::new(spanned(Expr::Bool(false))), 31);
        let checks = vec![LiveExpr::and(vec![
            LiveExpr::not(enabled),
            LiveExpr::state_changed(None, 32),
        ])];
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);

        assert!(try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &checks,
            &[],
            &[true],
            &[],
        )
        .expect("missing ENABLED cache is populated"));

        assert_eq!(
            super::super::get_enabled_cached(source.fingerprint(), 31),
            Some(false)
        );
        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert_eq!(info.action_check_masks.len(), 3);
        assert!(info.action_check_masks.get(0).unwrap().get(0));
        assert!(!info.action_check_masks.get(1).unwrap().get(0));
        assert!(info.action_check_masks.get(2).unwrap().get(0));
        super::super::clear_enabled_cache();
    }

    #[test]
    fn exact_raw_masks_reject_missing_payload_before_mutation() {
        let (mut checker, source, changed) = owned_checker_with_duplicate_edges();
        let checks = vec![LiveExpr::state_changed(None, 13)];
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        checker
            .graph
            .update_node_info(&source_node, |info| {
                info.action_check_masks =
                    vec![CheckMask::from_indices(&[9]); info.successors.len()].into();
            })
            .expect("seed sentinel masks")
            .expect("source node");
        checker.remove_owned_exact_state_for_test(changed.fingerprint());

        let error =
            try_populate_action_masks_from_exact_raw_fps(&mut checker, &checks, &[], &[true], &[])
                .expect_err("missing exact payload must fail closed");
        assert!(error.to_string().contains("missing endpoint payload"));

        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert!(info.action_check_masks.iter().all(|mask| mask.get(9)));
    }

    #[test]
    fn exact_raw_masks_reject_missing_source_adjacency_before_mutation() {
        let (mut checker, source, _) = owned_checker_with_duplicate_edges();
        let checks = vec![LiveExpr::state_changed(None, 15)];
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        checker
            .graph
            .update_node_info(&source_node, |info| {
                info.action_check_masks =
                    vec![CheckMask::from_indices(&[9]); info.successors.len()].into();
            })
            .expect("seed sentinel masks")
            .expect("source node");
        checker.state_successor_fps.remove(&source.fingerprint());

        let error =
            try_populate_action_masks_from_exact_raw_fps(&mut checker, &checks, &[], &[true], &[])
                .expect_err("missing exact adjacency must fail closed");
        assert!(error.to_string().contains("missing successor adjacency"));

        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert!(info.action_check_masks.iter().all(|mask| mask.get(9)));
    }

    #[test]
    fn exact_raw_masks_decline_non_owned_graph_before_mutation() {
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        let source = State::from_pairs([("x", Value::int(0))]);
        let changed = State::from_pairs([("x", Value::int(1))]);
        let source_for_next = source.clone();
        let changed_for_next = changed.clone();
        let mut next = move |state: &State| {
            if state == &source_for_next {
                Ok(vec![changed_for_next.clone()])
            } else {
                Ok(Vec::new())
            }
        };
        checker
            .explore_state_graph_direct(std::slice::from_ref(&source), &mut next)
            .expect("non-owned graph fixture");
        let source_node = super::super::BehaviorGraphNode::new(source.fingerprint(), 0);
        checker
            .graph
            .update_node_info(&source_node, |info| {
                info.action_check_masks =
                    vec![CheckMask::from_indices(&[9]); info.successors.len()].into();
            })
            .expect("seed sentinel masks")
            .expect("source node");

        assert!(!try_populate_action_masks_from_exact_raw_fps(
            &mut checker,
            &[LiveExpr::state_changed(None, 14)],
            &[],
            &[true],
            &[],
        )
        .expect("non-owned graph declines fast path"));

        let info = checker
            .graph()
            .get_node_info(&source_node)
            .expect("source node");
        assert!(info.action_check_masks.iter().all(|mask| mask.get(9)));
    }
}
