// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! SCC constraint satisfaction and witness cycle construction for liveness checking.
//!
//! Implements the algorithms that verify whether a strongly connected component
//! satisfies AE/EA constraints and promise obligations, and constructs witness
//! cycles for counterexample traces.
//!
//! ## Error propagation contract
//!
//! Functions in this module that touch `self.graph` or `self.tableau` must return
//! `EvalResult<T>` so invariant violations cannot be silently dropped.

use super::ea_bitmask_query::EaEdgeCheck;
#[cfg(test)]
use super::SccEdgeList;
use super::{BehaviorGraphNode, LivenessChecker};
use crate::error::{EvalError, EvalResult};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// A required waypoint that a witness cycle must visit: either a specific SCC
/// node (state-level milestone) or a specific intra-SCC edge (action-level
/// milestone). Selected either from precomputed bitmasks
/// ([`LivenessChecker::build_witness_cycle_in_scc`]) or by authoritative
/// interpreter evaluation
/// ([`LivenessChecker::build_witness_cycle_in_scc_authoritative`]), then stitched
/// into a closed cycle by
/// [`LivenessChecker::stitch_cycle_through_milestones`].
#[derive(Clone, Copy)]
pub(super) enum Milestone {
    Node(BehaviorGraphNode),
    Edge(BehaviorGraphNode, BehaviorGraphNode),
}

impl LivenessChecker {
    /// Return an internal error for a behavior graph node that should exist but doesn't.
    ///
    /// Every SCC node must be present in the behavior graph — this is enforced by
    /// `add_successor()` which inserts target nodes on first encounter. A missing
    /// node indicates a graph construction bug that must not be silently ignored.
    fn missing_graph_node_error(context: &str, node: &BehaviorGraphNode) -> EvalError {
        EvalError::Internal {
            message: format!(
                "behavior graph invariant violated: {context} — node {node} missing from graph"
            ),
            span: None,
        }
    }

    /// Return an internal error for a tableau node index that should exist but doesn't.
    fn missing_tableau_node_error(context: &str, node: &BehaviorGraphNode) -> EvalError {
        EvalError::Internal {
            message: format!(
                "tableau invariant violated: {context} — tableau node {} missing for behavior node {node}",
                node.tableau_idx
            ),
            span: None,
        }
    }

    /// Resolve a state used by the authoritative alternative-cycle scan.
    ///
    /// Legacy/shared modes historically tolerate absent concrete payloads and
    /// can fall back to fingerprint-only handling. Owned compact mode promises
    /// a complete checker-owned payload table, so absence there is an invariant
    /// failure and must not silently erase a genuine milestone.
    fn authoritative_fallback_state_by_fp(
        &self,
        fp: crate::state::Fingerprint,
        context: &str,
    ) -> EvalResult<Option<crate::state::State>> {
        let state = self.graph.get_state_by_fp(fp);
        if self.graph.has_owned_state_cache() && state.is_none() {
            return Err(EvalError::Internal {
                message: format!(
                    "owned compact cache invariant violated: {context} is missing payload {fp}"
                ),
                span: None,
            });
        }
        Ok(state)
    }

    /// Build the edge list for an SCC, filtering by membership and allowed_edges.
    /// Compute once per SCC and reuse across PEM checks to avoid redundant State clones.
    ///
    /// Returns an error if any SCC node is missing from the behavior graph, which
    /// would indicate a graph construction bug (invariant violation).
    ///
    /// Note: No longer used in production (#2572 — witness construction uses bitmask
    /// lookups). Retained for tests that verify error propagation from missing nodes.
    #[cfg(test)]
    pub(super) fn build_scc_edges(
        &self,
        scc: &super::super::tarjan::Scc,
        allowed_edges: Option<&FxHashSet<(BehaviorGraphNode, BehaviorGraphNode)>>,
    ) -> EvalResult<SccEdgeList> {
        let scc_set = scc.build_node_set();
        scc.nodes()
            .iter()
            .map(|from| {
                let from_info = self.graph.try_get_node_info(from)?.ok_or_else(|| {
                    Self::missing_graph_node_error("build_scc_edges: node info", from)
                })?;
                let from_state = self
                    .graph
                    .get_state(from)
                    .ok_or_else(|| Self::missing_graph_node_error("build_scc_edges: state", from))?
                    .clone();
                let valid_successors: Vec<_> = from_info
                    .successors()
                    .iter()
                    .filter(|to| {
                        scc_set.contains(*to)
                            && allowed_edges.map_or(true, |ae| ae.contains(&(*from, **to)))
                    })
                    .map(|to| {
                        let state = self.graph.get_state(to).ok_or_else(|| {
                            Self::missing_graph_node_error("build_scc_edges: successor state", to)
                        })?;
                        Ok((*to, state.clone()))
                    })
                    .collect::<EvalResult<Vec<_>>>()?;
                Ok((*from, from_state, valid_successors))
            })
            .collect::<EvalResult<SccEdgeList>>()
    }

    /// Build a witness cycle through an SCC that visits all required milestones.
    ///
    /// Uses precomputed bitmasks (#2572) for AE state and action witness finding,
    /// eliminating expression re-evaluation and State cloning. The bitmasks were
    /// populated by `populate_node_check_masks` and stored in `NodeInfo`.
    ///
    /// # Arguments
    /// * `ae_state_idx` - PEM's AE state check indices into `check_state`
    /// * `ae_action_idx` - PEM's AE action check indices into `check_action`
    pub(super) fn build_witness_cycle_in_scc(
        &self,
        scc: &super::super::tarjan::Scc,
        ea_check: Option<&EaEdgeCheck>,
        ae_state_idx: &[usize],
        ae_action_idx: &[usize],
    ) -> EvalResult<Option<Vec<BehaviorGraphNode>>> {
        if scc.is_empty() {
            return Ok(None);
        }

        // Self-loop SCC.
        if scc.len() == 1 {
            let node = scc.nodes()[0];
            let has_self_loop = {
                let info = self.graph.try_get_node_info(&node)?.ok_or_else(|| {
                    Self::missing_graph_node_error("build_witness_cycle: self-loop check", &node)
                })?;
                let edge_allowed = if let Some(ec) = ea_check {
                    ec.try_allows_edge_pair(&self.graph, &node, &node)?
                } else {
                    true
                };
                info.successors().contains(&node) && edge_allowed
            };
            return Ok(has_self_loop.then_some(vec![node, node]));
        }

        let scc_set = scc.build_node_set();

        // Promise witnesses (tableau-based; identical for bitmask and
        // authoritative milestone selection).
        let Some(mut milestones) = self.collect_promise_milestones(scc)? else {
            return Ok(None);
        };

        // AEState witnesses via precomputed bitmasks (#2572).
        // Each ae_state_idx entry is an index into the plan's check_state array.
        // The corresponding bit in NodeInfo.state_check_mask tells us if the check
        // is satisfied without re-evaluating the expression.
        for &check_idx in ae_state_idx {
            let mut found = None;
            for node in scc.nodes() {
                if let Some(info) = self.graph.try_get_node_info(node)? {
                    if info.state_check_mask.get(check_idx) {
                        found = Some(*node);
                        break;
                    }
                }
            }
            let Some(node) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Node(node));
        }

        // AEAction witnesses via precomputed bitmasks (#2572, #2890).
        // Each ae_action_idx entry is an index into the plan's check_action array.
        // The corresponding bit in NodeInfo.action_check_masks[succ_idx] tells us
        // if the check is satisfied on that edge.
        for &check_idx in ae_action_idx {
            let mut found = None;
            for node in scc.nodes() {
                let info = self.graph.try_get_node_info(node)?.ok_or_else(|| {
                    Self::missing_graph_node_error("build_witness_cycle: AE action", node)
                })?;
                for (succ_idx, succ) in info.successors().iter().enumerate() {
                    if !scc_set.contains(succ) {
                        continue;
                    }
                    if let Some(ec) = ea_check {
                        let passes = self
                            .graph
                            .try_get_node_info(succ)?
                            .is_some_and(|ti| ec.allows_edge(&info, succ_idx, &ti));
                        if !passes {
                            continue;
                        }
                    }
                    let has_bit = info
                        .action_check_masks
                        .get(succ_idx)
                        .is_some_and(|mask| mask.get(check_idx));
                    if has_bit {
                        found = Some((*node, *succ));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            let Some((from, to)) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Edge(from, to));
        }

        self.stitch_cycle_through_milestones(scc, &scc_set, ea_check, &milestones)
    }

    /// Build a witness cycle through an SCC picking AE milestones by AUTHORITATIVE
    /// interpreter evaluation instead of the precomputed check bitmasks.
    ///
    /// Completeness fallback for #liveness-refute-skip (N11): the bitmask-milestone
    /// builder [`build_witness_cycle_in_scc`] picks the FIRST SCC node/edge whose
    /// precomputed check bit is set, but those bits are fallible in the
    /// FALSE-POSITIVE direction. When the constructed cycle is authoritatively
    /// refuted by the witness gate, a DIFFERENT cycle in the SAME SCC may still
    /// genuinely satisfy the PEM's constraints (e.g. the bitmask's chosen
    /// milestone node is a false positive but another SCC node truly satisfies the
    /// check). Skipping the whole SCC in that case is an incompleteness bug (a
    /// missed violation / false HOLD). This builder re-selects each AE milestone by
    /// evaluating the check with `eval_live_check_expr` over the reseeded SCC nodes,
    /// so a genuine milestone is found whenever one exists.
    ///
    /// Milestone SELECTION is authoritative here; the returned cycle is still fed
    /// through `witness_cycle_satisfies_pem` by the caller for the final soundness
    /// verdict (including EA re-verification).
    pub(super) fn build_witness_cycle_in_scc_authoritative(
        &mut self,
        scc: &super::super::tarjan::Scc,
        ea_check: Option<&EaEdgeCheck>,
        plan: &super::GroupedLivenessPlan,
        pem: &super::PemPlan,
    ) -> EvalResult<Option<Vec<BehaviorGraphNode>>> {
        if scc.is_empty() {
            return Ok(None);
        }
        // Single-node SCC: the only possible cycle is the self-loop `[node, node]`,
        // which the caller already re-verified authoritatively via the gate before
        // reaching this fallback. If that was refuted, no alternative cycle exists
        // in a one-node SCC, so there is nothing new to try.
        if scc.len() == 1 {
            return Ok(None);
        }

        // Preamble (#liveness-wf): materialize + reseed successors + clear the
        // ENABLED cache over the SCC nodes so authoritative ENABLED evaluation sees
        // the complete post-BFS successor set (same rationale as
        // `witness_cycle_satisfies_pem`).
        let scc_nodes: Vec<BehaviorGraphNode> = scc.nodes().to_vec();
        self.reseed_state_successors_for_gate(&scc_nodes)?;

        let scc_set = scc.build_node_set();
        let Some(mut milestones) = self.collect_promise_milestones(scc)? else {
            return Ok(None);
        };

        // AE-state milestones by AUTHORITATIVE evaluation (not the bitmask). Pick
        // the first SCC node where the check genuinely holds.
        for &check_idx in &pem.ae_state_idx {
            let check = plan.check_state[check_idx].clone();
            let mut found = None;
            for node in &scc_nodes {
                let Some(state) = self.authoritative_fallback_state_by_fp(
                    node.state_fp,
                    "authoritative AE-state source",
                )?
                else {
                    continue;
                };
                if self.eval_live_check_expr(&check, &state, None, None)? {
                    found = Some(*node);
                    break;
                }
            }
            let Some(node) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Node(node));
        }

        // AE-action milestones by AUTHORITATIVE evaluation over intra-SCC,
        // EA-allowed edges. Pick the first edge where the action check genuinely
        // holds.
        for &check_idx in &pem.ae_action_idx {
            let check = plan.check_action[check_idx].clone();
            let mut found = None;
            'nodes: for node in &scc_nodes {
                let Some(from_state) = self.authoritative_fallback_state_by_fp(
                    node.state_fp,
                    "authoritative AE-action source",
                )?
                else {
                    continue;
                };
                let succ_list: Vec<BehaviorGraphNode> = self
                    .graph
                    .try_get_node_info(node)?
                    .ok_or_else(|| {
                        Self::missing_graph_node_error("authoritative AE-action source info", node)
                    })?
                    .successors()
                    .iter()
                    .copied()
                    .filter(|s| scc_set.contains(s))
                    .collect();
                for succ in succ_list {
                    if let Some(ec) = ea_check {
                        if !ec.try_allows_edge_pair(&self.graph, node, &succ)? {
                            continue;
                        }
                    }
                    let Some(to_state) = self.authoritative_fallback_state_by_fp(
                        succ.state_fp,
                        "authoritative AE-action destination",
                    )?
                    else {
                        continue;
                    };
                    crate::eval::clear_for_run_reset();
                    crate::liveness::clear_enabled_cache();
                    if self.eval_live_check_expr(&check, &from_state, Some(&to_state), None)? {
                        found = Some((*node, succ));
                        break 'nodes;
                    }
                }
            }
            let Some((from, to)) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Edge(from, to));
        }

        self.stitch_cycle_through_milestones(scc, &scc_set, ea_check, &milestones)
    }

    /// Collect the promise-fulfilment milestones for an SCC (tableau-based).
    ///
    /// Returns `None` if any promise cannot be fulfilled anywhere in the SCC (the
    /// SCC then cannot host a witness cycle for this tableau). Shared by both the
    /// bitmask and authoritative milestone builders so the two never diverge.
    pub(super) fn collect_promise_milestones(
        &self,
        scc: &super::super::tarjan::Scc,
    ) -> EvalResult<Option<Vec<Milestone>>> {
        let mut milestones: Vec<Milestone> = Vec::new();
        for promise in &self.promises {
            let mut found = None;
            for node in scc.nodes() {
                let tnode = self.tableau.node(node.tableau_idx).ok_or_else(|| {
                    Self::missing_tableau_node_error("build_witness_cycle: promise witness", node)
                })?;
                if tnode.particle().is_fulfilling(promise) {
                    found = Some(*node);
                    break;
                }
            }
            let Some(node) = found else {
                return Ok(None);
            };
            milestones.push(Milestone::Node(node));
        }
        Ok(Some(milestones))
    }

    /// Stitch a sequence of milestones into a closed witness cycle within an SCC.
    ///
    /// Picks a start node (preferring the first milestone), routes through every
    /// milestone in order via `find_path_within_scc` (EA-filtered), and closes the
    /// loop back to the start. Returns `None` if any required segment is
    /// unreachable under the EA edge filter. Shared by the bitmask and
    /// authoritative milestone builders.
    pub(super) fn stitch_cycle_through_milestones(
        &self,
        scc: &super::super::tarjan::Scc,
        scc_set: &FxHashSet<BehaviorGraphNode>,
        ea_check: Option<&EaEdgeCheck>,
        milestones: &[Milestone],
    ) -> EvalResult<Option<Vec<BehaviorGraphNode>>> {
        // Pick a start node (prefer a milestone).
        let start = milestones
            .iter()
            .map(|m| match m {
                Milestone::Node(n) => *n,
                Milestone::Edge(from, _) => *from,
            })
            .next()
            .unwrap_or_else(|| scc.nodes()[0]);

        let mut cycle = vec![start];
        let mut current = start;

        for milestone in milestones {
            match *milestone {
                Milestone::Node(target) => {
                    let Some(path) =
                        self.find_path_within_scc(current, target, scc_set, ea_check)?
                    else {
                        return Ok(None);
                    };
                    cycle.extend(path.into_iter().skip(1));
                    current = target;
                }
                Milestone::Edge(from, to) => {
                    let Some(path) = self.find_path_within_scc(current, from, scc_set, ea_check)?
                    else {
                        return Ok(None);
                    };
                    cycle.extend(path.into_iter().skip(1));

                    if let Some(ec) = ea_check {
                        if !ec.try_allows_edge_pair(&self.graph, &from, &to)? {
                            return Ok(None);
                        }
                    }
                    cycle.push(to);
                    current = to;
                }
            }
        }

        // Close the cycle by finding a path from current back to start.
        // Since start is already at the beginning of the cycle, we DON'T include it again.
        // The cycle should be [start, ..., last_node] where last_node -> start is the implicit back-edge.
        let Some(back_path) = self.find_path_within_scc(current, start, scc_set, ea_check)? else {
            return Ok(None);
        };
        // Skip first (current, already in cycle) and last (start, already at beginning)
        let back_len = back_path.len();
        if back_len > 2 {
            cycle.extend(back_path.into_iter().skip(1).take(back_len - 2));
        }
        // If back_path is just [current, start] (length 2), we add nothing - current already has edge to start

        Ok(Some(cycle))
    }

    pub(super) fn find_path_within_scc(
        &self,
        start: BehaviorGraphNode,
        goal: BehaviorGraphNode,
        scc_set: &FxHashSet<BehaviorGraphNode>,
        ea_check: Option<&EaEdgeCheck>,
    ) -> EvalResult<Option<Vec<BehaviorGraphNode>>> {
        // Special case: start == goal
        // We need to find either a self-loop OR a non-trivial cycle back to start
        if start == goal {
            // Check if there's a self-loop edge
            let start_info = self.graph.try_get_node_info(&start)?.ok_or_else(|| {
                Self::missing_graph_node_error("find_path_within_scc: start node info", &start)
            })?;
            let edge_allowed = if let Some(ec) = ea_check {
                ec.try_allows_edge_pair(&self.graph, &start, &start)?
            } else {
                true
            };
            let has_self_loop = start_info.successors().contains(&start) && edge_allowed;
            if has_self_loop {
                return Ok(Some(vec![start]));
            }
            // No self-loop - must find a non-trivial cycle through other SCC nodes
            // Fall through to BFS, but DON'T mark goal as visited initially so we can find it
        }

        // When start == goal, we need to find a non-trivial cycle.
        // Don't mark goal as visited initially so we can reach it through other nodes.
        let finding_cycle = start == goal;

        let mut visited: FxHashSet<BehaviorGraphNode> = FxHashSet::default();
        let mut parent: FxHashMap<BehaviorGraphNode, BehaviorGraphNode> = FxHashMap::default();
        let mut queue: VecDeque<BehaviorGraphNode> = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        let mut found_goal = false;
        while let Some(node) = queue.pop_front() {
            let info = self.graph.try_get_node_info(&node)?.ok_or_else(|| {
                Self::missing_graph_node_error("find_path_within_scc: BFS node info", &node)
            })?;

            for succ in info.successors() {
                if !scc_set.contains(succ) {
                    continue;
                }
                // For cycle finding, allow reaching goal (which is start) even though it's in visited
                let is_goal = *succ == goal;
                if !is_goal && visited.contains(succ) {
                    continue;
                }
                if let Some(ec) = ea_check {
                    if !ec.try_allows_edge_pair(&self.graph, &node, succ)? {
                        continue;
                    }
                }

                if !is_goal {
                    visited.insert(*succ);
                }
                parent.insert(*succ, node);

                if is_goal {
                    found_goal = true;
                    break;
                }
                queue.push_back(*succ);
            }

            if found_goal {
                break;
            }
        }

        if !found_goal && !finding_cycle {
            // For non-cycle case, check if goal was visited
            if !visited.contains(&goal) {
                return Ok(None);
            }
        } else if !found_goal {
            return Ok(None);
        }

        let mut path = Vec::new();
        let mut cur = goal;
        path.push(cur);
        while let Some(&p) = parent.get(&cur) {
            path.push(p);
            if p == start {
                break;
            }
            cur = p;
        }
        path.reverse();
        Ok(Some(path))
    }

    pub(super) fn is_trivial_scc_with_ea_check(
        &self,
        scc: &super::super::tarjan::Scc,
        ea_check: Option<&EaEdgeCheck>,
    ) -> EvalResult<bool> {
        if scc.len() != 1 {
            return Ok(false);
        }
        let node = scc.nodes()[0];
        let info = self
            .graph
            .try_get_node_info(&node)?
            .ok_or_else(|| Self::missing_graph_node_error("is_trivial_scc: node info", &node))?;
        let edge_allowed = if let Some(ec) = ea_check {
            ec.try_allows_edge_pair(&self.graph, &node, &node)?
        } else {
            true
        };
        let has_self_loop = info.successors().contains(&node) && edge_allowed;
        Ok(!has_self_loop)
    }

    pub(super) fn scc_fulfills_promises(
        &self,
        scc: &super::super::tarjan::Scc,
    ) -> crate::error::EvalResult<bool> {
        for promise in &self.promises {
            let mut fulfilled_somewhere = false;
            for node in scc.nodes() {
                let tnode = self.tableau.node(node.tableau_idx).ok_or_else(|| {
                    Self::missing_tableau_node_error("scc_fulfills_promises", node)
                })?;
                if tnode.particle().is_fulfilling(promise) {
                    fulfilled_somewhere = true;
                    break;
                }
            }
            if !fulfilled_somewhere {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
