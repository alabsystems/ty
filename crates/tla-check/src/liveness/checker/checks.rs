// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Top-level liveness checking entrypoint.

use super::ea_bitmask_query::{EaEdgeCheck, SccAggregateMasks};
use super::types::CounterexampleFingerprintPath;
use super::{
    BehaviorGraphNode, CounterexamplePath, GroupedLivenessPlan, InlineCheckResults,
    LivenessChecker, LivenessResult, TirProgram,
};
use crate::error::EvalResult;
use crate::liveness::debug::liveness_profile;
use rustc_hash::FxHashSet;
use std::time::Instant;

/// Outcome of the authoritative AE re-verification of a candidate witness cycle.
///
/// Part of #liveness-fp-only-false-hold: "states unavailable" is a distinct
/// outcome from "refuted" — conflating them silently converted every genuine
/// violation into a HOLD in fingerprint-only mode.
#[derive(Debug)]
pub(super) enum WitnessAeVerdict {
    /// Every AE constraint was authoritatively confirmed on the cycle.
    Confirmed,
    /// Some AE constraint authoritatively evaluates false on the whole cycle
    /// (the candidate came from a bitmask false-positive); reject the witness.
    Refuted,
    /// A cycle node's concrete state could not be materialized; the verdict is
    /// unknown and the caller must materialize states and re-run the check.
    StatesUnavailable {
        missing_fp: crate::state::Fingerprint,
    },
}

impl LivenessChecker {
    /// Check liveness with multiple PEM disjuncts against a shared graph.
    ///
    /// Uses per-node precomputed check bitmasks (#2572) so each unique check
    /// expression is evaluated once during mask population, stored in
    /// `NodeInfo.state_check_mask` and `NodeInfo.action_check_masks`. Per-PEM
    /// allowed-edge sets are assembled via bitmask operations, and PEMs
    /// sharing the same EA signature share a single Tarjan pass.
    ///
    /// Preserves per-PEM SCC correctness from #2047 (no union over
    /// heterogeneous EA sets). AE constraints are checked per-PEM.
    ///
    /// TLC reference: `LiveWorker.java:1280-1284`.
    /// Part of #3174: Bitmask-only mode — cross-property per-tag caches removed.
    pub fn check_liveness_grouped(
        &mut self,
        plan: &GroupedLivenessPlan,
        max_fairness_tag: u32,
    ) -> LivenessResult {
        self.check_liveness_grouped_with_inline_cache(plan, max_fairness_tag, None, None)
    }

    /// Part of #3174: Bitmask-only mode — cross-property per-tag caches removed.
    pub(crate) fn check_liveness_grouped_with_inline_cache(
        &mut self,
        plan: &GroupedLivenessPlan,
        max_fairness_tag: u32,
        inline_results: Option<InlineCheckResults<'_>>,
        tir: Option<&TirProgram<'_>>,
    ) -> LivenessResult {
        // Determine which check indices are used by ANY PEM for ANY purpose
        // (EA or AE). Evaluating all referenced checks upfront enables O(1)
        // bitmask lookups during SCC constraint checking (#2364 Approach G).
        let mut action_used = vec![false; plan.check_action.len()];
        let mut state_used = vec![false; plan.check_state.len()];
        for pem in &plan.pems {
            for &i in &pem.ea_action_idx {
                if i < action_used.len() {
                    action_used[i] = true;
                }
            }
            for &i in &pem.ea_state_idx {
                if i < state_used.len() {
                    state_used[i] = true;
                }
            }
            for &i in &pem.ae_action_idx {
                if i < action_used.len() {
                    action_used[i] = true;
                }
            }
            for &i in &pem.ae_state_idx {
                if i < state_used.len() {
                    state_used[i] = true;
                }
            }
        }

        // Populate per-node check bitmasks (#2572): evaluate each check once
        // across all graph nodes/edges, store results in NodeInfo fields.
        let populate_start = Instant::now();
        if let Err(e) = self.populate_node_check_masks_with_inline_cache(
            &plan.check_action,
            &plan.check_state,
            &action_used,
            &state_used,
            max_fairness_tag,
            inline_results,
            tir,
        ) {
            return LivenessResult::EvalFailure { error: e };
        }
        // The topology is complete once masks have been populated. Compact
        // in-memory successor rows to dense-id CSR before Tarjan reaches peak
        // RSS. This operation is idempotent because authoritative witness
        // retry may repopulate masks and re-enter this method on the same
        // graph; disk-backed graphs deliberately remain record-local.
        if let Err(e) = self.graph.pack_inmemory_successors() {
            return LivenessResult::EvalFailure { error: e };
        }
        let profile = liveness_profile();
        if profile {
            eprintln!(
                "  check_liveness_grouped: populate_node_check_masks: {:.3}s",
                populate_start.elapsed().as_secs_f64()
            );
        }

        // Group PEMs by EA signature for Tarjan deduplication.
        let mut ea_groups: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> = Vec::new();
        for (pem_idx, pem) in plan.pems.iter().enumerate() {
            let mut found = false;
            for (ga, gs, group_pems) in &mut ea_groups {
                if *ga == pem.ea_action_idx && *gs == pem.ea_state_idx {
                    group_pems.push(pem_idx);
                    found = true;
                    break;
                }
            }
            if !found {
                ea_groups.push((
                    pem.ea_action_idx.clone(),
                    pem.ea_state_idx.clone(),
                    vec![pem_idx],
                ));
            }
        }

        // Per unique-EA-signature: build inline edge check from precomputed
        // bitmasks (#2704), run Tarjan once, check PEM AE constraints against SCCs.
        for (ea_action_idx, ea_state_idx, pem_indices) in &ea_groups {
            // 1. Build inline edge check from EA indices (#2704).
            // Replaces prior FxHashSet<(BGNode, BGNode)> materialization.
            let ea_check = EaEdgeCheck::new(ea_action_idx, ea_state_idx);

            // 2. Run Tarjan once for this EA signature.
            // Edge filter now reads bitmasks inline via EaEdgeCheck (#2704).
            let tarjan_start = Instant::now();
            let scc_result = if let Some(ref ec) = ea_check {
                crate::liveness::tarjan::find_sccs_with_edge_filter(
                    &self.graph,
                    &|from_info, succ_idx, _to, to_info| {
                        ec.allows_edge(from_info, succ_idx, to_info)
                    },
                )
            } else {
                self.find_sccs()
            };
            if profile {
                eprintln!(
                    "  check_liveness_grouped: tarjan: {:.3}s (sccs={})",
                    tarjan_start.elapsed().as_secs_f64(),
                    scc_result.sccs.len()
                );
            }

            if !scc_result.errors.is_empty() {
                return LivenessResult::RuntimeFailure {
                    reason: format!(
                        "Tarjan SCC algorithm invariant violation: {}",
                        scc_result.errors.join("; ")
                    ),
                };
            }

            // 3. Pre-filter SCCs once per EA group (#2456).
            //    Both triviality and promise fulfillment depend only on the
            //    EA group's allowed_edges and the shared tableau/promises,
            //    NOT on the PEM's AE constraints. Pre-filtering converts
            //    O(PEM × total_SCCs × (triviality + promises)) to
            //    O(total_SCCs × (triviality + promises)).
            let prefilter_start = Instant::now();
            let candidate_sccs: Vec<&crate::liveness::tarjan::Scc> = {
                let mut result = Vec::new();
                for scc in &scc_result.sccs {
                    match self.is_trivial_scc_with_ea_check(scc, ea_check.as_ref()) {
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(e) => {
                            return LivenessResult::RuntimeFailure {
                                reason: format!("error checking SCC triviality: {e}"),
                            }
                        }
                    }
                    // Promise fulfillment is PEM-independent (uses self.promises).
                    match self.scc_fulfills_promises(scc) {
                        Ok(true) => result.push(scc),
                        Ok(false) => {}
                        Err(e) => {
                            return LivenessResult::RuntimeFailure {
                                reason: format!("error checking SCC promises: {e}"),
                            }
                        }
                    }
                }
                result
            };

            if profile {
                eprintln!(
                    "  check_liveness_grouped: prefilter_sccs: {:.3}s (candidates={}/{})",
                    prefilter_start.elapsed().as_secs_f64(),
                    candidate_sccs.len(),
                    scc_result.sccs.len()
                );
            }

            // Skip all PEMs if no candidate SCCs exist.
            if candidate_sccs.is_empty() {
                continue;
            }

            // Precompute per-SCC aggregate bitmasks for O(1) AE constraint checks.
            // The aggregate state mask is the union of all nodes' state_check_mask.
            // The aggregate action mask is the union of all intra-SCC edges'
            // action_check_masks. When a required AE bit is absent from the
            // aggregate, the SCC cannot satisfy that constraint for ANY PEM,
            // allowing early skip without per-node iteration.
            let scc_aggregates: Vec<SccAggregateMasks> = match Self::try_build_scc_aggregates(
                &candidate_sccs,
                ea_check.as_ref(),
                &self.graph,
            ) {
                Ok(agg) => agg,
                Err(error) => {
                    return LivenessResult::RuntimeFailure {
                        reason: format!("error building SCC aggregate masks: {error}"),
                    }
                }
            };

            // 4. For each PEM sharing this EA signature, check AE constraints
            //    against only the candidate SCCs using per-node bitmasks (#2572).
            //    No expression evaluation or HashMap lookup in this loop.
            for &pem_idx in pem_indices.iter() {
                let pem = &plan.pems[pem_idx];

                // Pre-build required masks for this PEM's AE constraints.
                // O(1) contains_all checks replace O(scc_size) per-node scans
                // when the aggregate mask proves the constraint is unsatisfiable.
                let required_ae_state = super::CheckMask::from_indices(&pem.ae_state_idx);
                let required_ae_action = super::CheckMask::from_indices(&pem.ae_action_idx);

                for (scc_idx, scc) in candidate_sccs.iter().enumerate() {
                    let agg = &scc_aggregates[scc_idx];

                    // Fast aggregate check: if the SCC's union of all state masks
                    // doesn't cover all required AE state bits, skip immediately.
                    // This avoids O(scc_size) per-node iteration when a fairness
                    // action is disabled in all states of the SCC.
                    if !pem.ae_state_idx.is_empty()
                        && !agg.state_mask.contains_all(&required_ae_state)
                    {
                        continue;
                    }

                    // Fast aggregate check for AE action constraints.
                    if !pem.ae_action_idx.is_empty()
                        && !agg.action_mask.contains_all(&required_ae_action)
                    {
                        continue;
                    }

                    // Aggregate passed — the SCC *might* satisfy the AE constraints.
                    // For AE state, the aggregate is exact (each bit is existential
                    // per check index), so no per-node fallback needed.
                    // For AE action, the aggregate is also exact: each bit in the
                    // aggregate means at least one intra-SCC edge has that bit set.
                    // Both try_scc_ae_state_satisfied and try_scc_ae_action_satisfied
                    // do per-index existential checks, which the aggregate already
                    // answers. Skip straight to witness construction.
                    //
                    // Note: The aggregate check is equivalent to the per-node check
                    // because both are existential (exists node with bit set). The
                    // aggregate union captures exactly this — if bit i is set in the
                    // aggregate, at least one node/edge has it set.

                    // Pass PEM indices for bitmask-based witness finding (#2572).
                    // No constraints installation needed — witness construction
                    // uses precomputed bitmasks instead of expression evaluation.
                    let mut cycle_nodes: Vec<BehaviorGraphNode> = match self
                        .build_witness_cycle_in_scc(
                            scc,
                            ea_check.as_ref(),
                            &pem.ae_state_idx,
                            &pem.ae_action_idx,
                        ) {
                        Ok(Some(cycle)) => cycle,
                        Ok(None) => continue,
                        Err(e) => {
                            return LivenessResult::RuntimeFailure {
                                reason: format!("error constructing counterexample cycle: {e}"),
                            }
                        }
                    };

                    // Soundness gate (#liveness-wf): re-verify the candidate witness
                    // cycle against the AUTHORITATIVE interpreter before reporting a
                    // violation. The per-edge/per-node check bitmasks are populated
                    // during BFS from the *explored* successor set, and ENABLED of a
                    // subscripted fairness action (`WF_e(A)`/`SF_e(A)`) computed over
                    // that set can be spuriously `false` (the action's witness
                    // successor may be absent from the recorded slice, or the inline
                    // evaluator's next-state caches may be polluted). A spurious
                    // `~ENABLED = true` makes the WF/SF AE-action constraint trivially
                    // satisfiable, yielding an UNSOUND liveness counterexample
                    // (e.g. SingleLaneBridge, where TLC proves the property HOLDS).
                    //
                    // Here we recompute each AE constraint directly with
                    // `eval_live_check_expr`, which resolves ENABLED via the post-BFS
                    // behavior-graph successors (the complete, authoritative set used
                    // by consistency checking). The witness is only accepted if every
                    // AE-state constraint holds at some cycle node and every AE-action
                    // constraint holds on some cycle edge: a bitmask false-positive is
                    // rejected, so we never report a violation for a cycle that is not
                    // genuinely fairness-satisfying.
                    //
                    // SOUNDNESS (both directions, #liveness-fp-only-false-hold): the
                    // gate needs the cycle's CONCRETE states. In fingerprint-only mode
                    // with the inline-bitmask fast path, the behavior graph may hold no
                    // concrete states at all. Treating "state unavailable" as a refuted
                    // witness (the old behavior) silently converted EVERY genuine
                    // violation into a HOLD — a missed-violation soundness bug (e.g.
                    // `<>[](x=0)` over a fair 3-cycle reported as holding). Instead we
                    // now surface `CandidateStatesUnavailable`, and the caller must
                    // materialize the state cache and re-run; it must never map this
                    // outcome to Satisfied or Violated directly.
                    //
                    // EA SYMMETRY (#liveness-ea-gate / N5): the gate historically
                    // re-verified only AE conjuncts and was SKIPPED entirely for
                    // EA-only PEMs, so a PEM's `<>[]c` conjuncts (e.g. SF's
                    // `<>[]~ENABLED`) were trusted STRAIGHT from the same distrusted
                    // bitmasks. `witness_cycle_satisfies_pem` now authoritatively
                    // re-verifies EA conjuncts too (universally over the cycle), and
                    // the gate runs whenever the PEM has ANY EA or AE conjunct.
                    if !pem.ae_state_idx.is_empty()
                        || !pem.ae_action_idx.is_empty()
                        || !pem.ea_state_idx.is_empty()
                        || !pem.ea_action_idx.is_empty()
                    {
                        match self.witness_cycle_satisfies_pem(&cycle_nodes, plan, pem) {
                            Ok(WitnessAeVerdict::Confirmed) => {}
                            Ok(WitnessAeVerdict::Refuted) => {
                                // Completeness fallback (#liveness-refute-skip / N11):
                                // the bitmask-milestone candidate was authoritatively
                                // refuted, but the milestone bitmasks are fallible in
                                // the FALSE-POSITIVE direction — a DIFFERENT cycle in
                                // the SAME SCC may still genuinely satisfy this PEM's
                                // constraints. Before skipping the whole SCC, rebuild
                                // the candidate picking milestones by AUTHORITATIVE
                                // interpreter evaluation. Only when no genuinely
                                // satisfying cycle exists do we skip the SCC.
                                match self.build_witness_cycle_in_scc_authoritative(
                                    scc,
                                    ea_check.as_ref(),
                                    plan,
                                    pem,
                                ) {
                                    Ok(Some(auth_cycle)) => {
                                        match self.witness_cycle_satisfies_pem(
                                            &auth_cycle,
                                            plan,
                                            pem,
                                        ) {
                                            Ok(WitnessAeVerdict::Confirmed) => {
                                                cycle_nodes = auth_cycle;
                                            }
                                            Ok(WitnessAeVerdict::Refuted) => continue,
                                            Ok(WitnessAeVerdict::StatesUnavailable {
                                                missing_fp,
                                            }) => {
                                                return LivenessResult::CandidateStatesUnavailable {
                                                    missing_fp,
                                                };
                                            }
                                            Err(e) => {
                                                return LivenessResult::RuntimeFailure {
                                                    reason: format!(
                                                        "error re-verifying authoritative \
                                                         counterexample fairness: {e}"
                                                    ),
                                                };
                                            }
                                        }
                                    }
                                    Ok(None) => continue,
                                    Err(e) => {
                                        return LivenessResult::RuntimeFailure {
                                            reason: format!(
                                                "error constructing authoritative \
                                                 counterexample cycle: {e}"
                                            ),
                                        };
                                    }
                                }
                            }
                            Ok(WitnessAeVerdict::StatesUnavailable { missing_fp }) => {
                                return LivenessResult::CandidateStatesUnavailable { missing_fp };
                            }
                            Err(e) => {
                                return LivenessResult::RuntimeFailure {
                                    reason: format!(
                                        "error re-verifying counterexample fairness: {e}"
                                    ),
                                };
                            }
                        }
                    }

                    return self.violation_result_for_cycle(&cycle_nodes);
                }
            }
        }

        LivenessResult::Satisfied
    }

    /// Materialize the final counterexample representation for a confirmed
    /// witness cycle.
    ///
    /// Owned compact mode promises complete concrete payloads, so it must take
    /// the concrete trace path even when a probe is missing. That path resolves
    /// every fingerprint fail-closed; only modes without an owned cache may
    /// deliberately fall back to a fingerprint-only counterexample.
    pub(super) fn violation_result_for_cycle(
        &self,
        cycle_nodes: &[BehaviorGraphNode],
    ) -> LivenessResult {
        let Some(first_node) = cycle_nodes.first() else {
            return LivenessResult::RuntimeFailure {
                reason: "cannot construct a counterexample from an empty confirmed witness cycle"
                    .into(),
            };
        };

        if self.graph.has_owned_state_cache()
            || self.graph.get_state_by_fp(first_node.state_fp).is_some()
        {
            let (prefix, cycle) = match self.build_counterexample(cycle_nodes) {
                Ok(value) => value,
                Err(error) => {
                    return LivenessResult::RuntimeFailure {
                        reason: format!("error constructing counterexample trace: {error}"),
                    };
                }
            };
            return LivenessResult::Violated { prefix, cycle };
        }

        let (prefix, cycle) = match self.build_counterexample_fingerprints(cycle_nodes) {
            Ok(value) => value,
            Err(error) => {
                return LivenessResult::RuntimeFailure {
                    reason: format!("error constructing fingerprint-only counterexample: {error}"),
                };
            }
        };
        LivenessResult::ViolatedFingerprints { prefix, cycle }
    }

    /// Re-verify, with the authoritative interpreter, that a candidate witness
    /// cycle genuinely satisfies ALL of a PEM's conjuncts — both the AE
    /// (always-eventually / `[]<>`) and the EA (eventually-always / `<>[]`)
    /// constraints. See the call site for the soundness rationale (#liveness-wf,
    /// #liveness-ea-gate / N5).
    ///
    /// # Acceptance condition (formal justification)
    ///
    /// The behavior graph encodes `fairness /\ ~property` in DNF; this PEM is one
    /// clause, decomposed into `<>[]` (EA) and `[]<>` (AE) conjuncts. A lasso
    /// `prefix . cycle^omega` is a genuine counterexample for this clause iff:
    ///
    /// 1. the cycle is reachable from an initial state (guaranteed: the cycle was
    ///    found inside an SCC of the reachable behavior graph),
    /// 2. every EA conjunct `<>[]c` holds: `c` is true on EVERY edge/node of the
    ///    cycle. Tarjan ran on the EA-filtered edge set, but that filter reads the
    ///    DISTRUSTED per-node check bitmasks — a bitmask false-positive on `c`
    ///    (e.g. a spurious `~ENABLED` from an incomplete recorded successor slice)
    ///    would let an unfair cycle survive. So EA conjuncts are re-verified here
    ///    with the authoritative interpreter, UNIVERSALLY over the cycle (a finite
    ///    cycle repeated forever satisfies `<>[]c` iff `c` holds at every node /
    ///    on every real edge).
    /// 3. every AE conjunct `[]<>c` holds: `c` is true INFINITELY OFTEN, which for
    ///    a finite cycle repeated forever means `c` holds at >= 1 cycle node
    ///    (state-level `c`) or on >= 1 real cycle edge (action-level `c`).
    ///
    /// Fairness soundness is condition 3 applied to the WF/SF conjuncts: e.g.
    /// `WF_e(A)` contributes `[]<>(~ENABLED <<A>>_e \/ <<A>>_e)`, so the cycle is
    /// only weakly fair if at some point around the loop the action is disabled or
    /// taken. Note that AE constraints are EXISTENTIAL over the cycle — requiring
    /// them at EVERY node would wrongly reject genuine counterexamples (e.g.
    /// `[]<>~P` from negating `<>[]P` needs `~P` at only one cycle node); EA
    /// constraints are UNIVERSAL, exactly the reverse.
    ///
    /// Returns `Confirmed` only when every AE-state constraint holds at some node
    /// of the cycle, every AE-action constraint holds on some edge of the cycle,
    /// every EA-state constraint holds at every node, and every EA-action
    /// constraint holds on every real edge, all evaluated via
    /// `eval_live_check_expr` (which resolves ENABLED against the complete
    /// post-BFS behavior-graph successors). Returns `Refuted` when a constraint
    /// authoritatively fails on the cycle. Returns `StatesUnavailable` when a
    /// cycle node's concrete state cannot be materialized (fingerprint-only
    /// graph): the verdict is then UNKNOWN and the caller must materialize states
    /// and re-check — mapping it to either verdict here would be unsound (false
    /// VIOLATION if accepted, false HOLD if refuted; the latter was the
    /// #liveness-fp-only-false-hold bug).
    ///
    /// The cycle is the closed walk `cycle_nodes[0] -> cycle_nodes[1] -> ... ->
    /// cycle_nodes[last] -> cycle_nodes[0]`; edges are consecutive pairs plus the
    /// implicit back-edge from the last node to the first (skipped when the witness
    /// already repeats the start node at the end, to avoid a degenerate self pair).
    pub(super) fn witness_cycle_satisfies_pem(
        &mut self,
        cycle_nodes: &[BehaviorGraphNode],
        plan: &GroupedLivenessPlan,
        pem: &super::PemPlan,
    ) -> EvalResult<WitnessAeVerdict> {
        if cycle_nodes.is_empty() {
            return Ok(WitnessAeVerdict::Refuted);
        }

        // Materialize the cycle's node states once. If any node's concrete state
        // is unavailable (fingerprint-only graph without a state cache), the gate
        // cannot decide either way — report StatesUnavailable so the caller can
        // materialize states and re-run. See the doc comment for why silently
        // refuting here is unsound (systematic false HOLDs).
        let mut node_states: Vec<crate::state::State> = Vec::with_capacity(cycle_nodes.len());
        for n in cycle_nodes {
            match self.graph.get_state_by_fp(n.state_fp) {
                Some(st) => node_states.push(st.clone()),
                None if self.graph.has_owned_state_cache() => {
                    return Err(Self::behavior_graph_invariant_error(format!(
                        "owned compact cache is missing authoritative cycle payload {}",
                        n.state_fp
                    )))
                }
                None => {
                    return Ok(WitnessAeVerdict::StatesUnavailable {
                        missing_fp: n.state_fp,
                    })
                }
            }
        }

        // Soundness (#liveness-wf): re-verification ENABLED must see the COMPLETE
        // state-graph successors of every cycle node. Prepare authoritative
        // successor adjacency and clear the ENABLED cache so no stale value from
        // an incomplete earlier computation is reused. See
        // `reseed_state_successors_for_gate` for the full rationale.
        self.reseed_state_successors_for_gate(cycle_nodes)?;

        // AE-state: each required state check must hold at >= 1 cycle node.
        for &si in &pem.ae_state_idx {
            let check = plan.check_state[si].clone();
            let mut found = false;
            for st in &node_states {
                if self.eval_live_check_expr(&check, st, None, None)? {
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(WitnessAeVerdict::Refuted);
            }
        }

        // EA-state (#liveness-ea-gate / N5): each EA `<>[]c` conjunct must hold at
        // EVERY cycle node. Enforced only by the DISTRUSTED per-node bitmasks via
        // the Tarjan edge filter until now; re-verify authoritatively so a bitmask
        // false-positive on `c` (e.g. a spurious `~ENABLED` for SF's `<>[]~ENABLED`)
        // cannot yield an unsound counterexample.
        for &si in &pem.ea_state_idx {
            let check = plan.check_state[si].clone();
            for st in &node_states {
                if !self.eval_live_check_expr(&check, st, None, None)? {
                    return Ok(WitnessAeVerdict::Refuted);
                }
            }
        }

        // Build the edge list: consecutive pairs + implicit back-edge. Each pair
        // MUST be a real behavior-graph edge — `<A>_vars` is just a predicate over
        // two states, so evaluating it on a non-adjacent pair could spuriously
        // report the action as "taken" and accept an unfair cycle.
        let n = cycle_nodes.len();
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n);
        if n == 1 {
            // A stitched witness can be a single SCC node with a self-loop.
            // Include that real edge so EA-action checks do not vacuously pass
            // over an empty edge set (and AE-action checks can witness it).
            edges.push((0, 0));
        } else {
            for i in 0..n.saturating_sub(1) {
                edges.push((i, i + 1));
            }
        }
        // Implicit back-edge last -> first, unless the witness already ends where it
        // started (a degenerate [.., start] tail that would create a self pair).
        if n >= 2 && cycle_nodes[n - 1] != cycle_nodes[0] {
            edges.push((n - 1, 0));
        }
        let mut real_edges: Vec<(usize, usize)> = Vec::with_capacity(edges.len());
        for (from_i, to_i) in edges {
            let from = cycle_nodes[from_i];
            let to = cycle_nodes[to_i];
            let from_info = self.graph.try_get_node_info(&from)?.ok_or_else(|| {
                Self::behavior_graph_invariant_error(format!(
                    "authoritative cycle source node {from} is missing from the behavior graph"
                ))
            })?;
            if !from_info.successors().contains(&to) {
                return Err(Self::behavior_graph_invariant_error(format!(
                    "authoritative cycle pair {from} -> {to} is not a behavior-graph edge"
                )));
            }
            real_edges.push((from_i, to_i));
        }

        // AE-action: each required action check must hold on >= 1 real cycle edge.
        // The gate is on the rare counterexample-reporting path, so a full eval
        // cache reset per edge is affordable and removes any residual cross-edge
        // cache pollution from this authoritative re-verification.
        for &ai in &pem.ae_action_idx {
            let check = plan.check_action[ai].clone();
            let mut found = false;
            for &(from_i, to_i) in &real_edges {
                let from_st = &node_states[from_i];
                let to_st = &node_states[to_i];
                crate::eval::clear_for_run_reset();
                crate::liveness::clear_enabled_cache();
                if self.eval_live_check_expr(&check, from_st, Some(to_st), None)? {
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(WitnessAeVerdict::Refuted);
            }
        }

        // EA-action (#liveness-ea-gate / N5): each EA `<>[]<A>_v` conjunct must hold
        // on EVERY real cycle edge (UNIVERSAL, the reverse of the existential
        // AE-action check above). A single edge where the action is authoritatively
        // not taken refutes `<>[]<A>_v`.
        for &ai in &pem.ea_action_idx {
            let check = plan.check_action[ai].clone();
            for &(from_i, to_i) in &real_edges {
                let from_st = &node_states[from_i];
                let to_st = &node_states[to_i];
                crate::eval::clear_for_run_reset();
                crate::liveness::clear_enabled_cache();
                if !self.eval_live_check_expr(&check, from_st, Some(to_st), None)? {
                    return Ok(WitnessAeVerdict::Refuted);
                }
            }
        }

        Ok(WitnessAeVerdict::Confirmed)
    }

    /// Prepare authoritative successor adjacency for the given behavior-graph
    /// nodes and clear the ENABLED cache. Shared by the
    /// witness re-verification gate (`witness_cycle_satisfies_pem`) and the
    /// authoritative fallback cycle builder (`build_witness_cycle_in_scc_authoritative`).
    ///
    /// Soundness (#liveness-wf): authoritative ENABLED re-evaluation must see the
    /// COMPLETE state-graph successors of every node. Legacy and shared-cache
    /// paths can have incomplete checker-local successor data, so they are
    /// reseeded from the union of behavior-graph edges across every tableau node
    /// with the same state fingerprint.
    ///
    /// Owned compact exploration is different: it records complete successor
    /// fingerprints before tableau consistency pruning and retains every payload.
    /// Its fingerprint adjacency is therefore more authoritative than behavior
    /// edges, which can legitimately omit an inconsistent successor. Preserve
    /// that entry and remove any stale full-state entry that would otherwise take
    /// precedence during ENABLED evaluation.
    pub(super) fn reseed_state_successors_for_gate(
        &mut self,
        nodes: &[BehaviorGraphNode],
    ) -> EvalResult<()> {
        crate::liveness::clear_enabled_cache();
        let mut seeded: FxHashSet<crate::state::Fingerprint> = FxHashSet::default();
        for node in nodes {
            if !seeded.insert(node.state_fp) {
                continue;
            }
            if self.graph.has_owned_state_cache() {
                if !self.state_successor_fps.contains_key(&node.state_fp) {
                    return Err(Self::behavior_graph_invariant_error(format!(
                        "owned compact cache is missing complete successor adjacency for gate source {}",
                        node.state_fp
                    )));
                }
                self.state_successors.remove(&node.state_fp);
                continue;
            }
            let mut succ_states: Vec<crate::state::State> = Vec::new();
            let mut succ_seen: FxHashSet<crate::state::Fingerprint> = FxHashSet::default();
            for bg_node in self.graph.node_keys() {
                if bg_node.state_fp != node.state_fp {
                    continue;
                }
                let info = self.graph.try_get_node_info(&bg_node)?.ok_or_else(|| {
                    Self::behavior_graph_invariant_error(format!(
                        "gate reseed source node {bg_node} from node_keys is missing"
                    ))
                })?;
                for succ in info.successors() {
                    if succ_seen.insert(succ.state_fp) {
                        if let Some(st) = self.graph.get_state_by_fp(succ.state_fp) {
                            succ_states.push(st.clone());
                        }
                    }
                }
            }
            self.state_successors
                .insert(node.state_fp, std::sync::Arc::new(succ_states));
        }
        Ok(())
    }

    /// Build a counterexample trace from a cycle in the behavior graph
    ///
    /// Returns (prefix, cycle) where:
    /// - prefix: Path from initial state to the start of the cycle
    /// - cycle: The violating cycle itself
    pub(super) fn build_counterexample(
        &self,
        cycle_nodes: &[BehaviorGraphNode],
    ) -> EvalResult<(CounterexamplePath, CounterexamplePath)> {
        let (prefix, cycle) = self.build_counterexample_fingerprints(cycle_nodes)?;
        let prefix = self.graph.resolve_fingerprint_trace(&prefix)?;
        let cycle = self.graph.resolve_fingerprint_trace(&cycle)?;
        Ok((prefix, cycle))
    }

    /// Build a fingerprint-only counterexample from a cycle in the behavior graph.
    pub(super) fn build_counterexample_fingerprints(
        &self,
        cycle_nodes: &[BehaviorGraphNode],
    ) -> EvalResult<(CounterexampleFingerprintPath, CounterexampleFingerprintPath)> {
        if cycle_nodes.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let cycle_start = cycle_nodes[0];
        let prefix_trace = self.graph.reconstruct_fingerprint_trace(cycle_start)?;
        let mut cycle = Vec::with_capacity(cycle_nodes.len());
        for node in cycle_nodes {
            if self.graph.try_get_node_info(node)?.is_none() {
                return Err(Self::behavior_graph_invariant_error(format!(
                    "counterexample cycle references missing node {node}"
                )));
            }
            cycle.push((node.state_fp, node.tableau_idx));
        }

        let prefix = if prefix_trace.len() > 1 {
            prefix_trace[..prefix_trace.len() - 1].to_vec()
        } else {
            Vec::new()
        };

        Ok((prefix, cycle))
    }
}
