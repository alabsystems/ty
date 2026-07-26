// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! IC3 proof obligation blocking: block_all, block_one, and related helpers
//! (reduce_cube_from_core, find_blocking_frame, dynamic_ctg_params).

use super::config::{
    consecution_verify_interval_full, Ic3Result, MAX_CROSSCHECK_FAILURES, MAX_OBLIGATION_DEPTH,
    MAX_SOLVER_REBUILDS_PER_FRAME, MAX_SPURIOUS_INIT_PREDS, MAX_TOTAL_CROSSCHECK_FAILURES,
    MAX_UNKNOWN_REQUEUES, UNKNOWN_FALLBACK_THRESHOLD,
};
use super::engine::Ic3Engine;
use super::frame::Lemma;
use super::obligation::ProofObligation;
use super::validate::LemmaVerdict;
use crate::sat_types::{Lit, SatResult, SatSolver, SolverBackend, Var};

impl Ic3Engine {
    /// Process all proof obligations.
    pub(super) fn block_all(&mut self) -> Result<(), Ic3Result> {
        let max_frame = self.frames.depth();
        while let Some(po) = self.obligations.pop(max_frame) {
            if self.is_cancelled() {
                return Err(Ic3Result::Unknown {
                    reason: "cancelled".into(),
                });
            }
            // Skip obligations whose cubes are already blocked by existing lemmas.
            // This check is cheap (subsumption test on sorted lemma lists) and avoids
            // redundant SAT calls for cubes that were blocked by a generalized lemma
            // from a prior iteration. Note: we still go through block_one for frame-0
            // obligations (counterexample verification) even if "blocked" by the frame
            // data structure, because the frame check is an over-approximation.
            if po.frame > 0 && self.frames.is_blocked(po.frame, &po.cube) {
                continue;
            }
            self.block_one(po)?;
        }
        Ok(())
    }

    /// Try to block a single proof obligation.
    pub(super) fn block_one(&mut self, mut po: ProofObligation) -> Result<(), Ic3Result> {
        // Depth limit: if the obligation chain is too deep, give up on this
        // branch. The BMC engine is better suited for deep counterexamples.
        // This prevents runaway depth explosion on circuits with very long
        // reachability chains (e.g., large shift registers, counters).
        if po.depth > MAX_OBLIGATION_DEPTH {
            return Ok(());
        }

        // Early init-subsumption skip (#4074): if the PO's cube is truly consistent
        // with Init AND the PO is at a frame > 0, skip it entirely.
        //
        // Rationale: Init states are always reachable (they're in F_0 by definition),
        // so they can never be blocked by IC3 lemmas. A cube consistent with Init
        // at frame k > 0 means the consecution check would find a predecessor
        // (since the init state transitions to itself or another reachable state),
        // creating an obligation at frame k-1, which eventually descends to frame 0
        // where verify_trace fails (spurious) or succeeds (real CEX).
        //
        // CONVERGENCE FIX (#4104): Use precise SAT-based check instead of the fast
        // over-approximation.
        if po.frame > 0 && self.cube_sat_consistent_with_init(&po.cube) {
            if std::env::var("IC3_DEBUG").is_ok() {
                eprintln!(
                    "IC3 skip_init: frame={} depth={} cube_len={} (init-consistent at frame>0)",
                    po.frame,
                    po.depth,
                    po.cube.len(),
                );
            }
            return Ok(());
        }

        // Trivial containment check (#4074):
        // If the cube is already blocked by a lemma at this frame or higher,
        // push the PO up to the frame where it's blocked instead of doing
        // a redundant SAT check.
        if po.frame > 0 {
            if let Some(higher_frame) = self.find_blocking_frame(po.frame, &po.cube) {
                if higher_frame > po.frame && higher_frame < self.frames.depth() {
                    po.frame = higher_frame + 1;
                    self.obligations.push(po);
                }
                // If blocked at po.frame exactly, just skip (already handled).
                return Ok(());
            }
        }

        if po.frame == 0 {
            // CONVERGENCE FIX (#4074): Fast-path for init-inconsistent cubes.
            let init_consistent = self.cube_consistent_with_init(&po.cube);
            if !init_consistent {
                if std::env::var("IC3_DEBUG").is_ok() {
                    eprintln!(
                        "IC3 block_one: frame=0 depth={} SKIP_VERIFY (init-inconsistent) cube_len={}",
                        po.depth,
                        po.cube.len(),
                    );
                }
                // Block the spurious cube at frame 0 ONLY.
                //
                // SOUNDNESS FIX (#4092): Init-inconsistent cubes are only guaranteed
                // unreachable from Init (frame 0), NOT from all frames. Adding them
                // to all frame solvers is unsound: a cube like {v2} (v2=true) is
                // init-inconsistent (Init has v2=0), but v2=true IS reachable at
                // higher frames through the transition relation.
                //
                // The original code `for s in &mut self.solvers { s.add_clause(...) }`
                // caused false UNSAT on circuits like the 3-deep shift register:
                // blocking {v2} at frame 0 added [~v2] to ALL solvers, which falsely
                // constrained higher frames to never have v2=true.
                let neg_cube: Vec<Lit> = po.cube.iter().map(|l| !*l).collect();
                let lemma = Lemma::from_blocked_cube(&po.cube);
                self.frames.add_lemma(0, lemma.clone());
                if !self.solvers.is_empty() {
                    self.solvers[0].add_clause(&neg_cube);
                }
                return Ok(());
            }

            // Cube is init-consistent: might be a real counterexample.
            // Verify the full counterexample trace using BMC-style unrolling.
            let trace_ok = self.verify_trace(&po);
            if std::env::var("IC3_DEBUG").is_ok() {
                eprintln!(
                    "IC3 block_one: frame=0 depth={} verify_trace={} cube_len={} cube_init_consistent=true",
                    po.depth,
                    trace_ok,
                    po.cube.len(),
                );
            }
            if trace_ok {
                let trace = self.extract_trace(&po);
                return Err(Ic3Result::Unsafe {
                    depth: po.depth,
                    trace,
                });
            }

            // Spurious counterexample: verify_trace failed despite the fast
            // init-consistency check returning true.
            //
            // CONVERGENCE FIX (#4104): Use precise SAT-based init check.
            let truly_in_init = self.cube_sat_consistent_with_init(&po.cube);
            if !truly_in_init {
                if std::env::var("IC3_DEBUG").is_ok() {
                    eprintln!(
                        "IC3 block_one: frame=0 depth={} BLOCK_SPURIOUS (init-inconsistent via SAT) cube_len={}",
                        po.depth,
                        po.cube.len(),
                    );
                }
                // SOUNDNESS FIX (#4092): Only add to solver[0], not all solvers.
                // See comment in the init-inconsistent fast path above for full explanation.
                let neg_cube: Vec<Lit> = po.cube.iter().map(|l| !*l).collect();
                let lemma = Lemma::from_blocked_cube(&po.cube);
                self.frames.add_lemma(0, lemma.clone());
                if !self.solvers.is_empty() {
                    self.solvers[0].add_clause(&neg_cube);
                }
            } else {
                self.spurious_init_pred_count += 1;
                if std::env::var("IC3_DEBUG").is_ok() {
                    eprintln!(
                        "IC3 block_one: frame=0 depth={} REQUEUE_SUCCESSOR (truly in Init, verify_trace spurious) cube_len={} spurious_count={}",
                        po.depth,
                        po.cube.len(),
                        self.spurious_init_pred_count,
                    );
                }
                // FIX (#4105): When the predecessor is truly in Init but verify_trace
                // fails, the predecessor cube is too abstract. Re-queue the successor
                // PO so IC3 can try to block it with a different predecessor.
                //
                // LOOP BREAKER (#4105): After MAX_SPURIOUS_INIT_PREDS consecutive
                // spurious init-consistent predecessors, stop re-queuing the successor.
                // On constraint-heavy circuits (e.g., microban_1: 124 constraints,
                // 23 latches), the verify_trace check may fail systematically because
                // the partial cube from the lift solver is too abstract to reconstruct
                // a concrete trace through 124 constraints. Re-queuing just rediscovers
                // the same pattern. Dropping the successor is sound: IC3's frame
                // sequence still over-approximates reachability, and the unblocked
                // cube will be re-examined at the next depth level.
                if self.spurious_init_pred_count <= MAX_SPURIOUS_INIT_PREDS {
                    if let Some(successor) = po.next.map(|n| *n) {
                        if successor.frame > 0 {
                            self.obligations.push(successor);
                        }
                    }
                } else if std::env::var("IC3_DEBUG").is_ok() {
                    eprintln!(
                        "IC3 block_one: frame=0 depth={} DROP_SPURIOUS (spurious_count={} > {}) — \
                         stopping successor re-queue to break infinite loop (#4105)",
                        po.depth, self.spurious_init_pred_count, MAX_SPURIOUS_INIT_PREDS,
                    );
                }
            }
            return Ok(());
        }

        let solver_idx = po.frame - 1;
        // Block check WITHOUT !cube strengthening (strengthen=false).
        let assumptions = self.prime_cube(&po.cube);

        // Domain-restricted consecution (#4059, #4091).
        // Domain is computed once inside build_consecution_domain_solver and
        // returned alongside the solver to avoid double-computation (#4081).
        let used_domain_restriction;
        let result = if let Some((mut domain_solver, domain)) =
            self.build_consecution_domain_solver(po.frame, &po.cube)
        {
            used_domain_restriction = true;
            self.domain_stats
                .record(domain.len(), self.max_var as usize + 1, true);

            domain_solver.set_cancelled(self.cancelled.clone());
            let domain_result = domain_solver.solve(&assumptions);
            if domain_result == SatResult::Unsat {
                SatResult::Unsat
            } else {
                // Use the full COI domain (not just cube vars) for set_domain
                // on the frame solver fallback. The COI includes AND-gate fanin,
                // input variables, and next-state variables — all needed for
                // ay-sat's domain-restricted BCP to work correctly (#4091).
                let domain_vars: Vec<Var> = (0..=self.max_var)
                    .filter(|&i| domain.contains(Var(i)))
                    .map(Var)
                    .collect();
                if self.solvers[solver_idx].is_poisoned() {
                    if self.solver_rebuild_budget_exceeded(solver_idx) {
                        SatResult::Sat // Conservative: treat as Sat (#4105)
                    } else {
                        self.rebuild_solver_at(solver_idx);
                        // small_circuit_mode (#4259, ay#8802): skip set_domain so
                        // ay-sat uses search_propagate_standard (plain BCP).
                        if !self.config.small_circuit_mode {
                            self.solvers[solver_idx].set_domain(&domain_vars);
                        }
                        let full_result = self.solvers[solver_idx].solve(&assumptions);
                        if !self.config.small_circuit_mode {
                            self.solvers[solver_idx].clear_domain();
                        }
                        full_result
                    }
                } else {
                    // small_circuit_mode (#4259, ay#8802): skip set_domain so
                    // ay-sat uses search_propagate_standard (plain BCP).
                    if !self.config.small_circuit_mode {
                        self.solvers[solver_idx].set_domain(&domain_vars);
                    }
                    let full_result = self.solvers[solver_idx].solve(&assumptions);
                    if !self.config.small_circuit_mode {
                        self.solvers[solver_idx].clear_domain();
                    }
                    full_result
                }
            }
        } else {
            used_domain_restriction = false;
            self.domain_stats
                .record(0, self.max_var as usize + 1, false);
            if self.solvers[solver_idx].is_poisoned() {
                if self.solver_rebuild_budget_exceeded(solver_idx) {
                    SatResult::Sat // Conservative: treat as Sat (#4105)
                } else {
                    self.rebuild_solver_at(solver_idx);
                    self.solvers[solver_idx].solve(&assumptions)
                }
            } else {
                self.solvers[solver_idx].solve(&assumptions)
            }
        };

        // Track consecution query result (#4121 diagnostics).
        self.consecution_stats.total_queries += 1;
        match result {
            SatResult::Unsat => self.consecution_stats.unsat_results += 1,
            SatResult::Sat => self.consecution_stats.sat_results += 1,
            SatResult::Unknown => self.consecution_stats.unknown_results += 1,
        }
        if used_domain_restriction {
            self.consecution_stats.domain_restricted += 1;
        } else {
            self.consecution_stats.full_solver += 1;
        }

        match result {
            SatResult::Unsat => {
                self.unknown_count = 0;
                // Reset spurious init-pred counter on successful block (#4105).
                // The counter only matters for consecutive spurious predecessors
                // without any progress. A successful block means IC3 is making
                // progress, so reset the counter.
                self.spurious_init_pred_count = 0;

                // Cube blocked — generalize with MIC.
                //
                // Parent lemma MIC seeding (CAV'23 #4150): when enabled and the PO
                // has a parent, extract the parent's cube and pass it to MIC. The
                // MIC function will intersect the current cube with the parent's
                // blocking lemma, producing a tighter starting point.
                let parent_cube_for_mic: Option<Vec<Lit>> = if self.config.parent_lemma_mic {
                    po.next.as_ref().map(|parent| parent.cube.clone())
                } else {
                    None
                };
                let parent_ref = parent_cube_for_mic.as_deref();

                let generalized = if self.config.parent_lemma_mic && parent_ref.is_some() {
                    // Use parent-seeded MIC variants.
                    if self.config.dynamic {
                        let (dyn_ctg_max, dyn_ctg_limit) = Self::dynamic_ctg_params(&po);
                        if self.config.multi_lift_orderings >= 2 {
                            self.mic_multi_order_with_parent_seed_params(
                                po.frame,
                                po.cube.clone(),
                                parent_ref,
                                dyn_ctg_max,
                                dyn_ctg_limit,
                            )
                        } else {
                            self.mic_with_parent_seed_params(
                                po.frame,
                                po.cube.clone(),
                                parent_ref,
                                dyn_ctg_max,
                                dyn_ctg_limit,
                            )
                        }
                    } else if self.config.multi_lift_orderings >= 2 {
                        self.mic_multi_order_with_parent_seed(po.frame, po.cube.clone(), parent_ref)
                    } else {
                        self.mic_with_parent_seed(po.frame, po.cube.clone(), parent_ref)
                    }
                } else if self.config.dynamic {
                    let (dyn_ctg_max, dyn_ctg_limit) = Self::dynamic_ctg_params(&po);
                    if self.config.multi_lift_orderings >= 2 {
                        self.mic_multi_order_with_params(
                            po.frame,
                            po.cube.clone(),
                            dyn_ctg_max,
                            dyn_ctg_limit,
                        )
                    } else {
                        self.mic_with_params(po.frame, po.cube.clone(), dyn_ctg_max, dyn_ctg_limit)
                    }
                } else if self.config.multi_lift_orderings >= 2 {
                    self.mic_multi_order(po.frame, po.cube.clone())
                } else {
                    self.mic(po.frame, po.cube.clone())
                };

                // SOUNDNESS CHECK (#4092): Refuse init-consistent lemmas.
                if self.cube_sat_consistent_with_init(&generalized) {
                    return Ok(());
                }

                // SOUNDNESS CHECK (#4092, #4121): Independent consecution verification.
                // Uses adaptive verification interval based on clause-to-latch ratio:
                // high-ratio circuits verify every consecution (interval=1), low-ratio
                // circuits sample every 10th. This catches ay-sat false UNSAT on
                // constraint-heavy circuits without excessive overhead on simple ones.
                //
                // CONVERGENCE FIX (#4105): Skip cross-check entirely when disabled.
                // On clause-heavy circuits (ratio > 5x), SimpleSolver's basic DPLL
                // without clause learning produces false SAT, causing every ay-sat
                // UNSAT result to be rejected. Disabling the cross-check and trusting
                // ay-sat (with validate_invariant_budgeted as final soundness net) is
                // the correct response for these circuits.
                if self.ts.latch_vars.len() <= 60 && !self.crosscheck_disabled {
                    self.consecution_verify_counter += 1;
                    let verify_interval = consecution_verify_interval_full(
                        self.ts.trans_clauses.len(),
                        self.ts.constraint_lits.len(),
                        self.ts.latch_vars.len(),
                    );
                    // Small-circuit fast path (#4259, #4288): verify_interval ==
                    // usize::MAX signals "skip cross-check entirely". Happens on
                    // <30 latches where SimpleSolver DPLL is unreliable on
                    // clause-dense circuits. Post-convergence validation still
                    // guards soundness. Short-circuit here so the modulo/budget
                    // machinery below is bypassed cleanly.
                    let should_verify = if verify_interval == usize::MAX {
                        false
                    } else {
                        self.consecution_verify_counter % verify_interval == 0
                    };
                    let frame_failures = self
                        .crosscheck_failures
                        .get(solver_idx)
                        .copied()
                        .unwrap_or(0);
                    // For clause-heavy circuits (verify_interval==1), use a tighter
                    // total-failure threshold to trigger the cross-check disable sooner.
                    // These circuits produce cross-check disagreements at a higher rate
                    // because SimpleSolver can't handle the constraint density (#4105).
                    let effective_total_threshold = if verify_interval == 1 {
                        MAX_TOTAL_CROSSCHECK_FAILURES / 2
                    } else {
                        MAX_TOTAL_CROSSCHECK_FAILURES
                    };
                    let needs_global_fallback =
                        self.total_crosscheck_failures >= effective_total_threshold;
                    let needs_frame_fallback = frame_failures >= MAX_CROSSCHECK_FAILURES;
                    if should_verify
                        && frame_failures < MAX_CROSSCHECK_FAILURES
                        && !needs_global_fallback
                    {
                        if !self.verify_consecution_independent(po.frame, &generalized, true) {
                            if std::env::var("IC3_DEBUG").is_ok() {
                                eprintln!(
                                    "IC3 CROSS-CHECK FAIL: frame={} cube_len={} failures={}/{} — ay-sat false UNSAT, \
                                     SimpleSolver disagrees.",
                                    po.frame,
                                    generalized.len(),
                                    frame_failures + 1,
                                    self.total_crosscheck_failures + 1,
                                );
                            }
                            if solver_idx < self.crosscheck_failures.len() {
                                self.crosscheck_failures[solver_idx] += 1;
                            }
                            self.total_crosscheck_failures += 1;
                            if let Some(pred) = self.consecution_simple_fallback(po.frame, &po.cube)
                            {
                                if self.cube_sat_consistent_with_init(&pred) {
                                    self.obligations.push(po);
                                    return Ok(());
                                }
                                self.obligations.push(ProofObligation::new(
                                    po.frame - 1,
                                    pred,
                                    po.depth + 1,
                                    Some(po.clone()),
                                ));
                                self.obligations.push(po);
                                return Ok(());
                            }
                        }
                    } else if needs_global_fallback {
                        // Global cross-check budget exhausted (#4105, #4121).
                        //
                        // On clause-heavy circuits (verify_interval==1), SimpleSolver
                        // is the problem, not ay-sat. SimpleSolver's basic DPLL without
                        // clause learning produces false SAT on constraint-dense formulas
                        // (e.g., microban_1: 124 constraints, 879 trans_clauses, 23
                        // latches). Falling back to SimpleSolver makes IC3 unable to
                        // solve anything.
                        //
                        // Instead: disable cross-checking entirely and trust ay-sat.
                        // The post-convergence validate_invariant_budgeted() provides
                        // the ultimate soundness safety net.
                        //
                        // For low-ratio circuits (verify_interval > 1), fall back to
                        // SimpleSolver as before -- those circuits are simple enough
                        // that SimpleSolver works correctly.
                        if verify_interval == 1 {
                            eprintln!(
                                "IC3: cross-check budget exhausted on clause-heavy circuit \
                                 (total={}, threshold={}, ratio={:.1}x). \
                                 Disabling cross-check, trusting ay-sat (#4105).",
                                self.total_crosscheck_failures,
                                effective_total_threshold,
                                self.ts.trans_clauses.len() as f64
                                    / self.ts.latch_vars.len().max(1) as f64,
                            );
                            self.crosscheck_disabled = true;
                        } else if self.solver_backend != SolverBackend::Simple {
                            eprintln!(
                                "IC3: global cross-check budget exhausted (total={}, threshold={}). \
                                 Falling back to SimpleSolver (#4121).",
                                self.total_crosscheck_failures,
                                effective_total_threshold,
                            );
                            self.fallback_solver_backend();
                        }
                    } else if needs_frame_fallback {
                        if std::env::var("IC3_DEBUG").is_ok() {
                            eprintln!(
                                "IC3: cross-check budget exhausted at frame {} (frame_failures={}, total={}). \
                                 Disabling cross-check for this frame.",
                                po.frame,
                                frame_failures,
                                self.total_crosscheck_failures,
                            );
                        }
                    }
                }

                let (push_frame, pushed_cube) = self.push_lemma(po.frame, generalized);
                let lemma = Lemma::from_blocked_cube(&pushed_cube);
                let target_frame = (push_frame - 1).min(self.frames.depth() - 1);

                // Per-lemma consecution verification (#4121 diagnostics).
                //
                // When IC3_VERIFY_LEMMAS was set at engine construction,
                // independently verify EVERY lemma before adding it to the frame
                // sequence. This catches ay-sat false UNSAT at the earliest possible
                // point, before unsound lemmas propagate. Expensive (doubles SAT
                // calls), but invaluable for diagnosing which benchmarks trigger
                // ay-sat bugs.
                //
                // The lemma is verified AT ITS PLACEMENT, `target_frame` (#4560).
                // Adding at delta index t claims ¬lemma holds in F_1..F_t, and
                // because the frames are nested (F_1 ⊆ … ⊆ F_t) the single
                // consecution step into F_t — over F_{t-1}, the weakest of the
                // step formulas — implies all the earlier ones. Verifying at
                // `po.frame` instead (as this code once did) checks a claim the
                // placement does not make whenever the two differ: for a root
                // obligation at frame == depth the placement clamps to
                // depth - 1, and the po.frame check demands the lemma also
                // survive one step PAST its placement — refuting perfectly
                // sound lemmas (e.g. shift registers whose bad state is exactly
                // one frame deeper than the current unrolling). A placement of
                // 0 claims only init-disjointness, which the #4092 check above
                // already established — nothing further to verify.
                //
                // When not set, this code is a no-op and the existing cross-check
                // + validate_invariant_budgeted provide the soundness net.
                if target_frame >= 1 && self.verify_lemmas {
                    match self.verify_lemma_consecution(target_frame, &pushed_cube) {
                        LemmaVerdict::Verified => {
                            self.consecution_stats.lemmas_verified += 1;
                        }
                        LemmaVerdict::Refuted { .. } => {
                            self.consecution_stats.lemmas_rejected += 1;
                            eprintln!(
                                "IC3 LEMMA REJECTED: frame={} target_frame={} cube_len={} \
                                 pushed_len={} total_rejected={} — independent verification \
                                 refuted the generalized lemma; falling back to the \
                                 unreduced cube",
                                po.frame,
                                target_frame,
                                po.cube.len(),
                                pushed_cube.len(),
                                self.consecution_stats.lemmas_rejected,
                            );
                            // PROGRESS GUARANTEE (#4560).
                            //
                            // The generalized lemma is refuted, so it must not be
                            // added — but the obligation must not be re-queued bare
                            // either. IC3's termination rests on every
                            // obligation-processing step doing one of two things:
                            //
                            //   (i) strengthening the frame system with a lemma
                            //       that blocks the obligation's cube, or
                            //  (ii) enqueueing a strictly-lower-frame predecessor
                            //       obligation, a descent that bottoms out at
                            //       frame 0 (real counterexample, or a spurious
                            //       one that frame 0 resolves).
                            //
                            // The old code here did neither: it re-queued the same
                            // obligation with no state change, and a deterministic
                            // engine then repeats the identical block/generalize/
                            // reject computation forever. That livelock was masked
                            // only by the (removed) drop_po heuristic, which
                            // eventually discarded the obligation — silently
                            // trading away completeness.
                            //
                            // Progress is restored soundly in two stages:
                            //
                            // 1. Fall back to the UNREDUCED cube: ¬po.cube is the
                            //    weakest lemma that still blocks this obligation
                            //    (route (i)), and it is exactly what the primary
                            //    consecution UNSAT above justifies — only MIC's
                            //    literal drops and push_lemma's core shrinking are
                            //    forfeited, i.e. the parts the refutation actually
                            //    called into question. Its natural placement is
                            //    po.frame (the frames are nested, so the step into
                            //    F_{po.frame} covers every earlier step), clamped
                            //    to the frame array like the normal path.
                            //
                            // 2. Verify that fallback too. If even the unreduced
                            //    cube is refuted, the validated model contradicts
                            //    the primary UNSAT outright — a machine-checked
                            //    solver false-UNSAT. The model is then a genuine
                            //    predecessor: a state in the frame satisfying
                            //    ¬po.cube whose successor satisfies po.cube'
                            //    (the very assumptions of the query), so descend
                            //    on it exactly like the SAT arm below (route (ii)).
                            //    The chain link is a real transition into po.cube,
                            //    keeping verify_trace's BMC unrolling meaningful.
                            //
                            // Each pass through this arm therefore either adds a
                            // lemma blocking po.cube or enqueues a strictly-lower
                            // frame obligation whose resolution strengthens some
                            // frame with a lemma excluding the (finitely many)
                            // witness states: over a finite state space the cycle
                            // terminates. No unverified lemma is added, and no
                            // obligation is ever discarded.
                            let fb_target = po.frame.min(self.frames.depth() - 1);
                            let fb_verdict = if fb_target >= 1 {
                                self.verify_lemma_consecution(fb_target, &po.cube)
                            } else {
                                LemmaVerdict::Verified
                            };
                            match fb_verdict {
                                LemmaVerdict::Verified => {
                                    self.consecution_stats.lemmas_verified += 1;
                                    let fb_lemma = Lemma::from_blocked_cube(&po.cube);
                                    if std::env::var("IC3_DEBUG").is_ok() {
                                        eprintln!(
                                            "IC3 block_one: frame={} BLOCKED (unreduced fallback) \
                                             cube_len={} target_frame={} lemma={:?}",
                                            po.frame,
                                            po.cube.len(),
                                            fb_target,
                                            fb_lemma.lits,
                                        );
                                    }
                                    self.frames.add_lemma(fb_target, fb_lemma.clone());
                                    if fb_target > 0 {
                                        self.earliest_changed_frame =
                                            self.earliest_changed_frame.min(fb_target);
                                    }
                                    let start = usize::from(fb_target != 0);
                                    for s in &mut self.solvers[start..=fb_target] {
                                        s.add_clause(&fb_lemma.lits);
                                    }
                                    if let Some(ref mut pp) = self.predprop_solver {
                                        pp.add_lemma(&fb_lemma.lits);
                                    }
                                    return Ok(());
                                }
                                LemmaVerdict::Refuted { predecessor } => {
                                    eprintln!(
                                        "IC3 LEMMA REJECTED: frame={} unreduced cube also \
                                         refuted — validated model contradicts the primary \
                                         consecution UNSAT (solver false-UNSAT); descending \
                                         on the witness predecessor",
                                        po.frame,
                                    );
                                    // Failed block: same accounting as the SAT arm.
                                    po.bump_act();
                                    if self.cube_sat_consistent_with_init(&predecessor) {
                                        // In-Init witness: route through the frame-0
                                        // handler, which verifies the full chain via
                                        // BMC unrolling.
                                        let pred_po = ProofObligation::new(
                                            0,
                                            predecessor,
                                            po.depth + 1,
                                            Some(po),
                                        );
                                        self.obligations.push(pred_po);
                                    } else {
                                        self.obligations.push(ProofObligation::new(
                                            fb_target - 1,
                                            predecessor,
                                            po.depth + 1,
                                            Some(po.clone()),
                                        ));
                                        self.obligations.push(po);
                                    }
                                    return Ok(());
                                }
                            }
                        }
                    }
                }

                if std::env::var("IC3_DEBUG").is_ok() {
                    eprintln!(
                        "IC3 block_one: frame={} BLOCKED cube_len={} mic_len={} push_frame={} target_frame={} lemma={:?}",
                        po.frame,
                        po.cube.len(),
                        pushed_cube.len(),
                        push_frame,
                        target_frame,
                        lemma.lits,
                    );
                }
                self.frames.add_lemma(target_frame, lemma.clone());
                if target_frame > 0 {
                    self.earliest_changed_frame = self.earliest_changed_frame.min(target_frame);
                }
                let start = usize::from(target_frame != 0);
                for s in &mut self.solvers[start..=target_frame] {
                    s.add_clause(&lemma.lits);
                }
                if let Some(ref mut pp) = self.predprop_solver {
                    pp.add_lemma(&lemma.lits);
                }
                Ok(())
            }
            SatResult::Sat => {
                self.unknown_count = 0;

                // Failed block: a predecessor exists, so this obligation goes
                // back on the queue. Count the failure — the +1-per-failed-block
                // activity is what the dynamic generalization strategy
                // (arXiv:2501.02480 Alg. 5) reads, via `po.next`, when the
                // predecessor spawned below is eventually blocked
                // (see `dynamic_ctg_params`).
                po.bump_act();

                // Per-latch model-unassign pre-filter (#4091 Phase 3).
                //
                // Soundness guard (#4509): the per-latch ay-sat `flip_to_none`
                // pre-filter is unreliable as a state-essential check on the
                // shift-register / counter family — it reports latches as
                // flippable even when the SAT assumptions on the primed
                // target literals require those latches to hold a specific
                // value. Excluding such "flippable" latches from `state_lits`
                // either yields an empty cube (no constraints on the
                // predecessor at all) or a cube containing only don't-care
                // latches (the actual essentials are dropped). In both cases
                // the lift returns a wrong/incomplete predecessor, and the
                // frame-0 SAT-init shortcut pushes a chain whose verify_trace
                // can never confirm the counterexample.
                //
                // The downstream ternary prefilter in `lift_with_ternary`
                // performs a proper circuit-based don't-care check, and the
                // UNSAT-core extraction then keeps only the minimal essential
                // literals. Both steps cover what this pre-filter was
                // intended to skip, so we always pass the full latch set and
                // rely on those (correct) passes for reduction.
                //
                // We still call `minimize_model` (unchanged) to keep ay-sat's
                // internal model in the trail-trimmed form that downstream
                // queries assume.
                self.solvers[solver_idx].minimize_model(&self.ts.latch_vars);
                let essential_latches = self.ts.latch_vars.clone();

                // SAT-based predecessor lifting.
                let pred = {
                    let Ic3Engine {
                        ref mut lift,
                        ref solvers,
                        ref ts,
                        ref config,
                        ref ternary_sim,
                        ref reverse_next,
                        ..
                    } = *self;
                    let mut p = lift.lift_with_ternary(
                        solvers[solver_idx].as_ref(),
                        &assumptions,
                        &essential_latches,
                        &ts.input_vars,
                        Some(ternary_sim),
                        Some(reverse_next),
                    );
                    if config.internal_signals && !ts.internal_signals.is_empty() {
                        let isig_lits = Self::extract_state_from_solver(
                            solvers[solver_idx].as_ref(),
                            &ts.internal_signals,
                        );
                        p.extend(isig_lits);
                    }
                    p
                };
                // Init-consistent predecessor: may be a real counterexample (#4074, #4139).
                //
                // If the predecessor IS in Init, then Init can reach po.cube in one
                // step. Create a frame-0 PO for the predecessor with the original PO
                // as its successor, rather than only re-queuing the original PO.
                // This routes the chain straight to the frame-0 handler, which
                // verifies the full trace via BMC unrolling (#4139) — re-queuing
                // alone would leave get_bad() to rediscover the same bad cube.
                if self.cube_sat_consistent_with_init(&pred) {
                    let pred_po = ProofObligation::new(0, pred, po.depth + 1, Some(po));
                    self.obligations.push(pred_po);
                    return Ok(());
                }
                self.obligations.push(ProofObligation::new(
                    po.frame - 1,
                    pred,
                    po.depth + 1,
                    Some(po.clone()),
                ));
                self.obligations.push(po);
                Ok(())
            }
            SatResult::Unknown => {
                if self.solvers[solver_idx].is_poisoned() {
                    if !self.solver_rebuild_budget_exceeded(solver_idx) {
                        self.rebuild_solver_at(solver_idx);
                    }
                    self.unknown_count = 0;
                } else {
                    self.unknown_count += 1;
                    if self.unknown_count >= UNKNOWN_FALLBACK_THRESHOLD
                        && self.solver_backend != SolverBackend::Simple
                    {
                        self.fallback_solver_backend();
                    }
                }
                po.unknown_requeues += 1;
                if po.unknown_requeues <= MAX_UNKNOWN_REQUEUES {
                    self.obligations.push(po);
                } else if std::env::var("IC3_DEBUG").is_ok() {
                    eprintln!(
                        "IC3 DROP_UNKNOWN: frame={} depth={} requeues={} — dropping PO after \
                         {} Unknown results",
                        po.frame, po.depth, po.unknown_requeues, MAX_UNKNOWN_REQUEUES,
                    );
                }
                Ok(())
            }
        }
    }

    /// Reduce a cube using the UNSAT core from the solver.
    #[allow(dead_code)]
    pub(super) fn reduce_cube_from_core(&self, solver_idx: usize, cube: &[Lit]) -> Vec<Lit> {
        let Some(core) = self.solvers[solver_idx].unsat_core() else {
            return cube.to_vec();
        };
        if core.is_empty() {
            return cube.to_vec();
        }
        let mut core_latch_vars = rustc_hash::FxHashSet::default();
        for &core_lit in &core {
            if let Some(&latch_var) = self.reverse_next.get(&core_lit.var()) {
                core_latch_vars.insert(latch_var);
            }
        }
        let reduced: Vec<Lit> = cube
            .iter()
            .filter(|lit| core_latch_vars.contains(&lit.var()))
            .copied()
            .collect();
        if reduced.is_empty() {
            cube.to_vec()
        } else {
            reduced
        }
    }

    /// Find the lowest frame >= start_frame where the cube is already blocked.
    pub(super) fn find_blocking_frame(&self, start_frame: usize, cube: &[Lit]) -> Option<usize> {
        let clause = Lemma::from_blocked_cube(cube);
        for i in start_frame..self.frames.frames.len() {
            if self.frames.frames[i].has_subsuming(&clause) {
                return Some(i);
            }
        }
        for lemma in &self.inf_lemmas {
            if lemma.subsumes(&clause) {
                return Some(self.frames.frames.len());
            }
        }
        None
    }

    /// Compute dynamic CTG parameters for generalizing a proof obligation.
    ///
    /// Implements the dynamic adjustment of generalization strategies
    /// published in arXiv:2501.02480 (§IV, Alg. 5): an obligation's
    /// activity is passed to its successor's generalization decision, so the
    /// blocking of `po` reads the activity of `po.next` — the successor
    /// obligation `po` was spawned to unblock — to select how aggressively
    /// MIC may pursue counterexamples-to-generalization. A root obligation
    /// (no successor) generalizes without CTG. Returns
    /// `(ctg_max, ctg_limit)`.
    pub(super) fn dynamic_ctg_params(po: &ProofObligation) -> (usize, usize) {
        /// Activity at which plain CTG generalization switches on
        /// (arXiv:2501.02480 Sec. V-A).
        const CTG_THRESHOLD: f64 = 10.0;
        /// Activity at which extended CTG (EXCTG) switches on
        /// (arXiv:2501.02480 Sec. V-A).
        const EXCTG_THRESHOLD: f64 = 40.0;

        let act = po.next.as_ref().map_or(0.0, |succ| succ.act);

        if act >= EXCTG_THRESHOLD {
            // EXCTG regime: limit = (act - 40)^0.3 * 2 + 5 with ctg_max = 5,
            // per arXiv:2501.02480 Alg. 5.
            let limit = ((act - EXCTG_THRESHOLD).powf(0.3) * 2.0 + 5.0).round() as usize;
            (5, limit)
        } else if act >= CTG_THRESHOLD {
            // CTG regime: ctg_max = (act - 10) / 10 + 2, single blocking
            // attempt per CTG, per arXiv:2501.02480 Alg. 5.
            let ctg_max = ((act - CTG_THRESHOLD) as usize / 10) + 2;
            (ctg_max, 1)
        } else {
            (0, 0)
        }
    }
}
