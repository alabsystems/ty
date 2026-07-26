// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Streaming diff-based successor processing (Part of #3027 Phase 3).
//!
//! Uses split borrows to separate `&mut self.ctx` (for enumeration) from
//! `&self.state_storage` (for dedup), enabling inline fingerprinting and
//! dedup during enumeration via ClosureSink callback. Only new (post-dedup)
//! diffs (~5% of successors) are collected for deferred invariant checking.
//!
//! The borrow conflict between `&mut self.ctx` and `&self.state_storage` is
//! solved by extracting individual field borrows from `self` into the closure
//! scope. See Phase 2 (`ClosureSink` + `ControlFlow` propagation) for the
//! underlying DiffSink infrastructure.

use std::ops::ControlFlow;

use super::super::frontier::BfsFrontier;
use super::super::run_helpers::BfsProfile;
use super::super::{compute_diff_fingerprint_with_xor, CheckError, Fingerprint, ModelChecker};
use super::iter_state::BfsIterState;
use super::observer::CompositeObserver;
use super::storage_modes::BfsStorage;
use super::successor_processing::{BfsIterOutcome, PendingSuccessor};
use super::BfsStepParams;
use crate::enumerate::{ClosureSink, DiffSink};
use crate::state::DiffSuccessor;
use crate::storage::LookupOutcome;
use crate::{ConfigCheckError, EvalCheckError};

impl ModelChecker<'_> {
    /// Streaming variant of diff successor processing (Part of #3027 Phase 3).
    ///
    /// Uses split borrows to separate `&mut self.ctx` (for enumeration) from
    /// `&self.state_storage` (for dedup), enabling inline fingerprinting and
    /// dedup during enumeration. Only new (post-dedup) diffs are collected
    /// for deferred invariant checking and admission.
    ///
    /// Returns `Some(outcome)` if diffs were processed, `None` if enumeration
    /// returned None (caller should fall through to full-state path).
    ///
    /// Precondition: no implied actions, no constraints.
    /// Caller checks these before invoking.
    pub(super) fn process_diff_successors_streaming<
        S: BfsStorage,
        Q: BfsFrontier<Entry = S::QueueEntry>,
    >(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
    ) -> Option<BfsIterOutcome> {
        // Hybrid per-action dispatch (item 4 M0): classify once (btree reaches
        // the checker only through this streaming diff path). Inert unless
        // `TY_HYBRID_FLAT_VIEW` is set.
        self.ensure_hybrid_dispatch_ready();
        let &BfsStepParams {
            registry,
            current_depth: _current_depth,
            succ_depth,
            current_level: _current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();
        let current_array = iter_state.array();
        let raw_next_name = match self.trace.cached_next_name.clone() {
            Some(name) => name,
            None => {
                return Some(BfsIterOutcome::Terminate(self.bfs_error_return(
                    iter_state,
                    storage,
                    ConfigCheckError::MissingNext.into(),
                )));
            }
        };

        // Coverage fallback: streaming doesn't support coverage collection.
        if self.coverage.collect && !self.coverage.actions.is_empty() {
            return None;
        }

        let cache_for_liveness = self.liveness_cache.cache_for_liveness;

        // ================================================================
        // Phase A: Enumeration with inline dedup (split borrows)
        //
        // The borrow conflict: enumeration needs `&mut self.ctx`, but dedup
        // needs `&self.state_storage.seen_fps`. Rust allows disjoint field
        // borrows when explicit, so we extract individual fields from `self`.
        //
        // The closure does fingerprint + dedup inline during enumeration.
        // Only new states (~5% of successors) are collected for Phase B.
        // ================================================================
        // WP-26: Phase A is the streaming engine's counterpart to the batch
        // engine's `interp_enum` + `interp_succ_build` + `fp_to_state` buckets —
        // it enumerates, fingerprints and dedups every successor without ever
        // materializing one. Measuring both makes the two engines comparable on
        // the same spec (`TY_HYBRID_PERF_DEBUG=1`, no other gate required).
        let t_stream_a = self.hybrid_dispatch.perf.start();
        let prof_t0 = prof.now();

        // Accumulators — captured by the ClosureSink closure.
        let mut new_diffs: Vec<(DiffSuccessor, Fingerprint, Option<u64>, bool)> = Vec::new();
        let mut total_count: usize = 0;
        let mut succ_fps_for_liveness: Option<Vec<Fingerprint>> = cache_for_liveness.then(Vec::new);
        let mut storage_fault: Option<crate::storage::StorageFault> = None;
        let mut leaf_tir_used = false;

        // Phase A body: split borrows for enumeration + inline dedup.
        let enum_result: Result<bool, CheckError> = (|| {
            // Split borrows: extract ctx separately from other fields.
            // Rust allows disjoint borrows of different struct fields.
            let ctx = &mut self.ctx;
            let module = &self.module;
            let compiled = &self.compiled;
            let state_storage = &self.state_storage;
            let tir_parity = &self.tir_parity;
            let trace = &self.trace;
            // Nested-set A6 diff-path monitor hook: disjoint &mut borrow of the
            // monitors so the ClosureSink can OBSERVE every successor's board on
            // the fast diff path (escape-only; the fingerprint is unaffected).
            let nested_set_monitors = &mut self.nested_set_monitors;

            let _state_guard = ctx.bind_state_env_guard(current_array.env_ref());

            // Use cached resolved name to avoid per-state resolve_op_name + String alloc.
            let resolved_next_name = trace
                .cached_resolved_next_name
                .as_deref()
                .unwrap_or_else(|| ctx.resolve_op_name(&raw_next_name));
            let def = module
                .op_defs
                .get(resolved_next_name)
                .ok_or(ConfigCheckError::MissingNext)?;

            // Part of #3294: extract TIR leaf for streaming path.
            // TirProgram borrows from tir_parity (immutable), disjoint from ctx (mutable).
            let tir_program = tir_parity.as_ref().and_then(|p| {
                p.make_tir_program_for_selected_eval_name(&raw_next_name, resolved_next_name)
            });

            // ClosureSink: fingerprint + dedup inline, collect only new states.
            // Captures disjoint borrows: state_storage (for dedup), registry +
            // current_array (for fingerprinting), local accumulators.
            let mut sink = ClosureSink::new(|diff: DiffSuccessor| -> ControlFlow<()> {
                total_count = total_count
                    .checked_add(1)
                    .expect("raw successor generation count overflowed usize");

                // Nested-set A6: the per-successor escape MONITOR on the HOT diff
                // path — unbypassable. Observes the board BEFORE dedup/store; fails
                // closed on escape (bails the var to raw value_fingerprint, the
                // SAME fp the diff already produces, so the verdict is identical).
                ModelChecker::observe_diff_monitors_escape(nested_set_monitors, &diff.changes);

                // Fingerprint computation (standalone, no ctx needed).
                // Same logic as the batch fast path: use pre-computed fp from
                // compiled action, or compute from changes if deferred/stale.
                let (succ_fp, cached_xor) = if diff.fingerprint.0 == 0 {
                    let (fp_val, xor) =
                        compute_diff_fingerprint_with_xor(current_array, &diff.changes, registry);
                    (fp_val, Some(xor))
                } else {
                    (diff.fingerprint, None)
                };

                // Collect liveness fps (all successors, including duplicates).
                if let Some(ref mut fps) = succ_fps_for_liveness {
                    fps.push(succ_fp);
                }

                // Inline dedup: check against seen fingerprint set.
                // Only needs &state_storage (shared borrow), not &mut ctx.
                let already_seen = match state_storage.seen_fps.contains_checked(succ_fp) {
                    LookupOutcome::Present => true,
                    LookupOutcome::StorageFault(fault) => {
                        storage_fault = Some(fault);
                        return ControlFlow::Break(());
                    }
                    LookupOutcome::Absent => false,
                    _ => unreachable!(),
                };

                // Defer duplicate payload authorization to Phase B where the
                // successor payload can be materialized and compared.
                new_diffs.push((diff, succ_fp, cached_xor, already_seen));
                ControlFlow::Continue(())
            });

            // Part of #3354 Slice 1: unified-only successor generation.
            // All successor enumeration routes through the canonical AST/TIR
            // unified path. The compiled split-action dispatch is removed.
            // Part of #3923: pc-guard hoisting — pass pc_var_idx to the unified
            // enumerator so Or branches with non-matching pc guards are skipped.
            // Part of #3805: pc_var_idx is set independently of the dispatch table,
            // enabling guard hoisting for multi-process PlusCal specs.
            let pc_var_idx = compiled.pc_var_idx;
            leaf_tir_used = tir_program.is_some();
            crate::enumerate::enumerate_successors_array_as_diffs_into_with_pc_hoist(
                ctx,
                def,
                current_array,
                &module.vars,
                &mut sink,
                tir_program.as_ref(),
                pc_var_idx,
            )
            .map_err(EvalCheckError::Eval)?;

            let had_raw = sink.count() > 0;
            Ok(had_raw)
        })();
        // ---- Split borrow scope ends: &mut self is available again ----

        let had_raw_successors = match enum_result {
            Ok(had) => had,
            Err(e) => {
                prof.accum_succ_gen(prof_t0);
                return Some(BfsIterOutcome::Terminate(
                    self.bfs_error_return(iter_state, storage, e),
                ));
            }
        };
        prof.accum_succ_gen(prof_t0);
        super::super::hybrid_dispatch::perf_acc(
            &mut self.hybrid_dispatch.perf.stream_phase_a_ns,
            t_stream_a,
        );
        self.hybrid_dispatch.perf.stream_parents += 1;
        self.hybrid_dispatch.perf.stream_succ_count += total_count as u64;

        // Storage fault during inline dedup — terminal error.
        if let Some(fault) = storage_fault {
            iter_state.return_to(storage, self);
            return Some(BfsIterOutcome::Terminate(self.storage_fault_result(fault)));
        }
        #[cfg(debug_assertions)]
        {
            let _ = self.debug_bfs_state_header(
                fp,
                current_array,
                _current_depth,
                total_count,
                "diff-stream",
            );
        }

        let current_array_for_tir =
            (self.tir_parity.is_some() && !leaf_tir_used).then(|| current_array.clone());

        let mut state_observer =
            CompositeObserver::state_completion(self.exploration.check_deadlock, false);
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            total_count == 0,
            had_raw_successors,
            None,
            &[],
        ) {
            return Some(outcome);
        }

        prof.count_successors(total_count);
        self.record_transitions(total_count);
        self.record_raw_successors_generated(total_count);

        self.ctx.set_tlc_level(succ_level);

        // ================================================================
        // Phase B: Process new (post-dedup) diffs with full &mut self.
        //
        // Only the ~5% of successors that passed dedup are processed here.
        // Each needs: materialize → recheck fp if changed → invariant →
        // admit → enqueue. These operations need &mut self.ctx which is
        // now available again after enumeration completed.
        // ================================================================
        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);
        // WP-26: Phase B is paid for post-dedup NEW successors only — the term
        // the batch engine has no equivalent of, because the batch engine
        // materializes every successor before it knows whether it is new.
        let t_stream_b = self.hybrid_dispatch.perf.start();
        for (mut diff, succ_fp, cached_xor, was_pre_seen) in new_diffs {
            prof.count_new_state();

            // Materialize lazy values (needs &self.ctx, available now).
            let materialized = match crate::materialize::materialize_diff_changes(
                &self.ctx,
                &mut diff.changes,
                self.compiled.spec_may_produce_lazy,
            ) {
                Ok(m) => m,
                Err(e) => {
                    return Some(BfsIterOutcome::Terminate(self.bfs_error_return(
                        iter_state,
                        storage,
                        EvalCheckError::Eval(e).into(),
                    )));
                }
            };

            // If materialization changed values, recompute fingerprint and
            // re-check dedup (the pre-materialization fp may now be different).
            let (final_fp, combined_xor) = if materialized {
                let (fp_val, xor) =
                    compute_diff_fingerprint_with_xor(iter_state.array(), &diff.changes, registry);
                (fp_val, xor)
            } else {
                // No materialization: use original fp. Compute combined_xor
                // from cache or fresh (Part of #1564).
                let xor = cached_xor.unwrap_or_else(|| {
                    compute_diff_fingerprint_with_xor(iter_state.array(), &diff.changes, registry).1
                });
                (succ_fp, xor)
            };

            // An exact ArrayFp64 witness can authorize a duplicate directly
            // against the virtual `base + changes` state. Lazy values have
            // already been forced and the final fingerprint corrected above,
            // so this preserves evaluation errors while avoiding the full base
            // clone and heap-backed CompactValue clones for duplicate edges.
            // A failed/absent proof always falls through to the existing
            // materialize-and-authorize path. Keep hybrid shadow validation
            // observable by disabling this skip while hybrid dispatch is active.
            let hybrid_dispatch_active = self.hybrid_dispatch_active();
            let fingerprint_may_be_seen =
                (was_pre_seen && final_fp == succ_fp) || (materialized && final_fp != succ_fp);
            if !hybrid_dispatch_active
                && fingerprint_may_be_seen
                && storage.confirm_seen_array_diff_duplicate(
                    final_fp,
                    iter_state.array(),
                    &diff.changes,
                ) == Some(true)
            {
                continue;
            }

            // Hybrid per-action dispatch (item 4 M0): capture changed vars BEFORE
            // `diff` is consumed by `into_array_state_with_fp`.
            let hybrid_changed: Option<Vec<crate::var_index::VarIndex>> =
                hybrid_dispatch_active.then(|| diff.changes.iter().map(|(vi, _)| *vi).collect());

            let succ = diff.into_array_state_with_fp(iter_state.array(), final_fp, combined_xor);

            // Validating shadow (pure — never changes `succ` or the enqueue set).
            if let Some(changed) = hybrid_changed.as_deref() {
                self.hybrid_shadow_validate_diff(iter_state.array(), &succ, changed, registry);
            }

            if was_pre_seen && final_fp == succ_fp {
                let current = Some((iter_state.fp(), iter_state.array()));
                if let Err(result) =
                    storage.enforce_seen_successor_duplicate(final_fp, &succ, current, self)
                {
                    iter_state.return_to(storage, self);
                    return Some(BfsIterOutcome::Terminate(result));
                }
                continue;
            }

            // Re-check dedup with the corrected fingerprint after materialization.
            if materialized && final_fp != succ_fp {
                let already_seen = match self
                    .check_seen_successor_with_profile(iter_state, storage, final_fp, &succ, prof)
                {
                    Ok(seen) => seen,
                    Err(outcome) => return Some(outcome),
                };
                if already_seen {
                    continue;
                }
            }

            // Part of #3352: the streaming diff path used split actions to
            // generate successors but never replayed the selected Next operator
            // through `transition_holds_via_tir`, so bare-INSTANCE wrapper specs
            // stayed on the compiled/AST lane with zero TIR probe hits. Match
            // the batch diff/full-array paths by replaying TIR on admitted
            // successors when leaf TIR was not already used during enumeration.
            if self.tir_parity.is_some() && !leaf_tir_used {
                let tir_current_array = current_array_for_tir
                    .as_ref()
                    .expect("TIR replay snapshot should exist when replay is enabled");
                match self.transition_holds_via_tir(&raw_next_name, tir_current_array, &succ) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(error) => {
                        return Some(BfsIterOutcome::Terminate(
                            self.bfs_error_return(iter_state, storage, error),
                        ));
                    }
                }
                if let Err(error) =
                    self.maybe_check_tir_parity_transition(&raw_next_name, tir_current_array, &succ)
                {
                    return Some(BfsIterOutcome::Terminate(
                        self.bfs_error_return(iter_state, storage, error),
                    ));
                }
            }

            if let Err(outcome) = self.finish_prefiltered_successor(
                iter_state,
                storage,
                queue,
                prof,
                &mut admitted_observer,
                succ,
                PendingSuccessor {
                    parent_fp: fp,
                    succ_fp: final_fp,
                    succ_depth,
                    succ_level,
                },
            ) {
                return Some(outcome);
            }
        }
        super::super::hybrid_dispatch::perf_acc(
            &mut self.hybrid_dispatch.perf.stream_phase_b_ns,
            t_stream_b,
        );

        // Cache liveness fingerprints (mode-specific).
        if let Err(error) = storage.cache_diff_liveness(fp, succ_fps_for_liveness, self) {
            return Some(BfsIterOutcome::Terminate(
                self.bfs_error_return(iter_state, storage, error),
            ));
        }

        // Part of #3784: record monolithic successor count to cooperative metrics.
        self.record_cooperative_monolithic_successors(total_count);

        // Part of #3850: record monolithic action eval for tiered JIT.
        self.record_action_eval_for_tier(0, total_count as u64);

        // Part of #3910: record JIT next-state dispatch for `--show-tiers` report.
        self.record_monolithic_next_state_dispatch();

        // Return current state to storage.
        iter_state.return_to(storage, self);

        Some(BfsIterOutcome::Continue)
    }
}
