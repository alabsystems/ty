// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! No-trace (fingerprint-only) BFS mode: initial state generation and BFS exploration loop.
//!
//! This module handles the fingerprint-indexed path where trace reconstruction
//! can avoid the legacy full-state store. Some paths still retain canonical
//! payload witnesses so shared dedup admission can fail closed on fingerprint
//! collisions instead of silently suppressing a different state.

use super::bfs::{FingerprintOnlyStorage, NoTraceQueueEntry};
#[cfg(debug_assertions)]
use super::debug::debug_states;
use super::frontier::BfsFrontier;
use super::{
    check_error_to_result, ArrayState, BulkStateHandle, BulkStateStorage, CheckResult, Fingerprint,
    ModelChecker, VecDeque,
};
use crate::TraceLocationStorage;
use crate::{ConfigCheckError, EvalCheckError, RuntimeCheckError};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::check) struct PreCodegenInitSeenTelemetry {
    pub retired: bool,
    pub witnesses_before: usize,
    pub witnesses_after_retire: usize,
    pub capacity_before: usize,
    pub capacity_after_retire: usize,
    pub reconstructed: bool,
    pub reconstructed_witnesses: usize,
}

#[cfg(test)]
thread_local! {
    static PRE_CODEGEN_INIT_SEEN_TELEMETRY:
        std::cell::Cell<PreCodegenInitSeenTelemetry> =
            const { std::cell::Cell::new(PreCodegenInitSeenTelemetry {
                retired: false,
                witnesses_before: 0,
                witnesses_after_retire: 0,
                capacity_before: 0,
                capacity_after_retire: 0,
                reconstructed: false,
                reconstructed_witnesses: 0,
            }) };
}

impl<'a> ModelChecker<'a> {
    /// mem2 (dead bulk-init drop) kill-switch.
    ///
    /// `TY_DROP_DEAD_BULK_INIT=0` (or `off`/`false`) restores retention of the
    /// bulk init-state tree in `FingerprintOnlyStorage` even when the direct
    /// flat init frontier makes it dead. Default: enabled.
    fn drop_dead_bulk_init_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| {
            std::env::var("TY_DROP_DEAD_BULK_INIT")
                .map(|v| {
                    !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
                })
                .unwrap_or(true)
        })
    }

    fn rekey_checkpoint_depth(&mut self, old_fp: Fingerprint, new_fp: Fingerprint) {
        if let Some(depth) = self.trace.depths.remove(&old_fp) {
            self.trace.depths.insert(new_fp, depth);
        }
    }

    /// Whether the FP64 init payload map can be retired while post-layout
    /// trust-codegen specialization runs.
    ///
    /// The bulk store plus its queue handles remain the exact reconstruction
    /// authority. Keep this gate narrower than ordinary flat-BFS admission:
    /// every generated bulk row must have one queued, fingerprinted handle in
    /// index order, and liveness must not retain a second init-state tree.
    fn can_retire_init_seen_before_codegen(
        &self,
        bulk_initial: &BulkStateStorage,
        queue: &VecDeque<(NoTraceQueueEntry, usize, u64)>,
    ) -> bool {
        Self::drop_dead_bulk_init_enabled()
            && self.fp_only_flat_witness_policy_enabled()
            && (self.flat_state_primary || self.native_fused_flat_frontier_admission_candidate())
            && self.should_use_flat_bfs()
            && self.liveness_cache.init_states.is_empty()
            && !self.state_storage.seen.is_empty()
            && self.state_storage.seen.len() == queue.len()
            && queue.len() == bulk_initial.len()
            && self
                .flat_bfs_adapter
                .as_ref()
                .is_some_and(|adapter| adapter.is_fully_flat())
            && queue
                .iter()
                .enumerate()
                .all(|(expected_idx, (entry, _, _))| match entry {
                    NoTraceQueueEntry::Bulk(handle) => {
                        handle.index as usize == expected_idx
                            && handle
                                .fingerprint
                                .is_some_and(|fp| self.state_storage.seen.contains_key(&fp))
                    }
                    NoTraceQueueEntry::Owned { .. }
                    | NoTraceQueueEntry::Witness { .. }
                    | NoTraceQueueEntry::Flat { .. } => false,
                })
    }

    /// Drop all but one entry from the wide FP64 -> ArrayState init witness
    /// table before codegen.
    ///
    /// The retained representative is required by
    /// `try_activate_compiled_fingerprinting`: an empty map makes that decision
    /// fall back to `flat_state_primary`, which is also valid for fully-flat
    /// compound layouts and would therefore misclassify them as scalar. No
    /// state admission occurs until the map is either rebuilt or superseded by
    /// the CompiledFlat re-fingerprint pass.
    fn retire_init_seen_before_codegen(
        &mut self,
        bulk_initial: &BulkStateStorage,
        queue: &VecDeque<(NoTraceQueueEntry, usize, u64)>,
    ) -> bool {
        #[cfg(test)]
        PRE_CODEGEN_INIT_SEEN_TELEMETRY.with(|telemetry| {
            telemetry.set(PreCodegenInitSeenTelemetry::default());
        });
        if !self.can_retire_init_seen_before_codegen(bulk_initial, queue) {
            return false;
        }

        #[cfg(test)]
        let witnesses_before = self.state_storage.seen.len();
        #[cfg(test)]
        let capacity_before = self.state_storage.seen.capacity();
        let retained_witness = self
            .state_storage
            .seen
            .iter()
            .next()
            .map(|(fp, state)| (*fp, state.clone()));
        self.state_storage.seen = crate::state::fp_hashmap();
        if let Some((fp, state)) = retained_witness {
            self.state_storage.seen.insert(fp, state);
        }

        #[cfg(test)]
        PRE_CODEGEN_INIT_SEEN_TELEMETRY.with(|telemetry| {
            telemetry.set(PreCodegenInitSeenTelemetry {
                retired: true,
                witnesses_before,
                witnesses_after_retire: self.state_storage.seen.len(),
                capacity_before,
                capacity_after_retire: self.state_storage.seen.capacity(),
                reconstructed: false,
                reconstructed_witnesses: 0,
            });
        });
        true
    }

    /// Rebuild the exact ArrayState collision witnesses for the queued init
    /// subset after codegen selects a non-CompiledFlat fingerprint domain.
    ///
    /// Queue membership is authoritative. The current retirement gate requires
    /// the queue and bulk store to be an exact one-to-one, index-ordered initial
    /// wavefront. Build into a fresh map and publish only after every handle
    /// validates and materializes.
    #[allow(clippy::result_large_err)]
    fn reconstruct_init_seen_from_bulk_queue(
        &mut self,
        bulk_initial: &BulkStateStorage,
        queue: &VecDeque<(NoTraceQueueEntry, usize, u64)>,
        scratch: &mut ArrayState,
    ) -> Result<(), CheckResult> {
        let mut rebuilt = crate::state::fp_hashmap_with_capacity(queue.len());
        for (entry, _, _) in queue {
            let NoTraceQueueEntry::Bulk(handle) = entry else {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(
                        "pre-codegen init witness reconstruction requires Bulk-only queue entries"
                            .to_string(),
                    )
                    .into(),
                    self.stats.clone(),
                ));
            };
            if handle.index as usize >= bulk_initial.len() {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(format!(
                        "pre-codegen init witness handle {} exceeds bulk length {}",
                        handle.index,
                        bulk_initial.len(),
                    ))
                    .into(),
                    self.stats.clone(),
                ));
            }
            let Some(fp) = handle.fingerprint else {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(
                        "pre-codegen init witness reconstruction requires cached fingerprints"
                            .to_string(),
                    )
                    .into(),
                    self.stats.clone(),
                ));
            };

            scratch.overwrite_from_slice(bulk_initial.get_state(handle.index));
            crate::materialize::materialize_array_state(
                &self.ctx,
                scratch,
                self.compiled.spec_may_produce_lazy,
            )
            .map_err(|error| {
                check_error_to_result(EvalCheckError::Eval(error).into(), &self.stats)
            })?;
            if rebuilt.insert(fp, scratch.clone()).is_some() {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(format!(
                        "duplicate queued init fingerprint {fp:?} during witness reconstruction",
                    ))
                    .into(),
                    self.stats.clone(),
                ));
            }
        }
        self.state_storage.seen = rebuilt;

        #[cfg(test)]
        PRE_CODEGEN_INIT_SEEN_TELEMETRY.with(|telemetry| {
            let mut row = telemetry.get();
            row.reconstructed = true;
            row.reconstructed_witnesses = self.state_storage.seen.len();
            telemetry.set(row);
        });
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::check) fn test_pre_codegen_init_seen_telemetry(
        &self,
    ) -> PreCodegenInitSeenTelemetry {
        PRE_CODEGEN_INIT_SEEN_TELEMETRY.with(std::cell::Cell::get)
    }

    /// Terminal result for a state that cannot be encoded in the fixed flat
    /// layout (scalar > i64, sequence over proven capacity, ...).
    ///
    /// Graceful flat-overflow handling: this replaces the former fail-stop
    /// asserts/panics. The typed error propagates to the CLI, which retries
    /// the check once with flat state storage disabled.
    fn flat_layout_unsupported_result(
        &self,
        context: &str,
        err: &crate::state::FlatSerializationError,
    ) -> CheckResult {
        check_error_to_result(
            crate::CheckError::flat_layout_unsupported_value(format!("{context}: {err}")),
            &self.stats,
        )
    }

    fn can_build_flat_initial_frontier_direct(
        &self,
        bulk_initial: &BulkStateStorage,
        queue: &VecDeque<(NoTraceQueueEntry, usize, u64)>,
    ) -> bool {
        if !(self.flat_state_primary || self.native_fused_flat_frontier_admission_active())
            || !self
                .flat_bfs_adapter
                .as_ref()
                .is_some_and(|adapter| adapter.is_fully_flat())
            || queue.len() != bulk_initial.len()
        {
            return false;
        }

        queue
            .iter()
            .enumerate()
            .all(|(expected_idx, (entry, _, _))| match entry {
                NoTraceQueueEntry::Bulk(handle) => {
                    handle.index as usize == expected_idx && handle.fingerprint.is_some()
                }
                NoTraceQueueEntry::Owned { .. }
                | NoTraceQueueEntry::Witness { .. }
                | NoTraceQueueEntry::Flat { .. } => false,
            })
    }

    #[allow(clippy::result_large_err)]
    fn refingerprint_initial_states_into_flat_frontier(
        &mut self,
        bulk_initial: &BulkStateStorage,
        queue: &mut VecDeque<(NoTraceQueueEntry, usize, u64)>,
        bulk_scratch: &mut ArrayState,
        layout: std::sync::Arc<crate::state::StateLayout>,
    ) -> Result<super::bfs::flat_frontier::FlatBfsFrontier, CheckResult> {
        let fresh_seen_fps = self
            .state_storage
            .seen_fps
            .fresh_empty_clone()
            .map_err(|fault| self.storage_fault_result(fault))?;
        self.state_storage.replace_seen_fps(fresh_seen_fps);
        // Initial-state admission happened before the flat layout and compiled
        // fingerprint domain were available, so fingerprint-only mode retained
        // an ArrayState payload witness for every FP64 init state in `seen`.
        // This direct path re-admits every init state from `bulk_initial` into
        // the CompiledFlat domain, where compact flat payload witnesses replace
        // those tree-form states. Replace the map instead of clearing it so its
        // now-dead bucket allocation is released before the flat frontier is
        // built. If compact witnesses are disabled, however, re-admission needs
        // the ArrayState map again; retain its capacity in that fallback mode.
        // The non-direct re-fingerprint path keeps its existing `clear` and
        // capacity reuse behavior.
        if self.fp_only_flat_witness_active() {
            self.state_storage.seen = crate::state::fp_hashmap();
        } else {
            self.state_storage.seen.clear();
        }

        if !self.liveness_cache.init_states.is_empty() {
            let mut liveness_buffer = vec![0i64; layout.total_slots()];
            // Phase 1 (immutable): compute the flat-domain fingerprints. Fail
            // closed if a state does not fit the fixed flat layout — never
            // fingerprint/dedup a collapsed all-zeros buffer, which would alias
            // distinct states. Propagate the typed error so the CLI can retry
            // without flat storage. (audit-2026-07 #6)
            let mut new_liveness_fps = Vec::with_capacity(self.liveness_cache.init_states.len());
            for (_fp, arr) in &self.liveness_cache.init_states {
                crate::state::FlatState::try_write_array_state_into(
                    arr,
                    &layout,
                    &mut liveness_buffer,
                )
                .map_err(|err| {
                    self.flat_layout_unsupported_result(
                        "direct flat init frontier (liveness init)",
                        &err,
                    )
                })?;
                new_liveness_fps.push(super::invariants::fingerprint_flat_compiled(
                    &liveness_buffer,
                ));
            }
            // Phase 2 (mutable): apply the rekeying only after every state
            // encoded successfully.
            for ((fp, _arr), new_fp) in self
                .liveness_cache
                .init_states
                .iter_mut()
                .zip(new_liveness_fps)
            {
                *fp = new_fp;
            }
        }

        let mut flat_queue =
            super::bfs::flat_frontier::FlatBfsFrontier::with_capacity(layout.clone(), queue.len());
        let mut flat_buffer = vec![0i64; layout.total_slots()];
        while let Some((entry, depth, trace_loc)) = queue.pop_front() {
            let NoTraceQueueEntry::Bulk(handle) = entry else {
                unreachable!("direct flat init frontier requires bulk-only queue entries");
            };
            let old_fp = handle
                .fingerprint
                .expect("direct flat init frontier requires cached init fingerprint");

            bulk_scratch.overwrite_from_slice(bulk_initial.get_state(handle.index));
            // Fail closed on an unencodable state rather than fingerprinting/
            // dedup'ing an all-zeros buffer that aliases distinct states.
            // Propagate the typed error (graceful flat-overflow handling) so
            // the CLI retries without flat storage. (audit-2026-07 #6)
            crate::state::FlatState::try_write_array_state_into(
                bulk_scratch,
                &layout,
                &mut flat_buffer,
            )
            .map_err(|err| {
                self.flat_layout_unsupported_result("direct flat init frontier", &err)
            })?;
            let new_fp = super::invariants::fingerprint_flat_compiled(&flat_buffer);

            if !self.mark_state_seen_checked_with_current(new_fp, bulk_scratch, None, 0, None)? {
                continue;
            }
            if let Some(trace_loc) = self.trace.trace_locs.get(&old_fp) {
                let _ = self.trace.trace_locs.insert(new_fp, trace_loc);
            }
            self.rekey_checkpoint_depth(old_fp, new_fp);

            flat_queue.push_raw_buffer(&flat_buffer, new_fp, depth, trace_loc);
        }

        self.stats.initial_states = self.states_count();
        self.stats.states_found = self.states_count();

        if super::debug::bytecode_vm_stats_enabled() && flat_queue.flat_pushed() > 0 {
            eprintln!(
                "[flat-bfs] prepared {} init states directly in flat frontier \
                 (seen_fps reset to xxh3 domain)",
                flat_queue.flat_pushed(),
            );
        }

        Ok(flat_queue)
    }

    /// Generate initial states for no-trace (fingerprint-only) BFS mode.
    ///
    /// Tries streaming enumeration first (avoids Vec<State> OrdMap overhead),
    /// then falls back to the Vec<State> path. Returns (BulkStateStorage, initial queue)
    /// or an early CheckResult on error/violation.
    #[allow(clippy::result_large_err, clippy::type_complexity)]
    fn init_states_no_trace(
        &mut self,
        init_name: &str,
        registry: &crate::var_index::VarRegistry,
        bulk_scratch: &mut ArrayState,
    ) -> Result<(BulkStateStorage, VecDeque<(NoTraceQueueEntry, usize, u64)>), CheckResult> {
        // Part of #3305: streaming invariant scan — O(1) memory per state.
        // For specs like Einstein (~199M init states), this finds invariant
        // violations without materializing the full state space into BulkStateStorage.
        self.scan_init_invariants_streaming(init_name)?;

        let init_generated: usize;
        let result = if let Some(bulk_init) =
            self.solve_predicate_for_states_to_bulk_prechecked(init_name)?
        {
            let init_generated_count = bulk_init.enumeration.generated;
            self.stats.raw_initial_states_generated = init_generated_count;
            let storage = bulk_init.storage;
            let mut queue: VecDeque<(NoTraceQueueEntry, usize, u64)> = VecDeque::new();
            queue.reserve(storage.len());

            let num_states = u32::try_from(storage.len()).map_err(|_| {
                CheckResult::from_error(
                    ConfigCheckError::Setup(format!(
                        "too many initial states ({}) for u32 BulkStateStorage index",
                        storage.len()
                    ))
                    .into(),
                    self.stats.clone(),
                )
            })?;

            for idx in 0..num_states {
                // --max-states: stop admitting initial states at the limit.
                if self.init_state_limit_reached() {
                    break;
                }
                bulk_scratch.overwrite_from_slice(storage.get_state(idx));
                let fp = self.prepare_prechecked_initial_state(bulk_scratch)?;

                if !self.mark_state_seen_checked_with_current(fp, bulk_scratch, None, 0, None)? {
                    debug_eprintln!(debug_states(), "DUP INIT STATE {}", fp);
                    continue;
                }
                if self.track_liveness_init_states() {
                    self.liveness_cache
                        .init_states
                        .push((fp, bulk_scratch.clone()));
                }
                let trace_loc = self.trace.last_inserted_trace_loc;
                #[cfg(debug_assertions)]
                if debug_states() {
                    let state = bulk_scratch.to_state(registry);
                    eprintln!("INIT STATE {} via Init: {:?}", fp, state);
                }
                queue.push_back((
                    NoTraceQueueEntry::Bulk(BulkStateHandle::with_fingerprint(idx, fp)),
                    0,
                    trace_loc,
                ));
            }

            init_generated = init_generated_count;
            self.stats.initial_states = self.states_count();
            (storage, queue)
        } else {
            match self.generate_initial_states_to_bulk(init_name) {
                Ok(Some(bulk_init)) => {
                    let init_generated_count = bulk_init.enumeration.generated;
                    self.stats.raw_initial_states_generated = init_generated_count;
                    let storage = bulk_init.storage;
                    // Streaming successful! Process states from BulkStateStorage directly.
                    let mut queue: VecDeque<(NoTraceQueueEntry, usize, u64)> = VecDeque::new();
                    queue.reserve(storage.len());

                    let num_states = u32::try_from(storage.len()).map_err(|_| {
                        CheckResult::from_error(
                            ConfigCheckError::Setup(format!(
                                "too many initial states ({}) for u32 BulkStateStorage index",
                                storage.len()
                            ))
                            .into(),
                            self.stats.clone(),
                        )
                    })?;

                    // Part of #254: Set TLC level for TLCGet("level") - TLC uses 1-based indexing
                    // Initial states are at level 1 in TLC
                    self.ctx.set_tlc_level(1);

                    for idx in 0..num_states {
                        // --max-states: stop admitting initial states at the limit.
                        if self.init_state_limit_reached() {
                            break;
                        }
                        // Load state into scratch buffer for constraint/invariant checking
                        bulk_scratch.overwrite_from_slice(storage.get_state(idx));

                        // Part of #2473: Use shared check_init_state helper
                        let (fp, violation) = match self.check_init_state(bulk_scratch, true)? {
                            Some(result) => result,
                            None => continue,
                        };

                        // Part of #2708: atomic test-and-set — skip if already present.
                        if !self.mark_state_seen_checked_with_current(
                            fp,
                            bulk_scratch,
                            None,
                            0,
                            None,
                        )? {
                            debug_eprintln!(debug_states(), "DUP INIT STATE {}", fp);
                            continue;
                        }
                        // Part of #3175: cache init states for post-BFS liveness
                        if self.track_liveness_init_states() {
                            self.liveness_cache
                                .init_states
                                .push((fp, bulk_scratch.clone()));
                        }
                        // Part of #2881 Step 3: capture trace_loc for queue entry.
                        let trace_loc = self.trace.last_inserted_trace_loc;
                        if let Some(violation) = violation {
                            self.handle_init_violation(violation, fp, || {
                                bulk_scratch.to_state(registry)
                            })?;
                        } else {
                            #[cfg(debug_assertions)]
                            if debug_states() {
                                let state = bulk_scratch.to_state(registry);
                                eprintln!("INIT STATE {} via Init: {:?}", fp, state);
                            }
                        }
                        queue.push_back((
                            NoTraceQueueEntry::Bulk(BulkStateHandle::with_fingerprint(idx, fp)),
                            0,
                            trace_loc,
                        ));
                    }

                    init_generated = init_generated_count;
                    self.stats.initial_states = self.states_count();
                    (storage, queue)
                }
                // Part of #1433: Propagate actual eval errors instead of silently falling back.
                // Part of #2789: Route through check_error_to_result so ExitRequested
                // maps to LimitReached(Exit) instead of being misreported as Error.
                Err(e) => {
                    return Err(check_error_to_result(e, &self.stats));
                }
                Ok(None) => {
                    // Streaming not possible - fall back to Vec<State> path
                    let (initial_states, raw_initial_states_generated) =
                        self.constrained_initial_states(init_name)?;
                    init_generated = raw_initial_states_generated;

                    // Part of #254: Set TLC level for TLCGet("level") - TLC uses 1-based indexing
                    // Initial states are at level 1 in TLC
                    self.ctx.set_tlc_level(1);

                    // Convert to BulkStateStorage and check invariants in a single pass
                    // Part of #595: Handle continue_on_error for initial states
                    let mut bulk_storage =
                        BulkStateStorage::new(registry.len(), initial_states.len());
                    let mut queue: VecDeque<(NoTraceQueueEntry, usize, u64)> = VecDeque::new();
                    queue.reserve(initial_states.len());

                    for state in initial_states {
                        // --max-states: stop admitting initial states at the limit.
                        if self.init_state_limit_reached() {
                            break;
                        }
                        // Part of #2473: Use shared check_init_state helper
                        // check_constraints=false: already filtered by constrained_initial_states
                        let mut arr = ArrayState::from_state(&state, registry);
                        let (fp, violation) = match self.check_init_state(&mut arr, false)? {
                            Some(result) => result,
                            None => continue,
                        };

                        // Part of #2708: atomic test-and-set — push state then check
                        // dedup via InsertOutcome (skip enqueue if already present).
                        let idx = bulk_storage.push_from_state(&state, registry);
                        if !self.mark_state_seen_checked_with_current(fp, &arr, None, 0, None)? {
                            debug_eprintln!(debug_states(), "DUP INIT STATE {}", fp);
                            continue;
                        }
                        // Part of #3175: cache init states for post-BFS liveness
                        if self.track_liveness_init_states() {
                            self.liveness_cache.init_states.push((fp, arr.clone()));
                        }
                        // Part of #2881 Step 3: capture trace_loc for queue entry.
                        let trace_loc = self.trace.last_inserted_trace_loc;
                        if let Some(violation) = violation {
                            self.handle_init_violation(violation, fp, || state.clone())?;
                        } else {
                            debug_eprintln!(
                                debug_states(),
                                "INIT STATE {} via Init: {:?}",
                                fp,
                                state
                            );
                        }
                        queue.push_back((
                            NoTraceQueueEntry::Bulk(BulkStateHandle::with_fingerprint(idx, fp)),
                            0,
                            trace_loc,
                        ));
                    }

                    self.stats.initial_states = self.states_count();
                    (bulk_storage, queue)
                }
            }
        };

        // Initialize states_found with initial states count
        self.stats.states_found = self.states_count();
        // Part of #2163: report both pre-dedup generated count and post-dedup distinct count
        self.report_init_progress(init_generated, self.stats.states_found);
        Ok(result)
    }

    /// Run the no-trace BFS loop using the unified `run_bfs_loop` implementation.
    ///
    /// Part of #2133: Delegates to `run_bfs_loop<FingerprintOnlyStorage>` instead of
    /// maintaining a separate copy of the BFS loop body.
    pub(in crate::check) fn check_impl_no_trace_mode(&mut self, init_name: &str) -> CheckResult {
        let registry = self.ctx.var_registry().clone();
        self.initialize_checkpoint_timing();
        // Part of #3175: Prepare inline liveness cache before BFS so that
        // record_inline_liveness_results() records bitmask data during BFS.
        // Without this, the post-BFS bitmask fast path is unavailable and
        // populate_node_check_masks falls back to eval which needs full states
        // that aren't stored in fingerprint-only mode.
        self.prepare_inline_liveness_cache();

        // Scratch ArrayState used to process bulk-backed states without per-state allocation.
        let mut bulk_scratch = ArrayState::new(registry.len());

        // Part of #1801: route init-state violations through finalize_terminal_result
        // so storage-error precedence applies even to early invariant violations.
        let (bulk_initial, mut queue) =
            match self.init_states_no_trace(init_name, &registry, &mut bulk_scratch) {
                Ok(result) => result,
                Err(result) => return self.finalize_terminal_result_with_storage(result),
            };

        // Part of #3986 / #4287: Infer flat i64 state layout from a wavefront of
        // initial states. Sampling multiple init states catches variable-shape
        // mismatches (e.g., IntArray lengths that differ across initials) that
        // single-state inference cannot detect. When shapes disagree, the
        // conflicting variable falls back to Dynamic, preventing index-out-of-
        // bounds crashes in `write_int_array_slots`/`write_record_slots` during
        // FlatState materialization.
        if !queue.is_empty() {
            let sample_size = std::cmp::min(bulk_initial.len(), 1024);
            if sample_size <= 1 {
                // Single init state: fall back to the original path.
                self.infer_flat_state_layout(&bulk_scratch);
            } else {
                let mut sample: Vec<ArrayState> = Vec::with_capacity(sample_size);
                let num_states = u32::try_from(bulk_initial.len()).unwrap_or(u32::MAX);
                let stride = std::cmp::max(1, bulk_initial.len() / sample_size);
                let mut idx: usize = 0;
                while sample.len() < sample_size && (idx as u32) < num_states {
                    let mut st = ArrayState::new(registry.len());
                    st.overwrite_from_slice(bulk_initial.get_state(idx as u32));
                    sample.push(st);
                    idx = idx.saturating_add(stride);
                }
                // Always include the last-processed init state for parity with
                // the previous single-state inference path.
                if sample.is_empty() {
                    sample.push(bulk_scratch.clone());
                }
                self.infer_flat_state_layout_from_wavefront(&sample);
            }
        }

        // Part of #3910: Upgrade JIT invariant cache with compound layout info
        // inferred from the first initial state. Uses bulk_scratch which holds
        // the last processed init state — sufficient for layout inference since
        // all states share the same variable types.
        let mut direct_flat_initial_frontier = None;
        if !queue.is_empty() {
            // The flat layout is now known. If it qualifies for the native-fused
            // fast path, release any auto-detected POR (mutually exclusive with
            // that path) before fingerprint activation, which also refuses to
            // activate while POR is set. Sound: releasing auto-POR never changes
            // the reachable-state set. See the method doc for the full rationale.
            self.maybe_release_auto_por_for_native_fused_admission();
            // The native compiler's transient allocations overlap the full FP64
            // initial-state collision map. On direct-flat candidates the bulk
            // arena + cached queue handles are an exact reconstruction authority,
            // so release only the redundant map during codegen. The map is either
            // superseded by compact CompiledFlat witnesses below or reconstructed
            // before BFS if final engine selection stays in ArrayFp64.
            let retired_initial_seen_before_codegen =
                self.retire_init_seen_before_codegen(&bulk_initial, &queue);
            self.upgrade_jit_cache_with_layout(&bulk_scratch);
            // AUTO engine-selection post-compile coverage gate: now that the
            // trust-cg cache is in final (layout-promoted) form, route to the
            // interpreter if native coverage/admission shows native is not
            // beneficial. Must run before fingerprint activation so the
            // fingerprint domain stays consistent with the chosen engine.
            self.auto_select_post_compile_trust_cg_gate();
            // Part of #3986: Verify that the flat BFS layout and native ABI layout agree
            // on buffer format. Log warning if incompatible.
            self.verify_layout_compatibility();
            // WP-11 slice 2: fail-closed flat-symmetry admission + install
            // (TY_FLAT_SYMMETRY=1 gated, default OFF — a no-op env probe
            // otherwise). Must run after layout inference (needs the fully-flat
            // adapter) and BEFORE `freeze_bfs_fingerprint_domain` + the init
            // re-fingerprint below, so init states and successors share the
            // flat-symmetry canonical hash domain from the first commit.
            self.maybe_install_flat_symmetry_canonicalizer();
            // Part of #3987: Activate compiled xxh3 fingerprinting if all conditions
            // are met (all-scalar, no VIEW, no SYMMETRY). Activates AFTER init states
            // are fingerprinted (since we need to see init states to verify all-scalar).
            // Init state fingerprints are re-hashed with xxh3 below to ensure consistency.
            self.try_activate_compiled_fingerprinting();

            // Part of #3987 / #4281: Re-fingerprint init states with xxh3 after activation.
            // Init states were fingerprinted with FP64 during init_states_no_trace() and
            // inserted into `seen_fps`. When xxh3 is now active, successors will be hashed
            // with xxh3; to keep `seen_fps` in a single domain we RESET the set and
            // re-insert xxh3 fingerprints for the init states. Keeping the stale FP64
            // entries alongside the xxh3 entries (the previous "phantom" approach) doubles
            // `states_count()` because FP64 and xxh3 fingerprints of the same state never
            // match, so both are counted. This caused the small-spec regression in #4281
            // (HourClock 12 → 24, ABCorrectness 20 → 26, AsynchInterface 12 → 20,
            // MCTwoPhase 4 → 5) after Stage 2c removed the old JIT feature gate.
            //
            // The reset is safe: no BFS successors have been inserted into `seen_fps` yet
            // (we're between init_states_no_trace and the BFS loop), and the FP64 entries
            // being dropped carry no information that isn't re-derived from the xxh3
            // re-fingerprint pass below.
            //
            // We use `FlatState::fingerprint_compiled()` (the same function the successor
            // path uses) instead of `array_state_fingerprint_xxh3` when a flat layout is
            // available. The two functions agree on scalar values but differ on heap-
            // wrapped integers (TAG_HEAP); using the same function as the successor path
            // guarantees domain equality.
            //
            // Part of #4281 follow-up (CatEvenBoxes/CatOddBoxes regression):
            // Re-fingerprinting must also fire when `flat_state_primary` is true even
            // if `jit_compiled_fp_active` is false. The flat-state-primary successor
            // path (`process_flat_state_primary_successors` in `full_state_successors.rs`)
            // unconditionally uses `flat.fingerprint_compiled()` for successor dedup.
            // When the init wavefront contains variables that hash differently between
            // FP64 (TAG_HEAP byte-hash) and the flat i64 representation — concretely,
            // `ScalarString` (flat encodes as NameId intern) and `ScalarModelValue` —
            // the two domains produce different fingerprints for the same state value.
            // `jit_compiled_fp_active` requires pure Int/Bool, but
            // `flat_state_primary` only requires `is_all_scalar()` which includes
            // `ScalarString` / `ScalarModelValue`. Without this re-fingerprint,
            // successors hashed with `fingerprint_compiled` never match init states
            // hashed with FP64, inflating the distinct-state count (e.g., Cat specs
            // saw exactly 2× states: 48 → 96, 30 → 60).
            //
            // Part of #4281 follow-up (HourClock2 PROPERTY regression):
            // The `flat_state_primary` reason must ONLY fire when the successor path
            // will actually use `fingerprint_compiled()`. In
            // `full_state_successors.rs::process_full_state_successors`, the flat-
            // primary path (line 111) is gated on the absence of batch-path triggers:
            //   flat_state_primary
            //     && !has_eval_implied_actions
            //     && !has_constraints
            //     && !has_por
            //     && !has_coverage
            //     && !has_symmetry
            //     && !has_view
            // When any batch-path trigger is set (e.g., PROPERTY with implied actions
            // like HC2's `[][HCnxt2]_hr`), successors are routed through the batch
            // path, which calls `array_state_fingerprint()` → FP64 cache. If we
            // re-fingerprint init states to xxh3 here regardless, `seen_fps` ends up
            // in xxh3 while successors arrive in FP64, double-counting every reachable
            // state (HC2: 12 → 24). Mirror the successor-path gate so the re-fingerprint
            // domain matches the domain the BFS will actually produce.
            //
            // `jit_compiled_fp_active` remains a sufficient condition on its own
            // because the `try_activate_compiled_fingerprinting` path in run_prepare
            // already refuses activation when any batch-path trigger is set
            // (run_prepare.rs:1216–1237), so this flag implies the flat/xxh3 path.
            // Freeze the fingerprint domain BEFORE the init states are committed
            // to the seen set below and before any successor is hashed. This is
            // the single point where init and successors must agree on the hash
            // domain; capturing it here makes a later mid-run AUTO lazy compile
            // (which installs the native fused level and would otherwise flip
            // `bfs_fingerprint_domain()` from ArrayFp64 to CompiledFlat) a no-op
            // for fingerprinting, so the distinct-state count stays exact.
            self.freeze_bfs_fingerprint_domain();
            // WP-11 slice 2: the flat-symmetry canonical domain re-uses the
            // exact CompiledFlat init re-fingerprint machinery below — init
            // states were hashed in the interpreter SymmetryCanonical domain
            // before the canonicalizer existed (layout inference needs an init
            // state), so `seen_fps` must be reset and re-seeded with
            // canonical-buffer hashes before any successor is committed. The
            // per-site hashes route through `flat_domain_refingerprint`, which
            // is byte-identical to `fingerprint_compiled()` for CompiledFlat
            // and adds the lexmin rewrite for FlatSymmetryCanonical.
            let flat_symmetry_domain_refp = matches!(
                self.bfs_fingerprint_domain(),
                super::fingerprint::BfsFingerprintDomain::FlatSymmetryCanonical
            );
            let need_flat_domain_refp =
                self.uses_compiled_bfs_fingerprint_domain() || flat_symmetry_domain_refp;
            if retired_initial_seen_before_codegen && !need_flat_domain_refp {
                if let Err(candidate) = self.reconstruct_init_seen_from_bulk_queue(
                    &bulk_initial,
                    &queue,
                    &mut bulk_scratch,
                ) {
                    return self.finalize_terminal_result_with_storage(candidate);
                }
            }
            // The direct flat-frontier init path stores raw buffers whose queue
            // fingerprints are computed inside
            // `refingerprint_initial_states_into_flat_frontier` without the
            // canonicalization hook; keep flat-symmetry runs on the (slower,
            // init-only) reset + re-insert branch instead (fail closed).
            let use_flat_for_direct_init = self.should_use_flat_bfs() && !flat_symmetry_domain_refp;
            if need_flat_domain_refp {
                let layout = self.flat_bfs_adapter.as_ref().map(|a| a.layout().clone());
                if use_flat_for_direct_init
                    && layout.is_some()
                    && self.can_build_flat_initial_frontier_direct(&bulk_initial, &queue)
                {
                    match self.refingerprint_initial_states_into_flat_frontier(
                        &bulk_initial,
                        &mut queue,
                        &mut bulk_scratch,
                        layout.clone().expect("layout checked above"),
                    ) {
                        Ok(flat_queue) => {
                            direct_flat_initial_frontier = Some(flat_queue);
                        }
                        Err(candidate) => {
                            return self.finalize_terminal_result_with_storage(candidate);
                        }
                    }
                } else {
                    // Drop the FP64 phantoms while preserving the configured backend.
                    let fresh_seen_fps = match self.state_storage.seen_fps.fresh_empty_clone() {
                        Ok(storage) => storage,
                        Err(fault) => {
                            let candidate = self.storage_fault_result(fault);
                            return self.finalize_terminal_result_with_storage(candidate);
                        }
                    };
                    self.state_storage.replace_seen_fps(fresh_seen_fps);
                    self.state_storage.seen.clear();

                    let mut init_states = std::mem::take(&mut self.liveness_cache.init_states);
                    for (fp, arr) in &mut init_states {
                        let old_fp = *fp;
                        let new_fp = if let Some(ref layout) = layout {
                            match crate::state::FlatState::try_from_array_state(
                                arr,
                                std::sync::Arc::clone(layout),
                            ) {
                                Ok(mut flat) => self.flat_domain_refingerprint(&mut flat),
                                Err(err) => {
                                    let candidate = self.flat_layout_unsupported_result(
                                        "liveness init refingerprint",
                                        &err,
                                    );
                                    return self.finalize_terminal_result_with_storage(candidate);
                                }
                            }
                        } else {
                            self.array_state_fingerprint_xxh3(arr)
                        };
                        *fp = new_fp;
                        if let Some(trace_loc) = self.trace.trace_locs.get(&old_fp) {
                            let _ = self.trace.trace_locs.insert(new_fp, trace_loc);
                        }
                        self.rekey_checkpoint_depth(old_fp, new_fp);
                    }
                    self.liveness_cache.init_states = init_states;
                    let num_states = u32::try_from(bulk_initial.len()).unwrap_or(0);
                    for idx in 0..num_states {
                        bulk_scratch.overwrite_from_slice(bulk_initial.get_state(idx));
                        let xxh3_fp = if let Some(ref layout) = layout {
                            match crate::state::FlatState::try_from_array_state(
                                &bulk_scratch,
                                std::sync::Arc::clone(layout),
                            ) {
                                Ok(mut flat) => self.flat_domain_refingerprint(&mut flat),
                                Err(err) => {
                                    let candidate = self.flat_layout_unsupported_result(
                                        "init state refingerprint",
                                        &err,
                                    );
                                    return self.finalize_terminal_result_with_storage(candidate);
                                }
                            }
                        } else {
                            self.array_state_fingerprint_xxh3(&bulk_scratch)
                        };
                        if let Err(candidate) = self.mark_state_seen_checked_with_current(
                            xxh3_fp,
                            &bulk_scratch,
                            None,
                            0,
                            None,
                        ) {
                            return self.finalize_terminal_result_with_storage(candidate);
                        }
                    }
                    let mut refingerprinted_queue = VecDeque::with_capacity(queue.len());
                    while let Some((entry, depth, trace_loc)) = queue.pop_front() {
                        let entry = match entry {
                            NoTraceQueueEntry::Bulk(mut handle) => {
                                let old_fp = handle.fingerprint;
                                bulk_scratch
                                    .overwrite_from_slice(bulk_initial.get_state(handle.index));
                                let fp = if let Some(ref layout) = layout {
                                    match crate::state::FlatState::try_from_array_state(
                                        &bulk_scratch,
                                        std::sync::Arc::clone(layout),
                                    ) {
                                        Ok(mut flat) => self.flat_domain_refingerprint(&mut flat),
                                        Err(err) => {
                                            let candidate = self.flat_layout_unsupported_result(
                                                "init queue refingerprint",
                                                &err,
                                            );
                                            return self
                                                .finalize_terminal_result_with_storage(candidate);
                                        }
                                    }
                                } else {
                                    self.array_state_fingerprint_xxh3(&bulk_scratch)
                                };
                                handle.fingerprint = Some(fp);
                                if let Some(old_fp) = old_fp {
                                    self.rekey_checkpoint_depth(old_fp, fp);
                                }
                                NoTraceQueueEntry::Bulk(handle)
                            }
                            NoTraceQueueEntry::Owned { state, .. } => {
                                let fp = if let Some(ref layout) = layout {
                                    match crate::state::FlatState::try_from_array_state(
                                        &state,
                                        std::sync::Arc::clone(layout),
                                    ) {
                                        Ok(mut flat) => self.flat_domain_refingerprint(&mut flat),
                                        Err(err) => {
                                            let candidate = self.flat_layout_unsupported_result(
                                                "init queue refingerprint (owned)",
                                                &err,
                                            );
                                            return self
                                                .finalize_terminal_result_with_storage(candidate);
                                        }
                                    }
                                } else {
                                    self.array_state_fingerprint_xxh3(&state)
                                };
                                NoTraceQueueEntry::Owned { state, fp }
                            }
                            witness @ NoTraceQueueEntry::Witness { .. } => {
                                // Witness handles are enabled only after init
                                // refingerprinting and storage construction.
                                witness
                            }
                            NoTraceQueueEntry::Flat { flat, .. } => {
                                // Non-mutating: the queued buffer must stay the
                                // RAW state (the explored representative), only
                                // its dedup fingerprint is canonical-domain.
                                let fp = self.flat_domain_fingerprint_of(&flat);
                                NoTraceQueueEntry::Flat { flat, fp }
                            }
                        };
                        refingerprinted_queue.push_back((entry, depth, trace_loc));
                    }
                    queue = refingerprinted_queue;
                    // `states_found` tracks unique seen-set size; the reset + re-insert
                    // resets the count to exactly the unique init-state count.
                    self.stats.initial_states = self.states_count();
                    self.stats.states_found = self.states_count();
                    if super::debug::bytecode_vm_stats_enabled() {
                        let reason = if self.jit_compiled_fp_active {
                            "jit_compiled_fp_active"
                        } else {
                            "flat_state_primary"
                        };
                        eprintln!(
                            "[jit-fp] Re-fingerprinted {} init states with xxh3 \
                             (seen_fps reset to xxh3 domain, reason={})",
                            num_states, reason,
                        );
                    }
                }
            }
        }

        // Part of #4126: Determine whether flat BFS should be active.
        //
        // Auto-detection: when the inferred layout is both roundtrip-verified
        // and safe for default flat admission, flat BFS activates automatically.
        // `TY_NO_FLAT_BFS=1` can force-disable.
        let use_flat = self.should_use_flat_bfs();

        // Log activation status.
        if use_flat {
            if let Some(ref adapter) = self.flat_bfs_adapter {
                let source = if self.config.use_flat_state == Some(true) {
                    "config.use_flat_state=true"
                } else {
                    "auto-detected (flat layout safe)"
                };
                telemetry_eprintln!(
                    "[flat-bfs] active ({}): {} slots/state, {} bytes/state, fully_flat={}",
                    source,
                    adapter.num_slots(),
                    adapter.bytes_per_state(),
                    adapter.is_fully_flat(),
                );
            }
        }
        if crate::check::debug::trust_cg_native_fused_strict() && !use_flat {
            return CheckResult::from_error(
                RuntimeCheckError::Internal(
                    "strict trust-codegen native fused requirement failed: flat BFS is not active"
                        .to_string(),
                )
                .into(),
                self.stats.clone(),
            );
        }

        // Nested-set dynamic-universe DISCOVERY (A4) — SHADOW / LOG-ONLY.
        // Behavior-neutral: gated behind `TY_NESTED_SET_DISCOVERY=1` (no-op
        // otherwise). The seeds are read from `bulk_initial` BEFORE it is moved
        // into `FingerprintOnlyStorage`. Never substitutes a layout — the run
        // stays byte-identical. See `run_bfs_full::shadow_discover_*` for the
        // full rationale. (No-op fast-path means no `bulk_initial` scan on a
        // normal run.)
        if std::env::var_os("TY_NESTED_SET_DISCOVERY").is_some_and(|v| v == "1") {
            let seed_count = std::cmp::min(bulk_initial.len(), 1024);
            let mut seeds: Vec<ArrayState> = Vec::with_capacity(seed_count);
            for idx in 0..seed_count {
                let mut st = ArrayState::new(registry.len());
                st.overwrite_from_slice(bulk_initial.get_state(idx as u32));
                seeds.push(st);
            }
            self.shadow_discover_nested_set_universes_from_seeds(&registry, seeds);
        }

        // Nested-set A5 — FREEZE + per-successor escape MONITOR (soundness gate).
        // Default-on when promotion is enabled AND the spec has a set-of-sets
        // state var (the SlidingPuzzles `board`); a no-op (no sampling, no
        // `bulk_initial` scan) otherwise, so non-nested specs are byte-identical.
        // Seeds are read from `bulk_initial` BEFORE it is moved into storage.
        if crate::state::nested_set_promotion_enabled() {
            let seed_count = std::cmp::min(bulk_initial.len(), 1024);
            let mut seeds: Vec<ArrayState> = Vec::with_capacity(seed_count);
            for idx in 0..seed_count {
                let mut st = ArrayState::new(registry.len());
                st.overwrite_from_slice(bulk_initial.get_state(idx as u32));
                seeds.push(st);
            }
            self.freeze_nested_set_monitors_from_seeds(&registry, &seeds);
        }

        // Step B — native slide-kernel arm. DEFAULT-ON via the static
        // recognizer (arms only when `Next` is PROVEN to be the rigid-unit
        // slide relation — `slide_recognize`); force-armable with
        // `TY_NESTED_SET_SLIDE=1`, killable with `TY_NO_NESTED_SET_SLIDE=1`.
        // Uses the INIT states only (no sampling), so arming is instant.
        // Seeds are read from `bulk_initial` BEFORE it is moved into storage.
        // Cheap default-path gate: the recognizer only accepts single-variable
        // specs, so multi-var specs skip even the seed copy.
        {
            use super::run_bfs_full::{nested_set_slide_arm_mode, SlideArmMode};
            let mode = nested_set_slide_arm_mode();
            let gate_ok = match mode {
                SlideArmMode::Off => false,
                SlideArmMode::Forced => true,
                SlideArmMode::Auto => registry.len() == 1,
            };
            if gate_ok {
                let seed_count = std::cmp::min(bulk_initial.len(), 1024);
                let mut seeds: Vec<ArrayState> = Vec::with_capacity(seed_count);
                for idx in 0..seed_count {
                    let mut st = ArrayState::new(registry.len());
                    st.overwrite_from_slice(bulk_initial.get_state(idx as u32));
                    seeds.push(st);
                }
                self.arm_nested_set_slide_kernel_from_seeds(&registry, &seeds);
            }
        }

        // mem2 (dead bulk-init drop): when the direct flat init frontier holds
        // every initial state, the bulk tree copy is DEAD — the frontier and all
        // successors are `Flat`, and `FingerprintOnlyStorage` reads `bulk_initial`
        // only to re-materialize `NoTraceQueueEntry::Bulk` entries, of which none
        // remain (the direct-flat path drained the bulk queue into flat buffers).
        // Retaining it held the full init-state tree materialization through BFS
        // (GameOfLife: 65536 grid functions, the dominant peak). Drop it here,
        // AFTER the nested-set/slide seed scans above have read it. Kill-switch:
        // TY_DROP_DEAD_BULK_INIT=0 restores retention.
        let bulk_initial =
            if direct_flat_initial_frontier.is_some() && Self::drop_dead_bulk_init_enabled() {
                BulkStateStorage::empty(registry.len())
            } else {
                bulk_initial
            };
        let mut storage = FingerprintOnlyStorage::new(bulk_initial, registry.len());
        let use_array_witness_frontier = storage.configure_array_witness_frontier(self, use_flat);

        // Part of #2881 Step 3: enable lazy trace index for the BFS loop.
        // Initial states above populated trace_locs eagerly (small count).
        // The BFS expansion loop below processes potentially millions of states,
        // so skipping trace_locs.insert per state eliminates the last per-state
        // HashMap write. The index is built lazily from the trace file if/when
        // trace reconstruction is needed (invariant violation, liveness check).
        self.trace.lazy_trace_index = true;

        // Part of #4126: Use arena-backed FlatBfsFrontier when flat BFS is active.
        // This stores flat i64 state buffers contiguously in a FlatStateStore arena
        // instead of individual Box<[i64]> per NoTraceQueueEntry::Flat, eliminating
        // per-state heap allocation on the BFS hot path and providing cache-friendly
        // sequential access during frontier iteration.
        if use_flat {
            let mut flat_queue = if let Some(flat_queue) = direct_flat_initial_frontier.take() {
                flat_queue
            } else {
                let layout = self
                    .flat_bfs_adapter
                    .as_ref()
                    .expect("invariant: flat_bfs_adapter present when use_flat")
                    .layout()
                    .clone();
                let mut flat_queue = super::bfs::flat_frontier::FlatBfsFrontier::with_capacity(
                    layout.clone(),
                    queue.len(),
                );
                // Part of #3986: Convert init states to FlatState when flat_state_primary.
                // When flat_state_primary=true, all vars are scalar and the flat i64
                // buffer is the primary BFS representation. Converting Bulk init states
                // to Flat entries ensures they go directly into the FlatBfsFrontier's
                // contiguous arena (hot path) instead of the fallback VecDeque (cold path).
                // This only changes the queued fingerprint when the run's canonical BFS
                // fingerprint domain is the flat-compiled domain. If a batch-path trigger
                // such as PROPERTY implied actions is active, successors still use FP64
                // fingerprints; the initial queue entry must stay in that same domain so
                // liveness parent keys and liveness init roots line up.
                {
                    if self.flat_state_primary || self.native_fused_flat_frontier_admission_active()
                    {
                        let use_compiled_init_fp = self.uses_compiled_bfs_fingerprint_domain();
                        let mut scratch = ArrayState::new(registry.len());
                        let mut converted = 0u64;
                        while let Some((entry, depth, trace_loc)) = queue.pop_front() {
                            match entry {
                                NoTraceQueueEntry::Bulk(handle) => {
                                    // Load the init state from BulkStateStorage.
                                    scratch.overwrite_from_slice(
                                        storage.bulk_initial.get_state(handle.index),
                                    );
                                    // Convert to FlatState via the inferred layout.
                                    // Fail closed (typed error, CLI retries
                                    // without flat) when the state cannot be
                                    // encoded in the fixed flat layout.
                                    let flat = match crate::state::FlatState::try_from_array_state(
                                        &scratch,
                                        std::sync::Arc::clone(&layout),
                                    ) {
                                        Ok(flat) => flat,
                                        Err(err) => {
                                            let candidate = self.flat_layout_unsupported_result(
                                                "init frontier flat conversion",
                                                &err,
                                            );
                                            return self
                                                .finalize_terminal_result_with_storage(candidate);
                                        }
                                    };
                                    // Use the same fingerprint domain that successors will use.
                                    // Flat queue storage is orthogonal to the dedup domain: with
                                    // implied actions, constraints, POR, coverage, VIEW, or
                                    // symmetry, the batch successor path fingerprints ArrayState
                                    // values with FP64 even though the frontier stores FlatState
                                    // buffers.
                                    let fp = if use_compiled_init_fp {
                                        flat.fingerprint_compiled()
                                    } else {
                                        handle
                                            .fingerprint
                                            .expect("invariant: init state handle has fingerprint")
                                    };
                                    flat_queue.push((
                                        NoTraceQueueEntry::Flat { flat, fp },
                                        depth,
                                        trace_loc,
                                    ));
                                    converted += 1;
                                }
                                other => {
                                    // Owned or already-Flat entries pass through.
                                    flat_queue.push((other, depth, trace_loc));
                                }
                            }
                        }
                        if super::debug::bytecode_vm_stats_enabled() && converted > 0 {
                            eprintln!(
                                "[flat-bfs] converted {} init states from Bulk to Flat for arena storage",
                                converted,
                            );
                        }
                        // mem2 (dead bulk-init drop): every init state was just
                        // converted from Bulk to Flat in `flat_queue`, so the bulk
                        // tree copy in `storage` is now dead — `FingerprintOnlyStorage`
                        // reads `bulk_initial` only to re-materialize
                        // `NoTraceQueueEntry::Bulk` entries, of which none remain
                        // (all successors are Flat/Owned). Retaining it held the full
                        // init-state tree materialization through BFS (GameOfLife:
                        // 65536 grid functions, the dominant transient peak). Free it
                        // here. Kill-switch: TY_DROP_DEAD_BULK_INIT=0.
                        if Self::drop_dead_bulk_init_enabled() {
                            storage.bulk_initial = BulkStateStorage::empty(registry.len());
                        }
                    } else {
                        // Transfer initial states from VecDeque to FlatBfsFrontier (fallback path).
                        while let Some(entry) = queue.pop_front() {
                            flat_queue.push(entry);
                        }
                    }
                }
                flat_queue
            };

            // Part of #3988 / #4171: When the compiled BFS level is available
            // and enabled, use the compiled level loop that processes the frontier
            // directly from the contiguous arena. This bypasses the interpreter
            // entirely — action dispatch + fingerprint + first-level dedup +
            // invariant checking all run in native compiled code.
            //
            // The `should_use_compiled_bfs()` check respects the hierarchy:
            //   1. config.use_compiled_bfs=false -> disabled
            //   2. TY_NO_COMPILED_BFS=1 -> disabled
            //   3. config.use_compiled_bfs=true -> enabled (if level ready)
            //   4. Auto-detect: enabled when all-scalar + all JIT-compiled
            {
                if self.should_use_compiled_bfs() {
                    if flat_queue.has_fallback_entries() || flat_queue.remaining_flat_count() == 0 {
                        eprintln!(
                            "[compiled-bfs] compiled BFS disabled: frontier has no flat parents ready"
                        );
                        if crate::check::debug::trust_cg_native_fused_strict() {
                            return CheckResult::from_error(
                                RuntimeCheckError::Internal(
                                    "strict trust-codegen native fused requirement failed: frontier has no flat parents ready"
                                        .to_string(),
                                )
                                .into(),
                                self.stats.clone(),
                            );
                        }
                    } else {
                        // Report the activation source for diagnostics.
                        let source = if self.config.use_compiled_bfs == Some(true) {
                            "config.use_compiled_bfs=true"
                        } else {
                            "auto-detected (all-scalar, fully JIT-compiled)"
                        };
                        telemetry_eprintln!(
                            "[compiled-bfs] activating compiled BFS loop ({source})"
                        );
                        self.record_engine_tier(true);
                        let result = self.run_compiled_bfs_loop(&mut storage, &mut flat_queue);
                        flat_queue.report_stats();
                        return result;
                    }
                } else if crate::check::debug::trust_cg_native_fused_strict() {
                    return CheckResult::from_error(
                        RuntimeCheckError::Internal(
                            "strict trust-codegen native fused requirement failed: compiled BFS did not activate"
                                .to_string(),
                        )
                        .into(),
                        self.stats.clone(),
                    );
                } else if self.compiled_bfs_step.is_some() {
                    // Compiled BFS step is available but auto-detection criteria
                    // not met (e.g. compound types in state). Log once.
                    let has_compound = self
                        .flat_bfs_adapter
                        .as_ref()
                        .is_some_and(|a| !a.is_fully_flat());
                    if has_compound {
                        eprintln!(
                            "[compiled-bfs] compiled BFS step available but state has \
                             compound types — interpreter path used."
                        );
                    } else {
                        eprintln!(
                            "[compiled-bfs] compiled BFS step available but not activated \
                             (auto-detection criteria not met)."
                        );
                    }
                }
            }

            self.record_engine_tier(false);
            // Part B (Tier-1 #5, auto tier-up): drive the interpreter BFS loop,
            // but allow a one-time hot-swap to the compiled BFS loop once an
            // AUTO-mode lazy trust-cg compile fires mid-run and installs
            // compiled-BFS artifacts (`maybe_trigger_trust_cg_lazy_compile` ->
            // `initialize_trust_cg_cache`). The interpreter loop yields control
            // back here at a level boundary when
            // `trust_cg_should_hot_swap_to_compiled_bfs()` holds (frontier still
            // in the flat arena, layout flat-primary safe), at which point the
            // remaining frontier is handed to `run_compiled_bfs_loop`. Each
            // pass re-checks the admission guards, so an unsafe frontier (e.g.
            // fallback entries) simply keeps running on the interpreter.
            //
            // The loop runs at most twice in practice: the interpreter pass,
            // then (if the lazy compile fired and admission holds) a single
            // compiled pass to drain the rest. The bound is provided by the
            // hot-swap predicate becoming false once the compiled loop owns the
            // frontier (it never returns control with the predicate still set),
            // but we additionally cap re-entries defensively.
            let mut result = self.run_bfs_loop(&mut storage, &mut flat_queue);
            let mut swaps_remaining = 1u8;
            while swaps_remaining > 0
                && self.trust_cg_should_hot_swap_to_compiled_bfs()
                && !flat_queue.has_fallback_entries()
                && flat_queue.remaining_flat_count() > 0
            {
                swaps_remaining -= 1;
                eprintln!(
                    "[compiled-bfs] auto tier-up: lazy compile installed compiled-BFS \
                     artifacts mid-run; hot-swapping interpreter loop to the compiled \
                     BFS loop at a level boundary ({} flat parents remaining)",
                    flat_queue.remaining_flat_count(),
                );
                self.record_engine_tier(true);
                result = self.run_compiled_bfs_loop(&mut storage, &mut flat_queue);
            }
            flat_queue.report_stats();
            result
        } else if use_array_witness_frontier {
            self.record_engine_tier(false);
            let mut witness_queue = super::bfs::witness_frontier::WitnessBfsFrontier::new();
            while let Some(entry) = queue.pop_front() {
                witness_queue.push(entry);
            }
            self.run_bfs_loop(&mut storage, &mut witness_queue)
        } else {
            self.record_engine_tier(false);
            self.run_bfs_loop(&mut storage, &mut queue)
        }
    }
}
