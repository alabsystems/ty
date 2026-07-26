// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Full-state BFS mode: initial state generation and BFS exploration loop.
//!
//! This module handles the `store_full_states` path where `ArrayState` objects
//! are kept in a `HashMap` for trace reconstruction.

use super::bfs::FullStateStorage;
#[cfg(debug_assertions)]
use super::debug::debug_states;
use super::{
    check_error_to_result, ArrayState, CheckResult, Fingerprint, ModelChecker, State, Trace,
    VecDeque,
};
use crate::{ConfigCheckError, EvalCheckError};

/// Arm mode for the nested-set slide kernel (Step B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::check) enum SlideArmMode {
    /// `TY_NO_NESTED_SET_SLIDE=1` — kill switch: never arm.
    Off,
    /// `TY_NESTED_SET_SLIDE=1` — force-arm from the INIT bounding box (the
    /// original opt-in override; no recognizer proof required).
    Forced,
    /// Default: arm ONLY when the static recognizer PROVES the spec's `Next`
    /// is the rigid-unit-slide relation (`slide_recognize`).
    Auto,
}

/// Resolve the slide-kernel arm mode from the environment. The kill switch
/// wins over the force flag.
pub(in crate::check) fn nested_set_slide_arm_mode() -> SlideArmMode {
    if std::env::var_os("TY_NO_NESTED_SET_SLIDE").is_some_and(|v| v == "1") {
        SlideArmMode::Off
    } else if std::env::var_os("TY_NESTED_SET_SLIDE").is_some_and(|v| v == "1") {
        SlideArmMode::Forced
    } else {
        SlideArmMode::Auto
    }
}

/// Number of leading states whose kernel successors are crosschecked against
/// the interpreter when the DEFAULT (recognizer) arm is active — the always-on
/// tripwire. Bounded, so the cost is a run-constant. Sized empirically: at
/// ~2.3ms interpreter successor generation per Klotski state, 8 states is
/// ~18ms ≈ 1% of even the SMALLEST armed run (SlidingPuzzles, ~1.6s), and a
/// vanishing fraction of anything larger. Detection power is front-loaded
/// anyway: a wrongly-armed relation diverges at the very first states (the
/// kernel and interpreter disagree on the successor SET of the init states
/// themselves), so the window is about catching a recognizer false-accept
/// immediately, not sampling deep into the run. Divergence DISARMS the kernel
/// and the run falls back to the interpreter.
pub(in crate::check) const SLIDE_TRIPWIRE_STATES: usize = 8;

impl ModelChecker<'_> {
    /// Generate initial states, then apply state constraints (CONSTRAINT directive).
    ///
    /// Part of #2133 (phase 2): shared fallback helper used by both full-state
    /// and no-trace initialization paths to avoid duplicated filtering logic.
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn constrained_initial_states(
        &mut self,
        init_name: &str,
    ) -> Result<(Vec<State>, usize), CheckResult> {
        let (initial_states, raw_initial_states_generated) = self
            .generate_initial_states_with_raw_count(init_name)
            .map_err(|error| check_error_to_result(error, &self.stats))?;
        // Make the complete raw count available to any terminal constraint
        // error before the outer init-progress report runs.
        self.stats.raw_initial_states_generated = raw_initial_states_generated;

        let registry = self.ctx.var_registry().clone();
        let mut constrained_initial_states = Vec::with_capacity(initial_states.len());
        for state in initial_states {
            let arr = ArrayState::from_state(&state, &registry);
            match self.check_state_constraints_array(&arr) {
                Ok(true) => constrained_initial_states.push(state),
                Ok(false) => {}
                Err(error) => {
                    return Err(check_error_to_result(error, &self.stats));
                }
            }
        }

        Ok((constrained_initial_states, raw_initial_states_generated))
    }

    /// Process a single initial state through all pre-admission checks.
    ///
    /// Part of #2473: shared helper used by both full-state and no-trace init paths
    /// to eliminate duplicated constraint/materialize/fingerprint/invariant logic.
    ///
    /// Performs (in order): state constraint checking (if `check_constraints`), lazy
    /// value materialization, fingerprinting, deduplication, and invariant checking.
    ///
    /// Returns:
    /// - `Ok(None)` — state pruned by constraints or duplicate fingerprint
    /// - `Ok(Some((fp, violation)))` — state should be admitted; `violation` is `Some`
    ///   if an invariant was violated (continue-on-error mode)
    /// - `Err(CheckResult)` — fatal error (eval error, storage fault)
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn check_init_state(
        &mut self,
        arr: &mut ArrayState,
        check_constraints: bool,
    ) -> Result<Option<(Fingerprint, Option<String>)>, CheckResult> {
        // Check state constraints (CONSTRAINT directive) if not already filtered
        if check_constraints {
            match self.check_state_constraints_array(arr) {
                Ok(true) => {}
                Ok(false) => return Ok(None),
                Err(e) => {
                    return Err(check_error_to_result(e, &self.stats));
                }
            }
        }

        // Part of #2018: Materialize lazy values before fingerprinting.
        // Part of #2356/#2777: Route through check_error_to_result so
        // ExitRequested maps to LimitReached(Exit).
        if let Err(e) = crate::materialize::materialize_array_state(
            &self.ctx,
            arr,
            self.compiled.spec_may_produce_lazy,
        ) {
            return Err(check_error_to_result(
                EvalCheckError::Eval(e).into(),
                &self.stats,
            ));
        }

        // Compute fingerprint for deduplication.
        //
        // Part of #2708: the dedup check that was here (`is_state_seen_checked`)
        // created a TOCTOU gap — the caller performs the actual atomic
        // `mark_state_seen_*` insertion, so a separate contains_checked is
        // redundant for sequential mode and racy for parallel mode. The caller
        // now uses the InsertOutcome from `mark_state_seen_*` to decide whether
        // to enqueue, matching TLC's FPSet.put() single-operation pattern.
        let fp = self
            .array_state_fingerprint(arr)
            .map_err(|e| CheckResult::from_error(e, self.stats.clone()))?;

        // Check invariants
        // Part of #1117: Set trace context for TLCExt Trace support
        self.set_trace_context_for_init_array(arr);
        let violation = match self.check_invariants_array(arr) {
            Ok(v) => {
                self.clear_trace_context();
                v
            }
            Err(e) => {
                self.clear_trace_context();
                return Err(check_error_to_result(e, &self.stats));
            }
        };
        if violation.is_some() {
            return Ok(Some((fp, violation)));
        }

        // Check property init predicates (#2834): non-Always state-level terms
        // from PROPERTY entries (e.g., M!Init in "M!Init /\ [][M!Next]_M!vars").
        let init_pred_violation = match crate::checker_ops::check_property_init_predicates(
            &mut self.ctx,
            &self.compiled.property_init_predicates,
            arr,
        ) {
            Ok(v) => v,
            Err(e) => return Err(check_error_to_result(e, &self.stats)),
        };

        Ok(Some((fp, init_pred_violation)))
    }

    /// Handle an invariant violation during init-state processing.
    ///
    /// Part of #2473: shared helper for violation recording/trace-construction.
    /// The `make_trace_state` closure lazily constructs the State for the error
    /// trace, avoiding allocation in the continue-on-error case.
    ///
    /// Returns `Ok(())` if continue_on_error absorbed the violation, or
    /// `Err(CheckResult)` (InvariantViolation or PropertyViolation) if the checker should stop.
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn handle_init_violation(
        &mut self,
        violation: String,
        fp: Fingerprint,
        make_trace_state: impl FnOnce() -> State,
    ) -> Result<(), CheckResult> {
        if self.record_invariant_violation(violation.clone(), fp) {
            let trace = Trace::from_states(vec![make_trace_state()]);
            // Part of #2676: check if this invariant was promoted from a PROPERTY entry.
            return Err(
                if self
                    .compiled
                    .promoted_property_invariants
                    .contains(&violation)
                {
                    CheckResult::PropertyViolation {
                        property: violation,
                        kind: crate::check::api::PropertyViolationKind::StateLevel,
                        trace,
                        stats: self.stats.clone(),
                    }
                } else {
                    CheckResult::InvariantViolation {
                        invariant: violation,
                        trace,
                        stats: self.stats.clone(),
                    }
                },
            );
        }
        Ok(())
    }

    /// Generate initial states for full-state BFS mode.
    ///
    /// Tries streaming enumeration first (avoids Vec<State> OrdMap overhead),
    /// then falls back to the Vec<State> path. Returns the initial BFS queue
    /// or an early CheckResult on error/violation.
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn init_states_full_state(
        &mut self,
        init_name: &str,
        registry: &crate::var_index::VarRegistry,
    ) -> Result<VecDeque<(Fingerprint, usize)>, CheckResult> {
        // Part of #3305: streaming invariant scan — O(1) memory per state.
        // For specs like Einstein (~199M init states), this finds invariant
        // violations without materializing the full state space into BulkStateStorage.
        self.scan_init_invariants_streaming(init_name)?;

        let mut queue: VecDeque<(Fingerprint, usize)> = VecDeque::new();
        let mut init_generated: usize = 0;
        let used_streaming = if let Some(bulk_init) =
            self.solve_predicate_for_states_to_bulk_prechecked(init_name)?
        {
            let init_generated_count = bulk_init.enumeration.generated;
            self.stats.raw_initial_states_generated = init_generated_count;
            let bulk_storage = bulk_init.storage;
            let mut scratch = ArrayState::new(registry.len());
            let num_states = u32::try_from(bulk_storage.len()).map_err(|_| {
                CheckResult::from_error(
                    ConfigCheckError::Setup(format!(
                        "too many initial states ({}) for u32 BulkStateStorage index",
                        bulk_storage.len()
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
                scratch.overwrite_from_slice(bulk_storage.get_state(idx));
                let fp = self.prepare_prechecked_initial_state(&mut scratch)?;

                #[cfg(debug_assertions)]
                if debug_states() {
                    let state = scratch.to_state(registry);
                    eprintln!("INIT STATE {fp} via Init: {state:?}");
                }

                #[cfg(feature = "memory-stats")]
                {
                    crate::value::memory_stats::inc_array_state();
                    crate::value::memory_stats::inc_array_state_bytes(scratch.len());
                }
                let mut arr = scratch.clone();
                if self.compiled.cached_view_name.is_none() && self.symmetry.perms.is_empty() {
                    let _ = arr.fingerprint(registry);
                }

                let liveness_arr = if self.track_liveness_init_states() {
                    Some(arr.clone())
                } else {
                    None
                };
                if !self.mark_state_seen_owned_checked(fp, arr, None, 0)? {
                    debug_eprintln!(debug_states(), "DUP INIT STATE {}", fp);
                    continue;
                }
                if let Some(liveness_arr) = liveness_arr {
                    self.liveness_cache.init_states.push((fp, liveness_arr));
                }
                queue.push_back((fp, 0));
            }

            init_generated = init_generated_count;
            self.stats.initial_states = queue.len();
            true
        } else {
            let streaming_result = self.generate_initial_states_to_bulk(init_name);
            // Part of #1433: Propagate actual eval errors instead of silently falling back.
            // Ok(None) = streaming not available (fall back to Vec<State> path).
            // Err(e)   = real eval error (propagate immediately).
            match streaming_result {
                Err(e) => {
                    return Err(check_error_to_result(e, &self.stats));
                }
                Ok(None) => false,
                Ok(Some(bulk_init)) => {
                    let init_generated_count = bulk_init.enumeration.generated;
                    self.stats.raw_initial_states_generated = init_generated_count;
                    let bulk_storage = bulk_init.storage;
                    // Streaming successful! Process states from BulkStateStorage directly.
                    // Filter by constraints and add to seen.
                    let mut scratch = ArrayState::new(registry.len());
                    let num_states = u32::try_from(bulk_storage.len()).map_err(|_| {
                        CheckResult::from_error(
                            ConfigCheckError::Setup(format!(
                                "too many initial states ({}) for u32 BulkStateStorage index",
                                bulk_storage.len()
                            ))
                            .into(),
                            self.stats.clone(),
                        )
                    })?;

                    // Part of #254: Set TLC level for TLCGet("level") - TLC uses 1-based indexing
                    // Initial states are at level 1 in TLC
                    self.ctx.set_tlc_level(1);

                    // Filter by constraints, check invariants, and store states
                    for idx in 0..num_states {
                        // --max-states: stop admitting initial states at the limit.
                        if self.init_state_limit_reached() {
                            break;
                        }
                        scratch.overwrite_from_slice(bulk_storage.get_state(idx));

                        // Part of #2473: Use shared check_init_state helper
                        let (fp, violation) = match self.check_init_state(&mut scratch, true)? {
                            Some(result) => result,
                            None => continue,
                        };

                        #[cfg(debug_assertions)]
                        if debug_states() {
                            let state = scratch.to_state(registry);
                            eprintln!("INIT STATE {fp} via Init: {state:?}");
                        }

                        // Create ArrayState for storage (clone from scratch)
                        #[cfg(feature = "memory-stats")]
                        {
                            crate::value::memory_stats::inc_array_state();
                            crate::value::memory_stats::inc_array_state_bytes(scratch.len());
                        }
                        let mut arr = scratch.clone();
                        if self.compiled.cached_view_name.is_none()
                            && self.symmetry.perms.is_empty()
                        {
                            let _ = arr.fingerprint(registry);
                        }

                        // Part of #2708: clone for liveness cache BEFORE the
                        // move into mark_state_seen, push only after dedup succeeds.
                        let liveness_arr = if self.track_liveness_init_states() {
                            Some(arr.clone())
                        } else {
                            None
                        };
                        // Part of #2708: atomic test-and-set replaces the old two-step
                        // is_state_seen_checked + mark_state_seen pattern. The return
                        // value is the dedup authority — skip if already present.
                        if !self.mark_state_seen_owned_checked(fp, arr, None, 0)? {
                            debug_eprintln!(debug_states(), "DUP INIT STATE {}", fp);
                            continue;
                        }
                        // Part of #3175: cache init states for post-BFS liveness
                        if let Some(liveness_arr) = liveness_arr {
                            self.liveness_cache.init_states.push((fp, liveness_arr));
                        }
                        if let Some(violation) = violation {
                            self.handle_init_violation(violation, fp, || {
                                scratch.to_state(registry)
                            })?;
                        }
                        // Initial states are at depth 0.
                        queue.push_back((fp, 0));
                    }

                    init_generated = init_generated_count;
                    self.stats.initial_states = queue.len();
                    true
                }
            }
        };

        // Fall back to Vec<State> path if streaming not available
        if !used_streaming {
            let (initial_states, raw_initial_states_generated) =
                self.constrained_initial_states(init_name)?;
            init_generated = raw_initial_states_generated;

            // Part of #254: Set TLC level for TLCGet("level") - TLC uses 1-based indexing
            // Initial states are at level 1 in TLC
            self.ctx.set_tlc_level(1);

            // Check initial states and mark as seen in a single pass.
            // Part of #595: Handle continue_on_error for initial states.
            for state in &initial_states {
                // --max-states: stop admitting initial states at the limit.
                if self.init_state_limit_reached() {
                    break;
                }
                // Part of #158: Use from_state (NOT from_state_with_fp) so fingerprint
                // is computed fresh using registry order. from_state_with_fp copies
                // State's alphabetical-order fingerprint, causing mismatch.
                let mut arr = ArrayState::from_state(state, registry);

                // Part of #2473: Use shared check_init_state helper
                // check_constraints=false: already filtered by constrained_initial_states
                let (fp, violation) = match self.check_init_state(&mut arr, false)? {
                    Some(result) => result,
                    None => continue,
                };

                // Part of #2708: clone for liveness BEFORE the move, push
                // only after dedup succeeds (prevents duplicate cache entries).
                let liveness_arr = if self.track_liveness_init_states() {
                    Some(arr.clone())
                } else {
                    None
                };
                // Part of #2708: atomic test-and-set — skip if already present.
                if !self.mark_state_seen_owned_checked(fp, arr, None, 0)? {
                    debug_eprintln!(debug_states(), "DUP INIT STATE {}", fp);
                    continue;
                }
                // Part of #3175: cache init states for post-BFS liveness
                if let Some(liveness_arr) = liveness_arr {
                    self.liveness_cache.init_states.push((fp, liveness_arr));
                }
                if let Some(violation) = violation {
                    self.handle_init_violation(violation, fp, || state.clone())?;
                } else {
                    debug_eprintln!(debug_states(), "INIT STATE {} via Init: {:?}", fp, state);
                }
                // Initial states are at depth 0.
                queue.push_back((fp, 0));
            }

            // Part of #2163: set initial_states to post-dedup count, consistent
            // with the streaming path (was initial_states.len() — pre-dedup).
            self.stats.initial_states = queue.len();

            // Explicitly drop Vec<State> to release OrdMap memory
            drop(initial_states);
        }

        // Initialize states_found with initial states count
        self.stats.states_found = self.states_count();
        // Part of #2163: report both pre-dedup generated count and post-dedup distinct count
        self.report_init_progress(init_generated, self.stats.states_found);
        Ok(queue)
    }

    /// Run the full-state BFS loop using the unified `run_bfs_loop` implementation.
    ///
    /// Part of #2133: Delegates to `run_bfs_loop<FullStateStorage>` instead of
    /// maintaining a separate copy of the BFS loop body.
    pub(in crate::check) fn check_impl_full_state_mode(&mut self, init_name: &str) -> CheckResult {
        let registry = self.ctx.var_registry().clone();
        self.initialize_checkpoint_timing();
        self.prepare_inline_liveness_cache();

        // Part of #1801: route init-state violations through finalize_terminal_result
        // so storage-error precedence applies even to early invariant violations.
        let mut queue = match self.init_states_full_state(init_name, &registry) {
            Ok(q) => q,
            Err(result) => return self.finalize_terminal_result_with_storage(result),
        };

        // Part of #3986 / #4287: Infer flat i64 state layout from a wavefront
        // of initial states when more than one is available. Single-state
        // inference cannot detect variable-shape mismatches across init states
        // (e.g., an IntFunc whose length depends on another variable), which
        // later causes `write_int_array_slots` to panic with index-out-of-bounds
        // during FlatState materialization.
        let mut wavefront_sample: Vec<ArrayState> = Vec::new();
        if wavefront_sample.is_empty() {
            wavefront_sample.extend(
                self.liveness_cache
                    .init_states
                    .iter()
                    .take(1024)
                    .map(|(_, a)| a.clone()),
            );
        }
        if wavefront_sample.is_empty() {
            wavefront_sample.extend(self.state_storage.seen.values().take(1024).cloned());
        }
        if wavefront_sample.len() >= 2 {
            self.infer_flat_state_layout_from_wavefront(&wavefront_sample);
        } else if let Some(first_init) = wavefront_sample.into_iter().next() {
            self.infer_flat_state_layout(&first_init);
        }

        // Part of #3910: Upgrade JIT invariant cache with compound layout info
        // inferred from the first initial state. This enables native record/function
        // access in JIT-compiled invariants instead of falling back to the interpreter.
        if let Some(first_init) = self.get_first_init_state_for_layout() {
            // The flat layout is now known. If it qualifies for the
            // native-fused / compiled-BFS fast path, release any auto-detected
            // POR (mutually exclusive with that path). Mirrors the no-trace path
            // (run_bfs_notrace.rs); sound because releasing auto-POR never
            // changes the reachable-state set. Full-state storage stays on the
            // FP64 successor path, so the flat-primary release candidate is a
            // no-op here today, but keeping the call symmetric guards against
            // future full-state compiled-BFS admission diverging silently.
            self.maybe_release_auto_por_for_native_fused_admission();
            self.upgrade_jit_cache_with_layout(&first_init);
            // AUTO engine-selection post-compile coverage gate (see
            // `auto_select_post_compile_trust_cg_gate`). Route to the interpreter
            // when native coverage/admission shows native is not beneficial.
            self.auto_select_post_compile_trust_cg_gate();
            // Part of #3986: Verify that the flat BFS layout and native ABI layout agree
            // on buffer format. Log warning if incompatible.
            self.verify_layout_compatibility();
            // Part of #3987 / #4281: Compiled xxh3 fingerprinting is gated off in
            // the full-state path by `try_activate_compiled_fingerprinting`
            // (`store_full_states == true` short-circuits activation). Full-state
            // mode needs single-domain consistency between the `seen` HashMap
            // (keyed on FP64 from init_states_full_state) and `seen_fps`, plus
            // downstream liveness/trace reconstruction that dereferences
            // `seen.get(fp)` using the same fingerprint domain produced by BFS.
            // Re-keying the populated `seen` HashMap mid-run would be invasive
            // and error-prone, so we simply stay on FP64 for the whole full-state
            // run. xxh3 remains enabled for the notrace path (run_bfs_notrace.rs)
            // where `seen_fps` is the sole state store.
            self.try_activate_compiled_fingerprinting();
        }

        // Nested-set dynamic-universe DISCOVERY (A4) — SHADOW / LOG-ONLY.
        //
        // Behavior-neutral: gated entirely behind `TY_NESTED_SET_DISCOVERY=1`
        // (no-op otherwise), it samples successors, derives the would-be
        // two-level `NestedSetBitmask` universe for each set-of-sets state var,
        // validates that sampled boards round-trip through the A3 codec, and
        // LOGS the result. It NEVER substitutes a layout — `self.flat_state_*`
        // is untouched — so the run stays byte-identical. Promotion + the
        // per-successor monitor are A5.
        self.shadow_discover_nested_set_universes(init_name);

        // Nested-set A5 — FREEZE + per-successor escape MONITOR (the soundness
        // gate). Default-on when promotion is enabled AND the spec has a
        // set-of-sets state var; a no-op (no sampling) otherwise, so non-nested
        // specs stay byte-identical. Installs `nested_set_monitors`, which the
        // dedup-fingerprint hook (`array_state_fingerprint`) consults on EVERY
        // successor and fails closed on any out-of-universe board.
        self.freeze_nested_set_monitors(init_name);

        // Step B — native slide-kernel arm. DEFAULT-ON via the static
        // recognizer (arms only when `Next` is PROVEN to be the rigid-unit
        // slide relation — see `slide_recognize`); force-armable with
        // `TY_NESTED_SET_SLIDE=1`, killable with `TY_NO_NESTED_SET_SLIDE=1`.
        // When armed, the variable's successors are generated by word-ops over
        // piece bitmasks. Not-armed → byte-identical interpreter run.
        self.arm_nested_set_slide_kernel();

        let mut storage = FullStateStorage;
        // Full-state (trace-capable) mode always runs the interpreter loop —
        // compiled BFS is a notrace-path capability. Record the tier so
        // engine provenance covers this path too.
        self.record_engine_tier(false);
        self.run_bfs_loop(&mut storage, &mut queue)
    }

    /// SHADOW (nested-set A4): sample successors, derive the would-be
    /// `NestedSetBitmask` universe for each set-of-sets state var, validate the
    /// A3-codec round-trip, and LOG it. Never promotes a layout — fully
    /// behavior-neutral, gated behind `TY_NESTED_SET_DISCOVERY=1`.
    fn shadow_discover_nested_set_universes(&mut self, init_name: &str) {
        // Env gate (fast pre-check): the entire pass is a no-op unless
        // explicitly enabled, so a normal run (incl. spec_regression) is
        // byte-identical — not even the seed gather runs.
        if std::env::var_os("TY_NESTED_SET_DISCOVERY").is_none_or(|v| v != "1") {
            return;
        }
        let _ = init_name;
        let registry = self.ctx.var_registry().clone();
        // Seed states: prefer cached init `ArrayState`s, else the seen store.
        let mut seeds: Vec<ArrayState> = self
            .liveness_cache
            .init_states
            .iter()
            .take(1024)
            .map(|(_, a)| a.clone())
            .collect();
        if seeds.is_empty() {
            seeds.extend(self.state_storage.seen.values().take(1024).cloned());
        }
        self.shadow_discover_nested_set_universes_from_seeds(&registry, seeds);
    }

    /// Nested-set A5 freeze entry (full-state path): gather seed init states and
    /// install the per-successor escape monitors. No-op unless promotion is
    /// enabled and a set-of-sets state var exists.
    fn freeze_nested_set_monitors(&mut self, init_name: &str) {
        if !crate::state::nested_set_promotion_enabled() {
            return;
        }
        let _ = init_name;
        let registry = self.ctx.var_registry().clone();
        let mut seeds: Vec<ArrayState> = self
            .liveness_cache
            .init_states
            .iter()
            .take(1024)
            .map(|(_, a)| a.clone())
            .collect();
        if seeds.is_empty() {
            seeds.extend(self.state_storage.seen.values().take(1024).cloned());
        }
        if seeds.is_empty() {
            return;
        }
        self.freeze_nested_set_monitors_from_seeds(&registry, &seeds);
    }

    /// Step B — arm the native slide-kernel successor fast-path.
    ///
    /// Three modes (see [`nested_set_slide_arm_mode`]):
    ///
    /// * **Auto (default)** — arm ONLY when the static recognizer
    ///   ([`super::slide_recognize`]) PROVES the spec's `Next` is the
    ///   rigid-unit-slide relation, with the EXACT evaluated `Pos` grid.
    ///   Anything unproven: no arm, byte-identical interpreter run.
    /// * **Forced (`TY_NESTED_SET_SLIDE=1`)** — the original opt-in override:
    ///   arm blindly from the INIT bounding box (validated by the per-state
    ///   crosscheck, `TY_NESTED_SET_SLIDE_CROSSCHECK=1`).
    /// * **Off (`TY_NO_NESTED_SET_SLIDE=1`)** — kill switch: never arm.
    ///
    /// Uses only the INIT states (no sampling BFS), so arming is instant.
    fn arm_nested_set_slide_kernel(&mut self) {
        let mode = nested_set_slide_arm_mode();
        if mode == SlideArmMode::Off {
            return;
        }
        let registry = self.ctx.var_registry().clone();
        // Cheap default-path gate: the recognizer only ever accepts a
        // single-variable spec, so multi-var specs skip even the seed copy.
        if mode == SlideArmMode::Auto && registry.len() != 1 {
            return;
        }
        let seeds: Vec<ArrayState> = self
            .liveness_cache
            .init_states
            .iter()
            .take(1024)
            .map(|(_, a)| a.clone())
            .collect();
        let seeds = if seeds.is_empty() {
            self.state_storage
                .seen
                .values()
                .take(1024)
                .cloned()
                .collect()
        } else {
            seeds
        };
        if seeds.is_empty() {
            return;
        }
        self.arm_nested_set_slide_kernel_from_seeds(&registry, &seeds);
    }

    /// Step B — arm the slide kernel from explicit seed states. Shared by the
    /// full-state and fingerprint-only (notrace) BFS entries. Dispatches on
    /// [`nested_set_slide_arm_mode`]; a no-op when nothing is proven / fits.
    pub(in crate::check) fn arm_nested_set_slide_kernel_from_seeds(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        seeds: &[ArrayState],
    ) {
        match nested_set_slide_arm_mode() {
            SlideArmMode::Off => {}
            SlideArmMode::Forced
                if self.coverage.collect && !self.coverage.default_dead_action_tracking => {}
            SlideArmMode::Forced => self.arm_nested_set_slide_forced(registry, seeds),
            SlideArmMode::Auto => self.arm_nested_set_slide_auto(registry, seeds),
        }
    }

    /// DEFAULT arm path: recognizer-proven only, fail closed on everything
    /// else — every `return` below leaves the run byte-identical on the
    /// interpreter. On success, arms with the EXACT evaluated `Pos` grid and
    /// enables the first-[`SLIDE_TRIPWIRE_STATES`]-states tripwire.
    fn arm_nested_set_slide_auto(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        seeds: &[ArrayState],
    ) {
        if seeds.is_empty() || registry.len() != 1 {
            return;
        }
        // Config surfaces whose semantics flow through the per-action
        // successor machinery that an armed run short-circuits (or that need
        // successor structure beyond the reachable-set/verdict): fail closed.
        if !self.auto_slide_arm_config_allows() {
            return;
        }
        let vi = 0usize;
        // Every INIT state's board must be a set-of-sets value.
        let init_boards: Vec<crate::Value> = seeds
            .iter()
            .map(|s| crate::Value::from(&s.values()[vi]))
            .collect();
        if !init_boards.iter().all(crate::state::is_nested_set_value) {
            return;
        }
        // THE PROOF: `Next` must statically BE the rigid-unit-slide relation.
        let Some(next_name) = self
            .trace
            .cached_next_name
            .clone()
            .or_else(|| self.config.next.clone())
        else {
            return;
        };
        let Some(recognized) = super::slide_recognize::recognize_slide_next(&self.ctx, &next_name)
        else {
            return;
        };
        debug_assert_eq!(recognized.board_var_idx, vi);
        // Value-level preconditions on the INIT boards (cells inside the
        // recognized grid, pieces pairwise disjoint — see the constructor).
        let board_refs: Vec<&crate::Value> = init_boards.iter().collect();
        let Some(arm) = crate::state::SlideKernelArm::try_arm_recognized(
            recognized.board_var_idx,
            recognized.positions,
            &board_refs,
        ) else {
            return;
        };
        let var_name = registry
            .name(crate::var_index::VarIndex::new(vi))
            .to_string();
        eprintln!(
            "[nested-set-slide] ARMED (recognized slide relation; native word-op successors): \
             var '{var_name}' grid={} cells [tripwire: first {} states kernel-vs-interpreter]{}",
            arm.geometry.num_positions(),
            SLIDE_TRIPWIRE_STATES,
            if std::env::var_os("TY_NESTED_SET_SLIDE_CROSSCHECK").is_some_and(|v| v == "1") {
                " [crosscheck ON: kernel vs interpreter per state]"
            } else {
                ""
            },
        );
        // The recognized kernel preserves the reachable-state graph but has no
        // action-attribution channel. Only the implicit default V2 diagnostic
        // may yield; explicit display, track-only, and guided coverage were
        // rejected by `auto_slide_arm_config_allows` above.
        self.yield_default_dead_action_tracking_to_slide();
        self.nested_set_slide_tripwire = SLIDE_TRIPWIRE_STATES;
        self.nested_set_slide_arm = Some(arm);
    }

    fn yield_default_dead_action_tracking_to_slide(&mut self) {
        if !self.coverage.default_dead_action_tracking {
            return;
        }
        self.coverage.collect = false;
        self.coverage.default_dead_action_tracking = false;
        self.coverage.native_fast_path_skipped = false;
        self.stats.coverage = None;
    }

    /// Config gates for the DEFAULT (recognizer) arm. The kernel bypasses the
    /// per-action successor dispatch, so any configured feature that consumes
    /// per-action structure or successor-graph detail beyond the reachable
    /// set fails closed to the interpreter. (The FORCED path skips these — it
    /// is an explicit validation override.)
    ///
    /// `ALIAS` deliberately does NOT gate the arm: ty (like TLC) evaluates the
    /// alias operator only when FORMATTING an already-reconstructed error
    /// trace (`Trace::apply_alias`, called post-check from the CLI on the
    /// final `CheckResult`). The kernel changes only HOW successors are
    /// enumerated — the successor SET is recognizer-proven identical, full
    /// `ArrayState`s flow through the normal BFS parent tracking, and trace
    /// reconstruction + alias formatting run on the same concrete states
    /// either way (validated end-to-end on `SlidingPuzzles_anim`, whose
    /// `Next` IS the inherited slide relation and whose only config delta is
    /// `ALIAS AnimAlias`).
    fn auto_slide_arm_config_allows(&self) -> bool {
        self.config.symmetry.is_none()
            && self.config.view.is_none()
            && self.config.constraints.is_empty()
            && self.config.action_constraints.is_empty()
            && self.config.properties.is_empty()
            && self.config.trace_invariants.is_empty()
            && self.config.postcondition.is_none()
            && self.config.terminal.is_none()
            && !self.config.por_enabled
            // Any liveness machinery (incl. fairness from a SPECIFICATION)
            // needs the real successor-edge structure: fail closed.
            && !self.liveness_cache.cache_for_liveness
            // Explicit action attribution needs the per-action path. Implicit
            // default tracking may yield after the recognizer proof succeeds.
            && (!self.coverage.collect || self.coverage.default_dead_action_tracking)
    }

    /// FORCED arm path (`TY_NESTED_SET_SLIDE=1`) — the original opt-in
    /// override, unchanged: INIT-bounding-box grid, no recognizer, validated
    /// by the per-state crosscheck.
    fn arm_nested_set_slide_forced(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        seeds: &[ArrayState],
    ) {
        if seeds.is_empty() {
            return;
        }
        let nested_var_indices = Self::nested_set_var_indices(registry, seeds);
        let Some(&vi) = nested_var_indices.first() else {
            eprintln!("[nested-set-slide] no set-of-sets state var found; not armed");
            return;
        };
        let init_boards: Vec<crate::Value> = seeds
            .iter()
            .filter_map(|s| {
                let v = crate::Value::from(&s.values()[vi]);
                crate::state::is_nested_set_value(&v).then_some(v)
            })
            .collect();
        let board_refs: Vec<&crate::Value> = init_boards.iter().collect();
        match crate::state::SlideKernelArm::try_arm(vi, &board_refs) {
            Some(arm) => {
                let var_name = registry
                    .name(crate::var_index::VarIndex::new(vi))
                    .to_string();
                eprintln!(
                    "[nested-set-slide] ARMED (native word-op successors): var '{var_name}' \
                     grid={} cells{}",
                    arm.geometry.num_positions(),
                    if std::env::var_os("TY_NESTED_SET_SLIDE_CROSSCHECK").is_some_and(|v| v == "1")
                    {
                        " [crosscheck ON: kernel vs interpreter per state]"
                    } else {
                        ""
                    },
                );
                self.yield_default_dead_action_tracking_to_slide();
                self.nested_set_slide_arm = Some(arm);
            }
            None => {
                eprintln!(
                    "[nested-set-slide] board var did not fit the slide-grid shape; not armed \
                     (falling back to interpreter)"
                );
            }
        }
    }

    /// FREEZE (nested-set discovery A5): when `NESTED_SET_PROMOTION_ENABLED`,
    /// run the successor-aware sampler over the seed states, derive the frozen
    /// universe per set-of-sets var, and install a per-successor escape
    /// [`crate::state::NestedSetVarMonitor`] for each. Default-on (NOT
    /// env-gated): triggers ONLY when a set-of-sets state var exists, so every
    /// spec without one is byte-identical and never pays for the pass.
    ///
    /// Soundness: the frozen universe is the discovered (sampled) one. A board
    /// reachable in the real run but outside the sampled universe ESCAPES and the
    /// monitor fails closed (bails the var to the interpreter's raw
    /// `value_fingerprint`, same domain → no aliasing). For `SlidingPuzzles` the
    /// sampler (bounded BFS) covers a superset of the real run, so 0 escapes.
    pub(in crate::check) fn freeze_nested_set_monitors_from_seeds(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        seeds: &[ArrayState],
    ) {
        if !crate::state::nested_set_promotion_enabled() {
            return;
        }
        let Some(next_name) = self.config.next.clone() else {
            return;
        };
        let nested_var_indices = Self::nested_set_var_indices(registry, seeds);
        if nested_var_indices.is_empty() {
            // Fast path for every non-nested spec: byte-identical, no sampling.
            return;
        }

        // CHEAP DISCOVERY (gap 2): when `Next` is the recognized rigid-slide
        // relation, the complete piece-shape universe is derivable STATICALLY
        // (all grid-fitting translates of the INIT pieces) with NO sampling BFS
        // and NO full-state retention — see `derive_nested_set_universe_static`.
        // Only vars WITHOUT a static universe fall back to the expensive sampler,
        // so a recognized spec (`SlidingPuzzles`) pays ~O(#pieces × grid) instead
        // of a full reachable-space re-exploration.
        let static_universes = self.try_static_nested_set_universes(registry, seeds);
        let needs_sampling: Vec<usize> = nested_var_indices
            .iter()
            .copied()
            .filter(|vi| !static_universes.contains_key(vi))
            .collect();
        let samples = if needs_sampling.is_empty() {
            std::collections::BTreeMap::new()
        } else {
            self.sample_nested_set_boards(registry, &next_name, seeds, &needs_sampling)
        };

        for &vi in &nested_var_indices {
            let var_name = registry
                .name(crate::var_index::VarIndex::new(vi))
                .to_string();
            let (discovered, provenance, sampled_count) =
                if let Some(discovered) = static_universes.get(&vi) {
                    (discovered.clone(), "static", 0usize)
                } else {
                    let board_samples = samples.get(&vi).map(Vec::as_slice).unwrap_or(&[]);
                    let sample_refs: Vec<&crate::Value> = board_samples.iter().collect();
                    let Some(discovered) = crate::state::derive_nested_set_universe(&sample_refs)
                    else {
                        eprintln!(
                            "[nested-set] A5: var '{var_name}' did not converge to a finite \
                             universe (|distinct boards|={}); staying Dynamic (no monitor)",
                            board_samples.len(),
                        );
                        continue;
                    };
                    (discovered, "sampled", board_samples.len())
                };
            let Some(monitor) = crate::state::freeze_nested_set_var(vi, &discovered) else {
                eprintln!("[nested-set] A5: var '{var_name}' freeze failed; staying Dynamic");
                continue;
            };
            telemetry_eprintln!(
                "[nested-set] A5 PROMOTED (monitor_enforced, {provenance}): var '{var_name}' \
                 |inner_universe|={} |outer_universe|={} distinct_boards_sampled={} slots={}",
                discovered.inner_len,
                discovered.outer_len,
                sampled_count,
                monitor.slot_count(),
            );
            self.nested_set_monitors.push(monitor);
        }
    }

    /// CHEAP DISCOVERY (gap 2): derive the nested-set universe STATICALLY (no
    /// sampling BFS) for any board variable whose `Next` the static recognizer
    /// PROVES is the rigid-unit-slide relation. Under that proof every reachable
    /// board's pieces are rigid translates of the INIT pieces, so the complete
    /// piece-shape universe is exactly the grid-fitting translates of the INIT
    /// pieces — enumerable in O(#pieces × grid) with zero exploration. Returns a
    /// per-var map of the derived universes; empty when nothing is recognized
    /// (the caller then falls back to the sampler for those vars).
    fn try_static_nested_set_universes(
        &self,
        registry: &crate::var_index::VarRegistry,
        seeds: &[ArrayState],
    ) -> std::collections::BTreeMap<usize, crate::state::DiscoveredNestedSet> {
        let mut out = std::collections::BTreeMap::new();
        // The slide recognizer only accepts single-variable specs.
        if registry.len() != 1 || seeds.is_empty() {
            return out;
        }
        let Some(next_name) = self
            .trace
            .cached_next_name
            .clone()
            .or_else(|| self.config.next.clone())
        else {
            return out;
        };
        let Some(recognized) = super::slide_recognize::recognize_slide_next(&self.ctx, &next_name)
        else {
            return out;
        };
        let vi = recognized.board_var_idx;
        let init_boards: Vec<crate::Value> = seeds
            .iter()
            .filter_map(|s| {
                let v = crate::Value::from(&s.values()[vi]);
                crate::state::is_nested_set_value(&v).then_some(v)
            })
            .collect();
        if init_boards.is_empty() {
            return out;
        }
        let board_refs: Vec<&crate::Value> = init_boards.iter().collect();
        if let Some(discovered) =
            crate::state::derive_nested_set_universe_static(&recognized.positions, &board_refs)
        {
            out.insert(vi, discovered);
        }
        out
    }

    /// Identify set-of-sets ("nested set") state vars from the seeds. A var
    /// qualifies if it holds a non-empty set-of-sets in any seed.
    fn nested_set_var_indices(
        registry: &crate::var_index::VarRegistry,
        seeds: &[ArrayState],
    ) -> Vec<usize> {
        let mut nested_var_indices: Vec<usize> = Vec::new();
        for (idx, _name) in registry.iter() {
            let i = idx.as_usize();
            let is_nested = seeds
                .iter()
                .any(|s| crate::state::is_nested_set_value(&crate::Value::from(&s.values()[i])));
            if is_nested {
                nested_var_indices.push(i);
            }
        }
        nested_var_indices
    }

    /// Successor-aware sampler (shared by shadow-discovery A4 and freeze A5/A6): a
    /// bounded BFS prefix over the seed states, collecting every distinct board
    /// value seen for each nested-set var (INIT plus the successors that reveal
    /// the sliding piece-shapes). The bounds keep the pass cheap on huge specs;
    /// both are overridable (diagnostic-only) so a full-space convergence check
    /// is possible. Returns per-var distinct-board samples (deduped by value).
    ///
    /// FOLD (A6): the prefix now expands successors via the SAME fast
    /// array-native enumeration the main BFS uses
    /// (`generate_successors_filtered_array` over `ArrayState`s), NOT the slow
    /// `State`/`OrdMap` `generate_successors` path. The discovery prefix therefore
    /// costs roughly one fast BFS pass (reusing the main expansion machinery)
    /// instead of one slow State-materializing pass — the bulk of the A5 "extra
    /// BFS" regression. Board values are read directly off the flat `ArrayState`
    /// (`Value::from(&arr.values()[vi])`), avoiding the per-state OrdMap build.
    fn sample_nested_set_boards(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        next_name: &str,
        seeds: &[ArrayState],
        nested_var_indices: &[usize],
    ) -> std::collections::BTreeMap<usize, Vec<crate::Value>> {
        let _ = next_name; // array-native enumeration resolves Next internally.
        let max_sampled_states: usize = std::env::var("TY_NESTED_SET_MAX_STATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);
        let max_successor_expansions: usize = std::env::var("TY_NESTED_SET_MAX_EXPANSIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);

        // Per-var collected board samples (deduped by value).
        let mut board_samples: std::collections::BTreeMap<usize, Vec<crate::Value>> =
            nested_var_indices
                .iter()
                .map(|&i| (i, Vec::new()))
                .collect();
        let mut seen_boards: std::collections::BTreeMap<
            usize,
            std::collections::BTreeSet<crate::Value>,
        > = nested_var_indices
            .iter()
            .map(|&i| (i, std::collections::BTreeSet::new()))
            .collect();

        // Record the board value of each nested-set var directly off the flat
        // ArrayState (no OrdMap / State materialization on the sampler hot path).
        let record_array =
            |arr: &ArrayState,
             board_samples: &mut std::collections::BTreeMap<usize, Vec<crate::Value>>,
             seen_boards: &mut std::collections::BTreeMap<
                usize,
                std::collections::BTreeSet<crate::Value>,
            >| {
                for &vi in nested_var_indices {
                    if vi >= arr.values().len() {
                        continue;
                    }
                    let val = crate::Value::from(&arr.values()[vi]);
                    if crate::state::is_nested_set_value(&val)
                        && seen_boards
                            .get_mut(&vi)
                            .is_some_and(|seen| seen.insert(val.clone()))
                    {
                        board_samples.entry(vi).or_default().push(val);
                    }
                }
            };

        // Frontier of ArrayStates to expand; seed with init states. Fast
        // array-native enumeration mirrors the main BFS expansion.
        let mut frontier: std::collections::VecDeque<ArrayState> =
            std::collections::VecDeque::new();
        let mut global_seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for arr in seeds.iter() {
            let arr = arr.clone();
            record_array(&arr, &mut board_samples, &mut seen_boards);
            frontier.push_back(arr);
        }

        let mut expansions = 0usize;
        let mut sampled = frontier.len();
        while let Some(arr) = frontier.pop_front() {
            if expansions >= max_successor_expansions || sampled >= max_sampled_states {
                break;
            }
            expansions += 1;
            let result = match self.generate_successors_filtered_array(&arr) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for mut succ in result.successors {
                if sampled >= max_sampled_states {
                    break;
                }
                let fp = succ.fingerprint(registry).0;
                if !global_seen.insert(fp) {
                    continue;
                }
                sampled += 1;
                record_array(&succ, &mut board_samples, &mut seen_boards);
                frontier.push_back(succ);
            }
        }
        board_samples
    }

    /// Shared shadow-discovery body: given seed init `ArrayState`s, run the
    /// successor-aware sampler + derive + validate + log. Behavior-neutral; the
    /// env gate is rechecked here so a caller cannot accidentally promote.
    pub(in crate::check) fn shadow_discover_nested_set_universes_from_seeds(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        seeds: Vec<ArrayState>,
    ) {
        if std::env::var_os("TY_NESTED_SET_DISCOVERY").is_none_or(|v| v != "1") {
            return;
        }
        let Some(next_name) = self.config.next.clone() else {
            return;
        };
        let registry = registry.clone();
        if seeds.is_empty() {
            eprintln!("[nested-set] discovery: no seed states; skipping");
            return;
        }

        // Identify set-of-sets ("nested set") state vars from the seeds. A var
        // qualifies if it holds a non-empty set-of-sets in any seed.
        let nested_var_indices = Self::nested_set_var_indices(&registry, &seeds);
        if nested_var_indices.is_empty() {
            eprintln!("[nested-set] discovery: no set-of-sets state vars found; skipping");
            return;
        }

        // Successor-aware sampler (shared with the A5 freeze path).
        let board_samples =
            self.sample_nested_set_boards(&registry, &next_name, &seeds, &nested_var_indices);

        // Derive + validate + log per nested-set var.
        for &vi in &nested_var_indices {
            let var_name = registry
                .name(crate::var_index::VarIndex::new(vi))
                .to_string();
            let samples = board_samples.get(&vi).map(Vec::as_slice).unwrap_or(&[]);
            let sample_refs: Vec<&crate::Value> = samples.iter().collect();
            let Some(discovered) = crate::state::derive_nested_set_universe(&sample_refs) else {
                eprintln!(
                    "[nested-set] discovery: var '{}' did not converge to a finite NestedSetBitmask universe (|distinct boards|={}); staying Dynamic (shadow)",
                    var_name,
                    samples.len(),
                );
                continue;
            };
            let report = crate::state::validate_nested_set_roundtrip(&discovered, &sample_refs);
            eprintln!(
                "[nested-set] discovery (SHADOW, not promoted): var '{}' \
                 |inner_universe|={} |outer_universe|={} distinct_boards_sampled={} \
                 roundtrip_ok={} escapes={} slots={}",
                var_name,
                discovered.inner_len,
                discovered.outer_len,
                report.sampled_boards,
                report.roundtrip_ok,
                report.escapes,
                discovered.layout.slot_count(),
            );
        }
    }
}
