// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Full-state successor processing for BFS iterations.
//!
//! Part of #2677 Phase 2/3: the full-state (explicit fingerprinting) path
//! for successor generation and pipeline processing. This is the fallback
//! path when diff-based processing is unavailable or returns `None`.
//!
//! Called from the BFS loop in `engine.rs`. Uses shared helpers from
//! `successor_processing.rs`.
//!
//! Part of #2881 Step 1: fused successor processing — materialize, fingerprint,
//! implied actions, dedup, invariant check, and enqueue happen in a single pass
//! per successor. This eliminates the intermediate `Vec<(ArrayState, Fingerprint)>`,
//! enables early termination on invariant violations, and matches TLC's
//! `Worker.addElement()` streaming pattern.

use super::super::frontier::BfsFrontier;
use super::super::run_helpers::{BfsProfile, FlatPrefilteredSuccessorResult};
use super::super::{ArrayState, Fingerprint, ModelChecker};
use super::compiled_step_trait::{CompiledBfsStepScratch, CompiledStepOutput};
use super::iter_state::BfsIterState;
use super::observer::{CompositeObserver, ExplorationSignal, SuccessorObservationCtx};
use super::storage_modes::BfsStorage;
use super::successor_processing::{BfsAdmissionBatch, BfsIterOutcome, PendingSuccessor};
use super::BfsStepParams;
use crate::EvalCheckError;
use std::cell::RefCell;

thread_local! {
    static COMPILED_BFS_STEP_SCRATCH: RefCell<CompiledBfsStepScratch> =
        RefCell::new(CompiledBfsStepScratch::new(0));
}

// ============ WP-34 lever 2: parent-delta lazy-value normalization ==========
//
// WP-29 measured `batch_consume` at 5,957 ms over btree's 2,820,090 successors
// and named the shape of the fix: the streaming engine fingerprints and dedups
// WITHOUT materializing, while the batch consumer normalizes EVERY successor
// first. The normalization is `materialize_array_state`, whose cost is a
// `has_lazy_state_value` walk of every state variable — for btree that means
// re-walking `childOf` (32 entries), `valOf` (32) and `keysOf` (8 sets) on all
// 2.82M successors, ~87% of which the very next dedup probe throws away.
//
// The lever-2 pattern applies directly: a per-action successor differs from its
// parent in only a handful of variables, and every variable it did NOT write is
// the parent's own payload. So walk the parent ONCE per parent, and then per
// successor walk only the variables whose compact slot is not bit-identical to
// a parent slot already proven lazy-free.
//
// SOUNDNESS: a variable is skipped only under a proof of VALUE IDENTITY with a
// parent variable the per-parent walk proved lazy-free:
//   * an inline compact slot (Bool / i61 int / interned String / model value)
//     can hold no `LazyFunc`, `SetPred` or `Closure` at all — no parent needed;
//   * `CompactValue::bits_eq` is `true` only for identical inline slots or the
//     SAME `Box<Value>` pointer;
//   * `value_payload_identical` compares the `Arc` INSIDE two heap slots, which
//     is what a per-action successor actually shares with its parent: the
//     reconstruction re-boxes the `Value` (so the box pointers differ) but
//     clones the same payload `Arc`.
// Each is exact identity, never a heuristic. A skipped variable therefore holds
// exactly a value already proven free of lazy payloads, which is precisely when
// `materialize_array_state` leaves it untouched. The resulting state, its
// fingerprint, and any error raised are identical to the unconditional call —
// this is work ELISION, not deferral, so no successor can reach dedup or
// storage un-normalized. Any width disagreement falls back to the plain call.

/// Per-parent lazy-value profile: which of the parent's variables are already
/// free of lazy payloads. Only built when the spec's AST can produce lazy
/// values at all; otherwise the consumer keeps the original early-returning
/// call.
struct ParentLazyProfile {
    lazy_free: Vec<bool>,
}

impl ParentLazyProfile {
    fn build(parent: &ArrayState) -> Self {
        let len = parent.len();
        let mut lazy_free = Vec::with_capacity(len);
        for i in 0..len {
            let cv = parent.get_compact(crate::var_index::VarIndex::new(i));
            lazy_free.push(if cv.is_heap() {
                !crate::materialize::has_lazy_state_value(cv.as_heap_value())
            } else {
                true
            });
        }
        Self { lazy_free }
    }
}

/// Whether two heap `Value`s share the same payload allocation. `true` proves
/// they are the SAME value; `false` proves nothing and the caller scans.
///
/// Only the variants a state variable can actually hold are listed; everything
/// else (including every lazy variant — `SetPred`, `LazyFunc`, `Closure` — which
/// must always be scanned) answers `false`.
fn value_payload_identical(a: &crate::Value, b: &crate::Value) -> bool {
    use crate::Value as V;
    use tla_value::Rp;
    match (a, b) {
        (V::Set(x), V::Set(y)) => Rp::ptr_eq(x, y),
        (V::Func(x), V::Func(y)) => Rp::ptr_eq(x, y),
        (V::IntFunc(x), V::IntFunc(y)) => Rp::ptr_eq(x, y),
        (V::Bag(x), V::Bag(y)) => Rp::ptr_eq(x, y),
        (V::Seq(x), V::Seq(y)) => Rp::ptr_eq(x, y),
        (V::Tuple(x), V::Tuple(y)) => Rp::ptr_eq(x, y),
        (V::String(x), V::String(y)) => Rp::ptr_eq(x, y),
        (V::ModelValue(x), V::ModelValue(y)) => Rp::ptr_eq(x, y),
        (V::Int(x), V::Int(y)) => Rp::ptr_eq(x, y),
        (V::Interval(x), V::Interval(y)) => Rp::ptr_eq(x, y),
        (V::Record(x), V::Record(y)) => x.storage_ptr_identity() == y.storage_ptr_identity(),
        _ => false,
    }
}

/// Outcome of one successor's normalization: `(scanned_vars, skipped_all)`.
type LazyScanOutcome = (u32, bool);

/// Normalize `succ` against `parent`, walking only the variables not proven
/// identical to a lazy-free parent variable. Semantically identical to
/// `materialize_array_state(ctx, succ, true)`.
fn materialize_delta_from_parent(
    ctx: &crate::eval::EvalCtx,
    profile: &ParentLazyProfile,
    parent: &ArrayState,
    succ: &mut ArrayState,
) -> crate::eval::EvalResult<LazyScanOutcome> {
    use crate::var_index::VarIndex;

    let len = succ.len();
    if len != profile.lazy_free.len() || len != parent.len() {
        // Width disagreement: fall back to the unconditional pass.
        crate::materialize::materialize_array_state(ctx, succ, true)?;
        return Ok((len as u32, false));
    }
    let mut scanned = 0u32;
    for i in 0..len {
        let idx = VarIndex::new(i);
        let needs_scan = {
            let cv = succ.get_compact(idx);
            if !cv.is_heap() {
                // Inline slot: no lazy payload is representable.
                false
            } else if !profile.lazy_free[i] {
                true
            } else {
                let pv = parent.get_compact(idx);
                !(cv.bits_eq(pv)
                    || (pv.is_heap()
                        && value_payload_identical(cv.as_heap_value(), pv.as_heap_value())))
            }
        };
        if !needs_scan {
            continue;
        }
        scanned += 1;
        let materialized = {
            let cv = succ.get_compact(idx);
            let value = cv.as_heap_value();
            if crate::materialize::has_lazy_state_value(value) {
                Some(crate::materialize::materialize_value(ctx, value)?)
            } else {
                None
            }
        };
        if let Some(materialized) = materialized {
            succ.set(idx, materialized);
        }
    }
    Ok((scanned, scanned == 0))
}

#[inline]
const fn monolithic_jit_route_ready(has_coverage: bool, jit_ready: bool) -> bool {
    !has_coverage && jit_ready
}

impl ModelChecker<'_> {
    /// Try the compiled BFS step using caller-owned successor scratch.
    ///
    /// This mirrors `try_compiled_bfs_step` without forcing trust_cg's scoped step
    /// output through an owned `FlatBfsStepOutput`. The returned slices borrow
    /// `scratch` and are consumed before the scratch is reused.
    fn try_compiled_bfs_step_scoped<'a>(
        &mut self,
        current_array: &ArrayState,
        scratch: &'a mut CompiledBfsStepScratch,
    ) -> Option<CompiledStepOutput<'a>> {
        if self.jit_monolithic_disabled {
            return None;
        }
        let state_len = self.compiled_bfs_step.as_ref()?.state_len();

        self.jit_state_scratch.reserve(state_len);
        if !super::super::invariants::flatten_state_to_i64_selective(
            current_array,
            &mut self.jit_state_scratch,
            &[],
        ) {
            return None;
        }

        let output = {
            let step = self.compiled_bfs_step.as_ref()?;
            step.step_flat_scoped(&self.jit_state_scratch, scratch)
        };

        match output {
            Ok(output) => Some(output),
            Err(e) => {
                eprintln!("[jit] CompiledBfsStep error: {e} -- disabling");
                self.jit_monolithic_disabled = true;
                self.compiled_bfs_step = None;
                self.compiled_bfs_level = None;
                None
            }
        }
    }

    #[allow(clippy::result_large_err)]
    fn cache_full_state_batch_liveness<S: BfsStorage>(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        parent_fp: Fingerprint,
        liveness_data: &[(ArrayState, Fingerprint)],
    ) -> Result<(), BfsIterOutcome> {
        if let Err(error) = storage.cache_full_liveness(parent_fp, liveness_data, self) {
            return Err(BfsIterOutcome::Terminate(
                self.bfs_error_return(iter_state, storage, error),
            ));
        }
        Ok(())
    }

    /// Process full-state successors for one BFS iteration.
    ///
    /// Part of #2677 Phase 2, refactored in #2881 Step 1.
    ///
    /// Uses a fused single-pass pipeline matching TLC's `Worker.addElement()`
    /// pattern: each successor is materialized, fingerprinted, checked for
    /// implied actions, deduped, invariant-checked, and enqueued in one pass.
    /// This eliminates the intermediate `Vec<(ArrayState, Fingerprint)>` and
    /// enables early termination on invariant violations — if the first
    /// successor violates an invariant, remaining successors are not processed.
    pub(super) fn process_full_state_successors<
        S: BfsStorage,
        Q: BfsFrontier<Entry = S::QueueEntry>,
    >(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
    ) -> BfsIterOutcome {
        let &BfsStepParams {
            registry: _registry,
            current_depth: _current_depth,
            succ_depth,
            current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();
        let has_eval_implied_actions = !self.compiled.eval_implied_actions.is_empty();
        let cache_for_liveness = self.liveness_cache.cache_for_liveness;
        let _next_uses_tir_eval = self.cached_next_uses_tir_eval();

        // Part of #3027: Try streaming path for the common case.
        // Streaming uses split borrows to do fingerprint + dedup inline during
        // enumeration. Falls back to the batch path below for specs with implied
        // actions, constraints, action tagging, VIEW, symmetry, or coverage
        // collection (which need ctx/state-aware fingerprinting or tagged generation).
        //
        // Part of #3294: TIR eval no longer gates streaming — tir_leaf is now
        // threaded through the streaming enumeration API.
        let has_constraints =
            !self.config.constraints.is_empty() || !self.config.action_constraints.is_empty();
        // POR currently runs through the per-action batch dispatcher, which computes
        // enabled actions and applies ample-set reduction before materialization.
        let has_por = self.por.independence.is_some();
        // Part of #3354 Slice 1: per-action tagging requires compiled split
        // actions which are being removed from the sequential checker.
        // The tagged path is disabled; monolithic enumeration is always used.
        let use_tagged = false;
        let has_coverage = self.coverage.collect && !self.coverage.actions.is_empty();
        let has_symmetry = !self.symmetry.perms.is_empty();
        let has_view = self.compiled.cached_view_name.is_some();

        // Part of #3986: Flat state primary path. When all state variables are
        // scalar (Int/Bool), no VIEW, no SYMMETRY, use the flat [i64] buffer
        // as the primary BFS representation. This eliminates the
        // ArrayState→eval→ArrayState interpreter sandwich:
        //   - Pop FlatState from frontier (already done by resolve())
        //   - generate_successors_filtered_flat() → Vec<FlatState>
        //   - FlatState::fingerprint_compiled() for xxh3 dedup
        //   - Unflatten to ArrayState ONLY for new states (cold path)
        //
        // Gated: flat_state_primary + jit feature + no complex features.
        if self.flat_state_primary
            && !has_eval_implied_actions
            && !has_constraints
            && !has_por
            && !has_coverage
            && !has_symmetry
            && !has_view
        {
            if let Some(layout) = self.flat_state_layout.clone() {
                return self.process_flat_state_primary_successors(
                    iter_state,
                    storage,
                    queue,
                    params,
                    prof,
                    layout,
                    cache_for_liveness,
                );
            }
        }

        // Part of #4034: Try compiled BFS step first. This performs the entire
        // BFS inner loop (action dispatch, inline fingerprinting, dedup,
        // invariant checking) in a single native compiled function.
        // The compiled step uses its own AtomicFpSet for first-level dedup;
        // successors still pass through the model checker's global seen set.
        //
        // This path is only available when ALL actions AND ALL invariants
        // are JIT-compiled and the state is fully flat (no compound types).
        // It bypasses implied actions, constraints, POR, coverage, symmetry,
        // and VIEW — those features require the interpreter path.
        if self.compiled_bfs_step.is_some()
            && !has_eval_implied_actions
            && !has_constraints
            && !has_por
            && !has_coverage
            && !has_symmetry
            && !has_view
        {
            let compiled_outcome = COMPILED_BFS_STEP_SCRATCH.with(|scratch| {
                let mut step_scratch = scratch.borrow_mut();
                self.try_compiled_bfs_step_scoped(iter_state.array(), &mut step_scratch)
                    .map(|output| {
                        self.process_compiled_bfs_output(
                            iter_state, storage, queue, params, prof, output,
                        )
                    })
            });
            if let Some(outcome) = compiled_outcome {
                return outcome;
            }
        }

        // Part of #3910: When JIT is ready for all actions, skip streaming and
        // use the batch path below which has JIT dispatch wired. This ensures
        // JIT-compiled native code is used instead of the interpreter.
        //
        // Part of #4030: When JIT is ready and validation is complete, use the
        // fused path that does JIT eval + fingerprint + dedup inline — zero
        // intermediate Vec allocations for duplicate states.
        // Monolithic JIT does not emit per-action attribution. Explicit
        // coverage (including strict track-only mode) stays on the split-action
        // dispatcher even after every action has promoted.
        let jit_ready =
            monolithic_jit_route_ready(has_coverage, self.jit_monolithic_ready());

        if jit_ready {
            // Part of #4030: Fused JIT dispatch path. Runs JIT actions inline
            // with fingerprint + dedup, eliminating per-action Vec clones.
            //
            // During the validation period (first N states after JIT activation),
            // we use the old two-phase path which collects successors into a Vec
            // for cross-checking against the interpreter. After validation
            // completes, the fused path is used.
            {
                if self.jit_validation_remaining == 0 {
                    // Post-validation: fused zero-allocation path.
                    return self.process_jit_fused_successors(
                        iter_state,
                        storage,
                        queue,
                        params,
                        prof,
                        has_eval_implied_actions,
                        has_constraints,
                        cache_for_liveness,
                    );
                }
                // Validation still active: use old two-phase path for cross-checking.
                let prof_t0 = prof.now();
                let jit_flat_result = self.try_jit_monolithic_successors(iter_state.array());
                if let Some(jit_flat_succs) = jit_flat_result {
                    prof.accum_succ_gen(prof_t0);
                    return self.process_jit_flat_successors(
                        iter_state,
                        storage,
                        queue,
                        params,
                        prof,
                        jit_flat_succs,
                        has_eval_implied_actions,
                        has_constraints,
                        cache_for_liveness,
                    );
                }
            }
        }

        // Part of #3968: Also check hybrid JIT readiness. When some actions
        // are JIT-compiled, skip the streaming path and fall through to the
        // batch path which routes through per-action dispatch.
        let jit_hybrid = self.jit_hybrid_ready();

        // Item 4 M0-G2: hybrid flat-view NATIVE dispatch also lives in the
        // batch per-action path (the differential + candidate matching need
        // action boundaries), so yield streaming exactly like hybrid JIT
        // does. Only ever true under TY_HYBRID_FLAT_VIEW=1 + TY_HYBRID_NATIVE=1
        // with >=1 hybrid-compiled action — the default streaming selection is
        // unchanged.
        let trust_cg_hybrid = self.trust_cg_hybrid_action_dispatch_ready();

        if !jit_ready
            && !jit_hybrid
            && !trust_cg_hybrid
            && !has_eval_implied_actions
            && !has_constraints
            && !has_por
            && !use_tagged
            && !has_coverage
            && !has_symmetry
            && !has_view
            // Step B: when the native slide kernel is armed (recognizer-proven
            // default or `TY_NESTED_SET_SLIDE=1` force-arm), route through the
            // batch path so `generate_successors_array_raw` can intercept with
            // the word-op kernel. `None` (every unrecognized spec) leaves
            // streaming untouched.
            && self.nested_set_slide_arm.is_none()
        // Nested-set A6: the streaming path is NO LONGER excluded when a
        // monitor is installed. The streaming ClosureSink now calls the
        // monitor's escape-only hook (`observe_diff_monitors_escape`) on every
        // successor's board, so the monitor stays unbypassable on the fast path
        // and fails closed on escape. The streaming fingerprint
        // (`compute_diff_fingerprint_with_xor`) already produces
        // `value_fingerprint(board)`, byte-matching the monitored `dedup_fp` —
        // verdict identical.
        {
            return self
                .process_full_state_successors_streaming(iter_state, storage, queue, params, prof);
        }

        // Batch path: specs with implied actions, constraints, action tagging,
        // coverage collection, VIEW, or symmetry, or JIT not available.

        // Interpreter path: JIT not available or returned None.
        let prof_t0 = prof.now();
        let (succ_result, succ_action_tags) =
        // Step 1: Generate all successors (batch).
        // Part of #3100: Use tagged generation when liveness provenance map is active,
        // so we can skip re-evaluating ActionPred leaves whose action tag is known.
        if use_tagged {
            match self.generate_successors_filtered_array_tagged(iter_state.array()) {
                Ok((result, tags)) => (result, tags),
                Err(e) => {
                    return BfsIterOutcome::Terminate(
                        self.bfs_error_return(iter_state, storage, e),
                    );
                }
            }
        } else {
            // TRUE-only ENABLED provenance (#3208 redo of #3100): arm the
            // per-state witness scratch STRICTLY around this state's successor
            // generation. Frames/emissions recorded inside witness ENABLED for
            // exactly this parent state; the guard disarms on scope exit
            // (including the error path), so later evaluations — invariants,
            // implied actions, the inline liveness recorder itself — can never
            // attribute their own enumerations to this state. The recorded
            // witnesses survive disarm and are consumed (fingerprint-checked)
            // by the ENABLED leaf evaluator via `witnessed_true`. No-op unless
            // the inline fairness plan registered provenance-eligible leaves.
            let _prov_arm = cache_for_liveness
                .then(|| crate::liveness::enabled_provenance::arm_state_guard(fp));
            // WP-26: grand total of the per-action batch engine, so the finer
            // buckets (`interp_enum`, `interp_succ_build`, `constraints`,
            // `fp_to_state`, `parent_setup`, plus the native buckets) can be
            // subtracted to expose whatever per-action-loop overhead is left.
            let t_batch = self.hybrid_dispatch.perf.start();
            let batch = self.generate_successors_array_raw(iter_state.array());
            super::super::hybrid_dispatch::perf_acc(
                &mut self.hybrid_dispatch.perf.batch_gen_ns,
                t_batch,
            );
            match batch {
                Ok(result) => (result, Vec::new()),
                Err(e) => {
                    return BfsIterOutcome::Terminate(
                        self.bfs_error_return(iter_state, storage, e),
                    );
                }
            }
        };
        let valid_successors = succ_result.successors;
        prof.accum_succ_gen(prof_t0);

        // Part of #4013: Check if any successor has an action tag (from JIT or
        // tagged generation). When true, collect tags for liveness provenance.
        let has_action_tags = use_tagged;

        #[cfg(debug_assertions)]
        let (state_tlc_fp, need_detail_log, debug_actions_this_state) = self
            .debug_bfs_state_header(
                fp,
                iter_state.array(),
                _current_depth,
                valid_successors.len(),
                "",
            );

        self.ctx.set_tlc_level(succ_level);

        let succ_count = valid_successors.len();
        let mut candidate_observer = CompositeObserver::candidate_successors(has_constraints);
        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);
        let mut admission_batch = BfsAdmissionBatch::new();
        let mut observable_successor_count = 0usize;

        // When liveness caching is active, collect successor data for the
        // storage-mode-specific `cache_full_liveness` method. When inactive
        // (the common performance-critical case), no allocation or cloning.
        let mut liveness_data: Vec<(ArrayState, Fingerprint)> = if cache_for_liveness {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };
        // Part of #3100, #4013: Parallel vector of action tags for liveness provenance.
        // Enabled when tagged generation is used OR JIT produced action tags.
        let mut liveness_action_tags: Vec<Option<usize>> = if has_action_tags {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };

        // Debug logging data (only in debug builds)
        #[cfg(debug_assertions)]
        let mut debug_succ_data: Vec<(Fingerprint, ArrayState, Option<usize>)> = if need_detail_log
        {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };

        // ================================================================
        // Fused pass: materialize → fingerprint → constraint observers →
        // implied actions → dedup → invariant observer → enqueue.
        //
        // Part of #2881 Step 1: replaces the former 3-pass approach
        // (fingerprint-all → implied-actions-all → core-step-batch) with a
        // single pass that enables early termination on invariant violations.
        // ================================================================
        // WP-26: the batch engine's per-successor consumer, paid over EVERY
        // successor including the ~87% that dedup discards. The streaming
        // engine's equivalent work is split — fingerprint + dedup inline in
        // Phase A (never materializing a duplicate), full processing in Phase B
        // for post-dedup survivors only.
        let t_consume = self.hybrid_dispatch.perf.start();
        // WP-34 lever 2: walk the parent's variables ONCE, so each successor
        // only has to walk the ones it actually rewrote. Skipped entirely when
        // the spec's AST cannot produce lazy values (`materialize_array_state`
        // early-returns in that case anyway).
        let lazy_profile = (self.compiled.spec_may_produce_lazy
            && super::super::hybrid_dispatch::consume_delta_materialize_enabled())
        .then(|| ParentLazyProfile::build(iter_state.array()));
        for (succ_idx, mut arr) in valid_successors.into_iter().enumerate() {
            let action_tag = if has_action_tags {
                succ_action_tags.get(succ_idx).copied().unwrap_or(None)
            } else {
                None
            };

            // --- Materialize lazy values ---
            let t_mat = self.hybrid_dispatch.perf.start_consume();
            let mat_result = match lazy_profile.as_ref() {
                Some(profile) => {
                    materialize_delta_from_parent(&self.ctx, profile, iter_state.array(), &mut arr)
                }
                // Either the spec cannot produce lazy values (the call
                // early-returns), or `TY_HYBRID_CONSUME_DELTA=0` restored the
                // unconditional whole-state normalization.
                None => crate::materialize::materialize_array_state(
                    &self.ctx,
                    &mut arr,
                    self.compiled.spec_may_produce_lazy,
                )
                .map(|_| (0u32, false)),
            };
            let mat_err = match mat_result {
                Ok((scanned, skipped_all)) => {
                    let perf = &mut self.hybrid_dispatch.perf;
                    perf.consume_succ += 1;
                    if lazy_profile.is_some() {
                        perf.consume_lazy_vars_scanned += u64::from(scanned);
                        perf.consume_lazy_vars_total += arr.len() as u64;
                        if skipped_all {
                            perf.consume_lazy_scan_skipped += 1;
                        }
                    }
                    None
                }
                Err(e) => Some(e),
            };
            super::super::hybrid_dispatch::perf_acc(
                &mut self.hybrid_dispatch.perf.consume_materialize_ns,
                t_mat,
            );
            if let Some(e) = mat_err {
                if let Err(outcome) =
                    self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
                {
                    return outcome;
                }
                return BfsIterOutcome::Terminate(self.bfs_error_return(
                    iter_state,
                    storage,
                    EvalCheckError::Eval(e).into(),
                ));
            }

            // --- Fingerprint ---
            let t_fp_bucket = self.hybrid_dispatch.perf.start_consume();
            let prof_t_fp = prof.now();
            let succ_fp = match self.array_state_fingerprint(&mut arr) {
                Ok(fp_val) => fp_val,
                Err(e) => {
                    if let Err(outcome) =
                        self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
                    {
                        return outcome;
                    }
                    return BfsIterOutcome::Terminate(
                        self.bfs_error_return(iter_state, storage, e),
                    );
                }
            };
            prof.accum_fingerprint(prof_t_fp);
            super::super::hybrid_dispatch::perf_acc(
                &mut self.hybrid_dispatch.perf.consume_fp_ns,
                t_fp_bucket,
            );

            // Part of #1281: ACTION_CONSTRAINT expressions reference TLCGet("level")
            // which should see the *current* (parent) state's level, not the
            // successor's level. The eval context was set to succ_level above
            // (for VIEW fingerprinting), so toggle to current_level for constraint
            // evaluation, then restore succ_level afterwards.
            let t_observe = self.hybrid_dispatch.perf.start_consume();
            if has_constraints {
                self.ctx.set_tlc_level(current_level);
            }
            let observation = candidate_observer.observe_successor(
                self,
                &SuccessorObservationCtx {
                    current: iter_state.array(),
                    parent_fp: fp,
                    succ: &arr,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            );
            super::super::hybrid_dispatch::perf_acc(
                &mut self.hybrid_dispatch.perf.consume_observe_ns,
                t_observe,
            );
            match observation {
                Ok(ExplorationSignal::Continue) => {
                    if has_constraints {
                        self.ctx.set_tlc_level(succ_level);
                    }
                }
                Ok(ExplorationSignal::Skip) => {
                    if has_constraints {
                        self.ctx.set_tlc_level(succ_level);
                    }
                    continue;
                }
                Ok(ExplorationSignal::Stop(result)) => {
                    if let Err(outcome) =
                        self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
                    {
                        return outcome;
                    }
                    iter_state.return_to(storage, self);
                    return BfsIterOutcome::Terminate(result);
                }
                Err(error) => {
                    if let Err(outcome) =
                        self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
                    {
                        return outcome;
                    }
                    return BfsIterOutcome::Terminate(
                        self.bfs_error_return(iter_state, storage, error),
                    );
                }
            }

            observable_successor_count += 1;
            prof.count_successors(1);
            self.record_transitions(1);

            // --- Collect for liveness caching ---
            if cache_for_liveness {
                liveness_data.push((arr.clone(), succ_fp));
                if has_action_tags {
                    liveness_action_tags.push(action_tag);
                }
            }

            // --- Debug data ---
            #[cfg(debug_assertions)]
            if need_detail_log {
                debug_succ_data.push((succ_fp, arr.clone(), None));
            }

            // --- Eval-based implied actions (#2983, #3354 Slice 4) ---
            // Part of #3140: Skip stuttering transitions.
            if has_eval_implied_actions && succ_fp != fp {
                let outcome = crate::checker_ops::check_eval_implied_actions_for_transition(
                    &mut self.ctx,
                    &self.compiled.eval_implied_actions,
                    iter_state.array(),
                    fp,
                    &arr,
                    succ_fp,
                );
                // A terminal implied-action outcome (a violation that finalizes,
                // or an error) makes `handle_implied_action_outcome` consume
                // `iter_state` via `return_to`. `flush_admission_batch` needs
                // `iter_state.array()`, so it MUST run BEFORE that hand-off —
                // mirroring the `ExplorationSignal::Stop` path above, which
                // flushes and only then returns the current state. The former
                // order (handle → flush) accessed `iter_state.array()` after the
                // array had already been returned, panicking with "array already
                // returned" on the very first refinement/implied-action violation
                // (e.g. an abstract action removed so a concrete step has no
                // matching image). Flushing early is harmless when the outcome is
                // non-terminal (continue-on-error): the batch simply drains to
                // storage and the loop continues with a fresh batch.
                if !matches!(
                    outcome,
                    crate::checker_ops::InvariantOutcome::Ok
                        | crate::checker_ops::InvariantOutcome::ViolationContinued
                ) {
                    if let Err(outcome) =
                        self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
                    {
                        return outcome;
                    }
                }
                if let Some(result) = self.handle_implied_action_outcome(
                    iter_state, storage, outcome, fp, &arr, succ_fp, succ_depth,
                ) {
                    return BfsIterOutcome::Terminate(result);
                }
            }

            // --- Inline core step: dedup → invariant → admit → enqueue ---
            // Two-phase dedup: the read-only prefilter avoids cloning most
            // duplicates, while `admit_successor` remains the authority for the
            // TOCTOU window where another worker wins the insert race.
            // Fatal invariant traces are staged on the terminal path only,
            // keeping the common case clone-free.
            let t_dedup_bucket = self.hybrid_dispatch.perf.start_consume();
            let prof_t_dedup = prof.now();
            let is_seen = match self.is_state_seen_checked(succ_fp) {
                Ok(seen) => seen,
                Err(result) => {
                    if let Err(outcome) =
                        self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
                    {
                        return outcome;
                    }
                    iter_state.return_to(storage, self);
                    return BfsIterOutcome::Terminate(result);
                }
            };
            if is_seen {
                prof.accum_dedup(prof_t_dedup);
                super::super::hybrid_dispatch::perf_acc(
                    &mut self.hybrid_dispatch.perf.consume_dedup_ns,
                    t_dedup_bucket,
                );
                continue;
            }
            prof.accum_dedup(prof_t_dedup);
            super::super::hybrid_dispatch::perf_acc(
                &mut self.hybrid_dispatch.perf.consume_dedup_ns,
                t_dedup_bucket,
            );
            self.hybrid_dispatch.perf.consume_survivors += 1;

            let t_finish = self.hybrid_dispatch.perf.start_consume();
            let finish_outcome = self.finish_prefiltered_successor_batched(
                iter_state,
                storage,
                queue,
                prof,
                &mut admitted_observer,
                &mut admission_batch,
                arr,
                PendingSuccessor {
                    parent_fp: fp,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            );
            super::super::hybrid_dispatch::perf_acc(
                &mut self.hybrid_dispatch.perf.consume_finish_ns,
                t_finish,
            );
            if let Err(outcome) = finish_outcome {
                return outcome;
            }
        }

        if let Err(outcome) =
            self.flush_admission_batch(iter_state, storage, queue, &mut admission_batch)
        {
            return outcome;
        }
        super::super::hybrid_dispatch::perf_acc(
            &mut self.hybrid_dispatch.perf.batch_consume_ns,
            t_consume,
        );

        // --- Post-loop: debug logging (before return_to, needs iter_state.array()) ---
        #[cfg(debug_assertions)]
        if need_detail_log {
            self.debug_log_bfs_successors(
                fp,
                state_tlc_fp,
                _current_depth,
                iter_state.array(),
                _registry,
                succ_result.had_raw_successors,
                debug_actions_this_state,
                &debug_succ_data,
            );
        }

        let liveness_tags = if has_action_tags {
            &liveness_action_tags[..]
        } else {
            &[]
        };
        let mut state_observer = CompositeObserver::state_completion(
            self.exploration.check_deadlock,
            self.inline_liveness_active(),
        );
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            observable_successor_count == 0,
            succ_result.had_raw_successors,
            cache_for_liveness.then_some(liveness_data.as_slice()),
            liveness_tags,
        ) {
            return outcome;
        }
        if let Err(outcome) =
            self.cache_full_state_batch_liveness(iter_state, storage, fp, &liveness_data)
        {
            return outcome;
        }

        // Part of #3784: record monolithic successor count to cooperative metrics.
        self.record_cooperative_monolithic_successors(observable_successor_count);

        // Part of #3850: record monolithic action eval for tiered JIT.
        // This is the interpreter path (JIT exits via fused/two-phase above).
        self.record_action_eval_for_tier(0, observable_successor_count as u64);
        // Part of #3910: record JIT next-state dispatch for `--show-tiers` report.
        self.record_monolithic_next_state_dispatch();

        // Return parent state to storage.
        iter_state.return_to(storage, self);

        BfsIterOutcome::Continue
    }

    /// Fused JIT action dispatch + BFS dedup pipeline.
    ///
    /// Runs JIT-compiled actions inline with fingerprint and dedup checks,
    /// eliminating intermediate Vec allocations. Each action's output is
    /// fingerprinted directly from the reusable scratch buffer. Only new
    /// (non-duplicate) states are unflattened to ArrayState.
    ///
    /// This replaces the two-phase approach (try_jit_monolithic_successors
    /// collecting JitFlatSuccessors + process_jit_flat_successors iterating
    /// them), which cloned the output and input buffers for every enabled
    /// action. Since ~80-95% of successors are duplicates, the two-phase
    /// approach wasted those clones.
    ///
    /// Part of #4030: Eliminate per-action Vec clone overhead in JIT dispatch.
    #[allow(clippy::too_many_arguments)]
    fn process_jit_fused_successors<S: BfsStorage, Q: BfsFrontier<Entry = S::QueueEntry>>(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
        has_eval_implied_actions: bool,
        has_constraints: bool,
        cache_for_liveness: bool,
    ) -> BfsIterOutcome {
        use super::super::invariants::{
            fingerprint_jit_flat_successor, fingerprint_jit_flat_successor_incremental,
            unflatten_i64_to_array_state_with_input,
        };
        use super::super::run_helpers::JIT_WARMUP_THRESHOLD;

        let &BfsStepParams {
            registry: _registry,
            current_depth: _current_depth,
            succ_depth,
            current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();
        let num_actions = self.jit_action_lookup_keys.len();

        // Extract state_var_count from the cache.
        let state_var_count = match self.jit_next_state_cache.as_ref() {
            Some(c) => c.state_var_count(),
            None => {
                // Should not happen — caller checked jit_monolithic_ready.
                return self.process_full_state_successors_streaming(
                    iter_state, storage, queue, params, prof,
                );
            }
        };

        // Flatten parent state for JIT evaluation.
        if !self.prepare_jit_next_state(iter_state.array()) {
            return self
                .process_full_state_successors_streaming(iter_state, storage, queue, params, prof);
        }

        // Part of #4030: Warmup timing tracks ONLY JIT eval time (not fingerprint/dedup)
        // for fair comparison with the interpreter's successor-generation-only timing.
        let warmup_sampling = self.jit_perf_monitor.2 < JIT_WARMUP_THRESHOLD;

        // Part of #4030: Defer jit_state_scratch clone to the cold path.
        // Only 5-20% of successors are new (non-duplicate). The remaining 80-95%
        // are duplicates where the clone would be wasted. We snapshot lazily
        // only when a new state needs unflatten with compound variable support.
        // For all-scalar states, the input snapshot is never needed.
        let state_all_scalar = self.jit_state_all_scalar;
        let mut jit_input_snapshot: Option<Vec<i64>> = None;

        // Get the registry for flat fingerprinting (O(1) Arc clone).
        let registry = self.ctx.var_registry().clone();
        let use_compiled_xxh3 = self.jit_compiled_fp_active && state_all_scalar;
        let parent_incremental_base_xor =
            (!use_compiled_xxh3).then(|| iter_state.array().incremental_fp_base(&registry).0);

        #[cfg(debug_assertions)]
        let (_state_tlc_fp, need_detail_log, debug_actions_this_state) = self
            .debug_bfs_state_header(
                fp,
                iter_state.array(),
                _current_depth,
                0, // count not yet known
                "[jit-fused]",
            );

        self.ctx.set_tlc_level(succ_level);

        let mut candidate_observer = CompositeObserver::candidate_successors(has_constraints);
        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);
        let mut observable_successor_count = 0usize;
        let mut had_raw = false;
        let mut enabled_action_count = 0usize;

        let can_dedup_before_materialize =
            !has_constraints && !has_eval_implied_actions && !cache_for_liveness;
        #[cfg(debug_assertions)]
        let can_dedup_before_materialize = can_dedup_before_materialize && !need_detail_log;

        let mut liveness_data: Vec<(ArrayState, Fingerprint)> = if cache_for_liveness {
            Vec::with_capacity(num_actions)
        } else {
            Vec::new()
        };
        // Pre-size to mirror its 1:1 sibling `liveness_data` above: both are pushed in
        // lockstep under `cache_for_liveness` in the per-action loop, so this Vec grows to
        // exactly `num_actions`. Capacity-only (no element/order/count change) — removes
        // the 0→…→num_actions realloc cascade per explored state on liveness runs.
        let mut liveness_action_tags: Vec<Option<usize>> = if cache_for_liveness {
            Vec::with_capacity(num_actions)
        } else {
            Vec::new()
        };

        #[cfg(debug_assertions)]
        let mut debug_succ_data: Vec<(Fingerprint, ArrayState, Option<usize>)> = if need_detail_log
        {
            Vec::with_capacity(num_actions)
        } else {
            Vec::new()
        };

        // Ensure scratch buffer is sized correctly.
        if self.jit_action_out_scratch.len() < state_var_count {
            self.jit_action_out_scratch.resize(state_var_count, 0);
        }

        // Part of #4030: Track cumulative JIT eval time separately from
        // fingerprint/dedup for fair warmup gate comparison.
        let mut jit_eval_ns: u64 = 0;

        // === Fused JIT action loop + BFS dedup pipeline ===
        // For each action: eval via JIT, fingerprint from scratch, dedup, unflatten only if new.
        for action_idx in 0..num_actions {
            // Check key validity (empty = can't be JIT-compiled).
            if self.jit_action_lookup_keys[action_idx].is_empty() {
                // Action can't be JIT-compiled — fall back to interpreter for entire state.
                // This shouldn't happen since jit_all_next_state_compiled is checked by caller,
                // but handle gracefully.
                return self.fallback_to_interpreter_path(iter_state, storage, queue, params, prof);
            }

            // Part of #4030: Skip compound scratch clearing for all-scalar states.
            // clear_compound_scratch() touches a thread-local (TLS access) per action.
            // For specs where all state variables are ints/bools, the compound scratch
            // is never used — skipping it saves one TLS access per action per state.
            if !state_all_scalar {
                tla_jit_abi::clear_compound_scratch();
            }

            self.next_state_dispatch.total += 1;

            // Part of #4030: Time only the JIT eval, not fingerprint/dedup.
            let eval_t0 = if warmup_sampling {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Evaluate the action via JIT into the reusable scratch buffer.
            let eval_result = {
                let cache = match self.jit_next_state_cache.as_ref() {
                    Some(c) => c,
                    None => {
                        return self.fallback_to_interpreter_path(
                            iter_state, storage, queue, params, prof,
                        );
                    }
                };
                let key = &self.jit_action_lookup_keys[action_idx];
                cache.eval_action_into(
                    key,
                    &self.jit_state_scratch,
                    &mut self.jit_action_out_scratch,
                )
            };

            // Accumulate JIT eval time (excludes fingerprint/dedup).
            if let Some(t0) = eval_t0 {
                jit_eval_ns += t0.elapsed().as_nanos() as u64;
            }

            match eval_result {
                Some(Ok(true)) => {
                    // Action enabled — successor is in jit_action_out_scratch.
                    self.next_state_dispatch.jit_hit += 1;
                    had_raw = true;
                    enabled_action_count += 1;

                    // --- Fingerprint directly from scratch buffer (no clone!) ---
                    // Part of #3987: When compiled xxh3 fingerprinting is active,
                    // hash the raw i64 buffer directly with xxh3 SIMD — single call,
                    // no per-variable type dispatch, no combined_xor tracking needed.
                    // This replaces the per-variable FP64 computation below.
                    //
                    // Part of #4030: Otherwise try incremental fingerprint first
                    // (O(changed_vars)), fall back to full scan (O(total_vars)) if
                    // parent lacks fp_cache or compound variables changed.
                    let prof_t_fp = prof.now();
                    let (succ_fp, succ_combined_xor, mut arr_opt) = if use_compiled_xxh3 {
                        // Part of #3987: Compiled xxh3 fast path. Single SIMD hash
                        // of the raw scratch buffer. No per-variable type dispatch.
                        let fp = super::super::invariants::fingerprint_flat_compiled(
                            &self.jit_action_out_scratch[..state_var_count],
                        );
                        (fp, None, None)
                    } else {
                        match fingerprint_jit_flat_successor_incremental(
                            iter_state.array(),
                            &self.jit_action_out_scratch,
                            &self.jit_state_scratch,
                            state_var_count,
                            parent_incremental_base_xor.expect(
                                "incremental base xor must be present when compiled xxh3 is disabled",
                            ),
                            &registry,
                        )
                        .or_else(|| {
                            // Incremental failed (no fp_cache, compound changed, or buffer mismatch).
                            // Fall back to full O(n) scan.
                            let input_ref = jit_input_snapshot.as_deref();
                            fingerprint_jit_flat_successor(
                                iter_state.array(),
                                &self.jit_action_out_scratch,
                                state_var_count,
                                input_ref,
                                &registry,
                            )
                        }) {
                            Some((flat_fp, xor)) => (flat_fp, Some(xor), None),
                            None => {
                                // Compound variable modified — need full unflatten for fingerprint.
                                // Part of #4030: Snapshot now (first time only).
                                if jit_input_snapshot.is_none() {
                                    jit_input_snapshot = Some(self.jit_state_scratch.clone());
                                }
                                let mut arr = unflatten_i64_to_array_state_with_input(
                                    iter_state.array(),
                                    &self.jit_action_out_scratch,
                                    state_var_count,
                                    jit_input_snapshot.as_deref(),
                                );
                                if let Err(e) = crate::materialize::materialize_array_state(
                                    &self.ctx,
                                    &mut arr,
                                    self.compiled.spec_may_produce_lazy,
                                ) {
                                    return BfsIterOutcome::Terminate(self.bfs_error_return(
                                        iter_state,
                                        storage,
                                        EvalCheckError::Eval(e).into(),
                                    ));
                                }
                                let fp_val = match self.array_state_fingerprint(&mut arr) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        return BfsIterOutcome::Terminate(
                                            self.bfs_error_return(iter_state, storage, e),
                                        );
                                    }
                                };
                                // combined_xor is available from the arr's fp_cache after
                                // array_state_fingerprint sets it.
                                let xor = arr.fp_cache.as_ref().map(|c| c.combined_xor);
                                (fp_val, xor, Some(arr))
                            }
                        }
                    };
                    prof.accum_fingerprint(prof_t_fp);

                    if can_dedup_before_materialize {
                        observable_successor_count += 1;
                        prof.count_successors(1);
                        self.record_transitions(1);

                        // Hot path: if the flat fingerprint is already present,
                        // skip ArrayState reconstruction and materialization.
                        let prof_t_dedup = prof.now();
                        let is_seen = match self.is_state_seen_checked(succ_fp) {
                            Ok(seen) => seen,
                            Err(result) => {
                                iter_state.return_to(storage, self);
                                return BfsIterOutcome::Terminate(result);
                            }
                        };
                        if is_seen {
                            prof.accum_dedup(prof_t_dedup);
                            continue;
                        }
                        prof.accum_dedup(prof_t_dedup);

                        let mut arr = match arr_opt.take() {
                            Some(a) => a,
                            None => {
                                if jit_input_snapshot.is_none() {
                                    jit_input_snapshot = Some(self.jit_state_scratch.clone());
                                }
                                let mut a = unflatten_i64_to_array_state_with_input(
                                    iter_state.array(),
                                    &self.jit_action_out_scratch,
                                    state_var_count,
                                    jit_input_snapshot.as_deref(),
                                );
                                if let Err(e) = crate::materialize::materialize_array_state(
                                    &self.ctx,
                                    &mut a,
                                    self.compiled.spec_may_produce_lazy,
                                ) {
                                    return BfsIterOutcome::Terminate(self.bfs_error_return(
                                        iter_state,
                                        storage,
                                        EvalCheckError::Eval(e).into(),
                                    ));
                                }
                                a
                            }
                        };

                        if let Some(xor) = succ_combined_xor {
                            arr.set_cached_fingerprint_with_xor(succ_fp, xor);
                        } else {
                            arr.set_cached_fingerprint(succ_fp);
                        }

                        if let Err(outcome) = self.finish_prefiltered_successor(
                            iter_state,
                            storage,
                            queue,
                            prof,
                            &mut admitted_observer,
                            arr,
                            PendingSuccessor {
                                parent_fp: fp,
                                succ_fp,
                                succ_depth,
                                succ_level,
                            },
                        ) {
                            return outcome;
                        }

                        continue;
                    }

                    // --- Unflatten after fingerprinting. When semantic consumers
                    // need full values before dedup, keep the conservative path.
                    let mut arr = match arr_opt.take() {
                        Some(a) => a,
                        None => {
                            // Scalar fast-path fingerprint succeeded but state is new.
                            // Part of #4030: Snapshot now if not yet done.
                            if jit_input_snapshot.is_none() {
                                jit_input_snapshot = Some(self.jit_state_scratch.clone());
                            }
                            let mut a = unflatten_i64_to_array_state_with_input(
                                iter_state.array(),
                                &self.jit_action_out_scratch,
                                state_var_count,
                                jit_input_snapshot.as_deref(),
                            );
                            if let Err(e) = crate::materialize::materialize_array_state(
                                &self.ctx,
                                &mut a,
                                self.compiled.spec_may_produce_lazy,
                            ) {
                                return BfsIterOutcome::Terminate(self.bfs_error_return(
                                    iter_state,
                                    storage,
                                    EvalCheckError::Eval(e).into(),
                                ));
                            }
                            a
                        }
                    };

                    // Part of #4030: Store combined_xor so this state can participate
                    // in incremental fingerprinting when it becomes a BFS parent.
                    if let Some(xor) = succ_combined_xor {
                        arr.set_cached_fingerprint_with_xor(succ_fp, xor);
                    } else {
                        arr.set_cached_fingerprint(succ_fp);
                    }

                    // --- Constraint observers ---
                    if has_constraints {
                        self.ctx.set_tlc_level(current_level);
                    }
                    match candidate_observer.observe_successor(
                        self,
                        &SuccessorObservationCtx {
                            current: iter_state.array(),
                            parent_fp: fp,
                            succ: &arr,
                            succ_fp,
                            succ_depth,
                            succ_level,
                        },
                    ) {
                        Ok(ExplorationSignal::Continue) => {
                            if has_constraints {
                                self.ctx.set_tlc_level(succ_level);
                            }
                        }
                        Ok(ExplorationSignal::Skip) => {
                            if has_constraints {
                                self.ctx.set_tlc_level(succ_level);
                            }
                            continue;
                        }
                        Ok(ExplorationSignal::Stop(result)) => {
                            iter_state.return_to(storage, self);
                            return BfsIterOutcome::Terminate(result);
                        }
                        Err(error) => {
                            return BfsIterOutcome::Terminate(
                                self.bfs_error_return(iter_state, storage, error),
                            );
                        }
                    }

                    observable_successor_count += 1;
                    prof.count_successors(1);
                    self.record_transitions(1);

                    // --- Collect for liveness caching ---
                    if cache_for_liveness {
                        liveness_data.push((arr.clone(), succ_fp));
                        liveness_action_tags.push(Some(action_idx));
                    }

                    // --- Debug data ---
                    #[cfg(debug_assertions)]
                    if need_detail_log {
                        debug_succ_data.push((succ_fp, arr.clone(), Some(action_idx)));
                    }

                    // --- Eval-based implied actions ---
                    if has_eval_implied_actions && succ_fp != fp {
                        let outcome = crate::checker_ops::check_eval_implied_actions_for_transition(
                            &mut self.ctx,
                            &self.compiled.eval_implied_actions,
                            iter_state.array(),
                            fp,
                            &arr,
                            succ_fp,
                        );
                        if let Some(result) = self.handle_implied_action_outcome(
                            iter_state, storage, outcome, fp, &arr, succ_fp, succ_depth,
                        ) {
                            return BfsIterOutcome::Terminate(result);
                        }
                    }

                    // --- Dedup check (after transition bookkeeping) ---
                    let prof_t_dedup = prof.now();
                    let is_seen = match self.is_state_seen_checked(succ_fp) {
                        Ok(seen) => seen,
                        Err(result) => {
                            iter_state.return_to(storage, self);
                            return BfsIterOutcome::Terminate(result);
                        }
                    };
                    if is_seen {
                        prof.accum_dedup(prof_t_dedup);
                        continue;
                    }
                    prof.accum_dedup(prof_t_dedup);

                    // --- Invariant check + admit + enqueue ---
                    if let Err(outcome) = self.finish_prefiltered_successor(
                        iter_state,
                        storage,
                        queue,
                        prof,
                        &mut admitted_observer,
                        arr,
                        PendingSuccessor {
                            parent_fp: fp,
                            succ_fp,
                            succ_depth,
                            succ_level,
                        },
                    ) {
                        return outcome;
                    }
                }
                Some(Ok(false)) => {
                    // Action disabled (guard=false) — no successor, no allocation.
                    self.next_state_dispatch.jit_hit += 1;
                }
                Some(Err(_)) => {
                    self.next_state_dispatch.jit_error += 1;
                    // Part of #4012: Disable only this action, not all JIT.
                    // The fused path falls back to interpreter for this state,
                    // but future states can still use JIT for other actions.
                    if action_idx < self.jit_disabled_actions.len() {
                        self.jit_disabled_actions[action_idx] = true;
                    }
                    return self
                        .fallback_to_interpreter_path(iter_state, storage, queue, params, prof);
                }
                None => {
                    // Not compiled or FallbackNeeded — abandon fused path.
                    let has_action = self.jit_next_state_cache.as_ref().map_or(false, |c| {
                        c.contains_action(&self.jit_action_lookup_keys[action_idx])
                    });
                    if has_action {
                        self.next_state_dispatch.jit_fallback += 1;
                    } else {
                        self.next_state_dispatch.jit_not_compiled += 1;
                    }
                    return self
                        .fallback_to_interpreter_path(iter_state, storage, queue, params, prof);
                }
            }
        }

        // Part of #4030: Record JIT diagnostic timing.
        if self.jit_diag_enabled {
            static DIAG_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let count = DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 10 || count % 100_000 == 0 {
                eprintln!(
                    "[jit-diag] state {}: fused dispatch, {} enabled actions, {} new states",
                    count, enabled_action_count, observable_successor_count,
                );
            }
        }

        // Part of #4030: Warmup gate timing — use only JIT eval time (not
        // fingerprint/dedup) for fair comparison with interpreter timing.
        // Previously, the entire fused-path elapsed time was attributed to JIT,
        // making JIT appear ~57% slower than it actually was.
        if warmup_sampling {
            self.jit_perf_monitor.0 += jit_eval_ns;
            self.jit_perf_monitor.2 += 1;
        }

        // Part of #4031: Warmup gate decision.
        if self.jit_perf_monitor.2 == JIT_WARMUP_THRESHOLD {
            self.evaluate_jit_warmup_gate();
            // If warmup gate disabled JIT, the next state will route to interpreter.
            // Current state's successors were already processed correctly above.
        }

        // --- Post-loop ---
        #[cfg(debug_assertions)]
        if need_detail_log {
            self.debug_log_bfs_successors(
                fp,
                _state_tlc_fp,
                _current_depth,
                iter_state.array(),
                _registry,
                had_raw,
                debug_actions_this_state,
                &debug_succ_data,
            );
        }

        let liveness_tags = if !liveness_action_tags.is_empty() {
            &liveness_action_tags[..]
        } else {
            &[]
        };
        let mut state_observer = CompositeObserver::state_completion(
            self.exploration.check_deadlock,
            self.inline_liveness_active(),
        );
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            observable_successor_count == 0,
            had_raw,
            cache_for_liveness.then_some(liveness_data.as_slice()),
            liveness_tags,
        ) {
            return outcome;
        }
        if let Err(outcome) =
            self.cache_full_state_batch_liveness(iter_state, storage, fp, &liveness_data)
        {
            return outcome;
        }

        self.record_cooperative_monolithic_successors(observable_successor_count);

        iter_state.return_to(storage, self);
        BfsIterOutcome::Continue
    }

    /// Flat state primary BFS path.
    ///
    /// When `flat_state_primary=true`, ALL state variables are scalar (Int/Bool)
    /// and the state layout has been verified via roundtrip. This path:
    ///
    /// 1. Converts the current ArrayState to FlatState (one-time per parent).
    /// 2. Calls `generate_successors_filtered_flat()` → Vec<FlatState>.
    /// 3. Fingerprints each successor via `FlatState::fingerprint_compiled()`
    ///    (xxh3 SIMD on raw i64 buffer — single call, no per-variable dispatch).
    /// 4. Dedup via `is_state_seen_checked()`.
    /// 5. For NEW states only (5-20% of successors): unflatten to ArrayState
    ///    for invariant checking and enqueue.
    ///
    /// This eliminates the interpreter sandwich (FlatState → ArrayState → eval →
    /// ArrayState → FlatState) that dominates the JIT hot path. The flat buffer
    /// IS the state — JIT actions read/write i64[] directly.
    ///
    /// Part of #3986: Flat i64 state as primary BFS representation.
    #[allow(clippy::too_many_arguments)]
    fn process_flat_state_primary_successors<
        S: BfsStorage,
        Q: BfsFrontier<Entry = S::QueueEntry>,
    >(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
        layout: std::sync::Arc<crate::state::StateLayout>,
        cache_for_liveness: bool,
    ) -> BfsIterOutcome {
        let &BfsStepParams {
            registry: _registry,
            current_depth: _current_depth,
            succ_depth,
            current_level: _current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();

        // Convert parent ArrayState to FlatState for flat successor generation.
        // Graceful flat-overflow handling: a parent that cannot be encoded in
        // the fixed flat layout (scalar > i64, sequence over capacity) is a
        // typed terminal error — never a panic — so the CLI can retry the
        // check with flat state storage disabled.
        let parent_flat = match crate::state::FlatState::try_from_array_state(
            iter_state.array(),
            std::sync::Arc::clone(&layout),
        ) {
            Ok(flat) => flat,
            Err(err) => {
                let error = crate::CheckError::flat_layout_unsupported_value(err.to_string());
                return BfsIterOutcome::Terminate(
                    self.bfs_error_return(iter_state, storage, error),
                );
            }
        };

        if self.flat_successor_prefilter_streaming_candidate(cache_for_liveness) {
            match self.generate_successors_filtered_flat_prefiltered(
                &parent_flat,
                prof,
                cache_for_liveness,
            ) {
                Ok(Some(prefiltered)) => {
                    return self.process_flat_state_primary_prefiltered_successors(
                        iter_state,
                        storage,
                        queue,
                        params,
                        prof,
                        prefiltered,
                    );
                }
                Ok(None) => {
                    // Preserve the existing interpreter/JIT fallback behavior
                    // whenever the narrow streaming gate cannot cover a state.
                }
                Err(result) => {
                    iter_state.return_to(storage, self);
                    return BfsIterOutcome::Terminate(result);
                }
            }
        }

        // Generate successors in flat domain. Falls back to interpreter sandwich
        // for actions not in the JIT cache.
        let prof_t0 = prof.now();
        let succ_result = match self.generate_successors_filtered_flat(&parent_flat) {
            Ok(result) => result,
            Err(e) => {
                return BfsIterOutcome::Terminate(self.bfs_error_return(iter_state, storage, e));
            }
        };
        let flat_successors = succ_result.successors;
        let had_raw = succ_result.had_raw_successors;
        prof.accum_succ_gen(prof_t0);

        let registry = self.ctx.var_registry().clone();

        #[cfg(debug_assertions)]
        let (_state_tlc_fp, need_detail_log, debug_actions_this_state) = self
            .debug_bfs_state_header(
                fp,
                iter_state.array(),
                _current_depth,
                flat_successors.len(),
                "[flat-primary]",
            );

        self.ctx.set_tlc_level(succ_level);

        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);
        let mut observable_successor_count = 0usize;

        let mut liveness_data: Vec<(ArrayState, Fingerprint)> = if cache_for_liveness {
            Vec::with_capacity(flat_successors.len())
        } else {
            Vec::new()
        };

        #[cfg(debug_assertions)]
        let mut debug_succ_data: Vec<(Fingerprint, ArrayState, Option<usize>)> = if need_detail_log
        {
            Vec::with_capacity(flat_successors.len())
        } else {
            Vec::new()
        };

        for flat_succ in flat_successors {
            // --- Fingerprint via compiled xxh3 (single SIMD call on raw i64 buffer) ---
            let prof_t_fp = prof.now();
            let succ_fp = flat_succ.fingerprint_compiled();
            prof.accum_fingerprint(prof_t_fp);

            let mut arr_opt = if cache_for_liveness {
                let mut arr = flat_succ.to_array_state(&registry);
                if let Err(e) = crate::materialize::materialize_array_state(
                    &self.ctx,
                    &mut arr,
                    self.compiled.spec_may_produce_lazy,
                ) {
                    return BfsIterOutcome::Terminate(self.bfs_error_return(
                        iter_state,
                        storage,
                        EvalCheckError::Eval(e).into(),
                    ));
                }
                arr.set_cached_fingerprint(succ_fp);
                Some(arr)
            } else {
                None
            };

            observable_successor_count += 1;
            prof.count_successors(1);
            self.record_transitions(1);

            if let Some(arr) = arr_opt.as_ref() {
                liveness_data.push((arr.clone(), succ_fp));
            }

            // --- Dedup check (zero allocation for duplicates!) ---
            let prof_t_dedup = prof.now();
            let is_seen = match self.is_state_seen_checked(succ_fp) {
                Ok(seen) => seen,
                Err(result) => {
                    iter_state.return_to(storage, self);
                    return BfsIterOutcome::Terminate(result);
                }
            };
            if is_seen {
                prof.accum_dedup(prof_t_dedup);
                // Hot path: 80-95% of successors are duplicates — ZERO allocation.
                continue;
            }
            prof.accum_dedup(prof_t_dedup);

            // --- New state: unflatten to ArrayState (cold path, ~5-20% of successors) ---
            let arr = match arr_opt.take() {
                Some(arr) => arr,
                None => {
                    let mut arr = flat_succ.to_array_state(&registry);
                    if let Err(e) = crate::materialize::materialize_array_state(
                        &self.ctx,
                        &mut arr,
                        self.compiled.spec_may_produce_lazy,
                    ) {
                        return BfsIterOutcome::Terminate(self.bfs_error_return(
                            iter_state,
                            storage,
                            EvalCheckError::Eval(e).into(),
                        ));
                    }
                    arr.set_cached_fingerprint(succ_fp);
                    arr
                }
            };

            // --- Debug data ---
            #[cfg(debug_assertions)]
            if need_detail_log {
                debug_succ_data.push((succ_fp, arr.clone(), None));
            }

            // --- Invariant check + admit + enqueue ---
            if let Err(outcome) = self.finish_prefiltered_successor(
                iter_state,
                storage,
                queue,
                prof,
                &mut admitted_observer,
                arr,
                PendingSuccessor {
                    parent_fp: fp,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            ) {
                return outcome;
            }
        }

        // --- Post-loop ---
        #[cfg(debug_assertions)]
        if need_detail_log {
            self.debug_log_bfs_successors(
                fp,
                _state_tlc_fp,
                _current_depth,
                iter_state.array(),
                _registry,
                had_raw,
                debug_actions_this_state,
                &debug_succ_data,
            );
        }

        let mut state_observer = CompositeObserver::state_completion(
            self.exploration.check_deadlock,
            self.inline_liveness_active(),
        );
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            observable_successor_count == 0,
            had_raw,
            cache_for_liveness.then_some(liveness_data.as_slice()),
            &[], // No action tags in flat-primary path
        ) {
            return outcome;
        }
        if let Err(outcome) =
            self.cache_full_state_batch_liveness(iter_state, storage, fp, &liveness_data)
        {
            return outcome;
        }

        // Record cooperative metrics.
        self.record_cooperative_monolithic_successors(observable_successor_count);
        self.record_action_eval_for_tier(0, observable_successor_count as u64);
        self.record_monolithic_next_state_dispatch();

        // Return parent state to storage.
        iter_state.return_to(storage, self);

        BfsIterOutcome::Continue
    }

    /// Finish flat-primary successors that already passed the read-only
    /// fingerprint prefilter during generation.
    ///
    /// Only dedup-absent candidates are materialized here. The storage backend's
    /// `admit_successor` remains authoritative and may still reject races.
    #[allow(clippy::too_many_arguments)]
    fn process_flat_state_primary_prefiltered_successors<
        S: BfsStorage,
        Q: BfsFrontier<Entry = S::QueueEntry>,
    >(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
        prefiltered: FlatPrefilteredSuccessorResult,
    ) -> BfsIterOutcome {
        let &BfsStepParams {
            registry: _registry,
            current_depth: _current_depth,
            succ_depth,
            current_level: _current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();
        let registry = self.ctx.var_registry().clone();
        let FlatPrefilteredSuccessorResult {
            successors,
            raw_successor_count,
            had_raw_successors,
        } = prefiltered;

        #[cfg(debug_assertions)]
        let (_state_tlc_fp, need_detail_log, debug_actions_this_state) = self
            .debug_bfs_state_header(
                fp,
                iter_state.array(),
                _current_depth,
                raw_successor_count,
                "[flat-primary-prefilter]",
            );

        self.ctx.set_tlc_level(succ_level);

        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);

        prof.count_successors(raw_successor_count);
        self.record_transitions(raw_successor_count);

        #[cfg(debug_assertions)]
        let mut debug_succ_data: Vec<(Fingerprint, ArrayState, Option<usize>)> = if need_detail_log
        {
            Vec::with_capacity(successors.len())
        } else {
            Vec::new()
        };

        for flat_succ in successors {
            let succ_fp = flat_succ.fingerprint;
            let mut arr = flat_succ.flat.to_array_state(&registry);
            if let Err(e) = crate::materialize::materialize_array_state(
                &self.ctx,
                &mut arr,
                self.compiled.spec_may_produce_lazy,
            ) {
                return BfsIterOutcome::Terminate(self.bfs_error_return(
                    iter_state,
                    storage,
                    EvalCheckError::Eval(e).into(),
                ));
            }
            arr.set_cached_fingerprint(succ_fp);

            #[cfg(debug_assertions)]
            if need_detail_log {
                debug_succ_data.push((succ_fp, arr.clone(), None));
            }

            if let Err(outcome) = self.finish_prefiltered_successor(
                iter_state,
                storage,
                queue,
                prof,
                &mut admitted_observer,
                arr,
                PendingSuccessor {
                    parent_fp: fp,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            ) {
                return outcome;
            }
        }

        #[cfg(debug_assertions)]
        if need_detail_log {
            self.debug_log_bfs_successors(
                fp,
                _state_tlc_fp,
                _current_depth,
                iter_state.array(),
                _registry,
                had_raw_successors,
                debug_actions_this_state,
                &debug_succ_data,
            );
        }

        let mut state_observer = CompositeObserver::state_completion(
            self.exploration.check_deadlock,
            self.inline_liveness_active(),
        );
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            raw_successor_count == 0,
            had_raw_successors,
            None,
            &[],
        ) {
            return outcome;
        }
        if let Err(outcome) = self.cache_full_state_batch_liveness(iter_state, storage, fp, &[]) {
            return outcome;
        }

        self.record_cooperative_monolithic_successors(raw_successor_count);
        self.record_action_eval_for_tier(0, raw_successor_count as u64);
        self.record_monolithic_next_state_dispatch();

        iter_state.return_to(storage, self);

        BfsIterOutcome::Continue
    }

    /// Fall back to the interpreter path when JIT fused dispatch encounters an error.
    ///
    /// Part of #4030: Clean fallback from fused JIT to interpreter.
    fn fallback_to_interpreter_path<S: BfsStorage, Q: BfsFrontier<Entry = S::QueueEntry>>(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
    ) -> BfsIterOutcome {
        // Use the streaming interpreter path as fallback.
        self.process_full_state_successors_streaming(iter_state, storage, queue, params, prof)
    }

    /// Process JIT flat successors with deferred unflatten (legacy two-phase path).
    ///
    /// Used during the JIT validation period (first N states after JIT activation)
    /// where successor counts are cross-checked against the interpreter. After
    /// validation completes, the fused path (process_jit_fused_successors) is used
    /// instead, eliminating per-action Vec clones.
    ///
    /// Part of #4032: Eliminate per-action unflatten.
    #[allow(clippy::too_many_arguments)]
    fn process_jit_flat_successors<S: BfsStorage, Q: BfsFrontier<Entry = S::QueueEntry>>(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
        flat_succs: Vec<(super::super::run_helpers::JitFlatSuccessor, Option<usize>)>,
        has_eval_implied_actions: bool,
        has_constraints: bool,
        cache_for_liveness: bool,
    ) -> BfsIterOutcome {
        let &BfsStepParams {
            registry: _registry,
            current_depth: _current_depth,
            succ_depth,
            current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();

        let has_action_tags = flat_succs.iter().any(|(_, tag)| tag.is_some());

        #[cfg(debug_assertions)]
        let (state_tlc_fp, need_detail_log, debug_actions_this_state) = self
            .debug_bfs_state_header(
                fp,
                iter_state.array(),
                _current_depth,
                flat_succs.len(),
                "[jit-flat]",
            );

        self.ctx.set_tlc_level(succ_level);

        let succ_count = flat_succs.len();
        let mut candidate_observer = CompositeObserver::candidate_successors(has_constraints);
        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);
        let mut observable_successor_count = 0usize;

        let mut liveness_data: Vec<(ArrayState, Fingerprint)> = if cache_for_liveness {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };
        let mut liveness_action_tags: Vec<Option<usize>> = if has_action_tags {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };

        #[cfg(debug_assertions)]
        let mut debug_succ_data: Vec<(Fingerprint, ArrayState, Option<usize>)> = if need_detail_log
        {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };

        let had_raw = !flat_succs.is_empty();

        // Get the registry for flat fingerprinting.
        let registry = self.ctx.var_registry().clone();

        for (flat_succ, action_tag) in flat_succs {
            // --- Step 1: Try flat fingerprint (no Value allocation) ---
            let prof_t_fp = prof.now();
            let (succ_fp, mut arr_opt) = if self.jit_compiled_fp_active {
                // Part of #3987: Compiled xxh3 fast path — single SIMD hash.
                (flat_succ.compiled_xxh3_fingerprint(), None)
            } else if let Some(flat_fp) =
                flat_succ.try_flat_fingerprint(iter_state.array(), &registry)
            {
                // Fast path: fingerprint computed directly from flat buffer.
                (flat_fp, None)
            } else {
                // Fallback: compound variable was modified, need full unflatten.
                let mut arr = flat_succ.to_array_state(iter_state.array());
                if let Err(e) = crate::materialize::materialize_array_state(
                    &self.ctx,
                    &mut arr,
                    self.compiled.spec_may_produce_lazy,
                ) {
                    return BfsIterOutcome::Terminate(self.bfs_error_return(
                        iter_state,
                        storage,
                        EvalCheckError::Eval(e).into(),
                    ));
                }
                let fp_val = match self.array_state_fingerprint(&mut arr) {
                    Ok(f) => f,
                    Err(e) => {
                        return BfsIterOutcome::Terminate(
                            self.bfs_error_return(iter_state, storage, e),
                        );
                    }
                };
                (fp_val, Some(arr))
            };
            prof.accum_fingerprint(prof_t_fp);

            // --- Step 2: Unflatten if not done already (dedup stays later) ---
            let mut arr = match arr_opt.take() {
                Some(a) => a,
                None => {
                    let mut a = flat_succ.to_array_state(iter_state.array());
                    if let Err(e) = crate::materialize::materialize_array_state(
                        &self.ctx,
                        &mut a,
                        self.compiled.spec_may_produce_lazy,
                    ) {
                        return BfsIterOutcome::Terminate(self.bfs_error_return(
                            iter_state,
                            storage,
                            EvalCheckError::Eval(e).into(),
                        ));
                    }
                    a
                }
            };

            arr.set_cached_fingerprint(succ_fp);

            // --- Constraint observers ---
            if has_constraints {
                self.ctx.set_tlc_level(current_level);
            }
            match candidate_observer.observe_successor(
                self,
                &SuccessorObservationCtx {
                    current: iter_state.array(),
                    parent_fp: fp,
                    succ: &arr,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            ) {
                Ok(ExplorationSignal::Continue) => {
                    if has_constraints {
                        self.ctx.set_tlc_level(succ_level);
                    }
                }
                Ok(ExplorationSignal::Skip) => {
                    if has_constraints {
                        self.ctx.set_tlc_level(succ_level);
                    }
                    continue;
                }
                Ok(ExplorationSignal::Stop(result)) => {
                    iter_state.return_to(storage, self);
                    return BfsIterOutcome::Terminate(result);
                }
                Err(error) => {
                    return BfsIterOutcome::Terminate(
                        self.bfs_error_return(iter_state, storage, error),
                    );
                }
            }

            observable_successor_count += 1;
            prof.count_successors(1);
            self.record_transitions(1);

            if cache_for_liveness {
                liveness_data.push((arr.clone(), succ_fp));
                if has_action_tags {
                    liveness_action_tags.push(action_tag);
                }
            }

            #[cfg(debug_assertions)]
            if need_detail_log {
                debug_succ_data.push((succ_fp, arr.clone(), None));
            }

            if has_eval_implied_actions && succ_fp != fp {
                let outcome = crate::checker_ops::check_eval_implied_actions_for_transition(
                    &mut self.ctx,
                    &self.compiled.eval_implied_actions,
                    iter_state.array(),
                    fp,
                    &arr,
                    succ_fp,
                );
                if let Some(result) = self.handle_implied_action_outcome(
                    iter_state, storage, outcome, fp, &arr, succ_fp, succ_depth,
                ) {
                    return BfsIterOutcome::Terminate(result);
                }
            }

            // --- Step 4: Dedup check (after transition bookkeeping) ---
            let prof_t_dedup = prof.now();
            let is_seen = match self.is_state_seen_checked(succ_fp) {
                Ok(seen) => seen,
                Err(result) => {
                    iter_state.return_to(storage, self);
                    return BfsIterOutcome::Terminate(result);
                }
            };
            if is_seen {
                prof.accum_dedup(prof_t_dedup);
                continue;
            }
            prof.accum_dedup(prof_t_dedup);

            if let Err(outcome) = self.finish_prefiltered_successor(
                iter_state,
                storage,
                queue,
                prof,
                &mut admitted_observer,
                arr,
                PendingSuccessor {
                    parent_fp: fp,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            ) {
                return outcome;
            }
        }

        #[cfg(debug_assertions)]
        if need_detail_log {
            self.debug_log_bfs_successors(
                fp,
                state_tlc_fp,
                _current_depth,
                iter_state.array(),
                _registry,
                had_raw,
                debug_actions_this_state,
                &debug_succ_data,
            );
        }

        let liveness_tags = if has_action_tags {
            &liveness_action_tags[..]
        } else {
            &[]
        };
        let mut state_observer = CompositeObserver::state_completion(
            self.exploration.check_deadlock,
            self.inline_liveness_active(),
        );
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            observable_successor_count == 0,
            had_raw,
            cache_for_liveness.then_some(liveness_data.as_slice()),
            liveness_tags,
        ) {
            return outcome;
        }
        if let Err(outcome) =
            self.cache_full_state_batch_liveness(iter_state, storage, fp, &liveness_data)
        {
            return outcome;
        }

        self.record_cooperative_monolithic_successors(observable_successor_count);

        iter_state.return_to(storage, self);
        BfsIterOutcome::Continue
    }

    /// Process the output of a compiled BFS step with deferred unflatten.
    ///
    /// Uses the same zero-allocation-for-duplicates pattern as the fused JIT
    /// dispatch path: flat fingerprint first (no Value allocation), dedup
    /// check, and only unflatten to ArrayState for genuinely new states.
    /// Since 80-95% of successors are duplicates, this avoids most of the
    /// ArrayState allocation overhead.
    ///
    /// The compiled step provides fast native successor generation (action
    /// dispatch + inline fingerprint + first-level dedup via AtomicFpSet);
    /// this method performs second-level dedup against the model checker's
    /// global seen set and handles invariant checking with proper trace
    /// reconstruction.
    ///
    /// Part of #3988: Compiled BFS step with deferred unflatten.
    /// Part of #4034: Wire CompiledBfsStep into model checker BFS loop.
    fn process_compiled_bfs_output<S: BfsStorage, Q: BfsFrontier<Entry = S::QueueEntry>>(
        &mut self,
        iter_state: &mut BfsIterState,
        storage: &mut S,
        queue: &mut Q,
        params: &BfsStepParams<'_>,
        prof: &mut BfsProfile,
        output: CompiledStepOutput<'_>,
    ) -> BfsIterOutcome {
        use super::super::invariants::{
            fingerprint_jit_flat_successor, unflatten_i64_to_array_state_with_input,
        };

        let &BfsStepParams {
            registry: _registry,
            current_depth: _current_depth,
            succ_depth,
            current_level: _current_level,
            succ_level,
        } = params;
        let fp = iter_state.fp();
        let cache_for_liveness = self.liveness_cache.cache_for_liveness;
        let state_var_count = output.state_len();
        let had_raw_successors = output.generated_count() > 0;
        let succ_count = output.successor_count();

        prof.count_successors(succ_count);
        self.record_transitions(succ_count);

        self.ctx.set_tlc_level(succ_level);

        // Get the registry for flat fingerprinting (clone to avoid borrow conflict).
        let registry = self.ctx.var_registry().clone();

        let has_trace_inv = !self.config.trace_invariants.is_empty();
        let skip_inv = self.cooperative_invariants_proved();
        let mut admitted_observer =
            CompositeObserver::admitted_successors_maybe_skip(has_trace_inv, skip_inv);
        let mut observable_successor_count = 0usize;

        let mut liveness_data: Vec<(ArrayState, Fingerprint)> = if cache_for_liveness {
            Vec::with_capacity(succ_count)
        } else {
            Vec::new()
        };

        // Fused pass with deferred unflatten: for each flat successor borrowed
        // from the compiled step scratch, try flat fingerprinting first (zero
        // allocation), then dedup, and only unflatten to ArrayState for new
        // states.
        let successor_result = output.for_each_successor(|flat_succ| {
            // --- Step 1: Try flat fingerprint (no Value allocation) ---
            let prof_t_fp = prof.now();
            let (succ_fp, mut arr_opt) = match fingerprint_jit_flat_successor(
                iter_state.array(),
                flat_succ,
                state_var_count,
                None, // No JIT input snapshot for compiled step
                &registry,
            ) {
                Some((flat_fp, _xor)) => (flat_fp, None),
                None => {
                    // Compound variable modified — need full unflatten for fingerprint.
                    let mut arr = unflatten_i64_to_array_state_with_input(
                        iter_state.array(),
                        flat_succ,
                        state_var_count,
                        None,
                    );
                    if let Err(e) = crate::materialize::materialize_array_state(
                        &self.ctx,
                        &mut arr,
                        self.compiled.spec_may_produce_lazy,
                    ) {
                        return Err(BfsIterOutcome::Terminate(self.bfs_error_return(
                            iter_state,
                            storage,
                            crate::EvalCheckError::Eval(e).into(),
                        )));
                    }
                    let fp_val = match self.array_state_fingerprint(&mut arr) {
                        Ok(f) => f,
                        Err(e) => {
                            return Err(BfsIterOutcome::Terminate(
                                self.bfs_error_return(iter_state, storage, e),
                            ));
                        }
                    };
                    (fp_val, Some(arr))
                }
            };
            prof.accum_fingerprint(prof_t_fp);

            observable_successor_count += 1;

            if cache_for_liveness {
                let arr = match arr_opt.take() {
                    Some(arr) => arr,
                    None => {
                        let mut arr = unflatten_i64_to_array_state_with_input(
                            iter_state.array(),
                            flat_succ,
                            state_var_count,
                            None,
                        );
                        if let Err(e) = crate::materialize::materialize_array_state(
                            &self.ctx,
                            &mut arr,
                            self.compiled.spec_may_produce_lazy,
                        ) {
                            return Err(BfsIterOutcome::Terminate(self.bfs_error_return(
                                iter_state,
                                storage,
                                crate::EvalCheckError::Eval(e).into(),
                            )));
                        }
                        arr.set_cached_fingerprint(succ_fp);
                        arr
                    }
                };
                liveness_data.push((arr.clone(), succ_fp));
                arr_opt = Some(arr);
            }

            // --- Step 2: Dedup check (no allocation for duplicates!) ---
            let prof_t_dedup = prof.now();
            let is_seen = match self.is_state_seen_checked(succ_fp) {
                Ok(seen) => seen,
                Err(result) => {
                    iter_state.return_to(storage, self);
                    return Err(BfsIterOutcome::Terminate(result));
                }
            };
            if is_seen {
                prof.accum_dedup(prof_t_dedup);
                // Hot path: duplicate state — ZERO allocation.
                return Ok(());
            }
            prof.accum_dedup(prof_t_dedup);

            // --- Step 3: New state — unflatten only now (cold path) ---
            let arr = match arr_opt.take() {
                Some(a) => a,
                None => {
                    // Scalar fast-path fingerprint succeeded but state is new.
                    let mut a = unflatten_i64_to_array_state_with_input(
                        iter_state.array(),
                        flat_succ,
                        state_var_count,
                        None,
                    );
                    if let Err(e) = crate::materialize::materialize_array_state(
                        &self.ctx,
                        &mut a,
                        self.compiled.spec_may_produce_lazy,
                    ) {
                        return Err(BfsIterOutcome::Terminate(self.bfs_error_return(
                            iter_state,
                            storage,
                            crate::EvalCheckError::Eval(e).into(),
                        )));
                    }
                    a
                }
            };

            // finish_prefiltered_successor handles invariant checking,
            // trace recording, and enqueuing the successor.
            self.finish_prefiltered_successor(
                iter_state,
                storage,
                queue,
                prof,
                &mut admitted_observer,
                arr,
                PendingSuccessor {
                    parent_fp: fp,
                    succ_fp,
                    succ_depth,
                    succ_level,
                },
            )?;
            Ok(())
        });
        if let Err(outcome) = successor_result {
            return outcome;
        }

        let mut state_observer = CompositeObserver::state_completion(
            self.exploration.check_deadlock,
            self.inline_liveness_active(),
        );
        if let Err(outcome) = self.run_state_completion_observers(
            iter_state,
            storage,
            &mut state_observer,
            observable_successor_count == 0,
            had_raw_successors,
            cache_for_liveness.then_some(liveness_data.as_slice()),
            &[], // No action tags from compiled step
        ) {
            return outcome;
        }
        if let Err(outcome) =
            self.cache_full_state_batch_liveness(iter_state, storage, fp, &liveness_data)
        {
            return outcome;
        }

        self.record_cooperative_monolithic_successors(observable_successor_count);

        iter_state.return_to(storage, self);
        BfsIterOutcome::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::model_checker::bfs::compiled_step_trait::{
        BfsStepError, CompiledBfsStep, FlatBfsStepOutput,
    };
    use crate::config::Config;
    use crate::test_support::parse_module;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn explicit_coverage_vetoes_monolithic_jit_route() {
        assert!(monolithic_jit_route_ready(false, true));
        assert!(!monolithic_jit_route_ready(true, true));
        assert!(!monolithic_jit_route_ready(false, false));
    }

    struct ScopedOnlyCompiledStep {
        scoped_calls: Arc<AtomicUsize>,
        owned_calls: Arc<AtomicUsize>,
    }

    impl CompiledBfsStep for ScopedOnlyCompiledStep {
        fn state_len(&self) -> usize {
            1
        }

        fn step_flat(&self, _state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
            self.owned_calls.fetch_add(1, Ordering::SeqCst);
            panic!("full-state compiled BFS path must use step_flat_scoped");
        }

        fn step_flat_scoped<'a>(
            &self,
            state: &[i64],
            scratch: &'a mut CompiledBfsStepScratch,
        ) -> Result<CompiledStepOutput<'a>, BfsStepError> {
            self.scoped_calls.fetch_add(1, Ordering::SeqCst);
            scratch.clear();
            let start = scratch.append_successor_template(state)?;
            scratch.successor_mut(start, self.state_len())?[0] += 1;
            let output = scratch.output_ref(self.state_len(), 1, 1, true, None, None)?;
            Ok(CompiledStepOutput::from_borrowed(output, self.state_len()))
        }
    }

    #[test]
    fn compiled_bfs_scoped_step_helper_avoids_owned_flat_materialization() {
        let module = parse_module(
            r#"
---- MODULE ScopedCompiledBfsStepTest ----
VARIABLE x
Init == x = 41
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let scoped_calls = Arc::new(AtomicUsize::new(0));
        let owned_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.compiled_bfs_step = Some(Box::new(ScopedOnlyCompiledStep {
            scoped_calls: scoped_calls.clone(),
            owned_calls: owned_calls.clone(),
        }));

        let current = ArrayState::from_values(vec![crate::Value::SmallInt(41)]);
        let mut scratch = CompiledBfsStepScratch::new(1);
        let output = checker
            .try_compiled_bfs_step_scoped(&current, &mut scratch)
            .expect("scoped compiled step should run");

        assert!(output.is_borrowed());
        let mut successors = Vec::new();
        output
            .for_each_successor(|successor| {
                successors.push(successor.to_vec());
                Ok::<(), std::convert::Infallible>(())
            })
            .expect("successor collection is infallible");

        assert_eq!(successors, vec![vec![42]]);
        assert_eq!(scoped_calls.load(Ordering::SeqCst), 1);
        assert_eq!(owned_calls.load(Ordering::SeqCst), 0);
    }
}
