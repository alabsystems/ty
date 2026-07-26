// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS exploration engine: entry points for sequential model checking.
//!
//! Part of #2351: extracted from `run_bfs_common.rs`.
//! Part of #2356 Phase 4 Step 4d: the BFS loop body is now in `worker_loop.rs`
//! (`run_bfs_worker`). This module provides the entry points that construct
//! `SequentialTransport` and delegate to the unified loop.

use super::super::frontier::BfsFrontier;
use super::super::{CheckResult, LimitType, ModelChecker};
use super::storage_modes::BfsStorage;
use super::transport::BfsWorkerConfig;
use super::transport_seq::SequentialTransport;
use super::worker_loop::{run_bfs_worker, BfsLoopOutcome};
use crate::arena::init_worker_arena;
use crate::shared_verdict::Verdict;
use crate::storage::FingerprintPayloadWitnesses;
use tla_eval::eval_arena::init_thread_arena;

fn release_after_terminal_bfs<S: BfsStorage>(
    storage: &mut S,
    queue: &mut impl BfsFrontier<Entry = S::QueueEntry>,
    compiled_flat_payload_witnesses: &mut FingerprintPayloadWitnesses,
    depth_limit_reached: bool,
    frontier_exhausted: bool,
) {
    // A depth-limited run drained the active cursor only by truncating states
    // at the boundary. Preserve its storage exactly as the compiled loop does;
    // only a complete, unbounded exhaustion makes BFS-only payloads disposable.
    if super::release_compiled_payload_witnesses_after_terminal_bfs(
        compiled_flat_payload_witnesses,
        frontier_exhausted,
        depth_limit_reached.then_some(LimitType::Depth),
    ) {
        storage.release_after_complete_bfs();
        queue.release_after_complete_bfs();
    }
}

/// Whether the `TY_RP_VALUE=1` opt-in for the non-atomic `Rp` refcount fast path
/// is set. Read once and cached; default (unset/`0`) keeps atomic `Rp` behavior,
/// which is byte-identical to `Arc`.
fn rp_value_opt_in() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(2); // 2 = uninitialized
    match CACHE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("TY_RP_VALUE")
                .map(|v| v == "1")
                .unwrap_or(false);
            CACHE.store(on as u8, Ordering::Relaxed);
            on
        }
    }
}

/// Install (or arm) the non-atomic single-threaded `Rp` mode for a sequential
/// BFS exploration, returning an RAII guard, or `None` (atomic mode unchanged)
/// when the opt-in is off or the run is NOT provably single-threaded.
///
/// Three cases, soundest-first:
///
/// 1. **Standalone sequential run** (no portfolio lane, no cooperative lane):
///    the exploration is single-threaded from the first state — enable
///    non-atomic mode immediately (`scoped_single_threaded`).
/// 2. **Fused CDEMC BFS lane** (`cooperative` is `Some` — the default engine):
///    the orchestrator's symbolic lanes (BMC/PDR/k-Induction) and the
///    wavefront compressor are LIVE when BFS starts, and the symbolic lanes
///    run `EvalCtx` evaluation on their own threads (module loading, operator
///    expansion, SMT translation) — real `Value`/`Rp` refcount traffic that
///    may share allocations with the BFS thread (interned/global values). So
///    the loop must START atomic. But every one of those threads is
///    registered on the `SharedCooperativeState` and, on specs the
///    translators reject (all-compound specs — the common perf-relevant
///    case), they all terminate within milliseconds. We therefore ARM a
///    deferred flip (`rp_deferred_nonatomic_armed`): the sequential transport
///    polls `aux_lanes_terminated()` — which flips to `true` only after every
///    auxiliary thread has been fully `join()`ed (a real `pthread_join`, TLS
///    destructors included) — and only then enables non-atomic mode. The
///    returned guard here is a `pause_single_threaded` bracket: a no-op now
///    (mode is already atomic) whose drop RESTORES atomic mode on every
///    return/panic path of the loop, covering a mid-loop flip.
/// 3. **Standalone portfolio lane** (`portfolio_verdict` set without
///    `cooperative`): the portfolio orchestrator's racing lanes are NOT
///    termination-tracked — fail closed, stay atomic.
///
/// Independently of the case split: `TY_JIT=1` (the tiered `tla-jit` path)
/// spawns an ASYNC background compile thread at Tier-1 promotion that runs
/// CONCURRENTLY with the BFS loop while holding a clone of the bytecode chunk
/// (`tla-tir` depends on `tla-value`, so that thread performs `Rp` refcount
/// ops unbracketed). Fail closed whenever it is set. (The trust-cg AUTO
/// compile batches are different: they spawn+join under a
/// `pause_single_threaded` bracket and are safe in every case.)
fn rp_nonatomic_exploration_guard(mc: &mut ModelChecker<'_>) -> Option<tla_value::rp::RpModeGuard> {
    if !rp_value_opt_in() {
        return None;
    }
    // TY_JIT async tier compilation runs an untracked concurrent thread that
    // touches `Value`s: never engage non-atomic mode alongside it.
    if crate::check::debug::jit_enabled() {
        return None;
    }
    #[cfg(feature = "ay")]
    if mc.cooperative.is_some() {
        // Fused CDEMC lane: defer engagement until every auxiliary
        // orchestrator thread has fully terminated (see doc above). The
        // pause guard restores atomic mode on drop no matter where the flip
        // happened.
        mc.rp_deferred_nonatomic_armed = true;
        mc.rp_deferred_poll_tick = 0;
        return Some(tla_value::rp::pause_single_threaded());
    }
    if mc.portfolio_verdict.is_none() {
        // SAFETY: no concurrent engine lane is attached and this is the
        // sequential checker, so the exploration is single-threaded except for
        // self-bracketed JIT compile batches (`rp::pause_single_threaded`).
        return Some(unsafe { tla_value::rp::scoped_single_threaded() });
    }
    // Standalone portfolio racing lanes: untracked threads — fail closed.
    None
}

impl ModelChecker<'_> {
    /// Deferred engagement of the non-atomic `Rp` refcount fast path for the
    /// fused CDEMC BFS lane. Called from the sequential transport on every
    /// dequeued state; O(1 branch) when disarmed, and while armed it pays the
    /// auxiliary-thread inspection only every 64th state.
    ///
    /// Engages non-atomic mode iff EVERY auxiliary fused-orchestrator thread
    /// (BMC/PDR/k-Induction lanes + wavefront compressor) has fully
    /// terminated — verified by `SharedCooperativeState::aux_lanes_terminated`,
    /// which `join()`s each thread (full `pthread_join`, TLS destructors
    /// included) before reporting `true`.
    ///
    /// SOUNDNESS of the flip: at the moment `aux_lanes_terminated()` first
    /// returns `true`, the live threads of the fused orchestration are exactly
    /// (a) this BFS thread and (b) the orchestrator thread, which is parked in
    /// `bfs_handle.join()` until this loop finishes and performs no
    /// `Value`/`Rp` operations while parked. Everything the dead lanes ever
    /// shared with BFS is either `Rp`-free by construction (`FrontierSample` /
    /// wavefront formulas carry `BmcValue` — owned scalars/strings/vecs; the
    /// cooperative hub is atomics + `BmcValue` channels; verdicts are
    /// atomics), or its refcount traffic happened-before the `join()`s.
    /// Post-BFS Value work (channel drains, cross-validation replay) runs on
    /// the orchestrator thread only AFTER `bfs_handle.join()` returns, i.e.
    /// after the loop's RAII guard has already restored atomic mode. The
    /// trust-cg JIT compile batches reachable from this loop bracket
    /// themselves with `pause_single_threaded`. Hence, from the flip until the
    /// guard restores atomic mode, at most one thread performs `Rp` refcount
    /// ops at any instant.
    #[cfg(feature = "ay")]
    pub(in crate::check::model_checker) fn maybe_engage_rp_nonatomic_deferred(&mut self) {
        if !self.rp_deferred_nonatomic_armed {
            return;
        }
        self.rp_deferred_poll_tick = self.rp_deferred_poll_tick.wrapping_add(1);
        if self.rp_deferred_poll_tick & 0x3F != 0 {
            return;
        }
        let Some(coop) = self.cooperative.as_ref() else {
            self.rp_deferred_nonatomic_armed = false;
            return;
        };
        if coop.aux_lanes_terminated() {
            self.rp_deferred_nonatomic_armed = false;
            // SAFETY: see the SOUNDNESS argument above — every other thread
            // that could touch an `Rp` refcount has been joined, and the
            // enclosing `run_bfs_loop` guard restores atomic mode on exit.
            unsafe { tla_value::rp::set_single_threaded(true) };
            telemetry_eprintln!(
                "[rp] non-atomic refcount mode engaged (fused: all symbolic lanes + \
                 wavefront thread terminated and joined)"
            );
        }
    }

    /// Unified BFS exploration loop, generic over storage mode.
    ///
    /// Calls `run_bfs_loop_core` for the BFS iteration, then
    /// `finish_check_after_bfs` for post-loop finalization (liveness,
    /// postcondition, storage error checks).
    ///
    /// Part of #2133.
    pub(in crate::check::model_checker) fn run_bfs_loop<S: BfsStorage>(
        &mut self,
        storage: &mut S,
        queue: &mut impl BfsFrontier<Entry = S::QueueEntry>,
    ) -> CheckResult {
        // Non-atomic `Rp` refcount fast path (gated on `TY_RP_VALUE=1`).
        //
        // SOUNDNESS: this is the sequential `ModelChecker` (the parallel path is
        // `ParallelChecker`), so the BFS itself is single-threaded. Non-atomic
        // mode is UB if another thread touches an `Rp`'s refcount concurrently.
        // Standalone runs (no portfolio, no cooperative lane) enable it
        // immediately; the fused CDEMC lane arms a DEFERRED flip that engages
        // only after every auxiliary orchestrator thread has been fully joined
        // (see `rp_nonatomic_exploration_guard` and
        // `maybe_engage_rp_nonatomic_deferred`). The only remaining thread
        // source reachable from this loop is an AUTO-mode trust-cg JIT compile
        // batch, which brackets itself with `rp::pause_single_threaded()`
        // (forcing atomic while its worker threads are live). The guard
        // restores the previous (atomic) mode on drop, so the mode is correct
        // on every return/panic path.
        let _rp_mode = rp_nonatomic_exploration_guard(self);
        let result = match self.run_bfs_loop_core(storage, queue) {
            BfsLoopOutcome::Terminated(result) => *result,
            BfsLoopOutcome::Complete {
                depth_limit_reached,
                frontier_exhausted,
            } => {
                // Discard BFS-only storage only after proven exhaustion.
                // Portfolio early exit may already have dequeued an item that
                // still belongs to a resumable frontier.
                release_after_terminal_bfs(
                    storage,
                    queue,
                    &mut self.state_storage.compiled_flat_payload_witnesses,
                    depth_limit_reached,
                    frontier_exhausted,
                );
                // This is an active-memory census value: exhausted storage was
                // released above, while an early-completion frontier remains
                // fully accounted. Process VmHWM retains the historical peak.
                let active_payload_witness_bytes = storage.payload_witness_memory_bytes();
                self.finish_check_after_bfs(
                    depth_limit_reached.then_some(LimitType::Depth),
                    false,
                    active_payload_witness_bytes,
                )
            }
        };
        // Disarm the deferred Rp flip (fused mode): the loop is over; the
        // `_rp_mode` guard drop below restores atomic mode.
        #[cfg(feature = "ay")]
        {
            self.rp_deferred_nonatomic_armed = false;
        }
        // Part of #3717: publish BFS verdict to portfolio SharedVerdict.
        if let Some(ref sv) = self.portfolio_verdict {
            let verdict = match &result {
                CheckResult::Success(_) => Verdict::Satisfied,
                CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. } => Verdict::Violated,
                _ => Verdict::Unknown,
            };
            sv.publish(verdict);
        }
        // Part of #3767: publish BFS verdict to cooperative SharedVerdict.
        // This enables the symbolic lane (BMC/PDR) to observe BFS completion
        // and exit early in fused CDEMC mode.
        #[cfg(feature = "ay")]
        if let Some(ref coop) = self.cooperative {
            let verdict = match &result {
                CheckResult::Success(_) => Verdict::Satisfied,
                CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. } => Verdict::Violated,
                _ => Verdict::Unknown,
            };
            coop.verdict.publish(verdict);
            // Part of #4002: signal BFS completion so cooperative BMC loop
            // exits cleanly even when verdict is Unknown.
            coop.mark_bfs_complete();
        }
        result
    }

    /// Core BFS iteration loop, generic over storage mode.
    ///
    /// Constructs a [`SequentialTransport`] and delegates to the unified
    /// [`run_bfs_worker`] loop body. Returns `BfsLoopOutcome` so callers
    /// can select `resume_mode` when calling `finish_check_after_bfs`.
    ///
    /// Part of #2356 Phase 4 Step 4d: replaces the previous ~140-line inline
    /// loop with a single call to the unified BFS worker loop.
    pub(in crate::check::model_checker) fn run_bfs_loop_core<S: BfsStorage>(
        &mut self,
        storage: &mut S,
        queue: &mut impl BfsFrontier<Entry = S::QueueEntry>,
    ) -> BfsLoopOutcome {
        // Part of #4215: Seal the fingerprint algorithm before BFS processing begins.
        // After this point, `try_activate_compiled_fingerprinting` will panic in debug
        // builds if called, providing a structural guarantee against mid-run algorithm
        // switches that could cause domain separation violations.
        #[cfg(debug_assertions)]
        {
            self.fp_algorithm_sealed = true;
        }

        // Freeze the fingerprint domain so a mid-run AUTO lazy compile cannot
        // flip it under the dedup set. Idempotent: run_bfs_notrace already froze
        // it (before committing init states); this covers the trace/full and
        // resume entry paths that reach the loop without that pre-commit freeze.
        self.freeze_bfs_fingerprint_domain();

        // Part of #3580: Initialize eval arena on the main thread for sequential BFS.
        init_thread_arena();

        // Part of #3990: Initialize worker arena for successor state allocation.
        init_worker_arena();

        // Pre-size the global dedup set once before the loop, mirroring the
        // per-level `local_seen` sizing in trust-cg (bfs_level.rs). Init states
        // are already inserted, so `states_count()` reflects the distinct init
        // count; reserving that-with-a-floor headroom eliminates the early
        // rehash cascade as the in-memory backend grows from empty. This is a
        // pure allocation hint — `reserve` is a no-op on fixed-capacity backends
        // and never changes admission, dedup, or any verdict.
        let reserve_hint = self.states_count().max(1024);
        self.state_storage.seen_fps.reserve(reserve_hint);

        let config = BfsWorkerConfig {
            max_depth: self.exploration.max_depth,
            // Part of #3: plumb the wall-clock deadline into the unified loop.
            deadline: self.exploration.deadline,
        };
        // Part of #3717: clone the Arc to avoid holding an immutable borrow on
        // `self` across the mutable borrow needed by SequentialTransport.
        let portfolio_verdict = self.portfolio_verdict.clone();
        // Part of #3767: clone the cooperative Arc so we can reference its
        // verdict field after `self` is mutably borrowed by SequentialTransport.
        #[cfg(feature = "ay")]
        let cooperative = self.cooperative.clone();
        let mut transport = SequentialTransport::new(self, storage, queue);
        // Part of #3767: use cooperative verdict for early-exit if in fused mode,
        // falling back to the standalone portfolio verdict.
        #[cfg(feature = "ay")]
        let verdict_ref = cooperative
            .as_ref()
            .map(|c| c.verdict.as_ref())
            .or(portfolio_verdict.as_deref());
        #[cfg(not(feature = "ay"))]
        let verdict_ref = portfolio_verdict.as_deref();
        run_bfs_worker(&mut transport, &config, verdict_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::model_checker::{ArrayState, Fingerprint, State, VecDeque};

    struct ReleaseProbeStorage {
        payload: Vec<u8>,
        release_calls: usize,
    }

    impl BfsStorage for ReleaseProbeStorage {
        type QueueEntry = u8;

        fn release_after_complete_bfs(&mut self) {
            self.payload = Vec::new();
            self.release_calls += 1;
        }

        fn payload_witness_memory_bytes(&self) -> usize {
            self.payload.capacity()
        }

        fn dequeue(
            &mut self,
            _entry: Self::QueueEntry,
            _mc: &mut ModelChecker,
        ) -> Result<Option<(Fingerprint, ArrayState, usize)>, CheckResult> {
            unimplemented!("release-policy probe never dequeues")
        }

        fn return_current(&mut self, _fp: Fingerprint, _state: ArrayState, _mc: &mut ModelChecker) {
            unimplemented!("release-policy probe never returns a state")
        }

        fn admit_successor(
            &mut self,
            _fp: Fingerprint,
            _state: ArrayState,
            _parent_fp: Option<Fingerprint>,
            _current: Option<(Fingerprint, &ArrayState)>,
            _depth: usize,
            _mc: &mut ModelChecker,
        ) -> Result<Option<Self::QueueEntry>, CheckResult> {
            unimplemented!("release-policy probe never admits a successor")
        }

        fn enforce_seen_successor_duplicate(
            &mut self,
            _fp: Fingerprint,
            _candidate: &ArrayState,
            _current: Option<(Fingerprint, &ArrayState)>,
            _mc: &mut ModelChecker,
        ) -> Result<(), CheckResult> {
            unimplemented!("release-policy probe never checks duplicates")
        }

        fn use_diffs(&self, _mc: &ModelChecker) -> bool {
            unimplemented!("release-policy probe never selects successor mode")
        }

        fn checkpoint_frontier(
            &self,
            _current: &ArrayState,
            _queue: &impl BfsFrontier<Entry = Self::QueueEntry>,
            _registry: &crate::var_index::VarRegistry,
            _mc: &mut ModelChecker,
        ) -> VecDeque<State> {
            unimplemented!("release-policy probe never checkpoints")
        }

        fn cache_diff_liveness(
            &self,
            _parent_fp: Fingerprint,
            _succ_fps: Option<Vec<Fingerprint>>,
            _mc: &mut ModelChecker,
        ) -> Result<(), crate::check::CheckError> {
            unimplemented!("release-policy probe never caches liveness")
        }

        fn cache_full_liveness(
            &self,
            _parent_fp: Fingerprint,
            _successors: &[(ArrayState, Fingerprint)],
            _mc: &mut ModelChecker,
        ) -> Result<(), crate::check::CheckError> {
            unimplemented!("release-policy probe never caches liveness")
        }
    }

    #[test]
    fn depth_limited_completion_retains_payload_and_frontier_storage() {
        let mut storage = ReleaseProbeStorage {
            payload: Vec::with_capacity(1024),
            release_calls: 0,
        };
        storage.payload.extend([1, 2, 3]);
        let payload_capacity = storage.payload.capacity();
        let mut queue = VecDeque::with_capacity(256);
        queue.push_back(9);
        let queue_capacity = queue.capacity();
        let mut compiled_witnesses = FingerprintPayloadWitnesses::new();
        compiled_witnesses.record_flat_i64_slots_if_absent(Fingerprint(7), &[4, 5, 6]);
        let compiled_witness_bytes = compiled_witnesses.estimated_memory_bytes();

        release_after_terminal_bfs(
            &mut storage,
            &mut queue,
            &mut compiled_witnesses,
            true,
            true,
        );

        assert_eq!(storage.release_calls, 0);
        assert_eq!(storage.payload, [1, 2, 3]);
        assert_eq!(storage.payload.capacity(), payload_capacity);
        assert_eq!(queue.front(), Some(&9));
        assert_eq!(queue.capacity(), queue_capacity);
        assert_eq!(
            compiled_witnesses.confirm_flat_i64_slots(Fingerprint(7), &[4, 5, 6]),
            Some(true)
        );
        assert_eq!(
            compiled_witnesses.estimated_memory_bytes(),
            compiled_witness_bytes
        );
    }

    #[test]
    fn normal_completion_releases_local_frontier_and_compiled_witness_storage() {
        let mut storage = ReleaseProbeStorage {
            payload: Vec::with_capacity(1024),
            release_calls: 0,
        };
        storage.payload.extend([1, 2, 3]);
        let mut queue = VecDeque::with_capacity(256);
        queue.push_back(9);
        let mut compiled_witnesses = FingerprintPayloadWitnesses::new();
        let fresh_compiled_census = compiled_witnesses.census();
        let fresh_compiled_bytes = compiled_witnesses.estimated_memory_bytes();
        for n in 0..128 {
            compiled_witnesses.record_flat_i64_slots_if_absent(Fingerprint(n), &[n as i64]);
        }
        assert!(compiled_witnesses.estimated_memory_bytes() > 0);

        release_after_terminal_bfs(
            &mut storage,
            &mut queue,
            &mut compiled_witnesses,
            false,
            true,
        );

        assert_eq!(storage.release_calls, 1);
        assert_eq!(storage.payload.capacity(), 0);
        assert!(queue.is_empty());
        assert_eq!(queue.capacity(), 0);
        assert_eq!(compiled_witnesses.census(), fresh_compiled_census);
        assert_eq!(
            compiled_witnesses.estimated_memory_bytes(),
            fresh_compiled_bytes
        );
    }
}
