// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cooperative Dual-Engine Model Checking (CDEMC) shared state.
//!
//! Cross-engine communication hub for fused BFS + symbolic checking.
//! BFS concrete states seed symbolic engines via [`FrontierSample`],
//! symbolic proofs prune BFS work via [`invariants_proved`].
//!
//! # Architecture
//!
//! ```text
//! BFS ──frontier_tx──▶ SharedCooperativeState ──frontier_rx──▶ BMC
//!                            │
//!                     invariants_proved ◀── PDR (on Safe result)
//!                            │
//!                       SharedVerdict  ◀── any lane (first wins)
//! ```

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender};
use tla_core::ast::Expr;
use tla_core::Spanned;

#[cfg(feature = "ay")]
use crate::check::oracle::{ActionOracle, ActionRoute};
use crate::shared_verdict::SharedVerdict;
use tla_ay::BmcValue;

// Re-export extracted types so existing `use crate::cooperative_state::X` paths continue to work.
pub(crate) use crate::action_metrics::ActionMetrics;
pub(crate) use crate::convergence::ConvergenceTracker;
pub(crate) use crate::per_invariant::PerInvariantProofState;
pub(crate) use crate::smt_compat::is_expr_smt_compatible;

/// A sampled concrete state from the BFS frontier for symbolic seeding.
///
/// Sent from BFS to the cooperative BMC lane at level boundaries.
/// Each sample represents one fully-explored BFS state that can serve
/// as a starting point for deeper symbolic exploration.
#[derive(Debug, Clone)]
pub(crate) struct FrontierSample {
    /// BFS depth at which this state was discovered.
    pub(crate) depth: usize,
    /// Variable name → concrete value assignments.
    ///
    /// Uses `BmcValue` from `tla_ay`. Supports scalars (Bool/Int) and
    /// compound types (Set, Sequence, Record, Function, Tuple) that the
    /// BMC translator's `assert_concrete_state` can encode in SMT.
    /// States with lazy/non-enumerable types are skipped during sampling.
    pub(crate) assignments: Vec<(String, BmcValue)>,
}

/// Shared state for cooperative dual-engine model checking.
///
/// Thread-safe hub enabling BFS and symbolic engines to exchange information
/// at runtime. All fields use atomic types or bounded channels for lock-free,
/// non-blocking coordination.
///
/// # Ownership
///
/// Created by `FusedOrchestrator`, shared via `Arc<SharedCooperativeState>`
/// across BFS, BMC, and PDR threads.
pub(crate) struct SharedCooperativeState {
    /// Portfolio-style verdict for cross-lane early termination.
    pub(crate) verdict: Arc<SharedVerdict>,

    /// BFS → Symbolic: sampled frontier states at level boundaries.
    frontier_tx: Sender<FrontierSample>,
    frontier_rx: Receiver<FrontierSample>,

    /// Symbolic → BFS: binary "all invariants proved" flag.
    pub(crate) invariants_proved: AtomicBool,

    /// Per-invariant proof tracking for partial PDR results.
    pub(crate) per_invariant: PerInvariantProofState,

    /// BFS depth watermark: the deepest level fully explored by BFS.
    pub(crate) bfs_depth_watermark: AtomicUsize,

    /// Per-action branching metrics for oracle routing.
    pub(crate) action_metrics: Vec<ActionMetrics>,

    /// Per-action routing oracle for engine selection decisions.
    #[cfg(feature = "ay")]
    oracle: Mutex<ActionOracle>,

    /// Action name to index mapping for name-based oracle queries.
    action_name_index: FxHashMap<String, usize>,

    /// Convergence tracker for BFS new-state discovery rates.
    convergence_tracker: Mutex<ConvergenceTracker>,

    /// Count of frontier samples successfully sent to the symbolic lane.
    pub(crate) frontier_samples_sent: AtomicU64,

    /// Count of wavefront formulas successfully sent by the compressor thread.
    /// Part of #3794.
    pub(crate) wavefronts_sent: AtomicU64,

    /// Count of wavefront formulas skipped due to low entropy.
    /// Part of #3794.
    pub(crate) wavefronts_skipped_low_entropy: AtomicU64,

    /// Count of wavefront formulas consumed by the BMC lane. Part of #3834.
    pub(crate) wavefronts_consumed: AtomicU64,

    /// Count of invariant violations induced by wavefront-seeded BMC. Part of #3834.
    pub(crate) wavefront_induced_violations: AtomicU64,

    /// Wavefront compressor → BMC: compressed frontier formulas.
    wavefront_tx: Sender<crate::check::wavefront::WavefrontFormula>,
    wavefront_rx: Receiver<crate::check::wavefront::WavefrontFormula>,

    /// PDR → BMC: learned inductive lemmas (partial invariants).
    ///
    /// PDR publishes TLA+ AST expressions representing proved properties.
    /// BMC polls these and asserts them as additional assumptions at each
    /// depth, pruning the search space to only states that satisfy all
    /// known lemmas. Each engine independently translates the shared AST
    /// to its own solver terms.
    ///
    /// Part of #3835, Epic #3830 (Perfection Plan F2).
    learned_lemmas: Mutex<Vec<Spanned<Expr>>>,

    /// Maximum BMC bounded depth reached (for lane budget progress tracking).
    /// Part of #3841.
    bmc_max_depth: AtomicU64,

    /// Atomic PDR lemma counter (lock-free complement to `learned_lemmas` Vec).
    /// Part of #3841.
    pdr_lemma_count_atomic: AtomicU64,

    /// The depth `k` at which k-induction proved safety (0 = not yet proved).
    kinduction_proved_k: AtomicU64,

    /// The current depth being explored by the k-induction lane (progress tracking).
    kinduction_current_k: AtomicU64,

    // BMC incremental deepening + wavefront integration fields
    bmc_current_seed_depth: AtomicU64,
    bmc_seeds_completed: AtomicU64,
    bmc_total_unsat_depth_steps: AtomicU64,
    bfs_frontier_depth_hint: AtomicUsize,
    bmc_seeds_deprioritized: AtomicU64,

    // Starvation prevention fields (Part of #4004)
    /// Count of wavefront formulas dropped (skipped) due to BMC backpressure.
    wavefronts_dropped_backpressure: AtomicU64,
    /// Count of frontier samples dropped due to BMC backpressure.
    frontier_samples_dropped_backpressure: AtomicU64,
    /// Count of translator refreshes triggered by learned clause accumulation.
    /// Part of #4006.
    translator_refreshes: AtomicU64,
    /// Flag set by BFS when it has completed (regardless of result).
    /// BMC cooperative loop checks this to exit cleanly when BFS finishes
    /// without publishing a resolved verdict (e.g., Unknown/error results).
    /// Part of #4002.
    bfs_complete: AtomicBool,

    /// Cross-thread interrupt handles for in-flight symbolic-lane SMT solvers.
    ///
    /// Symbolic lanes (cooperative BMC, k-induction) register the interrupt
    /// handle of every solver instance they create. When a definitive verdict
    /// is resolved, the fused orchestrator flips all registered handles so a
    /// solve that is *inside* `check_sat` aborts at its next internal control
    /// check instead of running to its (up to 300s) timeout — the
    /// verdict-latency defect where a solved race still held the fused join
    /// hostage. Best-effort: lanes must still poll `is_resolved()` at their
    /// own boundaries.
    solver_interrupts: Mutex<Vec<Arc<AtomicBool>>>,
    /// Latched once `interrupt_registered_solvers` has fired, so handles
    /// registered *after* the interrupt are flipped immediately (no race
    /// between a lane creating a fresh translator and the teardown sweep).
    solvers_interrupted: AtomicBool,
    /// Flag set when the BMC lane has exited (any path: result, error, panic,
    /// or cooperative teardown). The wavefront compressor thread and the BFS
    /// frontier sampler check it to stop producing seeds nobody will consume —
    /// the fused-mode CPU waste when BMC degrades instantly (e.g., translation
    /// failure on unsupported operators) while BFS keeps exploring.
    bmc_lane_done: AtomicBool,

    /// Join handles of EVERY auxiliary thread the fused orchestrator spawns
    /// besides the BFS lane itself (the BMC / PDR / k-Induction lane workers
    /// and the wavefront compressor). Registered by the orchestrator BEFORE
    /// the BFS lane thread is spawned, then sealed via
    /// [`seal_aux_lane_registration`](Self::seal_aux_lane_registration).
    ///
    /// Consumed by [`aux_lanes_terminated`](Self::aux_lanes_terminated): once
    /// every registered thread reports `is_finished()`, they are fully
    /// `join()`ed (a real `pthread_join`, so even thread-local destructors
    /// have run) and drained. This is the soundness witness that lets the
    /// sequential BFS lane enable the non-atomic `Rp` refcount fast path
    /// (`TY_RP_VALUE=1`) mid-run: after the join, no other thread in the
    /// fused orchestration can ever touch a `Value`/`Rp` refcount again.
    aux_lane_handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// Set once the orchestrator has registered ALL auxiliary thread handles.
    /// `aux_lanes_terminated` fails closed (returns `false`) until sealed, so
    /// an early poll can never observe an incomplete handle set as "all done".
    aux_lanes_sealed: AtomicBool,
    /// Latched result of [`aux_lanes_terminated`](Self::aux_lanes_terminated)
    /// so post-join polls are a single atomic load.
    aux_lanes_terminated: AtomicBool,
}

impl SharedCooperativeState {
    /// Create a new cooperative state hub.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(action_count: usize) -> Self {
        Self::with_invariant_count(action_count, 0)
    }

    /// Create a new cooperative state hub with per-invariant tracking.
    pub(crate) fn with_invariant_count(action_count: usize, invariant_count: usize) -> Self {
        let (frontier_tx, frontier_rx) = crossbeam_channel::bounded(64);
        let (wavefront_tx, wavefront_rx) = crossbeam_channel::bounded(8);

        let action_metrics = (0..action_count)
            .map(|_| ActionMetrics {
                total_successors: AtomicU64::new(0),
                times_evaluated: AtomicU64::new(0),
                smt_compatible: AtomicBool::new(false),
                jit_compiled: AtomicBool::new(false),
            })
            .collect();

        #[cfg(feature = "ay")]
        let oracle = Mutex::new(ActionOracle::all_bfs(action_count));

        Self {
            verdict: Arc::new(SharedVerdict::new()),
            frontier_tx,
            frontier_rx,
            invariants_proved: AtomicBool::new(false),
            per_invariant: PerInvariantProofState::new(invariant_count),
            bfs_depth_watermark: AtomicUsize::new(0),
            action_metrics,
            #[cfg(feature = "ay")]
            oracle,
            action_name_index: FxHashMap::default(),
            convergence_tracker: Mutex::new(ConvergenceTracker::new()),
            frontier_samples_sent: AtomicU64::new(0),
            wavefronts_sent: AtomicU64::new(0),
            wavefronts_skipped_low_entropy: AtomicU64::new(0),
            wavefronts_consumed: AtomicU64::new(0),
            wavefront_induced_violations: AtomicU64::new(0),
            wavefront_tx,
            wavefront_rx,
            learned_lemmas: Mutex::new(Vec::new()),
            bmc_max_depth: AtomicU64::new(0),
            pdr_lemma_count_atomic: AtomicU64::new(0),
            kinduction_proved_k: AtomicU64::new(0),
            kinduction_current_k: AtomicU64::new(0),
            bmc_current_seed_depth: AtomicU64::new(0),
            bmc_seeds_completed: AtomicU64::new(0),
            bmc_total_unsat_depth_steps: AtomicU64::new(0),
            bfs_frontier_depth_hint: AtomicUsize::new(0),
            bmc_seeds_deprioritized: AtomicU64::new(0),
            wavefronts_dropped_backpressure: AtomicU64::new(0),
            frontier_samples_dropped_backpressure: AtomicU64::new(0),
            translator_refreshes: AtomicU64::new(0),
            bfs_complete: AtomicBool::new(false),
            solver_interrupts: Mutex::new(Vec::new()),
            solvers_interrupted: AtomicBool::new(false),
            bmc_lane_done: AtomicBool::new(false),
            aux_lane_handles: Mutex::new(Vec::new()),
            aux_lanes_sealed: AtomicBool::new(false),
            aux_lanes_terminated: AtomicBool::new(false),
        }
    }

    /// Register the join handle of an auxiliary fused-orchestrator thread
    /// (symbolic lane worker or wavefront compressor). Must be called on the
    /// orchestrator thread BEFORE the BFS lane is spawned, and followed by
    /// [`seal_aux_lane_registration`](Self::seal_aux_lane_registration) once
    /// every auxiliary thread is registered.
    pub(crate) fn register_aux_lane_handle(&self, handle: std::thread::JoinHandle<()>) {
        self.aux_lane_handles
            .lock()
            .expect("aux_lane_handles poisoned")
            .push(handle);
    }

    /// Declare auxiliary-thread registration complete. Until this is called,
    /// [`aux_lanes_terminated`](Self::aux_lanes_terminated) fails closed.
    pub(crate) fn seal_aux_lane_registration(&self) {
        self.aux_lanes_sealed.store(true, Ordering::Release);
    }

    /// Whether EVERY registered auxiliary thread (BMC/PDR/k-Induction lanes +
    /// wavefront compressor) has fully terminated — not merely delivered its
    /// result, but actually been `join()`ed, so its thread-local destructors
    /// have run and it can never execute another instruction.
    ///
    /// Fail-closed by construction:
    /// - returns `false` until the orchestrator seals registration (so a poll
    ///   racing registration can never see an incomplete set as terminated);
    /// - only joins once every handle reports `is_finished()` (so the joins
    ///   are near-instant and never stall the polling BFS thread);
    /// - latches the result, making steady-state polls one atomic load.
    ///
    /// A panicked auxiliary thread still counts as terminated (its `join()`
    /// error payload is dropped here); termination — not success — is the
    /// property the caller needs.
    pub(crate) fn aux_lanes_terminated(&self) -> bool {
        if self.aux_lanes_terminated.load(Ordering::Acquire) {
            return true;
        }
        if !self.aux_lanes_sealed.load(Ordering::Acquire) {
            return false;
        }
        // try_lock: never block the BFS hot loop on the (uncontended) mutex.
        let Ok(mut handles) = self.aux_lane_handles.try_lock() else {
            return false;
        };
        if handles.iter().any(|h| !h.is_finished()) {
            return false;
        }
        // Every thread has finished its main closure; join for full
        // termination (TLS destructors included). Near-instant here.
        for handle in handles.drain(..) {
            // A panic payload is dropped on this thread; all other auxiliary
            // threads are already finished, and the mode flip that this
            // result gates happens strictly AFTER these joins.
            let _ = handle.join();
        }
        self.aux_lanes_terminated.store(true, Ordering::Release);
        true
    }

    /// Resize the action metrics vector to match the actual number of
    /// detected actions discovered at runtime.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn resize_action_metrics(&mut self, action_count: usize) {
        if self.action_metrics.len() == action_count {
            return;
        }
        self.action_metrics = (0..action_count)
            .map(|_| ActionMetrics {
                total_successors: AtomicU64::new(0),
                times_evaluated: AtomicU64::new(0),
                smt_compatible: AtomicBool::new(false),
                jit_compiled: AtomicBool::new(false),
            })
            .collect();
    }

    /// Send a frontier sample to the symbolic engine (non-blocking).
    pub(crate) fn send_frontier_sample(&self, sample: FrontierSample) -> bool {
        if self.frontier_tx.try_send(sample).is_ok() {
            self.frontier_samples_sent.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Receive a frontier sample from BFS (blocking with timeout).
    pub(crate) fn recv_frontier_sample(
        &self,
        timeout: std::time::Duration,
    ) -> Option<FrontierSample> {
        match self.frontier_rx.recv_timeout(timeout) {
            Ok(sample) => Some(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                self.verdict.cancel();
                None
            }
        }
    }

    /// Get an `Arc` clone of the shared verdict.
    pub(crate) fn verdict_arc(&self) -> Arc<SharedVerdict> {
        self.verdict.clone()
    }

    /// Check whether any lane has resolved the verdict.
    #[inline]
    pub(crate) fn is_resolved(&self) -> bool {
        self.verdict.is_resolved()
    }

    /// Check whether all invariants have been symbolically proved.
    #[inline]
    pub(crate) fn invariants_proved(&self) -> bool {
        self.invariants_proved.load(Ordering::Acquire)
    }

    /// Mark all invariants as proved (called by PDR on Safe result).
    pub(crate) fn set_invariants_proved(&self) {
        self.per_invariant.mark_all_proved();
        self.invariants_proved.store(true, Ordering::Release);
    }

    /// Send a compressed wavefront formula to the BMC lane.
    pub(crate) fn send_wavefront(
        &self,
        formula: crate::check::wavefront::WavefrontFormula,
    ) -> bool {
        self.wavefront_tx.try_send(formula).is_ok()
    }

    /// Receive a compressed wavefront formula from the compressor thread.
    pub(crate) fn recv_wavefront(
        &self,
        timeout: std::time::Duration,
    ) -> Option<crate::check::wavefront::WavefrontFormula> {
        match self.wavefront_rx.recv_timeout(timeout) {
            Ok(formula) => Some(formula),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                self.verdict.cancel();
                None
            }
        }
    }

    /// Record that a wavefront formula was successfully sent by the compressor.
    /// Part of #3794.
    pub(crate) fn record_wavefront_sent(&self) {
        self.wavefronts_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a wavefront was skipped due to low entropy.
    /// Part of #3794.
    pub(crate) fn record_wavefront_skipped_low_entropy(&self) {
        self.wavefronts_skipped_low_entropy
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of wavefronts successfully sent by the compressor.
    /// Part of #3794.
    #[inline]
    pub(crate) fn wavefronts_sent(&self) -> u64 {
        self.wavefronts_sent.load(Ordering::Relaxed)
    }

    /// Get the number of wavefronts skipped due to low entropy.
    /// Part of #3794.
    // diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn wavefronts_skipped_low_entropy(&self) -> u64 {
        self.wavefronts_skipped_low_entropy.load(Ordering::Relaxed)
    }

    /// Increment the count of wavefront formulas consumed by BMC. Part of #3834.
    pub(crate) fn record_wavefront_consumed(&self) {
        self.wavefronts_consumed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the count of wavefront-induced violations found by BMC. Part of #3834.
    pub(crate) fn record_wavefront_induced_violation(&self) {
        self.wavefront_induced_violations
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of wavefronts consumed by BMC. Part of #3834.
    #[inline]
    pub(crate) fn wavefronts_consumed(&self) -> u64 {
        self.wavefronts_consumed.load(Ordering::Relaxed)
    }

    /// Get the number of wavefront-induced violations found by BMC. Part of #3834.
    // diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn wavefront_induced_violations(&self) -> u64 {
        self.wavefront_induced_violations.load(Ordering::Relaxed)
    }

    /// Publish a learned inductive lemma from PDR to BMC.
    ///
    /// PDR calls this when it proves a frame clause or safety property.
    /// The lemma is a TLA+ AST expression that holds in all reachable states.
    /// BMC will poll these and assert them as additional assumptions to prune
    /// its search space.
    ///
    /// Part of #3835, Epic #3830 (Perfection Plan F2).
    pub(crate) fn publish_lemma(&self, lemma: Spanned<Expr>) {
        if let Ok(mut guard) = self.learned_lemmas.lock() {
            guard.push(lemma);
        }
    }

    /// Poll learned lemmas from PDR, returning only new lemmas since `cursor`.
    ///
    /// The caller maintains a cursor (initially 0) tracking how many lemmas
    /// it has already consumed. Returns a tuple of `(new_lemmas, new_cursor)`.
    /// This avoids re-processing previously seen lemmas without requiring
    /// the shared state to track per-consumer offsets.
    ///
    /// Part of #3835, Epic #3830 (Perfection Plan F2).
    pub(crate) fn poll_learned_lemmas(&self, cursor: usize) -> (Vec<Spanned<Expr>>, usize) {
        if let Ok(guard) = self.learned_lemmas.lock() {
            let total = guard.len();
            if total > cursor {
                let new_lemmas = guard[cursor..].to_vec();
                (new_lemmas, total)
            } else {
                (Vec::new(), cursor)
            }
        } else {
            (Vec::new(), cursor)
        }
    }

    /// Get the total count of learned lemmas published so far.
    ///
    /// Part of #3835, Epic #3830 (Perfection Plan F2).
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn learned_lemma_count(&self) -> usize {
        self.learned_lemmas
            .lock()
            .map(|guard| guard.len())
            .unwrap_or(0)
    }

    // =========================================================================
    // Lane budget progress accessors (Part of #3841)
    // =========================================================================

    /// Get the maximum BMC bounded depth reached.
    ///
    /// Used by [`LaneBudgetManager`] to detect BMC progress.
    #[inline]
    pub(crate) fn bmc_max_depth(&self) -> u64 {
        self.bmc_max_depth.load(Ordering::Relaxed)
    }

    /// Update the maximum BMC depth (monotonically increasing).
    #[inline]
    pub(crate) fn update_bmc_max_depth(&self, depth: u64) {
        self.bmc_max_depth.fetch_max(depth, Ordering::Relaxed);
    }

    /// Get the PDR lemma count (lock-free atomic read).
    ///
    /// Used by [`LaneBudgetManager`] to detect PDR progress.
    // diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn pdr_lemma_count(&self) -> u64 {
        self.pdr_lemma_count_atomic.load(Ordering::Relaxed)
    }

    /// Increment the atomic PDR lemma counter.
    ///
    /// Should be called alongside `publish_lemma` for lock-free progress tracking.
    #[inline]
    pub(crate) fn increment_pdr_lemma_count(&self) {
        self.pdr_lemma_count_atomic.fetch_add(1, Ordering::Relaxed);
    }

    // =========================================================================
    // k-Induction progress tracking
    // =========================================================================

    pub(crate) fn set_kinduction_proved_k(&self, k: usize) {
        self.kinduction_proved_k.store(k as u64, Ordering::Release);
    }

    // diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn kinduction_proved_k(&self) -> u64 {
        self.kinduction_proved_k.load(Ordering::Acquire)
    }

    pub(crate) fn update_kinduction_current_k(&self, k: usize) {
        self.kinduction_current_k.store(k as u64, Ordering::Relaxed);
    }

    // diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn kinduction_current_k(&self) -> u64 {
        self.kinduction_current_k.load(Ordering::Relaxed)
    }

    /// Mark a specific invariant as proved by PDR.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn mark_invariant_proved(&self, index: usize) {
        self.per_invariant.mark_proved(index);
        if self.per_invariant.all_proved() {
            self.invariants_proved.store(true, Ordering::Release);
        }
    }

    /// Check whether a specific invariant has been proved by PDR.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn is_invariant_proved(&self, index: usize) -> bool {
        self.per_invariant.is_proved(index)
    }

    /// Check if any (but not all) invariants have been proved.
    #[inline]
    pub(crate) fn has_partial_proofs(&self) -> bool {
        let count = self.per_invariant.proved_count();
        count > 0 && count < self.per_invariant.total_count()
    }

    /// Update the BFS depth watermark.
    pub(crate) fn update_bfs_depth(&self, depth: usize) {
        self.bfs_depth_watermark.fetch_max(depth, Ordering::Relaxed);
    }

    /// Record that action `action_id` produced `successor_count` successors.
    #[inline]
    pub(crate) fn record_action_firing(&self, action_id: usize, successor_count: u64) {
        if let Some(metrics) = self.action_metrics.get(action_id) {
            metrics.record_firing(successor_count);
        }
    }

    /// Record monolithic Next relation firing across all actions.
    #[inline]
    pub(crate) fn record_monolithic_firing(&self, total_successors: u64) {
        if self.action_metrics.is_empty() {
            return;
        }
        for (i, metrics) in self.action_metrics.iter().enumerate() {
            if i == 0 {
                metrics.record_firing(total_successors);
            } else {
                metrics.times_evaluated.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Mark action `action_id` as SMT-compatible based on static analysis.
    pub(crate) fn mark_smt_compatible(&self, action_id: usize, compatible: bool) {
        if let Some(metrics) = self.action_metrics.get(action_id) {
            metrics.smt_compatible.store(compatible, Ordering::Relaxed);
        }
    }

    /// Mark action `action_id` as JIT-compiled to native code.
    ///
    /// Part of #3854.
    // trait/shape-required or diagnostics-only
    #[allow(dead_code)]
    pub(crate) fn mark_jit_compiled(&self, action_id: usize) {
        if let Some(metrics) = self.action_metrics.get(action_id) {
            metrics.jit_compiled.store(true, Ordering::Relaxed);
        }
    }

    /// Check whether action `action_id` has been JIT-compiled.
    ///
    /// Part of #3854.
    // trait/shape-required or diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn is_action_jit_compiled(&self, action_id: usize) -> bool {
        self.action_metrics
            .get(action_id)
            .map_or(false, |m| m.jit_compiled.load(Ordering::Relaxed))
    }

    /// Query the oracle's routing decision for a single action.
    // trait/shape-required or diagnostics-only
    #[allow(dead_code)]
    #[cfg(feature = "ay")]
    pub(crate) fn query_route(&self, action_index: usize) -> ActionRoute {
        self.oracle
            .lock()
            .map(|guard| guard.route(action_index))
            .unwrap_or(ActionRoute::BfsOnly)
    }

    /// Trigger an oracle reroute based on accumulated metrics.
    #[cfg(feature = "ay")]
    pub(crate) fn maybe_reroute_oracle(&self, current_depth: usize) -> bool {
        let smt_flags: Vec<bool> = self
            .action_metrics
            .iter()
            .map(|m| m.smt_compatible.load(Ordering::Relaxed))
            .collect();

        self.oracle
            .lock()
            .map(|mut guard| guard.reroute(current_depth, &self.action_metrics, &smt_flags))
            .unwrap_or(false)
    }

    /// Initialize the oracle with SMT compatibility flags.
    #[cfg(feature = "ay")]
    pub(crate) fn initialize_oracle(&self, smt_compatible: &[bool]) {
        if let Ok(mut guard) = self.oracle.lock() {
            *guard = ActionOracle::new(&self.action_metrics, smt_compatible);
        }
    }

    /// Register action names for name-based oracle lookups.
    pub(crate) fn register_action_names(&mut self, names: &[String]) {
        self.action_name_index.clear();
        for (i, name) in names.iter().enumerate() {
            self.action_name_index.insert(name.clone(), i);
        }
    }

    /// Record BFS convergence data at a depth boundary.
    pub(crate) fn record_convergence(&self, depth: usize, new_states: u64, total_states: u64) {
        if let Ok(mut tracker) = self.convergence_tracker.lock() {
            tracker.record(depth, new_states, total_states);
        }
    }

    // =========================================================================
    // BMC incremental deepening + wavefront integration accessors
    // =========================================================================

    pub(crate) fn bmc_start_seed(&self) {
        self.bmc_current_seed_depth.store(0, Ordering::Relaxed);
    }

    pub(crate) fn bmc_report_depth_progress(&self, depth: u64) {
        self.bmc_current_seed_depth.store(depth, Ordering::Relaxed);
        self.update_bmc_max_depth(depth);
    }

    pub(crate) fn bmc_complete_seed(&self, max_unsat_depth: u64) {
        self.bmc_seeds_completed.fetch_add(1, Ordering::Relaxed);
        self.bmc_total_unsat_depth_steps
            .fetch_add(max_unsat_depth, Ordering::Relaxed);
    }

    // diagnostics-only
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn bmc_current_seed_depth(&self) -> u64 {
        self.bmc_current_seed_depth.load(Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn bmc_seeds_completed(&self) -> u64 {
        self.bmc_seeds_completed.load(Ordering::Relaxed)
    }

    #[must_use]
    pub(crate) fn bmc_avg_seed_depth(&self) -> f64 {
        let completed = self.bmc_seeds_completed.load(Ordering::Relaxed);
        if completed == 0 {
            return 0.0;
        }
        let total_steps = self.bmc_total_unsat_depth_steps.load(Ordering::Relaxed);
        total_steps as f64 / completed as f64
    }

    pub(crate) fn update_bfs_frontier_depth_hint(&self, depth: usize) {
        self.bfs_frontier_depth_hint
            .fetch_max(depth, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn bfs_frontier_depth_hint(&self) -> usize {
        self.bfs_frontier_depth_hint.load(Ordering::Relaxed)
    }

    pub(crate) fn record_bmc_seed_deprioritized(&self) {
        self.bmc_seeds_deprioritized.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn bmc_seeds_deprioritized(&self) -> u64 {
        self.bmc_seeds_deprioritized.load(Ordering::Relaxed)
    }

    // =========================================================================
    // Starvation prevention (Part of #4004)
    // =========================================================================

    /// Record that a wavefront was dropped due to BMC backpressure.
    pub(crate) fn record_wavefront_dropped_backpressure(&self) {
        self.wavefronts_dropped_backpressure
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get the count of wavefronts dropped due to backpressure.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn wavefronts_dropped_backpressure(&self) -> u64 {
        self.wavefronts_dropped_backpressure.load(Ordering::Relaxed)
    }

    /// Record that a frontier sample was dropped due to BMC backpressure.
    pub(crate) fn record_frontier_sample_dropped_backpressure(&self) {
        self.frontier_samples_dropped_backpressure
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get the count of frontier samples dropped due to backpressure.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn frontier_samples_dropped_backpressure(&self) -> u64 {
        self.frontier_samples_dropped_backpressure
            .load(Ordering::Relaxed)
    }

    /// Record that the BMC translator was refreshed to clear accumulated state.
    /// Part of #4006.
    pub(crate) fn record_translator_refresh(&self) {
        self.translator_refreshes.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the count of translator refreshes.
    /// Part of #4006.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn translator_refreshes(&self) -> u64 {
        self.translator_refreshes.load(Ordering::Relaxed)
    }

    /// Detect whether BMC is starved: the wavefront channel is full while
    /// BMC has consumed far fewer wavefronts than have been sent.
    ///
    /// Returns the starvation gap (sent - consumed). A gap larger than
    /// [`STARVATION_THRESHOLD`] triggers wavefront skipping in BMC.
    ///
    /// Part of #4004.
    #[must_use]
    pub(crate) fn wavefront_starvation_gap(&self) -> u64 {
        let sent = self.wavefronts_sent.load(Ordering::Relaxed);
        let consumed = self.wavefronts_consumed.load(Ordering::Relaxed);
        sent.saturating_sub(consumed)
    }

    /// Drain and discard all pending wavefront formulas from the channel,
    /// keeping only the most recent one (if any). Returns the number of
    /// wavefronts drained.
    ///
    /// This is the core backpressure mechanism: when BMC falls behind,
    /// intermediate wavefronts are discarded so BMC can catch up to the
    /// current frontier rather than processing stale intermediate states.
    ///
    /// Part of #4004.
    pub(crate) fn drain_stale_wavefronts(
        &self,
    ) -> (usize, Option<crate::check::wavefront::WavefrontFormula>) {
        let mut drained = 0;
        let mut latest = None;
        while let Ok(formula) = self.wavefront_rx.try_recv() {
            if let Some(prev) = latest.replace(formula) {
                // The previous one is now stale — count it as dropped.
                drop(prev);
                drained += 1;
            }
        }
        (drained, latest)
    }

    /// Drain and discard all pending frontier samples from the channel,
    /// keeping only the most recent one (if any). Returns the number of
    /// samples drained.
    ///
    /// Part of #4004.
    pub(crate) fn drain_stale_frontier_samples(&self) -> (usize, Option<FrontierSample>) {
        let mut drained = 0;
        let mut latest = None;
        while let Ok(sample) = self.frontier_rx.try_recv() {
            if let Some(prev) = latest.replace(sample) {
                drop(prev);
                drained += 1;
            }
        }
        (drained, latest)
    }

    #[must_use]
    pub(crate) fn should_prioritize_seed(&self, seed_depth: usize) -> bool {
        let hint = self.bfs_frontier_depth_hint.load(Ordering::Relaxed);
        if hint == 0 || seed_depth >= hint {
            return true;
        }
        hint.saturating_sub(seed_depth) <= 2
    }

    // =========================================================================
    // BFS completion signaling (Part of #4002)
    // =========================================================================

    /// Mark BFS as complete. Called when the BFS lane finishes, regardless
    /// of whether it published a resolved verdict. This allows the cooperative
    /// BMC loop to detect BFS completion and exit cleanly instead of spinning
    /// forever waiting for frontier states that will never arrive.
    pub(crate) fn mark_bfs_complete(&self) {
        self.bfs_complete.store(true, Ordering::Release);
    }

    /// Check whether the BFS lane has completed.
    #[inline]
    pub(crate) fn is_bfs_complete(&self) -> bool {
        self.bfs_complete.load(Ordering::Acquire)
    }

    // =========================================================================
    // Symbolic-lane teardown (verdict-latency fix)
    // =========================================================================

    /// Register a solver's cross-thread interrupt handle for cooperative
    /// teardown. Called by symbolic lanes (cooperative BMC, k-induction) for
    /// every solver instance they create. If the teardown sweep has already
    /// fired, the handle is flipped immediately so a translator created
    /// mid-teardown cannot start a fresh long solve.
    pub(crate) fn register_solver_interrupt(&self, handle: Arc<AtomicBool>) {
        if let Ok(mut handles) = self.solver_interrupts.lock() {
            handles.push(handle.clone());
        }
        // Re-check AFTER publishing the handle: closes the race where
        // `interrupt_registered_solvers` swept between our push and its
        // flag store (either the sweep saw our handle, or we see its flag).
        if self.solvers_interrupted.load(Ordering::Acquire) {
            handle.store(true, Ordering::Release);
        }
    }

    /// Interrupt every registered in-flight symbolic solver.
    ///
    /// Called by the fused orchestrator the moment a definitive verdict
    /// exists, so solves stuck *inside* `check_sat` abort at their next
    /// internal control check instead of holding the run hostage. Idempotent;
    /// handles registered later are flipped on registration.
    pub(crate) fn interrupt_registered_solvers(&self) {
        self.solvers_interrupted.store(true, Ordering::Release);
        if let Ok(handles) = self.solver_interrupts.lock() {
            for handle in handles.iter() {
                handle.store(true, Ordering::Release);
            }
        }
    }

    /// Mark the BMC lane as exited (any path). Stops the wavefront compressor
    /// and the BFS frontier sampler from producing seeds nobody consumes.
    pub(crate) fn mark_bmc_lane_done(&self) {
        self.bmc_lane_done.store(true, Ordering::Release);
    }

    /// Check whether the BMC lane has exited.
    #[inline]
    pub(crate) fn is_bmc_lane_done(&self) -> bool {
        self.bmc_lane_done.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_cooperative_state_creation() {
        let state = SharedCooperativeState::new(3);
        assert!(!state.is_resolved());
        assert!(!state.invariants_proved());
        assert_eq!(state.bfs_depth_watermark.load(Ordering::Relaxed), 0);
        assert_eq!(state.action_metrics.len(), 3);
    }

    #[test]
    fn test_frontier_sample_send_recv() {
        let state = Arc::new(SharedCooperativeState::new(0));
        let sample = FrontierSample {
            depth: 5,
            assignments: vec![
                ("x".to_string(), BmcValue::Int(42)),
                ("y".to_string(), BmcValue::Bool(true)),
            ],
        };
        assert!(state.send_frontier_sample(sample));
        let received = state
            .recv_frontier_sample(std::time::Duration::from_millis(100))
            .expect("should receive sample");
        assert_eq!(received.depth, 5);
        assert_eq!(received.assignments.len(), 2);
    }

    #[test]
    fn test_frontier_channel_bounded() {
        let state = SharedCooperativeState::new(0);
        for i in 0..64 {
            assert!(state.send_frontier_sample(FrontierSample {
                depth: i,
                assignments: vec![],
            }));
        }
        assert!(!state.send_frontier_sample(FrontierSample {
            depth: 65,
            assignments: vec![],
        }));
    }

    #[test]
    fn test_invariants_proved_flag() {
        let state = SharedCooperativeState::new(0);
        assert!(!state.invariants_proved());
        state.set_invariants_proved();
        assert!(state.invariants_proved());
    }

    #[test]
    fn test_bfs_depth_watermark() {
        let state = SharedCooperativeState::new(0);
        state.update_bfs_depth(5);
        assert_eq!(state.bfs_depth_watermark.load(Ordering::Relaxed), 5);
        state.update_bfs_depth(3);
        assert_eq!(state.bfs_depth_watermark.load(Ordering::Relaxed), 5);
        state.update_bfs_depth(10);
        assert_eq!(state.bfs_depth_watermark.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_per_invariant_proof_tracking() {
        let state = SharedCooperativeState::with_invariant_count(0, 3);
        assert!(!state.invariants_proved());
        assert!(!state.has_partial_proofs());
        assert_eq!(state.per_invariant.proved_count(), 0);

        state.mark_invariant_proved(0);
        assert!(!state.invariants_proved());
        assert!(state.has_partial_proofs());
        assert!(state.is_invariant_proved(0));
        assert!(!state.is_invariant_proved(1));
        assert!(!state.is_invariant_proved(2));
        assert_eq!(state.per_invariant.proved_count(), 1);

        state.mark_invariant_proved(1);
        assert!(!state.invariants_proved());
        assert!(state.has_partial_proofs());
        assert_eq!(state.per_invariant.proved_count(), 2);

        state.mark_invariant_proved(2);
        assert!(state.invariants_proved());
        assert!(!state.has_partial_proofs());
        assert_eq!(state.per_invariant.proved_count(), 3);
    }

    #[test]
    fn test_per_invariant_mark_all_proved() {
        let state = SharedCooperativeState::with_invariant_count(0, 3);
        state.set_invariants_proved();
        assert!(state.invariants_proved());
        assert!(state.is_invariant_proved(0));
        assert!(state.is_invariant_proved(1));
        assert!(state.is_invariant_proved(2));
        assert!(state.per_invariant.all_proved());
    }

    #[test]
    fn test_per_invariant_double_mark_no_double_count() {
        let state = SharedCooperativeState::with_invariant_count(0, 2);
        assert!(state.per_invariant.mark_proved(0));
        assert!(!state.per_invariant.mark_proved(0));
        assert_eq!(state.per_invariant.proved_count(), 1);
    }

    #[test]
    fn test_per_invariant_out_of_bounds() {
        let state = SharedCooperativeState::with_invariant_count(0, 2);
        assert!(!state.is_invariant_proved(5));
        assert!(!state.per_invariant.mark_proved(5));
    }

    #[test]
    fn test_per_invariant_collect_unproved() {
        let state = SharedCooperativeState::with_invariant_count(0, 4);
        state.mark_invariant_proved(1);
        state.mark_invariant_proved(3);
        let mut unproved = Vec::new();
        state.per_invariant.collect_unproved_indices(&mut unproved);
        assert_eq!(unproved, vec![0, 2]);
    }

    #[test]
    fn test_per_invariant_zero_invariants() {
        let state = SharedCooperativeState::with_invariant_count(0, 0);
        assert!(state.per_invariant.all_proved());
        assert!(!state.has_partial_proofs());
    }

    #[test]
    fn test_concurrent_frontier_access() {
        let state = Arc::new(SharedCooperativeState::new(0));
        let state_sender = state.clone();
        let state_receiver = state.clone();

        std::thread::scope(|scope| {
            scope.spawn(move || {
                for i in 0..10 {
                    state_sender.send_frontier_sample(FrontierSample {
                        depth: i,
                        assignments: vec![("x".to_string(), BmcValue::Int(i as i64))],
                    });
                }
            });

            scope.spawn(move || {
                let mut received = 0;
                while received < 10 {
                    if state_receiver
                        .recv_frontier_sample(std::time::Duration::from_millis(500))
                        .is_some()
                    {
                        received += 1;
                    }
                }
            });
        });

        assert_eq!(state.frontier_samples_sent.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_verdict_arc_shares_same_verdict() {
        let state = SharedCooperativeState::new(0);
        let verdict = state.verdict_arc();
        verdict.publish(crate::shared_verdict::Verdict::Satisfied);
        assert!(state.is_resolved());
        assert_eq!(
            state.verdict.get(),
            Some(crate::shared_verdict::Verdict::Satisfied)
        );
    }

    #[test]
    fn test_action_metrics_record_firing() {
        let state = SharedCooperativeState::new(3);
        state.record_action_firing(0, 5);
        assert_eq!(
            state.action_metrics[0]
                .times_evaluated
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            state.action_metrics[0]
                .total_successors
                .load(Ordering::Relaxed),
            5
        );
        state.record_action_firing(0, 3);
        assert_eq!(
            state.action_metrics[0]
                .times_evaluated
                .load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            state.action_metrics[0]
                .total_successors
                .load(Ordering::Relaxed),
            8
        );
        assert_eq!(
            state.action_metrics[1]
                .times_evaluated
                .load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_action_metrics_out_of_range_is_noop() {
        let state = SharedCooperativeState::new(1);
        state.record_action_firing(5, 10);
        state.mark_smt_compatible(5, true);
        assert_eq!(
            state.action_metrics[0]
                .times_evaluated
                .load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_monolithic_firing_distributes_correctly() {
        let state = SharedCooperativeState::new(3);
        state.record_monolithic_firing(10);
        assert_eq!(
            state.action_metrics[0]
                .total_successors
                .load(Ordering::Relaxed),
            10
        );
        assert_eq!(
            state.action_metrics[0]
                .times_evaluated
                .load(Ordering::Relaxed),
            1
        );
        for i in 1..3 {
            assert_eq!(
                state.action_metrics[i]
                    .total_successors
                    .load(Ordering::Relaxed),
                0
            );
            assert_eq!(
                state.action_metrics[i]
                    .times_evaluated
                    .load(Ordering::Relaxed),
                1
            );
        }
    }

    #[test]
    fn test_monolithic_firing_empty_actions_is_noop() {
        let state = SharedCooperativeState::new(0);
        assert!(
            state.action_metrics.is_empty(),
            "zero-action state should have empty action_metrics"
        );
        state.record_monolithic_firing(10);
        // After recording on zero actions, metrics should remain empty
        assert!(
            state.action_metrics.is_empty(),
            "recording monolithic firing with zero actions should be a no-op"
        );
    }

    #[test]
    fn test_smt_compatible_marking() {
        let state = SharedCooperativeState::new(3);
        for m in &state.action_metrics {
            assert!(!m.smt_compatible.load(Ordering::Relaxed));
        }
        state.mark_smt_compatible(0, true);
        state.mark_smt_compatible(2, true);
        assert!(state.action_metrics[0]
            .smt_compatible
            .load(Ordering::Relaxed));
        assert!(!state.action_metrics[1]
            .smt_compatible
            .load(Ordering::Relaxed));
        assert!(state.action_metrics[2]
            .smt_compatible
            .load(Ordering::Relaxed));
    }

    #[test]
    fn test_resize_action_metrics() {
        let mut state = SharedCooperativeState::new(0);
        assert_eq!(state.action_metrics.len(), 0);
        state.resize_action_metrics(4);
        assert_eq!(state.action_metrics.len(), 4);
        for m in &state.action_metrics {
            assert_eq!(m.times_evaluated.load(Ordering::Relaxed), 0);
            assert_eq!(m.total_successors.load(Ordering::Relaxed), 0);
        }
        state.record_action_firing(0, 5);
        state.resize_action_metrics(4);
        assert_eq!(
            state.action_metrics[0]
                .total_successors
                .load(Ordering::Relaxed),
            5,
            "same-size resize should not reset data"
        );
    }

    #[test]
    fn test_register_action_names() {
        let mut state = SharedCooperativeState::new(3);
        state.register_action_names(&["Send".to_string(), "Recv".to_string(), "Tick".to_string()]);
        assert_eq!(state.action_name_index.get("Send"), Some(&0));
        assert_eq!(state.action_name_index.get("Recv"), Some(&1));
        assert_eq!(state.action_name_index.get("Tick"), Some(&2));
        assert_eq!(state.action_name_index.get("Unknown"), None);
    }

    #[test]
    fn test_register_action_names_replaces_previous() {
        let mut state = SharedCooperativeState::new(2);
        state.register_action_names(&["A".to_string(), "B".to_string()]);
        assert_eq!(state.action_name_index.len(), 2);
        state.register_action_names(&["X".to_string(), "Y".to_string()]);
        assert_eq!(state.action_name_index.len(), 2);
        assert_eq!(state.action_name_index.get("A"), None);
        assert_eq!(state.action_name_index.get("X"), Some(&0));
    }

    // =========================================================================
    // PDR lemma sharing tests. Part of #3835.
    // =========================================================================

    #[test]
    fn test_publish_and_poll_lemmas() {
        use tla_core::ast::Expr;
        use tla_core::Spanned;

        let state = SharedCooperativeState::new(0);
        assert_eq!(state.learned_lemma_count(), 0);

        // Poll with no lemmas returns empty.
        let (lemmas, cursor) = state.poll_learned_lemmas(0);
        assert!(lemmas.is_empty());
        assert_eq!(cursor, 0);

        // Publish two lemmas.
        state.publish_lemma(Spanned::dummy(Expr::Bool(true)));
        state.publish_lemma(Spanned::dummy(Expr::Bool(false)));
        assert_eq!(state.learned_lemma_count(), 2);

        // Poll from cursor 0 returns both.
        let (lemmas, cursor) = state.poll_learned_lemmas(0);
        assert_eq!(lemmas.len(), 2);
        assert_eq!(cursor, 2);

        // Poll from cursor 2 returns none (already consumed).
        let (lemmas, cursor2) = state.poll_learned_lemmas(cursor);
        assert!(lemmas.is_empty());
        assert_eq!(cursor2, 2);

        // Publish a third lemma.
        state.publish_lemma(Spanned::dummy(Expr::Bool(true)));
        assert_eq!(state.learned_lemma_count(), 3);

        // Poll from cursor 2 returns only the new one.
        let (lemmas, cursor3) = state.poll_learned_lemmas(cursor);
        assert_eq!(lemmas.len(), 1);
        assert_eq!(cursor3, 3);
    }

    #[test]
    fn test_publish_lemma_concurrent() {
        use tla_core::ast::Expr;
        use tla_core::Spanned;

        let state = Arc::new(SharedCooperativeState::new(0));
        let state_pub = state.clone();
        let state_poll = state.clone();

        std::thread::scope(|scope| {
            // Publisher thread: publish 20 lemmas.
            scope.spawn(move || {
                for _ in 0..20 {
                    state_pub.publish_lemma(Spanned::dummy(Expr::Bool(true)));
                }
            });

            // Poller thread: poll until all 20 are consumed.
            scope.spawn(move || {
                let mut cursor = 0;
                let mut total = 0;
                while total < 20 {
                    let (new_lemmas, new_cursor) = state_poll.poll_learned_lemmas(cursor);
                    total += new_lemmas.len();
                    cursor = new_cursor;
                    if total < 20 {
                        std::thread::yield_now();
                    }
                }
                assert_eq!(total, 20);
                assert_eq!(cursor, 20);
            });
        });

        assert_eq!(state.learned_lemma_count(), 20);
    }

    // =========================================================================
    // Starvation prevention tests. Part of #4004.
    // =========================================================================

    #[test]
    fn test_wavefront_starvation_gap_initially_zero() {
        let state = SharedCooperativeState::new(0);
        assert_eq!(state.wavefront_starvation_gap(), 0);
    }

    #[test]
    fn test_wavefront_starvation_gap_tracks_sent_minus_consumed() {
        let state = SharedCooperativeState::new(0);
        // Simulate 10 sent, 3 consumed.
        for _ in 0..10 {
            state.record_wavefront_sent();
        }
        for _ in 0..3 {
            state.record_wavefront_consumed();
        }
        assert_eq!(state.wavefront_starvation_gap(), 7);
    }

    #[test]
    fn test_wavefront_starvation_gap_zero_when_consumed_matches_sent() {
        let state = SharedCooperativeState::new(0);
        for _ in 0..5 {
            state.record_wavefront_sent();
            state.record_wavefront_consumed();
        }
        assert_eq!(state.wavefront_starvation_gap(), 0);
    }

    #[test]
    fn test_backpressure_counters() {
        let state = SharedCooperativeState::new(0);
        assert_eq!(state.wavefronts_dropped_backpressure(), 0);
        assert_eq!(state.frontier_samples_dropped_backpressure(), 0);

        state.record_wavefront_dropped_backpressure();
        state.record_wavefront_dropped_backpressure();
        state.record_frontier_sample_dropped_backpressure();

        assert_eq!(state.wavefronts_dropped_backpressure(), 2);
        assert_eq!(state.frontier_samples_dropped_backpressure(), 1);
    }

    #[test]
    fn test_translator_refresh_counter() {
        let state = SharedCooperativeState::new(0);
        assert_eq!(state.translator_refreshes(), 0);
        state.record_translator_refresh();
        state.record_translator_refresh();
        assert_eq!(state.translator_refreshes(), 2);
    }

    #[test]
    fn test_drain_stale_wavefronts_empty_channel() {
        let state = SharedCooperativeState::new(0);
        let (drained, latest) = state.drain_stale_wavefronts();
        assert_eq!(drained, 0);
        assert!(latest.is_none());
    }

    #[test]
    fn test_drain_stale_frontier_samples_empty_channel() {
        let state = SharedCooperativeState::new(0);
        let (drained, latest) = state.drain_stale_frontier_samples();
        assert_eq!(drained, 0);
        assert!(latest.is_none());
    }

    #[test]
    fn test_drain_stale_frontier_samples_keeps_latest() {
        let state = SharedCooperativeState::new(0);
        // Fill frontier channel with 5 samples.
        for i in 0..5 {
            state.send_frontier_sample(FrontierSample {
                depth: i,
                assignments: vec![("x".to_string(), BmcValue::Int(i as i64))],
            });
        }
        let (drained, latest) = state.drain_stale_frontier_samples();
        assert_eq!(drained, 4, "should drain 4 intermediate samples");
        let latest = latest.expect("should keep the latest sample");
        assert_eq!(latest.depth, 4, "latest sample should be the most recent");
    }

    #[test]
    fn test_drain_stale_frontier_samples_single_item() {
        let state = SharedCooperativeState::new(0);
        state.send_frontier_sample(FrontierSample {
            depth: 7,
            assignments: vec![],
        });
        let (drained, latest) = state.drain_stale_frontier_samples();
        assert_eq!(drained, 0, "single item means nothing to drain");
        let latest = latest.expect("should return the single item");
        assert_eq!(latest.depth, 7);
    }
}
