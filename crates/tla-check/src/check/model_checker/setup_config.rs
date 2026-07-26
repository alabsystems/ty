// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Configuration mutator and accessor surface for `ModelChecker`.
//!
//! Extracted from `setup.rs` as part of #2359 Phase 2 decomposition.
//! Contains setter/getter methods that configure exploration parameters,
//! storage backends, callbacks, and checkpoint settings.

use super::{
    Arc, Duration, FairnessConstraint, Fingerprint, FingerprintSet, InitProgressCallback,
    ModelChecker, PathBuf, ProgressCallback, TraceFile, TraceLocationsStorage,
};

/// Derive internal memory limit from env var or from the RSS limit.
///
/// Checks `TY_INTERNAL_MEMORY_LIMIT` first (bytes, 0 = disabled).
/// Falls back to 75% of the RSS `limit_bytes`, reserving 25% for code
/// segments, stack, eval context, and allocator overhead.
///
/// Part of #4080: OOM safety — hard internal memory cap.
pub(crate) fn internal_memory_limit_from_env_or_default(rss_limit_bytes: usize) -> usize {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let env_override = *CACHED.get_or_init(|| {
        std::env::var("TY_INTERNAL_MEMORY_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    });
    match env_override {
        Some(0) => 0, // Explicitly disabled
        Some(v) => v,
        None => {
            // 75% of the RSS limit reserved for internal stores.
            (rss_limit_bytes as f64 * 0.75) as usize
        }
    }
}

impl<'a> ModelChecker<'a> {
    fn refresh_liveness_successor_storage(&mut self, storage: &dyn FingerprintSet) {
        if !self.liveness_cache.cache_for_liveness {
            return;
        }

        let use_disk = match crate::liveness::debug::disk_successors_override() {
            Some(force) => force,
            None => {
                crate::liveness::debug::use_disk_successors()
                    || storage.prefers_disk_successor_graph()
            }
        };

        if use_disk == self.liveness_cache.successors.is_disk() {
            return;
        }

        self.liveness_cache.successors = if use_disk {
            crate::storage::SuccessorGraph::disk().expect("disk successor graph creation failed")
        } else {
            crate::storage::SuccessorGraph::default()
        };
    }

    /// Set fairness constraints from a resolved SPECIFICATION formula
    ///
    /// These constraints will be conjoined with negated liveness properties
    /// during liveness checking.
    pub fn set_fairness(&mut self, fairness: Vec<FairnessConstraint>) {
        self.liveness_cache.fairness = fairness;
    }

    /// Enable per-action coverage and the verbose `--coverage` REPORT.
    ///
    /// This both tracks per-action coverage (`collect`) and shows the report
    /// (`display`). Used by the `--coverage` flag.
    pub fn set_collect_coverage(&mut self, collect: bool) {
        self.coverage.display = collect;
        self.coverage.collect = collect;
        self.coverage.default_dead_action_tracking = false;
        self.coverage.native_fast_path_skipped = false;
    }

    /// Enable per-action coverage TRACKING without the verbose report.
    ///
    /// V2 vacuity gate (TRUST_VACUITY_GATE §1.A): the CLI turns this on by
    /// default so dead-action detection is available and the dead-action WARNING
    /// is default-on, while the full coverage report stays gated behind
    /// `--coverage`. Does not enable display.
    pub fn set_track_coverage(&mut self, track: bool) {
        self.coverage.collect = track;
        self.coverage.default_dead_action_tracking = false;
        self.coverage.native_fast_path_skipped = false;
    }

    /// Enable default-on per-action coverage TRACKING (the V2 vacuity gate),
    /// UNLESS the native-fused fast path is structurally viable for this run.
    ///
    /// Coverage collection makes `compiled_bfs_level_eligible()` reject the
    /// native fused level (it checks every invariant / state constraint natively
    /// but cannot emit per-action coverage), silently forcing the ~20x-slower
    /// interpreter for flat safety specs (e.g. EWD998Small: 4.4s native → 87s
    /// interpreter). Dead-action detection is a NON-correctness vacuity WARNING,
    /// so for a safety-only run that trust-cg will compile we drop the default
    /// tracking and keep the fast path. This never changes any reachable-state /
    /// invariant / constraint verdict.
    ///
    /// IMPORTANT PAIRING: when this skips coverage it also signals
    /// `coverage.native_fast_path_skipped`, and `maybe_initialize_trust_cg_cache_
    /// eager_or_defer` then compiles NATIVE EAGERLY (from the first state) instead
    /// of running the interpreter until the lazy threshold. That pairing is
    /// REQUIRED for soundness: with coverage off, the interpreter's coverage-off
    /// action-set path is not the authoritative enumerator, and the mixed
    /// interpreter→native hot-swap over-counts (observed 1,636,305 vs the correct
    /// 1,520,618); running the WHOLE exploration natively avoids it (native alone
    /// is verified verdict-identical).
    ///
    /// Explicit display, track-only, and coverage-guided modes always keep
    /// tracking on and take the per-action path, regardless of setter order.
    ///
    /// Must be called BEFORE `check()` (before `setup_actions_and_por` wires the
    /// enumerator).
    pub fn set_default_dead_action_coverage(&mut self) {
        if self.coverage.display
            || self.coverage.coverage_guided
            || (self.coverage.collect && !self.coverage.default_dead_action_tracking)
        {
            self.coverage.collect = true;
            self.coverage.default_dead_action_tracking = false;
            self.coverage.native_fast_path_skipped = false;
            return;
        }
        self.coverage.default_dead_action_tracking = true;
        let native_fast_path_viable = self.native_fast_path_coverage_skippable();
        self.coverage.collect = !native_fast_path_viable;
        // Record WHY collection is off so the post-compile AUTO gate
        // (`auto_select_post_compile_trust_cg_gate`) can re-enable it if native
        // turns out not to be beneficial (the perf rationale for the skip never
        // materializes when the run falls back to the interpreter).
        self.coverage.native_fast_path_skipped = native_fast_path_viable;
    }

    /// Whether default-on dead-action coverage should be skipped (and native
    /// compiled eagerly) because the native-fused fast path is structurally
    /// viable AND native is expected to pay off.
    ///
    /// Scope (all must hold):
    /// - no explicit `--coverage` report requested (`!coverage.display`);
    /// - safety-only (`config.properties.is_empty()` — liveness keeps coverage
    ///   and the interpreter);
    /// - trust-cg will structurally compile this run (`should_use_trust_cg`).
    ///
    /// # Why the former STATE-CONSTRAINT requirement was dropped
    ///
    /// This gate previously ALSO required a `CONSTRAINT` (or the
    /// `TY_RECORD_SET_NATIVE` opt-in). That requirement silently stranded every
    /// unconstrained flat safety spec on the interpreter: default dead-action
    /// coverage stayed on, which vetoes `compiled_bfs_level_eligible()`, so the
    /// AUTO-mode native fused level compiled but never executed. On DijkstraMutex
    /// Safety-4 (no constraint, 33.3M states) that meant a 64-action fused level
    /// compiled in ~70ms yet the whole exhaustive run executed on the interpreter
    /// at ~1/7th of native throughput — TIMING OUT past 400s where the native
    /// fused loop finishes in ~50s (TLC: 348s). The interpreter's per-successor
    /// `ArrayState` materialization (`DiffSuccessor::into_array_state` →
    /// `Arc::make_mut` → deep `Value::clone` of the compound function-valued state
    /// vars) is ~82% of its self-time; the native fused loop works on flat i64
    /// buffers and pays none of it.
    ///
    /// The constraint requirement was a proxy for "native pays off", added to
    /// protect UNBOUNDED specs whose per-state native work is expensive (the
    /// historical example was GameOfLife's 8-neighbour grid sums, which once
    /// regressed to a timeout when forced native). That proxy is now stale and
    /// overbroad:
    /// - GameOfLife's `Next` no longer builds a usable native fused level at all
    ///   (its nested per-cell scoring is not lowered), so the coverage-off path
    ///   simply runs the *interpreter* — and coverage-off is itself FASTER than
    ///   the coverage-on default (measured 45.5s vs 69.0s at N=4, exact 65536
    ///   states) because it drops per-action fired-bitset maintenance. So the
    ///   very spec the constraint gate protected is IMPROVED, not regressed.
    /// - When native DOES build a fused level for an unconstrained flat safety
    ///   spec (DijkstraMutex-class: cheap branching actions over a small flat
    ///   vector), eager native is a large win.
    ///
    /// Engine choice only — verdicts and the exact reachable-state count are
    /// unchanged either way (native-fused and coverage-off interpreter are both
    /// count-/verdict-validated against the coverage-on interpreter). The eager
    /// compile below (`maybe_initialize_trust_cg_cache_eager_or_defer`) is REQUIRED
    /// for count soundness of the coverage-off path: it compiles from the first
    /// state so the whole exploration runs on one engine (no mixed
    /// interpreter→native hot-swap). The ~70-400ms eager compile is well under
    /// TLC's JVM startup, so small specs do not regress against TLC.
    ///
    /// The `TY_RECORD_SET_NATIVE` opt-in remains an explicit way to force this on
    /// (it still implies `should_use_trust_cg`), but is no longer load-bearing for
    /// the constraint-free case since that is now the default.
    pub(in crate::check) fn native_fast_path_coverage_skippable(&self) -> bool {
        !self.coverage.display
            && self.config.properties.is_empty()
            && crate::check::model_checker::trust_cg_dispatch::should_use_trust_cg(
                self.trust_cg_structurally_vetoed(),
            )
    }

    /// Enable or disable coverage-guided exploration.
    ///
    /// When enabled, the BFS frontier uses a hybrid priority queue that
    /// directs exploration toward states exercising rare/uncovered actions.
    /// Implicitly enables coverage collection.
    ///
    /// `mix_ratio` controls how often to pick from the priority queue:
    /// - 8 (default): every 8th state is coverage-guided
    /// - 1: always prefer coverage-guided (pure priority)
    /// - 0: never use priority (pure BFS)
    pub fn set_coverage_guided(&mut self, enabled: bool, mix_ratio: usize) {
        self.coverage.coverage_guided = enabled;
        self.coverage.default_dead_action_tracking = false;
        self.coverage.native_fast_path_skipped = false;
        self.coverage.mix_ratio = mix_ratio;
        if enabled {
            // Coverage-guided implies coverage collection
            self.coverage.collect = true;
            self.coverage.tracker = Some(crate::coverage::guided::CoverageTracker::new());
        }
    }

    /// Check if coverage-guided exploration is enabled.
    #[allow(dead_code)] // Coverage-guided not yet wired into BFS loop
    pub(crate) fn is_coverage_guided(&self) -> bool {
        self.coverage.coverage_guided
    }

    /// Get a mutable reference to the coverage tracker (if active).
    #[allow(dead_code)] // Coverage-guided not yet wired into BFS loop
    pub(in crate::check::model_checker) fn coverage_tracker_mut(
        &mut self,
    ) -> Option<&mut crate::coverage::guided::CoverageTracker> {
        self.coverage.tracker.as_mut()
    }

    pub(in crate::check::model_checker) fn update_coverage_totals(&mut self) {
        if let Some(ref mut coverage) = self.stats.coverage {
            coverage.total_states = self.stats.states_found;
            coverage.total_transitions = self.stats.transitions;
        }
    }

    /// Enable or disable deadlock checking
    pub fn set_deadlock_check(&mut self, check: bool) {
        self.exploration.check_deadlock = check;
    }

    /// Force pure explicit-state BFS: skip the symbolic inductive-safety
    /// certificate shortcut so the run genuinely enumerates reachable states.
    /// Used by the certifying-verification eval oracle (an engine-diverse
    /// explicit re-check that must not be short-circuited by the symbolic path).
    pub fn set_force_explicit_bfs(&mut self, force: bool) {
        self.exploration.force_explicit_bfs = force;
    }

    /// Enable or disable automatic symmetry detection for this checker instance.
    ///
    /// Overrides the `TY_AUTO_SYMMETRY` environment variable. Tests and library
    /// embedders MUST use this instead of `std::env::set_var("TY_AUTO_SYMMETRY", ..)`:
    /// environment mutation is process-global and races with concurrently-running
    /// checkers (it silently enabled symmetry reduction in unrelated parallel
    /// tests, collapsing their expected state counts).
    pub fn set_auto_symmetry(&mut self, enabled: bool) {
        self.symmetry.auto_symmetry_override = Some(enabled);
    }

    /// Set whether stuttering transitions are allowed.
    ///
    /// When `true` (`[A]_v` form), the liveness checker adds self-loop edges to the
    /// behavior graph so that infinite stuttering is visible to SCC analysis. This is
    /// required for correct liveness verdicts on specs using `[][Next]_vars`.
    ///
    /// When `false` (`<<A>>_v` form), no stuttering self-loops are injected.
    pub fn set_stuttering_allowed(&mut self, allowed: bool) {
        self.exploration.stuttering_allowed = allowed;
    }

    /// Enable continue-on-error mode (like TLC's -continue flag)
    ///
    /// When enabled, exploration continues after finding an invariant or property
    /// violation. The first violation is recorded but exploration continues until
    /// the full state space is exhausted (or limits are reached). Final stats
    /// include all reachable states.
    ///
    /// This is useful for:
    /// - Getting stable state counts for error specs (comparable with TLC -continue)
    /// - Finding all reachable states even when some violate invariants
    pub fn set_continue_on_error(&mut self, continue_on_error: bool) {
        self.exploration.continue_on_error = continue_on_error;
    }

    /// Record an invariant violation, returning whether to stop exploration.
    ///
    /// In continue_on_error mode, this records the first violation and returns `false`
    /// to signal that exploration should continue. Otherwise returns `true` to stop.
    ///
    /// Note: This only applies to invariant violations, NOT deadlocks. Deadlocks always
    /// stop immediately, matching TLC's `-continue` behavior which only continues after
    /// invariant violations.
    pub(in crate::check::model_checker) fn record_invariant_violation(
        &mut self,
        invariant: String,
        state_fp: Fingerprint,
    ) -> bool {
        if self.exploration.continue_on_error {
            // Record first violation but continue exploring (TLC -continue mode)
            if self.exploration.first_violation.is_none()
                && self.exploration.first_action_property_violation.is_none()
            {
                self.exploration.first_violation = Some((invariant, state_fp));
            }
            false // Continue exploring
        } else {
            true // Stop immediately
        }
    }

    /// Record an action-level PROPERTY violation.
    pub(in crate::check::model_checker) fn record_action_property_violation(
        &mut self,
        property: String,
        state_fp: Fingerprint,
    ) -> bool {
        if self.exploration.continue_on_error {
            if self.exploration.first_violation.is_none()
                && self.exploration.first_action_property_violation.is_none()
            {
                self.exploration.first_action_property_violation = Some((property, state_fp));
            }
            false
        } else {
            true
        }
    }

    /// Set maximum number of states to explore
    ///
    /// When this limit is reached, model checking stops with `CheckResult::LimitReached`.
    /// This is useful for unbounded specifications that would otherwise run indefinitely.
    pub fn set_max_states(&mut self, limit: usize) {
        self.exploration.max_states = Some(limit);
    }

    /// Set maximum BFS depth to explore
    ///
    /// When this limit is reached, model checking stops with `CheckResult::LimitReached`.
    /// Depth 0 = initial states, depth 1 = first successors, etc.
    pub fn set_max_depth(&mut self, limit: usize) {
        self.exploration.max_depth = Some(limit);
    }

    /// Part of #3: Set a wall-clock deadline for BFS exploration.
    ///
    /// When the deadline is reached, the unified BFS worker loop stops cleanly
    /// (polled every `DEADLINE_CHECK_INTERVAL` states) with a partial
    /// `CheckResult::LimitReached { limit_type: Time }`. This is the time
    /// backstop that prevents an unbounded spec from running forever or growing
    /// the state space until the process OOMs. A partial result is sound: no
    /// invariant is claimed proven, so callers leave properties unresolved.
    pub fn set_deadline(&mut self, deadline: std::time::Instant) {
        self.exploration.deadline = Some(deadline);
    }

    /// Part of #3: Set a wall-clock time budget (relative to now) for BFS.
    ///
    /// Convenience wrapper over [`set_deadline`](Self::set_deadline) that
    /// converts a `Duration` budget into an absolute `Instant` deadline. A
    /// zero or saturating-overflow budget is treated as "no backstop".
    pub fn set_time_budget(&mut self, budget: std::time::Duration) {
        if let Some(deadline) = std::time::Instant::now().checked_add(budget) {
            self.exploration.deadline = Some(deadline);
        }
    }

    /// Part of #2751 Phase 2+3: Set a memory limit for threshold-triggered stop.
    ///
    /// When RSS reaches 85% of `limit_bytes`, exploration stops gracefully
    /// with a `LimitReached { limit_type: Memory }` result.
    ///
    /// Also auto-derives an internal memory hard cap at 75% of the limit
    /// unless an explicit internal limit was already set.
    pub fn set_memory_limit(&mut self, limit_bytes: usize) {
        self.exploration.memory_policy = Some(crate::memory::MemoryPolicy::new(limit_bytes));
        // Publish for process-global guards that cannot reach this policy
        // (the liveness graph store's growth guard honors the user's grant).
        crate::memory::publish_configured_memory_limit(limit_bytes);
        // Part of #4080: auto-derive internal memory cap from RSS limit.
        if self.exploration.internal_memory_limit.is_none() {
            self.exploration.internal_memory_limit =
                Some(internal_memory_limit_from_env_or_default(limit_bytes));
        }
    }

    /// Part of #4080: Set a hard cap on estimated internal memory (bytes).
    ///
    /// When the sum of all in-memory stores (FP set + seen + depths + queue)
    /// exceeds this limit, BFS stops gracefully. 0 = disabled.
    pub fn set_internal_memory_limit(&mut self, limit_bytes: usize) {
        self.exploration.internal_memory_limit = if limit_bytes > 0 {
            Some(limit_bytes)
        } else {
            None
        };
    }

    /// Part of #3282: Set a disk usage limit in bytes.
    ///
    /// When disk-backed storage (DiskFingerprintSet) would exceed this limit,
    /// exploration stops gracefully with `LimitReached { limit_type: Disk }`.
    pub fn set_disk_limit(&mut self, limit_bytes: usize) {
        self.exploration.disk_limit_bytes = Some(limit_bytes);
    }

    /// Apply all resource limits from a [`ResourceBudget`] in a single call.
    ///
    /// Only applies non-zero limits (zero = unlimited in the budget contract).
    /// `timeout_secs` is managed at the CLI/runner layer and is not applied here.
    pub fn apply_budget(&mut self, budget: &crate::resource_budget::ResourceBudget) {
        if budget.max_states > 0 {
            self.set_max_states(budget.max_states);
        }
        if budget.max_depth > 0 {
            self.set_max_depth(budget.max_depth);
        }
        if budget.memory_bytes > 0 {
            self.set_memory_limit(budget.memory_bytes);
        }
        if budget.disk_bytes > 0 {
            self.set_disk_limit(budget.disk_bytes);
        }
    }

    /// Set a shared verdict for portfolio racing.
    ///
    /// When set, the BFS worker loop checks the verdict every 4096 states
    /// and exits early if another lane has resolved. After BFS completes,
    /// the result is published to the verdict so other lanes can exit.
    ///
    /// Part of #3717.
    pub fn set_portfolio_verdict(&mut self, verdict: Arc<crate::shared_verdict::SharedVerdict>) {
        self.portfolio_verdict = Some(verdict);
    }

    /// Set cooperative state for fused BFS+symbolic mode (CDEMC).
    ///
    /// When set, the BFS worker loop:
    /// - Checks `invariants_proved` periodically and skips invariant evaluation
    /// - Samples frontier states at level boundaries for symbolic seeding
    /// - Checks the cooperative verdict for early exit
    ///
    /// Part of #3767, Epic #3762.
    #[cfg(feature = "ay")]
    pub(crate) fn set_cooperative_state(
        &mut self,
        state: Arc<crate::cooperative_state::SharedCooperativeState>,
    ) {
        self.cooperative = Some(state);
    }

    /// Set a progress callback to receive periodic updates during model checking
    ///
    /// The callback is called approximately every `interval` states (default: 1000).
    /// This is useful for long-running model checks to show progress to users.
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.hooks.progress_callback = Some(callback);
    }

    /// Set an init progress callback to receive a one-shot update after Init completes.
    ///
    /// This is useful for tool integrations that want to reflect the transition from
    /// initial-state enumeration to reachability exploration.
    pub fn set_init_progress_callback(&mut self, callback: InitProgressCallback) {
        self.hooks.init_progress_callback = Some(callback);
    }

    /// Sync TLC config to the evaluation context for TLCGet("config") support
    ///
    /// The `mode` parameter specifies the exploration mode:
    /// - "bfs" for exhaustive model checking
    /// - "generate" for simulation/random behavior generation
    pub(in crate::check::model_checker) fn sync_tlc_config(&mut self, mode: &str) {
        use crate::eval::TlcConfig;
        let config = TlcConfig::new(
            Arc::from(mode),
            self.exploration.max_depth.map_or(-1, |d| d as i64),
            self.exploration.check_deadlock,
        );
        self.ctx.set_tlc_config(config);
    }

    /// Set the collision detection mode for fingerprint-based state storage.
    ///
    /// - `None`: Zero overhead (default). Only theoretical collision probability
    ///   is reported.
    /// - `Sampling { interval }`: Store one full state per N admitted states and
    ///   verify fingerprint uniqueness. Catches collisions with probability 1/N.
    /// - `Full`: Store all states and verify every fingerprint is unique.
    ///   Catches all collisions but doubles memory usage.
    ///
    /// Must be called before `check()`. Results are reported in `CheckStats`.
    pub fn set_collision_check_mode(
        &mut self,
        mode: crate::collision_detection::CollisionCheckMode,
    ) {
        if mode.is_active() {
            self.collision_detector =
                Some(crate::collision_detection::CollisionDetector::new(mode));
        } else {
            self.collision_detector = None;
        }
    }

    /// Set how often progress is reported (in number of states)
    ///
    /// Default is 1000 states. Setting to 0 disables progress reporting.
    #[cfg(test)]
    pub(crate) fn set_progress_interval(&mut self, interval: usize) {
        self.hooks.progress_interval = interval;
    }

    /// Set whether to store full states for trace reconstruction
    ///
    /// When `store` is true (legacy mode):
    /// - Full states are forced as the primary trace-reconstruction store.
    /// - Faster trace reconstruction (no replay needed)
    /// - Also enables eager full-state access for liveness replay/diagnostics
    ///
    /// When `store` is false (default, #88):
    /// - Fingerprints and trace-file/replay data remain the primary storage path
    /// - Canonical payload witnesses may still be retained for fail-closed
    ///   collision-checked dedup admission
    /// - Counterexample traces reconstructed via temp trace file (unless disabled)
    /// - Liveness still works via BFS-time caching and replay (#3175)
    ///
    /// Default is false.
    pub fn set_store_states(&mut self, store: bool) {
        self.state_storage.store_full_states = store;
        self.refresh_liveness_cache_requirement();
    }

    /// Mark the trace as degraded (test-only helper).
    #[cfg(test)]
    pub fn set_trace_degraded(&mut self, degraded: bool) {
        self.trace.trace_degraded = degraded;
    }

    /// Set whether to auto-create a temp trace file for fingerprint-only mode
    ///
    /// When true (default): Creates a temporary trace file automatically if
    /// `store_full_states` is false and no explicit trace file is set.
    ///
    /// When false (--no-trace mode): No trace file is created, traces are
    /// completely unavailable for maximum memory efficiency.
    pub fn set_auto_create_trace_file(&mut self, auto_create: bool) {
        self.trace.auto_create_trace_file = auto_create;
    }

    /// Set the fingerprint storage backend.
    ///
    /// This allows using memory-mapped storage for large state spaces that
    /// exceed available RAM. Must be called before `check()`.
    ///
    /// Only used when `store_full_states` is false (no-trace mode).
    /// When `store_full_states` is true, full states are stored in a HashMap
    /// regardless of this setting.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use tla_check::{Config, FingerprintSet, FingerprintStorage, ModelChecker};
    /// use tla_core::{lower, parse_to_syntax_tree, FileId};
    ///
    /// let src = "---- MODULE Counter ----\n\
    /// VARIABLE x\n\
    /// Init == x = 0\n\
    /// Next == x' = x + 1 /\\ x < 1\n\
    /// ====";
    /// let tree = parse_to_syntax_tree(src);
    /// let lowered = lower(FileId(0), &tree);
    /// let module = lowered.module.expect("valid module");
    ///
    /// let config = Config::parse("INIT Init\nNEXT Next\n").expect("valid config");
    /// let mut checker = ModelChecker::new(&module, &config);
    /// checker.set_store_states(false); // Enable no-trace mode
    ///
    /// let storage = FingerprintStorage::mmap(10_000_000, None).expect("mmap storage");
    /// checker.set_fingerprint_storage(Arc::new(storage) as Arc<dyn FingerprintSet>);
    /// let _result = checker.check();
    /// ```
    pub fn set_fingerprint_storage(&mut self, storage: Arc<dyn FingerprintSet>) {
        self.refresh_liveness_successor_storage(storage.as_ref());
        self.state_storage.replace_seen_fps(storage);
    }

    /// Enable disk-based trace storage for large state space exploration.
    ///
    /// When enabled, the model checker writes (predecessor_loc, fingerprint) pairs
    /// to a disk file instead of keeping full states in memory. This significantly
    /// reduces memory usage while still enabling counterexample trace reconstruction.
    ///
    /// When a violation is found, the trace is reconstructed by:
    /// 1. Walking backward through the trace file to collect fingerprints
    /// 2. Replaying from the initial state, generating successors and matching
    ///    by fingerprint until the error state is reached
    ///
    /// # Arguments
    ///
    /// * `trace_file` - The trace file to use for storage
    ///
    /// # Notes
    ///
    /// - Trace file mode is incompatible with `store_full_states = true` (the two
    ///   approaches are mutually exclusive)
    /// - Trace reconstruction is slower than in-memory trace storage because states
    ///   must be regenerated from fingerprints
    /// - Part of #3175: Liveness checking now works with trace file mode
    pub fn set_trace_file(&mut self, trace_file: TraceFile) {
        self.trace.trace_file = Some(trace_file);
        // Trace file mode implies we don't store full states in memory
        self.state_storage.store_full_states = false;
        // Part of #3175: keep liveness active even with trace file.
        // BFS-time inline evaluation doesn't need stored full states.
        self.refresh_liveness_cache_requirement();
    }

    /// Set the trace location storage for fingerprint-to-offset mapping.
    ///
    /// By default, trace locations are stored in memory. For large state spaces,
    /// use `TraceLocationsStorage::mmap()` to scale beyond available RAM.
    ///
    /// # Arguments
    ///
    /// * `storage` - The trace location storage to use
    pub fn set_trace_locations_storage(&mut self, storage: TraceLocationsStorage) {
        self.trace.trace_locs = storage;
    }

    /// Enable checkpoint saving during model checking.
    ///
    /// Checkpoints are saved periodically to the specified directory, allowing
    /// interrupted model checking runs to be resumed.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory to save checkpoint files
    /// * `interval` - Interval between checkpoints
    pub fn set_checkpoint(&mut self, dir: PathBuf, interval: Duration) {
        self.checkpoint.dir = Some(dir);
        self.checkpoint.interval = interval;
    }

    /// Set the spec and config paths for checkpoint metadata.
    ///
    /// These paths are stored in checkpoint metadata to help verify on resume
    /// that the checkpoint matches the current spec/config. SHA-256 content
    /// hashes are computed eagerly so that file modifications between
    /// checkpoint save and resume are detected even when the path is unchanged.
    pub fn set_checkpoint_paths(&mut self, spec_path: Option<String>, config_path: Option<String>) {
        if let Some(spec_path) = spec_path.as_deref() {
            let path = std::path::Path::new(spec_path);
            self.ctx
                .set_input_base_dir(path.parent().map(std::path::Path::to_path_buf));
            self.checkpoint.spec_hash = crate::checkpoint::compute_file_hash(path);
        }
        if let Some(config_path) = config_path.as_deref() {
            self.checkpoint.config_hash =
                crate::checkpoint::compute_file_hash(std::path::Path::new(config_path));
        }
        self.checkpoint.spec_path = spec_path;
        self.checkpoint.config_path = config_path;
    }
}
