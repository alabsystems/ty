// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS model checker entry point: setup, validation, and dispatch to
//! full-state or no-trace mode.
//!
//! The BFS implementation is split across submodules:
//! - `run_prepare`: Pre-BFS preparation (constants, symmetry, VIEW, compilation)
//! - `run_checks`: Post-BFS validation (ASSUME, POSTCONDITION)
//! - `run_bfs_full`: Full-state mode BFS loop
//! - `run_bfs_notrace`: No-trace (fingerprint-only) mode BFS loop
//! - `run_helpers`: Shared BFS helpers (invariant checks, deadlock, checkpointing, profiling)
//! - `run_gen`: State generation (initial states, successors, pilot sampling)
//! - `run_monitoring`: Resource monitoring (progress, memory/disk pressure, state space estimation)

use super::super::api::{check_error_to_result, CheckResult, InitProgress};
#[cfg(debug_assertions)]
use super::debug::{
    debug_states, debug_successors_actions, debug_successors_actions_all_states, ty_debug,
};
use super::mc_struct::ModelChecker;
use crate::constants::bind_constants_from_config;
use crate::coverage::{detect_actions, CoverageStats};
use crate::storage::FingerprintSet;
use crate::trace_file::TraceFile;
use crate::{ConfigCheckError, EvalCheckError};
use std::sync::Arc;
use std::time::Instant;

pub(super) use super::run_monitoring::ProgressAction;

#[cfg(feature = "ay")]
#[inline]
const fn inductive_safety_certificate_route_allowed(
    force_explicit_bfs: bool,
    coverage_collect: bool,
    default_dead_action_tracking: bool,
) -> bool {
    !force_explicit_bfs && (!coverage_collect || default_dead_action_tracking)
}

impl ModelChecker<'_> {
    pub(in crate::check) fn report_init_progress(
        &mut self,
        states_generated: usize,
        distinct_states: usize,
    ) {
        self.stats.raw_initial_states_generated = states_generated;
        if let Some(ref callback) = self.hooks.init_progress_callback {
            let init = InitProgress {
                states_generated,
                distinct_states,
            };
            callback(&init);
        }
    }

    /// Attach the current fingerprint-storage counters to a terminal result.
    pub(in crate::check) fn with_current_storage_stats(
        &mut self,
        result: CheckResult,
    ) -> CheckResult {
        // Exact terminal liveness may have released only the physical
        // membership table. Preserve the authoritative pre-release snapshot in
        // results while observability paths such as the memory census continue
        // to query the backend's current (now smaller) residency directly.
        let storage_stats = if self.state_storage.retired_seen_fps_len.is_some() {
            self.stats.storage_stats
        } else {
            FingerprintSet::stats(&*self.state_storage.seen_fps)
        };
        self.stats.storage_stats = storage_stats;
        result.with_storage_stats(storage_stats)
    }

    /// Finalize terminal-result precedence, then attach current storage stats.
    pub(in crate::check) fn finalize_terminal_result_with_storage(
        &mut self,
        candidate: CheckResult,
    ) -> CheckResult {
        let result = self.finalize_terminal_result(candidate);
        self.with_current_storage_stats(result)
    }

    /// Auto-create temp trace storage when configured.
    ///
    /// Part of #3178: creates trace file in both full-state and fp-only modes.
    /// In full-state mode, the trace file replaces the in-memory `parents`
    /// HashMap for parent tracking, reducing per-state memory by 16 bytes.
    pub(super) fn maybe_auto_create_trace_file(&mut self) {
        if self.trace.auto_create_trace_file && self.trace.trace_file.is_none() {
            match TraceFile::create_temp() {
                Ok(tf) => {
                    self.trace.trace_file = Some(tf);
                }
                Err(e) => {
                    // Part of #1433: warn instead of silently swallowing.
                    // TLC treats trace file failure as fatal; TY degrades gracefully
                    // but must inform the user that error traces will be unavailable.
                    eprintln!("WARNING: failed to create temp trace file: {e}");
                    eprintln!("  Error traces will be unavailable for this run.");
                }
            }
        }
    }

    /// Reset checkpoint timer when periodic checkpointing is enabled.
    pub(super) fn initialize_checkpoint_timing(&mut self) {
        if self.checkpoint.dir.is_some() {
            self.checkpoint.last_time = Some(Instant::now());
        }
    }

    /// Detect actions in the Next relation and set up coverage tracking and POR state.
    pub(super) fn setup_actions_and_por(&mut self, next_name: &str) {
        // SOUNDNESS: the detected-action ASTs are wrapped in a single Arc and
        // shared (never deep-cloned) by every per-action successor path. The
        // unified enumerator's pointer-keyed caches require the call-site
        // nodes to be run-stable allocations; see `CoverageState::actions`.
        let actions: Arc<Vec<_>> = match self.module.op_defs.get(next_name) {
            Some(next_def) => Arc::new(detect_actions(next_def)),
            None => return,
        };
        self.stats.detected_actions = actions.iter().map(|a| a.name.clone()).collect();
        self.stats.detected_action_ids = actions.iter().map(|a| a.id.to_string()).collect();

        if self.coverage.collect {
            let mut coverage = CoverageStats::new();
            for action in actions.iter() {
                coverage.register_action(action);
            }
            self.coverage.actions = Arc::clone(&actions);
            self.stats.coverage = Some(coverage);
        } else {
            // Keep detected actions available for:
            // - `TY_DEBUG_STATES` action attribution
            // - Part of #3910: JIT per-action next-state dispatch
            // - Native fast-path coverage skip: the post-compile AUTO gate
            //   (`auto_select_post_compile_trust_cg_gate`) may re-enable
            //   default dead-action coverage if native turns out not to be
            //   beneficial; it must reuse these SAME run-stable Arc'd action
            //   allocations (see the SOUNDNESS note above).
            let keep_for_jit = self.jit_next_state_cache.is_some()
                || self.pending_jit_compilation.is_some()
                || self.action_bytecode.is_some()
                || self.coverage.native_fast_path_skipped;

            #[cfg(debug_assertions)]
            if keep_for_jit
                || debug_states()
                || debug_successors_actions()
                || debug_successors_actions_all_states()
            {
                self.coverage.actions = Arc::clone(&actions);
            } else {
                self.coverage.actions = Arc::new(Vec::new());
            }
            #[cfg(not(debug_assertions))]
            if keep_for_jit {
                self.coverage.actions = Arc::clone(&actions);
            } else {
                self.coverage.actions = Arc::new(Vec::new());
            }
            self.stats.coverage = None;
        }

        // Build POR analysis inputs when requested or auto-detected.
        //
        // Part of #3993: Auto-POR enables partial order reduction automatically
        // when the independence analysis finds independent action pairs. This
        // matches SPIN's behavior where POR is the default for concurrent specs.
        //
        // POR is disabled when liveness properties are present because the C3
        // BFS proviso is insufficient for liveness — it only guarantees no
        // exploration cycles in safety BFS, but liveness checking requires
        // the "ignoring proviso" (Peled 1996) or "strong proviso" which we
        // do not yet implement.
        let has_liveness = self.config.has_liveness_properties();
        if has_liveness && self.config.por_enabled {
            eprintln!(
                "POR: disabled — liveness properties present (C3 BFS proviso is insufficient for liveness)"
            );
        }

        // Auto-POR: when not explicitly enabled, check if auto-detection should
        // run the independence analysis. Config.auto_por overrides the env var;
        // when None, TY_AUTO_POR env var controls (default: enabled).
        let auto_por = crate::por::resolve_auto_por(self.config.auto_por);

        // Auto-detected POR is a heuristic and is mutually exclusive with the
        // native-fused flat-frontier fast path (admission rejects runs where
        // `por.independence` is set, since POR routes successors through the
        // per-action interpreter path). We cannot yet tell whether this run will
        // qualify for native-fused admission — that depends on the flat layout,
        // which is only inferred after init states are computed. So we set up
        // auto-POR optimistically here and release it later, post-layout, in
        // `maybe_release_auto_por_for_native_fused_admission` when the resolved
        // layout is admitted to the native-fused level (a 10x+ win that dwarfs
        // POR's interleaving pruning). Releasing auto-detected POR is always
        // sound — it only prunes provably-equivalent interleavings and never
        // changes the reachable-state set or any invariant result. An explicit
        // POR request (`config.por_enabled`) is never released here.

        // Auto-symmetry × auto-POR: when auto-detected symmetry engaged during
        // prepare (run_prepare), release auto-POR for this run. The combination
        // of orbit canonicalization and ample-set pruning is not validated, and
        // symmetry's orbit reduction (measured ~5-10x on symmetric corpora)
        // dwarfs auto-POR's pruning. Releasing auto-detected POR is always
        // sound (see above). An explicit POR request never coexists with
        // auto-symmetry: run_prepare hard-disables the auto path when
        // `config.por_enabled` is set, so explicit POR is unaffected here.
        let auto_symmetry_engaged = self.symmetry.auto_detected && !self.symmetry.perms.is_empty();

        // POR is disabled when trace invariants (--trace-inv) are present.
        // Trace invariants are history-dependent: they are evaluated by
        // TraceInvariantObserver only on states the exploration admits, and
        // the ample-set C2 visibility set is built solely from STATE
        // invariants — it cannot see which writes distinguish histories. An
        // ample set that is invisible to every state invariant can still
        // prune the only history violating a trace invariant, yielding a
        // false PASS. Disabling the reduction only restores full-Next
        // enumeration (never changes the reachable-state set), so this gate
        // is sound by construction.
        let has_trace_invariants = !self.config.trace_invariants.is_empty();
        if has_trace_invariants && self.config.por_enabled {
            eprintln!(
                "POR: disabled — trace invariants present (ample-set C2 visibility cannot see history-dependent properties)"
            );
        }

        let por_candidate = (self.config.por_enabled || auto_por)
            && !auto_symmetry_engaged
            && !actions.is_empty()
            && !has_liveness
            && !has_trace_invariants
            && actions.len() >= 2;

        if por_candidate {
            // POR dependency extraction needs the full action body including primed
            // assignments and UNCHANGED to compute read/write sets. The standard
            // expansion (allow_primed=false) skips primed operators, so a named
            // action operator whose body directly contains a prime survives in
            // `actions` (the enumeration/coverage decomposition) as an un-split
            // operator reference. `extract_detected_action_dependencies` expands
            // EACH coverage action's expression with primes and extracts ONE
            // read/write/unchanged set per action, unioning over any internal
            // disjuncts the expansion reveals (a sound over-approximation). The
            // resulting matrix is therefore indexed by `actions` — the SAME list
            // the successor enumerator feeds to `compute_ample_set` — by
            // construction, fixing the coverage-vs-with-primes index mismatch
            // that previously made C1/C2 read off the wrong rows and then
            // fail-closed-skipped POR entirely. (audit-2026-07 #11)
            let action_dependencies =
                crate::por::extract_detected_action_dependencies(&self.ctx, &actions);
            debug_assert_eq!(
                action_dependencies.len(),
                actions.len(),
                "independence matrix must be indexed by the enumeration decomposition"
            );
            let independence = crate::por::IndependenceMatrix::compute(&action_dependencies);

            let indep_pairs = independence.count_independent_pairs();
            let total_pairs = independence.total_pairs();

            // Auto-POR gate: if this was auto-detected (not explicitly requested),
            // only enable POR when there are actually independent pairs. No point
            // routing through the slower per-action path with zero reduction.
            if !self.config.por_enabled && indep_pairs == 0 {
                // No independent pairs found — skip POR setup entirely.
                // The actions are already set in coverage.actions if needed.
                #[cfg(debug_assertions)]
                if ty_debug() {
                    eprintln!(
                        "Auto-POR: {} actions analyzed, 0/{} independent pairs — POR not beneficial",
                        actions.len(),
                        total_pairs,
                    );
                }
                return;
            }

            // Report independence analysis results
            #[cfg(debug_assertions)]
            if ty_debug() {
                let source = if self.config.por_enabled {
                    "explicit"
                } else {
                    "auto"
                };
                if indep_pairs > 0 {
                    eprintln!(
                        "POR ({}): {} actions, {}/{} independent pairs ({:.1}%)",
                        source,
                        actions.len(),
                        indep_pairs,
                        total_pairs,
                        if total_pairs > 0 {
                            100.0 * indep_pairs as f64 / total_pairs as f64
                        } else {
                            0.0
                        }
                    );
                }
            }

            // Build visibility set from PROPERTY-promoted and config-level
            // invariant expressions with operator expansion.
            // Part of #3354 Slice 4 + #3449: both PROPERTY and config invariants
            // go through expand_operators so wrapper operators (e.g. Inv == TypeOK)
            // are inlined before dependency extraction.
            let mut visibility = crate::por::VisibilitySet::new();

            // PROPERTY-promoted invariants (from classification pipeline).
            for (_name, expr) in &self.compiled.eval_state_invariants {
                visibility.extend_from_expanded_expr(&self.ctx, expr);
            }

            // Config-level INVARIANT entries (name-only strings from .cfg).
            // Resolve to operator bodies and expand through wrapper operators.
            for inv_name in &self.config.invariants {
                if let Some(def) = self.ctx.get_op(inv_name) {
                    visibility.extend_from_expanded_expr(&self.ctx, &def.body);
                } else {
                    // Config invariant name not found in operator definitions.
                    // validate_config_ops() should have caught this earlier; fall
                    // back to treating all actions as visible to keep exploration sound.
                    eprintln!(
                        "POR: config invariant '{}' not found in op_defs, disabling reduction",
                        inv_name
                    );
                    visibility.mark_all_visible();
                    break;
                }
            }

            // Static no-benefit gate (auto-POR only): when the C2 visibility
            // analysis makes EVERY action visible (each action writes a
            // variable that the checked invariants read), `compute_ample_set`
            // can never select a proper subset — the ample set is always the
            // full enabled set and POR yields exactly 0 reduction while still
            // paying the slower per-action enumeration path on every state.
            // Skip auto-POR engagement entirely in that case. This is sound
            // by construction: it only disables a reduction that could never
            // fire, leaving exploration identical to whole-Next enumeration.
            // Explicit `--por` is honored (it still runs soundly with zero
            // reduction), matching the existing `indep_pairs == 0` gate which
            // is also auto-only.
            let all_actions_visible = action_dependencies
                .iter()
                .all(|deps| visibility.is_action_visible(deps));
            if !self.config.por_enabled && all_actions_visible {
                eprintln!(
                    "POR: skipped — all {} actions are visible to the checked invariants \
                     (ample set would always be the full enabled set; 0 reduction possible)",
                    actions.len()
                );
                return;
            }

            self.por.independence = Some(independence);
            self.por.visibility = visibility;

            // POR requires per-action enumeration - populate coverage_actions if not already set
            if self.coverage.actions.is_empty() {
                self.coverage.actions = actions;
                // Record that the actions exist only for POR so the
                // low-benefit auto-POR release can retire them and return
                // the run to the faster whole-Next path.
                self.por.actions_populated_for_por = true;
            }
        }
    }

    pub(super) fn check_impl(&mut self) -> CheckResult {
        if let Some(err) = self.module.setup_error.take() {
            return CheckResult::from_error(err, self.stats.clone());
        }

        // Sync TLC config for TLCGet("config") support (must happen before ASSUME checking)
        self.sync_tlc_config("bfs");

        // Validate init_name (check_impl-specific: resume path skips init)
        let init_name = match &self.config.init {
            Some(name) => name.clone(),
            None => {
                // Toolbox-generated "constant-expression evaluation" models often contain only
                // ASSUME statements and do not provide INIT/NEXT. Check for assume-only model
                // below after constant binding.
                if self.config.next.is_none()
                    && self.config.specification.is_none()
                    && self.module.vars.is_empty()
                    && self.config.invariants.is_empty()
                    && self.config.properties.is_empty()
                    && !self.module.assumes.is_empty()
                {
                    // Bind constants first so ASSUME expressions evaluate correctly
                    if let Err(e) = bind_constants_from_config(&mut self.ctx, self.config) {
                        // Part of #2356/#2777: Route through check_error_to_result so
                        // ExitRequested maps to LimitReached(Exit).
                        return check_error_to_result(EvalCheckError::Eval(e).into(), &self.stats);
                    }
                    // Check ASSUME statements
                    if let Some(result) = self.check_assumes() {
                        super::print_eval_profile_stats();
                        return result;
                    }
                    super::print_eval_profile_stats();
                    return CheckResult::Success(self.stats.clone());
                }
                return CheckResult::from_error(
                    ConfigCheckError::MissingInit.into(),
                    self.stats.clone(),
                );
            }
        };

        // Shared BFS setup: constant binding, symmetry, VIEW, next validation,
        // invariant compilation, operator expansion, action compilation
        let next_name = match self.prepare_bfs_common() {
            Ok(name) => name,
            Err(result) => return result,
        };

        // Cache init name for trace reconstruction from fingerprints
        self.trace.cached_init_name = Some(init_name.clone());

        // Check ASSUME statements after constant binding (done in prepare_bfs_common).
        // TLC checks all assumptions and stops if any evaluate to FALSE.
        // Part of #1031: Use eval_entry to enable operator result caching.
        if let Some(result) = self.check_assumes() {
            super::print_eval_profile_stats();
            return result;
        }

        // FIX B: SOUND inductive infinite-state SAFETY CERTIFICATE.
        //
        // Before BFS enumerates (which never terminates on an unbounded spec
        // like `x' = x + 1`), attempt a complete symbolic proof that the spec
        // is safe AND deadlock-free. This reuses the established "check_module
        // may return a symbolic verdict" precedent of the symbolic-deferral
        // hook directly below. It returns Success(Safe) ONLY on a complete
        // proof; on ANY failure it falls through to the unchanged BFS, so it is
        // verdict-preserving by construction (see `try_inductive_safety_
        // certificate` for the full soundness argument). ay-gated; on non-ay
        // builds this hook is absent and check_module is unchanged.
        // The certifying-verification eval oracle forces pure explicit BFS so it
        // is a genuine engine-diverse re-check; skip the symbolic shortcut then.
        // Explicit coverage and strict track-only coverage also require concrete
        // action attribution, which the certificate does not emit. The implicit
        // default dead-action diagnostic may yield because it is non-authoritative.
        #[cfg(feature = "ay")]
        if inductive_safety_certificate_route_allowed(
            self.exploration.force_explicit_bfs,
            self.coverage.collect,
            self.coverage.default_dead_action_tracking,
        ) {
            if let Some(result) = self.try_inductive_safety_certificate() {
                super::print_eval_profile_stats();
                return result;
            }
        }

        // Part of #3282: Pre-exploration state space estimation.
        // After constants are bound, extract constraints from Init and estimate
        // the initial state space. Warn if it exceeds configured limits.
        self.maybe_warn_state_space_estimate(&init_name);

        // Detect actions and initialize coverage/POR state.
        self.setup_actions_and_por(&next_name);

        // Auto-create temp trace file for fingerprint-only mode (#88)
        // This enables trace reconstruction while using 42x less memory than full-state storage.
        // Skip if user explicitly set a trace file, enabled full-state storage, or disabled auto-creation.
        self.maybe_auto_create_trace_file();

        // Part of #2955: Freeze name interner for lock-free lookup during BFS.
        tla_core::name_intern::freeze_interner();

        if self.state_storage.store_full_states {
            self.check_impl_full_state_mode(&init_name)
        } else {
            self.check_impl_no_trace_mode(&init_name)
        }
    }

    /// FIX B: attempt the SOUND inductive infinite-state safety certificate.
    ///
    /// Returns `Some(Success(Safe))` ONLY when the certificate discharges a
    /// COMPLETE proof: every configured invariant is inductive (directly or
    /// after sound interval strengthening) AND — when `config.check_deadlock` —
    /// Next is provably deadlock-free under the inductive invariant. On any
    /// failure, Unknown, or non-decomposable Next structure it returns `None`
    /// and the caller runs unchanged BFS. The capability is gated on a
    /// divergence trigger (Next accumulates arithmetic on a state var) so finite
    /// specs are not taxed with SMT solves.
    ///
    /// SOUNDNESS (proof obligations, all discharged in `ay_bmc`):
    ///   1. inductive J with J => Safety (J is a conjunction INCLUDING Safety);
    ///   2. J => Enabled(Next) (deadlock-freedom) via guard extraction.
    /// J inductive + J => Safety proves Safety in every reachable state; J =>
    /// Enabled(Next) proves no reachable deadlock. Only with BOTH do we return
    /// Safe. Any Unknown/error => fall through (never Unknown => Safe).
    #[cfg(feature = "ay")]
    fn try_inductive_safety_certificate(&mut self) -> Option<CheckResult> {
        use crate::ay_bmc::{try_inductive_safety_certificate, InductiveSafetyCertificate};

        match try_inductive_safety_certificate(
            &self.ctx,
            self.config,
            &self.module.vars,
            self.config.check_deadlock,
        ) {
            InductiveSafetyCertificate::Safe => {
                eprintln!(
                    "[BFS] inductive-safety certificate proven (safe + deadlock-free) — \
                     skipping unbounded BFS enumeration"
                );
                // Proof certificate: no explicit states were enumerated, so the
                // representative stats carry states_found = 0. This is a proof,
                // not an exploration; consumers that read states_found on a spec
                // reaching this path see the certificate's zero count.
                Some(CheckResult::Success(self.stats.clone()))
            }
            InductiveSafetyCertificate::FallThrough => None,
        }
    }
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;
