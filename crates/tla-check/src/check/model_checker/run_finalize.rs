// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Terminal-result finalization and post-BFS completion.
//!
//! Contains the storage-error precedence gate (`finalize_terminal_result`),
//! the post-BFS orchestration (`finish_check_after_bfs`), and the deferred-
//! violation / depth-limit finalization (`finalize_bfs`).

use super::{
    print_enum_profile_stats, print_eval_profile_stats, CheckResult, LimitType, ModelChecker,
};
use crate::storage::FingerprintSet;

impl<'a> ModelChecker<'a> {
    /// Enforce storage-error precedence over any semantic outcome.
    ///
    /// If fingerprint storage has recorded errors (disk I/O failures, overflow),
    /// the storage error supersedes the candidate result. This ensures we never
    /// report a semantic outcome (deadlock, violation, success) from a run that
    /// already lost fingerprint-set soundness.
    ///
    /// Part of #1785: all terminal BFS outcomes must pass through this gate.
    pub(in crate::check) fn finalize_terminal_result(&self, candidate: CheckResult) -> CheckResult {
        // Nested-set A6: per-monitor observe/escape summary on EVERY terminal
        // result (including the invariant-violation exit, which does not reach
        // `finalize_bfs`). `encoded` counts boards the monitor OBSERVED in-universe
        // on the hot dedup path (via either the batch encode hook or the
        // diff/streaming escape-only hook); `escapes`/`bailed` report fail-closed.
        for monitor in &self.nested_set_monitors {
            let var_name = self
                .ctx
                .var_registry()
                .name(crate::var_index::VarIndex::new(monitor.var_idx))
                .to_string();
            telemetry_eprintln!(
                "[nested-set] A6 monitor summary: var '{}' encoded={} escapes={} bailed={} \
                 (monitor observed every successor board; fail-closed on escape; {}-slot mask)",
                var_name,
                monitor.encoded_count,
                monitor.escape_count,
                monitor.bailed,
                monitor.slot_count(),
            );
        }
        let property_check = self.run_diagnostics.property_check_snapshot();
        let backend_capability_report = self.backend_capability_report_json();
        let engine_provenance = self.engine_provenance_json();
        let result = if let Some(storage_error) = self.check_fingerprint_storage_errors() {
            storage_error.with_property_check_stats(property_check)
        } else {
            candidate.with_property_check_stats(property_check)
        };
        result
            .with_backend_capability_report(backend_capability_report)
            .with_engine_provenance(engine_provenance)
    }

    /// Post-BFS finalization shared by both normal and resume paths.
    ///
    /// When `resume_mode` is `true`, liveness checking is rejected (not yet
    /// supported for checkpoint resume) instead of being run. All other
    /// post-BFS steps — profile stats, finalize_bfs, storage-error
    /// precedence, and POSTCONDITION — are identical.
    ///
    /// Part of #1812: eliminates structural duplication between
    /// `finish_check_after_bfs` and the former `finish_resume_after_bfs`.
    pub(in crate::check) fn finish_check_after_bfs(
        &mut self,
        limit_reached: Option<LimitType>,
        resume_mode: bool,
        active_payload_witness_bytes: usize,
    ) -> CheckResult {
        // Part of #2665: capture storage stats before any terminal return so
        // every CheckResult clone includes backend counters.
        self.stats.storage_stats = FingerprintSet::stats(&*self.state_storage.seen_fps);

        // Print detailed enumeration profile if enabled (has its own flag check)
        print_enum_profile_stats();
        // Part of #188: Print eval() call count for performance analysis
        print_eval_profile_stats();
        // Part of #4126: Report flat BFS adapter statistics.
        if let Some(ref adapter) = self.flat_bfs_adapter {
            adapter.report_stats();
        }
        self.log_jit_dispatch_summary();
        self.log_jit_verify_summary();
        // Finalize stats and check for early-exit conditions (depth limit, continue-on-error).
        // Part of #1785: route through finalize_terminal_result for storage error precedence.
        if let Some(result) = self.finalize_bfs(limit_reached) {
            return self.finalize_terminal_result(result);
        }

        // Storage overflow or disk lookup I/O failures make exploration incomplete.
        // Fail before liveness/postcondition so we never report a semantic outcome
        // from a run that already lost fingerprint-set soundness.
        if let Some(result) = self.check_fingerprint_storage_errors() {
            return result;
        }

        if resume_mode {
            // Part of #1793: fail loudly instead of returning Success when temporal
            // properties were not checked on a resumed run.
            //
            // Part of #1812: after #3175/#3205, fingerprint-only mode supports
            // liveness during fresh BFS runs, but resume still does not persist
            // the BFS-time liveness caches needed to replay temporal checks.
            // Resumed runs must therefore reject unchecked temporal properties in
            // both full-state and fingerprint-only modes.
            let has_liveness_properties = self.config.has_liveness_properties();
            let skip_liveness_flag = super::debug::skip_liveness();

            if has_liveness_properties && !skip_liveness_flag {
                return self.finalize_terminal_result(CheckResult::from_error(
                    crate::LivenessCheckError::Generic(format!(
                        "Checkpoint resume does not yet support PROPERTY/liveness checking \
                         in full-state or fingerprint-only mode. Temporal properties were NOT checked: {}. \
                         Re-run without --resume to verify liveness.",
                        self.config.properties.join(", ")
                    ))
                    .into(),
                    self.stats.clone(),
                ));
            }
        } else {
            // Debug-only container census for peak-RSS attribution (TY_MEM_CENSUS=1).
            self.emit_mem_census("post-bfs", active_payload_witness_bytes);
            // Check liveness properties (temporal formulas) after safety checking passes.
            let liveness_result = self.run_liveness_checking(false);
            self.emit_mem_census("post-liveness", active_payload_witness_bytes);
            // Part of #liveness-bfs-state-seed: free any speculative BFS-time
            // state seed the liveness phase did not consume.
            self.discard_unused_fp_only_seed();
            if let Some(result) = liveness_result {
                return self.finalize_terminal_result(result);
            }
        }

        // Evaluate POSTCONDITION after model checking completes (TLC parity).
        if let Some(result) = self.check_postcondition() {
            return self.finalize_terminal_result(result);
        }

        // V1 vacuity gate: a "Success" with zero reachable states but a declared
        // checkable basis (Init/Next/invariant/property) proved nothing.
        // Design: TRUST_VACUITY_GATE §1.A (V1).
        if self.stats.states_found == 0 && self.config.declares_checkable_basis() {
            return self.finalize_terminal_result(CheckResult::Vacuous {
                reason: crate::vacuity::VacuityReason::EmptyReachableSet,
                stats: self.stats.clone(),
            });
        }

        self.finalize_terminal_result(CheckResult::Success(self.stats.clone()))
    }

    // Debug successor helpers (debug_successor_flags, debug_print_state_line,
    // debug_log_successor_details) are in run_debug.rs.

    /// Finalize BFS exploration: update stats, check depth/state limits, and return deferred violations.
    pub(in crate::check) fn finalize_bfs(
        &mut self,
        limit_reached: Option<LimitType>,
    ) -> Option<CheckResult> {
        self.stats.states_found = self.states_count();
        self.update_coverage_totals();
        // V2/V3 vacuity gate (TRUST_VACUITY_GATE §1.A): harvest dead-action and
        // vacuous-invariant WARNINGs from the always-on coverage data, then drop
        // the verbose `CoverageStats` from `stats` unless `--coverage` asked for it.
        self.harvest_vacuity_warnings();
        self.stats.property_check = self.run_diagnostics.property_check_snapshot();

        // Populate symmetry reduction statistics into CheckStats.
        self.populate_symmetry_stats();

        // Part of #2841: copy FP collision counters to stats for CLI reporting.
        self.stats.fp_dedup_collisions = self.debug.seen_tlc_fp_dedup_collisions;
        self.stats.internal_fp_collisions = self.debug.internal_fp_collisions;

        // Copy collision detection stats to CheckStats.
        if let Some(ref detector) = self.collision_detector {
            self.stats.collision_check_mode = detector.mode();
            self.stats.collision_check_stats = detector.stats();
        }

        // Hybrid per-action dispatch (item 4 M0): end-of-run routing summary
        // (only prints when TY_HYBRID_FLAT_VIEW is set). `mismatch_fallback`
        // MUST be 0 — a nonzero value means the projection diverged from the
        // interpreter (fail-closed, but the loud alarm).
        self.report_hybrid_dispatch_summary();

        // Part of #3850: log tiered JIT summary at end of BFS when promotions occurred.
        // Part of #3910: detailed `--show-tiers` report when TY_SHOW_TIERS=1.
        if let Some(ref manager) = self.tier_manager {
            let summary = manager.tier_summary();
            if summary.tier1 > 0 || summary.tier2 > 0 {
                eprintln!("[jit] Tier summary: {summary}");
            }
            // Full tier report when `--show-tiers` / TY_SHOW_TIERS=1 is set.
            let res = {
                static SHOW_TIERS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *SHOW_TIERS.get_or_init(|| std::env::var("TY_SHOW_TIERS").is_ok_and(|v| v == "1"))
            };
            if res {
                eprint!("{}", self.format_tier_report());
            }
        }
        // Always print next-state dispatch counters when stats mode is on.
        let res = {
            static VM_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *VM_STATS.get_or_init(|| std::env::var("TY_BYTECODE_VM_STATS").as_deref() == Ok("1"))
        };
        if res {
            let ns = &self.next_state_dispatch;
            if ns.total > 0 {
                eprintln!(
                    "[jit] Next-state dispatch: hits={}, fallbacks={}, not_compiled={}, errors={}, total={}",
                    ns.jit_hit, ns.jit_fallback, ns.jit_not_compiled, ns.jit_error, ns.total,
                );
            }
        }
        self.value_action_vm.report_summary();

        // Part of #3993: Populate POR reduction statistics into CheckStats
        // and report when reduction was active.
        if let Some(ref independence) = self.por.independence {
            let stats = &self.por.stats;
            // Part of #3993 Phase 11: independence counts now always available.
            let indep_pairs = independence.count_independent_pairs();
            let total_pairs = independence.total_pairs();

            self.stats.por_reduction = crate::check::api::PorReductionStats {
                action_count: self.stats.detected_actions.len(),
                independent_pairs: indep_pairs,
                total_pairs,
                states_reduced: stats.reductions,
                states_processed: stats.total_states,
                actions_skipped: stats.actions_skipped,
                auto_detected: !self.config.por_enabled,
            };

            // Part of #3993 Phase 11: emit POR diagnostic summary with action names.
            if indep_pairs > 0 || stats.total_states > 0 {
                eprintln!(
                    "{}",
                    independence.diagnostic_summary(&self.stats.detected_actions)
                );
            }
            if stats.total_states > 0 {
                let pct = 100.0 * stats.reductions as f64 / stats.total_states as f64;
                eprintln!(
                    "POR: {}/{} states reduced ({:.1}%), {} actions skipped",
                    stats.reductions, stats.total_states, pct, stats.actions_skipped,
                );
            }
        }

        // If we stopped early due to a depth or state limit, report which one.
        if let Some(limit_type) = limit_reached {
            return Some(CheckResult::LimitReached {
                limit_type,
                stats: self.stats.clone(),
            });
        }

        // Part of #595: If we recorded a violation in continue_on_error mode, return it now
        // with final stats (full state space was explored).
        if let Some((property, fp)) = self.exploration.first_action_property_violation.take() {
            let trace = self.reconstruct_trace(fp);
            return Some(CheckResult::PropertyViolation {
                property,
                kind: crate::check::api::PropertyViolationKind::ActionLevel,
                trace,
                stats: self.stats.clone(),
            });
        }

        if let Some((invariant, fp)) = self.exploration.first_violation.take() {
            let trace = self.reconstruct_trace(fp);
            if self
                .compiled
                .state_property_violation_names
                .contains(&invariant)
            {
                return Some(CheckResult::PropertyViolation {
                    property: invariant,
                    kind: crate::check::api::PropertyViolationKind::StateLevel,
                    trace,
                    stats: self.stats.clone(),
                });
            }
            // VIOLATED-class kernel certificate (Feature 3): when the live counterexample is a
            // finite single-Int-variable trace in the embeddable fragment, ground-evaluate it into a
            // Clean-kernel-CHECKED finite-trace witness (Init(s0) ∧ ⋀ Next(sᵢ,sᵢ₊₁) ∧ ¬Safety(sₙ)).
            // Fail-closed: emit nothing extra when out of fragment — the InvariantViolation below is
            // returned exactly as before.
            #[cfg(feature = "clean-cic")]
            self.emit_violated_trace_kernel_cert(&invariant, &trace);
            return Some(CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: self.stats.clone(),
            });
        }

        None
    }

    /// Feature 3 — promote a LIVE safety counterexample to a kernel-checked VIOLATED-trace
    /// certificate. Extracts the spec's `Init`/`Next`/`Safety` ASTs and the concrete single-Int-
    /// variable value sequence from the reconstructed `trace`, then calls
    /// [`crate::cleancic::certify_violated_trace`] (ground-evaluate + `TypeChecker::check_type`).
    /// Strictly fail-closed: any precondition miss (multi-var, non-Int, missing op def, out of the
    /// Int comparison fragment, or a trace leg the kernel will not accept) is a silent no-op — the
    /// violation is still reported normally. The kernel re-check ([`verify_violated_trace`]) is the
    /// arbiter, so we never claim a cert the kernel did not actually accept.
    #[cfg(feature = "clean-cic")]
    pub(in crate::check::model_checker) fn emit_violated_trace_kernel_cert(
        &self,
        invariant: &str,
        trace: &crate::check::api::Trace,
    ) -> Option<()> {
        use tla_value::value::Value;
        // Need at least an initial state and the Init/Next operator names.
        if trace.states.is_empty() {
            return None;
        }
        let init_name = self.trace.cached_init_name.as_ref()?;
        let next_name = self.trace.cached_next_name.as_ref()?;
        let init = &self.module.op_defs.get(init_name)?.body.node;
        let next = &self.module.op_defs.get(next_name)?.body.node;
        // The violated invariant's definition is the Safety predicate.
        let safety = &self.module.op_defs.get(invariant)?.body.node;

        // Exactly one state variable, Int-valued in EVERY trace state → the single-var Int fragment.
        let var: String = {
            let mut names = self.module.vars.iter();
            let first = names.next()?;
            if names.next().is_some() {
                return None; // more than one state variable — out of fragment
            }
            first.to_string()
        };
        let to_i64 = |v: &Value| -> Option<i64> {
            match v {
                Value::SmallInt(n) => Some(*n),
                Value::Int(b) => b.to_string().parse::<i64>().ok(),
                _ => None,
            }
        };
        let mut vals: Vec<i64> = Vec::with_capacity(trace.states.len());
        for st in &trace.states {
            vals.push(to_i64(st.get(&var)?)?);
        }

        let bytes = crate::cleancic::certify_violated_trace(init, next, safety, &var, &vals)?;
        // Obligation-aware re-check (the arbiter): re-derive + re-run the kernel on the carried term.
        if !crate::cleancic::verify_violated_trace(init, next, safety, &var, &vals, &bytes) {
            return None; // fail-closed: never report a cert the re-check rejects
        }
        eprintln!(
            "[violated-trace] KERNEL-CERTIFIED: a {}-state counterexample to `{invariant}` is \
             witnessed by a {}-byte Clean-kernel-checked finite-trace CIC term \
             (Init(s0) ∧ ⋀ Next(sᵢ,sᵢ₊₁) ∧ ¬Safety(s{}))",
            vals.len(),
            bytes.len(),
            vals.len() - 1,
        );
        Some(())
    }

    /// V2/V3 vacuity gate: harvest dead-action (V2) and vacuously-true-invariant
    /// (V3) WARNINGs into `stats.vacuity_warnings`, then suppress the verbose
    /// `CoverageStats` from `stats` unless `--coverage` requested its display.
    ///
    /// Design: TRUST_VACUITY_GATE §1.A (V2/V3). These are default-on WARNINGs;
    /// `--strict-vacuity` (CLI) promotes them to exit 3.
    pub(in crate::check) fn harvest_vacuity_warnings(&mut self) {
        // V2 — never-enabled (dead) actions. Only meaningful when at least one
        // state was reached and actions were detected; an empty reachable set is
        // already covered by V1 and must not also spam dead-action warnings.
        if self.stats.states_found > 0 {
            if let Some(ref coverage) = self.stats.coverage {
                let dead: Vec<String> = coverage
                    .dead_actions()
                    .into_iter()
                    .map(str::to_string)
                    .collect();
                if !dead.is_empty() {
                    self.stats
                        .vacuity_warnings
                        .push(crate::vacuity::VacuityWarning::DeadActions(dead));
                }
            }
        }

        // V3 — vacuously-true invariants (the two sound static special-cases).
        for inv_name in &self.config.invariants {
            if let Some(def) = self.module.op_defs.get(inv_name) {
                if let Some(warning) =
                    crate::coverage::invariant_tension::classify_invariant(inv_name, &def.body)
                {
                    self.stats.vacuity_warnings.push(warning);
                }
            }
        }

        // Suppress the verbose coverage report unless the user asked for it.
        if !self.coverage.display {
            self.stats.coverage = None;
        }
    }
}

#[cfg(test)]
#[path = "run_finalize_tests.rs"]
mod run_finalize_tests;
