// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Model checker setup: construction, configuration, and top-level entry points.
//!
//! Heavy implementation details are split into private submodules:
//! - `setup_imports`: import graph resolution and operator/variable collection
//! - `setup_build`: constructor assembly (`new_with_extends_impl`)
//! - `setup_config`: setter/getter configuration surface

#[path = "setup_build.rs"]
mod setup_build;
#[path = "setup_config.rs"]
pub(crate) mod setup_config;
#[path = "setup_imports.rs"]
pub(crate) mod setup_imports;

// Re-import parent module items so child modules can access them via `super::`.
use super::debug;
use super::module_set_validation;
use super::{
    Arc, CapacityStatus, CheckError, CheckResult, CheckStats, CheckpointState, CompiledSpec,
    CoverageState, DebugDiagnostics, Duration, ExplorationControl, Expr, FairnessConstraint,
    Fingerprint, FingerprintSet, FxHashMap, InitProgressCallback, LivenessCacheState, LivenessMode,
    ModelChecker, Module, ModuleState, OperatorDef, PathBuf, PeriodicLivenessState, PorState,
    ProgressCallback, RuntimeHooksState, Spanned, StateStorage, SymmetryState, TraceFile,
    TraceLocationsStorage, TraceState,
};
use crate::eval::stack_safe;
use crate::Config;
use tla_core::ast::Unit;

impl<'a> ModelChecker<'a> {
    const TRACE_DEGRADED_WARNING: &'static str =
        "WARNING: Counterexample trace may be incomplete due to I/O errors during model checking.";

    /// Create a new model checker
    pub fn new(module: &'a Module, config: &'a Config) -> Self {
        Self::new_with_extends(module, &[], config)
    }

    /// Create a new model checker with additional loaded modules.
    ///
    /// Despite the historical name, `extended_modules` is **not** "EXTENDS-only".
    /// It must be a *loaded-module superset* for the whole run:
    ///
    /// - Include every non-stdlib module that may be referenced, via `EXTENDS` or `INSTANCE`
    ///   (including transitive and nested instance dependencies).
    /// - Put the modules that contribute to the **unqualified** operator namespace first, in a
    ///   TLC-shaped deterministic order (the `EXTENDS` closure and standalone `INSTANCE` imports).
    ///   Remaining loaded modules may follow in any deterministic order.
    ///
    /// Missing referenced non-stdlib modules are treated as a setup error.
    pub fn new_with_extends(
        module: &'a Module,
        extended_modules: &[&Module],
        config: &'a Config,
    ) -> Self {
        // Construction is a semantic-input boundary for callers that use the
        // checker type directly. Pointer-keyed evaluator caches must not carry
        // facts from an earlier module whose allocations have been recycled.
        crate::clear_thread_local_eval_caches();
        // Part of #758: module loading and stdlib/operator expansion can recurse deeply on some
        // specs. Guard construction so callers don't need a special thread stack size.
        stack_safe(|| Self::new_with_extends_impl(module, extended_modules, config))
    }

    /// Provide a source path for a given FileId to enable TLC-style line/col location rendering.
    ///
    /// If a path is not registered (or cannot be read), location rendering falls back to byte
    /// offsets (e.g., "bytes 0-0 of module M").
    pub fn register_file_path(&mut self, file_id: tla_core::FileId, path: std::path::PathBuf) {
        // Keep IO builtins (JsonDeserialize/ndJsonDeserialize) spec-relative.
        // We anchor to the root module's directory when available.
        if let Some(module_name) = self.module.file_id_to_name.get(&file_id) {
            if module_name == &self.module.root_name {
                self.ctx
                    .set_input_base_dir(path.parent().map(std::path::Path::to_path_buf));
            }
        }
        self.module.file_id_to_path.entry(file_id).or_insert(path);
    }

    /// Run the model checker
    pub fn check(&mut self) -> CheckResult {
        if let Some(result) =
            crate::check::runtime_config_validation_result(self.config, &self.stats)
        {
            return result;
        }
        // A checker may be constructed before another checker runs. Reassert
        // the TLS boundary here so delayed direct-API execution cannot inherit
        // semantic facts or certificate samples from that intervening run.
        crate::clear_thread_local_eval_caches();
        let _model_check_run_guard = crate::intern::ModelCheckRunGuard::begin();
        let _subset_profile_guard = crate::enumerate::subset_profile::RunGuard::begin();
        // Per-run diagnostics: reset this checker's own counters and install
        // them in a thread-local scope for the run. The legacy globals are NOT
        // reset here — concurrent runs in the same process own their counts.
        self.run_diagnostics.reset();
        let run_diagnostics = std::sync::Arc::clone(&self.run_diagnostics);
        // Part of #3351: enable TIR eval probe when TIR_EVAL_STATS=1 is set.
        let tir_stats = crate::tir_mode::tir_eval_stats_requested();
        if tir_stats {
            tla_eval::tir::enable_tir_eval_probe();
        }
        // Part of #758: Some evaluation/expansion paths can recurse deeply enough to overflow
        // constrained per-thread stacks (tests, embedded callers, small worker stacks). Guard the
        // top-level run to avoid a hard-abort stack overflow.
        //
        // The diagnostics scope is installed INSIDE the stack_safe closure
        // because stack_safe may run the body on a freshly spawned thread.
        let result = stack_safe(|| {
            let _diag_scope = crate::run_diagnostics::RunDiagnosticsScope::enter(run_diagnostics);
            self.check_impl()
        })
        .with_suppressed_guard_errors(self.run_diagnostics.take_suppressed_guard_errors());
        // V1 vacuity gate (TRUST_VACUITY_GATE §1.A): an over-constrained Init that
        // enumerates zero initial states surfaces as an InitCannotEnumerate error
        // on the direct sequential path. When the module declares a checkable
        // basis, that is the empty-reachable-set vacuity — remap FAILED → VACUOUS.
        let result = self.remap_empty_init_to_vacuous(result);
        self.emit_terminal_warnings();
        if tir_stats {
            Self::emit_tir_eval_stats();
        }
        // Debug-gated (TY_IMPLIED_VC_DEBUG=1) implied-action verdict-cache
        // hit/miss summary; no-op otherwise.
        crate::checker_ops::implied_verdict_cache_debug_summary();
        // Debug-gated exact sequential invariant TRUE-cache summary.
        self.log_invariant_verdict_cache_summary();
        // Debug-gated exact sequential state-constraint verdict-cache summary.
        self.log_state_constraint_verdict_cache_summary();
        // Guarantee the checker's DEFINITIVE terminal verdict is published to
        // the portfolio/cooperative race on EVERY exit path of `check()`.
        //
        // The two BFS-loop finalizers (engine.rs / compiled_bfs_loop.rs)
        // publish only on their fall-through exit; the compiled BFS loop's
        // violation path early-returns through
        // `finalize_terminal_result_with_storage`, which publishes NOTHING.
        // In fused/CDEMC mode that left the shared verdict UNRESOLVED after a
        // found violation, so no cooperative-termination machinery ever fired:
        // PDR/k-Induction kept solving to their own limits and the fused join
        // waited on them — the DieHard verdict-latency defect (BFS violation
        // in 0.03s, verdict reported after ~216s). Publishing here is
        // idempotent (`publish` is first-writer-wins and `Unknown` is a no-op)
        // and soundness-neutral: it states exactly the checker's own terminal
        // result.
        {
            let verdict = match &result {
                CheckResult::Success(_) => crate::shared_verdict::Verdict::Satisfied,
                CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. } => crate::shared_verdict::Verdict::Violated,
                _ => crate::shared_verdict::Verdict::Unknown,
            };
            if let Some(ref sv) = self.portfolio_verdict {
                sv.publish(verdict);
            }
            #[cfg(feature = "ay")]
            if let Some(ref coop) = self.cooperative {
                coop.verdict.publish(verdict);
            }
        }
        // Part of #4002 (follow-up): guarantee BFS completion is signalled on
        // EVERY exit path of `check()`. The two BFS-loop finalizers
        // (engine.rs / compiled_bfs_loop.rs) only set this when a BFS loop
        // actually ran; `check_impl` has many early returns (setup error,
        // MissingInit, assume-only model, prepare_bfs_common failure,
        // inductive-safety-certificate success) that never reach a loop. Without
        // this, in fused mode those early returns leave the cooperative BMC and
        // wavefront-compressor lanes spinning forever (they exit only on
        // `is_resolved() || is_bfs_complete()`), blocking the `thread::scope`
        // join — a hang. `mark_bfs_complete()` is an idempotent flag store, so
        // re-marking after a loop already set it is a no-op.
        #[cfg(feature = "ay")]
        if let Some(ref coop) = self.cooperative {
            coop.mark_bfs_complete();
        }
        result
    }

    /// V1 vacuity gate: remap an empty-Init ("no solutions") error to the
    /// distinct `VACUOUS` verdict when the module declares a checkable basis.
    ///
    /// Design: TRUST_VACUITY_GATE §1.A (V1).
    fn remap_empty_init_to_vacuous(&self, result: CheckResult) -> CheckResult {
        if let CheckResult::Error { error, stats, .. } = &result {
            if crate::adaptive::is_empty_init_no_solutions(error)
                && self.config.declares_checkable_basis()
            {
                return CheckResult::Vacuous {
                    reason: crate::vacuity::VacuityReason::EmptyReachableSet,
                    stats: stats.clone(),
                };
            }
        }
        result
    }

    /// Print TIR eval coverage stats to stderr. Part of #3351 Phase 3.
    fn emit_tir_eval_stats() {
        use std::io::Write as _;
        let snapshot = tla_eval::tir::tir_eval_probe_snapshot();
        if snapshot.is_empty() {
            let _ = writeln!(
                std::io::stderr().lock(),
                "[TIR_EVAL_STATS] No operators evaluated."
            );
            return;
        }
        let mut total_named = 0usize;
        let mut total_expr = 0usize;
        let _ = writeln!(
            std::io::stderr().lock(),
            "[TIR_EVAL_STATS] Operator coverage:"
        );
        for (name, counts) in &snapshot {
            let _ = writeln!(
                std::io::stderr().lock(),
                "  {name}: named_op_evals={}, expr_evals={} ({})",
                counts.named_op_evals,
                counts.expr_evals,
                if counts.expr_evals > 0 {
                    "TIR"
                } else {
                    "AST fallback"
                },
            );
            total_named += counts.named_op_evals;
            total_expr += counts.expr_evals;
        }
        let tir_ops = snapshot.values().filter(|c| c.expr_evals > 0).count();
        let ast_ops = snapshot.values().filter(|c| c.expr_evals == 0).count();
        let pct = if !snapshot.is_empty() {
            tir_ops as f64 / snapshot.len() as f64 * 100.0
        } else {
            0.0
        };
        let _ = writeln!(
            std::io::stderr().lock(),
            "[TIR_EVAL_STATS] Summary: {tir_ops}/{} operators via TIR ({pct:.1}%), \
             {ast_ops} AST fallback. Total evals: named={total_named}, expr={total_expr}.",
            snapshot.len(),
        );
    }

    pub(in crate::check::model_checker) fn emit_terminal_warnings(&self) {
        if self.trace.trace_degraded {
            use std::io::Write as _;

            let _ = writeln!(std::io::stderr().lock(), "{}", Self::TRACE_DEGRADED_WARNING);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    fn simple_checker() -> (Module, Config) {
        let src = r#"
---- MODULE SetupPathTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lowered = lower(FileId(0), &tree);
        let module = lowered.module.expect("lowered module");
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        (module, config)
    }

    #[test]
    fn register_file_path_sets_eval_input_base_dir_for_root_module() {
        let (module, config) = simple_checker();
        let mut checker = ModelChecker::new(&module, &config);

        let spec_path = std::path::PathBuf::from("/tmp/setup-path-test/Spec.tla");
        checker.register_file_path(FileId(0), spec_path.clone());

        assert_eq!(
            checker.ctx.input_base_dir(),
            spec_path.parent().map(std::path::Path::to_path_buf)
        );
    }

    #[test]
    fn set_checkpoint_paths_sets_eval_input_base_dir() {
        let (module, config) = simple_checker();
        let mut checker = ModelChecker::new(&module, &config);

        checker.set_checkpoint_paths(Some("/tmp/setup-path-test/Spec.tla".to_string()), None);

        assert_eq!(
            checker.ctx.input_base_dir(),
            Some(std::path::PathBuf::from("/tmp/setup-path-test"))
        );
    }

    #[test]
    fn set_checkpoint_accepts_duration() {
        let (module, config) = simple_checker();
        let mut checker = ModelChecker::new(&module, &config);
        let checkpoint_dir = std::path::PathBuf::from("/tmp/setup-path-test/checkpoint");
        let interval = Duration::from_secs(42);

        checker.set_checkpoint(checkpoint_dir.clone(), interval);

        assert_eq!(checker.checkpoint.dir, Some(checkpoint_dir));
        assert_eq!(checker.checkpoint.interval, interval);
    }
}
