// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Global state reset and memory/disk policy helpers.
//!
//! Extracted from `lib.rs` to keep the crate root focused on module
//! declarations and re-exports. Part of #3608.

use crate::{enumerate, guard_error_stats, intern, liveness, memory, resource_budget, value};

/// Reset all global state (interning tables + eval caches) between model checking runs.
///
/// Clears six global interning tables and all thread-local eval caches:
///
/// 1. **GLOBAL_INTERNER** (tla-check/intern): Value interning for parallel BFS.
///    Grows proportional to unique values in the state space. Not capped mid-run
///    because clearing would invalidate `ValueHandle` references.
///
/// 2. **STRING_INTERN_TABLE** (tla-value): String deduplication for O(1) equality.
///    Capped at 10,000 entries mid-run, but residual entries persist after completion.
///
/// 3. **SET_INTERN_TABLE** (tla-value): Small set deduplication (≤8 elements).
///    Capped at 10,000 entries mid-run.
///
/// 4. **INT_FUNC_INTERN_TABLE** (tla-value): Integer function deduplication (≤8 elements).
///    Capped at 10,000 entries mid-run.
///
/// 5. **MODEL_VALUE_REGISTRY** (tla-value): Model value name → index mapping.
///    Bounded by u16::MAX but accumulates across runs without clearing.
///
/// 6. **GLOBAL_INTERNER** (tla-core/name_intern): Operator/variable name interning.
///    Grows proportional to spec size. Typically bounded.
///
/// Additionally clears all thread-local eval caches via `tla_eval::clear_for_run_reset()`:
/// - SUBST_CACHE, OP_RESULT_CACHE, ZERO_ARG_OP_CACHE, THUNK_DEP_CACHE (state caches)
///   (Fix #2462: constant entries formerly in ZERO_ARG_OP_CONST_CACHE now in ZERO_ARG_OP_CACHE)
/// - LITERAL_CACHE (permanent caches)
/// - CHAINED_REF_CACHE, MODULE_REF_SCOPE_CACHE (module ref caches)
/// - TLC_REGISTERS (TLCSet/TLCGet thread-local storage)
///
/// If a guarded parse/check transaction is active, this call leaves global
/// interners intact and only clears the caller thread's evaluation caches. If
/// the process is quiescent, it atomically excludes new transactions while
/// clearing global state. Call [`crate::enter_model_check_context`] before
/// parsing or lowering input that must remain valid through a concurrent reset.
///
/// # When to Call
///
/// - After `ModelChecker::check()` returns (in the `Drop` impl)
/// - After `ParallelChecker::check()` returns
/// - Before starting a new model checking run (library callers running multiple specs)
///
/// # Memory Impact
///
/// For large specs (millions of states), the value interner can hold millions of entries.
/// Without clearing, this memory is leaked between runs. The other tables are typically
/// small (<10K entries each) but should still be cleared for hygiene.
pub fn reset_global_state() {
    // Part of #4067: the sentinel is a production boundary, not test-only
    // hygiene. Atomically prove quiescence and prevent a new run from starting
    // while global NameIds/ValueHandles are invalidated. If another guarded
    // parse/check transaction is active, only this thread's pointer-keyed
    // caches may be cleared.
    if !intern::try_acquire_reset_lock() {
        clear_thread_local_eval_caches();
        return;
    }
    struct ResetLockGuard;
    impl Drop for ResetLockGuard {
        fn drop(&mut self) {
            intern::release_reset_lock();
        }
    }
    let _reset_lock = ResetLockGuard;
    reset_global_state_impl();
}

/// Clear only the per-run, thread-local evaluation caches — never any
/// process-global interner or registry.
///
/// This is the subset of [`reset_global_state`] that is safe to run
/// concurrently from multiple threads: every cache it touches is thread-local
/// (or otherwise has no cross-thread effect), so it cannot invalidate another
/// thread's live `NameId`/`ValueHandle`. It is exactly what `reset_global_state`
/// falls back to when a model-check run is active.
///
/// Use this (instead of [`reset_global_state`]) when you only need to honor the
/// pointer-keyed-cache run-boundary contract — e.g. re-evaluating an
/// independently-parsed AST whose freed nodes could otherwise alias a stale
/// cache entry. Crucially it never clears the global name interner, so it is
/// safe to call from parallel test threads that hold live `NameId`s (clearing
/// the name interner there causes index-out-of-bounds panics).
pub fn clear_thread_local_eval_caches() {
    tla_eval::clear_for_run_reset();
    tla_eval::clear_diagnostic_counters();
    enumerate::clear_expr_analysis_caches();
    enumerate::clear_enum_subst_cache();
    enumerate::clear_const_domain_cache();
    enumerate::clear_state_independent_branch_caches();
    crate::check::model_checker::invariants::clear_invariant_dependency_caches();
    crate::checker_ops::clear_view_projection_cache();
    #[cfg(feature = "clean-cic")]
    crate::cleancic::clear_run_scoped_certificate_samples();
    tla_eval::clear_let_def_interning();
    liveness::clear_enabled_cache();
    liveness::clear_subscript_value_cache();
    liveness::clear_leaf_result_cache();
    guard_error_stats::reset();
    crate::checker_ops::reset_property_check_telemetry();
}

/// Unconditional reset of all global state, including global interners and
/// registries. Used by tests that specifically test the reset mechanism.
///
/// Part of #4067: In test builds, tries to acquire the sentinel with a short
/// timeout (5s). If acquired, clears everything (including name interner)
/// safely while no new model check runs can start. If not acquired (concurrent
/// runs active), clears everything EXCEPT the name interner to avoid
/// index-out-of-bounds panics in concurrent tests. The name interner is the
/// only global table where clearing causes crashes (NameId index lookups into
/// a now-empty vec); other tables use content-based lookups that gracefully
/// return fresh entries on miss.
///
/// Callers that need the name interner cleared should call
/// `wait_for_no_active_runs()` BEFORE calling this function.
#[cfg(test)]
pub(crate) fn reset_global_state_unchecked() {
    // Try to acquire the sentinel with a short timeout.
    for _ in 0..5_000 {
        if intern::try_acquire_reset_lock() {
            reset_global_state_impl();
            intern::release_reset_lock();
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // Could not acquire sentinel — concurrent runs are active.
    // Clear everything EXCEPT the name interner (which would cause
    // index-out-of-bounds panics in concurrent tests).
    intern::clear_global_value_interner();
    value::clear_string_intern_table();
    value::clear_set_intern_table();
    value::clear_int_func_intern_table();
    value::clear_model_value_registry();
    // SKIP: tla_core::clear_global_name_interner() — unsafe with concurrent runs
    tla_eval::clear_for_run_reset();
    tla_eval::clear_diagnostic_counters();
    enumerate::clear_expr_analysis_caches();
    enumerate::clear_enum_subst_cache();
    enumerate::clear_const_domain_cache();
    enumerate::clear_state_independent_branch_caches();
    crate::check::model_checker::invariants::clear_invariant_dependency_caches();
    crate::checker_ops::clear_view_projection_cache();
    #[cfg(feature = "clean-cic")]
    crate::cleancic::clear_run_scoped_certificate_samples();
    tla_eval::clear_let_def_interning();
    liveness::clear_enabled_cache();
    liveness::clear_subscript_value_cache();
    liveness::clear_leaf_result_cache();
    tla_value::ParallelValueInternRunGuard::reset_for_recovery();
    guard_error_stats::reset();
    crate::checker_ops::reset_property_check_telemetry();
}

fn reset_global_state_impl() {
    // Clear value interner (tla-check: fingerprint -> Value table)
    intern::clear_global_value_interner();

    // Clear string intern table (tla-value: string dedup for O(1) equality)
    value::clear_string_intern_table();

    // Clear set intern table (tla-value: small set dedup)
    value::clear_set_intern_table();

    // Clear int-func intern table (tla-value: integer function dedup)
    value::clear_int_func_intern_table();

    // Clear model value registry (tla-value: model value name -> index)
    value::clear_model_value_registry();

    // Clear name interner (tla-core: operator/variable name dedup)
    tla_core::clear_global_name_interner();

    // Clear all eval caches (tla-eval: operator result cache, zero-arg cache,
    // literal cache, module ref caches, TLC registers).
    // Part of #1331: Prover finding — eval caches were not cleared between runs.
    // Part of #2413: Use semantic lifecycle API (RunReset) instead of generic clear_all.
    tla_eval::clear_for_run_reset();

    // Clear bytecode VM diagnostic counters separately — these are NOT cleared
    // by clear_for_run_reset() because liveness checking calls RunReset within
    // a single model checking run, and stats must survive that phase boundary.
    tla_eval::clear_diagnostic_counters();

    // Part of #3144: Clear enumerate-module thread-local caches.
    // These are keyed by raw AST node pointer (`usize`), so stale entries from a
    // previous AST could silently return incorrect results if memory is reused.
    enumerate::clear_expr_analysis_caches();
    enumerate::clear_enum_subst_cache();
    enumerate::clear_const_domain_cache();
    enumerate::clear_state_independent_branch_caches();
    crate::check::model_checker::invariants::clear_invariant_dependency_caches();
    crate::checker_ops::clear_view_projection_cache();
    #[cfg(feature = "clean-cic")]
    crate::cleancic::clear_run_scoped_certificate_samples();
    tla_eval::clear_let_def_interning();

    // Part of #3384: Clear liveness thread-local caches.
    // These are keyed by (Fingerprint, tag) and can contain stale results from
    // a previous spec's liveness checking that poison subsequent runs.
    liveness::clear_enabled_cache();
    liveness::clear_subscript_value_cache();
    liveness::clear_leaf_result_cache();

    // Part of #3334: Defense-in-depth: ensure value interner parallel mode is
    // deactivated. ParallelChecker uses ParallelValueInternRunGuard for normal
    // teardown, but if a previous run was killed (e.g., ntest::timeout) mid-check,
    // the interners stay frozen. reset_for_recovery() is idempotent and safe to
    // call even when no guard is active.
    tla_value::ParallelValueInternRunGuard::reset_for_recovery();

    // Reset suppressed guard error counter for clean per-run diagnostics.
    guard_error_stats::reset();
    crate::checker_ops::reset_property_check_telemetry();
}

/// Auto-detect total physical RAM and return a default memory limit in bytes.
///
/// Returns `Some(limit)` where `limit` is 90% of total physical RAM divided
/// by the number of concurrent `ty` instances, or `None` on unsupported
/// platforms. Intended for CLI auto-configuration so that memory monitoring
/// is active by default (Part of #2751).
pub fn memory_policy_system_default() -> Option<usize> {
    memory::MemoryPolicy::from_system_default().map(|p| p.limit_bytes())
}

/// Return auto-detected memory budget info: `(limit_bytes, total_bytes, instances)`.
///
/// Used by the CLI runner to log the detected budget on startup.
pub fn memory_policy_system_default_info() -> Option<(usize, usize, usize)> {
    memory::MemoryPolicy::system_default_info()
}

/// Log the auto-detected memory budget to stderr.
pub fn log_memory_budget(limit_bytes: usize, total_bytes: usize, instances: usize) {
    memory::log_memory_budget(limit_bytes, total_bytes, instances);
}

/// Auto-detect a safe disk limit based on available disk space on the CWD filesystem.
///
/// Returns `Some(limit)` where `limit` is 80% of available disk space in bytes,
/// or `None` on unsupported platforms. Intended for CLI auto-configuration
/// so that disk monitoring is active by default (Part of #3282).
pub fn disk_limit_system_default() -> Option<usize> {
    let bytes = resource_budget::ResourceBudget::safe_defaults().disk_bytes;
    if bytes > 0 {
        Some(bytes)
    } else {
        None
    }
}

#[cfg(test)]
#[path = "reset_tests.rs"]
mod tests;
