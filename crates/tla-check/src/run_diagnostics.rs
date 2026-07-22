// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-run diagnostic counters (suppressed guard errors, implied-action
//! telemetry).
//!
//! These counters were previously process-global statics
//! (`guard_error_stats::SUPPRESSED_GUARD_EVAL_ERRORS`,
//! `checker_ops::invariants::IMPLIED_ACTION_*`). Because every
//! `ModelChecker::check()` reset them at run start and swapped them at run
//! end, CONCURRENT runs in one process (the norm under `cargo test`) stole
//! and zeroed each other's counts. Observed as flaky tests:
//! `test_resume_carries_suppressed_guard_errors` (count stolen/zeroed by a
//! concurrent run) and the implied-action telemetry assertions
//! (`implied_action_transition_checks == 0` after a concurrent
//! `reset_property_check_telemetry()`).
//!
//! The fix: each checker instance owns an `Arc<RunDiagnostics>`. Run entry
//! points install it in a thread-local for the duration of the run (and each
//! parallel worker thread installs the shared handle at thread start), so the
//! deep recording sites (`enumerate::guard_check`, `checker_ops::invariants`)
//! attribute counts to the run that produced them. Recording sites fall back
//! to the legacy global counters when no run scope is installed (e.g. unit
//! tests driving `enumerate` directly), preserving prior behavior there.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

/// Diagnostic counters scoped to a single model-checking run.
///
/// All counters are atomics because parallel BFS workers increment them
/// concurrently through the shared `Arc`.
#[derive(Default)]
pub(crate) struct RunDiagnostics {
    /// Action-level guard evaluation errors suppressed during guard pre-check.
    suppressed_guard_errors: AtomicUsize,
    /// Number of `[][A]_v` implied-action transition checks performed.
    implied_action_transition_checks: AtomicU64,
    /// Number of implied-action term evaluations performed.
    implied_action_term_evals: AtomicU64,
    /// Total wall time spent in implied-action checking (nanoseconds).
    implied_action_time_ns: AtomicU64,
}

impl RunDiagnostics {
    /// Reset all counters. Call at run start (a checker instance may run
    /// more than once, e.g. `check()` after a failed `check_with_resume()`).
    pub(crate) fn reset(&self) {
        self.suppressed_guard_errors.store(0, Relaxed);
        self.implied_action_transition_checks.store(0, Relaxed);
        self.implied_action_term_evals.store(0, Relaxed);
        self.implied_action_time_ns.store(0, Relaxed);
    }

    #[inline]
    pub(crate) fn record_suppressed_guard_error(&self) {
        self.suppressed_guard_errors.fetch_add(1, Relaxed);
    }

    #[inline]
    pub(crate) fn record_implied_action(&self, term_evals: u64, elapsed_ns: u64) {
        self.implied_action_transition_checks.fetch_add(1, Relaxed);
        self.implied_action_term_evals
            .fetch_add(term_evals, Relaxed);
        self.implied_action_time_ns.fetch_add(elapsed_ns, Relaxed);
    }

    /// Take the suppressed-guard-error count, resetting it to zero.
    pub(crate) fn take_suppressed_guard_errors(&self) -> usize {
        self.suppressed_guard_errors.swap(0, Relaxed)
    }

    /// Snapshot the implied-action telemetry for `CheckStats`.
    pub(crate) fn property_check_snapshot(&self) -> crate::check::PropertyCheckStats {
        crate::check::PropertyCheckStats {
            implied_action_transition_checks: self.implied_action_transition_checks.load(Relaxed),
            implied_action_term_evals: self.implied_action_term_evals.load(Relaxed),
            implied_action_time_ns: self.implied_action_time_ns.load(Relaxed),
        }
    }
}

thread_local! {
    /// The diagnostics handle for the run currently executing on this thread.
    static CURRENT_RUN_DIAGNOSTICS: RefCell<Option<Arc<RunDiagnostics>>> =
        const { RefCell::new(None) };
}

/// RAII guard installing a run's diagnostics handle on the current thread.
///
/// Install at every run-execution entry: sequential `check()` /
/// `check_with_resume()` bodies, the parallel coordinator, and each parallel
/// worker thread. Nested installs restore the previous handle on drop.
pub(crate) struct RunDiagnosticsScope {
    previous: Option<Arc<RunDiagnostics>>,
}

impl RunDiagnosticsScope {
    #[must_use]
    pub(crate) fn enter(diagnostics: Arc<RunDiagnostics>) -> Self {
        let previous = CURRENT_RUN_DIAGNOSTICS.with(|cell| cell.borrow_mut().replace(diagnostics));
        Self { previous }
    }
}

impl Drop for RunDiagnosticsScope {
    fn drop(&mut self) {
        CURRENT_RUN_DIAGNOSTICS.with(|cell| {
            *cell.borrow_mut() = self.previous.take();
        });
    }
}

/// Run `f` against the current run's diagnostics, if a run scope is installed
/// on this thread. Returns `None` (so callers can fall back to the legacy
/// process-global counters) otherwise.
#[inline]
pub(crate) fn with_current<R>(f: impl FnOnce(&RunDiagnostics) -> R) -> Option<R> {
    CURRENT_RUN_DIAGNOSTICS.with(|cell| cell.borrow().as_deref().map(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn scope_routes_records_to_installed_handle_and_restores_previous() {
        let outer = Arc::new(RunDiagnostics::default());
        let inner = Arc::new(RunDiagnostics::default());

        {
            let _outer_scope = RunDiagnosticsScope::enter(Arc::clone(&outer));
            with_current(|d| d.record_suppressed_guard_error()).expect("outer installed");
            {
                let _inner_scope = RunDiagnosticsScope::enter(Arc::clone(&inner));
                with_current(|d| d.record_suppressed_guard_error()).expect("inner installed");
                with_current(|d| d.record_implied_action(3, 17)).expect("inner installed");
            }
            // Previous handle restored after inner scope drops.
            with_current(|d| d.record_suppressed_guard_error()).expect("outer restored");
        }
        assert!(
            with_current(|_| ()).is_none(),
            "no scope installed after all guards drop"
        );

        assert_eq!(outer.take_suppressed_guard_errors(), 2);
        assert_eq!(inner.take_suppressed_guard_errors(), 1);
        let snapshot = inner.property_check_snapshot();
        assert_eq!(snapshot.implied_action_transition_checks, 1);
        assert_eq!(snapshot.implied_action_term_evals, 3);
        assert_eq!(snapshot.implied_action_time_ns, 17);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn reset_zeroes_all_counters() {
        let diags = RunDiagnostics::default();
        diags.record_suppressed_guard_error();
        diags.record_implied_action(2, 9);
        diags.reset();
        assert_eq!(diags.take_suppressed_guard_errors(), 0);
        let snapshot = diags.property_check_snapshot();
        assert_eq!(snapshot.implied_action_transition_checks, 0);
        assert_eq!(snapshot.implied_action_term_evals, 0);
        assert_eq!(snapshot.implied_action_time_ns, 0);
    }
}
