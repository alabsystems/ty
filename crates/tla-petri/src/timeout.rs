// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Timeout handling for MCC cooperative time confinement.
//!
//! MCC sets `BK_TIME_CONFINEMENT` as the wall-clock budget in seconds.
//! We leave a safety margin for output formatting and cleanup.

use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Safety margin in seconds subtracted from the time budget.
const SAFETY_MARGIN_SECS: u64 = 5;

/// Stack size for the examination worker thread spawned by
/// [`run_with_hard_deadline`] (256 MiB).
///
/// Every examination's main work runs on this single scoped worker. A scoped
/// thread otherwise inherits the platform default (~2 MiB on macOS/Linux), far
/// below the main thread's ~8 MiB. The preprocessing / analysis pipeline
/// recurses with depth that scales with the net: on wide nets (e.g.
/// NoC3x3-PT-8A — 317 places, 4293 transitions) it descends thousands of frames
/// deep (observed ~3285), overflowing a 2 MiB stack and aborting the WHOLE
/// process with SIGABRT instead of declining fail-closed. The recursion is
/// net-bounded (it terminates with a clean `CANNOT_COMPUTE` given enough stack),
/// so we give the worker a generous stack well above the deepest observed need.
/// The reservation is virtual (lazily backed) — no real memory cost until
/// touched. Fail-closed invariant: the examination worker must never abort the
/// process by overflowing its stack.
const EXAMINATION_WORKER_STACK_BYTES: usize = 256 * 1024 * 1024;

/// Margin in seconds subtracted from the time budget for the HARD watchdog
/// fire point. This is smaller than [`SAFETY_MARGIN_SECS`] on purpose: the
/// hard watchdog fires `SAFETY_MARGIN_SECS - HARD_MARGIN_SECS` seconds AFTER
/// the cooperative deadline, giving a cooperative finisher ample time to emit
/// its result before the watchdog ever fires (see [`run_with_hard_deadline`]).
///
/// BUDGET ARITHMETIC (the absolute-kill bound). The watchdog does ONE extra
/// short `recv_timeout(GRACE_SECS)` after the first hard timeout (see
/// [`run_with_hard_deadline`]), so the absolute latest the process is forcibly
/// killed is `hard + GRACE = (budget - HARD_MARGIN_SECS) + GRACE_SECS`. We pick
/// `HARD_MARGIN_SECS = 3` and `GRACE_SECS = 2` so:
///   - hard fire point    = budget - 3s  (strictly AFTER the cooperative
///     deadline at budget - 5s, by 2s — preserving the
///     no-premature-fire gap), and
///   - absolute kill      = (budget - 3) + 2 = budget - 1s  (<= budget — the
///     hard kill always lands inside the MCC wall-clock
///     budget, before the infrastructure SIGKILLs us).
const HARD_MARGIN_SECS: u64 = 3;

/// Grace window: after the FIRST hard `recv_timeout` returns `Timeout`, the
/// watchdog waits ONE more `recv_timeout(GRACE_SECS)` before declaring the
/// worker a runaway. This lets a worker that is finishing right at the hard
/// deadline (e.g. Philosophers-PT-000050 flushing its last verdicts ~1s past
/// the fire point) complete and send — yielding [`RunOutcome::Completed`] with
/// the real output, no fallback. Sized (with [`HARD_MARGIN_SECS`]) so
/// `hard + GRACE_SECS <= budget`; see the arithmetic on [`HARD_MARGIN_SECS`].
const GRACE_SECS: u64 = 2;

/// Compute the exploration deadline from CLI timeout or MCC environment.
///
/// Priority: CLI `--timeout` flag > `BK_TIME_CONFINEMENT` env var > None.
/// Subtracts a safety margin to ensure output is written before the
/// MCC infrastructure kills the process.
#[must_use]
pub fn compute_deadline(cli_timeout_secs: Option<u64>) -> Option<Instant> {
    let budget = cli_timeout_secs.or_else(|| {
        std::env::var("BK_TIME_CONFINEMENT")
            .ok()?
            .parse::<u64>()
            .ok()
    });

    budget.map(|secs| {
        let adjusted = secs.saturating_sub(SAFETY_MARGIN_SECS);
        Instant::now() + Duration::from_secs(adjusted)
    })
}

/// Compute the HARD watchdog duration (from now) for the in-process examination.
///
/// Priority mirrors [`compute_deadline`]: CLI `--timeout` flag >
/// `BK_TIME_CONFINEMENT` env var > None. Returns the [`Duration`] from NOW
/// until the HARD fire point = `budget - HARD_MARGIN_SECS` (saturating to a
/// small floor for tiny budgets), or `None` when no budget is set.
///
/// The hard fire point is `SAFETY_MARGIN_SECS - HARD_MARGIN_SECS` seconds
/// (2s with the current constants) AFTER the cooperative deadline returned by
/// [`compute_deadline`]. That gap, together with the emit-once gate in
/// `cli.rs`'s `on_timeout` (which suppresses the fallback once any line has been
/// flushed) and the grace recv (see [`run_with_hard_deadline`]), is the
/// no-double-emit invariant: a worker that respects the cooperative deadline
/// finishes and emits its result ~2s before the watchdog could ever fire.
#[must_use]
pub fn compute_hard_deadline(cli_timeout_secs: Option<u64>) -> Option<Duration> {
    let budget = cli_timeout_secs.or_else(|| {
        std::env::var("BK_TIME_CONFINEMENT")
            .ok()?
            .parse::<u64>()
            .ok()
    });

    budget.map(|secs| Duration::from_secs(secs.saturating_sub(HARD_MARGIN_SECS)))
}

/// Outcome of running a worker under the hard wall-clock watchdog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    /// The worker finished within the hard deadline.
    Completed,
    /// The worker did not finish within the hard deadline (genuine runaway).
    /// The worker thread is left running; the caller is expected to terminate
    /// the process (e.g. `std::process::exit`) after emitting a fail-closed
    /// CANNOT_COMPUTE.
    TimedOut,
}

/// Run `run` under a HARD wall-clock watchdog of `hard` duration.
///
/// Returns [`RunOutcome::Completed`] if `run` finishes within `hard`. If `run`
/// does not finish in time, `on_timeout` is invoked on the watchdog (main)
/// thread and, if it returns, the function yields [`RunOutcome::TimedOut`].
///
/// `on_timeout` is the caller-supplied runaway action. In production it emits a
/// fail-closed `CANNOT_COMPUTE` and calls `std::process::exit(0)` — and because
/// `process::exit` terminates the process, control never returns from
/// `on_timeout`, so the scope's join of the (still-running, genuinely stuck)
/// worker is never reached. Keeping `process::exit` in the injected closure
/// rather than in this function is what makes the function unit-testable: a
/// test passes an `on_timeout` that merely records the timeout, and its worker
/// is bounded so the scope join completes normally.
///
/// Implemented with [`std::thread::scope`] so the worker can BORROW non-`'static`
/// locals (e.g. `&model`, `&config`). The worker calls `run()` then signals
/// completion on a channel; the watchdog thread waits with `recv_timeout(hard)`:
///
/// - `Ok(())`        => the worker completed => [`RunOutcome::Completed`].
/// - `Err(Timeout)`  => the hard deadline elapsed. Before declaring a runaway we
///   do ONE more short `recv_timeout(GRACE_SECS)`: a worker finishing right at
///   the deadline (flushing its last verdicts) gets to complete and send,
///   yielding [`RunOutcome::Completed`] (its real output stands, no fallback).
///   Only if THAT grace recv ALSO times out is the worker genuinely stuck =>
///   call `on_timeout` (which, in production, never returns because it
///   `process::exit`s); if it does return, yield [`RunOutcome::TimedOut`].
/// - `Err(Disconnected)` => the worker panicked and dropped the sender without
///   sending; we return from the scope closure WITHOUT special handling so
///   `thread::scope` re-raises the panic on join. This preserves today's
///   panic-aborts behavior — we never swallow a panic into a verdict.
///
/// NOTE: `thread::scope` joins all spawned threads on scope exit, so this
/// function would block forever on a genuinely-stuck worker if it tried to
/// *return* `TimedOut` without first letting `on_timeout` terminate the
/// process. That is exactly why the production runaway-termination lives in
/// `on_timeout` (via `process::exit`): it fires BEFORE any join.
pub fn run_with_hard_deadline(
    hard: Duration,
    run: impl FnOnce() + Send,
    on_timeout: impl FnOnce() + Send,
) -> RunOutcome {
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel::<()>();
        std::thread::Builder::new()
            .name("ty-examination-worker".into())
            .stack_size(EXAMINATION_WORKER_STACK_BYTES)
            .spawn_scoped(scope, move || {
                run();
                // Best-effort: the watchdog thread may have already moved on
                // (timeout). A failed send simply means nobody is listening.
                let _ = tx.send(());
            })
            .expect("spawn examination worker thread");
        match rx.recv_timeout(hard) {
            Ok(()) => RunOutcome::Completed,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // The hard deadline elapsed. Give the worker ONE short grace
                // window to finish: a worker flushing its last verdicts right at
                // the deadline (e.g. Philosophers-PT-000050 ~1s past the fire
                // point) completes and sends here, so we treat it as Completed —
                // its real output stands, no fallback. `hard + GRACE_SECS` is
                // sized to stay within the MCC budget (see HARD_MARGIN_SECS).
                match rx.recv_timeout(Duration::from_secs(GRACE_SECS)) {
                    Ok(()) => RunOutcome::Completed,
                    Err(mpsc::RecvTimeoutError::Disconnected) => RunOutcome::Completed,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // Genuine runaway: stuck through both the hard deadline
                        // and the grace window. Run the caller's action BEFORE
                        // the scope join (which would otherwise block forever on
                        // the stuck worker). In production `on_timeout` calls
                        // `process::exit` and never returns, so the join below is
                        // never reached.
                        on_timeout();
                        RunOutcome::TimedOut
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Worker panicked. Return without special handling; the scope's
                // join will re-raise the panic, preserving abort-on-panic.
                RunOutcome::Completed
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize tests that mutate BK_TIME_CONFINEMENT to prevent env var races.
    /// Rust runs tests in parallel; concurrent set/remove of the same env var
    /// causes non-deterministic failures. Delegates to the single crate-wide env
    /// lock so these tests also serialize against env mutators in other modules.
    use crate::env_test_lock;

    #[test]
    fn test_compute_deadline_cli_timeout_returns_some() {
        let deadline = compute_deadline(Some(60));
        assert!(deadline.is_some());
        // Deadline should be in the future (60 - 5 = 55 seconds from now)
        let remaining = deadline.unwrap().duration_since(Instant::now());
        // Allow 2s tolerance for test execution time
        assert!(remaining.as_secs() >= 50);
        assert!(remaining.as_secs() <= 55);
    }

    #[test]
    fn test_compute_deadline_none_without_env() {
        let _guard = env_test_lock();
        let prev = std::env::var("BK_TIME_CONFINEMENT").ok();
        crate::env_guard::remove_var("BK_TIME_CONFINEMENT");
        let deadline = compute_deadline(None);
        assert!(deadline.is_none());
        if let Some(val) = prev {
            crate::env_guard::set_var("BK_TIME_CONFINEMENT", val);
        }
    }

    #[test]
    fn test_compute_deadline_safety_margin_subtracted() {
        // With a 10s timeout, deadline should be ~5s away (10 - SAFETY_MARGIN_SECS)
        let deadline = compute_deadline(Some(10));
        let remaining = deadline.unwrap().duration_since(Instant::now());
        assert!(remaining.as_secs() <= 5);
    }

    #[test]
    fn test_compute_deadline_small_timeout_saturates_to_zero() {
        // Timeout smaller than safety margin should not underflow
        let deadline = compute_deadline(Some(2));
        assert!(deadline.is_some());
        // 2 - 5 saturates to 0, so deadline is essentially now
    }

    #[test]
    fn test_compute_deadline_cli_overrides_env() {
        let _guard = env_test_lock();
        let prev = std::env::var("BK_TIME_CONFINEMENT").ok();
        crate::env_guard::set_var("BK_TIME_CONFINEMENT", "3600");
        // CLI says 20, env says 3600 — CLI wins
        let deadline = compute_deadline(Some(20));
        let remaining = deadline.unwrap().duration_since(Instant::now());
        // Should be ~15s (20-5), not ~3595s (3600-5)
        assert!(remaining.as_secs() <= 15);
        if let Some(val) = prev {
            crate::env_guard::set_var("BK_TIME_CONFINEMENT", val);
        } else {
            crate::env_guard::remove_var("BK_TIME_CONFINEMENT");
        }
    }

    // -- compute_hard_deadline -------------------------------------------------

    #[test]
    fn test_compute_hard_deadline_cli_timeout_returns_some() {
        // 60s budget => hard duration of 60 - HARD_MARGIN_SECS = 57s.
        let hard = compute_hard_deadline(Some(60));
        assert_eq!(hard, Some(Duration::from_secs(57)));
    }

    #[test]
    fn test_compute_hard_deadline_none_without_env() {
        let _guard = env_test_lock();
        let prev = std::env::var("BK_TIME_CONFINEMENT").ok();
        crate::env_guard::remove_var("BK_TIME_CONFINEMENT");
        assert!(compute_hard_deadline(None).is_none());
        if let Some(val) = prev {
            crate::env_guard::set_var("BK_TIME_CONFINEMENT", val);
        }
    }

    #[test]
    fn test_compute_hard_deadline_reads_env_when_no_cli() {
        let _guard = env_test_lock();
        let prev = std::env::var("BK_TIME_CONFINEMENT").ok();
        crate::env_guard::set_var("BK_TIME_CONFINEMENT", "30");
        assert_eq!(compute_hard_deadline(None), Some(Duration::from_secs(27)));
        if let Some(val) = prev {
            crate::env_guard::set_var("BK_TIME_CONFINEMENT", val);
        } else {
            crate::env_guard::remove_var("BK_TIME_CONFINEMENT");
        }
    }

    #[test]
    fn test_compute_hard_deadline_cli_overrides_env() {
        let _guard = env_test_lock();
        let prev = std::env::var("BK_TIME_CONFINEMENT").ok();
        crate::env_guard::set_var("BK_TIME_CONFINEMENT", "3600");
        // CLI says 20, env says 3600 — CLI wins => 20 - 3 = 17s.
        assert_eq!(
            compute_hard_deadline(Some(20)),
            Some(Duration::from_secs(17))
        );
        if let Some(val) = prev {
            crate::env_guard::set_var("BK_TIME_CONFINEMENT", val);
        } else {
            crate::env_guard::remove_var("BK_TIME_CONFINEMENT");
        }
    }

    #[test]
    fn test_compute_hard_deadline_small_timeout_saturates() {
        // Budget smaller than HARD_MARGIN_SECS must not underflow.
        assert_eq!(compute_hard_deadline(Some(1)), Some(Duration::from_secs(0)));
        assert_eq!(compute_hard_deadline(Some(0)), Some(Duration::from_secs(0)));
    }

    #[test]
    fn test_hard_deadline_is_after_cooperative_deadline() {
        // The hard fire point must be strictly LATER than the cooperative one
        // for any budget that exceeds the safety margin — this is the gap that
        // guarantees no premature fire. We compare against a SINGLE captured
        // `now` for the cooperative deadline, then bound the gap by the NOMINAL
        // 2s (hard 57s vs cooperative 55s) minus a small wall-clock-drift
        // tolerance so this stays stable even when the two `compute_*` calls
        // straddle a scheduling hiccup.
        let now = Instant::now();
        let coop = compute_deadline(Some(60)).expect("cooperative deadline");
        let hard = compute_hard_deadline(Some(60)).expect("hard deadline");
        let coop_from_now = coop.duration_since(now);
        // hard ~= 57s, coop ~= 55s; hard must be later by ~2s.
        assert!(
            hard > coop_from_now,
            "hard ({hard:?}) must be later than cooperative ({coop_from_now:?})"
        );
        // Nominal gap is 2s; allow up to 200ms of drift between the two calls.
        // (`hard > coop_from_now` asserted just above, so the gap never saturates.)
        assert!(
            hard.saturating_sub(coop_from_now) >= Duration::from_millis(1800),
            "hard ({hard:?}) - cooperative ({coop_from_now:?}) must be ~2s"
        );
    }

    /// BUDGET ARITHMETIC: the absolute hard kill — `hard + GRACE` — must land
    /// inside the MCC wall-clock budget (<= budget) AND strictly after the
    /// cooperative deadline (budget - SAFETY_MARGIN_SECS). With budget=60:
    /// hard=57, hard+grace=59 (<= 60), cooperative=55 (< 57).
    #[test]
    fn test_hard_plus_grace_within_budget_and_after_cooperative() {
        let budget = 60u64;
        let hard = compute_hard_deadline(Some(budget)).expect("hard");
        let hard_plus_grace = hard + Duration::from_secs(GRACE_SECS);
        assert!(
            hard_plus_grace <= Duration::from_secs(budget),
            "hard+grace ({hard_plus_grace:?}) must be <= budget ({budget}s)"
        );
        let cooperative_secs = budget - SAFETY_MARGIN_SECS;
        assert!(
            hard > Duration::from_secs(cooperative_secs),
            "hard ({hard:?}) must be strictly after the cooperative deadline ({cooperative_secs}s)"
        );
    }

    // -- run_with_hard_deadline ------------------------------------------------

    #[test]
    fn test_run_with_hard_deadline_completed_for_instant_closure() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let timed_out = AtomicBool::new(false);
        let outcome = run_with_hard_deadline(
            Duration::from_secs(5),
            || {},
            || timed_out.store(true, Ordering::SeqCst),
        );
        assert_eq!(outcome, RunOutcome::Completed);
        assert!(
            !timed_out.load(Ordering::SeqCst),
            "on_timeout must not fire"
        );
    }

    #[test]
    fn test_run_with_hard_deadline_timed_out_for_long_closure() {
        use std::sync::atomic::{AtomicBool, Ordering};
        // Hard deadline 50ms, worker sleeps far longer (but BOUNDED so the
        // scope join completes for the test). The worker must outlast BOTH the
        // hard deadline AND the grace window (GRACE_SECS = 2s); a 3s sleep is
        // comfortably past hard(50ms)+grace(2s), so the grace recv ALSO times
        // out and on_timeout fires. We assert the timeout was detected and the
        // action ran — proving the mechanism WITHOUT calling process::exit.
        let timed_out = AtomicBool::new(false);
        let outcome = run_with_hard_deadline(
            Duration::from_millis(50),
            || std::thread::sleep(Duration::from_secs(GRACE_SECS + 1)),
            || timed_out.store(true, Ordering::SeqCst),
        );
        assert_eq!(outcome, RunOutcome::TimedOut);
        assert!(
            timed_out.load(Ordering::SeqCst),
            "on_timeout must fire on a runaway worker"
        );
    }

    #[test]
    fn test_run_with_hard_deadline_completed_when_worker_finishes_in_grace_window() {
        use std::sync::atomic::{AtomicBool, Ordering};
        // Hard deadline 50ms: the first recv_timeout times out. The worker
        // finishes shortly AFTER (200ms total) — within the 2s grace window —
        // so the grace recv succeeds and the outcome is Completed, with the
        // fallback (on_timeout) NEVER invoked. This is the
        // Philosophers-PT-000050 "finishing right at the deadline" case.
        let timed_out = AtomicBool::new(false);
        let outcome = run_with_hard_deadline(
            Duration::from_millis(50),
            || std::thread::sleep(Duration::from_millis(200)),
            || timed_out.store(true, Ordering::SeqCst),
        );
        assert_eq!(
            outcome,
            RunOutcome::Completed,
            "worker finishing within the grace window must yield Completed"
        );
        assert!(
            !timed_out.load(Ordering::SeqCst),
            "on_timeout must NOT fire when the worker finishes in the grace window"
        );
    }
}
