// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Worker suspension barrier for periodic work (checkpoint, liveness).
//!
//! Implements TLC's `StateQueue.suspendAll()`/`resumeAll()` pattern adapted
//! for a lock-free work-stealing architecture. The main thread requests
//! suspension; workers block on the next `try_dequeue()` call; when all
//! workers are paused, the main thread is notified and can perform periodic
//! work (checkpointing, liveness checking) with no concurrent mutations.
//!
//! Part of #2749: Phase 1 prerequisite for parallel checkpoint/resume.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// Barrier that allows the main thread to suspend all BFS workers.
///
/// Workers call [`worker_check()`](WorkBarrier::worker_check) before each
/// dequeue attempt. If suspension is requested, they block until
/// [`resume_all()`](WorkBarrier::resume_all) is called.
///
/// Matches TLC's `StateQueue` suspension protocol: workers finish their
/// current state before suspending — no mid-computation interruption.
pub(crate) struct WorkBarrier {
    /// When true, workers should pause on next dequeue check.
    pause_requested: AtomicBool,
    /// Count of workers currently paused at the barrier.
    paused_count: AtomicUsize,
    /// Cumulative count of workers that have permanently departed (terminated)
    /// during the run — violation/error/RSS-limit/deadline exits that never
    /// pass back through `worker_check()`.
    ///
    /// Fix for N9: a departed worker is invisible to the pause protocol, so
    /// `suspend_all()` would block forever waiting for it to pause. Counting
    /// departures lets `suspend_all()` complete on `paused + departed ==
    /// num_workers` (i.e. no worker is still actively mutating shared state).
    /// This count is *cumulative* across suspend/resume cycles — a departed
    /// worker is gone for good — so it is NOT reset by `resume_all()`.
    departed_count: AtomicUsize,
    /// Total number of workers that must pause before `suspend_all` returns.
    num_workers: usize,
    /// Mutex protecting the resume condition.
    mu: Mutex<()>,
    /// Signaled when `paused_count + departed_count >= num_workers`
    /// (every worker is either paused or gone).
    all_paused: Condvar,
    /// Signaled when `pause_requested` is cleared (workers may resume).
    resume: Condvar,
}

impl WorkBarrier {
    /// Create a new barrier for `num_workers` worker threads.
    pub(crate) fn new(num_workers: usize) -> Self {
        Self {
            pause_requested: AtomicBool::new(false),
            paused_count: AtomicUsize::new(0),
            departed_count: AtomicUsize::new(0),
            num_workers,
            mu: Mutex::new(()),
            all_paused: Condvar::new(),
            resume: Condvar::new(),
        }
    }

    /// Main thread: request all workers to pause, block until every worker is
    /// either paused or has permanently departed.
    ///
    /// Returns `true` when *all* `num_workers` workers are genuinely paused at
    /// the barrier — a clean quiescent state in which the caller may safely
    /// snapshot state for checkpointing or run liveness checks.
    ///
    /// Returns `false` when the pause threshold was only reached because one or
    /// more workers *departed* (terminated on a violation/error/RSS-limit/
    /// deadline path). Fix for N9: this never blocks on a worker that will
    /// never pause. A departure always sets the shared stop flag, so the run is
    /// already winding down — the caller MUST skip its maintenance action (the
    /// checkpoint / periodic-liveness snapshot) and let finalization proceed.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (indicates a prior panic
    /// while holding the lock — an unrecoverable state for the checker).
    pub(crate) fn suspend_all(&self) -> bool {
        // Set the pause flag. Workers will see this on their next
        // `worker_check()` call via Relaxed load (fast path).
        self.pause_requested.store(true, Ordering::SeqCst);

        let mut guard = self.mu.lock().expect("WorkBarrier mutex poisoned");
        // Complete once no worker is still actively mutating shared state:
        // paused workers + departed workers == num_workers. Both counters are
        // mutated only under `mu` (worker_check / worker_exited), and both
        // signal `all_paused`, so there is no lost-wakeup window.
        while self.paused_count.load(Ordering::SeqCst) + self.departed_count.load(Ordering::SeqCst)
            < self.num_workers
        {
            guard = self
                .all_paused
                .wait(guard)
                .expect("WorkBarrier mutex poisoned");
        }
        // Clean suspension iff no departures contributed to reaching the
        // threshold. Any departure => degraded/terminating run => caller skips
        // the maintenance action.
        self.departed_count.load(Ordering::SeqCst) == 0
    }

    /// Main thread: resume all paused workers.
    ///
    /// Must be called after [`suspend_all()`](WorkBarrier::suspend_all) to
    /// allow workers to continue BFS exploration. Workers will wake and
    /// resume dequeuing work items.
    pub(crate) fn resume_all(&self) {
        let _guard = self.mu.lock().expect("WorkBarrier mutex poisoned");
        // Reset paused count BEFORE clearing pause flag to avoid a race
        // where a worker sees pause_requested=false but paused_count is
        // still non-zero from the previous suspension cycle.
        //
        // N9: `departed_count` is intentionally NOT reset — a departed worker
        // stays gone, so the reduced live-worker count carries into the next
        // suspend/resume cycle (the next `suspend_all()` completes on
        // `paused(live) + departed == num_workers`).
        self.paused_count.store(0, Ordering::SeqCst);
        self.pause_requested.store(false, Ordering::SeqCst);
        self.resume.notify_all();
    }

    /// Worker thread: signal permanent departure to the barrier.
    ///
    /// Fix for N9: called from each transport's `Drop` — the single choke
    /// point that fires exactly once per worker, precisely when its BFS run
    /// ends *without* pausing (a worker never re-enters `worker_check` after
    /// its run returns). This covers EVERY non-pausing termination in one
    /// place: the `snapshot_stop_and_send` + `BfsTermination::Exit` paths
    /// (violation, error, RSS-limit, and the N8 wall-clock deadline exit) AND
    /// the cascade where those exits set `stop_flag`, causing sibling workers
    /// to bail via the Done path in `try_dequeue` (which checks `stop_flag`
    /// before the barrier) instead of pausing.
    ///
    /// Increments the cumulative departed count and wakes a coordinator that
    /// may be blocked in `suspend_all()`, so it does not hang waiting for a
    /// worker that will never pause. Paused workers are blocked inside
    /// `worker_check` and have NOT dropped their transport, so they remain
    /// correctly counted as paused, not departed.
    pub(crate) fn worker_exited(&self) {
        // Take the lock so the increment + notify cannot race with
        // `suspend_all()`'s predicate re-check inside its wait loop.
        let _guard = self.mu.lock().expect("WorkBarrier mutex poisoned");
        self.departed_count.fetch_add(1, Ordering::SeqCst);
        // The pause threshold may now be reachable via (paused + departed);
        // wake the (single) coordinator waiting in suspend_all().
        self.all_paused.notify_one();
    }

    /// Worker thread: check if suspension is requested, block if so.
    ///
    /// Called by each worker before attempting to dequeue a work item.
    /// Fast path (no suspension): single `Relaxed` atomic load.
    /// Slow path (suspension requested): increment paused count, notify
    /// main thread if last to pause, block until resumed.
    ///
    /// Workers should flush any batched counters (e.g., `local_work_delta`)
    /// before calling this method to ensure the main thread sees accurate
    /// global state during the suspension window.
    #[inline]
    pub(crate) fn worker_check(&self) {
        // Fast path: no suspension requested. Cost: one Relaxed atomic load.
        if !self.pause_requested.load(Ordering::Relaxed) {
            return;
        }

        // Slow path: suspension requested. Acquire the lock for condvar wait.
        let mut guard = self.mu.lock().expect("WorkBarrier mutex poisoned");

        // Re-check under the lock: the main thread might have called
        // resume_all() between our Relaxed check and acquiring the lock.
        if !self.pause_requested.load(Ordering::SeqCst) {
            return;
        }

        // Increment paused count. If every remaining worker is now accounted
        // for — paused here plus any that have permanently departed — notify
        // the main thread. N9: without the `departed_count` term, the last
        // *live* worker to pause would not fire this notification when some
        // sibling has already departed, and `suspend_all()` could miss the
        // wake and hang.
        let prev = self.paused_count.fetch_add(1, Ordering::SeqCst);
        if prev + 1 + self.departed_count.load(Ordering::SeqCst) >= self.num_workers {
            self.all_paused.notify_one();
        }

        // Block until resume_all() clears the pause flag.
        while self.pause_requested.load(Ordering::SeqCst) {
            guard = self.resume.wait(guard).expect("WorkBarrier mutex poisoned");
        }
    }

    /// Returns true if the main thread has requested worker suspension.
    ///
    /// This is a fast `Relaxed` atomic check suitable for polling in the
    /// worker hot path. Workers should use this to decide whether to flush
    /// batched counters before calling [`worker_check()`](WorkBarrier::worker_check).
    #[inline]
    pub(crate) fn is_pause_requested(&self) -> bool {
        self.pause_requested.load(Ordering::Relaxed)
    }

    /// Returns true if all workers are currently suspended.
    ///
    /// Useful for the main thread to check barrier state without blocking.
    // Part of #2749: test-only until checkpoint UI needs suspension status.
    #[inline]
    #[cfg(test)]
    pub(crate) fn is_suspended(&self) -> bool {
        self.pause_requested.load(Ordering::SeqCst)
            && self.paused_count.load(Ordering::SeqCst) + self.departed_count.load(Ordering::SeqCst)
                >= self.num_workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_barrier_suspend_resume_basic() {
        let num_workers = 4;
        let barrier = Arc::new(WorkBarrier::new(num_workers));
        let counter = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicBool::new(true));

        // Spawn worker threads that loop checking the barrier
        let mut handles = Vec::new();
        for _ in 0..num_workers {
            let barrier = Arc::clone(&barrier);
            let counter = Arc::clone(&counter);
            let running = Arc::clone(&running);
            handles.push(thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    barrier.worker_check();
                    // Simulate doing work
                    counter.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                }
            }));
        }

        // Let workers run briefly
        thread::sleep(Duration::from_millis(10));

        // Suspend all workers
        barrier.suspend_all();

        // While suspended, counter should not change
        let snapshot = counter.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(10));
        let after = counter.load(Ordering::Relaxed);
        assert_eq!(snapshot, after, "counter changed while workers suspended");
        assert!(barrier.is_suspended());

        // Resume workers
        barrier.resume_all();

        // Workers should make progress again
        thread::sleep(Duration::from_millis(10));
        let final_count = counter.load(Ordering::Relaxed);
        assert!(final_count > after, "workers did not resume");

        // Shut down workers
        running.store(false, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_barrier_multiple_suspend_resume_cycles() {
        let num_workers = 3;
        let barrier = Arc::new(WorkBarrier::new(num_workers));
        let running = Arc::new(AtomicBool::new(true));
        let cycle_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..num_workers {
            let barrier = Arc::clone(&barrier);
            let running = Arc::clone(&running);
            let cycle_count = Arc::clone(&cycle_count);
            handles.push(thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    barrier.worker_check();
                    cycle_count.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                }
            }));
        }

        // Run multiple suspend/resume cycles
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(5));
            barrier.suspend_all();

            // Verify suspension holds
            let snap = cycle_count.load(Ordering::Relaxed);
            thread::sleep(Duration::from_millis(5));
            assert_eq!(
                snap,
                cycle_count.load(Ordering::Relaxed),
                "work done while suspended"
            );

            barrier.resume_all();
        }

        running.store(false, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Fix for N9: `suspend_all()` must not hang when a worker terminates
    /// (violation/error/RSS-limit/deadline) during a suspension window instead
    /// of pausing. With `num_workers` = N, one worker departing via
    /// `worker_exited()` plus N-1 workers pausing via `worker_check()` must let
    /// `suspend_all()` complete — and it must report `false` (a departure
    /// occurred, so the coordinator skips its maintenance action).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_suspend_all_completes_when_one_worker_departs() {
        let num_workers = 4;
        let barrier = Arc::new(WorkBarrier::new(num_workers));

        barrier.pause_requested.store(true, Ordering::SeqCst);

        // N-1 workers reach the barrier and pause.
        let mut handles = Vec::new();
        for _ in 0..(num_workers - 1) {
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.worker_check();
            }));
        }

        // The Nth worker never pauses — it terminates and signals departure.
        // Spawn it so the test exercises the concurrent wake, and to make a
        // regression (a hang) fail via the ntest timeout rather than block.
        let b = Arc::clone(&barrier);
        let departer = thread::spawn(move || {
            b.worker_exited();
        });

        // Without the N9 fix this blocks forever (only N-1 ever pause).
        let clean = barrier.suspend_all();
        assert!(
            !clean,
            "suspend_all must report a non-clean suspension when a worker departed"
        );
        assert!(barrier.is_suspended(), "barrier should read as suspended");

        // Resume the N-1 paused workers so their threads can exit.
        barrier.resume_all();
        departer.join().unwrap();
        for h in handles {
            h.join().unwrap();
        }
    }

    /// N9: after a departure, a subsequent suspend/resume cycle over the
    /// remaining live workers still completes (the departed count is cumulative
    /// and not reset by `resume_all`).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_suspend_all_reduced_live_count_after_departure() {
        let num_workers = 3;
        let barrier = Arc::new(WorkBarrier::new(num_workers));

        // One worker departs before any suspension.
        barrier.worker_exited();

        // Cycle 1: the two live workers pause; suspend completes on 2 + 1 == 3.
        barrier.pause_requested.store(true, Ordering::SeqCst);
        let mut handles = Vec::new();
        for _ in 0..(num_workers - 1) {
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || b.worker_check()));
        }
        assert!(
            !barrier.suspend_all(),
            "a prior departure makes every suspension non-clean"
        );
        barrier.resume_all();
        for h in handles {
            h.join().unwrap();
        }

        // Cycle 2: same reduced live set pauses again; still completes.
        barrier.pause_requested.store(true, Ordering::SeqCst);
        let mut handles = Vec::new();
        for _ in 0..(num_workers - 1) {
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || b.worker_check()));
        }
        assert!(!barrier.suspend_all());
        barrier.resume_all();
        for h in handles {
            h.join().unwrap();
        }
    }

    /// N9: all workers departing (none left to pause) still completes
    /// `suspend_all()` instead of hanging.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_suspend_all_completes_when_all_workers_depart() {
        let num_workers = 3;
        let barrier = Arc::new(WorkBarrier::new(num_workers));
        barrier.pause_requested.store(true, Ordering::SeqCst);
        for _ in 0..num_workers {
            barrier.worker_exited();
        }
        assert!(!barrier.suspend_all());
        barrier.resume_all();
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_barrier_single_worker() {
        let barrier = Arc::new(WorkBarrier::new(1));
        let paused = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));

        let b = Arc::clone(&barrier);
        let p = Arc::clone(&paused);
        let r = Arc::clone(&running);
        let handle = thread::spawn(move || {
            while r.load(Ordering::Relaxed) {
                b.worker_check();
                p.store(false, Ordering::Relaxed);
                thread::yield_now();
            }
        });

        thread::sleep(Duration::from_millis(5));
        barrier.suspend_all();
        assert!(barrier.is_suspended());
        barrier.resume_all();

        running.store(false, Ordering::Relaxed);
        handle.join().unwrap();
    }
}
