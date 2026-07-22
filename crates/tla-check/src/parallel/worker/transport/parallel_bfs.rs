// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `BfsTransport` implementation for `ParallelTransport` (#2356 Phase 4 Step 4c).
//!
//! Contains the trait impl (`try_dequeue`, `resolve`, `process_successors`, etc.),
//! and the `Drop` impl. The `process_successors_inner` pipeline is in
//! `successor_pipeline.rs`.
//!
//! Extracted from `transport/mod.rs` for file-size compliance. Part of #3389.

mod successor_pipeline;

use super::super::coordination::{
    finish_if_stop_requested, flush_local_work_delta_profiled, send_done_result,
    snapshot_stop_and_send,
};
use super::super::work_item::{BfsWorkItem, ResolvedArrayState};
use super::super::work_stealing::{poll_work_with_backoff, PollAction};
use super::super::{WorkerResult, BACKOFF_YIELD_THRESHOLD};
use super::ParallelTransport;
use crate::check::model_checker::bfs::transport::{BfsTermination, BfsTransport};
use crate::check::CheckError;
use crate::parallel::WORK_BATCH_SIZE;
use crate::state::{ArrayState, Fingerprint};
use std::sync::atomic::Ordering;
use tla_eval::eval_arena::ArenaStateGuard;

impl<T: BfsWorkItem> BfsTransport for ParallelTransport<T> {
    type WorkItem = (T, usize);

    fn try_dequeue(&mut self) -> Option<Self::WorkItem> {
        if self.shared_frontier_tail_mode_active() {
            return self.try_dequeue_shared_frontier_tail();
        }

        // Part of #3233: Fast path — pop from local queue without touching the
        // shared active_workers atomic. For a steady BFS with deep queues this
        // eliminates ~2 AcqRel atomic RMW per state (was ~16M for MCBoulanger).
        // TLC uses a similar approach: workers only signal shared state when they
        // enter/exit a waiting condition, not per state processed.
        if let Some(item) = self.local_queue.pop() {
            self.consecutive_empty = 0;
            // Still honour stop flag and barrier on the fast path so checkpoint
            // and liveness pauses are not delayed while the local queue is deep.
            if self.stop_flag.load(Ordering::Relaxed) {
                flush_local_work_delta_profiled(
                    &mut self.local_work_delta,
                    &self.work_remaining,
                    &mut self.stats,
                );
                send_done_result(&self.ctx, &mut self.stats, &self.result_tx);
                return None;
            }
            if self.barrier.is_pause_requested() {
                flush_local_work_delta_profiled(
                    &mut self.local_work_delta,
                    &self.work_remaining,
                    &mut self.stats,
                );
                self.barrier.worker_check();
            }
            return Some(item);
        }

        // Local queue empty. Keep the worker counted as active across short
        // empty streaks on the diff-heavy hot path, but publish idleness
        // immediately on non-diff fallback specs (VIEW / constraints /
        // action constraints). The fallback lane is not the #3285 throughput
        // target, and hiding emptiness there can strand tiny state spaces in
        // the work-stealing termination loop (#3390).
        if !self.use_diffs && self.worker_active {
            flush_local_work_delta_profiled(
                &mut self.local_work_delta,
                &self.work_remaining,
                &mut self.stats,
            );
            self.mark_worker_idle();
        }

        // The diff-heavy path keeps the recent hysteresis so transient misses
        // do not churn `active_workers` during the late-tail throughput lane.
        loop {
            if finish_if_stop_requested(
                &self.stop_flag,
                &self.ctx,
                &mut self.stats,
                &self.result_tx,
                &mut self.local_work_delta,
                &self.work_remaining,
            ) {
                return None;
            }
            // Part of #2749: Check barrier before attempting to dequeue.
            // Flush batched work delta first so the main thread sees accurate
            // work_remaining during the suspension window.
            if self.barrier.is_pause_requested() {
                flush_local_work_delta_profiled(
                    &mut self.local_work_delta,
                    &self.work_remaining,
                    &mut self.stats,
                );
                self.barrier.worker_check();
            }
            if self.shared_frontier_tail_mode_active() {
                return self.try_dequeue_shared_frontier_tail();
            }
            match poll_work_with_backoff(
                &self.local_queue,
                &self.injector,
                &self.stealers,
                &self.bootstrap_injector_budget,
                self.depth_limited,
                &self.ctx,
                &self.result_tx,
                &mut self.stats,
                &mut self.consecutive_empty,
                &mut self.local_work_delta,
                &self.work_remaining,
                &self.active_workers,
                self.num_workers,
                self.worker_id,
                &mut self.steal_cursor,
            ) {
                PollAction::Work(item) => {
                    // Transition back to active after successful steal.
                    self.mark_worker_active();
                    return Some(item);
                }
                PollAction::Continue => {
                    if self.worker_active && self.consecutive_empty >= BACKOFF_YIELD_THRESHOLD {
                        flush_local_work_delta_profiled(
                            &mut self.local_work_delta,
                            &self.work_remaining,
                            &mut self.stats,
                        );
                        self.mark_worker_idle();
                    }
                }
                PollAction::PromoteSharedFrontierTail => {
                    if self.worker_active {
                        flush_local_work_delta_profiled(
                            &mut self.local_work_delta,
                            &self.work_remaining,
                            &mut self.stats,
                        );
                        self.mark_worker_idle();
                    }
                    self.promote_shared_frontier_tail();
                    return self.try_dequeue_shared_frontier_tail();
                }
                PollAction::ReturnDone => return None,
            }
        }
    }

    fn resolve(
        &mut self,
        item: Self::WorkItem,
    ) -> Result<Option<(Fingerprint, ArrayState, usize)>, BfsTermination> {
        let (mut work_item, current_depth) = item;
        // Part of #3233: Only increment active_workers on the first call or
        // after an idle→active transition via steal. When the fast path in
        // try_dequeue() popped from the local queue, worker_active is already
        // true from the previous resolve() call — no atomic RMW needed.
        if !self.worker_active {
            self.mark_worker_active();
        }
        let base_fp = work_item.base_fingerprint(&self.var_registry);
        self.stats.states_explored += 1;
        let queue_size = self.local_queue.len();
        if queue_size > self.local_max_queue {
            self.local_max_queue = queue_size;
            let _ = self.max_queue.fetch_max(queue_size, Ordering::Relaxed);
        }
        // Track per-worker max local queue depth for load balance diagnostics.
        if queue_size > self.stats.max_local_queue_depth {
            self.stats.max_local_queue_depth = queue_size;
        }
        debug_assert!(
            !self.current_uses_decode_scratch,
            "decode scratch must be reclaimed before resolving the next work item"
        );
        let mut scratch = self.decode_scratch.take();
        let resolved =
            work_item.resolve_array_state(scratch.as_mut(), &self.var_registry, &self.mode);
        let (current_array, used_scratch) = match resolved {
            ResolvedArrayState::Owned(arr) => {
                self.decode_scratch = scratch;
                (arr, false)
            }
            ResolvedArrayState::UsedScratch => (
                scratch.expect("scratch-backed resolve must return the transport scratch"),
                true,
            ),
        };
        self.current_uses_decode_scratch = used_scratch;

        Ok(Some((base_fp, current_array, current_depth)))
    }

    fn report_progress(&mut self, _depth: usize, _queue_len: usize) -> Result<(), BfsTermination> {
        // Per-worker inline RSS check: supplements the coordinator's periodic
        // maintenance barrier with a lightweight check. A single
        // current_rss_bytes() syscall is cheap (~1μs on macOS/Linux).
        //
        // Part of resource-audit finding #10: the gate fires on whichever comes
        // first — RSS_CHECK_STATE_INTERVAL states OR RSS_CHECK_TIME_INTERVAL of
        // wall-clock time. A pure state-count cadence is not a memory bound: a
        // spec whose states hold large values could allocate past the budget
        // and be OOM-killed before the next state-count poll. The wall-clock
        // backstop makes the memory budget fail-closed.
        if let Some(critical) = self.critical_rss_bytes {
            if self.rss_check_gate.tick(std::time::Instant::now()) {
                if let Some(rss) = crate::memory::current_rss_bytes() {
                    if rss >= critical {
                        // Set stop flag so all workers stop promptly.
                        self.stop_flag.store(true, Ordering::Relaxed);
                        let stop_ctx = par_stop_ctx!(self);
                        snapshot_stop_and_send(
                            &self.ctx,
                            &mut self.stats,
                            &stop_ctx,
                            &self.result_tx,
                            |stats| WorkerResult::LimitReached {
                                limit_type: crate::check::LimitType::Memory,
                                stats,
                            },
                        );
                        // Result sent via channel; Exit terminates the BFS loop.
                        return Err(BfsTermination::Exit);
                    }
                }
            }
        }
        Ok(())
    }

    fn should_checkpoint(&self) -> bool {
        false
    }

    fn save_checkpoint(&mut self, _current: &ArrayState) {}

    fn check_state_limit(&mut self) -> Result<(), BfsTermination> {
        // Enforced via CAS in ParallelAdapter::admit(), not here.
        Ok(())
    }

    fn release_state(&mut self, _fp: Fingerprint, current: ArrayState) {
        self.reclaim_decode_scratch(current);
    }

    fn on_deadline_exceeded(&mut self, _fp: Fingerprint, current: ArrayState) -> BfsTermination {
        // Part of #3 + N8 (time backstop): the wall-clock deadline elapsed.
        // Reclaim scratch, set the shared stop flag so every sibling worker
        // winds down on its next poll, then SEND a partial
        // `LimitReached { Time }` result before exiting.
        //
        // N8 fix: this previously exited *silently*, so the coordinator saw no
        // terminal outcome for the run and defaulted to `Success` on a
        // partially explored state space — a false PASS. This now mirrors the
        // MEMORY-limit path in `report_progress` exactly: `snapshot_stop_and_send`
        // records `states_at_stop` and emits `LimitReached { Time }`, which the
        // coordinator maps to `CheckResult::LimitReached { Time }` (never
        // `Success`), so an interrupted exploration is never reported as a proof.
        self.reclaim_decode_scratch(current);
        self.stop_flag.store(true, Ordering::Relaxed);
        let stop_ctx = par_stop_ctx!(self);
        snapshot_stop_and_send(
            &self.ctx,
            &mut self.stats,
            &stop_ctx,
            &self.result_tx,
            |stats| WorkerResult::LimitReached {
                limit_type: crate::check::LimitType::Time,
                stats,
            },
        );
        BfsTermination::Exit
    }

    fn on_depth_limit_skip(&mut self) {
        self.depth_limit_reached.store(true, Ordering::Relaxed);
        self.local_work_delta -= 1;
        if self.local_work_delta <= -(WORK_BATCH_SIZE as isize) {
            flush_local_work_delta_profiled(
                &mut self.local_work_delta,
                &self.work_remaining,
                &mut self.stats,
            );
        }
    }

    fn process_successors(
        &mut self,
        fp: Fingerprint,
        current: ArrayState,
        _depth: usize,
        succ_depth: usize,
        current_level: u32,
        succ_level: u32,
    ) -> Result<(), BfsTermination> {
        // Part of #3580: keep the arena lifecycle panic-safe across successor eval.
        let _arena_state = ArenaStateGuard::new();
        let transitions_before = self.stats.transitions;
        let result =
            self.process_successors_inner(fp, &current, succ_depth, current_level, succ_level);
        // Part of #3910: Record action evaluation for tier promotion tracking.
        // Uses action_id=0 because the parallel BFS loop evaluates Next monolithically.
        if let Some(ref tier) = self.tier_state {
            let succ_count = self.stats.transitions.saturating_sub(transitions_before);
            tier.record_action_eval(0, succ_count as u64);
        }
        self.reclaim_decode_scratch(current);
        result
    }

    fn handle_error(
        &mut self,
        _fp: Fingerprint,
        current: ArrayState,
        error: CheckError,
    ) -> BfsTermination {
        self.reclaim_decode_scratch(current);
        let stop_ctx = par_stop_ctx!(self);
        snapshot_stop_and_send(
            &self.ctx,
            &mut self.stats,
            &stop_ctx,
            &self.result_tx,
            |stats| WorkerResult::Error(error, stats),
        );
        BfsTermination::Exit
    }

    fn output_profile(&self) {
        // Part of #3580: report per-worker arena allocation stats.
        tla_eval::eval_arena::report_arena_stats();
        // Part of #3990: report per-worker state arena stats.
        crate::arena::report_worker_arena_stats();
    }

    fn depth_limit_reached(&self) -> bool {
        self.depth_limit_reached.load(Ordering::Relaxed)
    }
}

impl<T: BfsWorkItem> Drop for ParallelTransport<T> {
    fn drop(&mut self) {
        self.mark_worker_idle();
        // N9 fix (deadlock-freedom): signal permanent departure to the
        // suspension barrier. A worker's transport is dropped exactly once,
        // when its BFS run ends — and a worker only ends its run *without*
        // pausing (it never re-enters `worker_check`). This is the single
        // choke point that covers EVERY non-pausing termination:
        //   * the `snapshot_stop_and_send` + `Exit` paths (violation, error,
        //     RSS-limit, and the N8 wall-clock deadline), AND
        //   * the cascade where those Exits set `stop_flag`, causing sibling
        //     workers to bail via the Done path in `try_dequeue` (which checks
        //     `stop_flag` before the barrier) instead of pausing.
        // Without counting this cascade, `suspend_all()` would still hang
        // waiting for workers that will never pause. Paused workers are blocked
        // inside `worker_check` and have NOT dropped their transport, so they
        // are correctly still counted as paused, not departed. In a clean run
        // no transport drops during a suspension window, so `suspend_all()`
        // still returns `true` (departed_count stays 0).
        self.barrier.worker_exited();
        if self.shared_frontier_tail_mode_active() {
            self.shared_frontier.finish();
        }
    }
}
