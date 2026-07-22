// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! The adaptive memory probe — one abstraction replacing ~15 scattered
//! `counter.is_multiple_of(512|4096) && exceeds_budget()` hacks.
//!
//! ## Why adaptive, and how it is *resource-aware*
//!
//! A fixed iteration cadence is a guess that is wrong for every workload: "poll
//! every 512 iterations" is gigabytes on a wide net (each iteration allocates a
//! ~100 KB marking) and nothing on a narrow one, and it says nothing about the
//! wall clock, so a slow, heavy loop checks its deadline far too rarely.
//!
//! [`MemoryProbe`] instead runs on a **time-gated, self-tuning cadence**:
//!
//! - **Hot path** ([`MemoryProbe::over_budget`]): a single counter decrement
//!   (~1 ns), no syscall, no clock read — cheap enough to call once per loop
//!   iteration (per successor / per pop / per interned node).
//! - **Cold path** (fires when the countdown expires): reads the clock *once*
//!   and uses that one read for BOTH the wall-clock deadline and the memory
//!   decision. It then **self-tunes the next stride** from the observed
//!   iterations-per-second so the cold path lands again after ~one target
//!   interval of wall-clock — automatically large for fast loops, small for
//!   slow loops. And the **target interval shrinks as the footprint
//!   approaches the ceiling** (rare & cheap when far, frequent & careful when
//!   near), so near-limit bursts are caught while calm phases pay almost
//!   nothing. There is no byte-rate extrapolation (robust to bursty,
//!   non-stationary allocation) — only wall time, which is monotone.
//!
//! The scheduling and decision logic are pure (they take the clock/footprint as
//! arguments), so the whole probe is deterministically unit-testable without a
//! real allocator or clock.
//!
//! One probe is owned per loop / per worker thread by `&mut` (it is plain data
//! and therefore `Send`); the process footprint it reads is a whole-process
//! signal, so independent per-thread probes coordinate implicitly — never share
//! one probe across threads.

use std::time::{Duration, Instant};

use crate::budget::MemoryBudget;

/// Countdown floor: the fewest hot-path iterations between cold-path visits.
/// Its only job is amortizing the clock read on the very fastest loops.
const STRIDE_MIN: u32 = 256;
/// Countdown ceiling: the most iterations we will skip before a cold-path
/// visit, bounding worst-case latency even if the speed estimate overshoots.
const STRIDE_MAX: u32 = 1 << 22; // ~4.2M
/// Initial stride: `1`, so the FIRST tick reaches the cold path and checks the
/// wall-clock DEADLINE. This is a correctness requirement, not just tuning — an
/// already-expired deadline must be caught on iteration one, like the
/// fixed-cadence guards that tested it at iteration 0. (The MEMORY budget is
/// deliberately NOT evaluated on that first cold path — see `warmup` — so this
/// early check cannot spuriously decline a tiny run on a busy host.) After the
/// first cold path the stride grows to the adaptive value.
const STRIDE_INIT: u32 = 1;

/// Target wall-clock between cold-path checks when far below the ceiling.
const INTERVAL_FAR: Duration = Duration::from_millis(50);
/// Target wall-clock between cold-path checks when at/over the ceiling.
const INTERVAL_NEAR: Duration = Duration::from_millis(1);

/// Why a probe latched a stop — so sites that map a memory trip and a deadline
/// trip to distinct decline values (e.g. the mu-calculus solver's
/// `MuAbort::NodeCapReached` vs `MuAbort::DeadlineExceeded`) keep that
/// distinction through the one probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trip {
    /// The self-footprint ceiling or the collective free-memory floor.
    Memory,
    /// The wall-clock deadline.
    Deadline,
}

/// Adaptive, self-tuning memory/deadline probe. See the module docs.
#[derive(Debug, Clone)]
pub struct MemoryProbe {
    budget: MemoryBudget,
    deadline: Option<Instant>,
    /// Hot-path countdown; the cold path fires when it reaches zero.
    countdown: u32,
    /// Hot-path iterations since the last cold-path visit (speed estimate).
    ticks_since_cold: u64,
    /// Wall-clock of the last cold-path visit.
    last_cold: Instant,
    /// Current target wall-clock interval between cold-path visits (shrinks as
    /// the footprint nears the ceiling).
    target_interval: Duration,
    /// Latched trip reason once declined; a decision to decline is final, so
    /// subsequent ticks short-circuit to the same reason.
    tripped: Option<Trip>,
    /// `true` until the first cold path completes. The first cold path (fired on
    /// tick 1 so an already-expired DEADLINE is caught immediately) deliberately
    /// does NOT yet evaluate the MEMORY budget: a just-started loop has a tiny
    /// footprint and is not a collective-pressure contributor, so tripping the
    /// zero-gate floor there would decline a small/fast run that the historic
    /// ~512-iteration cadence let complete; and a single tick over the
    /// sub-microsecond construction gap is not an honest speed sample, so
    /// extrapolating a stride from it would inflate it to `STRIDE_MAX` and defeat
    /// the deadline clamp. The next cold path spans a real interval and does the
    /// full memory decision + rate estimate.
    warmup: bool,
}

impl MemoryProbe {
    /// Create a probe for `budget`, optionally carrying a wall-clock
    /// `deadline` (checked on the same cold-path cadence, so the per-loop
    /// deadline poll collapses into this one probe).
    #[must_use]
    pub fn new(budget: MemoryBudget, deadline: Option<Instant>) -> Self {
        Self {
            budget,
            deadline,
            countdown: STRIDE_INIT,
            ticks_since_cold: 0,
            last_cold: Instant::now(),
            target_interval: INTERVAL_FAR,
            tripped: None,
            warmup: true,
        }
    }

    /// Cheap per-iteration call. Returns `true` when the loop must stop and
    /// decline (memory ceiling exceeded, machine collectively low, or the
    /// deadline passed). Call once per loop iteration and map `true` to the
    /// loop's existing decline value (`Ok(None)` / `break` / an abort variant
    /// / a shared stop flag).
    #[inline]
    pub fn over_budget(&mut self) -> bool {
        self.check().is_some()
    }

    /// Like [`Self::over_budget`] but reports WHY the probe declined, for sites
    /// that map a memory trip and a deadline trip to distinct values. `None`
    /// ⇒ keep going.
    #[inline]
    pub fn check(&mut self) -> Option<Trip> {
        // A decline is terminal. Once latched, every later call is O(1) and
        // issues NO syscall — some sites keep calling `over_budget()` after a
        // trip (e.g. loops that set a shared stop flag and finish the current
        // batch), and re-running the cold path's footprint/host-free probes on
        // each of those calls would be a real per-call syscall cost.
        if let Some(trip) = self.tripped {
            return Some(trip);
        }
        self.ticks_since_cold += 1;
        if self.countdown > 1 {
            self.countdown -= 1;
            return None;
        }
        self.cold_path()
    }

    /// The latched trip reason, if any (without ticking).
    #[must_use]
    pub fn tripped(&self) -> Option<Trip> {
        self.tripped
    }

    /// The wall-clock deadline this probe carries, if any. Lets a caller merge
    /// an already-armed deadline with a tighter one when re-arming.
    #[must_use]
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    #[cold]
    fn cold_path(&mut self) -> Option<Trip> {
        let now = Instant::now();
        let footprint = crate::platform::process_footprint_bytes();
        let host_free = crate::platform::host_free_bytes();
        self.evaluate(now, footprint, host_free)
    }

    /// Pure cold-path core (all impurity — the clock and the OS probes — is
    /// passed in), so the latch, deadline, budget decision, and stride
    /// self-tuning are unit-testable deterministically.
    fn evaluate(
        &mut self,
        now: Instant,
        footprint: Option<usize>,
        host_free: Option<usize>,
    ) -> Option<Trip> {
        if self.tripped.is_some() {
            return self.tripped;
        }
        // The deadline is checked on EVERY cold path, the warm-up one included:
        // an already-expired deadline (or one that passes during a tiny loop)
        // must decline promptly.
        if self.deadline.is_some_and(|d| now >= d) {
            self.tripped = Some(Trip::Deadline);
            return self.tripped;
        }
        if self.warmup {
            // First cold path: defer the memory decision and do not extrapolate
            // a stride from the single sub-microsecond sample (see `warmup`).
            self.warmup = false;
            self.countdown = STRIDE_MIN;
            self.last_cold = now;
            self.ticks_since_cold = 0;
            return None;
        }
        if let Some(fp) = footprint {
            if self.budget.over_budget(fp, host_free) {
                self.tripped = Some(Trip::Memory);
                return self.tripped;
            }
            // Tighten the target as we approach the ceiling.
            self.target_interval = interval_for_headroom(fp, self.budget.ceiling());
        }
        // Self-tune the next stride so the cold path recurs after ~one target
        // interval of wall-clock, from the observed iterations/sec — clamped,
        // and never past a near deadline.
        let elapsed = now.saturating_duration_since(self.last_cold);
        let mut next = estimate_stride(self.ticks_since_cold, elapsed, self.target_interval);
        if let Some(deadline) = self.deadline {
            next = clamp_before_deadline(next, self.ticks_since_cold, elapsed, now, deadline);
        }
        self.countdown = next;
        self.last_cold = now;
        self.ticks_since_cold = 0;
        None
    }
}

/// Target cold-path interval given the current footprint and ceiling: shrinks
/// quadratically from [`INTERVAL_FAR`] to [`INTERVAL_NEAR`] as usage rises from
/// 0 to 1. No ceiling ⇒ far (only the collective floor matters, and it moves
/// slowly). Pure.
fn interval_for_headroom(footprint: usize, ceiling: Option<usize>) -> Duration {
    let Some(ceiling) = ceiling else {
        return INTERVAL_FAR;
    };
    if ceiling == 0 {
        return INTERVAL_NEAR;
    }
    let usage = (footprint as f64 / ceiling as f64).clamp(0.0, 1.0);
    let slack = 1.0 - usage;
    let slack2 = slack * slack; // tighten fast near the top
    let far = INTERVAL_FAR.as_micros() as f64;
    let near = INTERVAL_NEAR.as_micros() as f64;
    let micros = near + (far - near) * slack2;
    Duration::from_micros(micros as u64)
}

/// Estimate the iteration stride that spans ~`target` wall-clock, from the
/// observed `ticks` over `elapsed`. Falls back to a small stride when no useful
/// sample exists (elapsed ~0). Pure; clamped to [`STRIDE_MIN`, `STRIDE_MAX`].
fn estimate_stride(ticks: u64, elapsed: Duration, target: Duration) -> u32 {
    let elapsed_ns = elapsed.as_nanos();
    if elapsed_ns == 0 || ticks == 0 {
        return STRIDE_MIN;
    }
    let target_ns = target.as_nanos();
    // ticks per target interval = ticks * (target / elapsed).
    let estimate = (ticks as u128).saturating_mul(target_ns) / elapsed_ns;
    estimate.clamp(STRIDE_MIN as u128, STRIDE_MAX as u128) as u32
}

/// Shrink `stride` so the next cold path lands before `deadline` (using the
/// same observed rate), so a memory-calm phase's large stride cannot delay a
/// near deadline check. Pure.
fn clamp_before_deadline(
    stride: u32,
    ticks: u64,
    elapsed: Duration,
    now: Instant,
    deadline: Instant,
) -> u32 {
    let remaining = deadline.saturating_duration_since(now);
    let elapsed_ns = elapsed.as_nanos();
    if elapsed_ns == 0 || ticks == 0 {
        return stride;
    }
    // ticks that fit in the remaining time at the observed rate; aim for ~half
    // so we check with margin before the deadline.
    let fit = (ticks as u128).saturating_mul(remaining.as_nanos()) / elapsed_ns / 2;
    let cap = fit.clamp(STRIDE_MIN as u128, STRIDE_MAX as u128) as u32;
    stride.min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget_ceiling(ceiling: usize) -> MemoryBudget {
        // ceiling only; healthy floor so the collective arm never fires here.
        MemoryBudget::from_thresholds(Some(ceiling), None, 0, 0)
    }

    #[test]
    fn first_tick_checks_then_subsequent_are_cheap_countdown() {
        // A generous ceiling and no deadline: healthy, so no trip ever.
        let mut p = MemoryProbe::new(budget_ceiling(usize::MAX), None);
        // First tick does a real check (STRIDE_INIT == 1) and reschedules; then
        // the next STRIDE_MIN-ish ticks are pure countdown. None trips.
        for _ in 0..(STRIDE_MIN + 8) {
            assert!(!p.over_budget());
        }
        assert!(p.tripped().is_none());
    }

    #[test]
    fn expired_deadline_trips_on_the_very_first_tick() {
        // Regression: a tiny loop with an already-expired deadline must decline
        // immediately, not run to completion before the first cold-path check.
        let past = Instant::now() - Duration::from_secs(1);
        let mut p = MemoryProbe::new(budget_ceiling(usize::MAX), Some(past));
        assert!(p.over_budget(), "expired deadline must trip on tick 1");
        assert_eq!(p.tripped(), Some(Trip::Deadline));
    }

    #[test]
    fn trips_and_latches_when_footprint_exceeds_ceiling() {
        let mut p = MemoryProbe::new(budget_ceiling(1000), None);
        let t0 = Instant::now();
        // First cold path is warm-up: memory is NOT evaluated yet.
        assert_eq!(p.evaluate(t0, Some(2000), Some(usize::MAX)), None);
        assert!(p.tripped().is_none());
        // Second cold path: over the ceiling ⇒ trip, reason Memory.
        let t1 = t0 + Duration::from_millis(10);
        assert_eq!(
            p.evaluate(t1, Some(2000), Some(usize::MAX)),
            Some(Trip::Memory)
        );
        assert_eq!(p.tripped(), Some(Trip::Memory));
        // Latched: even a now-healthy footprint keeps returning the same trip.
        assert_eq!(
            p.evaluate(t1, Some(1), Some(usize::MAX)),
            Some(Trip::Memory)
        );
    }

    #[test]
    fn warmup_defers_the_memory_decision_to_the_second_cold_path() {
        // Finding #4 regression: a starved host (host_free below floor) with a
        // TINY footprint must NOT trip on the first cold path (iteration 1) —
        // the old ~512-iteration cadence let such a small/fast run complete.
        // floor = 10_000, host_free = 1 (below floor), gate 0 (explorer-style).
        let mut p = MemoryProbe::new(
            MemoryBudget::from_thresholds(Some(usize::MAX), None, 10_000, 0),
            None,
        );
        let t0 = Instant::now();
        // Warm-up: floor NOT evaluated, no trip despite the starved host.
        assert_eq!(p.evaluate(t0, Some(64 * 1024), Some(1)), None);
        assert!(p.tripped().is_none());
        // Second cold path: now the floor fires.
        assert_eq!(
            p.evaluate(t0 + Duration::from_millis(10), Some(64 * 1024), Some(1)),
            Some(Trip::Memory)
        );
    }

    #[test]
    fn first_cold_path_does_not_extrapolate_a_runaway_stride() {
        // Finding #2 regression: a sub-microsecond construction-to-first-tick
        // gap must NOT inflate the stride (extrapolating 1 tick over ~0 elapsed
        // would clamp to STRIDE_MAX and defeat the deadline clamp). The warm-up
        // cold path fixes the countdown at STRIDE_MIN regardless.
        let mut p = MemoryProbe::new(budget_ceiling(usize::MAX), None);
        // A cold path an instant after construction (near-zero elapsed).
        assert_eq!(p.evaluate(p.last_cold, Some(1), Some(usize::MAX)), None);
        assert_eq!(
            p.countdown, STRIDE_MIN,
            "warm-up must not extrapolate a huge stride"
        );
    }

    #[test]
    fn trips_on_deadline_even_if_memory_is_fine() {
        let t0 = Instant::now();
        let deadline = t0; // already reached
        let mut p = MemoryProbe::new(budget_ceiling(usize::MAX), Some(deadline));
        assert_eq!(
            p.evaluate(t0, Some(1), Some(usize::MAX)),
            Some(Trip::Deadline)
        );
        assert_eq!(p.tripped(), Some(Trip::Deadline));
    }

    #[test]
    fn footprint_probe_failure_is_fail_soft() {
        let mut p = MemoryProbe::new(budget_ceiling(1000), None);
        let t0 = Instant::now();
        // No footprint reading ⇒ never trips on memory (only the deadline could).
        assert_eq!(
            p.evaluate(t0 + Duration::from_millis(100), None, None),
            None
        );
        assert!(p.tripped().is_none());
    }

    #[test]
    fn interval_tightens_toward_the_ceiling() {
        let far = interval_for_headroom(0, Some(1000));
        let mid = interval_for_headroom(500, Some(1000));
        let near = interval_for_headroom(990, Some(1000));
        assert!(far > mid && mid > near, "{far:?} {mid:?} {near:?}");
        assert!(far <= INTERVAL_FAR && near >= INTERVAL_NEAR);
        // No ceiling ⇒ far.
        assert_eq!(interval_for_headroom(9_999, None), INTERVAL_FAR);
    }

    #[test]
    fn stride_self_tunes_to_loop_speed() {
        let target = Duration::from_millis(50);
        // Fast loop: 1_000_000 ticks in 1ms ⇒ ~50M ticks/50ms, clamped to MAX.
        let fast = estimate_stride(1_000_000, Duration::from_millis(1), target);
        // Slow loop: 10 ticks in 100ms ⇒ ~5 ticks/50ms, clamped to MIN.
        let slow = estimate_stride(10, Duration::from_millis(100), target);
        assert_eq!(fast, STRIDE_MAX);
        assert_eq!(slow, STRIDE_MIN);
        // Middle: 100k ticks in 50ms ⇒ ~100k for a 50ms target.
        let mid = estimate_stride(100_000, Duration::from_millis(50), target);
        assert!(mid > STRIDE_MIN && mid < STRIDE_MAX);
        // Degenerate inputs ⇒ conservative MIN, no panic.
        assert_eq!(estimate_stride(0, target, target), STRIDE_MIN);
        assert_eq!(estimate_stride(100, Duration::ZERO, target), STRIDE_MIN);
    }

    #[test]
    fn deadline_clamp_shortens_stride_when_deadline_is_near() {
        let now = Instant::now();
        // rate: 1000 ticks in 10ms = 100 ticks/ms.
        let ticks = 1000;
        let elapsed = Duration::from_millis(10);
        // Far deadline (10s): no shortening below the estimate.
        let far = clamp_before_deadline(
            STRIDE_MAX,
            ticks,
            elapsed,
            now,
            now + Duration::from_secs(10),
        );
        // Near deadline (2ms): ~100 ticks/ms * 2ms / 2 = ~100, clamped to >=MIN.
        let near = clamp_before_deadline(
            STRIDE_MAX,
            ticks,
            elapsed,
            now,
            now + Duration::from_millis(2),
        );
        assert!(near < far || near == STRIDE_MIN);
        assert!(near >= STRIDE_MIN);
    }

    #[test]
    fn probe_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MemoryProbe>();
    }
}
