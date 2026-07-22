// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS profiling counters, accumulation, and report formatting.
//!
//! `BfsProfile` bundles the per-phase timers and counters used by the BFS
//! loop; `bfs_profile_lines` renders them to the enumeration-profile report.
//! The `ModelChecker::output_bfs_profile` associated function lives here too,
//! since it only depends on `BfsProfile` and `bfs_profile_lines`.

use crate::check::debug::profile_enum;
use crate::check::model_checker::ModelChecker;
use std::time::Instant;

/// Bundled BFS profiling counters and accumulator.
///
/// Used both as the accumulator during the BFS loop and as the snapshot
/// passed to `output_bfs_profile`. The `accum_*` and `count_*` methods
/// are no-ops when `do_profile` is false, avoiding scattered `if` checks.
///
/// Part of #2677: consolidated from 8 local variables in `run_bfs_loop`.
#[derive(Clone)]
pub(in crate::check) struct BfsProfile {
    pub do_profile: bool,
    pub start_time: Instant,
    pub succ_gen_us: u64,
    pub fingerprint_us: u64,
    pub dedup_us: u64,
    pub invariant_us: u64,
    pub jit_hits: u64,
    pub jit_misses: u64,
    pub total_successors: u64,
    pub new_states: u64,
    /// Part of #3990: arena allocation count (cumulative across resets).
    pub arena_allocs: u64,
    /// Part of #3990: arena bytes allocated (cumulative across resets).
    pub arena_bytes: u64,
    /// Part of #3990: arena reset count (number of BFS level boundaries).
    pub arena_resets: u64,
}

impl BfsProfile {
    /// Create a new accumulator. All counters start at zero.
    pub(in crate::check) fn new(start_time: Instant) -> Self {
        Self {
            do_profile: profile_enum(),
            start_time,
            succ_gen_us: 0,
            fingerprint_us: 0,
            dedup_us: 0,
            invariant_us: 0,
            jit_hits: 0,
            jit_misses: 0,
            total_successors: 0,
            new_states: 0,
            arena_allocs: 0,
            arena_bytes: 0,
            arena_resets: 0,
        }
    }

    /// Capture the current instant when profiling is enabled, else reuse start_time.
    #[inline(always)]
    pub(in crate::check) fn now(&self) -> Instant {
        if self.do_profile {
            Instant::now()
        } else {
            self.start_time
        }
    }

    /// Accumulate successor generation time from a captured instant.
    #[inline(always)]
    pub(in crate::check) fn accum_succ_gen(&mut self, t0: Instant) {
        if self.do_profile {
            self.succ_gen_us += t0.elapsed().as_micros() as u64;
        }
    }

    /// Accumulate fingerprinting time from a captured instant.
    #[inline(always)]
    pub(in crate::check) fn accum_fingerprint(&mut self, t0: Instant) {
        if self.do_profile {
            self.fingerprint_us += t0.elapsed().as_micros() as u64;
        }
    }

    /// Accumulate dedup check time from a captured instant.
    #[inline(always)]
    pub(in crate::check) fn accum_dedup(&mut self, t0: Instant) {
        if self.do_profile {
            self.dedup_us += t0.elapsed().as_micros() as u64;
        }
    }

    /// Record successors generated for this state.
    #[inline(always)]
    pub(in crate::check) fn count_successors(&mut self, n: usize) {
        if self.do_profile {
            self.total_successors += n as u64;
        }
    }

    /// Record one new (unseen) state discovered.
    #[inline(always)]
    pub(in crate::check) fn count_new_state(&mut self) {
        if self.do_profile {
            self.new_states += 1;
        }
    }

    /// Snapshot arena stats from the thread-local worker arena into this profile.
    ///
    /// Called at the end of BFS exploration (before profile output) so the arena
    /// allocation count, bytes, and reset count appear in the profile.
    ///
    /// Part of #3990: arena allocation metrics in BFS profile.
    pub(in crate::check) fn snapshot_arena_stats(&mut self) {
        if !self.do_profile {
            return;
        }
        crate::arena::with_worker_arena(|arena| {
            self.arena_allocs = arena.allocation_count() as u64;
            self.arena_bytes = arena.allocated_bytes() as u64;
            self.arena_resets = arena.reset_count() as u64;
        });
    }
}

pub(in crate::check::model_checker::run_helpers) fn bfs_profile_lines(
    total_us: u64,
    prof: &BfsProfile,
) -> Vec<String> {
    let other_us = total_us.saturating_sub(
        prof.succ_gen_us
            .saturating_add(prof.fingerprint_us)
            .saturating_add(prof.dedup_us)
            .saturating_add(prof.invariant_us),
    );
    let pct = |us: u64| -> f64 {
        if total_us > 0 {
            us as f64 / total_us as f64 * 100.0
        } else {
            0.0
        }
    };
    let mut lines = vec![
        "=== Enumeration Profile ===".to_string(),
        format!(
            "  Successor gen:   {:>8.3}s ({:>5.1}%)",
            prof.succ_gen_us as f64 / 1_000_000.0,
            pct(prof.succ_gen_us)
        ),
        format!(
            "  Fingerprinting:  {:>8.3}s ({:>5.1}%)",
            prof.fingerprint_us as f64 / 1_000_000.0,
            pct(prof.fingerprint_us)
        ),
        format!(
            "  Dedup check:     {:>8.3}s ({:>5.1}%)",
            prof.dedup_us as f64 / 1_000_000.0,
            pct(prof.dedup_us)
        ),
        format!(
            "  Invariant check: {:>8.3}s ({:>5.1}%)",
            prof.invariant_us as f64 / 1_000_000.0,
            pct(prof.invariant_us)
        ),
        format!(
            "  Other:           {:>8.3}s ({:>5.1}%)",
            other_us as f64 / 1_000_000.0,
            pct(other_us)
        ),
        "  ---".to_string(),
        format!("  Total:           {:>8.3}s", total_us as f64 / 1_000_000.0),
    ];
    if prof.new_states > 0 {
        lines.push(format!(
            "  Total successors: {} ({:.0}/state)",
            prof.total_successors,
            prof.total_successors as f64 / prof.new_states as f64
        ));
    } else {
        lines.push(format!(
            "  Total successors: {} (no new states)",
            prof.total_successors
        ));
    }
    lines.push(format!("  New states:       {}", prof.new_states));
    if prof.jit_hits > 0 || prof.jit_misses > 0 {
        lines.push(format!(
            "  JIT invariant:    hits={} misses={}",
            prof.jit_hits, prof.jit_misses
        ));
    }
    // Part of #3990: arena allocation stats.
    if prof.arena_allocs > 0 {
        lines.push(format!(
            "  Arena allocs:     {} ({:.1} MB, {} resets)",
            prof.arena_allocs,
            prof.arena_bytes as f64 / (1024.0 * 1024.0),
            prof.arena_resets,
        ));
    }
    lines
}

impl ModelChecker<'_> {
    /// Output BFS profiling results (no-trace mode).
    pub(in crate::check) fn output_bfs_profile(prof: &BfsProfile) {
        if !prof.do_profile {
            return;
        }
        let total_us = prof.start_time.elapsed().as_micros() as u64;
        for line in bfs_profile_lines(total_us, prof) {
            eprintln!("{line}");
        }
    }
}
