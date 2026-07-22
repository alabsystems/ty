// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Parallel fingerprint-collision diagnostics (#2841).

use super::{ArrayState, Fingerprint, FxBuildHasher, ParallelChecker};
use crate::collision_detection::{CollisionCheckMode, CollisionStats};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) struct CollisionDebugConfig {
    pub(super) track_seen_tlc_fp_dedup: bool,
    pub(super) seen_tlc_fp_dedup_collision_limit: usize,
    pub(super) track_internal_fp_collision: bool,
    pub(super) internal_fp_collision_limit: usize,
}

impl CollisionDebugConfig {
    pub(super) fn from_env() -> Option<Self> {
        let track_seen_tlc_fp_dedup = crate::check::debug::debug_seen_tlc_fp_dedup();
        let track_internal_fp_collision = crate::check::debug::debug_internal_fp_collision();
        if !track_seen_tlc_fp_dedup && !track_internal_fp_collision {
            return None;
        }

        Some(Self {
            track_seen_tlc_fp_dedup,
            seen_tlc_fp_dedup_collision_limit:
                crate::check::debug::debug_seen_tlc_fp_dedup_collision_limit(),
            track_internal_fp_collision,
            internal_fp_collision_limit: crate::check::debug::debug_internal_fp_collision_limit(),
        })
    }
}

pub(crate) struct ParallelCollisionDiagnostics {
    seen_tlc_fp_dedup: Option<DashMap<u64, Fingerprint, FxBuildHasher>>,
    seen_tlc_fp_dedup_collisions: AtomicU64,
    seen_tlc_fp_dedup_collision_limit: usize,
    internal_fp_collision: Option<DashMap<Fingerprint, u64, FxBuildHasher>>,
    internal_fp_collisions: AtomicU64,
    internal_fp_collision_limit: usize,
    /// Store/verify one state in every `sample_interval` (`1` ⇒ every state,
    /// which is "full" checking). Set from `CollisionCheckMode::Sampling`.
    sample_interval: u64,
    /// Monotonic counter for the sampling gate.
    sample_counter: AtomicU64,
    /// States actually stored+checked (after the sampling gate), for reporting.
    states_checked: AtomicU64,
}

impl ParallelCollisionDiagnostics {
    pub(super) fn from_env(shard_amount: usize) -> Option<Arc<Self>> {
        CollisionDebugConfig::from_env().map(|config| Arc::new(Self::new(config, shard_amount)))
    }

    /// Build the diagnostics for the user-facing `--collision-check` mode.
    ///
    /// The parallel path checks the SOUNDNESS-relevant event — two distinct
    /// states (distinct TLC canonical fingerprints) that hash to the same
    /// internal 64-bit fingerprint, which would make a worker silently skip a
    /// reachable state. `Full` checks every admitted state; `Sampling{interval}`
    /// checks one in every `interval`; `None` ⇒ no diagnostics.
    pub(super) fn for_collision_mode(
        mode: CollisionCheckMode,
        shard_amount: usize,
    ) -> Option<Self> {
        let interval = match mode {
            CollisionCheckMode::None => return None,
            CollisionCheckMode::Full => 1,
            CollisionCheckMode::Sampling { interval } => interval.max(1),
        };
        let mut diag = Self::new(
            CollisionDebugConfig {
                track_seen_tlc_fp_dedup: false,
                seen_tlc_fp_dedup_collision_limit: 0,
                track_internal_fp_collision: true,
                internal_fp_collision_limit: 100,
            },
            shard_amount,
        );
        diag.sample_interval = interval;
        Some(diag)
    }

    pub(super) fn new(config: CollisionDebugConfig, shard_amount: usize) -> Self {
        Self {
            seen_tlc_fp_dedup: config.track_seen_tlc_fp_dedup.then(|| {
                DashMap::with_hasher_and_shard_amount(FxBuildHasher::default(), shard_amount)
            }),
            seen_tlc_fp_dedup_collisions: AtomicU64::new(0),
            seen_tlc_fp_dedup_collision_limit: config.seen_tlc_fp_dedup_collision_limit,
            internal_fp_collision: config.track_internal_fp_collision.then(|| {
                DashMap::with_hasher_and_shard_amount(FxBuildHasher::default(), shard_amount)
            }),
            internal_fp_collisions: AtomicU64::new(0),
            internal_fp_collision_limit: config.internal_fp_collision_limit,
            sample_interval: 1,
            sample_counter: AtomicU64::new(0),
            states_checked: AtomicU64::new(0),
        }
    }

    pub(super) fn record_state(&self, fp: Fingerprint, array_state: &ArrayState, depth: usize) {
        // Sampling gate: with interval N > 1, only one state in every N is
        // materialized+checked. Computing the TLC fingerprint below is the
        // expensive part, so gate before it.
        if self.sample_interval > 1 {
            let n = self.sample_counter.fetch_add(1, Ordering::Relaxed);
            if !n.is_multiple_of(self.sample_interval) {
                return;
            }
        }
        self.states_checked.fetch_add(1, Ordering::Relaxed);
        let tlc_fp = if self.seen_tlc_fp_dedup.is_some() || self.internal_fp_collision.is_some() {
            let vals = array_state.materialize_values();
            crate::check::debug::tlc_fp_for_state_values(&vals).ok()
        } else {
            None
        };

        if let (Some(seen), Some(tlc_fp)) = (&self.seen_tlc_fp_dedup, tlc_fp) {
            match seen.entry(tlc_fp) {
                Entry::Vacant(entry) => {
                    entry.insert(fp);
                }
                Entry::Occupied(entry) => {
                    let first = *entry.get();
                    if first != fp {
                        let collisions = self
                            .seen_tlc_fp_dedup_collisions
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        if collisions <= self.seen_tlc_fp_dedup_collision_limit as u64 {
                            eprintln!(
                                "Warning: TLC FP dedup collision tlc={} first={:016x} now={:016x} depth={}",
                                crate::check::debug::fmt_tlc_fp(tlc_fp),
                                first.0,
                                fp.0,
                                depth
                            );
                        }
                    }
                }
            }
        }

        if let (Some(seen), Some(tlc_fp)) = (&self.internal_fp_collision, tlc_fp) {
            match seen.entry(fp) {
                Entry::Vacant(entry) => {
                    entry.insert(tlc_fp);
                }
                Entry::Occupied(entry) => {
                    let first_tlc = *entry.get();
                    if first_tlc != tlc_fp {
                        let collisions =
                            self.internal_fp_collisions.fetch_add(1, Ordering::Relaxed) + 1;
                        if collisions <= self.internal_fp_collision_limit as u64 {
                            eprintln!(
                                "Warning: internal FP collision internal={:016x} first_tlc={} now_tlc={} depth={}",
                                fp.0,
                                crate::check::debug::fmt_tlc_fp(first_tlc),
                                crate::check::debug::fmt_tlc_fp(tlc_fp),
                                depth
                            );
                        }
                    }
                }
            }
        }
    }

    pub(super) fn fp_dedup_collisions(&self) -> u64 {
        self.seen_tlc_fp_dedup_collisions.load(Ordering::Relaxed)
    }

    pub(super) fn internal_fp_collisions(&self) -> u64 {
        self.internal_fp_collisions.load(Ordering::Relaxed)
    }

    /// Snapshot as the shared [`CollisionStats`] for the run report. The
    /// soundness-relevant count is internal-fp collisions (distinct states that
    /// hash to the same internal fingerprint).
    pub(super) fn collision_stats(&self) -> CollisionStats {
        CollisionStats {
            states_sampled: self.states_checked.load(Ordering::Relaxed),
            collisions_detected: self.internal_fp_collisions(),
            // Per-collision details are surfaced via the stderr warnings above;
            // the report carries the aggregate count.
            collision_details: Vec::new(),
        }
    }
}

impl ParallelChecker {
    pub(super) fn fp_dedup_collisions(&self) -> u64 {
        self.collision_diagnostics
            .as_ref()
            .map_or(0, |diag| diag.fp_dedup_collisions())
    }

    pub(super) fn internal_fp_collisions(&self) -> u64 {
        self.collision_diagnostics
            .as_ref()
            .map_or(0, |diag| diag.internal_fp_collisions())
    }

    /// The user-facing collision-check mode configured via `--collision-check`.
    pub(super) fn collision_check_mode(&self) -> CollisionCheckMode {
        self.collision_check_mode
    }

    /// Collision statistics for the run report (empty when detection is off).
    pub(super) fn collision_check_stats(&self) -> CollisionStats {
        self.collision_diagnostics
            .as_ref()
            .map_or_else(CollisionStats::default, |diag| diag.collision_stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    fn state(values: &[i64]) -> ArrayState {
        ArrayState::from_values(values.iter().copied().map(Value::int).collect())
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_parallel_collision_counters_track_tlc_fp_dedup() {
        let diagnostics = ParallelCollisionDiagnostics::new(
            CollisionDebugConfig {
                track_seen_tlc_fp_dedup: true,
                seen_tlc_fp_dedup_collision_limit: 0,
                track_internal_fp_collision: false,
                internal_fp_collision_limit: 0,
            },
            4,
        );
        let arr = state(&[1, 2]);

        diagnostics.record_state(Fingerprint(0x1111), &arr, 0);
        diagnostics.record_state(Fingerprint(0x2222), &arr, 1);

        assert_eq!(diagnostics.fp_dedup_collisions(), 1);
        assert_eq!(diagnostics.internal_fp_collisions(), 0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_parallel_collision_counters_track_internal_fp() {
        let diagnostics = ParallelCollisionDiagnostics::new(
            CollisionDebugConfig {
                track_seen_tlc_fp_dedup: false,
                seen_tlc_fp_dedup_collision_limit: 0,
                track_internal_fp_collision: true,
                internal_fp_collision_limit: 0,
            },
            4,
        );

        diagnostics.record_state(Fingerprint(0x1111), &state(&[1]), 0);
        diagnostics.record_state(Fingerprint(0x1111), &state(&[2]), 1);

        assert_eq!(diagnostics.fp_dedup_collisions(), 0);
        assert_eq!(diagnostics.internal_fp_collisions(), 1);
    }

    #[test]
    fn test_for_collision_mode_none_is_disabled() {
        assert!(
            ParallelCollisionDiagnostics::for_collision_mode(CollisionCheckMode::None, 4).is_none()
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_for_collision_mode_full_detects_internal_fp_collision() {
        // The user-facing `--collision-check full` mode must catch the soundness
        // event: two DISTINCT states hashing to the same internal fingerprint.
        let diag = ParallelCollisionDiagnostics::for_collision_mode(CollisionCheckMode::Full, 4)
            .expect("full mode installs diagnostics");
        diag.record_state(Fingerprint(0x1111), &state(&[1]), 0);
        diag.record_state(Fingerprint(0x1111), &state(&[2]), 1); // collision!

        assert_eq!(diag.internal_fp_collisions(), 1);
        let stats = diag.collision_stats();
        assert_eq!(stats.collisions_detected, 1);
        assert_eq!(stats.states_sampled, 2);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_for_collision_mode_sampling_honors_interval() {
        // Sampling:2 checks one state in every two (n=0 and n=2 of 0..4).
        let diag = ParallelCollisionDiagnostics::for_collision_mode(
            CollisionCheckMode::Sampling { interval: 2 },
            4,
        )
        .expect("sampling mode installs diagnostics");
        for i in 0..4u64 {
            diag.record_state(Fingerprint(i), &state(&[i as i64]), 0);
        }
        assert_eq!(diag.collision_stats().states_sampled, 2);
    }
}
