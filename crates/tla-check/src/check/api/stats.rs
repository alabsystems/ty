// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Model checking statistics, progress reporting, and callback types.
//!
//! Split from `api.rs` for file-size hygiene (Part of #3427).

use crate::collision_detection::{CollisionCheckMode, CollisionStats};
use crate::coverage::CoverageStats;
use crate::storage::StorageStats;

/// Statistics from symmetry reduction during model checking.
///
/// Tracks the effectiveness of symmetry-based state space reduction,
/// including permutation group size, cache performance, and reduction factor.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SymmetryReductionStats {
    /// Number of permutations in the symmetry group (0 if no symmetry).
    pub permutation_count: usize,
    /// Number of independent symmetric model value sets detected.
    /// For example, `Permutations(A) \cup Permutations(B)` yields 2 groups.
    pub symmetry_groups: usize,
    /// Names of the symmetric model value sets (e.g., `["Procs", "Values"]`).
    pub group_names: Vec<String>,
    /// Total fingerprint computations that used the symmetry fp_cache (hits).
    pub fp_cache_hits: u64,
    /// Total fingerprint computations that required full canonical computation (misses).
    pub fp_cache_misses: u64,
    /// Number of states that were identified as non-canonical and folded into
    /// existing canonical representatives during BFS exploration.
    /// This is `states_without_symmetry - states_with_symmetry`.
    pub states_folded: u64,
    /// Symmetry reduction factor: ratio of unreduced to reduced state count.
    /// Value of 1.0 means no reduction; N! for full symmetry on N elements.
    /// Only meaningful after exploration completes.
    pub reduction_factor: f64,
    /// Whether symmetry was auto-detected (true) or explicitly configured (false).
    pub auto_detected: bool,
}

/// Statistics from Partial Order Reduction (POR) during model checking.
///
/// Part of #3993: Tracks the effectiveness of compile-time action independence
/// analysis and ample set computation for state space reduction.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PorReductionStats {
    /// Number of actions detected in the Next relation.
    pub action_count: usize,
    /// Number of independent action pairs detected by static analysis.
    pub independent_pairs: usize,
    /// Total number of action pairs (action_count choose 2).
    pub total_pairs: usize,
    /// Number of states where POR achieved reduction (ample set < enabled set).
    pub states_reduced: u64,
    /// Total number of states processed through the POR ample set computation.
    pub states_processed: u64,
    /// Total number of action evaluations skipped due to POR reduction.
    pub actions_skipped: u64,
    /// Whether POR was auto-detected (vs explicitly enabled with --por).
    ///
    /// Part of #3993: auto-POR runs the independence analysis automatically
    /// and enables POR when independent pairs are found.
    pub auto_detected: bool,
}

/// Statistics for PROPERTY checks that run outside ordinary invariant checks.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PropertyCheckStats {
    /// Number of transitions that entered the promoted implied-action checker.
    pub implied_action_transition_checks: u64,
    /// Number of promoted implied-action terms evaluated across those transitions.
    pub implied_action_term_evals: u64,
    /// Wall-clock time spent evaluating promoted implied-action PROPERTY terms.
    pub implied_action_time_ns: u64,
}

impl PropertyCheckStats {
    /// Returns `true` if any implied-action PROPERTY counter is non-zero.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.implied_action_transition_checks > 0
            || self.implied_action_term_evals > 0
            || self.implied_action_time_ns > 0
    }

    /// Implied-action PROPERTY checking time, in seconds.
    #[must_use]
    pub fn implied_action_time_seconds(&self) -> f64 {
        self.implied_action_time_ns as f64 / 1_000_000_000.0
    }
}

/// Statistics from model checking
///
/// Part of #3005: `collision_probability()` computes the fingerprint collision
/// probability estimate (birthday paradox) matching TLC's `reportSuccess`.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CheckStats {
    /// Number of distinct states explored
    pub states_found: usize,
    /// Number of initial states
    pub initial_states: usize,
    /// Number of initial states generated before state constraints and
    /// fingerprint deduplication are applied.
    ///
    /// TLC increments `numberOfInitialStates` in `DoInitFunctor.addElement`
    /// before legality/model-constraint checks. Keep this separate from
    /// `initial_states`, whose established TY meaning is the number admitted
    /// to the initial BFS frontier.
    pub raw_initial_states_generated: usize,
    /// Maximum queue depth (BFS frontier)
    pub max_queue_depth: usize,
    /// Number of transitions examined
    pub transitions: usize,
    /// Number of successors generated before state/action constraints or
    /// exploration reductions are applied.
    ///
    /// TLC increments its worker `statesGenerated` counter at this boundary,
    /// before checking `isGoodState`, state constraints, or action
    /// constraints. Keep this separate from `transitions`, whose established
    /// TY meaning is the post-filter transition work consumed by the checker.
    pub raw_successors_generated: usize,
    /// Maximum BFS depth reached
    pub max_depth: usize,
    /// Detected action names from Next relation (top-level disjuncts)
    pub detected_actions: Vec<String>,
    /// Detected action location IDs from Next relation (e.g., "7:10-13")
    pub detected_action_ids: Vec<String>,
    /// Optional coverage statistics (enabled via `ModelChecker::set_collect_coverage(true)`)
    pub coverage: Option<CoverageStats>,
    /// Number of dequeued fingerprints not found in the seen map.
    /// Non-zero indicates fingerprint collisions or a state-tracking bug.
    pub phantom_dequeues: usize,
    /// Number of action-level guard evaluation errors suppressed as "not a guard".
    /// Non-zero means the state count may be incorrect.
    pub suppressed_guard_errors: usize,
    /// Number of trace reconstruction calls triggered during checking.
    /// Part of #1121: Non-zero only when TLCExt!Trace is used by configured formulas.
    pub trace_reconstructions: usize,
    /// Number of TLC FP dedup collisions (same TLC FP, different internal FP).
    /// Non-zero indicates potential over-exploration. Requires TY_DEBUG_SEEN_TLCFP_DEDUP=1.
    /// Part of #2841: available in release builds.
    pub fp_dedup_collisions: u64,
    /// Number of internal FP collisions (same internal FP, different TLC FP).
    /// Non-zero indicates potential under-exploration. Requires TY_DEBUG_INTERNAL_FP_COLLISION=1.
    /// Part of #2841: available in release builds.
    pub internal_fp_collisions: u64,
    /// Fingerprint storage backend counters (memory/disk residency, I/O, evictions).
    /// Part of #2665: trait-level storage statistics for observability.
    pub storage_stats: StorageStats,
    /// Collision detection mode used during this run.
    pub collision_check_mode: CollisionCheckMode,
    /// Collision detection statistics (populated when collision checking is active).
    pub collision_check_stats: CollisionStats,
    /// Symmetry reduction statistics (populated when SYMMETRY is configured or auto-detected).
    pub symmetry_reduction: SymmetryReductionStats,
    /// Partial Order Reduction statistics (populated when --por is enabled).
    /// Part of #3993.
    pub por_reduction: PorReductionStats,
    /// PROPERTY-check timing and counters.
    ///
    /// Refinement properties like `M!Spec` may be promoted into BFS-time
    /// implied-action checks; this separates that cost from ordinary BFS.
    pub property_check: PropertyCheckStats,
    /// Summarizer-ready backend capability/admission evidence for this run.
    ///
    /// JSON/JSONL output serializes this under `backend_capability_report`, a
    /// key consumed by the `ty-mcc-summarize-evidence` binary.
    pub backend_capability_report: Option<serde_json::Value>,
    /// Engine-provenance record: which execution tier actually ran this check
    /// (interpreter / trust-cg per-action callout / trust-cg native-fused /
    /// parallel BFS / gpu), whether an AUTO tier-up hot-swap occurred mid-run, and the
    /// value-action VM engagement summary. Benchmark harnesses consume this so
    /// every measured row can attribute its result to the engine that produced
    /// it; a row without provenance cannot support a supremacy claim.
    pub engine_provenance: Option<serde_json::Value>,
    /// Default-on, non-fatal vacuity WARNINGs (V2 dead-actions, V3
    /// vacuously-true invariants). Computed unconditionally at finalize time.
    /// The CLI prints these; `--strict-vacuity` promotes them to exit 3 and
    /// `--allow-vacuous=<class>` downgrades a named class.
    ///
    /// Design: TRUST_VACUITY_GATE §1.A (V2/V3).
    pub vacuity_warnings: Vec<crate::vacuity::VacuityWarning>,
}

impl CheckStats {
    /// Total states generated (including duplicates), matching TLC's `getStatesGenerated()`.
    ///
    /// In TLC: `numberOfInitialStates + sum(worker.statesGenerated)`.
    /// In TY: `raw_initial_states_generated + raw_successors_generated`.
    ///
    /// Part of #3005.
    #[must_use]
    pub fn states_generated(&self) -> usize {
        self.raw_initial_states_generated
            .checked_add(self.raw_successors_generated)
            .expect("raw generated-state total overflowed usize")
    }

    /// Optimistic fingerprint collision probability estimate.
    ///
    /// Uses TLC's exact formula: `n * (g - n) / 2^64` where `n = states_found`
    /// (distinct states) and `g = states_generated` (total including duplicates).
    /// The quantity `(g - n)` counts the number of fingerprint lookups that found
    /// an existing state — each of which could have been a collision.
    ///
    /// TLC reference: `AbstractChecker.calculateOptimisticProbability`
    /// (`tlatools/org.lamport.tlatools/src/tlc2/tool/AbstractChecker.java:205`).
    ///
    /// Part of #3005.
    #[must_use]
    pub fn collision_probability(&self) -> f64 {
        let n = self.states_found as f64;
        let g = self.states_generated() as f64;
        // TLC formula: n * (g - n) / 2^64
        // When g == n (no duplicate encounters), probability is 0.
        n * (g - n) / (2.0_f64).powi(64)
    }

    /// Format collision probability as a string matching TLC's `ProbabilityToString`.
    ///
    /// TLC uses 2 significant digits in scientific notation (e.g., "1.2E-14").
    /// Returns "0.0" for zero probability.
    ///
    /// Part of #3005.
    #[must_use]
    pub fn collision_probability_string(&self) -> String {
        let prob = self.collision_probability();
        if prob == 0.0 {
            "0.0".to_string()
        } else {
            // 2 significant figures in scientific notation, matching TLC's MathContext(2)
            format!("{:.1E}", prob)
        }
    }
}

/// Progress information reported during model checking
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Progress {
    /// Number of distinct states found so far
    pub states_found: usize,
    /// Current BFS depth being explored
    pub current_depth: usize,
    /// Current queue size (BFS frontier)
    pub queue_size: usize,
    /// Number of transitions explored so far
    pub transitions: usize,
    /// Seconds elapsed since model checking started
    pub elapsed_secs: f64,
    /// States explored per second (0.0 if elapsed is 0)
    pub states_per_sec: f64,
    /// Resident set size in bytes, if available (Part of #2751).
    pub memory_usage_bytes: Option<u64>,
    /// Estimated total reachable states (updated as more levels complete).
    pub estimated_total_states: Option<usize>,
    /// Confidence in the estimate (0.0..1.0).
    pub estimate_confidence: Option<f64>,
    /// Compact human-readable estimate string (e.g. "~1.2M est. (+/-15%)").
    pub estimate_display: Option<String>,
}

/// Initial-state enumeration information reported once after Init completes.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct InitProgress {
    /// Number of initial states generated (TLC-style "states generated").
    pub states_generated: usize,
    /// Number of distinct initial states after deduplication (TLC-style "distinct states").
    pub distinct_states: usize,
}

/// Type alias for progress callback function
pub type ProgressCallback = Box<dyn Fn(&Progress) + Send + Sync>;

/// Type alias for init progress callback (called once after initial states are computed).
pub type InitProgressCallback = Box<dyn Fn(&InitProgress) + Send + Sync>;

#[cfg(test)]
mod collision_probability_tests {
    use super::*;

    /// Part of #3005: collision probability is zero for 0 states.
    #[test]
    #[allow(clippy::float_cmp)]
    fn zero_states() {
        // Exact comparison is safe: 0.0 * anything == 0.0 in IEEE 754.
        let stats = CheckStats::default();
        assert_eq!(stats.collision_probability(), 0.0);
        assert_eq!(stats.collision_probability_string(), "0.0");
    }

    /// Part of #3005: collision probability is zero when no duplicate encounters.
    /// When states_generated == states_found, every generated state was new,
    /// so (g - n) = 0 and the probability is 0.
    #[test]
    #[allow(clippy::float_cmp)]
    fn no_duplicates() {
        // Exact comparison is safe: (g - n) == 0.0, so n * 0.0 / 2^64 == 0.0 in IEEE 754.
        let stats = CheckStats {
            states_found: 100,
            initial_states: 1,
            raw_initial_states_generated: 1,
            transitions: 99,
            raw_successors_generated: 99,
            ..Default::default()
        };
        // states_generated = 1 + 99 = 100, same as states_found
        assert_eq!(stats.states_generated(), 100);
        assert_eq!(stats.collision_probability(), 0.0);
        assert_eq!(stats.collision_probability_string(), "0.0");
    }

    /// Part of #3005: collision probability with duplicate state encounters.
    /// TLC formula: n * (g - n) / 2^64
    #[test]
    fn with_duplicates() {
        let stats = CheckStats {
            states_found: 1000,
            initial_states: 1,
            raw_initial_states_generated: 1,
            // 2000 transitions total, 999 new states + 1001 duplicate encounters
            transitions: 2000,
            raw_successors_generated: 2000,
            ..Default::default()
        };
        // states_generated = 1 + 2000 = 2001
        // prob = 1000 * (2001 - 1000) / 2^64 = 1000 * 1001 / 2^64
        assert_eq!(stats.states_generated(), 2001);
        let prob = stats.collision_probability();
        let expected = 1000.0 * 1001.0 / (2.0_f64).powi(64);
        assert!(
            (prob - expected).abs() < 1e-25,
            "prob={prob}, expected={expected}"
        );
    }

    /// Part of #3005: collision probability for 10M distinct states, ~20M generated.
    /// Realistic scenario: BFS with moderate revisitation.
    #[test]
    fn ten_million_states() {
        let stats = CheckStats {
            states_found: 10_000_000,
            initial_states: 1,
            raw_initial_states_generated: 1,
            transitions: 20_000_000,
            raw_successors_generated: 20_000_000,
            ..Default::default()
        };
        // states_generated = 20_000_001, g - n = 10_000_001
        // prob = 10M * 10M / 2^64 ≈ 5.42e-6
        let prob = stats.collision_probability();
        assert!(prob > 5.0e-6 && prob < 6.0e-6, "prob={prob}");
    }

    /// Part of #3005: collision probability string format matches TLC (scientific notation).
    #[test]
    fn string_format() {
        let stats = CheckStats {
            states_found: 10_000_000,
            initial_states: 1,
            raw_initial_states_generated: 1,
            transitions: 20_000_000,
            raw_successors_generated: 20_000_000,
            ..Default::default()
        };
        let s = stats.collision_probability_string();
        // Should be in scientific notation like "5.4E-6"
        assert!(s.contains('E'), "expected scientific notation, got: {s}");
    }

    /// Part of #3005: states_generated matches raw initial-state plus raw
    /// successor generation, independently of admitted initial-state and
    /// transition accounting.
    #[test]
    fn states_generated_is_sum() {
        let stats = CheckStats {
            initial_states: 3,
            raw_initial_states_generated: 5,
            transitions: 42,
            raw_successors_generated: 37,
            ..Default::default()
        };
        assert_eq!(stats.states_generated(), 42);
        assert_eq!(stats.initial_states, 3);
        assert_eq!(stats.transitions, 42);
    }

    #[test]
    #[should_panic(expected = "raw generated-state total overflowed usize")]
    fn states_generated_fails_closed_on_overflow() {
        let stats = CheckStats {
            raw_initial_states_generated: usize::MAX,
            raw_successors_generated: 1,
            ..Default::default()
        };
        let _ = stats.states_generated();
    }
}
