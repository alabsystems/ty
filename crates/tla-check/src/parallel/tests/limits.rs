// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Exploration limit tests — max states, max depth, constants, model values

use super::parse_module;
use super::*;
use crate::storage::{DiskFingerprintSet, FingerprintSet};
use crate::LimitType;
use std::sync::Arc;
use tempfile::tempdir;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_max_states_limit() {
    let _serial = super::acquire_interner_lock();
    // Unbounded counter that would run forever without limits
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);
    checker.set_max_states(5);

    let result = checker.check();

    match result {
        CheckResult::LimitReached { limit_type, stats } => {
            assert_eq!(limit_type, LimitType::States);
            // Part of #2123: CAS-based admission control ensures exact limit enforcement
            // even with multiple workers. No overshoot for deterministic specs.
            assert_eq!(
                stats.states_found, 5,
                "parallel max_states should match sequential exact-count contract"
            );
        }
        other => panic!("Expected LimitReached(States), got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_max_depth_limit() {
    let _serial = super::acquire_interner_lock();
    // Unbounded counter that would run forever without limits
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);
    checker.set_max_depth(3);

    let result = checker.check();

    match result {
        CheckResult::LimitReached { limit_type, stats } => {
            assert_eq!(limit_type, LimitType::Depth);
            // Should have explored depth 0, 1, 2, 3 (up to 4 states)
            assert!(stats.states_found >= 3);
            assert!(stats.max_depth <= 3);
        }
        other => panic!("Expected LimitReached(Depth), got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_invariant_found_before_limit() {
    let _serial = super::acquire_interner_lock();
    // Counter with invariant that will be violated before hitting limit
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallValue == x < 3
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["SmallValue".to_string()],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);
    checker.set_max_states(100); // High limit that won't be reached

    let result = checker.check();

    match result {
        CheckResult::InvariantViolation {
            invariant, stats, ..
        } => {
            assert_eq!(invariant, "SmallValue");
            // Should find violation at x=3 before hitting 100 state limit
            assert!(stats.states_found < 100);
        }
        other => panic!("Expected InvariantViolation, got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_success_within_limits() {
    let _serial = super::acquire_interner_lock();
    // Bounded counter that terminates naturally
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["InRange".to_string()],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);
    checker.set_max_states(100);
    checker.set_max_depth(100);

    let result = checker.check();

    match result {
        CheckResult::Success(stats) => {
            // Should complete naturally with 4 states
            assert_eq!(stats.states_found, 4);
        }
        other => panic!("Expected Success, got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_depth_tracking() {
    let _serial = super::acquire_interner_lock();
    // Bounded counter to verify depth tracking
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 5 /\ x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);

    let result = checker.check();

    match result {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 6); // x=0,1,2,3,4,5
            assert_eq!(stats.max_depth, 5); // 0->1->2->3->4->5
        }
        other => panic!("Expected Success, got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_limits_single_worker() {
    let _serial = super::acquire_interner_lock();
    // Test limits with single worker for deterministic behavior
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 1);
    checker.set_deadlock_check(false);
    checker.set_max_states(10);

    let result = checker.check();

    match result {
        CheckResult::LimitReached { limit_type, stats } => {
            assert_eq!(limit_type, LimitType::States);
            assert_eq!(stats.states_found, 10);
        }
        other => panic!("Expected LimitReached(States), got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_with_constants() {
    let _serial = super::acquire_interner_lock();
    // Test that parallel checker correctly binds constants from config
    let src = r#"
---- MODULE WithConstants ----
CONSTANT N
VARIABLE x
Init == x = N
Next == x < N + 3 /\ x' = x + 1
Bounded == x < N + 5
====
"#;
    let module = parse_module(src);

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["Bounded".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "N".to_string(),
        crate::config::ConstantValue::Value("5".to_string()),
    );

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);

    let result = checker.check();

    match result {
        CheckResult::Success(stats) => {
            // N=5, starts at x=5, Next enabled while x < 8
            // States: x=5, x=6, x=7, x=8 (then x < 8 is false, deadlock disabled)
            assert_eq!(stats.states_found, 4);
            assert_eq!(stats.initial_states, 1);
        }
        other => panic!("Expected success with constants, got: {:?}", other),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parallel_with_model_value_set() {
    let _serial = super::acquire_interner_lock();
    // Test that parallel checker correctly binds model value set constants
    let src = r#"
---- MODULE WithModelValues ----
CONSTANT Procs
VARIABLE current
Init == current \in Procs
Next == current' \in Procs
AlwaysProc == current \in Procs
====
"#;
    let module = parse_module(src);

    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["AlwaysProc".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "Procs".to_string(),
        crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);

    let result = checker.check();

    match result {
        CheckResult::Success(stats) => {
            // Should find 2 states: current=p1, current=p2
            assert_eq!(stats.states_found, 2);
            assert_eq!(stats.initial_states, 2);
        }
        other => panic!("Expected success with model values, got: {:?}", other),
    }
}

/// Part of #2751: verify that the parallel checker returns `LimitReached(Memory)`
/// when the configured memory limit is below the process RSS.
///
/// Uses a 1-byte limit so any running process immediately exceeds the 95%
/// critical threshold on the first `run_periodic_maintenance` poll.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_parallel_memory_limit_triggers_limit_reached() {
    let _serial = super::acquire_interner_lock();
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);
    // 1 byte: any process exceeds this — triggers Critical immediately.
    checker.set_memory_limit(1);

    let result = checker.check();

    match result {
        CheckResult::LimitReached { limit_type, stats } => {
            assert_eq!(limit_type, LimitType::Memory);
            // Some states were explored before the periodic check fired.
            assert!(
                stats.states_found > 0,
                "should have explored at least 1 state before memory stop"
            );
        }
        other => panic!("Expected LimitReached(Memory), got: {:?}", other),
    }
}

/// N8 (soundness): a wall-clock time budget must fail CLOSED. A run that the
/// deadline interrupts before the state space is exhausted must NOT report
/// `Success` — it produces a partial `LimitReached { Time }` instead.
///
/// Regression: a nanosecond budget on a multi-worker run over a state space
/// larger than the per-worker `DEADLINE_CHECK_INTERVAL` previously returned
/// `Success` with a partial `states_found`, because the parallel
/// `on_deadline_exceeded` exited silently without sending any `WorkerResult`,
/// so the coordinator saw no terminal outcome and defaulted to `Success` — a
/// false PASS on a partially explored space.
///
/// The spec is a SINGLE-ACTION linear counter (`x` from 0 to 20,000), not a
/// multi-action grid. This matters: a grid of independent actions (each writing
/// a disjoint variable) is collapsed by partial-order reduction to a single
/// interleaving — for a 301×301 grid, POR shrinks 90,601 states down to the
/// ~601-state diagonal, which is *below* the 4,096-state `DEADLINE_CHECK_INTERVAL`,
/// so no worker ever reaches a deadline poll and the run legitimately finishes
/// (masking the regression this test guards). A linear counter has exactly one
/// enabled action per state, so POR cannot reduce it: all 20,001 states are
/// explored, guaranteeing at least one worker crosses the 4,096-state poll and
/// trips the 1 ns backstop well before exhausting the space.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_parallel_time_budget_yields_limit_reached_not_success() {
    let _serial = super::acquire_interner_lock();
    let src = r#"
---- MODULE LinCount ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 20000 /\ x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let mut checker = ParallelChecker::new(&module, &config, 2);
    checker.set_deadlock_check(false);
    // 1 ns: the deadline is already in the past by the first per-worker poll.
    checker.set_time_budget(std::time::Duration::from_nanos(1));

    let result = checker.check();

    match result {
        CheckResult::LimitReached { limit_type, stats } => {
            assert_eq!(
                limit_type,
                LimitType::Time,
                "time-budget interruption must report LimitReached(Time)"
            );
            // The exploration was interrupted mid-flight: strictly partial.
            assert!(
                stats.states_found < 20_001,
                "time backstop must fire before the 20,001-state space is exhausted, \
                 got states_found={}",
                stats.states_found
            );
        }
        CheckResult::Success(stats) => panic!(
            "N8 regression: time-budget-interrupted run reported Success on a partially \
             explored space (states_found={})",
            stats.states_found
        ),
        other => panic!("Expected LimitReached(Time), got: {:?}", other),
    }
}

/// Part of #3282: verify that the parallel checker returns `LimitReached(Disk)`
/// when disk-backed fingerprint storage exceeds the configured disk budget.
///
/// Uses a tiny `DiskFingerprintSet` primary capacity so the unbounded counter
/// spec spills fingerprints to disk almost immediately. The 1-byte limit then
/// forces the periodic maintenance loop to stop exploration once `disk_count`
/// becomes non-zero.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn test_parallel_disk_limit_triggers_limit_reached_with_disk_storage() {
    let _serial = super::acquire_interner_lock();
    let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
    let module = parse_module(src);

    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![],
        ..Default::default()
    };

    let backing_dir = tempdir().expect("create temp dir for disk-backed storage");
    let storage: Arc<dyn FingerprintSet> = Arc::new(
        DiskFingerprintSet::new(1, backing_dir.path().to_path_buf())
            .expect("create disk-backed fingerprint storage"),
    );

    // Single-worker parallel mode keeps the disk-eviction path deterministic
    // while still exercising the parallel checker's periodic maintenance loop.
    let mut checker = ParallelChecker::new(&module, &config, 1);
    checker.set_deadlock_check(false);
    checker.set_fingerprint_storage(storage);
    // 1 byte: the first evicted fingerprint makes `disk_count * 8` exceed the limit.
    checker.set_disk_limit(1);

    let result = checker.check();

    match result {
        CheckResult::LimitReached { limit_type, stats } => {
            assert_eq!(limit_type, LimitType::Disk);
            assert!(
                stats.states_found > 0,
                "should have explored at least 1 state before disk stop"
            );
            assert!(
                stats.storage_stats.disk_count > 0,
                "disk-backed storage should have persisted fingerprints before stopping"
            );
        }
        other => panic!("Expected LimitReached(Disk), got: {:?}", other),
    }
}
