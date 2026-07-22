// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! On-device counterexample-trace reconstruction for the BFS engine.
//!
//! A one-slot bounded counter `x' = x + 1` (0..=5) with the invariant `x < 3`.
//! The reachable set is {0,1,2,3,4,5}; the invariant first fails at x=3, whose
//! unique init->bad path is [0,1,2,3]. With `trace_on_violation`, the engine
//! must return exactly that path by walking device-side parent pointers.

use tla_gpu::{probe, run_bfs, GpuBfsConfig, GpuBfsSpec, GpuError};

fn cuda_available() -> bool {
    match probe() {
        Err(GpuError::Unavailable(reason)) => {
            eprintln!("skipping BFS trace test: {reason}");
            false
        }
        Err(other) => panic!("probe failed with non-availability error: {other}"),
        Ok(_) => true,
    }
}

/// Bounded counter: fires while x < bound, incrementing x. One action.
fn counter_actions(bound: i64) -> String {
    format!(
        "static __device__ int ty_gpu_action_0(const long long* s, long long* t) {{\n\
         \x20 if (s[0] >= {bound}) return 0;\n\
         \x20 t[0] = s[0] + 1;\n\
         \x20 return 1;\n}}\n"
    )
}

fn invariant_lt(limit: i64) -> String {
    format!(
        "static __device__ int ty_gpu_invariants_ok(const long long* s) {{\n\
         \x20 return s[0] < {limit};\n}}\n"
    )
}

fn spec(bound: i64, limit: i64) -> GpuBfsSpec {
    GpuBfsSpec {
        slots: 1,
        action_count: 1,
        actions_src: format!("{}{}", counter_actions(bound), invariant_lt(limit)),
        init_rows: vec![0],
        track_slot_stats: false,
    }
}

fn trace_config() -> GpuBfsConfig {
    GpuBfsConfig {
        table_bits: 16,
        frontier_cap_rows: 1 << 12,
        trace_on_violation: true,
        ..Default::default()
    }
}

#[test]
fn violation_trace_is_the_exact_path() {
    if !cuda_available() {
        return;
    }
    // x in 0..=5, invariant x < 3 -> bad at x=3, path [0,1,2,3].
    let outcome = run_bfs(&spec(5, 3), &trace_config()).expect("BFS should complete");
    assert!(outcome.violation.is_some(), "x=3 must violate x<3");
    let trace = outcome
        .violation_trace
        .expect("trace_on_violation set + violation found -> Some");
    let path: Vec<i64> = trace.iter().map(|row| row[0]).collect();
    assert_eq!(path, vec![0, 1, 2, 3], "init->bad path must be 0,1,2,3");
    // Structural checks: starts at the init state, ends at the violating state,
    // every step is a single valid transition (x -> x+1).
    assert_eq!(trace.first().unwrap(), &vec![0]);
    assert_eq!(trace.last().unwrap(), outcome.violation.as_ref().unwrap());
    for w in trace.windows(2) {
        assert_eq!(w[1][0], w[0][0] + 1, "each step is one x++ transition");
    }
}

#[test]
fn no_violation_leaves_trace_none() {
    if !cuda_available() {
        return;
    }
    // x in 0..=5, invariant x < 100 -> never violated; exhaustive, no trace.
    let outcome = run_bfs(&spec(5, 100), &trace_config()).expect("BFS should complete");
    assert!(outcome.violation.is_none(), "x<100 holds on 0..=5");
    assert!(outcome.violation_trace.is_none());
    assert_eq!(outcome.distinct_states, 6, "states 0..=5");
}

#[test]
fn init_state_violation_is_a_singleton_trace() {
    if !cuda_available() {
        return;
    }
    // Invariant x < 0 fails at the initial state x=0 itself.
    let outcome = run_bfs(&spec(5, 0), &trace_config()).expect("BFS should complete");
    let trace = outcome.violation_trace.expect("init violates -> Some");
    assert_eq!(trace, vec![vec![0]], "the initial state is its own trace");
}
