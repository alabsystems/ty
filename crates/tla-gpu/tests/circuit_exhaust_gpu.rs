// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU exhaustive-SAT (bounded-safety proof lane) validated against the CPU
//! oracle: SAT witnesses replay, UNSAT proofs agree, and the outer-combo
//! enumeration (V > 6) is exercised.

use tla_gpu::{
    exhaustive_sat_cpu, probe, run_circuit_exhaust, CircuitExhaustConfig, CircuitExhaustSpec,
    ExhaustOutcome, GpuError,
};

fn cuda_available() -> bool {
    match probe() {
        Err(GpuError::Unavailable(reason)) => {
            eprintln!("skipping exhaustive-SAT GPU test: {reason}");
            false
        }
        Err(other) => panic!("probe failed with non-availability error: {other}"),
        Ok(_) => true,
    }
}

fn never() -> bool {
    false
}

/// bad = conjunction of `n` inputs (vars 1..=n): SAT only at all-ones.
/// Returns (spec, the final conjunction var).
fn conjunction(n: u32) -> CircuitExhaustSpec {
    let mut gates = Vec::new();
    let mut acc = 1u32; // v1
    let mut next_gate = n + 1;
    for k in 2..=n {
        gates.push([next_gate, acc * 2, k * 2]); // out = acc & v_k (positive lits)
        acc = next_gate;
        next_gate += 1;
    }
    CircuitExhaustSpec {
        num_vars: (2 * n).max(2),
        gates,
        free_vars: (1..=n).collect(),
        bad_lits: vec![acc * 2],
        constraint_lits: vec![],
    }
}

fn run_gpu(spec: &CircuitExhaustSpec) -> ExhaustOutcome {
    run_circuit_exhaust(spec, &CircuitExhaustConfig::default(), &never).expect("engine runs")
}

#[test]
fn gpu_sat_witness_matches_and_replays() {
    if !cuda_available() {
        return;
    }
    // bad = a & b: SAT at a=b=1.
    let spec = CircuitExhaustSpec {
        num_vars: 4,
        gates: vec![[3, 2, 4]],
        free_vars: vec![1, 2],
        bad_lits: vec![6], // var3 = a&b
        constraint_lits: vec![],
    };
    match run_gpu(&spec) {
        ExhaustOutcome::Sat { assignment } => assert_eq!(assignment, vec![true, true]),
        other => panic!("expected Sat, got {other:?}"),
    }
}

#[test]
fn gpu_unsat_is_a_complete_proof() {
    if !cuda_available() {
        return;
    }
    // bad = a & b under constraint !a: forces a=0 → bad never true → UNSAT.
    let spec = CircuitExhaustSpec {
        num_vars: 4,
        gates: vec![[3, 2, 4]],
        free_vars: vec![1, 2],
        bad_lits: vec![6],
        constraint_lits: vec![3], // !var1
    };
    assert_eq!(run_gpu(&spec), ExhaustOutcome::Unsat);
}

#[test]
fn gpu_finds_witness_across_outer_combos() {
    if !cuda_available() {
        return;
    }
    // V=10 (> 6 → 16 outer combos): bad = AND of all 10 inputs, satisfied only
    // by the single all-ones assignment, which lives in one specific outer
    // combo × one specific inner lane. The GPU must find it and it must replay.
    let spec = conjunction(10);
    match run_gpu(&spec) {
        ExhaustOutcome::Sat { assignment } => {
            assert_eq!(
                assignment,
                vec![true; 10],
                "only all-ones satisfies the conjunction"
            );
        }
        other => panic!("expected Sat, got {other:?}"),
    }
}

#[test]
fn gpu_unsat_complete_over_many_outer_combos() {
    if !cuda_available() {
        return;
    }
    // V=12 (64 outer combos): conjunction of 12 inputs but constrained to
    // !v1, so the all-ones witness is excluded → no assignment → UNSAT, only
    // provable by exhausting all 2^12 assignments.
    let mut spec = conjunction(12);
    spec.constraint_lits = vec![3]; // !var1
    assert_eq!(run_gpu(&spec), ExhaustOutcome::Unsat);
}

#[test]
fn gpu_matches_cpu_oracle_across_sizes() {
    if !cuda_available() {
        return;
    }
    // Differential vs the CPU oracle across V spanning the inner/outer boundary.
    for n in [1u32, 3, 6, 7, 9] {
        let spec = conjunction(n);
        let gpu = run_gpu(&spec);
        let cpu = exhaustive_sat_cpu(&spec, 30);
        assert_eq!(gpu, cpu, "GPU vs CPU disagree at V={n}: {gpu:?} vs {cpu:?}");
        // The conjunction is always SAT (all-ones); confirm both say so.
        assert!(
            matches!(gpu, ExhaustOutcome::Sat { .. }),
            "conjunction is SAT at V={n}"
        );
    }
}

#[test]
fn gpu_single_combo_does_not_allocate_idle_thread_scratch() {
    if !cuda_available() {
        return;
    }
    // Regression for the DGX Spark host-OOM cascade: the old fixed 65,536
    // workers allocated 100,000 vars * 65,536 * 8 = 48.8 GiB even though an
    // empty free set has exactly one useful outer combination. The adaptive
    // plan uses one worker (~0.8 MiB) and still completely checks that space.
    let spec = CircuitExhaustSpec {
        num_vars: 100_000,
        gates: Vec::new(),
        free_vars: Vec::new(),
        bad_lits: Vec::new(),
        constraint_lits: Vec::new(),
    };
    assert_eq!(run_gpu(&spec), ExhaustOutcome::Unsat);
}
