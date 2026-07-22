// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU bit-parallel random simulation: the device twin of [`random_sim`].
//!
//! The CPU walker explores one walk at a time; the GPU engine
//! (`tla_gpu::circuit_sim`) runs `threads × 64` bit-packed lanes
//! concurrently — hundreds of thousands of independent random walks per
//! kernel launch, each drawing fresh inputs every attempt and advancing
//! latches only on constraint-satisfying lanes (identical retry semantics to
//! the scalar walker).
//!
//! Falsification-only, exactly like the CPU lane: a device hit is only a
//! CANDIDATE. Walks are deterministic in `(seed, thread)`, so the hit lane is
//! REPLAYED here on the CPU — same RNG stream (`tla_gpu::thread_rng_seed` +
//! `xorshift64`, one draw per input var per attempt; the lane's input bit is
//! the drawn word's bit L) — to rebuild the full trace, which must reproduce
//! the bad state at the reported attempt or the lane declines. The resulting
//! `CheckResult::Unsafe` then flows through the portfolio's independent
//! witness verification like any other engine's. No safety verdict ever
//! originates here: a clean device run returns `None` and the CPU walker
//! runs unchanged (different RNG mapping = extra diversity).
//!
//! Fail-closed: CUDA unavailable, oversized circuit, any driver/nvrtc error,
//! or a replay mismatch all return `None` → the caller falls back to the
//! scalar engine. Kill-switch `TY_AIGER_DISABLE_GPU_SIM` (diagnostic only).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::check_result::CheckResult;
use crate::sat_types::Lit;
use crate::transys::Transys;

use super::random_sim::{compute_topo_order, eval_lit_fast, extract_init_values};

/// Diagnostic kill-switch (default OFF = lane runs on CUDA hosts).
fn gpu_sim_disabled() -> bool {
    std::env::var("TY_AIGER_DISABLE_GPU_SIM")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true" | "yes"))
}

/// Mirror of the CPU walker's trace budget: bugs deeper than this are
/// ignored rather than returning an incomplete witness.
const MAX_TRACE_STEPS: usize = 10_000;

/// Try the GPU random-simulation lane. `Some(Unsafe)` = replayed, full-trace
/// bug witness (still subject to the portfolio's independent verification);
/// `None` = unavailable / declined / budget exhausted with no bug — the
/// caller runs the scalar walker unchanged.
pub(crate) fn try_gpu_random_sim(
    ts: &Transys,
    steps_per_walk: usize,
    seed: u64,
    cancelled: &Arc<AtomicBool>,
) -> Option<CheckResult> {
    if gpu_sim_disabled() {
        return None;
    }
    if tla_gpu::probe().is_err() {
        return None;
    }

    let topo_order = compute_topo_order(ts);
    let init_values = extract_init_values(ts);
    let num_vars = ts.max_var.checked_add(1)?;

    // Scalar depth-0 check (initial latches, inputs all-false) — mirrors the
    // CPU walker's step-0 semantics before any random draws.
    {
        let mut values = vec![false; num_vars as usize];
        for (idx, &latch_var) in ts.latch_vars.iter().enumerate() {
            values[latch_var.index()] = init_values[idx];
        }
        eval_gates(ts, &topo_order, &mut values);
        if is_bad(ts, &values) && constraints_satisfied(ts, &values) {
            let trace = vec![build_trace_step(ts, &values)];
            return Some(CheckResult::Unsafe { depth: 0, trace });
        }
    }

    let spec = tla_gpu::CircuitSimSpec {
        num_vars,
        gates: topo_order
            .iter()
            .map(|&gate_var| {
                let (rhs0, rhs1) = ts.and_defs[&gate_var];
                [gate_var.0, rhs0.0, rhs1.0]
            })
            .collect(),
        input_vars: ts.input_vars.iter().map(|v| v.0).collect(),
        latches: ts
            .latch_vars
            .iter()
            .enumerate()
            .map(|(idx, &latch_var)| {
                // A latch without a next-state literal retains its value
                // (positive self-literal), matching the scalar walker.
                let next = ts
                    .next_state
                    .get(&latch_var)
                    .copied()
                    .unwrap_or_else(|| Lit::pos(latch_var));
                (latch_var.0, next.0, init_values[idx])
            })
            .collect(),
        bad_lits: ts.bad_lits.iter().map(|l| l.0).collect(),
        constraint_lits: ts.constraint_lits.iter().map(|l| l.0).collect(),
    };

    let config = tla_gpu::CircuitSimConfig {
        base_seed: seed,
        max_attempts: u64::try_from(steps_per_walk.min(MAX_TRACE_STEPS)).unwrap_or(8192),
        ..Default::default()
    };
    let cancelled_flag = Arc::clone(cancelled);
    let hit = match tla_gpu::run_circuit_sim(&spec, &config, &move || {
        cancelled_flag.load(Ordering::Relaxed)
    }) {
        Ok(Some(hit)) => hit,
        Ok(None) => {
            eprintln!(
                "GpuRandomSim(seed={seed}): {} lanes x {} attempts clean; \
                 falling through to the scalar walker",
                u64::from(config.threads) * 64,
                config.max_attempts,
            );
            return None;
        }
        Err(e) => {
            eprintln!("GpuRandomSim: unavailable ({e}); falling back to the scalar walker");
            return None;
        }
    };

    eprintln!(
        "GpuRandomSim(seed={seed}): device hit at thread={} attempt={} lanes={:#x}; replaying",
        hit.thread, hit.attempt, hit.lane_mask,
    );
    let result = replay_hit(ts, &topo_order, &init_values, seed, &hit);
    if result.is_none() {
        eprintln!(
            "GpuRandomSim: replay did not reproduce the device hit; declining (engine fault)"
        );
    }
    result
}

/// Deterministically replay one lane of the hit thread's walk and rebuild
/// the full counterexample trace. Returns `None` on any mismatch with the
/// device report (fail-closed — never trust an unreproduced hit).
fn replay_hit(
    ts: &Transys,
    topo_order: &[crate::sat_types::Var],
    init_values: &[bool],
    base_seed: u64,
    hit: &tla_gpu::CircuitSimHit,
) -> Option<CheckResult> {
    if hit.lane_mask == 0 {
        return None;
    }
    let lane = hit.lane_mask.trailing_zeros();
    let num_vars = ts.max_var as usize + 1;
    let mut values = vec![false; num_vars];
    for (idx, &latch_var) in ts.latch_vars.iter().enumerate() {
        values[latch_var.index()] = init_values[idx];
    }
    let mut rng = tla_gpu::thread_rng_seed(base_seed, hit.thread);

    // Trace convention follows `Transys::verify_witness`: `trace[k]` holds
    // state `L_k` TOGETHER with the inputs applied at `L_k` (which produce
    // `L_{k+1}`), and the final step's `(latches, inputs)` make some bad
    // literal true. Constraint-violating draws are not transitions and
    // record no step (the RNG stream still advances — identical to the
    // device semantics).
    let mut trace: Vec<FxHashMap<String, bool>> = Vec::new();

    for attempt in 0..=hit.attempt {
        // Same draw order as the device kernel: one u64 per input var; this
        // lane's input bit is bit `lane` of the drawn word.
        for &input_var in &ts.input_vars {
            rng = tla_gpu::xorshift64(rng);
            values[input_var.index()] = (rng >> lane) & 1 != 0;
        }
        eval_gates(ts, topo_order, &mut values);
        if !constraints_satisfied(ts, &values) {
            if attempt == hit.attempt {
                return None; // device claimed bad here, but this lane was constraint-blocked
            }
            continue;
        }
        if is_bad(ts, &values) {
            if attempt != hit.attempt {
                return None; // earlier bad than the device's first hit — mismatch
            }
            if trace.len() >= MAX_TRACE_STEPS {
                return None;
            }
            trace.push(build_trace_step(ts, &values));
            return Some(CheckResult::Unsafe {
                depth: trace.len() - 1,
                trace,
            });
        }
        if attempt == hit.attempt {
            return None; // device claimed bad here, scalar disagrees
        }
        if trace.len() < MAX_TRACE_STEPS {
            trace.push(build_trace_step(ts, &values));
        }
        advance_latches(ts, &mut values);
    }
    None
}

fn eval_gates(ts: &Transys, topo_order: &[crate::sat_types::Var], values: &mut [bool]) {
    for &gate_var in topo_order {
        if let Some(&(rhs0, rhs1)) = ts.and_defs.get(&gate_var) {
            let v0 = eval_lit_fast(rhs0, values);
            let v1 = eval_lit_fast(rhs1, values);
            values[gate_var.index()] = v0 && v1;
        }
    }
}

fn is_bad(ts: &Transys, values: &[bool]) -> bool {
    ts.bad_lits.iter().any(|&lit| eval_lit_fast(lit, values))
}

fn constraints_satisfied(ts: &Transys, values: &[bool]) -> bool {
    ts.constraint_lits
        .iter()
        .all(|&lit| eval_lit_fast(lit, values))
}

fn advance_latches(ts: &Transys, values: &mut [bool]) {
    let next_vals: Vec<bool> = ts
        .latch_vars
        .iter()
        .map(|&latch_var| {
            ts.next_state.get(&latch_var).map_or_else(
                || values[latch_var.index()],
                |&next_lit| eval_lit_fast(next_lit, values),
            )
        })
        .collect();
    for (idx, &latch_var) in ts.latch_vars.iter().enumerate() {
        values[latch_var.index()] = next_vals[idx];
    }
}

fn build_trace_step(ts: &Transys, values: &[bool]) -> FxHashMap<String, bool> {
    let mut step = FxHashMap::default();
    for (idx, &latch_var) in ts.latch_vars.iter().enumerate() {
        step.insert(format!("l{idx}"), values[latch_var.index()]);
    }
    for (idx, &input_var) in ts.input_vars.iter().enumerate() {
        step.insert(format!("i{idx}"), values[input_var.index()]);
    }
    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_aag;

    fn cuda_available() -> bool {
        if tla_gpu::probe().is_err() {
            eprintln!("skipping GPU random-sim test: no usable CUDA device");
            return false;
        }
        true
    }

    #[test]
    fn gpu_sim_finds_input_conjunction_bug_with_verified_trace() {
        if !cuda_available() {
            return;
        }
        // Two inputs, bad = i0 AND i1: reachable at the first draw on ~1/4 of
        // lanes, so a hit is certain across 64 lanes.
        let circuit = parse_aag("aag 3 2 0 0 1 1\n2\n4\n6\n6 2 4\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let cancelled = Arc::new(AtomicBool::new(false));
        match try_gpu_random_sim(&ts, 1000, 42, &cancelled) {
            Some(CheckResult::Unsafe { trace, .. }) => {
                assert!(
                    ts.verify_witness(&trace).is_ok(),
                    "replayed GPU trace must be a valid circuit execution"
                );
            }
            other => panic!("expected Unsafe from GPU random sim, got {other:?}"),
        }
    }

    #[test]
    fn gpu_sim_latch_tracking_input_bug_replays() {
        if !cuda_available() {
            return;
        }
        // Latch copies the input; bad = latch. Needs one advance with a true
        // input — exercises the latch-advance + replay path.
        let circuit = parse_aag("aag 2 1 1 0 0 1\n2\n4 2\n4\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let cancelled = Arc::new(AtomicBool::new(false));
        match try_gpu_random_sim(&ts, 1000, 7, &cancelled) {
            Some(CheckResult::Unsafe { depth, trace }) => {
                assert!(depth >= 1, "bug needs at least one advance");
                assert!(
                    ts.verify_witness(&trace).is_ok(),
                    "replayed GPU trace must be a valid circuit execution"
                );
            }
            other => panic!("expected Unsafe from GPU random sim, got {other:?}"),
        }
    }

    #[test]
    fn gpu_sim_depth_zero_bug_via_scalar_precheck() {
        if !cuda_available() {
            return;
        }
        // Constant-TRUE bad output: caught by the scalar depth-0 pre-check
        // before any device work.
        let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let cancelled = Arc::new(AtomicBool::new(false));
        assert!(matches!(
            try_gpu_random_sim(&ts, 100, 1, &cancelled),
            Some(CheckResult::Unsafe { depth: 0, .. })
        ));
    }

    #[test]
    fn gpu_sim_safe_circuit_returns_none() {
        if !cuda_available() {
            return;
        }
        // Latch pinned to 0 (next = FALSE), bad = latch: unreachable — the
        // lane must come back clean (None), never a safety verdict.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let cancelled = Arc::new(AtomicBool::new(false));
        assert!(try_gpu_random_sim(&ts, 200, 3, &cancelled).is_none());
    }
}
