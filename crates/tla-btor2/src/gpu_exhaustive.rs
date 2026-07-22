// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU exhaustive bounded-model-checking lane for the DEFAULT BTOR2 portfolio.
//!
//! The BTOR2 mirror of tla-aiger's exhaustive-BMC lane. It bit-blasts the
//! program (the same [`BitblastedCircuit`] the random-sim [`crate::gpu_falsify`]
//! lane and the eligibility-gated SAT handoff use — already AIGER-lit encoded),
//! unrolls the transition relation `k` steps into ONE combinational circuit,
//! and enumerates ALL free-variable assignments on the GPU
//! ([`tla_gpu::run_circuit_exhaust`]). Because that enumeration is COMPLETE
//! within the free-variable cap, the two outcomes are exact:
//! - `BoundedSafe` — no bad state is reachable in ≤ k steps (a genuine bounded
//!   proof, not the "no random hit" of the probabilistic walker), and
//! - `Unsafe` — a bad state IS reachable within k steps; the satisfying
//!   assignment is replayed forward through the circuit to a WORD-LEVEL named
//!   trace (via the bit-blaster's `state_bits`/`input_bits` tables, the same
//!   trust base the CHC counterexamples use).
//!
//! Unlike the falsification-only random walker, this lane can PROVE bounded
//! safety, so its soundness gates are STRICTER — it is fail-closed:
//! - Declines on any latch whose reset is not the constant `0` or `1`. A
//!   NONDETERMINISTIC-reset latch (`reset == its own literal`) has no pinned
//!   initial value; pinning it to `false` (as the random-sim lane does — sound
//!   there because it only costs coverage) would UNDER-APPROXIMATE the initial
//!   set and could falsely prove `BoundedSafe`.
//! - Declines on any `constraint` (the flat `bad = ∨ⱼ bad@j` cannot model a
//!   per-step-prefix constraint without risking a dropped counterexample).
//! - Declines when the free set `(k+1)·num_inputs` exceeds the exhaustive cap,
//!   on an ineligible / oversized bit-blast, or on a non-CUDA host.
//!
//! Only the `Unsafe` outcome yields a portfolio verdict (a complete
//! falsification the random walk can miss); `BoundedSafe` and every decline
//! fall through to the word-level BMC/CHC engines unchanged — bounded safety is
//! not full safety without the completeness argument those engines supply.
//!
//! Kill-switch: `TY_BTOR2_DISABLE_GPU_SIM` (shared with `gpu_falsify`).

use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::bitblast::{bitblast, bitblast_eligible, BitblastedCircuit};
use crate::gpu_falsify::trace_step;
use crate::types::Btor2Program;

/// Width cap: every reconstructed state/input word must fit the trace's `i64`
/// carrier exactly (matches the random-sim lane).
const GPU_EXHAUST_MAX_WIDTH: u32 = 63;

fn gpu_exhaust_disabled() -> bool {
    std::env::var("TY_BTOR2_DISABLE_GPU_SIM")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true" | "yes"))
}

/// GPU exhaustive-BMC verdict at a fixed unroll depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuExhaustBmc {
    /// No bad state is reachable in ≤ k steps — a complete bounded proof.
    BoundedSafe,
    /// A bad state is reachable within k steps: the exhaustively-found
    /// counterexample, replayed to a word-level named trace.
    Unsafe {
        /// Index into the program's `bad_properties` of the violated property.
        bad_index: usize,
        /// Word-level counterexample trace (state + inputs per step, in the
        /// CHC assignment-key convention).
        trace: Vec<FxHashMap<String, i64>>,
    },
}

/// Evaluate a `u64` AIGER literal against a bit-value vector.
fn eval_lit(values: &[bool], lit: u64) -> bool {
    let v = values[(lit >> 1) as usize];
    if lit & 1 == 1 {
        !v
    } else {
        v
    }
}

/// Try to decide bounded safety at depth `max_depth` on the GPU exhaustive
/// lane. `None` = declined (ineligible shape / cap / no CUDA / disabled) — the
/// portfolio proceeds to the word-level engines unchanged.
pub(crate) fn try_gpu_exhaustive_bmc(
    program: &Btor2Program,
    max_depth: usize,
    deadline: Option<Instant>,
) -> Option<GpuExhaustBmc> {
    if gpu_exhaust_disabled() || program.bad_properties.is_empty() {
        return None;
    }
    if tla_gpu::probe().is_err() {
        return None;
    }
    bitblast_eligible(program, GPU_EXHAUST_MAX_WIDTH).ok()?;
    let circuit = bitblast(program, GPU_EXHAUST_MAX_WIDTH).ok()?;

    // Every mapped word must fit i64 exactly (array states expand past the cap).
    if circuit
        .state_bits
        .iter()
        .chain(&circuit.input_bits)
        .any(|(_, bits)| bits.len() > GPU_EXHAUST_MAX_WIDTH as usize)
    {
        return None;
    }
    // SOUNDNESS: every latch must have a CONSTANT initial value. `reset` is 0,
    // 1, or the latch's own literal (nondeterministic); decline anything but 0
    // or 1 so an uninitialized latch is never silently pinned to false.
    if circuit
        .latches
        .iter()
        .any(|&(_, _, reset)| reset != 0 && reset != 1)
    {
        return None;
    }
    // No per-step-prefix constraints (see module doc).
    if !circuit.constraints.is_empty() {
        return None;
    }

    let max_var = u32::try_from(circuit.max_var).ok()?;
    let num_inputs = circuit.inputs.len();
    let k = max_depth;
    let steps = k + 1;

    // Free set = the primary inputs at each of the k+1 steps.
    let v = steps.checked_mul(num_inputs)?;
    if v > tla_gpu::circuit_exhaust::MAX_FREE_VARS as usize {
        return None;
    }

    // Definition-before-use gate ordering (mirrors gpu_falsify::circuit_spec):
    // the flat exhaustive evaluator processes gates in vec order, so a gate's
    // operands must be produced earlier. The bit-blaster allocates gate vars
    // after their operands; decline if that ever fails rather than mis-evaluate.
    for &(lhs, rhs0, rhs1) in &circuit.ands {
        let out = u32::try_from(lhs).ok()? >> 1;
        if (u32::try_from(rhs0).ok()? >> 1) >= out || (u32::try_from(rhs1).ok()? >> 1) >= out {
            return None;
        }
    }

    // Per-step global variable numbering: var 0 stays the shared constant; every
    // other var v at step s is offset by s*max_var so the k+1 copies are disjoint.
    let per = max_var;
    let gvar = |var: u32, step: usize| -> u32 {
        if var == 0 {
            0
        } else {
            var + (step as u32) * per
        }
    };
    let glit = |lit: u64, step: usize| -> Option<u32> {
        let l = u32::try_from(lit).ok()?;
        Some(gvar(l >> 1, step) * 2 + (l & 1))
    };
    let num_vars = (steps as u32).checked_mul(per)?.checked_add(1)?;

    let mut gates: Vec<[u32; 3]> = Vec::new();
    let mut free_vars: Vec<u32> = Vec::new();

    // Pinned initial latches at step 0 become constant gates: reset 0 ->
    // AND(FALSE,FALSE), reset 1 -> AND(TRUE,TRUE) (AIGER lits: FALSE=0, TRUE=1).
    for &(curr, _, reset) in &circuit.latches {
        let curr_var = u32::try_from(curr).ok()? >> 1;
        let c = u32::from(reset == 1);
        gates.push([gvar(curr_var, 0), c, c]);
    }

    for step in 0..steps {
        // Free primary inputs at this step.
        for &input in &circuit.inputs {
            let input_var = u32::try_from(input).ok()? >> 1;
            free_vars.push(gvar(input_var, step));
        }
        // AND gates at this step (topological in var index).
        for &(lhs, rhs0, rhs1) in &circuit.ands {
            let out = u32::try_from(lhs).ok()? >> 1;
            gates.push([gvar(out, step), glit(rhs0, step)?, glit(rhs1, step)?]);
        }
        // Wire the next latch layer: latch@(step+1) = next_state@step (buffer).
        if step + 1 < steps {
            for &(curr, next, _) in &circuit.latches {
                let curr_var = u32::try_from(curr).ok()? >> 1;
                let nl = glit(next, step)?;
                gates.push([gvar(curr_var, step + 1), nl, nl]);
            }
        }
    }

    // Bad = OR over all steps of the bad literals.
    let mut bad_lits: Vec<u32> = Vec::new();
    for step in 0..steps {
        for &b in &circuit.bad {
            bad_lits.push(glit(b, step)?);
        }
    }
    if bad_lits.is_empty() {
        return Some(GpuExhaustBmc::BoundedSafe);
    }

    let spec = tla_gpu::CircuitExhaustSpec {
        num_vars,
        gates,
        free_vars,
        bad_lits,
        constraint_lits: Vec::new(),
    };
    let config = tla_gpu::CircuitExhaustConfig {
        deadline,
        ..Default::default()
    };
    let cancelled = move || deadline.is_some_and(|d| Instant::now() >= d);
    match tla_gpu::run_circuit_exhaust(&spec, &config, &cancelled) {
        Ok(tla_gpu::ExhaustOutcome::Unsat) => Some(GpuExhaustBmc::BoundedSafe),
        Ok(tla_gpu::ExhaustOutcome::Sat { assignment }) => {
            reconstruct_unsafe(&circuit, steps, num_inputs, &assignment)
        }
        Ok(tla_gpu::ExhaustOutcome::Declined(_)) | Err(_) => None,
    }
}

/// Replay the circuit forward under the found per-step input assignment and
/// build the word-level trace up to the FIRST step where a bad literal fires.
/// The assignment satisfied `∨ⱼ bad@j` in the unrolled circuit, which uses the
/// identical semantics (constant-pinned init, buffered latch layers), so the
/// replay is guaranteed to reach a bad state; if it somehow does not, decline
/// (engine fault) rather than return an unwitnessed `Unsafe`.
fn reconstruct_unsafe(
    circuit: &BitblastedCircuit,
    steps: usize,
    num_inputs: usize,
    assignment: &[bool],
) -> Option<GpuExhaustBmc> {
    let mut values = vec![false; (circuit.max_var as usize) + 1];
    // Constant-pinned initial latches (nondeterministic resets were declined).
    for &(curr, _, reset) in &circuit.latches {
        values[(curr >> 1) as usize] = reset == 1;
    }
    let mut trace: Vec<FxHashMap<String, i64>> = Vec::with_capacity(steps);

    for step in 0..steps {
        for (i, &input) in circuit.inputs.iter().enumerate() {
            values[(input >> 1) as usize] = *assignment.get(step * num_inputs + i)?;
        }
        for &(lhs, rhs0, rhs1) in &circuit.ands {
            values[(lhs >> 1) as usize] = eval_lit(&values, rhs0) && eval_lit(&values, rhs1);
        }
        if let Some(bad_index) = circuit.bad.iter().position(|&b| eval_lit(&values, b)) {
            trace.push(trace_step(circuit, &values));
            return Some(GpuExhaustBmc::Unsafe { bad_index, trace });
        }
        trace.push(trace_step(circuit, &values));
        // Two-phase latch advance: latch@(step+1) = next_state@step.
        let next_vals: Vec<bool> = circuit
            .latches
            .iter()
            .map(|&(_, next, _)| eval_lit(&values, next))
            .collect();
        for (&(curr, _, _), next) in circuit.latches.iter().zip(next_vals) {
            values[(curr >> 1) as usize] = next;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn cuda_available() -> bool {
        if tla_gpu::probe().is_err() {
            eprintln!("skipping BTOR2 GPU exhaustive-BMC test: no usable CUDA device");
            return false;
        }
        true
    }

    #[test]
    fn counter_reaching_bad_is_exhaustively_unsafe_with_word_trace() {
        if !cuda_available() {
            return;
        }
        // 2-bit counter from 0; bad when it reaches 3. Reachable in 3 steps.
        // Zero inputs => a single deterministic enumeration finds it exactly.
        let src = "\
1 sort bitvec 2
2 zero 1
3 state 1 counter
4 init 1 3 2
5 one 1
6 add 1 3 5
7 next 1 3 6
8 constd 1 3
9 eq 1 3 8
10 bad 9
";
        let program = parse(src).expect("parse");
        match try_gpu_exhaustive_bmc(&program, 8, None) {
            Some(GpuExhaustBmc::Unsafe { bad_index, trace }) => {
                assert_eq!(bad_index, 0);
                let counts: Vec<i64> = trace
                    .iter()
                    .map(|s| *s.get("counter").expect("counter in every step"))
                    .collect();
                assert_eq!(counts, vec![0, 1, 2, 3]);
            }
            other => panic!("expected exhaustive Unsafe, got {other:?}"),
        }
    }

    #[test]
    fn stuck_latch_is_bounded_safe() {
        if !cuda_available() {
            return;
        }
        // Latch pinned to 0 (next = itself, init 0); bad = latch. No bad in any
        // number of steps — a complete BoundedSafe proof (zero free vars).
        let src = "\
1 sort bitvec 1
2 zero 1
3 state 1 stuck
4 init 1 3 2
5 next 1 3 3
6 bad 3
";
        let program = parse(src).expect("parse");
        assert_eq!(
            try_gpu_exhaustive_bmc(&program, 8, None),
            Some(GpuExhaustBmc::BoundedSafe)
        );
    }

    #[test]
    fn input_guarded_bad_is_exhaustively_unsafe() {
        if !cuda_available() {
            return;
        }
        // bad iff the 1-bit input is 1 — the exhaustive enumeration over the
        // single free input finds it.
        let src = "\
1 sort bitvec 1
2 input 1 go
3 bad 2
";
        let program = parse(src).expect("parse");
        match try_gpu_exhaustive_bmc(&program, 0, None) {
            Some(GpuExhaustBmc::Unsafe { bad_index, trace }) => {
                assert_eq!(bad_index, 0);
                assert_eq!(trace.last().and_then(|s| s.get("go")), Some(&1));
            }
            other => panic!("expected exhaustive Unsafe, got {other:?}"),
        }
    }

    #[test]
    fn nondeterministic_reset_latch_declines() {
        // A nondeterministic-reset latch (no `init`, so reset == its own
        // literal) is free at step 0. The PROVING lane must DECLINE rather than
        // pin it to 0 and risk a false BoundedSafe — `bad = latch`, so the true
        // verdict is Unsafe (the initial value may be 1). Asserts None on every
        // host (non-CUDA declines at the probe; CUDA declines at the latch gate).
        let src = "\
1 sort bitvec 1
2 state 1 free
3 next 1 2 2
4 bad 2
";
        let program = parse(src).expect("parse");
        assert_eq!(try_gpu_exhaustive_bmc(&program, 8, None), None);
    }
}
