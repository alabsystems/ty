// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU bit-parallel falsification lane for the DEFAULT BTOR2 portfolio.
//!
//! Bit-blasts the program (the same `bitblast` the eligibility-gated SAT
//! handoff uses — [`BitblastedCircuit`] is already in AIGER literal
//! encoding), runs `tla_gpu::circuit_sim`'s bit-parallel random walker
//! (threads × 64 packed lanes), and on a device hit REPLAYS the hit lane
//! scalar-side over the same circuit to rebuild the full bit-level trace,
//! then maps it to a WORD-LEVEL named trace through the bitblaster's
//! `state_bits`/`input_bits` tables (the same assignment keys the CHC
//! counterexamples use).
//!
//! Falsification-only, fail-closed: it can only produce `Sat { trace }` for
//! one bad property (the others stay with the word-level engines); a clean,
//! ineligible, or CUDA-less run returns `None` and the portfolio proceeds
//! unchanged. UNSAT verdicts never originate here.
//!
//! Soundness anchors:
//! - The replay must reproduce the device hit bit-for-bit or the lane
//!   declines (never trust an unreproduced hit).
//! - The word-level trace is derived from the verified bit-level trace via
//!   the bit-blaster's own tables — the identical trust base the
//!   eligibility-gated SAT handoff already accepts for verdicts.
//! - Widths are capped at 63 bits so every reconstructed word fits the
//!   trace's `i64` carrier exactly.
//! - Nondeterministically-reset latches start at 0 in every lane — a legal
//!   initial value, so any found trace starts from a genuine initial state
//!   (an under-approximation: it only costs coverage, never soundness).
//!
//! Kill-switch: `TY_BTOR2_DISABLE_GPU_SIM` (diagnostic only).

use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::bitblast::{bitblast, bitblast_eligible, BitblastedCircuit};
use crate::types::Btor2Program;

/// Width cap for this lane: every state/input word must fit the trace's
/// `i64` carrier exactly.
const GPU_FALSIFY_MAX_WIDTH: u32 = 63;

/// Per-thread attempt budget (also bounds the replayed trace length).
const GPU_FALSIFY_MAX_ATTEMPTS: u64 = 8192;

fn gpu_falsify_disabled() -> bool {
    std::env::var("TY_BTOR2_DISABLE_GPU_SIM")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true" | "yes"))
}

/// A falsification result: the violated bad-property index plus the
/// word-level named counterexample trace (states + inputs per step, in the
/// CHC assignment-key convention).
pub(crate) struct GpuFalsification {
    pub(crate) bad_index: usize,
    pub(crate) trace: Vec<FxHashMap<String, i64>>,
}

/// Try the GPU falsification lane on the ORIGINAL (pre-COI) program so bad
/// indices and state names align with the caller's result vector. `None` =
/// unavailable / ineligible / clean within budget — the portfolio proceeds
/// unchanged.
pub(crate) fn try_gpu_falsify(
    program: &Btor2Program,
    deadline: Option<Instant>,
) -> Option<GpuFalsification> {
    if gpu_falsify_disabled() {
        return None;
    }
    if program.bad_properties.is_empty() {
        return None;
    }
    if tla_gpu::probe().is_err() {
        return None;
    }
    bitblast_eligible(program, GPU_FALSIFY_MAX_WIDTH).ok()?;
    let circuit = bitblast(program, GPU_FALSIFY_MAX_WIDTH).ok()?;
    // Every mapped word must fit i64 exactly (array states expand past the
    // scalar cap; decline them).
    if circuit
        .state_bits
        .iter()
        .chain(&circuit.input_bits)
        .any(|(_, bits)| bits.len() > GPU_FALSIFY_MAX_WIDTH as usize)
    {
        return None;
    }

    let spec = circuit_spec(&circuit)?;
    let config = tla_gpu::CircuitSimConfig {
        base_seed: 0x00B7_0125_EED5_EED5,
        max_attempts: GPU_FALSIFY_MAX_ATTEMPTS,
        deadline,
        ..Default::default()
    };
    let deadline_hit = move || deadline.is_some_and(|d| Instant::now() >= d);
    let hit = match tla_gpu::run_circuit_sim(&spec, &config, &deadline_hit) {
        Ok(Some(hit)) => hit,
        Ok(None) => {
            return None;
        }
        Err(e) => {
            eprintln!("BTOR2 GPU falsification unavailable ({e}); continuing word-level");
            return None;
        }
    };
    eprintln!(
        "BTOR2 GPU falsification: device hit at thread={} attempt={} lanes={:#x}; replaying",
        hit.thread, hit.attempt, hit.lane_mask,
    );
    let result = replay_hit(&circuit, &spec, config.base_seed, &hit);
    if result.is_none() {
        eprintln!(
            "BTOR2 GPU falsification: replay did not reproduce the device hit; \
             declining (engine fault)"
        );
    }
    result
}

/// Build the device spec from the bit-blasted circuit. `None` = the circuit
/// does not fit the engine's `u32` literal carrier or violates the
/// definition-before-use gate ordering the single-pass evaluator needs.
fn circuit_spec(circuit: &BitblastedCircuit) -> Option<tla_gpu::CircuitSimSpec> {
    let num_vars = u32::try_from(circuit.max_var.checked_add(1)?).ok()?;
    let lit = |l: u64| u32::try_from(l).ok();
    let mut gates = Vec::with_capacity(circuit.ands.len());
    for &(lhs, rhs0, rhs1) in &circuit.ands {
        let out = lit(lhs)? >> 1;
        let r0 = lit(rhs0)?;
        let r1 = lit(rhs1)?;
        // The bit-blaster allocates gate variables after their operands, so
        // creation order is definition-before-use; decline if that ever
        // fails to hold rather than mis-evaluate.
        if (r0 >> 1) >= out || (r1 >> 1) >= out {
            return None;
        }
        gates.push([out, r0, r1]);
    }
    let mut latches = Vec::with_capacity(circuit.latches.len());
    for &(curr, next, reset) in &circuit.latches {
        let curr_var = lit(curr)? >> 1;
        let next_lit = lit(next)?;
        // reset: 0 => init false, 1 => init true, curr (nondet) => init
        // false (a legal initial value; under-approximation, sound for SAT).
        let init = reset == 1;
        latches.push((curr_var, next_lit, init));
    }
    Some(tla_gpu::CircuitSimSpec {
        num_vars,
        gates,
        input_vars: circuit
            .inputs
            .iter()
            .map(|&l| lit(l).map(|v| v >> 1))
            .collect::<Option<_>>()?,
        latches,
        bad_lits: circuit.bad.iter().map(|&l| lit(l)).collect::<Option<_>>()?,
        constraint_lits: circuit
            .constraints
            .iter()
            .map(|&l| lit(l))
            .collect::<Option<_>>()?,
    })
}

fn eval_lit(values: &[bool], l: u32) -> bool {
    let v = values[(l >> 1) as usize];
    if l & 1 == 1 {
        !v
    } else {
        v
    }
}

fn eval_gates(spec: &tla_gpu::CircuitSimSpec, values: &mut [bool]) {
    for gate in &spec.gates {
        values[gate[0] as usize] = eval_lit(values, gate[1]) && eval_lit(values, gate[2]);
    }
}

fn word_of(values: &[bool], bits: &[u64]) -> i64 {
    let mut word = 0i64;
    for (i, &bit_lit) in bits.iter().enumerate() {
        let b = eval_lit(values, u32::try_from(bit_lit).unwrap_or(0));
        if b {
            word |= 1i64 << i;
        }
    }
    word
}

pub(crate) fn trace_step(circuit: &BitblastedCircuit, values: &[bool]) -> FxHashMap<String, i64> {
    let mut step = FxHashMap::default();
    for (name, bits) in circuit.state_bits.iter().chain(&circuit.input_bits) {
        step.insert(name.clone(), word_of(values, bits));
    }
    step
}

/// Deterministically replay one lane of the hit thread's walk over the
/// bit-blasted circuit and rebuild the word-level trace (verify_witness-style
/// convention: `trace[k]` = state_k plus the inputs applied at state_k; the
/// final step's assignment makes some bad literal true). Returns `None` on
/// any mismatch with the device report.
fn replay_hit(
    circuit: &BitblastedCircuit,
    spec: &tla_gpu::CircuitSimSpec,
    base_seed: u64,
    hit: &tla_gpu::CircuitSimHit,
) -> Option<GpuFalsification> {
    if hit.lane_mask == 0 {
        return None;
    }
    let lane = hit.lane_mask.trailing_zeros();
    let mut values = vec![false; spec.num_vars as usize];
    for &(latch_var, _, init) in &spec.latches {
        values[latch_var as usize] = init;
    }
    let mut rng = tla_gpu::thread_rng_seed(base_seed, hit.thread);
    let mut trace: Vec<FxHashMap<String, i64>> = Vec::new();

    for attempt in 0..=hit.attempt {
        for &input_var in &spec.input_vars {
            rng = tla_gpu::xorshift64(rng);
            values[input_var as usize] = (rng >> lane) & 1 != 0;
        }
        eval_gates(spec, &mut values);
        let ok = spec.constraint_lits.iter().all(|&l| eval_lit(&values, l));
        if !ok {
            if attempt == hit.attempt {
                return None; // device claimed bad here, but the lane was constraint-blocked
            }
            continue; // retry: RNG advanced, latches unchanged — device semantics
        }
        let bad_index = spec.bad_lits.iter().position(|&l| eval_lit(&values, l));
        if let Some(bad_index) = bad_index {
            if attempt != hit.attempt {
                return None; // earlier bad than the device's first hit — mismatch
            }
            trace.push(trace_step(circuit, &values));
            return Some(GpuFalsification { bad_index, trace });
        }
        if attempt == hit.attempt {
            return None; // device claimed bad here, scalar disagrees
        }
        trace.push(trace_step(circuit, &values));
        // Two-phase latch advance.
        let next_vals: Vec<bool> = spec
            .latches
            .iter()
            .map(|&(_, next_lit, _)| eval_lit(&values, next_lit))
            .collect();
        for (&(latch_var, _, _), next) in spec.latches.iter().zip(next_vals) {
            values[latch_var as usize] = next;
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
            eprintln!("skipping BTOR2 GPU falsification test: no usable CUDA device");
            return false;
        }
        true
    }

    #[test]
    fn counter_reaching_bad_is_falsified_with_word_trace() {
        if !cuda_available() {
            return;
        }
        // 2-bit counter from 0; bad when it reaches 3. Reachable in 3 steps.
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
        let result = try_gpu_falsify(&program, None).expect("bug should be found");
        assert_eq!(result.bad_index, 0);
        let last = result.trace.last().expect("non-empty trace");
        assert_eq!(
            last.get("counter"),
            Some(&3),
            "final step must show the counter at 3, got {last:?}"
        );
        // The word-level steps must count up from the initial value.
        let counts: Vec<i64> = result
            .trace
            .iter()
            .map(|s| *s.get("counter").expect("counter in every step"))
            .collect();
        assert_eq!(counts, vec![0, 1, 2, 3]);
    }

    #[test]
    fn input_guarded_bad_is_falsified() {
        if !cuda_available() {
            return;
        }
        // bad iff the 1-bit input is 1 — found at the first draw on half the
        // lanes.
        let src = "\
1 sort bitvec 1
2 input 1 go
3 bad 2
";
        let program = parse(src).expect("parse");
        let result = try_gpu_falsify(&program, None).expect("bug should be found");
        assert_eq!(result.bad_index, 0);
        let last = result.trace.last().expect("non-empty trace");
        assert_eq!(last.get("go"), Some(&1));
    }

    #[test]
    fn safe_program_returns_none() {
        if !cuda_available() {
            return;
        }
        // Latch pinned to 0 (next = itself, init 0); bad = latch. Unreachable.
        let src = "\
1 sort bitvec 1
2 zero 1
3 state 1 stuck
4 init 1 3 2
5 next 1 3 3
6 bad 3
";
        let program = parse(src).expect("parse");
        assert!(try_gpu_falsify(&program, None).is_none());
    }
}
