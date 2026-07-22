// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU exhaustive bounded model checking: unroll the transition relation `k`
//! steps into ONE combinational AIG, then enumerate ALL free-variable
//! assignments on the GPU ([`tla_gpu::run_circuit_exhaust`]). Because the
//! enumeration is complete, "no bad assignment" is a genuine bounded-safety
//! PROOF (`BoundedSafe`), not the Unknown a random walk returns.
//!
//! Sound-by-construction and fail-closed:
//! - Declines (→ `None`, run on the CPU) if any init clause is non-unit
//!   (relational / nondeterministic init couples latch values and breaks the
//!   clean free-variable bijection), or if there are invariant constraints
//!   (a per-step-prefix condition the flat bad = ∨ⱼ bad@j does not model —
//!   modelling it wrong could drop a real counterexample and falsely prove
//!   safe), or if the free set `V = (k+1)·inputs` exceeds the exhaustive cap.
//! - Every pinned latch becomes a constant gate; only primary inputs at each
//!   step are free. Latch wiring `latch@(j+1) = next@j` is a buffer gate
//!   `AND(next, next)`. Bad = OR of the per-step bad literals.
//!
//! A `BoundedSafe` verdict means: no bad state is reachable in ≤ k steps. It is
//! surfaced as `Unknown` for standalone BMC (safety needs k-induction /
//! completeness), but is exactly the base-case discharge k-induction consumes.

use crate::sat_types::Lit;
use crate::transys::Transys;
use tla_gpu::{run_circuit_exhaust, CircuitExhaustConfig, CircuitExhaustSpec, ExhaustOutcome};

use super::random_sim::{compute_topo_order, extract_init_values};

/// GPU exhaustive-BMC verdict at a fixed depth `k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuExhaustBmc {
    /// No bad state is reachable in ≤ k steps — a complete bounded proof.
    BoundedSafe,
    /// A bad state is reachable within k steps (a counterexample exists).
    Unsafe,
}

/// Try to decide bounded safety at depth `k` on the GPU. `None` = declined
/// (unsupported shape / cap / no CUDA) — the caller runs the CPU BMC.
pub(crate) fn try_gpu_exhaustive_bmc(
    ts: &Transys,
    depth: usize,
    cancelled: &dyn Fn() -> bool,
) -> Option<GpuExhaustBmc> {
    if tla_gpu::probe().is_err() {
        return None;
    }
    // Sound-only shapes: no relational init, no invariant constraints.
    if ts.init_clauses.iter().any(|c| c.lits.len() != 1) {
        return None;
    }
    // Every latch must be pinned to a constant at step 0. An AIGER latch with a
    // NONDETERMINISTIC reset (`reset == its own literal`) emits NO init clause
    // (transys.rs) and would be silently pinned to `false` by
    // `extract_init_values` below — exploring only half of its legal initial
    // states, under-approximating the reachable set, and risking a FALSE
    // `BoundedSafe` (a dropped counterexample from the `latch = true` initial
    // state). Such a latch is free at step 0, like a primary input, but the
    // flat pinning cannot model that. Decline; the CPU BMC / k-induction (which
    // leaves an uninitialized latch free) decides it soundly.
    let pinned_latches: std::collections::HashSet<u32> = ts
        .init_clauses
        .iter()
        .filter_map(|c| {
            let lit = *c.lits.first()?;
            if c.lits.len() != 1 || lit == Lit::FALSE || lit == Lit::TRUE {
                return None;
            }
            Some(lit.var().0)
        })
        .collect();
    if ts.latch_vars.iter().any(|v| !pinned_latches.contains(&v.0)) {
        return None;
    }
    if !ts.constraint_lits.is_empty() {
        return None;
    }
    let max_var = ts.max_var;
    if max_var == 0 && ts.bad_lits.is_empty() {
        return None;
    }
    let k = depth;
    // Free set = the primary inputs at each of the k+1 steps.
    let v = (k + 1).checked_mul(ts.num_inputs)?;
    if v > tla_gpu::circuit_exhaust::MAX_FREE_VARS as usize {
        return None;
    }

    // Per-step global variable numbering: var 0 stays the shared constant; every
    // other var v is offset by step*max_var so the k+1 copies are disjoint.
    let steps = k + 1;
    let per = max_var; // vars 1..=max_var per step
    let gvar = |var: u32, step: usize| -> u32 {
        if var == 0 {
            0
        } else {
            var + (step as u32) * per
        }
    };
    let glit = |lit: Lit, step: usize| -> u32 {
        let var = lit.var().0;
        gvar(var, step) * 2 + u32::from(lit.is_negated())
    };
    let num_vars = (steps as u32) * per + 1;

    let init_values = extract_init_values(ts);
    let topo = compute_topo_order(ts); // gate output vars, topological (index) order

    let mut gates: Vec<[u32; 3]> = Vec::new();
    let mut free_vars: Vec<u32> = Vec::new();

    // Pinned initial latches at step 0 become constant gates:
    //   value 0 -> AND(FALSE, FALSE);  value 1 -> AND(TRUE, TRUE).
    for (idx, &latch_var) in ts.latch_vars.iter().enumerate() {
        let c = if init_values[idx] {
            Lit::TRUE.0
        } else {
            Lit::FALSE.0
        };
        gates.push([gvar(latch_var.0, 0), c, c]);
    }

    for step in 0..steps {
        // Free primary inputs at this step.
        for &input in &ts.input_vars {
            free_vars.push(gvar(input.0, step));
        }
        // AND gates at this step (topological in var index).
        for &g in &topo {
            let (rhs0, rhs1) = ts.and_defs[&g];
            gates.push([gvar(g.0, step), glit(rhs0, step), glit(rhs1, step)]);
        }
        // Wire the next latch layer: latch@(step+1) = next_state@step (buffer).
        if step + 1 < steps {
            for &latch_var in &ts.latch_vars {
                let next = ts
                    .next_state
                    .get(&latch_var)
                    .copied()
                    .unwrap_or_else(|| Lit::pos(latch_var));
                let nl = glit(next, step);
                gates.push([gvar(latch_var.0, step + 1), nl, nl]);
            }
        }
    }

    // Bad = OR over all steps of the bad literals.
    let mut bad_lits: Vec<u32> = Vec::new();
    for step in 0..steps {
        for &b in &ts.bad_lits {
            bad_lits.push(glit(b, step));
        }
    }
    if bad_lits.is_empty() {
        // No bad literal anywhere → trivially safe at every depth.
        return Some(GpuExhaustBmc::BoundedSafe);
    }

    let spec = CircuitExhaustSpec {
        num_vars,
        gates,
        free_vars,
        bad_lits,
        constraint_lits: Vec::new(),
    };
    match run_circuit_exhaust(&spec, &CircuitExhaustConfig::default(), cancelled) {
        Ok(ExhaustOutcome::Unsat) => Some(GpuExhaustBmc::BoundedSafe),
        Ok(ExhaustOutcome::Sat { .. }) => Some(GpuExhaustBmc::Unsafe),
        Ok(ExhaustOutcome::Declined(_)) | Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_aag;

    fn ts(aag: &str) -> Transys {
        Transys::from_aiger(&parse_aag(aag).unwrap())
    }

    fn no_cancel() -> bool {
        false
    }

    #[test]
    fn latch_stays_zero_is_bounded_safe() {
        // 1 latch (var1), next = FALSE, bad = var1, init 0 -> never bad.
        let t = ts("aag 1 0 1 0 0 1\n2 0\n2\n");
        if tla_gpu::probe().is_err() {
            return;
        }
        assert_eq!(
            try_gpu_exhaustive_bmc(&t, 4, &no_cancel),
            Some(GpuExhaustBmc::BoundedSafe)
        );
    }

    #[test]
    fn toggling_latch_reaches_bad() {
        // next = !current, init 0: bad(var1) becomes true at depth 1.
        let t = ts("aag 1 0 1 0 0 1\n2 3\n2\n");
        if tla_gpu::probe().is_err() {
            return;
        }
        assert_eq!(
            try_gpu_exhaustive_bmc(&t, 0, &no_cancel),
            Some(GpuExhaustBmc::BoundedSafe),
            "safe at depth 0 (still 0)"
        );
        assert_eq!(
            try_gpu_exhaustive_bmc(&t, 1, &no_cancel),
            Some(GpuExhaustBmc::Unsafe),
            "toggles to 1 by depth 1"
        );
    }

    #[test]
    fn free_input_bad_is_unsafe() {
        // bad = a primary input: some input assignment makes it bad.
        // Header: M=1 I=1 L=0 O=0 A=0 B=1 (one bad, no gates).
        let t = ts("aag 1 1 0 0 0 1\n2\n2\n");
        if tla_gpu::probe().is_err() {
            return;
        }
        assert_eq!(
            try_gpu_exhaustive_bmc(&t, 0, &no_cancel),
            Some(GpuExhaustBmc::Unsafe)
        );
    }

    #[test]
    fn free_input_contradiction_is_bounded_safe() {
        // bad = input & !input (a gate) — no input assignment satisfies it, an
        // UNSAT proof over the free input.
        let t = ts("aag 2 1 0 0 1 1\n2\n4\n4 2 3\n");
        if tla_gpu::probe().is_err() {
            return;
        }
        assert_eq!(
            try_gpu_exhaustive_bmc(&t, 0, &no_cancel),
            Some(GpuExhaustBmc::BoundedSafe)
        );
    }

    #[test]
    fn uninitialized_latch_declines() {
        // Latch with a nondeterministic reset (`reset == its own literal`, the
        // AIGER-1.9 uninitialized encoding) has NO init clause and is free at
        // step 0. The lane must DECLINE (→ None) rather than pin it to `false`:
        // pinning would explore only the latch=0 initial state and could return
        // a FALSE BoundedSafe. Here `bad = latch`, so the true verdict is Unsafe
        // (the initial latch value may be 1). Asserts None on every host: a
        // non-CUDA host declines at the device probe; a CUDA host declines at
        // the uninitialized-latch gate.
        let t = ts("aag 1 0 1 0 0 1\n2 2 2\n2\n");
        assert_eq!(try_gpu_exhaustive_bmc(&t, 4, &no_cancel), None);
    }
}
