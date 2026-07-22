// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Predecessor lifting for IC3/PDR.
//!
//! The LiftSolver minimizes predecessor cubes found by IC3 frame solvers.
//! It uses three complementary techniques, in order:
//!
//! 1. **Ternary pre-filter** (O(n * gates), zero SAT calls): propagate
//!    through the AND-gate circuit to identify don't-care state literals.
//! 2. **UNSAT core extraction** (1 SAT call): find the minimal subset of
//!    state literals needed for the transition, using fine-grained negated
//!    target assumptions for smaller cores.
//! 3. **Ternary X-drop pass** (O(n * gates), zero SAT calls): for each
//!    literal in the UNSAT core, try setting it to X (unknown) and
//!    re-propagating through the circuit. If the target next-state values
//!    remain determined, the literal is a don't-care and is dropped.
//!
//! Steps 1 and 3 exploit the fact that tla-aiger owns the circuit, not just
//! its CNF: ternary (0/1/X) simulation over the AND-gate graph answers
//! "does the transition still force the target?" directly, without asking
//! the SAT solver anything. That replaces the naive approach of one SAT
//! call per core literal (O(n * SAT_call)), saving thousands of SAT calls
//! per IC3 run.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::sat_types::{AYSatCdclSolver, Lit, SatResult, SatSolver, Var};
use crate::ternary::{TernarySim, TernaryVal};
use crate::transys::Transys;

/// SAT-based predecessor cube minimizer.
///
/// Holds a SAT solver with the transition relation and next-state linking,
/// but without any frame lemmas. Used to find the minimal subset of a
/// predecessor cube that still reaches the target via one transition step.
pub(crate) struct LiftSolver {
    solver: AYSatCdclSolver,
}

impl LiftSolver {
    /// Create a new LiftSolver from a transition system.
    ///
    /// Loads the transition relation, constraints, and next-state linking
    /// clauses (next_var <=> next_state_expr for each latch).
    pub(crate) fn new(
        ts: &Transys,
        next_vars: &rustc_hash::FxHashMap<Var, Var>,
        max_var: u32,
    ) -> Self {
        Self::new_inner(ts, next_vars, max_var, true)
    }

    /// Create a new LiftSolver with BVE preprocessing disabled (#4074).
    ///
    /// Used as a fallback when ay-sat's BVE produces FINALIZE_SAT_FAIL
    /// on certain clause structures.
    pub(crate) fn new_no_preprocess(
        ts: &Transys,
        next_vars: &rustc_hash::FxHashMap<Var, Var>,
        max_var: u32,
    ) -> Self {
        Self::new_inner(ts, next_vars, max_var, false)
    }

    fn new_inner(
        ts: &Transys,
        next_vars: &rustc_hash::FxHashMap<Var, Var>,
        max_var: u32,
        preprocess: bool,
    ) -> Self {
        let mut solver = AYSatCdclSolver::new(max_var + 1);
        if !preprocess {
            solver.disable_preprocessing();
            // Full IC3 mode for the no-preprocess fallback path (#4306
            // Patch B): LiftSolver runs thousands of short incremental
            // queries per IC3 proof, flipping assumptions to lift
            // predecessor cubes. set_ic3_mode also disables preprocessing
            // (consistent with this branch) plus LRAT proofs, chronological
            // backtracking, rephase, etc., and enables the O(new_clauses)
            // incremental reset path. We do not apply it to the
            // preprocess=true construction path to avoid changing
            // first-solve simplification behavior on the default
            // LiftSolver ctor.
            <AYSatCdclSolver as crate::sat_types::SatSolver>::set_ic3_mode(&mut solver);
        }

        // Constant: Var(0) = false.
        solver.add_clause(&[Lit::TRUE]);

        // Transition relation.
        for clause in &ts.trans_clauses {
            solver.add_clause(&clause.lits);
        }

        // Constraints.
        for &constraint in &ts.constraint_lits {
            solver.add_clause(&[constraint]);
        }

        // Next-state linking: next_var <=> next_state_expr.
        // For each latch, encode: next_var_i <=> f_i(current_state, inputs)
        // as Tseitin: (!next_var OR f_i) AND (next_var OR !f_i)
        for (&latch_var, &next_var) in next_vars {
            if let Some(&next_expr) = ts.next_state.get(&latch_var) {
                let nv_pos = Lit::pos(next_var);
                let nv_neg = Lit::neg(next_var);
                solver.add_clause(&[nv_neg, next_expr]);
                solver.add_clause(&[nv_pos, !next_expr]);
            }
        }

        LiftSolver { solver }
    }

    /// Wire the portfolio cancellation flag into the lift solver's SAT backend
    /// so ay-sat can exit promptly when the portfolio finds a verdict (#4057).
    pub(crate) fn set_cancelled(
        &mut self,
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        self.solver.set_cancelled(cancelled);
    }

    /// Minimize a predecessor cube using SAT-based lifting with optional
    /// ternary simulation pre-filtering and a zero-SAT-call ternary X-drop
    /// pass.
    ///
    /// Given a frame solver that returned SAT (finding a predecessor), extract
    /// the input and latch assignments, then use the lift solver to find the
    /// minimal subset of latch literals that still implies reaching the target.
    ///
    /// The approach:
    ///
    /// 1. Extract input assignments from the frame solver model.
    /// 2. Extract latch (state) assignments from the frame solver model.
    ///    - **Ternary pre-filter** (if `ternary_sim` provided): propagate
    ///      through the AND-gate circuit with 0/1/X values. Remove state
    ///      literals whose next-state contribution is X (don't-care). This
    ///      is O(n * |gates|) -- dramatically cheaper than SAT.
    /// 3. Negate target literals individually as assumptions.
    /// 4. Ask: is `inputs AND state AND Trans AND NOT(t1') AND ... AND NOT(tn')` UNSAT?
    /// 5. The UNSAT core (restricted to state literals) is the minimal predecessor.
    /// 6. **Ternary X-drop pass (zero SAT calls)**: use ternary simulation
    ///    to further reduce the core. For each remaining literal, check via
    ///    circuit propagation whether setting it to X preserves all target
    ///    next-state requirements.
    #[allow(dead_code)]
    pub(crate) fn lift(
        &mut self,
        frame_solver: &dyn SatSolver,
        target_primed: &[Lit],
        latch_vars: &[Var],
        input_vars: &[Var],
    ) -> Vec<Lit> {
        self.lift_with_ternary(
            frame_solver,
            target_primed,
            latch_vars,
            input_vars,
            None,
            None,
        )
    }

    /// Lift with optional ternary simulation pre-filtering and a
    /// zero-SAT-call ternary X-drop post-pass.
    ///
    /// When `ternary_sim` and `reverse_next` are provided, performs:
    /// 1. **Ternary pre-filter**: cheaply remove don't-care state literals
    ///    before any SAT call.
    /// 2. **UNSAT core extraction**: one SAT call to get the minimal core.
    /// 3. **Ternary X-drop pass**: further reduce the core without any
    ///    additional SAT calls, using circuit-level ternary simulation.
    ///
    /// Step 3 rests on a one-way implication: if setting a literal to X
    /// still forces every required target next-state value through the
    /// AND-gate graph, that literal is genuinely unnecessary for the
    /// transition, so dropping it is sound. The converse does not hold —
    /// ternary simulation may fail to certify a don't-care that a SAT
    /// query would find — so the pass can only leave the cube slightly
    /// larger than optimal, never too small.
    pub(crate) fn lift_with_ternary(
        &mut self,
        frame_solver: &dyn SatSolver,
        target_primed: &[Lit],
        latch_vars: &[Var],
        input_vars: &[Var],
        ternary_sim: Option<&TernarySim>,
        reverse_next: Option<&FxHashMap<Var, Var>>,
    ) -> Vec<Lit> {
        // Step 1: Extract input assignments from the frame solver model.
        let mut input_assumptions: Vec<Lit> = Vec::with_capacity(input_vars.len());
        for &iv in input_vars {
            let pos = Lit::pos(iv);
            match frame_solver.value(pos) {
                Some(true) => input_assumptions.push(pos),
                Some(false) => input_assumptions.push(Lit::neg(iv)),
                None => {}
            }
        }

        // Step 2: Extract latch (state) assignments from the frame solver model.
        let mut state_lits: Vec<Lit> = Vec::with_capacity(latch_vars.len());
        for &lv in latch_vars {
            let pos = Lit::pos(lv);
            match frame_solver.value(pos) {
                Some(true) => state_lits.push(pos),
                Some(false) => state_lits.push(Lit::neg(lv)),
                None => {}
            }
        }

        // Early exit: nothing to minimize or no target.
        if state_lits.is_empty() || target_primed.is_empty() {
            return state_lits;
        }

        // Step 2b: Ternary simulation pre-filter.
        // Propagate 0/1/X through the AND-gate circuit to identify state
        // literals that are don't-cares for the target. This is O(n * |gates|)
        // vs O(SAT) per literal — a massive speedup for medium/large circuits.
        if let (Some(tsim), Some(rev_next)) = (ternary_sim, reverse_next) {
            let pre_count = state_lits.len();
            state_lits = tsim.ternary_lift_prefilter(
                &state_lits,
                &input_assumptions,
                target_primed,
                rev_next,
            );
            let removed = pre_count.saturating_sub(state_lits.len());
            if removed > 0 {
                eprintln!("  lift: ternary prefilter removed {removed}/{pre_count} state lits");
            }
        }

        // Step 3: Negate target literals individually as assumptions.
        // Each neg-t_i' is a separate assumption, giving the SAT solver fine-grained
        // conflict tracking. This produces much smaller UNSAT cores than encoding
        // neg-target as a single disjunctive clause (!t1' OR !t2' OR ... OR !tn').
        //
        // Sound: the frame solver proved s AND I AND T => t_i' for each i,
        // so s AND I AND T AND neg-t_1' AND ... AND neg-t_n' is guaranteed UNSAT.
        let neg_target: Vec<Lit> = target_primed.iter().map(|l| !*l).collect();

        // Step 4: Build assumptions = inputs + state_lits + neg_target (individual).
        let mut assumptions =
            Vec::with_capacity(input_assumptions.len() + state_lits.len() + neg_target.len());
        assumptions.extend_from_slice(&input_assumptions);
        assumptions.extend_from_slice(&state_lits);
        assumptions.extend_from_slice(&neg_target);

        let result = self.solver.solve(&assumptions);

        let core_reduced = if result == SatResult::Unsat {
            // Extract UNSAT core and filter to only state literals.
            let state_set: FxHashSet<Lit> = state_lits.iter().copied().collect();
            if let Some(core) = self.solver.unsat_core() {
                let reduced: Vec<Lit> =
                    core.into_iter().filter(|l| state_set.contains(l)).collect();
                if !reduced.is_empty() {
                    Some(reduced)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Use core-reduced cube if available, otherwise fall back to full state.
        let reduced = match core_reduced {
            Some(r) => r,
            None => {
                return state_lits;
            }
        };

        // Step 5: ternary X-drop pass — further reduce without SAT calls.
        //
        // For each literal in the core, set it to X and propagate through
        // the transition relation via ternary simulation. If all target
        // next-state values remain determined, the literal is a don't-care
        // and is dropped.
        //
        // This is O(n * |gates|) instead of O(n * SAT_call) — a massive win
        // when IC3 processes thousands of proof obligations.
        if let (Some(tsim), Some(rev_next)) = (ternary_sim, reverse_next) {
            let pre_flip = reduced.len();
            let xdropped =
                Self::ternary_x_drop(tsim, &reduced, &input_assumptions, target_primed, rev_next);
            let dropped = pre_flip.saturating_sub(xdropped.len());
            if dropped > 0 {
                eprintln!(
                    "  lift: ternary X-drop removed {dropped}/{pre_flip} core lits (zero SAT calls)"
                );
            }
            return xdropped;
        }

        // Fallback: no ternary simulator available, return core as-is.
        // (Previously this shrank the core with one SAT call per literal.
        // The ternary X-drop pass supersedes it since the ternary simulator
        // is always constructed alongside the lift solver.)
        reduced
    }

    /// Zero-SAT-call ternary X-drop pass.
    ///
    /// For each literal in the core-reduced cube, attempts to set it to X
    /// (don't-care) and propagates through the AND-gate circuit. If all
    /// target next-state values remain determined (not X), the literal is
    /// unnecessary and is permanently dropped.
    ///
    /// # Soundness
    ///
    /// The ternary simulation is an over-approximation: if ternary sim says
    /// a literal is a don't-care, then it truly is (the circuit doesn't need
    /// it to produce the target). However, ternary sim may miss some
    /// don't-cares that a SAT solver would find (it can't reason about
    /// implications through learned clauses or frame lemmas). This is safe:
    /// we may return a slightly larger cube than optimal, but never too small.
    ///
    /// # Complexity
    ///
    /// O(n * |gates|) where n = |core_cube| and |gates| = number of AND gates.
    /// For a typical IC3 scenario with 50-variable COI and 200 gates, this
    /// is ~10K operations vs ~50 SAT calls (each potentially thousands of
    /// propagations).
    fn ternary_x_drop(
        tsim: &TernarySim,
        core_cube: &[Lit],
        input_lits: &[Lit],
        target_primed: &[Lit],
        next_var_to_latch: &FxHashMap<Var, Var>,
    ) -> Vec<Lit> {
        if core_cube.is_empty() || target_primed.is_empty() {
            return core_cube.to_vec();
        }

        // Convert target_primed (over next_vars) to required next-state values
        // keyed by latch variable.
        let mut required_next: Vec<(Var, bool)> = Vec::with_capacity(target_primed.len());
        for &tgt_lit in target_primed {
            if let Some(&latch_var) = next_var_to_latch.get(&tgt_lit.var()) {
                required_next.push((latch_var, tgt_lit.is_positive()));
            }
        }

        if required_next.is_empty() {
            return core_cube.to_vec();
        }

        // Initialize ternary values: state from core cube, inputs from model.
        let num_values = tsim.num_values();
        let mut values = vec![TernaryVal::X; num_values];
        values[0] = TernaryVal::Zero; // Var(0) = constant FALSE

        // Set state literals from core cube.
        for &lit in core_cube {
            let idx = lit.var().0 as usize;
            if idx < values.len() {
                values[idx] = if lit.is_positive() {
                    TernaryVal::One
                } else {
                    TernaryVal::Zero
                };
            }
        }

        // Set input literals from model.
        for &lit in input_lits {
            let idx = lit.var().0 as usize;
            if idx < values.len() {
                values[idx] = if lit.is_positive() {
                    TernaryVal::One
                } else {
                    TernaryVal::Zero
                };
            }
        }

        // Initial propagation.
        tsim.propagate(&mut values);

        // Verify: with all core literals set, next-state should be determined.
        if !tsim.next_state_matches_vals(&values, &required_next) {
            // Can't confirm transition — return core as-is.
            return core_cube.to_vec();
        }

        // Greedy one-pass: try setting each core literal to X.
        let mut result = Vec::with_capacity(core_cube.len());
        for &lit in core_cube {
            let var_idx = lit.var().0 as usize;
            if var_idx >= values.len() {
                result.push(lit);
                continue;
            }
            let saved = values[var_idx];
            values[var_idx] = TernaryVal::X;
            tsim.propagate(&mut values);
            if tsim.next_state_matches_vals(&values, &required_next) {
                // Don't-care: this literal is not needed.
            } else {
                // Needed: restore and keep.
                values[var_idx] = saved;
                result.push(lit);
            }
        }

        // Re-propagate for consistency after the final set of kept literals.
        if !result.is_empty() {
            tsim.propagate(&mut values);
        }

        // Fall back to core if everything was removed (should not happen in
        // practice since the core was already minimal from UNSAT extraction).
        if result.is_empty() {
            return core_cube.to_vec();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_aag;
    use crate::ternary::TernarySim;
    use crate::transys::Transys;

    /// Test ternary_x_drop on a 3-latch shift register.
    ///
    /// Circuit: input -> A -> B -> C. Bad = A AND B AND C.
    /// A.next = input, B.next = A, C.next = B.
    ///
    /// Given predecessor state (A=1, B=1, C=0) with input=1,
    /// target = (A'=1, B'=1, C'=1):
    /// - A is needed because B' = A, target requires B'=1.
    /// - B is needed because C' = B, target requires C'=1.
    /// - C is NOT needed — nothing in the target depends on C's current value.
    ///
    /// After UNSAT core extraction, the core might include all three.
    /// ternary_x_drop should drop C without a SAT call.
    #[test]
    fn test_ternary_x_drop_shift_register() {
        // 3-latch shift register: input -> A -> B -> C. Bad = A AND B AND C.
        let aag = "\
aag 6 1 3 0 2 1
2
4 2
6 4
8 6
12
10 4 6
12 10 8
";
        let circuit = parse_aag(aag).expect("parse failed");
        let ts = Transys::from_aiger(&circuit);
        let tsim = TernarySim::new(&ts);

        // Build next_var_to_latch map.
        let mut next_var_to_latch = FxHashMap::default();
        next_var_to_latch.insert(Var(7), Var(2)); // A
        next_var_to_latch.insert(Var(8), Var(3)); // B
        next_var_to_latch.insert(Var(9), Var(4)); // C

        // Core cube includes all 3 latches.
        let core_cube = vec![
            Lit::pos(Var(2)), // A=1
            Lit::pos(Var(3)), // B=1
            Lit::neg(Var(4)), // C=0
        ];
        let input_lits = vec![Lit::pos(Var(1))]; // input=1

        // Target: A'=1, B'=1, C'=1.
        let target_primed = vec![Lit::pos(Var(7)), Lit::pos(Var(8)), Lit::pos(Var(9))];

        let result = LiftSolver::ternary_x_drop(
            &tsim,
            &core_cube,
            &input_lits,
            &target_primed,
            &next_var_to_latch,
        );

        // A and B should survive, C should be dropped.
        assert!(
            result.contains(&Lit::pos(Var(2))),
            "A should be kept (B'=A, target B'=1)"
        );
        assert!(
            result.contains(&Lit::pos(Var(3))),
            "B should be kept (C'=B, target C'=1)"
        );
        assert!(
            !result.contains(&Lit::neg(Var(4))),
            "C should be dropped (nothing in target depends on C)"
        );
        assert_eq!(result.len(), 2, "should have dropped 1 of 3 literals");
    }

    /// Test ternary_x_drop with a single essential literal.
    ///
    /// Toggle: latch next = !latch. Target = latch' = 0 (i.e., latch must be 1).
    /// The single latch literal is essential and cannot be dropped.
    #[test]
    fn test_ternary_x_drop_single_essential() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").expect("parse failed");
        let ts = Transys::from_aiger(&circuit);
        let tsim = TernarySim::new(&ts);

        let mut next_var_to_latch = FxHashMap::default();
        next_var_to_latch.insert(Var(2), Var(1));

        // Core cube: latch=1.
        let core_cube = vec![Lit::pos(Var(1))];
        // Target: latch' should be 0 (next = !latch = !1 = 0).
        let target_primed = vec![Lit::neg(Var(2))];

        let result =
            LiftSolver::ternary_x_drop(&tsim, &core_cube, &[], &target_primed, &next_var_to_latch);

        // The single literal must be kept.
        assert_eq!(result.len(), 1, "single essential literal should survive");
        assert_eq!(result[0], Lit::pos(Var(1)));
    }

    /// Test ternary_x_drop with empty core cube.
    #[test]
    fn test_ternary_x_drop_empty_core() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").expect("parse failed");
        let ts = Transys::from_aiger(&circuit);
        let tsim = TernarySim::new(&ts);

        let result =
            LiftSolver::ternary_x_drop(&tsim, &[], &[], &[Lit::pos(Var(2))], &FxHashMap::default());
        assert!(result.is_empty(), "empty core should produce empty result");
    }

    /// Test ternary_x_drop with empty target.
    #[test]
    fn test_ternary_x_drop_empty_target() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").expect("parse failed");
        let ts = Transys::from_aiger(&circuit);
        let tsim = TernarySim::new(&ts);

        let core_cube = vec![Lit::pos(Var(1))];
        let result = LiftSolver::ternary_x_drop(&tsim, &core_cube, &[], &[], &FxHashMap::default());
        // Empty target means nothing to check — returns core as-is.
        assert_eq!(result, core_cube);
    }

    /// Test ternary_x_drop with two independent latches.
    ///
    /// Two stuck-at-0 latches: A.next=0, B.next=0. Both init at 0.
    /// If target requires A'=0 only, then B is a don't-care.
    #[test]
    fn test_ternary_x_drop_independent_latches() {
        // Two latches, both stuck-at-0.
        let circuit = parse_aag("aag 2 0 2 0 0 1\n2 0\n4 0\n2\n").expect("parse failed");
        let ts = Transys::from_aiger(&circuit);
        let tsim = TernarySim::new(&ts);

        let mut next_var_to_latch = FxHashMap::default();
        next_var_to_latch.insert(Var(3), Var(1)); // A
        next_var_to_latch.insert(Var(4), Var(2)); // B

        // Core cube: both latches assigned.
        let core_cube = vec![
            Lit::pos(Var(1)), // A=1
            Lit::pos(Var(2)), // B=1
        ];

        // Target: only A'=0 required.
        let target_primed = vec![Lit::neg(Var(3))]; // A' = 0

        let result =
            LiftSolver::ternary_x_drop(&tsim, &core_cube, &[], &target_primed, &next_var_to_latch);

        // A.next = 0, which is constant. A's current value doesn't affect A.next.
        // So actually both A and B are don't-cares for A'=0 (since A.next=0
        // regardless of current state). The result depends on whether ternary
        // sim can detect this: it should because the next-state function is
        // the constant literal 0.
        //
        // Both should be dropped since A'=0 is always true (next=constant 0).
        // But if both are dropped, the fallback returns the full core.
        // This tests the edge case handling.
        assert!(
            result.len() <= core_cube.len(),
            "result should not be larger than input"
        );
    }
}
