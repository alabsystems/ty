// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! k-Induction engine for AIGER safety checking.
//!
//! Uses a single SAT solver instance (a memory-efficiency choice):
//! - Transition clauses are loaded incrementally for each unrolled step.
//! - `!bad` at depths 0..k-1 is asserted permanently (for the induction check).
//! - The init constraints are used as **assumptions** for the base case check:
//!   unit init clauses directly, non-unit init clauses via a single activation
//!   literal (`(!init_act OR C)` permanently, `init_act` assumed base-only) so
//!   the induction step is never constrained by init.
//! - The induction step check uses the bad literal at depth k as an assumption.
//!
//! Algorithm:
//! 1. Unroll transition to step k.
//! 2. **Induction step first** (k > 0): Assert `!bad` at 0..k-1 permanently.
//!    Check if `bad@k` is satisfiable (without init). If UNSAT -> SAFE.
//! 3. **Base case**: Check if `bad@k` is reachable from init by adding init
//!    constraints as assumptions. If SAT -> UNSAFE.
//! 4. Increment k and repeat.
//!
//! Checking induction BEFORE base is key: an induction success (UNSAT) is
//! conclusive and avoids wasting time on base case checks that can't prove safety.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::bmc::gpu_exhaustive::{try_gpu_exhaustive_bmc, GpuExhaustBmc};
use crate::check_result::CheckResult;
use crate::sat_types::{Lit, SatResult, SatSolver, SolverBackend};
use crate::transys::{Transys, VarRenamer};

/// Configuration for the k-induction engine.
#[derive(Debug, Clone, Default)]
pub struct KindConfig {
    /// Enable simple-path constraint in the inductive step solver.
    ///
    /// The simple-path constraint asserts that all states in the induction
    /// trace are pairwise distinct (no two time steps visit the same state).
    /// This strengthens the induction hypothesis, helping k-induction prove
    /// properties that are not directly k-inductive but become so when
    /// restricted to simple (non-looping) paths.
    pub simple_path: bool,

    /// Skip the base case (BMC) check. When true, only the induction step
    /// is checked. This is useful when BMC is already running in the portfolio
    /// and we only need the induction proof.
    pub skip_bmc: bool,
}

/// k-Induction engine.
///
/// Uses a single SAT solver for both base and step checks. Init constraints
/// are added as assumptions for the base case, while `!bad` at earlier depths
/// is asserted permanently for the induction step.
///
/// The single-solver design halves memory usage
/// compared to the two-solver approach.
pub struct KindEngine {
    ts: Transys,
    /// Single solver for both base and step checks.
    solver: Box<dyn SatSolver>,
    renamer: VarRenamer,
    /// How far we've unrolled the transition relation.
    unrolled_depth: usize,
    /// How far we've asserted `!bad` (for induction step).
    bad_asserted_depth: usize,
    /// Bad literal in base encoding.
    bad_base: Lit,
    /// Init constraint literals (used as assumptions for base case only).
    /// Contains the unit init clause literals plus, when the system has
    /// non-unit init clauses, the activation literal that enables them.
    init_assumption_lits: Vec<Lit>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
    /// Configuration.
    config: KindConfig,
}

impl KindEngine {
    /// Create a k-induction engine for `ts` with the default configuration.
    pub fn new(ts: Transys) -> Self {
        Self::with_config(ts, KindConfig::default())
    }

    /// Create a k-induction engine with the simple-path constraint enabled.
    pub fn new_simple_path(ts: Transys) -> Self {
        Self::with_config(
            ts,
            KindConfig {
                simple_path: true,
                skip_bmc: false,
            },
        )
    }

    /// Create a k-induction engine with a specific configuration.
    pub fn with_config(ts: Transys, config: KindConfig) -> Self {
        Self::with_config_and_backend(ts, config, SolverBackend::AYSat)
    }

    /// Create a k-induction engine with a specific configuration and solver backend.
    ///
    /// This allows the portfolio to spawn k-induction engines backed by different
    /// ay-sat configurations (Luby restarts, stable mode, VMTF branching, etc.)
    /// for diversity. Mirrors the backend selection pattern established for BMC.
    pub fn with_config_and_backend(
        ts: Transys,
        config: KindConfig,
        backend: SolverBackend,
    ) -> Self {
        let mut solver: Box<dyn SatSolver> = backend.make_solver(ts.max_var + 1);
        let mut renamer = VarRenamer::new(&ts);

        // Constrain Var(0) = false (AIGER convention: literal 0 = constant FALSE).
        solver.add_clause(&[Lit::TRUE]);

        // Initialize with trans at step 0
        renamer.ensure_step(0);
        solver.ensure_vars(renamer.max_allocated());
        ts.load_trans_renamed(solver.as_mut(), &|lit| renamer.rename_lit(lit, 0));

        // Activation literal guarding non-unit init clauses (soundness).
        //
        // ALL init constraints must be base-case-only: the induction-step
        // solve must be free of init constraints, otherwise its UNSAT no
        // longer proves k-inductiveness (the step degenerates into bounded
        // reachability from init, and an "UNSAT" step yields a false Safe).
        // Unit init clauses become plain assumptions; non-unit init clauses
        // C are added permanently as (!init_act OR C) and `init_act` is
        // assumed ONLY in the base-case solve, so the step solve leaves the
        // start state unconstrained by init.
        //
        // Allocated via the renamer's collision-safe aux allocator (the same
        // mechanism BMC uses for its any-bad accumulator), BEFORE
        // `get_bad_lit` pulls a fresh var from the solver, so the two
        // allocation namespaces stay disjoint.
        let init_act: Option<Lit> = if ts.init_clauses.iter().any(|c| c.lits.len() != 1) {
            let act_var = renamer.alloc_aux_var();
            solver.ensure_vars(renamer.max_allocated());
            Some(Lit::pos(act_var))
        } else {
            None
        };

        let bad_base = ts.get_bad_lit(solver.as_mut());

        // Collect init constraint literals for assumption-based base case.
        // For the base case: assumptions include init lits + init_act + bad@k.
        // For the step case: no init assumptions, just bad@k.
        let mut init_assumption_lits = Vec::new();
        let mut guarded_clause = Vec::new();
        for clause in &ts.init_clauses {
            if clause.lits.len() == 1 {
                // Unit clauses can be used directly as assumptions.
                init_assumption_lits.push(clause.lits[0]);
            } else {
                // Non-unit clauses are guarded by the activation literal so
                // they only bind when `init_act` is assumed (base case).
                let act = init_act.expect("init_act allocated for non-unit init clauses");
                guarded_clause.clear();
                guarded_clause.push(!act);
                guarded_clause.extend_from_slice(&clause.lits);
                solver.add_clause(&guarded_clause);
            }
        }
        if let Some(act) = init_act {
            init_assumption_lits.push(act);
        }

        KindEngine {
            ts,
            solver,
            renamer,
            unrolled_depth: 0,
            bad_asserted_depth: 0,
            bad_base,
            init_assumption_lits,
            cancelled: Arc::new(AtomicBool::new(false)),
            config,
        }
    }

    /// Set the cancellation flag.
    ///
    /// Propagates the flag to the SAT solver so ay-sat's CDCL loop can exit
    /// promptly when cancelled (#4057).
    pub fn set_cancelled(&mut self, cancelled: Arc<AtomicBool>) {
        self.solver.set_cancelled(cancelled.clone());
        self.cancelled = cancelled;
    }

    /// Run k-induction up to `max_k` steps.
    ///
    /// Uses an induction-first strategy:
    /// 1. Unroll to depth k.
    /// 2. Check induction step (no init, !bad@0..k-1, bad@k?) -- UNSAT -> SAFE.
    /// 3. Check base case (init + bad@k?) -- SAT -> UNSAFE.
    /// 4. Repeat with k+1.
    pub fn check(&mut self, max_k: usize) -> CheckResult {
        // GPU exhaustive base-case discharge (SAT lane, #follow-on).
        //
        // A single call at depth `max_k` unrolls the transition relation into
        // one combinational AIG and enumerates ALL free-variable assignments on
        // the GPU. `BoundedSafe` is a COMPLETE proof that no bad state is
        // reachable in <= max_k steps — semantically the conjunction of every
        // per-depth base UNSAT check the loop below performs (k = 0..=max_k).
        // So one BoundedSafe discharges the entire base obligation and the
        // per-depth base SAT solves can be skipped; the loop then only runs the
        // inductive step (which is what actually proves `Safe`).
        //
        // Purely additive and fail-closed: the carrier declines (`None`) on a
        // non-CUDA host, relational init, any constraint, or when the free set
        // exceeds the exhaustive cap; `Unsafe` and every decline leave
        // `base_discharged = false`, so the CPU per-depth base loop runs exactly
        // as before (and re-derives a `verify_witness`-checked counterexample
        // trace on the unsafe direction). A wrong/incomplete GPU result can only
        // decline — it can never fabricate a proof.
        let mut base_discharged = false;
        if !self.config.skip_bmc {
            let cancel = self.cancelled.clone();
            let cancel_fn = move || cancel.load(Ordering::Relaxed);
            if let Some(GpuExhaustBmc::BoundedSafe) =
                try_gpu_exhaustive_bmc(&self.ts, max_k, &cancel_fn)
            {
                base_discharged = true;
            }
        }

        for k in 0..=max_k {
            if self.cancelled.load(Ordering::Relaxed) {
                return CheckResult::Unknown {
                    reason: "cancelled".into(),
                };
            }

            // Unroll transition to step k
            if k > 0 {
                self.unroll_to(k);
                // Add simple-path constraints if enabled
                self.add_simple_path_constraint(k);
            }

            // Induction step check (for k > 0):
            // Assert !bad at steps 0..k-1, check if bad@k is satisfiable
            // WITHOUT init constraints. UNSAT -> property is k-inductive -> SAFE.
            if k > 0 {
                self.assert_no_bad_up_to(k - 1);

                let bad_at_k = self.renamer.rename_lit(self.bad_base, k);
                match self.solver.solve(&[bad_at_k]) {
                    SatResult::Unsat => {
                        if self.config.simple_path {
                            // Simple-path k-induction is theoretically sound
                            // (Sheeran, Singh, Stalmarck, SAT 2000). However,
                            // the step UNSAT can be VACUOUS if simple-path
                            // constraints alone exhaust the state space — i.e.,
                            // no simple path of length k+1 exists regardless
                            // of the property.
                            //
                            // Vacuity check: solve the step formula WITHOUT
                            // bad@k. If UNSAT, simple-path has exhausted the
                            // reachable states and the proof is vacuous —
                            // continue to the next k. If SAT, simple paths of
                            // length k+1 exist, so the step UNSAT genuinely
                            // means the property is k-inductive on simple
                            // paths.
                            //
                            // This replaces the blanket guard from #4039 that
                            // returned Unknown for ALL simple-path UNSAT,
                            // effectively disabling simple-path entirely.
                            // The original #4039 bug on microban_24 was caused
                            // by a ay-sat solver bug (ay#8661), not a
                            // theoretical limitation of simple-path k-induction.
                            match self.solver.solve(&[]) {
                                SatResult::Unsat => {
                                    // Vacuous: no simple paths of length k+1
                                    // exist. The step UNSAT is due to state-
                                    // space exhaustion, not k-inductiveness.
                                    // Continue to next k.
                                    eprintln!(
                                        "kind: simple-path vacuous at k={k} \
                                         (state space exhausted). Continuing.",
                                    );
                                }
                                SatResult::Sat => {
                                    // Non-vacuous: simple paths of length k+1
                                    // exist, but none reach a bad state. The
                                    // property is genuinely k-inductive on
                                    // simple paths.
                                    return CheckResult::Safe;
                                }
                                SatResult::Unknown => {
                                    // Solver inconclusive on vacuity check.
                                    // Be conservative: don't claim Safe.
                                    return CheckResult::Unknown {
                                        reason: format!(
                                            "simple-path vacuity check \
                                             inconclusive at k={k}"
                                        ),
                                    };
                                }
                            }
                        } else {
                            // Standard k-induction (no simple-path): step UNSAT
                            // means the property is genuinely k-inductive.
                            return CheckResult::Safe;
                        }
                    }
                    SatResult::Sat => {
                        // Induction step fails -- continue to larger k
                    }
                    SatResult::Unknown => {
                        return CheckResult::Unknown {
                            reason: format!("step solver unknown at k={k}"),
                        };
                    }
                }
            }

            // Base case: is bad reachable at step k from init?
            // Use init constraints as assumptions alongside bad@k.
            //
            // Skipped entirely when the GPU exhaustive lane already discharged
            // the whole 0..=max_k base obligation (`base_discharged`): every
            // per-depth solve here would be provably UNSAT.
            if !self.config.skip_bmc && !base_discharged {
                let bad_at_k = self.renamer.rename_lit(self.bad_base, k);
                let mut assumptions = self.init_assumption_lits.clone();
                assumptions.push(bad_at_k);

                match self.solver.solve(&assumptions) {
                    SatResult::Sat => {
                        let trace = self.extract_trace(k);
                        // Verify witness by simulating the circuit (#4103).
                        // If the trace doesn't actually reach a bad state,
                        // the SAT solver returned a spurious result.
                        if let Err(reason) = self.ts.verify_witness(&trace) {
                            eprintln!(
                                "k-ind: spurious SAT at k={k} (witness verification \
                                 failed: {reason}). Ignoring and continuing search.",
                            );
                            // Continue to next k instead of reporting UNSAFE
                        } else {
                            return CheckResult::Unsafe { depth: k, trace };
                        }
                    }
                    SatResult::Unsat => {
                        // Not reachable at step k, continue
                    }
                    SatResult::Unknown => {
                        return CheckResult::Unknown {
                            reason: format!("base solver unknown at k={k}"),
                        };
                    }
                }
            }
        }

        CheckResult::Unknown {
            reason: format!("k-induction reached bound {max_k}"),
        }
    }

    /// Unroll the transition relation to step k.
    fn unroll_to(&mut self, k: usize) {
        while self.unrolled_depth < k {
            let next_k = self.unrolled_depth + 1;
            self.renamer.ensure_step(next_k);
            self.solver.ensure_vars(self.renamer.max_allocated());

            let renamer = &self.renamer;
            self.ts
                .load_trans_renamed(self.solver.as_mut(), &|lit| renamer.rename_lit(lit, next_k));

            // Wire latches from step next_k-1 to step next_k
            for &latch_var in &self.ts.latch_vars {
                let next_lit = self.ts.next_state[&latch_var];
                let next_at_prev = self.renamer.rename_lit(next_lit, next_k - 1);
                let curr_at_k = Lit::pos(self.renamer.rename_var(latch_var, next_k));
                self.solver.add_clause(&[!curr_at_k, next_at_prev]);
                self.solver.add_clause(&[curr_at_k, !next_at_prev]);
            }

            self.unrolled_depth = next_k;
        }
    }

    /// Assert !bad at all steps from bad_asserted_depth+1 up to k (inclusive).
    /// These are permanent clauses for the induction step.
    fn assert_no_bad_up_to(&mut self, k: usize) {
        while self.bad_asserted_depth < k {
            let next_depth = self.bad_asserted_depth + 1;
            // We assert !bad at depth `bad_asserted_depth` (which is the step
            // we haven't asserted yet). But we need to be careful: at depth 0,
            // we haven't asserted anything yet.
            let bad_at_depth = self
                .renamer
                .rename_lit(self.bad_base, self.bad_asserted_depth);
            self.solver.add_clause(&[!bad_at_depth]);
            self.bad_asserted_depth = next_depth;
        }
        // Also assert at depth k itself if not yet done
        if self.bad_asserted_depth <= k {
            let bad_at_k = self.renamer.rename_lit(self.bad_base, k);
            self.solver.add_clause(&[!bad_at_k]);
            self.bad_asserted_depth = k + 1;
        }
    }

    /// Add simple-path constraint between step `new_step` and all previous steps.
    ///
    /// For each pair (i, new_step) with i < new_step, asserts that the latch
    /// states at steps i and new_step are pairwise distinct. Encoded as:
    ///   OR over all latches L: (L@i XOR L@new_step)
    /// which is equivalent to:
    ///   OR over all latches L: (L@i != L@new_step)
    ///
    /// This means at least one latch must differ between any two time steps,
    /// preventing the induction trace from revisiting states. This strengthens
    /// the induction hypothesis and helps prove properties that are not directly
    /// k-inductive but become so on simple paths.
    fn add_simple_path_constraint(&mut self, new_step: usize) {
        if !self.config.simple_path || self.ts.latch_vars.is_empty() {
            return;
        }

        for earlier_step in 0..new_step {
            // Build: OR_{latch L} (L@earlier XOR L@new_step)
            // XOR(a, b) = (a AND !b) OR (!a AND b)
            // To encode this in CNF without auxiliary variables for each latch,
            // we use the Tseitin-like approach: introduce a fresh diff variable
            // for each latch and assert the disjunction of diffs.
            //
            // For each latch L:
            //   diff_L <=> (L@earlier XOR L@new_step)
            //   Tseitin:
            //     diff_L -> (L@earlier XOR L@new_step):
            //       (!diff_L OR L@earlier OR L@new_step) AND (!diff_L OR !L@earlier OR !L@new_step)
            //     (L@earlier XOR L@new_step) -> diff_L:
            //       (diff_L OR L@earlier OR !L@new_step) AND (diff_L OR !L@earlier OR L@new_step)
            //
            // Then assert: OR_{latch L} diff_L

            let mut diff_lits = Vec::with_capacity(self.ts.latch_vars.len());

            for &latch_var in &self.ts.latch_vars {
                let l_earlier = Lit::pos(self.renamer.rename_var(latch_var, earlier_step));
                let l_new = Lit::pos(self.renamer.rename_var(latch_var, new_step));

                // Allocate the Tseitin diff variable through the RENAMER's aux
                // allocator, not `solver.new_var()`. The renamer owns the
                // variable-index namespace that `unroll_to` draws step-k latch
                // and gate variables from; a diff var minted directly from the
                // solver is invisible to the renamer, so the next `ensure_step`
                // re-hands that same index out as a future step variable. That
                // collision pins a later latch (e.g. l@k) to a stale diff
                // constraint, yielding spurious UNSAT in the step solve (and a
                // possible false Safe). Minting via the renamer keeps the two
                // allocators in lockstep so step vars never reuse a diff index.
                let diff_var = self.renamer.alloc_aux_var();
                self.solver.ensure_vars(self.renamer.max_allocated());
                let diff = Lit::pos(diff_var);

                // diff -> XOR(l_earlier, l_new):
                // !diff OR l_earlier OR l_new
                self.solver.add_clause(&[!diff, l_earlier, l_new]);
                // !diff OR !l_earlier OR !l_new
                self.solver.add_clause(&[!diff, !l_earlier, !l_new]);

                // XOR(l_earlier, l_new) -> diff:
                // diff OR l_earlier OR !l_new
                self.solver.add_clause(&[diff, l_earlier, !l_new]);
                // diff OR !l_earlier OR l_new
                self.solver.add_clause(&[diff, !l_earlier, l_new]);

                diff_lits.push(diff);
            }

            // At least one latch must differ: OR(diff_0, diff_1, ..., diff_n)
            self.solver.add_clause(&diff_lits);
        }
    }

    /// Extract a counterexample trace from the solver (named string keys).
    fn extract_trace(&self, depth: usize) -> Vec<rustc_hash::FxHashMap<String, bool>> {
        let mut trace = Vec::with_capacity(depth + 1);
        for k in 0..=depth {
            let mut step = rustc_hash::FxHashMap::default();
            for (idx, &latch_var) in self.ts.latch_vars.iter().enumerate() {
                let renamed = self.renamer.rename_var(latch_var, k);
                let val = self.solver.value(Lit::pos(renamed)).unwrap_or(false);
                step.insert(format!("l{idx}"), val);
            }
            for (idx, &input_var) in self.ts.input_vars.iter().enumerate() {
                let renamed = self.renamer.rename_var(input_var, k);
                let val = self.solver.value(Lit::pos(renamed)).unwrap_or(false);
                step.insert(format!("i{idx}"), val);
            }
            trace.push(step);
        }
        trace
    }

    /// Extract a literal-level witness trace from the solver.
    ///
    /// Each time step is a vector of literals over the original (base)
    /// latch and input variables. Call this after `check()` returns
    /// `CheckResult::Unsafe` to get the raw witness.
    pub fn extract_lit_trace(&self, depth: usize) -> Vec<Vec<Lit>> {
        let mut trace = Vec::with_capacity(depth + 1);
        for k in 0..=depth {
            let mut step_lits = Vec::new();
            for &latch_var in &self.ts.latch_vars {
                let renamed = self.renamer.rename_var(latch_var, k);
                match self.solver.value(Lit::pos(renamed)) {
                    Some(true) => step_lits.push(Lit::pos(latch_var)),
                    Some(false) => step_lits.push(Lit::neg(latch_var)),
                    None => {}
                }
            }
            for &input_var in &self.ts.input_vars {
                let renamed = self.renamer.rename_var(input_var, k);
                match self.solver.value(Lit::pos(renamed)) {
                    Some(true) => step_lits.push(Lit::pos(input_var)),
                    Some(false) => step_lits.push(Lit::neg(input_var)),
                    None => {}
                }
            }
            trace.push(step_lits);
        }
        trace
    }
}
