// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! IC3 invariant validation: validate_invariant_budgeted, validate_invariant,
//! verify_consecution_independent, consecution_simple_fallback.

use std::sync::atomic::Ordering;

use super::config::ValidationStrategy;
use super::engine::Ic3Engine;
use crate::sat_types::{Lit, SatResult, SatSolver, SimpleSolver};

/// Outcome of per-lemma consecution verification (`IC3_VERIFY_LEMMAS`,
/// [`Ic3Engine::verify_lemma_consecution`]).
///
/// The two arms deliberately mirror the two ways an IC3 blocking step is
/// allowed to make progress: `Verified` lets the caller add the lemma
/// (strengthening the frame system), while `Refuted` carries a concrete
/// witness the caller can act on — fall back to a weaker lemma the witness
/// does not refute, or descend on the witness as a new proof obligation.
/// There is no bare "rejected" arm on purpose: rejection without a witness
/// invites "try the same thing again", which is not a progress-preserving
/// outcome (see the `Refuted` handler in `block_one` for the full argument).
pub(super) enum LemmaVerdict {
    /// The independent solver confirms the consecution query is UNSAT, or
    /// could not produce a checkable refutation (Unknown / unverifiable
    /// model). The lemma may be added.
    Verified,
    /// The independent solver produced a model that was re-validated against
    /// every clause of the verification formula: a genuine
    /// counterexample-to-consecution. `predecessor` is that model's full
    /// latch valuation — a state in F_{frame-1} ∧ ¬cube with a successor in
    /// the cube — for the caller to descend on.
    Refuted { predecessor: Vec<Lit> },
}

impl Ic3Engine {
    /// Validate the claimed inductive invariant using an independent SAT check.
    ///
    /// After IC3 converges, collect ALL lemmas from all frames and check:
    /// 1. Init => Inv  (initial state satisfies the invariant)
    /// 2. Inv AND T AND constraints => Inv'  (inductiveness: invariant is preserved)
    /// 3. Inv => !Bad  (invariant implies safety)
    ///
    /// Returns `Some(true)` if valid, `Some(false)` if unsound, `None` if
    /// the time budget was exhausted or portfolio cancellation was triggered.
    pub(super) fn validate_invariant_budgeted(&self) -> Option<bool> {
        // Strategy::None skips all validation — assume correct (#4121).
        // ONLY safe when another portfolio member validates with Auto.
        if self.config.validation_strategy == ValidationStrategy::None {
            eprintln!("IC3 validate: SKIPPING all validation (strategy=None)");
            return Some(true);
        }

        let validate_start = std::time::Instant::now();

        // Compute constraint-to-latch ratio for tier selection (#4121).
        // This uses constraint_lits (AIGER environment constraints), not
        // trans_clauses (Tseitin encoding). High constraint_lits/latch ratios
        // indicate circuits like qspiflash where the constraint propagation
        // overhead dominates solver time.
        let num_latches = self.ts.latch_vars.len();
        let constraint_ratio = if num_latches > 0 {
            self.ts.constraint_lits.len() as f64 / num_latches as f64
        } else {
            0.0
        };
        let is_high_constraint_ratio = constraint_ratio > 5.0;

        // Budget scales with circuit size. When proof verification is
        // explicitly enabled (#4216), use more generous budgets for larger
        // circuits since this is the primary defense-in-depth mechanism.
        let proof_verify = super::config::proof_verification_enabled();
        let base_budget_secs: f64 = if num_latches < 60 {
            10.0
        } else if num_latches <= 200 {
            if proof_verify {
                45.0
            } else {
                30.0
            }
        } else {
            if proof_verify {
                90.0
            } else {
                60.0
            }
        };
        // Budget selection: high-ratio circuits get tighter budgets (#4121),
        // but only as a cap. Small circuits keep their existing 10s budget.
        let budget_secs: f64 = if is_high_constraint_ratio {
            base_budget_secs.min(15.0)
        } else {
            base_budget_secs
        };
        let should_abort = |start: &std::time::Instant| -> bool {
            start.elapsed().as_secs_f64() > budget_secs || self.cancelled.load(Ordering::Relaxed)
        };

        // Collect ALL lemmas (clauses) from all frames.
        let mut all_lemmas: Vec<Vec<Lit>> = Vec::new();
        for frame in &self.frames.frames {
            for lemma in &frame.lemmas {
                all_lemmas.push(lemma.lits.clone());
            }
        }
        for lemma in &self.inf_lemmas {
            all_lemmas.push(lemma.lits.clone());
        }
        eprintln!(
            "IC3 validate: {} total lemmas across {} frames + inf (constraint_ratio={:.1}, strategy={:?})",
            all_lemmas.len(),
            self.frames.frames.len(),
            constraint_ratio,
            self.config.validation_strategy,
        );

        // Check 1: Init => Inv
        {
            let mut init_solver = self.make_fast_validation_solver();
            init_solver.add_clause(&[Lit::TRUE]);
            for clause in &self.ts.init_clauses {
                init_solver.add_clause(&clause.lits);
            }
            for clause in &self.ts.trans_clauses {
                init_solver.add_clause(&clause.lits);
            }
            for &constraint in &self.ts.constraint_lits {
                init_solver.add_clause(&[constraint]);
            }

            for (i, lemma) in all_lemmas.iter().enumerate() {
                if i > 0 && i % 50 == 0 && should_abort(&validate_start) {
                    eprintln!(
                        "IC3 validate: BUDGET EXCEEDED at Init=>Inv check lemma {}/{} ({:.1}s)",
                        i,
                        all_lemmas.len(),
                        validate_start.elapsed().as_secs_f64(),
                    );
                    return None;
                }
                let neg_lits: Vec<Lit> = lemma.iter().map(|l| !*l).collect();
                if init_solver.solve(&neg_lits) == SatResult::Sat {
                    eprintln!(
                        "IC3 VALIDATE FAIL: Init does NOT satisfy lemma {}: {:?}",
                        i, lemma
                    );
                    return Some(false);
                }
            }
            eprintln!("IC3 validate: Init => Inv OK");
        }

        // Check 2: Inv AND T AND constraints => Inv'
        //
        // Strategy::SkipConsecution skips this check entirely (#4121).
        // For Auto strategy, consecution is ALWAYS checked -- skipping it
        // is unsound when ay-sat has bugs that cause IC3 to converge on
        // non-inductive invariants. The solver tier is selected based on
        // circuit characteristics:
        //
        // - high ratio (>5x): AYNoPreprocess (skip SimpleSolver overhead)
        // - <= 60 latches AND low ratio: SimpleSolver (max independence)
        // - all others: AYNoPreprocess (faster, no BVE risk)
        //
        // For large circuits (>200 latches), we use AYNoPreprocess with a
        // generous budget. If the budget is exhausted, we return None
        // (indeterminate) rather than trusting an unvalidated invariant.
        #[allow(deprecated)]
        let skip_consecution =
            self.config.validation_strategy == ValidationStrategy::SkipConsecution;
        let skip_check2 = skip_consecution;

        if !skip_check2 {
            // For high-ratio circuits, never use SimpleSolver even for small
            // latch counts — the constraint propagation overhead is too high.
            let use_simple = num_latches <= 60 && !is_high_constraint_ratio;

            if use_simple {
                // Self-inductiveness debug info (SimpleSolver path only).
                for (i, lemma) in all_lemmas.iter().enumerate() {
                    let mut base_solver = self.make_validation_solver();
                    base_solver.add_clause(&[Lit::TRUE]);
                    for clause in &self.ts.trans_clauses {
                        base_solver.add_clause(&clause.lits);
                    }
                    for &constraint in &self.ts.constraint_lits {
                        base_solver.add_clause(&[constraint]);
                    }
                    for clause in &self.next_link_clauses {
                        base_solver.add_clause(clause);
                    }
                    base_solver.add_clause(lemma);

                    let neg_primed: Vec<Lit> = lemma
                        .iter()
                        .map(|l| {
                            let var = l.var();
                            if let Some(&next_var) = self.next_vars.get(&var) {
                                if l.is_positive() {
                                    Lit::neg(next_var)
                                } else {
                                    Lit::pos(next_var)
                                }
                            } else {
                                !*l
                            }
                        })
                        .collect();

                    if base_solver.solve(&neg_primed) == SatResult::Sat {
                        eprintln!(
                            "IC3 validate: lemma {} NOT self-inductive (base only): {:?}",
                            i, lemma
                        );
                    }
                }
            }

            // Build the validation solver: SimpleSolver for small low-ratio,
            // AYNoPreprocess for medium or high-ratio circuits.
            let mut trans_solver = if use_simple {
                self.make_validation_solver()
            } else {
                self.make_fast_validation_solver()
            };
            trans_solver.add_clause(&[Lit::TRUE]);
            for clause in &self.ts.trans_clauses {
                trans_solver.add_clause(&clause.lits);
            }
            for &constraint in &self.ts.constraint_lits {
                trans_solver.add_clause(&[constraint]);
            }
            for clause in &self.next_link_clauses {
                trans_solver.add_clause(clause);
            }
            for lemma in &all_lemmas {
                trans_solver.add_clause(lemma);
            }

            for (i, lemma) in all_lemmas.iter().enumerate() {
                if should_abort(&validate_start) {
                    eprintln!(
                        "IC3 validate: BUDGET EXCEEDED at Inv=>Inv' check lemma {}/{} ({:.1}s)",
                        i,
                        all_lemmas.len(),
                        validate_start.elapsed().as_secs_f64(),
                    );
                    return None;
                }
                let neg_primed: Vec<Lit> = lemma
                    .iter()
                    .map(|l| {
                        let var = l.var();
                        if let Some(&next_var) = self.next_vars.get(&var) {
                            if l.is_positive() {
                                Lit::neg(next_var)
                            } else {
                                Lit::pos(next_var)
                            }
                        } else {
                            !*l
                        }
                    })
                    .collect();

                if trans_solver.solve(&neg_primed) == SatResult::Sat {
                    eprintln!(
                        "IC3 VALIDATE FAIL: Inv AND T does NOT preserve lemma {}: {:?}",
                        i, lemma
                    );
                    for &latch in &self.ts.latch_vars {
                        let val = trans_solver.value(Lit::pos(latch));
                        if let Some(v) = val {
                            eprint!("v{}={} ", latch.0, if v { "T" } else { "F" });
                        }
                    }
                    eprintln!();
                    for &latch in &self.ts.latch_vars {
                        if let Some(&next_var) = self.next_vars.get(&latch) {
                            let val = trans_solver.value(Lit::pos(next_var));
                            if let Some(v) = val {
                                eprint!("v{}'={} ", latch.0, if v { "T" } else { "F" });
                            }
                        }
                    }
                    eprintln!();
                    return Some(false);
                }
            }
            let solver_kind = if use_simple {
                "SimpleSolver"
            } else {
                "AYNoPreprocess"
            };
            eprintln!(
                "IC3 validate: Inv AND T => Inv' OK ({:.3}s, {})",
                validate_start.elapsed().as_secs_f64(),
                solver_kind,
            );
        } else {
            eprintln!(
                "IC3 validate: SKIPPING Inv=>Inv' check (validation_strategy=SkipConsecution, latches={}, constraint_ratio={:.1})",
                num_latches, constraint_ratio,
            );
        }

        // Check 3: Inv => !Bad
        {
            let mut bad_solver = self.make_fast_validation_solver();
            bad_solver.add_clause(&[Lit::TRUE]);
            for clause in &self.ts.trans_clauses {
                bad_solver.add_clause(&clause.lits);
            }
            for &constraint in &self.ts.constraint_lits {
                bad_solver.add_clause(&[constraint]);
            }
            for lemma in &all_lemmas {
                bad_solver.add_clause(lemma);
            }

            for &bad_lit in &self.ts.bad_lits {
                if should_abort(&validate_start) {
                    eprintln!(
                        "IC3 validate: BUDGET EXCEEDED at Inv=>!Bad check ({:.1}s)",
                        validate_start.elapsed().as_secs_f64(),
                    );
                    return None;
                }
                if bad_solver.solve(&[bad_lit]) == SatResult::Sat {
                    eprintln!(
                        "IC3 VALIDATE FAIL: Inv allows bad state! bad_lit={:?}",
                        bad_lit
                    );
                    for &latch in &self.ts.latch_vars {
                        let val = bad_solver.value(Lit::pos(latch));
                        if let Some(v) = val {
                            eprint!("v{}={} ", latch.0, if v { "T" } else { "F" });
                        }
                    }
                    eprintln!();
                    return Some(false);
                }
            }
            eprintln!(
                "IC3 validate: Inv => !Bad OK ({:.3}s)",
                validate_start.elapsed().as_secs_f64()
            );
        }

        let validate_elapsed = validate_start.elapsed();
        eprintln!(
            "IC3 validate: ALL CHECKS PASSED -- invariant is valid ({:.3}s, {} lemmas)",
            validate_elapsed.as_secs_f64(),
            all_lemmas.len(),
        );
        Some(true)
    }

    /// Wrapper for backward compatibility — returns bool.
    #[allow(dead_code)]
    pub(super) fn validate_invariant(&self) -> bool {
        self.validate_invariant_budgeted().unwrap_or(false)
    }

    /// Independent consecution verification using a fresh SimpleSolver.
    pub(super) fn verify_consecution_independent(
        &self,
        frame: usize,
        cube: &[Lit],
        strengthen: bool,
    ) -> bool {
        let mut solver = SimpleSolver::new();
        // Collect every clause as we add it, so a SAT model can be independently
        // VERIFIED below — SimpleSolver is known to false-SAT on clause-dense
        // formulas (see config.rs), which would otherwise reject GOOD ay-sat
        // lemmas and stall IC3.
        let mut clauses: Vec<Vec<Lit>> = Vec::new();
        let mut add = |solver: &mut SimpleSolver, lits: &[Lit]| {
            solver.add_clause(lits);
            clauses.push(lits.to_vec());
        };
        add(&mut solver, &[Lit::TRUE]);
        for clause in &self.ts.trans_clauses {
            add(&mut solver, &clause.lits);
        }
        for &constraint in &self.ts.constraint_lits {
            add(&mut solver, &[constraint]);
        }
        for clause in &self.next_link_clauses {
            add(&mut solver, clause);
        }
        if frame >= 2 {
            let upper = (frame - 1).min(self.frames.depth().saturating_sub(1));
            for f in 1..=upper {
                if f < self.frames.frames.len() {
                    for lemma in &self.frames.frames[f].lemmas {
                        add(&mut solver, &lemma.lits);
                    }
                }
            }
        }
        for lemma in &self.inf_lemmas {
            add(&mut solver, &lemma.lits);
        }
        if strengthen {
            let neg_cube: Vec<Lit> = cube.iter().map(|l| !*l).collect();
            add(&mut solver, &neg_cube);
        }
        let assumptions = self.prime_cube(cube);
        match solver.solve(&assumptions) {
            // UNSAT agrees with ay-sat ⇒ the lemma is inductive (passes).
            SatResult::Unsat => true,
            // SAT contradicts ay-sat's UNSAT. Trust it (reject the lemma) ONLY
            // if the returned model actually satisfies every clause + assumption
            // — that is a real counterexample, i.e. a genuine ay-sat false-UNSAT
            // catch. A model that fails to check is a SimpleSolver false-SAT: it
            // is discarded and ay-sat's UNSAT is trusted, so a correct lemma is
            // no longer spuriously rejected (the IC3-stall bug in config.rs).
            SatResult::Sat => !model_satisfies(&solver, &clauses, &assumptions),
            // Unknown: no counterexample proven ⇒ conservative (unchanged).
            SatResult::Unknown => false,
        }
    }

    /// Verify that a single generalized lemma satisfies the consecution property
    /// at a given frame, using a fresh solver independent of the IC3 frame solvers.
    ///
    /// Checks: F_{frame-1} AND Inv_inf AND T AND constraints AND next_link AND
    ///         !(cube) AND cube' is UNSAT.
    ///
    /// This is the per-lemma equivalent of Check 2 in `validate_invariant_budgeted`,
    /// but applied immediately before a lemma is added to the frame sequence. It
    /// catches ay-sat false UNSAT in the consecution query before the unsound lemma
    /// can propagate to higher frames.
    ///
    /// ## Faithful reconstruction of F_{frame-1}
    ///
    /// A refutation is only meaningful if it refutes the formula the primary
    /// solver actually decided. The primary consecution query for an obligation
    /// at `frame` runs on `solvers[frame - 1]`, and the frame system is
    /// delta-encoded: a lemma recorded at `frames.frames[t]` was added to
    /// solvers `1..=t`, so solver `j >= 1` holds exactly the lemmas with index
    /// `>= j`, while solver 0 holds the init clauses plus the index-0 lemmas.
    /// Therefore:
    ///
    /// - `frame == 1`: F_0 is `Init` (init clauses + frame-0 lemmas). Omitting
    ///   the init clauses — as this function once did — weakens the formula so
    ///   far that any unreachable ¬cube-state with a successor in the cube
    ///   "refutes" a perfectly valid lemma. On init-constrained circuits that
    ///   rejected every early frame-1 lemma and livelocked blocking (#4560).
    /// - `frame >= 2`: F_{frame-1} is the lemmas with delta index
    ///   `>= frame - 1`. Collecting indexes `1..=frame` instead (the old code)
    ///   was wrong in both directions: low-index lemmas that only hold at
    ///   earlier frames over-constrain the check (masking genuine false
    ///   UNSATs), and the missing high-index lemmas weaken it (manufacturing
    ///   refutations of valid lemmas).
    ///
    /// ## Verdict semantics
    ///
    /// [`LemmaVerdict::Refuted`] is returned only with a model that was
    /// re-checked against every clause of the verification formula
    /// (`model_satisfies`), and carries the model's full latch state as the
    /// counterexample-to-consecution witness: a state in F_{frame-1} ∧ ¬cube
    /// with a successor in the cube. `Unknown` and unverifiable SAT models
    /// (SimpleSolver false-SATs on clause-dense formulas — see config.rs)
    /// yield `Verified`: without a checkable witness there is nothing sound to
    /// act on, and the post-convergence `validate_invariant_budgeted` net
    /// still guards the final answer.
    ///
    /// Uses AYNoPreprocess for circuits with > 60 latches (SimpleSolver is too slow)
    /// and SimpleSolver for small circuits (maximum independence from ay-sat bugs).
    pub(super) fn verify_lemma_consecution(&self, frame: usize, cube: &[Lit]) -> LemmaVerdict {
        // Build a fresh solver with the transition relation.
        let use_simple = self.ts.latch_vars.len() <= 60
            && !super::config::is_high_constraint_circuit(
                self.ts.trans_clauses.len(),
                self.ts.constraint_lits.len(),
                self.ts.latch_vars.len(),
            );
        let mut solver = if use_simple {
            self.make_validation_solver()
        } else {
            self.make_fast_validation_solver()
        };

        // Collect every clause so a SAT model can be VERIFIED below (the
        // SimpleSolver path false-SATs on clause-dense formulas — see config.rs).
        let mut clauses: Vec<Vec<Lit>> = vec![vec![Lit::TRUE]];
        // Transition relation.
        for clause in &self.ts.trans_clauses {
            clauses.push(clause.lits.clone());
        }
        // Constraints.
        for &constraint in &self.ts.constraint_lits {
            clauses.push(vec![constraint]);
        }
        // Next-state linking clauses.
        for clause in &self.next_link_clauses {
            clauses.push(clause.clone());
        }
        // F_{frame-1}, faithfully reconstructed per the delta encoding
        // (see the doc comment above).
        if frame <= 1 {
            // F_0 = Init plus the frame-0-only lemmas held by solver 0.
            for clause in &self.ts.init_clauses {
                clauses.push(clause.lits.clone());
            }
            if let Some(frame0) = self.frames.frames.first() {
                for lemma in &frame0.lemmas {
                    clauses.push(lemma.lits.clone());
                }
            }
        } else {
            for f in (frame - 1)..self.frames.frames.len() {
                for lemma in &self.frames.frames[f].lemmas {
                    clauses.push(lemma.lits.clone());
                }
            }
        }
        // Infinity lemmas.
        for lemma in &self.inf_lemmas {
            clauses.push(lemma.lits.clone());
        }
        // Strengthening: !(cube), so the current state does NOT satisfy the cube
        // (the standard IC3 consecution check F_k AND T AND !(cube) AND cube').
        clauses.push(cube.iter().map(|l| !*l).collect());

        for clause in &clauses {
            solver.add_clause(clause);
        }

        // Check cube' (primed cube) as assumptions. On SAT, only refute the
        // lemma if the model actually checks out (a real ay-sat false-UNSAT
        // catch); discard an unverifiable SAT so a correct lemma passes.
        let assumptions = self.prime_cube(cube);
        match solver.solve(&assumptions) {
            SatResult::Unsat => LemmaVerdict::Verified,
            SatResult::Sat => {
                if model_satisfies(&*solver, &clauses, &assumptions) {
                    let predecessor =
                        Self::extract_state_from_solver(&*solver, &self.ts.latch_vars);
                    if predecessor.is_empty() {
                        // A validated model must assign the latches (they occur
                        // in the next-link clauses it was checked against); an
                        // empty extraction cannot become a proof obligation —
                        // the empty cube denotes ALL states. Nothing sound to
                        // descend on, so fall back to trusting the primary.
                        debug_assert!(
                            self.ts.latch_vars.is_empty(),
                            "validated consecution refutation assigned no latches"
                        );
                        LemmaVerdict::Verified
                    } else {
                        LemmaVerdict::Refuted { predecessor }
                    }
                } else {
                    LemmaVerdict::Verified
                }
            }
            SatResult::Unknown => LemmaVerdict::Verified,
        }
    }

    /// Perform consecution using SimpleSolver as a fallback when ay-sat produces
    /// false UNSAT (#4105).
    pub(super) fn consecution_simple_fallback(
        &self,
        frame: usize,
        cube: &[Lit],
    ) -> Option<Vec<Lit>> {
        let mut solver = SimpleSolver::new();
        solver.add_clause(&[Lit::TRUE]);
        for clause in &self.ts.trans_clauses {
            solver.add_clause(&clause.lits);
        }
        for &constraint in &self.ts.constraint_lits {
            solver.add_clause(&[constraint]);
        }
        for clause in &self.next_link_clauses {
            solver.add_clause(clause);
        }
        if frame >= 2 {
            let upper = (frame - 1).min(self.frames.depth().saturating_sub(1));
            for f in 1..=upper {
                if f < self.frames.frames.len() {
                    for lemma in &self.frames.frames[f].lemmas {
                        solver.add_clause(&lemma.lits);
                    }
                }
            }
        }
        for lemma in &self.inf_lemmas {
            solver.add_clause(&lemma.lits);
        }
        let assumptions = self.prime_cube(cube);
        if solver.solve(&assumptions) == SatResult::Sat {
            Some(Self::extract_state_from_solver(
                &solver,
                &self.ts.latch_vars,
            ))
        } else {
            None
        }
    }
}

/// Independently verify a SAT model: `true` iff every literal in every `clause`
/// and every `assumption` is satisfied by the solver's current assignment.
///
/// This is the discriminator that makes the independent consecution cross-checks
/// RELIABLE. SimpleSolver's basic DPLL is known to report false SAT on
/// clause-dense formulas (config.rs); a SAT verdict is only a genuine ay-sat
/// false-UNSAT catch if its model actually checks out. An unverifiable "SAT" is
/// a solver bug and must be discarded so a correct ay-sat lemma is not rejected.
/// A returned model is always complete over the relevant variables, so a `None`
/// (unassigned) literal counts as not-satisfying (fail closed toward "discard").
fn model_satisfies(solver: &dyn SatSolver, clauses: &[Vec<Lit>], assumptions: &[Lit]) -> bool {
    for &a in assumptions {
        if solver.value(a) != Some(true) {
            return false;
        }
    }
    for clause in clauses {
        if !clause.iter().any(|&l| solver.value(l) == Some(true)) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod model_verify_tests {
    use super::*;
    use crate::sat_types::Var;

    #[test]
    fn model_satisfies_accepts_valid_and_rejects_violations() {
        // (x1) /\ (~x1 \/ x2) forces the unique model x1=T, x2=T.
        let mut s = SimpleSolver::new();
        s.ensure_vars(3);
        let x1 = Lit::pos(Var(1));
        let x2 = Lit::pos(Var(2));
        let c1 = vec![x1];
        let c2 = vec![!x1, x2];
        s.add_clause(&c1);
        s.add_clause(&c2);
        assert_eq!(s.solve(&[]), SatResult::Sat);

        // The real model satisfies both clauses ⇒ a SAT verdict to TRUST.
        assert!(model_satisfies(&s, &[c1.clone(), c2.clone()], &[]));
        // A clause the model falsifies ⇒ NOT satisfied (a bogus-SAT would fail here).
        assert!(!model_satisfies(&s, &[vec![!x1]], &[]));
        // An assumption the model violates ⇒ NOT satisfied.
        assert!(!model_satisfies(&s, &[], &[!x1]));
    }
}
