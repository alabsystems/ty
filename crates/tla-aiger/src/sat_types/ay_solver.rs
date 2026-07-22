// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AY-sat CDCL solver wrapper for IC3/BMC engines.
//!
//! Wraps the ay-sat `Solver` with panic resilience, UNSAT core extraction,
//! push/pop scope management, and IC3-specific optimizations (domain restriction,
//! flip-to-none state lifting, conflict-budgeted solving).

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::{Ic3BucketQueueVsidsStats, Lit, SatResult, SatSolver, Var};

const AY_BUCKET_QUEUE_HEAP_FALLBACK_RESTARTS: u64 = 10;

/// ay-sat CDCL solver with panic resilience and IC3-optimized solving.
///
/// ay-sat may panic on certain clause structures (see shrink.rs overflow,
/// conflict_analysis.rs assertion). Rather than crashing the entire
/// portfolio thread, this wrapper catches panics via `catch_unwind` and
/// degrades gracefully to `SatResult::Unknown`. Once a panic is caught,
/// the solver is marked `poisoned` and all subsequent calls return
/// `Unknown` immediately — the internal ay-sat state is unreliable
/// after a panic.
pub struct AYSatCdclSolver {
    pub(crate) solver: ay_sat::Solver,
    pub(crate) num_vars: u32,
    model: Vec<Option<bool>>,
    last_core: Option<Vec<Lit>>,
    last_decision: Option<AYSolveDecision>,
    /// Set to `true` after catching a ay-sat panic. All subsequent calls
    /// return `SatResult::Unknown`. The solver cannot be recovered after
    /// a panic because ay-sat's internal invariants may be violated.
    pub(crate) poisoned: bool,
    /// Log of all permanent clauses for clone_solver() replay (#4062).
    clause_log: Vec<Vec<Lit>>,
    /// Adapter-level observations for ay-sat bucket-queue VSIDS on IC3 domain
    /// queries. ay-sat keeps the actual bucket flag private, so TLA records the
    /// public boundary: `set_domain()` under IC3 mode and per-query restarts.
    ic3_bucket_queue_stats: Ic3BucketQueueVsidsStats,
    active_ic3_domain_restart_base: Option<u64>,
}

/// AY-specific solve decision at the SAT adapter boundary.
///
/// This intentionally does not replace [`SatResult`]. The AIGER engines still
/// consume the coarse SAT API, while capability/evidence callers can inspect
/// why the adapter collapsed a non-definitive AY outcome to `Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "AY solve decisions carry solver boundary evidence"]
pub enum AYSolveDecision {
    Sat,
    Unsat,
    Unavailable(AYUnavailableReason),
    Unknown(AYUnknownReason),
    SolverError(AYSolverErrorReason),
    Deadline(AYDeadlineReason),
}

impl AYSolveDecision {
    /// Coarse result consumed by existing SAT engines.
    pub fn as_sat_result(self) -> SatResult {
        match self {
            Self::Sat => SatResult::Sat,
            Self::Unsat => SatResult::Unsat,
            Self::Unavailable(_) | Self::Unknown(_) | Self::SolverError(_) | Self::Deadline(_) => {
                SatResult::Unknown
            }
        }
    }

    /// Stable high-level category for capability evidence.
    pub fn kind_code(self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Unsat => "unsat",
            Self::Unavailable(_) => "unavailable",
            Self::Unknown(_) => "unknown",
            Self::SolverError(_) => "solver_error",
            Self::Deadline(_) => "deadline",
        }
    }

    /// Stable reason code for non-definitive outcomes and successful verdicts.
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::Sat => "ay_sat",
            Self::Unsat => "ay_unsat",
            Self::Unavailable(reason) => reason.reason_code(),
            Self::Unknown(reason) => reason.reason_code(),
            Self::SolverError(reason) => reason.reason_code(),
            Self::Deadline(reason) => reason.reason_code(),
        }
    }

    /// Stable status token used in hardware backend evidence rows.
    pub fn evidence_status_name(self) -> &'static str {
        match self {
            Self::Sat | Self::Unsat => "Available",
            Self::Unavailable(_) => "Unavailable",
            Self::Unknown(_) => "Unknown",
            Self::SolverError(_) => "SolverError",
            Self::Deadline(_) => "Deadline",
        }
    }

    /// Stable coarse SAT result token preserved for existing SAT consumers.
    pub fn evidence_sat_result_name(self) -> &'static str {
        match self.as_sat_result() {
            SatResult::Sat => "Sat",
            SatResult::Unsat => "Unsat",
            SatResult::Unknown => "Unknown",
        }
    }

    fn from_ay_unknown_reason(reason: Option<ay_sat::SatUnknownReason>) -> Self {
        match reason.unwrap_or(ay_sat::SatUnknownReason::Unspecified) {
            ay_sat::SatUnknownReason::Interrupted => Self::Deadline(AYDeadlineReason::Interrupted),
            ay_sat::SatUnknownReason::UnsupportedConfig => {
                Self::Unavailable(AYUnavailableReason::UnsupportedConfig)
            }
            ay_sat::SatUnknownReason::InvalidSatModel => {
                Self::SolverError(AYSolverErrorReason::InvalidSatModel)
            }
            ay_sat::SatUnknownReason::ProofFinalizationFailure => {
                Self::SolverError(AYSolverErrorReason::ProofFinalizationFailure)
            }
            ay_sat::SatUnknownReason::EmptyTheoryConflict => {
                Self::SolverError(AYSolverErrorReason::EmptyTheoryConflict)
            }
            ay_sat::SatUnknownReason::TheoryStop => Self::Unknown(AYUnknownReason::TheoryStop),
            ay_sat::SatUnknownReason::ExtensionUnknown => {
                Self::Unknown(AYUnknownReason::ExtensionUnknown)
            }
            ay_sat::SatUnknownReason::AssumptionUnknown => {
                Self::Unknown(AYUnknownReason::AssumptionUnknown)
            }
            ay_sat::SatUnknownReason::Unspecified => Self::Unknown(AYUnknownReason::Unspecified),
            _ => Self::Unknown(AYUnknownReason::Unspecified),
        }
    }
}

/// Why the AY adapter could not make a AY solve attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AYUnavailableReason {
    Poisoned,
    UnsupportedConfig,
}

impl AYUnavailableReason {
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::Poisoned => "ay_solver_poisoned",
            Self::UnsupportedConfig => "ay_unsupported_config",
        }
    }
}

/// Non-deadline, non-error AY `Unknown` reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AYUnknownReason {
    TheoryStop,
    ExtensionUnknown,
    AssumptionUnknown,
    Unspecified,
}

impl AYUnknownReason {
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::TheoryStop => "ay_theory_stop",
            Self::ExtensionUnknown => "ay_extension_unknown",
            Self::AssumptionUnknown => "ay_assumption_unknown",
            Self::Unspecified => "ay_unknown",
        }
    }
}

/// AY outcomes that indicate the adapter cannot trust the solver state/result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AYSolverErrorReason {
    Panic,
    InvalidSatModel,
    ProofFinalizationFailure,
    EmptyTheoryConflict,
}

impl AYSolverErrorReason {
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::Panic => "ay_solver_panic",
            Self::InvalidSatModel => "ay_invalid_sat_model",
            Self::ProofFinalizationFailure => "ay_proof_finalization_failure",
            Self::EmptyTheoryConflict => "ay_empty_theory_conflict",
        }
    }
}

/// AY work stopped because an external or adapter-owned deadline fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AYDeadlineReason {
    Interrupted,
    ConflictBudgetZero,
    ConflictBudgetExhausted,
}

impl AYDeadlineReason {
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::Interrupted => "ay_interrupted",
            Self::ConflictBudgetZero => "ay_conflict_budget_zero",
            Self::ConflictBudgetExhausted => "ay_conflict_budget_exhausted",
        }
    }
}

impl AYSatCdclSolver {
    /// Disable BVE and other preprocessing (#4074).
    ///
    /// Used as a fallback when ay-sat produces FINALIZE_SAT_FAIL
    /// (InvalidSatModel) on certain clause structures. Must be called
    /// before any clauses are added.
    ///
    /// NOTE: In the IC3 path, `solve_incremental_ic3()` never calls
    /// `preprocess()`, so BVE does not actually execute for IC3 SAT
    /// queries. This fallback is relevant for non-IC3 paths (e.g.,
    /// `solve_with_assumptions`) where preprocessing does run.
    pub fn disable_preprocessing(&mut self) {
        self.solver.set_preprocess_enabled(false);
    }

    /// Disable all periodic inprocessing in the underlying ay-sat solver (#4102).
    ///
    /// Calls `disable_all_inprocessing()` on the ay-sat `Solver`, which turns off
    /// all 16 inprocessing technique toggles: vivification, subsumption, probing,
    /// BVE, BCE, conditioning, decomposition, factorization, transitive reduction,
    /// HTR, gate extraction, congruence closure, sweep, CCE, and backbone detection.
    ///
    /// This is distinct from `disable_preprocessing()`: preprocessing runs once at
    /// the start (and is kept enabled for initial simplification), while inprocessing
    /// runs periodically between conflicts and is harmful for IC3's short incremental
    /// queries.
    pub fn disable_inprocessing(&mut self) {
        self.solver.disable_all_inprocessing();
    }

    /// Create an ay-sat CDCL solver sized for `num_vars` variables, with full
    /// initial preprocessing enabled.
    pub fn new(num_vars: u32) -> Self {
        let mut solver = ay_sat::Solver::new(num_vars as usize);
        // Enable full preprocessing (ay-sat defaults to quick mode which skips
        // heavier passes like BVE). IC3 frame solvers live for the entire proof
        // and benefit from thorough initial simplification. This activates the
        // same pass set that CaDiCaL runs by default.
        //
        // NOTE: In practice, IC3 uses solve_incremental_ic3() which never calls
        // preprocess(), so this setting only takes effect if the solver falls
        // back to solve_with_assumptions() (which calls preprocess() on first
        // solve with freeze/melt around assumption variables).
        solver.set_full_preprocessing(true);
        // NOTE: Periodic inprocessing is left ENABLED by default (#4102).
        //
        // BMC and k-induction make longer SAT calls that can benefit from
        // periodic inprocessing (vivification, subsumption, probing, etc.).
        // Only IC3 frame solvers need inprocessing disabled because IC3 makes
        // thousands of short incremental queries where inprocessing overhead
        // is harmful. IC3 solvers use `make_solver_no_inprocessing()` instead
        // of the default `make_solver()` to achieve this, or configure the
        // full IC3 query shape via `SolverBackend::make_solver_ic3_mode()` /
        // `set_ic3_mode()` on the solver after creation.
        AYSatCdclSolver {
            solver,
            num_vars,
            model: vec![None; num_vars as usize],
            last_core: None,
            last_decision: None,
            poisoned: false,
            clause_log: Vec::new(),
            ic3_bucket_queue_stats: Ic3BucketQueueVsidsStats::default(),
            active_ic3_domain_restart_base: None,
        }
    }

    /// Returns true if the solver has been poisoned by a prior ay-sat panic.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Typed decision from the most recent AY adapter solve call.
    pub fn last_decision(&self) -> Option<AYSolveDecision> {
        self.last_decision
    }

    /// Stable category code for the most recent AY adapter decision.
    pub fn last_decision_kind_code(&self) -> Option<&'static str> {
        self.last_decision.map(AYSolveDecision::kind_code)
    }

    /// Stable reason code for the most recent AY adapter decision.
    pub fn last_decision_reason_code(&self) -> Option<&'static str> {
        self.last_decision.map(AYSolveDecision::reason_code)
    }

    /// Solve under `assumptions` via the IC3-optimized incremental path,
    /// returning a typed decision. A poisoned solver returns `Unavailable`;
    /// a caught ay-sat panic poisons the solver and returns an error decision.
    pub fn solve_decision(&mut self, assumptions: &[Lit]) -> AYSolveDecision {
        if self.poisoned {
            return self
                .record_decision(AYSolveDecision::Unavailable(AYUnavailableReason::Poisoned));
        }
        for lit in assumptions {
            self.ensure_vars(lit.var().0);
        }
        let ay_assumptions: Vec<ay_sat::Literal> =
            assumptions.iter().map(|&l| Self::to_ay_lit(l)).collect();

        // Use IC3-optimized solve path: a stripped-down CDCL loop.
        // Skips inprocessing scheduling, theory callbacks,
        // proof logging, TLA tracing, progress reporting, Glucose EMA restarts,
        // lucky phases, walk init, observer notifications — all overhead that
        // IC3's thousands of short queries don't need. Falls back to standard
        // solve_with_assumptions if the IC3 path is unavailable.
        //
        // Wrap in catch_unwind to handle panics from shrink.rs overflow and
        // conflict_analysis.rs BUG (#4026).
        // SAFETY rationale for AssertUnwindSafe: after a panic we mark
        // the solver as poisoned and never call into ay-sat again, so
        // the potentially-inconsistent internal state is never observed.
        let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.solver.solve_incremental_ic3(&ay_assumptions)
        }));

        let result = match solve_result {
            Ok(r) => r,
            Err(panic_info) => return self.poison_from_panic("solve()", panic_info),
        };

        self.record_ic3_domain_query();
        let decision = self.decision_from_ay_result(&result, None, |_| true);
        self.record_decision(decision)
    }

    /// Solve under `assumptions` with a conflict budget, returning a typed
    /// decision. A zero budget yields a `Deadline` decision immediately, and the
    /// solver returns `Deadline` if the budget is exhausted before deciding.
    pub fn solve_with_budget_decision(
        &mut self,
        assumptions: &[Lit],
        max_conflicts: u64,
    ) -> AYSolveDecision {
        if max_conflicts == 0 {
            return self.record_decision(AYSolveDecision::Deadline(
                AYDeadlineReason::ConflictBudgetZero,
            ));
        }
        if self.poisoned {
            return self
                .record_decision(AYSolveDecision::Unavailable(AYUnavailableReason::Poisoned));
        }
        for lit in assumptions {
            self.ensure_vars(lit.var().0);
        }
        let ay_assumptions: Vec<ay_sat::Literal> =
            assumptions.iter().map(|&l| Self::to_ay_lit(l)).collect();

        // ay-sat checks should_stop every 100 conflicts and every 1000 decisions.
        // Compute the maximum number of callback invocations to allow.
        // For max_conflicts < 100, we allow 1 invocation (up to ~100 conflicts).
        let max_invocations = max_conflicts.div_ceil(100);
        let invocation_count = std::sync::atomic::AtomicU64::new(0);

        // Wrap in catch_unwind for ay-sat panic resilience (#4026).
        let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.solver
                .solve_with_assumptions_interruptible(&ay_assumptions, || {
                    let count = invocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    count >= max_invocations
                })
        }));

        let result = match solve_result {
            Ok(r) => r,
            Err(panic_info) => return self.poison_from_panic("solve_with_budget()", panic_info),
        };

        self.record_ic3_domain_query();
        let deadline = (invocation_count.load(std::sync::atomic::Ordering::Relaxed)
            > max_invocations)
            .then_some(AYDeadlineReason::ConflictBudgetExhausted);
        let decision = self.decision_from_ay_result(&result, deadline, |_| true);
        self.record_decision(decision)
    }

    /// Solve under `assumptions` with `temp_clause` added only for this call,
    /// returning a typed decision. The temporary clause does not persist into
    /// later solves.
    pub fn solve_with_temporary_clause_decision(
        &mut self,
        assumptions: &[Lit],
        temp_clause: &[Lit],
    ) -> AYSolveDecision {
        if self.poisoned {
            return self
                .record_decision(AYSolveDecision::Unavailable(AYUnavailableReason::Poisoned));
        }
        if temp_clause.is_empty() {
            return self.solve_decision(assumptions);
        }
        for lit in temp_clause {
            self.ensure_vars(lit.var().0);
        }
        for lit in assumptions {
            self.ensure_vars(lit.var().0);
        }

        // Save the user-facing variable count before push(). ay-sat's push()
        // creates an internal scope-selector variable that increments its
        // internal variable count. We use this bound to filter the UNSAT core:
        // any literal with var index >= num_vars_before_push is an internal
        // ay-sat variable (scope selector) and must not leak into the core
        // returned to IC3 (#4024).
        let num_vars_before_push = self.num_vars;

        // Push a new scope — clauses added within this scope are automatically
        // tagged with a scope selector and removed on pop().
        self.solver.push();

        // Add the temporary clause within the pushed scope.
        // ay-sat's add_clause() attaches a scope selector when inside a push scope.
        let ay_temp: Vec<ay_sat::Literal> =
            temp_clause.iter().map(|&l| Self::to_ay_lit(l)).collect();
        self.solver.add_clause(ay_temp);

        // Solve with the original assumptions using IC3-optimized path.
        // solve_incremental_ic3 handles scope selectors via compose_scope_assumptions.
        // Wrap in catch_unwind to handle ay-sat panics (#4026).
        let ay_assumptions: Vec<ay_sat::Literal> =
            assumptions.iter().map(|&l| Self::to_ay_lit(l)).collect();
        let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.solver.solve_incremental_ic3(&ay_assumptions)
        }));

        let result = match solve_result {
            Ok(r) => r,
            Err(panic_info) => {
                // Best-effort pop to clean up the pushed scope. If this also
                // panics, the solver is already poisoned so it doesn't matter.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = self.solver.pop();
                }));
                return self.poison_from_panic("solve_with_temporary_clause()", panic_info);
            }
        };

        let decision = self.decision_from_ay_result(&result, None, |l| {
            let var_idx = Self::from_ay_lit(l).var().0;
            var_idx < num_vars_before_push
        });

        self.record_ic3_domain_query();
        // Pop the scope — removes the temporary clause from the solver.
        let _ = self.solver.pop();

        self.record_decision(decision)
    }

    #[inline]
    fn to_ay_lit(lit: Lit) -> ay_sat::Literal {
        // Both use var*2+sign encoding.
        let var = ay_sat::Variable::new(lit.var().0);
        if lit.is_positive() {
            ay_sat::Literal::positive(var)
        } else {
            ay_sat::Literal::negative(var)
        }
    }

    #[inline]
    fn from_ay_lit(lit: ay_sat::Literal) -> Lit {
        let var = Var(lit.variable().id());
        if lit.is_positive() {
            Lit::pos(var)
        } else {
            Lit::neg(var)
        }
    }

    fn record_decision(&mut self, decision: AYSolveDecision) -> AYSolveDecision {
        self.last_decision = Some(decision);
        decision
    }

    fn record_ic3_domain_query(&mut self) {
        let Some(base_restarts) = self.active_ic3_domain_restart_base else {
            return;
        };
        let query_restarts = self.solver.num_restarts().saturating_sub(base_restarts);
        self.ic3_bucket_queue_stats.domain_queries += 1;
        self.ic3_bucket_queue_stats.max_query_restarts = self
            .ic3_bucket_queue_stats
            .max_query_restarts
            .max(query_restarts);
        if query_restarts > AY_BUCKET_QUEUE_HEAP_FALLBACK_RESTARTS {
            self.ic3_bucket_queue_stats.heap_fallback_queries += 1;
        }
    }

    fn poison_from_panic(
        &mut self,
        context: &str,
        panic_info: Box<dyn std::any::Any + Send>,
    ) -> AYSolveDecision {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        eprintln!("IC3: ay-sat panic caught in {context}: {msg}");
        self.poisoned = true;
        self.record_decision(AYSolveDecision::SolverError(AYSolverErrorReason::Panic))
    }

    fn decision_from_ay_result<F>(
        &mut self,
        result: &ay_sat::VerifiedAssumeResult,
        deadline: Option<AYDeadlineReason>,
        core_filter: F,
    ) -> AYSolveDecision
    where
        F: Fn(ay_sat::Literal) -> bool,
    {
        if result.is_sat() {
            // Extract model
            if let Some(model_vals) = result.model() {
                self.model
                    .resize(model_vals.len().max(self.num_vars as usize), None);
                for (i, &val) in model_vals.iter().enumerate() {
                    if i < self.model.len() {
                        self.model[i] = Some(val);
                    }
                }
            }
            self.last_core = None;
            AYSolveDecision::Sat
        } else if result.is_unsat() {
            self.last_core = result.unsat_core().map(|core| {
                core.iter()
                    .copied()
                    .filter(|&l| core_filter(l))
                    .map(Self::from_ay_lit)
                    .collect()
            });
            AYSolveDecision::Unsat
        } else {
            self.last_core = None;
            if let Some(reason) = deadline {
                AYSolveDecision::Deadline(reason)
            } else {
                AYSolveDecision::from_ay_unknown_reason(self.solver.last_unknown_reason())
            }
        }
    }
}

impl SatSolver for AYSatCdclSolver {
    fn ensure_vars(&mut self, n: u32) {
        while self.num_vars <= n {
            self.solver.new_var();
            self.num_vars += 1;
        }
        self.model.resize(self.num_vars as usize, None);
    }

    fn add_clause(&mut self, clause: &[Lit]) {
        if self.poisoned {
            return;
        }
        for lit in clause {
            self.ensure_vars(lit.var().0);
        }
        let ay_clause: Vec<ay_sat::Literal> = clause.iter().map(|&l| Self::to_ay_lit(l)).collect();
        // Use add_clause_global to ensure permanent clauses survive push/pop scopes.
        // The default add_clause would attach a scope selector if called inside a
        // push() scope (e.g., if someone adds a lemma during solve_with_temporary_clause).
        self.solver.add_clause_global(ay_clause);
        self.clause_log.push(clause.to_vec());
    }

    fn solve(&mut self, assumptions: &[Lit]) -> SatResult {
        self.solve_decision(assumptions).as_sat_result()
    }

    fn value(&self, lit: Lit) -> Option<bool> {
        let var_idx = lit.var().index();
        if var_idx >= self.model.len() {
            return None;
        }
        self.model[var_idx].map(|v| if lit.is_negated() { !v } else { v })
    }

    fn new_var(&mut self) -> Var {
        let v = Var(self.num_vars);
        self.solver.new_var();
        self.num_vars += 1;
        self.model.push(None);
        v
    }

    fn unsat_core(&self) -> Option<Vec<Lit>> {
        self.last_core.clone()
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn disable_inprocessing(&mut self) {
        self.solver.disable_all_inprocessing();
    }

    /// Enable full IC3/PDR mode in the underlying ay-sat solver (#4306 Patch B,
    /// ay#8569).
    ///
    /// This is a strict superset of `disable_inprocessing()`: in addition to
    /// disabling all 16 inprocessing techniques, it also disables preprocessing,
    /// LRAT proof logging, chronological backtracking, cold restarts, rephase,
    /// and flip search; locks the branching heuristic to stable-mode VSIDS; and
    /// keeps the bucket queue permanently active for O(1) variable selection on
    /// short domain-restricted queries.
    ///
    /// The big win for cal14-style benchmarks is the per-query incremental
    /// reset: with ic3_mode on, `reset_search_state_incremental()` skips ~80
    /// cold scheduling state resets per solve call (EMA counters, tick
    /// watermarks, effort demotion, etc.) that the IC3 CDCL loop never reads.
    /// Per ay#8569, this saves ~5-10us per query — small per call but
    /// multiplicative across thousands of incremental queries.
    ///
    /// Must be called before `solve()` to take effect. Idempotent.
    fn set_ic3_mode(&mut self) {
        self.solver.set_ic3_mode();
    }

    fn is_ic3_mode(&self) -> bool {
        self.solver.is_ic3_mode()
    }

    fn set_luby_restarts(&mut self, unit: u64) {
        self.solver.set_glucose_restarts(false);
        self.solver.set_restart_base(unit.max(1));
    }

    fn set_geometric_restarts(&mut self, initial: f64, factor: f64) {
        self.solver.set_geometric_restarts(initial, factor);
    }

    /// Wire the portfolio's cancellation flag into ay-sat's interrupt mechanism.
    ///
    /// ay-sat's CDCL loop checks `is_interrupted()` every ~1000 decisions
    /// (solve.rs:868). When the flag is set, the solver returns Unknown with
    /// reason `Interrupted`, allowing the thread to exit promptly instead of
    /// running to completion (#4057).
    fn set_cancelled(&mut self, cancelled: Arc<AtomicBool>) {
        self.solver.set_interrupt(cancelled);
    }

    fn clone_solver(&self) -> Option<Box<dyn SatSolver>> {
        if self.poisoned {
            return None;
        }
        let mut new_solver = AYSatCdclSolver::new(self.num_vars);
        for clause in &self.clause_log {
            new_solver.add_clause(clause);
        }
        Some(Box::new(new_solver))
    }

    /// Native incremental clone using ay-sat's `clone_for_incremental()`.
    ///
    /// Deep-copies the entire solver state: clause arena, watch lists, VSIDS
    /// heap + activities, trail, variable assignments/phases, conflict analysis
    /// state. The cloned solver inherits all learned clauses and VSIDS scores,
    /// making it immediately effective without cold-start overhead.
    ///
    /// This replaces the clause-log replay in `clone_solver()` for frame
    /// extension (#4062). The key benefit: learned clauses from solving
    /// previous frames carry forward to new frames, reducing redundant work.
    ///
    /// ay-sat `solver/clone.rs:48` — `clone_for_incremental()`.
    fn clone_for_incremental(&self) -> Option<Box<dyn SatSolver>> {
        if self.poisoned {
            return None;
        }
        let cloned_solver = self.solver.clone_for_incremental();
        Some(Box::new(AYSatCdclSolver {
            solver: cloned_solver,
            num_vars: self.num_vars,
            model: self.model.clone(),
            last_core: None,
            last_decision: None,
            poisoned: false,
            clause_log: self.clause_log.clone(),
            ic3_bucket_queue_stats: self.ic3_bucket_queue_stats,
            active_ic3_domain_restart_base: None,
        }))
    }

    /// Wire ay-sat's native domain restriction for IC3 queries.
    ///
    /// Activates domain-restricted BCP (`propagate_domain_bcp`) and
    /// bucket-queue VSIDS for small domains. COI-restricted SAT
    /// (arXiv:2502.13605 §3): each IC3 SAT call only processes
    /// variables in the cube's cone-of-influence.
    ///
    /// Backed by ay-sat's `set_domain`.
    fn set_domain(&mut self, vars: &[Var]) {
        if self.poisoned {
            return;
        }
        let ay_vars: Vec<ay_sat::Variable> =
            vars.iter().map(|v| ay_sat::Variable::new(v.0)).collect();
        self.solver.set_domain(&ay_vars);
        if self.solver.is_ic3_mode() {
            self.ic3_bucket_queue_stats.domain_sets += 1;
            self.active_ic3_domain_restart_base = Some(self.solver.num_restarts());
        } else {
            self.active_ic3_domain_restart_base = None;
        }
    }

    fn clear_domain(&mut self) {
        if self.poisoned {
            return;
        }
        self.active_ic3_domain_restart_base = None;
        self.solver.clear_domain();
    }

    fn ic3_bucket_queue_vsids_stats(&self) -> Ic3BucketQueueVsidsStats {
        self.ic3_bucket_queue_stats
    }

    /// Wire ay-sat's `flip_to_none` as the model-unassignment primitive.
    ///
    /// After a SAT result, asks ay-sat whether the model stays a model with
    /// `var` unassigned. Returns true if the assignment was retracted.
    fn unassign_model_value(&mut self, var: Var) -> bool {
        if self.poisoned {
            return false;
        }
        let ay_var = ay_sat::Variable::new(var.0);
        // Wrap in catch_unwind for panic resilience (#4026).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.solver.flip_to_none(ay_var)
        }));
        match result {
            Ok(flipped) => flipped,
            Err(_) => {
                self.poisoned = true;
                false
            }
        }
    }

    /// Wire ay-sat's `minimize_model` for bulk state lifting.
    ///
    /// Iterates the trail in reverse, flipping non-important variables.
    /// Returns the remaining assignment as a minimal cube of literals.
    fn minimize_model(&mut self, important_vars: &[Var]) -> Vec<Lit> {
        if self.poisoned {
            return Vec::new();
        }
        let ay_important: Vec<ay_sat::Variable> = important_vars
            .iter()
            .map(|v| ay_sat::Variable::new(v.0))
            .collect();
        // Wrap in catch_unwind for panic resilience (#4026).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.solver.minimize_model(&ay_important)
        }));
        match result {
            Ok(ay_lits) => ay_lits.iter().map(|&l| Self::from_ay_lit(l)).collect(),
            Err(_) => {
                self.poisoned = true;
                Vec::new()
            }
        }
    }

    /// Conflict-budgeted solve using ay-sat's `solve_with_assumptions_interruptible`.
    ///
    /// ay-sat's CDCL loop calls the `should_stop` callback every ~100 conflicts.
    /// We use an invocation counter to fire after `ceil(max_conflicts / 100)`
    /// callbacks. This gives ~100-conflict granularity, which is sufficient for
    /// FRTS preprocessing (where the goal is to cap at O(100) conflicts rather
    /// than running to completion on hard pairs).
    ///
    /// For easy problems (SAT/UNSAT reached before the first callback), the
    /// budget has no effect — the solver returns the correct definitive answer.
    fn solve_with_budget(&mut self, assumptions: &[Lit], max_conflicts: u64) -> SatResult {
        self.solve_with_budget_decision(assumptions, max_conflicts)
            .as_sat_result()
    }

    /// Override the default activation-literal pattern with ay-sat's native
    /// push/pop scope mechanism. This eliminates activation literal accumulation
    /// that causes O(n) solver degradation per MIC call (#4016).
    ///
    /// Before this fix, every `solve_with_temporary_clause` call created a new
    /// activation variable and added a permanent guarded clause. Over thousands
    /// of IC3 inductiveness checks, this accumulated thousands of dead variables
    /// and clauses in the solver, degrading performance.
    ///
    /// With push/pop, the temporary clause is physically removed after the solve,
    /// keeping solver state clean.
    fn solve_with_temporary_clause(
        &mut self,
        assumptions: &[Lit],
        temp_clause: &[Lit],
    ) -> SatResult {
        self.solve_with_temporary_clause_decision(assumptions, temp_clause)
            .as_sat_result()
    }
}

// CaDiCaL solver backend REMOVED. ay-sat is our SAT solver — we own the
// full stack. Portfolio diversity comes from ay-sat configuration variants
// (restart policies, branching heuristics, preprocessing toggles).

#[cfg(test)]
mod decision_tests {
    use super::{
        AYDeadlineReason, AYSatCdclSolver, AYSolveDecision, AYSolverErrorReason,
        AYUnavailableReason, AYUnknownReason,
    };
    use crate::sat_types::{Lit, SatResult, SatSolver, Var};

    #[test]
    fn ay_decision_records_successful_sat_and_unsat() {
        let mut sat = AYSatCdclSolver::new(3);
        sat.add_clause(&[Lit::pos(Var(1))]);

        let decision = sat.solve_decision(&[]);

        assert_eq!(decision, AYSolveDecision::Sat);
        assert_eq!(decision.as_sat_result(), SatResult::Sat);
        assert_eq!(sat.last_decision(), Some(AYSolveDecision::Sat));
        assert_eq!(sat.last_decision_kind_code(), Some("sat"));
        assert_eq!(sat.last_decision_reason_code(), Some("ay_sat"));
        assert_eq!(sat.value(Lit::pos(Var(1))), Some(true));

        let mut unsat = AYSatCdclSolver::new(3);
        unsat.add_clause(&[Lit::pos(Var(1))]);
        unsat.add_clause(&[Lit::neg(Var(1))]);

        let decision = unsat.solve_decision(&[]);

        assert_eq!(decision, AYSolveDecision::Unsat);
        assert_eq!(decision.as_sat_result(), SatResult::Unsat);
        assert_eq!(unsat.last_decision_kind_code(), Some("unsat"));
        assert_eq!(unsat.last_decision_reason_code(), Some("ay_unsat"));
    }

    #[test]
    fn ay_trait_solve_records_typed_decision_without_changing_result() {
        let mut solver = AYSatCdclSolver::new(3);
        solver.add_clause(&[Lit::pos(Var(1))]);

        let result = solver.solve(&[]);

        assert_eq!(result, SatResult::Sat);
        assert_eq!(solver.last_decision(), Some(AYSolveDecision::Sat));
    }

    #[test]
    fn ay_budget_zero_is_deadline_decision() {
        let mut solver = AYSatCdclSolver::new(3);
        solver.add_clause(&[Lit::pos(Var(1))]);

        let decision = solver.solve_with_budget_decision(&[], 0);

        assert_eq!(
            decision,
            AYSolveDecision::Deadline(AYDeadlineReason::ConflictBudgetZero)
        );
        assert_eq!(decision.as_sat_result(), SatResult::Unknown);
        assert_eq!(solver.last_decision_kind_code(), Some("deadline"));
        assert_eq!(
            solver.last_decision_reason_code(),
            Some("ay_conflict_budget_zero")
        );
    }

    #[test]
    fn ay_interrupted_unknown_is_deadline_decision() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let mut solver = AYSatCdclSolver::new(3);
        solver.add_clause(&[Lit::pos(Var(1)), Lit::pos(Var(2))]);
        solver.set_cancelled(Arc::new(AtomicBool::new(true)));

        let decision = solver.solve_decision(&[]);

        assert_eq!(
            decision,
            AYSolveDecision::Deadline(AYDeadlineReason::Interrupted)
        );
        assert_eq!(decision.as_sat_result(), SatResult::Unknown);
        assert_eq!(solver.last_decision_kind_code(), Some("deadline"));
        assert_eq!(solver.last_decision_reason_code(), Some("ay_interrupted"));
    }

    #[test]
    fn ay_poisoned_solver_is_unavailable_decision() {
        let mut solver = AYSatCdclSolver::new(3);
        solver.poisoned = true;

        let decision = solver.solve_decision(&[Lit::pos(Var(1))]);

        assert_eq!(
            decision,
            AYSolveDecision::Unavailable(AYUnavailableReason::Poisoned)
        );
        assert_eq!(decision.as_sat_result(), SatResult::Unknown);
        assert_eq!(solver.last_decision_kind_code(), Some("unavailable"));
        assert_eq!(
            solver.last_decision_reason_code(),
            Some("ay_solver_poisoned")
        );
    }

    #[test]
    fn ay_decision_evidence_tokens_are_stable() {
        let samples = [
            (AYSolveDecision::Sat, "sat", "ay_sat", "Available", "Sat"),
            (
                AYSolveDecision::Unsat,
                "unsat",
                "ay_unsat",
                "Available",
                "Unsat",
            ),
            (
                AYSolveDecision::Unavailable(AYUnavailableReason::UnsupportedConfig),
                "unavailable",
                "ay_unsupported_config",
                "Unavailable",
                "Unknown",
            ),
            (
                AYSolveDecision::Unknown(AYUnknownReason::TheoryStop),
                "unknown",
                "ay_theory_stop",
                "Unknown",
                "Unknown",
            ),
            (
                AYSolveDecision::SolverError(AYSolverErrorReason::Panic),
                "solver_error",
                "ay_solver_panic",
                "SolverError",
                "Unknown",
            ),
            (
                AYSolveDecision::Deadline(AYDeadlineReason::ConflictBudgetExhausted),
                "deadline",
                "ay_conflict_budget_exhausted",
                "Deadline",
                "Unknown",
            ),
        ];

        for (decision, kind, reason, status, sat_result) in samples {
            assert_eq!(decision.kind_code(), kind);
            assert_eq!(decision.reason_code(), reason);
            assert_eq!(decision.evidence_status_name(), status);
            assert_eq!(decision.evidence_sat_result_name(), sat_result);
        }
    }

    #[test]
    fn ay_temp_clause_decision_records_unsat() {
        let mut solver = AYSatCdclSolver::new(3);
        let a = Var(1);

        let decision = solver.solve_with_temporary_clause_decision(&[Lit::pos(a)], &[Lit::neg(a)]);

        assert_eq!(decision, AYSolveDecision::Unsat);
        assert_eq!(decision.as_sat_result(), SatResult::Unsat);
        assert_eq!(solver.last_decision_kind_code(), Some("unsat"));
        assert!(solver
            .unsat_core()
            .unwrap_or_default()
            .iter()
            .all(|lit| lit.var().0 < solver.num_vars));
    }

    #[test]
    fn ay_unknown_reason_mapping_keeps_categories_distinct() {
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(Some(ay_sat::SatUnknownReason::Interrupted)),
            AYSolveDecision::Deadline(AYDeadlineReason::Interrupted)
        );
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(Some(
                ay_sat::SatUnknownReason::UnsupportedConfig
            )),
            AYSolveDecision::Unavailable(AYUnavailableReason::UnsupportedConfig)
        );
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(Some(
                ay_sat::SatUnknownReason::InvalidSatModel
            )),
            AYSolveDecision::SolverError(AYSolverErrorReason::InvalidSatModel)
        );
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(Some(
                ay_sat::SatUnknownReason::ProofFinalizationFailure
            )),
            AYSolveDecision::SolverError(AYSolverErrorReason::ProofFinalizationFailure)
        );
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(Some(
                ay_sat::SatUnknownReason::EmptyTheoryConflict
            )),
            AYSolveDecision::SolverError(AYSolverErrorReason::EmptyTheoryConflict)
        );
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(Some(ay_sat::SatUnknownReason::TheoryStop)),
            AYSolveDecision::Unknown(AYUnknownReason::TheoryStop)
        );
        assert_eq!(
            AYSolveDecision::from_ay_unknown_reason(None),
            AYSolveDecision::Unknown(AYUnknownReason::Unspecified)
        );
    }
}
