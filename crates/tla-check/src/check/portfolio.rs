// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Portfolio orchestrator for parallel multi-strategy verification.
//!
//! Runs up to six verification lanes in parallel:
//! 1. **BFS** — explicit-state model checking (always runs)
//! 2. **Random** — random walk witness search for fast bug finding (always runs)
//! 3. **PDR** — ay-based IC3 symbolic safety proving (ay feature)
//! 4. **BMC** — ay-based bounded model checking for bug finding (ay feature)
//! 5. **k-Induction** — ay-based inductive safety proving (ay feature)
//! 6. **Analytical** — proof-gated pre-solve for supported interval-counter
//!    execution models and static finite-cardinality invariants.
//!
//! The first lane to reach a definitive result publishes its verdict via
//! [`SharedVerdict`], and the other lanes exit early on their next poll.
//!
//! Part of #3717.

use std::sync::Arc;

use num_traits::ToPrimitive;
use tla_core::ast::{Expr, Module, Unit};
use tla_mc_core::{
    AnalyticalSolveDecision, AnalyticalSolveDecisionReason, AnalyticalSolveDecisionStatus,
    AnalyticalSolvePortfolioLifecycle, BackendKind, CheckerArtifactIdentityFields,
    PreparedAnalyticalSolveKind, PreparedBackendFamilyDescriptor, PreparedCandidateLaneDescriptor,
    PreparedCanonicalIdentityDescriptor, PreparedCanonicalIdentityKind, PreparedCheckerProgram,
    PreparedFingerprintDescriptor, PreparedFingerprintScheme, PreparedProgramPayloadKind,
    PreparedPropertyKind, PreparedStorageKind, PreparedSymbolicProofKind, PreparedTransitionKind,
    PreparedValidationKind, PreparedValidationPlanDescriptor, ProblemKind, SetupTraceLaneKind,
    SharedEngineFrontendFamily, SolverFacet, ValidationReceipt, ValidationReceiptArtifactKind,
    ValidationReceiptValidatorKind,
};

use crate::analytical::bound_context::BoundAnalyticalContext;
use crate::analytical::finite_cardinality::{
    admit_module_set_finite_cardinality_invariant, FiniteCardinalityAdmissionCertificate,
};
use crate::analytical::interval_counter::{
    admit_module_interval_counter_execution_model, admit_module_interval_counters,
    IntervalCounterExecutionCertificate,
};
use crate::analytical::{AnalyticalAdmission, VerificationGate, VerifiedProof};
use crate::check::{CheckResult, CheckStats, ModelChecker, RandomWalkConfig, RandomWalkResult};
use crate::config::Config;
#[cfg(feature = "ay")]
use crate::eval::EvalCtx;
use crate::shared_verdict::{SharedVerdict, Verdict};

/// Default number of random walks per portfolio run.
const DEFAULT_RANDOM_WALKS: usize = 100;
/// Default maximum depth per random walk.
const DEFAULT_WALK_DEPTH: usize = 10_000;
/// CLI/API strategy name for the analytical-engine scaffold.
const ANALYTICAL_STRATEGY: &str = "analytical";
/// Canonicalization version used by analytical portfolio prepared-program ids.
const ANALYTICAL_CANONICALIZATION_VERSION: &str = "tla-check:analytical-module-shape:v1";
/// Prepared analytical semantic identity schema.
const ANALYTICAL_SEMANTIC_DIGEST_VERSION: &str = "tla-check:analytical-prepared-program:v2";
/// Shared local structural backend-family id for TLA analytical portfolio proofs.
const ANALYTICAL_STRUCTURAL_BACKEND_FAMILY: &str = "tla.analytical.structural";
/// Local TLA analytical cache/proof evidence is not cross-frontend reusable.
const ANALYTICAL_CACHE_COMPAT_FRONTEND_LOCAL_ONLY: &str = "frontend_local_only";
/// AY lanes may reuse cache/fingerprint evidence once their shared admission proves it.
#[cfg(feature = "ay")]
const ANALYTICAL_CACHE_COMPAT_FRONTEND_REUSABLE: &str = "frontend_reusable";
/// Portfolio-owned row schema for shared validation receipts attached to analytical decisions.
const SHARED_ENGINE_VALIDATION_RECEIPT_SCHEMA: &str =
    "tla-check.shared-engine.validation-receipt.v1";
/// Digest algorithm label used when a AY receipt validates a prepared fingerprint identity.
#[cfg(feature = "ay")]
const AY_SHARED_VALIDATION_DIGEST_ALGORITHM: &str = "ay_fingerprint_identity";
/// Digest algorithm label used for frontend-local certificate receipts.
const ANALYTICAL_ARTIFACT_DIGEST_ALGORITHM: &str = "fnv1a64";
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Structural/proof status for the analytical engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticalEligibility {
    /// A verified analytical proof replaced explicit state exploration.
    VerifiedExecutionModel,
    /// Verified static invariant facts replaced explicit state exploration.
    VerifiedStaticInvariant,
    /// The spec shape has a verified invariant proof but not a complete execution-model proof.
    StructurallyEligible,
    /// The spec shape is not eligible for analytical handling.
    StructurallyIneligible,
    /// The analytical scaffold was not requested, so no eligibility was assessed.
    NotAssessed,
}

impl AnalyticalEligibility {
    /// Stable machine-readable eligibility code used by CLI/reporting layers.
    pub fn code(self) -> &'static str {
        match self {
            AnalyticalEligibility::VerifiedExecutionModel => "verified_execution_model",
            AnalyticalEligibility::VerifiedStaticInvariant => "verified_static_invariant",
            AnalyticalEligibility::StructurallyEligible => "structurally_eligible",
            AnalyticalEligibility::StructurallyIneligible => "structurally_ineligible",
            AnalyticalEligibility::NotAssessed => "not_assessed",
        }
    }

    /// Human-readable eligibility wording used by CLI/reporting layers.
    pub fn wording(self) -> &'static str {
        match self {
            AnalyticalEligibility::VerifiedExecutionModel => {
                "verified analytical execution model; explicit state exploration was skipped"
            }
            AnalyticalEligibility::VerifiedStaticInvariant => {
                "verified static finite-cardinality invariant proof; explicit state exploration was skipped"
            }
            AnalyticalEligibility::StructurallyEligible => {
                "verified analytical invariant proof exists, but explicit exploration is still required"
            }
            AnalyticalEligibility::StructurallyIneligible => {
                "not structurally eligible for analytical handling"
            }
            AnalyticalEligibility::NotAssessed => {
                "analytical structural eligibility was not assessed"
            }
        }
    }
}

/// Which lane won the portfolio race.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortfolioWinner {
    /// Analytical proof resolved before runtime exploration.
    Analytical,
    /// BFS explicit-state model checking resolved first.
    Bfs,
    /// Random walk witness search resolved first.
    Random,
    /// PDR (IC3) symbolic checking resolved first.
    Pdr,
    /// BMC symbolic bug finding resolved first.
    Bmc,
    /// k-Induction symbolic proving resolved first.
    KInduction,
}

/// Result of a portfolio verification run.
#[derive(Debug)]
pub struct PortfolioResult {
    /// Which lane resolved the verdict first.
    pub winner: PortfolioWinner,
    /// Analytical structural eligibility, if the analytical lane was requested.
    pub analytical_eligibility: AnalyticalEligibility,
    /// Frontend-neutral analytical admission/routing evidence rows.
    pub analytical_solve_evidence: Vec<String>,
    /// Source-aware shared validation receipt rows backing analytical solve publication.
    pub shared_engine_validation_receipts: Vec<String>,
    /// The BFS result (always present — BFS always runs to completion or early exit).
    pub bfs_result: CheckResult,
    /// The random walk result (always present — random walks always run).
    pub random_result: Option<RandomWalkResult>,
    /// The PDR result, if the ay feature is enabled and PDR ran successfully.
    /// `None` when PDR is not available (ay feature disabled) or failed with an error.
    #[cfg(feature = "ay")]
    pub pdr_result: Option<Result<crate::ay_pdr::PdrResult, crate::ay_pdr::PdrError>>,
    /// AY CHC/PDR proof-replay evidence row from the PDR lane, when it ran.
    #[cfg(feature = "ay")]
    pub pdr_proof_replay_evidence: Option<String>,
    /// Frontend-neutral AY shared-engine metadata evidence consumed by this run.
    #[cfg(feature = "ay")]
    pub ay_shared_engine_evidence: Vec<String>,
    /// The BMC result, if the ay feature is enabled and BMC ran.
    #[cfg(feature = "ay")]
    pub bmc_result: Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>>,
    /// The k-induction result, if the ay feature is enabled and k-induction ran.
    #[cfg(feature = "ay")]
    pub kinduction_result: Option<
        Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
    >,
}

impl PortfolioResult {
    /// Reconcile `bfs_result` against the racing lanes, **fail-closed** — the
    /// portfolio-mode mirror of [`crate::check::fused::FusedResult::reconcile_masked_violation`].
    ///
    /// A racing lane that resolves the `Violated` verdict truncates the BFS
    /// lane into a result indistinguishable from a clean Success; reporting
    /// that `bfs_result` prints "No error has been found" with exit 0 while a
    /// lane holds a real counterexample (cex-loss). This method returns:
    ///
    /// - `SymbolicViolation` when the winning lane's counterexample is
    ///   trustworthy: a concrete random-walk violation (interpreter-executed,
    ///   inherently validated) or a symbolic counterexample the explicit-state
    ///   evaluator re-confirmed on demand.
    /// - `UnvalidatedSymbolicViolation` (fail closed) when a symbolic lane won
    ///   the `Violated` race but its counterexample could not be re-validated —
    ///   never "no error" on the strength of a race-truncated BFS result.
    /// - `FromBfs` otherwise (the common case — `bfs_result` is authoritative).
    pub fn reconcile_masked_violation(
        &self,
        module: &Module,
        config: &Config,
    ) -> crate::check::fused::ReconciledVerdict {
        use crate::check::fused::ReconciledVerdict;
        // A violation found by the explicit BFS lane is authoritative and is
        // already reported from `bfs_result` — nothing is masked.
        if matches!(
            self.bfs_result,
            CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. }
        ) {
            return ReconciledVerdict::FromBfs;
        }
        match self.winner {
            // BFS won, or a safety-proving lane won the Satisfied race
            // (certificate/receipt-gated elsewhere) — nothing is masked.
            PortfolioWinner::Bfs
            | PortfolioWinner::Analytical
            | PortfolioWinner::Pdr
            | PortfolioWinner::KInduction => ReconciledVerdict::FromBfs,
            // The random walk executes the spec with the concrete interpreter —
            // its counterexample is inherently validated.
            PortfolioWinner::Random => match &self.random_result {
                Some(RandomWalkResult::InvariantViolation {
                    invariant,
                    trace,
                    walk_id,
                    depth,
                }) => ReconciledVerdict::SymbolicViolation {
                    lane: "Random walk",
                    detail: format!(
                        "random walk {walk_id} found invariant '{invariant}' violated at \
                         depth {depth} (concrete interpreter execution)"
                    ),
                    invariant: Some(invariant.clone()),
                    trace: trace.clone(),
                },
                Some(RandomWalkResult::Deadlock {
                    trace,
                    walk_id,
                    depth,
                }) => ReconciledVerdict::SymbolicViolation {
                    lane: "Random walk",
                    detail: format!(
                        "random walk {walk_id} reached a deadlock at depth {depth} \
                         (concrete interpreter execution)"
                    ),
                    invariant: None,
                    trace: trace.clone(),
                },
                _ => ReconciledVerdict::FromBfs,
            },
            // `determine_winner` attributes every symbolic `Violated` race win
            // to Bmc (the catch-all): pair it with the lane result that
            // actually carries the counterexample and re-validate on demand.
            PortfolioWinner::Bmc => self.reconcile_symbolic_violated_race(module, config),
        }
    }

    /// Locate the cex-bearing symbolic lane result behind a `Violated` race
    /// win and re-validate it through the explicit-state evaluator (ay builds).
    #[cfg(feature = "ay")]
    fn reconcile_symbolic_violated_race(
        &self,
        module: &Module,
        config: &Config,
    ) -> crate::check::fused::ReconciledVerdict {
        use crate::check::cross_validation::{
            confirm_symbolic_cex_fail_closed, pdr_trace_to_bmc_states, CrossValidationSource,
        };
        use crate::check::fused::ReconciledVerdict;

        // Find the counterexample-bearing lane result.
        let (lane, source, trace) =
            if let Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample {
                trace, ..
            })) = &self.kinduction_result
            {
                (
                    "k-Induction",
                    CrossValidationSource::KInduction,
                    trace.clone(),
                )
            } else if let Some(Ok(crate::ay_bmc::BmcResult::Violation { trace, .. })) =
                &self.bmc_result
            {
                ("BMC", CrossValidationSource::Bmc, trace.clone())
            } else if let Some(Ok(crate::ay_pdr::PdrResult::Unsafe { trace })) = &self.pdr_result {
                (
                    "PDR",
                    CrossValidationSource::Pdr,
                    pdr_trace_to_bmc_states(trace),
                )
            } else if let Some(Ok(crate::ay_bmc::BmcResult::Deadlock { depth, .. })) =
                &self.bmc_result
            {
                // Deadlock counterexamples cannot be re-validated by the invariant
                // replay — fail closed rather than report "no error".
                return ReconciledVerdict::UnvalidatedSymbolicViolation {
                    lane: "BMC",
                    detail: format!(
                        "BMC reported a reachable deadlock at depth {depth}; deadlock \
                     counterexamples cannot be re-validated by the invariant replay — \
                     failing closed"
                    ),
                };
            } else {
                // Violated race resolved but no lane result carries the
                // counterexample (lost/abandoned lane) — fail closed.
                return ReconciledVerdict::UnvalidatedSymbolicViolation {
                    lane: "BMC",
                    detail: "a symbolic lane resolved the Violated race but no lane result \
                         carries the counterexample — failing closed"
                        .to_string(),
                };
            };

        let cv = confirm_symbolic_cex_fail_closed(module, config, &trace, source);
        if cv.engine_agrees {
            if let (Some(invariant), Some(trace)) = (cv.violated_invariant, cv.validated_trace) {
                ReconciledVerdict::SymbolicViolation {
                    lane,
                    detail: cv.detail,
                    invariant: Some(invariant),
                    trace,
                }
            } else {
                // Confirmation without the replayed payload — fail closed.
                ReconciledVerdict::UnvalidatedSymbolicViolation {
                    lane,
                    detail: cv.detail,
                }
            }
        } else {
            ReconciledVerdict::UnvalidatedSymbolicViolation {
                lane,
                detail: cv.detail,
            }
        }
    }

    /// Non-ay builds have no symbolic lanes; a `Violated` race win not owned by
    /// BFS or the random walk cannot carry a counterexample — fail closed.
    #[cfg(not(feature = "ay"))]
    fn reconcile_symbolic_violated_race(
        &self,
        _module: &Module,
        _config: &Config,
    ) -> crate::check::fused::ReconciledVerdict {
        crate::check::fused::ReconciledVerdict::UnvalidatedSymbolicViolation {
            lane: "BMC",
            detail: "a lane resolved the Violated race but no lane result carries the \
                     counterexample — failing closed"
                .to_string(),
        }
    }

    /// Run portfolio verification while preserving the source frontend that
    /// produced the TLA AST consumed by the solver lanes.
    ///
    /// This only changes descriptor/evidence identity. Solver setup and lane
    /// behavior remain the same TLA AST execution paths.
    pub fn run_with_frontend_source(
        module: &Module,
        checker_modules: &[&Module],
        config: &Config,
        strategy_filter: &[String],
        frontend_source_is_quint: bool,
    ) -> Self {
        let payload_kind = portfolio_payload_kind_for_frontend_source(frontend_source_is_quint);
        run_portfolio_with_payload_kind(
            module,
            checker_modules,
            config,
            strategy_filter,
            payload_kind,
            frontend_source_is_quint,
        )
    }
}

/// Run parallel multi-strategy portfolio verification.
///
/// Creates a [`SharedVerdict`] and spawns all lanes using [`std::thread::scope`].
/// The first lane to resolve wins; the other lanes exit early on their next poll.
///
/// # Arguments
///
/// * `module` - The loaded TLA+ module
/// * `checker_modules` - Additional modules loaded via EXTENDS/INSTANCE
/// * `config` - TLC configuration with INIT, NEXT, INVARIANT
/// * `strategy_filter` - If non-empty, only run strategies whose names are in this list.
///   Valid names: `"bfs"`, `"random"`, `"bmc"`, `"pdr"`, `"kinduction"`,
///   `"analytical"`.
///   If empty, all strategies run.
///
/// # Returns
///
/// A [`PortfolioResult`] containing the winner and all lane results.
///
/// Part of #3717, #3816.
pub fn run_portfolio(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    strategy_filter: &[String],
) -> PortfolioResult {
    run_portfolio_with_payload_kind(
        module,
        checker_modules,
        config,
        strategy_filter,
        PreparedProgramPayloadKind::Tla,
        false,
    )
}

fn run_portfolio_with_payload_kind(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    strategy_filter: &[String],
    payload_kind: PreparedProgramPayloadKind,
    frontend_source_is_quint: bool,
) -> PortfolioResult {
    let runtime_config = config.runtime_model_config();
    let config = &runtime_config;

    if analytical_requested(strategy_filter) {
        if let Some(result) =
            try_run_analytical_execution_model(module, checker_modules, config, payload_kind)
        {
            return result;
        }
        if let Some(result) =
            try_run_analytical_static_invariant_proof(module, checker_modules, config, payload_kind)
        {
            return result;
        }
    }

    let verdict = Arc::new(SharedVerdict::new());

    std::thread::scope(|scope| {
        let verdict_bfs = verdict.clone();
        let verdict_random = verdict.clone();
        let verdict_analytical = verdict.clone();
        #[cfg(feature = "ay")]
        let verdict_pdr = verdict.clone();
        #[cfg(feature = "ay")]
        let verdict_bmc = verdict.clone();
        #[cfg(feature = "ay")]
        let verdict_kind = verdict.clone();

        // Lane 1: BFS explicit-state model checking.
        let run_bfs = should_run_strategy(strategy_filter, "bfs");
        let bfs_handle = scope.spawn(move || {
            if !run_bfs {
                verdict_bfs.publish(Verdict::Unknown);
                // Return a minimal success result so the portfolio can still determine a winner.
                return CheckResult::Success(crate::check::CheckStats::default());
            }
            let mut checker = ModelChecker::new_with_extends(module, checker_modules, config);
            checker.set_frontend_source_is_quint(frontend_source_is_quint);
            checker.set_portfolio_verdict(verdict_bfs);
            checker.check()
        });

        // Lane 2: Random walk witness search (fast bug finding, zero memory).
        let run_random = should_run_strategy(strategy_filter, "random");
        let random_handle = scope.spawn(move || {
            let sv = verdict_random;
            if !run_random {
                sv.publish(Verdict::Unknown);
                return RandomWalkResult::NoViolationFound {
                    walks_completed: 0,
                    total_steps: 0,
                };
            }
            let mut checker = ModelChecker::new_with_extends(module, checker_modules, config);
            checker.set_frontend_source_is_quint(frontend_source_is_quint);
            let walk_config = RandomWalkConfig {
                num_walks: DEFAULT_RANDOM_WALKS,
                max_depth: DEFAULT_WALK_DEPTH,
                seed: None,
            };
            let result = checker.random_walk(&walk_config);
            // Publish random walk verdict to SharedVerdict.
            let v = match &result {
                RandomWalkResult::InvariantViolation { .. } | RandomWalkResult::Deadlock { .. } => {
                    Verdict::Violated
                }
                RandomWalkResult::NoViolationFound { .. } => Verdict::Unknown,
                RandomWalkResult::Error(_) => Verdict::Unknown,
            };
            sv.publish(v);
            result
        });

        // Lane 6: Analytical structural-eligibility scaffold.
        let run_analytical = analytical_requested(strategy_filter);
        let analytical_handle = scope.spawn(move || {
            if !run_analytical {
                return AnalyticalEligibility::NotAssessed;
            }
            let eligibility = analytical_eligibility_for_config(module, checker_modules, config);
            let _ = eligibility.wording();
            let _ = verdict_analytical;
            eligibility
        });

        // Lane 3: PDR symbolic checking (ay feature required).
        #[cfg(feature = "ay")]
        let run_pdr = should_run_strategy(strategy_filter, "pdr");
        #[cfg(feature = "ay")]
        let pdr_handle = scope.spawn(move || {
            let sv = verdict_pdr.clone();
            if !run_pdr {
                sv.publish(Verdict::Unknown);
                return Ok(crate::ay_pdr::unknown_pdr_run_with_missing_evidence(
                    "filtered out by strategy_filter",
                ));
            }
            let mut pdr_ctx = EvalCtx::new();
            pdr_ctx.load_module(module);
            // Default PDR config with 300s timeout (matches check_pdr() behavior).
            let mut pdr_config: tla_ay::PdrConfig = Default::default();
            pdr_config.solve_timeout = Some(std::time::Duration::from_secs(300));
            let pdr_run_result = crate::ay_pdr::check_pdr_with_portfolio_and_evidence(
                module,
                config,
                &pdr_ctx,
                pdr_config,
                Some(verdict_pdr),
            );
            // Publish PDR verdict to SharedVerdict.
            if let Ok(ref run) = pdr_run_result {
                let v = match &run.result {
                    // A PDR safety proof only encodes the invariant conjunction.
                    // Publishing Satisfied when the run carries OTHER obligations
                    // (deadlock-checking — on by default — PROPERTIES, trace
                    // invariants, or an unresolved SPECIFICATION) truncates the
                    // racing BFS lane into a vacuous Success, masking a reachable
                    // deadlock or liveness violation. The cooperative/fused path
                    // already gates this exact publish (ay_pdr.rs:709); the
                    // portfolio path was the residual ungated variant. Gate it
                    // identically: a symbolic safety win only resolves the slot
                    // when safety is the run's sole obligation; otherwise leave
                    // BFS authoritative (downgrade to Unknown).
                    crate::ay_pdr::PdrResult::Safe { .. }
                        if crate::ay_shared::symbolic_safety_proof_covers_all_obligations(
                            config,
                        ) =>
                    {
                        Verdict::Satisfied
                    }
                    crate::ay_pdr::PdrResult::Safe { .. } => Verdict::Unknown,
                    // SOUNDNESS (fail closed): a `Violated` publish truncates the
                    // racing BFS lane into a clean-looking result, so it may
                    // happen ONLY after the explicit-state evaluator re-confirmed
                    // the counterexample. A spurious CHC model publishes nothing —
                    // BFS keeps running, unharmed. Mirrors the cooperative/fused
                    // PDR lane gate (ay_pdr.rs).
                    crate::ay_pdr::PdrResult::Unsafe { trace } => {
                        let bmc_states =
                            crate::check::cross_validation::pdr_trace_to_bmc_states(trace);
                        if crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                            module,
                            config,
                            &bmc_states,
                            crate::check::cross_validation::CrossValidationSource::Pdr,
                        )
                        .engine_agrees
                        {
                            Verdict::Violated
                        } else {
                            telemetry_eprintln!(
                                "[portfolio] PDR unsafe trace failed explicit-evaluator \
                                 cross-validation — failing closed (not publishing Violated)"
                            );
                            Verdict::Unknown
                        }
                    }
                    crate::ay_pdr::PdrResult::Unknown { .. } => Verdict::Unknown,
                };
                let program = prepared_analytical_portfolio_program(
                    module,
                    checker_modules,
                    config,
                    payload_kind,
                );
                let decisions =
                    ay_runtime_analytical_solve_decisions(&program, Some(run), None, None);
                ay_publish_verdict_with_shared_validation_receipt_decisions(&sv, v, &decisions);
            }
            pdr_run_result
        });

        // Lane 4: BMC symbolic bug finding (ay feature required).
        #[cfg(feature = "ay")]
        let run_bmc = should_run_strategy(strategy_filter, "bmc");
        #[cfg(feature = "ay")]
        let bmc_handle = scope.spawn(move || {
            let sv = verdict_bmc.clone();
            if !run_bmc {
                sv.publish(Verdict::Unknown);
                return Ok(crate::ay_bmc::unknown_bmc_run_with_missing_evidence(
                    "filtered out by strategy_filter",
                ));
            }
            let mut bmc_ctx = EvalCtx::new();
            bmc_ctx.load_module(module);
            let bmc_config = crate::ay_bmc::BmcConfig {
                max_depth: 20,
                ..Default::default()
            };
            let bmc_result = crate::ay_bmc::check_bmc_with_portfolio_and_evidence(
                module,
                config,
                &bmc_ctx,
                bmc_config,
                Some(verdict_bmc),
            );
            if let Ok(ref run) = bmc_result {
                let v = match &run.result {
                    // SOUNDNESS (fail closed): publish `Violated` — which
                    // truncates the racing BFS lane — ONLY after the explicit-
                    // state evaluator re-confirmed the counterexample. Mirrors
                    // the cooperative/fused BMC lane gate (ay_bmc.rs).
                    crate::ay_bmc::BmcResult::Violation { trace, .. } => {
                        if crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                            module,
                            config,
                            trace,
                            crate::check::cross_validation::CrossValidationSource::Bmc,
                        )
                        .engine_agrees
                        {
                            Verdict::Violated
                        } else {
                            telemetry_eprintln!(
                                "[portfolio] BMC violation failed explicit-evaluator \
                                 cross-validation — failing closed (not publishing Violated)"
                            );
                            Verdict::Unknown
                        }
                    }
                    // A reachable deadlock is a property failure (mirrors BFS).
                    // Deadlock counterexamples cannot be re-validated by the
                    // invariant replay; the masked-violation reconciliation
                    // fails closed at reporting time instead.
                    crate::ay_bmc::BmcResult::Deadlock { .. } => Verdict::Violated,
                    crate::ay_bmc::BmcResult::BoundReached { .. } => Verdict::Unknown,
                    crate::ay_bmc::BmcResult::Unknown { .. } => Verdict::Unknown,
                };
                let program = prepared_analytical_portfolio_program(
                    module,
                    checker_modules,
                    config,
                    payload_kind,
                );
                let decisions =
                    ay_runtime_analytical_solve_decisions(&program, None, Some(run), None);
                ay_publish_verdict_with_shared_validation_receipt_decisions(&sv, v, &decisions);
            }
            bmc_result
        });

        // Lane 5: k-Induction symbolic proving (ay feature required).
        #[cfg(feature = "ay")]
        let run_kind = should_run_strategy(strategy_filter, "kinduction");
        #[cfg(feature = "ay")]
        let kind_handle = scope.spawn(move || {
            let sv = verdict_kind.clone();
            if !run_kind {
                sv.publish(Verdict::Unknown);
                return Ok(crate::ay_kinduction::KInductionResult::Unknown {
                    max_k: 0,
                    reason: "filtered out by strategy_filter".to_string(),
                });
            }
            let mut kind_ctx = EvalCtx::new();
            kind_ctx.load_module(module);
            let kind_config = crate::ay_kinduction::KInductionConfig::default();
            let kind_result = crate::ay_kinduction::check_kinduction_with_portfolio(
                module,
                config,
                &kind_ctx,
                kind_config,
                Some(verdict_kind),
            );
            if let Ok(ref result) = kind_result {
                let v = match result {
                    // Same obligation-coverage gate as the PDR lane above: a
                    // k-induction safety proof covers only the invariant
                    // conjunction, so it may resolve the shared slot to Satisfied
                    // ONLY when safety is the run's sole obligation. Otherwise it
                    // would truncate BFS into a vacuous Success and mask a
                    // reachable deadlock/liveness violation (cf. the gated
                    // cooperative path at ay_kinduction.rs:623).
                    crate::ay_kinduction::KInductionResult::Proved { .. }
                        if crate::ay_shared::symbolic_safety_proof_covers_all_obligations(
                            config,
                        ) =>
                    {
                        Verdict::Satisfied
                    }
                    crate::ay_kinduction::KInductionResult::Proved { .. } => Verdict::Unknown,
                    // SOUNDNESS (fail closed): a k-Induction base-case
                    // counterexample may publish `Violated` — truncating the
                    // racing BFS lane — ONLY after the explicit-state evaluator
                    // re-confirmed it. A lane whose base case it could not
                    // discharge must not win the race. Mirrors the
                    // cooperative/fused k-Induction gate (ay_kinduction.rs).
                    crate::ay_kinduction::KInductionResult::Counterexample { trace, .. } => {
                        if crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                            module,
                            config,
                            trace,
                            crate::check::cross_validation::CrossValidationSource::KInduction,
                        )
                        .engine_agrees
                        {
                            Verdict::Violated
                        } else {
                            telemetry_eprintln!(
                                "[portfolio] k-Induction base-case counterexample failed \
                                 explicit-evaluator cross-validation — failing closed \
                                 (not publishing Violated)"
                            );
                            Verdict::Unknown
                        }
                    }
                    crate::ay_kinduction::KInductionResult::Unknown { .. } => Verdict::Unknown,
                };
                let program = prepared_analytical_portfolio_program(
                    module,
                    checker_modules,
                    config,
                    payload_kind,
                );
                let decisions =
                    ay_runtime_analytical_solve_decisions(&program, None, None, Some(result));
                ay_publish_verdict_with_shared_validation_receipt_decisions(&sv, v, &decisions);
            }
            kind_result
        });

        let bfs_result = bfs_handle.join().expect("BFS thread panicked");
        let random_result = Some(random_handle.join().expect("Random walk thread panicked"));
        let analytical_eligibility = analytical_handle
            .join()
            .expect("Analytical scaffold thread panicked");

        #[cfg(feature = "ay")]
        let pdr_run_result = pdr_handle.join().expect("PDR thread panicked");
        #[cfg(feature = "ay")]
        let bmc_run_result = bmc_handle.join().expect("BMC thread panicked");
        #[cfg(feature = "ay")]
        let kinduction_run_result = kind_handle.join().expect("k-induction thread panicked");
        #[cfg(feature = "ay")]
        let pdr_proof_replay_evidence = pdr_run_result
            .as_ref()
            .ok()
            .map(|run| run.proof_replay_evidence.clone());

        // Determine winner based on which verdict was published first.
        let winner = determine_winner(&verdict, &bfs_result, &random_result);
        let analytical_evidence = analytical_solve_evidence_for_payload_kind(
            module,
            checker_modules,
            config,
            payload_kind,
            analytical_eligibility,
        );
        #[cfg(feature = "ay")]
        let analytical_evidence = {
            let mut analytical_evidence = analytical_evidence;
            let ay_runtime_evidence = ay_runtime_analytical_solve_evidence_for_payload_kind(
                module,
                checker_modules,
                config,
                payload_kind,
                pdr_run_result.as_ref().ok(),
                bmc_run_result.as_ref().ok(),
                kinduction_run_result.as_ref().ok(),
            );
            analytical_evidence
                .decision_rows
                .extend(ay_runtime_evidence.decision_rows);
            analytical_evidence
                .validation_receipt_rows
                .extend(ay_runtime_evidence.validation_receipt_rows);
            analytical_evidence
        };

        #[cfg(feature = "ay")]
        let pdr_result = Some(pdr_run_result.map(crate::ay_pdr::PdrRunResult::into_result));
        #[cfg(feature = "ay")]
        let bmc_result = Some(bmc_run_result.map(crate::ay_bmc::BmcRunResult::into_result));
        #[cfg(feature = "ay")]
        let kinduction_result = Some(kinduction_run_result);

        PortfolioResult {
            winner,
            analytical_eligibility,
            analytical_solve_evidence: analytical_evidence.decision_rows,
            shared_engine_validation_receipts: analytical_evidence.validation_receipt_rows,
            bfs_result,
            random_result,
            #[cfg(feature = "ay")]
            pdr_result,
            #[cfg(feature = "ay")]
            pdr_proof_replay_evidence,
            #[cfg(feature = "ay")]
            ay_shared_engine_evidence: ay_shared_engine_evidence_rows_for_payload_kind(
                module,
                checker_modules,
                config,
                payload_kind,
            ),
            #[cfg(feature = "ay")]
            bmc_result,
            #[cfg(feature = "ay")]
            kinduction_result,
        }
    })
}

/// Build frontend-neutral prepared-program metadata for TLA analytical portfolio candidates.
///
/// This is descriptor-only: it records analytical obligations admitted by the
/// existing structural proof gates, but it does not publish verdicts or change
/// backend selection.
#[must_use]
pub fn prepared_analytical_portfolio_program(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
) -> PreparedCheckerProgram {
    let runtime_config = config.runtime_model_config();
    let config = &runtime_config;
    let context = BoundAnalyticalContext::new(module, checker_modules, config);
    let root_module = context.root_module();
    let root_digest = root_module.source_independent_digest().to_string();
    let semantic_digest = prepared_analytical_semantic_digest(&context, config, payload_kind);
    let identity = format!("{}#analytical:{semantic_digest}", root_module.name());
    let semantic_fingerprint_identity =
        prepared_descriptor_id("tla.analytical.semantic_fingerprint", &semantic_digest);
    let identity_fields = CheckerArtifactIdentityFields::new()
        .with_cache_key(prepared_descriptor_id(
            "tla.analytical.prepared_cache",
            &semantic_digest,
        ))
        .with_source_fingerprint(root_digest.clone())
        .with_frontend_payload_identity(prepared_descriptor_id(
            "tla.analytical.frontend_payload",
            &semantic_digest,
        ))
        .with_prepared_program_fingerprint(semantic_digest.clone())
        .with_artifact_identity(prepared_descriptor_id(
            "tla.analytical.prepared_artifact",
            &semantic_digest,
        ))
        .with_storage_policy_identity("tla_state_slots")
        .with_storage_layout_fingerprint(prepared_descriptor_id(
            "tla.analytical.storage_layout",
            &root_digest,
        ))
        .with_fingerprint_policy_identity("tla_analytical_semantic_fingerprint_v1")
        .with_fingerprint_identity(semantic_fingerprint_identity);
    let mut program = PreparedCheckerProgram::new(
        identity.clone(),
        payload_kind,
        PreparedStorageKind::TlaStateSlots,
    )
    .with_identity_fields(identity_fields)
    .add_canonical_identity(
        PreparedCanonicalIdentityDescriptor::new(
            format!("module:{}", root_module.name()),
            PreparedCanonicalIdentityKind::CanonicalPayload,
            ANALYTICAL_CANONICALIZATION_VERSION,
        )
        .with_digest("fnv1a64", root_digest.clone()),
    );

    if let Some(next) = config.next.as_deref() {
        program = program.add_transition(next, PreparedTransitionKind::TlaAction);
    }

    program = add_prepared_properties(
        program,
        "invariant",
        &config.invariants,
        PreparedPropertyKind::Invariant,
    );
    program = add_prepared_properties(
        program,
        "trace_invariant",
        &config.trace_invariants,
        PreparedPropertyKind::Invariant,
    );
    program = add_prepared_properties(
        program,
        "state_constraint",
        &config.constraints,
        PreparedPropertyKind::StateConstraint,
    );
    program = add_prepared_properties(
        program,
        "action_constraint",
        &config.action_constraints,
        PreparedPropertyKind::ProofObligation,
    );
    program = add_prepared_properties(
        program,
        "property",
        &config.properties,
        PreparedPropertyKind::Ltl,
    );
    if config.check_deadlock {
        program = program.add_property("deadlock", PreparedPropertyKind::Deadlock);
    }

    let mut has_analytical_candidate = false;
    if let (Some(init), Some(next), Some(invariant)) = (
        config.init.as_deref(),
        config.next.as_deref(),
        config.invariants.first().map(String::as_str),
    ) {
        if analytical_config_obligations_supported(config) && config.invariants.len() == 1 {
            if let AnalyticalAdmission::VerifiedProof(verified) =
                admit_module_interval_counters(module, init, next, invariant)
            {
                let certificate = verified.certificate();
                program = program
                    .add_analytical_solve(
                        prepared_descriptor_id(
                            "tla.analytical.interval_counter.invariant",
                            invariant,
                        ),
                        PreparedAnalyticalSolveKind::LinearInvariant,
                        ProblemKind::Invariant,
                    )
                    .add_symbolic_proof(
                        prepared_descriptor_id("tla.analytical.interval_counter.proof", invariant),
                        PreparedSymbolicProofKind::InvariantProof,
                        ProblemKind::Invariant,
                    );

                if certificate.initial_states_cover_invariant()
                    && (!config.check_deadlock || certificate.transition_total_on_invariant())
                {
                    program = program.add_analytical_solve(
                        prepared_descriptor_id(
                            "tla.analytical.interval_counter.state_space",
                            invariant,
                        ),
                        PreparedAnalyticalSolveKind::StateSpaceCardinality,
                        ProblemKind::StateSpace,
                    );
                }
                if config.check_deadlock && certificate.transition_total_on_invariant() {
                    program = program.add_analytical_solve(
                        prepared_descriptor_id(
                            "tla.analytical.interval_counter.deadlock",
                            invariant,
                        ),
                        PreparedAnalyticalSolveKind::DeadlockFreedom,
                        ProblemKind::Deadlock,
                    );
                }
                has_analytical_candidate = true;
            }
        }
    }

    for invariant in &config.invariants {
        if let AnalyticalAdmission::VerifiedProof(_) =
            admit_module_set_finite_cardinality_invariant(module, checker_modules, invariant)
        {
            program = program
                .add_analytical_solve(
                    prepared_descriptor_id("tla.analytical.finite_cardinality", invariant),
                    PreparedAnalyticalSolveKind::UpperBounds,
                    ProblemKind::Invariant,
                )
                .add_symbolic_proof(
                    prepared_descriptor_id("tla.analytical.finite_cardinality.proof", invariant),
                    PreparedSymbolicProofKind::InvariantProof,
                    ProblemKind::Invariant,
                );
            has_analytical_candidate = true;
        }
    }

    if has_analytical_candidate {
        program = add_analytical_candidate_readiness(program, root_module.name(), &semantic_digest);
    }

    #[cfg(feature = "ay")]
    if ay_shared_engine_config_supported(config) {
        program = crate::ay_shared::add_ay_shared_engine_prepared_descriptors(
            program,
            &identity,
            &semantic_digest,
        );
    }

    program
}

fn add_analytical_candidate_readiness(
    program: PreparedCheckerProgram,
    module_name: &str,
    semantic_digest: &str,
) -> PreparedCheckerProgram {
    let candidate_identity = prepared_descriptor_id("tla.analytical.candidate", module_name);
    let proof_fingerprint_identity =
        prepared_descriptor_id("tla.analytical.proof_fingerprint", semantic_digest);
    let proof_artifact_identity =
        prepared_descriptor_id("tla.analytical.proof_artifact", module_name);

    program
        .add_backend_family(
            PreparedBackendFamilyDescriptor::new(
                ANALYTICAL_STRUCTURAL_BACKEND_FAMILY,
                BackendKind::LocalSymbolicExecution,
                ProblemKind::Invariant,
            )
            .with_facet(SolverFacet::Proof)
            .with_facet(SolverFacet::LinearIntegerArithmetic),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                ANALYTICAL_STRUCTURAL_BACKEND_FAMILY,
                SetupTraceLaneKind::Analytical,
            )
            .with_candidate_key(ANALYTICAL_STRATEGY)
            .with_cache_key(prepared_descriptor_id(
                "tla.analytical.proof_cache",
                semantic_digest,
            ))
            .with_candidate_identity(candidate_identity)
            .with_lane_identity(ANALYTICAL_STRUCTURAL_BACKEND_FAMILY)
            .with_fingerprint_policy_identity("tla_analytical_proof_fingerprint_v1")
            .with_fingerprint_identity(proof_fingerprint_identity.clone()),
        )
        .add_validation_plan(
            PreparedValidationPlanDescriptor::new(
                prepared_descriptor_id("tla.analytical.validation.structural_proof", module_name),
                PreparedValidationKind::StructuralProof,
                ProblemKind::Invariant,
            )
            .with_fingerprint(
                PreparedFingerprintDescriptor::new(
                    "tla_analytical_proof",
                    PreparedFingerprintScheme::CanonicalBytesSha256,
                    ANALYTICAL_CANONICALIZATION_VERSION,
                )
                .with_fingerprint_policy_identity("tla_analytical_proof_fingerprint_v1")
                .with_fingerprint_identity(proof_fingerprint_identity),
            )
            .with_artifact_identity(proof_artifact_identity),
        )
}

/// Render frontend-neutral analytical decision rows from prepared TLA
/// analytical descriptors.
///
/// These rows intentionally report structural eligibility only. They do not
/// claim a published analytical verdict until a proof/certificate fingerprint
/// and replay boundary are available.
#[cfg_attr(not(test), allow(dead_code))]
pub fn analytical_solve_decision_rows_for_prepared_program(
    program: &PreparedCheckerProgram,
    scope: &str,
) -> Vec<String> {
    analytical_solve_decisions_for_prepared_program(program)
        .into_iter()
        .map(|decision| decision.render_evidence_row(scope))
        .collect()
}

fn portfolio_payload_kind_for_frontend_source(
    frontend_source_is_quint: bool,
) -> PreparedProgramPayloadKind {
    if frontend_source_is_quint {
        PreparedProgramPayloadKind::Quint
    } else {
        PreparedProgramPayloadKind::Tla
    }
}

pub fn portfolio_evidence_scope_for_payload_kind(
    payload_kind: PreparedProgramPayloadKind,
) -> &'static str {
    match payload_kind {
        PreparedProgramPayloadKind::Tla => "TLA",
        PreparedProgramPayloadKind::Quint => "Quint",
        PreparedProgramPayloadKind::MccPetri => "MCC",
        PreparedProgramPayloadKind::Aiger => "AIGER",
        PreparedProgramPayloadKind::Btor2 => "BTOR2",
        PreparedProgramPayloadKind::VmtInterchange => "VMT",
        PreparedProgramPayloadKind::AYOnly => "AY",
        PreparedProgramPayloadKind::WitnessReplay => "Replay",
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AnalyticalSolveEvidenceRows {
    decision_rows: Vec<String>,
    validation_receipt_rows: Vec<String>,
}

fn analytical_solve_evidence_for_payload_kind(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
    eligibility: AnalyticalEligibility,
) -> AnalyticalSolveEvidenceRows {
    let program =
        prepared_analytical_portfolio_program(module, checker_modules, config, payload_kind);
    let decisions = match eligibility {
        AnalyticalEligibility::VerifiedExecutionModel
        | AnalyticalEligibility::VerifiedStaticInvariant => {
            verified_analytical_solve_decisions_for_prepared_program(&program, eligibility, None)
        }
        AnalyticalEligibility::StructurallyEligible
        | AnalyticalEligibility::StructurallyIneligible
        | AnalyticalEligibility::NotAssessed => {
            analytical_solve_decisions_for_prepared_program(&program)
        }
    };

    let scope = portfolio_evidence_scope_for_payload_kind(payload_kind);
    AnalyticalSolveEvidenceRows {
        decision_rows: decisions
            .iter()
            .map(|decision| decision.render_evidence_row(scope))
            .collect(),
        validation_receipt_rows: shared_engine_validation_receipt_evidence_rows_for_decisions(
            scope, &decisions,
        ),
    }
}

fn verified_analytical_solve_evidence_for_payload_kind<T: std::fmt::Debug>(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
    eligibility: AnalyticalEligibility,
    certificate: &T,
) -> AnalyticalSolveEvidenceRows {
    let program =
        prepared_analytical_portfolio_program(module, checker_modules, config, payload_kind);
    let artifact_evidence =
        verified_analytical_artifact_evidence(&program, eligibility, certificate);
    let decisions = verified_analytical_solve_decisions_for_prepared_program(
        &program,
        eligibility,
        artifact_evidence.as_ref(),
    );
    let scope = portfolio_evidence_scope_for_payload_kind(payload_kind);
    AnalyticalSolveEvidenceRows {
        decision_rows: decisions
            .iter()
            .map(|decision| decision.render_evidence_row(scope))
            .collect(),
        validation_receipt_rows: shared_engine_validation_receipt_evidence_rows_for_decisions(
            scope, &decisions,
        ),
    }
}

/// Render source-aware shared-engine validation receipt rows attached to analytical decisions.
///
/// The core receipt is intentionally frontend-neutral. This portfolio row keeps
/// that receipt shape but adds source/payload/lane context so a consumer can
/// distinguish a receipt-backed analytical solve from ordinary model-check
/// search or a display-only selected-engine string.
#[must_use]
pub fn shared_engine_validation_receipt_evidence_rows_for_decisions(
    scope: &str,
    decisions: &[AnalyticalSolveDecision],
) -> Vec<String> {
    decisions
        .iter()
        .flat_map(|decision| {
            decision.validation_receipts.iter().map(move |receipt| {
                render_shared_engine_validation_receipt_evidence_row(scope, decision, receipt)
            })
        })
        .collect()
}

fn render_shared_engine_validation_receipt_evidence_row(
    scope: &str,
    decision: &AnalyticalSolveDecision,
    receipt: &ValidationReceipt,
) -> String {
    let publication_blocker = decision
        .publication_blocker_reason()
        .map(AnalyticalSolveDecisionReason::code)
        .unwrap_or("none");
    let failure_reason = receipt.failure_reason.as_deref().unwrap_or("none");
    format!(
        "{} shared_engine_validation_receipt schema={} source_kind={} frontend_kind={} frontend_family={} payload_kind={} receipt_role=analytical_solve receipt_identity={} search_kind=analytical_solve model_check_search=false lane_kind={} lane={} backend_code={} solver_family={} problem={} decision_status={} explicit_state_relation={} prepared_program_identity={} candidate_identity={} lane_identity={} frontend_payload_identity={} artifact_identity={} validation_requirement={} validator_kind={} validation_artifact_kind={} validation_artifact_identity={} digest_algorithm={} digest={} receipt_status={} receipt_validation={} failure_reason={} validation_receipt_readiness={} publication_readiness={} publication_blocker={} fail_closed=true consumable_frontend_families={}",
        scope,
        SHARED_ENGINE_VALIDATION_RECEIPT_SCHEMA,
        decision.source_kind.code(),
        decision.source_kind.frontend_family_code(),
        decision.source_kind.frontend_family_code(),
        decision.payload_kind.code(),
        shared_engine_validation_receipt_identity(receipt),
        decision.lane.code(),
        decision.lane.code(),
        decision.backend.code(),
        decision.solver_family_code(),
        decision.problem.code(),
        decision.status.code(),
        decision.status.explicit_state_relation_code(),
        portfolio_evidence_optional(decision.prepared_program_identity.as_deref()),
        portfolio_evidence_value(&receipt.candidate_identity),
        portfolio_evidence_optional(decision.identities.lane_identity.as_deref()),
        portfolio_evidence_optional(decision.identities.frontend_payload_identity.as_deref()),
        portfolio_evidence_optional(decision.identities.artifact_identity.as_deref()),
        validation_requirements_value_for_portfolio(&decision.validation_requirements),
        receipt.validator_kind.code(),
        receipt.validation_artifact_kind.code(),
        portfolio_evidence_value(&receipt.validation_artifact_identity),
        portfolio_evidence_value(&receipt.digest_algorithm),
        portfolio_evidence_value(&receipt.digest),
        receipt.status.code(),
        if receipt.validate().is_ok() { "valid" } else { "invalid" },
        portfolio_evidence_value(failure_reason),
        decision.validation_receipt_readiness_code(),
        decision.publication_readiness_code(),
        publication_blocker,
        shared_engine_validation_receipt_frontend_families(),
    )
}

fn shared_engine_validation_receipt_identity(receipt: &ValidationReceipt) -> String {
    let mut hasher = StableEvidenceHasher::default();
    hasher.field("schema", SHARED_ENGINE_VALIDATION_RECEIPT_SCHEMA);
    hasher.field("validator_kind", receipt.validator_kind.code());
    hasher.field("artifact_kind", receipt.validation_artifact_kind.code());
    hasher.field("digest_algorithm", &receipt.digest_algorithm);
    hasher.field("digest", &receipt.digest);
    hasher.field(
        "prepared_program_identity",
        &receipt.prepared_program_identity,
    );
    hasher.field("candidate_identity", &receipt.candidate_identity);
    hasher.field(
        "validation_artifact_identity",
        &receipt.validation_artifact_identity,
    );
    hasher.field("status", receipt.status.code());
    hasher.optional("failure_reason", receipt.failure_reason.as_deref());
    format!("shared_engine.validation_receipt:{}", hasher.finish_hex())
}

fn shared_engine_validation_receipt_frontend_families() -> String {
    [
        SharedEngineFrontendFamily::TlaPlus,
        SharedEngineFrontendFamily::Quint,
        SharedEngineFrontendFamily::MccPetri,
        SharedEngineFrontendFamily::Aiger,
        SharedEngineFrontendFamily::Btor2,
        SharedEngineFrontendFamily::VmtTransitionSystem,
        SharedEngineFrontendFamily::AYAnalytical,
        SharedEngineFrontendFamily::WitnessReplay,
    ]
    .into_iter()
    .map(SharedEngineFrontendFamily::code)
    .collect::<Vec<_>>()
    .join(",")
}

fn validation_requirements_value_for_portfolio(requirements: &[PreparedValidationKind]) -> String {
    if requirements.is_empty() {
        "none".to_string()
    } else {
        requirements
            .iter()
            .map(|requirement| requirement.code())
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn portfolio_evidence_optional(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .map(portfolio_evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn portfolio_evidence_value(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        return "none".to_string();
    }
    value
        .chars()
        .map(|ch| if ch.is_whitespace() { '_' } else { ch })
        .collect()
}

/// Convert a AY proof-lane receipt into the shared validation receipt contract.
///
/// Validator-backed AY receipts become accepted shared receipts. Artifact-only
/// or missing AY receipts are retained as rejected shared receipts so analytical
/// publication fails closed with a machine-readable reason.
#[cfg(feature = "ay")]
#[must_use]
pub fn shared_validation_receipt_from_ay_proof_receipt(
    prepared_program_identity: impl Into<String>,
    candidate_identity: impl Into<String>,
    receipt: &tla_ay::AYProofValidationReceipt,
) -> ValidationReceipt {
    let prepared_program_identity = prepared_program_identity.into();
    let candidate_identity = candidate_identity.into();
    let validator = match receipt.validation_kind {
        tla_ay::AYProofValidationReceiptKind::OutputFormat => {
            ValidationReceiptValidatorKind::OutputFormat
        }
        tla_ay::AYProofValidationReceiptKind::Model
        | tla_ay::AYProofValidationReceiptKind::Certificate
        | tla_ay::AYProofValidationReceiptKind::Witness
        | tla_ay::AYProofValidationReceiptKind::ProofTranscript => {
            ValidationReceiptValidatorKind::AYProof
        }
    };
    let artifact_kind = match receipt.validation_kind {
        tla_ay::AYProofValidationReceiptKind::OutputFormat => {
            ValidationReceiptArtifactKind::Artifact
        }
        tla_ay::AYProofValidationReceiptKind::Model
        | tla_ay::AYProofValidationReceiptKind::Witness => ValidationReceiptArtifactKind::Witness,
        tla_ay::AYProofValidationReceiptKind::Certificate => {
            ValidationReceiptArtifactKind::Certificate
        }
        tla_ay::AYProofValidationReceiptKind::ProofTranscript => {
            ValidationReceiptArtifactKind::Proof
        }
    };

    let digest_algorithm = AY_SHARED_VALIDATION_DIGEST_ALGORITHM;
    let digest = receipt.validated_fingerprint_identity.clone();
    let validation_artifact_identity = receipt.validated_fingerprint_identity.clone();

    if receipt.status == tla_ay::AYProofValidationReceiptStatus::ValidatorBacked {
        ValidationReceipt::accepted(
            validator,
            digest_algorithm,
            digest,
            prepared_program_identity,
            candidate_identity,
            artifact_kind,
            validation_artifact_identity,
        )
    } else {
        ValidationReceipt::rejected(
            validator,
            digest_algorithm,
            digest,
            prepared_program_identity,
            candidate_identity,
            artifact_kind,
            validation_artifact_identity,
            format!("ay_validation_receipt_status_{}", receipt.status.code()),
        )
    }
}

/// Build a receipt-backed AY analytical solve decision from prepared descriptors.
///
/// This helper is used by frontends that want to publish a AY analytical win
/// through the shared receipt contract. A missing, rejected, or malformed
/// receipt remains attached to the decision and blocks publication.
#[cfg(feature = "ay")]
#[must_use]
pub fn ay_analytical_solve_decision_with_shared_validation_receipt(
    program: &PreparedCheckerProgram,
    lane: tla_ay::AYSharedEngineLane,
    status: AnalyticalSolveDecisionStatus,
    receipt: Option<ValidationReceipt>,
) -> Option<AnalyticalSolveDecision> {
    let solve_kind = crate::ay_shared::ay_shared_engine_prepared_solve_kind(lane)?;
    let solve = program
        .analytical_solves
        .iter()
        .find(|solve| solve.kind == solve_kind)?;
    let admission = crate::ay_shared::ay_shared_engine_lane_admission(program, lane);
    let candidate_lane = admission.prepared_candidate_lane(program);
    let semantic_digest = analytical_semantic_digest(program)?;
    let receipt_missing = receipt.is_none();

    let lifecycle = match admission.status {
        crate::ay_shared::AYSharedEngineAdmissionStatus::Admitted => {
            AnalyticalSolvePortfolioLifecycle::Published
        }
        crate::ay_shared::AYSharedEngineAdmissionStatus::Delayed => {
            AnalyticalSolvePortfolioLifecycle::Candidate
        }
        crate::ay_shared::AYSharedEngineAdmissionStatus::Blocked => {
            AnalyticalSolvePortfolioLifecycle::Rejected
        }
    };

    let mut decision = AnalyticalSolveDecision::new(
        status,
        program.source_kind,
        program.payload_kind,
        solve.problem,
    )
    .with_backend(crate::ay_shared::ay_shared_engine_backend(lane))
    .with_prepared_program_identity(program.identity.clone())
    .with_candidate_key(admission.candidate_key())
    .with_semantic_digest(semantic_digest)
    .with_validation_requirements([PreparedValidationKind::AYProof])
    .with_admission_fail_closed(true)
    .with_cache_fingerprint_compatibility(admission.cache_fingerprint_compatibility_code())
    .with_portfolio_lifecycle(lifecycle)
    .with_portfolio_rank(admission.portfolio_rank())
    .with_portfolio_candidate_id(solve.id.clone())
    .with_reason_code(admission.reason_code());

    if let Some(candidate_lane) = candidate_lane {
        decision.lane = candidate_lane.lane;
        decision = decision.with_prepared_candidate(program, candidate_lane);
        if receipt_missing {
            if let Some(fingerprint_identity) =
                ay_expected_candidate_fingerprint_identity(program, Some(candidate_lane))
            {
                decision = decision.with_proof_fingerprint(fingerprint_identity);
            }
        }
    }
    if let Some(receipt) = receipt {
        let expected_candidate_identity = candidate_lane
            .and_then(|lane| lane.identities.candidate_identity.as_deref())
            .unwrap_or_else(|| admission.candidate_key());
        let receipt = ay_validation_receipt_checked_for_prepared_route(
            receipt,
            program,
            candidate_lane,
            expected_candidate_identity,
            lane,
            status,
        );
        decision = decision.with_validation_receipt(receipt);
    }

    Some(decision)
}

#[cfg(feature = "ay")]
fn ay_validation_receipt_checked_for_prepared_route(
    receipt: ValidationReceipt,
    program: &PreparedCheckerProgram,
    candidate_lane: Option<&PreparedCandidateLaneDescriptor>,
    expected_candidate_identity: &str,
    lane: tla_ay::AYSharedEngineLane,
    status: AnalyticalSolveDecisionStatus,
) -> ValidationReceipt {
    if let Some(reason) = ay_validation_receipt_prepared_route_failure_reason(
        &receipt,
        program,
        candidate_lane,
        expected_candidate_identity,
        lane,
        status,
    ) {
        return ay_rejected_validation_receipt_for_prepared_route(
            receipt,
            program,
            candidate_lane,
            expected_candidate_identity,
            reason,
        );
    }

    if let Err(error) = receipt.validate() {
        let error_code = ay_failure_reason_token(error.to_string());
        return ay_rejected_validation_receipt_for_prepared_route(
            receipt,
            program,
            candidate_lane,
            expected_candidate_identity,
            format!(
                "invalid_validation_receipt:expected_prepared_program_identity={}:expected_candidate_identity={}:error={}",
                program.identity, expected_candidate_identity, error_code
            ),
        );
    }

    receipt
}

#[cfg(feature = "ay")]
fn ay_rejected_validation_receipt_for_prepared_route(
    receipt: ValidationReceipt,
    program: &PreparedCheckerProgram,
    candidate_lane: Option<&PreparedCandidateLaneDescriptor>,
    expected_candidate_identity: &str,
    failure_reason: impl Into<String>,
) -> ValidationReceipt {
    let expected_artifact_identity =
        ay_expected_candidate_fingerprint_identity(program, candidate_lane).unwrap_or_else(|| {
            format!("ay.shared_engine.validation_artifact:{expected_candidate_identity}")
        });
    let digest_algorithm = ay_receipt_identity_or(
        receipt.digest_algorithm,
        AY_SHARED_VALIDATION_DIGEST_ALGORITHM,
    );
    let digest = ay_receipt_identity_or(receipt.digest, &expected_artifact_identity);
    let prepared_program_identity =
        ay_receipt_identity_or(receipt.prepared_program_identity, &program.identity);
    let candidate_identity =
        ay_receipt_identity_or(receipt.candidate_identity, expected_candidate_identity);
    let validation_artifact_identity = ay_receipt_identity_or(
        receipt.validation_artifact_identity,
        &expected_artifact_identity,
    );

    ValidationReceipt::rejected(
        receipt.validator_kind,
        digest_algorithm,
        digest,
        prepared_program_identity,
        candidate_identity,
        receipt.validation_artifact_kind,
        validation_artifact_identity,
        failure_reason,
    )
}

#[cfg(feature = "ay")]
fn ay_expected_candidate_fingerprint_identity(
    program: &PreparedCheckerProgram,
    candidate_lane: Option<&PreparedCandidateLaneDescriptor>,
) -> Option<String> {
    let identities = candidate_lane
        .map(|lane| program.effective_candidate_lane_identity_fields(lane))
        .unwrap_or_else(|| program.effective_identity_fields());
    identities
        .fingerprint_identity
        .filter(|identity| identity.trim() != "none" && !identity.trim().is_empty())
}

#[cfg(feature = "ay")]
fn ay_receipt_identity_or(value: String, fallback: impl AsRef<str>) -> String {
    if ay_has_identity(Some(value.as_str())) {
        value
    } else {
        fallback.as_ref().to_string()
    }
}

#[cfg(feature = "ay")]
fn ay_validation_receipt_prepared_route_failure_reason(
    receipt: &ValidationReceipt,
    program: &PreparedCheckerProgram,
    candidate_lane: Option<&PreparedCandidateLaneDescriptor>,
    expected_candidate_identity: &str,
    lane: tla_ay::AYSharedEngineLane,
    status: AnalyticalSolveDecisionStatus,
) -> Option<String> {
    let identities = candidate_lane
        .map(|lane| program.effective_candidate_lane_identity_fields(lane))
        .unwrap_or_else(|| program.effective_identity_fields());
    let expected_fingerprint_identity = identities.fingerprint_identity.as_deref();
    let mut failures = Vec::new();

    if receipt.prepared_program_identity != program.identity {
        failures.push(format!(
            "prepared_program_identity_expected={}_actual={}",
            program.identity, receipt.prepared_program_identity
        ));
    }
    if receipt.candidate_identity != expected_candidate_identity {
        failures.push(format!(
            "candidate_identity_expected={}_actual={}",
            expected_candidate_identity, receipt.candidate_identity
        ));
    }
    if !ay_has_identity(identities.frontend_payload_identity.as_deref()) {
        failures.push("missing_frontend_payload_identity".to_string());
    }
    if !ay_has_identity(identities.storage_layout_fingerprint.as_deref()) {
        failures.push("missing_storage_layout_fingerprint".to_string());
    }
    if !ay_has_identity(identities.fingerprint_policy_identity.as_deref()) {
        failures.push("missing_fingerprint_policy_identity".to_string());
    }
    if !ay_has_identity(expected_fingerprint_identity) {
        failures.push("missing_fingerprint_identity".to_string());
    }
    if let Some(expected_fingerprint_identity) = expected_fingerprint_identity {
        if receipt.validation_artifact_identity != expected_fingerprint_identity {
            failures.push(format!(
                "validation_artifact_identity_expected={}_actual={}",
                expected_fingerprint_identity, receipt.validation_artifact_identity
            ));
        }
        if receipt.digest_algorithm != AY_SHARED_VALIDATION_DIGEST_ALGORITHM {
            failures.push(format!(
                "digest_algorithm_expected={}_actual={}",
                AY_SHARED_VALIDATION_DIGEST_ALGORITHM, receipt.digest_algorithm
            ));
        }
        if receipt.digest != expected_fingerprint_identity {
            failures.push(format!(
                "digest_identity_expected={}_actual={}",
                expected_fingerprint_identity, receipt.digest
            ));
        }
    }
    let expected_artifact_kind = ay_expected_validation_artifact_kind(lane, status);
    if receipt.validation_artifact_kind != expected_artifact_kind {
        failures.push(format!(
            "validation_artifact_kind_expected={}_actual={}",
            expected_artifact_kind.code(),
            receipt.validation_artifact_kind.code()
        ));
    }
    if receipt.validator_kind != ValidationReceiptValidatorKind::AYProof {
        failures.push(format!(
            "validator_kind_expected={}_actual={}",
            ValidationReceiptValidatorKind::AYProof.code(),
            receipt.validator_kind.code()
        ));
    }

    if failures.is_empty() {
        None
    } else {
        Some(format!("identity_mismatch:{}", failures.join(";")))
    }
}

#[cfg(feature = "ay")]
fn ay_expected_validation_artifact_kind(
    lane: tla_ay::AYSharedEngineLane,
    status: AnalyticalSolveDecisionStatus,
) -> ValidationReceiptArtifactKind {
    match status {
        AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay => match lane {
            tla_ay::AYSharedEngineLane::Pdr | tla_ay::AYSharedEngineLane::Chc => {
                ValidationReceiptArtifactKind::Proof
            }
            tla_ay::AYSharedEngineLane::AllSatEnumeration
            | tla_ay::AYSharedEngineLane::Bmc
            | tla_ay::AYSharedEngineLane::KInduction => ValidationReceiptArtifactKind::Witness,
        },
        AnalyticalSolveDecisionStatus::VerifiedExecutionModel
        | AnalyticalSolveDecisionStatus::VerifiedStaticInvariant => match lane {
            tla_ay::AYSharedEngineLane::Bmc | tla_ay::AYSharedEngineLane::AllSatEnumeration => {
                ValidationReceiptArtifactKind::Witness
            }
            tla_ay::AYSharedEngineLane::Chc
            | tla_ay::AYSharedEngineLane::Pdr
            | tla_ay::AYSharedEngineLane::KInduction => ValidationReceiptArtifactKind::Proof,
        },
        AnalyticalSolveDecisionStatus::StructurallyEligible
        | AnalyticalSolveDecisionStatus::StructurallyIneligible
        | AnalyticalSolveDecisionStatus::NotAssessed => ValidationReceiptArtifactKind::Proof,
    }
}

#[cfg(feature = "ay")]
fn ay_has_identity(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|value| !value.is_empty() && value != "none")
}

#[cfg(feature = "ay")]
fn ay_failure_reason_token(value: impl AsRef<str>) -> String {
    value
        .as_ref()
        .chars()
        .map(|ch| if ch.is_whitespace() { '_' } else { ch })
        .collect()
}

#[cfg(feature = "ay")]
fn ay_runtime_analytical_solve_evidence_for_payload_kind(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
    pdr_run: Option<&crate::ay_pdr::PdrRunResult>,
    bmc_run: Option<&crate::ay_bmc::BmcRunResult>,
    kinduction_result: Option<&crate::ay_kinduction::KInductionResult>,
) -> AnalyticalSolveEvidenceRows {
    let program =
        prepared_analytical_portfolio_program(module, checker_modules, config, payload_kind);
    let decisions =
        ay_runtime_analytical_solve_decisions(&program, pdr_run, bmc_run, kinduction_result);
    let scope = portfolio_evidence_scope_for_payload_kind(payload_kind);
    AnalyticalSolveEvidenceRows {
        decision_rows: decisions
            .iter()
            .map(|decision| decision.render_evidence_row(scope))
            .collect(),
        validation_receipt_rows: shared_engine_validation_receipt_evidence_rows_for_decisions(
            scope, &decisions,
        ),
    }
}

#[cfg(feature = "ay")]
fn ay_runtime_analytical_solve_decisions(
    program: &PreparedCheckerProgram,
    pdr_run: Option<&crate::ay_pdr::PdrRunResult>,
    bmc_run: Option<&crate::ay_bmc::BmcRunResult>,
    kinduction_result: Option<&crate::ay_kinduction::KInductionResult>,
) -> Vec<AnalyticalSolveDecision> {
    let mut decisions = Vec::new();

    if let Some(run) = pdr_run {
        if let Some(status) = ay_pdr_analytical_status(&run.result) {
            if let Some(decision) = ay_analytical_solve_decision_with_shared_validation_receipt(
                program,
                tla_ay::AYSharedEngineLane::Pdr,
                status,
                ay_shared_validation_receipt_from_pdr_run(program, run),
            ) {
                decisions.push(decision);
            }
        }
    }

    if let Some(run) = bmc_run {
        if let Some(status) = ay_bmc_analytical_status(&run.result) {
            if let Some(decision) = ay_analytical_solve_decision_with_shared_validation_receipt(
                program,
                tla_ay::AYSharedEngineLane::Bmc,
                status,
                ay_shared_validation_receipt_from_bmc_run(program, run),
            ) {
                decisions.push(decision);
            }
        }
    }

    if let Some(result) = kinduction_result {
        if let Some(status) = ay_kinduction_analytical_status(result) {
            if let Some(decision) = ay_analytical_solve_decision_with_shared_validation_receipt(
                program,
                tla_ay::AYSharedEngineLane::KInduction,
                status,
                None,
            ) {
                decisions.push(decision);
            }
        }
    }

    decisions
}

#[cfg(feature = "ay")]
fn ay_pdr_analytical_status(
    result: &crate::ay_pdr::PdrResult,
) -> Option<AnalyticalSolveDecisionStatus> {
    match result {
        crate::ay_pdr::PdrResult::Safe { .. } => {
            Some(AnalyticalSolveDecisionStatus::VerifiedExecutionModel)
        }
        crate::ay_pdr::PdrResult::Unsafe { .. } => {
            Some(AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay)
        }
        crate::ay_pdr::PdrResult::Unknown { .. } => None,
    }
}

#[cfg(feature = "ay")]
fn ay_bmc_analytical_status(
    result: &crate::ay_bmc::BmcResult,
) -> Option<AnalyticalSolveDecisionStatus> {
    match result {
        crate::ay_bmc::BmcResult::Violation { .. } => {
            Some(AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay)
        }
        // A reachable deadlock is a verified counterexample (the witnessing
        // trace ends in a stuck, successor-free state).
        crate::ay_bmc::BmcResult::Deadlock { .. } => {
            Some(AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay)
        }
        crate::ay_bmc::BmcResult::BoundReached { .. }
        | crate::ay_bmc::BmcResult::Unknown { .. } => None,
    }
}

#[cfg(feature = "ay")]
fn ay_shared_validation_receipt_from_bmc_run(
    program: &PreparedCheckerProgram,
    run: &crate::ay_bmc::BmcRunResult,
) -> Option<ValidationReceipt> {
    // Both a safety `Violation` and a reachable `Deadlock` are verified BMC
    // counterexamples (a witnessing trace through `depth`); they must each carry
    // a validator-backed receipt. `ay_bmc_analytical_status` already classifies
    // both as `VerifiedCounterexampleReplay`, so emitting a receipt for only one
    // would fail-closed downgrade a genuine deadlock verdict to Unknown.
    let (crate::ay_bmc::BmcResult::Violation { depth, .. }
    | crate::ay_bmc::BmcResult::Deadlock { depth, .. }) = &run.result
    else {
        return None;
    };
    let lane = tla_ay::AYSharedEngineLane::Bmc;
    let candidate_lane = crate::ay_shared::ay_shared_engine_lane_admission(program, lane)
        .prepared_candidate_lane(program)?;
    let identities = program.effective_candidate_lane_identity_fields(candidate_lane);
    let candidate_identity = identities.candidate_identity.clone()?;
    let fingerprint_identity = identities.fingerprint_identity.clone()?;
    let status = if run.solver_decision_profile.accepts_model_for_tla_boundary() {
        tla_ay::AYProofValidationReceiptStatus::ValidatorBacked
    } else {
        tla_ay::AYProofValidationReceiptStatus::ArtifactOnly
    };
    let ay_receipt = tla_ay::AYProofValidationReceipt::validator_backed(
        ay_bmc_validation_receipt_identity(program, &candidate_identity, run, *depth, status),
        tla_ay::AYProofValidationReceiptKind::Witness,
        ay_proof_obligation_identity(program, lane),
        fingerprint_identity,
    )
    .with_status(status);
    Some(shared_validation_receipt_from_ay_proof_receipt(
        program.identity.clone(),
        candidate_identity,
        &ay_receipt,
    ))
}

#[cfg(feature = "ay")]
fn ay_bmc_validation_receipt_identity(
    program: &PreparedCheckerProgram,
    candidate_identity: &str,
    run: &crate::ay_bmc::BmcRunResult,
    depth: usize,
    status: tla_ay::AYProofValidationReceiptStatus,
) -> String {
    let mut hasher = StableEvidenceHasher::default();
    hasher.field("schema", "tla-check:ay-bmc-validation-receipt:v1");
    hasher.field("prepared_program_identity", &program.identity);
    hasher.field("candidate_identity", candidate_identity);
    hasher.field("receipt_status", status.code());
    hasher.usize("depth", depth);
    hasher.bool(
        "model_consumer_accepted",
        run.solver_decision_profile.accepts_model_for_tla_boundary(),
    );
    hasher.bool("fail_closed", run.solver_decision_profile.fail_closed());
    hasher.field(
        "solver_decision_profile_evidence",
        run.solver_decision_profile.evidence_row(),
    );
    format!(
        "ay.shared_engine.bmc.validation_receipt:{}",
        hasher.finish_hex()
    )
}

#[cfg(feature = "ay")]
fn ay_kinduction_analytical_status(
    result: &crate::ay_kinduction::KInductionResult,
) -> Option<AnalyticalSolveDecisionStatus> {
    match result {
        crate::ay_kinduction::KInductionResult::Proved { .. } => {
            Some(AnalyticalSolveDecisionStatus::VerifiedExecutionModel)
        }
        crate::ay_kinduction::KInductionResult::Counterexample { .. } => {
            Some(AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay)
        }
        crate::ay_kinduction::KInductionResult::Unknown { .. } => None,
    }
}

#[cfg(feature = "ay")]
fn ay_shared_validation_receipt_from_pdr_run(
    program: &PreparedCheckerProgram,
    run: &crate::ay_pdr::PdrRunResult,
) -> Option<ValidationReceipt> {
    let lane = tla_ay::AYSharedEngineLane::Pdr;
    let candidate_lane = crate::ay_shared::ay_shared_engine_lane_admission(program, lane)
        .prepared_candidate_lane(program)?;
    let identities = program.effective_candidate_lane_identity_fields(candidate_lane);
    let candidate_identity = identities.candidate_identity.clone()?;
    let fingerprint_identity = identities.fingerprint_identity.clone()?;
    let boundary = run.proof_replay_boundary();
    let status = if boundary.accepts_proof_for_tla_boundary() {
        tla_ay::AYProofValidationReceiptStatus::ValidatorBacked
    } else if boundary.typed_consumer() || boundary.status() == "Available" {
        tla_ay::AYProofValidationReceiptStatus::ArtifactOnly
    } else {
        tla_ay::AYProofValidationReceiptStatus::Missing
    };
    let ay_receipt = tla_ay::AYProofValidationReceipt::validator_backed(
        ay_pdr_validation_receipt_identity(program, &candidate_identity, run, status),
        tla_ay::AYProofValidationReceiptKind::ProofTranscript,
        ay_proof_obligation_identity(program, lane),
        fingerprint_identity,
    )
    .with_status(status);
    Some(shared_validation_receipt_from_ay_proof_receipt(
        program.identity.clone(),
        candidate_identity,
        &ay_receipt,
    ))
}

#[cfg(feature = "ay")]
fn ay_pdr_validation_receipt_identity(
    program: &PreparedCheckerProgram,
    candidate_identity: &str,
    run: &crate::ay_pdr::PdrRunResult,
    status: tla_ay::AYProofValidationReceiptStatus,
) -> String {
    let boundary = run.proof_replay_boundary();
    let consumer = run.proof_consumer_evidence();
    let mut hasher = StableEvidenceHasher::default();
    hasher.field("schema", "tla-check:ay-pdr-validation-receipt:v1");
    hasher.field("prepared_program_identity", &program.identity);
    hasher.field("candidate_identity", candidate_identity);
    hasher.field("receipt_status", status.code());
    hasher.field("boundary_status_code", boundary.status_code());
    hasher.field("row_status_code", boundary.row_status_code());
    hasher.bool("accepted_as_proof", boundary.accepted_as_proof());
    hasher.bool(
        "trust_full_verifier_admissible",
        boundary.trust_full_verifier_admissible(),
    );
    hasher.field(
        "trust_full_verifier_non_admission_reason",
        boundary.trust_full_verifier_non_admission_reason(),
    );
    hasher.optional(
        "normalized_input_sha256",
        consumer.map(|evidence| evidence.normalized_input_sha256.as_str()),
    );
    hasher.optional(
        "property_sha256",
        consumer.map(|evidence| evidence.property_sha256.as_str()),
    );
    hasher.optional(
        "verification_level_code",
        consumer.map(|evidence| evidence.verification_level_code.as_str()),
    );
    format!(
        "ay.shared_engine.pdr.validation_receipt:{}",
        hasher.finish_hex()
    )
}

#[cfg(feature = "ay")]
fn ay_proof_obligation_identity(
    program: &PreparedCheckerProgram,
    lane: tla_ay::AYSharedEngineLane,
) -> String {
    let proof_kind = crate::ay_shared::ay_shared_engine_symbolic_proof_kind(lane);
    let problem = crate::ay_shared::ay_shared_engine_problem(lane);
    program
        .symbolic_proofs
        .iter()
        .find(|proof| proof.kind == proof_kind && proof.problem == problem)
        .map(|proof| proof.id.clone())
        .unwrap_or_else(|| format!("ay.shared_engine.{}.proof_obligation", lane.code()))
}

#[cfg(feature = "ay")]
fn ay_decisions_have_publishable_shared_validation_receipt(
    decisions: &[AnalyticalSolveDecision],
) -> bool {
    decisions
        .iter()
        .any(|decision| decision.publication_blocker_reason().is_none())
}

#[cfg(feature = "ay")]
fn ay_publish_verdict_with_shared_validation_receipt_decisions(
    shared: &SharedVerdict,
    verdict: Verdict,
    decisions: &[AnalyticalSolveDecision],
) -> bool {
    if ay_decisions_have_publishable_shared_validation_receipt(decisions) {
        shared.publish(verdict)
    } else {
        // Missing, rejected, artifact-only, or identity-mismatched receipts are
        // explicit fallback, not symbolic winners.
        shared.publish(Verdict::Unknown)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedAnalyticalArtifactEvidence {
    digest: String,
    artifact_identity: String,
}

fn verified_analytical_artifact_evidence<T: std::fmt::Debug>(
    program: &PreparedCheckerProgram,
    eligibility: AnalyticalEligibility,
    certificate: &T,
) -> Option<VerifiedAnalyticalArtifactEvidence> {
    let semantic_digest = analytical_semantic_digest(program)?;
    let mut hasher = StableEvidenceHasher::default();
    hasher.field("schema", "tla-check:analytical-verified-artifact:v1");
    hasher.field("semantic_digest", &semantic_digest);
    hasher.field("eligibility", analytical_eligibility_code(eligibility));
    hasher.field("certificate", &format!("{certificate:?}"));
    Some(VerifiedAnalyticalArtifactEvidence {
        digest: hasher.finish_hex(),
        artifact_identity: prepared_descriptor_id(
            "tla.analytical.verified_certificate",
            &semantic_digest,
        ),
    })
}

fn analytical_eligibility_code(eligibility: AnalyticalEligibility) -> &'static str {
    eligibility.code()
}

fn verified_analytical_solve_decisions_for_prepared_program(
    program: &PreparedCheckerProgram,
    eligibility: AnalyticalEligibility,
    artifact_evidence: Option<&VerifiedAnalyticalArtifactEvidence>,
) -> Vec<AnalyticalSolveDecision> {
    let analytical_lane = program.candidate_lanes.iter().find(|lane| {
        lane.lane == SetupTraceLaneKind::Analytical
            && lane.candidate_key.as_deref() == Some(ANALYTICAL_STRATEGY)
    });
    let Some(semantic_digest) = analytical_semantic_digest(program) else {
        return analytical_solve_decisions_for_prepared_program(program);
    };

    let mut decisions = Vec::new();
    for solve in &program.analytical_solves {
        let Some(status) = verified_status_for_solve(eligibility, solve.kind) else {
            continue;
        };
        let mut decision = AnalyticalSolveDecision::new(
            status,
            program.source_kind,
            program.payload_kind,
            solve.problem,
        )
        .with_backend(BackendKind::LocalSymbolicExecution)
        .with_prepared_program_identity(program.identity.clone())
        .with_candidate_key(ANALYTICAL_STRATEGY)
        .with_semantic_digest(semantic_digest.clone())
        .with_validation_requirements([PreparedValidationKind::StructuralProof])
        .with_admission_fail_closed(true)
        .with_cache_fingerprint_compatibility(ANALYTICAL_CACHE_COMPAT_FRONTEND_LOCAL_ONLY)
        .with_portfolio_lifecycle(AnalyticalSolvePortfolioLifecycle::Published)
        .with_portfolio_candidate_id(solve.id.clone())
        .with_reason_code("analytical_preemption_proof_verified");

        if let Some(lane) = analytical_lane {
            decision = decision.with_prepared_candidate(program, lane);
        }

        let candidate_identity = decision
            .identities
            .candidate_identity
            .as_deref()
            .or(decision.candidate_key.as_deref())
            .unwrap_or(ANALYTICAL_STRATEGY)
            .to_string();
        if let Some(artifact_evidence) = artifact_evidence {
            decision = decision.with_validation_receipt(ValidationReceipt::accepted(
                ValidationReceiptValidatorKind::StructuralProof,
                ANALYTICAL_ARTIFACT_DIGEST_ALGORITHM,
                artifact_evidence.digest.clone(),
                program.identity.clone(),
                candidate_identity,
                ValidationReceiptArtifactKind::Certificate,
                artifact_evidence.artifact_identity.clone(),
            ));
        }

        decisions.push(decision);
    }

    if decisions.is_empty() {
        analytical_solve_decisions_for_prepared_program(program)
    } else {
        decisions
    }
}

fn verified_status_for_solve(
    eligibility: AnalyticalEligibility,
    solve_kind: PreparedAnalyticalSolveKind,
) -> Option<AnalyticalSolveDecisionStatus> {
    match (eligibility, solve_kind) {
        (
            AnalyticalEligibility::VerifiedExecutionModel,
            PreparedAnalyticalSolveKind::LinearInvariant
            | PreparedAnalyticalSolveKind::StateSpaceCardinality
            | PreparedAnalyticalSolveKind::DeadlockFreedom,
        ) => Some(AnalyticalSolveDecisionStatus::VerifiedExecutionModel),
        (
            AnalyticalEligibility::VerifiedStaticInvariant,
            PreparedAnalyticalSolveKind::UpperBounds,
        ) => Some(AnalyticalSolveDecisionStatus::VerifiedStaticInvariant),
        _ => None,
    }
}

fn analytical_semantic_digest(program: &PreparedCheckerProgram) -> Option<String> {
    program
        .identities
        .prepared_program_fingerprint
        .as_ref()
        .or(program.identities.source_fingerprint.as_ref())
        .cloned()
}

pub fn analytical_solve_decisions_for_prepared_program(
    program: &PreparedCheckerProgram,
) -> Vec<AnalyticalSolveDecision> {
    let analytical_lane = program.candidate_lanes.iter().find(|lane| {
        lane.lane == SetupTraceLaneKind::Analytical
            && lane.candidate_key.as_deref() == Some(ANALYTICAL_STRATEGY)
    });

    program
        .analytical_solves
        .iter()
        .map(|solve| {
            #[cfg(feature = "ay")]
            if let Some(ay_lane) =
                crate::ay_shared::ay_shared_engine_lane_for_prepared_solve_kind(solve.kind)
            {
                let admission = crate::ay_shared::ay_shared_engine_lane_admission(program, ay_lane);
                let lifecycle = match admission.status {
                    crate::ay_shared::AYSharedEngineAdmissionStatus::Admitted => {
                        tla_mc_core::AnalyticalSolvePortfolioLifecycle::Admitted
                    }
                    crate::ay_shared::AYSharedEngineAdmissionStatus::Delayed => {
                        tla_mc_core::AnalyticalSolvePortfolioLifecycle::Candidate
                    }
                    crate::ay_shared::AYSharedEngineAdmissionStatus::Blocked => {
                        tla_mc_core::AnalyticalSolvePortfolioLifecycle::Rejected
                    }
                };
                let decision_reason = match admission.status {
                    crate::ay_shared::AYSharedEngineAdmissionStatus::Admitted => {
                        tla_mc_core::AnalyticalSolveDecisionReason::StructuralProofOnly
                    }
                    crate::ay_shared::AYSharedEngineAdmissionStatus::Delayed => {
                        tla_mc_core::AnalyticalSolveDecisionReason::PortfolioLifecycleBlocked
                    }
                    crate::ay_shared::AYSharedEngineAdmissionStatus::Blocked => {
                        tla_mc_core::AnalyticalSolveDecisionReason::UnsupportedFragment
                    }
                };
                let lane = admission.prepared_candidate_lane(program);
                let mut decision =
                    AnalyticalSolveDecision::from_prepared_solve(program, solve, lane)
                        .with_candidate_key(admission.candidate_key())
                        .with_backend(crate::ay_shared::ay_shared_engine_backend(ay_lane))
                        .with_validation_requirements([PreparedValidationKind::AYProof])
                        .with_admission_fail_closed(true)
                        .with_portfolio_lifecycle(lifecycle)
                        .with_portfolio_rank(admission.portfolio_rank())
                        .with_decision_reason(decision_reason)
                        .with_reason_code(admission.reason_code());
                if admission.status == crate::ay_shared::AYSharedEngineAdmissionStatus::Admitted {
                    decision = decision.with_cache_fingerprint_compatibility(
                        ANALYTICAL_CACHE_COMPAT_FRONTEND_REUSABLE,
                    );
                }
                return decision;
            }

            let lane = analytical_lane;

            AnalyticalSolveDecision::from_prepared_solve(program, solve, lane)
                .with_candidate_key(ANALYTICAL_STRATEGY)
                .with_admission_fail_closed(true)
                .with_cache_fingerprint_compatibility(ANALYTICAL_CACHE_COMPAT_FRONTEND_LOCAL_ONLY)
        })
        .collect()
}

fn analytical_requested(strategy_filter: &[String]) -> bool {
    strategy_filter.is_empty() || strategy_filter.iter().any(|s| s == ANALYTICAL_STRATEGY)
}

#[cfg(feature = "ay")]
fn ay_shared_engine_config_supported(config: &Config) -> bool {
    config.init.is_some() && config.next.is_some() && !config.invariants.is_empty()
}

#[cfg(feature = "ay")]
fn ay_shared_engine_evidence_rows_for_payload_kind(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
) -> Vec<String> {
    let program =
        prepared_analytical_portfolio_program(module, checker_modules, config, payload_kind);
    crate::ay_shared::ay_shared_engine_metadata_and_admission_evidence_rows(
        portfolio_evidence_scope_for_payload_kind(payload_kind),
        &program,
    )
}

fn prepared_analytical_semantic_digest(
    context: &BoundAnalyticalContext,
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
) -> String {
    let configured_names = context.configured_names();
    let mut hasher = StableEvidenceHasher::default();
    hasher.field("schema", ANALYTICAL_SEMANTIC_DIGEST_VERSION);
    hasher.field("payload_kind", payload_kind.code());

    let root_module = context.root_module();
    hasher.field("root_module", root_module.name());
    hasher.field(
        "root_digest",
        &root_module.source_independent_digest().to_string(),
    );
    hasher.usize("checker_module_count", context.checker_module_count());
    for checker_module in context.checker_modules() {
        hasher.field("checker_module", checker_module.name());
        hasher.field(
            "checker_digest",
            &checker_module.source_independent_digest().to_string(),
        );
    }

    hasher.optional("init", configured_names.init());
    hasher.optional("next", configured_names.next());
    hasher.string_slice("invariants", configured_names.invariants());
    hasher.string_slice("trace_invariants", configured_names.trace_invariants());
    hasher.string_slice("properties", configured_names.properties());
    hasher.string_slice("constraints", configured_names.constraints());
    hasher.string_slice("action_constraints", configured_names.action_constraints());
    hasher.optional("specification", configured_names.specification());
    hasher.bool("check_deadlock", config.check_deadlock);
    hasher.bool("check_deadlock_explicit", config.check_deadlock_explicit);
    hasher.optional("symmetry", config.symmetry.as_deref());
    hasher.optional("view", config.view.as_deref());
    hasher.optional("postcondition", config.postcondition.as_deref());
    hasher.optional("alias", config.alias.as_deref());
    hasher.field("terminal", &format!("{:?}", config.terminal));
    hasher.field("init_mode", &format!("{:?}", config.init_mode));
    hasher.bool("por_enabled", config.por_enabled);
    hasher.field("auto_por", &format!("{:?}", config.auto_por));

    let mut constant_names = config.constants.keys().collect::<Vec<_>>();
    constant_names.sort();
    hasher.usize("constants_len", constant_names.len());
    for name in constant_names {
        hasher.field("constant_name", name);
        if let Some(value) = config.constants.get(name) {
            hasher.field("constant_value", &value.to_string());
        }
    }
    hasher.string_slice("constants_order", &config.constants_order);
    hash_nested_string_map(&mut hasher, "module_overrides", &config.module_overrides);
    hash_nested_string_map(
        &mut hasher,
        "module_assignments",
        &config.module_assignments,
    );

    hasher.finish_hex()
}

fn hash_nested_string_map(
    hasher: &mut StableEvidenceHasher,
    label: &str,
    map: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) {
    let mut module_names = map.keys().collect::<Vec<_>>();
    module_names.sort();
    hasher.usize(label, module_names.len());
    for module_name in module_names {
        hasher.field("map_module", module_name);
        if let Some(entries) = map.get(module_name) {
            let mut names = entries.keys().collect::<Vec<_>>();
            names.sort();
            hasher.usize("map_entries", names.len());
            for name in names {
                hasher.field("map_key", name);
                if let Some(value) = entries.get(name) {
                    hasher.field("map_value", value);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct StableEvidenceHasher {
    hash: u64,
}

impl Default for StableEvidenceHasher {
    fn default() -> Self {
        Self {
            hash: FNV_OFFSET_BASIS,
        }
    }
}

impl StableEvidenceHasher {
    fn field(&mut self, name: &str, value: &str) {
        self.str(name);
        self.str(value);
    }

    fn optional(&mut self, name: &str, value: Option<&str>) {
        self.str(name);
        match value {
            Some(value) => {
                self.str("some");
                self.str(value);
            }
            None => self.str("none"),
        }
    }

    fn string_slice(&mut self, name: &str, values: &[String]) {
        self.str(name);
        self.usize("len", values.len());
        for value in values {
            self.str(value);
        }
    }

    fn bool(&mut self, name: &str, value: bool) {
        self.field(name, if value { "true" } else { "false" });
    }

    fn usize(&mut self, name: &str, value: usize) {
        self.field(name, &value.to_string());
    }

    fn str(&mut self, value: &str) {
        for byte in value.as_bytes() {
            self.byte(*byte);
        }
        self.byte(0xff);
    }

    fn byte(&mut self, byte: u8) {
        self.hash ^= u64::from(byte);
        self.hash = self.hash.wrapping_mul(FNV_PRIME);
    }

    fn finish_hex(self) -> String {
        format!("{:016x}", self.hash)
    }
}

fn add_prepared_properties(
    mut program: PreparedCheckerProgram,
    prefix: &str,
    names: &[String],
    kind: PreparedPropertyKind,
) -> PreparedCheckerProgram {
    for name in names {
        program = program.add_property(prepared_descriptor_id(prefix, name), kind);
    }
    program
}

fn prepared_descriptor_id(prefix: &str, name: &str) -> String {
    if name.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}:{name}")
    }
}

fn should_run_strategy(strategy_filter: &[String], name: &str) -> bool {
    strategy_filter.is_empty()
        || strategy_filter.iter().any(|s| s == name)
        || (name == "bfs" && analytical_requested(strategy_filter))
}

fn try_run_analytical_execution_model(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
) -> Option<PortfolioResult> {
    let verified = prove_configured_interval_counter_execution_model(module, config)?;
    let state_count = verified.certificate().state_count().to_usize()?;
    let certificate = verified.certificate().clone();

    let shared = SharedVerdict::new();
    let gate = VerificationGate::new(&shared);
    if !gate.publish_verified_proof(verified).published() {
        return None;
    }

    let stats = CheckStats {
        states_found: state_count,
        initial_states: state_count,
        ..Default::default()
    };
    let analytical_evidence = verified_analytical_solve_evidence_for_payload_kind(
        module,
        checker_modules,
        config,
        payload_kind,
        AnalyticalEligibility::VerifiedExecutionModel,
        &certificate,
    );

    Some(PortfolioResult {
        winner: PortfolioWinner::Analytical,
        analytical_eligibility: AnalyticalEligibility::VerifiedExecutionModel,
        analytical_solve_evidence: analytical_evidence.decision_rows,
        shared_engine_validation_receipts: analytical_evidence.validation_receipt_rows,
        bfs_result: CheckResult::Success(stats),
        random_result: Some(RandomWalkResult::NoViolationFound {
            walks_completed: 0,
            total_steps: 0,
        }),
        #[cfg(feature = "ay")]
        pdr_result: None,
        #[cfg(feature = "ay")]
        pdr_proof_replay_evidence: None,
        #[cfg(feature = "ay")]
        ay_shared_engine_evidence: ay_shared_engine_evidence_rows_for_payload_kind(
            module,
            checker_modules,
            config,
            payload_kind,
        ),
        #[cfg(feature = "ay")]
        bmc_result: None,
        #[cfg(feature = "ay")]
        kinduction_result: None,
    })
}

fn try_run_analytical_static_invariant_proof(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
    payload_kind: PreparedProgramPayloadKind,
) -> Option<PortfolioResult> {
    if !finite_cardinality_config_obligations_supported(config)
        || !analytical_runtime_config_ops_supported(module, checker_modules, config)
    {
        return None;
    }

    let verified = prove_configured_finite_cardinality_invariants(module, checker_modules, config)?;
    let certificate = verified.certificate().clone();

    let shared = SharedVerdict::new();
    let gate = VerificationGate::new(&shared);
    if !gate.publish_verified_proof(verified).published() {
        return None;
    }
    let analytical_evidence = verified_analytical_solve_evidence_for_payload_kind(
        module,
        checker_modules,
        config,
        payload_kind,
        AnalyticalEligibility::VerifiedStaticInvariant,
        &certificate,
    );

    Some(PortfolioResult {
        winner: PortfolioWinner::Analytical,
        analytical_eligibility: AnalyticalEligibility::VerifiedStaticInvariant,
        analytical_solve_evidence: analytical_evidence.decision_rows,
        shared_engine_validation_receipts: analytical_evidence.validation_receipt_rows,
        bfs_result: CheckResult::Success(CheckStats::default()),
        random_result: Some(RandomWalkResult::NoViolationFound {
            walks_completed: 0,
            total_steps: 0,
        }),
        #[cfg(feature = "ay")]
        pdr_result: None,
        #[cfg(feature = "ay")]
        pdr_proof_replay_evidence: None,
        #[cfg(feature = "ay")]
        ay_shared_engine_evidence: ay_shared_engine_evidence_rows_for_payload_kind(
            module,
            checker_modules,
            config,
            payload_kind,
        ),
        #[cfg(feature = "ay")]
        bmc_result: None,
        #[cfg(feature = "ay")]
        kinduction_result: None,
    })
}

fn prove_configured_interval_counter_execution_model(
    module: &Module,
    config: &Config,
) -> Option<VerifiedProof<IntervalCounterExecutionCertificate>> {
    if !analytical_config_obligations_supported(config) || config.invariants.len() != 1 {
        return None;
    }

    let init = config.init.as_deref()?;
    let next = config.next.as_deref()?;
    let invariant = config.invariants.first()?;
    match admit_module_interval_counter_execution_model(
        module,
        init,
        next,
        invariant,
        config.check_deadlock,
    ) {
        AnalyticalAdmission::VerifiedProof(verified) => Some(verified),
        AnalyticalAdmission::ReplayedViolation(_)
        | AnalyticalAdmission::Unknown(_)
        | AnalyticalAdmission::Ineligible(_) => None,
    }
}

fn prove_configured_finite_cardinality_invariants(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
) -> Option<VerifiedProof<Vec<FiniteCardinalityAdmissionCertificate>>> {
    if config.invariants.is_empty() {
        return None;
    }

    let mut certificates = Vec::with_capacity(config.invariants.len());
    for invariant in &config.invariants {
        match admit_module_set_finite_cardinality_invariant(module, checker_modules, invariant) {
            AnalyticalAdmission::VerifiedProof(verified) => {
                certificates.push(verified.into_certificate());
            }
            AnalyticalAdmission::ReplayedViolation(_)
            | AnalyticalAdmission::Unknown(_)
            | AnalyticalAdmission::Ineligible(_) => return None,
        }
    }

    Some(VerifiedProof::new(certificates))
}

fn analytical_eligibility_for_config(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
) -> AnalyticalEligibility {
    if analytical_config_obligations_supported(config) && config.invariants.len() == 1 {
        if prove_configured_interval_counter_execution_model(module, config).is_some() {
            return AnalyticalEligibility::VerifiedExecutionModel;
        }

        let Some(init) = config.init.as_deref() else {
            return finite_cardinality_eligibility_for_config(module, checker_modules, config);
        };
        let Some(next) = config.next.as_deref() else {
            return finite_cardinality_eligibility_for_config(module, checker_modules, config);
        };
        let Some(invariant) = config.invariants.first() else {
            return finite_cardinality_eligibility_for_config(module, checker_modules, config);
        };

        match admit_module_interval_counters(module, init, next, invariant) {
            AnalyticalAdmission::VerifiedProof(_) => {
                return AnalyticalEligibility::StructurallyEligible;
            }
            AnalyticalAdmission::ReplayedViolation(_)
            | AnalyticalAdmission::Unknown(_)
            | AnalyticalAdmission::Ineligible(_) => {}
        }
    }

    finite_cardinality_eligibility_for_config(module, checker_modules, config)
}

fn analytical_config_obligations_supported(config: &Config) -> bool {
    config.specification.is_none()
        && config.properties.is_empty()
        && config.trace_invariants.is_empty()
        && config.constraints.is_empty()
        && config.action_constraints.is_empty()
        && config.postcondition.is_none()
        && config.view.is_none()
        && config.symmetry.is_none()
        && config.terminal.is_none()
}

fn finite_cardinality_eligibility_for_config(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
) -> AnalyticalEligibility {
    if prove_configured_finite_cardinality_invariants(module, checker_modules, config).is_none() {
        return AnalyticalEligibility::StructurallyIneligible;
    }

    if finite_cardinality_config_obligations_supported(config)
        && analytical_runtime_config_ops_supported(module, checker_modules, config)
    {
        AnalyticalEligibility::VerifiedStaticInvariant
    } else {
        AnalyticalEligibility::StructurallyEligible
    }
}

fn finite_cardinality_config_obligations_supported(config: &Config) -> bool {
    analytical_config_obligations_supported(config)
        && !config.check_deadlock
        && !config.invariants.is_empty()
}

fn analytical_runtime_config_ops_supported(
    module: &Module,
    checker_modules: &[&Module],
    config: &Config,
) -> bool {
    if !module_set_declares_variable(module, checker_modules) {
        return false;
    }

    let Some(init) = config.init.as_deref() else {
        return false;
    };
    let Some(next) = config.next.as_deref() else {
        return false;
    };

    if module_set_declares_assumption(module, checker_modules) {
        return false;
    }

    module_set_contains_bool_literal_zero_arity_operator(module, checker_modules, init)
        && module_set_contains_bool_literal_zero_arity_operator(module, checker_modules, next)
}

fn module_set_declares_variable(module: &Module, checker_modules: &[&Module]) -> bool {
    std::iter::once(module)
        .chain(checker_modules.iter().copied())
        .any(module_declares_variable)
}

fn module_declares_variable(module: &Module) -> bool {
    module
        .units
        .iter()
        .any(|unit| matches!(&unit.node, Unit::Variable(names) if !names.is_empty()))
}

fn module_set_declares_assumption(module: &Module, checker_modules: &[&Module]) -> bool {
    std::iter::once(module)
        .chain(checker_modules.iter().copied())
        .any(module_declares_assumption)
}

fn module_declares_assumption(module: &Module) -> bool {
    module
        .units
        .iter()
        .any(|unit| matches!(&unit.node, Unit::Assume(_)))
}

fn module_set_contains_bool_literal_zero_arity_operator(
    module: &Module,
    checker_modules: &[&Module],
    name: &str,
) -> bool {
    if let Some(body) = module_zero_arity_operator_body(module, name) {
        return matches!(body, Expr::Bool(_));
    }

    for checker_module in checker_modules {
        if let Some(body) = module_zero_arity_operator_body(checker_module, name) {
            return matches!(body, Expr::Bool(_));
        }
    }

    false
}

fn module_zero_arity_operator_body<'a>(module: &'a Module, name: &str) -> Option<&'a Expr> {
    module.units.iter().find_map(|unit| match &unit.node {
        Unit::Operator(op) => {
            (op.name.node == name && op.params.is_empty()).then_some(&op.body.node)
        }
        _ => None,
    })
}

/// Determine which lane won the portfolio race by examining the published
/// verdict and correlating it with each lane's result.
fn determine_winner(
    verdict: &SharedVerdict,
    bfs_result: &CheckResult,
    random_result: &Option<RandomWalkResult>,
) -> PortfolioWinner {
    match verdict.get() {
        Some(Verdict::Satisfied) => {
            // BFS publishes Satisfied when all states explored with no violation.
            // PDR and k-induction publish Satisfied when they prove the invariant.
            // Random walk never publishes Satisfied (it's an under-approximation).
            match bfs_result {
                CheckResult::Success(_) => PortfolioWinner::Bfs,
                // Could be PDR or k-induction; default to Pdr since it's
                // the more common symbolic prover.
                _ => PortfolioWinner::Pdr,
            }
        }
        Some(Verdict::Violated) => {
            // Check if random walk found a violation first.
            if let Some(
                RandomWalkResult::InvariantViolation { .. } | RandomWalkResult::Deadlock { .. },
            ) = random_result
            {
                // Random walk may have found it first. Check if BFS also found a violation.
                match bfs_result {
                    CheckResult::InvariantViolation { .. }
                    | CheckResult::PropertyViolation { .. }
                    | CheckResult::LivenessViolation { .. } => {
                        // Both found violations; random walk is faster for shallow bugs.
                        // Heuristic: credit random walk since it has zero memory overhead.
                        PortfolioWinner::Random
                    }
                    _ => PortfolioWinner::Random,
                }
            } else {
                match bfs_result {
                    CheckResult::InvariantViolation { .. }
                    | CheckResult::PropertyViolation { .. }
                    | CheckResult::LivenessViolation { .. } => PortfolioWinner::Bfs,
                    // Could be PDR or BMC; default to Bmc since it's the
                    // primary symbolic bug finder.
                    _ => PortfolioWinner::Bmc,
                }
            }
        }
        _ => PortfolioWinner::Bfs, // fallback: BFS always produces a result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::parse_module;

    fn interval_counter_config(init: &str, next: &str, invariant: &str) -> Config {
        Config {
            init: Some(init.to_string()),
            next: Some(next.to_string()),
            invariants: vec![invariant.to_string()],
            ..Default::default()
        }
    }

    /// Test that the portfolio orchestrator runs and the SharedVerdict
    /// is resolved by at least one lane.
    #[test]
    fn test_portfolio_shared_verdict_resolves() {
        let sv = Arc::new(SharedVerdict::new());
        let sv1 = sv.clone();
        let sv2 = sv.clone();

        std::thread::scope(|scope| {
            // Lane 1: publishes Satisfied immediately.
            scope.spawn(move || {
                sv1.publish(Verdict::Satisfied);
            });

            // Lane 2: checks and exits early.
            scope.spawn(move || {
                // Busy-wait until resolved (in real code, this would be
                // interleaved with state processing every 4096 states).
                while !sv2.is_resolved() {
                    std::thread::yield_now();
                }
            });
        });

        assert!(sv.is_resolved());
        assert_eq!(sv.get(), Some(Verdict::Satisfied));
    }

    /// Test that PortfolioWinner variants are all distinguishable.
    #[test]
    fn test_portfolio_winner_variants() {
        let variants = [
            PortfolioWinner::Analytical,
            PortfolioWinner::Bfs,
            PortfolioWinner::Random,
            PortfolioWinner::Pdr,
            PortfolioWinner::Bmc,
            PortfolioWinner::KInduction,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    /// Test a 5-lane race where all lanes publish concurrently.
    #[test]
    fn test_portfolio_five_lane_race() {
        let sv = Arc::new(SharedVerdict::new());
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let sv = sv.clone();
                std::thread::spawn(move || {
                    let v = if i % 2 == 0 {
                        Verdict::Satisfied
                    } else {
                        Verdict::Violated
                    };
                    sv.publish(v)
                })
            })
            .collect();

        let wins: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // Exactly one thread should win.
        assert_eq!(wins.iter().filter(|&&w| w).count(), 1);
        assert!(sv.is_resolved());
    }

    #[test]
    fn test_analytical_strategy_forces_bfs_verifier() {
        let strategy_filter = vec![ANALYTICAL_STRATEGY.to_string()];
        assert!(should_run_strategy(&strategy_filter, "bfs"));
        assert!(!should_run_strategy(&strategy_filter, "random"));
    }

    #[test]
    fn test_analytical_structural_status_is_not_a_published_verdict() {
        let eligibility = AnalyticalEligibility::StructurallyEligible;
        assert!(
            eligibility.wording().contains("explicit exploration"),
            "structural status must not claim portfolio success"
        );
    }

    #[test]
    fn analytical_execution_model_pre_solve_skips_bfs_for_full_finite_domain_counter() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalFiniteDomain ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let strategy_filter = vec![ANALYTICAL_STRATEGY.to_string()];

        let result = run_portfolio(&module, &[], &config, &strategy_filter);

        assert_eq!(result.winner, PortfolioWinner::Analytical);
        assert_eq!(
            result.analytical_eligibility,
            AnalyticalEligibility::VerifiedExecutionModel
        );
        match result.bfs_result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.initial_states, 3);
                assert_eq!(stats.states_found, 3);
            }
            other => panic!("expected analytical success result, got {other:?}"),
        }
        assert!(result.analytical_solve_evidence.iter().any(|row| {
            row.contains("analytical_solve_decision")
                && row.contains("decision_status=verified_execution_model")
                && row.contains("semantic_digest=")
                && !row.contains("semantic_digest=none")
                && row.contains("cache_fingerprint_compatibility=frontend_local_only")
                && row.contains("admission_fail_closed=true")
                && row.contains("admission_disposition=analytical_preempt")
                && row.contains("replay_validation_authority=structural_proof")
                && row.contains("validation_receipt_readiness=ready")
                && row.contains("publication_readiness=ready")
                && row.contains("publication_blocker=none")
        }));
    }

    #[test]
    fn analytical_prepared_program_describes_interval_counter_candidates() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalPreparedInterval ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");

        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        );

        assert_eq!(program.payload_kind, PreparedProgramPayloadKind::Tla);
        assert_eq!(program.storage_kind, PreparedStorageKind::TlaStateSlots);
        assert!(program.identities.cache_key.is_some());
        assert!(program.identities.source_fingerprint.is_some());
        assert!(program.identities.prepared_program_fingerprint.is_some());
        assert!(program.identities.fingerprint_identity.is_some());
        assert_eq!(program.transitions.len(), 1);
        assert_eq!(program.transitions[0].id, "Next");
        assert!(program
            .properties
            .iter()
            .any(|property| property.id == "invariant:Inv"
                && property.kind == PreparedPropertyKind::Invariant));
        assert!(program.analytical_solves.iter().any(|solve| solve.id
            == "tla.analytical.interval_counter.invariant:Inv"
            && solve.kind == PreparedAnalyticalSolveKind::LinearInvariant
            && solve.problem == ProblemKind::Invariant));
        assert!(program.analytical_solves.iter().any(|solve| solve.id
            == "tla.analytical.interval_counter.state_space:Inv"
            && solve.kind == PreparedAnalyticalSolveKind::StateSpaceCardinality
            && solve.problem == ProblemKind::StateSpace));
        assert!(program.analytical_solves.iter().any(|solve| solve.id
            == "tla.analytical.interval_counter.deadlock:Inv"
            && solve.kind == PreparedAnalyticalSolveKind::DeadlockFreedom
            && solve.problem == ProblemKind::Deadlock));
        assert!(program.symbolic_proofs.iter().any(|proof| proof.id
            == "tla.analytical.interval_counter.proof:Inv"
            && proof.kind == PreparedSymbolicProofKind::InvariantProof
            && proof.problem == ProblemKind::Invariant));
        assert!(program
            .backend_families
            .iter()
            .any(|family| family.id == ANALYTICAL_STRUCTURAL_BACKEND_FAMILY
                && family.backend == BackendKind::LocalSymbolicExecution
                && family.facets.contains(&SolverFacet::Proof)));
        assert!(program
            .validations
            .contains(&PreparedValidationKind::StructuralProof));
        let expected_ay_extra_lanes = if cfg!(feature = "ay") { 5 } else { 0 };
        assert_eq!(program.candidate_lanes.len(), 1 + expected_ay_extra_lanes);
        assert!(program
            .candidate_lanes
            .iter()
            .any(|lane| lane.candidate_key.as_deref() == Some(ANALYTICAL_STRATEGY)));
        assert_eq!(program.validation_plans.len(), 1 + expected_ay_extra_lanes);
        assert_eq!(
            program.validation_plans[0].kind,
            PreparedValidationKind::StructuralProof
        );
        assert_eq!(
            program.canonical_identities.len(),
            1 + usize::from(cfg!(feature = "ay"))
        );

        let row = program.render_evidence_row("TY");
        let expected_analytical_solves = if cfg!(feature = "ay") { 7 } else { 3 };
        let expected_symbolic_proofs = if cfg!(feature = "ay") { 6 } else { 1 };
        let expected_backend_families = if cfg!(feature = "ay") { 6 } else { 1 };
        let expected_candidate_lanes = if cfg!(feature = "ay") { 6 } else { 1 };
        let expected_validation_plans = if cfg!(feature = "ay") { 6 } else { 1 };
        assert!(row.contains("payload_kind=tla"));
        assert!(row.contains(&format!("analytical_solves={expected_analytical_solves}")));
        assert!(row.contains(&format!("symbolic_proofs={expected_symbolic_proofs}")));
        assert!(row.contains(&format!("backend_families={expected_backend_families}")));
        assert!(row.contains(&format!("candidate_lanes={expected_candidate_lanes}")));
        assert!(row.contains(&format!("validation_plans={expected_validation_plans}")));
        let validation_rows = program.render_validation_plan_evidence_rows("TY");
        assert_eq!(validation_rows.len(), expected_validation_plans);
        assert!(validation_rows[0].contains("prepared_validation_plan"));
        assert!(validation_rows[0].contains("validation_kind=structural_proof"));
        assert!(validation_rows[0].contains("fingerprint_id=tla_analytical_proof"));
        assert!(
            validation_rows[0].contains("fingerprint_identity=tla.analytical.proof_fingerprint")
        );

        let decision_rows = analytical_solve_decision_rows_for_prepared_program(&program, "TY");
        assert_eq!(decision_rows.len(), expected_analytical_solves);
        assert!(decision_rows.iter().any(|row| {
            row
            .contains("analytical_solve_decision")
            && row.contains("source_kind=tla")
            && row.contains("payload_kind=tla")
            && row.contains("problem=invariant")
            && row.contains("decision_status=structurally_eligible")
            && row.contains("semantic_digest=")
            && !row.contains("semantic_digest=none")
            && row.contains("cache_fingerprint_compatibility=frontend_local_only")
            && row.contains("admission_fail_closed=true")
            && row.contains("publication_readiness=blocked")
            && row.contains("publication_blocker=structural_proof_only")
            && row.contains("candidate_key=analytical")
            && row.contains(
                "prepared_program_identity=PortfolioAnalyticalPreparedInterval#analytical"
            )
            && row.contains(
                "candidate_identity=tla.analytical.candidate:PortfolioAnalyticalPreparedInterval"
            )
            && row.contains(
                "portfolio_candidate_id=tla.analytical.interval_counter.invariant:Inv"
            )
        }));

        #[cfg(feature = "ay")]
        {
            assert!(program
                .candidate_lanes
                .iter()
                .any(|lane| lane.candidate_key.as_deref() == Some("ay_bmc")));
            assert!(program
                .candidate_lanes
                .iter()
                .any(|lane| lane.candidate_key.as_deref() == Some("ay_all_sat_enumeration")));
            assert!(program
                .analytical_solves
                .iter()
                .any(|solve| solve.id == "ay.shared_engine.bmc.solve"
                    && solve.kind == PreparedAnalyticalSolveKind::BoundedModelCheck
                    && solve.problem == ProblemKind::Bmc));
            assert!(program.analytical_solves.iter().any(|solve| solve.id
                == "ay.shared_engine.all_sat_enumeration.solve"
                && solve.kind == PreparedAnalyticalSolveKind::SmtQuery
                && solve.problem == ProblemKind::SymbolicExecution));
            assert!(program
                .symbolic_proofs
                .iter()
                .any(|proof| proof.id == "ay.shared_engine.chc.proof_obligation"
                    && proof.kind == PreparedSymbolicProofKind::ChcQuery
                    && proof.problem == ProblemKind::Chc));
            assert!(decision_rows.iter().any(|row| {
                row.contains("portfolio_candidate_id=ay.shared_engine.bmc.solve")
                    && row.contains("lane_kind=ay")
                    && row.contains("candidate_key=ay_bmc")
                    && row.contains("backend_code=ay_smt")
                    && row.contains("solver_family=ay")
                    && row.contains("semantic_digest=")
                    && !row.contains("semantic_digest=none")
                    && row.contains("cache_key=ay.shared_engine.cache")
                    && row.contains("cache_fingerprint_compatibility=frontend_reusable")
                    && row.contains("admission_fail_closed=true")
                    && row.contains("validation_requirements=ay_proof")
                    && row.contains("portfolio_rank=20")
                    && row.contains("reason_code=admitted_prerequisites_satisfied")
            }));
            assert!(decision_rows.iter().any(|row| {
                row.contains("portfolio_candidate_id=ay.shared_engine.pdr.solve")
                    && row.contains("candidate_key=ay_pdr")
                    && row.contains("backend_code=ay_chc")
            }));
        }
    }

    #[test]
    fn analytical_prepared_program_semantic_identity_includes_config_obligations() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalConfigIdentity ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
InvA == x \in 0..2
InvB == x \in 0..2
====
"#,
        );
        let config_a = interval_counter_config("Init", "Next", "InvA");
        let config_b = interval_counter_config("Init", "Next", "InvB");

        let program_a = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config_a,
            PreparedProgramPayloadKind::Tla,
        );
        let program_b = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config_b,
            PreparedProgramPayloadKind::Tla,
        );

        assert_ne!(program_a.identity, program_b.identity);
        assert_ne!(
            program_a.identities.prepared_program_fingerprint,
            program_b.identities.prepared_program_fingerprint
        );
        assert_ne!(
            program_a.identities.cache_key,
            program_b.identities.cache_key
        );
        assert!(
            analytical_solve_decision_rows_for_prepared_program(&program_a, "TY")
                .iter()
                .any(|row| row.contains("cache_fingerprint_compatibility=frontend_local_only"))
        );
        assert!(
            analytical_solve_decision_rows_for_prepared_program(&program_b, "TY")
                .iter()
                .any(|row| row.contains("cache_fingerprint_compatibility=frontend_local_only"))
        );
    }

    #[test]
    fn analytical_preemption_fails_closed_without_verified_artifact_receipt() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalMissingArtifactReceipt ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        );

        let mut decisions = verified_analytical_solve_decisions_for_prepared_program(
            &program,
            AnalyticalEligibility::VerifiedExecutionModel,
            None,
        );
        let decision = decisions.remove(0);
        let row = decision.render_evidence_row("TY");

        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert!(row.contains("decision_status=verified_execution_model"));
        assert!(row.contains("validation_receipt_readiness=unknown"));
        assert!(row.contains("publication_blocker=missing_artifact_fingerprint"));
        assert!(row.contains("admission_disposition=fail_closed_explicit_fallback"));
    }

    #[test]
    fn analytical_preemption_fails_closed_for_invalid_cache_compatibility() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalInvalidCacheCompatibility ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        );
        let verified = prove_configured_interval_counter_execution_model(&module, &config)
            .expect("interval counter fixture should produce a verified certificate");
        let artifact_evidence = verified_analytical_artifact_evidence(
            &program,
            AnalyticalEligibility::VerifiedExecutionModel,
            verified.certificate(),
        )
        .expect("prepared program should carry semantic identity");

        let mut decisions = verified_analytical_solve_decisions_for_prepared_program(
            &program,
            AnalyticalEligibility::VerifiedExecutionModel,
            Some(&artifact_evidence),
        );
        let decision = decisions
            .remove(0)
            .with_cache_fingerprint_compatibility("invalid_cross_frontend_mode");
        let row = decision.render_evidence_row("TY");

        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert!(row.contains("decision_status=verified_execution_model"));
        assert!(row.contains("validation_receipt_readiness=ready"));
        assert!(row.contains("publication_blocker=invalid_cache_fingerprint_compatibility"));
        assert!(row.contains("admission_disposition=fail_closed_explicit_fallback"));
    }

    #[cfg(feature = "ay")]
    fn ay_pdr_candidate_identity(program: &PreparedCheckerProgram) -> String {
        ay_candidate_identity(program, "ay_pdr")
    }

    #[cfg(feature = "ay")]
    fn ay_candidate_identity(program: &PreparedCheckerProgram, candidate_key: &str) -> String {
        program
            .candidate_lanes
            .iter()
            .find(|lane| lane.candidate_key.as_deref() == Some(candidate_key))
            .and_then(|lane| lane.identities.candidate_identity.clone())
            .expect("prepared program should include the requested ay candidate identity")
    }

    #[cfg(feature = "ay")]
    fn ay_pdr_fingerprint_identity(program: &PreparedCheckerProgram) -> String {
        ay_candidate_fingerprint_identity(program, "ay_pdr")
    }

    #[cfg(feature = "ay")]
    fn ay_candidate_fingerprint_identity(
        program: &PreparedCheckerProgram,
        candidate_key: &str,
    ) -> String {
        let lane = program
            .candidate_lanes
            .iter()
            .find(|lane| lane.candidate_key.as_deref() == Some(candidate_key))
            .expect("prepared program should include the requested ay candidate lane");
        program
            .effective_candidate_lane_identity_fields(lane)
            .fingerprint_identity
            .expect("prepared ay lane should include a fingerprint identity")
    }

    #[cfg(feature = "ay")]
    #[test]
    fn ay_shared_engine_validation_receipts_preserve_frontend_identity_for_all_payloads() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAYReceiptPayloads ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");

        for &payload_kind in PreparedProgramPayloadKind::shared_engine_payloads() {
            let program =
                prepared_analytical_portfolio_program(&module, &[], &config, payload_kind);
            let candidate_identity = ay_pdr_candidate_identity(&program);
            let fingerprint_identity = ay_pdr_fingerprint_identity(&program);
            let ay_receipt = tla_ay::AYProofValidationReceipt::validator_backed(
                format!("receipt:{}:pdr", payload_kind.code()),
                tla_ay::AYProofValidationReceiptKind::ProofTranscript,
                "proof_obligation:portfolio:ay_pdr",
                fingerprint_identity,
            );
            let shared_receipt = shared_validation_receipt_from_ay_proof_receipt(
                program.identity.clone(),
                candidate_identity,
                &ay_receipt,
            );
            let decision = ay_analytical_solve_decision_with_shared_validation_receipt(
                &program,
                tla_ay::AYSharedEngineLane::Pdr,
                AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
                Some(shared_receipt),
            )
            .expect("prepared ay_pdr solve should produce a shared decision");
            let rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
                portfolio_evidence_scope_for_payload_kind(payload_kind),
                &[decision],
            );
            let row = rows
                .first()
                .expect("accepted ay receipt should render a source-aware receipt row");

            assert!(row.contains("shared_engine_validation_receipt"));
            assert!(row.contains(&format!("source_kind={}", program.source_kind.code())));
            assert!(row.contains(&format!("payload_kind={}", payload_kind.code())));
            assert!(row.contains("receipt_role=analytical_solve"));
            assert!(row.contains("receipt_identity=shared_engine.validation_receipt"));
            assert!(row.contains("search_kind=analytical_solve"));
            assert!(row.contains("model_check_search=false"));
            assert!(row.contains("validator_kind=ay_proof"));
            assert!(row.contains("validation_artifact_kind=proof"));
            assert!(row.contains("receipt_status=accepted"));
            assert!(row.contains("receipt_validation=valid"));
            assert!(row.contains("publication_readiness=ready"));
            assert!(row.contains("publication_blocker=none"));
            assert!(row.contains("consumable_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        }
    }

    #[cfg(feature = "ay")]
    #[test]
    fn ay_analytical_solve_rejects_missing_and_invalid_shared_receipts() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAYReceiptRejects ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::MccPetri,
        );
        let expected_fingerprint_identity = ay_pdr_fingerprint_identity(&program);

        let missing_receipt_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::Pdr,
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            None,
        )
        .expect("prepared ay_pdr solve should produce a shared decision")
        .with_proof_fingerprint(expected_fingerprint_identity.clone());
        let missing_route = tla_mc_core::choose_analytical_solve_route(
            &[missing_receipt_decision.clone()],
            Some("bfs"),
        );
        assert_eq!(
            missing_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            missing_route.reason,
            AnalyticalSolveDecisionReason::MissingValidationReceipt
        );
        let missing_row = missing_receipt_decision.render_evidence_row("MCC");
        assert!(missing_row.contains("validation_receipt_readiness=unknown"));
        assert!(missing_row.contains("publication_blocker=missing_validation_receipt"));
        assert!(missing_row.contains("admission_disposition=fail_closed_explicit_fallback"));

        let artifact_only_ay_receipt = tla_ay::AYProofValidationReceipt::validator_backed(
            "receipt:mcc_petri:pdr:artifact_only",
            tla_ay::AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:portfolio:ay_pdr",
            expected_fingerprint_identity.clone(),
        )
        .with_status(tla_ay::AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_receipt = shared_validation_receipt_from_ay_proof_receipt(
            program.identity.clone(),
            ay_pdr_candidate_identity(&program),
            &artifact_only_ay_receipt,
        );
        let artifact_only_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::Pdr,
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            Some(artifact_only_receipt),
        )
        .expect("prepared ay_pdr solve should produce a shared decision");
        let artifact_only_route = tla_mc_core::choose_analytical_solve_route(
            &[artifact_only_decision.clone()],
            Some("bfs"),
        );
        assert_eq!(
            artifact_only_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            artifact_only_route.reason,
            AnalyticalSolveDecisionReason::RejectedValidationReceipt
        );
        let artifact_only_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "MCC",
            &[artifact_only_decision],
        );
        let artifact_only_row = artifact_only_rows
            .first()
            .expect("artifact-only receipt should render rejection evidence");
        assert!(artifact_only_row.contains("receipt_status=rejected"));
        assert!(artifact_only_row.contains("receipt_validation=valid"));
        assert!(
            artifact_only_row.contains("failure_reason=ay_validation_receipt_status_artifact_only")
        );
        assert!(artifact_only_row.contains("publication_blocker=rejected_validation_receipt"));

        let invalid_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "",
            expected_fingerprint_identity.clone(),
            program.identity.clone(),
            ay_pdr_candidate_identity(&program),
            ValidationReceiptArtifactKind::Proof,
            expected_fingerprint_identity.clone(),
        );
        let invalid_receipt_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::Pdr,
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            Some(invalid_receipt),
        )
        .expect("prepared ay_pdr solve should produce a shared decision");
        let invalid_route = tla_mc_core::choose_analytical_solve_route(
            &[invalid_receipt_decision.clone()],
            Some("bfs"),
        );
        assert_eq!(
            invalid_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            invalid_route.reason,
            AnalyticalSolveDecisionReason::RejectedValidationReceipt
        );
        let invalid_receipt_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "MCC",
            &[invalid_receipt_decision],
        );
        let invalid_receipt_row = invalid_receipt_rows
            .first()
            .expect("invalid receipt should still render rejection evidence");
        assert!(invalid_receipt_row.contains("source_kind=mcc_petri"));
        assert!(invalid_receipt_row.contains("payload_kind=mcc_petri"));
        assert!(invalid_receipt_row.contains("receipt_status=rejected"));
        assert!(invalid_receipt_row.contains("receipt_validation=valid"));
        assert!(invalid_receipt_row.contains("publication_readiness=blocked"));
        assert!(invalid_receipt_row.contains("publication_blocker=rejected_validation_receipt"));
        assert!(invalid_receipt_row.contains("failure_reason=identity_mismatch"));
        assert!(invalid_receipt_row.contains("fail_closed=true"));

        let pdr_unsafe_proof_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            AY_SHARED_VALIDATION_DIGEST_ALGORITHM,
            expected_fingerprint_identity.clone(),
            program.identity.clone(),
            ay_pdr_candidate_identity(&program),
            ValidationReceiptArtifactKind::Proof,
            expected_fingerprint_identity.clone(),
        );
        let pdr_unsafe_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::Pdr,
            AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
            Some(pdr_unsafe_proof_receipt),
        )
        .expect("prepared unsafe ay_pdr solve should produce a shared decision");
        let pdr_unsafe_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "MCC",
            &[pdr_unsafe_decision],
        );
        let pdr_unsafe_row = pdr_unsafe_rows
            .first()
            .expect("pdr unsafe proof receipt should render accepted evidence");
        assert!(pdr_unsafe_row.contains("receipt_status=accepted"));
        assert!(pdr_unsafe_row.contains("validation_artifact_kind=proof"));
        assert!(pdr_unsafe_row.contains("publication_readiness=ready"));
        assert!(pdr_unsafe_row.contains("publication_blocker=none"));

        let weak_digest_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            expected_fingerprint_identity.clone(),
            program.identity.clone(),
            ay_pdr_candidate_identity(&program),
            ValidationReceiptArtifactKind::Proof,
            expected_fingerprint_identity.clone(),
        );
        let weak_digest_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::Pdr,
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            Some(weak_digest_receipt),
        )
        .expect("prepared ay_pdr solve should produce a shared decision");
        let weak_digest_route = tla_mc_core::choose_analytical_solve_route(
            &[weak_digest_decision.clone()],
            Some("bfs"),
        );
        assert_eq!(
            weak_digest_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            weak_digest_route.reason,
            AnalyticalSolveDecisionReason::RejectedValidationReceipt
        );
        let weak_digest_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "MCC",
            &[weak_digest_decision],
        );
        let weak_digest_row = weak_digest_rows
            .first()
            .expect("weak digest receipt should render rejection evidence");
        assert!(weak_digest_row.contains("receipt_status=rejected"));
        assert!(weak_digest_row.contains("publication_blocker=rejected_validation_receipt"));
        assert!(weak_digest_row.contains("digest_algorithm_expected=ay_fingerprint_identity"));

        let wrong_artifact_kind_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            AY_SHARED_VALIDATION_DIGEST_ALGORITHM,
            expected_fingerprint_identity.clone(),
            program.identity.clone(),
            ay_pdr_candidate_identity(&program),
            ValidationReceiptArtifactKind::Witness,
            expected_fingerprint_identity.clone(),
        );
        let wrong_artifact_kind_decision =
            ay_analytical_solve_decision_with_shared_validation_receipt(
                &program,
                tla_ay::AYSharedEngineLane::Pdr,
                AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
                Some(wrong_artifact_kind_receipt),
            )
            .expect("prepared ay_pdr solve should produce a shared decision");
        let wrong_artifact_kind_route = tla_mc_core::choose_analytical_solve_route(
            &[wrong_artifact_kind_decision.clone()],
            Some("bfs"),
        );
        assert_eq!(
            wrong_artifact_kind_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            wrong_artifact_kind_route.reason,
            AnalyticalSolveDecisionReason::RejectedValidationReceipt
        );
        let wrong_artifact_kind_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "MCC",
            &[wrong_artifact_kind_decision],
        );
        let wrong_artifact_kind_row = wrong_artifact_kind_rows
            .first()
            .expect("wrong artifact kind receipt should render rejection evidence");
        assert!(wrong_artifact_kind_row.contains("receipt_status=rejected"));
        assert!(wrong_artifact_kind_row.contains("publication_blocker=rejected_validation_receipt"));
        assert!(wrong_artifact_kind_row
            .contains("validation_artifact_kind_expected=proof_actual=witness"));

        let mismatched_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "ay.proof.fingerprint:mcc_petri:pdr",
            "wrong-prepared-program",
            "wrong-candidate",
            ValidationReceiptArtifactKind::Proof,
            "ay.proof.fingerprint:mcc_petri:pdr",
        );
        let mismatched_receipt_decision =
            ay_analytical_solve_decision_with_shared_validation_receipt(
                &program,
                tla_ay::AYSharedEngineLane::Pdr,
                AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
                Some(mismatched_receipt),
            )
            .expect("prepared ay_pdr solve should produce a shared decision");
        let mismatched_route = tla_mc_core::choose_analytical_solve_route(
            &[mismatched_receipt_decision.clone()],
            Some("bfs"),
        );
        assert_eq!(
            mismatched_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            mismatched_route.reason,
            AnalyticalSolveDecisionReason::RejectedValidationReceipt
        );
        let mismatched_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "MCC",
            &[mismatched_receipt_decision],
        );
        let mismatched_row = mismatched_rows
            .first()
            .expect("mismatched receipt should still render rejection evidence");
        assert!(mismatched_row.contains("receipt_status=rejected"));
        assert!(mismatched_row.contains("receipt_validation=valid"));
        assert!(mismatched_row.contains("publication_blocker=rejected_validation_receipt"));
        assert!(mismatched_row.contains("failure_reason=identity_mismatch"));
    }

    #[cfg(feature = "ay")]
    #[test]
    fn ay_bmc_and_kinduction_publication_is_receipt_backed_and_missing_receipts_fail_closed() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAYMissingRuntimeReceipts ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        );

        let bmc_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::Bmc,
            AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
            None,
        )
        .expect("prepared ay_bmc solve should produce a shared decision");
        let kinduction_decision = ay_analytical_solve_decision_with_shared_validation_receipt(
            &program,
            tla_ay::AYSharedEngineLane::KInduction,
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            None,
        )
        .expect("prepared ay_k_induction solve should produce a shared decision");

        for (decision, expected_fingerprint) in [
            (
                bmc_decision,
                ay_candidate_fingerprint_identity(&program, "ay_bmc"),
            ),
            (
                kinduction_decision,
                ay_candidate_fingerprint_identity(&program, "ay_k_induction"),
            ),
        ] {
            let route =
                tla_mc_core::choose_analytical_solve_route(&[decision.clone()], Some("bfs"));
            assert_eq!(
                route.route,
                tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
            );
            assert_eq!(
                route.reason,
                AnalyticalSolveDecisionReason::MissingValidationReceipt
            );
            let row = decision.render_evidence_row("TLA");
            assert!(row.contains("validation_requirements=ay_proof"));
            assert!(row.contains(&format!("proof_fingerprint={expected_fingerprint}")));
            assert!(row.contains("publication_readiness=blocked"));
            assert!(row.contains("publication_blocker=missing_validation_receipt"));
            assert!(row.contains("admission_disposition=fail_closed_explicit_fallback"));

            let shared = SharedVerdict::new();
            assert!(
                !ay_publish_verdict_with_shared_validation_receipt_decisions(
                    &shared,
                    Verdict::Violated,
                    &[decision]
                )
            );
            assert_eq!(shared.get(), None);
        }

        let bmc_run = crate::ay_bmc::BmcRunResult {
            result: crate::ay_bmc::BmcResult::Violation {
                depth: 2,
                trace: Vec::new(),
            },
            solver_decision_profile:
                crate::symbolic_explore::AYSolveDecisionProfileEvidence::from_typed_fields_for_testing(
                    tla_ay::SolveDecision::Sat,
                    true,
                    None,
                    true,
                ),
        };
        let bmc_runtime_decisions =
            ay_runtime_analytical_solve_decisions(&program, None, Some(&bmc_run), None);
        assert_eq!(bmc_runtime_decisions.len(), 1);
        let bmc_runtime_decision = &bmc_runtime_decisions[0];
        assert_eq!(bmc_runtime_decision.publication_blocker_reason(), None);
        let bmc_receipt_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "TLA",
            &bmc_runtime_decisions,
        );
        let bmc_receipt_row = bmc_receipt_rows
            .first()
            .expect("runtime BMC violation should attach a validator-backed receipt");
        assert!(bmc_receipt_row.contains("lane_kind=ay"));
        assert!(bmc_receipt_row.contains("lane=ay"));
        assert!(bmc_receipt_row.contains("backend_code=ay_smt"));
        assert!(bmc_receipt_row.contains("problem=bmc"));
        assert!(bmc_receipt_row.contains("validation_artifact_kind=witness"));
        assert!(bmc_receipt_row.contains("receipt_status=accepted"));
        assert!(bmc_receipt_row.contains("publication_readiness=ready"));

        let shared = SharedVerdict::new();
        assert!(ay_publish_verdict_with_shared_validation_receipt_decisions(
            &shared,
            Verdict::Violated,
            &bmc_runtime_decisions,
        ));
        assert_eq!(shared.get(), Some(Verdict::Violated));

        let kinduction_result = crate::ay_kinduction::KInductionResult::Proved { k: 2 };
        let kinduction_runtime_decisions =
            ay_runtime_analytical_solve_decisions(&program, None, None, Some(&kinduction_result));
        assert_eq!(kinduction_runtime_decisions.len(), 1);
        let kinduction_runtime_decision = &kinduction_runtime_decisions[0];
        assert_eq!(
            kinduction_runtime_decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingValidationReceipt)
        );
        let kinduction_route =
            tla_mc_core::choose_analytical_solve_route(&kinduction_runtime_decisions, Some("bfs"));
        assert_eq!(
            kinduction_route.route,
            tla_mc_core::AnalyticalSolveRoute::ExplicitFallback
        );
        assert_eq!(
            kinduction_route.reason,
            AnalyticalSolveDecisionReason::MissingValidationReceipt
        );
        let kinduction_row = kinduction_runtime_decision.render_evidence_row("TLA");
        assert!(kinduction_row.contains("validation_requirements=ay_proof"));
        assert!(kinduction_row.contains(&format!(
            "proof_fingerprint={}",
            ay_candidate_fingerprint_identity(&program, "ay_k_induction")
        )));
        assert!(kinduction_row.contains("publication_readiness=blocked"));
        assert!(kinduction_row.contains("publication_blocker=missing_validation_receipt"));
        assert!(kinduction_row.contains("admission_disposition=fail_closed_explicit_fallback"));
        assert!(
            shared_engine_validation_receipt_evidence_rows_for_decisions(
                "TLA",
                &kinduction_runtime_decisions
            )
            .is_empty()
        );

        let shared = SharedVerdict::new();
        assert!(
            !ay_publish_verdict_with_shared_validation_receipt_decisions(
                &shared,
                Verdict::Satisfied,
                &kinduction_runtime_decisions,
            )
        );
        assert_eq!(shared.get(), None);
    }

    // REPRO (audit): a BMC `Deadlock` must attach a validator-backed receipt and
    // publish `Violated`, exactly like a `Violation`. Before the fix,
    // `ay_shared_validation_receipt_from_bmc_run` only matched `Violation`, so a
    // reachable deadlock produced a `VerifiedCounterexampleReplay` decision with
    // NO receipt -> publication_blocker=MissingValidationReceipt -> the verdict
    // was fail-closed DOWNGRADED to Unknown (deadlock lost in the portfolio).
    #[cfg(feature = "ay")]
    #[test]
    fn ay_bmc_deadlock_runtime_decision_is_receipt_backed_like_violation() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAYDeadlockRuntimeReceipt ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        );

        let bmc_run = crate::ay_bmc::BmcRunResult {
            result: crate::ay_bmc::BmcResult::Deadlock {
                depth: 2,
                trace: Vec::new(),
            },
            solver_decision_profile:
                crate::symbolic_explore::AYSolveDecisionProfileEvidence::from_typed_fields_for_testing(
                    tla_ay::SolveDecision::Sat,
                    true,
                    None,
                    true,
                ),
        };
        let bmc_runtime_decisions =
            ay_runtime_analytical_solve_decisions(&program, None, Some(&bmc_run), None);
        assert_eq!(bmc_runtime_decisions.len(), 1);
        let bmc_runtime_decision = &bmc_runtime_decisions[0];
        // The deadlock decision must NOT be blocked for a missing receipt.
        assert_eq!(
            bmc_runtime_decision.publication_blocker_reason(),
            None,
            "BMC Deadlock should attach a validator-backed receipt, not be blocked"
        );
        let bmc_receipt_rows = shared_engine_validation_receipt_evidence_rows_for_decisions(
            "TLA",
            &bmc_runtime_decisions,
        );
        let bmc_receipt_row = bmc_receipt_rows
            .first()
            .expect("runtime BMC deadlock should attach a validator-backed receipt");
        assert!(bmc_receipt_row.contains("problem=bmc"));
        assert!(bmc_receipt_row.contains("validation_artifact_kind=witness"));
        assert!(bmc_receipt_row.contains("receipt_status=accepted"));
        assert!(bmc_receipt_row.contains("publication_readiness=ready"));

        // The Violated verdict must actually publish (not be downgraded to Unknown).
        let shared = SharedVerdict::new();
        assert!(ay_publish_verdict_with_shared_validation_receipt_decisions(
            &shared,
            Verdict::Violated,
            &bmc_runtime_decisions,
        ));
        assert_eq!(shared.get(), Some(Verdict::Violated));
    }

    #[test]
    fn analytical_prepared_program_uses_payload_kind_for_quint_lowered_to_tla() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalPreparedQuint ----
EXTENDS FiniteSets
VARIABLE x
Init == TRUE
Next == TRUE
Inv == Cardinality({1, 2}) = 2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");

        let program = prepared_analytical_portfolio_program(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Quint,
        );

        assert_eq!(program.payload_kind, PreparedProgramPayloadKind::Quint);
        assert_eq!(program.source_kind.code(), "quint");
        assert!(program
            .analytical_solves
            .iter()
            .any(|solve| solve.id == "tla.analytical.finite_cardinality:Inv"
                && solve.kind == PreparedAnalyticalSolveKind::UpperBounds
                && solve.problem == ProblemKind::Invariant));
        assert!(program.symbolic_proofs.iter().any(|proof| proof.id
            == "tla.analytical.finite_cardinality.proof:Inv"
            && proof.kind == PreparedSymbolicProofKind::InvariantProof));
        assert!(program
            .render_evidence_row("TY")
            .contains("payload_kind=quint"));
        let decision_rows = analytical_solve_decision_rows_for_prepared_program(&program, "TY");
        assert!(decision_rows
            .iter()
            .any(|row| row.contains("source_kind=quint") && row.contains("payload_kind=quint")));
        assert!(decision_rows.iter().any(|row| {
            row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
                && row.contains("candidate_key=analytical")
                && row.contains("cache_fingerprint_compatibility=frontend_local_only")
                && row.contains("publication_readiness=blocked")
        }));
        #[cfg(feature = "ay")]
        assert!(decision_rows.iter().any(|row| {
            row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
                && row.contains("candidate_key=ay_all_sat_enumeration")
                && row.contains("cache_fingerprint_compatibility=frontend_reusable")
        }));
    }

    #[test]
    fn portfolio_evidence_scope_names_cover_shared_payload_kinds() {
        let mappings = [
            (PreparedProgramPayloadKind::Tla, "TLA"),
            (PreparedProgramPayloadKind::Quint, "Quint"),
            (PreparedProgramPayloadKind::MccPetri, "MCC"),
            (PreparedProgramPayloadKind::Aiger, "AIGER"),
            (PreparedProgramPayloadKind::Btor2, "BTOR2"),
            (PreparedProgramPayloadKind::VmtInterchange, "VMT"),
            (PreparedProgramPayloadKind::AYOnly, "AY"),
            (PreparedProgramPayloadKind::WitnessReplay, "Replay"),
        ];

        for (payload_kind, expected_scope) in mappings {
            assert_eq!(
                portfolio_evidence_scope_for_payload_kind(payload_kind),
                expected_scope
            );
        }
    }

    #[test]
    fn portfolio_analytical_evidence_preserves_quint_source_identity() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalQuintSource ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let strategy_filter = vec![ANALYTICAL_STRATEGY.to_string()];

        let result = PortfolioResult::run_with_frontend_source(
            &module,
            &[],
            &config,
            &strategy_filter,
            true,
        );

        assert_eq!(result.winner, PortfolioWinner::Analytical);
        assert!(result.analytical_solve_evidence.iter().any(|row| {
            row.contains("Quint analytical_solve_decision")
                && row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
                && row.contains("decision_status=verified_execution_model")
                && row.contains("cache_fingerprint_compatibility=frontend_local_only")
        }));
        assert!(!result.analytical_solve_evidence.iter().any(|row| {
            row.contains("analytical_solve_decision")
                && (row.contains("source_kind=tla") || row.contains("payload_kind=tla"))
        }));
        assert!(!result
            .analytical_solve_evidence
            .iter()
            .any(|row| row.contains("TY analytical_solve_decision")));
        assert!(result.shared_engine_validation_receipts.iter().any(|row| {
            row.contains("Quint shared_engine_validation_receipt")
                && row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
                && row.contains("receipt_role=analytical_solve")
                && row.contains("model_check_search=false")
                && row.contains("publication_readiness=ready")
        }));
        assert!(!result
            .shared_engine_validation_receipts
            .iter()
            .any(|row| row.contains("TY shared_engine_validation_receipt")));

        #[cfg(feature = "ay")]
        {
            assert!(result.ay_shared_engine_evidence.iter().any(|row| {
                row.contains("Quint ay_shared_engine_lane_admission")
                    && row.contains("source_kind=quint")
                    && row.contains("payload_kind=quint")
                    && row.contains("frontend_family=quint")
                    && row.contains("admission_status=admitted")
            }));
            assert!(!result.ay_shared_engine_evidence.iter().any(|row| {
                row.contains("ay_shared_engine_lane_admission")
                    && (row.contains("source_kind=tla") || row.contains("payload_kind=tla"))
            }));
        }
    }

    #[test]
    fn analytical_execution_model_normalizes_resolved_specification_config() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalResolvedSpecification ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let mut config = interval_counter_config("Init", "Next", "Inv");
        config.specification = Some("Spec".to_string());
        let strategy_filter = vec![ANALYTICAL_STRATEGY.to_string()];

        let result = run_portfolio(&module, &[], &config, &strategy_filter);

        assert_eq!(result.winner, PortfolioWinner::Analytical);
        assert_eq!(
            result.analytical_eligibility,
            AnalyticalEligibility::VerifiedExecutionModel
        );
    }

    #[test]
    fn analytical_execution_model_rejects_partial_init_before_portfolio_shortcut() {
        let module = parse_module(
            r#"
---- MODULE PortfolioAnalyticalPartialInit ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");

        assert!(try_run_analytical_execution_model(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        )
        .is_none());
        assert_eq!(
            analytical_eligibility_for_config(&module, &[], &config),
            AnalyticalEligibility::StructurallyEligible
        );
    }

    #[test]
    fn analytical_static_cardinality_pre_solve_skips_bfs_when_deadlock_disabled() {
        let module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinality ----
EXTENDS FiniteSets
VARIABLE x
Init == TRUE
Next == FALSE
Inv == Cardinality(SUBSET {1, 2}) = 4
====
"#,
        );
        let mut config = interval_counter_config("Init", "Next", "Inv");
        config.check_deadlock = false;
        let strategy_filter = vec![ANALYTICAL_STRATEGY.to_string()];

        let result = run_portfolio(&module, &[], &config, &strategy_filter);

        assert_eq!(result.winner, PortfolioWinner::Analytical);
        assert_eq!(
            result.analytical_eligibility,
            AnalyticalEligibility::VerifiedStaticInvariant
        );
        match result.bfs_result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.initial_states, 0);
                assert_eq!(stats.states_found, 0);
            }
            other => panic!("expected analytical static invariant success, got {other:?}"),
        }
    }

    #[test]
    fn analytical_static_cardinality_requires_deadlock_disabled_before_shortcut() {
        let module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinalityDeadlockOn ----
EXTENDS FiniteSets
VARIABLE x
Init == TRUE
Next == FALSE
Inv == Cardinality(SUBSET {1, 2}) = 4
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");

        assert!(try_run_analytical_static_invariant_proof(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        )
        .is_none());
        assert_eq!(
            analytical_eligibility_for_config(&module, &[], &config),
            AnalyticalEligibility::StructurallyEligible
        );
    }

    #[test]
    fn analytical_static_cardinality_proves_multiple_configured_invariants() {
        let module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinalityMulti ----
EXTENDS FiniteSets, Integers
VARIABLE x
Init == TRUE
Next == TRUE
Small == Cardinality({1, 2, 3}) = 3
Functions == Cardinality([1..2 -> {"a", "b"}]) = 4
====
"#,
        );
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Small".to_string(), "Functions".to_string()],
            ..Default::default()
        };
        config.check_deadlock = false;
        let strategy_filter = vec![ANALYTICAL_STRATEGY.to_string()];

        let result = run_portfolio(&module, &[], &config, &strategy_filter);

        assert_eq!(result.winner, PortfolioWinner::Analytical);
        assert_eq!(
            result.analytical_eligibility,
            AnalyticalEligibility::VerifiedStaticInvariant
        );
    }

    #[test]
    fn analytical_static_cardinality_does_not_bypass_missing_runtime_ops() {
        let module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinalityMissingNext ----
EXTENDS FiniteSets
VARIABLE x
Init == TRUE
Inv == Cardinality(SUBSET {1, 2}) = 4
====
"#,
        );
        let mut config = interval_counter_config("Init", "MissingNext", "Inv");
        config.check_deadlock = false;

        assert!(try_run_analytical_static_invariant_proof(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        )
        .is_none());
        assert_eq!(
            analytical_eligibility_for_config(&module, &[], &config),
            AnalyticalEligibility::StructurallyEligible
        );
    }

    #[test]
    fn analytical_static_cardinality_runtime_ops_resolve_owner_first() {
        let module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinalityShadowRoot ----
EXTENDS FiniteSets
VARIABLE x
Init == x = x
Next == TRUE
Inv == Cardinality(SUBSET {1, 2}) = 4
====
"#,
        );
        let checker_module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinalityShadowHelper ----
Init == TRUE
====
"#,
        );
        let checker_modules = [&checker_module];
        let mut config = interval_counter_config("Init", "Next", "Inv");
        config.check_deadlock = false;

        assert!(try_run_analytical_static_invariant_proof(
            &module,
            &checker_modules,
            &config,
            PreparedProgramPayloadKind::Tla,
        )
        .is_none());
        assert_eq!(
            analytical_eligibility_for_config(&module, &checker_modules, &config),
            AnalyticalEligibility::StructurallyEligible
        );
    }

    #[test]
    fn analytical_static_cardinality_does_not_bypass_assumes() {
        let module = parse_module(
            r#"
---- MODULE PortfolioStaticCardinalityAssume ----
EXTENDS FiniteSets
VARIABLE x
ASSUME FALSE
Init == TRUE
Next == TRUE
Inv == Cardinality(SUBSET {1, 2}) = 4
====
"#,
        );
        let mut config = interval_counter_config("Init", "Next", "Inv");
        config.check_deadlock = false;

        assert!(try_run_analytical_static_invariant_proof(
            &module,
            &[],
            &config,
            PreparedProgramPayloadKind::Tla,
        )
        .is_none());
        assert_eq!(
            analytical_eligibility_for_config(&module, &[], &config),
            AnalyticalEligibility::StructurallyEligible
        );
    }

    #[cfg(feature = "ay")]
    #[test]
    fn portfolio_pdr_lane_exposes_chc_proof_replay_evidence() {
        let module = parse_module(
            r#"
---- MODULE PortfolioPdrEvidence ----
VARIABLE x
Init == x = 0
Next == x' = x
Inv == x = 0
====
"#,
        );
        let config = interval_counter_config("Init", "Next", "Inv");
        let strategy_filter = vec!["pdr".to_string()];

        let result = run_portfolio(&module, &[], &config, &strategy_filter);

        assert!(result.pdr_result.is_some(), "PDR lane should run");
        let evidence = result
            .pdr_proof_replay_evidence
            .as_deref()
            .expect("portfolio should retain PDR proof/replay evidence");
        assert!(evidence.contains("TLA ay_chc_proof_replay_boundary"));
        assert!(evidence.contains("production_selected=false"));
        assert!(evidence.contains("fail_closed=true"));
        assert!(result.ay_shared_engine_evidence.iter().any(|row| {
            row.contains("TLA ay_shared_engine_metadata")
                && row.contains("lanes=all_sat_enumeration,bmc,chc,pdr,k_induction")
        }));
        assert!(result.ay_shared_engine_evidence.iter().any(|row| {
            row.contains("TLA ay_shared_engine_lane_metadata")
                && row.contains("lane=pdr")
                && row.contains("frontend_neutral=true")
        }));
        assert!(result.ay_shared_engine_evidence.iter().any(|row| {
            row.contains("TLA ay_shared_engine_lane_admission")
                && row.contains("lane=pdr")
                && row.contains("admission_status=admitted")
                && row.contains("reason_code=admitted_prerequisites_satisfied")
        }));
    }

    /// Test determine_winner with BFS satisfied.
    #[test]
    fn test_determine_winner_bfs_satisfied() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Satisfied);
        let bfs_result = CheckResult::Success(crate::check::CheckStats {
            states_found: 10,
            ..Default::default()
        });
        let random_result = Some(RandomWalkResult::NoViolationFound {
            walks_completed: 100,
            total_steps: 1000,
        });
        let winner = determine_winner(&sv, &bfs_result, &random_result);
        assert_eq!(winner, PortfolioWinner::Bfs);
    }

    /// Test determine_winner with random walk finding violation.
    #[test]
    fn test_determine_winner_random_violated() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Violated);
        let bfs_result = CheckResult::Success(crate::check::CheckStats {
            states_found: 10,
            ..Default::default()
        });
        let random_result = Some(RandomWalkResult::InvariantViolation {
            invariant: "TypeOK".to_string(),
            trace: crate::check::Trace {
                states: vec![],
                action_labels: vec![],
            },
            walk_id: 0,
            depth: 3,
        });
        let winner = determine_winner(&sv, &bfs_result, &random_result);
        assert_eq!(winner, PortfolioWinner::Random);
    }

    /// Test determine_winner fallback when verdict unresolved.
    #[test]
    fn test_determine_winner_unresolved_fallback() {
        let sv = SharedVerdict::new();
        let bfs_result = CheckResult::Success(crate::check::CheckStats {
            states_found: 10,
            ..Default::default()
        });
        let random_result = None;
        let winner = determine_winner(&sv, &bfs_result, &random_result);
        assert_eq!(winner, PortfolioWinner::Bfs);
    }
}
