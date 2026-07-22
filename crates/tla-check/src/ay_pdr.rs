// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AY-based PDR (Property-Directed Reachability) for symbolic safety checking
//!
//! This module provides IC3/PDR-based verification using the CHC solver from
//! `tla-ay`. Unlike explicit-state model checking, PDR can prove safety for
//! infinite-state systems by synthesizing inductive invariants.
//!
//! # Part of #642: Implement ay PDR symbolic safety check
//!
//! # Supported Subset
//!
//! - State variables: Bool, Int, String, and finite-domain functions
//! - Boolean operators: TRUE, FALSE, /\, \/, ~, =>, <=>
//! - Comparisons: =, #, <, <=, >, >=
//! - Arithmetic: +, -, *, unary -, \div, %
//! - Conditionals: IF THEN ELSE
//! - Action constructs: x', UNCHANGED x, UNCHANGED <<x, y>>
//! - Finite quantifiers over `SetEnum`, `Range`, `BOOLEAN`, and `SetFilter`
//! - Set membership over `BOOLEAN`, `Int`, `Nat`, `SetEnum`, `Range`, `SetFilter`,
//!   `SetBuilder`, and finite-domain function sets
//!
//! # Usage
//!
//! ```no_run
//! use tla_check::{check_pdr, PdrResult};
//! use tla_check::{Config, EvalCtx};
//! use tla_core::ast::Module;
//!
//! fn verify_spec(module: &Module, config: &Config, ctx: &EvalCtx) {
//!     match check_pdr(module, config, ctx) {
//!         Ok(PdrResult::Safe { invariant }) => {
//!             println!("Proved safe with invariant: {}", invariant);
//!         }
//!         Ok(PdrResult::Unsafe { trace }) => {
//!             println!("Found counterexample with {} states", trace.len());
//!         }
//!         Ok(PdrResult::Unknown { reason }) => {
//!             println!("Inconclusive: {}", reason);
//!         }
//!         Err(e) => {
//!             eprintln!("PDR error: {}", e);
//!         }
//!     }
//! }
//! ```
//!
//! # Design Notes
//!
//! PDR results are separate from explicit-state CheckResult because:
//! 1. PDR does not enumerate states, so metrics like `states_found` are misleading
//! 2. PDR produces different artifacts (invariant model vs state space coverage)
//! 3. Clear UX separation helps users understand which mode is active

mod expand;
pub(crate) mod generalize;

use std::sync::Arc;
use std::time::Instant;

use tla_ay::chc::{
    ChcProofTranscriptConsumerEvidence, ChcTranslator, PdrCheckResult, PdrProofCheckResult,
    PdrState,
};
use tla_ay::{PdrConfig, TlaSort};
use tla_core::ast::Module;

use crate::ay_shared;
use crate::check::CheckError;
use crate::config::Config;
use crate::eval::EvalCtx;
use crate::shared_verdict::SharedVerdict;

pub use expand::expand_operators_for_chc;
pub(crate) use generalize::{flatten_conjunction, generalize_lemma, LemmaGeneralizer};

// Re-exports for statistics tracking and batch processing.
// These are used by `check_pdr_with_generalization` and `generalize_lemmas_batch`
// which are currently wired in cooperative mode and available for external callers.
#[allow(unused_imports)]
pub(crate) use generalize::{AtomicGeneralizationStats, GeneralizedLemma};

/// Result of PDR-based safety verification
#[derive(Debug)]
pub enum PdrResult {
    /// Proven safe: all reachable states satisfy all invariants
    Safe {
        /// String representation of the synthesized invariant
        invariant: String,
    },
    /// Counterexample found: trace violates an invariant
    Unsafe {
        /// Counterexample trace (each element is a state with variable assignments)
        trace: Vec<PdrState>,
    },
    /// Inconclusive: PDR could not determine safety
    Unknown {
        /// Reason for the unknown result
        reason: String,
    },
}

const PDR_MISSING_TYPED_CHC_PROOF_TRANSCRIPT: &str = "missing_typed_chc_proof_transcript";
const PDR_MISSING_PUBLIC_CHC_METADATA: &str = "missing_public_chc_metadata";

/// TLA-check boundary around AY CHC proof/replay evidence.
///
/// Keep a structured, conservative boundary here so consumers can fail closed
/// without parsing rows or treating rendered strings as proof admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AYChcProofReplayEvidence {
    evidence: String,
    status: String,
    status_code: String,
    row_status_code: String,
    typed_consumer: bool,
    accepted_as_proof: bool,
    trust_full_verifier_admissible: bool,
    trust_full_verifier_non_admission_reason: String,
    fail_closed: bool,
    consumer_evidence: Option<ChcProofTranscriptConsumerEvidence>,
}

impl AYChcProofReplayEvidence {
    pub(crate) fn missing() -> Self {
        Self {
            evidence: missing_pdr_proof_replay_evidence(),
            status: "Unavailable".to_string(),
            status_code: PDR_MISSING_TYPED_CHC_PROOF_TRANSCRIPT.to_string(),
            row_status_code: PDR_MISSING_TYPED_CHC_PROOF_TRANSCRIPT.to_string(),
            typed_consumer: false,
            accepted_as_proof: false,
            trust_full_verifier_admissible: false,
            trust_full_verifier_non_admission_reason: PDR_MISSING_TYPED_CHC_PROOF_TRANSCRIPT
                .to_string(),
            fail_closed: true,
            consumer_evidence: None,
        }
    }

    fn from_ay_proof_replay_row(evidence: String) -> Self {
        Self {
            evidence,
            status: "Unavailable".to_string(),
            status_code: PDR_MISSING_PUBLIC_CHC_METADATA.to_string(),
            row_status_code: "typed_chc_proof_transcript".to_string(),
            typed_consumer: false,
            accepted_as_proof: false,
            trust_full_verifier_admissible: false,
            trust_full_verifier_non_admission_reason: PDR_MISSING_PUBLIC_CHC_METADATA.to_string(),
            fail_closed: true,
            consumer_evidence: None,
        }
    }

    fn from_ay_consumer_evidence(
        evidence: String,
        consumer_evidence: &ChcProofTranscriptConsumerEvidence,
    ) -> Self {
        let trust_full_verifier_admissible = consumer_evidence.trust_full_verifier_admissible;
        Self {
            evidence,
            status: "Available".to_string(),
            status_code: "typed_chc_consumer_evidence".to_string(),
            row_status_code: "typed_chc_proof_transcript".to_string(),
            typed_consumer: true,
            accepted_as_proof: consumer_evidence.accepted_for_consumer,
            trust_full_verifier_admissible,
            trust_full_verifier_non_admission_reason: consumer_evidence
                .trust_full_verifier_non_admission_reason
                .as_deref()
                .unwrap_or("none")
                .to_string(),
            fail_closed: !trust_full_verifier_admissible,
            consumer_evidence: Some(consumer_evidence.clone()),
        }
    }

    fn from_ay_checked_result(checked: &PdrProofCheckResult) -> Self {
        if let Some(consumer_evidence) = checked.proof_consumer_evidence.as_ref() {
            Self::from_ay_consumer_evidence(
                checked.proof_replay_evidence.clone(),
                consumer_evidence,
            )
        } else {
            Self::from_observable_proof_replay_row(checked.proof_replay_evidence.clone())
        }
    }

    fn from_observable_proof_replay_row(evidence: String) -> Self {
        if evidence == missing_pdr_proof_replay_evidence() {
            Self::missing()
        } else {
            Self::from_ay_proof_replay_row(evidence)
        }
    }

    /// Borrow the summarizer-ready CHC proof/replay evidence row.
    pub fn evidence_row(&self) -> &str {
        &self.evidence
    }

    /// TLA-check's structured metadata availability status.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// TLA-check's structured metadata availability code.
    pub fn status_code(&self) -> &str {
        &self.status_code
    }

    /// Status code of the AY-rendered evidence row, if known at the call site.
    pub fn row_status_code(&self) -> &str {
        &self.row_status_code
    }

    /// Whether TLA-check consumed typed CHC transcript fields.
    pub fn typed_consumer(&self) -> bool {
        self.typed_consumer
    }

    /// Whether AY accepted the transcript as a proof.
    pub fn accepted_as_proof(&self) -> bool {
        self.accepted_as_proof
    }

    /// Whether AY accepted the transcript for downstream consumers.
    pub fn accepted_for_consumer(&self) -> bool {
        self.accepted_as_proof
    }

    /// Whether AY's full-verifier trust boundary admitted this proof.
    pub fn trust_full_verifier_admissible(&self) -> bool {
        self.trust_full_verifier_admissible
    }

    /// AY/TLA-check non-admission reason code for the proof boundary.
    pub fn trust_full_verifier_non_admission_reason(&self) -> &str {
        &self.trust_full_verifier_non_admission_reason
    }

    /// Whether this boundary must fail closed.
    pub fn fail_closed(&self) -> bool {
        self.fail_closed
    }

    /// Whether this PDR proof is admissible at the TLA-check boundary.
    pub fn accepts_proof_for_tla_boundary(&self) -> bool {
        self.typed_consumer
            && self.accepted_as_proof
            && self.trust_full_verifier_admissible
            && !self.fail_closed
    }

    /// Borrow AY-owned typed CHC consumer evidence, when the proof API returned it.
    pub fn consumer_evidence(&self) -> Option<&ChcProofTranscriptConsumerEvidence> {
        self.consumer_evidence.as_ref()
    }
}

/// PDR result plus the CHC proof/replay boundary evidence row consumed from AY.
#[derive(Debug)]
pub struct PdrRunResult {
    /// Proven-safe, unsafe, or inconclusive PDR result.
    pub result: PdrResult,
    /// Stable evidence row describing the typed AY CHC proof/replay boundary.
    pub proof_replay_evidence: String,
    /// Structured CHC proof/replay boundary consumed without row parsing.
    pub proof_replay_boundary: AYChcProofReplayEvidence,
    /// Frontend-neutral shared AY PDR lane metadata consumed by this run.
    pub shared_engine_lane_evidence: String,
}

impl PdrRunResult {
    fn new(result: PdrResult, proof_replay_boundary: AYChcProofReplayEvidence) -> Self {
        let proof_replay_evidence = proof_replay_boundary.evidence_row().to_string();
        Self {
            result,
            proof_replay_evidence,
            proof_replay_boundary,
            shared_engine_lane_evidence: tla_ay::render_ay_shared_engine_lane_evidence(
                "TLA",
                tla_ay::AYSharedEngineLane::Pdr,
            ),
        }
    }

    /// Borrow the CHC proof/replay evidence row.
    pub fn proof_replay_evidence(&self) -> &str {
        &self.proof_replay_evidence
    }

    /// Borrow the frontend-neutral shared AY PDR lane metadata evidence row.
    pub fn shared_engine_lane_evidence(&self) -> &str {
        &self.shared_engine_lane_evidence
    }

    /// Return the structured CHC proof/replay boundary.
    pub fn proof_replay_boundary(&self) -> AYChcProofReplayEvidence {
        self.proof_replay_boundary.clone()
    }

    /// Borrow AY-owned typed CHC consumer evidence for this PDR run.
    pub fn proof_consumer_evidence(&self) -> Option<&ChcProofTranscriptConsumerEvidence> {
        self.proof_replay_boundary.consumer_evidence()
    }

    /// Consume this wrapper and return the legacy PDR result.
    pub fn into_result(self) -> PdrResult {
        self.result
    }
}

fn missing_pdr_proof_replay_evidence() -> String {
    tla_ay::chc::render_chc_proof_replay_boundary_evidence("TLA", None)
}

pub(crate) fn unknown_pdr_run_with_missing_evidence(reason: impl Into<String>) -> PdrRunResult {
    PdrRunResult::new(
        PdrResult::Unknown {
            reason: reason.into(),
        },
        AYChcProofReplayEvidence::missing(),
    )
}

fn map_checked_pdr_result(checked: PdrProofCheckResult) -> PdrRunResult {
    let proof_replay_boundary = AYChcProofReplayEvidence::from_ay_checked_result(&checked);
    let result = match checked.result {
        PdrCheckResult::Safe { invariant } => PdrResult::Safe { invariant },
        PdrCheckResult::Unsafe { trace } => PdrResult::Unsafe { trace },
        PdrCheckResult::Unknown { reason } => PdrResult::Unknown { reason },
    };
    PdrRunResult::new(result, proof_replay_boundary)
}

/// Error types specific to PDR checking
#[derive(Debug, thiserror::Error)]
pub enum PdrError {
    /// Missing Init or Next definition
    #[error("Missing specification: {0}")]
    MissingSpec(String),
    /// No invariants configured
    #[error("No invariants configured for PDR checking")]
    NoInvariants,
    /// Failed to infer variable sorts
    #[error("Sort inference failed: {0}")]
    SortInference(String),
    /// Expression not supported for CHC translation
    #[error("Unsupported expression: {0}")]
    UnsupportedExpr(String),
    /// CHC translation error
    #[error("CHC translation error: {0}")]
    TranslationError(String),
    /// General check error
    #[error("Check error: {0:?}")]
    CheckError(#[from] CheckError),
}

impl From<tla_ay::AYError> for PdrError {
    fn from(e: tla_ay::AYError) -> Self {
        PdrError::TranslationError(format!("{}", e))
    }
}

/// Run PDR-based safety verification on a TLA+ spec
///
/// This function:
/// 1. Resolves Init/Next from the config
/// 2. Builds Safety as conjunction of configured invariants
/// 3. Infers variable sorts (Bool/Int) from TypeOK or constraints
/// 4. Translates to CHC and runs PDR solver
///
/// # Arguments
/// * `module` - The loaded TLA+ module
/// * `config` - TLC configuration with INIT, NEXT, INVARIANT
/// * `ctx` - Evaluation context with loaded operators
///
/// # Returns
/// * `Ok(PdrResult)` - Safe, Unsafe, or Unknown
/// * `Err(PdrError)` - If translation or verification fails
pub fn check_pdr(module: &Module, config: &Config, ctx: &EvalCtx) -> Result<PdrResult, PdrError> {
    check_pdr_with_config(module, config, ctx, default_pdr_config())
}

/// Run PDR and return the result plus typed CHC proof/replay boundary evidence.
pub fn check_pdr_with_evidence(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
) -> Result<PdrRunResult, PdrError> {
    check_pdr_with_config_and_evidence(module, config, ctx, default_pdr_config())
}

/// Default PDR configuration with a bounded solve timeout.
///
/// Part of #2826: The default path installs a 300-second timeout so
/// production PDR runs cannot hang indefinitely. Use
/// `check_pdr_with_config` to bypass this for tests or tuning.
/// Override via `TY_PDR_TIMEOUT_SECS` env var.
fn default_pdr_config() -> PdrConfig {
    use std::time::Duration;

    let timeout_secs: u64 = std::env::var("TY_PDR_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let mut config = PdrConfig::default();
    config.solve_timeout = Some(Duration::from_secs(timeout_secs));
    config
}

/// Run PDR with custom solver configuration
pub fn check_pdr_with_config(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    pdr_config: PdrConfig,
) -> Result<PdrResult, PdrError> {
    check_pdr_with_config_and_evidence(module, config, ctx, pdr_config)
        .map(PdrRunResult::into_result)
}

/// Run PDR with custom solver configuration and preserve proof/replay evidence.
pub fn check_pdr_with_config_and_evidence(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    pdr_config: PdrConfig,
) -> Result<PdrRunResult, PdrError> {
    check_pdr_with_portfolio_and_evidence(module, config, ctx, pdr_config, None)
}

/// Run PDR with portfolio verdict for early termination.
///
/// When `portfolio_verdict` is `Some`, checks whether another lane has
/// already resolved before and after the (blocking) PDR solve call.
/// If resolved, returns `PdrResult::Unknown` immediately.
///
/// Part of #3717.
pub fn check_pdr_with_portfolio(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    pdr_config: PdrConfig,
    portfolio_verdict: Option<Arc<SharedVerdict>>,
) -> Result<PdrResult, PdrError> {
    check_pdr_with_portfolio_and_evidence(module, config, ctx, pdr_config, portfolio_verdict)
        .map(PdrRunResult::into_result)
}

/// Run PDR with portfolio verdict and preserve proof/replay evidence.
///
/// If the lane exits before invoking AY because another lane already resolved,
/// the returned evidence is an explicit fail-closed missing-transcript row.
pub fn check_pdr_with_portfolio_and_evidence(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    pdr_config: PdrConfig,
    portfolio_verdict: Option<Arc<SharedVerdict>>,
) -> Result<PdrRunResult, PdrError> {
    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(PdrError::MissingSpec)?;
    let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
    if vars.is_empty() {
        return Err(PdrError::MissingSpec(
            "No state variables declared".to_string(),
        ));
    }

    if config.invariants.is_empty() {
        return Err(PdrError::NoInvariants);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(PdrError::MissingSpec)?;

    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(PdrError::MissingSpec)?;
    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);

    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(PdrError::MissingSpec)?;
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);

    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(|e| PdrError::MissingSpec(e))?;
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);

    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    // 8. Create CHC translator and add clauses
    let var_refs: Vec<(&str, TlaSort)> = var_sorts
        .iter()
        .map(|(name, sort)| (name.as_str(), sort.clone()))
        .collect();

    let mut translator = ChcTranslator::new(&var_refs)?;

    translator.add_init(&init_expanded)?;
    translator.add_next(&next_expanded)?;
    translator.add_safety(&safety_expanded)?;

    // 9. Portfolio early-exit check before blocking solve (Part of #3717).
    if let Some(ref sv) = portfolio_verdict {
        if sv.is_resolved() {
            return Ok(unknown_pdr_run_with_missing_evidence(
                "portfolio verdict resolved by another lane",
            ));
        }
    }

    // 10. Run PDR solver
    let checked = translator.solve_pdr_with_proof_evidence(pdr_config)?;

    // 11. Portfolio early-exit check after solve returns (Part of #3717).
    if let Some(ref sv) = portfolio_verdict {
        if sv.is_resolved() {
            let proof_replay_boundary = AYChcProofReplayEvidence::from_ay_checked_result(&checked);
            return Ok(PdrRunResult::new(
                PdrResult::Unknown {
                    reason: String::from("portfolio verdict resolved by another lane during PDR"),
                },
                proof_replay_boundary,
            ));
        }
    }

    // 12. Map result to PdrResult
    Ok(map_checked_pdr_result(checked))
}

/// Outcome of an interruptible PDR solve. See [`solve_pdr_interruptible`].
enum InterruptiblePdr {
    /// The solve ran to completion (or hit its own `solve_deadline`).
    Completed(tla_ay::AYResult<PdrProofCheckResult>),
    /// Another cooperative lane resolved the verdict first; the solve was
    /// abandoned (and continues to completion on a detached worker, reaped at
    /// process exit).
    Interrupted,
}

/// Run a blocking PDR solve so it can be abandoned the instant another
/// cooperative lane resolves the verdict.
///
/// `ChcTranslator::solve_pdr_with_proof_evidence` only checks its
/// `solve_deadline` *between* SMT operations. On a spec whose PDR fixpoint does
/// not converge, that lets the solve hold the entire fused run hostage for the
/// full `solve_timeout` (300s in fused mode) even though BFS already produced
/// the authoritative verdict in milliseconds — a cooperative-termination defect
/// of the same family as the report's #1/#2/#4 hangs: the lane never learns the
/// race is already over (observed as the DifftraceTest fused-mode hang). The
/// existing pre-/post-solve `is_resolved()` guards cannot help because the
/// thread is *inside* the blocking call.
///
/// Running the solve on a detached worker and racing it against `is_resolved()`
/// bounds the lane's shutdown latency to a single poll interval, while
/// preserving PDR's full budget when no other lane resolves first. The worker
/// owns the translator + config and borrows nothing from the enclosing
/// `thread::scope`, so leaving it running is sound.
fn solve_pdr_interruptible(
    translator: ChcTranslator,
    pdr_config: PdrConfig,
    cooperative: &crate::cooperative_state::SharedCooperativeState,
) -> InterruptiblePdr {
    use std::sync::mpsc::{sync_channel, RecvTimeoutError};
    use std::time::Duration;

    let (tx, rx) = sync_channel(1);
    std::thread::spawn(move || {
        let _ = tx.send(translator.solve_pdr_with_proof_evidence(pdr_config));
    });
    loop {
        match rx.recv_timeout(Duration::from_millis(25)) {
            Ok(result) => return InterruptiblePdr::Completed(result),
            Err(RecvTimeoutError::Timeout) => {
                if cooperative.is_resolved() {
                    return InterruptiblePdr::Interrupted;
                }
            }
            // Worker panicked without sending — treat as abandoned rather than
            // blocking forever.
            Err(RecvTimeoutError::Disconnected) => return InterruptiblePdr::Interrupted,
        }
    }
}

/// Run PDR with cooperative result publishing and preserve proof/replay evidence.
pub(crate) fn check_pdr_cooperative_with_evidence(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    pdr_config: PdrConfig,
    cooperative: Arc<crate::cooperative_state::SharedCooperativeState>,
) -> Result<PdrRunResult, PdrError> {
    // Early exit: another lane already resolved.
    if cooperative.is_resolved() {
        return Ok(unknown_pdr_run_with_missing_evidence(
            "cooperative verdict already resolved",
        ));
    }

    // Prepare expanded expressions for both PDR solving and generalization.
    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(PdrError::MissingSpec)?;
    let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
    if vars.is_empty() {
        return Err(PdrError::MissingSpec(
            "No state variables declared".to_string(),
        ));
    }
    if config.invariants.is_empty() {
        return Err(PdrError::NoInvariants);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(PdrError::MissingSpec)?;
    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(PdrError::MissingSpec)?;
    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(PdrError::MissingSpec)?;
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);
    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(PdrError::MissingSpec)?;
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);
    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    // Build and run CHC/PDR solver.
    let var_refs: Vec<(&str, TlaSort)> = var_sorts
        .iter()
        .map(|(name, sort)| (name.as_str(), sort.clone()))
        .collect();
    let mut translator = ChcTranslator::new(&var_refs)?;
    translator.add_init(&init_expanded)?;
    translator.add_next(&next_expanded)?;
    translator.add_safety(&safety_expanded)?;

    // Portfolio early-exit check.
    if cooperative.is_resolved() {
        return Ok(unknown_pdr_run_with_missing_evidence(
            "cooperative verdict resolved before PDR solve",
        ));
    }

    // Run the (blocking) PDR solve interruptibly so a verdict resolved by
    // another lane mid-solve tears this lane down promptly instead of stalling
    // the whole fused run up to `solve_timeout`. See `solve_pdr_interruptible`.
    let checked = match solve_pdr_interruptible(translator, pdr_config, &cooperative) {
        InterruptiblePdr::Completed(result) => result?,
        InterruptiblePdr::Interrupted => {
            return Ok(unknown_pdr_run_with_missing_evidence(
                "cooperative verdict resolved during PDR solve (interrupted)",
            ));
        }
    };
    let proof_replay_boundary = AYChcProofReplayEvidence::from_ay_checked_result(&checked);

    // Portfolio early-exit check after solve.
    if cooperative.is_resolved() {
        return Ok(PdrRunResult::new(
            PdrResult::Unknown {
                reason: String::from("cooperative verdict resolved during PDR solve"),
            },
            proof_replay_boundary,
        ));
    }

    // Map result and publish cooperative results.
    let result = match checked.result {
        PdrCheckResult::Safe { invariant } => {
            // Generalize the safety expression and publish individual
            // lemmas to the cooperative state for BMC pruning.
            let gen_result =
                generalize_lemma(&var_sorts, &init_expanded, &next_expanded, &safety_expanded);

            // Publish the generalized lemma (or original if generalization failed).
            let lemma_expr = match gen_result {
                Ok(gen) => gen.expr,
                Err(_) => safety_expanded.clone(),
            };

            // Publish individual conjuncts as separate lemmas for finer-grained
            // BMC pruning. Each conjunct is an independently valid lemma.
            let mut conjuncts = Vec::new();
            flatten_conjunction(&lemma_expr, &mut conjuncts);
            for conjunct in &conjuncts {
                cooperative.publish_lemma(conjunct.clone());
                cooperative.increment_pdr_lemma_count();
            }

            cooperative.set_invariants_proved();
            // SOUNDNESS: PDR proves the safety invariant ONLY (no deadlock-freedom,
            // no liveness/temporal/trace obligations). Only let this resolve the
            // cooperative verdict — which makes the BFS lane exit early and report an
            // indistinguishable Success — when safety is the run's SOLE obligation.
            // Otherwise (deadlock-checking on, or PROPERTIES/trace invariants), leave
            // the verdict unresolved so BFS runs to completion and stays authoritative
            // for the obligations PDR did not verify; a symbolic-safe win must never
            // silently mask a reachable deadlock or a liveness violation. The
            // invariant proof above is still published as lemmas for cross-validation.
            if ay_shared::symbolic_safety_proof_covers_all_obligations(config) {
                // PROOF-PRODUCING: a symbolic Satisfied resolves the cooperative verdict
                // (and truncates BFS into a Success) ONLY when the proven invariant
                // re-discharges a STRICT, independently re-checkable certificate. An
                // unverifiable proof yields MissingVerifier, publish_analytical returns
                // false, the slot stays UNRESOLVED, and BFS stays authoritative — the
                // symbolic "proof" is never trusted as the user's verdict.
                let cert = crate::ay_bmc::strict_safety_certificate_state(ctx, config, &vars);
                cooperative
                    .verdict
                    .publish_analytical(crate::shared_verdict::Verdict::Satisfied, cert);
            }

            PdrResult::Safe { invariant }
        }
        PdrCheckResult::Unsafe { trace } => {
            // SOUNDNESS (fail closed): a CHC counterexample comes from the SMT
            // TRANSLATION of the spec and can be spurious. Publishing `Violated`
            // truncates the racing BFS lane into a result indistinguishable from
            // a clean Success, so it may happen ONLY after the explicit-state
            // evaluator has re-confirmed the counterexample. An unconfirmed
            // trace downgrades the lane result to Unknown (no publish) — BFS
            // keeps running, unharmed. Mirrors the BMC / k-Induction lanes.
            let bmc_states = crate::check::cross_validation::pdr_trace_to_bmc_states(&trace);
            let cv = crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                module,
                config,
                &bmc_states,
                crate::check::cross_validation::CrossValidationSource::Pdr,
            );
            if cv.engine_agrees {
                cooperative
                    .verdict
                    .publish(crate::shared_verdict::Verdict::Violated);
                PdrResult::Unsafe { trace }
            } else {
                telemetry_eprintln!(
                    "[ay-pdr-coop] CHC reported unsafe but the explicit-state evaluator did \
                     NOT confirm the counterexample ({}) — failing closed to Unknown \
                     (no verdict published)",
                    cv.detail
                );
                PdrResult::Unknown {
                    reason: format!(
                        "PDR reported unsafe but the explicit-state evaluator did not \
                         confirm the counterexample ({}) — failing closed",
                        cv.detail
                    ),
                }
            }
        }
        PdrCheckResult::Unknown { reason } => PdrResult::Unknown { reason },
    };

    Ok(PdrRunResult::new(result, proof_replay_boundary))
}

/// Run PDR with lemma generalization enabled.
///
/// Wraps the standard PDR flow with a post-processing generalization step.
/// When PDR proves safety, the invariant's individual conjuncts (safety
/// clauses) are generalized by dropping unnecessary literals. This produces
/// a stronger inductive invariant that converges faster when used as a seed.
///
/// Returns the PDR result along with optional generalization statistics.
#[allow(dead_code)]
pub fn check_pdr_with_generalization(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    pdr_config: PdrConfig,
    gen_stats: Option<&AtomicGeneralizationStats>,
) -> Result<PdrResult, PdrError> {
    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(PdrError::MissingSpec)?;
    let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
    if vars.is_empty() {
        return Err(PdrError::MissingSpec(
            "No state variables declared".to_string(),
        ));
    }

    if config.invariants.is_empty() {
        return Err(PdrError::NoInvariants);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(PdrError::MissingSpec)?;

    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(PdrError::MissingSpec)?;
    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);

    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(PdrError::MissingSpec)?;
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);

    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(PdrError::MissingSpec)?;
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);

    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    // Run the standard CHC/PDR solver.
    let var_refs: Vec<(&str, TlaSort)> = var_sorts
        .iter()
        .map(|(name, sort)| (name.as_str(), sort.clone()))
        .collect();

    let mut translator = ChcTranslator::new(&var_refs)?;
    translator.add_init(&init_expanded)?;
    translator.add_next(&next_expanded)?;
    translator.add_safety(&safety_expanded)?;

    let result = translator.solve_pdr(pdr_config)?;

    match result {
        PdrCheckResult::Safe { invariant } => {
            // Attempt to generalize the safety expression itself.
            // This tries to drop unnecessary conjuncts from the safety property
            // to find a weaker (but still inductive) invariant.
            let gen_start = Instant::now();
            let gen_result =
                generalize_lemma(&var_sorts, &init_expanded, &next_expanded, &safety_expanded);

            if let (Ok(gen), Some(stats)) = (&gen_result, gen_stats) {
                let elapsed_us = gen_start.elapsed().as_micros() as u64;
                stats.record(
                    gen.original_literal_count as u64,
                    gen.literals_dropped as u64,
                    elapsed_us,
                );
            }

            Ok(PdrResult::Safe { invariant })
        }
        PdrCheckResult::Unsafe { trace } => Ok(PdrResult::Unsafe { trace }),
        PdrCheckResult::Unknown { reason } => Ok(PdrResult::Unknown { reason }),
    }
}

/// Generalize a sequence of learned lemmas against the given spec.
///
/// This is a batch utility for external callers that have collected
/// lemma expressions from a PDR run and want to strengthen them
/// before using them as seeds for subsequent verification.
///
/// Returns generalized lemmas paired with per-lemma statistics.
#[allow(dead_code)]
pub(crate) fn generalize_lemmas_batch(
    var_sorts: &[(String, TlaSort)],
    init_expr: &tla_core::Spanned<tla_core::ast::Expr>,
    next_expr: &tla_core::Spanned<tla_core::ast::Expr>,
    lemmas: &[tla_core::Spanned<tla_core::ast::Expr>],
    stats: Option<&AtomicGeneralizationStats>,
) -> Vec<Result<GeneralizedLemma, PdrError>> {
    let generalizer = LemmaGeneralizer::new(var_sorts, init_expr, next_expr);

    lemmas
        .iter()
        .map(|lemma| {
            let start = Instant::now();
            let result = generalizer.generalize(lemma);
            if let (Ok(ref gen), Some(s)) = (&result, stats) {
                let elapsed_us = start.elapsed().as_micros() as u64;
                s.record(
                    gen.original_literal_count as u64,
                    gen.literals_dropped as u64,
                    elapsed_us,
                );
            }
            result
        })
        .collect()
}

#[cfg(test)]
mod tests;
