// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared analytical-solve decision and evidence contracts.
//!
//! Analytical lanes are engines, not benchmark exceptions. A verified
//! analytical proof may preempt explicit-state search; structural eligibility
//! alone must remain advisory and require a normal verifier lane.

use crate::backend_capability::{BackendKind, ProblemKind};
use crate::backend_evidence::NO_REASON_CODE;
use crate::prepared_program::{
    PreparedAnalyticalSolveDescriptor, PreparedCandidateLaneDescriptor, PreparedCheckerProgram,
    PreparedProgramPayloadKind, PreparedValidationKind,
};
use crate::setup_trace::{CheckerArtifactIdentityFields, CheckerSourceKind, SetupTraceLaneKind};
use crate::shared_engine_adoption::SharedEngineFrontendFamily;
use crate::validation_receipt::{
    ValidationReceipt, ValidationReceiptArtifactKind, ValidationReceiptValidatorKind,
};

/// No cache/fingerprint reuse claim was made for the analytical artifact.
pub const ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_NOT_DECLARED: &str = "not_declared";

/// Analytical cache/fingerprint evidence is source-frontend local only.
pub const ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_LOCAL_ONLY: &str =
    "frontend_local_only";

/// Analytical cache/fingerprint evidence is safe for compatible frontend families.
pub const ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_REUSABLE: &str = "frontend_reusable";

/// Shared-engine row kind for analytical-solve validation receipt evidence.
pub const ANALYTICAL_SOLVE_SHARED_ENGINE_VALIDATION_RECEIPT_ROW_KIND: &str =
    "shared_engine_validation_receipt";

/// Shared-engine schema for source-aware analytical-solve validation receipt rows.
pub const ANALYTICAL_SOLVE_SHARED_ENGINE_VALIDATION_RECEIPT_SCHEMA: &str =
    "ty.analytical-solve.shared-engine-validation-receipt.v1";

/// Digest label used when a AY receipt validates a prepared fingerprint identity.
pub const ANALYTICAL_SOLVE_AY_VALIDATION_DIGEST_ALGORITHM: &str = "ay_fingerprint_identity";

/// Proof/admission status for one analytical solve decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticalSolveDecisionStatus {
    /// A verified analytical execution model replaces explicit-state search.
    VerifiedExecutionModel,
    /// Verified static invariant facts replace explicit-state search for the configured obligations.
    VerifiedStaticInvariant,
    /// A replay-verified analytical witness or counterexample replaces explicit-state search.
    VerifiedCounterexampleReplay,
    /// A structural proof exists, but a normal verifier lane must still run.
    StructurallyEligible,
    /// The input is outside the analytical engine's supported fragment.
    StructurallyIneligible,
    /// The analytical lane was not requested or not evaluated.
    NotAssessed,
}

impl AnalyticalSolveDecisionStatus {
    /// Stable lowercase wire code for this status.
    pub fn code(self) -> &'static str {
        match self {
            Self::VerifiedExecutionModel => "verified_execution_model",
            Self::VerifiedStaticInvariant => "verified_static_invariant",
            Self::VerifiedCounterexampleReplay => "verified_counterexample_replay",
            Self::StructurallyEligible => "structurally_eligible",
            Self::StructurallyIneligible => "structurally_ineligible",
            Self::NotAssessed => "not_assessed",
        }
    }

    /// Rust-style variant name, for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::VerifiedExecutionModel => "VerifiedExecutionModel",
            Self::VerifiedStaticInvariant => "VerifiedStaticInvariant",
            Self::VerifiedCounterexampleReplay => "VerifiedCounterexampleReplay",
            Self::StructurallyEligible => "StructurallyEligible",
            Self::StructurallyIneligible => "StructurallyIneligible",
            Self::NotAssessed => "NotAssessed",
        }
    }

    /// Human wording for CLI/reporting layers.
    pub fn wording(self) -> &'static str {
        match self {
            Self::VerifiedExecutionModel => {
                "verified analytical execution model; explicit state exploration was skipped"
            }
            Self::VerifiedStaticInvariant => {
                "verified static finite-cardinality invariant proof; explicit state exploration was skipped"
            }
            Self::VerifiedCounterexampleReplay => {
                "replay-verified analytical witness; explicit state exploration was skipped"
            }
            Self::StructurallyEligible => {
                "verified analytical invariant proof exists, but explicit exploration is still required"
            }
            Self::StructurallyIneligible => "not structurally eligible for analytical handling",
            Self::NotAssessed => "analytical structural eligibility was not assessed",
        }
    }

    /// Wire code describing how this decision is published in results.
    pub fn publication_status_code(self) -> &'static str {
        match self {
            Self::VerifiedExecutionModel | Self::VerifiedStaticInvariant => "proof_verified",
            Self::VerifiedCounterexampleReplay => "witness_replayed",
            Self::StructurallyEligible => "requires_explicit_state_verifier",
            Self::StructurallyIneligible => "skipped_ineligible",
            Self::NotAssessed => "not_assessed",
        }
    }

    /// Wire code describing this decision's relationship to explicit-state search.
    pub fn explicit_state_relation_code(self) -> &'static str {
        match self {
            Self::VerifiedExecutionModel
            | Self::VerifiedStaticInvariant
            | Self::VerifiedCounterexampleReplay => "preempts_explicit_state_search",
            Self::StructurallyEligible => "requires_explicit_state_verifier",
            Self::StructurallyIneligible => "no_analytical_claim",
            Self::NotAssessed => "not_assessed",
        }
    }

    /// Whether this status authorizes skipping explicit-state search.
    ///
    /// True only for the three verified statuses.
    pub fn can_preempt_explicit_state(self) -> bool {
        matches!(
            self,
            Self::VerifiedExecutionModel
                | Self::VerifiedStaticInvariant
                | Self::VerifiedCounterexampleReplay
        )
    }

    /// The reason code that pairs with this status by default.
    pub fn default_reason(self) -> AnalyticalSolveDecisionReason {
        match self {
            Self::VerifiedExecutionModel => AnalyticalSolveDecisionReason::ExecutionModelVerified,
            Self::VerifiedStaticInvariant => AnalyticalSolveDecisionReason::StaticInvariantVerified,
            Self::VerifiedCounterexampleReplay => AnalyticalSolveDecisionReason::WitnessVerified,
            Self::StructurallyEligible => AnalyticalSolveDecisionReason::StructuralProofOnly,
            Self::StructurallyIneligible => AnalyticalSolveDecisionReason::UnsupportedFragment,
            Self::NotAssessed => AnalyticalSolveDecisionReason::NotRequested,
        }
    }

    /// The portfolio lifecycle stage that pairs with this status by default.
    pub fn default_portfolio_lifecycle(self) -> AnalyticalSolvePortfolioLifecycle {
        match self {
            Self::VerifiedExecutionModel
            | Self::VerifiedStaticInvariant
            | Self::VerifiedCounterexampleReplay => AnalyticalSolvePortfolioLifecycle::Published,
            Self::StructurallyEligible => AnalyticalSolvePortfolioLifecycle::Admitted,
            Self::StructurallyIneligible => AnalyticalSolvePortfolioLifecycle::Rejected,
            Self::NotAssessed => AnalyticalSolvePortfolioLifecycle::NotSelected,
        }
    }
}

/// Stable reason code for one analytical solve decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticalSolveDecisionReason {
    /// A verified execution-model proof authorizes the analytical answer.
    ExecutionModelVerified,
    /// A verified static-invariant proof authorizes the analytical answer.
    StaticInvariantVerified,
    /// A witness trace or symbolic witness was checked.
    WitnessVerified,
    /// A certificate object was checked.
    CertificateVerified,
    /// Structural facts are useful but insufficient for publication.
    StructuralProofOnly,
    /// The frontend payload or problem is outside the lane's supported fragment.
    UnsupportedFragment,
    /// The lane was not requested for this portfolio.
    NotRequested,
    /// Publication is blocked because no proof/witness/certificate fingerprint exists.
    MissingArtifactFingerprint,
    /// Publication is blocked because validation requirements are missing.
    MissingValidationRequirement,
    /// Publication is blocked because the semantic digest binding the frontend payload is missing.
    MissingSemanticDigest,
    /// Publication is blocked because no validation receipt has been attached.
    MissingValidationReceipt,
    /// Publication is blocked because analytical admission was not fail closed.
    AdmissionNotFailClosed,
    /// Publication is blocked because cache/fingerprint reuse evidence is incomplete.
    MissingCacheFingerprintCompatibility,
    /// Publication is blocked because cache/fingerprint reuse evidence is malformed.
    InvalidCacheFingerprintCompatibility,
    /// Publication is blocked because an attached validation receipt was rejected.
    RejectedValidationReceipt,
    /// Publication is blocked because an attached validation receipt is malformed.
    InvalidValidationReceipt,
    /// The portfolio lifecycle has not admitted publication.
    PortfolioLifecycleBlocked,
}

impl AnalyticalSolveDecisionReason {
    /// Stable lowercase wire code for this reason.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::ExecutionModelVerified => "execution_model_verified",
            Self::StaticInvariantVerified => "static_invariant_verified",
            Self::WitnessVerified => "witness_verified",
            Self::CertificateVerified => "certificate_verified",
            Self::StructuralProofOnly => "structural_proof_only",
            Self::UnsupportedFragment => "unsupported_fragment",
            Self::NotRequested => "not_requested",
            Self::MissingArtifactFingerprint => "missing_artifact_fingerprint",
            Self::MissingValidationRequirement => "missing_validation_requirement",
            Self::MissingSemanticDigest => "missing_semantic_digest",
            Self::MissingValidationReceipt => "missing_validation_receipt",
            Self::AdmissionNotFailClosed => "admission_not_fail_closed",
            Self::MissingCacheFingerprintCompatibility => "missing_cache_fingerprint_compatibility",
            Self::InvalidCacheFingerprintCompatibility => "invalid_cache_fingerprint_compatibility",
            Self::RejectedValidationReceipt => "rejected_validation_receipt",
            Self::InvalidValidationReceipt => "invalid_validation_receipt",
            Self::PortfolioLifecycleBlocked => "portfolio_lifecycle_blocked",
        }
    }
}

/// Receipt-level readiness for analytical publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticalSolveValidationReceiptReadiness {
    /// At least one valid accepted receipt is attached.
    Ready,
    /// A receipt is present but rejected or malformed.
    Blocked,
    /// Publication requirements exist but no receipt has been attached.
    Unknown,
}

impl AnalyticalSolveValidationReceiptReadiness {
    /// Stable lowercase wire code for this readiness state.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Blocked => "blocked",
            Self::Unknown => "unknown",
        }
    }
}

/// Lifecycle state for an analytical candidate inside a shared portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticalSolvePortfolioLifecycle {
    /// Candidate exists but has not been admitted.
    Candidate,
    /// Candidate is admitted but still requires normal verifier execution.
    Admitted,
    /// Candidate is executing or waiting on solver/proof work.
    Running,
    /// Candidate was verified and may publish its analytical answer.
    Published,
    /// Candidate was rejected by policy or validation.
    Rejected,
    /// Candidate was declined by the lane before execution.
    Declined,
    /// A stronger portfolio lane superseded this candidate.
    Shadowed,
    /// The portfolio did not select or assess this candidate.
    NotSelected,
}

impl AnalyticalSolvePortfolioLifecycle {
    /// Stable lowercase wire code for this lifecycle stage.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Admitted => "admitted",
            Self::Running => "running",
            Self::Published => "published",
            Self::Rejected => "rejected",
            Self::Declined => "declined",
            Self::Shadowed => "shadowed",
            Self::NotSelected => "not_selected",
        }
    }

    /// Whether this stage permits publishing the analytical answer
    /// (only [`Published`](Self::Published)).
    #[must_use]
    pub fn can_publish(self) -> bool {
        matches!(self, Self::Published)
    }
}

/// Shared route selected after analytical candidates have attached validation evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticalSolveRoute {
    /// A verified analytical candidate may publish instead of explicit exploration.
    AnalyticalWin,
    /// No analytical candidate may publish; run the configured explicit fallback lane.
    ExplicitFallback,
    /// No analytical candidate may publish and no explicit fallback lane was declared.
    ExplorationPortfolio,
}

impl AnalyticalSolveRoute {
    /// Stable lowercase wire code for this route.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::AnalyticalWin => "analytical_win",
            Self::ExplicitFallback => "explicit_fallback",
            Self::ExplorationPortfolio => "exploration_portfolio",
        }
    }
}

/// Frontend-neutral analytical routing result for a prepared portfolio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalSolveRoutingDecision {
    /// Route selected for the portfolio.
    pub route: AnalyticalSolveRoute,
    /// Index of the winning analytical decision, when one was selected.
    pub analytical_decision_index: Option<usize>,
    /// Candidate key of the winning analytical candidate, when any.
    pub analytical_candidate_key: Option<String>,
    /// Candidate key of the explicit fallback lane, when one was declared.
    pub explicit_fallback_candidate_key: Option<String>,
    /// Reason code explaining the route.
    pub reason: AnalyticalSolveDecisionReason,
}

impl AnalyticalSolveRoutingDecision {
    /// Wire code of the selected [`route`](Self::route).
    #[must_use]
    pub fn route_code(&self) -> &'static str {
        self.route.code()
    }

    /// Whether the route is [`AnalyticalSolveRoute::ExplicitFallback`].
    #[must_use]
    pub fn uses_explicit_fallback(&self) -> bool {
        self.route == AnalyticalSolveRoute::ExplicitFallback
    }

    /// The winning analytical decision from `decisions`, indexed by
    /// [`analytical_decision_index`](Self::analytical_decision_index).
    ///
    /// Returns `None` when no winner was selected or the index is out of range.
    #[must_use]
    pub fn analytical_winner<'a>(
        &self,
        decisions: &'a [AnalyticalSolveDecision],
    ) -> Option<&'a AnalyticalSolveDecision> {
        self.analytical_decision_index
            .and_then(|index| decisions.get(index))
    }
}

/// Frontend-neutral analytical solve decision row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalSolveDecision {
    /// Proof/admission status of this decision.
    pub status: AnalyticalSolveDecisionStatus,
    /// Source/interchange family the candidate came from.
    pub source_kind: CheckerSourceKind,
    /// Prepared-program payload kind being decided.
    pub payload_kind: PreparedProgramPayloadKind,
    /// Problem kind the analytical lane addressed.
    pub problem: ProblemKind,
    /// Execution lane that produced the decision.
    pub lane: SetupTraceLaneKind,
    /// Backend that produced the analytical answer.
    pub backend: BackendKind,
    /// Semantic digest binding the frontend payload, when known.
    pub semantic_digest: Option<String>,
    /// Identity of the prepared program, when known.
    pub prepared_program_identity: Option<String>,
    /// Artifact identity fields for evidence attribution.
    pub identities: CheckerArtifactIdentityFields,
    /// Validation kind actually applied, when any.
    pub validation: Option<PreparedValidationKind>,
    /// Validation kinds required before this decision may publish.
    pub validation_requirements: Vec<PreparedValidationKind>,
    /// Candidate key distinguishing this candidate in the portfolio.
    pub candidate_key: Option<String>,
    /// Portfolio lifecycle stage of the candidate.
    pub portfolio_lifecycle: AnalyticalSolvePortfolioLifecycle,
    /// Portfolio rank, when assigned.
    pub portfolio_rank: Option<u32>,
    /// Portfolio candidate id, when assigned.
    pub portfolio_candidate_id: Option<String>,
    /// Digest of an emitted proof, when one exists.
    pub proof_fingerprint: Option<String>,
    /// Digest of an emitted witness, when one exists.
    pub witness_fingerprint: Option<String>,
    /// Digest of an emitted certificate, when one exists.
    pub certificate_fingerprint: Option<String>,
    /// Validation receipts attached to this decision.
    pub validation_receipts: Vec<ValidationReceipt>,
    /// Artifact kinds a receipt must cover for publication.
    pub validation_artifact_requirements: Vec<ValidationReceiptArtifactKind>,
    /// Required receipt digest algorithm, when constrained.
    pub validation_digest_algorithm_requirement: Option<String>,
    /// Whether analytical admission was performed fail-closed.
    pub admission_fail_closed: bool,
    /// Cache/fingerprint reuse compatibility code (see the `ANALYTICAL_CACHE_FINGERPRINT_*` constants).
    pub cache_fingerprint_compatibility: String,
    /// Reason code for the decision.
    pub decision_reason: AnalyticalSolveDecisionReason,
    /// Optional free-form reason-code override carried in evidence.
    pub reason_code: Option<String>,
}

impl AnalyticalSolveDecision {
    /// Build a baseline decision for the given status, source, payload, and
    /// problem.
    ///
    /// Defaults the lane to analytical, the backend to local symbolic execution,
    /// requires a structural-proof validation, and derives the portfolio
    /// lifecycle and decision reason from `status`.
    pub fn new(
        status: AnalyticalSolveDecisionStatus,
        source_kind: CheckerSourceKind,
        payload_kind: PreparedProgramPayloadKind,
        problem: ProblemKind,
    ) -> Self {
        Self {
            status,
            source_kind,
            payload_kind,
            problem,
            lane: SetupTraceLaneKind::Analytical,
            backend: BackendKind::LocalSymbolicExecution,
            semantic_digest: None,
            prepared_program_identity: None,
            identities: CheckerArtifactIdentityFields::default(),
            validation: Some(PreparedValidationKind::StructuralProof),
            validation_requirements: vec![PreparedValidationKind::StructuralProof],
            candidate_key: None,
            portfolio_lifecycle: status.default_portfolio_lifecycle(),
            portfolio_rank: None,
            portfolio_candidate_id: None,
            proof_fingerprint: None,
            witness_fingerprint: None,
            certificate_fingerprint: None,
            validation_receipts: Vec::new(),
            validation_artifact_requirements: Vec::new(),
            validation_digest_algorithm_requirement: None,
            admission_fail_closed: true,
            cache_fingerprint_compatibility:
                ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_NOT_DECLARED.to_string(),
            decision_reason: status.default_reason(),
            reason_code: None,
        }
    }

    /// Build the shared decision baseline for one prepared analytical solve.
    ///
    /// Prepared descriptors prove structural eligibility only. They must not
    /// fabricate a verified/published analytical result; frontends add proof,
    /// witness, certificate, or replay policy after this baseline is linked.
    pub fn from_prepared_solve(
        program: &PreparedCheckerProgram,
        solve: &PreparedAnalyticalSolveDescriptor,
        lane: Option<&PreparedCandidateLaneDescriptor>,
    ) -> Self {
        let mut decision = Self::new(
            AnalyticalSolveDecisionStatus::StructurallyEligible,
            program.source_kind,
            program.payload_kind,
            solve.problem,
        )
        .with_backend(BackendKind::LocalSymbolicExecution)
        .with_validation(PreparedValidationKind::StructuralProof)
        .with_portfolio_lifecycle(AnalyticalSolvePortfolioLifecycle::Admitted)
        .with_decision_reason(AnalyticalSolveDecisionReason::StructuralProofOnly)
        .with_reason_code("prepared_descriptor_only")
        .with_prepared_program_identity(program.identity.clone())
        .with_portfolio_candidate_id(solve.id.clone());
        if let Some(semantic_digest) = program
            .identities
            .prepared_program_fingerprint
            .as_ref()
            .or(program.identities.source_fingerprint.as_ref())
        {
            decision = decision.with_semantic_digest(semantic_digest.clone());
        }

        if let Some(lane) = lane {
            decision.lane = lane.lane;
            decision.identities = program.effective_candidate_lane_identity_fields(lane);
            if let Some(candidate_key) = lane.candidate_key.as_ref() {
                if !candidate_key.is_empty() {
                    decision.candidate_key = Some(candidate_key.clone());
                }
            }
        } else {
            decision.identities = program.effective_identity_fields();
        }

        decision
    }

    /// Set [`candidate_key`](Self::candidate_key) (empty input is ignored).
    pub fn with_candidate_key(mut self, candidate_key: impl Into<String>) -> Self {
        let candidate_key = candidate_key.into();
        if !candidate_key.is_empty() {
            self.candidate_key = Some(candidate_key);
        }
        self
    }

    /// Set [`backend`](Self::backend).
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set [`semantic_digest`](Self::semantic_digest) (empty input is ignored).
    pub fn with_semantic_digest(mut self, digest: impl Into<String>) -> Self {
        let digest = digest.into();
        if !digest.is_empty() {
            self.semantic_digest = Some(digest);
        }
        self
    }

    /// Set [`prepared_program_identity`](Self::prepared_program_identity) (empty input is ignored).
    pub fn with_prepared_program_identity(mut self, identity: impl Into<String>) -> Self {
        let identity = identity.into();
        if !identity.is_empty() {
            self.prepared_program_identity = Some(identity);
        }
        self
    }

    /// Replace the whole [`identities`](Self::identities) set.
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = identities;
        self
    }

    /// Link this analytical decision to the shared prepared-program and
    /// candidate-lane identities that admitted it.
    pub fn with_prepared_candidate(
        mut self,
        program: &PreparedCheckerProgram,
        lane: &PreparedCandidateLaneDescriptor,
    ) -> Self {
        self.lane = lane.lane;
        self.prepared_program_identity = Some(program.identity.clone());
        self.identities = program.effective_candidate_lane_identity_fields(lane);
        self
    }

    /// Set the applied [`validation`](Self::validation) and add it to the
    /// required set.
    pub fn with_validation(mut self, validation: PreparedValidationKind) -> Self {
        self.validation = Some(validation);
        push_validation_requirement(&mut self.validation_requirements, validation);
        self
    }

    /// Add a required validation kind, defaulting the applied
    /// [`validation`](Self::validation) to it if none is set yet.
    pub fn with_validation_requirement(mut self, validation: PreparedValidationKind) -> Self {
        push_validation_requirement(&mut self.validation_requirements, validation);
        if self.validation.is_none() {
            self.validation = Some(validation);
        }
        self
    }

    /// Replace the required validation set with `validations`, taking the first
    /// as the applied [`validation`](Self::validation).
    pub fn with_validation_requirements<I>(mut self, validations: I) -> Self
    where
        I: IntoIterator<Item = PreparedValidationKind>,
    {
        self.validation_requirements.clear();
        self.validation = None;
        for validation in validations {
            push_validation_requirement(&mut self.validation_requirements, validation);
            if self.validation.is_none() {
                self.validation = Some(validation);
            }
        }
        self
    }

    /// Clear the applied validation and all validation requirements.
    pub fn without_validation(mut self) -> Self {
        self.validation = None;
        self.validation_requirements.clear();
        self
    }

    /// Add a required validation artifact kind.
    pub fn with_validation_artifact_requirement(
        mut self,
        artifact_kind: ValidationReceiptArtifactKind,
    ) -> Self {
        push_validation_artifact_requirement(
            &mut self.validation_artifact_requirements,
            artifact_kind,
        );
        self
    }

    /// Replace the required validation artifact kinds with `artifact_kinds`.
    pub fn with_validation_artifact_requirements<I>(mut self, artifact_kinds: I) -> Self
    where
        I: IntoIterator<Item = ValidationReceiptArtifactKind>,
    {
        self.validation_artifact_requirements.clear();
        for artifact_kind in artifact_kinds {
            push_validation_artifact_requirement(
                &mut self.validation_artifact_requirements,
                artifact_kind,
            );
        }
        self
    }

    /// Constrain the digest algorithm a validating receipt must use (empty clears it).
    pub fn with_validation_digest_algorithm_requirement(
        mut self,
        digest_algorithm: impl Into<String>,
    ) -> Self {
        self.validation_digest_algorithm_requirement = non_empty_string(digest_algorithm.into());
        self
    }

    /// Require an AY-proof shared-engine validation of the given artifact kind,
    /// pinned to the AY fingerprint-identity digest algorithm.
    pub fn with_ay_shared_engine_validation_requirement(
        self,
        artifact_kind: ValidationReceiptArtifactKind,
    ) -> Self {
        self.with_validation_requirement(PreparedValidationKind::AYProof)
            .with_validation_artifact_requirement(artifact_kind)
            .with_validation_digest_algorithm_requirement(
                ANALYTICAL_SOLVE_AY_VALIDATION_DIGEST_ALGORITHM,
            )
    }

    /// Set [`portfolio_lifecycle`](Self::portfolio_lifecycle).
    pub fn with_portfolio_lifecycle(
        mut self,
        lifecycle: AnalyticalSolvePortfolioLifecycle,
    ) -> Self {
        self.portfolio_lifecycle = lifecycle;
        self
    }

    /// Set [`portfolio_rank`](Self::portfolio_rank).
    pub fn with_portfolio_rank(mut self, rank: u32) -> Self {
        self.portfolio_rank = Some(rank);
        self
    }

    /// Set [`portfolio_candidate_id`](Self::portfolio_candidate_id) (empty input is ignored).
    pub fn with_portfolio_candidate_id(mut self, candidate_id: impl Into<String>) -> Self {
        let candidate_id = candidate_id.into();
        if !candidate_id.is_empty() {
            self.portfolio_candidate_id = Some(candidate_id);
        }
        self
    }

    /// Set [`proof_fingerprint`](Self::proof_fingerprint) (empty input is ignored).
    pub fn with_proof_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        let fingerprint = fingerprint.into();
        if !fingerprint.is_empty() {
            self.proof_fingerprint = Some(fingerprint);
        }
        self
    }

    /// Set [`witness_fingerprint`](Self::witness_fingerprint) (empty input is ignored).
    pub fn with_witness_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        let fingerprint = fingerprint.into();
        if !fingerprint.is_empty() {
            self.witness_fingerprint = Some(fingerprint);
        }
        self
    }

    /// Set [`certificate_fingerprint`](Self::certificate_fingerprint) (empty input is ignored).
    pub fn with_certificate_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        let fingerprint = fingerprint.into();
        if !fingerprint.is_empty() {
            self.certificate_fingerprint = Some(fingerprint);
        }
        self
    }

    /// Attach frontend-neutral validation evidence to this analytical decision.
    ///
    /// Accepted proof, witness, and certificate receipts provide the shared
    /// artifact fingerprint needed by publication readiness. Rejected or
    /// malformed receipts are still retained so the decision can describe why
    /// publication remains blocked.
    pub fn with_validation_receipt(mut self, receipt: ValidationReceipt) -> Self {
        if let Some(validation) = prepared_validation_kind_for_receipt(receipt.validator_kind) {
            push_validation_requirement(&mut self.validation_requirements, validation);
            self.validation = Some(validation);
        }

        if receipt.validate().is_ok() && receipt.status.is_accepted() {
            let fingerprint = validation_receipt_fingerprint(&receipt);
            match receipt.validation_artifact_kind {
                ValidationReceiptArtifactKind::Proof => {
                    self.proof_fingerprint = Some(fingerprint);
                }
                ValidationReceiptArtifactKind::Witness => {
                    self.witness_fingerprint = Some(fingerprint);
                }
                ValidationReceiptArtifactKind::Certificate => {
                    self.certificate_fingerprint = Some(fingerprint);
                }
                ValidationReceiptArtifactKind::Artifact => {}
            }
        }

        self.validation_receipts.push(receipt);
        self
    }

    /// Set [`admission_fail_closed`](Self::admission_fail_closed).
    pub fn with_admission_fail_closed(mut self, fail_closed: bool) -> Self {
        self.admission_fail_closed = fail_closed;
        self
    }

    /// Set [`cache_fingerprint_compatibility`](Self::cache_fingerprint_compatibility)
    /// to a raw compatibility code.
    pub fn with_cache_fingerprint_compatibility(
        mut self,
        compatibility: impl Into<String>,
    ) -> Self {
        self.cache_fingerprint_compatibility = compatibility.into();
        self
    }

    /// Declare frontend-reusable cache/fingerprint evidence: set the semantic
    /// digest, cache key, and fingerprint identity, and mark compatibility as
    /// `frontend_reusable`.
    pub fn with_frontend_reusable_cache_fingerprint(
        mut self,
        semantic_digest: impl Into<String>,
        cache_key: impl Into<String>,
        fingerprint_identity: impl Into<String>,
    ) -> Self {
        self.semantic_digest = non_empty_string(semantic_digest.into());
        self.identities.cache_key = non_empty_string(cache_key.into());
        self.identities.fingerprint_identity = non_empty_string(fingerprint_identity.into());
        self.cache_fingerprint_compatibility =
            ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_REUSABLE.to_string();
        self
    }

    /// Set [`decision_reason`](Self::decision_reason).
    pub fn with_decision_reason(mut self, reason: AnalyticalSolveDecisionReason) -> Self {
        self.decision_reason = reason;
        self
    }

    /// Set [`reason_code`](Self::reason_code) (empty or the no-reason sentinel is ignored).
    pub fn with_reason_code(mut self, reason_code: impl Into<String>) -> Self {
        let reason_code = reason_code.into();
        if !reason_code.is_empty() && reason_code != NO_REASON_CODE {
            self.reason_code = Some(reason_code);
        }
        self
    }

    /// Whether this decision's status authorizes skipping explicit-state search.
    pub fn can_preempt_explicit_state(&self) -> bool {
        self.status.can_preempt_explicit_state()
    }

    /// Whether any proof/witness/certificate fingerprint is present.
    pub fn has_publication_artifact_fingerprint(&self) -> bool {
        self.proof_fingerprint.is_some()
            || self.witness_fingerprint.is_some()
            || self.certificate_fingerprint.is_some()
    }

    /// Wire code naming the solver family of this decision's backend.
    pub fn solver_family_code(&self) -> &'static str {
        solver_family_code(self.backend)
    }

    /// Code summarizing the replay-validation authority of the attached receipts.
    pub fn replay_validation_authority_code(&self) -> String {
        replay_validation_authority_value(&self.validation_receipts)
    }

    /// Code summarizing how this decision is admitted into the portfolio:
    /// `analytical_preempt` when publication is unblocked,
    /// `fail_closed_explicit_fallback` when it could preempt but is blocked, or
    /// `candidate_only` otherwise.
    pub fn admission_disposition_code(&self) -> &'static str {
        if self.publication_blocker_reason().is_none() {
            "analytical_preempt"
        } else if self.can_preempt_explicit_state() {
            "fail_closed_explicit_fallback"
        } else {
            "candidate_only"
        }
    }

    /// Compute receipt-level publication readiness from the attached receipts
    /// and the decision's validation requirements.
    ///
    /// Returns [`Blocked`](AnalyticalSolveValidationReceiptReadiness::Blocked)
    /// if any receipt is invalid, rejected, or fails its artifact requirements;
    /// [`Ready`](AnalyticalSolveValidationReceiptReadiness::Ready) when every
    /// requirement is met by an accepted, valid receipt; otherwise
    /// [`Unknown`](AnalyticalSolveValidationReceiptReadiness::Unknown).
    pub fn validation_receipt_readiness(&self) -> AnalyticalSolveValidationReceiptReadiness {
        if self.validation_requirements.is_empty() || self.validation_receipts.is_empty() {
            return AnalyticalSolveValidationReceiptReadiness::Unknown;
        }
        if self
            .validation_receipts
            .iter()
            .any(|receipt| receipt.validate().is_err())
        {
            return AnalyticalSolveValidationReceiptReadiness::Blocked;
        }
        if self
            .validation_receipts
            .iter()
            .any(|receipt| !receipt.status.is_accepted())
        {
            return AnalyticalSolveValidationReceiptReadiness::Blocked;
        }
        if self
            .validation_receipts
            .iter()
            .any(|receipt| !validation_receipt_satisfies_artifact_requirements(receipt, self))
        {
            return AnalyticalSolveValidationReceiptReadiness::Blocked;
        }
        if self.validation_requirements.iter().all(|requirement| {
            self.validation_receipts
                .iter()
                .any(|receipt| validation_receipt_satisfies_requirement(receipt, *requirement))
        }) {
            return AnalyticalSolveValidationReceiptReadiness::Ready;
        }
        AnalyticalSolveValidationReceiptReadiness::Unknown
    }

    /// Wire code of the [`validation_receipt_readiness`](Self::validation_receipt_readiness).
    pub fn validation_receipt_readiness_code(&self) -> &'static str {
        self.validation_receipt_readiness().code()
    }

    /// Reason a receipt blocks publication, or `None` when receipts are ready.
    ///
    /// Maps `Unknown` to `MissingValidationReceipt` and `Blocked` to an
    /// `Invalid`/`Rejected` receipt reason depending on which check failed.
    pub fn validation_receipt_blocker_reason(&self) -> Option<AnalyticalSolveDecisionReason> {
        match self.validation_receipt_readiness() {
            AnalyticalSolveValidationReceiptReadiness::Ready => None,
            AnalyticalSolveValidationReceiptReadiness::Unknown => {
                Some(AnalyticalSolveDecisionReason::MissingValidationReceipt)
            }
            AnalyticalSolveValidationReceiptReadiness::Blocked => {
                if self
                    .validation_receipts
                    .iter()
                    .any(|receipt| receipt.validate().is_err())
                {
                    Some(AnalyticalSolveDecisionReason::InvalidValidationReceipt)
                } else if self.validation_receipts.iter().any(|receipt| {
                    !validation_receipt_satisfies_artifact_requirements(receipt, self)
                }) {
                    Some(AnalyticalSolveDecisionReason::InvalidValidationReceipt)
                } else {
                    Some(AnalyticalSolveDecisionReason::RejectedValidationReceipt)
                }
            }
        }
    }

    /// The first reason blocking analytical publication, or `None` if the
    /// decision may publish its analytical answer.
    ///
    /// Checks, in order: status preemption, portfolio lifecycle, fail-closed
    /// admission, presence of a validation requirement, receipt readiness,
    /// semantic digest, cache/fingerprint compatibility, an artifact
    /// fingerprint, and finally any still-unknown receipt readiness.
    pub fn publication_blocker_reason(&self) -> Option<AnalyticalSolveDecisionReason> {
        if !self.can_preempt_explicit_state() {
            return Some(self.decision_reason);
        }
        if !self.portfolio_lifecycle.can_publish() {
            return Some(AnalyticalSolveDecisionReason::PortfolioLifecycleBlocked);
        }
        if !self.admission_fail_closed {
            return Some(AnalyticalSolveDecisionReason::AdmissionNotFailClosed);
        }
        if self.validation_requirements.is_empty() {
            return Some(AnalyticalSolveDecisionReason::MissingValidationRequirement);
        }
        if self.validation_receipt_readiness() == AnalyticalSolveValidationReceiptReadiness::Blocked
        {
            return self.validation_receipt_blocker_reason();
        }
        if !has_identity(self.semantic_digest.as_deref()) {
            return Some(AnalyticalSolveDecisionReason::MissingSemanticDigest);
        }
        if let Some(reason) = cache_fingerprint_compatibility_blocker_reason(self) {
            return Some(reason);
        }
        if !self.has_publication_artifact_fingerprint() {
            return Some(AnalyticalSolveDecisionReason::MissingArtifactFingerprint);
        }
        if self.validation_receipt_readiness() == AnalyticalSolveValidationReceiptReadiness::Unknown
        {
            return self.validation_receipt_blocker_reason();
        }
        None
    }

    /// `"ready"` when [`publication_blocker_reason`](Self::publication_blocker_reason)
    /// is `None`, otherwise `"blocked"`.
    pub fn publication_readiness_code(&self) -> &'static str {
        if self.publication_blocker_reason().is_none() {
            "ready"
        } else {
            "blocked"
        }
    }

    /// Render a stable analytical-solve decision evidence row, prefixed by `scope`.
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let publication_blocker = self
            .publication_blocker_reason()
            .map(AnalyticalSolveDecisionReason::code)
            .unwrap_or(NO_REASON_CODE);
        format!(
            "{} analytical_solve_decision source_kind={} frontend_kind={} frontend_family={} payload_kind={} problem={} lane_kind={} lane={} backend={} backend_code={} solver_family={} semantic_digest={} prepared_program_identity={} cache_key={} frontend_payload_identity={} artifact_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} cache_fingerprint_compatibility={} batch_artifact_identity={} candidate_identity={} lane_identity={} decision_status={} status={} publication_status={} explicit_state_relation={} validation={} validation_requirements={} replay_validation_authority={} admission_fail_closed={} admission_disposition={} validation_receipt_readiness={} validation_receipt_identities={} validation_receipt_failures={} candidate_key={} portfolio_lifecycle={} portfolio_rank={} portfolio_candidate_id={} proof_fingerprint={} witness_fingerprint={} certificate_fingerprint={} publication_readiness={} publication_blocker={} decision_reason={} reason_code={}",
            scope,
            self.source_kind.code(),
            self.source_kind.frontend_family_code(),
            self.source_kind.frontend_family_code(),
            self.payload_kind.code(),
            self.problem.code(),
            self.lane.code(),
            self.lane.code(),
            self.backend.name(),
            self.backend.code(),
            self.solver_family_code(),
            evidence_optional(self.semantic_digest.as_deref()),
            evidence_optional(self.prepared_program_identity.as_deref()),
            evidence_optional(self.identities.cache_key.as_deref()),
            evidence_optional(self.identities.frontend_payload_identity.as_deref()),
            evidence_optional(self.identities.artifact_identity.as_deref()),
            evidence_optional(self.identities.storage_policy_identity.as_deref()),
            evidence_optional(self.identities.fingerprint_policy_identity.as_deref()),
            evidence_optional(self.identities.fingerprint_identity.as_deref()),
            evidence_value(&self.cache_fingerprint_compatibility),
            evidence_optional(self.identities.batch_artifact_identity.as_deref()),
            evidence_optional(self.identities.candidate_identity.as_deref()),
            evidence_optional(self.identities.lane_identity.as_deref()),
            self.status.code(),
            self.status.name(),
            self.status.publication_status_code(),
            self.status.explicit_state_relation_code(),
            self.validation
                .map(PreparedValidationKind::code)
                .unwrap_or(NO_REASON_CODE),
            validation_requirements_value(&self.validation_requirements),
            self.replay_validation_authority_code(),
            self.admission_fail_closed,
            self.admission_disposition_code(),
            self.validation_receipt_readiness_code(),
            validation_receipt_identities_value(&self.validation_receipts),
            validation_receipt_failures_value(&self.validation_receipts),
            evidence_optional(self.candidate_key.as_deref()),
            self.portfolio_lifecycle.code(),
            self.portfolio_rank
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| NO_REASON_CODE.to_string()),
            evidence_optional(self.portfolio_candidate_id.as_deref()),
            evidence_optional(self.proof_fingerprint.as_deref()),
            evidence_optional(self.witness_fingerprint.as_deref()),
            evidence_optional(self.certificate_fingerprint.as_deref()),
            self.publication_readiness_code(),
            publication_blocker,
            self.decision_reason.code(),
            self.reason_code.as_deref().unwrap_or(NO_REASON_CODE),
        )
    }

    /// Render one shared-engine validation-receipt evidence row per attached
    /// receipt, prefixed by `scope`.
    #[must_use]
    pub fn render_shared_engine_validation_receipt_evidence_rows(
        &self,
        scope: &str,
    ) -> Vec<String> {
        self.validation_receipts
            .iter()
            .map(|receipt| {
                let receipt_valid = receipt.validate().is_ok();
                let requirement_valid =
                    validation_receipt_satisfies_artifact_requirements(receipt, self);
                let receipt_validation = if receipt_valid && receipt.status.is_accepted() && requirement_valid {
                    "accepted"
                } else {
                    "invalid"
                };
                let publication_blocker = self
                    .publication_blocker_reason()
                    .map(AnalyticalSolveDecisionReason::code)
                    .unwrap_or(NO_REASON_CODE);
                let expected_artifact = validation_artifact_requirements_value(
                    &self.validation_artifact_requirements,
                );
                let expected_digest = self
                    .validation_digest_algorithm_requirement
                    .as_deref()
                    .unwrap_or(NO_REASON_CODE);
                format!(
                    "{} {} schema={} source_kind={} payload_kind={} origin_frontend={} model_check_search=false search_kind=analytical_solve backend_code={} solver_family={} semantic_digest={} prepared_program_identity={} candidate_key={} validation={} validation_requirements={} validation_artifact_requirements={} digest_algorithm_requirement={} validation_receipt_readiness={} receipt_validation={} receipt_validator={} validation_artifact_kind={} validation_artifact_kind_expected={}_actual={} digest_algorithm={} digest_algorithm_expected={}_actual={} validation_receipt_identity={} validation_artifact_identity={} publication_readiness={} publication_blocker={} fail_closed={} consumable_frontend_families={}",
                    scope,
                    ANALYTICAL_SOLVE_SHARED_ENGINE_VALIDATION_RECEIPT_ROW_KIND,
                    ANALYTICAL_SOLVE_SHARED_ENGINE_VALIDATION_RECEIPT_SCHEMA,
                    self.source_kind.code(),
                    self.payload_kind.code(),
                    self.source_kind.frontend_family_code(),
                    self.backend.code(),
                    self.solver_family_code(),
                    evidence_optional(self.semantic_digest.as_deref()),
                    evidence_optional(self.prepared_program_identity.as_deref()),
                    evidence_optional(self.candidate_key.as_deref()),
                    self.validation
                        .map(PreparedValidationKind::code)
                        .unwrap_or(NO_REASON_CODE),
                    validation_requirements_value(&self.validation_requirements),
                    expected_artifact,
                    evidence_value(expected_digest),
                    self.validation_receipt_readiness_code(),
                    receipt_validation,
                    receipt.validator_kind.code(),
                    receipt.validation_artifact_kind.code(),
                    expected_artifact,
                    receipt.validation_artifact_kind.code(),
                    evidence_value(receipt.digest_algorithm.as_str()),
                    evidence_value(expected_digest),
                    evidence_value(receipt.digest_algorithm.as_str()),
                    shared_engine_validation_receipt_identity(receipt),
                    evidence_value(receipt.validation_artifact_identity.as_str()),
                    self.publication_readiness_code(),
                    publication_blocker,
                    self.admission_fail_closed,
                    shared_engine_validation_receipt_frontend_families(),
                )
            })
            .collect()
    }
}

/// Select the shared analytical route after proof/replay receipts have been attached.
///
/// The helper is intentionally fail-closed: any publication blocker on the best
/// analytical candidate routes to the explicit fallback when one is declared.
#[must_use]
pub fn choose_analytical_solve_route(
    decisions: &[AnalyticalSolveDecision],
    explicit_fallback_candidate_key: Option<&str>,
) -> AnalyticalSolveRoutingDecision {
    if let Some((index, decision)) = publishable_analytical_decision(decisions) {
        return AnalyticalSolveRoutingDecision {
            route: AnalyticalSolveRoute::AnalyticalWin,
            analytical_decision_index: Some(index),
            analytical_candidate_key: decision.candidate_key.clone(),
            explicit_fallback_candidate_key: None,
            reason: decision.decision_reason,
        };
    }

    let reason =
        first_publication_blocker(decisions).unwrap_or(AnalyticalSolveDecisionReason::NotRequested);
    let explicit_fallback_candidate_key = explicit_fallback_candidate_key
        .filter(|candidate_key| !candidate_key.is_empty())
        .map(ToOwned::to_owned);
    let route = if explicit_fallback_candidate_key.is_some() {
        AnalyticalSolveRoute::ExplicitFallback
    } else {
        AnalyticalSolveRoute::ExplorationPortfolio
    };

    AnalyticalSolveRoutingDecision {
        route,
        analytical_decision_index: None,
        analytical_candidate_key: first_analytical_candidate_key(decisions),
        explicit_fallback_candidate_key,
        reason,
    }
}

fn publishable_analytical_decision(
    decisions: &[AnalyticalSolveDecision],
) -> Option<(usize, &AnalyticalSolveDecision)> {
    decisions
        .iter()
        .enumerate()
        .filter(|(_, decision)| decision.publication_blocker_reason().is_none())
        .min_by_key(|(index, decision)| {
            (
                decision.portfolio_rank.unwrap_or(u32::MAX),
                decision.candidate_key.as_deref().unwrap_or(NO_REASON_CODE),
                *index,
            )
        })
}

fn first_publication_blocker(
    decisions: &[AnalyticalSolveDecision],
) -> Option<AnalyticalSolveDecisionReason> {
    decisions
        .iter()
        .find_map(AnalyticalSolveDecision::publication_blocker_reason)
}

fn first_analytical_candidate_key(decisions: &[AnalyticalSolveDecision]) -> Option<String> {
    decisions
        .iter()
        .filter_map(|decision| decision.candidate_key.as_ref())
        .find(|candidate_key| !candidate_key.is_empty())
        .cloned()
}

fn push_validation_requirement(
    validations: &mut Vec<PreparedValidationKind>,
    validation: PreparedValidationKind,
) {
    if validations.contains(&validation) {
        return;
    }
    validations.push(validation);
    validations.sort_by(|left, right| left.code().cmp(right.code()));
}

fn push_validation_artifact_requirement(
    artifacts: &mut Vec<ValidationReceiptArtifactKind>,
    artifact: ValidationReceiptArtifactKind,
) {
    if artifacts.contains(&artifact) {
        return;
    }
    artifacts.push(artifact);
    artifacts.sort_by(|left, right| left.code().cmp(right.code()));
}

fn prepared_validation_kind_for_receipt(
    validator: ValidationReceiptValidatorKind,
) -> Option<PreparedValidationKind> {
    match validator {
        ValidationReceiptValidatorKind::Selftest => Some(PreparedValidationKind::Selftest),
        ValidationReceiptValidatorKind::TraceReplay => Some(PreparedValidationKind::TraceReplay),
        ValidationReceiptValidatorKind::WitnessReplay => {
            Some(PreparedValidationKind::WitnessReplay)
        }
        ValidationReceiptValidatorKind::CompleteGraph => {
            Some(PreparedValidationKind::CompleteGraph)
        }
        ValidationReceiptValidatorKind::SccCertificate => {
            Some(PreparedValidationKind::SccCertificate)
        }
        ValidationReceiptValidatorKind::AcceptingCycleCertificate => {
            Some(PreparedValidationKind::AcceptingCycleCertificate)
        }
        ValidationReceiptValidatorKind::StructuralProof => {
            Some(PreparedValidationKind::StructuralProof)
        }
        ValidationReceiptValidatorKind::AYProof => Some(PreparedValidationKind::AYProof),
        ValidationReceiptValidatorKind::OutputFormat => Some(PreparedValidationKind::OutputFormat),
        ValidationReceiptValidatorKind::CertificateValidation
        | ValidationReceiptValidatorKind::ProofReplay => None,
    }
}

fn validation_receipt_satisfies_requirement(
    receipt: &ValidationReceipt,
    requirement: PreparedValidationKind,
) -> bool {
    if receipt.validate().is_err() || !receipt.status.is_accepted() {
        return false;
    }
    let Some(receipt_validation) = prepared_validation_kind_for_receipt(receipt.validator_kind)
    else {
        return false;
    };
    receipt_validation == requirement
        || (requirement == PreparedValidationKind::StructuralProof
            && receipt_validation == PreparedValidationKind::AYProof)
}

fn validation_receipt_satisfies_artifact_requirements(
    receipt: &ValidationReceipt,
    decision: &AnalyticalSolveDecision,
) -> bool {
    if receipt.validate().is_err() {
        return false;
    }
    if !decision.validation_artifact_requirements.is_empty()
        && !decision
            .validation_artifact_requirements
            .contains(&receipt.validation_artifact_kind)
    {
        return false;
    }
    if let Some(expected) = decision.validation_digest_algorithm_requirement.as_deref() {
        if receipt.digest_algorithm != expected {
            return false;
        }
    }
    true
}

fn validation_receipt_fingerprint(receipt: &ValidationReceipt) -> String {
    format!("{}:{}", receipt.digest_algorithm, receipt.digest)
}

fn cache_fingerprint_compatibility_blocker_reason(
    decision: &AnalyticalSolveDecision,
) -> Option<AnalyticalSolveDecisionReason> {
    match decision.cache_fingerprint_compatibility.trim() {
        ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_NOT_DECLARED
        | ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_LOCAL_ONLY => None,
        ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_REUSABLE => {
            if has_identity(decision.identities.cache_key.as_deref())
                && has_identity(decision.identities.fingerprint_identity.as_deref())
            {
                None
            } else {
                Some(AnalyticalSolveDecisionReason::MissingCacheFingerprintCompatibility)
            }
        }
        "" | "none" => Some(AnalyticalSolveDecisionReason::MissingCacheFingerprintCompatibility),
        _ => Some(AnalyticalSolveDecisionReason::InvalidCacheFingerprintCompatibility),
    }
}

fn solver_family_code(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::ExternalAYBinary
        | BackendKind::AYSmt
        | BackendKind::AYSat
        | BackendKind::AYChc => "ay",
        BackendKind::LocalSymbolicExecution => "local_symbolic",
        BackendKind::ExplicitState => "explicit_state",
        BackendKind::NativeKernel => "native_kernel",
        BackendKind::AigerPortfolio | BackendKind::Btor2Portfolio => "hardware_portfolio",
    }
}

fn replay_validation_authority_value(receipts: &[ValidationReceipt]) -> String {
    let mut authorities = receipts
        .iter()
        .filter(|receipt| receipt.validate().is_ok() && receipt.status.is_accepted())
        .map(|receipt| receipt.validator_kind.code())
        .collect::<Vec<_>>();
    authorities.sort_unstable();
    authorities.dedup();
    if authorities.is_empty() {
        NO_REASON_CODE.to_string()
    } else {
        authorities.join(",")
    }
}

fn validation_receipt_identities_value(receipts: &[ValidationReceipt]) -> String {
    if receipts.is_empty() {
        return NO_REASON_CODE.to_string();
    }
    receipts
        .iter()
        .map(|receipt| {
            evidence_value(&format!(
                "{}:{}:{}:{}:{}:{}:{}:{}",
                receipt.validator_kind.code(),
                receipt.status.code(),
                receipt.validation_artifact_kind.code(),
                receipt.digest_algorithm.as_str(),
                receipt.digest.as_str(),
                receipt.prepared_program_identity.as_str(),
                receipt.candidate_identity.as_str(),
                receipt.validation_artifact_identity.as_str()
            ))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn validation_receipt_failures_value(receipts: &[ValidationReceipt]) -> String {
    let failures = receipts
        .iter()
        .filter_map(|receipt| receipt.failure_reason.as_deref())
        .map(evidence_value)
        .collect::<Vec<_>>();
    if failures.is_empty() {
        NO_REASON_CODE.to_string()
    } else {
        failures.join(",")
    }
}

fn validation_requirements_value(validations: &[PreparedValidationKind]) -> String {
    if validations.is_empty() {
        return NO_REASON_CODE.to_string();
    }
    validations
        .iter()
        .map(|validation| validation.code())
        .collect::<Vec<_>>()
        .join(",")
}

fn validation_artifact_requirements_value(artifacts: &[ValidationReceiptArtifactKind]) -> String {
    if artifacts.is_empty() {
        return NO_REASON_CODE.to_string();
    }
    artifacts
        .iter()
        .map(|artifact| artifact.code())
        .collect::<Vec<_>>()
        .join(",")
}

fn shared_engine_validation_receipt_identity(receipt: &ValidationReceipt) -> String {
    evidence_value(&format!(
        "{}:{}:{}:{}:{}",
        receipt.validator_kind.code(),
        receipt.validation_artifact_kind.code(),
        receipt.digest_algorithm,
        receipt.digest,
        receipt.validation_artifact_identity,
    ))
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
    .iter()
    .map(|family| family.code())
    .collect::<Vec<_>>()
    .join(",")
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .map(evidence_value)
        .unwrap_or_else(|| NO_REASON_CODE.to_string())
}

fn evidence_value(value: &str) -> String {
    value.replace(char::is_whitespace, "_")
}

fn has_identity(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|value| !value.is_empty() && value != NO_REASON_CODE)
}

fn non_empty_string(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn shared_engine_receipt_fields(row: &str) -> Option<BTreeMap<&str, &str>> {
        let mut fields = BTreeMap::new();
        for token in row.split_whitespace().skip(2) {
            let (key, value) = token.split_once('=')?;
            fields.insert(key, value);
        }
        Some(fields)
    }

    fn analytical_receipt_row_is_strictly_ready(row: &str) -> bool {
        let Some(fields) = shared_engine_receipt_fields(row) else {
            return false;
        };
        fields.get("schema") == Some(&ANALYTICAL_SOLVE_SHARED_ENGINE_VALIDATION_RECEIPT_SCHEMA)
            && fields.get("model_check_search") == Some(&"false")
            && fields.get("search_kind") == Some(&"analytical_solve")
            && fields.get("validation_receipt_readiness") == Some(&"ready")
            && fields.get("receipt_validation") == Some(&"accepted")
            && fields.get("publication_readiness") == Some(&"ready")
            && fields.get("publication_blocker") == Some(&NO_REASON_CODE)
            && fields.get("fail_closed") == Some(&"true")
    }

    #[test]
    fn verified_analytical_decision_preempts_explicit_state_search() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            CheckerSourceKind::Tla,
            PreparedProgramPayloadKind::Tla,
            ProblemKind::StateSpace,
        )
        .with_candidate_key("analytical")
        .with_semantic_digest("semantic tla digest")
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "abc",
            "prepared tla program",
            "analytical",
            ValidationReceiptArtifactKind::Proof,
            "ay analytical proof",
        ));

        assert!(decision.can_preempt_explicit_state());
        assert!(decision.has_publication_artifact_fingerprint());
        assert_eq!(decision.validation_receipt_readiness_code(), "ready");
        assert_eq!(decision.publication_readiness_code(), "ready");
        let row = decision.render_evidence_row("TY");
        assert!(row.contains("analytical_solve_decision"));
        assert!(row.contains("decision_status=verified_execution_model"));
        assert!(row.contains("frontend_kind=tla_plus"));
        assert!(row.contains("frontend_family=tla_plus"));
        assert!(row.contains("solver_family=local_symbolic"));
        assert!(row.contains("semantic_digest=semantic_tla_digest"));
        assert!(row.contains("publication_status=proof_verified"));
        assert!(row.contains("explicit_state_relation=preempts_explicit_state_search"));
        assert!(row.contains("lane_kind=analytical"));
        assert!(row.contains("candidate_key=analytical"));
        assert!(row.contains("portfolio_lifecycle=published"));
        assert!(row.contains("validation=ay_proof"));
        assert!(row.contains("proof_fingerprint=sha256:abc"));
        assert!(row.contains("validation_requirements=ay_proof,structural_proof"));
        assert!(row.contains("replay_validation_authority=ay_proof"));
        assert!(row.contains("admission_fail_closed=true"));
        assert!(row.contains("admission_disposition=analytical_preempt"));
        assert!(row.contains("cache_fingerprint_compatibility=not_declared"));
        assert!(row.contains("validation_receipt_readiness=ready"));
        assert!(row.contains(
            "validation_receipt_identities=ay_proof:accepted:proof:sha256:abc:prepared_tla_program:analytical:ay_analytical_proof"
        ));
        assert!(row.contains("publication_readiness=ready"));
        assert!(row.contains("publication_blocker=none"));
    }

    #[test]
    fn structural_only_decision_requires_normal_verifier_lane() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::StructurallyEligible,
            CheckerSourceKind::Quint,
            PreparedProgramPayloadKind::Quint,
            ProblemKind::Invariant,
        );

        assert!(!decision.can_preempt_explicit_state());
        assert_eq!(decision.publication_readiness_code(), "blocked");
        let row = decision.render_evidence_row("TY");
        assert!(row.contains("source_kind=quint"));
        assert!(row.contains("payload_kind=quint"));
        assert!(row.contains("decision_status=structurally_eligible"));
        assert!(row.contains("publication_status=requires_explicit_state_verifier"));
        assert!(row.contains("explicit_state_relation=requires_explicit_state_verifier"));
        assert!(row.contains("portfolio_lifecycle=admitted"));
        assert!(row.contains("decision_reason=structural_proof_only"));
        assert!(row.contains("publication_blocker=structural_proof_only"));
    }

    #[test]
    fn replay_verified_counterexample_status_can_preempt_explicit_state() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
            CheckerSourceKind::Aiger,
            PreparedProgramPayloadKind::Aiger,
            ProblemKind::Safety,
        )
        .with_candidate_key("ay_pdr")
        .with_semantic_digest("aiger transition semantic digest")
        .with_validation_requirements([PreparedValidationKind::WitnessReplay])
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::WitnessReplay,
            "sha256",
            "trace-ok",
            "prepared aiger program",
            "ay_pdr",
            ValidationReceiptArtifactKind::Witness,
            "replayed counterexample",
        ));

        assert!(decision.can_preempt_explicit_state());
        assert_eq!(decision.publication_readiness_code(), "ready");

        let row = decision.render_evidence_row("AIGER");
        assert!(row.contains("decision_status=verified_counterexample_replay"));
        assert!(row.contains("publication_status=witness_replayed"));
        assert!(row.contains("explicit_state_relation=preempts_explicit_state_search"));
        assert!(row.contains("decision_reason=witness_verified"));
        assert!(row.contains("witness_fingerprint=sha256:trace-ok"));
        assert!(row.contains("semantic_digest=aiger_transition_semantic_digest"));
        assert!(row.contains("replay_validation_authority=witness_replay"));
    }

    #[test]
    fn analytical_routing_fails_closed_to_explicit_fallback_without_receipt() {
        let decisions = vec![AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            ProblemKind::ExplicitReachability,
        )
        .with_candidate_key("ay_symbolic")
        .with_semantic_digest("petri marking semantic digest")
        .with_validation_requirements([PreparedValidationKind::WitnessReplay])
        .with_witness_fingerprint("sha256:witness")
        .with_portfolio_rank(1)];

        let route = choose_analytical_solve_route(&decisions, Some("explicit_bfs"));

        assert_eq!(route.route, AnalyticalSolveRoute::ExplicitFallback);
        assert_eq!(route.route_code(), "explicit_fallback");
        assert!(route.uses_explicit_fallback());
        assert_eq!(route.analytical_decision_index, None);
        assert_eq!(
            route.analytical_candidate_key.as_deref(),
            Some("ay_symbolic")
        );
        assert_eq!(
            route.explicit_fallback_candidate_key.as_deref(),
            Some("explicit_bfs")
        );
        assert_eq!(
            route.reason,
            AnalyticalSolveDecisionReason::MissingValidationReceipt
        );
        assert!(route.analytical_winner(&decisions).is_none());
    }

    #[test]
    fn analytical_routing_accepts_verified_witness_preemption() {
        let decisions = vec![
            AnalyticalSolveDecision::new(
                AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
                CheckerSourceKind::Btor2,
                PreparedProgramPayloadKind::Btor2,
                ProblemKind::Chc,
            )
            .with_candidate_key("ay_pdr")
            .with_semantic_digest("btor2 semantic digest")
            .with_validation_requirements([PreparedValidationKind::WitnessReplay])
            .with_portfolio_rank(4)
            .with_validation_receipt(ValidationReceipt::accepted(
                ValidationReceiptValidatorKind::WitnessReplay,
                "sha256",
                "btor2-witness",
                "prepared btor2 program",
                "ay_pdr",
                ValidationReceiptArtifactKind::Witness,
                "validated btor2 witness",
            )),
            AnalyticalSolveDecision::new(
                AnalyticalSolveDecisionStatus::StructurallyEligible,
                CheckerSourceKind::Btor2,
                PreparedProgramPayloadKind::Btor2,
                ProblemKind::Safety,
            )
            .with_candidate_key("explicit_bfs")
            .with_portfolio_rank(1),
        ];

        let route = choose_analytical_solve_route(&decisions, Some("explicit_bfs"));

        assert_eq!(route.route, AnalyticalSolveRoute::AnalyticalWin);
        assert_eq!(route.route_code(), "analytical_win");
        assert!(!route.uses_explicit_fallback());
        assert_eq!(route.analytical_decision_index, Some(0));
        assert_eq!(route.analytical_candidate_key.as_deref(), Some("ay_pdr"));
        assert_eq!(route.explicit_fallback_candidate_key, None);
        assert_eq!(route.reason, AnalyticalSolveDecisionReason::WitnessVerified);
        assert_eq!(
            route
                .analytical_winner(&decisions)
                .map(AnalyticalSolveDecision::publication_readiness_code),
            Some("ready")
        );
    }

    #[test]
    fn analytical_solve_records_frontend_neutral_admission_contract() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            CheckerSourceKind::AYOnly,
            PreparedProgramPayloadKind::AYOnly,
            ProblemKind::Smt,
        )
        .with_backend(BackendKind::AYSmt)
        .with_candidate_key("ay_smt")
        .with_frontend_reusable_cache_fingerprint(
            "solver semantic digest",
            "canonical solver cache",
            "solver object fingerprint namespace",
        )
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "solver-proof",
            "prepared ay program",
            "ay_smt",
            ValidationReceiptArtifactKind::Proof,
            "ay solver proof",
        ));

        assert_eq!(decision.publication_readiness_code(), "ready");
        assert_eq!(decision.solver_family_code(), "ay");
        assert_eq!(decision.replay_validation_authority_code(), "ay_proof");
        assert_eq!(decision.admission_disposition_code(), "analytical_preempt");

        let row = decision.render_evidence_row("CORE");
        assert!(row.contains("source_kind=ay_only"));
        assert!(row.contains("frontend_kind=ay_analytical"));
        assert!(row.contains("frontend_family=ay_analytical"));
        assert!(row.contains("backend_code=ay_smt"));
        assert!(row.contains("solver_family=ay"));
        assert!(row.contains("semantic_digest=solver_semantic_digest"));
        assert!(row.contains("cache_key=canonical_solver_cache"));
        assert!(row.contains("fingerprint_identity=solver_object_fingerprint_namespace"));
        assert!(row.contains("cache_fingerprint_compatibility=frontend_reusable"));
        assert!(row.contains("replay_validation_authority=ay_proof"));
        assert!(row.contains("admission_fail_closed=true"));
        assert!(row.contains("admission_disposition=analytical_preempt"));
        assert!(row.contains("publication_readiness=ready"));
    }

    #[test]
    fn analytical_solve_fails_closed_without_semantic_digest() {
        let decisions = vec![AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedStaticInvariant,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            ProblemKind::Invariant,
        )
        .with_candidate_key("petri_linear_invariant")
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "petri-proof",
            "prepared petri program",
            "petri_linear_invariant",
            ValidationReceiptArtifactKind::Proof,
            "petri invariant proof",
        ))];

        assert_eq!(
            decisions[0].publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingSemanticDigest)
        );

        let route = choose_analytical_solve_route(&decisions, Some("explicit_petri_bfs"));
        assert_eq!(route.route, AnalyticalSolveRoute::ExplicitFallback);
        assert_eq!(
            route.reason,
            AnalyticalSolveDecisionReason::MissingSemanticDigest
        );

        let row = decisions[0].render_evidence_row("MCC");
        assert!(row.contains("semantic_digest=none"));
        assert!(row.contains("admission_disposition=fail_closed_explicit_fallback"));
        assert!(row.contains("publication_blocker=missing_semantic_digest"));
    }

    #[test]
    fn analytical_solve_frontend_reusable_cache_requires_fingerprint_compatibility() {
        let decisions = vec![AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
            CheckerSourceKind::Btor2,
            PreparedProgramPayloadKind::Btor2,
            ProblemKind::Chc,
        )
        .with_backend(BackendKind::AYChc)
        .with_candidate_key("ay_pdr")
        .with_semantic_digest("btor2 transition semantic digest")
        .with_cache_fingerprint_compatibility(
            ANALYTICAL_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_REUSABLE,
        )
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::WitnessReplay,
            "sha256",
            "btor2-witness",
            "prepared btor2 program",
            "ay_pdr",
            ValidationReceiptArtifactKind::Witness,
            "validated btor2 witness",
        ))];

        assert_eq!(
            decisions[0].publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingCacheFingerprintCompatibility)
        );

        let route = choose_analytical_solve_route(&decisions, Some("explicit_btor2"));
        assert_eq!(route.route, AnalyticalSolveRoute::ExplicitFallback);
        assert_eq!(
            route.reason,
            AnalyticalSolveDecisionReason::MissingCacheFingerprintCompatibility
        );

        let row = decisions[0].render_evidence_row("BTOR2");
        assert!(row.contains("solver_family=ay"));
        assert!(row.contains("cache_fingerprint_compatibility=frontend_reusable"));
        assert!(row.contains("publication_blocker=missing_cache_fingerprint_compatibility"));
    }

    #[test]
    fn analytical_solve_non_fail_closed_admission_cannot_publish() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            CheckerSourceKind::Tla,
            PreparedProgramPayloadKind::Tla,
            ProblemKind::StateSpace,
        )
        .with_candidate_key("closed_form_cardinality")
        .with_semantic_digest("tla semantic digest")
        .with_admission_fail_closed(false)
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "cardinality-proof",
            "prepared tla program",
            "closed_form_cardinality",
            ValidationReceiptArtifactKind::Proof,
            "closed form proof",
        ));

        assert_eq!(
            decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::AdmissionNotFailClosed)
        );

        let row = decision.render_evidence_row("TY");
        assert!(row.contains("admission_fail_closed=false"));
        assert!(row.contains("admission_disposition=fail_closed_explicit_fallback"));
        assert!(row.contains("publication_blocker=admission_not_fail_closed"));
    }

    #[test]
    fn decision_links_prepared_candidate_lane_identity() {
        let program = PreparedCheckerProgram::new(
            "model#analytical",
            PreparedProgramPayloadKind::MccPetri,
            crate::prepared_program::PreparedStorageKind::PetriMarking,
        )
        .with_artifact_identity("prepared artifact")
        .with_fingerprint_identity("marking fingerprint")
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("mcc-structural", SetupTraceLaneKind::Analytical)
                .with_candidate_key("analytical")
                .with_candidate_identity("candidate structural")
                .with_lane_identity("lane analytical")
                .with_fingerprint_identity("proof fingerprint namespace"),
        );

        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::StructurallyEligible,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            ProblemKind::Invariant,
        )
        .with_prepared_candidate(&program, &program.candidate_lanes[0])
        .with_candidate_key("analytical")
        .with_portfolio_candidate_id("mcc-structural");

        assert_eq!(
            decision.prepared_program_identity.as_deref(),
            Some("model#analytical")
        );
        assert_eq!(
            decision.identities.candidate_identity.as_deref(),
            Some("candidate structural")
        );

        let row = decision.render_evidence_row("MCC");
        assert!(row.contains("prepared_program_identity=model#analytical"));
        assert!(row.contains("artifact_identity=prepared_artifact"));
        assert!(row.contains("fingerprint_identity=proof_fingerprint_namespace"));
        assert!(row.contains("candidate_identity=candidate_structural"));
        assert!(row.contains("lane_identity=lane_analytical"));
        assert!(row.contains("portfolio_candidate_id=mcc-structural"));
    }

    #[test]
    fn decision_from_prepared_solve_sets_shared_identity_baseline() {
        let program = PreparedCheckerProgram::new(
            "model#analytical",
            PreparedProgramPayloadKind::MccPetri,
            crate::prepared_program::PreparedStorageKind::PetriMarking,
        )
        .with_artifact_identity("prepared artifact")
        .with_fingerprint_identity("marking fingerprint")
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("mcc-structural", SetupTraceLaneKind::Analytical)
                .with_candidate_key("analytical")
                .with_candidate_identity("candidate structural")
                .with_lane_identity("lane analytical")
                .with_fingerprint_identity("proof fingerprint namespace"),
        );
        let solve = PreparedAnalyticalSolveDescriptor::new(
            "mcc.solve.linear",
            crate::prepared_program::PreparedAnalyticalSolveKind::LinearInvariant,
            ProblemKind::Invariant,
        );

        let decision = AnalyticalSolveDecision::from_prepared_solve(
            &program,
            &solve,
            Some(&program.candidate_lanes[0]),
        );

        assert_eq!(decision.source_kind, CheckerSourceKind::MccPetri);
        assert_eq!(decision.payload_kind, PreparedProgramPayloadKind::MccPetri);
        assert_eq!(decision.problem, ProblemKind::Invariant);
        assert_eq!(decision.lane, SetupTraceLaneKind::Analytical);
        assert_eq!(
            decision.status,
            AnalyticalSolveDecisionStatus::StructurallyEligible
        );
        assert_eq!(
            decision.decision_reason,
            AnalyticalSolveDecisionReason::StructuralProofOnly
        );
        assert_eq!(
            decision.reason_code.as_deref(),
            Some("prepared_descriptor_only")
        );
        assert_eq!(decision.candidate_key.as_deref(), Some("analytical"));
        assert_eq!(
            decision.portfolio_candidate_id.as_deref(),
            Some("mcc.solve.linear")
        );
        assert_eq!(
            decision.prepared_program_identity.as_deref(),
            Some("model#analytical")
        );
        assert_eq!(
            decision.identities.candidate_identity.as_deref(),
            Some("candidate structural")
        );
        assert_eq!(
            decision.identities.lane_identity.as_deref(),
            Some("lane analytical")
        );
        assert_eq!(
            decision.identities.fingerprint_identity.as_deref(),
            Some("proof fingerprint namespace")
        );
    }

    #[test]
    fn verified_decision_without_artifact_fingerprint_is_blocked() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedStaticInvariant,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            ProblemKind::Invariant,
        )
        .with_portfolio_rank(2)
        .with_semantic_digest("petri invariant semantic digest")
        .with_portfolio_candidate_id("mcc-linear-invariant");

        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert_eq!(
            decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingArtifactFingerprint)
        );

        let row = decision.render_evidence_row("MCC");
        assert!(row.contains("source_kind=mcc_petri"));
        assert!(row.contains("payload_kind=mcc_petri"));
        assert!(row.contains("portfolio_rank=2"));
        assert!(row.contains("portfolio_candidate_id=mcc-linear-invariant"));
        assert!(row.contains("proof_fingerprint=none"));
        assert!(row.contains("publication_blocker=missing_artifact_fingerprint"));
    }

    #[test]
    fn rejected_validation_receipt_blocks_publication() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            CheckerSourceKind::Aiger,
            PreparedProgramPayloadKind::Aiger,
            ProblemKind::Safety,
        )
        .with_witness_fingerprint("sha256:bad")
        .with_validation_receipt(ValidationReceipt::rejected(
            ValidationReceiptValidatorKind::WitnessReplay,
            "sha256",
            "bad",
            "prepared replay program",
            "native replay candidate",
            ValidationReceiptArtifactKind::Witness,
            "native witness",
            "trace step 7 missing assignment",
        ));

        assert_eq!(decision.validation_receipt_readiness_code(), "blocked");
        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert_eq!(
            decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::RejectedValidationReceipt)
        );

        let row = decision.render_evidence_row("AIGER");
        assert!(row.contains("validation_receipt_readiness=blocked"));
        assert!(row.contains(
            "validation_receipt_identities=witness_replay:rejected:witness:sha256:bad:prepared_replay_program:native_replay_candidate:native_witness"
        ));
        assert!(row.contains("validation_receipt_failures=trace_step_7_missing_assignment"));
        assert!(row.contains("publication_blocker=rejected_validation_receipt"));
    }

    #[test]
    fn missing_validation_receipt_keeps_publication_unknown_and_blocked() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedStaticInvariant,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            ProblemKind::Invariant,
        )
        .with_validation_requirement(PreparedValidationKind::OutputFormat)
        .with_semantic_digest("petri cert semantic digest")
        .with_certificate_fingerprint("sha256:cert");

        assert_eq!(decision.validation_receipt_readiness_code(), "unknown");
        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert_eq!(
            decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingValidationReceipt)
        );

        let row = decision.render_evidence_row("MCC");
        assert!(row.contains("validation_receipt_readiness=unknown"));
        assert!(row.contains("validation_receipt_identities=none"));
        assert!(row.contains("publication_blocker=missing_validation_receipt"));
    }

    #[test]
    fn validation_receipts_must_satisfy_every_declared_requirement() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedCounterexampleReplay,
            CheckerSourceKind::Btor2,
            PreparedProgramPayloadKind::Btor2,
            ProblemKind::Chc,
        )
        .with_candidate_key("ay_pdr")
        .with_semantic_digest("btor2 semantic digest")
        .with_validation_requirements([PreparedValidationKind::WitnessReplay])
        .with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "proof-only",
            "prepared btor2 program",
            "ay_pdr",
            ValidationReceiptArtifactKind::Proof,
            "validated btor2 proof",
        ));

        assert_eq!(decision.validation_receipt_readiness_code(), "unknown");
        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert_eq!(
            decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingValidationReceipt)
        );

        let row = decision.render_evidence_row("BTOR2");
        assert!(row.contains("validation_requirements=ay_proof,witness_replay"));
        assert!(row.contains("validation_receipt_readiness=unknown"));
        assert!(row.contains("publication_blocker=missing_validation_receipt"));
        assert!(row.contains("admission_disposition=fail_closed_explicit_fallback"));
    }

    #[test]
    fn lifecycle_and_validation_requirements_are_explicit_for_candidate_lane() {
        let decision = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            CheckerSourceKind::Aiger,
            PreparedProgramPayloadKind::Aiger,
            ProblemKind::Safety,
        )
        .with_portfolio_lifecycle(AnalyticalSolvePortfolioLifecycle::Running)
        .with_validation_requirements([
            PreparedValidationKind::OutputFormat,
            PreparedValidationKind::TraceReplay,
            PreparedValidationKind::OutputFormat,
        ])
        .with_witness_fingerprint("witness sha256 feed")
        .with_certificate_fingerprint("cert:sha256:beef")
        .with_decision_reason(AnalyticalSolveDecisionReason::WitnessVerified)
        .with_reason_code("bounded_witness_replayed");

        assert_eq!(decision.publication_readiness_code(), "blocked");
        assert_eq!(
            decision.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::PortfolioLifecycleBlocked)
        );

        let row = decision.render_evidence_row("AIGER");
        assert!(row.contains("portfolio_lifecycle=running"));
        assert!(row.contains("validation=output_format"));
        assert!(row.contains("validation_requirements=output_format,trace_replay"));
        assert!(row.contains("witness_fingerprint=witness_sha256_feed"));
        assert!(row.contains("certificate_fingerprint=cert:sha256:beef"));
        assert!(row.contains("decision_reason=witness_verified"));
        assert!(row.contains("reason_code=bounded_witness_replayed"));
        assert!(row.contains("publication_blocker=portfolio_lifecycle_blocked"));
    }

    #[test]
    fn analytical_solve_shared_engine_receipts_are_consumable_for_all_payloads() {
        for payload_kind in PreparedProgramPayloadKind::shared_engine_payloads() {
            let payload_code = payload_kind.code();
            let source_kind = payload_kind.source_kind();
            let prepared_program_identity = format!("prepared_program:{payload_code}:ay");
            let candidate_identity = format!("candidate:{payload_code}:ay_pdr");
            let fingerprint_identity = format!("ay.proof.fingerprint:{payload_code}:pdr");
            let identities = CheckerArtifactIdentityFields::new()
                .with_frontend_payload_identity(format!("frontend_payload:{payload_code}"))
                .with_artifact_identity(format!("artifact:{payload_code}:ay"))
                .with_candidate_identity(candidate_identity.clone())
                .with_lane_identity(format!("lane:{payload_code}:ay_pdr"));

            let decision = AnalyticalSolveDecision::new(
                AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
                source_kind,
                *payload_kind,
                ProblemKind::Safety,
            )
            .with_backend(BackendKind::AYSmt)
            .with_prepared_program_identity(prepared_program_identity.clone())
            .with_identity_fields(identities)
            .with_candidate_key("ay_pdr")
            .with_semantic_digest(format!("semantic:{payload_code}:ay"))
            .with_ay_shared_engine_validation_requirement(ValidationReceiptArtifactKind::Proof)
            .with_validation_receipt(ValidationReceipt::accepted(
                ValidationReceiptValidatorKind::AYProof,
                ANALYTICAL_SOLVE_AY_VALIDATION_DIGEST_ALGORITHM,
                fingerprint_identity.clone(),
                prepared_program_identity,
                candidate_identity,
                ValidationReceiptArtifactKind::Proof,
                fingerprint_identity,
            ));

            assert_eq!(decision.validation_receipt_readiness_code(), "ready");
            assert_eq!(decision.publication_readiness_code(), "ready");

            let rows = decision.render_shared_engine_validation_receipt_evidence_rows("CORE");
            assert_eq!(rows.len(), 1);
            let row = &rows[0];
            assert!(
                analytical_receipt_row_is_strictly_ready(row),
                "strict shared-engine consumer should accept row for {payload_code}: {row}"
            );

            let fields = shared_engine_receipt_fields(row).expect("strict key/value row");
            assert_eq!(fields["source_kind"], source_kind.code());
            assert_eq!(fields["payload_kind"], payload_code);
            assert_eq!(
                fields["origin_frontend"],
                source_kind.frontend_family_code()
            );
            assert_eq!(fields["model_check_search"], "false");
            assert_eq!(fields["search_kind"], "analytical_solve");
            assert_eq!(fields["validation_artifact_requirements"], "proof");
            assert_eq!(
                fields["digest_algorithm_requirement"],
                ANALYTICAL_SOLVE_AY_VALIDATION_DIGEST_ALGORITHM
            );
            assert_eq!(fields["validation_receipt_readiness"], "ready");
            assert_eq!(fields["fail_closed"], "true");
            assert!(fields["consumable_frontend_families"].contains("tla_plus"));
            assert!(fields["consumable_frontend_families"].contains("mcc_petri"));
            assert!(fields["consumable_frontend_families"].contains("aiger"));
            assert!(fields["consumable_frontend_families"].contains("btor2"));
            assert!(
                !fields["consumable_frontend_families"].contains("future_importer"),
                "future importers need a registered payload before default consumption"
            );
        }
    }

    #[test]
    fn analytical_solve_shared_engine_receipts_fail_closed_when_missing_or_mislabeled() {
        let prepared_program_identity = "prepared_program:mcc_petri:ay";
        let candidate_identity = "candidate:mcc_petri:ay_pdr";
        let fingerprint_identity = "ay.proof.fingerprint:mcc_petri:pdr";
        let base = AnalyticalSolveDecision::new(
            AnalyticalSolveDecisionStatus::VerifiedExecutionModel,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            ProblemKind::Safety,
        )
        .with_backend(BackendKind::AYChc)
        .with_prepared_program_identity(prepared_program_identity)
        .with_candidate_key("ay_pdr")
        .with_semantic_digest("semantic:mcc_petri:ay")
        .with_proof_fingerprint(fingerprint_identity)
        .with_ay_shared_engine_validation_requirement(ValidationReceiptArtifactKind::Proof);

        let missing_route =
            choose_analytical_solve_route(std::slice::from_ref(&base), Some("explicit_bfs"));
        assert_eq!(missing_route.route, AnalyticalSolveRoute::ExplicitFallback);
        assert_eq!(
            missing_route.reason,
            AnalyticalSolveDecisionReason::MissingValidationReceipt
        );
        assert_eq!(
            base.render_shared_engine_validation_receipt_evidence_rows("MCC")
                .len(),
            0
        );
        assert_eq!(
            base.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::MissingValidationReceipt)
        );

        let wrong_artifact_kind =
            base.clone()
                .with_validation_receipt(ValidationReceipt::accepted(
                    ValidationReceiptValidatorKind::AYProof,
                    ANALYTICAL_SOLVE_AY_VALIDATION_DIGEST_ALGORITHM,
                    fingerprint_identity,
                    prepared_program_identity,
                    candidate_identity,
                    ValidationReceiptArtifactKind::Witness,
                    fingerprint_identity,
                ));
        assert_eq!(
            wrong_artifact_kind.validation_receipt_readiness_code(),
            "blocked"
        );
        assert_eq!(
            wrong_artifact_kind.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::InvalidValidationReceipt)
        );
        let wrong_artifact_row = wrong_artifact_kind
            .render_shared_engine_validation_receipt_evidence_rows("MCC")
            .pop()
            .expect("mislabeled receipt should render fail-closed evidence");
        assert!(!analytical_receipt_row_is_strictly_ready(
            &wrong_artifact_row
        ));
        assert!(wrong_artifact_row.contains("receipt_validation=invalid"));
        assert!(wrong_artifact_row.contains("publication_blocker=invalid_validation_receipt"));
        assert!(
            wrong_artifact_row.contains("validation_artifact_kind_expected=proof_actual=witness")
        );

        let wrong_digest_algorithm = base.with_validation_receipt(ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "fnv1a64",
            fingerprint_identity,
            prepared_program_identity,
            candidate_identity,
            ValidationReceiptArtifactKind::Proof,
            fingerprint_identity,
        ));
        assert_eq!(
            wrong_digest_algorithm.publication_blocker_reason(),
            Some(AnalyticalSolveDecisionReason::InvalidValidationReceipt)
        );
        let wrong_digest_row = wrong_digest_algorithm
            .render_shared_engine_validation_receipt_evidence_rows("MCC")
            .pop()
            .expect("wrong digest label should render fail-closed evidence");
        assert!(!analytical_receipt_row_is_strictly_ready(&wrong_digest_row));
        assert!(wrong_digest_row.contains("receipt_validation=invalid"));
        assert!(wrong_digest_row
            .contains("digest_algorithm_expected=ay_fingerprint_identity_actual=fnv1a64"));
    }
}
