// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Portfolio strategy for BTOR2 hardware model checking.
//!
//! [`check_btor2_portfolio`] orchestrates the full checking pipeline on a parsed
//! program:
//!
//! 1. **COI reduction** — drop state variables and inputs outside the cone of
//!    influence of any `bad` property, shrinking the invariant predicate.
//! 2. **Simplification** — constant-fold and eliminate identities so the SMT
//!    kernel has less to chew on.
//! 3. **BMC preprocessing** — shallow bounded model checking to catch the many
//!    benchmarks with short counterexamples cheaply (default 20% of the budget).
//! 4. **Full CHC solving** — PDR / k-induction via the `ay-chc` adaptive
//!    portfolio for the remainder of the budget.
//!
//! Each stage is independently toggleable through [`PortfolioConfig`], and
//! [`check_btor2_portfolio_with_report`] additionally returns a shared backend
//! [`CapabilityReport`].
//!
//! The rest of the module exports the BTOR2 *evidence* surface: typed records
//! and validators ([`Btor2UnsafeProofReplayArtifact`],
//! [`Btor2ConcreteTraceReplayAcceptance`], [`Btor2ConcreteTraceReplayRejection`]
//! and the `btor2_hardware_replay_*` functions) that bind every SAFE/UNSAFE
//! verdict to a fail-closed, `ay-chc`-owned proof or replay artifact rather than
//! to any locally-produced witness.

use std::time::{Duration, Instant};

use ay_chc::{
    bmc_unsafe_trace_assignment_contract, engines, normalized_chc_input_sha256, AdaptiveConfig,
    AdaptivePortfolio, BmcConfig as AYBmcConfig, ChcProofArtifactDigest, ChcProofEvidenceOptions,
    ChcProofSolverIdentity, ChcProofTranscriptConsumerEvidence, ChcReplayEvidence,
    ChcReplayObligation, ChcReplayObligationArtifact, ChcReplayObligationKind, PdrConfig,
    VerifiedChcResult,
};
use rustc_hash::FxHashMap;
#[allow(unused_imports)]
use tla_mc_core::{
    ay_chc_capability, validate_hardware_replay_decision_evidence_row, BackendCapability,
    BackendDomain, BackendKind, CapabilityLaneDecision, CapabilityReport, CapabilityRole,
    HardwareReplayDecisionEvidenceError, HardwareReplayDecisionStatus,
    HardwareReplayPrimitiveAssignmentStatus, HardwareReplayPrimitiveConsumerStatus,
    HardwareReplayPrimitiveDecisionStatus, HardwareReplayPrimitiveRejectionReason,
    HardwareReplayPrimitiveStatus, ProblemKind, ProductionRoutingStatus, SolverFacet, SolverLimits,
    SymbolicExecutionDetection, SymbolicExecutionReason, UnsupportedReason,
    HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS, HARDWARE_REPLAY_DECISION_ROW_KIND,
    HARDWARE_REPLAY_DECISION_SCHEMA, HARDWARE_REPLAY_DECISION_SCHEMA_VERSION,
};

use crate::bmc::{bmc_preprocess, BmcConfig, BmcPreResult};
use crate::coi::{compute_coi, reduce_program};
use crate::error::Btor2Error;
use crate::shared_engine_evidence::btor2_shared_engine_evidence_rows;
use crate::to_chc::{translate_to_chc, StateVarEntry};
use crate::translate::Btor2CheckResult;
use crate::types::Btor2Program;

struct Btor2ReplayApiGateEvidence {
    verdict: &'static str,
    artifact_kind: &'static str,
    backend: BackendKind,
    replay_api: &'static str,
    replay_status: &'static str,
    acceptance_gate: &'static str,
    failure_policy: &'static str,
    evidence_basis: &'static str,
}

/// Static replay API gate evidence exported by the BTOR2 portfolio.
const BTOR2_REPLAY_API_GATES: &[Btor2ReplayApiGateEvidence] = &[
    Btor2ReplayApiGateEvidence {
        verdict: "safe",
        artifact_kind: "verified_chc_result_safe",
        backend: BackendKind::AYChc,
        replay_api: "ay_chc_verified_result",
        replay_status: "delegated_to_ay",
        acceptance_gate: "verified_chc_result_safe",
        failure_policy: "fail_closed_no_local_production",
        evidence_basis: "ay_chc_safe_proof",
    },
    Btor2ReplayApiGateEvidence {
        verdict: "unsafe",
        artifact_kind: "verified_chc_result_unsafe",
        backend: BackendKind::AYChc,
        replay_api: "ay_chc_verified_result",
        replay_status: "delegated_to_ay",
        acceptance_gate: "verified_chc_result_unsafe",
        failure_policy: "fail_closed_no_local_production",
        evidence_basis: "ay_chc_counterexample",
    },
    Btor2ReplayApiGateEvidence {
        verdict: "unsafe",
        artifact_kind: "query_clause_attribution",
        backend: BackendKind::AYChc,
        replay_api: "property_indices_match",
        replay_status: "proven_attribution_only",
        acceptance_gate: "query_clause_matches_property_indices",
        failure_policy: "multi_property_unknown_without_query_clause",
        evidence_basis: "counterexample_witness_query_clause",
    },
    Btor2ReplayApiGateEvidence {
        verdict: "unsafe",
        artifact_kind: "local_trace_replay",
        backend: BackendKind::Btor2Portfolio,
        replay_api: "none",
        replay_status: "not_available",
        acceptance_gate: "not_applicable",
        failure_policy: "do_not_report_local_replay",
        evidence_basis: "no_btor2_trace_replay_api",
    },
];

/// Configuration for the portfolio strategy.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Total time budget for the entire pipeline.
    pub time_budget: Option<Duration>,
    /// Enable COI reduction.
    pub enable_coi: bool,
    /// Enable expression simplification.
    pub enable_simplify: bool,
    /// Enable BMC preprocessing.
    pub enable_bmc: bool,
    /// BMC time budget fraction (0.0-1.0 of total budget).
    pub bmc_budget_fraction: f64,
    /// BMC maximum depth.
    pub bmc_max_depth: u32,
    /// Enable verbose output.
    pub verbose: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            time_budget: None,
            enable_coi: true,
            enable_simplify: true,
            enable_bmc: true,
            bmc_budget_fraction: 0.2,
            bmc_max_depth: 20,
            verbose: false,
        }
    }
}

/// Statistics from a portfolio run.
#[derive(Debug, Clone)]
pub struct PortfolioStats {
    /// Number of state variables before COI reduction.
    pub states_before_coi: usize,
    /// Number of state variables after COI reduction.
    pub states_after_coi: usize,
    /// Number of inputs before COI reduction.
    pub inputs_before_coi: usize,
    /// Number of inputs after COI reduction.
    pub inputs_after_coi: usize,
    /// Time spent in COI analysis.
    pub coi_time: Duration,
    /// Time spent in BMC preprocessing.
    pub bmc_time: Duration,
    /// Time spent in full CHC solving.
    pub chc_time: Duration,
    /// Total elapsed time.
    pub total_time: Duration,
    /// Which phase produced the result.
    pub result_phase: ResultPhase,
}

/// Which phase of the portfolio produced the final result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultPhase {
    /// The GPU bit-parallel falsification lane found the answer.
    Gpu,
    /// BMC preprocessing found the answer.
    Bmc,
    /// Full CHC solving found the answer.
    Chc,
    /// No answer found.
    None,
}

/// Real BTOR2 unsafe proof/replay artifact produced from a verified ay-chc
/// counterexample.
#[derive(Debug, Clone)]
pub struct Btor2UnsafeProofReplayArtifact {
    /// Number of BTOR2 bad properties in the translated program.
    pub bad_property_count: usize,
    /// Bad-property index attributed by the ay-chc query-clause witness.
    pub property_index: Option<usize>,
    /// CHC query clause index carried by the ay-chc derivation witness.
    pub query_clause: Option<usize>,
    /// How the violated BTOR2 property was attributed.
    pub property_attribution: &'static str,
    /// Number of concrete counterexample trace steps.
    pub trace_steps: usize,
    /// Number of BTOR2 state variables that must be assignment-complete per step.
    pub state_var_count: usize,
    /// Number of derivation witness entries carried by ay-chc.
    pub witness_entries: usize,
    /// SHA-256 of the normalized CHC input solved by ay-chc.
    pub normalized_chc_input_sha256: String,
    /// Replayable SMT-LIB obligations generated from the verified counterexample.
    pub replay_obligations: Vec<ChcReplayObligation>,
    /// Hash-bound descriptors for the generated replay obligations.
    pub replay_obligation_artifacts: Vec<ChcReplayObligationArtifact>,
    /// Typed ay-chc replay evidence binding for concrete trace replay consumers.
    pub replay_evidence: Option<ChcReplayEvidence>,
    /// Typed ay-chc consumer transcript evidence used to derive trace assignments.
    pub ay_consumer_evidence: Option<ChcProofTranscriptConsumerEvidence>,
    /// AY replay-obligation generation error when replay evidence is unavailable.
    pub replay_unavailable_reason: Option<String>,
    /// Backend evidence rows for this concrete artifact.
    pub evidence: Vec<String>,
}

/// Accepted BTOR2 concrete trace replay binding consumed from typed ay-chc
/// metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Btor2ConcreteTraceReplayAcceptance {
    /// SHA-256 over the accepted typed ay-chc replay evidence identity.
    pub replay_evidence_identity_sha256: String,
    /// SHA-256 of the normalized CHC input solved by ay-chc.
    pub normalized_chc_input_sha256: String,
    /// Number of typed trace-validity replay obligations accepted.
    pub trace_validity_obligations: usize,
    /// Digest identities for the accepted replay obligation artifacts.
    pub replay_obligation_identity_sha256: Vec<String>,
    /// AY-owned proof evidence status admitted with this replay binding.
    pub ay_proof_evidence_status: String,
    /// SHA-256 identity for the AY proof/consumer evidence binding.
    pub ay_proof_evidence_sha256: String,
}

/// Fail-closed reason for rejecting BTOR2 concrete trace replay evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Btor2ConcreteTraceReplayRejection {
    /// Evidence rows identify generated placeholder material.
    GeneratedPlaceholderEvidence,
    /// Required proof/replay boundary row is absent.
    MissingProofReplayBoundaryEvidence,
    /// Required unsafe replay API gate row is absent.
    MissingUnsafeReplayGateEvidence,
    /// The violated BTOR2 bad property was not attributed.
    MissingPropertyAttribution,
    /// Multi-property unsafe results must carry a query-clause attribution.
    MissingQueryClauseAttribution,
    /// The property attribution mode is not one of the proven modes.
    MissingProvenPropertyAttributionMode,
    /// ay-chc could not export concrete trace replay obligations.
    ConcreteTraceAssignmentsUnavailable(String),
    /// No typed ay-chc replay evidence was attached.
    MissingTypedAYReplayEvidence,
    /// No typed ay-chc consumer transcript evidence was attached.
    MissingTypedAYConsumerEvidence,
    /// The typed consumer evidence is bound to a different CHC input hash.
    AYConsumerEvidenceProblemHashMismatch {
        /// Expected normalized CHC input hash.
        expected: String,
        /// Hash carried by typed consumer evidence.
        actual: String,
    },
    /// The typed consumer evidence does not describe an accepted unsafe result.
    AYConsumerEvidenceResultMismatch {
        /// Verdict code carried by typed consumer evidence.
        verdict_code: String,
        /// Consumer acceptance bit carried by typed consumer evidence.
        accepted_for_consumer: bool,
        /// Model validation bit carried by typed consumer evidence.
        model_validated: bool,
        /// Verification level code carried by typed consumer evidence.
        verification_level_code: String,
    },
    /// Typed ay-chc consumer evidence did not carry unsafe trace material.
    MissingTypedAYUnsafeTrace,
    /// Typed ay-chc unsafe trace assignments are not complete enough for BTOR2 replay.
    TypedAYTraceAssignmentsIncomplete {
        /// Expected `(trace step, state argument)` assignment slots.
        expected_slots: usize,
        /// Total assignment fields present in AY consumer evidence.
        assignment_fields: usize,
        /// Assignments that carried `predicate_argument_index`.
        typed_predicate_argument_assignments: usize,
        /// Missing typed assignment slots for BTOR2 replay.
        missing_typed_predicate_argument_assignments: usize,
        /// Assignment field names present in the AY consumer evidence.
        present_assignment_fields: String,
    },
    /// No concrete counterexample digest was attached to typed ay-chc evidence.
    MissingTypedAYCounterexampleDigest,
    /// No typed trace-validity replay obligation was attached.
    MissingTypedTraceValidityReplayObligation,
    /// Replay obligation queries and digest descriptors do not line up.
    ReplayObligationDescriptorCountMismatch {
        /// Number of replay queries.
        obligations: usize,
        /// Number of digest descriptors.
        artifacts: usize,
    },
    /// The typed replay evidence does not bind the same obligations as the
    /// generated replay artifacts.
    ReplayEvidenceObligationMismatch,
    /// A replay obligation descriptor is not the digest of the query bytes.
    ReplayObligationDigestMismatch {
        /// Replay obligation index.
        index: usize,
    },
    /// A replay obligation query is not a self-contained replay SMT-LIB query.
    ReplayObligationNotReplayable {
        /// Human-readable obligation name.
        name: String,
    },
    /// The typed replay evidence is bound to a different CHC input hash.
    ReplayEvidenceProblemHashMismatch {
        /// Expected normalized CHC input hash.
        expected: String,
        /// Hash carried by typed replay evidence.
        actual: String,
    },
    /// The typed replay evidence does not claim unsafe counterexample proof
    /// material.
    ReplayEvidenceResultMismatch {
        /// Result carried by typed replay evidence.
        result: String,
        /// Proof status carried by typed replay evidence.
        proof_status: String,
    },
}

impl Btor2ConcreteTraceReplayRejection {
    /// Stable machine-readable reason code for evidence rows and tests.
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::GeneratedPlaceholderEvidence => "generated_placeholder_evidence",
            Self::MissingProofReplayBoundaryEvidence => "missing_proof_replay_boundary_evidence",
            Self::MissingUnsafeReplayGateEvidence => "missing_unsafe_replay_api_gate_evidence",
            Self::MissingPropertyAttribution => "missing_property_attribution",
            Self::MissingQueryClauseAttribution => "missing_query_clause_attribution",
            Self::MissingProvenPropertyAttributionMode => {
                "missing_proven_property_attribution_mode"
            }
            Self::ConcreteTraceAssignmentsUnavailable(_) => {
                "concrete_trace_assignments_unavailable"
            }
            Self::MissingTypedAYReplayEvidence => "missing_typed_ay_replay_evidence",
            Self::MissingTypedAYConsumerEvidence => "missing_typed_ay_consumer_evidence",
            Self::AYConsumerEvidenceProblemHashMismatch { .. } => {
                "typed_ay_consumer_evidence_problem_hash_mismatch"
            }
            Self::AYConsumerEvidenceResultMismatch { .. } => {
                "typed_ay_consumer_evidence_result_mismatch"
            }
            Self::MissingTypedAYUnsafeTrace => "missing_typed_ay_unsafe_trace",
            Self::TypedAYTraceAssignmentsIncomplete { .. } => {
                "typed_ay_trace_assignments_incomplete"
            }
            Self::MissingTypedAYCounterexampleDigest => "missing_typed_ay_counterexample_digest",
            Self::MissingTypedTraceValidityReplayObligation => {
                "missing_typed_trace_validity_replay_obligation"
            }
            Self::ReplayObligationDescriptorCountMismatch { .. } => {
                "replay_obligation_descriptor_count_mismatch"
            }
            Self::ReplayEvidenceObligationMismatch => "replay_evidence_obligation_mismatch",
            Self::ReplayObligationDigestMismatch { .. } => "replay_obligation_digest_mismatch",
            Self::ReplayObligationNotReplayable { .. } => "replay_obligation_not_replayable",
            Self::ReplayEvidenceProblemHashMismatch { .. } => {
                "replay_evidence_problem_hash_mismatch"
            }
            Self::ReplayEvidenceResultMismatch { .. } => "replay_evidence_result_mismatch",
        }
    }

    /// Map BTOR2-specific replay rejection onto the shared hardware replay
    /// primitive reason vocabulary.
    pub fn hardware_replay_rejection_reason(&self) -> HardwareReplayPrimitiveRejectionReason {
        match self {
            Self::GeneratedPlaceholderEvidence => {
                HardwareReplayPrimitiveRejectionReason::GeneratedPlaceholderEvidence
            }
            Self::MissingProofReplayBoundaryEvidence => {
                HardwareReplayPrimitiveRejectionReason::MissingProofReplayBoundaryEvidence
            }
            Self::MissingUnsafeReplayGateEvidence => {
                HardwareReplayPrimitiveRejectionReason::MissingUnsafeReplayGateEvidence
            }
            Self::MissingPropertyAttribution => {
                HardwareReplayPrimitiveRejectionReason::MissingPropertyAttribution
            }
            Self::MissingQueryClauseAttribution => {
                HardwareReplayPrimitiveRejectionReason::MissingQueryClauseAttribution
            }
            Self::MissingProvenPropertyAttributionMode => {
                HardwareReplayPrimitiveRejectionReason::MissingProvenPropertyAttributionMode
            }
            Self::ConcreteTraceAssignmentsUnavailable(_) => {
                HardwareReplayPrimitiveRejectionReason::ConcreteTraceAssignmentsUnavailable
            }
            Self::MissingTypedAYReplayEvidence => {
                HardwareReplayPrimitiveRejectionReason::MissingTypedAYReplayEvidence
            }
            Self::MissingTypedAYConsumerEvidence => {
                HardwareReplayPrimitiveRejectionReason::MissingTypedAYConsumerEvidence
            }
            Self::AYConsumerEvidenceProblemHashMismatch { .. } => {
                HardwareReplayPrimitiveRejectionReason::AYConsumerEvidenceProblemHashMismatch
            }
            Self::AYConsumerEvidenceResultMismatch { .. } => {
                HardwareReplayPrimitiveRejectionReason::AYConsumerEvidenceResultMismatch
            }
            Self::MissingTypedAYUnsafeTrace => {
                HardwareReplayPrimitiveRejectionReason::MissingTypedAYUnsafeTrace
            }
            Self::TypedAYTraceAssignmentsIncomplete { .. } => {
                HardwareReplayPrimitiveRejectionReason::TypedAYTraceAssignmentsIncomplete
            }
            Self::MissingTypedAYCounterexampleDigest => {
                HardwareReplayPrimitiveRejectionReason::MissingTypedAYCounterexampleDigest
            }
            Self::MissingTypedTraceValidityReplayObligation => {
                HardwareReplayPrimitiveRejectionReason::MissingTypedTraceValidityReplayObligation
            }
            Self::ReplayObligationDescriptorCountMismatch { .. } => {
                HardwareReplayPrimitiveRejectionReason::ReplayObligationDescriptorCountMismatch
            }
            Self::ReplayEvidenceObligationMismatch => {
                HardwareReplayPrimitiveRejectionReason::ReplayEvidenceObligationMismatch
            }
            Self::ReplayObligationDigestMismatch { .. } => {
                HardwareReplayPrimitiveRejectionReason::ReplayObligationDigestMismatch
            }
            Self::ReplayObligationNotReplayable { .. } => {
                HardwareReplayPrimitiveRejectionReason::ReplayObligationNotReplayable
            }
            Self::ReplayEvidenceProblemHashMismatch { .. } => {
                HardwareReplayPrimitiveRejectionReason::ReplayEvidenceProblemHashMismatch
            }
            Self::ReplayEvidenceResultMismatch { .. } => {
                HardwareReplayPrimitiveRejectionReason::ReplayEvidenceResultMismatch
            }
        }
    }
}

impl std::fmt::Display for Btor2ConcreteTraceReplayRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConcreteTraceAssignmentsUnavailable(reason) => {
                write!(formatter, "{}: {reason}", self.reason_code())
            }
            Self::ReplayObligationDescriptorCountMismatch {
                obligations,
                artifacts,
            } => write!(
                formatter,
                "{}: obligations={obligations} artifacts={artifacts}",
                self.reason_code()
            ),
            Self::ReplayObligationDigestMismatch { index } => {
                write!(formatter, "{}: index={index}", self.reason_code())
            }
            Self::ReplayObligationNotReplayable { name } => {
                write!(formatter, "{}: name={name}", self.reason_code())
            }
            Self::ReplayEvidenceProblemHashMismatch { expected, actual } => write!(
                formatter,
                "{}: expected={expected} actual={actual}",
                self.reason_code()
            ),
            Self::ReplayEvidenceResultMismatch {
                result,
                proof_status,
            } => write!(
                formatter,
                "{}: result={result} proof_status={proof_status}",
                self.reason_code()
            ),
            Self::AYConsumerEvidenceProblemHashMismatch { expected, actual } => write!(
                formatter,
                "{}: expected={expected} actual={actual}",
                self.reason_code()
            ),
            Self::AYConsumerEvidenceResultMismatch {
                verdict_code,
                accepted_for_consumer,
                model_validated,
                verification_level_code,
            } => write!(
                formatter,
                "{}: verdict_code={verdict_code} accepted_for_consumer={accepted_for_consumer} model_validated={model_validated} verification_level_code={verification_level_code}",
                self.reason_code()
            ),
            Self::TypedAYTraceAssignmentsIncomplete {
                expected_slots,
                assignment_fields,
                typed_predicate_argument_assignments,
                missing_typed_predicate_argument_assignments,
                present_assignment_fields,
            } => write!(
                formatter,
                "{}: expected_slots={expected_slots} assignment_fields={assignment_fields} typed_predicate_argument_assignments={typed_predicate_argument_assignments} missing_typed_predicate_argument_assignments={missing_typed_predicate_argument_assignments} present_assignment_fields={present_assignment_fields}",
                self.reason_code()
            ),
            other => formatter.write_str(other.reason_code()),
        }
    }
}

impl std::error::Error for Btor2ConcreteTraceReplayRejection {}

/// Consume BTOR2 concrete trace replay evidence through typed ay-chc metadata.
///
/// This intentionally ignores string-only success claims: replay is accepted
/// only when the artifact carries typed ay-chc replay evidence, trace-validity
/// SMT obligations, and matching hash-bound obligation descriptors.
pub fn btor2_accept_concrete_trace_replay(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> Result<Btor2ConcreteTraceReplayAcceptance, Btor2ConcreteTraceReplayRejection> {
    validate_btor2_replay_boundary_metadata(artifact)?;
    let ay_consumer_evidence = artifact.ay_consumer_evidence.as_ref();
    if let Some(evidence) = ay_consumer_evidence {
        validate_ay_consumer_evidence_for_btor2_replay(artifact, evidence)?;
    }
    if let Some(reason) = &artifact.replay_unavailable_reason {
        return Err(
            Btor2ConcreteTraceReplayRejection::ConcreteTraceAssignmentsUnavailable(reason.clone()),
        );
    }

    let replay_evidence = artifact
        .replay_evidence
        .as_ref()
        .ok_or(Btor2ConcreteTraceReplayRejection::MissingTypedAYReplayEvidence)?;
    let ay_consumer_evidence = ay_consumer_evidence
        .ok_or(Btor2ConcreteTraceReplayRejection::MissingTypedAYConsumerEvidence)?;
    if artifact.replay_obligations.len() != artifact.replay_obligation_artifacts.len() {
        return Err(
            Btor2ConcreteTraceReplayRejection::ReplayObligationDescriptorCountMismatch {
                obligations: artifact.replay_obligations.len(),
                artifacts: artifact.replay_obligation_artifacts.len(),
            },
        );
    }
    if artifact.replay_obligations.is_empty() {
        return Err(Btor2ConcreteTraceReplayRejection::MissingTypedTraceValidityReplayObligation);
    }

    if replay_evidence.problem_sha256 != artifact.normalized_chc_input_sha256 {
        return Err(
            Btor2ConcreteTraceReplayRejection::ReplayEvidenceProblemHashMismatch {
                expected: artifact.normalized_chc_input_sha256.clone(),
                actual: replay_evidence.problem_sha256.clone(),
            },
        );
    }
    if replay_evidence.result != "unsafe"
        || replay_evidence.proof_status != "verified-counterexample"
    {
        return Err(
            Btor2ConcreteTraceReplayRejection::ReplayEvidenceResultMismatch {
                result: replay_evidence.result.clone(),
                proof_status: replay_evidence.proof_status.clone(),
            },
        );
    }
    if replay_evidence.counterexample.is_none() {
        return Err(Btor2ConcreteTraceReplayRejection::MissingTypedAYCounterexampleDigest);
    }

    let mut trace_validity_obligations = 0;
    let mut artifact_identities = Vec::with_capacity(artifact.replay_obligation_artifacts.len());
    for (index, (obligation, descriptor)) in artifact
        .replay_obligations
        .iter()
        .zip(artifact.replay_obligation_artifacts.iter())
        .enumerate()
    {
        if obligation.kind == ChcReplayObligationKind::TraceValidity {
            trace_validity_obligations += 1;
        }
        let expected_query =
            ChcProofArtifactDigest::from_bytes("replay-obligation", obligation.smtlib.as_bytes());
        if descriptor.kind != obligation.kind
            || descriptor.query.sha256 != expected_query.sha256
            || descriptor.query.role != expected_query.role
        {
            return Err(
                Btor2ConcreteTraceReplayRejection::ReplayObligationDigestMismatch { index },
            );
        }
        if !is_replayable_btor2_trace_obligation(obligation, &artifact.normalized_chc_input_sha256)
        {
            return Err(
                Btor2ConcreteTraceReplayRejection::ReplayObligationNotReplayable {
                    name: obligation.name.clone(),
                },
            );
        }
        artifact_identities.push(descriptor.identity_sha256());
    }
    if trace_validity_obligations == 0 {
        return Err(Btor2ConcreteTraceReplayRejection::MissingTypedTraceValidityReplayObligation);
    }

    let mut evidence_identities: Vec<_> = replay_evidence
        .replay_obligations
        .iter()
        .map(ChcReplayObligationArtifact::identity_sha256)
        .collect();
    evidence_identities.sort();
    let mut sorted_artifact_identities = artifact_identities.clone();
    sorted_artifact_identities.sort();
    if evidence_identities != sorted_artifact_identities {
        return Err(Btor2ConcreteTraceReplayRejection::ReplayEvidenceObligationMismatch);
    }

    Ok(Btor2ConcreteTraceReplayAcceptance {
        replay_evidence_identity_sha256: replay_evidence.identity_sha256(),
        normalized_chc_input_sha256: artifact.normalized_chc_input_sha256.clone(),
        trace_validity_obligations,
        replay_obligation_identity_sha256: artifact_identities,
        ay_proof_evidence_status: ay_consumer_evidence.verification_level_code.clone(),
        ay_proof_evidence_sha256: ay_consumer_evidence.property_sha256.clone(),
    })
}

/// Classify BTOR2 unsafe replay evidence using the shared hardware replay
/// primitive boundary vocabulary.
pub fn btor2_hardware_replay_primitive_status(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> HardwareReplayPrimitiveStatus {
    let assignments = btor2_hardware_replay_decision_assignments(artifact);
    match btor2_accept_concrete_trace_replay(artifact) {
        Ok(_) => btor2_replay_primitive_status(
            HardwareReplayPrimitiveConsumerStatus::Accepted,
            HardwareReplayPrimitiveRejectionReason::None,
            "proven",
            "ay_chc",
            false,
            assignments,
        ),
        Err(error) => btor2_replay_primitive_status(
            HardwareReplayPrimitiveConsumerStatus::Rejected,
            error.hardware_replay_rejection_reason(),
            "not_available",
            "consumer_gate",
            matches!(
                error,
                Btor2ConcreteTraceReplayRejection::GeneratedPlaceholderEvidence
            ),
            assignments,
        ),
    }
}

/// Classify BTOR2 unsafe replay evidence as an actionable orchestration
/// decision using the shared hardware replay decision schema.
pub fn btor2_hardware_replay_decision_status(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> HardwareReplayDecisionStatus {
    let primitive_status = btor2_hardware_replay_primitive_status(artifact);
    let assignments = btor2_hardware_replay_decision_assignments(artifact);
    let acceptance = btor2_hardware_replay_decision_acceptance(artifact);
    HardwareReplayDecisionStatus {
        hardware: primitive_status.hardware,
        verdict: primitive_status.verdict,
        primitive: primitive_status.primitive,
        ay_backend_code: primitive_status.ay_backend_code,
        replay_api: primitive_status.replay_api,
        replay_status: primitive_status.replay_status,
        evidence_source: primitive_status.evidence_source,
        generated_placeholder: primitive_status.generated_placeholder,
        typed_assignment_source: assignments.typed_assignment_source,
        replay_assignment_status: assignments.replay_assignment_status,
        typed_assignment_required_slots: assignments.typed_assignment_required_slots,
        typed_assignment_present_slots: assignments.typed_assignment_present_slots,
        typed_assignment_missing_slots: assignments.typed_assignment_missing_slots,
        accepted_replay_evidence_identity_sha256: acceptance
            .accepted_replay_evidence_identity_sha256,
        accepted_trace_validity_obligations: acceptance.accepted_trace_validity_obligations,
        accepted_replay_obligation_identities_sha256: acceptance
            .accepted_replay_obligation_identities_sha256,
        accepted_ay_proof_evidence_status: acceptance.accepted_ay_proof_evidence_status,
        accepted_ay_proof_evidence_sha256: acceptance.accepted_ay_proof_evidence_sha256,
        consumer_status: primitive_status.consumer_status,
        rejection_reason: primitive_status.rejection_reason,
    }
}

/// Render the actionable BTOR2 hardware replay decision evidence row.
pub fn btor2_hardware_replay_decision_evidence(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> String {
    btor2_hardware_replay_decision_status(artifact).render_evidence_row()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Btor2HardwareReplayDecisionAssignments {
    typed_assignment_source: String,
    replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus,
    typed_assignment_required_slots: usize,
    typed_assignment_present_slots: usize,
    typed_assignment_missing_slots: usize,
}

impl Btor2HardwareReplayDecisionAssignments {
    fn missing(required_slots: usize) -> Self {
        Self {
            typed_assignment_source: "missing".to_string(),
            replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus::Missing,
            typed_assignment_required_slots: required_slots,
            typed_assignment_present_slots: 0,
            typed_assignment_missing_slots: required_slots,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Btor2HardwareReplayDecisionAcceptance {
    accepted_replay_evidence_identity_sha256: String,
    accepted_trace_validity_obligations: usize,
    accepted_replay_obligation_identities_sha256: String,
    accepted_ay_proof_evidence_status: String,
    accepted_ay_proof_evidence_sha256: String,
}

impl Btor2HardwareReplayDecisionAcceptance {
    fn none() -> Self {
        Self {
            accepted_replay_evidence_identity_sha256: "none".to_string(),
            accepted_trace_validity_obligations: 0,
            accepted_replay_obligation_identities_sha256: "none".to_string(),
            accepted_ay_proof_evidence_status: "none".to_string(),
            accepted_ay_proof_evidence_sha256: "none".to_string(),
        }
    }

    fn accepted(acceptance: Btor2ConcreteTraceReplayAcceptance) -> Self {
        Self {
            accepted_replay_evidence_identity_sha256: acceptance.replay_evidence_identity_sha256,
            accepted_trace_validity_obligations: acceptance.trace_validity_obligations,
            accepted_replay_obligation_identities_sha256: if acceptance
                .replay_obligation_identity_sha256
                .is_empty()
            {
                "none".to_string()
            } else {
                evidence_token(&acceptance.replay_obligation_identity_sha256.join(","))
            },
            accepted_ay_proof_evidence_status: evidence_token(&acceptance.ay_proof_evidence_status),
            accepted_ay_proof_evidence_sha256: acceptance.ay_proof_evidence_sha256,
        }
    }
}

fn btor2_hardware_replay_decision_assignments(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> Btor2HardwareReplayDecisionAssignments {
    let required_slots = artifact
        .trace_steps
        .saturating_mul(artifact.state_var_count);
    let Some(consumer_evidence) = artifact.ay_consumer_evidence.as_ref() else {
        return Btor2HardwareReplayDecisionAssignments::missing(required_slots);
    };
    let Ok(summary) = summarize_ay_consumer_trace_assignments(
        consumer_evidence,
        artifact.trace_steps,
        artifact.state_var_count,
    ) else {
        return Btor2HardwareReplayDecisionAssignments {
            typed_assignment_source: "ay_chc_consumer_evidence".to_string(),
            replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus::Missing,
            typed_assignment_required_slots: required_slots,
            typed_assignment_present_slots: 0,
            typed_assignment_missing_slots: required_slots,
        };
    };

    Btor2HardwareReplayDecisionAssignments {
        typed_assignment_source: "ay_chc_consumer_evidence".to_string(),
        replay_assignment_status: if summary.missing_typed_predicate_argument_assignments == 0 {
            HardwareReplayPrimitiveAssignmentStatus::Complete
        } else {
            HardwareReplayPrimitiveAssignmentStatus::Incomplete
        },
        typed_assignment_required_slots: summary.expected_slots,
        typed_assignment_present_slots: summary.projected_btor2_assignments,
        typed_assignment_missing_slots: summary.missing_typed_predicate_argument_assignments,
    }
}

fn btor2_hardware_replay_decision_acceptance(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> Btor2HardwareReplayDecisionAcceptance {
    btor2_accept_concrete_trace_replay(artifact).map_or_else(
        |_| Btor2HardwareReplayDecisionAcceptance::none(),
        Btor2HardwareReplayDecisionAcceptance::accepted,
    )
}

/// Validate a BTOR2 hardware replay decision row against the exported schema.
pub fn validate_btor2_hardware_replay_decision_evidence_row(
    row: &str,
) -> Result<(), HardwareReplayDecisionEvidenceError> {
    let expected_prefix = format!("BTOR2 {} ", HARDWARE_REPLAY_DECISION_ROW_KIND);
    if !row.starts_with(&expected_prefix) {
        return Err(HardwareReplayDecisionEvidenceError::WrongRowKind);
    }

    validate_hardware_replay_decision_evidence_row(row)
}

/// Validate the BTOR2 decision row against the current primitive status.
pub fn validate_btor2_hardware_replay_decision_evidence(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> Result<(), HardwareReplayDecisionEvidenceError> {
    let expected_prefix = format!("BTOR2 {} ", HARDWARE_REPLAY_DECISION_ROW_KIND);
    let mut rows = artifact
        .evidence
        .iter()
        .filter(|row| row.starts_with(&expected_prefix));
    let row = rows
        .next()
        .ok_or(HardwareReplayDecisionEvidenceError::MissingDecisionEvidence)?;
    if rows.next().is_some() {
        return Err(HardwareReplayDecisionEvidenceError::DuplicateDecisionEvidence);
    }

    validate_btor2_hardware_replay_decision_evidence_row(row)?;
    let expected_row = btor2_hardware_replay_decision_evidence(artifact);
    if row != &expected_row {
        return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "decision_row_does_not_match_current_primitive_status",
        ));
    }

    Ok(())
}

fn btor2_replay_primitive_status(
    consumer_status: HardwareReplayPrimitiveConsumerStatus,
    rejection_reason: HardwareReplayPrimitiveRejectionReason,
    replay_status: &'static str,
    evidence_source: &'static str,
    generated_placeholder: bool,
    assignments: Btor2HardwareReplayDecisionAssignments,
) -> HardwareReplayPrimitiveStatus {
    HardwareReplayPrimitiveStatus {
        hardware: "BTOR2",
        verdict: "unsafe",
        primitive: "unsafe_counterexample_trace",
        ay_backend_code: BackendKind::AYChc.code(),
        replay_api: "ay_chc_trace_validity_replay_obligations",
        replay_status,
        evidence_source,
        generated_placeholder,
        typed_assignment_source: assignments.typed_assignment_source,
        replay_assignment_status: assignments.replay_assignment_status,
        typed_assignment_required_slots: assignments.typed_assignment_required_slots,
        typed_assignment_present_slots: assignments.typed_assignment_present_slots,
        typed_assignment_missing_slots: assignments.typed_assignment_missing_slots,
        consumer_status,
        rejection_reason,
    }
}

fn validate_btor2_replay_boundary_metadata(
    artifact: &Btor2UnsafeProofReplayArtifact,
) -> Result<(), Btor2ConcreteTraceReplayRejection> {
    if artifact.evidence.iter().any(|row| {
        row.contains("generated_placeholder=true")
            || row.contains("MCC hardware_fallback")
            || row.contains("mcc-generated")
    }) {
        return Err(Btor2ConcreteTraceReplayRejection::GeneratedPlaceholderEvidence);
    }
    if !artifact
        .evidence
        .iter()
        .any(|row| row.as_str() == btor2_proof_replay_boundary_row())
    {
        return Err(Btor2ConcreteTraceReplayRejection::MissingProofReplayBoundaryEvidence);
    }
    if !artifact
        .evidence
        .iter()
        .any(|row| row.as_str() == btor2_unsafe_replay_gate_row())
    {
        return Err(Btor2ConcreteTraceReplayRejection::MissingUnsafeReplayGateEvidence);
    }
    if artifact.property_index.is_none() {
        return Err(Btor2ConcreteTraceReplayRejection::MissingPropertyAttribution);
    }
    if artifact.bad_property_count > 1 && artifact.query_clause.is_none() {
        return Err(Btor2ConcreteTraceReplayRejection::MissingQueryClauseAttribution);
    }
    if artifact.property_attribution != "query_clause"
        && artifact.property_attribution != "single_property"
    {
        return Err(Btor2ConcreteTraceReplayRejection::MissingProvenPropertyAttributionMode);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Btor2AYConsumerTraceAssignmentSummary {
    unsafe_trace_status: String,
    ay_trace_steps: usize,
    btor2_trace_steps: usize,
    state_var_count: usize,
    expected_slots: usize,
    assignment_fields: usize,
    typed_predicate_argument_assignments: usize,
    projected_btor2_assignments: usize,
    missing_typed_predicate_argument_assignments: usize,
    present_assignment_fields: Vec<String>,
}

impl Btor2AYConsumerTraceAssignmentSummary {
    fn assignment_status(&self) -> &'static str {
        if self.missing_typed_predicate_argument_assignments == 0 {
            "complete"
        } else {
            "incomplete"
        }
    }

    fn present_assignment_fields_token(&self) -> String {
        if self.present_assignment_fields.is_empty() {
            return "none".to_string();
        }
        evidence_token(&self.present_assignment_fields.join(","))
    }
}

fn validate_ay_consumer_evidence_for_btor2_replay(
    artifact: &Btor2UnsafeProofReplayArtifact,
    evidence: &ChcProofTranscriptConsumerEvidence,
) -> Result<Btor2AYConsumerTraceAssignmentSummary, Btor2ConcreteTraceReplayRejection> {
    if evidence.normalized_input_sha256 != artifact.normalized_chc_input_sha256 {
        return Err(
            Btor2ConcreteTraceReplayRejection::AYConsumerEvidenceProblemHashMismatch {
                expected: artifact.normalized_chc_input_sha256.clone(),
                actual: evidence.normalized_input_sha256.clone(),
            },
        );
    }
    if evidence.verdict_code != "unsafe"
        || !evidence.accepted_for_consumer
        || !evidence.model_validated
        || evidence.verification_level_code != "ay_chc_verified_counterexample"
    {
        return Err(
            Btor2ConcreteTraceReplayRejection::AYConsumerEvidenceResultMismatch {
                verdict_code: evidence.verdict_code.clone(),
                accepted_for_consumer: evidence.accepted_for_consumer,
                model_validated: evidence.model_validated,
                verification_level_code: evidence.verification_level_code.clone(),
            },
        );
    }
    let summary = summarize_ay_consumer_trace_assignments(
        evidence,
        artifact.trace_steps,
        artifact.state_var_count,
    )?;
    if summary.missing_typed_predicate_argument_assignments > 0 {
        return Err(
            Btor2ConcreteTraceReplayRejection::TypedAYTraceAssignmentsIncomplete {
                expected_slots: summary.expected_slots,
                assignment_fields: summary.assignment_fields,
                typed_predicate_argument_assignments: summary.typed_predicate_argument_assignments,
                missing_typed_predicate_argument_assignments: summary
                    .missing_typed_predicate_argument_assignments,
                present_assignment_fields: summary.present_assignment_fields_token(),
            },
        );
    }
    Ok(summary)
}

fn summarize_ay_consumer_trace_assignments(
    evidence: &ChcProofTranscriptConsumerEvidence,
    btor2_trace_steps: usize,
    state_var_count: usize,
) -> Result<Btor2AYConsumerTraceAssignmentSummary, Btor2ConcreteTraceReplayRejection> {
    let trace = evidence
        .unsafe_trace
        .as_ref()
        .ok_or(Btor2ConcreteTraceReplayRejection::MissingTypedAYUnsafeTrace)?;
    let mut present_assignment_fields = Vec::new();
    let mut assignment_fields = 0;
    let mut typed_predicate_argument_assignments = 0;
    let mut projected_btor2_assignments = 0;
    let mut missing_typed_predicate_argument_assignments = 0;

    for step_index in 0..btor2_trace_steps {
        let ay_step = trace
            .steps
            .iter()
            .find(|step| step.step_index == step_index as u64);
        if let Some(ay_step) = ay_step {
            assignment_fields += ay_step.assignments.len();
            for assignment in &ay_step.assignments {
                present_assignment_fields.push(format!(
                    "{}:{}:{}",
                    step_index,
                    assignment.name,
                    assignment
                        .predicate_argument_index
                        .map_or_else(|| "none".to_string(), |index| index.to_string())
                ));
                if assignment.predicate_argument_index.is_some() {
                    typed_predicate_argument_assignments += 1;
                }
            }
        }

        for argument_index in 0..state_var_count {
            if ay_step
                .and_then(|step| {
                    step.assignments.iter().find(|assignment| {
                        assignment.predicate_argument_index == Some(argument_index as u64)
                    })
                })
                .is_some()
            {
                projected_btor2_assignments += 1;
            } else {
                missing_typed_predicate_argument_assignments += 1;
            }
        }
    }

    present_assignment_fields.sort();
    present_assignment_fields.dedup();

    Ok(Btor2AYConsumerTraceAssignmentSummary {
        unsafe_trace_status: trace.status.clone(),
        ay_trace_steps: trace.steps.len(),
        btor2_trace_steps,
        state_var_count,
        expected_slots: btor2_trace_steps.saturating_mul(state_var_count),
        assignment_fields,
        typed_predicate_argument_assignments,
        projected_btor2_assignments,
        missing_typed_predicate_argument_assignments,
        present_assignment_fields,
    })
}

fn project_btor2_trace_assignments_from_ay_consumer_evidence(
    counterexample: &mut ay_chc::Counterexample,
    state_vars: &[StateVarEntry],
    evidence: &ChcProofTranscriptConsumerEvidence,
) -> Result<Btor2AYConsumerTraceAssignmentSummary, Btor2ConcreteTraceReplayRejection> {
    let trace = evidence
        .unsafe_trace
        .as_ref()
        .ok_or(Btor2ConcreteTraceReplayRejection::MissingTypedAYUnsafeTrace)?;

    for counterexample_step in &mut counterexample.steps {
        counterexample_step.assignments.retain(|name, _| {
            !state_vars
                .iter()
                .any(|state_var| state_var.var.name == *name)
        });
    }

    for ay_step in &trace.steps {
        let step_index = ay_step.step_index as usize;
        let Some(counterexample_step) = counterexample.steps.get_mut(step_index) else {
            continue;
        };
        for (argument_index, state_var) in state_vars.iter().enumerate() {
            let Some(assignment) = ay_step.assignments.iter().find(|assignment| {
                assignment.predicate_argument_index == Some(argument_index as u64)
            }) else {
                continue;
            };
            counterexample_step
                .assignments
                .insert(state_var.var.name.clone(), assignment.value);
            counterexample_step.assignments.insert(
                format!("{}_{}", state_var.var.name, step_index),
                assignment.value,
            );
        }
    }

    summarize_ay_consumer_trace_assignments(evidence, counterexample.steps.len(), state_vars.len())
}

fn btor2_ay_consumer_trace_assignment_row(
    evidence: &ChcProofTranscriptConsumerEvidence,
    summary: &Btor2AYConsumerTraceAssignmentSummary,
) -> String {
    let assignment_contract = bmc_unsafe_trace_assignment_contract();
    let assignment_contract_required_fields = assignment_contract.required_fields.join(",");
    let assignment_contract_supported_sorts = assignment_contract.supported_sort_families.join(",");
    let assignment_contract_fail_closed_sorts =
        assignment_contract.fail_closed_sort_families.join(",");
    format!(
        "BTOR2 ay_consumer_trace_assignments schema={} verdict_code={} backend_code={} accepted_for_consumer={} model_validated={} verification_level_code={} unsafe_trace_status={} ay_trace_steps={} btor2_trace_steps={} state_var_count={} assignment_fields={} typed_predicate_argument_assignments={} projected_btor2_assignments={} missing_typed_predicate_argument_assignments={} present_assignment_fields={} replay_assignment_status={} assignment_contract_schema={} assignment_contract_schema_version={} assignment_contract_scope={} assignment_contract_canonical_name_format={} assignment_contract_required_fields={} assignment_contract_supported_sort_families={} assignment_contract_fail_closed_sort_families={} assignment_contract_unsupported_sort_reason_code={} assignment_contract_value_out_of_range_reason_code={}",
        evidence_token(evidence.schema),
        evidence.verdict_code,
        evidence.backend_code,
        evidence.accepted_for_consumer,
        evidence.model_validated,
        evidence.verification_level_code,
        summary.unsafe_trace_status,
        summary.ay_trace_steps,
        summary.btor2_trace_steps,
        summary.state_var_count,
        summary.assignment_fields,
        summary.typed_predicate_argument_assignments,
        summary.projected_btor2_assignments,
        summary.missing_typed_predicate_argument_assignments,
        summary.present_assignment_fields_token(),
        summary.assignment_status(),
        evidence_token(assignment_contract.schema),
        assignment_contract.schema_version,
        evidence_token(assignment_contract.scope),
        evidence_token(assignment_contract.canonical_name_format),
        evidence_token(&assignment_contract_required_fields),
        evidence_token(&assignment_contract_supported_sorts),
        evidence_token(&assignment_contract_fail_closed_sorts),
        assignment_contract.unsupported_sort_reason_code,
        assignment_contract.value_out_of_range_reason_code,
    )
}

fn is_replayable_btor2_trace_obligation(
    obligation: &ChcReplayObligation,
    normalized_chc_input_sha256: &str,
) -> bool {
    obligation.kind == ChcReplayObligationKind::TraceValidity
        && obligation.smtlib.contains("; expected-result: sat")
        && obligation.smtlib.contains("(check-sat)")
        && obligation.smtlib.contains(normalized_chc_input_sha256)
}

fn btor2_proof_replay_boundary_row() -> &'static str {
    "BTOR2 proof_replay_boundary ay_backend_code=ay_chc safe_proof=ay_chc_verified_result safe_replay=not_available unsafe_witness=ay_chc_counterexample unsafe_replay=not_available witness_attribution=query_clause local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first"
}

fn btor2_unsafe_replay_gate_row() -> &'static str {
    "BTOR2 replay_api_gate verdict=unsafe artifact_kind=verified_chc_result_unsafe api_backend=AYChc api_backend_code=ay_chc replay_api=ay_chc_verified_result replay_status=delegated_to_ay acceptance_gate=verified_chc_result_unsafe failure_policy=fail_closed_no_local_production evidence_basis=ay_chc_counterexample production_routing_status_code=ay_first"
}

fn btor2_unsafe_trace_replay_evidence(
    normalized_chc_input_sha256: &str,
    time_budget: Duration,
    obligation_id: &str,
    counterexample_certificate: &str,
    replay_obligation_artifacts: &[ChcReplayObligationArtifact],
) -> Option<ChcReplayEvidence> {
    if replay_obligation_artifacts.is_empty() {
        return None;
    }

    let options = ChcProofEvidenceOptions::portfolio(time_budget, false);
    let solver = ChcProofSolverIdentity::new("bmc");
    let mut evidence = ChcReplayEvidence::new(
        normalized_chc_input_sha256,
        options.identity_sha256(),
        solver.identity_sha256(),
        obligation_id,
        "unsafe",
        "verified-counterexample",
    )
    .with_proof(ChcProofArtifactDigest::from_bytes(
        "proof-certificate",
        counterexample_certificate.as_bytes(),
    ))
    .with_counterexample(ChcProofArtifactDigest::counterexample_from_bytes(
        counterexample_certificate.as_bytes(),
    ));

    for artifact in replay_obligation_artifacts {
        evidence = evidence.with_replay_obligation(artifact.clone());
    }

    Some(evidence)
}

/// Build a real BTOR2 unsafe proof/replay artifact when the ay-chc portfolio
/// produces a verified counterexample.
///
/// Returns `Ok(None)` for safe/unknown runs; callers that require replay
/// evidence should fail closed on `None`.
pub fn btor2_unsafe_proof_replay_artifact(
    program: &Btor2Program,
    config: &PortfolioConfig,
) -> Result<Option<Btor2UnsafeProofReplayArtifact>, Btor2Error> {
    if program.bad_properties.is_empty() {
        return Ok(None);
    }

    let capability_report = btor2_portfolio_capability_report(program, config);
    let translation = translate_to_chc(program)?;
    let solved_problem = translation.problem.clone();
    let bmc_budget = config
        .time_budget
        .unwrap_or_else(|| Duration::from_secs(10));
    let bmc_config = AYBmcConfig::default()
        .with_max_depth(config.bmc_max_depth as usize)
        .with_time_budget(bmc_budget);
    let proof_run = engines::solve_bmc_proof(translation.problem, bmc_config).map_err(|error| {
        Btor2Error::ParseError {
            line: 0,
            message: format!("ay-chc BMC proof facade failed: {error}"),
        }
    })?;
    let result = proof_run.result.clone();
    let ay_consumer_evidence = proof_run.consumer_evidence(&solved_problem);

    let VerifiedChcResult::Unsafe(counterexample) = result else {
        return Ok(None);
    };

    let mut counterexample = counterexample.counterexample().clone();
    let assignment_summary = project_btor2_trace_assignments_from_ay_consumer_evidence(
        &mut counterexample,
        &translation.state_vars,
        &ay_consumer_evidence,
    )
    .map_err(|error| Btor2Error::ParseError {
        line: 0,
        message: error.to_string(),
    })?;
    let query_clause = counterexample
        .witness
        .as_ref()
        .and_then(|witness| witness.query_clause);
    let query_property_index = query_clause.and_then(|clause_idx| {
        translation
            .property_indices
            .iter()
            .position(|&property_clause| property_clause == clause_idx)
    });
    let bad_property_count = program.bad_properties.len();
    let (property_index, property_attribution) = match (query_property_index, bad_property_count) {
        (Some(index), _) => (Some(index), "query_clause"),
        (None, 1) => (Some(0), "single_property"),
        (None, _) => (None, "missing"),
    };
    let witness_entries = counterexample
        .witness
        .as_ref()
        .map_or(0, |witness| witness.entries.len());
    let normalized_sha = normalized_chc_input_sha256(&solved_problem);

    let assignment_unavailable_reason = if assignment_summary
        .missing_typed_predicate_argument_assignments
        == 0
    {
        None
    } else {
        Some(format!(
                "typed ay consumer unsafe_trace assignments incomplete: expected_slots={} assignment_fields={} typed_predicate_argument_assignments={} missing_typed_predicate_argument_assignments={} present_assignment_fields={}",
                assignment_summary.expected_slots,
                assignment_summary.assignment_fields,
                assignment_summary.typed_predicate_argument_assignments,
                assignment_summary.missing_typed_predicate_argument_assignments,
                assignment_summary.present_assignment_fields_token(),
            ))
    };
    let (replay_obligations, replay_unavailable_reason, replay_unavailable_reason_code) =
        if let Some(reason) = assignment_unavailable_reason {
            (
                Vec::new(),
                Some(reason),
                "typed_ay_trace_assignments_incomplete",
            )
        } else {
            match counterexample.trace_validity_replay_obligations(&solved_problem) {
                Ok(obligations) if !obligations.is_empty() => (obligations, None, "none"),
                Ok(_) => (
                    Vec::new(),
                    Some("ay-chc produced no trace-validity replay obligations".to_string()),
                    "concrete_trace_assignments_unavailable",
                ),
                Err(error) => (
                    Vec::new(),
                    Some(error.to_string()),
                    "concrete_trace_assignments_unavailable",
                ),
            }
        };
    let replay_obligation_artifacts: Vec<_> = replay_obligations
        .iter()
        .map(|obligation| {
            ChcReplayObligationArtifact::new(
                obligation.kind,
                ChcProofArtifactDigest::from_bytes(
                    "replay-obligation",
                    obligation.smtlib.as_bytes(),
                ),
            )
        })
        .collect();
    let counterexample_certificate = counterexample.to_certificate(&solved_problem);
    let replay_evidence = btor2_unsafe_trace_replay_evidence(
        &normalized_sha,
        bmc_budget,
        &format!("btor2:unsafe:{normalized_sha}"),
        &counterexample_certificate,
        &replay_obligation_artifacts,
    );

    let mut evidence = capability_report.evidence;
    evidence.push(btor2_ay_consumer_trace_assignment_row(
        &ay_consumer_evidence,
        &assignment_summary,
    ));
    evidence.push(format!(
        "BTOR2 real_proof_replay_artifact verdict=unsafe bad_property_count={} property_index={} query_clause={} property_attribution={} trace_steps={} state_var_count={} witness_entries={} ay_backend_code=ay_chc solver=ay_chc_bmc_only trace_assignment_source=ay_chc_consumer_evidence replay_api=ay_chc_trace_validity_replay_obligations replay_status={} replay_obligations={} replay_unavailable_reason={} replay_unavailable_reason_code={} typed_ay_consumer_evidence_status=present typed_replay_evidence_status={} typed_replay_evidence_sha256={} normalized_chc_input_sha256={} evidence_source=real_solver generated_placeholder=false",
        bad_property_count,
        evidence_usize(property_index),
        evidence_usize(query_clause),
        property_attribution,
        counterexample.steps.len(),
        translation.state_vars.len(),
        witness_entries,
        if replay_obligations.is_empty() {
            "not_available"
        } else {
            "generated"
        },
        replay_obligations.len(),
        replay_unavailable_reason
            .as_deref()
            .map(evidence_token)
            .unwrap_or_else(|| "none".to_string()),
        replay_unavailable_reason_code,
        if replay_evidence.is_some() {
            "bound"
        } else {
            "not_available"
        },
        replay_evidence
            .as_ref()
            .map(ChcReplayEvidence::identity_sha256)
            .unwrap_or_else(|| "none".to_string()),
        normalized_sha,
    ));
    for artifact in &replay_obligation_artifacts {
        evidence.push(format!(
            "BTOR2 real_replay_obligation kind={} query_sha256={} query_identity_sha256={} obligation_identity_sha256={} evidence_source=ay_chc generated_placeholder=false",
            artifact.kind.as_str(),
            artifact.query.sha256,
            artifact.query.identity_sha256(),
            artifact.identity_sha256(),
        ));
    }

    let mut artifact = Btor2UnsafeProofReplayArtifact {
        bad_property_count,
        property_index,
        query_clause,
        property_attribution,
        trace_steps: counterexample.steps.len(),
        state_var_count: translation.state_vars.len(),
        witness_entries,
        normalized_chc_input_sha256: normalized_sha,
        replay_obligations,
        replay_obligation_artifacts,
        replay_evidence,
        ay_consumer_evidence: Some(ay_consumer_evidence),
        replay_unavailable_reason,
        evidence,
    };
    let replay_primitive_status = btor2_hardware_replay_primitive_status(&artifact);
    artifact
        .evidence
        .push(replay_primitive_status.render_evidence_row());
    artifact
        .evidence
        .push(btor2_hardware_replay_decision_evidence(&artifact));
    Ok(Some(artifact))
}

fn evidence_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

fn evidence_token(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

/// Report the shared backend lanes the BTOR2 portfolio can use without solving.
pub fn btor2_portfolio_capability_report(
    program: &Btor2Program,
    config: &PortfolioConfig,
) -> CapabilityReport {
    let mut report = CapabilityReport::new(ProblemKind::Safety).with_limits(SolverLimits {
        time_budget: config.time_budget,
        max_depth: Some(config.bmc_max_depth),
        max_states: None,
        max_memory_bytes: None,
    });

    let portfolio_capability = BackendCapability::available(
        BackendDomain::Btor2,
        BackendKind::Btor2Portfolio,
        format!(
            "BTOR2 portfolio with {} bad properties, COI={}, BMC={}, CHC fallback",
            program.bad_properties.len(),
            config.enable_coi,
            config.enable_bmc
        ),
    )
    .for_problem(ProblemKind::Safety)
    .with_facets([
        SolverFacet::BitVector,
        SolverFacet::Bmc,
        SolverFacet::Chc,
        SolverFacet::Pdr,
        SolverFacet::KInduction,
    ])
    .with_role(CapabilityRole::Validation)
    .with_detail("BTOR2 local orchestration/preprocessing; solver verdicts delegate to ay-chc");
    select_btor2_lane(&mut report, portfolio_capability);

    if program.bad_properties.is_empty() {
        report.add_evidence(
            "BTOR2 portfolio has no bad properties; no ay-chc production lane required",
        );
        reject_no_property_ay_lane(&mut report, ProblemKind::Bmc);
        reject_no_property_ay_lane(&mut report, ProblemKind::Chc);
        reject_no_property_ay_sat_lane(&mut report);
        reject_missing_native_kernel(&mut report);
        add_btor2_symbolic_execution_evidence(&mut report, program);
        add_btor2_shared_engine_evidence(&mut report, program);
        add_btor2_routing_evidence(&mut report);
        return report;
    }

    if config.enable_bmc {
        select_btor2_lane(
            &mut report,
            btor2_ay_chc_lane(
                ProblemKind::Bmc,
                "BTOR2 BMC preprocessing delegates through ay-chc portfolio",
            ),
        );
    } else {
        report.add_evidence("BTOR2 BMC preprocessing disabled by policy");
        reject_disabled_bmc_lane(&mut report);
    }
    select_btor2_lane(
        &mut report,
        btor2_ay_chc_lane(
            ProblemKind::Chc,
            "BTOR2 full solving delegates through ay-chc adaptive portfolio",
        ),
    );
    reject_non_default_ay_sat_lane(&mut report);
    reject_missing_native_kernel(&mut report);
    add_btor2_symbolic_execution_evidence(&mut report, program);
    add_btor2_shared_engine_evidence(&mut report, program);
    add_btor2_routing_evidence(&mut report);
    report
}

fn select_btor2_lane(report: &mut CapabilityReport, capability: BackendCapability) {
    add_btor2_lane_evidence(report, CapabilityLaneDecision::Selected, &capability);
    report.select(capability);
}

fn reject_btor2_lane(report: &mut CapabilityReport, capability: BackendCapability) {
    add_btor2_lane_evidence(report, CapabilityLaneDecision::Rejected, &capability);
    report.reject(capability);
}

fn add_btor2_lane_evidence(
    report: &mut CapabilityReport,
    decision: CapabilityLaneDecision,
    capability: &BackendCapability,
) {
    report.add_evidence(capability.render_lane_evidence("BTOR2", decision));
    report.add_evidence(capability.render_lane_status_evidence("BTOR2", decision));
}

fn add_btor2_routing_evidence(report: &mut CapabilityReport) {
    report.add_evidence(report.render_production_routing_status_evidence("BTOR2"));
    report.add_evidence(format!(
        "BTOR2 routing_summary production_routing_status={} ay_selected_for_production={} has_unjustified_local_production={}",
        report.production_routing_status_name(),
        report.ay_selected_for_production(),
        report.has_unjustified_local_production()
    ));
    add_btor2_proof_replay_boundary_evidence(report);
    add_btor2_handoff_evidence(report);
    if let Some(reason_code) = report.rejection_reason_code(BackendKind::NativeKernel) {
        report.add_evidence(format!(
            "BTOR2 unsupported_reason backend={} code={reason_code}",
            BackendKind::NativeKernel.name(),
        ));
    }
}

fn add_btor2_proof_replay_boundary_evidence(report: &mut CapabilityReport) {
    report.add_evidence(format!(
        "BTOR2 proof_replay_boundary ay_backend_code={} safe_proof=ay_chc_verified_result safe_replay=not_available unsafe_witness=ay_chc_counterexample unsafe_replay=not_available witness_attribution=query_clause local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code={}",
        BackendKind::AYChc.code(),
        report.production_routing_status_code(),
    ));
    add_btor2_replay_api_gate_evidence(report);
}

fn add_btor2_replay_api_gate_evidence(report: &mut CapabilityReport) {
    for gate in BTOR2_REPLAY_API_GATES {
        report.add_evidence(format!(
            "BTOR2 replay_api_gate verdict={} artifact_kind={} api_backend={} api_backend_code={} replay_api={} replay_status={} acceptance_gate={} failure_policy={} evidence_basis={} production_routing_status_code={}",
            gate.verdict,
            gate.artifact_kind,
            gate.backend.name(),
            gate.backend.code(),
            gate.replay_api,
            gate.replay_status,
            gate.acceptance_gate,
            gate.failure_policy,
            gate.evidence_basis,
            report.production_routing_status_code(),
        ));
    }
}

fn add_btor2_handoff_evidence(report: &mut CapabilityReport) {
    let ay_handoffs: Vec<_> = report
        .selected
        .iter()
        .filter(|capability| capability.backend.is_ay())
        .map(|capability| ("delegated", capability.clone()))
        .chain(
            report
                .rejected
                .iter()
                .filter(|capability| capability.backend.is_ay())
                .map(|capability| ("rejected", capability.clone())),
        )
        .collect();
    for (handoff_status, capability) in ay_handoffs {
        report.add_evidence(format!(
            "BTOR2 ay_handoff handoff_status={handoff_status} from_backend={} to_backend={} to_backend_code={} to_problem={} to_role={} to_status={} reason_code={}",
            BackendKind::Btor2Portfolio.name(),
            capability.backend.name(),
            capability.backend.code(),
            capability.problem.map_or("None", ProblemKind::name),
            capability.role.code(),
            capability.status.code(),
            capability.normalized_reason_code()
        ));
        add_btor2_ay_handoff_detail_evidence(
            report,
            if handoff_status == "delegated" {
                CapabilityLaneDecision::Selected
            } else {
                CapabilityLaneDecision::Rejected
            },
            handoff_status,
            &capability,
        );
    }

    let native_handoffs: Vec<_> = report
        .selected
        .iter()
        .filter(|capability| capability.backend == BackendKind::NativeKernel)
        .map(|capability| ("available", capability.clone()))
        .chain(
            report
                .rejected
                .iter()
                .filter(|capability| capability.backend == BackendKind::NativeKernel)
                .map(|capability| ("deferred", capability.clone())),
        )
        .collect();
    for (handoff_status, capability) in native_handoffs {
        report.add_evidence(format!(
            "BTOR2 native_handoff handoff_status={handoff_status} from_backend={} to_backend={} to_backend_code={} to_problem={} to_role={} to_status={} reason_code={}",
            BackendKind::Btor2Portfolio.name(),
            BackendKind::NativeKernel.name(),
            BackendKind::NativeKernel.code(),
            capability.problem.map_or("None", ProblemKind::name),
            capability.role.code(),
            capability.status.code(),
            capability.normalized_reason_code()
        ));
    }
}

fn add_btor2_ay_handoff_detail_evidence(
    report: &mut CapabilityReport,
    decision: CapabilityLaneDecision,
    handoff_status: &str,
    capability: &BackendCapability,
) {
    let production_routing_status = report.production_routing_status_name();
    let production_routing_status_code = report.production_routing_status_code();
    let local_fallback_status = btor2_local_fallback_status(report);
    let evidence = format!(
        "BTOR2 ay_handoff_detail lane_status={} handoff_status={handoff_status} from_backend={} to_backend={} to_backend_code={} to_problem={} to_problem_code={} to_role={} to_status={} reason_code={} production_routing_status={} production_routing_status_code={} local_fallback_status={}",
        decision.action(),
        BackendKind::Btor2Portfolio.name(),
        capability.backend.name(),
        capability.backend.code(),
        capability.problem.map_or("None", ProblemKind::name),
        capability.problem.map_or("none", ProblemKind::code),
        capability.role.code(),
        capability.status.code(),
        capability.normalized_reason_code(),
        production_routing_status,
        production_routing_status_code,
        local_fallback_status
    );
    report.add_evidence(evidence);
}

fn btor2_local_fallback_status(report: &CapabilityReport) -> &'static str {
    match report.production_routing_status() {
        ProductionRoutingStatus::JustifiedLocalFallback => "justified_local_fallback",
        ProductionRoutingStatus::UnjustifiedLocalFallback => "unjustified_local_fallback",
        _ => "not_selected",
    }
}

fn add_btor2_symbolic_execution_evidence(report: &mut CapabilityReport, program: &Btor2Program) {
    if program.bad_properties.is_empty() {
        report.add_evidence(
            SymbolicExecutionDetection::not_detected()
                .render_evidence("BTOR2", ProblemKind::Safety),
        );
        return;
    }

    let detection =
        SymbolicExecutionDetection::ay_preferred(SymbolicExecutionReason::BitVectorFormula);
    report.add_evidence(detection.render_evidence("BTOR2", ProblemKind::Chc));
    report.add_evidence(detection.render_evidence("BTOR2", ProblemKind::Sat));
}

fn add_btor2_shared_engine_evidence(report: &mut CapabilityReport, program: &Btor2Program) {
    for evidence in btor2_shared_engine_evidence_rows(program) {
        report.add_evidence(evidence);
    }
}

fn btor2_ay_chc_lane(problem: ProblemKind, detail: &'static str) -> BackendCapability {
    ay_chc_capability(BackendDomain::Btor2, problem)
        .with_role(CapabilityRole::Production)
        .with_detail(detail)
}

fn reject_disabled_bmc_lane(report: &mut CapabilityReport) {
    reject_btor2_lane(
        report,
        BackendCapability::disabled(
            BackendDomain::Btor2,
            BackendKind::AYChc,
            UnsupportedReason::DisabledByPolicy("BTOR2 BMC preprocessing disabled"),
        )
        .for_problem(ProblemKind::Bmc)
        .with_facets([SolverFacet::BitVector, SolverFacet::Bmc, SolverFacet::Chc])
        .with_role(CapabilityRole::Production)
        .with_detail("BTOR2 configuration keeps the BMC preprocessing lane out of production"),
    );
}

fn reject_no_property_ay_lane(report: &mut CapabilityReport, problem: ProblemKind) {
    reject_btor2_lane(
        report,
        BackendCapability::unsupported(
            BackendDomain::Btor2,
            BackendKind::AYChc,
            UnsupportedReason::UnsupportedFragment("BTOR2 program has no bad properties"),
        )
        .for_problem(problem)
        .with_facets(ay_chc_problem_facets(problem))
        .with_role(CapabilityRole::Production)
        .with_detail("no property obligation exists for ay-chc to solve"),
    );
}

fn reject_no_property_ay_sat_lane(report: &mut CapabilityReport) {
    reject_btor2_lane(
        report,
        BackendCapability::unsupported(
            BackendDomain::Btor2,
            BackendKind::AYSat,
            UnsupportedReason::UnsupportedFragment("BTOR2 program has no bad properties"),
        )
        .for_problem(ProblemKind::Sat)
        .with_facets([SolverFacet::BitVector, SolverFacet::Sat])
        .with_role(CapabilityRole::Production)
        .with_detail("no property obligation exists for ay-sat to solve"),
    );
}

fn reject_non_default_ay_sat_lane(report: &mut CapabilityReport) {
    reject_btor2_lane(
        report,
        BackendCapability::disabled(
            BackendDomain::Btor2,
            BackendKind::AYSat,
            UnsupportedReason::DisabledByPolicy(
                "BTOR2 SAT/bitblast handoff is not a production default",
            ),
        )
        .for_problem(ProblemKind::Sat)
        .with_facets([SolverFacet::BitVector, SolverFacet::Sat])
        .with_role(CapabilityRole::Production)
        .with_detail("BTOR2 production defaults remain on the ay-chc portfolio"),
    );
}

fn ay_chc_problem_facets(problem: ProblemKind) -> [SolverFacet; 3] {
    match problem {
        ProblemKind::Bmc => [SolverFacet::BitVector, SolverFacet::Bmc, SolverFacet::Chc],
        ProblemKind::Chc => [SolverFacet::BitVector, SolverFacet::Chc, SolverFacet::Pdr],
        _ => [SolverFacet::BitVector, SolverFacet::Chc, SolverFacet::Pdr],
    }
}

fn reject_missing_native_kernel(report: &mut CapabilityReport) {
    reject_btor2_lane(
        report,
        BackendCapability::unsupported(
            BackendDomain::Btor2,
            BackendKind::NativeKernel,
            UnsupportedReason::NativeKernelUnavailable,
        )
        .for_problem(ProblemKind::NativeSuccessor)
        .with_facets([SolverFacet::NativeCodegen])
        .with_role(CapabilityRole::Validation)
        .with_detail("BTOR2 has no shared successor-kernel adapter yet"),
    );
}

/// Run the portfolio and return shared backend capability evidence with it.
///
/// Behaves like [`check_btor2_portfolio`] but additionally produces a
/// [`CapabilityReport`] describing the backend lanes exercised for this program.
///
/// # Errors
///
/// Propagates any [`Btor2Error`] from [`check_btor2_portfolio`].
pub fn check_btor2_portfolio_with_report(
    program: &Btor2Program,
    config: &PortfolioConfig,
) -> Result<(Vec<Btor2CheckResult>, PortfolioStats, CapabilityReport), Btor2Error> {
    let mut report = btor2_portfolio_capability_report(program, config);
    let (results, stats) = check_btor2_portfolio(program, config)?;
    report.add_evidence(format!(
        "BTOR2 portfolio result_phase={:?} states_before={} states_after={}",
        stats.result_phase, stats.states_before_coi, stats.states_after_coi
    ));
    Ok((results, stats, report))
}

/// Run the full portfolio strategy on a BTOR2 program.
///
/// Executes the configured stages (COI reduction, simplification, BMC
/// preprocessing, full CHC solving) under `config` and returns the per-`bad`
/// [`Btor2CheckResult`]s together with a [`PortfolioStats`] record of which
/// phase produced the answer and where time was spent. A program with no `bad`
/// properties returns an empty result vector and a zeroed `stats`.
///
/// # Errors
///
/// Returns [`Btor2Error`] if translating the (possibly COI-reduced) program to
/// CHC fails — e.g. a bitvector sort wider than [`MAX_BV_WIDTH`](crate::error::MAX_BV_WIDTH),
/// an unparseable constant, or an undefined node reference.
pub fn check_btor2_portfolio(
    program: &Btor2Program,
    config: &PortfolioConfig,
) -> Result<(Vec<Btor2CheckResult>, PortfolioStats), Btor2Error> {
    let start = Instant::now();

    if program.bad_properties.is_empty() {
        return Ok((
            vec![],
            PortfolioStats {
                states_before_coi: program.num_states,
                states_after_coi: program.num_states,
                inputs_before_coi: program.num_inputs,
                inputs_after_coi: program.num_inputs,
                coi_time: Duration::ZERO,
                bmc_time: Duration::ZERO,
                chc_time: Duration::ZERO,
                total_time: Duration::ZERO,
                result_phase: ResultPhase::None,
            },
        ));
    }

    let n = program.bad_properties.len();

    // -----------------------------------------------------------------------
    // Phase 1: COI reduction
    // -----------------------------------------------------------------------
    let coi_start = Instant::now();
    let working_program = if config.enable_coi {
        let coi = compute_coi(program);
        if config.verbose && (coi.eliminated_states > 0 || coi.eliminated_inputs > 0) {
            eprintln!(
                "COI: eliminated {}/{} states, {}/{} inputs",
                coi.eliminated_states,
                program.num_states,
                coi.eliminated_inputs,
                program.num_inputs,
            );
        }
        if coi.eliminated_states > 0 || coi.eliminated_inputs > 0 {
            reduce_program(program, &coi)
        } else {
            program.clone()
        }
    } else {
        program.clone()
    };
    let coi_time = coi_start.elapsed();

    let states_after = working_program.num_states;
    let inputs_after = working_program.num_inputs;

    // -----------------------------------------------------------------------
    // Phase 0g: GPU bit-parallel falsification (feature-independent; dlopen).
    // Runs on the COI-reduced program — the same program the BMC/CHC engines
    // check, so trace state names match their convention (COI preserves the
    // bad-property count and order: bads are the cone roots). Falsification-only: a device hit is
    // replayed scalar-side bit-for-bit and mapped to a word-level named
    // trace through the bit-blaster's tables (the same trust base as the
    // eligibility-gated SAT handoff); a clean / ineligible / CUDA-less run
    // falls through to COI+BMC+CHC unchanged. Budget: a short leading slice
    // so the lane can never starve the word-level engines.
    // -----------------------------------------------------------------------
    {
        let gpu_deadline = config.time_budget.map(|total| {
            let slice = (total.as_secs_f64() * 0.1).clamp(1.0, 10.0);
            start + Duration::from_secs_f64(slice)
        });
        if working_program.bad_properties.len() == n {
            if let Some(hit) = crate::gpu_falsify::try_gpu_falsify(&working_program, gpu_deadline) {
                let mut results = Vec::with_capacity(n);
                for i in 0..n {
                    if i == hit.bad_index {
                        results.push(Btor2CheckResult::Sat {
                            trace: hit.trace.clone(),
                            // Bit-level GPU trace is already replayable bit-for-bit;
                            // no word-level model needed.
                            model: None,
                        });
                    } else {
                        results.push(Btor2CheckResult::Unknown {
                            reason: "GPU falsification: other property violated".to_string(),
                        });
                    }
                }
                return Ok((
                    results,
                    PortfolioStats {
                        states_before_coi: program.num_states,
                        states_after_coi: states_after,
                        inputs_before_coi: program.num_inputs,
                        inputs_after_coi: inputs_after,
                        coi_time,
                        bmc_time: Duration::ZERO,
                        chc_time: Duration::ZERO,
                        total_time: start.elapsed(),
                        result_phase: ResultPhase::Gpu,
                    },
                ));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 0h: GPU exhaustive bounded model checking (feature-independent;
    // dlopen). The BTOR2 mirror of AIGER's exhaustive-BMC lane: unrolls the
    // bit-blasted transition relation `bmc_max_depth` steps into ONE
    // combinational circuit and enumerates ALL free-variable assignments on the
    // GPU. Within the free-variable cap it is a COMPLETE bounded decision, so it
    // can find a counterexample the probabilistic Phase-0g walker misses; the
    // satisfying assignment is replayed to a verified word-level trace. A
    // `BoundedSafe` proof or any decline (nondeterministic init / constraints /
    // over the cap / non-CUDA / ineligible) falls through to BMC+CHC unchanged —
    // bounded safety is not full safety without the completeness argument the
    // word-level engines supply.
    // -----------------------------------------------------------------------
    {
        let gpu_deadline = config.time_budget.map(|total| {
            let slice = (total.as_secs_f64() * 0.1).clamp(1.0, 10.0);
            start + Duration::from_secs_f64(slice)
        });
        if working_program.bad_properties.len() == n {
            if let Some(crate::gpu_exhaustive::GpuExhaustBmc::Unsafe { bad_index, trace }) =
                crate::gpu_exhaustive::try_gpu_exhaustive_bmc(
                    &working_program,
                    config.bmc_max_depth as usize,
                    gpu_deadline,
                )
            {
                let mut results = Vec::with_capacity(n);
                for i in 0..n {
                    if i == bad_index {
                        results.push(Btor2CheckResult::Sat {
                            trace: trace.clone(),
                            // Verified bit-level GPU trace — already replayable.
                            model: None,
                        });
                    } else {
                        results.push(Btor2CheckResult::Unknown {
                            reason: "GPU exhaustive BMC: other property violated".to_string(),
                        });
                    }
                }
                return Ok((
                    results,
                    PortfolioStats {
                        states_before_coi: program.num_states,
                        states_after_coi: states_after,
                        inputs_before_coi: program.num_inputs,
                        inputs_after_coi: inputs_after,
                        coi_time,
                        bmc_time: Duration::ZERO,
                        chc_time: Duration::ZERO,
                        total_time: start.elapsed(),
                        result_phase: ResultPhase::Gpu,
                    },
                ));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: BMC preprocessing
    // -----------------------------------------------------------------------
    let bmc_start = Instant::now();
    if config.enable_bmc {
        let bmc_budget = match config.time_budget {
            Some(total) => {
                let bmc_secs = total.as_secs_f64() * config.bmc_budget_fraction;
                Duration::from_secs_f64(bmc_secs.max(1.0))
            }
            None => Duration::from_secs(5),
        };

        let bmc_config = BmcConfig {
            max_depth: config.bmc_max_depth,
            time_budget: bmc_budget,
        };

        match bmc_preprocess(&working_program, &bmc_config)? {
            BmcPreResult::Unsafe { results } => {
                let bmc_time = bmc_start.elapsed();
                if config.verbose {
                    eprintln!("BMC: found result in {:.3}s", bmc_time.as_secs_f64());
                }
                return Ok((
                    results,
                    PortfolioStats {
                        states_before_coi: program.num_states,
                        states_after_coi: states_after,
                        inputs_before_coi: program.num_inputs,
                        inputs_after_coi: inputs_after,
                        coi_time,
                        bmc_time,
                        chc_time: Duration::ZERO,
                        total_time: start.elapsed(),
                        result_phase: ResultPhase::Bmc,
                    },
                ));
            }
            BmcPreResult::Inconclusive {
                depth_reached,
                elapsed,
            } => {
                if config.verbose {
                    eprintln!(
                        "BMC: inconclusive after depth {} ({:.3}s)",
                        depth_reached,
                        elapsed.as_secs_f64()
                    );
                }
            }
        }
    }
    let bmc_time = bmc_start.elapsed();

    // -----------------------------------------------------------------------
    // Phase 3: Full CHC solving
    // -----------------------------------------------------------------------
    let chc_start = Instant::now();

    // Calculate remaining time budget.
    let chc_budget = match config.time_budget {
        Some(total) => {
            let elapsed_so_far = start.elapsed();
            if elapsed_so_far >= total {
                // Out of time.
                let results: Vec<Btor2CheckResult> = (0..n)
                    .map(|_| Btor2CheckResult::Unknown {
                        reason: "portfolio: out of time after BMC phase".to_string(),
                    })
                    .collect();
                return Ok((
                    results,
                    PortfolioStats {
                        states_before_coi: program.num_states,
                        states_after_coi: states_after,
                        inputs_before_coi: program.num_inputs,
                        inputs_after_coi: inputs_after,
                        coi_time,
                        bmc_time,
                        chc_time: Duration::ZERO,
                        total_time: start.elapsed(),
                        result_phase: ResultPhase::None,
                    },
                ));
            }
            Some(total.checked_sub(elapsed_so_far).unwrap())
        }
        None => None,
    };

    let translation = translate_to_chc(&working_program)?;

    // Capture the CHC problem before it is moved into the portfolio, so a SAFE
    // verdict can be independently re-verified on a fresh solver below.
    let chc_problem = translation.problem.clone();

    let adaptive_config = match chc_budget {
        Some(budget) => AdaptiveConfig::with_budget(budget, config.verbose),
        None => AdaptiveConfig::default(),
    };
    let portfolio = AdaptivePortfolio::new(translation.problem, adaptive_config);
    let result = portfolio.solve();

    let chc_time = chc_start.elapsed();

    let results = match result {
        VerifiedChcResult::Safe(inv) => {
            // Independently re-verify the SAFE invariant on a fresh solver
            // before surfacing it. Fail-CLOSED: only a confirmed Ok(true) (the
            // invariant provably excludes every bad state) surfaces SAFE/Unsat.
            // An Err means the re-verification could not be completed, so the
            // safety proof is unverifiable -> Unknown, never SAFE. Ok(false)
            // means the invariant permits a bad state -> Unknown (soundness
            // alert).
            match engines::external_invariant_model_excludes_error(
                &chc_problem,
                inv.model(),
                &PdrConfig::default(),
            ) {
                Ok(true) => (0..n).map(|_| Btor2CheckResult::Unsat).collect(),
                Ok(false) => (0..n)
                    .map(|_| Btor2CheckResult::Unknown {
                        reason: "btor2/aiger SAFE invariant permits a bad state on \
                                 independent re-verify"
                            .to_string(),
                    })
                    .collect(),
                Err(e) => (0..n)
                    .map(|_| Btor2CheckResult::Unknown {
                        reason: format!(
                            "btor2/aiger SAFE invariant could not be independently \
                             re-verified: {e}"
                        ),
                    })
                    .collect(),
            }
        }
        VerifiedChcResult::Unsafe(cex) => {
            let trace: Vec<FxHashMap<String, i64>> = cex
                .counterexample()
                .steps
                .iter()
                .map(|step| {
                    step.assignments
                        .iter()
                        .map(|(name, value)| (name.clone(), *value))
                        .collect()
                })
                .collect();

            // Recover the concrete per-frame word-level model from the
            // derivation witness. Reconstructed over `working_program` (the
            // translated program); COI preserves state/input node ids, so the
            // model keys align with the original program for witness projection.
            let model = crate::word_replay::reconstruct_model(
                &working_program,
                &translation.state_vars,
                cex.counterexample(),
            );

            let violated_prop = cex
                .counterexample()
                .witness
                .as_ref()
                .and_then(|w| w.query_clause)
                .and_then(|clause_idx| {
                    translation
                        .property_indices
                        .iter()
                        .position(|&pi| pi == clause_idx)
                });

            let mut results = Vec::with_capacity(n);
            if let Some(prop_idx) = violated_prop {
                for i in 0..n {
                    if i == prop_idx {
                        results.push(Btor2CheckResult::Sat {
                            trace: trace.clone(),
                            model: model.clone(),
                        });
                    } else {
                        results.push(Btor2CheckResult::Unknown {
                            reason: "other property violated".to_string(),
                        });
                    }
                }
            } else if n == 1 {
                results.push(Btor2CheckResult::Sat { trace, model });
            } else {
                for _ in 0..n {
                    results.push(Btor2CheckResult::Unknown {
                        reason: "counterexample found but property unknown".to_string(),
                    });
                }
            }
            results
        }
        _ => (0..n)
            .map(|_| Btor2CheckResult::Unknown {
                reason: "solver budget exhausted".to_string(),
            })
            .collect(),
    };

    Ok((
        results,
        PortfolioStats {
            states_before_coi: program.num_states,
            states_after_coi: states_after,
            inputs_before_coi: program.num_inputs,
            inputs_after_coi: inputs_after,
            coi_time,
            bmc_time,
            chc_time,
            total_time: start.elapsed(),
            result_phase: ResultPhase::Chc,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Btor2Line, Btor2Node, Btor2Sort};
    use ay_chc::{Counterexample, CounterexampleStep};

    use std::collections::HashMap;
    use tla_mc_core::{CapabilityLaneDecision, CapabilityStatus, ProductionRoutingStatus};

    /// Simple counter: 0 -> 1 -> 2 -> 3, bad = (count == 3).
    fn make_counter_program() -> Btor2Program {
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(8));
        sorts.insert(10, Btor2Sort::BitVec(1));

        let lines = vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(8),
                args: vec![],
            },
            Btor2Line {
                id: 10,
                sort_id: 0,
                node: Btor2Node::SortBitVec(1),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 1,
                node: Btor2Node::Zero,
                args: vec![],
            },
            Btor2Line {
                id: 3,
                sort_id: 1,
                node: Btor2Node::State(1, Some("count".to_string())),
                args: vec![],
            },
            Btor2Line {
                id: 4,
                sort_id: 1,
                node: Btor2Node::Init(1, 3, 2),
                args: vec![3, 2],
            },
            Btor2Line {
                id: 5,
                sort_id: 1,
                node: Btor2Node::One,
                args: vec![],
            },
            Btor2Line {
                id: 6,
                sort_id: 1,
                node: Btor2Node::Add,
                args: vec![3, 5],
            },
            Btor2Line {
                id: 7,
                sort_id: 1,
                node: Btor2Node::Next(1, 3, 6),
                args: vec![3, 6],
            },
            Btor2Line {
                id: 8,
                sort_id: 1,
                node: Btor2Node::ConstD("3".to_string()),
                args: vec![],
            },
            Btor2Line {
                id: 9,
                sort_id: 10,
                node: Btor2Node::Eq,
                args: vec![3, 8],
            },
            Btor2Line {
                id: 11,
                sort_id: 0,
                node: Btor2Node::Bad(9),
                args: vec![9],
            },
        ];

        Btor2Program {
            lines,
            sorts,
            num_inputs: 0,
            num_states: 1,
            bad_properties: vec![11],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        }
    }

    fn evidence_contains(report: &CapabilityReport, expected: &str) -> bool {
        report.evidence.iter().any(|evidence| evidence == expected)
    }

    const BTOR2_PROOF_REPLAY_BOUNDARY_ROW: &str = "BTOR2 proof_replay_boundary ay_backend_code=ay_chc safe_proof=ay_chc_verified_result safe_replay=not_available unsafe_witness=ay_chc_counterexample unsafe_replay=not_available witness_attribution=query_clause local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first";

    const BTOR2_UNSAFE_REPLAY_GATE_ROW: &str = "BTOR2 replay_api_gate verdict=unsafe artifact_kind=verified_chc_result_unsafe api_backend=AYChc api_backend_code=ay_chc replay_api=ay_chc_verified_result replay_status=delegated_to_ay acceptance_gate=verified_chc_result_unsafe failure_policy=fail_closed_no_local_production evidence_basis=ay_chc_counterexample production_routing_status_code=ay_first";

    fn btor2_hardware_replay_decision_row(artifact: &Btor2UnsafeProofReplayArtifact) -> &str {
        artifact
            .evidence
            .iter()
            .find(|row| row.starts_with(&format!("BTOR2 {} ", HARDWARE_REPLAY_DECISION_ROW_KIND)))
            .expect("expected BTOR2 hardware replay decision evidence")
    }

    #[test]
    fn test_btor2_hardware_replay_decision_schema_contract_is_exported() {
        assert_eq!(
            HARDWARE_REPLAY_DECISION_ROW_KIND,
            "hardware_replay_decision"
        );
        assert_eq!(
            HARDWARE_REPLAY_DECISION_SCHEMA,
            "hardware_replay_primitive/v1"
        );
        assert_eq!(HARDWARE_REPLAY_DECISION_SCHEMA_VERSION, 1);
        assert_eq!(
            HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS,
            &[
                "schema",
                "verdict",
                "primitive",
                "decision_status",
                "accepted_replay_primitive",
                "blocked_by_typed_assignment_completeness",
                "blocked_by_placeholder",
                "consumer_status",
                "reason_code",
                "ay_backend_code",
                "replay_api",
                "replay_status",
                "typed_assignment_source",
                "replay_assignment_status",
                "typed_assignment_required_slots",
                "typed_assignment_present_slots",
                "typed_assignment_missing_slots",
                "evidence_source",
                "generated_placeholder",
            ]
        );
    }

    fn real_btor2_proof_replay_artifact() -> Btor2UnsafeProofReplayArtifact {
        let program = make_counter_program();
        let config = PortfolioConfig {
            time_budget: Some(Duration::from_secs(10)),
            ..PortfolioConfig::default()
        };
        btor2_unsafe_proof_replay_artifact(&program, &config)
            .expect("BTOR2 proof/replay artifact generation should not error")
            .expect("counter program should produce a verified unsafe artifact")
    }

    fn fixture_btor2_proof_replay_artifact() -> Btor2UnsafeProofReplayArtifact {
        let program = make_counter_program();
        let config = PortfolioConfig::default();
        let capability_report = btor2_portfolio_capability_report(&program, &config);
        let translation = translate_to_chc(&program).expect("counter program should translate");
        let solved_problem = translation.problem;
        let predicate = solved_problem.predicates()[0].id;
        // CounterexampleStep::new expects ay-chc's deterministic-iteration
        // map (`hashbrown::HashMap<_, _, foldhash::fast::FixedState>` via
        // `ay_core::kani_compat::DetHashMap`), not the rustc-hash
        // `FxHashMap` aliased at the top of this file.
        use ay_core::kani_compat::DetHashMap;
        let steps = [0_i64, 1, 2, 3]
            .into_iter()
            .enumerate()
            .map(|(time, value)| {
                let mut assignments: DetHashMap<String, i64> = DetHashMap::default();
                assignments.insert("count".to_string(), value);
                assignments.insert(format!("count_{time}"), value);
                CounterexampleStep::new(predicate, assignments)
            })
            .collect();
        let counterexample = Counterexample::new(steps);
        let normalized_sha = normalized_chc_input_sha256(&solved_problem);
        let replay_obligations = counterexample
            .trace_validity_replay_obligations(&solved_problem)
            .expect("fixture trace should produce a AY trace-validity replay obligation");
        let replay_obligation_artifacts: Vec<_> = replay_obligations
            .iter()
            .map(|obligation| {
                ChcReplayObligationArtifact::new(
                    obligation.kind,
                    ChcProofArtifactDigest::from_bytes(
                        "replay-obligation",
                        obligation.smtlib.as_bytes(),
                    ),
                )
            })
            .collect();
        let counterexample_certificate = counterexample.to_certificate(&solved_problem);
        let replay_evidence = btor2_unsafe_trace_replay_evidence(
            &normalized_sha,
            Duration::from_secs(10),
            &format!("btor2:fixture:unsafe:{normalized_sha}"),
            &counterexample_certificate,
            &replay_obligation_artifacts,
        );

        let mut evidence = capability_report.evidence;
        evidence.push(format!(
            "BTOR2 real_proof_replay_artifact verdict=unsafe bad_property_count=1 property_index=0 query_clause={} property_attribution=query_clause trace_steps={} state_var_count={} witness_entries=0 ay_backend_code=ay_chc solver=ay_chc_fixture_trace trace_assignment_source=btor2_fixture replay_api=ay_chc_trace_validity_replay_obligations replay_status=generated replay_obligations={} replay_unavailable_reason=none replay_unavailable_reason_code=none typed_ay_consumer_evidence_status=missing typed_replay_evidence_status={} typed_replay_evidence_sha256={} normalized_chc_input_sha256={} evidence_source=btor2_fixture generated_placeholder=false",
            translation.property_indices[0],
            counterexample.steps.len(),
            translation.state_vars.len(),
            replay_obligations.len(),
            if replay_evidence.is_some() {
                "bound"
            } else {
                "not_available"
            },
            replay_evidence
                .as_ref()
                .map(ChcReplayEvidence::identity_sha256)
                .unwrap_or_else(|| "none".to_string()),
            normalized_sha,
        ));
        for artifact in &replay_obligation_artifacts {
            evidence.push(format!(
                "BTOR2 real_replay_obligation kind={} query_sha256={} query_identity_sha256={} obligation_identity_sha256={} evidence_source=ay_chc generated_placeholder=false",
                artifact.kind.as_str(),
                artifact.query.sha256,
                artifact.query.identity_sha256(),
                artifact.identity_sha256(),
            ));
        }

        let mut artifact = Btor2UnsafeProofReplayArtifact {
            bad_property_count: 1,
            property_index: Some(0),
            query_clause: Some(translation.property_indices[0]),
            property_attribution: "query_clause",
            trace_steps: counterexample.steps.len(),
            state_var_count: translation.state_vars.len(),
            witness_entries: 0,
            normalized_chc_input_sha256: normalized_sha,
            replay_obligations,
            replay_obligation_artifacts,
            replay_evidence,
            ay_consumer_evidence: None,
            replay_unavailable_reason: None,
            evidence,
        };
        let replay_primitive_status = btor2_hardware_replay_primitive_status(&artifact);
        artifact
            .evidence
            .push(replay_primitive_status.render_evidence_row());
        artifact
            .evidence
            .push(btor2_hardware_replay_decision_evidence(&artifact));
        artifact
    }

    fn validate_btor2_artifact_metadata(
        artifact: &Btor2UnsafeProofReplayArtifact,
    ) -> Result<(), String> {
        if artifact.evidence.iter().any(|row| {
            row.contains("generated_placeholder=true")
                || row.contains("MCC hardware_fallback")
                || row.contains("mcc-generated")
        }) {
            return Err(
                "generated placeholder evidence is not a real proof replay artifact".into(),
            );
        }
        if !artifact
            .evidence
            .iter()
            .any(|row| row.as_str() == BTOR2_PROOF_REPLAY_BOUNDARY_ROW)
        {
            return Err("missing BTOR2 proof_replay_boundary evidence".into());
        }
        if !artifact
            .evidence
            .iter()
            .any(|row| row.as_str() == BTOR2_UNSAFE_REPLAY_GATE_ROW)
        {
            return Err("missing BTOR2 unsafe replay_api_gate evidence".into());
        }
        if artifact.property_index.is_none() {
            return Err("missing BTOR2 property attribution".into());
        }
        if artifact.bad_property_count > 1 && artifact.query_clause.is_none() {
            return Err("missing BTOR2 query-clause property attribution".into());
        }
        if artifact.property_attribution != "query_clause"
            && artifact.property_attribution != "single_property"
        {
            return Err("missing BTOR2 proven property attribution mode".into());
        }

        Ok(())
    }

    fn validate_real_btor2_proof_replay_artifact(
        artifact: &Btor2UnsafeProofReplayArtifact,
    ) -> Result<(), String> {
        validate_btor2_artifact_metadata(artifact)?;
        if artifact.replay_obligations.is_empty() || artifact.replay_obligation_artifacts.is_empty()
        {
            return Err("missing BTOR2 ay-chc trace-validity replay obligations".into());
        }
        if artifact.replay_obligations.len() != artifact.replay_obligation_artifacts.len() {
            return Err("BTOR2 replay obligation descriptors do not match obligations".into());
        }
        if !artifact
            .replay_obligations
            .iter()
            .any(|obligation| obligation.kind == ChcReplayObligationKind::TraceValidity)
        {
            return Err("missing BTOR2 trace-validity replay obligation".into());
        }
        if !artifact.replay_obligations.iter().all(|obligation| {
            obligation.smtlib.contains("; expected-result: sat")
                && obligation.smtlib.contains("(check-sat)")
                && obligation
                    .smtlib
                    .contains(&artifact.normalized_chc_input_sha256)
        }) {
            return Err("BTOR2 replay obligation is not a replayable SMT-LIB artifact".into());
        }
        if !artifact.evidence.iter().any(|row| {
            row.starts_with("BTOR2 real_proof_replay_artifact ")
                && row.contains("verdict=unsafe")
                && row.contains("ay_backend_code=ay_chc")
                && row.contains("replay_status=generated")
                && (row.contains("evidence_source=real_solver")
                    || row.contains("evidence_source=btor2_fixture"))
                && row.contains("generated_placeholder=false")
        }) {
            return Err("missing BTOR2 replay artifact evidence".into());
        }
        if !artifact.evidence.iter().any(|row| {
            row.starts_with("BTOR2 real_replay_obligation ")
                && row.contains("kind=trace-validity")
                && row.contains("evidence_source=ay_chc")
                && row.contains("generated_placeholder=false")
        }) {
            return Err("missing BTOR2 replay obligation digest evidence".into());
        }
        if !artifact.evidence.iter().any(|row| {
            row == &btor2_hardware_replay_primitive_status(artifact).render_evidence_row()
        }) {
            return Err("missing shared BTOR2 hardware replay primitive evidence".into());
        }
        validate_btor2_hardware_replay_decision_evidence(artifact).map_err(|err| {
            format!(
                "invalid BTOR2 hardware replay decision evidence: reason_code={} {err}",
                err.reason_code()
            )
        })?;

        Ok(())
    }

    #[test]
    fn test_portfolio_finds_bug() {
        let program = make_counter_program();
        let config = PortfolioConfig {
            time_budget: Some(Duration::from_secs(30)),
            enable_coi: true,
            enable_simplify: true,
            enable_bmc: true,
            bmc_budget_fraction: 0.3,
            bmc_max_depth: 10,
            verbose: false,
        };

        let (results, stats, report) =
            check_btor2_portfolio_with_report(&program, &config).expect("should succeed");
        assert_eq!(results.len(), 1);
        match &results[0] {
            Btor2CheckResult::Sat { trace, .. } => {
                assert!(!trace.is_empty());
            }
            other => panic!("expected Sat, got: {:?}", other),
        }
        assert_eq!(stats.states_before_coi, 1);
        assert!(report.has_selected(BackendKind::Btor2Portfolio));
        assert!(report.has_selected(BackendKind::AYChc));
        assert!(report.selected.iter().any(|capability| capability.backend
            == BackendKind::Btor2Portfolio
            && capability.role == CapabilityRole::Validation));
        assert!(report
            .selected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYChc
                && capability.problem == Some(ProblemKind::Bmc)
                && capability.role == CapabilityRole::Production));
        assert!(report
            .selected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYChc
                && capability.problem == Some(ProblemKind::Chc)
                && capability.role == CapabilityRole::Production));
        assert!(report.ay_selected_for_production());
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::AYFirst
        );
        assert_eq!(
            report.rejection_reason(BackendKind::NativeKernel),
            Some(&UnsupportedReason::NativeKernelUnavailable)
        );
        assert!(!report.has_unjustified_local_production());
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 selected_lane backend=AYChc role=Production problem=Bmc status=Available reason_code=none"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 selected_lane backend=AYChc role=Production problem=Chc status=Available reason_code=none"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=NativeKernel role=Validation problem=NativeSuccessor status=Unsupported reason_code=native_kernel_unavailable"));
        assert!(report
            .evidence
            .iter()
            .any(|evidence| evidence == "BTOR2 production_routing_status=AYFirst"));
    }

    #[test]
    fn test_portfolio_capability_report_emits_shared_lane_vocabulary() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=selected backend=Btor2Portfolio backend_code=btor2_portfolio backend_role=validation problem=Safety capability_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=selected backend=AYChc backend_code=ay_chc backend_role=production problem=Bmc capability_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=selected backend=AYChc backend_code=ay_chc backend_role=production problem=Chc capability_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=rejected backend=AYSat backend_code=ay_sat backend_role=production problem=Sat capability_status=disabled reason_code=disabled_by_policy"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=rejected backend=NativeKernel backend_code=native_kernel backend_role=validation problem=NativeSuccessor capability_status=unsupported reason_code=native_kernel_unavailable"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 routing_summary production_routing_status=AYFirst ay_selected_for_production=true has_unjustified_local_production=false"
        ));
    }

    #[test]
    fn test_portfolio_capability_report_emits_symbolic_execution_detection_rows() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(evidence_contains(
            &report,
            "BTOR2 symbolic_execution domain=btor2 status=AYPreferred status_code=ay_preferred problem=Chc reason=BitVectorFormula reason_code=bit_vector_formula preferred_backend=AYChc preferred_backend_code=ay_chc"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 symbolic_execution domain=btor2 status=AYPreferred status_code=ay_preferred problem=Sat reason=BitVectorFormula reason_code=bit_vector_formula preferred_backend=AYSat preferred_backend_code=ay_sat"
        ));

        let mut no_property_program = make_counter_program();
        no_property_program.bad_properties.clear();
        let no_property_report =
            btor2_portfolio_capability_report(&no_property_program, &PortfolioConfig::default());
        assert!(evidence_contains(
            &no_property_report,
            "BTOR2 symbolic_execution domain=btor2 status=NotDetected status_code=not_detected problem=Safety reason=None reason_code=none preferred_backend=None preferred_backend_code=none"
        ));
    }

    #[test]
    fn test_portfolio_capability_report_uses_core_evidence_renderers() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        let bmc_capability = report
            .selected
            .iter()
            .find(|capability| {
                capability.backend == BackendKind::AYChc
                    && capability.problem == Some(ProblemKind::Bmc)
            })
            .expect("BMC production lane should be selected");
        let selected_bmc_evidence =
            bmc_capability.render_lane_evidence("BTOR2", CapabilityLaneDecision::Selected);
        assert!(evidence_contains(&report, &selected_bmc_evidence));

        let native_capability = report
            .rejected
            .iter()
            .find(|capability| capability.backend == BackendKind::NativeKernel)
            .expect("native validation lane should be rejected");
        let rejected_native_evidence =
            native_capability.render_lane_evidence("BTOR2", CapabilityLaneDecision::Rejected);
        assert!(evidence_contains(&report, &rejected_native_evidence));

        let routing_status_evidence = report.render_production_routing_status_evidence("BTOR2");
        assert!(evidence_contains(&report, &routing_status_evidence));
    }

    #[test]
    fn test_portfolio_capability_report_emits_ay_and_native_handoff_vocabulary() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(evidence_contains(
            &report,
            "BTOR2 ay_handoff handoff_status=delegated from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Bmc to_role=production to_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 ay_handoff handoff_status=delegated from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Chc to_role=production to_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 ay_handoff handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Sat to_role=production to_status=disabled reason_code=disabled_by_policy"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 native_handoff handoff_status=deferred from_backend=Btor2Portfolio to_backend=NativeKernel to_backend_code=native_kernel to_problem=NativeSuccessor to_role=validation to_status=unsupported reason_code=native_kernel_unavailable"
        ));
    }

    #[test]
    fn test_portfolio_capability_report_emits_detailed_ay_handoff_status_rows() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(report
            .selected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYChc
                && capability.problem == Some(ProblemKind::Chc)));
        assert!(!report
            .selected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYSat));
        assert!(report
            .rejected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYSat
                && capability.problem == Some(ProblemKind::Sat)
                && capability.reason_code() == Some("disabled_by_policy")));
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::AYFirst
        );

        assert!(evidence_contains(
            &report,
            "BTOR2 ay_handoff_detail lane_status=selected handoff_status=delegated from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Chc to_problem_code=chc to_role=production to_status=available reason_code=none production_routing_status=AYFirst production_routing_status_code=ay_first local_fallback_status=not_selected"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 ay_handoff_detail lane_status=rejected handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Sat to_problem_code=sat to_role=production to_status=disabled reason_code=disabled_by_policy production_routing_status=AYFirst production_routing_status_code=ay_first local_fallback_status=not_selected"
        ));
    }

    #[test]
    fn test_portfolio_capability_report_emits_proof_replay_boundary() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(evidence_contains(
            &report,
            "BTOR2 proof_replay_boundary ay_backend_code=ay_chc safe_proof=ay_chc_verified_result safe_replay=not_available unsafe_witness=ay_chc_counterexample unsafe_replay=not_available witness_attribution=query_clause local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=selected backend=AYChc backend_code=ay_chc backend_role=production problem=Chc capability_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 shared_lane lane_status=rejected backend=NativeKernel backend_code=native_kernel backend_role=validation problem=NativeSuccessor capability_status=unsupported reason_code=native_kernel_unavailable"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 ay_handoff handoff_status=delegated from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Chc to_role=production to_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 native_handoff handoff_status=deferred from_backend=Btor2Portfolio to_backend=NativeKernel to_backend_code=native_kernel to_problem=NativeSuccessor to_role=validation to_status=unsupported reason_code=native_kernel_unavailable"
        ));
    }

    #[test]
    fn test_portfolio_capability_report_emits_real_replay_api_gates() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(evidence_contains(
            &report,
            "BTOR2 replay_api_gate verdict=safe artifact_kind=verified_chc_result_safe api_backend=AYChc api_backend_code=ay_chc replay_api=ay_chc_verified_result replay_status=delegated_to_ay acceptance_gate=verified_chc_result_safe failure_policy=fail_closed_no_local_production evidence_basis=ay_chc_safe_proof production_routing_status_code=ay_first"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 replay_api_gate verdict=unsafe artifact_kind=verified_chc_result_unsafe api_backend=AYChc api_backend_code=ay_chc replay_api=ay_chc_verified_result replay_status=delegated_to_ay acceptance_gate=verified_chc_result_unsafe failure_policy=fail_closed_no_local_production evidence_basis=ay_chc_counterexample production_routing_status_code=ay_first"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 replay_api_gate verdict=unsafe artifact_kind=query_clause_attribution api_backend=AYChc api_backend_code=ay_chc replay_api=property_indices_match replay_status=proven_attribution_only acceptance_gate=query_clause_matches_property_indices failure_policy=multi_property_unknown_without_query_clause evidence_basis=counterexample_witness_query_clause production_routing_status_code=ay_first"
        ));
        assert!(evidence_contains(
            &report,
            "BTOR2 replay_api_gate verdict=unsafe artifact_kind=local_trace_replay api_backend=Btor2Portfolio api_backend_code=btor2_portfolio replay_api=none replay_status=not_available acceptance_gate=not_applicable failure_policy=do_not_report_local_replay evidence_basis=no_btor2_trace_replay_api production_routing_status_code=ay_first"
        ));
        assert!(!report.evidence.iter().any(|evidence| evidence
            .contains("artifact_kind=local_trace_replay")
            && evidence.contains("replay_status=proven")));
    }

    #[test]
    fn test_fixture_btor2_proof_replay_artifact_validates_generated_chc() {
        let artifact = fixture_btor2_proof_replay_artifact();

        validate_real_btor2_proof_replay_artifact(&artifact)
            .expect("BTOR2 fixture proof/replay artifact should validate");
        let consumer_error = btor2_accept_concrete_trace_replay(&artifact)
            .expect_err("fixture replay must not bypass typed AY consumer evidence");
        assert_eq!(
            consumer_error.reason_code(),
            "missing_typed_ay_consumer_evidence"
        );
        let replay_primitive_status = btor2_hardware_replay_primitive_status(&artifact);
        assert_eq!(
            replay_primitive_status.consumer_status,
            HardwareReplayPrimitiveConsumerStatus::Rejected
        );
        assert_eq!(
            replay_primitive_status.reason_code(),
            "missing_typed_ay_consumer_evidence"
        );
        assert_eq!(artifact.property_index, Some(0));
        assert_eq!(artifact.bad_property_count, 1);
        assert_eq!(artifact.property_attribution, "query_clause");
        assert_eq!(artifact.trace_steps, 4);
        assert_eq!(artifact.state_var_count, 1);
        assert!(artifact.replay_unavailable_reason.is_none());
        assert!(artifact.replay_evidence.is_some());
        assert!(artifact.ay_consumer_evidence.is_none());
        assert_eq!(
            artifact.replay_obligations[0].kind,
            ChcReplayObligationKind::TraceValidity
        );
        assert!(
            artifact.replay_obligations[0]
                .smtlib
                .contains("trace-validity"),
            "expected trace-validity replay SMT-LIB, got {}",
            artifact.replay_obligations[0].smtlib
        );
        assert!(artifact.evidence.iter().any(|row| {
            row.starts_with("BTOR2 real_proof_replay_artifact ")
                && row.contains("property_index=0")
                && row.contains("property_attribution=")
                && row.contains("replay_status=generated")
                && row.contains("evidence_source=btor2_fixture")
                && row.contains("generated_placeholder=false")
        }));
        assert!(artifact.evidence.iter().any(|row| {
            row.starts_with("BTOR2 hardware_replay_primitive ")
                && row.contains("schema=hardware_replay_primitive/v1")
                && row.contains("primitive=unsafe_counterexample_trace")
                && row.contains("consumer_status=rejected")
                && row.contains("reason_code=missing_typed_ay_consumer_evidence")
                && row.contains("generated_placeholder=false")
        }));
        let replay_decision_status = btor2_hardware_replay_decision_status(&artifact);
        assert_eq!(
            replay_decision_status.decision_status(),
            HardwareReplayPrimitiveDecisionStatus::Blocked
        );
        assert!(!replay_decision_status.accepted_replay_primitive());
        assert!(!replay_decision_status.blocked_by_typed_assignment_completeness());
        assert!(!replay_decision_status.blocked_by_placeholder());
        assert_eq!(
            replay_decision_status.replay_assignment_status,
            HardwareReplayPrimitiveAssignmentStatus::Missing
        );
        assert_eq!(replay_decision_status.typed_assignment_source, "missing");
        assert_eq!(replay_decision_status.typed_assignment_required_slots, 4);
        assert_eq!(replay_decision_status.typed_assignment_present_slots, 0);
        assert_eq!(replay_decision_status.typed_assignment_missing_slots, 4);
        assert_eq!(
            replay_decision_status.accepted_replay_evidence_identity_sha256,
            "none"
        );
        assert_eq!(
            replay_decision_status.accepted_trace_validity_obligations,
            0
        );
        assert_eq!(
            replay_decision_status.accepted_replay_obligation_identities_sha256,
            "none"
        );
        assert_eq!(
            replay_decision_status.accepted_ay_proof_evidence_status,
            "none"
        );
        assert_eq!(
            replay_decision_status.accepted_ay_proof_evidence_sha256,
            "none"
        );
        let decision_row = btor2_hardware_replay_decision_row(&artifact);
        assert!(decision_row.starts_with("BTOR2 hardware_replay_decision "));
        assert!(decision_row.contains("schema=hardware_replay_primitive/v1"));
        assert!(decision_row.contains("decision_status=blocked"));
        assert!(decision_row.contains("accepted_replay_primitive=false"));
        assert!(decision_row.contains("blocked_by_typed_assignment_completeness=false"));
        assert!(decision_row.contains("blocked_by_placeholder=false"));
        assert!(decision_row.contains("typed_assignment_source=missing"));
        assert!(decision_row.contains("replay_assignment_status=missing"));
        assert!(decision_row.contains("typed_assignment_required_slots=4"));
        assert!(decision_row.contains("typed_assignment_present_slots=0"));
        assert!(decision_row.contains("typed_assignment_missing_slots=4"));
        assert!(decision_row.contains("accepted_replay_evidence_identity_sha256=none"));
        assert!(decision_row.contains("accepted_trace_validity_obligations=0"));
        assert!(decision_row.contains("accepted_replay_obligation_identities_sha256=none"));
        assert!(decision_row.contains("accepted_ay_proof_evidence_status=none"));
        assert!(decision_row.contains("accepted_ay_proof_evidence_sha256=none"));
        assert!(decision_row.contains("reason_code=missing_typed_ay_consumer_evidence"));
        validate_btor2_hardware_replay_decision_evidence(&artifact)
            .expect("fixture blocked decision evidence should validate");
    }

    #[test]
    fn test_btor2_hardware_replay_decision_validator_rejects_missing_and_malformed_rows() {
        let artifact = fixture_btor2_proof_replay_artifact();
        let decision_row = btor2_hardware_replay_decision_row(&artifact);

        let mut without_decision = artifact.clone();
        without_decision.evidence.retain(|row| {
            !row.starts_with(&format!("BTOR2 {} ", HARDWARE_REPLAY_DECISION_ROW_KIND))
        });
        let missing = validate_btor2_hardware_replay_decision_evidence(&without_decision)
            .expect_err("missing decision evidence must fail closed");
        assert_eq!(
            missing.reason_code(),
            "missing_hardware_replay_decision_evidence"
        );

        let mut duplicate = artifact.clone();
        duplicate.evidence.push(decision_row.to_string());
        let duplicate_error = validate_btor2_hardware_replay_decision_evidence(&duplicate)
            .expect_err("duplicate decision evidence must fail closed");
        assert_eq!(
            duplicate_error.reason_code(),
            "duplicate_hardware_replay_decision_evidence"
        );

        let missing_reason = decision_row
            .split_whitespace()
            .filter(|token| !token.starts_with("reason_code="))
            .collect::<Vec<_>>()
            .join(" ");
        let missing_field = validate_btor2_hardware_replay_decision_evidence_row(&missing_reason)
            .expect_err("missing required reason_code must fail closed");
        assert!(matches!(
            missing_field,
            HardwareReplayDecisionEvidenceError::MissingField("reason_code")
        ));

        let unsupported_schema = decision_row.replace(
            "schema=hardware_replay_primitive/v1",
            "schema=hardware_replay_primitive/v2",
        );
        let schema_error =
            validate_btor2_hardware_replay_decision_evidence_row(&unsupported_schema)
                .expect_err("unsupported schema must fail closed");
        assert_eq!(
            schema_error.reason_code(),
            "unsupported_hardware_replay_decision_schema"
        );

        let invalid_bool = decision_row.replace(
            "accepted_replay_primitive=false",
            "accepted_replay_primitive=maybe",
        );
        let bool_error = validate_btor2_hardware_replay_decision_evidence_row(&invalid_bool)
            .expect_err("invalid boolean field must fail closed");
        assert!(matches!(
            bool_error,
            HardwareReplayDecisionEvidenceError::InvalidField {
                field: "accepted_replay_primitive",
                ..
            }
        ));

        let mut stale = artifact.clone();
        stale.evidence.push(
            "BTOR2 MCC hardware_fallback verdict=unsafe generated_placeholder=true".to_string(),
        );
        let stale_error = validate_btor2_hardware_replay_decision_evidence(&stale)
            .expect_err("stale decision row must fail closed when placeholder evidence appears");
        assert!(matches!(
            stale_error,
            HardwareReplayDecisionEvidenceError::InconsistentDecision(
                "decision_row_does_not_match_current_primitive_status"
            )
        ));
    }

    #[test]
    fn test_real_btor2_solver_artifact_accepts_typed_trace_assignments() {
        let artifact = real_btor2_proof_replay_artifact();

        validate_btor2_artifact_metadata(&artifact)
            .expect("real solver artifact metadata should validate");
        validate_real_btor2_proof_replay_artifact(&artifact)
            .expect("typed AY assignments should produce replay obligations");
        assert!(!artifact.replay_obligations.is_empty());
        assert!(!artifact.replay_obligation_artifacts.is_empty());
        assert!(artifact.replay_unavailable_reason.is_none());
        let consumer_evidence = artifact
            .ay_consumer_evidence
            .as_ref()
            .expect("real solver artifact should carry typed AY consumer evidence");
        assert_eq!(consumer_evidence.verdict_code, "unsafe");
        assert!(consumer_evidence.accepted_for_consumer);
        assert!(consumer_evidence.model_validated);
        assert_eq!(
            consumer_evidence.verification_level_code,
            "ay_chc_verified_counterexample"
        );
        let consumer_trace = consumer_evidence
            .unsafe_trace
            .as_ref()
            .expect("unsafe AY consumer evidence should carry trace metadata");
        assert_eq!(consumer_trace.status, "validated_counterexample");
        let replay_acceptance = btor2_accept_concrete_trace_replay(&artifact)
            .expect("complete typed AY assignments should pass the concrete replay gate");
        assert_eq!(
            replay_acceptance.normalized_chc_input_sha256,
            artifact.normalized_chc_input_sha256
        );
        assert!(replay_acceptance.trace_validity_obligations > 0);
        let replay_primitive_status = btor2_hardware_replay_primitive_status(&artifact);
        assert_eq!(
            replay_primitive_status.consumer_status,
            HardwareReplayPrimitiveConsumerStatus::Accepted
        );
        assert_eq!(replay_primitive_status.reason_code(), "none");
        let assignment_row = artifact
            .evidence
            .iter()
            .find(|row| row.starts_with("BTOR2 ay_consumer_trace_assignments "))
            .expect("real solver artifact should summarize typed AY assignment fields");
        assert!(assignment_row.contains("accepted_for_consumer=true"));
        assert!(assignment_row.contains("model_validated=true"));
        assert!(assignment_row.contains("unsafe_trace_status=validated_counterexample"));
        assert!(assignment_row.contains("missing_typed_predicate_argument_assignments=0"));
        assert!(assignment_row.contains("replay_assignment_status=complete"));
        assert!(assignment_row
            .contains("assignment_contract_schema=ay.chc-bmc-unsafe-trace-assignment-contract_v1"));
        assert!(assignment_row.contains(
            "assignment_contract_canonical_name_format=__p_predicate_id__a_predicate_argument_index_"
        ));
        assert!(assignment_row.contains(
            "assignment_contract_required_fields=name_predicate_argument_index_sort_value"
        ));
        assert!(assignment_row
            .contains("assignment_contract_supported_sort_families=Bool_Int_BitVec_width_"));
        assert!(artifact.evidence.iter().any(|row| {
            row.starts_with("BTOR2 real_proof_replay_artifact ")
                && row.contains("replay_status=generated")
                && row.contains("replay_unavailable_reason_code=none")
                && row.contains("trace_assignment_source=ay_chc_consumer_evidence")
                && row.contains("typed_ay_consumer_evidence_status=present")
                && row.contains("evidence_source=real_solver")
                && row.contains("generated_placeholder=false")
        }));
        assert!(artifact.evidence.iter().any(|row| {
            row.starts_with("BTOR2 hardware_replay_primitive ")
                && row.contains("schema=hardware_replay_primitive/v1")
                && row.contains("consumer_status=accepted")
                && row.contains("reason_code=none")
                && row.contains("generated_placeholder=false")
        }));
        let replay_decision_status = btor2_hardware_replay_decision_status(&artifact);
        assert_eq!(
            replay_decision_status.decision_status(),
            HardwareReplayPrimitiveDecisionStatus::Accepted
        );
        assert!(replay_decision_status.accepted_replay_primitive());
        assert!(!replay_decision_status.blocked_by_typed_assignment_completeness());
        assert!(!replay_decision_status.blocked_by_placeholder());
        assert_eq!(replay_decision_status.reason_code(), "none");
        assert_eq!(
            replay_decision_status.replay_assignment_status,
            HardwareReplayPrimitiveAssignmentStatus::Complete
        );
        assert_eq!(
            replay_decision_status.typed_assignment_source,
            "ay_chc_consumer_evidence"
        );
        assert_eq!(replay_decision_status.typed_assignment_required_slots, 4);
        assert_eq!(replay_decision_status.typed_assignment_present_slots, 4);
        assert_eq!(replay_decision_status.typed_assignment_missing_slots, 0);
        assert_eq!(
            replay_decision_status.accepted_replay_evidence_identity_sha256,
            replay_acceptance.replay_evidence_identity_sha256
        );
        assert_eq!(
            replay_decision_status.accepted_trace_validity_obligations,
            replay_acceptance.trace_validity_obligations
        );
        assert_ne!(
            replay_decision_status.accepted_replay_obligation_identities_sha256,
            "none"
        );
        assert_eq!(
            replay_decision_status.accepted_ay_proof_evidence_status,
            replay_acceptance.ay_proof_evidence_status
        );
        assert_eq!(
            replay_decision_status.accepted_ay_proof_evidence_sha256,
            replay_acceptance.ay_proof_evidence_sha256
        );
        let decision_row = btor2_hardware_replay_decision_row(&artifact);
        assert!(decision_row.starts_with("BTOR2 hardware_replay_decision "));
        assert!(decision_row.contains("decision_status=accepted"));
        assert!(decision_row.contains("accepted_replay_primitive=true"));
        assert!(decision_row.contains("blocked_by_typed_assignment_completeness=false"));
        assert!(decision_row.contains("blocked_by_placeholder=false"));
        assert!(decision_row.contains("typed_assignment_source=ay_chc_consumer_evidence"));
        assert!(decision_row.contains("replay_assignment_status=complete"));
        assert!(decision_row.contains("typed_assignment_required_slots=4"));
        assert!(decision_row.contains("typed_assignment_present_slots=4"));
        assert!(decision_row.contains("typed_assignment_missing_slots=0"));
        assert!(decision_row.contains(&format!(
            "accepted_replay_evidence_identity_sha256={}",
            replay_acceptance.replay_evidence_identity_sha256
        )));
        assert!(decision_row.contains(&format!(
            "accepted_trace_validity_obligations={}",
            replay_acceptance.trace_validity_obligations
        )));
        assert!(decision_row.contains("accepted_replay_obligation_identities_sha256="));
        assert!(!decision_row.contains("accepted_replay_obligation_identities_sha256=none"));
        assert!(decision_row.contains(&format!(
            "accepted_ay_proof_evidence_status={}",
            replay_acceptance.ay_proof_evidence_status
        )));
        assert!(decision_row.contains(&format!(
            "accepted_ay_proof_evidence_sha256={}",
            replay_acceptance.ay_proof_evidence_sha256
        )));
        assert!(decision_row.contains("reason_code=none"));
        validate_btor2_hardware_replay_decision_evidence(&artifact)
            .expect("real accepted decision evidence should validate");

        let missing_acceptance_identity = decision_row.replace(
            &format!(
                "accepted_replay_evidence_identity_sha256={}",
                replay_acceptance.replay_evidence_identity_sha256
            ),
            "accepted_replay_evidence_identity_sha256=none",
        );
        assert!(matches!(
            validate_btor2_hardware_replay_decision_evidence_row(&missing_acceptance_identity),
            Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                "accepted_decision_requires_replay_evidence_identity"
            ))
        ));

        let missing_ay_proof_evidence = decision_row.replace(
            &format!(
                "accepted_ay_proof_evidence_sha256={}",
                replay_acceptance.ay_proof_evidence_sha256
            ),
            "accepted_ay_proof_evidence_sha256=none",
        );
        assert!(matches!(
            validate_btor2_hardware_replay_decision_evidence_row(&missing_ay_proof_evidence),
            Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                "accepted_decision_requires_ay_proof_evidence"
            ))
        ));
    }

    #[test]
    fn test_btor2_proof_replay_artifact_validator_fail_closes() {
        let artifact = fixture_btor2_proof_replay_artifact();
        let mut without_boundary = artifact.clone();
        without_boundary
            .evidence
            .retain(|row| row.as_str() != BTOR2_PROOF_REPLAY_BOUNDARY_ROW);
        let missing_boundary = validate_btor2_artifact_metadata(&without_boundary)
            .expect_err("missing proof/replay boundary must fail closed");
        assert!(missing_boundary.contains("proof_replay_boundary"));

        let mut without_obligations = artifact.clone();
        without_obligations.replay_obligations.clear();
        without_obligations.replay_obligation_artifacts.clear();
        let missing_obligations = validate_real_btor2_proof_replay_artifact(&without_obligations)
            .expect_err("string-only proof acceptance must fail closed");
        assert!(missing_obligations.contains("trace-validity replay obligations"));

        let mut row_only = artifact.clone();
        row_only.replay_obligations.clear();
        row_only.replay_obligation_artifacts.clear();
        row_only.replay_evidence = None;
        row_only.replay_unavailable_reason = None;
        let row_only_error = btor2_accept_concrete_trace_replay(&row_only)
            .expect_err("generated rows without typed ay metadata must fail closed");
        assert_eq!(
            row_only_error.reason_code(),
            "missing_typed_ay_replay_evidence"
        );

        let mut placeholder = artifact;
        placeholder.evidence.push(
            "BTOR2 MCC hardware_fallback verdict=unsafe generated_placeholder=true".to_string(),
        );
        let placeholder_error = validate_real_btor2_proof_replay_artifact(&placeholder)
            .expect_err("generated placeholder evidence must fail closed");
        assert!(placeholder_error.contains("generated placeholder"));
        let placeholder_consumer_error = btor2_accept_concrete_trace_replay(&placeholder)
            .expect_err("placeholder evidence must fail closed before typed replay acceptance");
        assert_eq!(
            placeholder_consumer_error.reason_code(),
            "generated_placeholder_evidence"
        );
        let placeholder_primitive_status = btor2_hardware_replay_primitive_status(&placeholder);
        assert_eq!(
            placeholder_primitive_status.consumer_status,
            HardwareReplayPrimitiveConsumerStatus::Rejected
        );
        assert_eq!(
            placeholder_primitive_status.reason_code(),
            "generated_placeholder_evidence"
        );
        assert!(placeholder_primitive_status.generated_placeholder);
    }

    #[test]
    fn test_portfolio_capability_report_handoff_records_rejected_ay_lanes() {
        let program = make_counter_program();
        let disabled_bmc_report = btor2_portfolio_capability_report(
            &program,
            &PortfolioConfig {
                enable_bmc: false,
                ..PortfolioConfig::default()
            },
        );
        assert!(evidence_contains(
            &disabled_bmc_report,
            "BTOR2 ay_handoff handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Bmc to_role=production to_status=disabled reason_code=disabled_by_policy"
        ));
        assert!(evidence_contains(
            &disabled_bmc_report,
            "BTOR2 ay_handoff handoff_status=delegated from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Chc to_role=production to_status=available reason_code=none"
        ));
        assert!(evidence_contains(
            &disabled_bmc_report,
            "BTOR2 ay_handoff handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Sat to_role=production to_status=disabled reason_code=disabled_by_policy"
        ));

        let mut no_property_program = make_counter_program();
        no_property_program.bad_properties.clear();
        let no_property_report =
            btor2_portfolio_capability_report(&no_property_program, &PortfolioConfig::default());
        assert!(evidence_contains(
            &no_property_report,
            "BTOR2 ay_handoff handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Bmc to_role=production to_status=unsupported reason_code=unsupported_fragment"
        ));
        assert!(evidence_contains(
            &no_property_report,
            "BTOR2 ay_handoff handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYChc to_backend_code=ay_chc to_problem=Chc to_role=production to_status=unsupported reason_code=unsupported_fragment"
        ));
        assert!(evidence_contains(
            &no_property_report,
            "BTOR2 ay_handoff handoff_status=rejected from_backend=Btor2Portfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Sat to_role=production to_status=unsupported reason_code=unsupported_fragment"
        ));
    }

    #[test]
    fn test_portfolio_capability_report_marks_disabled_bmc_without_changing_chc_lane() {
        let program = make_counter_program();
        let config = PortfolioConfig {
            enable_bmc: false,
            ..PortfolioConfig::default()
        };

        let report = btor2_portfolio_capability_report(&program, &config);

        assert!(!report
            .selected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYChc
                && capability.problem == Some(ProblemKind::Bmc)));
        assert!(report
            .selected
            .iter()
            .any(|capability| capability.backend == BackendKind::AYChc
                && capability.problem == Some(ProblemKind::Chc)
                && capability.role == CapabilityRole::Production));
        assert!(report
            .evidence
            .iter()
            .any(|evidence| evidence == "BTOR2 BMC preprocessing disabled by policy"));
        let bmc_rejection = report
            .rejected
            .iter()
            .find(|capability| {
                capability.backend == BackendKind::AYChc
                    && capability.problem == Some(ProblemKind::Bmc)
            })
            .expect("disabled BMC lane should be rejected with shared metadata");
        assert_eq!(bmc_rejection.status, CapabilityStatus::Disabled);
        assert_eq!(bmc_rejection.role, CapabilityRole::Production);
        assert_eq!(
            bmc_rejection.reason,
            Some(UnsupportedReason::DisabledByPolicy(
                "BTOR2 BMC preprocessing disabled"
            ))
        );
        assert_eq!(bmc_rejection.reason_code(), Some("disabled_by_policy"));
        assert!(bmc_rejection.facets.contains(&SolverFacet::BitVector));
        assert!(bmc_rejection.facets.contains(&SolverFacet::Bmc));
        assert!(!report.has_unjustified_local_production());
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=AYChc role=Production problem=Bmc status=Disabled reason_code=disabled_by_policy"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 selected_lane backend=AYChc role=Production problem=Chc status=Available reason_code=none"));
        assert!(report
            .evidence
            .iter()
            .any(|evidence| evidence == "BTOR2 production_routing_status=AYFirst"));
    }

    #[test]
    fn test_portfolio_capability_report_no_bad_properties_uses_no_solver_lane() {
        let mut program = make_counter_program();
        program.bad_properties.clear();
        let config = PortfolioConfig::default();

        let report = btor2_portfolio_capability_report(&program, &config);

        assert!(report.has_selected(BackendKind::Btor2Portfolio));
        assert!(!report.has_selected(BackendKind::AYChc));
        assert!(!report.ay_selected_for_production());
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::NoProductionSelection
        );
        assert_eq!(
            report.rejection_reason(BackendKind::NativeKernel),
            Some(&UnsupportedReason::NativeKernelUnavailable)
        );
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 portfolio has no bad properties; no ay-chc production lane required"));
        let ay_rejections: Vec<_> = report
            .rejected
            .iter()
            .filter(|capability| capability.backend.is_ay())
            .collect();
        assert_eq!(ay_rejections.len(), 3);
        assert!(ay_rejections.iter().all(|capability| {
            capability.status == CapabilityStatus::Unsupported
                && capability.role == CapabilityRole::Production
                && capability.reason_code() == Some("unsupported_fragment")
        }));
        assert!(ay_rejections
            .iter()
            .any(|capability| capability.problem == Some(ProblemKind::Bmc)
                && capability.facets.contains(&SolverFacet::Bmc)));
        assert!(ay_rejections
            .iter()
            .any(|capability| capability.problem == Some(ProblemKind::Chc)
                && capability.facets.contains(&SolverFacet::Pdr)));
        assert!(ay_rejections
            .iter()
            .any(|capability| capability.problem == Some(ProblemKind::Sat)
                && capability.facets.contains(&SolverFacet::Sat)));
        assert!(report.ay_rejected());
        assert!(!report.has_unjustified_local_production());
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=AYChc role=Production problem=Bmc status=Unsupported reason_code=unsupported_fragment"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=AYChc role=Production problem=Chc status=Unsupported reason_code=unsupported_fragment"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=AYSat role=Production problem=Sat status=Unsupported reason_code=unsupported_fragment"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=NativeKernel role=Validation problem=NativeSuccessor status=Unsupported reason_code=native_kernel_unavailable"));
        assert!(report
            .evidence
            .iter()
            .any(|evidence| evidence == "BTOR2 production_routing_status=NoProductionSelection"));
    }

    #[test]
    fn test_portfolio_capability_report_native_unsupported_metadata() {
        let program = make_counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        let native = report
            .rejected
            .iter()
            .find(|capability| capability.backend == BackendKind::NativeKernel)
            .expect("native kernel capability should be rejected with shared metadata");
        assert_eq!(native.status, CapabilityStatus::Unsupported);
        assert_eq!(native.role, CapabilityRole::Validation);
        assert_eq!(native.problem, Some(ProblemKind::NativeSuccessor));
        assert_eq!(
            native.reason,
            Some(UnsupportedReason::NativeKernelUnavailable)
        );
        assert_eq!(native.reason_code(), Some("native_kernel_unavailable"));
        assert!(native.facets.contains(&SolverFacet::NativeCodegen));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 rejected_lane backend=NativeKernel role=Validation problem=NativeSuccessor status=Unsupported reason_code=native_kernel_unavailable"));
        assert!(report.evidence.iter().any(|evidence| evidence
            == "BTOR2 unsupported_reason backend=NativeKernel code=native_kernel_unavailable"));
    }

    #[test]
    fn test_portfolio_with_irrelevant_state() {
        let mut program = make_counter_program();

        // Add irrelevant state "y".
        let y_sort_id = 1;
        program.lines.push(Btor2Line {
            id: 20,
            sort_id: y_sort_id,
            node: Btor2Node::State(y_sort_id, Some("y".to_string())),
            args: vec![],
        });
        program.lines.push(Btor2Line {
            id: 21,
            sort_id: y_sort_id,
            node: Btor2Node::Init(y_sort_id, 20, 2),
            args: vec![20, 2],
        });
        program.lines.push(Btor2Line {
            id: 22,
            sort_id: y_sort_id,
            node: Btor2Node::Next(y_sort_id, 20, 2),
            args: vec![20, 2],
        });
        program.num_states = 2;

        let config = PortfolioConfig {
            time_budget: Some(Duration::from_secs(30)),
            enable_coi: true,
            enable_simplify: true,
            enable_bmc: true,
            bmc_budget_fraction: 0.3,
            bmc_max_depth: 10,
            verbose: false,
        };

        let (results, stats) = check_btor2_portfolio(&program, &config).expect("should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(stats.states_before_coi, 2);
        assert_eq!(stats.states_after_coi, 1, "COI should eliminate y");
        match &results[0] {
            Btor2CheckResult::Sat { .. } => {}
            other => panic!("expected Sat, got: {:?}", other),
        }
    }

    #[test]
    fn test_portfolio_no_bad_properties() {
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(1));
        let program = Btor2Program {
            lines: vec![],
            sorts,
            num_inputs: 0,
            num_states: 0,
            bad_properties: vec![],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        };

        let config = PortfolioConfig::default();
        let (results, _stats) = check_btor2_portfolio(&program, &config).expect("should succeed");
        assert!(results.is_empty());
    }
}
