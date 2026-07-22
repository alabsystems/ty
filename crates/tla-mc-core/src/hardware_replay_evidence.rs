// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared, solver-free hardware replay evidence vocabulary.

use crate::backend_evidence::NO_REASON_CODE;
use crate::evidence_row::evidence_field;

/// Stable evidence schema for hardware replay primitive boundary rows.
pub const HARDWARE_REPLAY_PRIMITIVE_SCHEMA: &str = "hardware_replay_primitive/v1";

/// Stable row kind for proof/replay boundary evidence.
pub const HARDWARE_PROOF_REPLAY_BOUNDARY_ROW_KIND: &str = "proof_replay_boundary";

/// Schema ID for hardware proof/replay boundary rows.
pub const HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA: &str = HARDWARE_REPLAY_PRIMITIVE_SCHEMA;

/// Schema version for hardware proof/replay boundary rows.
pub const HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA_VERSION: u32 = 1;

/// Required key/value fields common to every proof/replay boundary row.
pub const HARDWARE_PROOF_REPLAY_BOUNDARY_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "ay_backend_code",
    "safe_proof",
    "safe_replay",
    "unsafe_witness",
    "unsafe_replay",
    "witness_attribution",
    "local_production_gate",
    "native_promotion_gate",
    "production_routing_status_code",
    "production_selected",
    "fail_closed",
];

/// Stable row kind for actionable hardware replay decision evidence.
pub const HARDWARE_REPLAY_DECISION_ROW_KIND: &str = "hardware_replay_decision";

/// Schema ID for hardware replay decision rows consumed by orchestration.
pub const HARDWARE_REPLAY_DECISION_SCHEMA: &str = HARDWARE_REPLAY_PRIMITIVE_SCHEMA;

/// Schema version for hardware replay decision rows.
pub const HARDWARE_REPLAY_DECISION_SCHEMA_VERSION: u32 = 1;

/// Required key/value fields common to every hardware replay decision row.
pub const HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS: &[&str] = &[
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
];

/// Replay/proof identity fields required for AY CHC trace-validity replay rows.
pub const HARDWARE_REPLAY_DECISION_AY_REPLAY_FIELDS: &[&str] = &[
    "accepted_replay_evidence_identity_sha256",
    "accepted_trace_validity_obligations",
    "accepted_replay_obligation_identities_sha256",
];

/// AY proof fields required for AY CHC trace-validity replay rows.
pub const HARDWARE_REPLAY_DECISION_AY_PROOF_FIELDS: &[&str] = &[
    "accepted_ay_proof_evidence_status",
    "accepted_ay_proof_evidence_sha256",
];

/// Consumer status for a hardware replay primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareReplayPrimitiveConsumerStatus {
    /// The replay primitive was accepted by the consumer gate.
    Accepted,
    /// The replay primitive was rejected by the consumer gate.
    Rejected,
}

impl HardwareReplayPrimitiveConsumerStatus {
    /// Stable status code for evidence rows.
    pub fn code(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }
}

/// Stable hardware replay decision status for orchestration consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareReplayPrimitiveDecisionStatus {
    /// The replay primitive was admitted for the hardware lane.
    Accepted,
    /// The replay primitive was blocked by a fail-closed consumer gate.
    Blocked,
}

impl HardwareReplayPrimitiveDecisionStatus {
    /// Stable status code for decision evidence rows.
    pub fn code(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Blocked => "blocked",
        }
    }
}

/// Assignment completeness status for a hardware replay primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareReplayPrimitiveAssignmentStatus {
    /// The replay evidence carries all assignment slots required by the consumer.
    Complete,
    /// The replay evidence carries assignment slots, but not enough for replay.
    Incomplete,
    /// The replay evidence does not expose typed assignment completeness.
    Missing,
}

impl HardwareReplayPrimitiveAssignmentStatus {
    /// Stable status code for evidence rows.
    pub fn code(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Incomplete => "incomplete",
            Self::Missing => "missing",
        }
    }
}

/// Validation error for hardware replay decision evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HardwareReplayDecisionEvidenceError {
    /// No hardware replay decision row was present.
    MissingDecisionEvidence,
    /// More than one hardware replay decision row was present.
    DuplicateDecisionEvidence,
    /// The row does not use the hardware replay decision row kind.
    WrongRowKind,
    /// A required key/value field is absent.
    MissingField(&'static str),
    /// The row uses an unsupported schema.
    UnsupportedSchema(String),
    /// A field value is syntactically invalid for this schema.
    InvalidField {
        /// Field name.
        field: &'static str,
        /// Field value.
        value: String,
    },
    /// Field values do not match the fail-closed decision contract.
    InconsistentDecision(&'static str),
}

impl HardwareReplayDecisionEvidenceError {
    /// Stable reason code for orchestration logs.
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::MissingDecisionEvidence => "missing_hardware_replay_decision_evidence",
            Self::DuplicateDecisionEvidence => "duplicate_hardware_replay_decision_evidence",
            Self::WrongRowKind => "wrong_hardware_replay_decision_row_kind",
            Self::MissingField(_) => "missing_hardware_replay_decision_field",
            Self::UnsupportedSchema(_) => "unsupported_hardware_replay_decision_schema",
            Self::InvalidField { .. } => "invalid_hardware_replay_decision_field",
            Self::InconsistentDecision(_) => "inconsistent_hardware_replay_decision",
        }
    }
}

impl std::fmt::Display for HardwareReplayDecisionEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDecisionEvidence => {
                write!(formatter, "missing hardware replay decision evidence")
            }
            Self::DuplicateDecisionEvidence => {
                write!(formatter, "duplicate hardware replay decision evidence")
            }
            Self::WrongRowKind => write!(formatter, "wrong hardware replay decision row kind"),
            Self::MissingField(field) => {
                write!(formatter, "missing hardware replay decision field: {field}")
            }
            Self::UnsupportedSchema(schema) => {
                write!(
                    formatter,
                    "unsupported hardware replay decision schema: {schema}"
                )
            }
            Self::InvalidField { field, value } => {
                write!(
                    formatter,
                    "invalid hardware replay decision field {field}={value}"
                )
            }
            Self::InconsistentDecision(reason) => {
                write!(formatter, "inconsistent hardware replay decision: {reason}")
            }
        }
    }
}

impl std::error::Error for HardwareReplayDecisionEvidenceError {}

/// Validation error for hardware proof/replay boundary evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HardwareProofReplayBoundaryEvidenceError {
    /// No hardware proof/replay boundary row was present.
    MissingBoundaryEvidence,
    /// More than one boundary row was present where one was expected.
    DuplicateBoundaryEvidence,
    /// The row does not use the proof/replay boundary row kind.
    WrongRowKind,
    /// A required key/value field is absent.
    MissingField(&'static str),
    /// The row uses an unsupported hardware namespace.
    UnsupportedHardware(String),
    /// The row uses an unsupported schema.
    UnsupportedSchema(String),
    /// A field value is syntactically invalid for this schema.
    InvalidField {
        /// Field name.
        field: &'static str,
        /// Field value.
        value: String,
    },
    /// Field values do not match the fail-closed boundary contract.
    InconsistentBoundary(&'static str),
}

impl HardwareProofReplayBoundaryEvidenceError {
    /// Stable reason code for orchestration logs.
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::MissingBoundaryEvidence => "missing_hardware_proof_replay_boundary_evidence",
            Self::DuplicateBoundaryEvidence => "duplicate_hardware_proof_replay_boundary_evidence",
            Self::WrongRowKind => "wrong_hardware_proof_replay_boundary_row_kind",
            Self::MissingField(_) => "missing_hardware_proof_replay_boundary_field",
            Self::UnsupportedHardware(_) => "unsupported_hardware_proof_replay_boundary_scope",
            Self::UnsupportedSchema(_) => "unsupported_hardware_proof_replay_boundary_schema",
            Self::InvalidField { .. } => "invalid_hardware_proof_replay_boundary_field",
            Self::InconsistentBoundary(_) => "inconsistent_hardware_proof_replay_boundary",
        }
    }
}

impl std::fmt::Display for HardwareProofReplayBoundaryEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBoundaryEvidence => {
                write!(formatter, "missing hardware proof replay boundary evidence")
            }
            Self::DuplicateBoundaryEvidence => {
                write!(
                    formatter,
                    "duplicate hardware proof replay boundary evidence"
                )
            }
            Self::WrongRowKind => write!(formatter, "wrong hardware proof replay row kind"),
            Self::MissingField(field) => {
                write!(formatter, "missing hardware proof replay field: {field}")
            }
            Self::UnsupportedHardware(hardware) => {
                write!(
                    formatter,
                    "unsupported hardware proof replay scope: {hardware}"
                )
            }
            Self::UnsupportedSchema(schema) => {
                write!(
                    formatter,
                    "unsupported hardware proof replay boundary schema: {schema}"
                )
            }
            Self::InvalidField { field, value } => {
                write!(
                    formatter,
                    "invalid hardware proof replay field {field}={value}"
                )
            }
            Self::InconsistentBoundary(reason) => {
                write!(
                    formatter,
                    "inconsistent hardware proof replay boundary: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for HardwareProofReplayBoundaryEvidenceError {}

/// Shared fail-closed reason vocabulary for hardware replay primitive gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HardwareReplayPrimitiveRejectionReason {
    /// No rejection.
    None,
    /// Evidence rows identify generated placeholder material.
    GeneratedPlaceholderEvidence,
    /// Required proof/replay boundary row is absent.
    MissingProofReplayBoundaryEvidence,
    /// Required unsafe replay API gate row is absent.
    MissingUnsafeReplayGateEvidence,
    /// AIGER did not carry a solver-produced replay artifact row.
    MissingRealReplayArtifactEvidence,
    /// No typed AY replay evidence was attached.
    MissingTypedAYReplayEvidence,
    /// No typed AY consumer transcript evidence was attached.
    MissingTypedAYConsumerEvidence,
    /// AY could not export concrete trace assignments for replay.
    ConcreteTraceAssignmentsUnavailable,
    /// No typed trace-validity replay obligation was attached.
    MissingTypedTraceValidityReplayObligation,
    /// Replay obligation descriptors do not match the replay evidence.
    ReplayEvidenceObligationMismatch,
    /// Replay evidence is bound to a different solved problem hash.
    ReplayEvidenceProblemHashMismatch,
    /// Replay evidence does not describe an accepted unsafe proof.
    ReplayEvidenceResultMismatch,
    /// Typed AY unsafe trace assignments are incomplete.
    TypedAYTraceAssignmentsIncomplete,
    /// Property attribution evidence is missing or not proven.
    MissingPropertyAttribution,
    /// Query-clause attribution is missing where it is required.
    MissingQueryClauseAttribution,
    /// Property attribution mode is not one of the proven modes.
    MissingProvenPropertyAttributionMode,
    /// Typed AY unsafe trace material is missing.
    MissingTypedAYUnsafeTrace,
    /// Typed AY counterexample digest is missing.
    MissingTypedAYCounterexampleDigest,
    /// Replay obligation descriptors and obligations have different counts.
    ReplayObligationDescriptorCountMismatch,
    /// A replay obligation digest does not match its query bytes.
    ReplayObligationDigestMismatch,
    /// A replay obligation is not a self-contained replay query.
    ReplayObligationNotReplayable,
    /// Typed AY consumer evidence is bound to a different problem hash.
    AYConsumerEvidenceProblemHashMismatch,
    /// Typed AY consumer evidence does not describe an accepted unsafe result.
    AYConsumerEvidenceResultMismatch,
}

impl HardwareReplayPrimitiveRejectionReason {
    /// Stable machine-readable reason code.
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::None => NO_REASON_CODE,
            Self::GeneratedPlaceholderEvidence => "generated_placeholder_evidence",
            Self::MissingProofReplayBoundaryEvidence => "missing_proof_replay_boundary_evidence",
            Self::MissingUnsafeReplayGateEvidence => "missing_unsafe_replay_api_gate_evidence",
            Self::MissingRealReplayArtifactEvidence => "missing_real_replay_artifact_evidence",
            Self::MissingTypedAYReplayEvidence => "missing_typed_ay_replay_evidence",
            Self::MissingTypedAYConsumerEvidence => "missing_typed_ay_consumer_evidence",
            Self::ConcreteTraceAssignmentsUnavailable => "concrete_trace_assignments_unavailable",
            Self::MissingTypedTraceValidityReplayObligation => {
                "missing_typed_trace_validity_replay_obligation"
            }
            Self::ReplayEvidenceObligationMismatch => "replay_evidence_obligation_mismatch",
            Self::ReplayEvidenceProblemHashMismatch => "replay_evidence_problem_hash_mismatch",
            Self::ReplayEvidenceResultMismatch => "replay_evidence_result_mismatch",
            Self::TypedAYTraceAssignmentsIncomplete => "typed_ay_trace_assignments_incomplete",
            Self::MissingPropertyAttribution => "missing_property_attribution",
            Self::MissingQueryClauseAttribution => "missing_query_clause_attribution",
            Self::MissingProvenPropertyAttributionMode => {
                "missing_proven_property_attribution_mode"
            }
            Self::MissingTypedAYUnsafeTrace => "missing_typed_ay_unsafe_trace",
            Self::MissingTypedAYCounterexampleDigest => "missing_typed_ay_counterexample_digest",
            Self::ReplayObligationDescriptorCountMismatch => {
                "replay_obligation_descriptor_count_mismatch"
            }
            Self::ReplayObligationDigestMismatch => "replay_obligation_digest_mismatch",
            Self::ReplayObligationNotReplayable => "replay_obligation_not_replayable",
            Self::AYConsumerEvidenceProblemHashMismatch => {
                "typed_ay_consumer_evidence_problem_hash_mismatch"
            }
            Self::AYConsumerEvidenceResultMismatch => "typed_ay_consumer_evidence_result_mismatch",
        }
    }
}

/// Typed status for a proof/replay boundary row consumed by MCC sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareProofReplayBoundaryStatus {
    /// Hardware consumer namespace, such as `AIGER` or `BTOR2`.
    pub hardware: &'static str,
    /// AY backend family that owns the solved evidence boundary.
    pub ay_backend_code: &'static str,
    /// Safe proof artifact or proof class covered by the boundary.
    pub safe_proof: &'static str,
    /// Safe replay API required by the hardware consumer.
    pub safe_replay: &'static str,
    /// Unsafe witness artifact or proof class covered by the boundary.
    pub unsafe_witness: &'static str,
    /// Unsafe replay API required by the hardware consumer.
    pub unsafe_replay: &'static str,
    /// Proven attribution source for the replayed witness.
    pub witness_attribution: &'static str,
    /// Gate preventing local-only proof production from being promoted.
    pub local_production_gate: &'static str,
    /// Gate preventing native promotion without replay acceptance.
    pub native_promotion_gate: &'static str,
    /// Current production routing status for the owning runtime sidecar.
    pub production_routing_status_code: &'static str,
    /// Whether this boundary alone selects a production answer lane.
    pub production_selected: bool,
    /// Whether missing/invalid replay evidence closes the hardware lane.
    pub fail_closed: bool,
}

impl HardwareProofReplayBoundaryStatus {
    /// Render the proof/replay boundary using the shared hardware vocabulary.
    pub fn render_evidence_row(&self) -> String {
        format!(
            "{} {} schema={} schema_version={} ay_backend_code={} safe_proof={} safe_replay={} unsafe_witness={} unsafe_replay={} witness_attribution={} local_production_gate={} native_promotion_gate={} production_routing_status_code={} production_selected={} fail_closed={}",
            self.hardware,
            HARDWARE_PROOF_REPLAY_BOUNDARY_ROW_KIND,
            HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA,
            HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA_VERSION,
            self.ay_backend_code,
            self.safe_proof,
            self.safe_replay,
            self.unsafe_witness,
            self.unsafe_replay,
            self.witness_attribution,
            self.local_production_gate,
            self.native_promotion_gate,
            self.production_routing_status_code,
            self.production_selected,
            self.fail_closed,
        )
    }
}

/// Canonical AIGER proof/replay boundary for MCC runtime sidecars.
pub fn aiger_hardware_proof_replay_boundary_status(
    production_routing_status_code: &'static str,
) -> HardwareProofReplayBoundaryStatus {
    HardwareProofReplayBoundaryStatus {
        hardware: "AIGER",
        ay_backend_code: "ay_sat",
        safe_proof: "aiger_safe_witness_validation",
        safe_replay: "validate_safe",
        unsafe_witness: "aiger_counterexample_trace",
        unsafe_replay: "transys_verify_witness",
        witness_attribution: "engine_trace",
        local_production_gate: "no_local_production",
        native_promotion_gate: "fail_closed",
        production_routing_status_code,
        production_selected: false,
        fail_closed: true,
    }
}

/// Canonical BTOR2 proof/replay boundary for MCC runtime sidecars.
pub fn btor2_hardware_proof_replay_boundary_status(
    production_routing_status_code: &'static str,
) -> HardwareProofReplayBoundaryStatus {
    HardwareProofReplayBoundaryStatus {
        hardware: "BTOR2",
        ay_backend_code: "ay_chc",
        safe_proof: "ay_chc_verified_result",
        safe_replay: "not_available",
        unsafe_witness: "ay_chc_counterexample",
        unsafe_replay: "not_available",
        witness_attribution: "query_clause",
        local_production_gate: "no_local_production",
        native_promotion_gate: "fail_closed",
        production_routing_status_code,
        production_selected: false,
        fail_closed: true,
    }
}

/// Canonical proof/replay boundaries required before hardware answer promotion.
pub fn runtime_hardware_proof_replay_boundary_statuses(
    production_routing_status_code: &'static str,
) -> [HardwareProofReplayBoundaryStatus; 2] {
    [
        aiger_hardware_proof_replay_boundary_status(production_routing_status_code),
        btor2_hardware_proof_replay_boundary_status(production_routing_status_code),
    ]
}

/// Typed status for the shared hardware replay primitive boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareReplayPrimitiveStatus {
    /// Hardware consumer namespace, such as `AIGER` or `BTOR2`.
    pub hardware: &'static str,
    /// Verdict guarded by this replay primitive.
    pub verdict: &'static str,
    /// Shared primitive being accepted or rejected.
    pub primitive: &'static str,
    /// AY backend family that owns the solved evidence boundary.
    pub ay_backend_code: &'static str,
    /// Replay API required for consumer admission.
    pub replay_api: &'static str,
    /// Replay API status.
    pub replay_status: &'static str,
    /// Source of the evidence being classified.
    pub evidence_source: &'static str,
    /// Whether the evidence was generated placeholder material.
    pub generated_placeholder: bool,
    /// Source of typed assignments used by the replay consumer.
    pub typed_assignment_source: String,
    /// Assignment completeness status reported at the replay boundary.
    pub replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus,
    /// Number of assignment slots required for replay.
    pub typed_assignment_required_slots: usize,
    /// Number of typed assignment slots present in the replay evidence.
    pub typed_assignment_present_slots: usize,
    /// Number of required typed assignment slots missing from replay evidence.
    pub typed_assignment_missing_slots: usize,
    /// Consumer gate status.
    pub consumer_status: HardwareReplayPrimitiveConsumerStatus,
    /// Fail-closed reason code, or `none` for accepted evidence.
    pub rejection_reason: HardwareReplayPrimitiveRejectionReason,
}

impl HardwareReplayPrimitiveStatus {
    /// Stable reason code for this status.
    pub fn reason_code(&self) -> &'static str {
        self.rejection_reason.reason_code()
    }

    /// Actionable lane decision derived from the primitive consumer status.
    pub fn decision_status(&self) -> HardwareReplayPrimitiveDecisionStatus {
        match self.consumer_status {
            HardwareReplayPrimitiveConsumerStatus::Accepted => {
                HardwareReplayPrimitiveDecisionStatus::Accepted
            }
            HardwareReplayPrimitiveConsumerStatus::Rejected => {
                HardwareReplayPrimitiveDecisionStatus::Blocked
            }
        }
    }

    /// Whether this decision admits the replay primitive for the hardware lane.
    pub fn accepted_replay_primitive(&self) -> bool {
        self.decision_status() == HardwareReplayPrimitiveDecisionStatus::Accepted
            && self.rejection_reason == HardwareReplayPrimitiveRejectionReason::None
    }

    /// Whether this decision is blocked by AY typed assignment completeness.
    pub fn blocked_by_typed_assignment_completeness(&self) -> bool {
        matches!(
            self.rejection_reason,
            HardwareReplayPrimitiveRejectionReason::ConcreteTraceAssignmentsUnavailable
                | HardwareReplayPrimitiveRejectionReason::TypedAYTraceAssignmentsIncomplete
        )
    }

    /// Whether this decision is blocked by generated placeholder material.
    pub fn blocked_by_placeholder(&self) -> bool {
        self.generated_placeholder
            || self.rejection_reason
                == HardwareReplayPrimitiveRejectionReason::GeneratedPlaceholderEvidence
    }

    /// Render the status using the shared hardware replay evidence vocabulary.
    pub fn render_evidence_row(&self) -> String {
        format!(
            "{} hardware_replay_primitive schema={} verdict={} primitive={} ay_backend_code={} replay_api={} replay_status={} typed_assignment_source={} replay_assignment_status={} typed_assignment_required_slots={} typed_assignment_present_slots={} typed_assignment_missing_slots={} consumer_status={} reason_code={} evidence_source={} generated_placeholder={}",
            self.hardware,
            HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
            self.verdict,
            self.primitive,
            self.ay_backend_code,
            self.replay_api,
            self.replay_status,
            self.typed_assignment_source,
            self.replay_assignment_status.code(),
            self.typed_assignment_required_slots,
            self.typed_assignment_present_slots,
            self.typed_assignment_missing_slots,
            self.consumer_status.code(),
            self.reason_code(),
            self.evidence_source,
            self.generated_placeholder,
        )
    }

    /// Render the actionable hardware lane decision row for MCC orchestration.
    pub fn render_decision_evidence_row(&self) -> String {
        HardwareReplayDecisionStatus::from_primitive(self).render_evidence_row()
    }
}

/// Typed status for the actionable hardware replay decision row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareReplayDecisionStatus {
    /// Hardware consumer namespace, such as `AIGER` or `BTOR2`.
    pub hardware: &'static str,
    /// Verdict guarded by this replay primitive.
    pub verdict: &'static str,
    /// Shared primitive being accepted or rejected.
    pub primitive: &'static str,
    /// AY backend family that owns the solved evidence boundary.
    pub ay_backend_code: &'static str,
    /// Replay API required for consumer admission.
    pub replay_api: &'static str,
    /// Replay API status.
    pub replay_status: &'static str,
    /// Source of the evidence being classified.
    pub evidence_source: &'static str,
    /// Whether the evidence was generated placeholder material.
    pub generated_placeholder: bool,
    /// Source of typed assignments used by the replay consumer.
    pub typed_assignment_source: String,
    /// Assignment completeness status reported at the replay boundary.
    pub replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus,
    /// Number of assignment slots required for replay.
    pub typed_assignment_required_slots: usize,
    /// Number of typed assignment slots present in the replay evidence.
    pub typed_assignment_present_slots: usize,
    /// Number of required typed assignment slots missing from replay evidence.
    pub typed_assignment_missing_slots: usize,
    /// SHA-256 identity for the accepted replay evidence, or `none`.
    pub accepted_replay_evidence_identity_sha256: String,
    /// Number of accepted trace-validity replay obligations.
    pub accepted_trace_validity_obligations: usize,
    /// Joined SHA-256 identities for accepted replay obligations, or `none`.
    pub accepted_replay_obligation_identities_sha256: String,
    /// AY-owned proof evidence status admitted with the accepted replay row.
    pub accepted_ay_proof_evidence_status: String,
    /// SHA-256 identity for the accepted AY proof evidence, or `none`.
    pub accepted_ay_proof_evidence_sha256: String,
    /// Consumer gate status.
    pub consumer_status: HardwareReplayPrimitiveConsumerStatus,
    /// Fail-closed reason code, or `none` for accepted evidence.
    pub rejection_reason: HardwareReplayPrimitiveRejectionReason,
}

impl HardwareReplayDecisionStatus {
    /// Build a decision row from primitive-only evidence with no AY proof identities.
    pub fn from_primitive(primitive: &HardwareReplayPrimitiveStatus) -> Self {
        Self {
            hardware: primitive.hardware,
            verdict: primitive.verdict,
            primitive: primitive.primitive,
            ay_backend_code: primitive.ay_backend_code,
            replay_api: primitive.replay_api,
            replay_status: primitive.replay_status,
            evidence_source: primitive.evidence_source,
            generated_placeholder: primitive.generated_placeholder,
            typed_assignment_source: primitive.typed_assignment_source.clone(),
            replay_assignment_status: primitive.replay_assignment_status,
            typed_assignment_required_slots: primitive.typed_assignment_required_slots,
            typed_assignment_present_slots: primitive.typed_assignment_present_slots,
            typed_assignment_missing_slots: primitive.typed_assignment_missing_slots,
            accepted_replay_evidence_identity_sha256: "none".to_string(),
            accepted_trace_validity_obligations: 0,
            accepted_replay_obligation_identities_sha256: "none".to_string(),
            accepted_ay_proof_evidence_status: "none".to_string(),
            accepted_ay_proof_evidence_sha256: "none".to_string(),
            consumer_status: primitive.consumer_status,
            rejection_reason: primitive.rejection_reason,
        }
    }

    /// Stable reason code for this status.
    pub fn reason_code(&self) -> &'static str {
        self.rejection_reason.reason_code()
    }

    /// Actionable lane decision derived from the primitive consumer status.
    pub fn decision_status(&self) -> HardwareReplayPrimitiveDecisionStatus {
        match self.consumer_status {
            HardwareReplayPrimitiveConsumerStatus::Accepted => {
                HardwareReplayPrimitiveDecisionStatus::Accepted
            }
            HardwareReplayPrimitiveConsumerStatus::Rejected => {
                HardwareReplayPrimitiveDecisionStatus::Blocked
            }
        }
    }

    /// Whether this decision admits the replay primitive for the hardware lane.
    pub fn accepted_replay_primitive(&self) -> bool {
        self.decision_status() == HardwareReplayPrimitiveDecisionStatus::Accepted
            && self.rejection_reason == HardwareReplayPrimitiveRejectionReason::None
    }

    /// Whether this decision is blocked by AY typed assignment completeness.
    pub fn blocked_by_typed_assignment_completeness(&self) -> bool {
        matches!(
            self.rejection_reason,
            HardwareReplayPrimitiveRejectionReason::ConcreteTraceAssignmentsUnavailable
                | HardwareReplayPrimitiveRejectionReason::TypedAYTraceAssignmentsIncomplete
        )
    }

    /// Whether this decision is blocked by generated placeholder material.
    pub fn blocked_by_placeholder(&self) -> bool {
        self.generated_placeholder
            || self.rejection_reason
                == HardwareReplayPrimitiveRejectionReason::GeneratedPlaceholderEvidence
    }

    /// Render the actionable hardware lane decision row for MCC orchestration.
    pub fn render_evidence_row(&self) -> String {
        let mut row = format!(
            "{} {} schema={} verdict={} primitive={} decision_status={} accepted_replay_primitive={} blocked_by_typed_assignment_completeness={} blocked_by_placeholder={} consumer_status={} reason_code={} ay_backend_code={} replay_api={} replay_status={} typed_assignment_source={} replay_assignment_status={} typed_assignment_required_slots={} typed_assignment_present_slots={} typed_assignment_missing_slots={}",
            self.hardware,
            HARDWARE_REPLAY_DECISION_ROW_KIND,
            HARDWARE_REPLAY_DECISION_SCHEMA,
            self.verdict,
            self.primitive,
            self.decision_status().code(),
            self.accepted_replay_primitive(),
            self.blocked_by_typed_assignment_completeness(),
            self.blocked_by_placeholder(),
            self.consumer_status.code(),
            self.reason_code(),
            self.ay_backend_code,
            self.replay_api,
            self.replay_status,
            self.typed_assignment_source,
            self.replay_assignment_status.code(),
            self.typed_assignment_required_slots,
            self.typed_assignment_present_slots,
            self.typed_assignment_missing_slots,
        );

        use std::fmt::Write;
        if self.replay_api == "ay_chc_trace_validity_replay_obligations" {
            let _ = write!(
                row,
                " accepted_replay_evidence_identity_sha256={} accepted_trace_validity_obligations={} accepted_replay_obligation_identities_sha256={} accepted_ay_proof_evidence_status={} accepted_ay_proof_evidence_sha256={}",
                self.accepted_replay_evidence_identity_sha256,
                self.accepted_trace_validity_obligations,
                self.accepted_replay_obligation_identities_sha256,
                self.accepted_ay_proof_evidence_status,
                self.accepted_ay_proof_evidence_sha256,
            );
        }

        let _ = write!(
            row,
            " evidence_source={} generated_placeholder={}",
            self.evidence_source, self.generated_placeholder
        );
        row
    }
}

/// Canonical blocked hardware replay decisions for MCC runtime sidecars.
pub fn runtime_blocked_hardware_replay_decision_statuses() -> [HardwareReplayDecisionStatus; 2] {
    [
        HardwareReplayDecisionStatus {
            hardware: "AIGER",
            verdict: "unsafe",
            primitive: "unsafe_counterexample_trace",
            ay_backend_code: "ay_sat",
            replay_api: "transys_verify_witness",
            replay_status: "not_available",
            evidence_source: "runtime_sidecar_gate",
            generated_placeholder: false,
            typed_assignment_source: "ay_sat_witness".to_string(),
            replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus::Missing,
            typed_assignment_required_slots: 0,
            typed_assignment_present_slots: 0,
            typed_assignment_missing_slots: 0,
            accepted_replay_evidence_identity_sha256: "none".to_string(),
            accepted_trace_validity_obligations: 0,
            accepted_replay_obligation_identities_sha256: "none".to_string(),
            accepted_ay_proof_evidence_status: "none".to_string(),
            accepted_ay_proof_evidence_sha256: "none".to_string(),
            consumer_status: HardwareReplayPrimitiveConsumerStatus::Rejected,
            rejection_reason:
                HardwareReplayPrimitiveRejectionReason::MissingRealReplayArtifactEvidence,
        },
        HardwareReplayDecisionStatus {
            hardware: "BTOR2",
            verdict: "unsafe",
            primitive: "unsafe_counterexample_trace",
            ay_backend_code: "ay_chc",
            replay_api: "ay_chc_trace_validity_replay_obligations",
            replay_status: "not_available",
            evidence_source: "runtime_sidecar_gate",
            generated_placeholder: false,
            typed_assignment_source: "ay_chc_consumer_evidence".to_string(),
            replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus::Missing,
            typed_assignment_required_slots: 0,
            typed_assignment_present_slots: 0,
            typed_assignment_missing_slots: 0,
            accepted_replay_evidence_identity_sha256: "none".to_string(),
            accepted_trace_validity_obligations: 0,
            accepted_replay_obligation_identities_sha256: "none".to_string(),
            accepted_ay_proof_evidence_status: "none".to_string(),
            accepted_ay_proof_evidence_sha256: "none".to_string(),
            consumer_status: HardwareReplayPrimitiveConsumerStatus::Rejected,
            rejection_reason:
                HardwareReplayPrimitiveRejectionReason::ConcreteTraceAssignmentsUnavailable,
        },
    ]
}

/// Validate a hardware proof/replay boundary row using the shared schema,
/// independent of the hardware namespace prefix.
pub fn validate_hardware_proof_replay_boundary_evidence_row(
    row: &str,
) -> Result<(), HardwareProofReplayBoundaryEvidenceError> {
    let mut tokens = row.split_whitespace();
    let Some(hardware_namespace) = tokens.next() else {
        return Err(HardwareProofReplayBoundaryEvidenceError::WrongRowKind);
    };
    if !matches!(hardware_namespace, "AIGER" | "BTOR2") {
        return Err(
            HardwareProofReplayBoundaryEvidenceError::UnsupportedHardware(
                hardware_namespace.to_string(),
            ),
        );
    }
    if tokens.next() != Some(HARDWARE_PROOF_REPLAY_BOUNDARY_ROW_KIND) {
        return Err(HardwareProofReplayBoundaryEvidenceError::WrongRowKind);
    }

    for field in HARDWARE_PROOF_REPLAY_BOUNDARY_REQUIRED_FIELDS {
        required_boundary_field(row, field)?;
    }

    let schema = required_boundary_field(row, "schema")?;
    if schema != HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA {
        return Err(HardwareProofReplayBoundaryEvidenceError::UnsupportedSchema(
            schema.to_string(),
        ));
    }

    let schema_version = boundary_usize_field(row, "schema_version")?;
    if schema_version != HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA_VERSION as usize {
        return Err(HardwareProofReplayBoundaryEvidenceError::InvalidField {
            field: "schema_version",
            value: schema_version.to_string(),
        });
    }

    require_boundary_field_value(row, "local_production_gate", "no_local_production")?;
    require_boundary_field_value(row, "native_promotion_gate", "fail_closed")?;
    if boundary_bool_field(row, "production_selected")? {
        return Err(
            HardwareProofReplayBoundaryEvidenceError::InconsistentBoundary(
                "boundary_cannot_select_production",
            ),
        );
    }
    if !boundary_bool_field(row, "fail_closed")? {
        return Err(
            HardwareProofReplayBoundaryEvidenceError::InconsistentBoundary(
                "boundary_must_fail_closed",
            ),
        );
    }
    if required_boundary_field(row, "production_routing_status_code")?.is_empty() {
        return Err(HardwareProofReplayBoundaryEvidenceError::MissingField(
            "production_routing_status_code",
        ));
    }

    match hardware_namespace {
        "AIGER" => {
            require_boundary_field_value(row, "ay_backend_code", "ay_sat")?;
            require_boundary_field_value(row, "safe_proof", "aiger_safe_witness_validation")?;
            require_boundary_field_value(row, "safe_replay", "validate_safe")?;
            require_boundary_field_value(row, "unsafe_witness", "aiger_counterexample_trace")?;
            require_boundary_field_value(row, "unsafe_replay", "transys_verify_witness")?;
            require_boundary_field_value(row, "witness_attribution", "engine_trace")?;
        }
        "BTOR2" => {
            require_boundary_field_value(row, "ay_backend_code", "ay_chc")?;
            require_boundary_field_value(row, "safe_proof", "ay_chc_verified_result")?;
            require_boundary_field_value(row, "safe_replay", "not_available")?;
            require_boundary_field_value(row, "unsafe_witness", "ay_chc_counterexample")?;
            require_boundary_field_value(row, "unsafe_replay", "not_available")?;
            require_boundary_field_value(row, "witness_attribution", "query_clause")?;
        }
        _ => unreachable!("unsupported hardware namespaces returned above"),
    }

    Ok(())
}

/// Validate a hardware replay decision row using the shared schema, independent
/// of the hardware namespace prefix.
pub fn validate_hardware_replay_decision_evidence_row(
    row: &str,
) -> Result<(), HardwareReplayDecisionEvidenceError> {
    let mut tokens = row.split_whitespace();
    let Some(hardware_namespace) = tokens.next() else {
        return Err(HardwareReplayDecisionEvidenceError::WrongRowKind);
    };
    if hardware_namespace.is_empty() {
        return Err(HardwareReplayDecisionEvidenceError::WrongRowKind);
    }
    if tokens.next() != Some(HARDWARE_REPLAY_DECISION_ROW_KIND) {
        return Err(HardwareReplayDecisionEvidenceError::WrongRowKind);
    }

    for field in HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS {
        required_decision_field(row, field)?;
    }

    let schema = required_decision_field(row, "schema")?;
    if schema != HARDWARE_REPLAY_DECISION_SCHEMA {
        return Err(HardwareReplayDecisionEvidenceError::UnsupportedSchema(
            schema.to_string(),
        ));
    }

    let decision_status = match required_decision_field(row, "decision_status")? {
        "accepted" => HardwareReplayPrimitiveDecisionStatus::Accepted,
        "blocked" => HardwareReplayPrimitiveDecisionStatus::Blocked,
        value => {
            return Err(HardwareReplayDecisionEvidenceError::InvalidField {
                field: "decision_status",
                value: value.to_string(),
            });
        }
    };
    let consumer_status = match required_decision_field(row, "consumer_status")? {
        "accepted" => HardwareReplayPrimitiveConsumerStatus::Accepted,
        "rejected" => HardwareReplayPrimitiveConsumerStatus::Rejected,
        value => {
            return Err(HardwareReplayDecisionEvidenceError::InvalidField {
                field: "consumer_status",
                value: value.to_string(),
            });
        }
    };
    let replay_assignment_status = match required_decision_field(row, "replay_assignment_status")? {
        "complete" => HardwareReplayPrimitiveAssignmentStatus::Complete,
        "incomplete" => HardwareReplayPrimitiveAssignmentStatus::Incomplete,
        "missing" => HardwareReplayPrimitiveAssignmentStatus::Missing,
        value => {
            return Err(HardwareReplayDecisionEvidenceError::InvalidField {
                field: "replay_assignment_status",
                value: value.to_string(),
            });
        }
    };

    let accepted_replay_primitive = decision_bool_field(row, "accepted_replay_primitive")?;
    let blocked_by_typed_assignment_completeness =
        decision_bool_field(row, "blocked_by_typed_assignment_completeness")?;
    let blocked_by_placeholder = decision_bool_field(row, "blocked_by_placeholder")?;
    let generated_placeholder = decision_bool_field(row, "generated_placeholder")?;
    let reason_code = required_decision_field(row, "reason_code")?;
    let replay_api = required_decision_field(row, "replay_api")?;
    let required_slots = decision_usize_field(row, "typed_assignment_required_slots")?;
    let present_slots = decision_usize_field(row, "typed_assignment_present_slots")?;
    let missing_slots = decision_usize_field(row, "typed_assignment_missing_slots")?;

    let validates_ay_replay_identity = replay_api == "ay_chc_trace_validity_replay_obligations"
        || HARDWARE_REPLAY_DECISION_AY_REPLAY_FIELDS
            .iter()
            .any(|field| evidence_field(row, field).is_some());
    let validates_ay_proof_evidence = replay_api == "ay_chc_trace_validity_replay_obligations"
        || HARDWARE_REPLAY_DECISION_AY_PROOF_FIELDS
            .iter()
            .any(|field| evidence_field(row, field).is_some());
    let accepted_replay_identity = if validates_ay_replay_identity {
        Some((
            required_decision_field(row, "accepted_replay_evidence_identity_sha256")?,
            decision_usize_field(row, "accepted_trace_validity_obligations")?,
            required_decision_field(row, "accepted_replay_obligation_identities_sha256")?,
        ))
    } else {
        None
    };
    let accepted_ay_proof_evidence = if validates_ay_proof_evidence {
        Some((
            required_decision_field(row, "accepted_ay_proof_evidence_status")?,
            required_decision_field(row, "accepted_ay_proof_evidence_sha256")?,
        ))
    } else {
        None
    };

    let typed_assignment_block = matches!(
        reason_code,
        "concrete_trace_assignments_unavailable" | "typed_ay_trace_assignments_incomplete"
    );
    if blocked_by_typed_assignment_completeness != typed_assignment_block {
        return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "typed_assignment_block_flag_mismatch",
        ));
    }
    if blocked_by_placeholder
        != (generated_placeholder || reason_code == "generated_placeholder_evidence")
    {
        return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "placeholder_block_flag_mismatch",
        ));
    }

    match decision_status {
        HardwareReplayPrimitiveDecisionStatus::Accepted => {
            if consumer_status != HardwareReplayPrimitiveConsumerStatus::Accepted {
                return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                    "accepted_decision_requires_accepted_consumer",
                ));
            }
            if !accepted_replay_primitive || reason_code != NO_REASON_CODE {
                return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                    "accepted_decision_requires_no_reason",
                ));
            }
            if blocked_by_typed_assignment_completeness
                || blocked_by_placeholder
                || generated_placeholder
            {
                return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                    "accepted_decision_cannot_be_blocked",
                ));
            }
            if replay_assignment_status != HardwareReplayPrimitiveAssignmentStatus::Complete
                || missing_slots != 0
                || present_slots < required_slots
            {
                return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                    "accepted_decision_requires_complete_assignments",
                ));
            }
            if let Some((
                accepted_replay_evidence_identity_sha256,
                accepted_trace_validity_obligations,
                accepted_replay_obligation_identities_sha256,
            )) = accepted_replay_identity
            {
                if accepted_replay_evidence_identity_sha256 == "none"
                    || accepted_trace_validity_obligations == 0
                    || accepted_replay_obligation_identities_sha256 == "none"
                {
                    return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                        "accepted_decision_requires_replay_evidence_identity",
                    ));
                }
            }
            if let Some((accepted_ay_proof_evidence_status, accepted_ay_proof_evidence_sha256)) =
                accepted_ay_proof_evidence
            {
                if accepted_ay_proof_evidence_status != "ay_chc_verified_counterexample"
                    || accepted_ay_proof_evidence_sha256 == "none"
                {
                    return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                        "accepted_decision_requires_ay_proof_evidence",
                    ));
                }
            }
        }
        HardwareReplayPrimitiveDecisionStatus::Blocked => {
            if consumer_status != HardwareReplayPrimitiveConsumerStatus::Rejected {
                return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                    "blocked_decision_requires_rejected_consumer",
                ));
            }
            if accepted_replay_primitive || reason_code == NO_REASON_CODE {
                return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                    "blocked_decision_requires_reason",
                ));
            }
            if let Some((
                accepted_replay_evidence_identity_sha256,
                accepted_trace_validity_obligations,
                accepted_replay_obligation_identities_sha256,
            )) = accepted_replay_identity
            {
                if accepted_replay_evidence_identity_sha256 != "none"
                    || accepted_trace_validity_obligations != 0
                    || accepted_replay_obligation_identities_sha256 != "none"
                {
                    return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                        "blocked_decision_cannot_claim_replay_evidence_identity",
                    ));
                }
            }
            if let Some((accepted_ay_proof_evidence_status, accepted_ay_proof_evidence_sha256)) =
                accepted_ay_proof_evidence
            {
                if accepted_ay_proof_evidence_status != "none"
                    || accepted_ay_proof_evidence_sha256 != "none"
                {
                    return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                        "blocked_decision_cannot_claim_ay_proof_evidence",
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Return whether a shared hardware replay decision row admits replay.
pub fn hardware_replay_decision_accepts_replay_primitive(
    row: &str,
) -> Result<bool, HardwareReplayDecisionEvidenceError> {
    validate_hardware_replay_decision_evidence_row(row)?;
    decision_bool_field(row, "accepted_replay_primitive")
}

fn required_boundary_field<'a>(
    row: &'a str,
    key: &'static str,
) -> Result<&'a str, HardwareProofReplayBoundaryEvidenceError> {
    evidence_field(row, key).ok_or(HardwareProofReplayBoundaryEvidenceError::MissingField(key))
}

fn require_boundary_field_value(
    row: &str,
    key: &'static str,
    expected: &'static str,
) -> Result<(), HardwareProofReplayBoundaryEvidenceError> {
    let value = required_boundary_field(row, key)?;
    if value == expected {
        Ok(())
    } else {
        Err(HardwareProofReplayBoundaryEvidenceError::InvalidField {
            field: key,
            value: value.to_string(),
        })
    }
}

fn boundary_bool_field(
    row: &str,
    key: &'static str,
) -> Result<bool, HardwareProofReplayBoundaryEvidenceError> {
    match required_boundary_field(row, key)? {
        "true" => Ok(true),
        "false" => Ok(false),
        value => Err(HardwareProofReplayBoundaryEvidenceError::InvalidField {
            field: key,
            value: value.to_string(),
        }),
    }
}

fn boundary_usize_field(
    row: &str,
    key: &'static str,
) -> Result<usize, HardwareProofReplayBoundaryEvidenceError> {
    let value = required_boundary_field(row, key)?;
    value
        .parse()
        .map_err(|_| HardwareProofReplayBoundaryEvidenceError::InvalidField {
            field: key,
            value: value.to_string(),
        })
}

fn required_decision_field<'a>(
    row: &'a str,
    key: &'static str,
) -> Result<&'a str, HardwareReplayDecisionEvidenceError> {
    evidence_field(row, key).ok_or(HardwareReplayDecisionEvidenceError::MissingField(key))
}

fn decision_bool_field(
    row: &str,
    key: &'static str,
) -> Result<bool, HardwareReplayDecisionEvidenceError> {
    match required_decision_field(row, key)? {
        "true" => Ok(true),
        "false" => Ok(false),
        value => Err(HardwareReplayDecisionEvidenceError::InvalidField {
            field: key,
            value: value.to_string(),
        }),
    }
}

fn decision_usize_field(
    row: &str,
    key: &'static str,
) -> Result<usize, HardwareReplayDecisionEvidenceError> {
    let value = required_decision_field(row, key)?;
    value
        .parse()
        .map_err(|_| HardwareReplayDecisionEvidenceError::InvalidField {
            field: key,
            value: value.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_replay_schema_contract_is_stable() {
        assert_eq!(
            HARDWARE_PROOF_REPLAY_BOUNDARY_ROW_KIND,
            "proof_replay_boundary"
        );
        assert_eq!(
            HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA,
            "hardware_replay_primitive/v1"
        );
        assert_eq!(HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA_VERSION, 1);
        assert_eq!(
            HARDWARE_PROOF_REPLAY_BOUNDARY_REQUIRED_FIELDS,
            &[
                "schema",
                "schema_version",
                "ay_backend_code",
                "safe_proof",
                "safe_replay",
                "unsafe_witness",
                "unsafe_replay",
                "witness_attribution",
                "local_production_gate",
                "native_promotion_gate",
                "production_routing_status_code",
                "production_selected",
                "fail_closed",
            ]
        );
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

    #[test]
    fn hardware_proof_replay_boundaries_render_and_validate_runtime_rows() {
        let statuses = runtime_hardware_proof_replay_boundary_statuses("ay_first");
        let rows = statuses
            .iter()
            .map(HardwareProofReplayBoundaryStatus::render_evidence_row)
            .collect::<Vec<_>>();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].starts_with("AIGER proof_replay_boundary "));
        assert!(rows[0].contains("schema=hardware_replay_primitive/v1"));
        assert!(rows[0].contains("production_routing_status_code=ay_first"));
        assert!(rows[0].contains("production_selected=false"));
        assert!(rows[0].contains("fail_closed=true"));
        assert!(rows[1].starts_with("BTOR2 proof_replay_boundary "));
        assert!(rows[1].contains("ay_backend_code=ay_chc"));

        for row in rows {
            validate_hardware_proof_replay_boundary_evidence_row(&row).unwrap();
        }
    }

    #[test]
    fn hardware_proof_replay_boundary_validator_rejects_non_fail_closed_boundary() {
        let row = aiger_hardware_proof_replay_boundary_status("ay_first").render_evidence_row();
        let row = row.replace(
            "native_promotion_gate=fail_closed",
            "native_promotion_gate=open",
        );

        assert_eq!(
            validate_hardware_proof_replay_boundary_evidence_row(&row),
            Err(HardwareProofReplayBoundaryEvidenceError::InvalidField {
                field: "native_promotion_gate",
                value: "open".to_string()
            })
        );
    }

    #[test]
    fn hardware_replay_reason_codes_are_stable() {
        assert_eq!(
            HardwareReplayPrimitiveRejectionReason::None.reason_code(),
            "none"
        );
        assert_eq!(
            HardwareReplayPrimitiveRejectionReason::TypedAYTraceAssignmentsIncomplete.reason_code(),
            "typed_ay_trace_assignments_incomplete"
        );
        assert_eq!(
            HardwareReplayPrimitiveRejectionReason::ReplayEvidenceProblemHashMismatch.reason_code(),
            "replay_evidence_problem_hash_mismatch"
        );
    }

    #[test]
    fn runtime_blocked_hardware_replay_decisions_validate_and_fail_closed() {
        let statuses = runtime_blocked_hardware_replay_decision_statuses();
        let rows = statuses
            .iter()
            .map(HardwareReplayDecisionStatus::render_evidence_row)
            .collect::<Vec<_>>();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].starts_with("AIGER hardware_replay_decision "));
        assert!(rows[0].contains("decision_status=blocked"));
        assert!(rows[0].contains("accepted_replay_primitive=false"));
        assert!(rows[0].contains("reason_code=missing_real_replay_artifact_evidence"));
        assert!(!rows[0].contains("accepted_replay_evidence_identity_sha256="));
        assert!(rows[1].starts_with("BTOR2 hardware_replay_decision "));
        assert!(rows[1].contains("decision_status=blocked"));
        assert!(rows[1].contains("accepted_replay_primitive=false"));
        assert!(rows[1].contains("blocked_by_typed_assignment_completeness=true"));
        assert!(rows[1].contains("accepted_replay_evidence_identity_sha256=none"));
        assert!(rows[1].contains("accepted_trace_validity_obligations=0"));
        assert!(rows[1].contains("accepted_ay_proof_evidence_sha256=none"));

        for row in rows {
            validate_hardware_replay_decision_evidence_row(&row).unwrap();
            assert!(!hardware_replay_decision_accepts_replay_primitive(&row).unwrap());
        }
    }

    #[test]
    fn hardware_replay_decision_renders_aiger_row_without_identity_fields() {
        let status = HardwareReplayPrimitiveStatus {
            hardware: "AIGER",
            verdict: "unsafe",
            primitive: "unsafe_counterexample_trace",
            ay_backend_code: "ay_sat",
            replay_api: "transys_verify_witness",
            replay_status: "proven",
            evidence_source: "real_solver",
            generated_placeholder: false,
            typed_assignment_source: "ay_sat_witness".to_string(),
            replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus::Complete,
            typed_assignment_required_slots: 2,
            typed_assignment_present_slots: 2,
            typed_assignment_missing_slots: 0,
            consumer_status: HardwareReplayPrimitiveConsumerStatus::Accepted,
            rejection_reason: HardwareReplayPrimitiveRejectionReason::None,
        };
        let row = status.render_decision_evidence_row();
        assert!(row.starts_with("AIGER hardware_replay_decision "));
        assert!(!row.contains("accepted_replay_evidence_identity_sha256="));
        validate_hardware_replay_decision_evidence_row(&row).unwrap();
    }

    #[test]
    fn hardware_replay_decision_validator_accepts_btor2_identity_row() {
        let row = "BTOR2 hardware_replay_decision schema=hardware_replay_primitive/v1 verdict=unsafe primitive=unsafe_counterexample_trace decision_status=accepted accepted_replay_primitive=true blocked_by_typed_assignment_completeness=false blocked_by_placeholder=false consumer_status=accepted reason_code=none ay_backend_code=ay_chc replay_api=ay_chc_trace_validity_replay_obligations replay_status=proven typed_assignment_source=ay_chc_consumer_evidence replay_assignment_status=complete typed_assignment_required_slots=4 typed_assignment_present_slots=4 typed_assignment_missing_slots=0 accepted_replay_evidence_identity_sha256=0123456789abcdef accepted_trace_validity_obligations=1 accepted_replay_obligation_identities_sha256=fedcba9876543210 accepted_ay_proof_evidence_status=ay_chc_verified_counterexample accepted_ay_proof_evidence_sha256=abcdef0123456789 evidence_source=real_solver generated_placeholder=false";
        validate_hardware_replay_decision_evidence_row(row).unwrap();
        assert!(hardware_replay_decision_accepts_replay_primitive(row).unwrap());
    }

    #[test]
    fn hardware_replay_decision_validator_rejects_incomplete_accepted_assignments() {
        let row = "AIGER hardware_replay_decision schema=hardware_replay_primitive/v1 verdict=unsafe primitive=unsafe_counterexample_trace decision_status=accepted accepted_replay_primitive=true blocked_by_typed_assignment_completeness=false blocked_by_placeholder=false consumer_status=accepted reason_code=none ay_backend_code=ay_sat replay_api=transys_verify_witness replay_status=proven typed_assignment_source=ay_sat_witness replay_assignment_status=incomplete typed_assignment_required_slots=4 typed_assignment_present_slots=3 typed_assignment_missing_slots=1 evidence_source=real_solver generated_placeholder=false";
        assert_eq!(
            validate_hardware_replay_decision_evidence_row(row),
            Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
                "accepted_decision_requires_complete_assignments"
            ))
        );
    }
}
