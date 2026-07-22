// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral validation receipt records.
//!
//! Receipts are the core-side handoff object for validators that accept or
//! reject a prepared candidate artifact. They intentionally identify the
//! prepared program, candidate lane, validator, digest, and validation artifact
//! without depending on frontend payload types.

use std::fmt;

use crate::evidence_row::evidence_field;
use crate::fingerprint_identity::SharedDuplicateAuthorization;
use thiserror::Error;

/// Stable row kind for shared validation receipt evidence.
pub const VALIDATION_RECEIPT_ROW_KIND: &str = "validation_receipt";

/// Stable schema label for shared validation receipt evidence.
pub const VALIDATION_RECEIPT_SCHEMA: &str = "ty.shared.validation_receipt.v1";

/// Stable schema version for shared validation receipt evidence.
pub const VALIDATION_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Fields every validation receipt evidence row publishes.
pub const VALIDATION_RECEIPT_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "validator_kind",
    "digest_algorithm",
    "digest",
    "prepared_program_identity",
    "candidate_identity",
    "validation_artifact_kind",
    "validation_artifact_identity",
    "status",
    "failure_reason",
];

/// Validator family that produced a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationReceiptValidatorKind {
    /// Engine self-test ran before hot execution.
    Selftest,
    /// Replay of a recorded trace.
    TraceReplay,
    /// Replay of a recorded witness/counterexample.
    WitnessReplay,
    /// Check that the explored graph is complete.
    CompleteGraph,
    /// Check of an SCC (strongly-connected-component) certificate.
    SccCertificate,
    /// Check of an accepting-cycle (liveness) certificate.
    AcceptingCycleCertificate,
    /// Check of a structural proof.
    StructuralProof,
    /// Check of an AY solver proof.
    AYProof,
    /// Check of the result's output format.
    OutputFormat,
    /// Generic certificate validation.
    CertificateValidation,
    /// Replay of an emitted proof.
    ProofReplay,
}

impl ValidationReceiptValidatorKind {
    /// Stable lowercase wire code for this validator (for example `"ay_proof"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Selftest => "selftest",
            Self::TraceReplay => "trace_replay",
            Self::WitnessReplay => "witness_replay",
            Self::CompleteGraph => "complete_graph",
            Self::SccCertificate => "scc_certificate",
            Self::AcceptingCycleCertificate => "accepting_cycle_certificate",
            Self::StructuralProof => "structural_proof",
            Self::AYProof => "ay_proof",
            Self::OutputFormat => "output_format",
            Self::CertificateValidation => "certificate_validation",
            Self::ProofReplay => "proof_replay",
        }
    }

    fn from_code(code: &str) -> Option<Self> {
        match code {
            "selftest" => Some(Self::Selftest),
            "trace_replay" => Some(Self::TraceReplay),
            "witness_replay" => Some(Self::WitnessReplay),
            "complete_graph" => Some(Self::CompleteGraph),
            "scc_certificate" => Some(Self::SccCertificate),
            "accepting_cycle_certificate" => Some(Self::AcceptingCycleCertificate),
            "structural_proof" => Some(Self::StructuralProof),
            "ay_proof" => Some(Self::AYProof),
            "output_format" => Some(Self::OutputFormat),
            "certificate_validation" => Some(Self::CertificateValidation),
            "proof_replay" => Some(Self::ProofReplay),
            _ => None,
        }
    }
}

impl fmt::Display for ValidationReceiptValidatorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Class of artifact a validator checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationReceiptArtifactKind {
    /// A generic checker artifact.
    Artifact,
    /// A proof object.
    Proof,
    /// A witness / counterexample.
    Witness,
    /// A certificate.
    Certificate,
}

impl ValidationReceiptArtifactKind {
    /// Stable lowercase wire code for this artifact kind (for example `"proof"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Artifact => "artifact",
            Self::Proof => "proof",
            Self::Witness => "witness",
            Self::Certificate => "certificate",
        }
    }

    fn from_code(code: &str) -> Option<Self> {
        match code {
            "artifact" => Some(Self::Artifact),
            "proof" => Some(Self::Proof),
            "witness" => Some(Self::Witness),
            "certificate" => Some(Self::Certificate),
            _ => None,
        }
    }
}

impl fmt::Display for ValidationReceiptArtifactKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Typed validation result for a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationReceiptStatus {
    /// The validator accepted the artifact.
    Accepted,
    /// The validator rejected the artifact (a failure reason is required).
    Rejected,
}

impl ValidationReceiptStatus {
    /// Stable lowercase wire code for this status (`"accepted"` / `"rejected"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }

    /// Whether this status is [`ValidationReceiptStatus::Accepted`].
    pub fn is_accepted(self) -> bool {
        self == Self::Accepted
    }

    fn from_code(code: &str) -> Option<Self> {
        match code {
            "accepted" => Some(Self::Accepted),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

impl fmt::Display for ValidationReceiptStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Frontend-neutral receipt proving that a validator accepted or rejected a
/// prepared candidate artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidationReceipt {
    /// Validator family that produced this receipt.
    pub validator_kind: ValidationReceiptValidatorKind,
    /// Digest algorithm naming how `digest` was computed (for example `sha256`).
    pub digest_algorithm: String,
    /// Digest of the validated artifact.
    pub digest: String,
    /// Identity of the prepared program the artifact belongs to.
    pub prepared_program_identity: String,
    /// Identity of the candidate lane that produced the artifact.
    pub candidate_identity: String,
    /// Class of artifact that was checked.
    pub validation_artifact_kind: ValidationReceiptArtifactKind,
    /// Identity of the specific validation artifact.
    pub validation_artifact_identity: String,
    /// Accept/reject outcome.
    pub status: ValidationReceiptStatus,
    /// Failure reason; required when [`status`](Self::status) is rejected and
    /// must be absent when accepted.
    pub failure_reason: Option<String>,
}

impl ValidationReceipt {
    /// Build an accepted receipt (no failure reason).
    #[allow(clippy::too_many_arguments)]
    pub fn accepted(
        validator_kind: ValidationReceiptValidatorKind,
        digest_algorithm: impl Into<String>,
        digest: impl Into<String>,
        prepared_program_identity: impl Into<String>,
        candidate_identity: impl Into<String>,
        validation_artifact_kind: ValidationReceiptArtifactKind,
        validation_artifact_identity: impl Into<String>,
    ) -> Self {
        Self {
            validator_kind,
            digest_algorithm: digest_algorithm.into(),
            digest: digest.into(),
            prepared_program_identity: prepared_program_identity.into(),
            candidate_identity: candidate_identity.into(),
            validation_artifact_kind,
            validation_artifact_identity: validation_artifact_identity.into(),
            status: ValidationReceiptStatus::Accepted,
            failure_reason: None,
        }
    }

    /// Build a rejected receipt carrying a failure reason.
    ///
    /// An empty `failure_reason` is normalized to `None`, which will cause
    /// [`validate`](Self::validate) to fail with
    /// [`RejectedReceiptMissingFailureReason`](ValidationReceiptValidationError::RejectedReceiptMissingFailureReason).
    #[allow(clippy::too_many_arguments)]
    pub fn rejected(
        validator_kind: ValidationReceiptValidatorKind,
        digest_algorithm: impl Into<String>,
        digest: impl Into<String>,
        prepared_program_identity: impl Into<String>,
        candidate_identity: impl Into<String>,
        validation_artifact_kind: ValidationReceiptArtifactKind,
        validation_artifact_identity: impl Into<String>,
        failure_reason: impl Into<String>,
    ) -> Self {
        Self {
            validator_kind,
            digest_algorithm: digest_algorithm.into(),
            digest: digest.into(),
            prepared_program_identity: prepared_program_identity.into(),
            candidate_identity: candidate_identity.into(),
            validation_artifact_kind,
            validation_artifact_identity: validation_artifact_identity.into(),
            status: ValidationReceiptStatus::Rejected,
            failure_reason: non_empty_string(failure_reason.into()),
        }
    }

    /// Attach a failure reason (empty string clears it to `None`).
    pub fn with_failure_reason(mut self, failure_reason: impl Into<String>) -> Self {
        self.failure_reason = non_empty_string(failure_reason.into());
        self
    }

    /// Whether this receipt satisfies the frontend-neutral contract.
    ///
    /// Convenience boolean wrapper around [`validate`](Self::validate).
    pub fn validates(&self) -> bool {
        validate_validation_receipt(self).is_ok()
    }

    /// Validate this receipt against the frontend-neutral contract.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationReceiptValidationError::MissingField`] if any required
    /// identity (`digest_algorithm`, `digest`, `prepared_program_identity`,
    /// `candidate_identity`, `validation_artifact_identity`) is blank or `none`;
    /// [`AcceptedReceiptHasFailureReason`](ValidationReceiptValidationError::AcceptedReceiptHasFailureReason)
    /// if an accepted receipt carries a failure reason; and
    /// [`RejectedReceiptMissingFailureReason`](ValidationReceiptValidationError::RejectedReceiptMissingFailureReason)
    /// if a rejected receipt lacks one.
    pub fn validate(&self) -> Result<(), ValidationReceiptValidationError> {
        validate_validation_receipt(self)
    }

    /// Duplicate-fingerprint authorization supplied by replay/proof receipts.
    ///
    /// Only accepted, internally valid proof, witness, and certificate receipts
    /// satisfy [`SharedCollisionPolicy::ProofWitnessRequired`]. Generic
    /// artifact receipts remain unconfirmed because they do not by themselves
    /// authorize suppressing a duplicate runtime value.
    pub fn proof_witness_duplicate_authorization(&self) -> SharedDuplicateAuthorization {
        let is_proof_witness_artifact = matches!(
            self.validation_artifact_kind,
            ValidationReceiptArtifactKind::Proof
                | ValidationReceiptArtifactKind::Witness
                | ValidationReceiptArtifactKind::Certificate
        );
        SharedDuplicateAuthorization::proof_witness(
            is_proof_witness_artifact
                && self.status.is_accepted()
                && validate_validation_receipt(self).is_ok(),
        )
    }

    /// Renders a stable evidence row for receipt consumers.
    pub fn render_evidence_row(&self, scope: &str) -> String {
        format!(
            "{} {} schema={} schema_version={} validator_kind={} digest_algorithm={} digest={} prepared_program_identity={} candidate_identity={} validation_artifact_kind={} validation_artifact_identity={} status={} failure_reason={}",
            scope,
            VALIDATION_RECEIPT_ROW_KIND,
            VALIDATION_RECEIPT_SCHEMA,
            VALIDATION_RECEIPT_SCHEMA_VERSION,
            self.validator_kind.code(),
            evidence_value(&self.digest_algorithm),
            evidence_value(&self.digest),
            evidence_value(&self.prepared_program_identity),
            evidence_value(&self.candidate_identity),
            self.validation_artifact_kind.code(),
            evidence_value(&self.validation_artifact_identity),
            self.status.code(),
            evidence_optional(self.failure_reason.as_deref())
        )
    }
}

/// Validation failures for malformed or internally inconsistent receipts.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationReceiptValidationError {
    /// The evidence row did not start with the validation-receipt row kind.
    #[error("wrong validation receipt row kind")]
    WrongRowKind,
    /// A required field was missing or blank (the field name is carried).
    #[error("missing validation receipt field: {0}")]
    MissingField(&'static str),
    /// The row's schema label did not match [`VALIDATION_RECEIPT_SCHEMA`].
    #[error("unsupported validation receipt schema: {0}")]
    UnsupportedSchema(String),
    /// A field held a value outside its allowed domain.
    #[error("invalid validation receipt field {field}={value}")]
    InvalidField {
        /// Name of the offending field.
        field: &'static str,
        /// Offending value as found in the row.
        value: String,
    },
    /// An accepted receipt carried a failure reason, which is not allowed.
    #[error("accepted validation receipt cannot carry failure reason")]
    AcceptedReceiptHasFailureReason,
    /// A rejected receipt lacked the required failure reason.
    #[error("rejected validation receipt requires failure reason")]
    RejectedReceiptMissingFailureReason,
}

/// Validates the frontend-neutral receipt contract.
///
/// # Errors
///
/// Returns [`ValidationReceiptValidationError::MissingField`] for any blank or
/// `none` required identity, and the status/failure-reason consistency errors
/// [`AcceptedReceiptHasFailureReason`](ValidationReceiptValidationError::AcceptedReceiptHasFailureReason)
/// or [`RejectedReceiptMissingFailureReason`](ValidationReceiptValidationError::RejectedReceiptMissingFailureReason).
pub fn validate_validation_receipt(
    receipt: &ValidationReceipt,
) -> Result<(), ValidationReceiptValidationError> {
    require_identity("digest_algorithm", &receipt.digest_algorithm)?;
    require_identity("digest", &receipt.digest)?;
    require_identity(
        "prepared_program_identity",
        &receipt.prepared_program_identity,
    )?;
    require_identity("candidate_identity", &receipt.candidate_identity)?;
    require_identity(
        "validation_artifact_identity",
        &receipt.validation_artifact_identity,
    )?;

    match receipt.status {
        ValidationReceiptStatus::Accepted => {
            if has_failure_reason(receipt.failure_reason.as_deref()) {
                return Err(ValidationReceiptValidationError::AcceptedReceiptHasFailureReason);
            }
        }
        ValidationReceiptStatus::Rejected => {
            if !has_failure_reason(receipt.failure_reason.as_deref()) {
                return Err(ValidationReceiptValidationError::RejectedReceiptMissingFailureReason);
            }
        }
    }

    Ok(())
}

/// Validate one rendered frontend-neutral validation receipt evidence row.
///
/// # Errors
///
/// Returns [`ValidationReceiptValidationError::WrongRowKind`] if the row does
/// not begin with the receipt row kind,
/// [`MissingField`](ValidationReceiptValidationError::MissingField) for any
/// absent required field,
/// [`UnsupportedSchema`](ValidationReceiptValidationError::UnsupportedSchema)
/// for a mismatched schema label,
/// [`InvalidField`](ValidationReceiptValidationError::InvalidField) for an
/// out-of-domain `schema_version`, `validator_kind`, `validation_artifact_kind`,
/// or `status`, and the status/failure-reason consistency errors.
pub fn validate_validation_receipt_evidence_row(
    row: &str,
) -> Result<(), ValidationReceiptValidationError> {
    let mut tokens = row.split_whitespace();
    if tokens.next().is_none() {
        return Err(ValidationReceiptValidationError::WrongRowKind);
    }
    if tokens.next() != Some(VALIDATION_RECEIPT_ROW_KIND) {
        return Err(ValidationReceiptValidationError::WrongRowKind);
    }

    for field in VALIDATION_RECEIPT_REQUIRED_FIELDS {
        required_row_field(row, field)?;
    }

    let schema = required_row_field(row, "schema")?;
    if schema != VALIDATION_RECEIPT_SCHEMA {
        return Err(ValidationReceiptValidationError::UnsupportedSchema(
            schema.to_string(),
        ));
    }

    let schema_version = required_row_field(row, "schema_version")?;
    if schema_version != VALIDATION_RECEIPT_SCHEMA_VERSION.to_string() {
        return Err(ValidationReceiptValidationError::InvalidField {
            field: "schema_version",
            value: schema_version.to_string(),
        });
    }

    let validator_kind = required_row_field(row, "validator_kind")?;
    if ValidationReceiptValidatorKind::from_code(validator_kind).is_none() {
        return Err(ValidationReceiptValidationError::InvalidField {
            field: "validator_kind",
            value: validator_kind.to_string(),
        });
    }

    let validation_artifact_kind = required_row_field(row, "validation_artifact_kind")?;
    if ValidationReceiptArtifactKind::from_code(validation_artifact_kind).is_none() {
        return Err(ValidationReceiptValidationError::InvalidField {
            field: "validation_artifact_kind",
            value: validation_artifact_kind.to_string(),
        });
    }

    for field in [
        "digest_algorithm",
        "digest",
        "prepared_program_identity",
        "candidate_identity",
        "validation_artifact_identity",
    ] {
        require_row_identity(row, field)?;
    }

    let status_code = required_row_field(row, "status")?;
    let status = ValidationReceiptStatus::from_code(status_code).ok_or_else(|| {
        ValidationReceiptValidationError::InvalidField {
            field: "status",
            value: status_code.to_string(),
        }
    })?;
    let failure_reason = required_row_field(row, "failure_reason")?;
    match status {
        ValidationReceiptStatus::Accepted => {
            if has_failure_reason(Some(failure_reason)) {
                return Err(ValidationReceiptValidationError::AcceptedReceiptHasFailureReason);
            }
        }
        ValidationReceiptStatus::Rejected => {
            if !has_failure_reason(Some(failure_reason)) {
                return Err(ValidationReceiptValidationError::RejectedReceiptMissingFailureReason);
            }
        }
    }

    Ok(())
}

fn require_identity(
    field: &'static str,
    value: &str,
) -> Result<(), ValidationReceiptValidationError> {
    if value.trim().is_empty() || value.trim() == "none" {
        Err(ValidationReceiptValidationError::MissingField(field))
    } else {
        Ok(())
    }
}

fn has_failure_reason(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|value| !value.is_empty() && value != "none")
}

fn required_row_field<'a>(
    row: &'a str,
    key: &'static str,
) -> Result<&'a str, ValidationReceiptValidationError> {
    evidence_field(row, key).ok_or(ValidationReceiptValidationError::MissingField(key))
}

fn require_row_identity(
    row: &str,
    field: &'static str,
) -> Result<(), ValidationReceiptValidationError> {
    require_identity(field, required_row_field(row, field)?)
}

fn non_empty_string(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn evidence_value(value: &str) -> String {
    if value.trim().is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint_identity::SharedCollisionPolicy;

    #[test]
    fn validation_receipt_accepts_proof_identity() {
        let receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "0123456789abcdef",
            "prepared hardware program",
            "ay chc candidate",
            ValidationReceiptArtifactKind::Proof,
            "ay proof certificate",
        );

        assert!(receipt.validate().is_ok());
        assert!(receipt.validates());

        let row = receipt.render_evidence_row("CORE");
        assert!(row.starts_with("CORE validation_receipt "));
        assert!(row.contains("schema=ty.shared.validation_receipt.v1"));
        assert!(row.contains("validator_kind=ay_proof"));
        assert!(row.contains("digest_algorithm=sha256"));
        assert!(row.contains("digest=0123456789abcdef"));
        assert!(row.contains("prepared_program_identity=prepared_hardware_program"));
        assert!(row.contains("candidate_identity=ay_chc_candidate"));
        assert!(row.contains("validation_artifact_kind=proof"));
        assert!(row.contains("validation_artifact_identity=ay_proof_certificate"));
        assert!(row.contains("status=accepted"));
        assert!(row.contains("failure_reason=none"));
        validate_validation_receipt_evidence_row(&row).unwrap();
    }

    #[test]
    fn validation_receipt_records_rejected_witness_reason() {
        let receipt = ValidationReceipt::rejected(
            ValidationReceiptValidatorKind::WitnessReplay,
            "sha256",
            "fedcba9876543210",
            "prepared replay program",
            "native candidate",
            ValidationReceiptArtifactKind::Witness,
            "unsafe trace witness",
            "trace step 7 missing assignment",
        );

        assert_eq!(receipt.status, ValidationReceiptStatus::Rejected);
        assert!(receipt.validate().is_ok());

        let row = receipt.render_evidence_row("CORE");
        assert!(row.contains("validator_kind=witness_replay"));
        assert!(row.contains("validation_artifact_kind=witness"));
        assert!(row.contains("status=rejected"));
        assert!(row.contains("failure_reason=trace_step_7_missing_assignment"));
        validate_validation_receipt_evidence_row(&row).unwrap();
    }

    #[test]
    fn validation_receipt_authorizes_proof_witness_duplicate_policy() {
        let accepted_proof = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "abc",
            "prepared tla program",
            "analytical",
            ValidationReceiptArtifactKind::Proof,
            "ay analytical proof",
        );
        assert_eq!(
            accepted_proof.proof_witness_duplicate_authorization(),
            SharedDuplicateAuthorization::ProofWitness
        );
        assert!(SharedCollisionPolicy::ProofWitnessRequired
            .authorizes_duplicate(accepted_proof.proof_witness_duplicate_authorization()));

        let rejected_witness = ValidationReceipt::rejected(
            ValidationReceiptValidatorKind::WitnessReplay,
            "sha256",
            "bad",
            "prepared replay program",
            "native replay candidate",
            ValidationReceiptArtifactKind::Witness,
            "native witness",
            "trace step 7 missing assignment",
        );
        assert_eq!(
            rejected_witness.proof_witness_duplicate_authorization(),
            SharedDuplicateAuthorization::Unconfirmed
        );

        let accepted_artifact = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::Selftest,
            "sha256",
            "artifact",
            "prepared program",
            "candidate",
            ValidationReceiptArtifactKind::Artifact,
            "generic artifact",
        );
        assert_eq!(
            accepted_artifact.proof_witness_duplicate_authorization(),
            SharedDuplicateAuthorization::Unconfirmed
        );
    }

    #[test]
    fn validation_receipt_rejects_missing_required_identities() {
        let receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::Selftest,
            "sha256",
            "",
            "prepared",
            "candidate",
            ValidationReceiptArtifactKind::Artifact,
            "selftest report",
        );

        assert_eq!(
            receipt.validate(),
            Err(ValidationReceiptValidationError::MissingField("digest"))
        );
    }

    #[test]
    fn validation_receipt_rejects_inconsistent_status_reason_pairs() {
        let accepted_with_reason = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::OutputFormat,
            "sha256",
            "abc123",
            "prepared",
            "candidate",
            ValidationReceiptArtifactKind::Certificate,
            "format certificate",
        )
        .with_failure_reason("format mismatch");
        assert_eq!(
            accepted_with_reason.validate(),
            Err(ValidationReceiptValidationError::AcceptedReceiptHasFailureReason)
        );

        let rejected_without_reason = ValidationReceipt {
            status: ValidationReceiptStatus::Rejected,
            failure_reason: None,
            ..ValidationReceipt::accepted(
                ValidationReceiptValidatorKind::CertificateValidation,
                "sha256",
                "abc123",
                "prepared",
                "candidate",
                ValidationReceiptArtifactKind::Certificate,
                "certificate",
            )
        };
        assert_eq!(
            rejected_without_reason.validate(),
            Err(ValidationReceiptValidationError::RejectedReceiptMissingFailureReason)
        );
    }

    #[test]
    fn validation_receipt_row_validation_rejects_malformed_rows() {
        let accepted = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "abc123",
            "prepared",
            "candidate",
            ValidationReceiptArtifactKind::Proof,
            "proof",
        )
        .render_evidence_row("CORE");
        let accepted_with_reason = accepted.replace(
            "failure_reason=none",
            "failure_reason=unexpected_failure_reason",
        );
        assert_eq!(
            validate_validation_receipt_evidence_row(&accepted_with_reason),
            Err(ValidationReceiptValidationError::AcceptedReceiptHasFailureReason)
        );

        let rejected = ValidationReceipt::rejected(
            ValidationReceiptValidatorKind::WitnessReplay,
            "sha256",
            "def456",
            "prepared",
            "candidate",
            ValidationReceiptArtifactKind::Witness,
            "witness",
            "trace failed",
        )
        .render_evidence_row("CORE");
        let rejected_without_reason =
            rejected.replace("failure_reason=trace_failed", "failure_reason=none");
        assert_eq!(
            validate_validation_receipt_evidence_row(&rejected_without_reason),
            Err(ValidationReceiptValidationError::RejectedReceiptMissingFailureReason)
        );

        let unknown_status = accepted.replace("status=accepted", "status=maybe");
        assert_eq!(
            validate_validation_receipt_evidence_row(&unknown_status),
            Err(ValidationReceiptValidationError::InvalidField {
                field: "status",
                value: "maybe".to_string()
            })
        );

        let missing_candidate = accepted.replace("candidate_identity=candidate", "");
        assert_eq!(
            validate_validation_receipt_evidence_row(&missing_candidate),
            Err(ValidationReceiptValidationError::MissingField(
                "candidate_identity"
            ))
        );
    }

    #[test]
    fn validation_receipt_required_fields_include_shared_identities() {
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"validator_kind"));
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"digest"));
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"prepared_program_identity"));
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"candidate_identity"));
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"validation_artifact_identity"));
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"status"));
        assert!(VALIDATION_RECEIPT_REQUIRED_FIELDS.contains(&"failure_reason"));
    }
}
