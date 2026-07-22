// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral prepared successor batch contracts.
//!
//! This module describes the runtime shape shared by explicit-state, native,
//! analytical, hardware, and replay frontends once a checker can produce
//! successor candidates that are ready for fingerprint admission. It carries
//! descriptor and evidence metadata only; concrete queues, arenas, and solver
//! handles remain frontend-owned.

use std::{fmt, hash::Hash};

use crate::{
    evidence_row::evidence_field,
    prepared_fingerprint_admission::{
        PreparedFingerprintAdmissionPlan, PreparedFingerprintPayloadWitnessKind,
    },
    prepared_program::{PreparedProgramPayloadKind, PreparedStorageKind},
    setup_trace::{CheckerArtifactIdentityFields, CheckerSourceKind, SetupTraceLaneKind},
    shared_engine_adoption::SharedEngineFrontendFamily,
};

/// Stable row kind for prepared successor batch evidence.
pub const PREPARED_SUCCESSOR_BATCH_ROW_KIND: &str = "prepared_successor_batch";

/// Stable schema label for prepared successor batch evidence.
pub const PREPARED_SUCCESSOR_BATCH_SCHEMA: &str = "ty.prepared_successor_batch.v1";

/// Stable schema version for prepared successor batch evidence.
pub const PREPARED_SUCCESSOR_BATCH_SCHEMA_VERSION: u32 = 1;

/// Fields every prepared successor batch evidence row publishes.
pub const PREPARED_SUCCESSOR_BATCH_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "source_kind",
    "frontend_family",
    "shared_engine_component",
    "descriptor_id",
    "payload_kind",
    "storage_kind",
    "lane_kind",
    "successor_payload_kind",
    "admission_kind",
    "admission_ready",
    "candidate_key",
    "prepared_program_identity",
    "prepared_lane_identity",
    "batch_artifact_identity",
    "fingerprint_admission_plan",
    "payload_witness",
    "storage_policy_identity",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "canonical_payload_available",
    "full_state_payload_available",
    "fingerprint_available",
    "sparse_delta_available",
    "parent_index_available",
    "raw_successor_metadata_complete",
    "compatible_frontend_families",
    "default_consumers",
    "remaining_compatible_frontend_families",
    "validation_status",
    "reason_code",
];

const PREPARED_SUCCESSOR_REJECTION_EMPTY_ID: &str = "empty_prepared_successor_batch_id";
const PREPARED_SUCCESSOR_REJECTION_UNKNOWN_SOURCE_KIND: &str = "unknown_source_kind";
const PREPARED_SUCCESSOR_REJECTION_SOURCE_PAYLOAD_MISMATCH: &str = "source_payload_mismatch";
const PREPARED_SUCCESSOR_REJECTION_UNKNOWN_STORAGE_KIND: &str = "unknown_storage_kind";
const PREPARED_SUCCESSOR_REJECTION_UNKNOWN_LANE_KIND: &str = "unknown_lane_kind";
const PREPARED_SUCCESSOR_REJECTION_MISSING_SOURCE_FAMILY: &str =
    "source_family_missing_from_compatible_frontends";
const PREPARED_SUCCESSOR_REJECTION_DEFAULT_CONSUMER_NOT_COMPATIBLE: &str =
    "default_consumer_not_compatible";
const PREPARED_SUCCESSOR_REJECTION_ADMISSION_SOURCE_MISMATCH: &str = "admission_source_mismatch";
const PREPARED_SUCCESSOR_REJECTION_ADMISSION_PAYLOAD_MISMATCH: &str = "admission_payload_mismatch";
const PREPARED_SUCCESSOR_REJECTION_ADMISSION_STORAGE_MISMATCH: &str = "admission_storage_mismatch";
const PREPARED_SUCCESSOR_REJECTION_ADMISSION_LANE_MISMATCH: &str = "admission_lane_mismatch";

/// Payload carried by a prepared successor item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedSuccessorPayloadKind {
    /// A full state plus canonical bytes/fingerprint are available.
    FullCanonicalState,
    /// Canonical bytes/fingerprint are available; materialization is deferred.
    CanonicalPayloadOnly,
    /// Only a fingerprint is available at the producer boundary.
    FingerprintOnly,
    /// Successor is represented as a sparse delta from the parent.
    SparseDelta,
    /// Successor is represented by a proof, replay, or certificate step.
    ProofOrReplayStep,
}

impl PreparedSuccessorPayloadKind {
    /// Stable evidence code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::FullCanonicalState => "full_canonical_state",
            Self::CanonicalPayloadOnly => "canonical_payload_only",
            Self::FingerprintOnly => "fingerprint_only",
            Self::SparseDelta => "sparse_delta",
            Self::ProofOrReplayStep => "proof_or_replay_step",
        }
    }

    /// Parse a stable evidence code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "full_canonical_state" => Some(Self::FullCanonicalState),
            "canonical_payload_only" => Some(Self::CanonicalPayloadOnly),
            "fingerprint_only" => Some(Self::FingerprintOnly),
            "sparse_delta" => Some(Self::SparseDelta),
            "proof_or_replay_step" => Some(Self::ProofOrReplayStep),
            _ => None,
        }
    }

    fn exposes_full_state(self) -> bool {
        matches!(self, Self::FullCanonicalState)
    }

    fn exposes_canonical_payload(self) -> bool {
        matches!(self, Self::FullCanonicalState | Self::CanonicalPayloadOnly)
    }

    fn exposes_fingerprint(self) -> bool {
        !matches!(self, Self::ProofOrReplayStep)
    }

    fn exposes_sparse_delta(self) -> bool {
        matches!(self, Self::SparseDelta)
    }
}

impl fmt::Display for PreparedSuccessorPayloadKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Admission path attached to a prepared successor batch descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedSuccessorAdmissionKind {
    /// No setup-validated admission plan is attached.
    None,
    /// A setup-validated prepared fingerprint admission plan is attached.
    PreparedFingerprint,
}

impl PreparedSuccessorAdmissionKind {
    /// Stable evidence code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PreparedFingerprint => "prepared_fingerprint",
        }
    }

    /// Parse a stable evidence code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "none" => Some(Self::None),
            "prepared_fingerprint" => Some(Self::PreparedFingerprint),
            _ => None,
        }
    }
}

impl fmt::Display for PreparedSuccessorAdmissionKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Validation error for a prepared successor batch descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSuccessorBatchDescriptorError {
    /// Stable machine-readable reason code.
    pub reason_code: &'static str,
    /// Human-readable detail.
    pub detail: String,
}

impl PreparedSuccessorBatchDescriptorError {
    fn new(reason_code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            reason_code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for PreparedSuccessorBatchDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.reason_code, self.detail)
    }
}

impl std::error::Error for PreparedSuccessorBatchDescriptorError {}

/// Frontend-neutral descriptor for a prepared successor batch producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSuccessorBatchDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Prepared source family that produced the batch.
    pub source_kind: CheckerSourceKind,
    /// Prepared payload family.
    pub payload_kind: PreparedProgramPayloadKind,
    /// Storage ABI represented by the successor payload.
    pub storage_kind: PreparedStorageKind,
    /// Runtime lane that consumes the successor batch.
    pub lane: SetupTraceLaneKind,
    /// Payload shape emitted by this producer.
    pub successor_payload_kind: PreparedSuccessorPayloadKind,
    /// Optional prepared candidate key.
    pub candidate_key: Option<String>,
    /// Optional prepared program identity.
    pub prepared_program_identity: Option<String>,
    /// Optional prepared candidate lane identity.
    pub prepared_lane_identity: Option<String>,
    /// Optional batch artifact identity.
    pub batch_artifact_identity: Option<String>,
    /// Optional setup-validated fingerprint admission plan.
    pub admission_plan: Option<PreparedFingerprintAdmissionPlan>,
    /// Shared identity fields for setup traces and evidence.
    pub identities: CheckerArtifactIdentityFields,
    /// Frontend families that can use the same prepared successor contract.
    pub compatible_frontend_families: Vec<SharedEngineFrontendFamily>,
    /// Families using this descriptor as a default consumer.
    pub default_consumers: Vec<SharedEngineFrontendFamily>,
    /// Whether producer metadata is complete enough for runtime admission rows.
    pub raw_successor_metadata_complete: bool,
    /// Whether parent indices are included in runtime successor items.
    pub parent_index_available: bool,
}

impl PreparedSuccessorBatchDescriptor {
    /// Build a prepared successor batch descriptor from explicit metadata.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        source_kind: CheckerSourceKind,
        payload_kind: PreparedProgramPayloadKind,
        storage_kind: PreparedStorageKind,
        lane: SetupTraceLaneKind,
        successor_payload_kind: PreparedSuccessorPayloadKind,
    ) -> Self {
        let compatible_frontend_families = source_kind
            .adoption_frontend_family()
            .into_iter()
            .collect::<Vec<_>>();
        let mut default_consumers = Vec::new();
        if let Some(source_family) = source_kind.adoption_frontend_family() {
            default_consumers.push(source_family);
        }
        Self {
            id: id.into(),
            source_kind,
            payload_kind,
            storage_kind,
            lane,
            successor_payload_kind,
            candidate_key: None,
            prepared_program_identity: None,
            prepared_lane_identity: None,
            batch_artifact_identity: None,
            admission_plan: None,
            identities: CheckerArtifactIdentityFields::default(),
            compatible_frontend_families,
            default_consumers,
            raw_successor_metadata_complete: true,
            parent_index_available: false,
        }
    }

    /// Attach a prepared candidate key.
    #[must_use]
    pub fn with_candidate_key(mut self, candidate_key: impl Into<String>) -> Self {
        self.candidate_key = non_empty_string(candidate_key.into());
        self
    }

    /// Attach a prepared program identity.
    #[must_use]
    pub fn with_prepared_program_identity(mut self, identity: impl Into<String>) -> Self {
        self.prepared_program_identity = non_empty_string(identity.into());
        self
    }

    /// Attach a prepared lane identity.
    #[must_use]
    pub fn with_prepared_lane_identity(mut self, identity: impl Into<String>) -> Self {
        self.prepared_lane_identity = non_empty_string(identity.into());
        self
    }

    /// Attach a batch artifact identity.
    #[must_use]
    pub fn with_batch_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        let identity = identity.into();
        self.identities = self
            .identities
            .with_batch_artifact_identity(identity.clone());
        self.batch_artifact_identity = non_empty_string(identity);
        self
    }

    /// Attach identity fields, preserving existing explicit fields.
    #[must_use]
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = self.identities.merged_with_fallback(&identities);
        self
    }

    /// Attach a setup-validated fingerprint admission plan.
    #[must_use]
    pub fn with_admission_plan(mut self, plan: PreparedFingerprintAdmissionPlan) -> Self {
        self.identities = self.identities.merged_with_fallback(&plan.identities);
        self.admission_plan = Some(plan);
        self
    }

    /// Override frontend family coverage and default consumers.
    #[must_use]
    pub fn with_frontend_families(
        mut self,
        compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        default_consumers: impl IntoIterator<Item = SharedEngineFrontendFamily>,
    ) -> Self {
        self.compatible_frontend_families =
            dedup_frontend_families(compatible_frontend_families.into_iter());
        self.default_consumers = dedup_frontend_families(default_consumers.into_iter());
        self
    }

    /// Mark whether parent indices are carried by runtime successor items.
    #[must_use]
    pub fn with_parent_index_available(mut self, available: bool) -> Self {
        self.parent_index_available = available;
        self
    }

    /// Mark whether raw successor metadata is complete.
    #[must_use]
    pub fn with_raw_successor_metadata_complete(mut self, complete: bool) -> Self {
        self.raw_successor_metadata_complete = complete;
        self
    }

    /// Admission kind for this descriptor.
    #[must_use]
    pub fn admission_kind(&self) -> PreparedSuccessorAdmissionKind {
        if self.admission_plan.is_some() {
            PreparedSuccessorAdmissionKind::PreparedFingerprint
        } else {
            PreparedSuccessorAdmissionKind::None
        }
    }

    /// Whether the descriptor is ready for runtime admission.
    #[must_use]
    pub fn admission_ready(&self) -> bool {
        self.admission_plan.is_some()
            && self.successor_payload_kind.exposes_fingerprint()
            && self.successor_payload_kind.exposes_canonical_payload()
    }

    /// Remaining compatible families that are not default consumers.
    #[must_use]
    pub fn remaining_compatible_frontend_families(&self) -> Vec<SharedEngineFrontendFamily> {
        self.compatible_frontend_families
            .iter()
            .copied()
            .filter(|family| !self.default_consumers.contains(family))
            .collect()
    }

    /// Validate the shared prepared successor batch contract.
    pub fn validate(&self) -> Result<(), PreparedSuccessorBatchDescriptorError> {
        if self.id.trim().is_empty() {
            return Err(PreparedSuccessorBatchDescriptorError::new(
                PREPARED_SUCCESSOR_REJECTION_EMPTY_ID,
                "prepared successor batch descriptor id must not be empty",
            ));
        }
        let Some(source_family) = self.source_kind.adoption_frontend_family() else {
            return Err(PreparedSuccessorBatchDescriptorError::new(
                PREPARED_SUCCESSOR_REJECTION_UNKNOWN_SOURCE_KIND,
                "prepared successor batch requires a known source frontend family",
            ));
        };
        if self.payload_kind.source_kind() != self.source_kind {
            return Err(PreparedSuccessorBatchDescriptorError::new(
                PREPARED_SUCCESSOR_REJECTION_SOURCE_PAYLOAD_MISMATCH,
                format!(
                    "source kind {} does not match prepared payload kind {}",
                    self.source_kind.code(),
                    self.payload_kind.code()
                ),
            ));
        }
        if self.storage_kind == PreparedStorageKind::Unknown {
            return Err(PreparedSuccessorBatchDescriptorError::new(
                PREPARED_SUCCESSOR_REJECTION_UNKNOWN_STORAGE_KIND,
                "prepared successor batch requires a concrete storage ABI",
            ));
        }
        if self.lane == SetupTraceLaneKind::Unknown {
            return Err(PreparedSuccessorBatchDescriptorError::new(
                PREPARED_SUCCESSOR_REJECTION_UNKNOWN_LANE_KIND,
                "prepared successor batch requires a concrete lane",
            ));
        }
        if !self.compatible_frontend_families.contains(&source_family) {
            return Err(PreparedSuccessorBatchDescriptorError::new(
                PREPARED_SUCCESSOR_REJECTION_MISSING_SOURCE_FAMILY,
                format!(
                    "source family {} must be listed as compatible",
                    source_family.code()
                ),
            ));
        }
        for default_consumer in &self.default_consumers {
            if !self.compatible_frontend_families.contains(default_consumer) {
                return Err(PreparedSuccessorBatchDescriptorError::new(
                    PREPARED_SUCCESSOR_REJECTION_DEFAULT_CONSUMER_NOT_COMPATIBLE,
                    format!(
                        "default consumer {} is not compatible",
                        default_consumer.code()
                    ),
                ));
            }
        }
        if let Some(plan) = &self.admission_plan {
            plan.validate_runtime_admission().map_err(|error| {
                PreparedSuccessorBatchDescriptorError::new(error.reason_code, error.detail)
            })?;
            if plan.source_kind != self.source_kind {
                return Err(PreparedSuccessorBatchDescriptorError::new(
                    PREPARED_SUCCESSOR_REJECTION_ADMISSION_SOURCE_MISMATCH,
                    "fingerprint admission plan source must match successor source",
                ));
            }
            if plan.payload_kind != self.payload_kind {
                return Err(PreparedSuccessorBatchDescriptorError::new(
                    PREPARED_SUCCESSOR_REJECTION_ADMISSION_PAYLOAD_MISMATCH,
                    "fingerprint admission plan payload must match successor payload",
                ));
            }
            if plan.storage_kind != self.storage_kind {
                return Err(PreparedSuccessorBatchDescriptorError::new(
                    PREPARED_SUCCESSOR_REJECTION_ADMISSION_STORAGE_MISMATCH,
                    "fingerprint admission plan storage must match successor storage",
                ));
            }
            if plan.lane != self.lane {
                return Err(PreparedSuccessorBatchDescriptorError::new(
                    PREPARED_SUCCESSOR_REJECTION_ADMISSION_LANE_MISMATCH,
                    "fingerprint admission plan lane must match successor lane",
                ));
            }
        }
        Ok(())
    }

    /// Render one prepared successor batch evidence row.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let validation = self.validate();
        let (validation_status, reason_code) = match validation {
            Ok(()) => ("accepted", "accepted"),
            Err(ref error) => ("rejected", error.reason_code),
        };
        let remaining = self.remaining_compatible_frontend_families();
        format!(
            "{} {} schema={} schema_version={} source_kind={} frontend_family={} shared_engine_component={} descriptor_id={} payload_kind={} storage_kind={} lane_kind={} successor_payload_kind={} admission_kind={} admission_ready={} candidate_key={} prepared_program_identity={} prepared_lane_identity={} batch_artifact_identity={} fingerprint_admission_plan={} payload_witness={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} canonical_payload_available={} full_state_payload_available={} fingerprint_available={} sparse_delta_available={} parent_index_available={} raw_successor_metadata_complete={} compatible_frontend_families={} default_consumers={} remaining_compatible_frontend_families={} validation_status={} reason_code={}",
            evidence_value(scope),
            PREPARED_SUCCESSOR_BATCH_ROW_KIND,
            PREPARED_SUCCESSOR_BATCH_SCHEMA,
            PREPARED_SUCCESSOR_BATCH_SCHEMA_VERSION,
            self.source_kind.code(),
            self.source_kind.frontend_family_code(),
            PREPARED_SUCCESSOR_BATCH_ROW_KIND,
            evidence_value(&self.id),
            self.payload_kind.code(),
            self.storage_kind.code(),
            self.lane.code(),
            self.successor_payload_kind.code(),
            self.admission_kind().code(),
            evidence_bool(self.admission_ready()),
            evidence_optional(self.candidate_key.as_deref()),
            evidence_optional(self.prepared_program_identity.as_deref()),
            evidence_optional(self.prepared_lane_identity.as_deref()),
            evidence_optional(self.batch_artifact_identity.as_deref()),
            evidence_optional(self.admission_plan.as_ref().map(|plan| plan.id.as_str())),
            evidence_payload_witness(self.admission_plan.as_ref()),
            evidence_optional(
                self.admission_plan
                    .as_ref()
                    .map(|plan| plan.dedup.storage_policy_identity())
                    .as_deref(),
            ),
            evidence_optional(
                self.admission_plan
                    .as_ref()
                    .map(|plan| plan.dedup.fingerprint.fingerprint_policy_identity())
                    .as_deref(),
            ),
            evidence_optional(
                self.admission_plan
                    .as_ref()
                    .map(|plan| plan.dedup.fingerprint.fingerprint_identity())
                    .as_deref(),
            ),
            evidence_bool(self.successor_payload_kind.exposes_canonical_payload()),
            evidence_bool(self.successor_payload_kind.exposes_full_state()),
            evidence_bool(self.successor_payload_kind.exposes_fingerprint()),
            evidence_bool(self.successor_payload_kind.exposes_sparse_delta()),
            evidence_bool(self.parent_index_available),
            evidence_bool(self.raw_successor_metadata_complete),
            evidence_frontend_families(&self.compatible_frontend_families),
            evidence_frontend_families(&self.default_consumers),
            evidence_frontend_families(&remaining),
            validation_status,
            reason_code,
        )
    }

    /// Validate one prepared successor batch evidence row.
    pub fn validate_evidence_row(row: &str) -> Result<(), String> {
        validate_prepared_successor_batch_evidence_row(row)
    }
}

/// Borrowed prepared successor item emitted by a frontend-neutral producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedSuccessorRef<'a, A, F> {
    /// Frontend action/transition identity.
    pub action: A,
    /// Optional precomputed fingerprint.
    pub fingerprint: Option<F>,
    /// Optional canonical payload bytes for duplicate authorization.
    pub canonical_payload: Option<&'a [u8]>,
    /// Optional materialized full-state payload as canonical storage bytes.
    pub full_state_payload: Option<&'a [u8]>,
    /// Optional sparse-delta bytes from the parent.
    pub sparse_delta_payload: Option<&'a [u8]>,
    /// Optional parent index in a batched parent arena.
    pub parent_index: Option<u32>,
}

impl<'a, A, F> PreparedSuccessorRef<'a, A, F> {
    /// Create a successor item with no optional payload evidence.
    #[must_use]
    pub fn new(action: A) -> Self {
        Self {
            action,
            fingerprint: None,
            canonical_payload: None,
            full_state_payload: None,
            sparse_delta_payload: None,
            parent_index: None,
        }
    }

    /// Attach a precomputed fingerprint.
    #[must_use]
    pub fn with_fingerprint(mut self, fingerprint: F) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }

    /// Attach canonical payload bytes.
    #[must_use]
    pub fn with_canonical_payload(mut self, payload: &'a [u8]) -> Self {
        self.canonical_payload = Some(payload);
        self
    }

    /// Attach full state payload.
    #[must_use]
    pub fn with_full_state_payload(mut self, payload: &'a [u8]) -> Self {
        self.full_state_payload = Some(payload);
        self
    }

    /// Attach sparse-delta payload bytes.
    #[must_use]
    pub fn with_sparse_delta_payload(mut self, payload: &'a [u8]) -> Self {
        self.sparse_delta_payload = Some(payload);
        self
    }

    /// Attach parent index.
    #[must_use]
    pub fn with_parent_index(mut self, parent_index: u32) -> Self {
        self.parent_index = Some(parent_index);
        self
    }

    /// Whether this successor carries the minimum fingerprint-admission data.
    #[must_use]
    pub fn is_admission_ready(&self) -> bool {
        self.fingerprint.is_some() && self.canonical_payload.is_some()
    }
}

/// Borrowed batch of prepared successors for one or more parents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSuccessorBatch<'a, A, F> {
    /// Descriptor shared by all items in this batch.
    pub descriptor: &'a PreparedSuccessorBatchDescriptor,
    /// Borrowed successor items.
    pub successors: Vec<PreparedSuccessorRef<'a, A, F>>,
    /// Raw successor count reported by the producer.
    pub generated_count: usize,
    /// Whether raw successor metadata is complete.
    pub raw_successor_metadata_complete: bool,
    /// First parent index without raw successors, if producer detected a gap.
    pub first_parent_without_raw_successors: Option<u32>,
}

impl<'a, A, F> PreparedSuccessorBatch<'a, A, F> {
    /// Build a prepared successor batch from a descriptor and successor rows.
    #[must_use]
    pub fn new(
        descriptor: &'a PreparedSuccessorBatchDescriptor,
        successors: Vec<PreparedSuccessorRef<'a, A, F>>,
    ) -> Self {
        let generated_count = successors.len();
        Self {
            descriptor,
            successors,
            generated_count,
            raw_successor_metadata_complete: descriptor.raw_successor_metadata_complete,
            first_parent_without_raw_successors: None,
        }
    }
}

/// Frontend-neutral producer of prepared successor refs for a mutable state.
pub trait PreparedSuccessorProvider<State: ?Sized> {
    /// Frontend action/transition identity.
    type Action: Clone;
    /// Fingerprint type emitted by this provider.
    type Fingerprint: Copy + Eq + Hash;

    /// Descriptor for the produced successor batch.
    fn prepared_successor_batch_descriptor(&self) -> &PreparedSuccessorBatchDescriptor;

    /// Enumerate prepared successors for `state`.
    ///
    /// Returns `false` if the visitor stopped enumeration early.
    fn for_each_prepared_successor(
        &mut self,
        state: &mut State,
        visit: &mut dyn FnMut(PreparedSuccessorRef<'_, Self::Action, Self::Fingerprint>) -> bool,
    ) -> bool;
}

fn non_empty_string(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn dedup_frontend_families(
    families: impl Iterator<Item = SharedEngineFrontendFamily>,
) -> Vec<SharedEngineFrontendFamily> {
    let mut result = Vec::new();
    for family in families {
        if !result.contains(&family) {
            result.push(family);
        }
    }
    result
}

fn evidence_bool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn evidence_payload_witness(plan: Option<&PreparedFingerprintAdmissionPlan>) -> &'static str {
    plan.map_or("none", |plan| match plan.payload_witness {
        PreparedFingerprintPayloadWitnessKind::TlaArrayFp64 => "tla_array_fp64",
        PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3 => "compiled_flat_xxh3",
        PreparedFingerprintPayloadWitnessKind::PetriMarkingCas => "petri_marking_cas",
        PreparedFingerprintPayloadWitnessKind::RegisterVectorCanonical => {
            "register_vector_canonical"
        }
        PreparedFingerprintPayloadWitnessKind::ValidationReceiptProof => "validation_receipt_proof",
    })
}

fn evidence_frontend_families(families: &[SharedEngineFrontendFamily]) -> String {
    if families.is_empty() {
        return "none".to_string();
    }
    families
        .iter()
        .map(|family| family.code())
        .collect::<Vec<_>>()
        .join(",")
}

fn evidence_value(value: &str) -> String {
    if value.is_empty() {
        return "none".to_string();
    }
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':' | '=') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn validate_prepared_successor_batch_evidence_row(row: &str) -> Result<(), String> {
    let mut fields = row.split_whitespace();
    let scope = fields
        .next()
        .ok_or_else(|| "missing evidence scope".to_string())?;
    if scope.is_empty() {
        return Err("missing evidence scope".to_string());
    }
    let row_kind = fields
        .next()
        .ok_or_else(|| "missing row kind".to_string())?;
    if row_kind != PREPARED_SUCCESSOR_BATCH_ROW_KIND {
        return Err(format!(
            "row kind must be {PREPARED_SUCCESSOR_BATCH_ROW_KIND}, got {row_kind}"
        ));
    }
    for field in PREPARED_SUCCESSOR_BATCH_REQUIRED_FIELDS {
        require_field(row, field)?;
    }
    require_field_value(row, "schema", PREPARED_SUCCESSOR_BATCH_SCHEMA)?;
    require_field_value(
        row,
        "schema_version",
        &PREPARED_SUCCESSOR_BATCH_SCHEMA_VERSION.to_string(),
    )?;
    require_known_code(row, "source_kind", |code| {
        !matches!(code, "unknown" | "none") && source_kind_from_code(code).is_some()
    })?;
    require_known_code(row, "frontend_family", |code| {
        !matches!(code, "unknown" | "none") && SharedEngineFrontendFamily::from_code(code).is_some()
    })?;
    require_known_code(row, "payload_kind", |code| {
        payload_kind_from_code(code).is_some()
    })?;
    require_known_code(row, "storage_kind", |code| {
        !matches!(code, "unknown" | "none") && storage_kind_from_code(code).is_some()
    })?;
    require_known_code(row, "lane_kind", |code| {
        !matches!(code, "unknown" | "none") && lane_kind_from_code(code).is_some()
    })?;
    require_known_code(row, "successor_payload_kind", |code| {
        PreparedSuccessorPayloadKind::from_code(code).is_some()
    })?;
    require_known_code(row, "admission_kind", |code| {
        PreparedSuccessorAdmissionKind::from_code(code).is_some()
    })?;
    for bool_field in [
        "admission_ready",
        "canonical_payload_available",
        "full_state_payload_available",
        "fingerprint_available",
        "sparse_delta_available",
        "parent_index_available",
        "raw_successor_metadata_complete",
    ] {
        require_known_code(row, bool_field, |code| matches!(code, "true" | "false"))?;
    }
    validate_frontend_family_list(row, "compatible_frontend_families")?;
    validate_frontend_family_list(row, "default_consumers")?;
    validate_frontend_family_list(row, "remaining_compatible_frontend_families")?;
    validate_successor_row_source_payload_pair(row)?;
    validate_successor_row_admission_fields(row)?;
    require_known_code(row, "validation_status", |code| {
        matches!(code, "accepted" | "rejected")
    })?;
    if require_field(row, "reason_code")?.is_empty() {
        return Err("field reason_code must not be empty".to_string());
    }
    Ok(())
}

fn validate_successor_row_source_payload_pair(row: &str) -> Result<(), String> {
    let source_kind = source_kind_from_code(require_field(row, "source_kind")?)
        .ok_or_else(|| "source kind must be known".to_string())?;
    let payload_kind = payload_kind_from_code(require_field(row, "payload_kind")?)
        .ok_or_else(|| "payload kind must be known".to_string())?;
    if payload_kind.source_kind() == source_kind {
        Ok(())
    } else {
        Err(format!(
            "source kind {} does not match payload kind {}",
            source_kind.code(),
            payload_kind.code()
        ))
    }
}

fn validate_successor_row_admission_fields(row: &str) -> Result<(), String> {
    let admission_kind = require_field(row, "admission_kind")?;
    let plan = require_field(row, "fingerprint_admission_plan")?;
    let payload_witness = require_field(row, "payload_witness")?;
    match admission_kind {
        "none" if (plan != "none" || payload_witness != "none") => {
            return Err("admission_kind=none requires no fingerprint admission plan".to_string());
        }
        "prepared_fingerprint" if (plan == "none" || payload_witness == "none") => {
            return Err(
                "prepared fingerprint admission requires a plan and payload witness".to_string(),
            );
        }
        _ => {}
    }

    let admission_ready = require_bool_field(row, "admission_ready")?;
    if admission_ready
        && (admission_kind != "prepared_fingerprint"
            || !require_bool_field(row, "canonical_payload_available")?
            || !require_bool_field(row, "fingerprint_available")?)
    {
        return Err(
            "admission_ready=true requires prepared fingerprint, canonical payload, and fingerprint"
                .to_string(),
        );
    }
    Ok(())
}

fn require_bool_field(row: &str, key: &str) -> Result<bool, String> {
    match require_field(row, key)? {
        "true" => Ok(true),
        "false" => Ok(false),
        value => Err(format!("field {key} has unknown value {value}")),
    }
}

fn require_field<'a>(row: &'a str, key: &str) -> Result<&'a str, String> {
    evidence_field(row, key).ok_or_else(|| format!("missing field {key}"))
}

fn require_field_value(row: &str, key: &str, expected: &str) -> Result<(), String> {
    let actual = require_field(row, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!("field {key} expected {expected}, got {actual}"))
    }
}

fn require_known_code(
    row: &str,
    key: &str,
    valid: impl FnOnce(&str) -> bool,
) -> Result<(), String> {
    let value = require_field(row, key)?;
    if valid(value) {
        Ok(())
    } else {
        Err(format!("field {key} has unknown value {value}"))
    }
}

fn validate_frontend_family_list(row: &str, key: &str) -> Result<(), String> {
    let value = require_field(row, key)?;
    if value == "none" {
        return Ok(());
    }
    for family in value.split(',') {
        if SharedEngineFrontendFamily::from_code(family).is_none() {
            return Err(format!("field {key} has unknown frontend family {family}"));
        }
    }
    Ok(())
}

fn source_kind_from_code(code: &str) -> Option<CheckerSourceKind> {
    match code {
        "tla" => Some(CheckerSourceKind::Tla),
        "quint" => Some(CheckerSourceKind::Quint),
        "mcc_petri" => Some(CheckerSourceKind::MccPetri),
        "aiger" => Some(CheckerSourceKind::Aiger),
        "btor2" => Some(CheckerSourceKind::Btor2),
        "vmt_interchange" => Some(CheckerSourceKind::VmtInterchange),
        "ay_only" => Some(CheckerSourceKind::AYOnly),
        "witness_replay" => Some(CheckerSourceKind::WitnessReplay),
        _ => None,
    }
}

fn payload_kind_from_code(code: &str) -> Option<PreparedProgramPayloadKind> {
    PreparedProgramPayloadKind::shared_engine_payloads()
        .iter()
        .copied()
        .find(|payload| payload.code() == code)
}

fn storage_kind_from_code(code: &str) -> Option<PreparedStorageKind> {
    match code {
        "tla_state_slots" => Some(PreparedStorageKind::TlaStateSlots),
        "petri_marking" => Some(PreparedStorageKind::PetriMarking),
        "hardware_registers" => Some(PreparedStorageKind::HardwareRegisters),
        "smt_variables" => Some(PreparedStorageKind::SmtVariables),
        "witness_steps" => Some(PreparedStorageKind::WitnessSteps),
        _ => None,
    }
}

fn lane_kind_from_code(code: &str) -> Option<SetupTraceLaneKind> {
    match code {
        "frontend" => Some(SetupTraceLaneKind::Frontend),
        "explicit_state" => Some(SetupTraceLaneKind::ExplicitState),
        "native" => Some(SetupTraceLaneKind::Native),
        "ay" => Some(SetupTraceLaneKind::AY),
        "analytical" => Some(SetupTraceLaneKind::Analytical),
        "replay" => Some(SetupTraceLaneKind::Replay),
        "fingerprint" => Some(SetupTraceLaneKind::Fingerprint),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn petri_descriptor() -> PreparedSuccessorBatchDescriptor {
        PreparedSuccessorBatchDescriptor::new(
            "mcc_petri.prepared_successor_batch.v1",
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
            SetupTraceLaneKind::ExplicitState,
            PreparedSuccessorPayloadKind::FullCanonicalState,
        )
        .with_admission_plan(PreparedFingerprintAdmissionPlan::petri_marking_cas(
            "mcc_petri.prepared_successor_batch.admission.v1",
            "pack_marking_config.v1",
        ))
        .with_candidate_key("interpreted")
        .with_prepared_program_identity("petri-prepared-program")
        .with_prepared_lane_identity("petri-explicit-lane")
        .with_batch_artifact_identity("petri-successor-batch-artifact")
    }

    #[test]
    fn descriptor_renders_and_validates_shared_successor_batch_evidence() {
        let descriptor = petri_descriptor();

        descriptor
            .validate()
            .expect("Petri prepared successor descriptor should validate");
        let row = descriptor.render_evidence_row("MCC");

        PreparedSuccessorBatchDescriptor::validate_evidence_row(&row)
            .expect("rendered row should validate");
        assert!(row.contains("schema=ty.prepared_successor_batch.v1"));
        assert!(row.contains("source_kind=mcc_petri"));
        assert!(row.contains("payload_kind=mcc_petri"));
        assert!(row.contains("successor_payload_kind=full_canonical_state"));
        assert!(row.contains("admission_kind=prepared_fingerprint"));
        assert!(row.contains("admission_ready=true"));
        assert!(row.contains("payload_witness=petri_marking_cas"));
        assert!(row.contains("storage_policy_identity="));
        assert!(row.contains("fingerprint_policy_identity="));
        assert!(row.contains("fingerprint_identity="));
        assert!(row.contains("compatible_frontend_families=mcc_petri"));
    }

    #[test]
    fn descriptor_rejects_empty_id_and_mismatched_admission_plan() {
        let empty = PreparedSuccessorBatchDescriptor::new(
            "",
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
            SetupTraceLaneKind::ExplicitState,
            PreparedSuccessorPayloadKind::FullCanonicalState,
        );
        assert_eq!(
            empty.validate().unwrap_err().reason_code,
            PREPARED_SUCCESSOR_REJECTION_EMPTY_ID
        );

        let mismatched = PreparedSuccessorBatchDescriptor::new(
            "bad",
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
            SetupTraceLaneKind::ExplicitState,
            PreparedSuccessorPayloadKind::FullCanonicalState,
        )
        .with_admission_plan(PreparedFingerprintAdmissionPlan::tla_array_fp64(
            "tla admission",
            "slots-v1",
        ));
        assert_eq!(
            mismatched.validate().unwrap_err().reason_code,
            PREPARED_SUCCESSOR_REJECTION_ADMISSION_SOURCE_MISMATCH
        );
    }

    #[test]
    fn descriptor_covers_tla_hardware_ay_and_replay_frontends() {
        let cases = [
            (
                CheckerSourceKind::Tla,
                PreparedProgramPayloadKind::Tla,
                PreparedStorageKind::TlaStateSlots,
                SetupTraceLaneKind::ExplicitState,
                PreparedSuccessorPayloadKind::CanonicalPayloadOnly,
            ),
            (
                CheckerSourceKind::Aiger,
                PreparedProgramPayloadKind::Aiger,
                PreparedStorageKind::HardwareRegisters,
                SetupTraceLaneKind::Fingerprint,
                PreparedSuccessorPayloadKind::SparseDelta,
            ),
            (
                CheckerSourceKind::Btor2,
                PreparedProgramPayloadKind::Btor2,
                PreparedStorageKind::HardwareRegisters,
                SetupTraceLaneKind::Fingerprint,
                PreparedSuccessorPayloadKind::SparseDelta,
            ),
            (
                CheckerSourceKind::VmtInterchange,
                PreparedProgramPayloadKind::VmtInterchange,
                PreparedStorageKind::SmtVariables,
                SetupTraceLaneKind::Fingerprint,
                PreparedSuccessorPayloadKind::SparseDelta,
            ),
            (
                CheckerSourceKind::AYOnly,
                PreparedProgramPayloadKind::AYOnly,
                PreparedStorageKind::SmtVariables,
                SetupTraceLaneKind::AY,
                PreparedSuccessorPayloadKind::ProofOrReplayStep,
            ),
            (
                CheckerSourceKind::WitnessReplay,
                PreparedProgramPayloadKind::WitnessReplay,
                PreparedStorageKind::WitnessSteps,
                SetupTraceLaneKind::Replay,
                PreparedSuccessorPayloadKind::ProofOrReplayStep,
            ),
        ];

        for (source, payload, storage, lane, successor_payload) in cases {
            let descriptor = PreparedSuccessorBatchDescriptor::new(
                format!("{}.successor_batch", source.code()),
                source,
                payload,
                storage,
                lane,
                successor_payload,
            );
            descriptor
                .validate()
                .expect("frontend descriptor should use shared successor contract");
            assert!(descriptor
                .compatible_frontend_families
                .contains(&source.adoption_frontend_family().unwrap()));
        }
    }

    #[test]
    fn successor_ref_carries_admission_payload_forms() {
        let canonical = [1_u8, 2, 3];
        let full = [5_u8, 8, 13];
        let delta = [0_u8, 2, 1];
        let successor = PreparedSuccessorRef::new("fire")
            .with_fingerprint(42_u128)
            .with_canonical_payload(&canonical)
            .with_full_state_payload(&full)
            .with_sparse_delta_payload(&delta)
            .with_parent_index(7);

        assert!(successor.is_admission_ready());
        assert_eq!(successor.action, "fire");
        assert_eq!(successor.fingerprint, Some(42));
        assert_eq!(successor.canonical_payload, Some(canonical.as_slice()));
        assert_eq!(successor.full_state_payload, Some(full.as_slice()));
        assert_eq!(successor.sparse_delta_payload, Some(delta.as_slice()));
        assert_eq!(successor.parent_index, Some(7));
    }
}
