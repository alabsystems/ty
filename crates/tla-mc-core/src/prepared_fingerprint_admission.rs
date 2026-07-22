// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Prepared-program fingerprint admission planning.
//!
//! This module is intentionally descriptor-only. It records the shared dedup
//! identity, duplicate evidence required by that identity, payload witness
//! shape, and prepared source/lane metadata without binding to any runtime
//! frontend or storage implementation.

use crate::{
    evidence_row::evidence_field as prepared_admission_evidence_field,
    fingerprint_identity::{
        SharedCollisionPolicy, SharedDedupIdentity, SharedDedupScope, SharedDedupStorageKind,
        SharedDuplicateAuthorization, SharedFingerprintAlgorithm, SharedFingerprintIdentity,
        SharedFingerprintIdentityRejection, SharedFingerprintValueKind,
    },
    prepared_program::{
        PreparedCandidateLaneDescriptor, PreparedCheckerProgram, PreparedFingerprintDescriptor,
        PreparedProgramPayloadKind, PreparedStorageKind,
    },
    setup_trace::{CheckerArtifactIdentityFields, CheckerSourceKind, SetupTraceLaneKind},
    shared_engine_adoption::SharedEngineFrontendFamily,
    storage::{FingerprintAdmission, FingerprintSet, StorageFault},
    validation_receipt::ValidationReceipt,
};

const PREPARED_FINGERPRINT_ADMISSION_BACKEND: &str = "prepared_fingerprint_admission";
const PREPARED_FINGERPRINT_ADMISSION_OPERATION: &str = "admit";
const PREPARED_FINGERPRINT_ADMISSION_SCHEMA: &str = "ty.prepared_fingerprint_admission.v1";
const PREPARED_FINGERPRINT_ADMISSION_SCHEMA_VERSION: u32 = 1;
const PREPARED_FINGERPRINT_ADMISSION_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "source_kind",
    "frontend_family",
    "shared_engine_component",
    "plan_id",
    "payload_kind",
    "storage_kind",
    "lane_kind",
    "candidate_key",
    "prepared_program_identity",
    "prepared_lane_identity",
    "payload_witness",
    "dedup_identity",
    "storage_policy_identity",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "collision_policy",
    "duplicate_authorization",
    "admission_status",
    "reason_code",
    "fail_closed",
    "compatible_frontend_families",
    "default_consumers",
    "remaining_compatible_frontend_families",
    "blockers",
];
const PREPARED_FINGERPRINT_ADMISSION_FUTURE_IMPORTER_BLOCKER: &str =
    "future_importer:awaiting_registered_importer_frontend";

const PREPARED_FINGERPRINT_REJECTION_EMPTY_PLAN_ID: &str = "empty_prepared_admission_id";
const PREPARED_FINGERPRINT_REJECTION_UNKNOWN_SOURCE_KIND: &str = "unknown_source_kind";
const PREPARED_FINGERPRINT_REJECTION_SOURCE_PAYLOAD_MISMATCH: &str = "source_payload_mismatch";
const PREPARED_FINGERPRINT_REJECTION_UNKNOWN_STORAGE_KIND: &str = "unknown_storage_kind";
const PREPARED_FINGERPRINT_REJECTION_EVIDENCE_ONLY_STORAGE: &str = "evidence_only_storage";
const PREPARED_FINGERPRINT_REJECTION_UNKNOWN_LANE_KIND: &str = "unknown_lane_kind";
const PREPARED_FINGERPRINT_REJECTION_LANE_MISMATCH: &str = "lane_mismatch";
const PREPARED_FINGERPRINT_REJECTION_MISSING_COLLISION_POLICY: &str = "missing_collision_policy";
const PREPARED_FINGERPRINT_REJECTION_COLLISION_POLICY_MISMATCH: &str = "collision_policy_mismatch";
const PREPARED_FINGERPRINT_REJECTION_DUPLICATE_AUTHORIZATION_MISMATCH: &str =
    "duplicate_authorization_mismatch";
const PREPARED_FINGERPRINT_REJECTION_CANONICAL_PAYLOAD_MISMATCH: &str =
    "canonical_payload_mismatch";
const PREPARED_FINGERPRINT_REJECTION_CANONICAL_PAYLOAD_UNSUPPORTED: &str =
    "canonical_payload_equality_unsupported";
const PREPARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REQUIRED: &str = "proof_witness_required";
const PREPARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REJECTED: &str = "proof_witness_rejected";
const PREPARED_FINGERPRINT_REJECTION_PROOF_WITNESS_UNSUPPORTED: &str = "proof_witness_unsupported";
const PREPARED_FINGERPRINT_REJECTION_STORAGE_POLICY_IDENTITY_MISMATCH: &str =
    "storage_policy_identity_mismatch";
const PREPARED_FINGERPRINT_REJECTION_FINGERPRINT_POLICY_IDENTITY_MISMATCH: &str =
    "fingerprint_policy_identity_mismatch";
const PREPARED_FINGERPRINT_REJECTION_FINGERPRINT_IDENTITY_MISMATCH: &str =
    "fingerprint_identity_mismatch";

/// Canonical payload evidence attached to a prepared fingerprint admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedFingerprintPayloadWitnessKind {
    /// TLC/TLA array state slots fingerprinted with the FP64 algorithm.
    TlaArrayFp64,
    /// Compiled/native flat canonical bytes fingerprinted with xxh3.
    CompiledFlatXxh3,
    /// Petri/MCC marking vector admitted into CAS dedup storage.
    PetriMarkingCas,
    /// Hardware, trust-ir, or solver register vector canonical bytes.
    RegisterVectorCanonical,
    /// Replay/proof/certificate validation receipt bytes.
    ValidationReceiptProof,
}

impl PreparedFingerprintPayloadWitnessKind {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaArrayFp64 => "tla_array_fp64",
            Self::CompiledFlatXxh3 => "compiled_flat_xxh3",
            Self::PetriMarkingCas => "petri_marking_cas",
            Self::RegisterVectorCanonical => "register_vector_canonical",
            Self::ValidationReceiptProof => "validation_receipt_proof",
        }
    }
}

/// Frontend-neutral plan for admitting a prepared fingerprint into dedup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedFingerprintAdmissionPlan {
    /// Human-readable plan id.
    pub id: String,
    /// Prepared program/source that produced the payload.
    pub source_kind: CheckerSourceKind,
    /// Prepared payload family.
    pub payload_kind: PreparedProgramPayloadKind,
    /// Prepared storage ABI for the payload witness.
    pub storage_kind: PreparedStorageKind,
    /// Prepared candidate lane that consumes the admission decision.
    pub lane: SetupTraceLaneKind,
    /// Shared dedup identity used for storage and equality.
    pub dedup: SharedDedupIdentity,
    /// Evidence required before a duplicate may be suppressed.
    pub duplicate_authorization: SharedDuplicateAuthorization,
    /// Payload witness kind used to confirm equal fingerprints.
    pub payload_witness: PreparedFingerprintPayloadWitnessKind,
    /// Optional prepared program identity.
    pub prepared_program_identity: Option<String>,
    /// Optional prepared candidate lane identity.
    pub prepared_lane_identity: Option<String>,
    /// Optional prepared candidate key.
    pub candidate_key: Option<String>,
    /// Prepared identity fields after merging program, lane, and dedup fields.
    pub identities: CheckerArtifactIdentityFields,
}

/// Evidence that a prepared fingerprint admission descriptor was validated at
/// setup time and is not revalidated by hot runtime handle calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedFingerprintAdmissionValidationEvidence {
    /// Full descriptor validations performed while creating the runtime handle.
    pub setup_descriptor_validation_count: u64,
    /// Full descriptor validations performed by hot runtime handle methods.
    pub hot_descriptor_validation_count: u64,
}

impl PreparedFingerprintAdmissionValidationEvidence {
    const SETUP_ONCE: Self = Self {
        setup_descriptor_validation_count: 1,
        hot_descriptor_validation_count: 0,
    };
}

/// Reusable duplicate evidence produced by a prepared admission check.
///
/// This is intentionally storage/frontend neutral: scalar sets, fused batches,
/// proof caches, and persisted stores can all report the same authorization
/// record after the prepared handle has checked policy compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedFingerprintDuplicateAuthorizationEvidence {
    /// Runtime collision policy observed at the storage/admission boundary.
    pub observed_collision_policy: SharedCollisionPolicy,
    /// Duplicate evidence accepted for that policy.
    pub authorization: SharedDuplicateAuthorization,
    /// Prepared payload witness contract used to validate the duplicate.
    pub payload_witness: PreparedFingerprintPayloadWitnessKind,
}

impl PreparedFingerprintDuplicateAuthorizationEvidence {
    /// Whether this evidence satisfies the observed fail-closed policy.
    #[must_use]
    pub fn satisfies_observed_policy(self) -> bool {
        self.observed_collision_policy
            .authorizes_duplicate(self.authorization)
    }
}

/// Per-call prepared admission counters for hot scalar and batch paths.
///
/// Setup validation counters live here too so consumers can combine setup rows
/// and hot rows without inventing frontend-specific counter names. Hot methods
/// on [`ValidatedPreparedFingerprintAdmissionPlan`] report zero descriptor
/// validations; setup validation is exposed by
/// [`ValidatedPreparedFingerprintAdmissionPlan::validation_evidence`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PreparedFingerprintAdmissionCounters {
    /// Descriptor validations performed while preparing this call outcome.
    pub setup_descriptor_validation_count: u64,
    /// Descriptor validations performed by the hot admission path.
    pub hot_descriptor_validation_count: u64,
    /// Fingerprints represented by this admission call/outcome.
    pub fingerprints_attempted: u64,
    /// Fingerprints newly inserted/admitted by this call/outcome.
    pub fingerprints_inserted: u64,
    /// Fingerprints suppressed as duplicates by this call/outcome.
    pub duplicate_fingerprints: u64,
    /// Duplicate authorization checks performed by this call/outcome.
    pub duplicate_authorization_checks: u64,
}

impl PreparedFingerprintAdmissionCounters {
    fn hot_scalar(admission: FingerprintAdmission, duplicate_authorization_checked: bool) -> Self {
        Self {
            fingerprints_attempted: 1,
            fingerprints_inserted: u64::from(admission.is_new()),
            duplicate_fingerprints: u64::from(admission.is_duplicate()),
            duplicate_authorization_checks: u64::from(duplicate_authorization_checked),
            ..Self::default()
        }
    }

    fn hot_batch(
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        duplicate_authorization_checked: bool,
    ) -> Self {
        let duplicate_fingerprints = if fault_present {
            0
        } else {
            attempted.saturating_sub(inserted_count) as u64
        };
        Self {
            fingerprints_attempted: attempted as u64,
            fingerprints_inserted: inserted_count as u64,
            duplicate_fingerprints,
            duplicate_authorization_checks: u64::from(duplicate_authorization_checked),
            ..Self::default()
        }
    }
}

/// Typed result for one prepared fingerprint admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedFingerprintAdmissionOutcome {
    /// Frontend-neutral storage admission decision.
    pub admission: FingerprintAdmission,
    /// Evidence used to authorize a duplicate, when this was a duplicate.
    pub duplicate_authorization: Option<PreparedFingerprintDuplicateAuthorizationEvidence>,
    /// Hot-path counters for this admission result.
    pub counters: PreparedFingerprintAdmissionCounters,
}

impl PreparedFingerprintAdmissionOutcome {
    fn new(
        admission: FingerprintAdmission,
        duplicate_authorization: Option<PreparedFingerprintDuplicateAuthorizationEvidence>,
    ) -> Self {
        Self {
            admission,
            duplicate_authorization,
            counters: PreparedFingerprintAdmissionCounters::hot_scalar(
                admission,
                duplicate_authorization.is_some(),
            ),
        }
    }
}

/// Typed result for prepared batch duplicate authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedFingerprintBatchAdmissionOutcome {
    /// Fingerprints attempted by the fault-free prefix of the batch.
    pub attempted: usize,
    /// Fingerprints inserted by the storage backend for that prefix.
    pub inserted_count: usize,
    /// Whether storage already reported a fault for this batch.
    pub fault_present: bool,
    /// Evidence used to authorize suppressed duplicates, when any were present.
    pub duplicate_authorization: Option<PreparedFingerprintDuplicateAuthorizationEvidence>,
    /// Hot-path counters for this batch result.
    pub counters: PreparedFingerprintAdmissionCounters,
}

impl PreparedFingerprintBatchAdmissionOutcome {
    fn new(
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        duplicate_authorization: Option<PreparedFingerprintDuplicateAuthorizationEvidence>,
    ) -> Self {
        Self {
            attempted,
            inserted_count,
            fault_present,
            duplicate_authorization,
            counters: PreparedFingerprintAdmissionCounters::hot_batch(
                attempted,
                inserted_count,
                fault_present,
                duplicate_authorization.is_some(),
            ),
        }
    }

    /// Number of duplicate fingerprints represented by this batch outcome.
    #[must_use]
    pub fn duplicate_count(self) -> usize {
        if self.fault_present {
            0
        } else {
            self.attempted.saturating_sub(self.inserted_count)
        }
    }
}

/// Setup-validated runtime handle for admitting prepared fingerprints.
///
/// Construct this once at setup/binding time, then use the admission methods
/// on the handle from hot loops. The descriptor remains available for evidence
/// and compatibility, but full prepared-plan validation is not repeated per
/// fingerprint admission.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidatedPreparedFingerprintAdmissionPlan {
    plan: PreparedFingerprintAdmissionPlan,
    validation_evidence: PreparedFingerprintAdmissionValidationEvidence,
}

impl ValidatedPreparedFingerprintAdmissionPlan {
    /// Validate and bind a prepared admission descriptor for runtime use.
    pub fn new(
        plan: PreparedFingerprintAdmissionPlan,
    ) -> Result<Self, SharedFingerprintIdentityRejection> {
        plan.validate_runtime_admission()?;
        Ok(Self {
            plan,
            validation_evidence: PreparedFingerprintAdmissionValidationEvidence::SETUP_ONCE,
        })
    }

    /// Descriptor that was validated at setup time.
    #[must_use]
    pub fn plan(&self) -> &PreparedFingerprintAdmissionPlan {
        &self.plan
    }

    /// Consume the runtime handle and recover its descriptor.
    #[must_use]
    pub fn into_plan(self) -> PreparedFingerprintAdmissionPlan {
        self.plan
    }

    /// Runtime validation evidence for downstream setup/admission rows.
    #[must_use]
    pub fn validation_evidence(&self) -> PreparedFingerprintAdmissionValidationEvidence {
        self.validation_evidence
    }

    /// Validate that the runtime collision policy observed by storage is the
    /// same fail-closed policy bound into this prepared admission handle.
    pub fn validate_collision_policy_binding(
        &self,
        observed_collision_policy: Option<SharedCollisionPolicy>,
    ) -> Result<SharedCollisionPolicy, SharedFingerprintIdentityRejection> {
        self.plan
            .validate_collision_policy_binding(observed_collision_policy)
    }

    /// Admit a fingerprint through this validated handle, using canonical
    /// payload comparison as the only duplicate authorization.
    pub fn admit_fingerprint_with_canonical_payload_comparison<F, S>(
        &self,
        fingerprint_set: &S,
        fingerprint: F,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault>
    where
        S: FingerprintSet<F> + ?Sized,
    {
        self.admit_fingerprint_with_canonical_payload_evidence(
            fingerprint_set,
            fingerprint,
            &mut canonical_payloads_equal,
        )
        .map(|outcome| outcome.admission)
    }

    /// Admit a fingerprint through this validated handle and return reusable
    /// scalar admission evidence/counters.
    pub fn admit_fingerprint_with_canonical_payload_evidence<F, S>(
        &self,
        fingerprint_set: &S,
        fingerprint: F,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintAdmissionOutcome, StorageFault>
    where
        S: FingerprintSet<F> + ?Sized,
    {
        let mut duplicate_authorization = None;
        let admission = {
            let mut authorize_duplicate = |policy| {
                let payloads_equal = canonical_payloads_equal()?;
                let authorization = self
                    .authorize_duplicate_with_canonical_payload_equality(policy, payloads_equal)?;
                duplicate_authorization =
                    Some(self.duplicate_authorization_evidence(policy, authorization));
                Ok(authorization)
            };
            fingerprint_set.admit_fingerprint_with_duplicate_authorization(
                fingerprint,
                &self.plan.dedup,
                &mut authorize_duplicate,
            )?
        };
        Ok(PreparedFingerprintAdmissionOutcome::new(
            admission,
            duplicate_authorization,
        ))
    }

    /// Admit an ordered fingerprint batch through this validated handle and
    /// authorize any suppressed duplicates with canonical payload evidence.
    pub fn admit_fingerprint_batch_with_canonical_payload_evidence<F, S>(
        &self,
        fingerprint_set: &S,
        fingerprints: &[F],
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintBatchAdmissionOutcome, StorageFault>
    where
        F: Copy,
        S: FingerprintSet<F> + ?Sized,
    {
        let batch = fingerprint_set.admit_fingerprint_batch(fingerprints)?;
        self.enforce_batch_duplicate_with_canonical_payload_evidence(
            batch.attempted_count(),
            batch.inserted_count(),
            false,
            &mut canonical_payloads_equal,
        )
    }

    /// Enforce this validated handle on an already computed admission decision.
    pub fn enforce_duplicate_with_canonical_payload_comparison(
        &self,
        admission: FingerprintAdmission,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault> {
        self.enforce_duplicate_with_canonical_payload_evidence(
            admission,
            &mut canonical_payloads_equal,
        )
        .map(|outcome| outcome.admission)
    }

    /// Enforce this validated handle on an already computed admission decision
    /// and return reusable scalar admission evidence/counters.
    pub fn enforce_duplicate_with_canonical_payload_evidence(
        &self,
        admission: FingerprintAdmission,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintAdmissionOutcome, StorageFault> {
        let mut duplicate_authorization = None;
        let admission = {
            let mut authorize_duplicate = |policy| {
                let payloads_equal = canonical_payloads_equal()?;
                let authorization = self
                    .authorize_duplicate_with_canonical_payload_equality(policy, payloads_equal)?;
                duplicate_authorization =
                    Some(self.duplicate_authorization_evidence(policy, authorization));
                Ok(authorization)
            };
            admission.enforce_shared_duplicate_authorization(
                &self.plan.dedup,
                &mut authorize_duplicate,
            )?
        };
        Ok(PreparedFingerprintAdmissionOutcome::new(
            admission,
            duplicate_authorization,
        ))
    }

    /// Enforce this validated handle on a batch admission summary.
    ///
    /// A storage fault already fails the batch, so this only validates duplicate
    /// authorization for fault-free attempted prefixes that suppressed at least
    /// one fingerprint.
    pub fn enforce_batch_duplicate_with_canonical_payload_comparison(
        &self,
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<(), StorageFault> {
        self.enforce_batch_duplicate_with_canonical_payload_evidence(
            attempted,
            inserted_count,
            fault_present,
            &mut canonical_payloads_equal,
        )
        .map(|_| ())
    }

    /// Enforce this validated handle on a batch admission summary and return
    /// reusable batch duplicate evidence/counters.
    pub fn enforce_batch_duplicate_with_canonical_payload_evidence(
        &self,
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintBatchAdmissionOutcome, StorageFault> {
        if fault_present || inserted_count >= attempted {
            return Ok(PreparedFingerprintBatchAdmissionOutcome::new(
                attempted,
                inserted_count,
                fault_present,
                None,
            ));
        }
        let outcome = self.enforce_duplicate_with_canonical_payload_evidence(
            FingerprintAdmission::Duplicate,
            &mut canonical_payloads_equal,
        )?;
        Ok(PreparedFingerprintBatchAdmissionOutcome::new(
            attempted,
            inserted_count,
            fault_present,
            outcome.duplicate_authorization,
        ))
    }

    /// Convert a canonical payload comparison result into the duplicate
    /// authorization required by this validated handle.
    pub fn authorize_duplicate_with_canonical_payload_equality(
        &self,
        observed_collision_policy: SharedCollisionPolicy,
        canonical_payloads_equal: bool,
    ) -> Result<SharedDuplicateAuthorization, StorageFault> {
        let policy = self
            .validate_collision_policy_binding(Some(observed_collision_policy))
            .map_err(|rejection| self.plan.admission_fault(rejection))?;
        if policy.requires_validation_receipt() {
            let rejection = SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_CANONICAL_PAYLOAD_UNSUPPORTED,
                "canonical payload equality cannot satisfy a proof/witness collision policy",
            );
            return Err(self.plan.admission_fault(rejection));
        }
        if !canonical_payloads_equal {
            let rejection = SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_CANONICAL_PAYLOAD_MISMATCH,
                "resident and candidate canonical payloads differ for the same fingerprint",
            );
            return Err(self.plan.admission_fault(rejection));
        }
        Ok(SharedDuplicateAuthorization::CanonicalPayloadEquality)
    }

    /// Convert proof/witness validation into the duplicate authorization
    /// required by proof/certificate-backed validated handles.
    pub fn authorize_duplicate_with_proof_witness(
        &self,
        observed_collision_policy: SharedCollisionPolicy,
        proof_witness_accepted: bool,
    ) -> Result<SharedDuplicateAuthorization, StorageFault> {
        let policy = self
            .validate_collision_policy_binding(Some(observed_collision_policy))
            .map_err(|rejection| self.plan.admission_fault(rejection))?;
        if !policy.requires_validation_receipt() {
            let rejection = SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_PROOF_WITNESS_UNSUPPORTED,
                "proof/witness authorization is only valid for proof/witness collision policies",
            );
            return Err(self.plan.admission_fault(rejection));
        }
        if self.plan.duplicate_authorization != SharedDuplicateAuthorization::ProofWitness {
            let rejection = SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REQUIRED,
                "prepared admission plan does not bind proof/witness duplicate authorization",
            );
            return Err(self.plan.admission_fault(rejection));
        }
        if !proof_witness_accepted {
            let rejection = SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REJECTED,
                "proof/witness validation did not authorize the duplicate fingerprint",
            );
            return Err(self.plan.admission_fault(rejection));
        }
        Ok(SharedDuplicateAuthorization::ProofWitness)
    }

    /// Convert a validation receipt into duplicate authorization for
    /// proof/witness-backed validated handles.
    pub fn authorize_duplicate_with_validation_receipt(
        &self,
        observed_collision_policy: SharedCollisionPolicy,
        receipt: &ValidationReceipt,
    ) -> Result<SharedDuplicateAuthorization, StorageFault> {
        self.authorize_duplicate_with_proof_witness(
            observed_collision_policy,
            receipt.proof_witness_duplicate_authorization()
                == SharedDuplicateAuthorization::ProofWitness,
        )
    }

    /// Enforce this validated handle on an already computed admission decision
    /// with proof/witness authorization evidence.
    pub fn enforce_duplicate_with_proof_witness_evidence(
        &self,
        admission: FingerprintAdmission,
        mut proof_witness_accepted: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintAdmissionOutcome, StorageFault> {
        let mut duplicate_authorization = None;
        let admission = {
            let mut authorize_duplicate = |policy| {
                let authorization =
                    self.authorize_duplicate_with_proof_witness(policy, proof_witness_accepted()?)?;
                duplicate_authorization =
                    Some(self.duplicate_authorization_evidence(policy, authorization));
                Ok(authorization)
            };
            admission.enforce_shared_duplicate_authorization(
                &self.plan.dedup,
                &mut authorize_duplicate,
            )?
        };
        Ok(PreparedFingerprintAdmissionOutcome::new(
            admission,
            duplicate_authorization,
        ))
    }

    /// Enforce this validated handle on a batch summary with proof/witness
    /// authorization evidence.
    pub fn enforce_batch_duplicate_with_proof_witness_evidence(
        &self,
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        mut proof_witness_accepted: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintBatchAdmissionOutcome, StorageFault> {
        if fault_present || inserted_count >= attempted {
            return Ok(PreparedFingerprintBatchAdmissionOutcome::new(
                attempted,
                inserted_count,
                fault_present,
                None,
            ));
        }
        let outcome = self.enforce_duplicate_with_proof_witness_evidence(
            FingerprintAdmission::Duplicate,
            &mut proof_witness_accepted,
        )?;
        Ok(PreparedFingerprintBatchAdmissionOutcome::new(
            attempted,
            inserted_count,
            fault_present,
            outcome.duplicate_authorization,
        ))
    }

    /// Admit a fingerprint and authorize duplicates with proof/witness
    /// validation evidence.
    pub fn admit_fingerprint_with_proof_witness_evidence<F, S>(
        &self,
        fingerprint_set: &S,
        fingerprint: F,
        mut proof_witness_accepted: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<PreparedFingerprintAdmissionOutcome, StorageFault>
    where
        S: FingerprintSet<F> + ?Sized,
    {
        let mut duplicate_authorization = None;
        let admission = {
            let mut authorize_duplicate = |policy| {
                let authorization =
                    self.authorize_duplicate_with_proof_witness(policy, proof_witness_accepted()?)?;
                duplicate_authorization =
                    Some(self.duplicate_authorization_evidence(policy, authorization));
                Ok(authorization)
            };
            fingerprint_set.admit_fingerprint_with_duplicate_authorization(
                fingerprint,
                &self.plan.dedup,
                &mut authorize_duplicate,
            )?
        };
        Ok(PreparedFingerprintAdmissionOutcome::new(
            admission,
            duplicate_authorization,
        ))
    }

    /// Admit a fingerprint and authorize duplicates with validation receipts.
    pub fn admit_fingerprint_with_validation_receipt_evidence<F, S>(
        &self,
        fingerprint_set: &S,
        fingerprint: F,
        mut validation_receipt: impl FnMut() -> Result<ValidationReceipt, StorageFault>,
    ) -> Result<PreparedFingerprintAdmissionOutcome, StorageFault>
    where
        S: FingerprintSet<F> + ?Sized,
    {
        let mut duplicate_authorization = None;
        let admission = {
            let mut authorize_duplicate = |policy| {
                let receipt = validation_receipt()?;
                let authorization =
                    self.authorize_duplicate_with_validation_receipt(policy, &receipt)?;
                duplicate_authorization =
                    Some(self.duplicate_authorization_evidence(policy, authorization));
                Ok(authorization)
            };
            fingerprint_set.admit_fingerprint_with_duplicate_authorization(
                fingerprint,
                &self.plan.dedup,
                &mut authorize_duplicate,
            )?
        };
        Ok(PreparedFingerprintAdmissionOutcome::new(
            admission,
            duplicate_authorization,
        ))
    }

    /// Enforce this validated handle on a batch summary with validation receipt
    /// authorization evidence.
    pub fn enforce_batch_duplicate_with_validation_receipt_evidence(
        &self,
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        mut validation_receipt: impl FnMut() -> Result<ValidationReceipt, StorageFault>,
    ) -> Result<PreparedFingerprintBatchAdmissionOutcome, StorageFault> {
        if fault_present || inserted_count >= attempted {
            return Ok(PreparedFingerprintBatchAdmissionOutcome::new(
                attempted,
                inserted_count,
                fault_present,
                None,
            ));
        }
        let outcome = self.enforce_duplicate_with_validation_receipt_evidence(
            FingerprintAdmission::Duplicate,
            &mut validation_receipt,
        )?;
        Ok(PreparedFingerprintBatchAdmissionOutcome::new(
            attempted,
            inserted_count,
            fault_present,
            outcome.duplicate_authorization,
        ))
    }

    /// Enforce this validated handle on an already computed admission decision
    /// with validation receipt authorization evidence.
    pub fn enforce_duplicate_with_validation_receipt_evidence(
        &self,
        admission: FingerprintAdmission,
        mut validation_receipt: impl FnMut() -> Result<ValidationReceipt, StorageFault>,
    ) -> Result<PreparedFingerprintAdmissionOutcome, StorageFault> {
        let mut duplicate_authorization = None;
        let admission = {
            let mut authorize_duplicate = |policy| {
                let receipt = validation_receipt()?;
                let authorization =
                    self.authorize_duplicate_with_validation_receipt(policy, &receipt)?;
                duplicate_authorization =
                    Some(self.duplicate_authorization_evidence(policy, authorization));
                Ok(authorization)
            };
            admission.enforce_shared_duplicate_authorization(
                &self.plan.dedup,
                &mut authorize_duplicate,
            )?
        };
        Ok(PreparedFingerprintAdmissionOutcome::new(
            admission,
            duplicate_authorization,
        ))
    }

    fn duplicate_authorization_evidence(
        &self,
        observed_collision_policy: SharedCollisionPolicy,
        authorization: SharedDuplicateAuthorization,
    ) -> PreparedFingerprintDuplicateAuthorizationEvidence {
        PreparedFingerprintDuplicateAuthorizationEvidence {
            observed_collision_policy,
            authorization,
            payload_witness: self.plan.payload_witness,
        }
    }

    /// Whether runtime duplicate evidence satisfies this validated handle.
    #[must_use]
    pub fn authorizes_duplicate(&self, authorization: SharedDuplicateAuthorization) -> bool {
        self.plan.authorizes_duplicate(authorization)
    }

    /// Fingerprint descriptor consumable by prepared-program descriptors.
    #[must_use]
    pub fn prepared_fingerprint_descriptor(&self) -> PreparedFingerprintDescriptor {
        self.plan.prepared_fingerprint_descriptor()
    }
}

impl TryFrom<PreparedFingerprintAdmissionPlan> for ValidatedPreparedFingerprintAdmissionPlan {
    type Error = SharedFingerprintIdentityRejection;

    fn try_from(plan: PreparedFingerprintAdmissionPlan) -> Result<Self, Self::Error> {
        Self::new(plan)
    }
}

impl TryFrom<&PreparedFingerprintAdmissionPlan> for ValidatedPreparedFingerprintAdmissionPlan {
    type Error = SharedFingerprintIdentityRejection;

    fn try_from(plan: &PreparedFingerprintAdmissionPlan) -> Result<Self, Self::Error> {
        Self::new(plan.clone())
    }
}

impl PreparedFingerprintAdmissionPlan {
    /// Stable row kind for prepared fingerprint admission evidence.
    pub const EVIDENCE_ROW_KIND: &'static str = PREPARED_FINGERPRINT_ADMISSION_BACKEND;
    /// Stable schema label for prepared fingerprint admission evidence.
    pub const EVIDENCE_SCHEMA: &'static str = PREPARED_FINGERPRINT_ADMISSION_SCHEMA;
    /// Stable schema version for prepared fingerprint admission evidence.
    pub const EVIDENCE_SCHEMA_VERSION: u32 = PREPARED_FINGERPRINT_ADMISSION_SCHEMA_VERSION;
    /// Fields every prepared fingerprint admission evidence row publishes.
    pub const EVIDENCE_REQUIRED_FIELDS: &'static [&'static str] =
        PREPARED_FINGERPRINT_ADMISSION_REQUIRED_FIELDS;

    /// Render this plan as one shared, frontend-neutral admission evidence row.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let validation = self.validate_runtime_admission();
        let (admission_status, reason_code) = match validation {
            Ok(()) => ("accepted", "accepted"),
            Err(ref rejection) => ("rejected", rejection.reason_code),
        };
        let compatible_frontend_families = prepared_admission_compatible_frontend_families(self);
        let default_consumers =
            prepared_admission_default_consumers(self, &compatible_frontend_families);
        let remaining_compatible_frontend_families = prepared_admission_remaining_families(
            &compatible_frontend_families,
            &default_consumers,
        );

        format!(
            "{} {} schema={} schema_version={} source_kind={} frontend_family={} shared_engine_component={} plan_id={} payload_kind={} storage_kind={} lane_kind={} candidate_key={} prepared_program_identity={} prepared_lane_identity={} payload_witness={} dedup_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} collision_policy={} duplicate_authorization={} admission_status={} reason_code={} fail_closed=true compatible_frontend_families={} default_consumers={} remaining_compatible_frontend_families={} blockers={}",
            prepared_admission_evidence_value(scope),
            Self::EVIDENCE_ROW_KIND,
            Self::EVIDENCE_SCHEMA,
            Self::EVIDENCE_SCHEMA_VERSION,
            self.source_kind.code(),
            self.source_kind.frontend_family_code(),
            Self::EVIDENCE_ROW_KIND,
            prepared_admission_evidence_value(&self.id),
            self.payload_kind.code(),
            self.storage_kind.code(),
            self.lane.code(),
            prepared_admission_evidence_optional(self.candidate_key.as_deref()),
            prepared_admission_evidence_optional(self.prepared_program_identity.as_deref()),
            prepared_admission_evidence_optional(self.prepared_lane_identity.as_deref()),
            self.payload_witness.code(),
            prepared_admission_evidence_value(&self.dedup.dedup_identity()),
            prepared_admission_evidence_value(&self.dedup.storage_policy_identity()),
            prepared_admission_evidence_value(
                &self.dedup.fingerprint.fingerprint_policy_identity()
            ),
            prepared_admission_evidence_value(&self.dedup.fingerprint.fingerprint_identity()),
            self.dedup.collision_policy.code(),
            self.duplicate_authorization.code(),
            admission_status,
            reason_code,
            prepared_admission_evidence_frontend_families(&compatible_frontend_families),
            prepared_admission_evidence_frontend_families(&default_consumers),
            prepared_admission_evidence_frontend_families(&remaining_compatible_frontend_families),
            PREPARED_FINGERPRINT_ADMISSION_FUTURE_IMPORTER_BLOCKER,
        )
    }

    /// Validate one prepared fingerprint admission evidence row.
    pub fn validate_evidence_row(row: &str) -> Result<(), String> {
        validate_prepared_fingerprint_admission_evidence_row(row)
    }

    /// Build a prepared fingerprint admission plan from explicit metadata.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        source_kind: CheckerSourceKind,
        payload_kind: PreparedProgramPayloadKind,
        storage_kind: PreparedStorageKind,
        lane: SetupTraceLaneKind,
        dedup: SharedDedupIdentity,
        duplicate_authorization: SharedDuplicateAuthorization,
        payload_witness: PreparedFingerprintPayloadWitnessKind,
    ) -> Self {
        let identities = dedup.identity_fields();
        Self {
            id: id.into(),
            source_kind,
            payload_kind,
            storage_kind,
            lane,
            dedup,
            duplicate_authorization,
            payload_witness,
            prepared_program_identity: None,
            prepared_lane_identity: None,
            candidate_key: None,
            identities,
        }
    }

    /// Attach prepared program metadata, preserving lane-specific identities
    /// when present and filling the rest from the program and dedup identity.
    #[must_use]
    pub fn with_prepared_program(mut self, program: &PreparedCheckerProgram) -> Self {
        self.source_kind = program.source_kind;
        self.payload_kind = program.payload_kind;
        self.storage_kind = program.storage_kind;
        self.prepared_program_identity = non_empty_string(program.identity.clone());
        self.identities = program
            .effective_identity_fields()
            .merged_with_fallback(&self.identities);
        self
    }

    /// Attach prepared candidate lane metadata using the prepared-program merge
    /// rules for source, lane, candidate key, and identity precedence.
    #[must_use]
    pub fn with_prepared_candidate_lane(
        mut self,
        program: &PreparedCheckerProgram,
        lane: &PreparedCandidateLaneDescriptor,
    ) -> Self {
        self = self.with_prepared_program(program);
        self.lane = lane.lane;
        self.candidate_key.clone_from(&lane.candidate_key);
        self.prepared_lane_identity = non_empty_string(lane.id.clone());
        self.identities = program
            .effective_candidate_lane_identity_fields(lane)
            .merged_with_fallback(&self.identities);
        self
    }

    /// Validate the shared runtime admission contract.
    pub fn validate(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        self.validate_runtime_admission()
    }

    /// Validate and bind this descriptor as a setup-once runtime handle.
    pub fn validated_runtime_handle(
        &self,
    ) -> Result<ValidatedPreparedFingerprintAdmissionPlan, SharedFingerprintIdentityRejection> {
        ValidatedPreparedFingerprintAdmissionPlan::new(self.clone())
    }

    /// Consume, validate, and bind this descriptor as a setup-once runtime
    /// handle.
    pub fn into_validated_runtime_handle(
        self,
    ) -> Result<ValidatedPreparedFingerprintAdmissionPlan, SharedFingerprintIdentityRejection> {
        ValidatedPreparedFingerprintAdmissionPlan::new(self)
    }

    /// Validate the prepared-program, storage, lane, frontend-family, and
    /// collision contract before a runtime fingerprint set is allowed to
    /// suppress duplicates under this plan.
    pub fn validate_runtime_admission(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        if self.id.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_EMPTY_PLAN_ID,
                "prepared fingerprint admission plan id must not be empty",
            ));
        }
        self.dedup.require_frontend_reusable_admission()?;

        let Some(frontend_family) = self.source_kind.adoption_frontend_family() else {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_UNKNOWN_SOURCE_KIND,
                "prepared fingerprint admission requires a known source frontend family",
            ));
        };
        if !self
            .dedup
            .fingerprint
            .reusable_frontend_families()
            .contains(&frontend_family)
        {
            return Err(SharedFingerprintIdentityRejection::new(
                "frontend_family_not_reusable",
                format!(
                    "frontend family {} is not registered for fingerprint identity {}",
                    frontend_family.code(),
                    self.dedup.fingerprint.fingerprint_identity()
                ),
            ));
        }
        if self.payload_kind.source_kind() != self.source_kind {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_SOURCE_PAYLOAD_MISMATCH,
                format!(
                    "source kind {} does not match prepared payload kind {}",
                    self.source_kind.code(),
                    self.payload_kind.code()
                ),
            ));
        }
        if self.storage_kind == PreparedStorageKind::Unknown {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_UNKNOWN_STORAGE_KIND,
                "prepared fingerprint admission requires a concrete storage ABI",
            ));
        }
        if self.dedup.storage == SharedDedupStorageKind::EvidenceOnly {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_EVIDENCE_ONLY_STORAGE,
                "runtime fingerprint admission requires active dedup storage, not evidence-only policy",
            ));
        }
        if self.lane == SetupTraceLaneKind::Unknown {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_UNKNOWN_LANE_KIND,
                "prepared fingerprint admission requires a concrete setup trace lane",
            ));
        }
        if self.lane != self.dedup.lane {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_LANE_MISMATCH,
                format!(
                    "prepared lane {} does not match dedup lane {}",
                    self.lane.code(),
                    self.dedup.lane.code()
                ),
            ));
        }
        self.validate_identity_binding()?;
        self.validate_collision_policy_binding(Some(self.dedup.collision_policy))?;
        Ok(())
    }

    /// Validate that the runtime collision policy observed by storage is the
    /// same fail-closed policy bound into this prepared admission plan.
    pub fn validate_collision_policy_binding(
        &self,
        observed_collision_policy: Option<SharedCollisionPolicy>,
    ) -> Result<SharedCollisionPolicy, SharedFingerprintIdentityRejection> {
        self.dedup.require_fail_closed()?;
        let observed_collision_policy = observed_collision_policy.ok_or_else(|| {
            SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_MISSING_COLLISION_POLICY,
                "runtime fingerprint admission requires an explicit collision policy",
            )
        })?;
        if observed_collision_policy != self.dedup.collision_policy {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_COLLISION_POLICY_MISMATCH,
                format!(
                    "runtime collision policy {} does not match prepared policy {}",
                    observed_collision_policy.code(),
                    self.dedup.collision_policy.code()
                ),
            ));
        }
        if !self
            .dedup
            .collision_policy
            .authorizes_duplicate(self.duplicate_authorization)
        {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_DUPLICATE_AUTHORIZATION_MISMATCH,
                format!(
                    "duplicate authorization {} does not satisfy collision policy {}",
                    self.duplicate_authorization.code(),
                    self.dedup.collision_policy.code()
                ),
            ));
        }
        Ok(observed_collision_policy)
    }

    /// Admit a fingerprint through this prepared plan, using canonical payload
    /// comparison as the only duplicate authorization.
    pub fn admit_fingerprint_with_canonical_payload_comparison<F, S>(
        &self,
        fingerprint_set: &S,
        fingerprint: F,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault>
    where
        S: FingerprintSet<F> + ?Sized,
    {
        let handle = self
            .validated_runtime_handle()
            .map_err(|rejection| self.admission_fault(rejection))?;
        handle.admit_fingerprint_with_canonical_payload_comparison(
            fingerprint_set,
            fingerprint,
            &mut canonical_payloads_equal,
        )
    }

    /// Enforce this prepared plan on an already computed admission decision.
    pub fn enforce_duplicate_with_canonical_payload_comparison(
        &self,
        admission: FingerprintAdmission,
        mut canonical_payloads_equal: impl FnMut() -> Result<bool, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault> {
        let handle = self
            .validated_runtime_handle()
            .map_err(|rejection| self.admission_fault(rejection))?;
        handle.enforce_duplicate_with_canonical_payload_comparison(
            admission,
            &mut canonical_payloads_equal,
        )
    }

    /// Convert a canonical payload comparison result into the duplicate
    /// authorization required by this plan.
    pub fn authorize_duplicate_with_canonical_payload_equality(
        &self,
        observed_collision_policy: SharedCollisionPolicy,
        canonical_payloads_equal: bool,
    ) -> Result<SharedDuplicateAuthorization, StorageFault> {
        let handle = self
            .validated_runtime_handle()
            .map_err(|rejection| self.admission_fault(rejection))?;
        handle.authorize_duplicate_with_canonical_payload_equality(
            observed_collision_policy,
            canonical_payloads_equal,
        )
    }

    /// Convert proof/witness validation into the duplicate authorization
    /// required by proof/certificate-backed plans.
    pub fn authorize_duplicate_with_proof_witness(
        &self,
        observed_collision_policy: SharedCollisionPolicy,
        proof_witness_accepted: bool,
    ) -> Result<SharedDuplicateAuthorization, StorageFault> {
        let handle = self
            .validated_runtime_handle()
            .map_err(|rejection| self.admission_fault(rejection))?;
        handle.authorize_duplicate_with_proof_witness(
            observed_collision_policy,
            proof_witness_accepted,
        )
    }

    /// Whether runtime duplicate evidence satisfies this prepared plan.
    #[must_use]
    pub fn authorizes_duplicate(&self, authorization: SharedDuplicateAuthorization) -> bool {
        authorization == self.duplicate_authorization
            && self
                .dedup
                .collision_policy
                .authorizes_duplicate(authorization)
    }

    /// Fingerprint descriptor consumable by prepared-program descriptors.
    #[must_use]
    pub fn prepared_fingerprint_descriptor(&self) -> PreparedFingerprintDescriptor {
        self.dedup.prepared_fingerprint_descriptor()
    }

    fn validate_identity_binding(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        let storage_policy_identity = self.dedup.storage_policy_identity();
        if self
            .identities
            .storage_policy_identity
            .as_deref()
            .is_some_and(|identity| identity != storage_policy_identity.as_str())
        {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_STORAGE_POLICY_IDENTITY_MISMATCH,
                format!(
                    "prepared storage identity does not match dedup storage policy {}",
                    storage_policy_identity
                ),
            ));
        }

        let fingerprint_policy_identity = self.dedup.fingerprint.fingerprint_policy_identity();
        if self
            .identities
            .fingerprint_policy_identity
            .as_deref()
            .is_some_and(|identity| identity != fingerprint_policy_identity.as_str())
        {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_FINGERPRINT_POLICY_IDENTITY_MISMATCH,
                format!(
                    "prepared fingerprint policy identity does not match {}",
                    fingerprint_policy_identity
                ),
            ));
        }

        let fingerprint_identity = self.dedup.fingerprint.fingerprint_identity();
        if self
            .identities
            .fingerprint_identity
            .as_deref()
            .is_some_and(|identity| identity != fingerprint_identity.as_str())
        {
            return Err(SharedFingerprintIdentityRejection::new(
                PREPARED_FINGERPRINT_REJECTION_FINGERPRINT_IDENTITY_MISMATCH,
                format!(
                    "prepared fingerprint identity does not match {}",
                    fingerprint_identity
                ),
            ));
        }
        Ok(())
    }

    fn admission_fault(&self, rejection: SharedFingerprintIdentityRejection) -> StorageFault {
        StorageFault::new(
            PREPARED_FINGERPRINT_ADMISSION_BACKEND,
            PREPARED_FINGERPRINT_ADMISSION_OPERATION,
            format!(
                "status_code=rejected reason_code={} fail_closed=true plan_id={} source_kind={} frontend_family={} payload_kind={} storage_kind={} dedup_storage_kind={} lane_kind={} collision_policy={} duplicate_authorization={} payload_witness={} dedup_identity={} detail={}",
                rejection.reason_code,
                self.id,
                self.source_kind.code(),
                self.source_kind.frontend_family_code(),
                self.payload_kind.code(),
                self.storage_kind.code(),
                self.dedup.storage.code(),
                self.lane.code(),
                self.dedup.collision_policy.code(),
                self.duplicate_authorization.code(),
                self.payload_witness.code(),
                self.dedup.dedup_identity(),
                rejection.detail,
            ),
        )
    }

    /// TLA array state slots admitted with TLC-compatible FP64.
    #[must_use]
    pub fn tla_array_fp64(
        id: impl Into<String>,
        canonicalization_version: impl Into<String>,
    ) -> Self {
        let fingerprint = SharedFingerprintIdentity::new(
            "tla array fp64",
            SharedFingerprintAlgorithm::TlaFingerprint64,
            SharedFingerprintValueKind::StateVector,
            canonicalization_version,
            "tla-array-state",
            64,
        )
        .with_canonical_domain("tla-array-state", "fp64");
        let dedup = SharedDedupIdentity::new(
            "tla array fp64 dedup",
            fingerprint,
            SharedDedupScope::StateSpace,
            SharedDedupStorageKind::ShardedInMemory,
            SetupTraceLaneKind::ExplicitState,
        )
        .with_collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality);
        Self::new(
            id,
            CheckerSourceKind::Tla,
            PreparedProgramPayloadKind::Tla,
            PreparedStorageKind::TlaStateSlots,
            SetupTraceLaneKind::ExplicitState,
            dedup,
            SharedDuplicateAuthorization::CanonicalPayloadEquality,
            PreparedFingerprintPayloadWitnessKind::TlaArrayFp64,
        )
    }

    /// Compiled/native flat state bytes admitted with xxh3.
    #[must_use]
    pub fn compiled_flat_xxh3(
        id: impl Into<String>,
        source_kind: CheckerSourceKind,
        payload_kind: PreparedProgramPayloadKind,
        canonicalization_version: impl Into<String>,
    ) -> Self {
        let fingerprint = SharedFingerprintIdentity::new(
            "compiled flat xxh3",
            SharedFingerprintAlgorithm::Xxh3U64,
            SharedFingerprintValueKind::State,
            canonicalization_version,
            "compiled-flat-state",
            64,
        )
        .with_canonical_domain("compiled-flat-state", "xxh3");
        let dedup = SharedDedupIdentity::new(
            "compiled flat xxh3 dedup",
            fingerprint,
            SharedDedupScope::FrontierBatch,
            SharedDedupStorageKind::Cas,
            SetupTraceLaneKind::Native,
        )
        .with_collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality);
        Self::new(
            id,
            source_kind,
            payload_kind,
            PreparedStorageKind::TlaStateSlots,
            SetupTraceLaneKind::Native,
            dedup,
            SharedDuplicateAuthorization::CanonicalPayloadEquality,
            PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3,
        )
    }

    /// Petri marking vector admitted with CAS-backed dedup storage.
    #[must_use]
    pub fn petri_marking_cas(
        id: impl Into<String>,
        canonicalization_version: impl Into<String>,
    ) -> Self {
        let fingerprint = SharedFingerprintIdentity::new(
            "petri marking cas",
            SharedFingerprintAlgorithm::CanonicalBytesSha256,
            SharedFingerprintValueKind::MarkingVector,
            canonicalization_version,
            "petri-marking",
            128,
        )
        .with_canonical_domain("petri-marking-vector", "sha256-truncated-u128");
        let dedup = SharedDedupIdentity::new(
            "petri marking cas dedup",
            fingerprint,
            SharedDedupScope::StateSpace,
            SharedDedupStorageKind::Cas,
            SetupTraceLaneKind::ExplicitState,
        )
        .with_collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality);
        Self::new(
            id,
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
            SetupTraceLaneKind::ExplicitState,
            dedup,
            SharedDuplicateAuthorization::CanonicalPayloadEquality,
            PreparedFingerprintPayloadWitnessKind::PetriMarkingCas,
        )
    }

    /// Hardware/register vector canonical bytes admitted through the shared
    /// fingerprint runtime. Source/payload stay explicit so AIGER, BTOR2, VMT,
    /// and future registered importers can bind the same storage contract
    /// without adding frontend-owned policies.
    #[must_use]
    pub fn register_vector_canonical(
        id: impl Into<String>,
        source_kind: CheckerSourceKind,
        payload_kind: PreparedProgramPayloadKind,
        canonicalization_version: impl Into<String>,
    ) -> Self {
        let fingerprint = SharedFingerprintIdentity::new(
            "register vector canonical",
            SharedFingerprintAlgorithm::CanonicalBytesSha256,
            SharedFingerprintValueKind::RegisterVector,
            canonicalization_version,
            "register-vector",
            128,
        )
        .with_canonical_domain("hardware-register-vector", "sha256-truncated-u128");
        let dedup = SharedDedupIdentity::new(
            "register vector canonical dedup",
            fingerprint,
            SharedDedupScope::StateSpace,
            SharedDedupStorageKind::Cas,
            SetupTraceLaneKind::Fingerprint,
        )
        .with_collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality);
        Self::new(
            id,
            source_kind,
            payload_kind,
            PreparedStorageKind::HardwareRegisters,
            SetupTraceLaneKind::Fingerprint,
            dedup,
            SharedDuplicateAuthorization::CanonicalPayloadEquality,
            PreparedFingerprintPayloadWitnessKind::RegisterVectorCanonical,
        )
    }
}

fn non_empty_string(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn prepared_admission_compatible_frontend_families(
    plan: &PreparedFingerprintAdmissionPlan,
) -> Vec<SharedEngineFrontendFamily> {
    let allowed = plan.dedup.fingerprint.reusable_frontend_families();
    let mut families = Vec::new();
    let mut push = |family| {
        if allowed.contains(&family) && !families.contains(&family) {
            families.push(family);
        }
    };

    match plan.payload_witness {
        PreparedFingerprintPayloadWitnessKind::TlaArrayFp64
        | PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3 => {
            push(SharedEngineFrontendFamily::TlaPlus);
            push(SharedEngineFrontendFamily::Quint);
        }
        PreparedFingerprintPayloadWitnessKind::PetriMarkingCas => {
            push(SharedEngineFrontendFamily::MccPetri);
        }
        PreparedFingerprintPayloadWitnessKind::RegisterVectorCanonical => {
            push(SharedEngineFrontendFamily::Aiger);
            push(SharedEngineFrontendFamily::Btor2);
            push(SharedEngineFrontendFamily::VmtTransitionSystem);
            push(SharedEngineFrontendFamily::AYAnalytical);
            push(SharedEngineFrontendFamily::WitnessReplay);
        }
        PreparedFingerprintPayloadWitnessKind::ValidationReceiptProof => {
            push(SharedEngineFrontendFamily::AYAnalytical);
            push(SharedEngineFrontendFamily::WitnessReplay);
        }
    }
    if let Some(source_family) = plan.source_kind.adoption_frontend_family() {
        push(source_family);
    }
    families
}

fn prepared_admission_default_consumers(
    plan: &PreparedFingerprintAdmissionPlan,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Vec<SharedEngineFrontendFamily> {
    let mut consumers = Vec::new();
    let mut push = |family| {
        if compatible_frontend_families.contains(&family) && !consumers.contains(&family) {
            consumers.push(family);
        }
    };

    match plan.payload_witness {
        PreparedFingerprintPayloadWitnessKind::RegisterVectorCanonical => {
            push(SharedEngineFrontendFamily::Aiger);
            push(SharedEngineFrontendFamily::Btor2);
        }
        PreparedFingerprintPayloadWitnessKind::ValidationReceiptProof => {
            push(SharedEngineFrontendFamily::AYAnalytical);
            push(SharedEngineFrontendFamily::WitnessReplay);
        }
        PreparedFingerprintPayloadWitnessKind::TlaArrayFp64
        | PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3
        | PreparedFingerprintPayloadWitnessKind::PetriMarkingCas => {}
    }
    if let Some(source_family) = plan.source_kind.adoption_frontend_family() {
        push(source_family);
    }
    if consumers.is_empty() {
        consumers.extend_from_slice(compatible_frontend_families);
    }
    consumers
}

fn prepared_admission_remaining_families(
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    default_consumers: &[SharedEngineFrontendFamily],
) -> Vec<SharedEngineFrontendFamily> {
    compatible_frontend_families
        .iter()
        .copied()
        .filter(|family| !default_consumers.contains(family))
        .collect()
}

fn prepared_admission_evidence_frontend_families(
    families: &[SharedEngineFrontendFamily],
) -> String {
    if families.is_empty() {
        return "none".to_string();
    }
    families
        .iter()
        .map(|family| family.code())
        .collect::<Vec<_>>()
        .join(",")
}

fn prepared_admission_evidence_value(value: &str) -> String {
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

fn prepared_admission_evidence_optional(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .map(prepared_admission_evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn validate_prepared_fingerprint_admission_evidence_row(row: &str) -> Result<(), String> {
    validate_prepared_admission_row_kind(row)?;
    require_prepared_admission_fields(row, PREPARED_FINGERPRINT_ADMISSION_REQUIRED_FIELDS)?;
    require_prepared_admission_field_value(row, "schema", PREPARED_FINGERPRINT_ADMISSION_SCHEMA)?;
    require_prepared_admission_field_value(
        row,
        "schema_version",
        &PREPARED_FINGERPRINT_ADMISSION_SCHEMA_VERSION.to_string(),
    )?;
    require_prepared_admission_field_value(
        row,
        "shared_engine_component",
        PREPARED_FINGERPRINT_ADMISSION_BACKEND,
    )?;
    validate_prepared_admission_source_and_payload(row)?;
    validate_prepared_admission_storage_kind(row)?;
    validate_prepared_admission_lane_kind(row)?;
    validate_prepared_admission_payload_witness(row)?;
    validate_prepared_admission_collision_policy(row)?;
    validate_prepared_admission_duplicate_authorization(row)?;
    validate_prepared_admission_status(row)?;
    require_prepared_admission_field_value(row, "fail_closed", "true")?;
    validate_prepared_admission_frontend_family_lists(row)?;
    for field in [
        "plan_id",
        "dedup_identity",
        "storage_policy_identity",
        "fingerprint_policy_identity",
        "fingerprint_identity",
    ] {
        require_prepared_admission_non_none_field(row, field)?;
    }
    Ok(())
}

fn validate_prepared_admission_row_kind(row: &str) -> Result<(), String> {
    let mut tokens = row.split_whitespace();
    tokens
        .next()
        .ok_or_else(|| "missing evidence scope".to_string())?;
    let kind = tokens
        .next()
        .ok_or_else(|| "missing prepared fingerprint admission row kind".to_string())?;
    if kind != PREPARED_FINGERPRINT_ADMISSION_BACKEND {
        return Err(format!(
            "wrong prepared fingerprint admission row kind: expected {}, got {kind}",
            PREPARED_FINGERPRINT_ADMISSION_BACKEND
        ));
    }
    Ok(())
}

fn require_prepared_admission_fields(row: &str, fields: &[&'static str]) -> Result<(), String> {
    for field in fields {
        require_prepared_admission_field(row, field)?;
    }
    Ok(())
}

fn require_prepared_admission_field<'a>(
    row: &'a str,
    field: &'static str,
) -> Result<&'a str, String> {
    prepared_admission_evidence_field(row, field).ok_or_else(|| format!("missing field {field}"))
}

fn require_prepared_admission_non_none_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_admission_field(row, field)?;
    if value == "none" {
        return Err(format!("field {field} must not be none"));
    }
    Ok(())
}

fn require_prepared_admission_field_value(
    row: &str,
    field: &'static str,
    expected: &str,
) -> Result<(), String> {
    let value = require_prepared_admission_field(row, field)?;
    if value != expected {
        return Err(format!("field {field} expected {expected}, got {value}"));
    }
    Ok(())
}

fn validate_prepared_admission_source_and_payload(row: &str) -> Result<(), String> {
    let source_kind = require_prepared_admission_field(row, "source_kind")?;
    let payload_kind = require_prepared_admission_field(row, "payload_kind")?;
    let frontend_family = require_prepared_admission_field(row, "frontend_family")?;
    if !is_prepared_admission_source_kind_code(source_kind) {
        return Err(format!(
            "field source_kind has unknown source kind: {source_kind}"
        ));
    }
    if !is_prepared_admission_payload_kind_code(payload_kind) {
        return Err(format!(
            "field payload_kind has unknown payload kind: {payload_kind}"
        ));
    }
    let expected_source_kind = prepared_admission_source_kind_for_payload(payload_kind)
        .ok_or_else(|| format!("field payload_kind has unknown payload kind: {payload_kind}"))?;
    if source_kind != expected_source_kind {
        return Err(format!(
            "payload_kind {payload_kind} requires source_kind {expected_source_kind}, got {source_kind}"
        ));
    }
    let expected_frontend_family = prepared_admission_frontend_family_for_source(source_kind)
        .ok_or_else(|| format!("field source_kind has unknown source kind: {source_kind}"))?;
    if frontend_family != expected_frontend_family {
        return Err(format!(
            "source_kind {source_kind} requires frontend_family {expected_frontend_family}, got {frontend_family}"
        ));
    }
    Ok(())
}

fn validate_prepared_admission_storage_kind(row: &str) -> Result<(), String> {
    let value = require_prepared_admission_field(row, "storage_kind")?;
    if is_prepared_admission_storage_kind_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field storage_kind has unknown storage kind: {value}"
        ))
    }
}

fn validate_prepared_admission_lane_kind(row: &str) -> Result<(), String> {
    let value = require_prepared_admission_field(row, "lane_kind")?;
    if is_prepared_admission_lane_kind_code(value) {
        Ok(())
    } else {
        Err(format!("field lane_kind has unknown lane kind: {value}"))
    }
}

fn validate_prepared_admission_payload_witness(row: &str) -> Result<(), String> {
    let value = require_prepared_admission_field(row, "payload_witness")?;
    if is_prepared_admission_payload_witness_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field payload_witness has unknown witness kind: {value}"
        ))
    }
}

fn validate_prepared_admission_collision_policy(row: &str) -> Result<(), String> {
    let value = require_prepared_admission_field(row, "collision_policy")?;
    if is_prepared_admission_collision_policy_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field collision_policy has unknown collision policy: {value}"
        ))
    }
}

fn validate_prepared_admission_duplicate_authorization(row: &str) -> Result<(), String> {
    let value = require_prepared_admission_field(row, "duplicate_authorization")?;
    if is_prepared_admission_duplicate_authorization_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field duplicate_authorization has unknown duplicate authorization: {value}"
        ))
    }
}

fn validate_prepared_admission_status(row: &str) -> Result<(), String> {
    let status = require_prepared_admission_field(row, "admission_status")?;
    let reason_code = require_prepared_admission_field(row, "reason_code")?;
    match status {
        "accepted" if reason_code == "accepted" => Ok(()),
        "accepted" => Err(format!(
            "accepted admission row must use reason_code accepted, got {reason_code}"
        )),
        "rejected" if reason_code != "accepted" && reason_code != "none" => Ok(()),
        "rejected" => Err("rejected admission row requires a rejection reason_code".to_string()),
        value => Err(format!(
            "field admission_status has unknown status: {value}"
        )),
    }
}

fn validate_prepared_admission_frontend_family_lists(row: &str) -> Result<(), String> {
    let compatible =
        validate_prepared_admission_frontend_family_list(row, "compatible_frontend_families")?;
    let default_consumers =
        validate_prepared_admission_frontend_family_list(row, "default_consumers")?;
    let remaining = validate_prepared_admission_frontend_family_list(
        row,
        "remaining_compatible_frontend_families",
    )?;
    for family in default_consumers.iter().chain(remaining.iter()) {
        if !compatible.contains(family) {
            return Err(format!(
                "frontend family {} is not listed in compatible_frontend_families",
                family.code()
            ));
        }
    }
    for family in &default_consumers {
        if remaining.contains(family) {
            return Err(format!(
                "frontend family {} cannot be both default and remaining",
                family.code()
            ));
        }
    }
    Ok(())
}

fn validate_prepared_admission_frontend_family_list(
    row: &str,
    field: &'static str,
) -> Result<Vec<SharedEngineFrontendFamily>, String> {
    let value = require_prepared_admission_field(row, field)?;
    if value == "none" {
        return Ok(Vec::new());
    }
    let mut families = Vec::new();
    for code in value.split(',') {
        if code.is_empty() {
            return Err(format!("field {field} contains an empty frontend family"));
        }
        let family = SharedEngineFrontendFamily::from_code(code)
            .ok_or_else(|| format!("field {field} has unknown frontend family: {code}"))?;
        if families.contains(&family) {
            return Err(format!(
                "field {field} contains duplicate frontend family: {code}"
            ));
        }
        families.push(family);
    }
    Ok(families)
}

fn is_prepared_admission_source_kind_code(value: &str) -> bool {
    matches!(
        value,
        "tla"
            | "quint"
            | "mcc_petri"
            | "aiger"
            | "btor2"
            | "vmt_interchange"
            | "ay_only"
            | "witness_replay"
            | "unknown"
    )
}

fn is_prepared_admission_payload_kind_code(value: &str) -> bool {
    matches!(
        value,
        "tla"
            | "quint"
            | "mcc_petri"
            | "aiger"
            | "btor2"
            | "vmt_interchange"
            | "ay_only"
            | "witness_replay"
    )
}

fn is_prepared_admission_storage_kind_code(value: &str) -> bool {
    matches!(
        value,
        "tla_state_slots"
            | "petri_marking"
            | "hardware_registers"
            | "smt_variables"
            | "witness_steps"
            | "unknown"
    )
}

fn is_prepared_admission_lane_kind_code(value: &str) -> bool {
    matches!(
        value,
        "frontend"
            | "explicit_state"
            | "native"
            | "ay"
            | "analytical"
            | "replay"
            | "fingerprint"
            | "unknown"
    )
}

fn is_prepared_admission_payload_witness_code(value: &str) -> bool {
    matches!(
        value,
        "tla_array_fp64"
            | "compiled_flat_xxh3"
            | "petri_marking_cas"
            | "register_vector_canonical"
            | "validation_receipt_proof"
    )
}

fn is_prepared_admission_collision_policy_code(value: &str) -> bool {
    matches!(
        value,
        "unchecked"
            | "canonical_payload_equality"
            | "proof_witness_required"
            | "reject_on_collision"
    )
}

fn is_prepared_admission_duplicate_authorization_code(value: &str) -> bool {
    matches!(
        value,
        "unconfirmed" | "canonical_payload_equality" | "proof_witness"
    )
}

fn prepared_admission_source_kind_for_payload(payload_kind: &str) -> Option<&'static str> {
    match payload_kind {
        "tla" => Some("tla"),
        "quint" => Some("quint"),
        "mcc_petri" => Some("mcc_petri"),
        "aiger" => Some("aiger"),
        "btor2" => Some("btor2"),
        "vmt_interchange" => Some("vmt_interchange"),
        "ay_only" => Some("ay_only"),
        "witness_replay" => Some("witness_replay"),
        _ => None,
    }
}

fn prepared_admission_frontend_family_for_source(source_kind: &str) -> Option<&'static str> {
    match source_kind {
        "tla" => Some("tla_plus"),
        "quint" => Some("quint"),
        "mcc_petri" => Some("mcc_petri"),
        "aiger" => Some("aiger"),
        "btor2" => Some("btor2"),
        "vmt_interchange" => Some("vmt_transition_system"),
        "ay_only" => Some("ay_analytical"),
        "witness_replay" => Some("witness_replay"),
        "unknown" => Some("unknown"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        storage::InMemoryFingerprintSet, SharedEngineFrontendFamily, ValidationReceipt,
        ValidationReceiptArtifactKind, ValidationReceiptValidatorKind,
    };

    fn validation_receipt_admission_plan() -> PreparedFingerprintAdmissionPlan {
        let fingerprint = SharedFingerprintIdentity::new(
            "ay validation receipt",
            SharedFingerprintAlgorithm::CanonicalBytesSha256,
            SharedFingerprintValueKind::ValidationReceipt,
            "validation-receipt-v1",
            "validation-receipt",
            256,
        )
        .with_canonical_domain("validation-receipt", "v1");
        let dedup = SharedDedupIdentity::new(
            "ay proof receipt dedup",
            fingerprint,
            SharedDedupScope::ProofQuery,
            SharedDedupStorageKind::Cas,
            SetupTraceLaneKind::AY,
        )
        .with_collision_policy(SharedCollisionPolicy::ProofWitnessRequired);
        PreparedFingerprintAdmissionPlan::new(
            "ay receipt-backed admission",
            CheckerSourceKind::AYOnly,
            PreparedProgramPayloadKind::AYOnly,
            PreparedStorageKind::SmtVariables,
            SetupTraceLaneKind::AY,
            dedup,
            SharedDuplicateAuthorization::ProofWitness,
            PreparedFingerprintPayloadWitnessKind::ValidationReceiptProof,
        )
    }

    fn accepted_proof_receipt() -> ValidationReceipt {
        ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "accepted-proof-digest",
            "prepared-ay-program",
            "ay-candidate",
            ValidationReceiptArtifactKind::Proof,
            "ay-proof",
        )
    }

    fn rejected_proof_receipt() -> ValidationReceipt {
        ValidationReceipt::rejected(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "rejected-proof-digest",
            "prepared-ay-program",
            "ay-candidate",
            ValidationReceiptArtifactKind::Proof,
            "ay-proof",
            "proof checker rejected certificate",
        )
    }

    #[test]
    fn prepared_fingerprint_tla_array_fp64_plan_captures_shared_identity_and_policy() {
        let plan =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1");

        assert_eq!(plan.source_kind, CheckerSourceKind::Tla);
        assert_eq!(plan.payload_kind, PreparedProgramPayloadKind::Tla);
        assert_eq!(plan.storage_kind, PreparedStorageKind::TlaStateSlots);
        assert_eq!(plan.lane, SetupTraceLaneKind::ExplicitState);
        assert_eq!(
            plan.payload_witness,
            PreparedFingerprintPayloadWitnessKind::TlaArrayFp64
        );
        assert_eq!(
            plan.duplicate_authorization,
            SharedDuplicateAuthorization::CanonicalPayloadEquality
        );
        assert_eq!(
            plan.dedup.fingerprint.algorithm,
            SharedFingerprintAlgorithm::TlaFingerprint64
        );
        assert_eq!(
            plan.dedup.fingerprint.value_kind,
            SharedFingerprintValueKind::StateVector
        );
        assert_eq!(
            plan.prepared_fingerprint_descriptor().scheme.code(),
            "tla_fingerprint64"
        );
        assert!(plan.validate().is_ok());
        assert!(plan.authorizes_duplicate(SharedDuplicateAuthorization::CanonicalPayloadEquality));
        assert!(!plan.authorizes_duplicate(SharedDuplicateAuthorization::ProofWitness));
    }

    #[test]
    fn prepared_fingerprint_compiled_flat_xxh3_plan_records_native_lane_metadata() {
        let program = PreparedCheckerProgram::new(
            "compiled program",
            PreparedProgramPayloadKind::Quint,
            PreparedStorageKind::TlaStateSlots,
        )
        .with_cache_key("program cache")
        .with_frontend_payload_identity("quint payload")
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("native lane", SetupTraceLaneKind::Native)
                .with_candidate_key("native candidate")
                .with_lane_identity("native lane identity"),
        );
        let lane = &program.candidate_lanes[0];
        let plan = PreparedFingerprintAdmissionPlan::compiled_flat_xxh3(
            "compiled flat admission",
            CheckerSourceKind::Quint,
            PreparedProgramPayloadKind::Quint,
            "flat-v1",
        )
        .with_prepared_candidate_lane(&program, lane);

        assert_eq!(plan.source_kind, CheckerSourceKind::Quint);
        assert_eq!(plan.payload_kind, PreparedProgramPayloadKind::Quint);
        assert_eq!(plan.lane, SetupTraceLaneKind::Native);
        assert_eq!(plan.candidate_key.as_deref(), Some("native candidate"));
        assert_eq!(
            plan.prepared_program_identity.as_deref(),
            Some("compiled program")
        );
        assert_eq!(plan.prepared_lane_identity.as_deref(), Some("native lane"));
        assert_eq!(plan.identities.cache_key.as_deref(), Some("program cache"));
        assert_eq!(
            plan.identities.frontend_payload_identity.as_deref(),
            Some("quint payload")
        );
        assert_eq!(
            plan.identities.lane_identity.as_deref(),
            Some("native lane identity")
        );
        assert_eq!(plan.dedup.storage, SharedDedupStorageKind::Cas);
        assert_eq!(
            plan.dedup.fingerprint.algorithm,
            SharedFingerprintAlgorithm::Xxh3U64
        );
        assert_eq!(plan.payload_witness.code(), "compiled_flat_xxh3");
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn prepared_fingerprint_petri_marking_cas_plan_captures_marking_dedup() {
        let plan =
            PreparedFingerprintAdmissionPlan::petri_marking_cas("petri marking admission", "m1");

        assert_eq!(plan.source_kind, CheckerSourceKind::MccPetri);
        assert_eq!(plan.payload_kind, PreparedProgramPayloadKind::MccPetri);
        assert_eq!(plan.storage_kind, PreparedStorageKind::PetriMarking);
        assert_eq!(
            plan.payload_witness,
            PreparedFingerprintPayloadWitnessKind::PetriMarkingCas
        );
        assert_eq!(plan.dedup.storage, SharedDedupStorageKind::Cas);
        assert_eq!(plan.dedup.scope, SharedDedupScope::StateSpace);
        assert_eq!(
            plan.dedup.fingerprint.value_kind,
            SharedFingerprintValueKind::MarkingVector
        );
        assert_eq!(
            plan.dedup.fingerprint.algorithm,
            SharedFingerprintAlgorithm::CanonicalBytesSha256
        );
        assert_eq!(plan.dedup.fingerprint.digest_bits, 128);
        assert_eq!(
            plan.identities.storage_policy_identity.as_deref(),
            Some(plan.dedup.storage_policy_identity().as_str())
        );
        assert_eq!(
            plan.identities.fingerprint_identity.as_deref(),
            Some(plan.dedup.fingerprint.fingerprint_identity().as_str())
        );
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn prepared_fingerprint_admission_renders_shared_rows_for_tla_petri_and_register_styles() {
        struct Case {
            scope: &'static str,
            plan: PreparedFingerprintAdmissionPlan,
            fields: Vec<(&'static str, &'static str)>,
        }

        let cases = vec![
            Case {
                scope: "TY",
                plan: PreparedFingerprintAdmissionPlan::tla_array_fp64(
                    "tla fp64 admission",
                    "slots-v1",
                ),
                fields: vec![
                    ("schema", PreparedFingerprintAdmissionPlan::EVIDENCE_SCHEMA),
                    ("schema_version", "1"),
                    ("source_kind", "tla"),
                    ("frontend_family", "tla_plus"),
                    ("payload_kind", "tla"),
                    ("storage_kind", "tla_state_slots"),
                    ("lane_kind", "explicit_state"),
                    ("plan_id", "tla_fp64_admission"),
                    ("payload_witness", "tla_array_fp64"),
                    ("collision_policy", "canonical_payload_equality"),
                    ("duplicate_authorization", "canonical_payload_equality"),
                    ("admission_status", "accepted"),
                    ("reason_code", "accepted"),
                    ("compatible_frontend_families", "tla_plus,quint"),
                    ("default_consumers", "tla_plus"),
                    ("remaining_compatible_frontend_families", "quint"),
                ],
            },
            Case {
                scope: "MCC",
                plan: PreparedFingerprintAdmissionPlan::petri_marking_cas(
                    "petri marking admission",
                    "m1",
                ),
                fields: vec![
                    ("source_kind", "mcc_petri"),
                    ("frontend_family", "mcc_petri"),
                    ("payload_kind", "mcc_petri"),
                    ("storage_kind", "petri_marking"),
                    ("lane_kind", "explicit_state"),
                    ("plan_id", "petri_marking_admission"),
                    ("payload_witness", "petri_marking_cas"),
                    ("compatible_frontend_families", "mcc_petri"),
                    ("default_consumers", "mcc_petri"),
                    ("remaining_compatible_frontend_families", "none"),
                ],
            },
            Case {
                scope: "BTOR2",
                plan: PreparedFingerprintAdmissionPlan::register_vector_canonical(
                    "btor2 register vector admission",
                    CheckerSourceKind::Btor2,
                    PreparedProgramPayloadKind::Btor2,
                    "register-vector-v1",
                ),
                fields: vec![
                    ("source_kind", "btor2"),
                    ("frontend_family", "btor2"),
                    ("payload_kind", "btor2"),
                    ("storage_kind", "hardware_registers"),
                    ("lane_kind", "fingerprint"),
                    ("plan_id", "btor2_register_vector_admission"),
                    ("payload_witness", "register_vector_canonical"),
                    (
                        "compatible_frontend_families",
                        "aiger,btor2,vmt_transition_system,ay_analytical,witness_replay",
                    ),
                    ("default_consumers", "aiger,btor2"),
                    (
                        "remaining_compatible_frontend_families",
                        "vmt_transition_system,ay_analytical,witness_replay",
                    ),
                ],
            },
        ];

        assert!(
            PreparedFingerprintAdmissionPlan::EVIDENCE_REQUIRED_FIELDS.contains(&"payload_witness")
        );
        assert!(PreparedFingerprintAdmissionPlan::EVIDENCE_REQUIRED_FIELDS
            .contains(&"compatible_frontend_families"));

        for case in cases {
            let row = case.plan.render_evidence_row(case.scope);
            assert!(
                row.starts_with(&format!(
                    "{} {} ",
                    case.scope,
                    PreparedFingerprintAdmissionPlan::EVIDENCE_ROW_KIND
                )),
                "row should start with scope and prepared admission kind: {row}"
            );
            PreparedFingerprintAdmissionPlan::validate_evidence_row(&row)
                .expect("rendered prepared admission row should validate");
            for (field, expected) in case.fields {
                assert_eq!(
                    prepared_admission_evidence_field(&row, field),
                    Some(expected),
                    "field {field} mismatch in row: {row}"
                );
            }
            assert_eq!(
                prepared_admission_evidence_field(&row, "shared_engine_component"),
                Some("prepared_fingerprint_admission")
            );
            assert_eq!(
                prepared_admission_evidence_field(&row, "fail_closed"),
                Some("true")
            );
            assert_ne!(
                prepared_admission_evidence_field(&row, "dedup_identity"),
                Some("none")
            );
            assert_ne!(
                prepared_admission_evidence_field(&row, "fingerprint_identity"),
                Some("none")
            );
        }
    }

    #[test]
    fn prepared_fingerprint_admission_evidence_validator_rejects_missing_or_mismatched_fields() {
        let row =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1")
                .render_evidence_row("TY");

        let missing_witness = row.replacen(" payload_witness=tla_array_fp64", "", 1);
        let missing_error =
            PreparedFingerprintAdmissionPlan::validate_evidence_row(&missing_witness)
                .expect_err("validator should reject rows missing the witness contract");
        assert!(missing_error.contains("missing field payload_witness"));

        let mismatched_frontend =
            row.replacen("frontend_family=tla_plus", "frontend_family=mcc_petri", 1);
        let mismatch_error =
            PreparedFingerprintAdmissionPlan::validate_evidence_row(&mismatched_frontend)
                .expect_err("validator should reject source/frontend-family drift");
        assert!(mismatch_error.contains("requires frontend_family tla_plus"));
    }

    #[test]
    fn prepared_fingerprint_admission_rejects_mismatched_duplicate_authorization() {
        let plan = PreparedFingerprintAdmissionPlan::petri_marking_cas("petri", "m1");
        let invalid = PreparedFingerprintAdmissionPlan {
            duplicate_authorization: SharedDuplicateAuthorization::ProofWitness,
            ..plan
        };

        let err = invalid
            .validate()
            .expect_err("proof witness must not satisfy payload equality policy");
        assert_eq!(err.reason_code, "duplicate_authorization_mismatch");
    }

    #[test]
    fn prepared_fingerprint_admission_rejects_missing_collision_policy() {
        let plan =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1");

        let err = plan
            .validate_collision_policy_binding(None)
            .expect_err("runtime admission must bind an explicit collision policy");

        assert_eq!(err.reason_code, "missing_collision_policy");
    }

    #[test]
    fn validated_prepared_fingerprint_admission_handle_validates_at_setup() {
        let plan =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1");

        let handle = plan
            .validated_runtime_handle()
            .expect("valid descriptor should create a runtime handle");

        assert_eq!(handle.plan().id, "tla fp64 admission");
        assert_eq!(
            handle.prepared_fingerprint_descriptor().scheme.code(),
            "tla_fingerprint64"
        );
        assert_eq!(
            handle.validation_evidence(),
            PreparedFingerprintAdmissionValidationEvidence {
                setup_descriptor_validation_count: 1,
                hot_descriptor_validation_count: 0,
            }
        );
        assert!(handle.authorizes_duplicate(SharedDuplicateAuthorization::CanonicalPayloadEquality));
    }

    #[test]
    fn invalid_prepared_fingerprint_descriptor_cannot_create_validated_handle() {
        let plan = PreparedFingerprintAdmissionPlan::petri_marking_cas("petri", "m1");
        let invalid = PreparedFingerprintAdmissionPlan {
            storage_kind: PreparedStorageKind::Unknown,
            ..plan
        };

        let err = invalid
            .into_validated_runtime_handle()
            .expect_err("unknown storage ABI must fail before runtime binding");

        assert_eq!(err.reason_code, "unknown_storage_kind");
    }

    #[test]
    fn validated_prepared_fingerprint_hot_calls_do_not_revalidate_full_descriptor() {
        let set = InMemoryFingerprintSet::default();
        let plan =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1");
        let mut handle = plan
            .validated_runtime_handle()
            .expect("valid descriptor should create a runtime handle");

        // Public callers cannot mutate the private handle. This intentionally
        // poisons descriptor-only metadata to prove the hot handle path does not
        // rerun full prepared-plan validation per fingerprint.
        handle.plan.id.clear();

        let first = handle
            .admit_fingerprint_with_canonical_payload_comparison(&set, 101_u64, || {
                panic!("new admission must not compare duplicate payloads")
            })
            .expect("validated handle should admit new fingerprints");
        let duplicate = handle
            .admit_fingerprint_with_canonical_payload_comparison(&set, 101_u64, || Ok(true))
            .expect("validated handle should suppress authorized duplicates");

        assert_eq!(first, FingerprintAdmission::New);
        assert_eq!(duplicate, FingerprintAdmission::Duplicate);
        assert_eq!(set.len(), 1);
        assert_eq!(
            handle.validation_evidence().hot_descriptor_validation_count,
            0
        );
    }

    #[test]
    fn descriptor_convenience_methods_revalidate_but_runtime_handle_uses_setup_snapshot() {
        let descriptor =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1");
        let handle = descriptor
            .validated_runtime_handle()
            .expect("valid descriptor should create a runtime handle");
        let mut invalid_descriptor = descriptor.clone();
        invalid_descriptor.id.clear();
        let descriptor_set = InMemoryFingerprintSet::default();

        let descriptor_error = invalid_descriptor
            .admit_fingerprint_with_canonical_payload_comparison(&descriptor_set, 707_u64, || {
                panic!("invalid descriptor must fail before storage admission")
            })
            .expect_err("descriptor convenience API may validate per call");

        assert_eq!(descriptor_error.backend, "prepared_fingerprint_admission");
        assert!(descriptor_error
            .detail
            .contains("reason_code=empty_prepared_admission_id"));
        assert_eq!(descriptor_set.len(), 0);

        let handle_set = InMemoryFingerprintSet::default();
        let handle_admission = handle
            .admit_fingerprint_with_canonical_payload_comparison(&handle_set, 707_u64, || {
                panic!("new handle admission must not compare duplicate payloads")
            })
            .expect("setup-bound handle should use its validated snapshot");

        assert_eq!(handle_admission, FingerprintAdmission::New);
        assert_eq!(
            handle.validation_evidence(),
            PreparedFingerprintAdmissionValidationEvidence {
                setup_descriptor_validation_count: 1,
                hot_descriptor_validation_count: 0,
            }
        );
    }

    #[test]
    fn validated_prepared_fingerprint_handle_preserves_fail_closed_duplicate_checks() {
        let set = InMemoryFingerprintSet::default();
        let handle =
            PreparedFingerprintAdmissionPlan::petri_marking_cas("petri marking admission", "m1")
                .into_validated_runtime_handle()
                .expect("valid descriptor should create a runtime handle");

        assert_eq!(
            handle.admit_fingerprint_with_canonical_payload_comparison(&set, 42_u64, || {
                panic!("new admission must not compare duplicate payloads")
            }),
            Ok(FingerprintAdmission::New)
        );
        let payload_error = handle
            .admit_fingerprint_with_canonical_payload_comparison(&set, 42_u64, || Ok(false))
            .expect_err("same fingerprint with different payload must fail closed");

        assert_eq!(payload_error.backend, "prepared_fingerprint_admission");
        assert_eq!(payload_error.operation, "admit");
        assert!(payload_error
            .detail
            .contains("reason_code=canonical_payload_mismatch"));

        let collision_error = handle
            .authorize_duplicate_with_canonical_payload_equality(
                SharedCollisionPolicy::RejectOnCollision,
                true,
            )
            .expect_err("observed collision policy drift must fail closed");

        assert_eq!(collision_error.backend, "prepared_fingerprint_admission");
        assert!(collision_error
            .detail
            .contains("reason_code=collision_policy_mismatch"));
    }

    #[test]
    fn validated_prepared_fingerprint_enforces_computed_duplicate_admissions() {
        let handle =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1")
                .into_validated_runtime_handle()
                .expect("valid descriptor should create a runtime handle");

        let new = handle
            .enforce_duplicate_with_canonical_payload_comparison(FingerprintAdmission::New, || {
                panic!("new admission must not compare duplicate payloads")
            })
            .expect("new admission should pass through");
        let duplicate = handle
            .enforce_duplicate_with_canonical_payload_comparison(
                FingerprintAdmission::Duplicate,
                || Ok(true),
            )
            .expect("payload-confirmed duplicate should suppress");
        handle
            .enforce_batch_duplicate_with_canonical_payload_comparison(3, 2, false, || Ok(true))
            .expect("batch with payload-confirmed duplicate should suppress");
        handle
            .enforce_batch_duplicate_with_canonical_payload_comparison(3, 2, true, || {
                panic!("faulted batch should not compare duplicate payloads")
            })
            .expect("storage fault owns batch failure before duplicate enforcement");

        assert_eq!(new, FingerprintAdmission::New);
        assert_eq!(duplicate, FingerprintAdmission::Duplicate);
        assert_eq!((), ());
        assert_eq!((), ());

        let mismatch = handle
            .enforce_batch_duplicate_with_canonical_payload_comparison(2, 1, false, || Ok(false))
            .expect_err("batch duplicate without payload proof must fail closed");
        assert_eq!(mismatch.backend, "prepared_fingerprint_admission");
        assert!(mismatch
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
        assert_eq!(
            handle.validation_evidence(),
            PreparedFingerprintAdmissionValidationEvidence {
                setup_descriptor_validation_count: 1,
                hot_descriptor_validation_count: 0,
            }
        );
    }

    #[test]
    fn prepared_fingerprint_typed_canonical_outcome_rejects_payload_mismatch() {
        let handle =
            PreparedFingerprintAdmissionPlan::petri_marking_cas("petri marking admission", "m1")
                .into_validated_runtime_handle()
                .expect("valid descriptor should create a runtime handle");

        let duplicate = handle
            .enforce_duplicate_with_canonical_payload_evidence(
                FingerprintAdmission::Duplicate,
                || Ok(true),
            )
            .expect("payload-confirmed duplicate should suppress");
        let evidence = duplicate
            .duplicate_authorization
            .expect("duplicate outcome should record authorization evidence");

        assert_eq!(duplicate.admission, FingerprintAdmission::Duplicate);
        assert_eq!(
            evidence.observed_collision_policy,
            SharedCollisionPolicy::CanonicalPayloadEquality
        );
        assert_eq!(
            evidence.authorization,
            SharedDuplicateAuthorization::CanonicalPayloadEquality
        );
        assert_eq!(
            evidence.payload_witness,
            PreparedFingerprintPayloadWitnessKind::PetriMarkingCas
        );
        assert!(evidence.satisfies_observed_policy());
        assert_eq!(
            duplicate.counters,
            PreparedFingerprintAdmissionCounters {
                fingerprints_attempted: 1,
                fingerprints_inserted: 0,
                duplicate_fingerprints: 1,
                duplicate_authorization_checks: 1,
                ..PreparedFingerprintAdmissionCounters::default()
            }
        );

        let mismatch = handle
            .enforce_duplicate_with_canonical_payload_evidence(
                FingerprintAdmission::Duplicate,
                || Ok(false),
            )
            .expect_err("same fingerprint with mismatched payload must fail closed");
        assert_eq!(mismatch.backend, "prepared_fingerprint_admission");
        assert!(mismatch
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
    }

    #[test]
    fn prepared_fingerprint_batch_outcome_reuses_duplicate_authorization_evidence() {
        let handle =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1")
                .into_validated_runtime_handle()
                .expect("valid descriptor should create a runtime handle");

        let batch = handle
            .enforce_batch_duplicate_with_canonical_payload_evidence(8, 5, false, || Ok(true))
            .expect("payload-confirmed batch duplicates should suppress");
        let evidence = batch
            .duplicate_authorization
            .expect("batch duplicate should record authorization evidence");

        assert_eq!(batch.duplicate_count(), 3);
        assert_eq!(
            evidence.observed_collision_policy,
            SharedCollisionPolicy::CanonicalPayloadEquality
        );
        assert_eq!(
            evidence.authorization,
            SharedDuplicateAuthorization::CanonicalPayloadEquality
        );
        assert_eq!(
            evidence.payload_witness,
            PreparedFingerprintPayloadWitnessKind::TlaArrayFp64
        );
        assert_eq!(
            batch.counters,
            PreparedFingerprintAdmissionCounters {
                fingerprints_attempted: 8,
                fingerprints_inserted: 5,
                duplicate_fingerprints: 3,
                duplicate_authorization_checks: 1,
                ..PreparedFingerprintAdmissionCounters::default()
            }
        );

        let no_duplicate = handle
            .enforce_batch_duplicate_with_canonical_payload_evidence(4, 4, false, || {
                panic!("batch without duplicates must not compare payloads")
            })
            .expect("batch without duplicates should pass through");
        assert_eq!(no_duplicate.duplicate_authorization, None);
        assert_eq!(no_duplicate.duplicate_count(), 0);
        assert_eq!(no_duplicate.counters.duplicate_authorization_checks, 0);

        let faulted = handle
            .enforce_batch_duplicate_with_canonical_payload_evidence(4, 1, true, || {
                panic!("faulted batch must not compare duplicate payloads")
            })
            .expect("storage fault owns batch failure before duplicate enforcement");
        assert_eq!(faulted.duplicate_authorization, None);
        assert_eq!(faulted.duplicate_count(), 0);
        assert_eq!(faulted.counters.duplicate_authorization_checks, 0);
    }

    #[test]
    fn validated_prepared_fingerprint_admits_storage_batch_with_canonical_evidence() {
        let set = InMemoryFingerprintSet::default();
        let handle =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1")
                .into_validated_runtime_handle()
                .expect("valid descriptor should create a runtime handle");

        let batch = handle
            .admit_fingerprint_batch_with_canonical_payload_evidence(
                &set,
                &[10_u64, 11, 10, 12],
                || Ok(true),
            )
            .expect("payload-confirmed batch duplicates should suppress");
        let evidence = batch
            .duplicate_authorization
            .expect("batch duplicate should record authorization evidence");

        assert_eq!(batch.attempted, 4);
        assert_eq!(batch.inserted_count, 3);
        assert_eq!(batch.duplicate_count(), 1);
        assert_eq!(
            evidence.authorization,
            SharedDuplicateAuthorization::CanonicalPayloadEquality
        );
        assert_eq!(
            batch.counters,
            PreparedFingerprintAdmissionCounters {
                fingerprints_attempted: 4,
                fingerprints_inserted: 3,
                duplicate_fingerprints: 1,
                duplicate_authorization_checks: 1,
                ..PreparedFingerprintAdmissionCounters::default()
            }
        );
        assert_eq!(set.len(), 3);

        let no_duplicate = handle
            .admit_fingerprint_batch_with_canonical_payload_evidence(&set, &[13_u64, 14], || {
                panic!("batch without duplicates must not compare payloads")
            })
            .expect("duplicate-free batch should not require payload comparison");
        assert_eq!(no_duplicate.duplicate_authorization, None);
        assert_eq!(no_duplicate.duplicate_count(), 0);

        let mismatch = handle
            .admit_fingerprint_batch_with_canonical_payload_evidence(&set, &[10_u64], || Ok(false))
            .expect_err("batch duplicate without payload proof must fail closed");
        assert_eq!(mismatch.backend, "prepared_fingerprint_admission");
        assert!(mismatch
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
    }

    #[test]
    fn validated_prepared_fingerprint_proof_witness_path_is_fail_closed() {
        let plan = validation_receipt_admission_plan();
        let handle = plan
            .into_validated_runtime_handle()
            .expect("proof/witness descriptor should validate for receipt fingerprints");

        let accepted = handle
            .authorize_duplicate_with_proof_witness(
                SharedCollisionPolicy::ProofWitnessRequired,
                true,
            )
            .expect("accepted proof witness should authorize duplicate");
        assert_eq!(accepted, SharedDuplicateAuthorization::ProofWitness);

        let rejected = handle
            .authorize_duplicate_with_proof_witness(
                SharedCollisionPolicy::ProofWitnessRequired,
                false,
            )
            .expect_err("rejected proof witness must fail closed");
        assert_eq!(rejected.backend, "prepared_fingerprint_admission");
        assert!(rejected
            .detail
            .contains("reason_code=proof_witness_rejected"));

        let payload_only = handle
            .authorize_duplicate_with_canonical_payload_equality(
                SharedCollisionPolicy::ProofWitnessRequired,
                true,
            )
            .expect_err("payload equality cannot satisfy proof/witness policy");
        assert!(payload_only
            .detail
            .contains("reason_code=canonical_payload_equality_unsupported"));
    }

    #[test]
    fn validated_prepared_fingerprint_receipt_and_proof_paths_share_typed_evidence() {
        let set = InMemoryFingerprintSet::default();
        let handle = validation_receipt_admission_plan()
            .into_validated_runtime_handle()
            .expect("proof/witness descriptor should validate for receipt fingerprints");

        let first = handle
            .admit_fingerprint_with_validation_receipt_evidence(&set, 909_u64, || {
                panic!("new validation receipt admission must not validate duplicate receipts")
            })
            .expect("new proof receipt fingerprint should admit");
        assert_eq!(first.admission, FingerprintAdmission::New);
        assert_eq!(first.duplicate_authorization, None);
        assert_eq!(
            first.counters,
            PreparedFingerprintAdmissionCounters {
                fingerprints_attempted: 1,
                fingerprints_inserted: 1,
                ..PreparedFingerprintAdmissionCounters::default()
            }
        );

        let duplicate = handle
            .admit_fingerprint_with_validation_receipt_evidence(&set, 909_u64, || {
                Ok(accepted_proof_receipt())
            })
            .expect("accepted validation receipt should authorize duplicate");
        let receipt_evidence = duplicate
            .duplicate_authorization
            .expect("duplicate receipt admission should record authorization evidence");
        assert_eq!(duplicate.admission, FingerprintAdmission::Duplicate);
        assert_eq!(
            receipt_evidence.observed_collision_policy,
            SharedCollisionPolicy::ProofWitnessRequired
        );
        assert_eq!(
            receipt_evidence.authorization,
            SharedDuplicateAuthorization::ProofWitness
        );
        assert_eq!(
            receipt_evidence.payload_witness,
            PreparedFingerprintPayloadWitnessKind::ValidationReceiptProof
        );
        assert!(receipt_evidence.satisfies_observed_policy());
        assert_eq!(duplicate.counters.duplicate_authorization_checks, 1);

        let proof_batch = handle
            .enforce_batch_duplicate_with_proof_witness_evidence(6, 2, false, || Ok(true))
            .expect("accepted proof witness should authorize batch duplicates");
        assert_eq!(proof_batch.duplicate_count(), 4);
        assert_eq!(proof_batch.duplicate_authorization, Some(receipt_evidence));
        assert_eq!(
            proof_batch.counters,
            PreparedFingerprintAdmissionCounters {
                fingerprints_attempted: 6,
                fingerprints_inserted: 2,
                duplicate_fingerprints: 4,
                duplicate_authorization_checks: 1,
                ..PreparedFingerprintAdmissionCounters::default()
            }
        );

        let receipt_batch = handle
            .enforce_batch_duplicate_with_validation_receipt_evidence(5, 3, false, || {
                Ok(accepted_proof_receipt())
            })
            .expect("accepted validation receipt should authorize batch duplicates");
        assert_eq!(receipt_batch.duplicate_count(), 2);
        assert_eq!(
            receipt_batch.duplicate_authorization,
            Some(receipt_evidence)
        );

        let rejected = handle
            .admit_fingerprint_with_validation_receipt_evidence(&set, 909_u64, || {
                Ok(rejected_proof_receipt())
            })
            .expect_err("rejected validation receipt must fail closed");
        assert_eq!(rejected.backend, "prepared_fingerprint_admission");
        assert!(rejected
            .detail
            .contains("reason_code=proof_witness_rejected"));
    }

    #[test]
    fn prepared_fingerprint_hot_outcomes_keep_setup_and_hot_validation_counters_separate() {
        let mut handle =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1")
                .into_validated_runtime_handle()
                .expect("valid descriptor should create a runtime handle");

        assert_eq!(
            handle.validation_evidence(),
            PreparedFingerprintAdmissionValidationEvidence {
                setup_descriptor_validation_count: 1,
                hot_descriptor_validation_count: 0,
            }
        );
        handle.plan.id.clear();

        let scalar = handle
            .enforce_duplicate_with_canonical_payload_evidence(
                FingerprintAdmission::Duplicate,
                || Ok(true),
            )
            .expect("hot duplicate authorization should use setup snapshot");
        let batch = handle
            .enforce_batch_duplicate_with_canonical_payload_evidence(3, 1, false, || Ok(true))
            .expect("hot batch authorization should use setup snapshot");

        assert_eq!(scalar.counters.setup_descriptor_validation_count, 0);
        assert_eq!(scalar.counters.hot_descriptor_validation_count, 0);
        assert_eq!(batch.counters.setup_descriptor_validation_count, 0);
        assert_eq!(batch.counters.hot_descriptor_validation_count, 0);
        assert_eq!(
            handle.validation_evidence(),
            PreparedFingerprintAdmissionValidationEvidence {
                setup_descriptor_validation_count: 1,
                hot_descriptor_validation_count: 0,
            }
        );
    }

    #[test]
    fn prepared_fingerprint_shared_api_admits_tla_and_petri_duplicates() {
        let set = InMemoryFingerprintSet::default();
        let tla_plan =
            PreparedFingerprintAdmissionPlan::tla_array_fp64("tla fp64 admission", "slots-v1");
        let petri_plan =
            PreparedFingerprintAdmissionPlan::petri_marking_cas("petri marking admission", "m1");

        let first_tla = tla_plan
            .admit_fingerprint_with_canonical_payload_comparison(&set, 11_u64, || {
                panic!("new TLA state-slot admission must not compare duplicate payloads")
            })
            .expect("new TLA fingerprint should admit");
        let duplicate_tla = tla_plan
            .admit_fingerprint_with_canonical_payload_comparison(&set, 11_u64, || Ok(true))
            .expect("payload-confirmed TLA duplicate should suppress");

        let first_petri = petri_plan
            .admit_fingerprint_with_canonical_payload_comparison(&set, 23_u64, || {
                panic!("new Petri marking admission must not compare duplicate payloads")
            })
            .expect("new Petri fingerprint should admit");
        let duplicate_petri = petri_plan
            .admit_fingerprint_with_canonical_payload_comparison(&set, 23_u64, || Ok(true))
            .expect("payload-confirmed Petri duplicate should suppress");

        assert_eq!(first_tla, FingerprintAdmission::New);
        assert_eq!(duplicate_tla, FingerprintAdmission::Duplicate);
        assert_eq!(first_petri, FingerprintAdmission::New);
        assert_eq!(duplicate_petri, FingerprintAdmission::Duplicate);
    }

    #[test]
    fn prepared_fingerprint_shared_api_rejects_payload_mismatch() {
        let set = InMemoryFingerprintSet::default();
        let plan =
            PreparedFingerprintAdmissionPlan::petri_marking_cas("petri marking admission", "m1");

        assert_eq!(
            plan.admit_fingerprint_with_canonical_payload_comparison(&set, 42_u64, || {
                panic!("new admission must not compare duplicate payloads")
            }),
            Ok(FingerprintAdmission::New)
        );
        let error = plan
            .admit_fingerprint_with_canonical_payload_comparison(&set, 42_u64, || Ok(false))
            .expect_err("same fingerprint with different payload must fail closed");

        assert_eq!(error.backend, "prepared_fingerprint_admission");
        assert_eq!(error.operation, "admit");
        assert!(error.detail.contains("status_code=rejected"));
        assert!(error
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
        assert!(error.detail.contains("fail_closed=true"));
        assert!(error
            .detail
            .contains("collision_policy=canonical_payload_equality"));
        assert!(error.detail.contains("payload_witness=petri_marking_cas"));
    }

    #[test]
    fn prepared_fingerprint_register_vector_plan_records_hardware_metadata() {
        let plan = PreparedFingerprintAdmissionPlan::register_vector_canonical(
            "btor2 register vector admission",
            CheckerSourceKind::Btor2,
            PreparedProgramPayloadKind::Btor2,
            "register-vector-v1",
        );

        assert!(plan.validate_runtime_admission().is_ok());
        assert_eq!(plan.source_kind.frontend_family_code(), "btor2");
        assert_eq!(plan.storage_kind, PreparedStorageKind::HardwareRegisters);
        assert_eq!(
            plan.payload_witness,
            PreparedFingerprintPayloadWitnessKind::RegisterVectorCanonical
        );
        assert_eq!(
            plan.dedup.fingerprint.value_kind,
            SharedFingerprintValueKind::RegisterVector
        );
        assert_eq!(plan.dedup.storage, SharedDedupStorageKind::Cas);

        let families = plan.dedup.fingerprint.reusable_frontend_families();
        assert!(families.contains(&SharedEngineFrontendFamily::Aiger));
        assert!(families.contains(&SharedEngineFrontendFamily::Btor2));
        assert!(families.contains(&SharedEngineFrontendFamily::VmtTransitionSystem));
        assert!(!families.contains(&SharedEngineFrontendFamily::FutureImporter));
    }
}
