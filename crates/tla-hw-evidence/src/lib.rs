// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Frontend-parameterized shared-engine adoption evidence for the hardware
//! frontends (AIGER and BTOR2).
//!
//! The AIGER and BTOR2 frontends emit the same family of shared-engine
//! adoption evidence rows that bind their lowering shapes to the shared
//! prepared-program, AY proof, replay, and hardware state fingerprint contracts
//! exposed by [`tla_mc_core`]. The bulk of that machinery (admission plans,
//! evidence-row rendering, identity digests, validator-backed proof/witness
//! lanes) is frontend-neutral; only a handful of names/constants and the
//! frontend-specific [`PreparedCheckerProgram`] construction differ.
//!
//! This crate captures the neutral machinery once. Each frontend supplies its
//! per-frontend names/constants and its program builder through the
//! [`HardwareFrontend`] trait; the generic [`SharedEngineEvidence`] bundle and
//! the free helpers in this module then produce byte-identical evidence rows.

use std::marker::PhantomData;

use tla_ay::{
    AYFrontendFamily, AYProofValidationReceipt, AYProofValidationReceiptKind, AYSharedEngineLane,
    AYSharedProofLaneDescriptor,
};
use tla_mc_core::CheckerSourceKind;
use tla_mc_core::{
    PreparedCheckerProgram, PreparedFingerprintAdmissionPlan, PreparedFingerprintDescriptor,
    PreparedFingerprintScheme, PreparedProgramPayloadKind, PreparedValidationKind,
    PreparedValidationPlanDescriptor, ProblemKind, SharedEngineAdoptionEvidence,
    SharedEngineAdoptionFamilyBlocker, SharedEngineAdoptionLevel, SharedEngineFrontendFamily,
    ValidationReceipt, ValidationReceiptArtifactKind, ValidationReceiptValidatorKind,
};

/// Hardware register-vector compatibility set shared by every hardware frontend.
pub const HARDWARE_REGISTER_VECTOR_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
/// Default register-vector consumers shared by every hardware frontend.
pub const HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS: &str = "aiger,btor2";
/// Remaining register-vector compatible families shared by every hardware frontend.
pub const HARDWARE_REGISTER_VECTOR_REMAINING_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "vmt_transition_system,ay_analytical,witness_replay";
/// Tracked register-vector blockers shared by every hardware frontend.
pub const HARDWARE_REGISTER_VECTOR_BLOCKERS: &str =
    "future_importer:awaiting_registered_importer_frontend";

/// Per-frontend names, constants, and the frontend-specific prepared-program
/// construction that distinguish two otherwise-identical hardware evidence
/// builders (e.g. AIGER vs BTOR2).
///
/// Implementors are zero-sized marker types: every method is associated and
/// `Self` is never instantiated.
pub trait HardwareFrontend {
    /// The frontend-specific input (e.g. an AIGER circuit or a BTOR2 program).
    type Input;

    // --- Render label + origin identity ------------------------------------

    /// Uppercase row label, e.g. `"AIGER"` / `"BTOR2"`.
    const LABEL: &'static str;
    /// Lowercase origin-frontend token, e.g. `"aiger"` / `"btor2"`.
    const ORIGIN_FRONTEND: &'static str;
    /// Shared-engine component name (frontend-neutral, kept per-frontend for clarity).
    const SHARED_ENGINE_COMPONENT: &'static str;
    /// Owning team/component recorded as the responsible party in the
    /// shared-engine adoption row.
    const SHARED_ENGINE_OWNER: &'static str;
    /// First beneficiary / portfolio name, e.g. `"aiger_portfolio"`.
    const PORTFOLIO: &'static str;
    /// Second beneficiary for the shared-engine adoption row.
    const SHARED_ENGINE_SECOND_BENEFICIARY: &'static str;
    /// Extraction status reported in evidence rows.
    const SHARED_ENGINE_EXTRACTION_STATUS: &'static str;
    /// Acceptance-test command reported in the adoption row.
    const ACCEPTANCE_TEST: &'static str;
    /// Digest algorithm used for prepared-program identity digests.
    const PREPARED_PROGRAM_DIGEST_ALGORITHM: &'static str;

    // --- Fingerprint / register identities ---------------------------------

    /// Stable hardware register-layout identity.
    const REGISTER_LAYOUT_IDENTITY: &'static str;
    /// State (register vector) canonicalization label.
    const STATE_CANONICALIZATION: &'static str;

    // --- Admission plan ----------------------------------------------------

    /// Human-readable admission plan description.
    const ADMISSION_DESCRIPTION: &'static str;
    /// Source-kind for admission plans / dedup rows.
    const CHECKER_SOURCE_KIND: CheckerSourceKind;
    /// Prepared-program payload kind.
    const PROGRAM_PAYLOAD_KIND: PreparedProgramPayloadKind;

    // --- Generic adoption prerequisites ------------------------------------

    /// Description of the AY proof receipt prerequisite (e.g. "ay sat proof receipt").
    const AY_PROOF_RECEIPT_PREREQUISITE: &'static str;

    // --- AY proof / replay identities --------------------------------------

    /// Identity prefix for the AY safety candidate (e.g. `aiger.ay_sat.safety_candidate`).
    const AY_SAFETY_CANDIDATE_PREFIX: &'static str;
    /// Identity prefix for the AY proof artifact (e.g. `aiger.ay_sat.proof_artifact`).
    const AY_PROOF_ARTIFACT_PREFIX: &'static str;
    /// Identity prefix for the AY proof fingerprint (e.g. `aiger.ay_sat.proof`).
    const AY_PROOF_FINGERPRINT_PREFIX: &'static str;
    /// Identity prefix for the replay counterexample candidate.
    const REPLAY_COUNTEREXAMPLE_CANDIDATE_PREFIX: &'static str;
    /// Identity prefix for the replay counterexample artifact.
    const REPLAY_COUNTEREXAMPLE_ARTIFACT_PREFIX: &'static str;
    /// Identity prefix for the replay counterexample fingerprint.
    const REPLAY_COUNTEREXAMPLE_PREFIX: &'static str;

    // --- AY proof-lane descriptor / receipts -------------------------------

    /// Shared-engine lane bound by the generalized AY proof lane.
    const AY_SHARED_ENGINE_LANE: AYSharedEngineLane;
    /// AY frontend family bound by the generalized AY proof lane.
    const AY_FRONTEND_FAMILY: AYFrontendFamily;
    /// Proof obligation identity for the register-vector safety obligation.
    const AY_PROOF_OBLIGATION_IDENTITY: &'static str;
    /// Receipt identity for the generalized AY proof lane.
    const AY_PROOF_LANE_RECEIPT_IDENTITY: &'static str;
    /// Receipt identity for the generalized witness lane.
    const AY_WITNESS_LANE_RECEIPT_IDENTITY: &'static str;
    /// First beneficiary for the generalized AY proof-lane adoption row.
    const AY_PROOF_LANE_FIRST_BENEFICIARY: &'static str;
    /// Second beneficiary for the generalized AY proof-lane adoption row.
    const AY_PROOF_LANE_SECOND_BENEFICIARY: &'static str;

    // --- Frontend-specific construction ------------------------------------

    /// Build the shared prepared-program descriptor for this frontend's input.
    fn prepared_checker_program(input: &Self::Input) -> PreparedCheckerProgram;
}

/// Frontend-parameterized evidence bundle for the shared prepared-program
/// adoption contract.
///
/// The bundle is generic over the frontend `F` so that the rendering methods can
/// recover the per-frontend names/constants without a turbofish at every call
/// site. The `F` parameter is carried only as a zero-sized [`PhantomData`].
pub struct SharedEngineEvidence<F: HardwareFrontend> {
    /// The shared prepared-program descriptor.
    pub prepared_program: PreparedCheckerProgram,
    /// Stable FNV-1a digest over the prepared-program identity rows.
    pub prepared_program_digest: String,
    /// Register-vector admission plan for this prepared program.
    pub register_vector_admission: PreparedFingerprintAdmissionPlan,
    /// Shared-engine adoption evidence.
    pub adoption: SharedEngineAdoptionEvidence,
    /// AY proof validation receipt.
    pub ay_proof_receipt: ValidationReceipt,
    /// Proof-replay validation receipt.
    pub replay_receipt: ValidationReceipt,
    /// Validator-backed receipt for the generalized AY proof lane.
    pub ay_proof_lane_receipt: AYProofValidationReceipt,
    /// Validator-backed receipt for the generalized witness lane.
    pub ay_witness_lane_receipt: AYProofValidationReceipt,
    _frontend: PhantomData<fn() -> F>,
}

impl<F: HardwareFrontend> std::fmt::Debug for SharedEngineEvidence<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEngineEvidence")
            .field("prepared_program", &self.prepared_program)
            .field("prepared_program_digest", &self.prepared_program_digest)
            .field("register_vector_admission", &self.register_vector_admission)
            .field("adoption", &self.adoption)
            .field("ay_proof_receipt", &self.ay_proof_receipt)
            .field("replay_receipt", &self.replay_receipt)
            .field("ay_proof_lane_receipt", &self.ay_proof_lane_receipt)
            .field("ay_witness_lane_receipt", &self.ay_witness_lane_receipt)
            .finish()
    }
}

impl<F: HardwareFrontend> Clone for SharedEngineEvidence<F> {
    fn clone(&self) -> Self {
        Self {
            prepared_program: self.prepared_program.clone(),
            prepared_program_digest: self.prepared_program_digest.clone(),
            register_vector_admission: self.register_vector_admission.clone(),
            adoption: self.adoption.clone(),
            ay_proof_receipt: self.ay_proof_receipt.clone(),
            replay_receipt: self.replay_receipt.clone(),
            ay_proof_lane_receipt: self.ay_proof_lane_receipt.clone(),
            ay_witness_lane_receipt: self.ay_witness_lane_receipt.clone(),
            _frontend: PhantomData,
        }
    }
}

impl<F: HardwareFrontend> SharedEngineEvidence<F> {
    /// Build the evidence bundle for frontend `F` from its input.
    pub fn from_input(input: &F::Input) -> Self {
        let prepared_program = F::prepared_checker_program(input);
        let prepared_program_digest = prepared_program_identity_digest::<F>(&prepared_program);
        let register_vector_admission = register_vector_admission_plan::<F>(&prepared_program);
        let adoption = SharedEngineAdoptionEvidence::new(
            F::ORIGIN_FRONTEND,
            F::SHARED_ENGINE_COMPONENT,
            F::PORTFOLIO,
            F::SHARED_ENGINE_SECOND_BENEFICIARY,
            F::SHARED_ENGINE_EXTRACTION_STATUS,
            F::SHARED_ENGINE_OWNER,
            F::ACCEPTANCE_TEST,
        )
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level3,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::MccPetri,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::AYAnalytical,
                SharedEngineFrontendFamily::WitnessReplay,
            ],
            [SharedEngineAdoptionFamilyBlocker::new(
                SharedEngineFrontendFamily::FutureImporter,
                "awaiting registered importer frontend",
            )],
        )
        .with_generic_prerequisite("prepared program identity")
        .with_generic_prerequisite(F::AY_PROOF_RECEIPT_PREREQUISITE)
        .with_generic_prerequisite("proof replay receipt")
        .with_generic_prerequisite("hardware state fingerprint identity");

        let ay_proof_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            F::PREPARED_PROGRAM_DIGEST_ALGORITHM,
            prepared_program_digest.clone(),
            prepared_program.identity.clone(),
            prepared_identity(F::AY_SAFETY_CANDIDATE_PREFIX, &prepared_program.identity),
            ValidationReceiptArtifactKind::Proof,
            prepared_identity(F::AY_PROOF_ARTIFACT_PREFIX, &prepared_program.identity),
        );
        let replay_receipt = ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::ProofReplay,
            F::PREPARED_PROGRAM_DIGEST_ALGORITHM,
            prepared_program_digest.clone(),
            prepared_program.identity.clone(),
            prepared_identity(
                F::REPLAY_COUNTEREXAMPLE_CANDIDATE_PREFIX,
                &prepared_program.identity,
            ),
            ValidationReceiptArtifactKind::Witness,
            prepared_identity(
                F::REPLAY_COUNTEREXAMPLE_ARTIFACT_PREFIX,
                &prepared_program.identity,
            ),
        );
        let ay_proof_lane_receipt = ay_proof_lane_receipt::<F>(&prepared_program);
        let ay_witness_lane_receipt = ay_witness_lane_receipt::<F>(&prepared_program);

        Self {
            prepared_program,
            prepared_program_digest,
            register_vector_admission,
            adoption,
            ay_proof_receipt,
            replay_receipt,
            ay_proof_lane_receipt,
            ay_witness_lane_receipt,
            _frontend: PhantomData,
        }
    }

    /// Render all shared-engine adoption rows for frontend `F`.
    pub fn render_evidence_rows(&self) -> Vec<String> {
        let mut rows = Vec::new();
        rows.push(self.adoption.render_evidence_row(F::LABEL));
        rows.push(self.prepared_program.render_evidence_row(F::LABEL));
        rows.extend(
            self.prepared_program
                .render_frontend_extension_evidence_rows(F::LABEL),
        );
        rows.extend(
            self.prepared_program
                .render_candidate_lane_evidence_rows(F::LABEL),
        );
        rows.extend(
            self.prepared_program
                .render_validation_plan_evidence_rows(F::LABEL),
        );
        rows.extend(register_vector_admission_evidence_rows::<F>(
            &self.prepared_program,
            &self.register_vector_admission,
        ));
        rows.push(self.ay_proof_receipt.render_evidence_row(F::LABEL));
        rows.push(self.replay_receipt.render_evidence_row(F::LABEL));
        if let Some(row) = ay_proof_lane_adoption_evidence_row::<F>(
            &self.prepared_program,
            Some(&self.ay_proof_lane_receipt),
            Some(&self.ay_witness_lane_receipt),
        ) {
            rows.push(row);
        }
        rows
    }
}

/// Render all shared-engine adoption rows for frontend `F` from its input.
pub fn shared_engine_evidence_rows<F: HardwareFrontend>(input: &F::Input) -> Vec<String> {
    SharedEngineEvidence::<F>::from_input(input).render_evidence_rows()
}

/// Build the shared prepared fingerprint admission plan for frontend `F`.
///
/// Starts from the frontend's [`register_vector_admission_base_plan`] and binds
/// it to `program`. If `program` carries a `hardware_state_fingerprint`
/// candidate lane the plan is bound to that lane (so dedup/fingerprint identities
/// reference the lane); otherwise it is bound to the program as a whole.
pub fn register_vector_admission_plan<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
) -> PreparedFingerprintAdmissionPlan {
    let base = register_vector_admission_base_plan::<F>();
    if let Some(lane) = program
        .candidate_lanes
        .iter()
        .find(|lane| lane.candidate_key.as_deref() == Some("hardware_state_fingerprint"))
    {
        base.with_prepared_candidate_lane(program, lane)
    } else {
        base.with_prepared_program(program)
    }
}

/// Build the base register-vector admission plan for frontend `F`.
pub fn register_vector_admission_base_plan<F: HardwareFrontend>() -> PreparedFingerprintAdmissionPlan
{
    PreparedFingerprintAdmissionPlan::register_vector_canonical(
        F::ADMISSION_DESCRIPTION,
        F::CHECKER_SOURCE_KIND,
        F::PROGRAM_PAYLOAD_KIND,
        F::STATE_CANONICALIZATION,
    )
}

fn register_vector_admission_evidence_rows<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
    admission: &PreparedFingerprintAdmissionPlan,
) -> Vec<String> {
    vec![
        admission
            .dedup
            .fingerprint
            .render_evidence_row(F::LABEL, admission.source_kind),
        admission
            .dedup
            .fingerprint
            .render_validation_evidence_row(F::LABEL, admission.source_kind),
        admission
            .dedup
            .render_evidence_row(F::LABEL, admission.source_kind),
        admission
            .dedup
            .render_validation_evidence_row(F::LABEL, admission.source_kind),
        prepared_fingerprint_admission_evidence_row::<F>(admission),
        hardware_transition_system_adoption_evidence_row::<F>(program, admission),
    ]
}

fn prepared_fingerprint_admission_evidence_row<F: HardwareFrontend>(
    admission: &PreparedFingerprintAdmissionPlan,
) -> String {
    let validation = admission.validate_runtime_admission();
    let (admission_status, reason_code) = match validation {
        Ok(()) => ("accepted", "accepted"),
        Err(ref rejection) => ("rejected", rejection.reason_code),
    };
    format!(
        "{} prepared_fingerprint_admission schema=ty.prepared_fingerprint_admission.v1 schema_version=1 source_kind={} frontend_family={} shared_engine_component=prepared_fingerprint_admission plan_id={} payload_kind={} storage_kind={} lane_kind={} candidate_key={} prepared_program_identity={} prepared_lane_identity={} payload_witness={} dedup_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} collision_policy={} duplicate_authorization={} admission_status={} reason_code={} fail_closed=true compatible_frontend_families={} default_consumers={} remaining_compatible_frontend_families={} blockers={}",
        F::LABEL,
        admission.source_kind.code(),
        admission.source_kind.frontend_family_code(),
        evidence_token(&admission.id),
        admission.payload_kind.code(),
        admission.storage_kind.code(),
        admission.lane.code(),
        evidence_option(admission.candidate_key.as_deref()),
        evidence_option(admission.prepared_program_identity.as_deref()),
        evidence_option(admission.prepared_lane_identity.as_deref()),
        admission.payload_witness.code(),
        evidence_token(&admission.dedup.dedup_identity()),
        evidence_token(&admission.dedup.storage_policy_identity()),
        evidence_token(&admission.dedup.fingerprint.fingerprint_policy_identity()),
        evidence_token(&admission.dedup.fingerprint.fingerprint_identity()),
        admission.dedup.collision_policy.code(),
        admission.duplicate_authorization.code(),
        admission_status,
        reason_code,
        HARDWARE_REGISTER_VECTOR_COMPATIBLE_FRONTEND_FAMILIES,
        HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS,
        HARDWARE_REGISTER_VECTOR_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        HARDWARE_REGISTER_VECTOR_BLOCKERS,
    )
}

fn hardware_transition_system_adoption_evidence_row<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
    admission: &PreparedFingerprintAdmissionPlan,
) -> String {
    format!(
        "{} hardware_transition_system_adoption schema=ty.hardware.transition_system_adoption.v1 schema_version=1 origin_frontend={} shared_engine_component=prepared_checker_program prepared_program_identity={} payload_kind={} storage_kind={} transition_kind=hardware_next_state transition_count={} property_count={} register_vector_identity={} fingerprint_admission_plan={} dedup_identity={} fingerprint_identity={} ay_analytical_lane=receipt_backed witness_replay_lane=receipt_backed compatible_frontend_families={} default_consumers={} remaining_compatible_frontend_families={} blockers={}",
        F::LABEL,
        F::ORIGIN_FRONTEND,
        evidence_token(&program.identity),
        program.payload_kind.code(),
        program.storage_kind.code(),
        program.transitions.len(),
        program.properties.len(),
        F::REGISTER_LAYOUT_IDENTITY,
        evidence_token(&admission.id),
        evidence_token(&admission.dedup.dedup_identity()),
        evidence_token(&admission.dedup.fingerprint.fingerprint_identity()),
        HARDWARE_REGISTER_VECTOR_COMPATIBLE_FRONTEND_FAMILIES,
        HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS,
        HARDWARE_REGISTER_VECTOR_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        HARDWARE_REGISTER_VECTOR_BLOCKERS,
    )
}

/// Build the generalized AY proof-lane descriptor for frontend `F`.
pub fn ay_proof_lane_descriptor<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
) -> AYSharedProofLaneDescriptor {
    AYSharedProofLaneDescriptor::new(
        F::AY_SHARED_ENGINE_LANE,
        F::AY_FRONTEND_FAMILY,
        program.identity.clone(),
        F::AY_PROOF_OBLIGATION_IDENTITY,
    )
    .with_proof_fingerprint_identity(prepared_identity(
        F::AY_PROOF_FINGERPRINT_PREFIX,
        &program.identity,
    ))
    .with_witness_fingerprint_identity(prepared_identity(
        F::REPLAY_COUNTEREXAMPLE_PREFIX,
        &program.identity,
    ))
}

/// Build a validator-backed receipt for the generalized AY proof lane.
pub fn ay_proof_lane_receipt<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
) -> AYProofValidationReceipt {
    AYProofValidationReceipt::validator_backed(
        F::AY_PROOF_LANE_RECEIPT_IDENTITY,
        AYProofValidationReceiptKind::ProofTranscript,
        F::AY_PROOF_OBLIGATION_IDENTITY,
        prepared_identity(F::AY_PROOF_FINGERPRINT_PREFIX, &program.identity),
    )
}

/// Build a validator-backed receipt for the generalized witness lane.
pub fn ay_witness_lane_receipt<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
) -> AYProofValidationReceipt {
    AYProofValidationReceipt::validator_backed(
        F::AY_WITNESS_LANE_RECEIPT_IDENTITY,
        AYProofValidationReceiptKind::Witness,
        F::AY_PROOF_OBLIGATION_IDENTITY,
        prepared_identity(F::REPLAY_COUNTEREXAMPLE_PREFIX, &program.identity),
    )
}

/// Render generalized AY proof-lane adoption only after receipt validation succeeds.
pub fn ay_proof_lane_adoption_evidence_row<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
    proof_receipt: Option<&AYProofValidationReceipt>,
    witness_receipt: Option<&AYProofValidationReceipt>,
) -> Option<String> {
    ay_proof_lane_descriptor::<F>(program).render_hardware_adoption_evidence_with_receipts(
        F::LABEL,
        "hardware_ay_proof_lane_adoption",
        F::REGISTER_LAYOUT_IDENTITY,
        F::AY_PROOF_LANE_FIRST_BENEFICIARY,
        F::AY_PROOF_LANE_SECOND_BENEFICIARY,
        proof_receipt,
        witness_receipt,
    )
}

/// Stable FNV-1a (64-bit) digest over the prepared-program identity rows for
/// frontend `F`, formatted as a 16-character lowercase hex string.
///
/// The digest hashes the newline-terminated identity rows produced by
/// `prepared_program_identity_rows`, giving a deterministic prepared-program
/// fingerprint that is fed into the AY proof / replay receipts so that proofs
/// can be bound to the exact program they were produced for.
pub fn prepared_program_identity_digest<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
) -> String {
    let mut hash = FNV1A64_OFFSET;
    for row in prepared_program_identity_rows::<F>(program) {
        hash = fnv1a64_update(hash, row.as_bytes());
        hash = fnv1a64_update(hash, b"\n");
    }
    format!("{hash:016x}")
}

/// Build a validation-plan descriptor with a canonical-bytes-sha256 fingerprint.
///
/// `fingerprint_identity_prefix` and `artifact_identity_prefix` are expanded
/// against `identity` via [`prepared_identity`].
pub fn validation_plan(
    identity: &str,
    kind: PreparedValidationKind,
    problem: ProblemKind,
    plan_id: &'static str,
    fingerprint_id: &'static str,
    canonicalization: &'static str,
    fingerprint_policy: &'static str,
    fingerprint_identity_prefix: &'static str,
    artifact_identity_prefix: &'static str,
) -> PreparedValidationPlanDescriptor {
    PreparedValidationPlanDescriptor::new(plan_id, kind, problem)
        .with_fingerprint(
            PreparedFingerprintDescriptor::new(
                fingerprint_id,
                PreparedFingerprintScheme::CanonicalBytesSha256,
                canonicalization,
            )
            .with_fingerprint_policy_identity(fingerprint_policy)
            .with_fingerprint_identity(prepared_identity(fingerprint_identity_prefix, identity)),
        )
        .with_artifact_identity(prepared_identity(artifact_identity_prefix, identity))
}

fn prepared_program_identity_rows<F: HardwareFrontend>(
    program: &PreparedCheckerProgram,
) -> Vec<String> {
    let mut rows = Vec::new();
    rows.push(program.render_evidence_row(F::LABEL));
    rows.extend(program.render_frontend_extension_evidence_rows(F::LABEL));
    rows.extend(program.render_candidate_lane_evidence_rows(F::LABEL));
    rows.extend(program.render_validation_plan_evidence_rows(F::LABEL));
    rows
}

/// Expand a `prefix:<token>` prepared identity, normalizing `identity` to an
/// evidence token.
pub fn prepared_identity(prefix: &str, identity: &str) -> String {
    format!("{prefix}:{}", evidence_token(identity))
}

/// Render an optional value as an evidence token, or `"none"` when absent.
pub fn evidence_option(value: Option<&str>) -> String {
    value
        .map(evidence_token)
        .unwrap_or_else(|| "none".to_string())
}

/// Normalize a value into an evidence token (ASCII-alphanumeric plus
/// `-_.:=` survive; everything else becomes `_`; empty becomes `"none"`).
pub fn evidence_token(value: &str) -> String {
    if value.is_empty() {
        return String::from("none");
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

const FNV1A64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}
