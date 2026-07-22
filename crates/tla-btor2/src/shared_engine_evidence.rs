// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared-engine adoption evidence for the BTOR2 hardware frontend.
//!
//! These rows bind the BTOR2 lowering lane to the same prepared-program, AY
//! proof, replay, and hardware fingerprint contracts used by other frontends.
//! The frontend-neutral machinery lives in `tla-hw-evidence`; this module wires
//! the BTOR2 frontend (its names/constants and its program builder) into that
//! shared builder and re-exposes the BTOR2-named public API.

use tla_ay::{
    AYFrontendFamily, AYProofValidationReceipt, AYSharedEngineLane, AYSharedProofLaneDescriptor,
};
use tla_hw_evidence::{
    ay_proof_lane_adoption_evidence_row, ay_proof_lane_descriptor, ay_proof_lane_receipt,
    ay_witness_lane_receipt, prepared_identity, prepared_program_identity_digest,
    register_vector_admission_base_plan, register_vector_admission_plan, validation_plan,
    HardwareFrontend, SharedEngineEvidence,
};
use tla_mc_core::CheckerSourceKind;
use tla_mc_core::{
    BackendKind, PreparedAnalyticalSolveKind, PreparedBackendFamilyDescriptor,
    PreparedCandidateLaneDescriptor, PreparedCanonicalIdentityDescriptor,
    PreparedCanonicalIdentityKind, PreparedCheckerProgram, PreparedFingerprintAdmissionPlan,
    PreparedFrontendExtensionDescriptor, PreparedFrontendExtensionKind, PreparedProgramPayloadKind,
    PreparedPropertyKind, PreparedStorageKind, PreparedSymbolicProofKind, PreparedTransitionKind,
    PreparedValidationKind, ProblemKind, SetupTraceLaneKind, SolverFacet,
};

use crate::types::{Btor2Node, Btor2Program};

const BTOR2_REGISTER_LAYOUT_IDENTITY: &str = "btor2.hardware_register_layout.v1";
const BTOR2_PROOF_FINGERPRINT_POLICY: &str = "hardware_ay_proof_fingerprint.v1";
const BTOR2_REPLAY_FINGERPRINT_POLICY: &str = "hardware_replay_fingerprint.v1";
const BTOR2_PREPARED_CANONICALIZATION: &str = "btor2-hardware-safety-v1";
const BTOR2_STATE_CANONICALIZATION: &str = "btor2-state-vector-v1";
const BTOR2_AY_PROOF_CANONICALIZATION: &str = "ay-chc-proof-v1";
const BTOR2_REPLAY_CANONICALIZATION: &str = "btor2-counterexample-replay-v1";
const BTOR2_PREPARED_CANONICAL_IDENTITY: &str = "btor2.prepared_program";
const BTOR2_AY_LANE_IDENTITY: &str = "shared_ay_chc";
const BTOR2_AY_BMC_LANE_IDENTITY: &str = "shared_ay_bmc";
const BTOR2_AY_PDR_LANE_IDENTITY: &str = "shared_ay_pdr";
const BTOR2_AY_K_INDUCTION_LANE_IDENTITY: &str = "shared_ay_k_induction";
const BTOR2_REPLAY_LANE_IDENTITY: &str = "shared_hardware_proof_replay";
const BTOR2_FINGERPRINT_LANE_IDENTITY: &str = "shared_hardware_state_fingerprint";
const BTOR2_AY_PROOF_OBLIGATION_IDENTITY: &str = "btor2.hardware_register_safety_obligation.v1";

/// Frontend marker binding the shared hardware evidence builder to BTOR2.
pub struct Btor2Frontend;

impl HardwareFrontend for Btor2Frontend {
    type Input = Btor2Program;

    const LABEL: &'static str = "BTOR2";
    const ORIGIN_FRONTEND: &'static str = "btor2";
    const SHARED_ENGINE_COMPONENT: &'static str = "tla_mc_core.prepared_checker_program";
    const SHARED_ENGINE_OWNER: &'static str = "shared_high_performance_engine";
    const PORTFOLIO: &'static str = "btor2_portfolio";
    const SHARED_ENGINE_SECOND_BENEFICIARY: &'static str = "aiger_portfolio";
    const SHARED_ENGINE_EXTRACTION_STATUS: &'static str = "shared-core-ready";
    const ACCEPTANCE_TEST: &'static str = "cargo test -p tla-btor2 shared_engine_evidence";
    const PREPARED_PROGRAM_DIGEST_ALGORITHM: &'static str = "fnv1a64";

    const REGISTER_LAYOUT_IDENTITY: &'static str = BTOR2_REGISTER_LAYOUT_IDENTITY;
    const STATE_CANONICALIZATION: &'static str = BTOR2_STATE_CANONICALIZATION;

    const ADMISSION_DESCRIPTION: &'static str = "btor2 register-vector prepared admission";
    const CHECKER_SOURCE_KIND: CheckerSourceKind = CheckerSourceKind::Btor2;
    const PROGRAM_PAYLOAD_KIND: PreparedProgramPayloadKind = PreparedProgramPayloadKind::Btor2;

    const AY_PROOF_RECEIPT_PREREQUISITE: &'static str = "ay chc proof receipt";

    const AY_SAFETY_CANDIDATE_PREFIX: &'static str = "btor2.ay_chc.safety_candidate";
    const AY_PROOF_ARTIFACT_PREFIX: &'static str = "btor2.ay_chc.proof_artifact";
    const AY_PROOF_FINGERPRINT_PREFIX: &'static str = "btor2.ay_chc.proof";
    const REPLAY_COUNTEREXAMPLE_CANDIDATE_PREFIX: &'static str =
        "btor2.replay.counterexample_candidate";
    const REPLAY_COUNTEREXAMPLE_ARTIFACT_PREFIX: &'static str =
        "btor2.replay.counterexample_artifact";
    const REPLAY_COUNTEREXAMPLE_PREFIX: &'static str = "btor2.replay.counterexample";

    const AY_SHARED_ENGINE_LANE: AYSharedEngineLane = AYSharedEngineLane::Chc;
    const AY_FRONTEND_FAMILY: AYFrontendFamily = AYFrontendFamily::Btor2;
    const AY_PROOF_OBLIGATION_IDENTITY: &'static str = BTOR2_AY_PROOF_OBLIGATION_IDENTITY;
    const AY_PROOF_LANE_RECEIPT_IDENTITY: &'static str =
        "btor2.hardware_ay_proof_lane.validation_receipt";
    const AY_WITNESS_LANE_RECEIPT_IDENTITY: &'static str =
        "btor2.hardware_ay_witness_lane.validation_receipt";
    const AY_PROOF_LANE_FIRST_BENEFICIARY: &'static str = "btor2_hardware_register_vector";
    const AY_PROOF_LANE_SECOND_BENEFICIARY: &'static str = "replay_shared_ay_proof_lanes";

    fn prepared_checker_program(input: &Self::Input) -> PreparedCheckerProgram {
        btor2_prepared_checker_program(input)
    }
}

/// BTOR2 evidence bundle for the shared prepared-program adoption contract.
///
/// Construct via [`Btor2SharedEngineEvidence::from_input`]; render rows via
/// [`SharedEngineEvidence::render_evidence_rows`] or
/// [`btor2_shared_engine_evidence_rows`].
pub type Btor2SharedEngineEvidence = SharedEngineEvidence<Btor2Frontend>;

/// Build the BTOR2 shared prepared-program descriptor.
pub fn btor2_prepared_checker_program(program: &Btor2Program) -> PreparedCheckerProgram {
    let identity = btor2_program_identity(program);
    let register_vector_admission = btor2_register_vector_admission_base_plan();
    let register_storage_policy_identity =
        register_vector_admission.dedup.storage_policy_identity();
    let register_fingerprint_policy_identity = register_vector_admission
        .dedup
        .fingerprint
        .fingerprint_policy_identity();
    let register_fingerprint_identity = register_vector_admission
        .dedup
        .fingerprint
        .fingerprint_identity();
    let mut prepared = PreparedCheckerProgram::new(
        identity.clone(),
        PreparedProgramPayloadKind::Btor2,
        PreparedStorageKind::HardwareRegisters,
    )
    .with_canonical_payload_identity(btor2_prepared_identity(
        "btor2.canonical_payload",
        &identity,
    ))
    .with_source_identity(btor2_prepared_identity("btor2.source", &identity))
    .with_config_identity(btor2_prepared_identity("btor2.config", "default"))
    .with_examination_identity(btor2_prepared_identity("btor2.examination", "safety"))
    .with_cache_key(btor2_prepared_identity("btor2.prepared.cache", &identity))
    .with_source_fingerprint(btor2_prepared_identity(
        "btor2.source_fingerprint",
        &identity,
    ))
    .with_frontend_payload_identity(btor2_prepared_identity("btor2.payload", &identity))
    .with_frontend_payload_fingerprint(btor2_prepared_identity(
        "btor2.payload_fingerprint",
        &identity,
    ))
    .with_artifact_identity(btor2_prepared_identity("btor2.prepared_program", &identity))
    .with_storage_layout_fingerprint(BTOR2_REGISTER_LAYOUT_IDENTITY)
    .with_storage_policy_identity(register_storage_policy_identity.clone())
    .with_fingerprint_policy_identity(register_fingerprint_policy_identity.clone())
    .with_fingerprint_identity(register_fingerprint_identity.clone())
    .with_transition_descriptor_fingerprint(btor2_prepared_identity(
        "btor2.transition_descriptor",
        &identity,
    ))
    .with_property_descriptor_fingerprint(btor2_prepared_identity(
        "btor2.property_descriptor",
        &identity,
    ))
    .with_validation_plan_fingerprint(btor2_prepared_identity("btor2.validation_plan", &identity))
    .with_fingerprint(register_vector_admission.prepared_fingerprint_descriptor())
    .add_frontend_extension(
        PreparedFrontendExtensionDescriptor::new(
            "btor2.frontend_adapter",
            PreparedFrontendExtensionKind::Btor2,
            ProblemKind::Safety,
        )
        .with_frontend_payload_identity(btor2_prepared_identity("btor2.payload", &identity))
        .with_artifact_identity(btor2_prepared_identity("btor2.adapter", &identity))
        .with_storage_policy_identity(register_storage_policy_identity.clone())
        .with_fingerprint_policy_identity(register_fingerprint_policy_identity.clone())
        .with_fingerprint_identity(register_fingerprint_identity.clone()),
    );

    let next_count = btor2_next_count(program).max(1);
    for index in 0..next_count {
        prepared = prepared.add_transition(
            format!("btor2.next_state.{index}"),
            PreparedTransitionKind::HardwareNextState,
        );
    }
    for (index, property_id) in program.bad_properties.iter().enumerate() {
        prepared = prepared.add_property(
            format!("btor2.bad.{index}.line.{property_id}"),
            PreparedPropertyKind::BadState,
        );
    }

    prepared = prepared
        .add_analytical_solve(
            "btor2.ay_chc.bmc",
            PreparedAnalyticalSolveKind::BoundedModelCheck,
            ProblemKind::Bmc,
        )
        .add_analytical_solve(
            "btor2.ay_chc.pdr",
            PreparedAnalyticalSolveKind::PdrSafety,
            ProblemKind::Chc,
        )
        .add_analytical_solve(
            "btor2.ay_chc.k_induction",
            PreparedAnalyticalSolveKind::KInduction,
            ProblemKind::Chc,
        )
        .add_symbolic_proof(
            "btor2.ay_chc.query",
            PreparedSymbolicProofKind::ChcQuery,
            ProblemKind::Chc,
        )
        .add_symbolic_proof(
            "btor2.ay_chc.pdr_proof",
            PreparedSymbolicProofKind::PdrSafetyProof,
            ProblemKind::Chc,
        )
        .add_symbolic_proof(
            "btor2.ay_chc.proof_certificate",
            PreparedSymbolicProofKind::ProofCertificate,
            ProblemKind::Chc,
        )
        .add_symbolic_proof(
            "btor2.ay_chc.model_extraction",
            PreparedSymbolicProofKind::ModelExtraction,
            ProblemKind::Chc,
        )
        .add_backend_family(
            PreparedBackendFamilyDescriptor::new(
                "btor2.ay_chc",
                BackendKind::AYChc,
                ProblemKind::Chc,
            )
            .with_facet(SolverFacet::InProcess)
            .with_facet(SolverFacet::BitVector)
            .with_facet(SolverFacet::Chc)
            .with_facet(SolverFacet::Bmc)
            .with_facet(SolverFacet::Pdr)
            .with_facet(SolverFacet::KInduction)
            .with_facet(SolverFacet::Proof)
            .with_facet(SolverFacet::Witness),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("btor2.ay_chc.safety", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_chc_safety")
                .with_candidate_identity(btor2_prepared_identity(
                    "btor2.ay_chc.safety_candidate",
                    &identity,
                ))
                .with_lane_identity(BTOR2_AY_LANE_IDENTITY)
                .with_fingerprint_policy_identity(BTOR2_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(btor2_prepared_identity(
                    "btor2.ay_chc.proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("btor2.ay_chc.chc", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_chc")
                .with_candidate_identity(btor2_prepared_identity(
                    "btor2.ay_chc.chc_candidate",
                    &identity,
                ))
                .with_lane_identity(BTOR2_AY_LANE_IDENTITY)
                .with_fingerprint_policy_identity(BTOR2_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(btor2_prepared_identity(
                    "btor2.ay_chc.chc_proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("btor2.ay_chc.bmc", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_bmc")
                .with_candidate_identity(btor2_prepared_identity(
                    "btor2.ay_chc.bmc_candidate",
                    &identity,
                ))
                .with_lane_identity(BTOR2_AY_BMC_LANE_IDENTITY)
                .with_fingerprint_policy_identity(BTOR2_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(btor2_prepared_identity(
                    "btor2.ay_chc.bmc_proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("btor2.ay_chc.pdr", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_pdr")
                .with_candidate_identity(btor2_prepared_identity(
                    "btor2.ay_chc.pdr_candidate",
                    &identity,
                ))
                .with_lane_identity(BTOR2_AY_PDR_LANE_IDENTITY)
                .with_fingerprint_policy_identity(BTOR2_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(btor2_prepared_identity(
                    "btor2.ay_chc.pdr_proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "btor2.ay_chc.k_induction",
                SetupTraceLaneKind::AY,
            )
            .with_candidate_key("ay_k_induction")
            .with_candidate_identity(btor2_prepared_identity(
                "btor2.ay_chc.k_induction_candidate",
                &identity,
            ))
            .with_lane_identity(BTOR2_AY_K_INDUCTION_LANE_IDENTITY)
            .with_fingerprint_policy_identity(BTOR2_PROOF_FINGERPRINT_POLICY)
            .with_fingerprint_identity(btor2_prepared_identity(
                "btor2.ay_chc.k_induction_proof",
                &identity,
            )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "btor2.replay.counterexample",
                SetupTraceLaneKind::Replay,
            )
            .with_candidate_key("ay_chc_verified_result_replay")
            .with_candidate_identity(btor2_prepared_identity(
                "btor2.replay.counterexample_candidate",
                &identity,
            ))
            .with_lane_identity(BTOR2_REPLAY_LANE_IDENTITY)
            .with_fingerprint_policy_identity(BTOR2_REPLAY_FINGERPRINT_POLICY)
            .with_fingerprint_identity(btor2_prepared_identity(
                "btor2.replay.counterexample",
                &identity,
            )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "btor2.fingerprint.hardware_state",
                SetupTraceLaneKind::Fingerprint,
            )
            .with_candidate_key("hardware_state_fingerprint")
            .with_candidate_identity(btor2_prepared_identity(
                "btor2.fingerprint.state",
                &identity,
            ))
            .with_lane_identity(BTOR2_FINGERPRINT_LANE_IDENTITY)
            .with_storage_policy_identity(register_storage_policy_identity.clone())
            .with_fingerprint_policy_identity(register_fingerprint_policy_identity.clone())
            .with_fingerprint_identity(register_fingerprint_identity.clone()),
        )
        .add_validation_plan(btor2_validation_plan(
            &identity,
            PreparedValidationKind::AYProof,
            ProblemKind::Chc,
            "btor2.validation.ay_proof",
            "btor2.ay_chc.proof_fingerprint",
            BTOR2_AY_PROOF_CANONICALIZATION,
            BTOR2_PROOF_FINGERPRINT_POLICY,
            "btor2.ay_chc.proof",
            "btor2.ay_chc.proof_artifact",
        ))
        .add_validation_plan(btor2_validation_plan(
            &identity,
            PreparedValidationKind::WitnessReplay,
            ProblemKind::Safety,
            "btor2.validation.counterexample_replay",
            "btor2.replay.counterexample_fingerprint",
            BTOR2_REPLAY_CANONICALIZATION,
            BTOR2_REPLAY_FINGERPRINT_POLICY,
            "btor2.replay.counterexample",
            "btor2.replay.counterexample_artifact",
        ))
        .add_validation_plan(btor2_validation_plan(
            &identity,
            PreparedValidationKind::OutputFormat,
            ProblemKind::Safety,
            "btor2.validation.output_format",
            "btor2.output.format_fingerprint",
            BTOR2_PREPARED_CANONICALIZATION,
            "hardware_output_format_fingerprint.v1",
            "btor2.output.format",
            "btor2.output.format_artifact",
        ))
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            BTOR2_PREPARED_CANONICAL_IDENTITY,
            PreparedCanonicalIdentityKind::PreparedProgram,
            BTOR2_PREPARED_CANONICALIZATION,
        ));

    prepared
}

/// Render all BTOR2 shared-engine adoption rows for a program.
pub fn btor2_shared_engine_evidence_rows(program: &Btor2Program) -> Vec<String> {
    Btor2SharedEngineEvidence::from_input(program).render_evidence_rows()
}

/// Build the shared prepared fingerprint admission plan for BTOR2 registers.
pub fn btor2_register_vector_admission_plan(
    program: &PreparedCheckerProgram,
) -> PreparedFingerprintAdmissionPlan {
    register_vector_admission_plan::<Btor2Frontend>(program)
}

fn btor2_register_vector_admission_base_plan() -> PreparedFingerprintAdmissionPlan {
    register_vector_admission_base_plan::<Btor2Frontend>()
}

/// Build the generalized AY proof-lane descriptor for BTOR2 register-vector safety.
pub fn btor2_ay_proof_lane_descriptor(
    program: &PreparedCheckerProgram,
) -> AYSharedProofLaneDescriptor {
    ay_proof_lane_descriptor::<Btor2Frontend>(program)
}

/// Build a validator-backed receipt for the generalized BTOR2 AY proof lane.
pub fn btor2_ay_proof_lane_receipt(program: &PreparedCheckerProgram) -> AYProofValidationReceipt {
    ay_proof_lane_receipt::<Btor2Frontend>(program)
}

/// Build a validator-backed receipt for the generalized BTOR2 witness lane.
pub fn btor2_ay_witness_lane_receipt(program: &PreparedCheckerProgram) -> AYProofValidationReceipt {
    ay_witness_lane_receipt::<Btor2Frontend>(program)
}

/// Render generalized AY proof-lane adoption only after receipt validation succeeds.
pub fn btor2_ay_proof_lane_adoption_evidence_row(
    program: &PreparedCheckerProgram,
    proof_receipt: Option<&AYProofValidationReceipt>,
    witness_receipt: Option<&AYProofValidationReceipt>,
) -> Option<String> {
    ay_proof_lane_adoption_evidence_row::<Btor2Frontend>(program, proof_receipt, witness_receipt)
}

/// Stable FNV-1a digest over the prepared-program identity rows.
pub fn btor2_prepared_program_identity_digest(program: &PreparedCheckerProgram) -> String {
    prepared_program_identity_digest::<Btor2Frontend>(program)
}

fn btor2_validation_plan(
    identity: &str,
    kind: PreparedValidationKind,
    problem: ProblemKind,
    plan_id: &'static str,
    fingerprint_id: &'static str,
    canonicalization: &'static str,
    fingerprint_policy: &'static str,
    fingerprint_identity_prefix: &'static str,
    artifact_identity_prefix: &'static str,
) -> tla_mc_core::PreparedValidationPlanDescriptor {
    validation_plan(
        identity,
        kind,
        problem,
        plan_id,
        fingerprint_id,
        canonicalization,
        fingerprint_policy,
        fingerprint_identity_prefix,
        artifact_identity_prefix,
    )
}

fn btor2_program_identity(program: &Btor2Program) -> String {
    format!(
        "btor2:safety:lines={}:sorts={}:inputs={}:states={}:next={}:bad={}:constraints={}:fairness={}:justice={}",
        program.lines.len(),
        program.sorts.len(),
        program.num_inputs,
        program.num_states,
        btor2_next_count(program),
        program.bad_properties.len(),
        program.constraints.len(),
        program.fairness.len(),
        program.justice.len(),
    )
}

fn btor2_next_count(program: &Btor2Program) -> usize {
    program
        .lines
        .iter()
        .filter(|line| matches!(&line.node, Btor2Node::Next(..)))
        .count()
}

fn btor2_prepared_identity(prefix: &str, identity: &str) -> String {
    prepared_identity(prefix, identity)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::portfolio::{btor2_portfolio_capability_report, PortfolioConfig};
    use crate::types::{Btor2Line, Btor2Node, Btor2Sort};
    use tla_ay::AYProofValidationReceiptStatus;
    use tla_mc_core::{
        validate_prepared_candidate_lane_evidence_row,
        validate_prepared_checker_program_evidence_row,
        validate_prepared_frontend_extension_evidence_row,
        validate_prepared_validation_plan_evidence_row,
        validate_shared_engine_adoption_evidence_row, validate_validation_receipt_evidence_row,
        PreparedProgramPayloadKind, PreparedStorageKind, ValidationReceiptStatus,
    };

    #[test]
    fn btor2_shared_engine_evidence_binds_second_beneficiary_and_receipts() {
        let program = counter_program();
        let evidence = Btor2SharedEngineEvidence::from_input(&program);

        evidence.adoption.validate().unwrap();
        evidence.register_vector_admission.validate().unwrap();
        evidence.ay_proof_receipt.validate().unwrap();
        evidence.replay_receipt.validate().unwrap();
        assert!(btor2_ay_proof_lane_descriptor(&evidence.prepared_program)
            .can_publish_with_receipt(Some(&evidence.ay_proof_lane_receipt)));
        assert!(btor2_ay_proof_lane_descriptor(&evidence.prepared_program)
            .can_publish_with_receipt(Some(&evidence.ay_witness_lane_receipt)));
        assert_eq!(
            evidence.prepared_program.payload_kind,
            PreparedProgramPayloadKind::Btor2
        );
        assert_eq!(
            evidence.prepared_program.storage_kind,
            PreparedStorageKind::HardwareRegisters
        );
        assert_eq!(
            evidence.register_vector_admission.payload_witness.code(),
            "register_vector_canonical"
        );
        assert_eq!(
            evidence
                .register_vector_admission
                .prepared_program_identity
                .as_deref(),
            Some(evidence.prepared_program.identity.as_str())
        );
        assert_eq!(
            evidence.register_vector_admission.candidate_key.as_deref(),
            Some("hardware_state_fingerprint")
        );
        assert_eq!(evidence.prepared_program.frontend_extensions.len(), 1);
        assert_eq!(evidence.prepared_program.properties.len(), 1);
        assert_eq!(
            evidence.ay_proof_receipt.status,
            ValidationReceiptStatus::Accepted
        );
        assert_eq!(
            evidence.replay_receipt.status,
            ValidationReceiptStatus::Accepted
        );
        assert_eq!(
            evidence.ay_proof_receipt.digest,
            evidence.prepared_program_digest
        );
        assert_eq!(evidence.prepared_program_digest.len(), 16);

        let rows = evidence.render_evidence_rows();
        let adoption_row = rows
            .iter()
            .find(|row| row.starts_with("BTOR2 shared_engine_adoption "))
            .expect("adoption row");
        validate_shared_engine_adoption_evidence_row(adoption_row).unwrap();
        assert!(adoption_row.contains("origin_frontend=btor2"));
        assert!(adoption_row.contains("second_beneficiary=aiger_portfolio"));
        assert!(adoption_row.contains("owner=shared_high_performance_engine"));
        assert!(adoption_row.contains("extraction_status=shared-core-ready"));
        assert!(adoption_row.contains("adoption_level=level-3"));
        assert!(adoption_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains("default_compatible_frontend_families=aiger,btor2"));
        assert!(adoption_row.contains(
            "remaining_compatible_frontend_families=tla_plus,quint,mcc_petri,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(adoption_row.contains("blocker_status=tracked-blockers"));
        assert!(!adoption_row.contains("adoption_not_yet_recorded"));

        let register_storage_policy_identity = evidence
            .register_vector_admission
            .dedup
            .storage_policy_identity();
        let register_fingerprint_policy_identity = evidence
            .register_vector_admission
            .dedup
            .fingerprint
            .fingerprint_policy_identity();
        let register_fingerprint_identity = evidence
            .register_vector_admission
            .dedup
            .fingerprint
            .fingerprint_identity();
        let prepared_row = rows
            .iter()
            .find(|row| row.starts_with("BTOR2 prepared_checker_program "))
            .expect("prepared checker program row");
        validate_prepared_checker_program_evidence_row(prepared_row).unwrap();
        assert!(
            prepared_row.contains("payload_kind=btor2")
                && prepared_row.contains("storage_kind=hardware_registers")
                && prepared_row.contains("frontend_extensions=1")
                && prepared_row.contains(&format!(
                    "storage_policy_identity={register_storage_policy_identity}"
                ))
                && prepared_row.contains(&format!(
                    "fingerprint_policy_identity={register_fingerprint_policy_identity}"
                ))
                && prepared_row.contains(&format!(
                    "fingerprint_identity={register_fingerprint_identity}"
                ))
        );
        assert!(prepared_row.contains("fingerprint_id=register_vector_canonical"));
        let extension_row = rows
            .iter()
            .find(|row| row.starts_with("BTOR2 prepared_frontend_extension "))
            .expect("frontend extension row");
        validate_prepared_frontend_extension_evidence_row(extension_row).unwrap();
        assert!(
            extension_row.contains("extension_kind=btor2")
                && extension_row.contains("extension_payload_kind=btor2")
                && extension_row.contains("extension_storage_kind=hardware_registers")
                && extension_row.contains(&format!(
                    "storage_policy_identity={register_storage_policy_identity}"
                ))
                && extension_row.contains(&format!(
                    "fingerprint_policy_identity={register_fingerprint_policy_identity}"
                ))
        );
        let ay_lane_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 prepared_candidate_lane ")
                    && row.contains("lane_kind=ay")
                    && row.contains("candidate_key=ay_chc_safety")
                    && row.contains("lane_identity=shared_ay_chc")
            })
            .expect("ay candidate lane");
        validate_prepared_candidate_lane_evidence_row(ay_lane_row).unwrap();
        let candidate_keys: std::collections::HashSet<&str> = evidence
            .prepared_program
            .candidate_lanes
            .iter()
            .filter_map(|lane| lane.candidate_key.as_deref())
            .collect();
        for expected_key in [
            "ay_chc_safety",
            "ay_chc",
            "ay_bmc",
            "ay_pdr",
            "ay_k_induction",
            "ay_chc_verified_result_replay",
            "hardware_state_fingerprint",
        ] {
            assert!(
                candidate_keys.contains(expected_key),
                "missing candidate key {expected_key}"
            );
        }
        for (candidate_key, lane_identity) in [
            ("ay_chc", BTOR2_AY_LANE_IDENTITY),
            ("ay_bmc", BTOR2_AY_BMC_LANE_IDENTITY),
            ("ay_pdr", BTOR2_AY_PDR_LANE_IDENTITY),
            ("ay_k_induction", BTOR2_AY_K_INDUCTION_LANE_IDENTITY),
        ] {
            let alias_row = rows
                .iter()
                .find(|row| {
                    row.starts_with("BTOR2 prepared_candidate_lane ")
                        && row.contains("lane_kind=ay")
                        && row.contains(&format!("candidate_key={candidate_key}"))
                        && row.contains(&format!("lane_identity={lane_identity}"))
                })
                .expect("canonical shared ay alias lane");
            validate_prepared_candidate_lane_evidence_row(alias_row).unwrap();
        }
        let replay_lane_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 prepared_candidate_lane ")
                    && row.contains("lane_kind=replay")
                    && row.contains("candidate_key=ay_chc_verified_result_replay")
                    && row.contains("lane_identity=shared_hardware_proof_replay")
            })
            .expect("replay candidate lane");
        validate_prepared_candidate_lane_evidence_row(replay_lane_row).unwrap();
        let fingerprint_lane_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 prepared_candidate_lane ")
                    && row.contains("lane_kind=fingerprint")
                    && row.contains("candidate_key=hardware_state_fingerprint")
                    && row.contains("lane_identity=shared_hardware_state_fingerprint")
            })
            .expect("fingerprint candidate lane");
        validate_prepared_candidate_lane_evidence_row(fingerprint_lane_row).unwrap();
        assert!(fingerprint_lane_row.contains(&format!(
            "storage_policy_identity={register_storage_policy_identity}"
        )));
        assert!(fingerprint_lane_row.contains(&format!(
            "fingerprint_policy_identity={register_fingerprint_policy_identity}"
        )));
        for row in rows
            .iter()
            .filter(|row| row.starts_with("BTOR2 prepared_candidate_lane "))
        {
            validate_prepared_candidate_lane_evidence_row(row).unwrap();
        }
        let validation_plan_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 prepared_validation_plan ")
                    && row.contains("validation_kind=ay_proof")
                    && row.contains("fingerprint_scheme=canonical_bytes_sha256")
            })
            .expect("ay proof validation plan");
        validate_prepared_validation_plan_evidence_row(validation_plan_row).unwrap();
        assert!(
            validation_plan_row.contains("required=true")
                && validation_plan_row.contains("fail_closed=true")
        );
        for row in rows
            .iter()
            .filter(|row| row.starts_with("BTOR2 prepared_validation_plan "))
        {
            validate_prepared_validation_plan_evidence_row(row).unwrap();
        }
        let ay_receipt_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 validation_receipt ")
                    && row.contains("validator_kind=ay_proof")
                    && row.contains("digest_algorithm=fnv1a64")
                    && row.contains("validation_artifact_kind=proof")
            })
            .expect("ay proof receipt");
        validate_validation_receipt_evidence_row(ay_receipt_row).unwrap();
        let replay_receipt_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 validation_receipt ")
                    && row.contains("validator_kind=proof_replay")
                    && row.contains("validation_artifact_kind=witness")
            })
            .expect("proof replay receipt");
        validate_validation_receipt_evidence_row(replay_receipt_row).unwrap();

        let shared_fingerprint_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 shared_fingerprint_identity ")
                    && row.contains("source_kind=btor2")
                    && row.contains("value_kind=register_vector")
            })
            .expect("shared register-vector fingerprint row");
        assert!(shared_fingerprint_row.contains("frontend_family_reusable=true"));
        assert!(shared_fingerprint_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        let shared_dedup_row = rows
            .iter()
            .find(|row| {
                row.starts_with("BTOR2 shared_dedup_identity ")
                    && row.contains("source_kind=btor2")
                    && row.contains("fingerprint_value_kind=register_vector")
            })
            .expect("shared register-vector dedup row");
        assert!(shared_dedup_row.contains("storage_kind=cas"));
        assert!(shared_dedup_row.contains("collision_policy=canonical_payload_equality"));
        assert!(shared_dedup_row.contains("collision_fail_closed=true"));
        assert!(rows.iter().any(|row| {
            row.starts_with("BTOR2 shared_dedup_identity_validation ")
                && row.contains("status_code=accepted")
                && row.contains("fail_closed=true")
        }));
        let admission_row = rows
            .iter()
            .find(|row| row.starts_with("BTOR2 prepared_fingerprint_admission "))
            .expect("prepared fingerprint admission row");
        assert!(admission_row.contains("shared_engine_component=prepared_fingerprint_admission"));
        assert!(admission_row.contains("payload_witness=register_vector_canonical"));
        assert!(admission_row.contains("admission_status=accepted"));
        assert!(admission_row.contains("default_consumers=aiger,btor2"));
        assert!(admission_row.contains(
            "remaining_compatible_frontend_families=vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(admission_row
            .contains("blockers=future_importer:awaiting_registered_importer_frontend"));
        let transition_row = rows
            .iter()
            .find(|row| row.starts_with("BTOR2 hardware_transition_system_adoption "))
            .expect("hardware transition-system row");
        assert!(transition_row.contains("shared_engine_component=prepared_checker_program"));
        assert!(transition_row.contains("transition_kind=hardware_next_state"));
        assert!(transition_row.contains("ay_analytical_lane=receipt_backed"));
        assert!(transition_row.contains("witness_replay_lane=receipt_backed"));
        assert!(transition_row.contains("default_consumers=aiger,btor2"));

        let proof_lane_adoption_row = rows
            .iter()
            .find(|row| row.starts_with("BTOR2 hardware_ay_proof_lane_adoption "))
            .expect("generalized AY proof-lane adoption row");
        assert!(proof_lane_adoption_row.contains("origin_frontend=btor2"));
        assert!(proof_lane_adoption_row.contains("shared_engine_component=analytical_ay_proof"));
        assert!(proof_lane_adoption_row.contains("lane=chc"));
        assert!(proof_lane_adoption_row
            .contains("proof_obligation_identity=btor2.hardware_register_safety_obligation.v1"));
        assert!(proof_lane_adoption_row.contains("prepared_program_identity=btor2:safety:"));
        assert!(proof_lane_adoption_row
            .contains("register_vector_identity=btor2.hardware_register_layout.v1"));
        assert!(proof_lane_adoption_row.contains(
            "validation_receipt_requirement=validator_backed_receipt_for_proof_obligation_fingerprint"
        ));
        assert!(proof_lane_adoption_row.contains("validation_status=validator_backed"));
        assert!(proof_lane_adoption_row
            .contains("proof_receipt_identity=btor2.hardware_ay_proof_lane.validation_receipt"));
        assert!(proof_lane_adoption_row.contains("proof_receipt_validation_kind=proof_transcript"));
        assert!(proof_lane_adoption_row.contains("proof_receipt_status=validator_backed"));
        assert!(proof_lane_adoption_row
            .contains("proof_receipt_validated_fingerprint_identity=btor2.ay_chc.proof:"));
        assert!(proof_lane_adoption_row.contains(
            "witness_receipt_identity=btor2.hardware_ay_witness_lane.validation_receipt"
        ));
        assert!(proof_lane_adoption_row.contains("witness_receipt_validation_kind=witness"));
        assert!(proof_lane_adoption_row.contains("witness_receipt_status=validator_backed"));
        assert!(proof_lane_adoption_row.contains(
            "witness_receipt_validated_fingerprint_identity=btor2.replay.counterexample:"
        ));
        assert!(
            proof_lane_adoption_row.contains("first_beneficiary=btor2_hardware_register_vector")
        );
        assert!(proof_lane_adoption_row.contains("second_beneficiary=replay_shared_ay_proof_lanes"));
        assert!(proof_lane_adoption_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(proof_lane_adoption_row.contains("extraction_status=shared-core-ready"));
        assert!(proof_lane_adoption_row.contains("blocker_status=tracked-blockers"));
        assert!(proof_lane_adoption_row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
    }

    #[test]
    fn btor2_ay_proof_lane_publication_fails_closed_without_validator_receipt() {
        let program = counter_program();
        let evidence = Btor2SharedEngineEvidence::from_input(&program);
        let artifact_only_receipt = evidence
            .ay_proof_lane_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_witness_receipt = evidence
            .ay_witness_lane_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);

        assert!(btor2_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            None,
            Some(&evidence.ay_witness_lane_receipt)
        )
        .is_none());
        assert!(btor2_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            Some(&artifact_only_receipt),
            Some(&evidence.ay_witness_lane_receipt)
        )
        .is_none());
        assert!(btor2_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            Some(&evidence.ay_proof_lane_receipt),
            None
        )
        .is_none());
        assert!(btor2_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            Some(&evidence.ay_proof_lane_receipt),
            Some(&artifact_only_witness_receipt)
        )
        .is_none());
    }

    #[test]
    fn btor2_capability_report_publishes_shared_engine_evidence() {
        let program = counter_program();
        let report = btor2_portfolio_capability_report(&program, &PortfolioConfig::default());

        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 shared_engine_adoption ")
                && row.contains("second_beneficiary=aiger_portfolio")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 prepared_checker_program ")
                && row.contains("payload_kind=btor2")
                && row.contains("frontend_extensions=1")
                && row.contains("candidate_lanes=7")
        }));
        for candidate_key in [
            "ay_chc_safety",
            "ay_chc",
            "ay_bmc",
            "ay_pdr",
            "ay_k_induction",
        ] {
            assert!(report.evidence.iter().any(|row| {
                row.starts_with("BTOR2 prepared_candidate_lane ")
                    && row.contains(&format!("candidate_key={candidate_key}"))
            }));
        }
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 prepared_frontend_extension ")
                && row.contains("extension_kind=btor2")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 shared_dedup_identity ")
                && row.contains("fingerprint_value_kind=register_vector")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 prepared_fingerprint_admission ")
                && row.contains("default_consumers=aiger,btor2")
                && row.contains("admission_status=accepted")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 hardware_transition_system_adoption ")
                && row.contains("ay_analytical_lane=receipt_backed")
                && row.contains("witness_replay_lane=receipt_backed")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 validation_receipt ") && row.contains("validator_kind=ay_proof")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("BTOR2 hardware_ay_proof_lane_adoption ")
                && row.contains("second_beneficiary=replay_shared_ay_proof_lanes")
                && row.contains("witness_receipt_validation_kind=witness")
                && row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri")
        }));
    }

    fn counter_program() -> Btor2Program {
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(8));
        sorts.insert(10, Btor2Sort::BitVec(1));

        Btor2Program {
            lines: vec![
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
            ],
            sorts,
            num_inputs: 0,
            num_states: 1,
            bad_properties: vec![11],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        }
    }
}
