// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared-engine adoption evidence for the AIGER hardware frontend.
//!
//! The rows built here are intentionally frontend-neutral: they bind the AIGER
//! lowering shape to the shared prepared-program, AY proof, replay, and hardware
//! state fingerprint contracts exposed by `tla-mc-core`. The frontend-neutral
//! machinery lives in `tla-hw-evidence`; this module wires the AIGER frontend
//! (its names/constants and its program builder) into that shared builder and
//! re-exposes the AIGER-named public API.

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

use crate::types::AigerCircuit;

const AIGER_REGISTER_LAYOUT_IDENTITY: &str = "aiger.hardware_register_layout.v1";
const AIGER_PROOF_FINGERPRINT_POLICY: &str = "hardware_ay_proof_fingerprint.v1";
const AIGER_REPLAY_FINGERPRINT_POLICY: &str = "hardware_replay_fingerprint.v1";
const AIGER_PREPARED_CANONICALIZATION: &str = "aiger-hardware-safety-v1";
const AIGER_STATE_CANONICALIZATION: &str = "aiger-latch-vector-v1";
const AIGER_AY_PROOF_CANONICALIZATION: &str = "ay-sat-proof-v1";
const AIGER_REPLAY_CANONICALIZATION: &str = "aiger-counterexample-replay-v1";
const AIGER_PREPARED_CANONICAL_IDENTITY: &str = "aiger.prepared_program";
const AIGER_AY_LANE_IDENTITY: &str = "shared_ay_sat";
const AIGER_AY_BMC_LANE_IDENTITY: &str = "shared_ay_bmc";
const AIGER_AY_PDR_LANE_IDENTITY: &str = "shared_ay_pdr";
const AIGER_AY_K_INDUCTION_LANE_IDENTITY: &str = "shared_ay_k_induction";
const AIGER_REPLAY_LANE_IDENTITY: &str = "shared_hardware_proof_replay";
const AIGER_FINGERPRINT_LANE_IDENTITY: &str = "shared_hardware_state_fingerprint";
const AIGER_AY_PROOF_OBLIGATION_IDENTITY: &str = "aiger.hardware_register_safety_obligation.v1";

/// Frontend marker binding the shared hardware evidence builder to AIGER.
pub struct AigerFrontend;

impl HardwareFrontend for AigerFrontend {
    type Input = AigerCircuit;

    const LABEL: &'static str = "AIGER";
    const ORIGIN_FRONTEND: &'static str = "aiger";
    const SHARED_ENGINE_COMPONENT: &'static str = "tla_mc_core.prepared_checker_program";
    const SHARED_ENGINE_OWNER: &'static str = "shared_high_performance_engine";
    const PORTFOLIO: &'static str = "aiger_portfolio";
    const SHARED_ENGINE_SECOND_BENEFICIARY: &'static str = "btor2_portfolio";
    const SHARED_ENGINE_EXTRACTION_STATUS: &'static str = "shared-core-ready";
    const ACCEPTANCE_TEST: &'static str = "cargo test -p tla-aiger shared_engine_evidence";
    const PREPARED_PROGRAM_DIGEST_ALGORITHM: &'static str = "fnv1a64";

    const REGISTER_LAYOUT_IDENTITY: &'static str = AIGER_REGISTER_LAYOUT_IDENTITY;
    const STATE_CANONICALIZATION: &'static str = AIGER_STATE_CANONICALIZATION;

    const ADMISSION_DESCRIPTION: &'static str = "aiger register-vector prepared admission";
    const CHECKER_SOURCE_KIND: CheckerSourceKind = CheckerSourceKind::Aiger;
    const PROGRAM_PAYLOAD_KIND: PreparedProgramPayloadKind = PreparedProgramPayloadKind::Aiger;

    const AY_PROOF_RECEIPT_PREREQUISITE: &'static str = "ay sat proof receipt";

    const AY_SAFETY_CANDIDATE_PREFIX: &'static str = "aiger.ay_sat.safety_candidate";
    const AY_PROOF_ARTIFACT_PREFIX: &'static str = "aiger.ay_sat.proof_artifact";
    const AY_PROOF_FINGERPRINT_PREFIX: &'static str = "aiger.ay_sat.proof";
    const REPLAY_COUNTEREXAMPLE_CANDIDATE_PREFIX: &'static str =
        "aiger.replay.counterexample_candidate";
    const REPLAY_COUNTEREXAMPLE_ARTIFACT_PREFIX: &'static str =
        "aiger.replay.counterexample_artifact";
    const REPLAY_COUNTEREXAMPLE_PREFIX: &'static str = "aiger.replay.counterexample";

    const AY_SHARED_ENGINE_LANE: AYSharedEngineLane = AYSharedEngineLane::Bmc;
    const AY_FRONTEND_FAMILY: AYFrontendFamily = AYFrontendFamily::Aiger;
    const AY_PROOF_OBLIGATION_IDENTITY: &'static str = AIGER_AY_PROOF_OBLIGATION_IDENTITY;
    const AY_PROOF_LANE_RECEIPT_IDENTITY: &'static str =
        "aiger.hardware_ay_proof_lane.validation_receipt";
    const AY_WITNESS_LANE_RECEIPT_IDENTITY: &'static str =
        "aiger.hardware_ay_witness_lane.validation_receipt";
    const AY_PROOF_LANE_FIRST_BENEFICIARY: &'static str = "aiger_hardware_register_vector";
    const AY_PROOF_LANE_SECOND_BENEFICIARY: &'static str = "tla_and_mcc_shared_ay_proof_lanes";

    fn prepared_checker_program(input: &Self::Input) -> PreparedCheckerProgram {
        aiger_prepared_checker_program(input)
    }
}

/// AIGER evidence bundle for the shared prepared-program adoption contract.
///
/// Construct via [`AigerSharedEngineEvidence::from_input`]; render rows via
/// [`SharedEngineEvidence::render_evidence_rows`] or
/// [`aiger_shared_engine_evidence_rows`].
pub type AigerSharedEngineEvidence = SharedEngineEvidence<AigerFrontend>;

/// Build the AIGER shared prepared-program descriptor.
pub fn aiger_prepared_checker_program(circuit: &AigerCircuit) -> PreparedCheckerProgram {
    let identity = aiger_circuit_identity(circuit);
    let register_vector_admission = aiger_register_vector_admission_base_plan();
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
    let mut program = PreparedCheckerProgram::new(
        identity.clone(),
        PreparedProgramPayloadKind::Aiger,
        PreparedStorageKind::HardwareRegisters,
    )
    .with_canonical_payload_identity(aiger_prepared_identity(
        "aiger.canonical_payload",
        &identity,
    ))
    .with_source_identity(aiger_prepared_identity("aiger.source", &identity))
    .with_config_identity(aiger_prepared_identity("aiger.config", "default"))
    .with_examination_identity(aiger_prepared_identity("aiger.examination", "safety"))
    .with_cache_key(aiger_prepared_identity("aiger.prepared.cache", &identity))
    .with_source_fingerprint(aiger_prepared_identity(
        "aiger.source_fingerprint",
        &identity,
    ))
    .with_frontend_payload_identity(aiger_prepared_identity("aiger.payload", &identity))
    .with_frontend_payload_fingerprint(aiger_prepared_identity(
        "aiger.payload_fingerprint",
        &identity,
    ))
    .with_artifact_identity(aiger_prepared_identity("aiger.prepared_program", &identity))
    .with_storage_layout_fingerprint(AIGER_REGISTER_LAYOUT_IDENTITY)
    .with_storage_policy_identity(register_storage_policy_identity.clone())
    .with_fingerprint_policy_identity(register_fingerprint_policy_identity.clone())
    .with_fingerprint_identity(register_fingerprint_identity.clone())
    .with_transition_descriptor_fingerprint(aiger_prepared_identity(
        "aiger.transition_descriptor",
        &identity,
    ))
    .with_property_descriptor_fingerprint(aiger_prepared_identity(
        "aiger.property_descriptor",
        &identity,
    ))
    .with_validation_plan_fingerprint(aiger_prepared_identity("aiger.validation_plan", &identity))
    .with_fingerprint(register_vector_admission.prepared_fingerprint_descriptor())
    .add_transition(
        "aiger.next_state",
        PreparedTransitionKind::HardwareNextState,
    )
    .add_frontend_extension(
        PreparedFrontendExtensionDescriptor::new(
            "aiger.frontend_adapter",
            PreparedFrontendExtensionKind::Aiger,
            ProblemKind::Safety,
        )
        .with_frontend_payload_identity(aiger_prepared_identity("aiger.payload", &identity))
        .with_artifact_identity(aiger_prepared_identity("aiger.adapter", &identity))
        .with_storage_policy_identity(register_storage_policy_identity.clone())
        .with_fingerprint_policy_identity(register_fingerprint_policy_identity.clone())
        .with_fingerprint_identity(register_fingerprint_identity.clone()),
    );

    for index in 0..circuit.bad.len() {
        program =
            program.add_property(format!("aiger.bad.{index}"), PreparedPropertyKind::BadState);
    }
    if circuit.bad.is_empty() {
        for index in 0..circuit.outputs.len() {
            program = program.add_property(
                format!("aiger.output.{index}"),
                PreparedPropertyKind::BadState,
            );
        }
    }

    program = program
        .add_analytical_solve(
            "aiger.ay_sat.bmc",
            PreparedAnalyticalSolveKind::BoundedModelCheck,
            ProblemKind::Bmc,
        )
        .add_analytical_solve(
            "aiger.ay_sat.k_induction",
            PreparedAnalyticalSolveKind::KInduction,
            ProblemKind::KInduction,
        )
        .add_analytical_solve(
            "aiger.ay_sat.pdr",
            PreparedAnalyticalSolveKind::PdrSafety,
            ProblemKind::Sat,
        )
        .add_symbolic_proof(
            "aiger.ay_sat.pdr_proof",
            PreparedSymbolicProofKind::PdrSafetyProof,
            ProblemKind::Sat,
        )
        .add_symbolic_proof(
            "aiger.ay_sat.k_induction_proof",
            PreparedSymbolicProofKind::KInduction,
            ProblemKind::KInduction,
        )
        .add_symbolic_proof(
            "aiger.ay_sat.proof_certificate",
            PreparedSymbolicProofKind::ProofCertificate,
            ProblemKind::Sat,
        )
        .add_backend_family(
            PreparedBackendFamilyDescriptor::new(
                "aiger.ay_sat",
                BackendKind::AYSat,
                ProblemKind::Sat,
            )
            .with_facet(SolverFacet::InProcess)
            .with_facet(SolverFacet::Sat)
            .with_facet(SolverFacet::Bmc)
            .with_facet(SolverFacet::KInduction)
            .with_facet(SolverFacet::Pdr)
            .with_facet(SolverFacet::Incremental)
            .with_facet(SolverFacet::Assumptions)
            .with_facet(SolverFacet::Proof)
            .with_facet(SolverFacet::Witness),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("aiger.ay_sat.safety", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_sat_safety")
                .with_candidate_identity(aiger_prepared_identity(
                    "aiger.ay_sat.safety_candidate",
                    &identity,
                ))
                .with_lane_identity(AIGER_AY_LANE_IDENTITY)
                .with_fingerprint_policy_identity(AIGER_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(aiger_prepared_identity(
                    "aiger.ay_sat.proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("aiger.ay_sat.bmc", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_bmc")
                .with_candidate_identity(aiger_prepared_identity(
                    "aiger.ay_sat.bmc_candidate",
                    &identity,
                ))
                .with_lane_identity(AIGER_AY_BMC_LANE_IDENTITY)
                .with_fingerprint_policy_identity(AIGER_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(aiger_prepared_identity(
                    "aiger.ay_sat.bmc_proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("aiger.ay_sat.pdr", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_pdr")
                .with_candidate_identity(aiger_prepared_identity(
                    "aiger.ay_sat.pdr_candidate",
                    &identity,
                ))
                .with_lane_identity(AIGER_AY_PDR_LANE_IDENTITY)
                .with_fingerprint_policy_identity(AIGER_PROOF_FINGERPRINT_POLICY)
                .with_fingerprint_identity(aiger_prepared_identity(
                    "aiger.ay_sat.pdr_proof",
                    &identity,
                )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "aiger.ay_sat.k_induction",
                SetupTraceLaneKind::AY,
            )
            .with_candidate_key("ay_k_induction")
            .with_candidate_identity(aiger_prepared_identity(
                "aiger.ay_sat.k_induction_candidate",
                &identity,
            ))
            .with_lane_identity(AIGER_AY_K_INDUCTION_LANE_IDENTITY)
            .with_fingerprint_policy_identity(AIGER_PROOF_FINGERPRINT_POLICY)
            .with_fingerprint_identity(aiger_prepared_identity(
                "aiger.ay_sat.k_induction_proof",
                &identity,
            )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "aiger.replay.counterexample",
                SetupTraceLaneKind::Replay,
            )
            .with_candidate_key("counterexample_trace_replay")
            .with_candidate_identity(aiger_prepared_identity(
                "aiger.replay.counterexample_candidate",
                &identity,
            ))
            .with_lane_identity(AIGER_REPLAY_LANE_IDENTITY)
            .with_fingerprint_policy_identity(AIGER_REPLAY_FINGERPRINT_POLICY)
            .with_fingerprint_identity(aiger_prepared_identity(
                "aiger.replay.counterexample",
                &identity,
            )),
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "aiger.fingerprint.hardware_state",
                SetupTraceLaneKind::Fingerprint,
            )
            .with_candidate_key("hardware_state_fingerprint")
            .with_candidate_identity(aiger_prepared_identity(
                "aiger.fingerprint.state",
                &identity,
            ))
            .with_lane_identity(AIGER_FINGERPRINT_LANE_IDENTITY)
            .with_storage_policy_identity(register_storage_policy_identity.clone())
            .with_fingerprint_policy_identity(register_fingerprint_policy_identity.clone())
            .with_fingerprint_identity(register_fingerprint_identity.clone()),
        )
        .add_validation_plan(aiger_validation_plan(
            &identity,
            PreparedValidationKind::AYProof,
            ProblemKind::Sat,
            "aiger.validation.ay_proof",
            "aiger.ay_sat.proof_fingerprint",
            AIGER_AY_PROOF_CANONICALIZATION,
            AIGER_PROOF_FINGERPRINT_POLICY,
            "aiger.ay_sat.proof",
            "aiger.ay_sat.proof_artifact",
        ))
        .add_validation_plan(aiger_validation_plan(
            &identity,
            PreparedValidationKind::WitnessReplay,
            ProblemKind::Safety,
            "aiger.validation.counterexample_replay",
            "aiger.replay.counterexample_fingerprint",
            AIGER_REPLAY_CANONICALIZATION,
            AIGER_REPLAY_FINGERPRINT_POLICY,
            "aiger.replay.counterexample",
            "aiger.replay.counterexample_artifact",
        ))
        .add_validation_plan(aiger_validation_plan(
            &identity,
            PreparedValidationKind::OutputFormat,
            ProblemKind::Safety,
            "aiger.validation.output_format",
            "aiger.output.format_fingerprint",
            AIGER_PREPARED_CANONICALIZATION,
            "hardware_output_format_fingerprint.v1",
            "aiger.output.format",
            "aiger.output.format_artifact",
        ))
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            AIGER_PREPARED_CANONICAL_IDENTITY,
            PreparedCanonicalIdentityKind::PreparedProgram,
            AIGER_PREPARED_CANONICALIZATION,
        ));

    program
}

/// Render all AIGER shared-engine adoption rows for a circuit.
pub fn aiger_shared_engine_evidence_rows(circuit: &AigerCircuit) -> Vec<String> {
    AigerSharedEngineEvidence::from_input(circuit).render_evidence_rows()
}

/// Build the shared prepared fingerprint admission plan for AIGER registers.
pub fn aiger_register_vector_admission_plan(
    program: &PreparedCheckerProgram,
) -> PreparedFingerprintAdmissionPlan {
    register_vector_admission_plan::<AigerFrontend>(program)
}

fn aiger_register_vector_admission_base_plan() -> PreparedFingerprintAdmissionPlan {
    register_vector_admission_base_plan::<AigerFrontend>()
}

/// Build the generalized AY proof-lane descriptor for AIGER register-vector safety.
pub fn aiger_ay_proof_lane_descriptor(
    program: &PreparedCheckerProgram,
) -> AYSharedProofLaneDescriptor {
    ay_proof_lane_descriptor::<AigerFrontend>(program)
}

/// Build a validator-backed receipt for the generalized AIGER AY proof lane.
pub fn aiger_ay_proof_lane_receipt(program: &PreparedCheckerProgram) -> AYProofValidationReceipt {
    ay_proof_lane_receipt::<AigerFrontend>(program)
}

/// Build a validator-backed receipt for the generalized AIGER witness lane.
pub fn aiger_ay_witness_lane_receipt(program: &PreparedCheckerProgram) -> AYProofValidationReceipt {
    ay_witness_lane_receipt::<AigerFrontend>(program)
}

/// Render generalized AY proof-lane adoption only after receipt validation succeeds.
pub fn aiger_ay_proof_lane_adoption_evidence_row(
    program: &PreparedCheckerProgram,
    proof_receipt: Option<&AYProofValidationReceipt>,
    witness_receipt: Option<&AYProofValidationReceipt>,
) -> Option<String> {
    ay_proof_lane_adoption_evidence_row::<AigerFrontend>(program, proof_receipt, witness_receipt)
}

/// Stable FNV-1a digest over the prepared-program identity rows.
pub fn aiger_prepared_program_identity_digest(program: &PreparedCheckerProgram) -> String {
    prepared_program_identity_digest::<AigerFrontend>(program)
}

fn aiger_validation_plan(
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

fn aiger_circuit_identity(circuit: &AigerCircuit) -> String {
    format!(
        "aiger:safety:maxvar={}:inputs={}:latches={}:ands={}:bad={}:outputs={}:constraints={}:justice={}:fairness={}",
        circuit.maxvar,
        circuit.inputs.len(),
        circuit.latches.len(),
        circuit.ands.len(),
        circuit.bad.len(),
        circuit.outputs.len(),
        circuit.constraints.len(),
        circuit.justice.len(),
        circuit.fairness.len(),
    )
}

fn aiger_prepared_identity(prefix: &str, identity: &str) -> String {
    prepared_identity(prefix, identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_aag;
    use crate::portfolio::{aiger_portfolio_capability_report, EngineConfig, PortfolioConfig};
    use std::collections::HashSet;
    use std::time::Duration;
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
    fn aiger_shared_engine_evidence_binds_second_beneficiary_and_receipts() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let evidence = AigerSharedEngineEvidence::from_input(&circuit);

        evidence.adoption.validate().unwrap();
        evidence.register_vector_admission.validate().unwrap();
        evidence.ay_proof_receipt.validate().unwrap();
        evidence.replay_receipt.validate().unwrap();
        assert!(aiger_ay_proof_lane_descriptor(&evidence.prepared_program)
            .can_publish_with_receipt(Some(&evidence.ay_proof_lane_receipt)));
        assert!(aiger_ay_proof_lane_descriptor(&evidence.prepared_program)
            .can_publish_with_receipt(Some(&evidence.ay_witness_lane_receipt)));
        assert_eq!(
            evidence.prepared_program.payload_kind,
            PreparedProgramPayloadKind::Aiger
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
            .find(|row| row.starts_with("AIGER shared_engine_adoption "))
            .expect("adoption row");
        validate_shared_engine_adoption_evidence_row(adoption_row).unwrap();
        assert!(adoption_row.contains("origin_frontend=aiger"));
        assert!(adoption_row.contains("second_beneficiary=btor2_portfolio"));
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
            .find(|row| row.starts_with("AIGER prepared_checker_program "))
            .expect("prepared checker program row");
        validate_prepared_checker_program_evidence_row(prepared_row).unwrap();
        assert!(
            prepared_row.contains("payload_kind=aiger")
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
            .find(|row| row.starts_with("AIGER prepared_frontend_extension "))
            .expect("frontend extension row");
        validate_prepared_frontend_extension_evidence_row(extension_row).unwrap();
        assert!(
            extension_row.contains("extension_kind=aiger")
                && extension_row.contains("extension_payload_kind=aiger")
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
                row.starts_with("AIGER prepared_candidate_lane ")
                    && row.contains("lane_kind=ay")
                    && row.contains("candidate_key=ay_sat_safety")
                    && row.contains("lane_identity=shared_ay_sat")
            })
            .expect("ay candidate lane");
        validate_prepared_candidate_lane_evidence_row(ay_lane_row).unwrap();
        let candidate_keys: HashSet<&str> = evidence
            .prepared_program
            .candidate_lanes
            .iter()
            .filter_map(|lane| lane.candidate_key.as_deref())
            .collect();
        for expected_key in [
            "ay_sat_safety",
            "ay_bmc",
            "ay_pdr",
            "ay_k_induction",
            "counterexample_trace_replay",
            "hardware_state_fingerprint",
        ] {
            assert!(
                candidate_keys.contains(expected_key),
                "missing candidate key {expected_key}"
            );
        }
        for (candidate_key, lane_identity) in [
            ("ay_bmc", AIGER_AY_BMC_LANE_IDENTITY),
            ("ay_pdr", AIGER_AY_PDR_LANE_IDENTITY),
            ("ay_k_induction", AIGER_AY_K_INDUCTION_LANE_IDENTITY),
        ] {
            let alias_row = rows
                .iter()
                .find(|row| {
                    row.starts_with("AIGER prepared_candidate_lane ")
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
                row.starts_with("AIGER prepared_candidate_lane ")
                    && row.contains("lane_kind=replay")
                    && row.contains("candidate_key=counterexample_trace_replay")
                    && row.contains("lane_identity=shared_hardware_proof_replay")
            })
            .expect("replay candidate lane");
        validate_prepared_candidate_lane_evidence_row(replay_lane_row).unwrap();
        let fingerprint_lane_row = rows
            .iter()
            .find(|row| {
                row.starts_with("AIGER prepared_candidate_lane ")
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
            .filter(|row| row.starts_with("AIGER prepared_candidate_lane "))
        {
            validate_prepared_candidate_lane_evidence_row(row).unwrap();
        }
        let validation_plan_row = rows
            .iter()
            .find(|row| {
                row.starts_with("AIGER prepared_validation_plan ")
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
            .filter(|row| row.starts_with("AIGER prepared_validation_plan "))
        {
            validate_prepared_validation_plan_evidence_row(row).unwrap();
        }
        let ay_receipt_row = rows
            .iter()
            .find(|row| {
                row.starts_with("AIGER validation_receipt ")
                    && row.contains("validator_kind=ay_proof")
                    && row.contains("digest_algorithm=fnv1a64")
                    && row.contains("validation_artifact_kind=proof")
            })
            .expect("ay proof receipt");
        validate_validation_receipt_evidence_row(ay_receipt_row).unwrap();
        let replay_receipt_row = rows
            .iter()
            .find(|row| {
                row.starts_with("AIGER validation_receipt ")
                    && row.contains("validator_kind=proof_replay")
                    && row.contains("validation_artifact_kind=witness")
            })
            .expect("proof replay receipt");
        validate_validation_receipt_evidence_row(replay_receipt_row).unwrap();

        let shared_fingerprint_row = rows
            .iter()
            .find(|row| {
                row.starts_with("AIGER shared_fingerprint_identity ")
                    && row.contains("source_kind=aiger")
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
                row.starts_with("AIGER shared_dedup_identity ")
                    && row.contains("source_kind=aiger")
                    && row.contains("fingerprint_value_kind=register_vector")
            })
            .expect("shared register-vector dedup row");
        assert!(shared_dedup_row.contains("storage_kind=cas"));
        assert!(shared_dedup_row.contains("collision_policy=canonical_payload_equality"));
        assert!(shared_dedup_row.contains("collision_fail_closed=true"));
        assert!(rows.iter().any(|row| {
            row.starts_with("AIGER shared_dedup_identity_validation ")
                && row.contains("status_code=accepted")
                && row.contains("fail_closed=true")
        }));
        let admission_row = rows
            .iter()
            .find(|row| row.starts_with("AIGER prepared_fingerprint_admission "))
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
            .find(|row| row.starts_with("AIGER hardware_transition_system_adoption "))
            .expect("hardware transition-system row");
        assert!(transition_row.contains("shared_engine_component=prepared_checker_program"));
        assert!(transition_row.contains("transition_kind=hardware_next_state"));
        assert!(transition_row.contains("ay_analytical_lane=receipt_backed"));
        assert!(transition_row.contains("witness_replay_lane=receipt_backed"));
        assert!(transition_row.contains("default_consumers=aiger,btor2"));

        let proof_lane_adoption_row = rows
            .iter()
            .find(|row| row.starts_with("AIGER hardware_ay_proof_lane_adoption "))
            .expect("generalized AY proof-lane adoption row");
        assert!(proof_lane_adoption_row.contains("origin_frontend=aiger"));
        assert!(proof_lane_adoption_row.contains("shared_engine_component=analytical_ay_proof"));
        assert!(proof_lane_adoption_row.contains("lane=bmc"));
        assert!(proof_lane_adoption_row
            .contains("proof_obligation_identity=aiger.hardware_register_safety_obligation.v1"));
        assert!(proof_lane_adoption_row.contains("prepared_program_identity=aiger:safety:"));
        assert!(proof_lane_adoption_row
            .contains("register_vector_identity=aiger.hardware_register_layout.v1"));
        assert!(proof_lane_adoption_row.contains(
            "validation_receipt_requirement=validator_backed_receipt_for_proof_obligation_fingerprint"
        ));
        assert!(proof_lane_adoption_row.contains("validation_status=validator_backed"));
        assert!(proof_lane_adoption_row
            .contains("proof_receipt_identity=aiger.hardware_ay_proof_lane.validation_receipt"));
        assert!(proof_lane_adoption_row.contains("proof_receipt_validation_kind=proof_transcript"));
        assert!(proof_lane_adoption_row.contains("proof_receipt_status=validator_backed"));
        assert!(proof_lane_adoption_row
            .contains("proof_receipt_validated_fingerprint_identity=aiger.ay_sat.proof:"));
        assert!(proof_lane_adoption_row.contains(
            "witness_receipt_identity=aiger.hardware_ay_witness_lane.validation_receipt"
        ));
        assert!(proof_lane_adoption_row.contains("witness_receipt_validation_kind=witness"));
        assert!(proof_lane_adoption_row.contains("witness_receipt_status=validator_backed"));
        assert!(proof_lane_adoption_row.contains(
            "witness_receipt_validated_fingerprint_identity=aiger.replay.counterexample:"
        ));
        assert!(
            proof_lane_adoption_row.contains("first_beneficiary=aiger_hardware_register_vector")
        );
        assert!(proof_lane_adoption_row
            .contains("second_beneficiary=tla_and_mcc_shared_ay_proof_lanes"));
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
    fn aiger_ay_proof_lane_publication_fails_closed_without_validator_receipt() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let evidence = AigerSharedEngineEvidence::from_input(&circuit);
        let artifact_only_receipt = evidence
            .ay_proof_lane_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_witness_receipt = evidence
            .ay_witness_lane_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);

        assert!(aiger_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            None,
            Some(&evidence.ay_witness_lane_receipt)
        )
        .is_none());
        assert!(aiger_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            Some(&artifact_only_receipt),
            Some(&evidence.ay_witness_lane_receipt)
        )
        .is_none());
        assert!(aiger_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            Some(&evidence.ay_proof_lane_receipt),
            None
        )
        .is_none());
        assert!(aiger_ay_proof_lane_adoption_evidence_row(
            &evidence.prepared_program,
            Some(&evidence.ay_proof_lane_receipt),
            Some(&artifact_only_witness_receipt)
        )
        .is_none());
    }

    #[test]
    fn aiger_capability_report_publishes_shared_engine_evidence() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(1),
            engines: vec![EngineConfig::Bmc { step: 1 }],
            max_depth: 1,
            preprocess: Default::default(),
        };
        let report = aiger_portfolio_capability_report(&circuit, &config);

        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER shared_engine_adoption ")
                && row.contains("second_beneficiary=btor2_portfolio")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER prepared_checker_program ")
                && row.contains("payload_kind=aiger")
                && row.contains("frontend_extensions=1")
                && row.contains("candidate_lanes=6")
        }));
        for candidate_key in ["ay_sat_safety", "ay_bmc", "ay_pdr", "ay_k_induction"] {
            assert!(report.evidence.iter().any(|row| {
                row.starts_with("AIGER prepared_candidate_lane ")
                    && row.contains(&format!("candidate_key={candidate_key}"))
            }));
        }
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER prepared_frontend_extension ")
                && row.contains("extension_kind=aiger")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER shared_dedup_identity ")
                && row.contains("fingerprint_value_kind=register_vector")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER prepared_fingerprint_admission ")
                && row.contains("default_consumers=aiger,btor2")
                && row.contains("admission_status=accepted")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER hardware_transition_system_adoption ")
                && row.contains("ay_analytical_lane=receipt_backed")
                && row.contains("witness_replay_lane=receipt_backed")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER validation_receipt ") && row.contains("validator_kind=ay_proof")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("AIGER hardware_ay_proof_lane_adoption ")
                && row.contains("second_beneficiary=tla_and_mcc_shared_ay_proof_lanes")
                && row.contains("witness_receipt_validation_kind=witness")
                && row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri")
        }));
    }
}
