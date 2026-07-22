// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::marking::{pack_marking_config, MarkingConfig, PreparedMarking, TokenWidth};
use crate::petri_net::PetriNet;
use crate::portfolio::ExactOrUnknownStatus;
use tla_mc_core::{
    current_native_frontend_families, CheckerSourceKind, FingerprintCanonicalPayload,
    FingerprintDomainKey, FingerprintDomainProjection, FingerprintDomainStoragePolicy,
    PreparedProgramPayloadKind, PreparedStorageKind, SetupTraceLaneKind, SharedCollisionPolicy,
    SharedDedupScope, SharedDedupStorageKind, SharedFingerprintAlgorithm,
    SharedFingerprintCanonicalDomain, SharedFingerprintValueKind, SharedNativeAbiSignature,
    SharedNativeAbiValueKind, SharedNativeAdmission, SharedNativeAdmissionDisposition,
    SharedNativeContract, SharedNativeContractIdentity, SharedNativeContractKind,
    SharedNativeEvidenceKind, SharedNativeEvidencePolicy, SharedNativeEvidenceRequirement,
    SharedNativeInstallAuthority, SharedNativeLayoutContract, SharedNativeLayoutKind,
    SharedNativePlanningIdentity, SharedNativeVectorContract, VALIDATION_RECEIPT_SCHEMA,
    VALIDATION_RECEIPT_SCHEMA_VERSION,
};

const SHARED_PETRI_PREPARED_NATIVE_CANDIDATE_SCHEMA: &str =
    "ty.shared_engine.petri.prepared_native_candidate.v1";
const SHARED_PETRI_PREPARED_NATIVE_CANDIDATE_SCHEMA_VERSION: u32 = 1;
const SHARED_PETRI_NATIVE_CONTRACT_MANIFEST_SCHEMA: &str =
    "ty.shared_engine.petri.native_contract_manifest.v1";
const SHARED_PETRI_NATIVE_CONTRACT_MANIFEST_SCHEMA_VERSION: u32 = 1;
const SHARED_PETRI_NATIVE_ENGINE_READINESS_ROW_KIND: &str = "petri_native_shared_engine_readiness";
const SHARED_PETRI_NATIVE_ENGINE_READINESS_SCHEMA: &str =
    "ty.shared_engine.petri.native_engine_readiness.v1";
const SHARED_PETRI_NATIVE_ENGINE_READINESS_SCHEMA_VERSION: u32 = 1;
const SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_ROW_KIND: &str =
    "petri_native_planning_fingerprint_identity";
const SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_SCHEMA: &str =
    "ty.shared_engine.petri.planning_fingerprint_identity.v1";
const SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_SCHEMA_VERSION: u32 = 1;
const SHARED_ENGINE_PREPARED_PROGRAM_COMPONENT: &str = "tla_mc_core.prepared_checker_program";
const SHARED_ENGINE_ORIGIN_FRONTEND: &str = "mcc_petri";
const SHARED_ENGINE_OWNER: &str = "shared_high_performance_engine";
const SHARED_ENGINE_FIRST_BENEFICIARY: &str = "mcc_petri_runtime_storage";
const SHARED_ENGINE_SECOND_BENEFICIARY: &str =
    "trust_cg_batch_identity_contract,ay_analytical,witness_replay";
const SHARED_ENGINE_EXTRACTION_STATUS: &str = "frontend-local-with-tracked-extraction";
const SHARED_ENGINE_BLOCKER_STATUS: &str = "tracked-blockers";
const SHARED_ENGINE_GENERIC_PREREQUISITES: &str = "prepared_checker_program_descriptor,marking_storage_identity,transition_relation_descriptor,state_predicate_descriptor,native_candidate_descriptor,validation_plan_descriptor";
const SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer";
const SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES: &str = "none";
const SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES: &str = "none";
const SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES: &str =
    SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES;
const SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS: &str =
    "tla_plus:needs_state_vector_native_layout_manifest,quint:needs_source_identity_preserving_native_manifest,mcc_petri:missing_native_install_validation_parity_and_callable_receipts,aiger:needs_register_vector_native_layout_manifest,btor2:needs_bitvector_register_native_layout_manifest,vmt_transition_system:needs_transition_system_native_layout_manifest,ay_analytical:needs_native_helper_validation_receipt,witness_replay:needs_replay_validation_receipt_adapter,future_importer:awaiting_registered_importer_frontend";
const SHARED_ENGINE_ADOPTION_LEVEL: &str = "level-0";
const SHARED_ENGINE_ADOPTION_MATRIX_FIELDS: &str = "origin_frontend,shared_owner,first_beneficiary,second_beneficiary,compatible_frontend_families,default_compatible_frontend_families,downstream_beneficiary_families,remaining_compatible_frontend_families,frontend_family_blockers,generic_prerequisites,shared_engine_prerequisite,compile_artifact_handoff_schema,compile_artifact_handoff_owner,compile_artifact_handoff_status,compile_artifact_handoff_blocker_code,native_adoption_blocker,exact_or_unknown,frontend_neutral_kernel_layout_fingerprint,validation_receipt_status,parity_receipt_status,callable_receipt_status,production_gate_status,acceptance_test,acceptance_evidence";
const SHARED_ENGINE_ACCEPTANCE_TEST: &str = "cargo_test_-p_tla-petri_--lib_explorer::setup";
const SHARED_ENGINE_ACCEPTANCE_EVIDENCE: &str =
    "prepared_native_candidate_row,shared_native_contract_manifest_row,shared_native_engine_readiness_row";
const SHARED_PETRI_PREPARED_PROGRAM_IDENTITY: &str = "mcc_petri.prepared_program";
const SHARED_PETRI_TRANSITION_DESCRIPTOR: &str = "shared_petri_transition_relation";
const SHARED_PETRI_PREDICATE_DESCRIPTOR: &str = "shared_petri_state_predicate";
const SHARED_PETRI_NATIVE_CANDIDATE_KEY: &str = "trust_cg_native";
const SHARED_PETRI_NATIVE_LANE_KIND: &str = "native";
const SHARED_PETRI_NATIVE_LANE_IDENTITY: &str = "shared_native_successor";
const SHARED_PETRI_FINGERPRINT_POLICY: &str = "petri_marking_fingerprint_chain.v1";
const SHARED_PETRI_FINGERPRINT_CHAIN_FIELDS: &str = "prepared_program_identity,transition_relation_identity,predicate_identity,candidate_identity,lane_identity,marking_layout_identity";
const SHARED_PETRI_FINGERPRINT_HELPER_SYMBOL: &str =
    "crate::explorer::fingerprint::fingerprint_marking";
const SHARED_PETRI_FINGERPRINT_SEED_IDENTITY: &str = "unseeded_sha256_packed_marking_bytes.v1";
const SHARED_PETRI_FINGERPRINT_CANONICAL_DOMAIN: &str = "place-token-marking";
const SHARED_PETRI_FINGERPRINT_CANONICAL_DOMAIN_VERSION: &str = "u64-vector-v1";
const SHARED_PETRI_FINGERPRINT_CANONICALIZATION_VERSION: &str =
    "pack_marking_config.packed_bytes.sha256_u128.v1";
const SHARED_PETRI_FINGERPRINT_STORAGE_CONFIG: &str =
    "petri-shared-native-validation-fingerprint-domain.v1";
const SHARED_PETRI_CACHE_NAMESPACE: &str = "mcc_petri.shared_native.validation_cache.v1";
const SHARED_PETRI_CACHE_REUSE_POLICY: &str = "frontend_local_only";
const SHARED_PETRI_CANONICALIZATION: &str = "petri-marking-transition-predicate-v1";
const SHARED_PETRI_VALIDATION_PLAN: &str = "fail_closed_exact_or_unknown";
const SHARED_PETRI_VALIDATION_ARTIFACT_KIND: &str = "native_candidate_descriptor";
const SHARED_PETRI_KERNEL_METADATA_SCHEMA: &str = "tla_ir.whole_program_kernel_metadata.v1";
const SHARED_PETRI_KERNEL_METADATA_IDENTITY_BASIS: &str =
    "tla_ir.whole_program_kernel_metadata.canonical_identity.v1";
const SHARED_PETRI_KERNEL_METADATA_SOURCE: &str = "local_tla_ir_compatible";
const SHARED_PETRI_KERNEL_METADATA_BLOCKER: &str =
    "tla-ir_metadata_crate_not_a_default_tla-petri_dependency";
const SHARED_PETRI_KERNEL_LAYOUT_KIND: &str = "petri_marking_i64_vector";
const SHARED_PETRI_NATIVE_CONTRACT_SYMBOL: &str = "petri_marking_successor_predicate_batch";
const SHARED_PETRI_NATIVE_CONTRACT_ARTIFACT_IDENTITY: &str =
    "mcc_petri.shared_native_contract.trust_cg_native.v1";
const SHARED_PETRI_NATIVE_CONTRACT_FRONTEND_PAYLOAD_IDENTITY: &str = "mcc_petri.marking_vector";
const SHARED_PETRI_NATIVE_CONTRACT_TARGET_ABI: &str =
    "extern_c.petri_marking_successor_predicate_batch.v1";
const SHARED_PETRI_NATIVE_CONTRACT_TRANSPORT: &str = "tla_mc_core.native_contract.v1";
const SHARED_PETRI_NATIVE_VECTOR_IDENTITY: &str = "petri_marking_vector.u64_tokens.v1";
const SHARED_PETRI_NATIVE_VECTOR_OPS: &str =
    "petri_successor_delta_and_predicate_eval.prerequisites.v1";
const SHARED_PETRI_NATIVE_VECTOR_GUARD: &str =
    "petri_native_contract_validation_only_until_trust_cg_batch_manifest";
const SHARED_PETRI_BATCH_ARTIFACT_COMPONENT: &str = "batch_native_artifact_identity";
const SHARED_PETRI_DIGEST_SOURCE: &str = "petri_native_contract_manifest";
const SHARED_PETRI_PREPARED_TRUST_IR_REUSE_IDENTITY_PREFIX: &str =
    "trust_cg_prepared_trust_ir_reuse";
const SHARED_PETRI_PREPARED_TRUST_IR_REUSE_SCOPE: &str = "shared_engine_frontend_neutral_batch";
const SHARED_PETRI_PREPARED_TRUST_IR_REUSE: &str = "deferred_until_trust_ir_manifest";
const SHARED_PETRI_PREPARED_IDENTITY_BASIS: &str = "petri_marking_successor_predicate_semantic_v1";
const SHARED_PETRI_FRONTEND_FIELDS_IN_CHECKSUMS: &str =
    "net_name,place_ids,place_names,transition_ids,transition_names,initial_marking,arcs";
const SHARED_PETRI_EXPORT_SET_IDENTITY_BASIS: &str = "petri_successor_predicate_symbol_set_v1";
const SHARED_PETRI_ALIAS_RESOLUTION_IDENTITY_BASIS: &str =
    "petri_marking_successor_predicate_alias_resolution_v1";
const SHARED_PETRI_EXPORT_SURFACE_IDENTITY_BASIS: &str =
    "petri_marking_successor_predicate_export_surface_v1";
const SHARED_PETRI_NATIVE_REQUIREMENTS_IDENTITY_BASIS: &str =
    "petri_marking_successor_predicate_native_requirements_v1";
const SHARED_PETRI_READINESS_SECOND_BENEFICIARY: &str = SHARED_ENGINE_SECOND_BENEFICIARY;
const SHARED_PETRI_READINESS_FRONTEND_FAMILIES: &str = "mcc_petri";
const SHARED_PETRI_FUTURE_FRONTEND_FAMILY_READINESS: &str =
    "deferred_until_core_shared_adoption_schema";
const SHARED_PETRI_SHARD_COMPATIBILITY_STATUS: &str = "deferred_until_trust_ir_trust_cg_manifest";
const SHARED_PETRI_SHARD_COMPATIBILITY_SCOPE: &str = "marking_vector_batch_partitionable";
const SHARED_PETRI_SHARD_IDENTITY_STATUS: &str = "deferred_until_trust_ir_trust_cg_manifest";
const SHARED_PETRI_SHARD_IDENTITY_PROVIDER: &str = "future_trust_ir_trust_cg_manifest";
const SHARED_PETRI_SHARD_REQUIRED_FIELDS: &str = "source_kind,payload_kind,storage_kind,layout_identity,symbol,semantic_checksum,layout_checksum,manifest_checksum";
const SHARED_PETRI_CACHE_FINGERPRINT_COMPATIBILITY: &str = "frontend_local_only";
const SHARED_PETRI_FINGERPRINT_COMPATIBILITY_STATUS: &str = "validation_only_declared";
const SHARED_PETRI_CACHE_COMPATIBILITY_STATUS: &str = "validation_only_frontend_local";
const SHARED_PETRI_FINGERPRINT_ADMISSION_SURFACE: &str =
    "shared_fingerprint_state_vector_admission";
const SHARED_PETRI_FINGERPRINT_ADMISSION_SEMANTICS: &str =
    "default_consumer,compatible_consumer,blocked";
const SHARED_PETRI_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
const SHARED_PETRI_FINGERPRINT_ADMISSION_DEFAULT_FRONTEND_FAMILIES: &str = "tla_plus,mcc_petri";
const SHARED_PETRI_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES: &str =
    "future_importer:awaiting_registered_importer_frontend";
const SHARED_PETRI_ARTIFACT_IDENTITY_STATUS: &str = "contract_template_only";
const SHARED_PETRI_ARTIFACT_DIGEST_STATUS: &str = "per_artifact_digest_missing";
const SHARED_PETRI_NATIVE_PARITY_RECEIPT_SCHEMA: &str =
    "ty.petri.native_successor.parity_receipt.v1";
const SHARED_PETRI_NATIVE_PARITY_RECEIPT_STATUS: &str = "missing";
const SHARED_PETRI_NATIVE_PARITY_RECEIPT_BLOCKER_CODE: &str = "missing_parity_receipt";
const SHARED_PETRI_NATIVE_PARITY_RECEIPT_GATE_API: &str =
    "tla_petri::petri_native_successor_parity_receipt_gate";
const SHARED_PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE: &str =
    "exact_successor_parity_trace,native_vs_explicit_state_replay_receipt";
const SHARED_PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA: &str =
    "ty.petri.native_successor.callable_receipt.v1";
const SHARED_PETRI_NATIVE_CALLABLE_RECEIPT_STATUS: &str = "missing";
const SHARED_PETRI_NATIVE_CALLABLE_RECEIPT_BLOCKER_CODE: &str = "missing_callable_receipt";
const SHARED_PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API: &str =
    "tla_petri::petri_native_successor_callable_receipt_gate";
const SHARED_PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE: &str =
    "compile_artifact_handoff,runtime_readiness_packet,native_install_gate_packet,call_packet,callable_pointer";
const SHARED_PETRI_VALIDATION_RECEIPT_STATUS: &str = "missing";
const SHARED_PETRI_VALIDATION_RECEIPT_BLOCKER_CODE: &str = "missing_validation_receipt";
const SHARED_PETRI_VALIDATION_RECEIPT_GATE_API: &str =
    "tla_mc_core::validate_validation_receipt_evidence_row";
const SHARED_PETRI_VALIDATION_RECEIPT_REQUIRED_EVIDENCE: &str =
    "accepted_shared_validation_receipt_for_native_successor_candidate";
const SHARED_PETRI_PRODUCTION_GATE: &str =
    "native_install_validation_parity_and_callable_receipts_required";
const SHARED_PETRI_PRODUCTION_GATE_STATUS: &str =
    "blocked_missing_native_install_validation_parity_and_callable_receipts";
const SHARED_PETRI_PRODUCTION_GATE_REQUIRED_RECEIPTS: &str =
    "native_install_receipt,validation_receipt,parity_receipt,callable_receipt";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PREREQUISITE: &str =
    "trust_cg_petri_compile_artifact_handoff";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_SCHEMA: &str =
    "trust-cg.petri.native_successor.compile_artifact_handoff.v1";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_OWNER: &str = "trust-cg";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_SOURCE: &str =
    "trust-cg.petri_native_successor_compile_artifact_handoff";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_STATUS: &str = "blocked";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODE: &str =
    "missing_trust_cg_petri_compile_artifact_handoff";
const SHARED_PETRI_NATIVE_ADOPTION_BLOCKER: &str =
    "trust_cg_petri_compile_artifact_handoff_required_before_native_adoption";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_API: &str =
    "trust-cg::petri_native_successor_compile_artifact_handoff_evidence";
#[cfg(feature = "trust-cg-petri-native")]
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR_SOURCE: &str =
    "tla_trust_cg::PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR";
#[cfg(not(feature = "trust-cg-petri-native"))]
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR_FALLBACK_SOURCE: &str =
    "tla_petri.local_compile_artifact_handoff_contract";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_SURFACE: &str =
    "petri_native_successor_compile_artifact_handoff";
#[cfg(not(feature = "trust-cg-petri-native"))]
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_SCHEMA_VERSION: u32 = 1;
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_REQUIRED_FIELDS: &str =
    "compiled_artifact.native_payload_sha256,compiled_artifact.entry_symbol,compiled_artifact.callable_pointer,compiled_artifact.executable_region_sha256,compiled_artifact.lifetime_owner,compiled_artifact.current_generation";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_STATUS_CODES: &str = "ready,blocked";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODES: &str =
    "missing_native_payload_sha256,missing_entry_symbol,missing_callable_pointer,missing_executable_region_sha256,missing_lifetime_owner,missing_current_generation";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_INPUT_TYPE: &str =
    "PetriNativeSuccessorCompileArtifactHandoffInput";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_TYPE: &str =
    "PetriNativeSuccessorCompileArtifactHandoffEvidence";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_TYPE: &str =
    "PetriNativeSuccessorCompileArtifactHandoffBlocker";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCER_REQUIREMENTS: &str =
    "InstalledArtifact.native_payload_sha256,entry_symbol,callable_pointer,executable_region_sha256,lifetime_owner,current_generation";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_CONSUMER_REQUIREMENTS: &str =
    "status_ready,sha256_identity,entry_symbol_match,callable_pointer_present,executable_region_present,lifetime_owner_present,current_generation_present";
#[cfg(feature = "trust-cg-petri-native")]
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BRIDGE_STATUS: &str =
    "descriptor_available_production_blocked";
#[cfg(not(feature = "trust-cg-petri-native"))]
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BRIDGE_FALLBACK_STATUS: &str =
    "local_contract_only_production_blocked";
const SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCTION_REQUIREMENT: &str =
    "ready_handoff_plus_native_install_validation_parity_and_callable_receipts";
const SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_STATUS: &str = "validation_only_missing_receipts";
const SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_REQUIRED_FIELDS: &str =
    "prepared_program_identity,candidate_identity,lane_identity,layout_checksum,semantic_checksum,source_checksum,payload_checksum,manifest_checksum,fingerprint_domain_identity,fingerprint_domain_acceptance_identity,cache_namespace_identity,cache_reuse_policy,cache_digest,prepared_trust_ir_reuse_identity";
const SHARED_PETRI_TRUST_CG_BATCH_CACHE_REUSE_STATUS: &str =
    "blocked_until_native_install_validation_parity_and_callable_receipts";
const SHARED_PETRI_TRUST_CG_BATCH_CACHE_REUSE_BLOCKER_CODE: &str =
    "missing_native_install_validation_parity_and_callable_receipts";

/// Shared explorer-specific preparation used by all execution backends.
pub(crate) struct ExplorationSetup {
    pub(crate) marking_config: MarkingConfig,
    pub(crate) pack_capacity: usize,
    pub(crate) num_places: usize,
    pub(crate) num_transitions: usize,
    pub(crate) initial_packed: Box<[u8]>,
}

impl ExplorationSetup {
    pub(crate) fn analyze(net: &PetriNet) -> Self {
        let prepared = PreparedMarking::analyze(net);
        let pack_capacity = prepared.packed_capacity();
        let marking_config = prepared.config;
        let num_places = marking_config.num_places;
        let num_transitions = net.num_transitions();

        let mut pack_buf = Vec::with_capacity(pack_capacity);
        pack_marking_config(&net.initial_marking, &marking_config, &mut pack_buf);
        let initial_packed: Box<[u8]> = pack_buf.as_slice().into();

        Self {
            marking_config,
            pack_capacity,
            num_places,
            num_transitions,
            initial_packed,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn render_shared_native_candidate_evidence_row(
        &self,
        scope: &str,
        validation_status: ExactOrUnknownStatus,
    ) -> String {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup(self, validation_status)
            .render_evidence_row(scope)
    }

    pub(crate) fn render_shared_native_candidate_evidence_row_for_net(
        &self,
        scope: &str,
        validation_status: ExactOrUnknownStatus,
        net: &PetriNet,
    ) -> String {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup_and_net(
            self,
            net,
            validation_status,
        )
        .render_evidence_row(scope)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn shared_native_contract(&self) -> SharedNativeContract {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup(
            self,
            ExactOrUnknownStatus::Unknown,
        )
        .shared_native_contract()
    }

    pub(crate) fn shared_native_contract_for_net(&self, net: &PetriNet) -> SharedNativeContract {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup_and_net(
            self,
            net,
            ExactOrUnknownStatus::Unknown,
        )
        .shared_native_contract()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn render_shared_native_contract_evidence_row(&self, scope: &str) -> String {
        self.shared_native_contract().render_evidence_row(scope)
    }

    pub(crate) fn render_shared_native_contract_evidence_row_for_net(
        &self,
        scope: &str,
        net: &PetriNet,
    ) -> String {
        self.shared_native_contract_for_net(net)
            .render_evidence_row(scope)
    }

    pub(crate) fn render_core_shared_native_planning_identity_evidence_row_for_net(
        &self,
        scope: &str,
        net: &PetriNet,
    ) -> String {
        self.shared_native_contract_for_net(net)
            .render_planning_identity_evidence_row(scope)
    }

    pub(crate) fn render_shared_native_contract_manifest_evidence_row(
        &self,
        scope: &str,
        net: &PetriNet,
    ) -> String {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup_and_net(
            self,
            net,
            ExactOrUnknownStatus::Unknown,
        )
        .render_contract_manifest_evidence_row(scope)
    }

    pub(crate) fn render_shared_native_engine_readiness_evidence_row(
        &self,
        scope: &str,
        net: &PetriNet,
    ) -> String {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup_and_net(
            self,
            net,
            ExactOrUnknownStatus::Unknown,
        )
        .render_shared_engine_readiness_evidence_row(scope)
    }

    pub(crate) fn shared_native_planning_fingerprint_identity_fields_for_net(
        &self,
        net: &PetriNet,
    ) -> Vec<(&'static str, String)> {
        SharedPetriPreparedNativeCandidateDescriptor::from_setup_and_net(
            self,
            net,
            ExactOrUnknownStatus::Unknown,
        )
        .evidence
        .shared_native_planning_fingerprint_identity_fields()
    }

    pub(crate) fn render_shared_planning_fingerprint_identity_evidence_row(
        &self,
        scope: &str,
        net: &PetriNet,
    ) -> String {
        let fields = self.shared_native_planning_fingerprint_identity_fields_for_net(net);
        render_shared_planning_fingerprint_identity_evidence_row(scope, &fields)
    }
}

/// Shared-engine descriptor for Petri transition/predicate/native candidate rows.
#[derive(Debug, Clone)]
pub(crate) struct SharedPetriPreparedNativeCandidateDescriptor {
    num_places: usize,
    num_transitions: usize,
    pack_capacity: usize,
    validation_status: ExactOrUnknownStatus,
    frontend_neutral_kernel_layout_fingerprint: String,
    evidence: SharedPetriNativeContractEvidence,
}

impl SharedPetriPreparedNativeCandidateDescriptor {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn for_net(net: &PetriNet, validation_status: ExactOrUnknownStatus) -> Self {
        let setup = ExplorationSetup::analyze(net);
        Self::from_setup(&setup, validation_status)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_setup(
        setup: &ExplorationSetup,
        validation_status: ExactOrUnknownStatus,
    ) -> Self {
        let frontend_neutral_kernel_layout_fingerprint = shared_petri_descriptor_digest(setup);
        let evidence = SharedPetriNativeContractEvidence::from_setup(setup, None);
        Self {
            num_places: setup.num_places,
            num_transitions: setup.num_transitions,
            pack_capacity: setup.pack_capacity,
            validation_status,
            frontend_neutral_kernel_layout_fingerprint,
            evidence,
        }
    }

    pub(crate) fn from_setup_and_net(
        setup: &ExplorationSetup,
        net: &PetriNet,
        validation_status: ExactOrUnknownStatus,
    ) -> Self {
        let frontend_neutral_kernel_layout_fingerprint = shared_petri_descriptor_digest(setup);
        let evidence = SharedPetriNativeContractEvidence::from_setup(setup, Some(net));
        Self {
            num_places: setup.num_places,
            num_transitions: setup.num_transitions,
            pack_capacity: setup.pack_capacity,
            validation_status,
            frontend_neutral_kernel_layout_fingerprint,
            evidence,
        }
    }

    pub(crate) fn render_evidence_row(&self, scope: &str) -> String {
        let compile_artifact_handoff = SharedPetriCompileArtifactHandoffBridge::current();
        format!(
            "{scope} prepared_native_candidate_shared_vocab schema={schema} schema_version={schema_version} \
             shared_engine_component={component} shared_engine_origin_frontend={origin_frontend} \
             origin_frontend={origin_frontend} shared_owner={owner} first_beneficiary={first_beneficiary} \
             second_beneficiary={second_beneficiary} shared_engine_owner={owner} \
             extraction_status={extraction_status} blocker_status={blocker_status} \
             adoption_matrix_fields={adoption_matrix_fields} generic_prerequisites={generic_prerequisites} \
             compatible_frontend_families={compatible_frontend_families} \
             default_compatible_frontend_families={default_compatible_frontend_families} \
             downstream_beneficiary_families={downstream_beneficiary_families} \
             remaining_compatible_frontend_families={remaining_compatible_frontend_families} \
             frontend_family_blockers={frontend_family_blockers} acceptance_test={acceptance_test} \
             acceptance_evidence={acceptance_evidence} prepared_program_identity={prepared_program_identity} \
             transition_descriptor={transition_descriptor} predicate_descriptor={predicate_descriptor} \
             candidate_key={candidate_key} lane_kind={lane_kind} lane_identity={lane_identity} \
             fingerprint_policy_identity={fingerprint_policy_identity} fingerprint_chain_fields={fingerprint_chain_fields} \
             fingerprint_chain_digest_algorithm=fnv1a64 fingerprint_chain_digest={kernel_layout_fingerprint} \
             fingerprint_domain_identity={fingerprint_domain_identity} fingerprint_domain_acceptance_identity={fingerprint_domain_acceptance_identity} \
             cache_namespace_identity={cache_namespace_identity} cache_reuse_policy={cache_reuse_policy} cache_digest={cache_digest} \
             fingerprint_admission_surface={fingerprint_admission_surface} fingerprint_admission_semantics={fingerprint_admission_semantics} \
             fingerprint_admission_compatible_frontend_families={fingerprint_admission_compatible_frontend_families} \
             fingerprint_admission_default_frontend_families={fingerprint_admission_default_frontend_families} \
             fingerprint_admission_blocked_frontend_families={fingerprint_admission_blocked_frontend_families} \
             kernel_metadata_schema={kernel_metadata_schema} kernel_metadata_identity_basis={kernel_metadata_identity_basis} \
             kernel_metadata_source={kernel_metadata_source} kernel_metadata_blocker={kernel_metadata_blocker} \
             kernel_layout_kind={kernel_layout_kind} frontend_neutral_kernel_layout_fingerprint_algorithm=fnv1a64 \
             frontend_neutral_kernel_layout_fingerprint={kernel_layout_fingerprint} \
             manifest_checksum_algorithm=fnv1a64 manifest_checksum={manifest_checksum} \
             layout_checksum_algorithm=fnv1a64 layout_checksum={layout_checksum} \
             semantic_checksum_algorithm=fnv1a64 semantic_checksum={semantic_checksum} \
             source_checksum_algorithm=fnv1a64 source_checksum={source_checksum} \
             payload_checksum_algorithm=fnv1a64 payload_checksum={payload_checksum} \
             canonicalization={canonicalization} validation_plan={validation_plan} \
             validation_artifact_kind={validation_artifact_kind} validation_status={validation_status} \
             validation_receipt_status={validation_receipt_status} parity_receipt_status={parity_receipt_status} \
             production_gate={production_gate} production_gate_status={production_gate_status} \
             production_gate_required_receipts={production_gate_required_receipts} \
             shared_engine_prerequisite={compile_artifact_handoff_prerequisite} \
             compile_artifact_handoff_schema={compile_artifact_handoff_schema} \
             compile_artifact_handoff_schema_version={compile_artifact_handoff_schema_version} \
             compile_artifact_handoff_owner={compile_artifact_handoff_owner} \
             compile_artifact_handoff_evidence_source={compile_artifact_handoff_evidence_source} \
             compile_artifact_handoff_descriptor_source={compile_artifact_handoff_descriptor_source} \
             compile_artifact_handoff_surface={compile_artifact_handoff_surface} \
             compile_artifact_handoff_api={compile_artifact_handoff_api} \
             compile_artifact_handoff_input_type={compile_artifact_handoff_input_type} \
             compile_artifact_handoff_evidence_type={compile_artifact_handoff_evidence_type} \
             compile_artifact_handoff_blocker_type={compile_artifact_handoff_blocker_type} \
             compile_artifact_handoff_required_fields={compile_artifact_handoff_required_fields} \
             compile_artifact_handoff_status_codes={compile_artifact_handoff_status_codes} \
             compile_artifact_handoff_blocker_codes={compile_artifact_handoff_blocker_codes} \
             compile_artifact_handoff_blocker_codes_count={compile_artifact_handoff_blocker_codes_count} \
             compile_artifact_handoff_bridge_status={compile_artifact_handoff_bridge_status} \
             compile_artifact_handoff_producer_requirements={compile_artifact_handoff_producer_requirements} \
             compile_artifact_handoff_consumer_requirements={compile_artifact_handoff_consumer_requirements} \
             compile_artifact_handoff_production_requirement={compile_artifact_handoff_production_requirement} \
             compile_artifact_handoff_status={compile_artifact_handoff_status} \
             compile_artifact_handoff_blocker_code={compile_artifact_handoff_blocker_code} \
             native_adoption_blocker={native_adoption_blocker} \
             exact_or_unknown={exact_or_unknown} fail_closed=true model_identity=generic \
             places={places} transitions={transitions} pack_capacity={pack_capacity}",
            schema = SHARED_PETRI_PREPARED_NATIVE_CANDIDATE_SCHEMA,
            schema_version = SHARED_PETRI_PREPARED_NATIVE_CANDIDATE_SCHEMA_VERSION,
            component = SHARED_ENGINE_PREPARED_PROGRAM_COMPONENT,
            origin_frontend = SHARED_ENGINE_ORIGIN_FRONTEND,
            owner = SHARED_ENGINE_OWNER,
            first_beneficiary = SHARED_ENGINE_FIRST_BENEFICIARY,
            second_beneficiary = SHARED_ENGINE_SECOND_BENEFICIARY,
            extraction_status = SHARED_ENGINE_EXTRACTION_STATUS,
            blocker_status = SHARED_ENGINE_BLOCKER_STATUS,
            adoption_matrix_fields = SHARED_ENGINE_ADOPTION_MATRIX_FIELDS,
            generic_prerequisites = SHARED_ENGINE_GENERIC_PREREQUISITES,
            compatible_frontend_families = SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            default_compatible_frontend_families =
                SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
            downstream_beneficiary_families = SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES,
            remaining_compatible_frontend_families =
                SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
            frontend_family_blockers = SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
            acceptance_test = SHARED_ENGINE_ACCEPTANCE_TEST,
            acceptance_evidence = SHARED_ENGINE_ACCEPTANCE_EVIDENCE,
            prepared_program_identity = SHARED_PETRI_PREPARED_PROGRAM_IDENTITY,
            transition_descriptor = SHARED_PETRI_TRANSITION_DESCRIPTOR,
            predicate_descriptor = SHARED_PETRI_PREDICATE_DESCRIPTOR,
            candidate_key = SHARED_PETRI_NATIVE_CANDIDATE_KEY,
            lane_kind = SHARED_PETRI_NATIVE_LANE_KIND,
            lane_identity = SHARED_PETRI_NATIVE_LANE_IDENTITY,
            fingerprint_policy_identity = SHARED_PETRI_FINGERPRINT_POLICY,
            fingerprint_chain_fields = SHARED_PETRI_FINGERPRINT_CHAIN_FIELDS,
            kernel_layout_fingerprint = self.frontend_neutral_kernel_layout_fingerprint,
            fingerprint_domain_identity = self.evidence.fingerprint_domain_identity,
            fingerprint_domain_acceptance_identity = self.evidence.fingerprint_policy_identity,
            cache_namespace_identity = self.evidence.cache_namespace_identity,
            cache_reuse_policy = SHARED_PETRI_CACHE_REUSE_POLICY,
            cache_digest = self.evidence.cache_digest,
            fingerprint_admission_surface = SHARED_PETRI_FINGERPRINT_ADMISSION_SURFACE,
            fingerprint_admission_semantics = SHARED_PETRI_FINGERPRINT_ADMISSION_SEMANTICS,
            fingerprint_admission_compatible_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES,
            fingerprint_admission_default_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_DEFAULT_FRONTEND_FAMILIES,
            fingerprint_admission_blocked_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES,
            kernel_metadata_schema = SHARED_PETRI_KERNEL_METADATA_SCHEMA,
            kernel_metadata_identity_basis = SHARED_PETRI_KERNEL_METADATA_IDENTITY_BASIS,
            kernel_metadata_source = SHARED_PETRI_KERNEL_METADATA_SOURCE,
            kernel_metadata_blocker = SHARED_PETRI_KERNEL_METADATA_BLOCKER,
            kernel_layout_kind = SHARED_PETRI_KERNEL_LAYOUT_KIND,
            manifest_checksum = self.evidence.manifest_checksum,
            layout_checksum = self.evidence.layout_checksum,
            semantic_checksum = self.evidence.semantic_checksum,
            source_checksum = self.evidence.source_checksum,
            payload_checksum = self.evidence.payload_checksum,
            canonicalization = SHARED_PETRI_CANONICALIZATION,
            validation_plan = SHARED_PETRI_VALIDATION_PLAN,
            validation_artifact_kind = SHARED_PETRI_VALIDATION_ARTIFACT_KIND,
            validation_status = self.validation_status.validation_status_code(),
            validation_receipt_status = SHARED_PETRI_VALIDATION_RECEIPT_STATUS,
            parity_receipt_status = SHARED_PETRI_NATIVE_PARITY_RECEIPT_STATUS,
            production_gate = SHARED_PETRI_PRODUCTION_GATE,
            production_gate_status = SHARED_PETRI_PRODUCTION_GATE_STATUS,
            production_gate_required_receipts = SHARED_PETRI_PRODUCTION_GATE_REQUIRED_RECEIPTS,
            compile_artifact_handoff_prerequisite =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PREREQUISITE,
            compile_artifact_handoff_schema = compile_artifact_handoff.schema,
            compile_artifact_handoff_schema_version = compile_artifact_handoff.schema_version,
            compile_artifact_handoff_owner = SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_OWNER,
            compile_artifact_handoff_evidence_source =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_SOURCE,
            compile_artifact_handoff_descriptor_source =
                compile_artifact_handoff.descriptor_source,
            compile_artifact_handoff_surface = compile_artifact_handoff.surface_name,
            compile_artifact_handoff_api = compile_artifact_handoff.api,
            compile_artifact_handoff_input_type = compile_artifact_handoff.input_type,
            compile_artifact_handoff_evidence_type = compile_artifact_handoff.evidence_type,
            compile_artifact_handoff_blocker_type = compile_artifact_handoff.blocker_type,
            compile_artifact_handoff_required_fields = compile_artifact_handoff.required_fields,
            compile_artifact_handoff_status_codes = compile_artifact_handoff.status_codes,
            compile_artifact_handoff_blocker_codes = compile_artifact_handoff.blocker_codes,
            compile_artifact_handoff_blocker_codes_count =
                compile_artifact_handoff.blocker_code_count,
            compile_artifact_handoff_bridge_status = compile_artifact_handoff.bridge_status,
            compile_artifact_handoff_producer_requirements =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCER_REQUIREMENTS,
            compile_artifact_handoff_consumer_requirements =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_CONSUMER_REQUIREMENTS,
            compile_artifact_handoff_production_requirement =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCTION_REQUIREMENT,
            compile_artifact_handoff_status = SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_STATUS,
            compile_artifact_handoff_blocker_code =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODE,
            native_adoption_blocker = SHARED_PETRI_NATIVE_ADOPTION_BLOCKER,
            exact_or_unknown = self.validation_status.code(),
            places = self.num_places,
            transitions = self.num_transitions,
            pack_capacity = self.pack_capacity,
        )
    }

    pub(crate) fn shared_native_contract(&self) -> SharedNativeContract {
        let identity = SharedNativeContractIdentity::new(
            SHARED_PETRI_PREPARED_PROGRAM_IDENTITY,
            SHARED_PETRI_NATIVE_CANDIDATE_KEY,
            SHARED_PETRI_NATIVE_LANE_IDENTITY,
        )
        .with_source_fingerprint(self.evidence.source_checksum.clone())
        .with_frontend_payload_identity(SHARED_PETRI_NATIVE_CONTRACT_FRONTEND_PAYLOAD_IDENTITY)
        .with_plan_reuse_manifest(
            shared_petri_prepared_trust_ir_reuse_identity(&self.evidence.semantic_checksum),
            self.evidence.manifest_checksum.clone(),
        )
        .with_trust_ir_identity(SHARED_PETRI_KERNEL_METADATA_IDENTITY_BASIS)
        .with_transport_identity(SHARED_PETRI_NATIVE_CONTRACT_TRANSPORT)
        .with_semantic_digest(self.evidence.semantic_checksum.clone())
        .with_cache_digest(self.evidence.cache_digest.clone())
        .with_fingerprint_domain_identity(self.evidence.fingerprint_domain_identity.clone())
        .with_cas_identity(self.evidence.fingerprint_policy_identity.clone())
        .with_cache_identity(self.evidence.cache_namespace_identity.clone())
        .with_cache_namespace_identity(self.evidence.cache_namespace_identity.clone())
        .with_cache_reuse_policy(SHARED_PETRI_CACHE_REUSE_POLICY)
        .with_artifact_identity(SHARED_PETRI_NATIVE_CONTRACT_ARTIFACT_IDENTITY)
        .with_artifact_fingerprint(self.evidence.manifest_checksum.clone())
        .with_target_abi_identity(SHARED_PETRI_NATIVE_CONTRACT_TARGET_ABI)
        .with_storage_layout_fingerprint(self.evidence.layout_checksum.clone())
        .with_proof_policy_identity(SHARED_PETRI_VALIDATION_PLAN);

        let layout = self.shared_native_layout_contract(&self.evidence.layout_checksum);
        let evidence_policy =
            SharedNativeEvidencePolicy::fail_closed(SharedNativeInstallAuthority::ValidationOnly)
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::ManifestMetadata,
                    self.evidence.manifest_checksum.clone(),
                ))
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::LayoutChecksum,
                    self.evidence.layout_checksum.clone(),
                ))
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::SemanticChecksum,
                    self.evidence.semantic_checksum.clone(),
                ))
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::ValidationReceipt,
                    SHARED_PETRI_VALIDATION_PLAN,
                ));

        let admission = SharedNativeAdmission::accepted_fail_closed(
            SharedNativeInstallAuthority::ValidationOnly,
        )
        .with_disposition(SharedNativeAdmissionDisposition::ProfileOnly)
        .with_layout_checksum(self.evidence.layout_checksum.clone())
        .with_semantic_checksum(self.evidence.semantic_checksum.clone());

        SharedNativeContract::new(
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
            SharedNativeContractKind::SuccessorKernel,
            Self::shared_successor_predicate_abi(),
            identity,
        )
        .with_lane_kind(SetupTraceLaneKind::Native)
        .with_layout(layout)
        .with_evidence_policy(evidence_policy)
        .with_admission(admission)
    }

    pub(crate) fn render_contract_manifest_evidence_row(&self, scope: &str) -> String {
        self.evidence.render_evidence_row(
            scope,
            self.num_places,
            self.num_transitions,
            self.pack_capacity,
        )
    }

    pub(crate) fn render_shared_engine_readiness_evidence_row(&self, scope: &str) -> String {
        self.evidence.render_shared_engine_readiness_evidence_row(
            scope,
            self.num_places,
            self.num_transitions,
            self.pack_capacity,
        )
    }

    fn shared_successor_predicate_abi() -> SharedNativeAbiSignature {
        SharedNativeAbiSignature::new("extern_c", SHARED_PETRI_NATIVE_CONTRACT_SYMBOL)
            .with_param("input_markings", SharedNativeAbiValueKind::Ptr)
            .with_param("input_count", SharedNativeAbiValueKind::U32)
            .with_param("place_count", SharedNativeAbiValueKind::U32)
            .with_param("transition_plan", SharedNativeAbiValueKind::Ptr)
            .with_param("predicate_plan", SharedNativeAbiValueKind::Ptr)
            .with_param("output_markings", SharedNativeAbiValueKind::Ptr)
            .with_param("output_parent_indices", SharedNativeAbiValueKind::Ptr)
            .with_param("output_counts", SharedNativeAbiValueKind::Ptr)
            .with_param("diagnostics", SharedNativeAbiValueKind::Ptr)
            .with_return(SharedNativeAbiValueKind::U32)
    }

    fn shared_native_layout_contract(
        &self,
        layout_fingerprint: &str,
    ) -> SharedNativeLayoutContract {
        let mut layout = SharedNativeLayoutContract::new(
            SharedNativeLayoutKind::PetriMarking,
            SHARED_PETRI_KERNEL_LAYOUT_KIND,
        )
        .with_fingerprint(layout_fingerprint)
        .with_state_len(saturating_u32(self.num_places));

        if self.num_places > 0 {
            layout = layout.with_vector_contract(
                SharedNativeVectorContract::new(
                    SHARED_PETRI_NATIVE_VECTOR_IDENTITY,
                    SharedNativeAbiValueKind::U64,
                    saturating_u32(self.num_places),
                    saturating_u32(self.num_places),
                    64,
                    64,
                )
                .with_operations_identity(SHARED_PETRI_NATIVE_VECTOR_OPS)
                .with_feature_guard(SHARED_PETRI_NATIVE_VECTOR_GUARD),
            );
        }

        layout
    }
}

#[derive(Debug, Clone)]
struct SharedPetriNativeContractEvidence {
    source_checksum: String,
    payload_checksum: String,
    manifest_checksum: String,
    layout_checksum: String,
    semantic_checksum: String,
    cache_digest: String,
    fingerprint_domain_identity: String,
    fingerprint_policy_identity: String,
    cache_namespace_identity: String,
    token_width: &'static str,
    packed_len: usize,
    excluded_place_count: usize,
}

#[derive(Debug, Clone)]
struct SharedPetriCompileArtifactHandoffBridge {
    descriptor_source: &'static str,
    bridge_status: &'static str,
    schema: &'static str,
    schema_version: u32,
    surface_name: &'static str,
    api: &'static str,
    input_type: &'static str,
    evidence_type: &'static str,
    blocker_type: &'static str,
    required_fields: String,
    status_codes: String,
    blocker_codes: String,
    blocker_code_count: usize,
}

impl SharedPetriCompileArtifactHandoffBridge {
    fn current() -> Self {
        #[cfg(feature = "trust-cg-petri-native")]
        {
            let descriptor =
                tla_trust_cg::PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR;
            Self {
                descriptor_source: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR_SOURCE,
                bridge_status: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BRIDGE_STATUS,
                schema: descriptor.schema,
                schema_version: descriptor.schema_version,
                surface_name: descriptor.name,
                api: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_API,
                input_type: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_INPUT_TYPE,
                evidence_type: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_TYPE,
                blocker_type: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_TYPE,
                required_fields: descriptor.required_fields.join(","),
                status_codes: descriptor.status_codes.join(","),
                blocker_codes: descriptor.blocker_codes.join(","),
                blocker_code_count: descriptor.blocker_codes.len(),
            }
        }

        #[cfg(not(feature = "trust-cg-petri-native"))]
        {
            Self {
                descriptor_source: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR_FALLBACK_SOURCE,
                bridge_status: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BRIDGE_FALLBACK_STATUS,
                schema: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_SCHEMA,
                schema_version: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_SCHEMA_VERSION,
                surface_name: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_SURFACE,
                api: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_API,
                input_type: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_INPUT_TYPE,
                evidence_type: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_TYPE,
                blocker_type: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_TYPE,
                required_fields: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_REQUIRED_FIELDS.to_string(),
                status_codes: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_STATUS_CODES.to_string(),
                blocker_codes: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODES.to_string(),
                blocker_code_count: SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODES
                    .split(',')
                    .count(),
            }
        }
    }
}

impl SharedPetriNativeContractEvidence {
    fn from_setup(setup: &ExplorationSetup, net: Option<&PetriNet>) -> Self {
        let layout_checksum = shared_petri_layout_checksum(setup);
        let semantic_checksum = shared_petri_semantic_checksum(setup, net);
        let source_checksum = shared_petri_source_checksum(setup, net);
        let payload_checksum =
            shared_petri_payload_checksum(&layout_checksum, &semantic_checksum, &source_checksum);
        let fingerprint_domain = shared_petri_fingerprint_domain_key(&layout_checksum)
            .unwrap_or_else(|error| {
                panic!("Petri shared fingerprint domain must be valid: {error}")
            });
        let fingerprint_domain_identity = fingerprint_domain.stable_identity();
        let fingerprint_policy_identity = fingerprint_domain
            .accepted_fail_closed_policy_identity()
            .unwrap_or_else(|error| {
                panic!("Petri shared fingerprint domain policy must fail closed: {error}")
            });
        let cache_namespace_identity = SHARED_PETRI_CACHE_NAMESPACE.to_string();
        let cache_digest = shared_petri_cache_digest(
            &fingerprint_domain_identity,
            &fingerprint_policy_identity,
            &cache_namespace_identity,
            &payload_checksum,
        );
        let manifest_checksum = shared_petri_manifest_checksum(
            &layout_checksum,
            &semantic_checksum,
            &source_checksum,
            &payload_checksum,
            &cache_digest,
            &fingerprint_domain_identity,
        );

        Self {
            source_checksum,
            payload_checksum,
            manifest_checksum,
            layout_checksum,
            semantic_checksum,
            cache_digest,
            fingerprint_domain_identity,
            fingerprint_policy_identity,
            cache_namespace_identity,
            token_width: token_width_code(setup.marking_config.width),
            packed_len: setup.marking_config.packed_len,
            excluded_place_count: setup
                .marking_config
                .excluded_places()
                .iter()
                .filter(|excluded| **excluded)
                .count(),
        }
    }

    fn render_evidence_row(
        &self,
        scope: &str,
        num_places: usize,
        num_transitions: usize,
        pack_capacity: usize,
    ) -> String {
        let shard_identity_key = shared_petri_shard_identity(
            &self.layout_checksum,
            &self.semantic_checksum,
            &self.manifest_checksum,
        );
        let compile_artifact_handoff = SharedPetriCompileArtifactHandoffBridge::current();
        format!(
            "{scope} shared_native_contract_manifest schema={schema} schema_version={schema_version} \
             source_kind=mcc_petri payload_kind=mcc_petri storage_kind=petri_marking \
             layout_kind=petri_marking layout_identity={layout_identity} symbol={symbol} \
             prepared_program_identity={prepared_program_identity} candidate_identity={candidate_identity} \
             lane_identity={lane_identity} manifest_metadata_schema={manifest_metadata_schema} \
             manifest_metadata_source={manifest_metadata_source} manifest_metadata_blocker={manifest_metadata_blocker} \
             origin_frontend={origin_frontend} shared_engine_component={component} shared_owner={owner} \
             first_beneficiary={first_beneficiary} second_beneficiary={second_beneficiary} \
             extraction_status={extraction_status} \
             adoption_level={adoption_level} compatible_frontend_families={compatible_frontend_families} \
             default_compatible_frontend_families={default_compatible_frontend_families} \
             downstream_beneficiary_families={downstream_beneficiary_families} \
             remaining_compatible_frontend_families={remaining_compatible_frontend_families} \
             frontend_family_blockers={frontend_family_blockers} blocker_status={blocker_status} \
             generic_prerequisites={generic_prerequisites} acceptance_test={acceptance_test} \
             acceptance_evidence={acceptance_evidence} \
             manifest_checksum_algorithm=fnv1a64 manifest_checksum={manifest_checksum} \
             layout_checksum_algorithm=fnv1a64 layout_checksum={layout_checksum} \
             semantic_checksum_algorithm=fnv1a64 semantic_checksum={semantic_checksum} \
             source_checksum_algorithm=fnv1a64 source_checksum={source_checksum} \
             payload_checksum_algorithm=fnv1a64 payload_checksum={payload_checksum} \
             fingerprint_algorithm=canonical_bytes_sha256 fingerprint_helper_symbol={fingerprint_helper_symbol} \
             fingerprint_seed_identity={fingerprint_seed_identity} fingerprint_domain_identity={fingerprint_domain_identity} \
             fingerprint_domain_acceptance_identity={fingerprint_policy_identity} collision_policy=canonical_payload_equality \
             fingerprint_admission_surface={fingerprint_admission_surface} fingerprint_admission_semantics={fingerprint_admission_semantics} \
             fingerprint_admission_compatible_frontend_families={fingerprint_admission_compatible_frontend_families} \
             fingerprint_admission_default_frontend_families={fingerprint_admission_default_frontend_families} \
             fingerprint_admission_blocked_frontend_families={fingerprint_admission_blocked_frontend_families} \
             cache_namespace_identity={cache_namespace_identity} cache_reuse_policy={cache_reuse_policy} \
             cache_digest_algorithm=fnv1a64 cache_digest={cache_digest} \
             artifact_identity={artifact_identity} artifact_identity_kind=contract_template \
             artifact_identity_status={artifact_identity_status} artifact_digest_status={artifact_digest_status} \
             validation_only=true shard_compatibility_status={shard_compatibility_status} \
             shard_compatibility_scope={shard_compatibility_scope} shard_identity_status={shard_identity_status} \
             shard_identity_provider={shard_identity_provider} shard_identity_key={shard_identity_key} \
             shard_required_fields={shard_required_fields} \
             fingerprint_compatibility_status={fingerprint_compatibility_status} \
             fingerprint_compatibility=canonical_bytes_sha256 \
             cache_compatibility_status={cache_compatibility_status} \
             cache_fingerprint_compatibility={cache_fingerprint_compatibility} \
             parity_receipt_required=true parity_receipt_schema={parity_receipt_schema} \
             parity_receipt_status={parity_receipt_status} parity_receipt_blocker_code={parity_receipt_blocker_code} \
             parity_receipt_gate_api={parity_receipt_gate_api} parity_receipt_required_evidence={parity_receipt_required_evidence} \
             validation_receipt_required=true validation_receipt_schema={validation_receipt_schema} \
             validation_receipt_schema_version={validation_receipt_schema_version} \
             validation_receipt_status={validation_receipt_status} \
             validation_receipt_blocker_code={validation_receipt_blocker_code} \
             validation_receipt_gate_api={validation_receipt_gate_api} \
             validation_receipt_required_evidence={validation_receipt_required_evidence} \
             production_gate={production_gate} production_gate_status={production_gate_status} \
             production_gate_required_receipts={production_gate_required_receipts} \
             shared_engine_prerequisite={compile_artifact_handoff_prerequisite} \
             compile_artifact_handoff_schema={compile_artifact_handoff_schema} \
             compile_artifact_handoff_schema_version={compile_artifact_handoff_schema_version} \
             compile_artifact_handoff_owner={compile_artifact_handoff_owner} \
             compile_artifact_handoff_evidence_source={compile_artifact_handoff_evidence_source} \
             compile_artifact_handoff_descriptor_source={compile_artifact_handoff_descriptor_source} \
             compile_artifact_handoff_surface={compile_artifact_handoff_surface} \
             compile_artifact_handoff_api={compile_artifact_handoff_api} \
             compile_artifact_handoff_input_type={compile_artifact_handoff_input_type} \
             compile_artifact_handoff_evidence_type={compile_artifact_handoff_evidence_type} \
             compile_artifact_handoff_blocker_type={compile_artifact_handoff_blocker_type} \
             compile_artifact_handoff_required_fields={compile_artifact_handoff_required_fields} \
             compile_artifact_handoff_status_codes={compile_artifact_handoff_status_codes} \
             compile_artifact_handoff_blocker_codes={compile_artifact_handoff_blocker_codes} \
             compile_artifact_handoff_blocker_codes_count={compile_artifact_handoff_blocker_codes_count} \
             compile_artifact_handoff_bridge_status={compile_artifact_handoff_bridge_status} \
             compile_artifact_handoff_producer_requirements={compile_artifact_handoff_producer_requirements} \
             compile_artifact_handoff_consumer_requirements={compile_artifact_handoff_consumer_requirements} \
             compile_artifact_handoff_production_requirement={compile_artifact_handoff_production_requirement} \
             compile_artifact_handoff_status={compile_artifact_handoff_status} \
             compile_artifact_handoff_blocker_code={compile_artifact_handoff_blocker_code} \
             native_adoption_blocker={native_adoption_blocker} \
             required_evidence=manifest_metadata,layout_checksum,semantic_checksum,validation_receipt \
             manifest_metadata_status=present layout_checksum_status=present semantic_checksum_status=present \
             install_authority=validation_only admission_disposition=profile_only \
             production_selected=false fail_closed=true token_width={token_width} places={places} transitions={transitions} \
             packed_len={packed_len} excluded_places={excluded_places} pack_capacity={pack_capacity}",
            schema = SHARED_PETRI_NATIVE_CONTRACT_MANIFEST_SCHEMA,
            schema_version = SHARED_PETRI_NATIVE_CONTRACT_MANIFEST_SCHEMA_VERSION,
            layout_identity = SHARED_PETRI_KERNEL_LAYOUT_KIND,
            symbol = SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
            prepared_program_identity = SHARED_PETRI_PREPARED_PROGRAM_IDENTITY,
            candidate_identity = SHARED_PETRI_NATIVE_CANDIDATE_KEY,
            lane_identity = SHARED_PETRI_NATIVE_LANE_IDENTITY,
            manifest_metadata_schema = SHARED_PETRI_KERNEL_METADATA_SCHEMA,
            manifest_metadata_source = SHARED_PETRI_KERNEL_METADATA_SOURCE,
            manifest_metadata_blocker = SHARED_PETRI_KERNEL_METADATA_BLOCKER,
            origin_frontend = SHARED_ENGINE_ORIGIN_FRONTEND,
            component = SHARED_ENGINE_PREPARED_PROGRAM_COMPONENT,
            owner = SHARED_ENGINE_OWNER,
            first_beneficiary = SHARED_ENGINE_FIRST_BENEFICIARY,
            second_beneficiary = SHARED_ENGINE_SECOND_BENEFICIARY,
            extraction_status = SHARED_ENGINE_EXTRACTION_STATUS,
            adoption_level = SHARED_ENGINE_ADOPTION_LEVEL,
            compatible_frontend_families = SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            default_compatible_frontend_families =
                SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
            downstream_beneficiary_families = SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES,
            remaining_compatible_frontend_families =
                SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
            frontend_family_blockers = SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
            blocker_status = SHARED_ENGINE_BLOCKER_STATUS,
            generic_prerequisites = SHARED_ENGINE_GENERIC_PREREQUISITES,
            acceptance_test = SHARED_ENGINE_ACCEPTANCE_TEST,
            acceptance_evidence = SHARED_ENGINE_ACCEPTANCE_EVIDENCE,
            manifest_checksum = self.manifest_checksum,
            layout_checksum = self.layout_checksum,
            semantic_checksum = self.semantic_checksum,
            source_checksum = self.source_checksum,
            payload_checksum = self.payload_checksum,
            fingerprint_helper_symbol = SHARED_PETRI_FINGERPRINT_HELPER_SYMBOL,
            fingerprint_seed_identity = SHARED_PETRI_FINGERPRINT_SEED_IDENTITY,
            fingerprint_domain_identity = self.fingerprint_domain_identity,
            fingerprint_policy_identity = self.fingerprint_policy_identity,
            fingerprint_admission_surface = SHARED_PETRI_FINGERPRINT_ADMISSION_SURFACE,
            fingerprint_admission_semantics = SHARED_PETRI_FINGERPRINT_ADMISSION_SEMANTICS,
            fingerprint_admission_compatible_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES,
            fingerprint_admission_default_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_DEFAULT_FRONTEND_FAMILIES,
            fingerprint_admission_blocked_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES,
            cache_namespace_identity = self.cache_namespace_identity,
            cache_reuse_policy = SHARED_PETRI_CACHE_REUSE_POLICY,
            cache_digest = self.cache_digest,
            artifact_identity = SHARED_PETRI_NATIVE_CONTRACT_ARTIFACT_IDENTITY,
            artifact_identity_status = SHARED_PETRI_ARTIFACT_IDENTITY_STATUS,
            artifact_digest_status = SHARED_PETRI_ARTIFACT_DIGEST_STATUS,
            shard_compatibility_status = SHARED_PETRI_SHARD_COMPATIBILITY_STATUS,
            shard_compatibility_scope = SHARED_PETRI_SHARD_COMPATIBILITY_SCOPE,
            shard_identity_status = SHARED_PETRI_SHARD_IDENTITY_STATUS,
            shard_identity_provider = SHARED_PETRI_SHARD_IDENTITY_PROVIDER,
            shard_identity_key = shard_identity_key,
            shard_required_fields = SHARED_PETRI_SHARD_REQUIRED_FIELDS,
            fingerprint_compatibility_status = SHARED_PETRI_FINGERPRINT_COMPATIBILITY_STATUS,
            cache_compatibility_status = SHARED_PETRI_CACHE_COMPATIBILITY_STATUS,
            cache_fingerprint_compatibility = SHARED_PETRI_CACHE_FINGERPRINT_COMPATIBILITY,
            parity_receipt_schema = SHARED_PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
            parity_receipt_status = SHARED_PETRI_NATIVE_PARITY_RECEIPT_STATUS,
            parity_receipt_blocker_code = SHARED_PETRI_NATIVE_PARITY_RECEIPT_BLOCKER_CODE,
            parity_receipt_gate_api = SHARED_PETRI_NATIVE_PARITY_RECEIPT_GATE_API,
            parity_receipt_required_evidence = SHARED_PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
            validation_receipt_schema = VALIDATION_RECEIPT_SCHEMA,
            validation_receipt_schema_version = VALIDATION_RECEIPT_SCHEMA_VERSION,
            validation_receipt_status = SHARED_PETRI_VALIDATION_RECEIPT_STATUS,
            validation_receipt_blocker_code = SHARED_PETRI_VALIDATION_RECEIPT_BLOCKER_CODE,
            validation_receipt_gate_api = SHARED_PETRI_VALIDATION_RECEIPT_GATE_API,
            validation_receipt_required_evidence =
                SHARED_PETRI_VALIDATION_RECEIPT_REQUIRED_EVIDENCE,
            production_gate = SHARED_PETRI_PRODUCTION_GATE,
            production_gate_status = SHARED_PETRI_PRODUCTION_GATE_STATUS,
            production_gate_required_receipts = SHARED_PETRI_PRODUCTION_GATE_REQUIRED_RECEIPTS,
            compile_artifact_handoff_prerequisite =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PREREQUISITE,
            compile_artifact_handoff_schema = compile_artifact_handoff.schema,
            compile_artifact_handoff_schema_version = compile_artifact_handoff.schema_version,
            compile_artifact_handoff_owner = SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_OWNER,
            compile_artifact_handoff_evidence_source =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_SOURCE,
            compile_artifact_handoff_descriptor_source =
                compile_artifact_handoff.descriptor_source,
            compile_artifact_handoff_surface = compile_artifact_handoff.surface_name,
            compile_artifact_handoff_api = compile_artifact_handoff.api,
            compile_artifact_handoff_input_type = compile_artifact_handoff.input_type,
            compile_artifact_handoff_evidence_type = compile_artifact_handoff.evidence_type,
            compile_artifact_handoff_blocker_type = compile_artifact_handoff.blocker_type,
            compile_artifact_handoff_required_fields = compile_artifact_handoff.required_fields,
            compile_artifact_handoff_status_codes = compile_artifact_handoff.status_codes,
            compile_artifact_handoff_blocker_codes = compile_artifact_handoff.blocker_codes,
            compile_artifact_handoff_blocker_codes_count =
                compile_artifact_handoff.blocker_code_count,
            compile_artifact_handoff_bridge_status = compile_artifact_handoff.bridge_status,
            compile_artifact_handoff_producer_requirements =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCER_REQUIREMENTS,
            compile_artifact_handoff_consumer_requirements =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_CONSUMER_REQUIREMENTS,
            compile_artifact_handoff_production_requirement =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCTION_REQUIREMENT,
            compile_artifact_handoff_status = SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_STATUS,
            compile_artifact_handoff_blocker_code =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODE,
            native_adoption_blocker = SHARED_PETRI_NATIVE_ADOPTION_BLOCKER,
            token_width = self.token_width,
            places = num_places,
            transitions = num_transitions,
            packed_len = self.packed_len,
            excluded_places = self.excluded_place_count,
            pack_capacity = pack_capacity,
        )
    }

    fn render_shared_engine_readiness_evidence_row(
        &self,
        scope: &str,
        num_places: usize,
        num_transitions: usize,
        pack_capacity: usize,
    ) -> String {
        let export_set_digest = shared_petri_contract_digest(
            "export_set",
            &[
                SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
                SHARED_PETRI_NATIVE_CANDIDATE_KEY,
                SHARED_PETRI_NATIVE_LANE_IDENTITY,
            ],
        );
        let alias_resolution_digest = shared_petri_contract_digest(
            "alias_resolution",
            &[
                SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
                SHARED_PETRI_NATIVE_CONTRACT_TARGET_ABI,
            ],
        );
        let export_surface_digest = shared_petri_contract_digest(
            "export_surface",
            &[
                &export_set_digest,
                &alias_resolution_digest,
                &self.layout_checksum,
            ],
        );
        let artifact_link_digest = shared_petri_contract_digest(
            "artifact_link",
            &[
                &export_surface_digest,
                &self.semantic_checksum,
                SHARED_PETRI_NATIVE_CONTRACT_TARGET_ABI,
            ],
        );
        let native_requirements_digest = shared_petri_contract_digest(
            "native_requirements",
            &[
                &self.layout_checksum,
                &self.semantic_checksum,
                &self.payload_checksum,
                SHARED_PETRI_NATIVE_VECTOR_OPS,
                SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
            ],
        );
        let shard_identity_key = shared_petri_shard_identity(
            &self.layout_checksum,
            &self.semantic_checksum,
            &self.manifest_checksum,
        );
        let readiness_identity = shared_petri_readiness_identity(&self.manifest_checksum);
        let prepared_trust_ir_reuse_identity =
            shared_petri_prepared_trust_ir_reuse_identity(&self.semantic_checksum);
        let compile_artifact_handoff = SharedPetriCompileArtifactHandoffBridge::current();

        format!(
            "{scope} {row_kind} schema={schema} schema_version={schema_version} \
             readiness_identity={readiness_identity} readiness_mode=validation_only \
             prepared_trust_ir_reuse_identity={prepared_trust_ir_reuse_identity} \
             prepared_trust_ir_reuse_identity_status={prepared_trust_ir_reuse_status} \
             origin_frontend={origin_frontend} diagnostic_module_family={origin_frontend} \
             shared_engine_component={shared_engine_component} digest_source={digest_source} \
             prepared_semantic_digest={prepared_semantic_digest} artifact_link_digest={artifact_link_digest} \
             artifact_cache_digest={artifact_cache_digest} batch_artifact_identity={batch_artifact_identity} \
             batch_artifact_identity_kind=contract_template batch_artifact_digest_status={artifact_digest_status} \
             export_set_identity_basis={export_set_identity_basis} export_set_digest={export_set_digest} \
             alias_resolution_identity_basis={alias_resolution_identity_basis} alias_resolution_digest={alias_resolution_digest} \
             export_surface_identity_basis={export_surface_identity_basis} export_surface_digest={export_surface_digest} \
             native_requirements_identity_basis={native_requirements_identity_basis} native_requirements_digest={native_requirements_digest} \
             readiness_owner={shared_owner} primary_beneficiary={first_beneficiary} secondary_beneficiary={second_beneficiary} \
             first_beneficiary={first_beneficiary} second_beneficiary={second_beneficiary} \
             readiness_frontend_families={readiness_frontend_families} future_frontend_family_readiness={future_frontend_family_readiness} \
             adoption_level={adoption_level} compatible_frontend_families={compatible_frontend_families} \
             default_compatible_frontend_families={default_compatible_frontend_families} \
             downstream_beneficiary_families={downstream_beneficiary_families} \
             remaining_compatible_frontend_families={remaining_compatible_frontend_families} \
             frontend_family_blockers={frontend_family_blockers} \
             extraction_status={extraction_status} \
             blocker_status={blocker_status} prepared_identity_basis={prepared_identity_basis} \
             generic_prerequisites={generic_prerequisites} acceptance_test={acceptance_test} \
             acceptance_evidence={acceptance_evidence} \
             checksum_scope=layout_semantic_source_payload_cache frontend_fields_in_checksums={frontend_fields_in_checksums} \
             prepared_trust_ir_reuse={prepared_trust_ir_reuse} \
             prepared_trust_ir_reuse_scope={prepared_trust_ir_reuse_scope} source_kind=mcc_petri payload_kind=mcc_petri \
             storage_kind=petri_marking layout_kind=petri_marking layout_identity={layout_identity} symbol={symbol} \
             manifest_checksum={manifest_checksum} layout_checksum={layout_checksum} semantic_checksum={semantic_checksum} \
             source_checksum={source_checksum} payload_checksum={payload_checksum} validation_only=true readiness_status=validation_only \
             install_authority=validation_only admission_disposition=profile_only \
             validation_receipt_required=true validation_receipt_schema={validation_receipt_schema} \
             validation_receipt_schema_version={validation_receipt_schema_version} \
             validation_receipt_status={validation_receipt_status} \
             validation_receipt_blocker_code={validation_receipt_blocker_code} \
             validation_receipt_gate_api={validation_receipt_gate_api} \
             validation_receipt_required_evidence={validation_receipt_required_evidence} \
             shard_compatibility_status={shard_compatibility_status} shard_compatibility_scope={shard_compatibility_scope} \
             shard_identity_status={shard_identity_status} \
             shard_identity_provider={shard_identity_provider} shard_identity_key={shard_identity_key} \
             shard_required_fields={shard_required_fields} trust_ir_shard_identity_status={shard_identity_status} \
             trust_cg_shard_identity_status={shard_identity_status} \
             fingerprint_compatibility_status={fingerprint_compatibility_status} \
             fingerprint_compatibility=canonical_bytes_sha256 fingerprint_domain_identity={fingerprint_domain_identity} \
             fingerprint_domain_acceptance_identity={fingerprint_policy_identity} \
             fingerprint_admission_surface={fingerprint_admission_surface} fingerprint_admission_semantics={fingerprint_admission_semantics} \
             fingerprint_admission_compatible_frontend_families={fingerprint_admission_compatible_frontend_families} \
             fingerprint_admission_default_frontend_families={fingerprint_admission_default_frontend_families} \
             fingerprint_admission_blocked_frontend_families={fingerprint_admission_blocked_frontend_families} \
             cache_compatibility_status={cache_compatibility_status} \
             cache_fingerprint_compatibility={cache_fingerprint_compatibility} cache_namespace_identity={cache_namespace_identity} \
             cache_reuse_policy={cache_reuse_policy} cache_digest={cache_digest} artifact_identity={artifact_identity} \
             artifact_identity_kind=contract_template artifact_identity_status={artifact_identity_status} \
             artifact_digest_status={artifact_digest_status} parity_receipt_required=true \
             parity_receipt_schema={parity_receipt_schema} parity_receipt_status={parity_receipt_status} \
             parity_receipt_blocker_code={parity_receipt_blocker_code} \
             parity_receipt_gate_api={parity_receipt_gate_api} parity_receipt_required_evidence={parity_receipt_required_evidence} \
             production_gate={production_gate} production_gate_status={production_gate_status} \
             production_gate_required_receipts={production_gate_required_receipts} \
             shared_engine_prerequisite={compile_artifact_handoff_prerequisite} \
             compile_artifact_handoff_schema={compile_artifact_handoff_schema} \
             compile_artifact_handoff_schema_version={compile_artifact_handoff_schema_version} \
             compile_artifact_handoff_owner={compile_artifact_handoff_owner} \
             compile_artifact_handoff_evidence_source={compile_artifact_handoff_evidence_source} \
             compile_artifact_handoff_descriptor_source={compile_artifact_handoff_descriptor_source} \
             compile_artifact_handoff_surface={compile_artifact_handoff_surface} \
             compile_artifact_handoff_api={compile_artifact_handoff_api} \
             compile_artifact_handoff_input_type={compile_artifact_handoff_input_type} \
             compile_artifact_handoff_evidence_type={compile_artifact_handoff_evidence_type} \
             compile_artifact_handoff_blocker_type={compile_artifact_handoff_blocker_type} \
             compile_artifact_handoff_required_fields={compile_artifact_handoff_required_fields} \
             compile_artifact_handoff_status_codes={compile_artifact_handoff_status_codes} \
             compile_artifact_handoff_blocker_codes={compile_artifact_handoff_blocker_codes} \
             compile_artifact_handoff_blocker_codes_count={compile_artifact_handoff_blocker_codes_count} \
             compile_artifact_handoff_bridge_status={compile_artifact_handoff_bridge_status} \
             compile_artifact_handoff_producer_requirements={compile_artifact_handoff_producer_requirements} \
             compile_artifact_handoff_consumer_requirements={compile_artifact_handoff_consumer_requirements} \
             compile_artifact_handoff_production_requirement={compile_artifact_handoff_production_requirement} \
             compile_artifact_handoff_status={compile_artifact_handoff_status} \
             compile_artifact_handoff_blocker_code={compile_artifact_handoff_blocker_code} \
             native_adoption_blocker={native_adoption_blocker} \
             production_selected=false fail_closed=true \
             places={places} transitions={transitions} pack_capacity={pack_capacity}",
            row_kind = SHARED_PETRI_NATIVE_ENGINE_READINESS_ROW_KIND,
            schema = SHARED_PETRI_NATIVE_ENGINE_READINESS_SCHEMA,
            schema_version = SHARED_PETRI_NATIVE_ENGINE_READINESS_SCHEMA_VERSION,
            readiness_identity = readiness_identity,
            prepared_trust_ir_reuse_identity = prepared_trust_ir_reuse_identity,
            prepared_trust_ir_reuse_status = SHARED_PETRI_PREPARED_TRUST_IR_REUSE,
            origin_frontend = SHARED_ENGINE_ORIGIN_FRONTEND,
            shared_engine_component = SHARED_PETRI_BATCH_ARTIFACT_COMPONENT,
            digest_source = SHARED_PETRI_DIGEST_SOURCE,
            prepared_semantic_digest = self.semantic_checksum,
            artifact_link_digest = artifact_link_digest,
            artifact_cache_digest = self.cache_digest,
            batch_artifact_identity = SHARED_PETRI_NATIVE_CONTRACT_ARTIFACT_IDENTITY,
            export_set_identity_basis = SHARED_PETRI_EXPORT_SET_IDENTITY_BASIS,
            export_set_digest = export_set_digest,
            alias_resolution_identity_basis = SHARED_PETRI_ALIAS_RESOLUTION_IDENTITY_BASIS,
            alias_resolution_digest = alias_resolution_digest,
            export_surface_identity_basis = SHARED_PETRI_EXPORT_SURFACE_IDENTITY_BASIS,
            export_surface_digest = export_surface_digest,
            native_requirements_identity_basis = SHARED_PETRI_NATIVE_REQUIREMENTS_IDENTITY_BASIS,
            native_requirements_digest = native_requirements_digest,
            shared_owner = SHARED_ENGINE_OWNER,
            first_beneficiary = SHARED_ENGINE_FIRST_BENEFICIARY,
            second_beneficiary = SHARED_PETRI_READINESS_SECOND_BENEFICIARY,
            readiness_frontend_families = SHARED_PETRI_READINESS_FRONTEND_FAMILIES,
            future_frontend_family_readiness = SHARED_PETRI_FUTURE_FRONTEND_FAMILY_READINESS,
            adoption_level = SHARED_ENGINE_ADOPTION_LEVEL,
            compatible_frontend_families = SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            default_compatible_frontend_families =
                SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
            downstream_beneficiary_families = SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES,
            remaining_compatible_frontend_families =
                SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
            frontend_family_blockers = SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
            extraction_status = SHARED_ENGINE_EXTRACTION_STATUS,
            blocker_status = SHARED_ENGINE_BLOCKER_STATUS,
            prepared_identity_basis = SHARED_PETRI_PREPARED_IDENTITY_BASIS,
            generic_prerequisites = SHARED_ENGINE_GENERIC_PREREQUISITES,
            acceptance_test = SHARED_ENGINE_ACCEPTANCE_TEST,
            acceptance_evidence = SHARED_ENGINE_ACCEPTANCE_EVIDENCE,
            frontend_fields_in_checksums = SHARED_PETRI_FRONTEND_FIELDS_IN_CHECKSUMS,
            prepared_trust_ir_reuse = SHARED_PETRI_PREPARED_TRUST_IR_REUSE,
            prepared_trust_ir_reuse_scope = SHARED_PETRI_PREPARED_TRUST_IR_REUSE_SCOPE,
            layout_identity = SHARED_PETRI_KERNEL_LAYOUT_KIND,
            symbol = SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
            manifest_checksum = self.manifest_checksum,
            layout_checksum = self.layout_checksum,
            semantic_checksum = self.semantic_checksum,
            source_checksum = self.source_checksum,
            payload_checksum = self.payload_checksum,
            artifact_digest_status = SHARED_PETRI_ARTIFACT_DIGEST_STATUS,
            validation_receipt_schema = VALIDATION_RECEIPT_SCHEMA,
            validation_receipt_schema_version = VALIDATION_RECEIPT_SCHEMA_VERSION,
            validation_receipt_status = SHARED_PETRI_VALIDATION_RECEIPT_STATUS,
            validation_receipt_blocker_code = SHARED_PETRI_VALIDATION_RECEIPT_BLOCKER_CODE,
            validation_receipt_gate_api = SHARED_PETRI_VALIDATION_RECEIPT_GATE_API,
            validation_receipt_required_evidence =
                SHARED_PETRI_VALIDATION_RECEIPT_REQUIRED_EVIDENCE,
            shard_compatibility_status = SHARED_PETRI_SHARD_COMPATIBILITY_STATUS,
            shard_compatibility_scope = SHARED_PETRI_SHARD_COMPATIBILITY_SCOPE,
            shard_identity_status = SHARED_PETRI_SHARD_IDENTITY_STATUS,
            shard_identity_provider = SHARED_PETRI_SHARD_IDENTITY_PROVIDER,
            shard_identity_key = shard_identity_key,
            shard_required_fields = SHARED_PETRI_SHARD_REQUIRED_FIELDS,
            fingerprint_compatibility_status = SHARED_PETRI_FINGERPRINT_COMPATIBILITY_STATUS,
            fingerprint_domain_identity = self.fingerprint_domain_identity,
            fingerprint_policy_identity = self.fingerprint_policy_identity,
            fingerprint_admission_surface = SHARED_PETRI_FINGERPRINT_ADMISSION_SURFACE,
            fingerprint_admission_semantics = SHARED_PETRI_FINGERPRINT_ADMISSION_SEMANTICS,
            fingerprint_admission_compatible_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES,
            fingerprint_admission_default_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_DEFAULT_FRONTEND_FAMILIES,
            fingerprint_admission_blocked_frontend_families =
                SHARED_PETRI_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES,
            cache_compatibility_status = SHARED_PETRI_CACHE_COMPATIBILITY_STATUS,
            cache_fingerprint_compatibility = SHARED_PETRI_CACHE_FINGERPRINT_COMPATIBILITY,
            cache_namespace_identity = self.cache_namespace_identity,
            cache_reuse_policy = SHARED_PETRI_CACHE_REUSE_POLICY,
            cache_digest = self.cache_digest,
            artifact_identity = SHARED_PETRI_NATIVE_CONTRACT_ARTIFACT_IDENTITY,
            artifact_identity_status = SHARED_PETRI_ARTIFACT_IDENTITY_STATUS,
            production_gate = SHARED_PETRI_PRODUCTION_GATE,
            production_gate_status = SHARED_PETRI_PRODUCTION_GATE_STATUS,
            production_gate_required_receipts = SHARED_PETRI_PRODUCTION_GATE_REQUIRED_RECEIPTS,
            compile_artifact_handoff_prerequisite =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PREREQUISITE,
            compile_artifact_handoff_schema = compile_artifact_handoff.schema,
            compile_artifact_handoff_schema_version = compile_artifact_handoff.schema_version,
            compile_artifact_handoff_owner = SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_OWNER,
            compile_artifact_handoff_evidence_source =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_SOURCE,
            compile_artifact_handoff_descriptor_source =
                compile_artifact_handoff.descriptor_source,
            compile_artifact_handoff_surface = compile_artifact_handoff.surface_name,
            compile_artifact_handoff_api = compile_artifact_handoff.api,
            compile_artifact_handoff_input_type = compile_artifact_handoff.input_type,
            compile_artifact_handoff_evidence_type = compile_artifact_handoff.evidence_type,
            compile_artifact_handoff_blocker_type = compile_artifact_handoff.blocker_type,
            compile_artifact_handoff_required_fields = compile_artifact_handoff.required_fields,
            compile_artifact_handoff_status_codes = compile_artifact_handoff.status_codes,
            compile_artifact_handoff_blocker_codes = compile_artifact_handoff.blocker_codes,
            compile_artifact_handoff_blocker_codes_count =
                compile_artifact_handoff.blocker_code_count,
            compile_artifact_handoff_bridge_status = compile_artifact_handoff.bridge_status,
            compile_artifact_handoff_producer_requirements =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCER_REQUIREMENTS,
            compile_artifact_handoff_consumer_requirements =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_CONSUMER_REQUIREMENTS,
            compile_artifact_handoff_production_requirement =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_PRODUCTION_REQUIREMENT,
            compile_artifact_handoff_status = SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_STATUS,
            compile_artifact_handoff_blocker_code =
                SHARED_PETRI_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODE,
            native_adoption_blocker = SHARED_PETRI_NATIVE_ADOPTION_BLOCKER,
            parity_receipt_schema = SHARED_PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
            parity_receipt_status = SHARED_PETRI_NATIVE_PARITY_RECEIPT_STATUS,
            parity_receipt_blocker_code = SHARED_PETRI_NATIVE_PARITY_RECEIPT_BLOCKER_CODE,
            parity_receipt_gate_api = SHARED_PETRI_NATIVE_PARITY_RECEIPT_GATE_API,
            parity_receipt_required_evidence = SHARED_PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
            places = num_places,
            transitions = num_transitions,
            pack_capacity = pack_capacity,
        )
    }

    fn shared_native_planning_fingerprint_identity_fields(&self) -> Vec<(&'static str, String)> {
        let prepared_trust_ir_reuse_identity =
            shared_petri_prepared_trust_ir_reuse_identity(&self.semantic_checksum);
        let planning_identity =
            SharedNativePlanningIdentity::new(current_native_frontend_families().iter().copied())
                .with_source_fingerprint(self.source_checksum.clone())
                .with_plan_reuse_manifest(
                    prepared_trust_ir_reuse_identity.clone(),
                    self.manifest_checksum.clone(),
                )
                .with_fingerprint_domain_identity(self.fingerprint_domain_identity.clone())
                .with_cas_identity(self.fingerprint_policy_identity.clone())
                .with_cache_identity(self.cache_namespace_identity.clone())
                .with_cache_reuse_policy(SHARED_PETRI_CACHE_REUSE_POLICY);
        planning_identity
            .validate(CheckerSourceKind::MccPetri)
            .unwrap_or_else(|error| {
                panic!("Petri shared planning identity must be valid: {error}")
            });

        vec![
            (
                "row_kind",
                SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_ROW_KIND.to_string(),
            ),
            (
                "schema",
                SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_SCHEMA.to_string(),
            ),
            (
                "schema_version",
                SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_SCHEMA_VERSION.to_string(),
            ),
            (
                "source_kind",
                CheckerSourceKind::MccPetri.code().to_string(),
            ),
            (
                "frontend_kind",
                CheckerSourceKind::MccPetri
                    .frontend_family_code()
                    .to_string(),
            ),
            (
                "planning_identity_status",
                SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_STATUS.to_string(),
            ),
            (
                "planning_identity_digest",
                planning_identity.stable_identity(),
            ),
            (
                "core_native_planning_identity",
                planning_identity.stable_identity(),
            ),
            (
                "frontend_family_scope_identity",
                planning_identity.frontend_family_scope_identity(),
            ),
            (
                "frontend_family_reusable",
                planning_identity.frontend_family_reusable().to_string(),
            ),
            (
                "planning_identity_required_fields",
                SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_REQUIRED_FIELDS.to_string(),
            ),
            (
                "prepared_program_identity",
                SHARED_PETRI_PREPARED_PROGRAM_IDENTITY.to_string(),
            ),
            (
                "candidate_identity",
                SHARED_PETRI_NATIVE_CANDIDATE_KEY.to_string(),
            ),
            (
                "lane_identity",
                SHARED_PETRI_NATIVE_LANE_IDENTITY.to_string(),
            ),
            ("layout_checksum", self.layout_checksum.clone()),
            ("semantic_checksum", self.semantic_checksum.clone()),
            ("source_checksum", self.source_checksum.clone()),
            ("payload_checksum", self.payload_checksum.clone()),
            ("manifest_checksum", self.manifest_checksum.clone()),
            (
                "fingerprint_domain_identity",
                self.fingerprint_domain_identity.clone(),
            ),
            (
                "fingerprint_domain_acceptance_identity",
                self.fingerprint_policy_identity.clone(),
            ),
            (
                "cache_namespace_identity",
                self.cache_namespace_identity.clone(),
            ),
            (
                "cache_reuse_policy",
                SHARED_PETRI_CACHE_REUSE_POLICY.to_string(),
            ),
            ("cache_digest", self.cache_digest.clone()),
            (
                "prepared_trust_ir_reuse_identity",
                prepared_trust_ir_reuse_identity,
            ),
            (
                "trust_cg_batch_cache_reuse_status",
                SHARED_PETRI_TRUST_CG_BATCH_CACHE_REUSE_STATUS.to_string(),
            ),
            (
                "trust_cg_batch_cache_reuse_blocker_code",
                SHARED_PETRI_TRUST_CG_BATCH_CACHE_REUSE_BLOCKER_CODE.to_string(),
            ),
            (
                "validation_receipt_status",
                SHARED_PETRI_VALIDATION_RECEIPT_STATUS.to_string(),
            ),
            (
                "parity_receipt_status",
                SHARED_PETRI_NATIVE_PARITY_RECEIPT_STATUS.to_string(),
            ),
            (
                "callable_receipt_status",
                SHARED_PETRI_NATIVE_CALLABLE_RECEIPT_STATUS.to_string(),
            ),
            (
                "production_gate_status",
                SHARED_PETRI_PRODUCTION_GATE_STATUS.to_string(),
            ),
        ]
    }
}

fn shared_petri_descriptor_digest(setup: &ExplorationSetup) -> String {
    let mut hash = FNV1A64_OFFSET;
    for part in [
        SHARED_ENGINE_PREPARED_PROGRAM_COMPONENT,
        SHARED_ENGINE_ORIGIN_FRONTEND,
        SHARED_ENGINE_OWNER,
        SHARED_ENGINE_FIRST_BENEFICIARY,
        SHARED_ENGINE_SECOND_BENEFICIARY,
        SHARED_ENGINE_GENERIC_PREREQUISITES,
        SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
        SHARED_ENGINE_ADOPTION_MATRIX_FIELDS,
        SHARED_PETRI_PREPARED_PROGRAM_IDENTITY,
        SHARED_PETRI_TRANSITION_DESCRIPTOR,
        SHARED_PETRI_PREDICATE_DESCRIPTOR,
        SHARED_PETRI_NATIVE_CANDIDATE_KEY,
        SHARED_PETRI_NATIVE_LANE_IDENTITY,
        SHARED_PETRI_FINGERPRINT_POLICY,
        SHARED_PETRI_FINGERPRINT_CHAIN_FIELDS,
        SHARED_PETRI_KERNEL_METADATA_SCHEMA,
        SHARED_PETRI_KERNEL_METADATA_IDENTITY_BASIS,
        SHARED_PETRI_KERNEL_METADATA_SOURCE,
        SHARED_PETRI_KERNEL_LAYOUT_KIND,
        SHARED_PETRI_CANONICALIZATION,
    ] {
        hash = fnv1a64_update(hash, part.as_bytes());
        hash = fnv1a64_update(hash, b"\n");
    }
    for value in [
        setup.num_places as u64,
        setup.num_transitions as u64,
        setup.pack_capacity as u64,
    ] {
        hash = fnv1a64_update(hash, &value.to_le_bytes());
    }
    format!("{hash:016x}")
}

fn shared_petri_layout_checksum(setup: &ExplorationSetup) -> String {
    let mut hash = fnv1a64_seeded("layout");
    hash = fnv1a64_update_str(hash, SHARED_ENGINE_ORIGIN_FRONTEND);
    hash = fnv1a64_update_str(hash, SHARED_PETRI_KERNEL_LAYOUT_KIND);
    hash = fnv1a64_update_str(hash, token_width_code(setup.marking_config.width));
    hash = fnv1a64_update_usize(hash, setup.marking_config.num_places);
    hash = fnv1a64_update_usize(hash, setup.marking_config.packed_len);
    hash = fnv1a64_update_usize(hash, setup.pack_capacity);
    for excluded in setup.marking_config.excluded_places() {
        hash = fnv1a64_update_bool(hash, *excluded);
    }
    prefixed_fnv1a64(hash)
}

fn shared_petri_semantic_checksum(setup: &ExplorationSetup, net: Option<&PetriNet>) -> String {
    let mut hash = fnv1a64_seeded("semantic");
    hash = fnv1a64_update_str(hash, SHARED_PETRI_TRANSITION_DESCRIPTOR);
    hash = fnv1a64_update_str(hash, SHARED_PETRI_PREDICATE_DESCRIPTOR);
    hash = fnv1a64_update_usize(hash, setup.num_places);
    hash = fnv1a64_update_usize(hash, setup.num_transitions);
    match net {
        Some(net) => {
            hash = fnv1a64_update_str(hash, "net_available");
            for transition in &net.transitions {
                hash = fnv1a64_update_str(hash, &transition.id);
                hash = fnv1a64_update_optional_str(hash, transition.name.as_deref());
                hash = fnv1a64_update_usize(hash, transition.inputs.len());
                for arc in &transition.inputs {
                    hash = fnv1a64_update_u64(hash, u64::from(arc.place.0));
                    hash = fnv1a64_update_u64(hash, arc.weight);
                }
                hash = fnv1a64_update_usize(hash, transition.outputs.len());
                for arc in &transition.outputs {
                    hash = fnv1a64_update_u64(hash, u64::from(arc.place.0));
                    hash = fnv1a64_update_u64(hash, arc.weight);
                }
            }
        }
        None => {
            hash = fnv1a64_update_str(hash, "setup_only");
        }
    }
    prefixed_fnv1a64(hash)
}

fn shared_petri_source_checksum(setup: &ExplorationSetup, net: Option<&PetriNet>) -> String {
    let mut hash = fnv1a64_seeded("source");
    hash = fnv1a64_update_str(hash, SHARED_ENGINE_ORIGIN_FRONTEND);
    hash = fnv1a64_update_usize(hash, setup.num_places);
    hash = fnv1a64_update_usize(hash, setup.num_transitions);
    for byte in setup.initial_packed.iter() {
        hash = fnv1a64_update_u64(hash, u64::from(*byte));
    }
    if let Some(net) = net {
        hash = fnv1a64_update_optional_str(hash, net.name.as_deref());
        for place in &net.places {
            hash = fnv1a64_update_str(hash, &place.id);
            hash = fnv1a64_update_optional_str(hash, place.name.as_deref());
        }
        for marking in &net.initial_marking {
            hash = fnv1a64_update_u64(hash, *marking);
        }
    }
    prefixed_fnv1a64(hash)
}

fn shared_petri_payload_checksum(
    layout_checksum: &str,
    semantic_checksum: &str,
    source_checksum: &str,
) -> String {
    let mut hash = fnv1a64_seeded("payload");
    hash = fnv1a64_update_str(hash, SHARED_PETRI_NATIVE_CONTRACT_FRONTEND_PAYLOAD_IDENTITY);
    hash = fnv1a64_update_str(hash, SHARED_PETRI_NATIVE_CONTRACT_SYMBOL);
    hash = fnv1a64_update_str(hash, layout_checksum);
    hash = fnv1a64_update_str(hash, semantic_checksum);
    hash = fnv1a64_update_str(hash, source_checksum);
    prefixed_fnv1a64(hash)
}

fn shared_petri_cache_digest(
    fingerprint_domain_identity: &str,
    fingerprint_policy_identity: &str,
    cache_namespace_identity: &str,
    payload_checksum: &str,
) -> String {
    let mut hash = fnv1a64_seeded("cache");
    hash = fnv1a64_update_str(hash, fingerprint_domain_identity);
    hash = fnv1a64_update_str(hash, fingerprint_policy_identity);
    hash = fnv1a64_update_str(hash, cache_namespace_identity);
    hash = fnv1a64_update_str(hash, SHARED_PETRI_CACHE_REUSE_POLICY);
    hash = fnv1a64_update_str(hash, payload_checksum);
    prefixed_fnv1a64(hash)
}

fn shared_petri_manifest_checksum(
    layout_checksum: &str,
    semantic_checksum: &str,
    source_checksum: &str,
    payload_checksum: &str,
    cache_digest: &str,
    fingerprint_domain_identity: &str,
) -> String {
    let mut hash = fnv1a64_seeded("manifest");
    for part in [
        SHARED_PETRI_NATIVE_CONTRACT_MANIFEST_SCHEMA,
        SHARED_PETRI_KERNEL_METADATA_SCHEMA,
        SHARED_PETRI_NATIVE_CONTRACT_ARTIFACT_IDENTITY,
        SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
        SHARED_PETRI_NATIVE_CONTRACT_TARGET_ABI,
        layout_checksum,
        semantic_checksum,
        source_checksum,
        payload_checksum,
        cache_digest,
        fingerprint_domain_identity,
    ] {
        hash = fnv1a64_update_str(hash, part);
    }
    prefixed_fnv1a64(hash)
}

fn shared_petri_contract_digest(label: &str, parts: &[&str]) -> String {
    let mut hash = fnv1a64_seeded(label);
    for part in parts {
        hash = fnv1a64_update_str(hash, part);
    }
    prefixed_fnv1a64(hash)
}

fn shared_petri_readiness_identity(manifest_checksum: &str) -> String {
    format!("mcc_petri_shared_native_readiness:{manifest_checksum}")
}

fn shared_petri_prepared_trust_ir_reuse_identity(semantic_checksum: &str) -> String {
    format!(
        "{}:{}:{}:{}",
        SHARED_PETRI_PREPARED_TRUST_IR_REUSE_IDENTITY_PREFIX,
        SHARED_PETRI_PREPARED_TRUST_IR_REUSE_SCOPE,
        SHARED_PETRI_PREPARED_IDENTITY_BASIS,
        semantic_checksum
    )
}

fn shared_petri_shard_identity(
    layout_checksum: &str,
    semantic_checksum: &str,
    manifest_checksum: &str,
) -> String {
    shared_petri_contract_digest(
        "shard_identity",
        &[
            SHARED_ENGINE_ORIGIN_FRONTEND,
            SHARED_PETRI_NATIVE_CONTRACT_SYMBOL,
            SHARED_PETRI_SHARD_REQUIRED_FIELDS,
            layout_checksum,
            semantic_checksum,
            manifest_checksum,
        ],
    )
}

fn render_shared_planning_fingerprint_identity_evidence_row(
    scope: &str,
    fields: &[(&'static str, String)],
) -> String {
    let row_kind = fields
        .iter()
        .find_map(|(key, value)| (*key == "row_kind").then_some(value.as_str()))
        .unwrap_or(SHARED_PETRI_PLANNING_FINGERPRINT_IDENTITY_ROW_KIND);
    let mut row = format!("{scope} {row_kind}");
    for (key, value) in fields {
        if *key == "row_kind" {
            continue;
        }
        row.push(' ');
        row.push_str(key);
        row.push('=');
        row.push_str(value);
    }
    row
}

fn shared_petri_fingerprint_domain_key(
    layout_checksum: &str,
) -> Result<FingerprintDomainKey, tla_mc_core::SharedFingerprintIdentityRejection> {
    FingerprintDomainKey::builder(SharedFingerprintAlgorithm::CanonicalBytesSha256)
        .helper_symbol(SHARED_PETRI_FINGERPRINT_HELPER_SYMBOL)
        .seed_identity(SHARED_PETRI_FINGERPRINT_SEED_IDENTITY)
        .canonical_payload(FingerprintCanonicalPayload::new(
            SharedFingerprintValueKind::MarkingVector,
            SharedFingerprintCanonicalDomain::new(
                SHARED_PETRI_FINGERPRINT_CANONICAL_DOMAIN,
                SHARED_PETRI_FINGERPRINT_CANONICAL_DOMAIN_VERSION,
            ),
            SHARED_PETRI_FINGERPRINT_CANONICALIZATION_VERSION,
            128,
        ))
        .layout_digest(layout_checksum)
        .projection(FingerprintDomainProjection::Full)
        .storage_policy(
            FingerprintDomainStoragePolicy::new(
                SharedDedupScope::StateSpace,
                SharedDedupStorageKind::EvidenceOnly,
            )
            .with_storage_config_identity(SHARED_PETRI_FINGERPRINT_STORAGE_CONFIG),
        )
        .collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality)
        .build()
}

fn token_width_code(width: TokenWidth) -> &'static str {
    match width {
        TokenWidth::U8 => "u8",
        TokenWidth::U16 => "u16",
        TokenWidth::U64 => "u64",
    }
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
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

fn fnv1a64_seeded(label: &str) -> u64 {
    let hash = fnv1a64_update(FNV1A64_OFFSET, b"ty-petri-shared-native");
    fnv1a64_update_str(hash, label)
}

fn fnv1a64_update_str(hash: u64, value: &str) -> u64 {
    let hash = fnv1a64_update(hash, value.as_bytes());
    fnv1a64_update(hash, b"\n")
}

fn fnv1a64_update_optional_str(hash: u64, value: Option<&str>) -> u64 {
    match value {
        Some(value) => fnv1a64_update_str(hash, value),
        None => fnv1a64_update_str(hash, "none"),
    }
}

fn fnv1a64_update_usize(hash: u64, value: usize) -> u64 {
    fnv1a64_update_u64(hash, value as u64)
}

fn fnv1a64_update_u64(hash: u64, value: u64) -> u64 {
    let hash = fnv1a64_update(hash, &value.to_le_bytes());
    fnv1a64_update(hash, b"\n")
}

fn fnv1a64_update_bool(hash: u64, value: bool) -> u64 {
    fnv1a64_update_str(hash, if value { "true" } else { "false" })
}

fn prefixed_fnv1a64(hash: u64) -> String {
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionInfo};

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn simple_net() -> PetriNet {
        PetriNet {
            name: Some("descriptor-fixture".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        }
    }

    fn evidence_field<'a>(row: &'a str, key: &str) -> &'a str {
        let prefix = format!("{key}=");
        row.split_whitespace()
            .find_map(|field| field.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("{key} should be present in evidence row: {row}"))
    }

    fn planning_identity_evidence_value(value: &str) -> String {
        value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                    ch
                } else {
                    '_'
                }
            })
            .collect()
    }

    #[test]
    fn shared_native_candidate_descriptor_uses_frontend_neutral_vocabulary() {
        let net = simple_net();
        let setup = ExplorationSetup::analyze(&net);
        let descriptor = SharedPetriPreparedNativeCandidateDescriptor::for_net(
            &net,
            ExactOrUnknownStatus::Unknown,
        );

        let row = descriptor.render_evidence_row("MCC");
        let setup_row =
            setup.render_shared_native_candidate_evidence_row("MCC", ExactOrUnknownStatus::Unknown);

        assert!(row.starts_with("MCC prepared_native_candidate_shared_vocab "));
        assert!(setup_row.starts_with("MCC prepared_native_candidate_shared_vocab "));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains("origin_frontend=mcc_petri"));
        assert!(row.contains("shared_owner=shared_high_performance_engine"));
        assert!(row.contains("first_beneficiary=mcc_petri_runtime_storage"));
        assert!(row.contains(
            "second_beneficiary=trust_cg_batch_identity_contract,ay_analytical,witness_replay"
        ));
        assert!(row.contains("extraction_status=frontend-local-with-tracked-extraction"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains("adoption_matrix_fields=origin_frontend,shared_owner,first_beneficiary,second_beneficiary,compatible_frontend_families,default_compatible_frontend_families,downstream_beneficiary_families,remaining_compatible_frontend_families,frontend_family_blockers,generic_prerequisites,shared_engine_prerequisite,compile_artifact_handoff_schema,compile_artifact_handoff_owner,compile_artifact_handoff_status,compile_artifact_handoff_blocker_code,native_adoption_blocker,exact_or_unknown,frontend_neutral_kernel_layout_fingerprint,validation_receipt_status,parity_receipt_status,callable_receipt_status,production_gate_status,acceptance_test,acceptance_evidence"));
        assert!(row.contains("generic_prerequisites=prepared_checker_program_descriptor,marking_storage_identity,transition_relation_descriptor,state_predicate_descriptor,native_candidate_descriptor,validation_plan_descriptor"));
        assert!(row.contains("acceptance_test=cargo_test_-p_tla-petri_--lib_explorer::setup"));
        assert!(row.contains("acceptance_evidence=prepared_native_candidate_row,shared_native_contract_manifest_row,shared_native_engine_readiness_row"));
        assert!(row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(row.contains("transition_descriptor=shared_petri_transition_relation"));
        assert!(row.contains("predicate_descriptor=shared_petri_state_predicate"));
        assert!(row.contains("candidate_key=trust_cg_native"));
        assert!(row.contains("lane_identity=shared_native_successor"));
        assert!(row.contains("fingerprint_chain_fields=prepared_program_identity,transition_relation_identity,predicate_identity,candidate_identity,lane_identity,marking_layout_identity"));
        assert!(row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(row.contains(
            "fingerprint_domain_acceptance_identity=accepted_fail_closed_fingerprint_domain"
        ));
        assert!(
            row.contains("fingerprint_admission_surface=shared_fingerprint_state_vector_admission")
        );
        assert!(row.contains(
            "fingerprint_admission_semantics=default_consumer,compatible_consumer,blocked"
        ));
        assert!(row.contains("fingerprint_admission_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(row.contains("fingerprint_admission_default_frontend_families=tla_plus,mcc_petri"));
        assert!(row.contains("fingerprint_admission_blocked_frontend_families=future_importer:awaiting_registered_importer_frontend"));
        assert!(
            row.contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1")
        );
        assert!(row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(row.contains("cache_digest=fnv1a64:"));
        assert!(row.contains("kernel_metadata_schema=tla_ir.whole_program_kernel_metadata.v1"));
        assert!(row.contains("kernel_metadata_source=local_tla_ir_compatible"));
        assert!(row.contains(
            "kernel_metadata_blocker=tla-ir_metadata_crate_not_a_default_tla-petri_dependency"
        ));
        assert!(row.contains("kernel_layout_kind=petri_marking_i64_vector"));
        assert!(row.contains("frontend_neutral_kernel_layout_fingerprint_algorithm=fnv1a64"));
        assert!(row.contains("frontend_neutral_kernel_layout_fingerprint="));
        assert!(row.contains("manifest_checksum=fnv1a64:"));
        assert!(row.contains("layout_checksum=fnv1a64:"));
        assert!(row.contains("semantic_checksum=fnv1a64:"));
        assert!(row.contains("source_checksum=fnv1a64:"));
        assert!(row.contains("payload_checksum=fnv1a64:"));
        assert!(row.contains("shared_engine_prerequisite=trust_cg_petri_compile_artifact_handoff"));
        assert!(row.contains(
            "compile_artifact_handoff_schema=trust-cg.petri.native_successor.compile_artifact_handoff.v1"
        ));
        assert!(row.contains("compile_artifact_handoff_owner=trust-cg"));
        assert!(row.contains(
            "compile_artifact_handoff_evidence_source=trust-cg.petri_native_successor_compile_artifact_handoff"
        ));
        assert_compile_artifact_handoff_bridge_blocks_production(&row);
        assert!(row.contains(
            "compile_artifact_handoff_blocker_code=missing_trust_cg_petri_compile_artifact_handoff"
        ));
        assert!(row.contains(
            "native_adoption_blocker=trust_cg_petri_compile_artifact_handoff_required_before_native_adoption"
        ));
        assert!(row.contains("validation_status=validation_unknown"));
        assert!(row.contains("exact_or_unknown=unknown"));
        assert!(row.contains("fail_closed=true"));
        assert!(row.contains("model_identity=generic"));
    }

    #[test]
    fn shared_native_contract_maps_petri_successor_predicate_prerequisites() {
        let net = simple_net();
        let setup = ExplorationSetup::analyze(&net);
        let contract = setup.shared_native_contract();

        contract
            .validate()
            .expect("Petri shared native contract should validate");
        assert_eq!(contract.source_kind, CheckerSourceKind::MccPetri);
        assert_eq!(contract.payload_kind, PreparedProgramPayloadKind::MccPetri);
        assert_eq!(contract.storage_kind, PreparedStorageKind::PetriMarking);
        assert_eq!(
            contract.contract_kind,
            SharedNativeContractKind::SuccessorKernel
        );
        assert_eq!(contract.lane_kind, SetupTraceLaneKind::Native);
        assert_eq!(contract.abi.symbol, SHARED_PETRI_NATIVE_CONTRACT_SYMBOL);
        assert_eq!(contract.layout.kind, SharedNativeLayoutKind::PetriMarking);
        assert_eq!(contract.layout.identity, SHARED_PETRI_KERNEL_LAYOUT_KIND);
        assert_eq!(contract.layout.state_len, Some(2));
        assert_eq!(contract.layout.vector_contracts.len(), 1);
        assert_eq!(
            contract.layout.vector_contracts[0].identity,
            SHARED_PETRI_NATIVE_VECTOR_IDENTITY
        );
        assert_eq!(contract.layout.vector_contracts[0].logical_lanes, 2);
        assert_eq!(contract.layout.vector_contracts[0].element_bits, 64);
        assert_eq!(
            contract.layout.vector_contracts[0]
                .operations_identity
                .as_deref(),
            Some(SHARED_PETRI_NATIVE_VECTOR_OPS)
        );
        assert!(contract.layout.vector_contracts[0]
            .feature_guards
            .contains(&SHARED_PETRI_NATIVE_VECTOR_GUARD.to_string()));
        assert!(contract.evidence_policy.fail_closed);
        assert_eq!(
            contract.evidence_policy.required_evidence_codes(),
            vec![
                "manifest_metadata",
                "layout_checksum",
                "semantic_checksum",
                "validation_receipt"
            ]
        );
        assert!(!contract.admission.can_publish_callable());

        let row = setup.render_shared_native_contract_evidence_row("MCC");
        assert!(row.starts_with("MCC shared_native_contract "));
        assert!(row.contains("schema=ty.shared.native_contract.v1"));
        assert!(row.contains("source_kind=mcc_petri"));
        assert!(row.contains("frontend_kind=mcc_petri"));
        assert!(row.contains("payload_kind=mcc_petri"));
        assert!(row.contains("storage_kind=petri_marking"));
        assert!(row.contains("contract_kind=successor_kernel"));
        assert!(row.contains("lane_kind=native"));
        assert!(row.contains("symbol=petri_marking_successor_predicate_batch"));
        assert!(row.contains("abi_params=input_markings:ptr,input_count:u32,place_count:u32,transition_plan:ptr,predicate_plan:ptr,output_markings:ptr,output_parent_indices:ptr,output_counts:ptr,diagnostics:ptr"));
        assert!(row.contains("abi_returns=u32"));
        assert!(row.contains("layout_kind=petri_marking"));
        assert!(row.contains("layout_identity=petri_marking_i64_vector"));
        assert!(row.contains("prepared_program_identity=mcc_petri.prepared_program"));
        assert!(row.contains("candidate_identity=trust_cg_native"));
        assert!(row.contains("lane_identity=shared_native_successor"));
        assert!(row.contains("source_fingerprint=fnv1a64:"));
        assert!(row.contains("plan_reuse_manifest_id=trust_cg_prepared_trust_ir_reuse:shared_engine_frontend_neutral_batch:petri_marking_successor_predicate_semantic_v1:fnv1a64:"));
        assert!(row.contains("plan_reuse_manifest_digest=fnv1a64:"));
        assert!(row.contains("frontend_payload_identity=mcc_petri.marking_vector"));
        assert!(row.contains(
            "trust_ir_identity=tla_ir.whole_program_kernel_metadata.canonical_identity.v1"
        ));
        assert!(row.contains("semantic_digest=fnv1a64:"));
        assert!(row.contains("cache_digest=fnv1a64:"));
        assert!(row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(row.contains("cas_identity=accepted_fail_closed_fingerprint_domain"));
        assert!(row.contains("cache_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(
            row.contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1")
        );
        assert!(row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(row.contains("storage_layout_fingerprint=fnv1a64:"));
        assert!(
            row.contains("artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1")
        );
        assert!(
            row.contains("target_abi_identity=extern_c.petri_marking_successor_predicate_batch.v1")
        );
        assert!(row.contains(
            "required_evidence=manifest_metadata,layout_checksum,semantic_checksum,validation_receipt"
        ));
        assert!(row.contains("install_authority=validation_only"));
        assert!(row.contains("evidence_fail_closed=true"));
        assert!(row.contains("admission_status=accepted"));
        assert!(row.contains("admission_disposition=profile_only"));
        assert!(row.contains("admission_authority=validation_only"));
        assert!(row.contains("admission_reason=accepted_evidence"));
        assert!(row.contains("admission_fail_closed=true"));
        assert!(row.contains("production_selected=false"));

        let net_contract_row =
            setup.render_shared_native_contract_evidence_row_for_net("MCC", &net);
        let core_planning_row =
            setup.render_core_shared_native_planning_identity_evidence_row_for_net("MCC", &net);
        let petri_planning_row =
            setup.render_shared_planning_fingerprint_identity_evidence_row("MCC", &net);
        assert!(core_planning_row.starts_with("MCC shared_native_planning_identity "));
        assert!(core_planning_row.contains("schema=ty.shared.native_planning_identity.v1"));
        assert!(core_planning_row.contains("source_kind=mcc_petri"));
        assert!(core_planning_row.contains("frontend_kind=mcc_petri"));
        assert_eq!(
            evidence_field(&core_planning_row, "source_fingerprint"),
            planning_identity_evidence_value(evidence_field(
                &net_contract_row,
                "source_fingerprint"
            ))
        );
        assert_eq!(
            evidence_field(&core_planning_row, "plan_reuse_manifest_id"),
            planning_identity_evidence_value(evidence_field(
                &net_contract_row,
                "plan_reuse_manifest_id"
            ))
        );
        assert_eq!(
            evidence_field(&core_planning_row, "plan_reuse_manifest_digest"),
            planning_identity_evidence_value(evidence_field(
                &net_contract_row,
                "plan_reuse_manifest_digest"
            ))
        );
        assert_eq!(
            evidence_field(&core_planning_row, "fingerprint_domain_identity"),
            planning_identity_evidence_value(evidence_field(
                &net_contract_row,
                "fingerprint_domain_identity"
            ))
        );
        assert_eq!(
            evidence_field(&core_planning_row, "cas_identity"),
            planning_identity_evidence_value(evidence_field(&net_contract_row, "cas_identity"))
        );
        assert_eq!(
            evidence_field(&core_planning_row, "cache_identity"),
            evidence_field(&net_contract_row, "cache_identity")
        );
        assert!(core_planning_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(core_planning_row.contains("frontend_family_reusable=true"));
        assert_eq!(
            planning_identity_evidence_value(evidence_field(
                &net_contract_row,
                "native_planning_identity"
            )),
            evidence_field(&core_planning_row, "native_planning_identity")
        );
        assert_eq!(
            evidence_field(&core_planning_row, "native_planning_identity"),
            planning_identity_evidence_value(evidence_field(
                &petri_planning_row,
                "core_native_planning_identity"
            ))
        );
        assert_eq!(
            evidence_field(&petri_planning_row, "planning_identity_digest"),
            evidence_field(&petri_planning_row, "core_native_planning_identity")
        );

        let manifest_row = setup.render_shared_native_contract_manifest_evidence_row("MCC", &net);
        assert!(manifest_row.starts_with("MCC shared_native_contract_manifest "));
        assert!(manifest_row.contains("schema=ty.shared_engine.petri.native_contract_manifest.v1"));
        assert!(manifest_row.contains("source_kind=mcc_petri"));
        assert!(manifest_row.contains("payload_kind=mcc_petri"));
        assert!(manifest_row.contains("storage_kind=petri_marking"));
        assert!(manifest_row.contains("layout_kind=petri_marking"));
        assert!(manifest_row.contains("layout_identity=petri_marking_i64_vector"));
        assert!(manifest_row.contains("symbol=petri_marking_successor_predicate_batch"));
        assert!(manifest_row.contains("manifest_checksum=fnv1a64:"));
        assert!(manifest_row.contains("layout_checksum=fnv1a64:"));
        assert!(manifest_row.contains("semantic_checksum=fnv1a64:"));
        assert!(manifest_row.contains("source_checksum=fnv1a64:"));
        assert!(manifest_row.contains("payload_checksum=fnv1a64:"));
        assert!(manifest_row.contains("fingerprint_algorithm=canonical_bytes_sha256"));
        assert!(manifest_row.contains(
            "fingerprint_helper_symbol=crate::explorer::fingerprint::fingerprint_marking"
        ));
        assert!(manifest_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(manifest_row.contains(
            "fingerprint_domain_acceptance_identity=accepted_fail_closed_fingerprint_domain"
        ));
        assert!(manifest_row
            .contains("fingerprint_admission_surface=shared_fingerprint_state_vector_admission"));
        assert!(manifest_row.contains(
            "fingerprint_admission_semantics=default_consumer,compatible_consumer,blocked"
        ));
        assert!(manifest_row.contains("fingerprint_admission_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(manifest_row
            .contains("fingerprint_admission_default_frontend_families=tla_plus,mcc_petri"));
        assert!(manifest_row.contains("fingerprint_admission_blocked_frontend_families=future_importer:awaiting_registered_importer_frontend"));
        assert!(manifest_row
            .contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(manifest_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(manifest_row
            .contains("artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1"));
        assert!(manifest_row.contains("artifact_identity_kind=contract_template"));
        assert!(manifest_row.contains("artifact_identity_status=contract_template_only"));
        assert!(manifest_row.contains("artifact_digest_status=per_artifact_digest_missing"));
        assert!(manifest_row.contains("validation_only=true"));
        assert!(manifest_row.contains("origin_frontend=mcc_petri"));
        assert!(
            manifest_row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
        );
        assert!(manifest_row.contains("shared_owner=shared_high_performance_engine"));
        assert!(manifest_row.contains("first_beneficiary=mcc_petri_runtime_storage"));
        assert!(manifest_row.contains(
            "second_beneficiary=trust_cg_batch_identity_contract,ay_analytical,witness_replay"
        ));
        assert!(manifest_row.contains("extraction_status=frontend-local-with-tracked-extraction"));
        assert!(manifest_row.contains("adoption_level=level-0"));
        assert!(manifest_row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(manifest_row.contains("default_compatible_frontend_families=none"));
        assert!(manifest_row.contains("downstream_beneficiary_families=none"));
        assert!(manifest_row.contains("remaining_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(manifest_row.contains("frontend_family_blockers=tla_plus:needs_state_vector_native_layout_manifest,quint:needs_source_identity_preserving_native_manifest,mcc_petri:missing_native_install_validation_parity_and_callable_receipts,aiger:needs_register_vector_native_layout_manifest,btor2:needs_bitvector_register_native_layout_manifest,vmt_transition_system:needs_transition_system_native_layout_manifest,ay_analytical:needs_native_helper_validation_receipt,witness_replay:needs_replay_validation_receipt_adapter,future_importer:awaiting_registered_importer_frontend"));
        assert!(manifest_row.contains("blocker_status=tracked-blockers"));
        assert!(manifest_row.contains("generic_prerequisites=prepared_checker_program_descriptor,marking_storage_identity,transition_relation_descriptor,state_predicate_descriptor,native_candidate_descriptor,validation_plan_descriptor"));
        assert!(
            manifest_row.contains("acceptance_test=cargo_test_-p_tla-petri_--lib_explorer::setup")
        );
        assert!(manifest_row.contains("acceptance_evidence=prepared_native_candidate_row,shared_native_contract_manifest_row,shared_native_engine_readiness_row"));
        assert!(manifest_row
            .contains("shard_compatibility_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(
            manifest_row.contains("shard_compatibility_scope=marking_vector_batch_partitionable")
        );
        assert!(manifest_row
            .contains("shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(manifest_row.contains("shard_identity_provider=future_trust_ir_trust_cg_manifest"));
        assert!(manifest_row.contains("shard_identity_key=fnv1a64:"));
        assert!(manifest_row.contains("shard_required_fields=source_kind,payload_kind,storage_kind,layout_identity,symbol,semantic_checksum,layout_checksum,manifest_checksum"));
        assert!(manifest_row.contains("fingerprint_compatibility_status=validation_only_declared"));
        assert!(manifest_row.contains("fingerprint_compatibility=canonical_bytes_sha256"));
        assert!(manifest_row.contains("cache_compatibility_status=validation_only_frontend_local"));
        assert!(manifest_row.contains("cache_fingerprint_compatibility=frontend_local_only"));
        assert!(manifest_row.contains("parity_receipt_required=true"));
        assert!(manifest_row
            .contains("parity_receipt_schema=ty.petri.native_successor.parity_receipt.v1"));
        assert!(manifest_row.contains("parity_receipt_status=missing"));
        assert!(manifest_row.contains("parity_receipt_blocker_code=missing_parity_receipt"));
        assert!(manifest_row.contains(
            "parity_receipt_gate_api=tla_petri::petri_native_successor_parity_receipt_gate"
        ));
        assert!(manifest_row.contains("parity_receipt_required_evidence=exact_successor_parity_trace,native_vs_explicit_state_replay_receipt"));
        assert!(manifest_row.contains("validation_receipt_required=true"));
        assert!(manifest_row.contains("validation_receipt_schema=ty.shared.validation_receipt.v1"));
        assert!(manifest_row.contains("validation_receipt_schema_version=1"));
        assert!(manifest_row.contains("validation_receipt_status=missing"));
        assert!(manifest_row.contains("validation_receipt_blocker_code=missing_validation_receipt"));
        assert!(manifest_row.contains(
            "validation_receipt_gate_api=tla_mc_core::validate_validation_receipt_evidence_row"
        ));
        assert!(manifest_row.contains("validation_receipt_required_evidence=accepted_shared_validation_receipt_for_native_successor_candidate"));
        assert!(manifest_row.contains(
            "production_gate=native_install_validation_parity_and_callable_receipts_required"
        ));
        assert!(manifest_row.contains(
            "production_gate_status=blocked_missing_native_install_validation_parity_and_callable_receipts"
        ));
        assert!(manifest_row.contains(
            "production_gate_required_receipts=native_install_receipt,validation_receipt,parity_receipt,callable_receipt"
        ));
        assert!(manifest_row
            .contains("shared_engine_prerequisite=trust_cg_petri_compile_artifact_handoff"));
        assert!(manifest_row.contains(
            "compile_artifact_handoff_schema=trust-cg.petri.native_successor.compile_artifact_handoff.v1"
        ));
        assert!(manifest_row.contains("compile_artifact_handoff_owner=trust-cg"));
        assert!(manifest_row.contains(
            "compile_artifact_handoff_evidence_source=trust-cg.petri_native_successor_compile_artifact_handoff"
        ));
        assert_compile_artifact_handoff_bridge_blocks_production(&manifest_row);
        assert!(manifest_row.contains(
            "compile_artifact_handoff_blocker_code=missing_trust_cg_petri_compile_artifact_handoff"
        ));
        assert!(manifest_row.contains(
            "native_adoption_blocker=trust_cg_petri_compile_artifact_handoff_required_before_native_adoption"
        ));
        assert!(manifest_row.contains("manifest_metadata_status=present"));
        assert!(manifest_row.contains("layout_checksum_status=present"));
        assert!(manifest_row.contains("semantic_checksum_status=present"));
        assert!(manifest_row.contains("validation_receipt_status=missing"));
        assert!(manifest_row.contains("install_authority=validation_only"));
        assert!(manifest_row.contains("admission_disposition=profile_only"));
        assert!(manifest_row.contains("production_selected=false"));
        assert!(manifest_row.contains("fail_closed=true"));

        let readiness_row =
            setup.render_shared_native_engine_readiness_evidence_row("MCC", &simple_net());
        assert!(readiness_row.starts_with("MCC petri_native_shared_engine_readiness "));
        assert!(readiness_row.contains("schema=ty.shared_engine.petri.native_engine_readiness.v1"));
        assert!(readiness_row.contains("readiness_identity=mcc_petri_shared_native_readiness:"));
        assert!(readiness_row.contains("readiness_mode=validation_only"));
        assert!(readiness_row.contains("prepared_trust_ir_reuse_identity=trust_cg_prepared_trust_ir_reuse:shared_engine_frontend_neutral_batch:petri_marking_successor_predicate_semantic_v1:fnv1a64:"));
        assert!(readiness_row
            .contains("prepared_trust_ir_reuse_identity_status=deferred_until_trust_ir_manifest"));
        assert!(readiness_row.contains("origin_frontend=mcc_petri"));
        assert!(readiness_row.contains("diagnostic_module_family=mcc_petri"));
        assert!(readiness_row.contains("shared_engine_component=batch_native_artifact_identity"));
        assert!(readiness_row.contains("digest_source=petri_native_contract_manifest"));
        assert!(readiness_row.contains("prepared_semantic_digest=fnv1a64:"));
        assert!(readiness_row.contains("artifact_link_digest=fnv1a64:"));
        assert!(readiness_row.contains("artifact_cache_digest=fnv1a64:"));
        assert!(readiness_row.contains(
            "batch_artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1"
        ));
        assert!(readiness_row.contains("batch_artifact_identity_kind=contract_template"));
        assert!(readiness_row.contains("batch_artifact_digest_status=per_artifact_digest_missing"));
        assert!(readiness_row
            .contains("export_set_identity_basis=petri_successor_predicate_symbol_set_v1"));
        assert!(readiness_row.contains("export_set_digest=fnv1a64:"));
        assert!(readiness_row.contains(
            "alias_resolution_identity_basis=petri_marking_successor_predicate_alias_resolution_v1"
        ));
        assert!(readiness_row.contains("alias_resolution_digest=fnv1a64:"));
        assert!(readiness_row.contains(
            "export_surface_identity_basis=petri_marking_successor_predicate_export_surface_v1"
        ));
        assert!(readiness_row.contains("export_surface_digest=fnv1a64:"));
        assert!(readiness_row.contains(
            "native_requirements_identity_basis=petri_marking_successor_predicate_native_requirements_v1"
        ));
        assert!(readiness_row.contains("native_requirements_digest=fnv1a64:"));
        assert!(readiness_row.contains("readiness_owner=shared_high_performance_engine"));
        assert!(readiness_row.contains("primary_beneficiary=mcc_petri_runtime_storage"));
        assert!(readiness_row.contains("first_beneficiary=mcc_petri_runtime_storage"));
        assert!(readiness_row.contains(
            "secondary_beneficiary=trust_cg_batch_identity_contract,ay_analytical,witness_replay"
        ));
        assert!(readiness_row.contains(
            "second_beneficiary=trust_cg_batch_identity_contract,ay_analytical,witness_replay"
        ));
        assert!(readiness_row.contains("readiness_frontend_families=mcc_petri"));
        assert!(readiness_row.contains(
            "future_frontend_family_readiness=deferred_until_core_shared_adoption_schema"
        ));
        assert!(readiness_row.contains("adoption_level=level-0"));
        assert!(readiness_row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(readiness_row.contains("default_compatible_frontend_families=none"));
        assert!(readiness_row.contains("downstream_beneficiary_families=none"));
        assert!(readiness_row.contains("remaining_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(readiness_row.contains("frontend_family_blockers=tla_plus:needs_state_vector_native_layout_manifest,quint:needs_source_identity_preserving_native_manifest,mcc_petri:missing_native_install_validation_parity_and_callable_receipts,aiger:needs_register_vector_native_layout_manifest,btor2:needs_bitvector_register_native_layout_manifest,vmt_transition_system:needs_transition_system_native_layout_manifest,ay_analytical:needs_native_helper_validation_receipt,witness_replay:needs_replay_validation_receipt_adapter,future_importer:awaiting_registered_importer_frontend"));
        assert!(readiness_row.contains("generic_prerequisites=prepared_checker_program_descriptor,marking_storage_identity,transition_relation_descriptor,state_predicate_descriptor,native_candidate_descriptor,validation_plan_descriptor"));
        assert!(
            readiness_row.contains("acceptance_test=cargo_test_-p_tla-petri_--lib_explorer::setup")
        );
        assert!(readiness_row.contains("acceptance_evidence=prepared_native_candidate_row,shared_native_contract_manifest_row,shared_native_engine_readiness_row"));
        assert!(readiness_row.contains("checksum_scope=layout_semantic_source_payload_cache"));
        assert!(readiness_row.contains("frontend_fields_in_checksums=net_name,place_ids,place_names,transition_ids,transition_names,initial_marking,arcs"));
        assert!(readiness_row.contains("prepared_trust_ir_reuse=deferred_until_trust_ir_manifest"));
        assert!(readiness_row
            .contains("prepared_trust_ir_reuse_scope=shared_engine_frontend_neutral_batch"));
        assert!(readiness_row.contains("storage_kind=petri_marking"));
        assert!(readiness_row.contains("layout_kind=petri_marking"));
        assert!(readiness_row.contains("symbol=petri_marking_successor_predicate_batch"));
        assert!(readiness_row.contains("validation_only=true"));
        assert!(readiness_row.contains("readiness_status=validation_only"));
        assert!(readiness_row.contains("validation_receipt_required=true"));
        assert!(readiness_row.contains("validation_receipt_schema=ty.shared.validation_receipt.v1"));
        assert!(readiness_row.contains("validation_receipt_schema_version=1"));
        assert!(readiness_row.contains("validation_receipt_status=missing"));
        assert!(
            readiness_row.contains("validation_receipt_blocker_code=missing_validation_receipt")
        );
        assert!(readiness_row.contains(
            "validation_receipt_gate_api=tla_mc_core::validate_validation_receipt_evidence_row"
        ));
        assert!(readiness_row.contains("validation_receipt_required_evidence=accepted_shared_validation_receipt_for_native_successor_candidate"));
        assert!(readiness_row
            .contains("shard_compatibility_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(
            readiness_row.contains("shard_compatibility_scope=marking_vector_batch_partitionable")
        );
        assert!(readiness_row
            .contains("shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(readiness_row.contains("shard_identity_provider=future_trust_ir_trust_cg_manifest"));
        assert!(readiness_row.contains("shard_identity_key=fnv1a64:"));
        assert!(readiness_row
            .contains("trust_ir_shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(readiness_row
            .contains("trust_cg_shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(readiness_row.contains("fingerprint_compatibility_status=validation_only_declared"));
        assert!(readiness_row.contains("fingerprint_compatibility=canonical_bytes_sha256"));
        assert!(readiness_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(readiness_row.contains(
            "fingerprint_domain_acceptance_identity=accepted_fail_closed_fingerprint_domain"
        ));
        assert!(readiness_row
            .contains("fingerprint_admission_surface=shared_fingerprint_state_vector_admission"));
        assert!(readiness_row.contains(
            "fingerprint_admission_semantics=default_consumer,compatible_consumer,blocked"
        ));
        assert!(readiness_row.contains("fingerprint_admission_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(readiness_row
            .contains("fingerprint_admission_default_frontend_families=tla_plus,mcc_petri"));
        assert!(readiness_row.contains("fingerprint_admission_blocked_frontend_families=future_importer:awaiting_registered_importer_frontend"));
        assert!(readiness_row.contains("cache_compatibility_status=validation_only_frontend_local"));
        assert!(readiness_row.contains("cache_fingerprint_compatibility=frontend_local_only"));
        assert!(readiness_row
            .contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(readiness_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(readiness_row.contains("cache_digest=fnv1a64:"));
        assert!(readiness_row
            .contains("artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1"));
        assert!(readiness_row.contains("artifact_identity_kind=contract_template"));
        assert!(readiness_row.contains("artifact_identity_status=contract_template_only"));
        assert!(readiness_row.contains("artifact_digest_status=per_artifact_digest_missing"));
        assert!(readiness_row.contains("parity_receipt_required=true"));
        assert!(readiness_row
            .contains("parity_receipt_schema=ty.petri.native_successor.parity_receipt.v1"));
        assert!(readiness_row.contains("parity_receipt_status=missing"));
        assert!(readiness_row.contains("parity_receipt_blocker_code=missing_parity_receipt"));
        assert!(readiness_row.contains(
            "parity_receipt_gate_api=tla_petri::petri_native_successor_parity_receipt_gate"
        ));
        assert!(readiness_row.contains("parity_receipt_required_evidence=exact_successor_parity_trace,native_vs_explicit_state_replay_receipt"));
        assert!(readiness_row.contains(
            "production_gate=native_install_validation_parity_and_callable_receipts_required"
        ));
        assert!(readiness_row.contains(
            "production_gate_status=blocked_missing_native_install_validation_parity_and_callable_receipts"
        ));
        assert!(readiness_row.contains(
            "production_gate_required_receipts=native_install_receipt,validation_receipt,parity_receipt,callable_receipt"
        ));
        assert!(readiness_row
            .contains("shared_engine_prerequisite=trust_cg_petri_compile_artifact_handoff"));
        assert!(readiness_row.contains(
            "compile_artifact_handoff_schema=trust-cg.petri.native_successor.compile_artifact_handoff.v1"
        ));
        assert!(readiness_row.contains("compile_artifact_handoff_owner=trust-cg"));
        assert!(readiness_row.contains(
            "compile_artifact_handoff_evidence_source=trust-cg.petri_native_successor_compile_artifact_handoff"
        ));
        assert_compile_artifact_handoff_bridge_blocks_production(&readiness_row);
        assert!(readiness_row.contains(
            "compile_artifact_handoff_blocker_code=missing_trust_cg_petri_compile_artifact_handoff"
        ));
        assert!(readiness_row.contains(
            "native_adoption_blocker=trust_cg_petri_compile_artifact_handoff_required_before_native_adoption"
        ));
        assert!(readiness_row.contains("production_selected=false"));
        assert!(readiness_row.contains("fail_closed=true"));
    }

    fn assert_compile_artifact_handoff_bridge_blocks_production(row: &str) {
        assert!(
            row.contains("compile_artifact_handoff_schema_version=1"),
            "handoff bridge should expose a concrete schema version: {row}"
        );
        assert!(row.contains(
            "compile_artifact_handoff_surface=petri_native_successor_compile_artifact_handoff"
        ));
        assert!(row.contains(
            "compile_artifact_handoff_api=trust-cg::petri_native_successor_compile_artifact_handoff_evidence"
        ));
        assert!(row.contains(
            "compile_artifact_handoff_input_type=PetriNativeSuccessorCompileArtifactHandoffInput"
        ));
        assert!(row.contains(
            "compile_artifact_handoff_evidence_type=PetriNativeSuccessorCompileArtifactHandoffEvidence"
        ));
        assert!(row.contains(
            "compile_artifact_handoff_blocker_type=PetriNativeSuccessorCompileArtifactHandoffBlocker"
        ));
        assert!(row.contains("compile_artifact_handoff_required_fields=compiled_artifact.native_payload_sha256,compiled_artifact.entry_symbol,compiled_artifact.callable_pointer,compiled_artifact.executable_region_sha256,compiled_artifact.lifetime_owner,compiled_artifact.current_generation"));
        assert!(row.contains("compile_artifact_handoff_status_codes=ready,blocked"));
        assert!(row.contains("compile_artifact_handoff_blocker_codes=missing_native_payload_sha256,missing_entry_symbol,missing_callable_pointer,missing_executable_region_sha256,missing_lifetime_owner,missing_current_generation"));
        assert!(row.contains("compile_artifact_handoff_blocker_codes_count=6"));
        assert!(row.contains("compile_artifact_handoff_bridge_status="));
        assert!(row.contains("production_blocked"));
        assert!(row.contains("compile_artifact_handoff_producer_requirements=InstalledArtifact.native_payload_sha256,entry_symbol,callable_pointer,executable_region_sha256,lifetime_owner,current_generation"));
        assert!(row.contains("compile_artifact_handoff_consumer_requirements=status_ready,sha256_identity,entry_symbol_match,callable_pointer_present,executable_region_present,lifetime_owner_present,current_generation_present"));
        assert!(row.contains("compile_artifact_handoff_production_requirement=ready_handoff_plus_native_install_validation_parity_and_callable_receipts"));
        assert!(row.contains("compile_artifact_handoff_status=blocked"));
        assert!(
            row.contains("production_selected=false")
                || row.contains("native_production_selected=false")
                || row.contains(
                    "production_gate_status=blocked_missing_native_install_validation_parity_and_callable_receipts"
                ),
            "handoff bridge should expose a fail-closed production gate: {row}"
        );
        assert!(row.contains("fail_closed=true"));
        assert!(!row.contains("compile_artifact_handoff_status=ready"));
        assert!(!row.contains("production_selected=true"));
        assert!(!row.contains("native_production_selected=true"));
    }
}
