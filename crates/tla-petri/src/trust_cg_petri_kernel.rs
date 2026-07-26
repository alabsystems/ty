// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Scaffolding for the trust-codegen Petri transition/predicate kernel lane.
//!
//! This module is deliberately pure Rust for now. It defines the flat `i64`
//! marking layout and parity helpers that later native trust-codegen kernels must match.
//! The default MCC execution path is unchanged; opt-in parity checks can be
//! enabled by integration code without trusting kernel output.
//!
//! # Staged scaffolding — what is live vs deferred
//!
//! This is intentional native-Petri-kernel scaffolding. Do not delete items that
//! look unused before checking the layers below; most "dead" symbols are
//! staged/future API wired together but not yet reached from a fully-enabled
//! production route.
//!
//! Live production path (NOT dead): the capability-report surface
//! [`petri_native_successor_capability_report`] is reached on every MCC run via
//! `mcc_backend_evidence::petri_native_capability_report_deadline_aware` →
//! `build_petri_mcc_capability_report` → `mcc_backend_capability_report`, which
//! `model::render::run_examination_for_model` calls before each examination. It
//! emits *diagnostic* backend-capability evidence only and never produces or
//! changes a verdict (native successor *execution* adoption is decided separately
//! in `explorer::observer`).
//!
//! Deferred lanes: the executable native successor/predicate kernels and the
//! parity-promotion gate are not yet wired. The richer trust-ir handoff /
//! verification-bundle helpers are additionally gated behind the
//! `trust-cg-petri-native` cargo feature (off by default). Several string/schema
//! contract constants (e.g. the `ay_trust_mc_native_bundle` `pub const` API-name
//! mirrors, downstream-contract `_SCHEMA`/`_SCHEMA_VERSION` twins) are referenced
//! only from the in-module `#[cfg(test)]` suite that pins the staged contract
//! shape; they are deliberately retained ahead of their production wiring.
//!
//! Because of this mix of staged-but-interconnected items, dead-code is
//! suppressed crate-wide via the `#![allow(dead_code)]` in `lib.rs` rather than
//! per item: enabling the lint here surfaces ~37 transitively-dead symbols (the
//! unwired entry points and everything they reference), so per-item annotation
//! would be pure churn with no signal. Tighten to per-item `#[allow]` only once
//! the executable lanes are wired and the dead set shrinks to a handful.

use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::portfolio::ExactOrUnknownStatus;
use crate::resolved_predicate::{eval_predicate, ResolvedIntExpr, ResolvedPredicate};
use tla_jit_abi::{
    KernelArtifactAdoptionEvidence, KernelArtifactChecksum, KernelArtifactChecksums,
    KernelStateDomain, KernelSymbolSignature, SuccessorKernelDescriptor, SuccessorKernelShape,
    TY_KERNEL_ARTIFACT_CONSUMER, TY_PREDICATE_KERNEL_EVIDENCE_METADATA,
    TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA,
};
use tla_mc_core::{
    BackendCapability, BackendDomain, BackendKind, CapabilityReport, CapabilityRole, ProblemKind,
    SolverFacet, UnsupportedReason, VALIDATION_RECEIPT_SCHEMA, VALIDATION_RECEIPT_SCHEMA_VERSION,
};

pub(crate) const ENABLE_TRANSITION_PARITY_ENV: &str = "TY_MCC_TRUST_CG_PETRI_PARITY";
pub(crate) const ENABLE_NATIVE_CANDIDATE_ENV: &str = "TY_MCC_TRUST_CG_PETRI_NATIVE";
pub(crate) const ENABLE_NATIVE_CANDIDATE_STRICT_ENV: &str = "TY_MCC_TRUST_CG_PETRI_NATIVE_STRICT";
pub(crate) const PETRI_KERNEL_ABI_VERSION: u32 = 1;
/// Maximum `num_places × num_transitions` the native successor capability probe
/// will attempt to codegen. The plan cache + native IR generation is
/// Θ(places × transitions); beyond this budget the (non-cancellable) probe
/// worker would allocate tens of GB and OOM the process. Declining native
/// admission above it is verdict-preserving (the interpreter runs instead).
/// 16e6 ≈ a 4000×4000 net — well beyond any net whose state space is
/// explicitly enumerable, so native admission loses nothing in practice.
pub(crate) const NATIVE_SUCCESSOR_MAX_CELLS: usize = 16_000_000;
pub(crate) const PETRI_NATIVE_SUCCESSOR_DESCRIPTOR_NAME: &str = "petri-all-transitions";
pub(crate) const PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL: &str = "ty_petri_all_transition_successors";
pub(crate) const PETRI_NATIVE_PREDICATE_ENTRY_SYMBOL: &str = "ty_petri_state_predicate";
const PETRI_NATIVE_SUCCESSOR_POLICY: &str = "trust-cg-petri-native is not parity promoted";
const PETRI_NATIVE_PREDICATE_DETAIL: &str =
    "trust-cg Petri native predicate kernel is deferred; Rust flat predicate evaluation remains parity-only";
const TRUST_CG_PETRI_NATIVE_FEATURE: &str = "trust-cg-petri-native";
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_REQUIRED_REV: &str =
    "222785e293636ac6c63b20525151aef2ccd586c1";
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV: &str =
    "3fafb62434db0a5b2bd4027a988a7fed74bd8679";
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_DEPENDENCY_BLOCKER: &str = "tla-petri was built without the trust-cg-petri-native feature, so the trust-ir transport identity type is not linked";
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_ABSENT_BLOCKER: &str =
    "NativeVerificationBundle was not supplied to the Petri native capability report";
const TRUST_IR_NATIVE_VERIFICATION_EXPECTED_FIELDS: &str =
    "transport,source,module,bundle,target_abi_digest";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA: &str = "trust_ir.native.transport_identity.v2";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA_VERSION: u32 = 2;
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SCHEMA: &str =
    "ty.petri.native_transport_identity.producer_contract.v1";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SCHEMA_VERSION: u32 = 1;
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_API: &str =
    "trust_cg_petri_native::petri_native_successor_verification_bundle";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SOURCE: &str =
    "PetriTrustIrTransportIdentityProducerContract";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_REQUIRED_OUTPUT: &str =
    "trust_ir::NativeVerificationBundle";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_BUNDLE_SOURCE_NONE: &str = "none";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_DIGEST_NONE: &str = "none";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_NONE: &str = "none";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_INPUT_NONE: &str = "none";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_STATUS_AVAILABLE: &str = "available";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_STATUS_BLOCKED: &str = "blocked";
const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_BLOCKER_NONE: &str = "none";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_PROJECT_CODE: &str = "trust-ir";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_COMPONENT: &str =
    "native_verification_bundle_handoff_manifest";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_COMPLETENESS_COMPONENT: &str =
    "native_verification_bundle_handoff_completeness";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA: &str =
    "trust_ir.native.petri_successor.native_verification_bundle_handoff.v1";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION: u32 = 1;
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_IDENTITY_COMPONENT: &str =
    "native_verification_bundle_handoff_manifest_identity";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_CONTRACT_HEALTH_COMPONENT: &str =
    "native_verification_bundle_handoff_contract_health";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_COMPONENT: &str =
    "native_verification_bundle_handoff_diagnostic_fixture_manifest";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_ROUND_TRIP_COMPONENT: &str =
    "native_verification_bundle_handoff_diagnostic_fixture_manifest_round_trip";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_ROUND_TRIP_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.diagnostic_fixture_manifest.round_trip_report.v1";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_ROUND_TRIP_SCHEMA_VERSION:
    u32 = 1;
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_COMPONENT: &str =
    "native_verification_bundle_handoff_replay_contract_surface";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_ROUND_TRIP_COMPONENT:
    &str = "native_verification_bundle_handoff_replay_contract_surface_round_trip";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_REPORT_IDENTITY_COMPONENT: &str =
    "native_verification_bundle_handoff_replay_contract_report_identity";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_JSON_MANIFEST_BINDING_COMPONENT:
    &str = "native_verification_bundle_handoff_replay_contract_json_manifest_binding";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT: &str =
    "native_semantic_bridge_proof_identity";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_REPLAY_HEALTH_COMPONENT: &str =
    "native_semantic_bridge_proof_identity_replay_component_health";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_COMPONENT: &str =
    "petri_successor_trust_mc_chc_proof_evidence_identity";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_REPLAY_HEALTH_COMPONENT: &str =
    "petri_successor_trust_mc_chc_proof_evidence_identity_replay_component_health";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_TY_MCC_SHARED_PRIMITIVE_MANIFEST_COMPONENT: &str =
    "ty_mcc_shared_primitive_manifest";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_IR_HARDWARE_VECTOR_CONTRACT_SET_COMPONENT: &str = "hardware_vector_contract_set";
#[cfg(feature = "trust-cg-petri-native")]
const PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_API: &str =
    "trust-cg::petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle";
#[cfg(feature = "trust-cg-petri-native")]
const PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA: &str =
    tla_trust_cg::PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA;
#[cfg(feature = "trust-cg-petri-native")]
const AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_API: &str =
    "ay_trust_mc_native_bundle::solve_trust_mc_petri_successor_native_verification_bundle";
#[cfg(feature = "trust-cg-petri-native")]
const AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CONSUMER_ACCEPTANCE_API: &str =
    "ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleReport::accept_for_consumer";
#[cfg(feature = "trust-cg-petri-native")]
const AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_REQUIRED_AY_REV: &str = "7fe72a4d";
#[cfg(feature = "trust-cg-petri-native")]
const AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV: &str =
    "035e84f25ffe983f4c1a0d8f2cb1d5f945d3bdee";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_CG_PETRI_NATIVE_SEMANTIC_BRIDGE_REQUIRED_TRUST_CG_REV: &str = "dd0d5338";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_CG_PETRI_NATIVE_SEMANTIC_BRIDGE_CURRENT_TRUST_CG_REV: &str =
    "98e3ffb6ae59b803a93a3f09f72dd497810ac5b4";
const TRUST_CG_PETRI_NATIVE_ADMISSION_KIND: &str = "petri_successor";
const TRUST_CG_PETRI_NATIVE_ADMISSION_SURFACE: &str = "native_successor";
const TRUST_CG_PETRI_NATIVE_ADMISSION_API: &str =
    "trust-cg::petri_native_successor_admission_from_trust_ir_bundle";
const TRUST_CG_PETRI_NATIVE_ADMISSION_BUNDLE_API: &str =
    "NativeVerificationBundle::native_evidence_consumption_report";
const TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON: &str =
    "missing_trust_ir_transport_identity";
const TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_SCHEMA: &str =
    "trust-cg.petri.native_successor.execution_plan.v1";
const TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_API: &str =
    "trust-cg::petri_native_successor_execution_plan_from_trust_ir_bundle";
const TRUST_CG_PETRI_NATIVE_EXECUTION_EXPECTED_API: &str =
    "PetriNativeSuccessorExecutionExpected::canary_callable";
const TRUST_CG_PETRI_NATIVE_TRAMPOLINE_CONTRACT_API: &str =
    "trust-cg::petri_native_successor_trampoline_contract";
const TRUST_CG_PETRI_NATIVE_INSTALL_PACKET_API: &str =
    "trust-cg::petri_native_successor_install_packet_from_trust_ir_bundle";
const TRUST_CG_PETRI_NATIVE_EXECUTION_STATE_ALIGNMENT_BYTES: u32 = 8;
const TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE: &str = "available";
const TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON: &str =
    "missing_callable_pointer_handoff";
const TRUST_CG_PETRI_NATIVE_MISSING_CONCRETE_CALLABLE_POINTER_REASON: &str =
    "missing_concrete_callable_pointer";
const TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_API: &str =
    "trust-cg::petri_native_successor_call_packet_from_trust_ir_bundle";
const TRUST_CG_PETRI_NATIVE_CALLABLE_POINTER_TYPE: &str = "PetriNativeSuccessorCallablePointer";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_TYPE: &str = "PetriNativeSuccessorCallPacket";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_SCHEMA: &str =
    "trust-cg.petri.native_successor.call_packet.v1";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_REQUIRED_TRUST_CG_REV: &str = "2d31fd8b";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV: &str =
    "98e3ffb6ae59b803a93a3f09f72dd497810ac5b4";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_DEPENDENCY: &str =
    "trust-cg::petri_native_successor_downstream_contract_descriptor.call_packet";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_UPSTREAM_ASK: &str =
    "expose_petri_native_successor_call_packet_descriptor";
const TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_NO_UPSTREAM_ASK: &str = "none";
const TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_BLOCKER: &str = "missing_concrete_callable_packet";
const TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_UPSTREAM_ASK: &str =
    "provide_runtime_callable_pointer_and_accepted_install_packet";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_API: &str =
    "trust-cg::petri_native_successor_runtime_readiness_packet";
const TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_API: &str =
    "trust-cg::petri_native_successor_execution_authority_decision";
const TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_SUMMARY_API: &str =
    "PetriNativeSuccessorExecutionAuthorityDecision::compact_authority_summary";
const TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_MANIFEST_VALIDATION_API: &str =
    "PetriNativeSuccessorExecutionAuthorityDecision::manifest_validation_report";
const TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_SUMMARY_VALIDATION_API: &str =
    "trust-cg::validate_petri_native_successor_execution_authority_summary_rows";
const TRUST_CG_PETRI_NATIVE_PRODUCTION_SELECTION_API: &str =
    "trust-cg::petri_native_successor_production_selection_decision";
const PETRI_NATIVE_ROUTE_SELECTION_SCHEMA: &str = "ty.petri.native_successor.route_selection.v1";
const PETRI_NATIVE_ROUTE_SELECTION_SCHEMA_VERSION: u32 = 1;
const PETRI_NATIVE_ROUTE_SELECTION_API: &str = "tla_petri::petri_native_successor_route_selection";
const PETRI_NATIVE_ROUTE_SELECTION_SAFE_CRITERIA: &str =
    "producer_admission,producer_execution_authority,producer_production_selection,parity_enabled,parity_receipt_available,validation_receipt_available,callable_receipt_available,native_runtime_callable_impl";
const PETRI_NATIVE_ROUTE_SELECTION_BLOCKER_ISSUES: &str =
    "alabsystems/ty#4455,alabsystems/ty#4458,alabsystems/trust-cg#881";
const PETRI_NATIVE_ROUTE_SELECTION_LANE_NATIVE: &str = "native_successor";
const PETRI_NATIVE_ROUTE_SELECTION_LANE_FALLBACK: &str = "explicit_state";
const PETRI_NATIVE_ROUTE_SELECTION_STATUS_SELECTED: &str = "selected";
const PETRI_NATIVE_ROUTE_SELECTION_STATUS_FAIL_CLOSED: &str = "fail_closed";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_NONE: &str = "none";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_MISSING_TRANSPORT: &str =
    "missing_trust_ir_transport_identity";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_PRODUCER_ADMISSION: &str =
    "producer_admission_not_installable";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_EXECUTION_AUTHORITY: &str =
    "producer_execution_authority_not_authorized";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_PRODUCTION_SELECTION: &str =
    "producer_production_selection_not_selected";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY: &str = "parity_evidence_required";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT: &str = "missing_parity_receipt";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT: &str = "missing_validation_receipt";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT: &str = "missing_callable_receipt";
const PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL: &str =
    "native_runtime_callable_impl_missing";
const PETRI_NATIVE_ROUTE_SELECTION_TODO: &str =
    "wire_executable_native_successor_runtime_and_parity_promotion_gate";
const PETRI_NATIVE_ROUTE_SELECTION_TODO_SELECTED: &str = "none";
const PETRI_NATIVE_ROUTE_SELECTION_BLOCKER_ISSUES_SELECTED: &str = "none";
const PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE: bool = true;
const PETRI_NATIVE_PARITY_RECEIPT_SCHEMA: &str = "ty.petri.native_successor.parity_receipt.v1";
const PETRI_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION: u32 = 1;
const PETRI_NATIVE_PARITY_RECEIPT_GATE_API: &str =
    "tla_petri::petri_native_successor_parity_receipt_gate";
const PETRI_NATIVE_PARITY_RECEIPT_STATUS_ACCEPTED: &str = "accepted";
const PETRI_NATIVE_PARITY_RECEIPT_STATUS_MISSING: &str = "missing";
const PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE: &str =
    "exact_successor_parity_trace,native_vs_explicit_state_replay_receipt";
const PETRI_NATIVE_VALIDATION_RECEIPT_GATE_API: &str =
    "tla_mc_core::validate_validation_receipt_evidence_row";
const PETRI_NATIVE_VALIDATION_RECEIPT_STATUS_ACCEPTED: &str = "accepted";
const PETRI_NATIVE_VALIDATION_RECEIPT_STATUS_MISSING: &str = "missing";
const PETRI_NATIVE_VALIDATION_RECEIPT_REQUIRED_EVIDENCE: &str =
    "accepted_shared_validation_receipt_for_native_successor_candidate";
const PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA: &str = "ty.petri.native_successor.callable_receipt.v1";
const PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA_VERSION: u32 = 1;
const PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API: &str =
    "tla_petri::petri_native_successor_callable_receipt_gate";
const PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_ACCEPTED: &str = "accepted";
const PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_MISSING: &str = "missing";
const PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE: &str =
    "compile_artifact_handoff,runtime_readiness_packet,native_install_gate_packet,call_packet,callable_pointer";
const PETRI_NATIVE_ROUTE_SHARED_ENGINE_OWNER: &str = "shared_high_performance_engine";
const PETRI_NATIVE_ROUTE_SHARED_ENGINE_COMPONENT: &str =
    "tla_mc_core.prepared_checker_program,tla_ir.whole_program_kernel,trust_cg.batch_native_artifact_identity,tla_mc_core.validation_receipt";
const PETRI_NATIVE_ROUTE_ORIGIN_FRONTEND: &str = "mcc_petri";
const PETRI_NATIVE_ROUTE_FIRST_BENEFICIARY: &str = "mcc_petri_runtime_storage";
const PETRI_NATIVE_ROUTE_SECOND_BENEFICIARY: &str =
    "trust_cg_batch_identity_contract,ay_analytical,witness_replay";
const PETRI_NATIVE_ROUTE_EXTRACTION_STATUS: &str = "frontend-local-with-tracked-extraction";
const PETRI_NATIVE_ROUTE_ADOPTION_LEVEL: &str = "level-0";
const PETRI_NATIVE_ROUTE_GENERIC_PREREQUISITES: &str = "prepared_checker_program_descriptor,marking_storage_identity,transition_relation_descriptor,state_predicate_descriptor,native_candidate_descriptor,validation_plan_descriptor,accepted_validation_receipt,accepted_parity_receipt,accepted_callable_receipt";
const PETRI_NATIVE_ROUTE_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer";
const PETRI_NATIVE_ROUTE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES: &str = "none";
const PETRI_NATIVE_ROUTE_DOWNSTREAM_BENEFICIARY_FAMILIES: &str = "none";
const PETRI_NATIVE_ROUTE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES: &str =
    PETRI_NATIVE_ROUTE_COMPATIBLE_FRONTEND_FAMILIES;
const PETRI_NATIVE_ROUTE_FRONTEND_FAMILY_BLOCKERS: &str =
    "tla_plus:needs_state_vector_native_layout_manifest,quint:needs_source_identity_preserving_native_manifest,mcc_petri:missing_native_install_validation_parity_and_callable_receipts,aiger:needs_register_vector_native_layout_manifest,btor2:needs_bitvector_register_native_layout_manifest,vmt_transition_system:needs_transition_system_native_layout_manifest,ay_analytical:needs_native_helper_validation_receipt,witness_replay:needs_replay_validation_receipt_adapter,future_importer:awaiting_registered_importer_frontend";
const PETRI_NATIVE_ROUTE_FRONTEND_FAMILY_BLOCKERS_SELECTED: &str =
    "tla_plus:needs_state_vector_native_layout_manifest,quint:needs_source_identity_preserving_native_manifest,mcc_petri:none,aiger:needs_register_vector_native_layout_manifest,btor2:needs_bitvector_register_native_layout_manifest,vmt_transition_system:needs_transition_system_native_layout_manifest,ay_analytical:needs_native_helper_validation_receipt,witness_replay:needs_replay_validation_receipt_adapter,future_importer:awaiting_registered_importer_frontend";
const PETRI_NATIVE_ROUTE_BLOCKER_STATUS: &str = "tracked-blockers";
const PETRI_NATIVE_ROUTE_BLOCKER_STATUS_SELECTED: &str = "mcc_petri-cleared";
const PETRI_NATIVE_ROUTE_PRODUCTION_GATE: &str =
    "native_install_validation_parity_and_callable_receipts_required";
const PETRI_NATIVE_ROUTE_PRODUCTION_GATE_STATUS: &str =
    "blocked_missing_native_install_validation_parity_and_callable_receipts";
const PETRI_NATIVE_ROUTE_PRODUCTION_GATE_STATUS_SELECTED: &str = "selected";
const PETRI_NATIVE_ROUTE_PRODUCTION_GATE_REQUIRED_RECEIPTS: &str =
    "native_install_receipt,validation_receipt,parity_receipt,callable_receipt";
const PETRI_NATIVE_SHARED_READINESS_ADMISSION_SCHEMA: &str =
    "ty.petri.native_successor.shared_readiness_admission.v1";
const PETRI_NATIVE_SHARED_READINESS_ADMISSION_SCHEMA_VERSION: u32 = 1;
const PETRI_NATIVE_SHARED_READINESS_ADMISSION_API: &str =
    "tla_petri::petri_native_successor_shared_readiness_admission";
const PETRI_NATIVE_SHARED_READINESS_ADMISSION_SOURCE: &str =
    "PetriNativeSharedReadinessAdmissionContract";
const PETRI_NATIVE_SHARED_SOLVER_FAMILIES: &str =
    "explicit_state,native_successor,analytical_ay,witness_replay,hardware_transition_system,future_importer";
const PETRI_NATIVE_SHARED_PAYLOAD_IDENTITY_SOURCE: &str = "pnml_hlpnml_import_adapter";
const PETRI_NATIVE_SHARED_PAYLOAD_IDENTITY_REQUIRED_FIELDS: &str =
    "source_payload_digest,normalized_payload_digest,examination_identity,replay_identity";
const PETRI_NATIVE_SHARED_PAYLOAD_IDENTITY_STATUS: &str = "required_before_trusted_native";
const PETRI_NATIVE_SHARED_LAYOUT_IDENTITY: &str = "petri_marking_i64_vector";
const PETRI_NATIVE_SHARED_LAYOUT_FINGERPRINT_ALGORITHM: &str = "fnv1a64";
const PETRI_NATIVE_SHARED_FINGERPRINT_DOMAIN_IDENTITY: &str =
    "fingerprint_domain_key:canonical_bytes_sha256";
const PETRI_NATIVE_SHARED_FINGERPRINT_ADMISSION_CONTRACT: &str = "prepared_fingerprint_admission";
const PETRI_NATIVE_SHARED_FINGERPRINT_ADMISSION_STATUS: &str = "blocked_for_trusted_native";
const PETRI_NATIVE_SHARED_TRUSTED_PRODUCTION_BLOCKERS: &str =
    "canonical_payload_identity,layout_fingerprint_admission,accepted_validation_receipt,accepted_parity_receipt,accepted_callable_receipt,native_runtime_callable_impl,history_suite_parity,end_to_end_speedup_evidence";
const PETRI_NATIVE_SHARED_EXACT_OR_UNKNOWN_GUARD: &str =
    "native_output_unknown_until_explicit_replay_validation";
const PETRI_NATIVE_SHARED_PERFORMANCE_CLAIM_STATUS: &str = "not_claimed";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_INSTALLED_ARTIFACT_API: &str =
    "InstalledArtifact::petri_native_successor_runtime_readiness_packet";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_INSTALLED_ARTIFACT_REQUIRED_TRUST_CG_REV: &str =
    "690f04d7";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_TYPE: &str =
    "PetriNativeSuccessorRuntimeReadinessPacket";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_SCHEMA: &str =
    "trust-cg.petri.native_successor.runtime_readiness_packet.v1";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_REQUIRED_TRUST_CG_REV: &str = "502f8928";
const TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_CURRENT_TRUST_CG_REV: &str =
    "98e3ffb6ae59b803a93a3f09f72dd497810ac5b4";
const TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_API: &str =
    "trust-cg::petri_native_successor_mock_executable_call_dry_run";
const TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_SCHEMA: &str =
    "trust-cg.petri.native_successor.mock_executable_call.v1";
const TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_ROLE: &str = "test_diagnostic_only";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_API: &str =
    "trust-cg::petri_native_successor_compile_artifact_handoff_evidence";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_INPUT_TYPE: &str =
    "PetriNativeSuccessorCompileArtifactHandoffInput";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_TYPE: &str =
    "PetriNativeSuccessorCompileArtifactHandoffEvidence";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_BLOCKER_TYPE: &str =
    "PetriNativeSuccessorCompileArtifactHandoffBlocker";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_REQUIRED_TRUST_CG_REV: &str = "df133c3f";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_CURRENT_TRUST_CG_REV: &str =
    "98e3ffb6ae59b803a93a3f09f72dd497810ac5b4";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_API: &str =
    "InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_TYPE: &str = "InstalledArtifact";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_REQUIRED_TRUST_CG_REV: &str =
    "00597478";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_NATIVE_LIBRARY_BRIDGE_API: &str =
    "NativeLibrary::petri_native_successor_installed_artifact";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_INSTALLED_ARTIFACT_FIELD: &str =
    "petri_native_successor_capability_report.installed_artifact";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_INSTALLED_ARTIFACT: &str =
    "missing_ty_installed_artifact_wiring";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_CG_COMPILE_ARTIFACT_CACHE_TELEMETRY_PROBE_KEY_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_NOT_ATTEMPTED: &str =
    "not_attempted";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_AVAILABLE: &str = "available";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED: &str = "blocked";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_WIRING_STATUS_AVAILABLE: &str = "available";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_WIRING_STATUS_MISSING_INSTALLED_ARTIFACT: &str =
    "missing_installed_artifact";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE: &str = "none";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_UNAVAILABLE: &str = "unavailable";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_CALLABLE_CONTRACT: &str = "callable_contract";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_INSTALLED_ARTIFACT: &str = "installed_artifact";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_SEMANTIC_SUCCESSOR_BRIDGE: &str =
    "semantic_successor_bridge";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_COMPILE_ARTIFACT_HANDOFF: &str =
    "compile_artifact_handoff";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_RUNTIME_READINESS: &str = "runtime_readiness";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_EXECUTION_AUTHORITY: &str =
    "execution_authority";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PRODUCTION_SELECTION: &str =
    "production_selection";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PARITY_GATE: &str = "parity_gate";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PARITY_RECEIPT: &str = "parity_receipt_gate";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_VALIDATION_RECEIPT: &str =
    "validation_receipt_gate";
const TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_CALLABLE_RECEIPT: &str = "callable_receipt_gate";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_ENTRY_SYMBOL_SOURCE_PETRI: &str =
    "petri_successor_entry_symbol";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_ENTRY_SYMBOL_SOURCE_CONTRACT: &str =
    "callable_contract.entry_function";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_NATIVE_PAYLOAD_SOURCE_CONTRACT: &str =
    "callable_contract.native_payload_sha256";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_NATIVE_PAYLOAD: &str =
    "NativeLibrary::replay_report_metadata.native_payload_sha256";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_ENTRY_SYMBOL: &str =
    "PetriNativeSuccessorCallableContract::entry_function";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_CALLABLE_POINTER: &str =
    "NativeLibrary::diagnose_published_symbol_ptr.pointer";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_EXECUTABLE_REGION: &str =
    "JitSymbolPublicationProof::buffer_base..buffer_end";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_LIFETIME_OWNER: &str = "NativeLibrary";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_CURRENT_GENERATION: &str =
    "NativeInstallGateInput::current_generation";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_NATIVE_PAYLOAD: &str =
    "ExecutableBuffer::replay_report_metadata.properties.native_payload_sha256";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_ENTRY_SYMBOL: &str =
    "JitReplayReportMetadata::entry_symbol";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_CALLABLE_POINTER: &str =
    "JitSymbolPublicationProof::pointer";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_EXECUTABLE_REGION: &str =
    "JitSymbolPublicationProof::executable_region_sha256";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_LIFETIME_OWNER: &str =
    "PetriNativeSuccessorCallableLifetimeProof::lifetime_owner";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_CURRENT_GENERATION: &str =
    "PetriNativeSuccessorCallableLifetimeProof::current_generation";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_NO_COMPILED_LIBRARY: &str =
    "petri_native_capability_report_has_no_compiled_NativeLibrary";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_ENTRY_SYMBOL: &str =
    "compiled_artifact_has_no_entry_symbol";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_CALLABLE_POINTER: &str =
    "compiled_artifact_has_no_callable_pointer";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_EXECUTABLE_REGION: &str =
    "compiled_artifact_has_no_executable_region_sha256";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_LIFETIME_OWNER: &str =
    "compiled_artifact_has_no_lifetime_owner";
const TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_CURRENT_GENERATION: &str =
    "compiled_artifact_has_no_current_generation";
const TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API: &str =
    "trust-cg::petri_native_successor_downstream_contract_descriptor";
const TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_SCHEMA: &str =
    "trust-cg.petri.native_successor.downstream_contract.v1";
const TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_REQUIRED_TRUST_CG_REV: &str = "50cf1169";
const TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV: &str =
    "98e3ffb6ae59b803a93a3f09f72dd497810ac5b4";
const TRUST_CG_PETRI_NATIVE_TRUST_IR_BUNDLE_IDENTITY_API: &str =
    "trust-cg::petri_native_successor_trust_ir_bundle_identity_descriptor";
const TRUST_CG_PETRI_NATIVE_TRUST_IR_BUNDLE_IDENTITY_REQUIRED_TRUST_CG_REV: &str = "497f4540";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrustCgPetriNativeReadinessStatus {
    Available,
    Unavailable,
    Missing,
}

impl TrustCgPetriNativeReadinessStatus {
    fn code(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Unavailable => "unavailable",
            Self::Missing => "missing",
        }
    }
}

#[cfg(feature = "trust-cg-petri-native")]
type TrustCgPetriCallPacketBuilder = fn(
    &trust_ir::NativeVerificationBundle,
    tla_trust_cg::PetriNativeSuccessorExecutionExpected<'_>,
    &tla_trust_cg::PetriNativeSuccessorTrampolineContract,
    tla_trust_cg::PetriNativeSuccessorCallablePointer,
) -> Result<
    tla_trust_cg::PetriNativeSuccessorCallPacket,
    tla_trust_cg::NativeInstallGateRejectionCode,
>;

#[cfg(feature = "trust-cg-petri-native")]
type TrustCgPetriRuntimeReadinessPacketBuilder =
    fn(
        Option<&tla_trust_cg::PetriNativeSuccessorCallPacket>,
        Option<&tla_trust_cg::NativeInstallGatePacket>,
        Option<&tla_trust_cg::PetriNativeSuccessorTrampolineContract>,
        Option<&tla_trust_cg::PetriNativeSuccessorCallableLifetimeProof>,
        Option<&tla_trust_cg::PetriNativeSuccessorRuntimeAbiProof>,
        u64,
    ) -> tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket;

#[cfg(feature = "trust-cg-petri-native")]
type TrustCgPetriCompileArtifactHandoffEvidenceBuilder =
    fn(
        tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffInput<'_>,
    ) -> tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence;

#[cfg(feature = "trust-cg-petri-native")]
type TrustCgPetriInstalledArtifactHandoffEvidenceBuilder =
    fn(
        &tla_trust_cg::InstalledArtifact,
        Option<&str>,
    ) -> tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence;

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone, Copy)]
struct TrustCgPetriCallPacketSurface {
    api: &'static str,
    schema: &'static str,
    schema_version: u32,
    call_packet_type: &'static str,
    callable_pointer_type: &'static str,
    descriptor_available: bool,
    descriptor_source: &'static str,
    descriptor_status_code: &'static str,
    descriptor_authoritative: bool,
    descriptor_dependency: &'static str,
    descriptor_upstream_ask: &'static str,
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_call_packet_surface() -> TrustCgPetriCallPacketSurface {
    let _: TrustCgPetriCallPacketBuilder =
        tla_trust_cg::petri_native_successor_call_packet_from_trust_ir_bundle;
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorCallPacket>();
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorCallablePointer>();
    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let descriptor = downstream_contract.call_packet;
    let descriptor_health =
        tla_trust_cg::validate_petri_native_successor_call_packet_contract_descriptor_rows(
            &descriptor.manifest_rows(),
        );
    let descriptor_available = descriptor_health.status
        == tla_trust_cg::PetriNativeSuccessorCallPacketContractHealthStatus::Healthy;

    TrustCgPetriCallPacketSurface {
        api: TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_API,
        schema: tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA,
        schema_version: tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA_VERSION,
        call_packet_type: TRUST_CG_PETRI_NATIVE_CALL_PACKET_TYPE,
        callable_pointer_type: TRUST_CG_PETRI_NATIVE_CALLABLE_POINTER_TYPE,
        descriptor_available,
        descriptor_source: TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_DEPENDENCY,
        descriptor_status_code: descriptor.status_code,
        descriptor_authoritative: descriptor.is_authoritative(),
        descriptor_dependency: TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_DEPENDENCY,
        descriptor_upstream_ask: if descriptor.upstream_pending {
            TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_UPSTREAM_ASK
        } else {
            TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_NO_UPSTREAM_ASK
        },
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone, Copy)]
struct TrustCgPetriCompileArtifactHandoffSurface {
    api: &'static str,
    installed_artifact_api: &'static str,
    installed_artifact_type: &'static str,
    installed_artifact_required_trust_cg_rev: &'static str,
    schema: &'static str,
    schema_version: u32,
    input_type: &'static str,
    evidence_type: &'static str,
    blocker_type: &'static str,
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_compile_artifact_handoff_surface() -> TrustCgPetriCompileArtifactHandoffSurface {
    let _: TrustCgPetriCompileArtifactHandoffEvidenceBuilder =
        tla_trust_cg::petri_native_successor_compile_artifact_handoff_evidence;
    let _: TrustCgPetriInstalledArtifactHandoffEvidenceBuilder =
        tla_trust_cg::InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence;
    let _ = std::mem::size_of::<tla_trust_cg::InstalledArtifact>();
    let _ =
        std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffInput<'_>>();
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence>();
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker>();
    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let compile_artifact_handoff_surface = downstream_contract.compile_artifact_handoff;
    debug_assert_eq!(
        compile_artifact_handoff_surface.schema,
        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA
    );

    TrustCgPetriCompileArtifactHandoffSurface {
        api: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_API,
        installed_artifact_api: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_API,
        installed_artifact_type: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_TYPE,
        installed_artifact_required_trust_cg_rev:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_REQUIRED_TRUST_CG_REV,
        schema: compile_artifact_handoff_surface.schema,
        schema_version: compile_artifact_handoff_surface.schema_version,
        input_type: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_INPUT_TYPE,
        evidence_type: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_EVIDENCE_TYPE,
        blocker_type: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_BLOCKER_TYPE,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone)]
struct TrustCgPetriCompileArtifactHandoffAttempt {
    evidence: tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
    installed_artifact_available: bool,
    real_artifact_source: &'static str,
    entry_symbol_source: &'static str,
    native_payload_source: &'static str,
    ty_wiring_status: &'static str,
    ty_wiring_blocker: &'static str,
    ty_required_field: &'static str,
    missing_ty_artifact_field: &'static str,
    missing_trust_cg_artifact_field: &'static str,
    missing_artifact_blocker: &'static str,
    next_production_api: &'static str,
    next_production_input: &'static str,
    next_production_reason_code: &'static str,
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Default)]
struct TrustCgPetriRuntimeReadinessInputs {
    trampoline_contract: Option<tla_trust_cg::PetriNativeSuccessorTrampolineContract>,
    install_packet: Option<tla_trust_cg::NativeInstallGatePacket>,
    call_packet: Option<tla_trust_cg::PetriNativeSuccessorCallPacket>,
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone)]
struct AYTrustMcNativeVerificationBundleFacadeEvidence {
    accepted_for_consumer: bool,
    fail_closed: bool,
    status_code: &'static str,
    reason_code: &'static str,
    consumer_acceptance_api: &'static str,
    consumer_rejection_status_code: &'static str,
    consumer_rejection_reason_code: &'static str,
    consumer_rejection_code: &'static str,
    consumer_rejection_fail_closed: bool,
    consumer_rejection_ready_for_trust_mc_chc_handoff: bool,
    model_validated: bool,
    verification_level_code: &'static str,
    proof_replay_status_code: &'static str,
    ready_for_trust_mc_chc_handoff: bool,
    matched_trust_mc_request_count: usize,
    matched_trust_mc_chc_request_count: usize,
    matched_trust_mc_evidence_count: usize,
    matched_trust_mc_artifact_count: usize,
    model_acceptance_accepted_for_consumer: bool,
    model_acceptance_fail_closed: bool,
    model_acceptance_status_code: &'static str,
    model_acceptance_reason_code: &'static str,
    model_acceptance_api: &'static str,
    model_acceptance_consumer_acceptance_api: &'static str,
    model_acceptance_consumer_rejection_status_code: &'static str,
    model_acceptance_consumer_rejection_reason_code: &'static str,
    model_acceptance_consumer_rejection_fail_closed: bool,
    model_acceptance_proof_handoff_ready: bool,
    model_acceptance_ready_for_solver_validation: bool,
    model_acceptance_solver_model_validation_present: bool,
    model_acceptance_solver_model_validation_accepted: bool,
    model_acceptance_solver_artifact_bytes_validated: bool,
    model_acceptance_solver_model_artifact_bytes_digest: String,
    model_acceptance_solver_replay_transcript_artifact_bytes_digest: String,
    model_acceptance_trust_ir_artifact_byte_attachment_count: usize,
    model_acceptance_trust_ir_artifact_byte_resolution_status_codes: String,
    model_acceptance_trust_ir_artifact_byte_resolution_reason_codes: String,
    model_acceptance_trust_ir_artifact_byte_resolution_authority_codes: String,
    model_acceptance_trust_ir_authoritative_artifact_requirement_count: usize,
    model_acceptance_trust_ir_authoritative_artifact_requirement_roles: String,
    model_acceptance_trust_ir_unauthoritative_artifact_requirement_roles: String,
    model_acceptance_trust_ir_authoritative_artifact_requirement_kinds: String,
    model_acceptance_trust_ir_unauthoritative_artifact_requirement_kinds: String,
    model_acceptance_trust_ir_authoritative_artifact_bytes_count: usize,
    model_acceptance_trust_mc_chc_proof_handoff_status_code: &'static str,
    model_acceptance_trust_mc_chc_proof_handoff_reason_code: &'static str,
    model_acceptance_trust_mc_chc_proof_handoff_schema: String,
    model_acceptance_trust_mc_chc_proof_handoff_schema_version: u32,
    model_acceptance_trust_mc_chc_proof_handoff_fail_closed: bool,
    model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_name: String,
    model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_kind_code: String,
    model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_digest: String,
    model_acceptance_trust_mc_chc_proof_handoff_model_artifact_name: String,
    model_acceptance_trust_mc_chc_proof_handoff_model_artifact_kind_code: String,
    model_acceptance_trust_mc_chc_proof_handoff_model_artifact_digest: String,
    model_acceptance_trust_mc_chc_model_validation_status_code: &'static str,
    model_acceptance_trust_mc_chc_model_validation_reason_code: &'static str,
    model_acceptance_trust_mc_chc_model_validation_schema: String,
    model_acceptance_trust_mc_chc_model_validation_schema_version: u32,
    model_acceptance_trust_mc_chc_model_validation_fail_closed: bool,
    model_acceptance_trust_mc_chc_model_validation_model_artifact_name: String,
    model_acceptance_trust_mc_chc_model_validation_model_artifact_kind_code: String,
    model_acceptance_trust_mc_chc_model_validation_model_artifact_digest: String,
    semantic_bridge_proof_identity_schema: &'static str,
    semantic_bridge_proof_identity_schema_version: u32,
    semantic_bridge_proof_identity_digest: String,
    semantic_bridge_fail_closed: bool,
    semantic_bridge_status_code: &'static str,
    semantic_bridge_reason_code: &'static str,
    semantic_bridge_evidence_status_code: &'static str,
}

#[cfg(feature = "trust-cg-petri-native")]
impl AYTrustMcNativeVerificationBundleFacadeEvidence {
    fn is_accepted_for_native_production(&self) -> bool {
        self.model_acceptance_accepted_for_consumer
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[allow(non_camel_case_types)]
mod ay_trust_mc_native_bundle {

    pub const TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_SCHEMA: &str =
        "ay.chc.trust_mc_native_verification_bundle_facade.v2";
    pub const TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_SCHEMA_VERSION: u32 = 2;
    pub const TRUST_MC_PETRI_SUCCESSOR_CHC_MODEL_ACCEPTANCE_SCHEMA: &str =
        "ay.chc.trust_mc_petri_successor_model_acceptance.v1";
    pub const TRUST_MC_PETRI_SUCCESSOR_CHC_MODEL_ACCEPTANCE_SCHEMA_VERSION: u32 = 1;
    pub const TRUST_MC_NATIVE_VERIFICATION_BUNDLE_PROBLEM: &str =
        "trust_mc_native_verification_bundle";
    pub const TRUST_MC_NATIVE_VERIFICATION_BUNDLE_BACKEND_CODE: &str =
        "ay_chc_trust_mc_native_bundle";
    pub const TRUST_MC_NATIVE_VERIFICATION_BUNDLE_DOMAIN: &str = "native_bundle";
    pub const TRUST_MC_NATIVE_VERIFICATION_BUNDLE_SCOPE: &str = "trust_mc_native_chc";
    pub const PETRI_SUCCESSOR_TRUST_MC_CHC_MODEL_ACCEPTANCE_REPORT_API_NAME: &str =
        "ay::chc::trust_mc_petri_successor_chc_model_acceptance_report";
    // Consumer-acceptance API name. Must equal trust-ir's
    // `PETRI_SUCCESSOR_TRUST_MC_CHC_CONSUMER_ACCEPTANCE_API_NAME` (the upstream
    // descriptor that production acceptance reporting resolves via
    // `production_consumer_acceptance_api_name`). Hardcoded — like the sibling
    // MODEL_ACCEPTANCE_REPORT name above — to avoid pulling the optional
    // `trust-ir` dep into this always-compiled facade.
    pub const PETRI_SUCCESSOR_TRUST_MC_CHC_CONSUMER_ACCEPTANCE_API_NAME: &str =
        "ay::chc::TrustMcPetriSuccessorChcModelAcceptanceReport::accept_for_consumer";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TrustMcConsumerRejection {
        pub status_code: &'static str,
        pub reason_code: &'static str,
        pub consumer_rejection_code: &'static str,
        pub fail_closed: bool,
        pub ready_for_trust_mc_chc_handoff: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum trust_mcNativeVerificationBundleConsumerDecision {
        Accepted,
        Rejected(TrustMcConsumerRejection),
    }

    #[derive(Debug, Clone)]
    pub struct SemanticBridgeProofIdentity {
        pub schema: &'static str,
        pub schema_version: u32,
        pub digest: String,
        pub fail_closed: bool,
        pub status_code: &'static str,
        pub reason_code: &'static str,
        pub evidence_status_code: &'static str,
    }

    #[derive(Debug, Clone)]
    pub struct trust_mcNativeVerificationBundleReport {
        pub schema: &'static str,
        pub schema_version: u32,
        pub problem: &'static str,
        pub preferred_backend_code: &'static str,
        pub domain: &'static str,
        pub scope: &'static str,
        pub status_code: &'static str,
        pub reason_code: &'static str,
        pub model_validated: bool,
        pub verification_level_code: &'static str,
        pub proof_replay_status_code: &'static str,
        pub ready_for_trust_mc_chc_handoff: bool,
        pub trust_mc_request_count: usize,
        pub trust_mc_evidence_count: usize,
        pub native_evidence_entry_count: usize,
        pub matched_trust_mc_request_count: usize,
        pub matched_trust_mc_chc_request_count: usize,
        pub matched_trust_mc_evidence_count: usize,
        pub matched_trust_mc_artifact_count: usize,
        pub matched_trust_mc_request_ids: Vec<u32>,
        pub matched_trust_mc_request_mode_codes: Vec<&'static str>,
        pub matched_trust_mc_request_digests: Vec<String>,
        pub matched_trust_mc_evidence_digests: Vec<String>,
        pub matched_trust_mc_artifact_kind_codes: Vec<&'static str>,
        pub semantic_bridge_status_code: &'static str,
        pub semantic_bridge_reason_code: &'static str,
        pub semantic_bridge_evidence_status_code: &'static str,
        pub semantic_bridge_relation_code: &'static str,
        pub semantic_bridge_function_index: u32,
        pub semantic_bridge_formula_schema: String,
        pub semantic_bridge_digest: String,
        pub semantic_bridge_proof_obligation_index: Option<u32>,
        pub semantic_bridge_proof_status_code: Option<&'static str>,
        pub semantic_bridge_proof_digest: Option<String>,
        pub semantic_bridge_evidence_digest: Option<String>,
        pub semantic_bridge_report: trust_ir::NativeSemanticBridgeReport,
        rejection: TrustMcConsumerRejection,
    }

    impl trust_mcNativeVerificationBundleReport {
        pub fn semantic_bridge_proof_identity(&self) -> SemanticBridgeProofIdentity {
            SemanticBridgeProofIdentity {
                schema: self.semantic_bridge_report.proof_identity_schema(),
                schema_version: self.semantic_bridge_report.proof_identity_schema_version(),
                digest: self
                    .semantic_bridge_report
                    .proof_identity_digest()
                    .to_string(),
                fail_closed: self.semantic_bridge_report.fail_closed(),
                status_code: self.semantic_bridge_report.status_code(),
                reason_code: self.semantic_bridge_report.reason_code(),
                evidence_status_code: self.semantic_bridge_report.evidence_status_code(),
            }
        }

        pub fn consumer_decision(&self) -> trust_mcNativeVerificationBundleConsumerDecision {
            trust_mcNativeVerificationBundleConsumerDecision::Rejected(self.rejection)
        }

        pub fn accept_for_consumer(&self) -> Result<(), TrustMcConsumerRejection> {
            Err(self.rejection)
        }
    }

    #[derive(Debug, Clone)]
    pub struct TrustMcPetriSuccessorChcModelAcceptanceReport {
        pub schema: &'static str,
        pub schema_version: u32,
        pub status_code: &'static str,
        pub reason_code: &'static str,
        pub fail_closed: bool,
        pub proof_handoff_ready: bool,
        pub ready_for_solver_validation: bool,
        pub solver_model_validation_present: bool,
        pub solver_model_validation_accepted: bool,
        pub trust_mc_chc_proof_handoff_status_code: &'static str,
        pub trust_mc_chc_proof_handoff_reason_code: &'static str,
        pub trust_mc_chc_model_validation_status_code: &'static str,
        pub trust_mc_chc_model_validation_reason_code: &'static str,
        pub model_artifact_digest: Option<String>,
        pub proof_identity_digest: Option<String>,
        pub replay_transcript_digest: Option<String>,
        pub solver_model_artifact_digest: Option<String>,
        pub solver_proof_identity_digest: Option<String>,
        pub solver_replay_transcript_digest: Option<String>,
        pub solver_artifact_bytes_validated: bool,
        pub solver_model_artifact_bytes_digest: Option<String>,
        pub solver_replay_transcript_artifact_bytes_digest: Option<String>,
        pub solver_validation_digest: Option<String>,
        pub solver_identity_count: usize,
        pub trust_mc_chc_model_validation_readiness_report:
            trust_ir::PetriSuccessorTrustMcChcModelValidationReadinessReport,
        rejection: TrustMcConsumerRejection,
    }

    impl TrustMcPetriSuccessorChcModelAcceptanceReport {
        pub fn accept_for_consumer(&self) -> Result<(), TrustMcConsumerRejection> {
            Err(self.rejection)
        }
    }

    #[derive(Debug, Clone)]
    pub struct TrustMcPetriSuccessorChcLoweringReport {
        pub ready_for_trust_mc_chc_handoff: bool,
    }

    #[derive(Debug, Clone)]
    pub struct TrustMcPetriSuccessorNativeRouteAdmissionDecision {
        pub schema: &'static str,
        pub schema_version: u32,
        pub status_code: &'static str,
        pub reason_code: &'static str,
        pub accepted_for_consumer: bool,
        pub fail_closed: bool,
    }

    impl TrustMcPetriSuccessorNativeRouteAdmissionDecision {
        pub fn to_key_value_rows(&self) -> Vec<(String, String)> {
            vec![
                ("schema".to_owned(), self.schema.to_owned()),
                ("schema_version".to_owned(), self.schema_version.to_string()),
                ("status_code".to_owned(), self.status_code.to_owned()),
                ("reason_code".to_owned(), self.reason_code.to_owned()),
                (
                    "accepted_for_consumer".to_owned(),
                    self.accepted_for_consumer.to_string(),
                ),
                ("fail_closed".to_owned(), self.fail_closed.to_string()),
                (
                    "helpers".to_owned(),
                    "ay_trust_mc_native_bundle::trust_mc_petri_successor_native_route_admission_decision".to_owned(),
                ),
                (
                    "validators".to_owned(),
                    "ay_trust_mc_native_bundle::validate_trust_mc_petri_successor_native_route_admission_decision".to_owned(),
                ),
            ]
        }
    }

    fn proof_status_code(status: trust_ir::ProofStatus) -> &'static str {
        match status {
            trust_ir::ProofStatus::Pending => "pending",
            trust_ir::ProofStatus::Discharged => "discharged",
            trust_ir::ProofStatus::Failed => "failed",
            trust_ir::ProofStatus::Trusted => "trusted",
            // Kernel-checkable CIC proof term — strictly stronger than `Trusted`.
            trust_ir::ProofStatus::Certified => "certified",
        }
    }

    pub fn solve_trust_mc_petri_successor_native_verification_bundle(
        bundle: &trust_ir::NativeVerificationBundle,
        function: trust_ir::FuncId,
    ) -> trust_mcNativeVerificationBundleReport {
        let semantic_bridge_report = bundle.petri_successor_semantic_bridge_report(function);
        let binding_report = bundle.petri_successor_trust_mc_chc_binding_report(function);
        let proof_handoff_report =
            bundle.petri_successor_trust_mc_chc_proof_handoff_report(function);
        let model_validation_report =
            bundle.petri_successor_trust_mc_chc_model_validation_readiness_report(function);
        let rejection = TrustMcConsumerRejection {
            status_code: "blocked",
            reason_code: "chc_problem_lowering_unavailable",
            consumer_rejection_code: "chc_problem_lowering_unavailable",
            fail_closed: true,
            ready_for_trust_mc_chc_handoff: proof_handoff_report.is_ready(),
        };
        let matched_trust_mc_request_ids = binding_report
            .request
            .map(|request| vec![request.index()])
            .unwrap_or_default();
        let matched_trust_mc_request_digests = binding_report
            .request_digest
            .map(|digest| vec![digest.to_string()])
            .unwrap_or_default();
        let matched_trust_mc_evidence_digests = binding_report
            .evidence_digest
            .map(|digest| vec![digest.to_string()])
            .unwrap_or_default();
        let mut matched_trust_mc_artifact_kind_codes = Vec::new();
        if let Some(artifact) = &binding_report.horn_clause_artifact {
            matched_trust_mc_artifact_kind_codes.push(artifact.kind.code());
        }
        if proof_handoff_report.replay_transcript_artifact.is_some() {
            matched_trust_mc_artifact_kind_codes.push("replay_transcript");
        }
        if proof_handoff_report.model_artifact.is_some() {
            matched_trust_mc_artifact_kind_codes.push("trust_mc_model");
        }
        trust_mcNativeVerificationBundleReport {
            schema: TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_SCHEMA,
            schema_version: TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_SCHEMA_VERSION,
            problem: TRUST_MC_NATIVE_VERIFICATION_BUNDLE_PROBLEM,
            preferred_backend_code: TRUST_MC_NATIVE_VERIFICATION_BUNDLE_BACKEND_CODE,
            domain: TRUST_MC_NATIVE_VERIFICATION_BUNDLE_DOMAIN,
            scope: TRUST_MC_NATIVE_VERIFICATION_BUNDLE_SCOPE,
            status_code: "blocked",
            reason_code: "chc_problem_lowering_unavailable",
            model_validated: model_validation_report.model_validated,
            verification_level_code: "typed_handoff",
            proof_replay_status_code: proof_handoff_report.status_code(),
            ready_for_trust_mc_chc_handoff: proof_handoff_report.is_ready(),
            trust_mc_request_count: usize::from(binding_report.request.is_some()),
            trust_mc_evidence_count: usize::from(binding_report.evidence_digest.is_some()),
            native_evidence_entry_count: bundle.evidence_bundles.len(),
            matched_trust_mc_request_count: usize::from(binding_report.request.is_some()),
            matched_trust_mc_chc_request_count: usize::from(binding_report.request.is_some()),
            matched_trust_mc_evidence_count: usize::from(binding_report.evidence_digest.is_some()),
            matched_trust_mc_artifact_count: matched_trust_mc_artifact_kind_codes.len(),
            matched_trust_mc_request_ids,
            matched_trust_mc_request_mode_codes: if binding_report.request.is_some() {
                vec!["chc"]
            } else {
                Vec::new()
            },
            matched_trust_mc_request_digests,
            matched_trust_mc_evidence_digests,
            matched_trust_mc_artifact_kind_codes,
            semantic_bridge_status_code: semantic_bridge_report.status_code(),
            semantic_bridge_reason_code: semantic_bridge_report.reason_code(),
            semantic_bridge_evidence_status_code: semantic_bridge_report.evidence_status_code(),
            semantic_bridge_relation_code: semantic_bridge_report.bridge.relation.code(),
            semantic_bridge_function_index: function.index(),
            semantic_bridge_formula_schema: semantic_bridge_report.bridge.formula_schema.clone(),
            semantic_bridge_digest: semantic_bridge_report.bridge_digest.to_string(),
            semantic_bridge_proof_obligation_index: semantic_bridge_report
                .proof_obligation
                .map(|proof| proof.index()),
            semantic_bridge_proof_status_code: semantic_bridge_report
                .proof_status
                .map(proof_status_code),
            semantic_bridge_proof_digest: semantic_bridge_report
                .proof_digest
                .map(|digest| digest.to_string()),
            semantic_bridge_evidence_digest: semantic_bridge_report
                .evidence_digest
                .map(|digest| digest.to_string()),
            semantic_bridge_report,
            rejection,
        }
    }

    pub fn trust_mc_petri_successor_chc_model_acceptance_report(
        bundle: &trust_ir::NativeVerificationBundle,
        function: trust_ir::FuncId,
    ) -> TrustMcPetriSuccessorChcModelAcceptanceReport {
        let readiness =
            bundle.petri_successor_trust_mc_chc_model_validation_readiness_report(function);
        let proof_handoff = &readiness.proof_handoff_report;
        let rejection = TrustMcConsumerRejection {
            status_code: "rejected",
            reason_code: "proof_handoff_blocked",
            consumer_rejection_code: readiness.reason_code(),
            fail_closed: true,
            ready_for_trust_mc_chc_handoff: proof_handoff.is_ready(),
        };
        TrustMcPetriSuccessorChcModelAcceptanceReport {
            schema: TRUST_MC_PETRI_SUCCESSOR_CHC_MODEL_ACCEPTANCE_SCHEMA,
            schema_version: TRUST_MC_PETRI_SUCCESSOR_CHC_MODEL_ACCEPTANCE_SCHEMA_VERSION,
            status_code: "rejected",
            reason_code: "proof_handoff_blocked",
            fail_closed: true,
            proof_handoff_ready: proof_handoff.is_ready(),
            ready_for_solver_validation: readiness.is_ready_for_solver_validation(),
            solver_model_validation_present: readiness.model_validated,
            solver_model_validation_accepted: readiness.model_validated,
            trust_mc_chc_proof_handoff_status_code: proof_handoff.status_code(),
            trust_mc_chc_proof_handoff_reason_code: proof_handoff.reason_code(),
            trust_mc_chc_model_validation_status_code: readiness.status_code(),
            trust_mc_chc_model_validation_reason_code: readiness.reason_code(),
            model_artifact_digest: readiness
                .model_artifact_digest
                .map(|digest| digest.to_string()),
            proof_identity_digest: proof_handoff
                .proof_identity_digest
                .map(|digest| digest.to_string()),
            replay_transcript_digest: proof_handoff
                .replay_transcript_digest
                .map(|digest| digest.to_string()),
            solver_model_artifact_digest: readiness
                .model_artifact_digest
                .map(|digest| digest.to_string()),
            solver_proof_identity_digest: proof_handoff
                .proof_identity_digest
                .map(|digest| digest.to_string()),
            solver_replay_transcript_digest: proof_handoff
                .replay_transcript_digest
                .map(|digest| digest.to_string()),
            solver_artifact_bytes_validated: false,
            solver_model_artifact_bytes_digest: None,
            solver_replay_transcript_artifact_bytes_digest: None,
            solver_validation_digest: None,
            solver_identity_count: readiness.solver_identities.len(),
            trust_mc_chc_model_validation_readiness_report: readiness,
            rejection,
        }
    }

    pub fn trust_mc_petri_successor_chc_lowering_report(
        bundle: &trust_ir::NativeVerificationBundle,
        function: trust_ir::FuncId,
    ) -> TrustMcPetriSuccessorChcLoweringReport {
        let proof_handoff = bundle.petri_successor_trust_mc_chc_proof_handoff_report(function);
        TrustMcPetriSuccessorChcLoweringReport {
            ready_for_trust_mc_chc_handoff: proof_handoff.is_ready(),
        }
    }

    pub fn trust_mc_petri_successor_native_route_admission_decision_from_reports(
        ay_report: &trust_mcNativeVerificationBundleReport,
        lowering_report: &TrustMcPetriSuccessorChcLoweringReport,
        model_acceptance_report: &TrustMcPetriSuccessorChcModelAcceptanceReport,
    ) -> TrustMcPetriSuccessorNativeRouteAdmissionDecision {
        let accepted_for_consumer = ay_report.accept_for_consumer().is_ok()
            && model_acceptance_report.accept_for_consumer().is_ok()
            && lowering_report.ready_for_trust_mc_chc_handoff;
        TrustMcPetriSuccessorNativeRouteAdmissionDecision {
            schema: "ay.chc.trust_mc_petri_successor_native_route_admission.v1",
            schema_version: 1,
            status_code: if accepted_for_consumer {
                "accepted"
            } else {
                "blocked"
            },
            reason_code: if accepted_for_consumer {
                "accepted"
            } else {
                "chc_problem_lowering_unavailable"
            },
            accepted_for_consumer,
            fail_closed: !accepted_for_consumer,
        }
    }

    pub fn trust_mc_petri_successor_native_route_admission_decision(
        bundle: &trust_ir::NativeVerificationBundle,
        function: trust_ir::FuncId,
    ) -> TrustMcPetriSuccessorNativeRouteAdmissionDecision {
        let ay_report = solve_trust_mc_petri_successor_native_verification_bundle(bundle, function);
        let lowering_report = trust_mc_petri_successor_chc_lowering_report(bundle, function);
        let model_acceptance_report =
            trust_mc_petri_successor_chc_model_acceptance_report(bundle, function);
        trust_mc_petri_successor_native_route_admission_decision_from_reports(
            &ay_report,
            &lowering_report,
            &model_acceptance_report,
        )
    }

    pub fn validate_trust_mc_petri_successor_native_route_admission_key_value_rows(
        decision: &TrustMcPetriSuccessorNativeRouteAdmissionDecision,
        rows: &[(String, String)],
    ) -> TrustMcPetriSuccessorNativeRouteAdmissionDecision {
        let mut validated = decision.clone();
        let has_schema = rows
            .iter()
            .any(|(key, value)| key == "schema" && value == decision.schema);
        if !has_schema {
            validated.status_code = "blocked";
            validated.reason_code = "invalid_route_admission_rows";
            validated.accepted_for_consumer = false;
            validated.fail_closed = true;
        }
        validated
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone, Copy)]
struct TrustCgPetriNextProductionBlocker<'a> {
    source: &'static str,
    api: &'a str,
    input: &'a str,
    evidence: &'a str,
    reason_code: &'a str,
    status_code: &'a str,
    blocker_stage: &'a str,
    blocker_code: &'a str,
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_compile_artifact_missing_ty_field(
    blocker: Option<tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker>,
) -> &'static str {
    match blocker {
        None => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingNativePayloadSha256) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_NATIVE_PAYLOAD
        }
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingEntrySymbol) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_ENTRY_SYMBOL
        }
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingCallablePointer) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_CALLABLE_POINTER
        }
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingExecutableRegionSha256,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_EXECUTABLE_REGION,
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingLifetimeOwner) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_LIFETIME_OWNER
        }
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingCurrentGeneration) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TY_CURRENT_GENERATION
        }
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_compile_artifact_missing_trust_cg_field(
    blocker: Option<tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker>,
) -> &'static str {
    match blocker {
        None => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingNativePayloadSha256) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_NATIVE_PAYLOAD
        }
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingEntrySymbol) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_ENTRY_SYMBOL
        }
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingCallablePointer) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_CALLABLE_POINTER
        }
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingExecutableRegionSha256,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_EXECUTABLE_REGION,
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingLifetimeOwner) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_LIFETIME_OWNER
        }
        Some(tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingCurrentGeneration) => {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_TRUST_CG_CURRENT_GENERATION
        }
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_compile_artifact_population_blocker(
    blocker: Option<tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker>,
) -> &'static str {
    match blocker {
        None => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingNativePayloadSha256,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_NO_COMPILED_LIBRARY,
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingEntrySymbol,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_ENTRY_SYMBOL,
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingCallablePointer,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_CALLABLE_POINTER,
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingExecutableRegionSha256,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_EXECUTABLE_REGION,
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingLifetimeOwner,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_LIFETIME_OWNER,
        Some(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffBlocker::MissingCurrentGeneration,
        ) => TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_CURRENT_GENERATION,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_runtime_readiness_inputs(
    bundle: &trust_ir::NativeVerificationBundle,
    state_bytes: u64,
    target_abi_digest: Option<trust_ir::ProofDigest>,
    compile_artifact_handoff: &tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
) -> TrustCgPetriRuntimeReadinessInputs {
    let Some(native_payload_sha256) = compile_artifact_handoff.native_payload_sha256.as_deref()
    else {
        return TrustCgPetriRuntimeReadinessInputs::default();
    };
    let Some(entry_symbol) = compile_artifact_handoff.entry_symbol.as_deref() else {
        return TrustCgPetriRuntimeReadinessInputs::default();
    };

    let mut expected = tla_trust_cg::PetriNativeSuccessorExecutionExpected::canary_callable(
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
        state_bytes,
    )
    .with_native_payload_sha256(native_payload_sha256);
    if let Some(target_abi_digest) = target_abi_digest {
        expected = expected.with_target_abi_digest(target_abi_digest);
    }

    let Some(callable_contract) =
        tla_trust_cg::petri_native_successor_execution_plan_from_trust_ir_bundle(bundle, expected)
            .callable_contract
    else {
        return TrustCgPetriRuntimeReadinessInputs::default();
    };
    let Some(trampoline_contract) = tla_trust_cg::petri_native_successor_trampoline_contract(
        &callable_contract,
        entry_symbol,
        native_payload_sha256,
    ) else {
        return TrustCgPetriRuntimeReadinessInputs::default();
    };

    let install_expected = expected.with_trampoline_contract(&trampoline_contract);
    let install_packet = tla_trust_cg::petri_native_successor_install_packet_from_trust_ir_bundle(
        bundle,
        install_expected,
        &trampoline_contract,
    )
    .ok();
    let call_packet = install_packet.as_ref().and_then(|packet| {
        let callable_pointer = compile_artifact_handoff.callable_pointer?;
        tla_trust_cg::petri_native_successor_call_packet_from_trust_ir_bundle(
            bundle,
            install_expected.with_native_install_gate_packet(packet),
            &trampoline_contract,
            callable_pointer,
        )
        .ok()
    });

    TrustCgPetriRuntimeReadinessInputs {
        trampoline_contract: Some(trampoline_contract),
        install_packet,
        call_packet,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_compile_artifact_handoff_attempt(
    callable_contract: Option<&tla_trust_cg::PetriNativeSuccessorCallableContract>,
    installed_artifact: Option<&tla_trust_cg::InstalledArtifact>,
    lookup_entry_symbol: Option<&str>,
) -> TrustCgPetriCompileArtifactHandoffAttempt {
    if let Some(installed_artifact) = installed_artifact {
        let mut evidence = installed_artifact
            .petri_native_successor_compile_artifact_handoff_evidence(Some(
                lookup_entry_symbol.unwrap_or(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL),
            ));
        if evidence.entry_symbol.as_deref() != Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL) {
            evidence.entry_symbol = Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL.to_string());
            evidence.compile_artifact_handoff_sha256 =
                evidence.canonical_compile_artifact_handoff_sha256();
        }
        let missing_ty_artifact_field =
            trust_cg_petri_compile_artifact_missing_ty_field(evidence.blocker);
        let missing_trust_cg_artifact_field =
            trust_cg_petri_compile_artifact_missing_trust_cg_field(evidence.blocker);
        let missing_artifact_blocker =
            trust_cg_petri_compile_artifact_population_blocker(evidence.blocker);
        let next_production_input = if evidence.is_ready() {
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE
        } else {
            evidence
                .required_field
                .unwrap_or(TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE)
        };
        let next_production_reason_code = if evidence.is_ready() {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            evidence
                .reason_code
                .unwrap_or(TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE)
        };

        return TrustCgPetriCompileArtifactHandoffAttempt {
            evidence,
            installed_artifact_available: true,
            real_artifact_source: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_INSTALLED_ARTIFACT,
            entry_symbol_source: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_API,
            native_payload_source: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_API,
            ty_wiring_status: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_WIRING_STATUS_AVAILABLE,
            ty_wiring_blocker: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
            ty_required_field: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
            missing_ty_artifact_field,
            missing_trust_cg_artifact_field,
            missing_artifact_blocker,
            next_production_api: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_API,
            next_production_input,
            next_production_reason_code,
        };
    }

    let (entry_symbol, entry_symbol_source) = callable_contract.map_or(
        (
            PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_ENTRY_SYMBOL_SOURCE_PETRI,
        ),
        |contract| {
            (
                contract.entry_function.as_str(),
                TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_ENTRY_SYMBOL_SOURCE_CONTRACT,
            )
        },
    );
    let mut input = tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffInput::default()
        .with_entry_symbol(entry_symbol);
    let native_payload_source = if let Some(contract) = callable_contract {
        input = input.with_native_payload_sha256(contract.native_payload_sha256.as_str());
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_NATIVE_PAYLOAD_SOURCE_CONTRACT
    } else {
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_UNAVAILABLE
    };
    let evidence = tla_trust_cg::petri_native_successor_compile_artifact_handoff_evidence(input);
    let real_artifact_source = if callable_contract.is_some() {
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_CALLABLE_CONTRACT
    } else {
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE
    };

    TrustCgPetriCompileArtifactHandoffAttempt {
        evidence,
        installed_artifact_available: false,
        real_artifact_source,
        entry_symbol_source,
        native_payload_source,
        ty_wiring_status:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_WIRING_STATUS_MISSING_INSTALLED_ARTIFACT,
        ty_wiring_blocker:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_INSTALLED_ARTIFACT,
        ty_required_field: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_INSTALLED_ARTIFACT_FIELD,
        missing_ty_artifact_field:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_INSTALLED_ARTIFACT_FIELD,
        missing_trust_cg_artifact_field: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
        missing_artifact_blocker:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_INSTALLED_ARTIFACT,
        next_production_api: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_INSTALLED_ARTIFACT_API,
        next_production_input:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_MISSING_INSTALLED_ARTIFACT_FIELD,
        next_production_reason_code:
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_BLOCKER_MISSING_INSTALLED_ARTIFACT,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone, Copy)]
struct TrustCgPetriRuntimeReadinessSurface {
    api: &'static str,
    installed_artifact_api: &'static str,
    installed_artifact_required_trust_cg_rev: &'static str,
    schema: &'static str,
    schema_version: u32,
    packet_type: &'static str,
    mock_executable_call_api: &'static str,
    mock_executable_call_schema: &'static str,
    mock_executable_call_schema_version: u32,
    mock_executable_call_role: &'static str,
    mock_executable_call_descriptor_available: bool,
    mock_executable_call_descriptor_authoritative: bool,
    mock_executable_call_descriptor_source: &'static str,
    mock_executable_call_descriptor_name: &'static str,
    mock_executable_call_gate_enabled: bool,
    mock_executable_call_gate_kind: &'static str,
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_runtime_readiness_surface() -> TrustCgPetriRuntimeReadinessSurface {
    let _: TrustCgPetriRuntimeReadinessPacketBuilder =
        tla_trust_cg::petri_native_successor_runtime_readiness_packet;
    let _: fn(
        &tla_trust_cg::InstalledArtifact,
        Option<&str>,
        Option<&tla_trust_cg::NativeInstallGatePacket>,
        Option<&tla_trust_cg::PetriNativeSuccessorTrampolineContract>,
        Option<&tla_trust_cg::PetriNativeSuccessorCallPacket>,
        Option<u64>,
    ) -> tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket =
        tla_trust_cg::InstalledArtifact::petri_native_successor_runtime_readiness_packet;
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket>();
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorRuntimeReadinessStatus>();
    let _ = std::mem::size_of::<tla_trust_cg::PetriNativeSuccessorRuntimeReadinessBlocker>();
    let _: fn(
        &tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
        Option<&tla_trust_cg::PetriNativeSuccessorCallPacket>,
        &tla_trust_cg::PetriNativeSuccessorMockExecutableCallGate,
        &[u8],
        &[u8],
    ) -> tla_trust_cg::PetriNativeSuccessorMockExecutableCallReport =
        tla_trust_cg::petri_native_successor_mock_executable_call_dry_run;
    let production_gate =
        tla_trust_cg::PetriNativeSuccessorMockExecutableCallGate::disabled_for_production();
    debug_assert!(!production_gate.enabled);
    debug_assert_eq!(production_gate.gate_kind, "production_fail_closed");
    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let runtime_readiness_surface = downstream_contract.runtime_readiness;
    let mock_executable_call_surface = downstream_contract.mock_executable_call;
    debug_assert_eq!(
        downstream_contract.schema,
        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_DOWNSTREAM_CONTRACT_SCHEMA
    );
    debug_assert_eq!(
        runtime_readiness_surface.schema,
        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_PACKET_SCHEMA
    );
    debug_assert_eq!(
        mock_executable_call_surface.schema,
        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_SCHEMA
    );
    debug_assert_eq!(
        downstream_contract.trust_ir_native_bundle_identity,
        tla_trust_cg::petri_native_successor_trust_ir_bundle_identity_descriptor()
    );
    debug_assert_eq!(
        downstream_contract.trust_ir_native_bundle_identity,
        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_TRUST_IR_BUNDLE_IDENTITY_DESCRIPTOR
    );

    TrustCgPetriRuntimeReadinessSurface {
        api: TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_API,
        installed_artifact_api: TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_INSTALLED_ARTIFACT_API,
        installed_artifact_required_trust_cg_rev:
            TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_INSTALLED_ARTIFACT_REQUIRED_TRUST_CG_REV,
        schema: runtime_readiness_surface.schema,
        schema_version: runtime_readiness_surface.schema_version,
        packet_type: TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_TYPE,
        mock_executable_call_api: TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_API,
        mock_executable_call_schema: mock_executable_call_surface.schema,
        mock_executable_call_schema_version: mock_executable_call_surface.schema_version,
        mock_executable_call_role: TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_ROLE,
        mock_executable_call_descriptor_available: true,
        mock_executable_call_descriptor_authoritative: true,
        mock_executable_call_descriptor_source: TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API,
        mock_executable_call_descriptor_name: mock_executable_call_surface.name,
        mock_executable_call_gate_enabled: production_gate.enabled,
        mock_executable_call_gate_kind: production_gate.gate_kind,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[path = "trust_cg_petri_native.rs"]
mod native;

#[cfg(feature = "trust-cg-petri-native")]
#[allow(unused_imports)]
pub(crate) use native::{
    checked_native_all_transition_successors_cached_into, petri_native_successor_batch_candidate,
    PetriNativeAllTransitionConfig, PetriNativeAllTransitionKernel,
    PetriNativeCallableSuccessorBatch, PetriNativeSuccessorBatchCandidate,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PetriKernelError {
    CacheAbiVersionMismatch {
        expected: u32,
        actual: u32,
    },
    CachePlaceCountMismatch {
        expected: usize,
        actual: usize,
    },
    CacheTransitionCountMismatch {
        expected: usize,
        actual: usize,
    },
    CachePlanCountMismatch {
        expected: usize,
        actual: usize,
    },
    CachePlanTransitionMismatch {
        index: usize,
        transition: TransitionIdx,
    },
    TransitionOutOfBounds {
        transition: TransitionIdx,
        transition_count: usize,
    },
    PlaceOutOfBounds {
        place: PlaceIdx,
        place_count: usize,
    },
    StateLenMismatch {
        expected: usize,
        actual: usize,
    },
    TokenExceedsI64 {
        place: usize,
        value: u64,
    },
    NegativeFlatToken {
        place: usize,
        value: i64,
    },
    ArcWeightExceedsI64 {
        transition: TransitionIdx,
        place: PlaceIdx,
        weight: u64,
    },
    TokenOverflow {
        place: PlaceIdx,
        value: i64,
        delta: i64,
    },
    ConstantExceedsI64 {
        value: u64,
    },
    IntExprOverflow {
        detail: String,
    },
    ParityMismatch {
        transition: TransitionIdx,
        detail: String,
    },
    PredicateParityMismatch {
        detail: String,
    },
    NativeCompile {
        detail: String,
    },
    NativeSymbol {
        detail: String,
    },
    NativeStatus {
        status: PetriNativeAllSuccessorsStatus,
        detail: String,
    },
    NativeCandidateMismatch {
        detail: String,
    },
    NativeOutputCountExceedsCapacity {
        count: usize,
        capacity: usize,
    },
    CountExceedsU32 {
        what: &'static str,
        count: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PetriNativeAllSuccessorsStatus {
    Ok,
    InvalidAbi,
    BufferOverflow,
    TokenOverflow,
    Unsupported,
    Unknown(u32),
}

impl PetriNativeAllSuccessorsStatus {
    #[must_use]
    pub(crate) fn as_raw(self) -> u32 {
        match self {
            Self::Ok => 0,
            Self::InvalidAbi => 1,
            Self::BufferOverflow => 2,
            Self::TokenOverflow => 3,
            Self::Unsupported => 4,
            Self::Unknown(value) => value,
        }
    }

    #[must_use]
    pub(crate) fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Ok,
            1 => Self::InvalidAbi,
            2 => Self::BufferOverflow,
            3 => Self::TokenOverflow,
            4 => Self::Unsupported,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PetriKernelLayout {
    place_count: usize,
}

impl PetriKernelLayout {
    #[must_use]
    pub(crate) fn for_net(net: &PetriNet) -> Self {
        Self {
            place_count: net.num_places(),
        }
    }

    #[must_use]
    pub(crate) fn state_len(self) -> usize {
        self.place_count
    }

    fn check_state_len(self, actual: usize) -> Result<(), PetriKernelError> {
        if actual == self.place_count {
            Ok(())
        } else {
            Err(PetriKernelError::StateLenMismatch {
                expected: self.place_count,
                actual,
            })
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PetriKernelScratch {
    flat_in: Vec<i64>,
    flat_out: Vec<i64>,
    interpreter_out: Vec<u64>,
    native_transition_ids: Vec<u32>,
}

impl PetriKernelScratch {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransitionKernelPlan {
    transition: TransitionIdx,
    inputs: Vec<(PlaceIdx, i64)>,
    outputs: Vec<(PlaceIdx, i64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PetriKernelPlanCache {
    abi_version: u32,
    place_count: usize,
    transition_count: usize,
    plans: Vec<TransitionKernelPlan>,
}

impl PetriKernelPlanCache {
    pub(crate) fn for_net(net: &PetriNet) -> Result<Self, PetriKernelError> {
        let mut plans = Vec::with_capacity(net.num_transitions());
        for transition in 0..net.num_transitions() {
            plans.push(build_transition_plan(
                net,
                TransitionIdx(transition as u32),
            )?);
        }
        Ok(Self {
            abi_version: PETRI_KERNEL_ABI_VERSION,
            place_count: net.num_places(),
            transition_count: net.num_transitions(),
            plans,
        })
    }

    pub(crate) fn validate_for_net(
        &self,
        net: &PetriNet,
    ) -> Result<PetriKernelLayout, PetriKernelError> {
        if self.abi_version != PETRI_KERNEL_ABI_VERSION {
            return Err(PetriKernelError::CacheAbiVersionMismatch {
                expected: PETRI_KERNEL_ABI_VERSION,
                actual: self.abi_version,
            });
        }
        if self.place_count != net.num_places() {
            return Err(PetriKernelError::CachePlaceCountMismatch {
                expected: net.num_places(),
                actual: self.place_count,
            });
        }
        if self.transition_count != net.num_transitions() {
            return Err(PetriKernelError::CacheTransitionCountMismatch {
                expected: net.num_transitions(),
                actual: self.transition_count,
            });
        }
        if self.plans.len() != self.transition_count {
            return Err(PetriKernelError::CachePlanCountMismatch {
                expected: self.transition_count,
                actual: self.plans.len(),
            });
        }
        for (index, plan) in self.plans.iter().enumerate() {
            if plan.transition.0 as usize != index {
                return Err(PetriKernelError::CachePlanTransitionMismatch {
                    index,
                    transition: plan.transition,
                });
            }
        }
        Ok(PetriKernelLayout {
            place_count: self.place_count,
        })
    }

    fn plan(&self, transition: TransitionIdx) -> Result<&TransitionKernelPlan, PetriKernelError> {
        self.plans
            .get(transition.0 as usize)
            .ok_or(PetriKernelError::TransitionOutOfBounds {
                transition,
                transition_count: self.plans.len(),
            })
    }

    fn plans(&self) -> &[TransitionKernelPlan] {
        &self.plans
    }
}

fn petri_native_successor_state_bytes(state_len: u32) -> u64 {
    u64::from(state_len) * std::mem::size_of::<i64>() as u64
}

pub(crate) fn petri_native_successor_capability_report(net: &PetriNet) -> CapabilityReport {
    petri_native_successor_capability_report_inner(
        net,
        PetriTrustIrTransportIdentityEvidence::Auto(std::marker::PhantomData),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
#[allow(dead_code)]
pub(crate) fn petri_native_successor_capability_report_with_verification_bundle(
    net: &PetriNet,
    bundle: Option<&trust_ir::NativeVerificationBundle>,
) -> CapabilityReport {
    let evidence = bundle.map_or(
        PetriTrustIrTransportIdentityEvidence::BundleUnavailable,
        PetriTrustIrTransportIdentityEvidence::Bundle,
    );
    petri_native_successor_capability_report_inner(net, evidence)
}

fn petri_native_successor_capability_report_inner(
    net: &PetriNet,
    transport_identity_evidence: PetriTrustIrTransportIdentityEvidence<'_>,
) -> CapabilityReport {
    let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
    let gate = NativeJitFailClosedGate::from_env();
    let state_len = match u32::try_from(net.num_places()) {
        Ok(state_len) => state_len,
        Err(_) => {
            reject_native_successor(
                &mut report,
                UnsupportedReason::TooLarge("place count exceeds successor-kernel ABI"),
                format!(
                    "Petri place count {} exceeds shared successor-kernel ABI",
                    net.num_places()
                ),
            );
            return report;
        }
    };
    let max_successors = match u32::try_from(net.num_transitions()) {
        Ok(max_successors) => max_successors,
        Err(_) => {
            reject_native_successor(
                &mut report,
                UnsupportedReason::TooLarge("transition count exceeds successor-kernel ABI"),
                format!(
                    "Petri transition count {} exceeds shared successor-kernel ABI",
                    net.num_transitions()
                ),
            );
            return report;
        }
    };

    // Wide-net OOM guard. Building the per-transition plan cache and the native
    // successor codegen below is Θ(places × transitions); on AirplaneLD-PT-4000
    // (28 019 places × 32 008 transitions) the codegen worker allocates tens of
    // GB of IR. The deadline-aware wrapper only caps the *wait* on this probe —
    // the worker thread itself is not cancellable, so it keeps allocating past
    // the cap and OOMs the whole process. Decline native admission up front for
    // nets past a practical codegen budget: the capability report is a pure
    // successor-path optimization (native vs. interpreted), never a verdict, so
    // declining is strictly verdict-preserving — the interpreter runs unchanged.
    let cells = net.num_places().saturating_mul(net.num_transitions());
    if cells > NATIVE_SUCCESSOR_MAX_CELLS {
        reject_native_successor(
            &mut report,
            UnsupportedReason::TooLarge("net exceeds native successor codegen budget"),
            format!(
                "Petri net {} places × {} transitions = {cells} cells exceeds native \
                 successor codegen budget {NATIVE_SUCCESSOR_MAX_CELLS}; using interpreter",
                net.num_places(),
                net.num_transitions(),
            ),
        );
        return report;
    }

    let plan_cache = match PetriKernelPlanCache::for_net(net) {
        Ok(plan_cache) => plan_cache,
        Err(error) => {
            reject_native_successor(
                &mut report,
                unsupported_reason_for_kernel_error(&error),
                format!("Petri native successor plan cache rejected model: {error:?}"),
            );
            return report;
        }
    };
    let setup = crate::explorer::ExplorationSetup::analyze(net);
    report.add_evidence(setup.render_shared_native_candidate_evidence_row_for_net(
        "MCC",
        ExactOrUnknownStatus::Unknown,
        net,
    ));
    report.add_evidence(setup.render_shared_native_contract_evidence_row_for_net("MCC", net));
    report.add_evidence(
        setup.render_core_shared_native_planning_identity_evidence_row_for_net("MCC", net),
    );
    report.add_evidence(setup.render_shared_native_contract_manifest_evidence_row("MCC", net));
    report.add_evidence(setup.render_shared_native_engine_readiness_evidence_row("MCC", net));
    report.add_evidence(setup.render_shared_planning_fingerprint_identity_evidence_row("MCC", net));

    let shape = SuccessorKernelShape::new(state_len, 0, 0, max_successors);
    let state_bytes = petri_native_successor_state_bytes(state_len);
    let descriptor = SuccessorKernelDescriptor::new(PETRI_NATIVE_SUCCESSOR_DESCRIPTOR_NAME, shape);
    report.add_evidence(format!(
        "successor kernel descriptor name={} state_len={} max_successors={} buffer_slots={:?} requires_parity={}",
        descriptor.name,
        descriptor.shape.state_len,
        descriptor.shape.max_successors,
        descriptor.shape.successor_buffer_slots(),
        descriptor.requires_parity
    ));
    add_kernel_artifact_contract_evidence(
        &mut report,
        "successor",
        "deferred",
        &petri_successor_kernel_artifact_adoption_evidence(),
    );
    add_kernel_artifact_contract_evidence(
        &mut report,
        "predicate",
        "deferred",
        &petri_predicate_kernel_artifact_adoption_evidence(),
    );

    add_petri_native_shared_readiness_admission_evidence(
        &mut report,
        net,
        &plan_cache,
        gate,
        ExactOrUnknownStatus::Unknown,
    );
    let transport_identity_evidence = match transport_identity_evidence {
        PetriTrustIrTransportIdentityEvidence::Auto(_) => {
            default_trust_ir_transport_identity_evidence(net, &plan_cache)
        }
        evidence => evidence,
    };
    let native_route_decision = add_native_verification_bundle_evidence(
        &mut report,
        transport_identity_evidence,
        state_bytes,
        net,
        &plan_cache,
        gate,
    );
    let route_selection = native_route_decision.route_selection;
    let gate = native_route_decision.gate;
    add_native_jit_fail_closed_gate_evidence(&mut report, gate, &route_selection);

    let successor_capability = native_capability_with_evidence(
        &mut report,
        "successor",
        "validation_only",
        true,
        if route_selection.selected_for_native_execution {
            BackendCapability::available(
                BackendDomain::PetriMcc,
                BackendKind::NativeKernel,
                "trust-cg Petri native successor route selected by producer-owned evidence",
            )
            .for_problem(ProblemKind::NativeSuccessor)
            .with_facets([SolverFacet::NativeCodegen])
            .with_role(CapabilityRole::Production)
            .with_detail(gate.successor_detail(&route_selection))
        } else {
            BackendCapability::disabled(
                BackendDomain::PetriMcc,
                BackendKind::NativeKernel,
                UnsupportedReason::DisabledByPolicy(PETRI_NATIVE_SUCCESSOR_POLICY),
            )
            .for_problem(ProblemKind::NativeSuccessor)
            .with_facets([SolverFacet::NativeCodegen])
            .with_role(CapabilityRole::Validation)
            .with_detail(gate.successor_detail(&route_selection))
        },
    );
    if route_selection.selected_for_native_execution {
        report.select(successor_capability);
    } else {
        report.reject(successor_capability);
    }
    let predicate_capability = native_capability_with_evidence(
        &mut report,
        "predicate",
        "deferred",
        true,
        BackendCapability::unsupported(
            BackendDomain::PetriMcc,
            BackendKind::NativeKernel,
            UnsupportedReason::NativeKernelUnavailable,
        )
        .for_problem(ProblemKind::Safety)
        .with_facets([
            SolverFacet::NativeCodegen,
            SolverFacet::LinearIntegerArithmetic,
        ])
        .with_role(CapabilityRole::Validation)
        .with_detail(PETRI_NATIVE_PREDICATE_DETAIL),
    );
    report.reject(predicate_capability);
    report
}

#[derive(Debug)]
// Boxing the large variant would alter the enum layout and require touching
// every construction/match site (some in other modules); the size disparity is
// acceptable for this short-lived, single-instance evidence value.
#[allow(clippy::large_enum_variant)]
enum PetriTrustIrTransportIdentityEvidence<'a> {
    Auto(std::marker::PhantomData<&'a ()>),
    DependencyUnavailable(std::marker::PhantomData<&'a ()>),
    #[cfg(feature = "trust-cg-petri-native")]
    BundleUnavailable,
    #[cfg(feature = "trust-cg-petri-native")]
    BundleProductionBlocked(native::PetriNativeVerificationBundleProductionBlocker),
    #[cfg(feature = "trust-cg-petri-native")]
    Bundle(&'a trust_ir::NativeVerificationBundle),
    #[cfg(feature = "trust-cg-petri-native")]
    ProducedBundle {
        bundle: trust_ir::NativeVerificationBundle,
        installed_artifact: PetriNativeInstalledArtifactEvidence,
    },
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone)]
// Boxing the large variant would alter the enum layout and require touching
// every construction/match site (some in other modules); the size disparity is
// acceptable for this short-lived, single-instance evidence value.
#[allow(clippy::large_enum_variant)]
enum PetriNativeInstalledArtifactEvidence {
    NotAttempted,
    Available(native::PetriNativeInstalledArtifact),
    Blocked(native::PetriNativeInstalledArtifactProductionBlocker),
}

#[cfg(feature = "trust-cg-petri-native")]
struct PetriNativeInstalledArtifactEvidenceRef<'a> {
    artifact: Option<&'a tla_trust_cg::InstalledArtifact>,
    lookup_entry_symbol: Option<&'a str>,
    status: &'static str,
    reason_code: &'a str,
    production_path: &'a str,
    missing_api: &'a str,
    blocker: &'a str,
    upstream_ask: &'a str,
}

#[cfg(feature = "trust-cg-petri-native")]
impl PetriNativeInstalledArtifactEvidence {
    fn as_ref(&self) -> PetriNativeInstalledArtifactEvidenceRef<'_> {
        match self {
            Self::NotAttempted => PetriNativeInstalledArtifactEvidenceRef {
                artifact: None,
                lookup_entry_symbol: None,
                status: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_NOT_ATTEMPTED,
                reason_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                production_path: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                missing_api: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                blocker: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                upstream_ask: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
            },
            Self::Available(artifact) => PetriNativeInstalledArtifactEvidenceRef {
                artifact: Some(&artifact.artifact),
                lookup_entry_symbol: Some(artifact.lookup_entry_symbol()),
                status: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_AVAILABLE,
                reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
                production_path: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_NATIVE_LIBRARY_BRIDGE_API,
                missing_api: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                blocker: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                upstream_ask: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
            },
            Self::Blocked(blocker) => PetriNativeInstalledArtifactEvidenceRef {
                artifact: None,
                lookup_entry_symbol: None,
                status: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED,
                reason_code: blocker.reason_code,
                production_path: blocker.production_path,
                missing_api: blocker.missing_api,
                blocker: blocker.detail.as_str(),
                upstream_ask: blocker.upstream_ask,
            },
        }
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn petri_native_successor_validation_only_native_evidence_bundle(
    mut bundle: trust_ir::NativeVerificationBundle,
) -> trust_ir::NativeVerificationBundle {
    if !bundle.evidence_bundles.is_empty() {
        return bundle;
    }

    let mut evidence_bundles = Vec::with_capacity(bundle.requests.len());
    for request in &bundle.requests {
        let artifact = petri_native_successor_validation_only_native_evidence_artifact(
            request,
            bundle.trust_ir_module_digest,
        );
        let Ok(evidence_bundle) = bundle.evidence_bundle_for_request(request, vec![artifact])
        else {
            return bundle;
        };
        evidence_bundles.push(evidence_bundle);
    }

    if !evidence_bundles.is_empty() {
        bundle.evidence_bundles = evidence_bundles;
    }

    bundle
}

#[cfg(feature = "trust-cg-petri-native")]
fn petri_native_successor_runtime_native_evidence_bundle(
    mut bundle: trust_ir::NativeVerificationBundle,
    installed_artifact: PetriNativeInstalledArtifactEvidenceRef<'_>,
) -> trust_ir::NativeVerificationBundle {
    let Some(_) = installed_artifact.artifact else {
        return bundle;
    };
    if bundle.requests.is_empty()
        || trust_cg_petri_native_evidence_profile(&bundle).native_execution_artifact_available()
    {
        return bundle;
    }

    let request = bundle.requests[0].clone();
    let native_artifact =
        petri_native_successor_runtime_native_evidence_artifact(&request, installed_artifact);
    append_petri_native_evidence_artifact_for_request(&mut bundle, &request, native_artifact);
    bundle
}

#[cfg(feature = "trust-cg-petri-native")]
fn petri_native_successor_runtime_native_evidence_artifact(
    request: &trust_ir::NativeVerificationRequest,
    installed_artifact: PetriNativeInstalledArtifactEvidenceRef<'_>,
) -> trust_ir::NativeEvidenceArtifact {
    let artifact = installed_artifact
        .artifact
        .expect("runtime native evidence artifact requires installed artifact");
    let handoff = artifact.petri_native_successor_compile_artifact_handoff_evidence(
        installed_artifact.lookup_entry_symbol,
    );
    let artifact_digest_material = format!(
        "producer=ty-petri native_compiled_artifact=true request={} request_digest={} native_payload_sha256={} executable_region_sha256={} lifetime_owner={} current_generation={}",
        request.id(),
        request.stable_digest(),
        handoff.native_payload_sha256.as_deref().unwrap_or("none"),
        handoff.executable_region_sha256.as_deref().unwrap_or("none"),
        handoff.lifetime_owner.as_deref().unwrap_or("none"),
        handoff.current_generation.unwrap_or(0),
    );
    trust_ir::NativeEvidenceArtifact::new(
        format!("ty-petri-native-compiled-artifact-{}", request.id()),
        trust_ir::NativeEvidenceArtifactKind::NativeCompiledArtifact,
        trust_ir::ProofDigest::sha256_domain(
            "ty.petri.native.compiled_artifact_evidence.v1",
            artifact_digest_material.as_bytes(),
        ),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn petri_native_successor_runtime_native_receipt_available(
    bundle: &trust_ir::NativeVerificationBundle,
    installed_artifact: PetriNativeInstalledArtifactEvidenceRef<'_>,
) -> bool {
    if bundle.validate().is_err() {
        return false;
    }
    let (Some(_), Some(request)) = (installed_artifact.artifact, bundle.requests.first()) else {
        return false;
    };
    let expected =
        petri_native_successor_runtime_native_evidence_artifact(request, installed_artifact);
    bundle.evidence_bundles.iter().any(|evidence| {
        evidence.request() == request.id()
            && evidence.verifier_suite() == request.verifier_suite()
            && evidence
                .artifacts()
                .iter()
                .any(|artifact| artifact == &expected)
    })
}

#[cfg(feature = "trust-cg-petri-native")]
fn append_petri_native_evidence_artifact_for_request(
    bundle: &mut trust_ir::NativeVerificationBundle,
    request: &trust_ir::NativeVerificationRequest,
    artifact: trust_ir::NativeEvidenceArtifact,
) -> bool {
    if let Some(existing) = bundle.evidence_bundles.iter_mut().find(|evidence| {
        evidence.request() == request.id() && evidence.verifier_suite() == request.verifier_suite()
    }) {
        let artifacts = match existing {
            trust_ir::request::NativeEvidenceBundle::TrustVc(evidence) => &mut evidence.artifacts,
            trust_ir::request::NativeEvidenceBundle::TrustMc(evidence) => &mut evidence.artifacts,
            trust_ir::request::NativeEvidenceBundle::TrustWp(evidence) => &mut evidence.artifacts,
        };
        if !artifacts.iter().any(|existing| existing == &artifact) {
            artifacts.push(artifact);
        }
        return true;
    }

    let Ok(evidence) = bundle.evidence_bundle_for_request(request, vec![artifact]) else {
        return false;
    };
    bundle.evidence_bundles.push(evidence);
    true
}

#[cfg(feature = "trust-cg-petri-native")]
fn petri_native_successor_validation_only_native_evidence_artifact(
    request: &trust_ir::NativeVerificationRequest,
    trust_ir_module_digest: trust_ir::ProofDigest,
) -> trust_ir::NativeEvidenceArtifact {
    // Metadata-only evidence makes trust-ir bundle consumption explicit without
    // claiming solver proof or authorizing native activation.
    let artifact_digest_material = format!(
        "producer=ty-petri validation_only=true module_digest={} request={} request_digest={}",
        trust_ir_module_digest,
        request.id(),
        request.stable_digest()
    );

    trust_ir::NativeEvidenceArtifact::new(
        format!("ty-petri-validation-only-native-evidence-{}", request.id()),
        trust_ir::NativeEvidenceArtifactKind::BackendCapabilityMetadata,
        trust_ir::ProofDigest::sha256_domain(
            "ty.petri.native.validation_only_native_evidence.v1",
            artifact_digest_material.as_bytes(),
        ),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Default)]
struct TrustCgPetriNativeEvidenceProfile {
    artifact_count: usize,
    backend_metadata_artifact_count: usize,
    semantic_proof_artifact_count: usize,
    native_execution_artifact_count: usize,
    other_artifact_count: usize,
    metadata_request_ids: Vec<String>,
    metadata_request_digests: Vec<String>,
    metadata_artifact_digests: Vec<String>,
    trust_ir_module_digest: String,
}

#[cfg(feature = "trust-cg-petri-native")]
impl TrustCgPetriNativeEvidenceProfile {
    fn metadata_only(&self) -> bool {
        self.artifact_count > 0
            && self.backend_metadata_artifact_count == self.artifact_count
            && self.semantic_proof_artifact_count == 0
            && self.native_execution_artifact_count == 0
            && self.other_artifact_count == 0
    }

    fn semantic_proof_available(&self) -> bool {
        self.semantic_proof_artifact_count > 0
    }

    fn native_execution_artifact_available(&self) -> bool {
        self.native_execution_artifact_count > 0
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn native_jit_receipt_evidence_for_bundle(
    bundle: &trust_ir::NativeVerificationBundle,
    semantic_bundle: &trust_ir::NativeVerificationBundle,
    installed_artifact: PetriNativeInstalledArtifactEvidenceRef<'_>,
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
) -> NativeJitReceiptEvidence {
    NativeJitReceiptEvidence {
        validation_receipt_available: native::petri_native_successor_semantic_receipt_available(
            semantic_bundle,
            net,
            cache,
        ),
        parity_receipt_available: petri_native_successor_runtime_native_receipt_available(
            bundle,
            installed_artifact,
        ),
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_native_evidence_profile(
    bundle: &trust_ir::NativeVerificationBundle,
) -> TrustCgPetriNativeEvidenceProfile {
    let mut profile = TrustCgPetriNativeEvidenceProfile {
        trust_ir_module_digest: bundle.trust_ir_module_digest.to_string(),
        ..Default::default()
    };

    for evidence in &bundle.evidence_bundles {
        for artifact in evidence.artifacts() {
            profile.artifact_count += 1;
            match artifact.kind {
                trust_ir::NativeEvidenceArtifactKind::BackendCapabilityMetadata => {
                    profile.backend_metadata_artifact_count += 1;
                    profile
                        .metadata_request_ids
                        .push(evidence.request().to_string());
                    let request_digest = bundle
                        .requests
                        .iter()
                        .find(|request| request.id() == evidence.request())
                        .map(|request| request.stable_digest().to_string())
                        .unwrap_or_else(|| "missing_request_digest".to_owned());
                    profile.metadata_request_digests.push(request_digest);
                    profile
                        .metadata_artifact_digests
                        .push(artifact.digest.to_string());
                }
                trust_ir::NativeEvidenceArtifactKind::NativeCompiledArtifact => {
                    profile.native_execution_artifact_count += 1;
                }
                kind if trust_cg_petri_native_semantic_proof_artifact_kind(kind) => {
                    profile.semantic_proof_artifact_count += 1;
                }
                _ => {
                    profile.other_artifact_count += 1;
                }
            }
        }
    }

    profile
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_cg_petri_native_semantic_proof_artifact_kind(
    kind: trust_ir::NativeEvidenceArtifactKind,
) -> bool {
    matches!(
        kind,
        trust_ir::NativeEvidenceArtifactKind::TrustVcCertificateImport
            | trust_ir::NativeEvidenceArtifactKind::TrustVcMergedCertificate
            | trust_ir::NativeEvidenceArtifactKind::TrustMcHornClauses
            | trust_ir::NativeEvidenceArtifactKind::TrustMcPdrTrace
            | trust_ir::NativeEvidenceArtifactKind::TrustMcModel
            | trust_ir::NativeEvidenceArtifactKind::TrustWpVerificationCondition
            | trust_ir::NativeEvidenceArtifactKind::TrustWpReplayTrace
            | trust_ir::NativeEvidenceArtifactKind::TrustWpAbducedPrecondition
            | trust_ir::NativeEvidenceArtifactKind::ReplayTranscript
            | trust_ir::NativeEvidenceArtifactKind::Btor2Trace
            | trust_ir::NativeEvidenceArtifactKind::Btor2Proof
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn push_trust_cg_native_evidence_profile_fields(
    fields: &mut Vec<(String, String)>,
    profile: &TrustCgPetriNativeEvidenceProfile,
) {
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_backend_metadata_artifacts",
        profile.backend_metadata_artifact_count,
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_semantic_proof_artifacts",
        profile.semantic_proof_artifact_count,
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_native_execution_artifacts",
        profile.native_execution_artifact_count,
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_other_artifacts",
        profile.other_artifact_count,
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_metadata_only",
        profile.metadata_only(),
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_semantic_proof_available",
        profile.semantic_proof_available(),
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_native_execution_artifact_available",
        profile.native_execution_artifact_available(),
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_metadata_request_ids",
        join_strings_or_none(&profile.metadata_request_ids),
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_metadata_request_digests",
        join_strings_or_none(&profile.metadata_request_digests),
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_metadata_module_digest",
        &profile.trust_ir_module_digest,
    );
    push_trust_cg_native_admission_field(
        fields,
        "native_evidence_metadata_artifact_digests",
        join_strings_or_none(&profile.metadata_artifact_digests),
    );
}

fn default_trust_ir_transport_identity_evidence<'a>(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
) -> PetriTrustIrTransportIdentityEvidence<'a> {
    #[cfg(feature = "trust-cg-petri-native")]
    {
        match native::petri_native_successor_verification_bundle(net, cache) {
            native::PetriNativeVerificationBundleProduction::Available(bundle) => {
                let installed_artifact =
                    match native::petri_native_successor_installed_artifact(&bundle) {
                        native::PetriNativeInstalledArtifactProduction::Available(artifact) => {
                            PetriNativeInstalledArtifactEvidence::Available(artifact)
                        }
                        native::PetriNativeInstalledArtifactProduction::Blocked(blocker) => {
                            PetriNativeInstalledArtifactEvidence::Blocked(blocker)
                        }
                    };
                PetriTrustIrTransportIdentityEvidence::ProducedBundle {
                    bundle,
                    installed_artifact,
                }
            }
            native::PetriNativeVerificationBundleProduction::Blocked(blocker) => {
                PetriTrustIrTransportIdentityEvidence::BundleProductionBlocked(blocker)
            }
        }
    }
    #[cfg(not(feature = "trust-cg-petri-native"))]
    {
        let _ = (net, cache);
        PetriTrustIrTransportIdentityEvidence::DependencyUnavailable(std::marker::PhantomData)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeJitFailClosedGate {
    feature_enabled: bool,
    native_requested: bool,
    strict_requested: bool,
    parity_enabled: bool,
    parity_receipt_available: bool,
    validation_receipt_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeJitReceiptEvidence {
    parity_receipt_available: bool,
    validation_receipt_available: bool,
}

impl NativeJitFailClosedGate {
    fn from_env() -> Self {
        Self {
            feature_enabled: cfg!(feature = "trust-cg-petri-native"),
            native_requested: env_flag_enabled(ENABLE_NATIVE_CANDIDATE_ENV),
            strict_requested: env_flag_enabled(ENABLE_NATIVE_CANDIDATE_STRICT_ENV),
            parity_enabled: env_flag_enabled(ENABLE_TRANSITION_PARITY_ENV),
            parity_receipt_available: false,
            validation_receipt_available: false,
        }
    }

    fn with_receipt_evidence(self, evidence: NativeJitReceiptEvidence) -> Self {
        Self {
            parity_receipt_available: evidence.parity_receipt_available,
            validation_receipt_available: evidence.validation_receipt_available,
            ..self
        }
    }

    fn parity_receipt_status_code(self) -> &'static str {
        if self.parity_receipt_available {
            PETRI_NATIVE_PARITY_RECEIPT_STATUS_ACCEPTED
        } else {
            PETRI_NATIVE_PARITY_RECEIPT_STATUS_MISSING
        }
    }

    fn parity_receipt_reason_code(self) -> &'static str {
        if self.parity_receipt_available {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT
        }
    }

    fn validation_receipt_status_code(self) -> &'static str {
        if self.validation_receipt_available {
            PETRI_NATIVE_VALIDATION_RECEIPT_STATUS_ACCEPTED
        } else {
            PETRI_NATIVE_VALIDATION_RECEIPT_STATUS_MISSING
        }
    }

    fn validation_receipt_reason_code(self) -> &'static str {
        if self.validation_receipt_available {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT
        }
    }

    fn successor_detail(self, route_selection: &PetriNativeRouteSelection) -> String {
        let mode_detail = if route_selection.selected_for_native_execution {
            "trust-cg Petri native successor backend selected by producer-owned semantic, parity, validation, and callable receipts"
        } else {
            "trust-cg Petri native successor backend remains on explicit-state fallback until all native production receipts are accepted"
        };
        format!(
            "{}; \
             fail_closed={} feature={} feature_enabled={} native_env={} native_requested={} \
             strict_env={} strict_requested={} parity_env={} parity_enabled={} \
             parity_receipt_required=true parity_receipt_available={} parity_receipt_schema={} \
             parity_receipt_reason_code={} \
             validation_receipt_required=true validation_receipt_available={} validation_receipt_schema={} \
             validation_receipt_reason_code={} \
             callable_receipt_required=true callable_receipt_available={} callable_receipt_status_code={} callable_receipt_schema={} \
             callable_receipt_reason_code={} callable_receipt_gate_api={} \
             native_runtime_callable_impl_available={} runtime_readiness_status_code={} runtime_readiness_reason_code={} \
             production_selected={} trust_ir_transport_identity_available={} \
             route_selection_status_code={} route_selection_reason_code={} \
             route_selection_selected_lane={} route_selection_safe_criteria={} \
             shared_engine_owner={} shared_engine_component={} generic_prerequisites={} \
             trust_ir_required_rev={} trust_ir_current_rev={} expected_fields={}",
            mode_detail,
            route_selection.fail_closed,
            TRUST_CG_PETRI_NATIVE_FEATURE,
            self.feature_enabled,
            ENABLE_NATIVE_CANDIDATE_ENV,
            self.native_requested,
            ENABLE_NATIVE_CANDIDATE_STRICT_ENV,
            self.strict_requested,
            ENABLE_TRANSITION_PARITY_ENV,
            self.parity_enabled,
            self.parity_receipt_available,
            PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
            self.parity_receipt_reason_code(),
            self.validation_receipt_available,
            VALIDATION_RECEIPT_SCHEMA,
            self.validation_receipt_reason_code(),
            route_selection.callable_receipt_available,
            route_selection.callable_receipt_status_code(),
            PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA,
            route_selection.callable_receipt_reason_code,
            PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API,
            route_selection.native_runtime_callable_impl_available,
            route_selection.runtime_readiness_status_code(),
            route_selection.runtime_readiness_reason_code,
            route_selection.selected_for_native_execution,
            route_selection.transport_identity_available,
            route_selection.status_code,
            route_selection.reason_code,
            route_selection.selected_lane,
            PETRI_NATIVE_ROUTE_SELECTION_SAFE_CRITERIA,
            PETRI_NATIVE_ROUTE_SHARED_ENGINE_OWNER,
            PETRI_NATIVE_ROUTE_SHARED_ENGINE_COMPONENT,
            PETRI_NATIVE_ROUTE_GENERIC_PREREQUISITES,
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_REQUIRED_REV,
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV,
            TRUST_IR_NATIVE_VERIFICATION_EXPECTED_FIELDS,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PetriNativeRouteSelection {
    transport_identity_available: bool,
    producer_admission: bool,
    producer_execution_authority: bool,
    producer_production_selection: bool,
    parity_enabled: bool,
    parity_receipt_available: bool,
    validation_receipt_available: bool,
    callable_receipt_available: bool,
    native_runtime_callable_impl_available: bool,
    selected_for_native_execution: bool,
    fail_closed: bool,
    status_code: &'static str,
    reason_code: &'static str,
    selected_lane: &'static str,
    producer_admission_reason_code: String,
    producer_execution_authority_reason_code: String,
    producer_production_selection_reason_code: String,
    parity_receipt_reason_code: String,
    validation_receipt_reason_code: String,
    callable_receipt_reason_code: String,
    runtime_readiness_reason_code: String,
}

#[derive(Debug, Clone, Copy)]
struct PetriNativeRouteSelectionInput<'a> {
    transport_identity_available: bool,
    producer_admission: bool,
    producer_execution_authority: bool,
    producer_production_selection: bool,
    parity_enabled: bool,
    parity_receipt_available: bool,
    validation_receipt_available: bool,
    callable_receipt_available: bool,
    native_runtime_callable_impl_available: bool,
    producer_admission_reason_code: &'a str,
    producer_execution_authority_reason_code: &'a str,
    producer_production_selection_reason_code: &'a str,
    parity_receipt_reason_code: &'a str,
    validation_receipt_reason_code: &'a str,
    callable_receipt_reason_code: &'a str,
    runtime_readiness_reason_code: &'a str,
}

impl PetriNativeRouteSelection {
    fn production_gate_status(&self) -> &'static str {
        if self.selected_for_native_execution {
            PETRI_NATIVE_ROUTE_PRODUCTION_GATE_STATUS_SELECTED
        } else {
            PETRI_NATIVE_ROUTE_PRODUCTION_GATE_STATUS
        }
    }

    fn frontend_family_blockers(&self) -> &'static str {
        if self.selected_for_native_execution {
            PETRI_NATIVE_ROUTE_FRONTEND_FAMILY_BLOCKERS_SELECTED
        } else {
            PETRI_NATIVE_ROUTE_FRONTEND_FAMILY_BLOCKERS
        }
    }

    fn blocker_status(&self) -> &'static str {
        if self.selected_for_native_execution {
            PETRI_NATIVE_ROUTE_BLOCKER_STATUS_SELECTED
        } else {
            PETRI_NATIVE_ROUTE_BLOCKER_STATUS
        }
    }

    fn blocker_issue_refs(&self) -> &'static str {
        if self.selected_for_native_execution {
            PETRI_NATIVE_ROUTE_SELECTION_BLOCKER_ISSUES_SELECTED
        } else {
            PETRI_NATIVE_ROUTE_SELECTION_BLOCKER_ISSUES
        }
    }

    fn todo(&self) -> &'static str {
        if self.selected_for_native_execution {
            PETRI_NATIVE_ROUTE_SELECTION_TODO_SELECTED
        } else {
            PETRI_NATIVE_ROUTE_SELECTION_TODO
        }
    }

    fn callable_receipt_status_code(&self) -> &'static str {
        if self.callable_receipt_available {
            PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        } else {
            PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_MISSING
        }
    }

    fn runtime_readiness_status_code(&self) -> &'static str {
        if self.native_runtime_callable_impl_available {
            PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        } else {
            PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_MISSING
        }
    }

    fn evaluate(input: PetriNativeRouteSelectionInput<'_>) -> Self {
        let reason_code = if !input.transport_identity_available {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_MISSING_TRANSPORT
        } else if !input.producer_admission {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PRODUCER_ADMISSION
        } else if !input.producer_execution_authority {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_EXECUTION_AUTHORITY
        } else if !input.producer_production_selection {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PRODUCTION_SELECTION
        } else if !input.parity_enabled {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY
        } else if !input.parity_receipt_available {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT
        } else if !input.validation_receipt_available {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT
        } else if !input.callable_receipt_available {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT
        } else if !input.native_runtime_callable_impl_available {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL
        } else {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_NONE
        };
        let selected_for_native_execution = reason_code == PETRI_NATIVE_ROUTE_SELECTION_REASON_NONE;

        Self {
            transport_identity_available: input.transport_identity_available,
            producer_admission: input.producer_admission,
            producer_execution_authority: input.producer_execution_authority,
            producer_production_selection: input.producer_production_selection,
            parity_enabled: input.parity_enabled,
            parity_receipt_available: input.parity_receipt_available,
            validation_receipt_available: input.validation_receipt_available,
            callable_receipt_available: input.callable_receipt_available,
            native_runtime_callable_impl_available: input.native_runtime_callable_impl_available,
            selected_for_native_execution,
            fail_closed: !selected_for_native_execution,
            status_code: if selected_for_native_execution {
                PETRI_NATIVE_ROUTE_SELECTION_STATUS_SELECTED
            } else {
                PETRI_NATIVE_ROUTE_SELECTION_STATUS_FAIL_CLOSED
            },
            reason_code,
            selected_lane: if selected_for_native_execution {
                PETRI_NATIVE_ROUTE_SELECTION_LANE_NATIVE
            } else {
                PETRI_NATIVE_ROUTE_SELECTION_LANE_FALLBACK
            },
            producer_admission_reason_code: input.producer_admission_reason_code.to_owned(),
            producer_execution_authority_reason_code: input
                .producer_execution_authority_reason_code
                .to_owned(),
            producer_production_selection_reason_code: input
                .producer_production_selection_reason_code
                .to_owned(),
            parity_receipt_reason_code: input.parity_receipt_reason_code.to_owned(),
            validation_receipt_reason_code: input.validation_receipt_reason_code.to_owned(),
            callable_receipt_reason_code: input.callable_receipt_reason_code.to_owned(),
            runtime_readiness_reason_code: input.runtime_readiness_reason_code.to_owned(),
        }
    }
}

fn add_petri_native_shared_readiness_admission_evidence(
    report: &mut CapabilityReport,
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    gate: NativeJitFailClosedGate,
    exactness: ExactOrUnknownStatus,
) {
    let layout_matches_payload = cache.place_count == net.num_places()
        && cache.transition_count == net.num_transitions()
        && cache.plans().len() == net.num_transitions();

    report.add_evidence(format!(
        "Petri native_jit shared_readiness_admission source={} schema={} schema_version={} api={} \
         shared_engine_owner={} shared_engine_component={} origin_frontend={} solver_family_scope={} \
         compatible_frontend_families={} default_compatible_frontend_families={} \
         payload_identity_required=true payload_identity_source={} payload_identity_required_fields={} \
         payload_identity_status={} payload_identity_admission_status=blocked_for_trusted_native \
         payload_identity_exact_match_required=true layout_identity={} layout_abi_version={} \
         layout_place_count={} layout_transition_count={} layout_plan_count={} payload_place_count={} \
         payload_transition_count={} layout_matches_payload={} layout_fingerprint_required=true \
         layout_fingerprint_algorithm={} layout_fingerprint_admission_status=validation_only_declared \
         layout_fingerprint_exact_match_required=true fingerprint_domain_identity={} \
         fingerprint_admission_contract={} fingerprint_admission_status={} \
         fingerprint_admission_authority=validation_only fingerprint_admission_default_consumers=none \
         parity_required=true parity_env={} parity_enabled={} parity_receipt_required=true \
         parity_receipt_status={} parity_receipt_schema={} validation_receipt_required=true \
         validation_receipt_status={} validation_receipt_schema={} callable_receipt_required=true \
         callable_receipt_status={} callable_receipt_reason_code={} callable_receipt_schema={} \
         callable_receipt_gate_api={} callable_receipt_required_evidence={} \
         native_runtime_callable_impl_available=false runtime_readiness_status_code=missing \
         runtime_readiness_reason_code={} exact_or_unknown={} \
         validation_status={} exact_or_unknown_guard={} native_output_trusted=false \
         trusted_production_admitted=false trusted_production_blockers={} \
         production_gate={} production_gate_status={} production_gate_required_receipts={} \
         performance_claim_status={} production_selected=false fail_closed=true",
        PETRI_NATIVE_SHARED_READINESS_ADMISSION_SOURCE,
        PETRI_NATIVE_SHARED_READINESS_ADMISSION_SCHEMA,
        PETRI_NATIVE_SHARED_READINESS_ADMISSION_SCHEMA_VERSION,
        PETRI_NATIVE_SHARED_READINESS_ADMISSION_API,
        PETRI_NATIVE_ROUTE_SHARED_ENGINE_OWNER,
        PETRI_NATIVE_ROUTE_SHARED_ENGINE_COMPONENT,
        PETRI_NATIVE_ROUTE_ORIGIN_FRONTEND,
        PETRI_NATIVE_SHARED_SOLVER_FAMILIES,
        PETRI_NATIVE_ROUTE_COMPATIBLE_FRONTEND_FAMILIES,
        PETRI_NATIVE_ROUTE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
        PETRI_NATIVE_SHARED_PAYLOAD_IDENTITY_SOURCE,
        PETRI_NATIVE_SHARED_PAYLOAD_IDENTITY_REQUIRED_FIELDS,
        PETRI_NATIVE_SHARED_PAYLOAD_IDENTITY_STATUS,
        PETRI_NATIVE_SHARED_LAYOUT_IDENTITY,
        cache.abi_version,
        cache.place_count,
        cache.transition_count,
        cache.plans().len(),
        net.num_places(),
        net.num_transitions(),
        layout_matches_payload,
        PETRI_NATIVE_SHARED_LAYOUT_FINGERPRINT_ALGORITHM,
        PETRI_NATIVE_SHARED_FINGERPRINT_DOMAIN_IDENTITY,
        PETRI_NATIVE_SHARED_FINGERPRINT_ADMISSION_CONTRACT,
        PETRI_NATIVE_SHARED_FINGERPRINT_ADMISSION_STATUS,
        ENABLE_TRANSITION_PARITY_ENV,
        gate.parity_enabled,
        gate.parity_receipt_status_code(),
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
        gate.validation_receipt_status_code(),
        VALIDATION_RECEIPT_SCHEMA,
        PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_MISSING,
        PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA,
        PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API,
        PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE,
        PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL,
        exactness.code(),
        exactness.validation_status_code(),
        PETRI_NATIVE_SHARED_EXACT_OR_UNKNOWN_GUARD,
        PETRI_NATIVE_SHARED_TRUSTED_PRODUCTION_BLOCKERS,
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE,
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE_STATUS,
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE_REQUIRED_RECEIPTS,
        PETRI_NATIVE_SHARED_PERFORMANCE_CLAIM_STATUS,
    ));
}

fn add_petri_native_route_selection_evidence(
    report: &mut CapabilityReport,
    route_selection: &PetriNativeRouteSelection,
) {
    report.add_evidence(format!(
        "Petri native_jit route_selection source=PetriNativeRouteSelection schema={} schema_version={} api={} selected_lane={} status_code={} reason_code={} safe_class_criteria={} producer_admission={} producer_admission_reason_code={} producer_execution_authority={} producer_execution_authority_reason_code={} producer_production_selection={} producer_production_selection_reason_code={} parity_required=true parity_enabled={} parity_receipt_required=true parity_receipt_available={} parity_receipt_reason_code={} parity_receipt_schema={} parity_receipt_schema_version={} parity_receipt_gate_api={} parity_receipt_required_evidence={} validation_receipt_required=true validation_receipt_available={} validation_receipt_reason_code={} validation_receipt_schema={} validation_receipt_schema_version={} validation_receipt_gate_api={} validation_receipt_required_evidence={} callable_receipt_required=true callable_receipt_available={} callable_receipt_status_code={} callable_receipt_reason_code={} callable_receipt_schema={} callable_receipt_schema_version={} callable_receipt_gate_api={} callable_receipt_required_evidence={} native_runtime_callable_impl_available={} transport_identity_available={} runtime_readiness_status_code={} runtime_readiness_reason_code={} shared_engine_owner={} shared_engine_component={} origin_frontend={} first_beneficiary={} second_beneficiary={} extraction_status={} adoption_level={} compatible_frontend_families={} default_compatible_frontend_families={} downstream_beneficiary_families={} remaining_compatible_frontend_families={} frontend_family_blockers={} blocker_status={} generic_prerequisites={} production_gate={} production_gate_status={} production_gate_required_receipts={} production_selected={} fail_closed={} blocker_issue_refs={} todo={}",
        PETRI_NATIVE_ROUTE_SELECTION_SCHEMA,
        PETRI_NATIVE_ROUTE_SELECTION_SCHEMA_VERSION,
        PETRI_NATIVE_ROUTE_SELECTION_API,
        route_selection.selected_lane,
        route_selection.status_code,
        route_selection.reason_code,
        PETRI_NATIVE_ROUTE_SELECTION_SAFE_CRITERIA,
        route_selection.producer_admission,
        route_selection.producer_admission_reason_code,
        route_selection.producer_execution_authority,
        route_selection.producer_execution_authority_reason_code,
        route_selection.producer_production_selection,
        route_selection.producer_production_selection_reason_code,
        route_selection.parity_enabled,
        route_selection.parity_receipt_available,
        route_selection.parity_receipt_reason_code,
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION,
        PETRI_NATIVE_PARITY_RECEIPT_GATE_API,
        PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
        route_selection.validation_receipt_available,
        route_selection.validation_receipt_reason_code,
        VALIDATION_RECEIPT_SCHEMA,
        VALIDATION_RECEIPT_SCHEMA_VERSION,
        PETRI_NATIVE_VALIDATION_RECEIPT_GATE_API,
        PETRI_NATIVE_VALIDATION_RECEIPT_REQUIRED_EVIDENCE,
        route_selection.callable_receipt_available,
        route_selection.callable_receipt_status_code(),
        route_selection.callable_receipt_reason_code,
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA,
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA_VERSION,
        PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API,
        PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE,
        route_selection.native_runtime_callable_impl_available,
        route_selection.transport_identity_available,
        route_selection.runtime_readiness_status_code(),
        route_selection.runtime_readiness_reason_code,
        PETRI_NATIVE_ROUTE_SHARED_ENGINE_OWNER,
        PETRI_NATIVE_ROUTE_SHARED_ENGINE_COMPONENT,
        PETRI_NATIVE_ROUTE_ORIGIN_FRONTEND,
        PETRI_NATIVE_ROUTE_FIRST_BENEFICIARY,
        PETRI_NATIVE_ROUTE_SECOND_BENEFICIARY,
        PETRI_NATIVE_ROUTE_EXTRACTION_STATUS,
        PETRI_NATIVE_ROUTE_ADOPTION_LEVEL,
        PETRI_NATIVE_ROUTE_COMPATIBLE_FRONTEND_FAMILIES,
        PETRI_NATIVE_ROUTE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
        PETRI_NATIVE_ROUTE_DOWNSTREAM_BENEFICIARY_FAMILIES,
        PETRI_NATIVE_ROUTE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        route_selection.frontend_family_blockers(),
        route_selection.blocker_status(),
        PETRI_NATIVE_ROUTE_GENERIC_PREREQUISITES,
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE,
        route_selection.production_gate_status(),
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE_REQUIRED_RECEIPTS,
        route_selection.selected_for_native_execution,
        route_selection.fail_closed,
        route_selection.blocker_issue_refs(),
        route_selection.todo(),
    ));
}

#[derive(Debug, Clone)]
struct PetriNativeRouteDecision {
    route_selection: PetriNativeRouteSelection,
    gate: NativeJitFailClosedGate,
}

fn add_native_jit_fail_closed_gate_evidence(
    report: &mut CapabilityReport,
    gate: NativeJitFailClosedGate,
    route_selection: &PetriNativeRouteSelection,
) {
    report.add_evidence(format!(
        "Petri native_jit fail_closed_gate feature={} feature_enabled={} native_env={} native_requested={} strict_env={} strict_requested={} parity_env={} parity_enabled={} parity_receipt_required=true parity_receipt_available={} parity_receipt_status_code={} parity_receipt_reason_code={} parity_receipt_schema={} parity_receipt_schema_version={} parity_receipt_gate_api={} parity_receipt_required_evidence={} validation_receipt_required=true validation_receipt_available={} validation_receipt_status_code={} validation_receipt_reason_code={} validation_receipt_schema={} validation_receipt_schema_version={} validation_receipt_gate_api={} validation_receipt_required_evidence={} callable_receipt_required=true callable_receipt_available={} callable_receipt_status_code={} callable_receipt_reason_code={} callable_receipt_schema={} callable_receipt_schema_version={} callable_receipt_gate_api={} callable_receipt_required_evidence={} native_runtime_callable_impl_available={} runtime_readiness_status_code={} runtime_readiness_reason_code={} production_gate={} production_gate_status={} production_gate_required_receipts={} production_selected={} fail_closed={} reason_code={}",
        TRUST_CG_PETRI_NATIVE_FEATURE,
        gate.feature_enabled,
        ENABLE_NATIVE_CANDIDATE_ENV,
        gate.native_requested,
        ENABLE_NATIVE_CANDIDATE_STRICT_ENV,
        gate.strict_requested,
        ENABLE_TRANSITION_PARITY_ENV,
        gate.parity_enabled,
        gate.parity_receipt_available,
        gate.parity_receipt_status_code(),
        gate.parity_receipt_reason_code(),
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION,
        PETRI_NATIVE_PARITY_RECEIPT_GATE_API,
        PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
        gate.validation_receipt_available,
        gate.validation_receipt_status_code(),
        gate.validation_receipt_reason_code(),
        VALIDATION_RECEIPT_SCHEMA,
        VALIDATION_RECEIPT_SCHEMA_VERSION,
        PETRI_NATIVE_VALIDATION_RECEIPT_GATE_API,
        PETRI_NATIVE_VALIDATION_RECEIPT_REQUIRED_EVIDENCE,
        route_selection.callable_receipt_available,
        route_selection.callable_receipt_status_code(),
        route_selection.callable_receipt_reason_code,
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA,
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA_VERSION,
        PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API,
        PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE,
        route_selection.native_runtime_callable_impl_available,
        route_selection.runtime_readiness_status_code(),
        route_selection.runtime_readiness_reason_code,
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE,
        route_selection.production_gate_status(),
        PETRI_NATIVE_ROUTE_PRODUCTION_GATE_REQUIRED_RECEIPTS,
        route_selection.selected_for_native_execution,
        route_selection.fail_closed,
        route_selection.reason_code,
    ));
}

fn add_native_verification_bundle_evidence(
    report: &mut CapabilityReport,
    evidence: PetriTrustIrTransportIdentityEvidence<'_>,
    state_bytes: u64,
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    gate: NativeJitFailClosedGate,
) -> PetriNativeRouteDecision {
    #[cfg(not(feature = "trust-cg-petri-native"))]
    let _ = (net, cache);

    #[cfg(feature = "trust-cg-petri-native")]
    {
        add_trust_cg_compile_artifact_cache_telemetry_evidence(report);
        add_trust_cg_host_jit_pgo_provenance_evidence(report);
        add_trust_ir_native_verification_bundle_handoff_replay_evidence(report);
    }

    match evidence {
        PetriTrustIrTransportIdentityEvidence::Auto(_)
        | PetriTrustIrTransportIdentityEvidence::DependencyUnavailable(_) => {
            add_trust_ir_transport_identity_unavailable_evidence(
                report,
                false,
                None,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_DEPENDENCY_BLOCKER,
            );
            add_trust_cg_native_admission_blocker_for_missing_transport(
                report,
                false,
                TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_DEPENDENCY_BLOCKER,
            );
            add_trust_cg_native_execution_plan_blocker_for_missing_transport(
                report,
                false,
                TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_DEPENDENCY_BLOCKER,
                state_bytes,
            );
            let route_selection =
                PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
                    transport_identity_available: false,
                    producer_admission: false,
                    producer_execution_authority: false,
                    producer_production_selection: false,
                    parity_enabled: gate.parity_enabled,
                    parity_receipt_available: gate.parity_receipt_available,
                    validation_receipt_available: gate.validation_receipt_available,
                    callable_receipt_available: false,
                    native_runtime_callable_impl_available:
                        PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE,
                    producer_admission_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                    producer_execution_authority_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                    producer_production_selection_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                    parity_receipt_reason_code: gate.parity_receipt_reason_code(),
                    validation_receipt_reason_code: gate.validation_receipt_reason_code(),
                    callable_receipt_reason_code:
                        PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
                    runtime_readiness_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                });
            add_petri_native_route_selection_evidence(report, &route_selection);
            PetriNativeRouteDecision {
                route_selection,
                gate,
            }
        }
        #[cfg(feature = "trust-cg-petri-native")]
        PetriTrustIrTransportIdentityEvidence::BundleUnavailable => {
            add_trust_ir_transport_identity_unavailable_evidence(
                report,
                true,
                None,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_ABSENT_BLOCKER,
            );
            add_trust_cg_native_admission_blocker_for_missing_transport(
                report,
                true,
                TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_ABSENT_BLOCKER,
            );
            add_trust_cg_native_execution_plan_blocker_for_missing_transport(
                report,
                true,
                TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_ABSENT_BLOCKER,
                state_bytes,
            );
            let route_selection =
                PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
                    transport_identity_available: false,
                    producer_admission: false,
                    producer_execution_authority: false,
                    producer_production_selection: false,
                    parity_enabled: gate.parity_enabled,
                    parity_receipt_available: gate.parity_receipt_available,
                    validation_receipt_available: gate.validation_receipt_available,
                    callable_receipt_available: false,
                    native_runtime_callable_impl_available:
                        PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE,
                    producer_admission_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                    producer_execution_authority_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                    producer_production_selection_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                    parity_receipt_reason_code: gate.parity_receipt_reason_code(),
                    validation_receipt_reason_code: gate.validation_receipt_reason_code(),
                    callable_receipt_reason_code:
                        PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
                    runtime_readiness_reason_code:
                        TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
                });
            add_petri_native_route_selection_evidence(report, &route_selection);
            PetriNativeRouteDecision {
                route_selection,
                gate,
            }
        }
        #[cfg(feature = "trust-cg-petri-native")]
        PetriTrustIrTransportIdentityEvidence::BundleProductionBlocked(blocker) => {
            report.add_evidence(format!(
                "Petri native_jit trust_ir_transport_identity unavailable required_trust_ir_rev={} current_trust_ir_rev={} cargo_dependency=true api=NativeVerificationBundle::transport_identity reason_code={} bundle_production_path={} missing_api=\"{}\" blocker=\"{}\" upstream_ask=\"{}\" expected_fields={} production_selected=false fail_closed=true",
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_REQUIRED_REV,
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV,
                blocker.reason_code,
                blocker.production_path,
                blocker.missing_api,
                blocker.detail,
                blocker.upstream_ask,
                TRUST_IR_NATIVE_VERIFICATION_EXPECTED_FIELDS,
            ));
            add_trust_ir_transport_identity_producer_contract_evidence(
                report,
                PetriTrustIrTransportIdentityProducerContractEvidence {
                    cargo_dependency: true,
                    status_code: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_STATUS_BLOCKED,
                    reason_code: blocker.reason_code,
                    bundle_source: blocker.production_path,
                    bundle_validated: false,
                    producer: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_NONE,
                    input: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_INPUT_NONE,
                    transport_identity_available: false,
                    module_digest: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_DIGEST_NONE,
                    transport_digest: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_DIGEST_NONE,
                    blocker: blocker.detail,
                },
            );
            add_trust_cg_native_admission_blocker_for_missing_transport(
                report,
                true,
                blocker.reason_code,
                blocker.detail,
            );
            add_trust_cg_native_execution_plan_blocker_for_missing_transport(
                report,
                true,
                blocker.reason_code,
                blocker.detail,
                state_bytes,
            );
            let route_selection =
                PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
                    transport_identity_available: false,
                    producer_admission: false,
                    producer_execution_authority: false,
                    producer_production_selection: false,
                    parity_enabled: gate.parity_enabled,
                    parity_receipt_available: gate.parity_receipt_available,
                    validation_receipt_available: gate.validation_receipt_available,
                    callable_receipt_available: false,
                    native_runtime_callable_impl_available:
                        PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE,
                    producer_admission_reason_code: blocker.reason_code,
                    producer_execution_authority_reason_code: blocker.reason_code,
                    producer_production_selection_reason_code: blocker.reason_code,
                    parity_receipt_reason_code: gate.parity_receipt_reason_code(),
                    validation_receipt_reason_code: gate.validation_receipt_reason_code(),
                    callable_receipt_reason_code:
                        PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
                    runtime_readiness_reason_code: blocker.reason_code,
                });
            add_petri_native_route_selection_evidence(report, &route_selection);
            PetriNativeRouteDecision {
                route_selection,
                gate,
            }
        }
        #[cfg(feature = "trust-cg-petri-native")]
        PetriTrustIrTransportIdentityEvidence::Bundle(bundle) => {
            add_trust_ir_transport_identity_available_evidence(
                report,
                bundle,
                "external_supplied",
                false,
            );
            add_trust_cg_native_admission_blocker_for_bundle(
                report,
                bundle,
                "external_supplied",
                false,
            );
            let installed_artifact = PetriNativeInstalledArtifactEvidence::NotAttempted;
            let gate = gate.with_receipt_evidence(native_jit_receipt_evidence_for_bundle(
                bundle,
                bundle,
                installed_artifact.as_ref(),
                net,
                cache,
            ));
            let route_selection = add_trust_cg_native_execution_plan_blocker_for_bundle(
                report,
                bundle,
                bundle,
                "external_supplied",
                false,
                state_bytes,
                cache,
                installed_artifact.as_ref(),
                gate,
            );
            add_petri_native_route_selection_evidence(report, &route_selection);
            PetriNativeRouteDecision {
                route_selection,
                gate,
            }
        }
        #[cfg(feature = "trust-cg-petri-native")]
        PetriTrustIrTransportIdentityEvidence::ProducedBundle {
            bundle,
            installed_artifact,
        } => {
            let semantic_bundle = bundle;
            let receipt_bundle = petri_native_successor_runtime_native_evidence_bundle(
                semantic_bundle.clone(),
                installed_artifact.as_ref(),
            );
            let receipt_bundle =
                petri_native_successor_validation_only_native_evidence_bundle(receipt_bundle);
            add_trust_ir_transport_identity_available_evidence(
                report,
                &receipt_bundle,
                "petri_native_production_path",
                true,
            );
            add_trust_cg_native_admission_blocker_for_bundle(
                report,
                &receipt_bundle,
                "petri_native_production_path",
                true,
            );
            let gate = gate.with_receipt_evidence(native_jit_receipt_evidence_for_bundle(
                &receipt_bundle,
                &semantic_bundle,
                installed_artifact.as_ref(),
                net,
                cache,
            ));
            let route_selection = add_trust_cg_native_execution_plan_blocker_for_bundle(
                report,
                &semantic_bundle,
                &semantic_bundle,
                "petri_native_production_path",
                true,
                state_bytes,
                cache,
                installed_artifact.as_ref(),
                gate,
            );
            add_petri_native_route_selection_evidence(report, &route_selection);
            PetriNativeRouteDecision {
                route_selection,
                gate,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PetriTrustIrTransportIdentityProducerContractEvidence<'a> {
    cargo_dependency: bool,
    status_code: &'a str,
    reason_code: &'a str,
    bundle_source: &'a str,
    bundle_validated: bool,
    producer: &'a str,
    input: &'a str,
    transport_identity_available: bool,
    module_digest: &'a str,
    transport_digest: &'a str,
    blocker: &'a str,
}

fn add_trust_ir_transport_identity_producer_contract_evidence(
    report: &mut CapabilityReport,
    evidence: PetriTrustIrTransportIdentityProducerContractEvidence<'_>,
) {
    report.add_evidence(format!(
        "Petri native_jit trust_ir_transport_identity_producer_contract source={} schema={} schema_version={} producer_package=tla-petri producer_api={} consumer=trust-cg consumer_api={} consumer_contract=NativeInstallGateAdmissionSummary requested_authority=validation_only install_authority=none required_output={} transport_identity_api=NativeVerificationBundle::transport_identity validation_api=NativeVerificationBundle::validate transport_identity_schema={} transport_identity_schema_version={} required_trust_ir_rev={} current_trust_ir_rev={} cargo_dependency={} status_code={} reason_code={} bundle_source={} bundle_validated={} producer={} input={} transport_identity_available={} module_digest={} transport_digest={} expected_fields={} native_promotion_authorized=false production_selected=false fail_closed=true blocker=\"{}\"",
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SOURCE,
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SCHEMA,
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SCHEMA_VERSION,
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_API,
        TRUST_CG_PETRI_NATIVE_ADMISSION_API,
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_REQUIRED_OUTPUT,
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA,
        TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA_VERSION,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_REQUIRED_REV,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV,
        evidence.cargo_dependency,
        evidence.status_code,
        evidence.reason_code,
        evidence.bundle_source,
        evidence.bundle_validated,
        evidence.producer,
        evidence.input,
        evidence.transport_identity_available,
        evidence.module_digest,
        evidence.transport_digest,
        TRUST_IR_NATIVE_VERIFICATION_EXPECTED_FIELDS,
        evidence.blocker,
    ));
}

fn add_trust_ir_transport_identity_unavailable_evidence(
    report: &mut CapabilityReport,
    cargo_dependency: bool,
    reason_code: Option<&'static str>,
    blocker: &'static str,
) {
    let producer_reason_code =
        reason_code.unwrap_or(TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON);
    let reason_code = reason_code
        .map(|code| format!(" reason_code={code}"))
        .unwrap_or_default();
    report.add_evidence(format!(
        "Petri native_jit trust_ir_transport_identity unavailable required_trust_ir_rev={} current_trust_ir_rev={} cargo_dependency={} api=NativeVerificationBundle::transport_identity{} blocker=\"{}\" expected_fields={} production_selected=false fail_closed=true",
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_REQUIRED_REV,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV,
        cargo_dependency,
        reason_code,
        blocker,
        TRUST_IR_NATIVE_VERIFICATION_EXPECTED_FIELDS,
    ));
    add_trust_ir_transport_identity_producer_contract_evidence(
        report,
        PetriTrustIrTransportIdentityProducerContractEvidence {
            cargo_dependency,
            status_code: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_STATUS_BLOCKED,
            reason_code: producer_reason_code,
            bundle_source: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_BUNDLE_SOURCE_NONE,
            bundle_validated: false,
            producer: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_NONE,
            input: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_INPUT_NONE,
            transport_identity_available: false,
            module_digest: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_DIGEST_NONE,
            transport_digest: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_DIGEST_NONE,
            blocker,
        },
    );
}

fn add_trust_cg_native_admission_blocker_for_missing_transport(
    report: &mut CapabilityReport,
    cargo_dependency: bool,
    reason_code: &str,
    blocker: &str,
) {
    report.add_evidence(format!(
        "trust-cg trust_cg_admission_blocker source=NativeInstallGateAdmissionSummary source_package=trust-cg-codegen package=trust-cg-codegen schema=trust-cg.phase6.native_install_gate.admission_summary.v1 schema_version=1 consumer=mcc consumer_mode=petri_successor kind=petri_native_successor surface=mcc_replay disposition=rejected status_code=rejected rejection_code={} reason_code={} requested_authority=active_callable install_authority=none cargo_dependency={} trust_ir_transport_identity_available=false trust_ir_bundle_consumed=false admission_api={} blocker=\"{}\" production_selected=false fail_closed=true",
        reason_code,
        reason_code,
        cargo_dependency,
        TRUST_CG_PETRI_NATIVE_ADMISSION_API,
        blocker,
    ));
}

fn add_trust_cg_native_execution_plan_blocker_for_missing_transport(
    report: &mut CapabilityReport,
    cargo_dependency: bool,
    reason_code: &str,
    blocker: &str,
    state_bytes: u64,
) {
    let callable_handoff_available = cargo_dependency;
    let callable_handoff_reason_code = if callable_handoff_available {
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    } else {
        TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON
    };
    let callable_handoff_blocker = if callable_handoff_available {
        TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_BLOCKER
    } else {
        TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON
    };
    let callable_pointer_reason_code = if cargo_dependency {
        TRUST_CG_PETRI_NATIVE_MISSING_CONCRETE_CALLABLE_POINTER_REASON
    } else {
        TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON
    };
    let call_packet_api_status = if cargo_dependency {
        TrustCgPetriNativeReadinessStatus::Available
    } else {
        TrustCgPetriNativeReadinessStatus::Unavailable
    };
    let execution_plan_status = TrustCgPetriNativeReadinessStatus::Missing;
    let install_packet_status = TrustCgPetriNativeReadinessStatus::Missing;
    let concrete_callable_pointer_status = TrustCgPetriNativeReadinessStatus::Missing;
    let concrete_callable_packet_status = TrustCgPetriNativeReadinessStatus::Missing;
    let runtime_readiness_status_code = if cargo_dependency {
        "blocked"
    } else {
        TrustCgPetriNativeReadinessStatus::Unavailable.code()
    };
    #[cfg(feature = "trust-cg-petri-native")]
    let call_packet_surface = trust_cg_petri_call_packet_surface();
    #[cfg(feature = "trust-cg-petri-native")]
    let (
        call_packet_descriptor_available,
        call_packet_descriptor_source,
        call_packet_descriptor_status_code,
        call_packet_descriptor_authoritative,
        call_packet_descriptor_upstream_ask,
    ) = (
        call_packet_surface.descriptor_available,
        call_packet_surface.descriptor_source,
        call_packet_surface.descriptor_status_code,
        call_packet_surface.descriptor_authoritative,
        call_packet_surface.descriptor_upstream_ask,
    );
    #[cfg(not(feature = "trust-cg-petri-native"))]
    let (
        call_packet_descriptor_available,
        call_packet_descriptor_source,
        call_packet_descriptor_status_code,
        call_packet_descriptor_authoritative,
        call_packet_descriptor_upstream_ask,
    ) = (
        false,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_DEPENDENCY,
        TrustCgPetriNativeReadinessStatus::Unavailable.code(),
        false,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_UPSTREAM_ASK,
    );
    report.add_evidence(format!(
        "trust-cg petri_native_successor_execution_plan source=MissingNativeVerificationBundle schema={} schema_version={} consumer=mcc kind={} surface={} disposition=rejected status_code=rejected rejection_code={} reason_code={} cargo_dependency={} trust_ir_transport_identity_available=false trust_ir_bundle_consumed=false execution_plan_api={} expected_api={} trampoline_contract_api={} install_packet_api={} call_packet_api={} call_packet_schema={} call_packet_schema_version={} call_packet_type={} callable_pointer_type={} call_packet_required_trust_cg_rev={} call_packet_current_trust_cg_rev={} call_packet_descriptor_available={} call_packet_descriptor_source={} call_packet_descriptor_status_code={} call_packet_descriptor_authoritative={} call_packet_descriptor_dependency={} call_packet_descriptor_upstream_ask={} runtime_readiness_api={} runtime_readiness_schema={} runtime_readiness_schema_version={} runtime_readiness_packet_type={} runtime_readiness_required_trust_cg_rev={} runtime_readiness_current_trust_cg_rev={} runtime_readiness_packet_available=false runtime_readiness_status_code={} runtime_readiness_ready_for_runtime_call=false runtime_readiness_reason_code={} runtime_readiness_blocker_stage=trust_ir_transport_identity runtime_readiness_required_evidence=NativeVerificationBundle runtime_readiness_packet_sha256=none mock_executable_call_api={} mock_executable_call_schema={} mock_executable_call_schema_version={} mock_executable_call_role={} mock_executable_call_production_enabled=false call_packet_api_available={} call_packet_api_status_code={} call_packet_type_available={} callable_pointer_type_available={} entry_function={} input_state_bytes={} output_state_bytes={} state_alignment_bytes={} execution_plan_available=false execution_plan_status_code={} execution_plan_reason_code={} callable_contract_available=false trampoline_contract_available=false install_packet_available=false install_packet_status_code={} install_packet_reason_code={} call_packet_available=false call_packet_reason_code={} callable_pointer_available=false callable_pointer_reason_code={} concrete_callable_pointer_required=true concrete_callable_pointer_available=false concrete_callable_pointer_status_code={} concrete_callable_packet_required=true concrete_callable_packet_available=false concrete_callable_packet_status_code={} call_packet_readiness_status_code={} call_packet_readiness_blocker={} callable_authorized=false callable_authorized_reason_code={} callable_handoff_available={} callable_handoff_reason_code={} callable_handoff_blocker={} callable_handoff_upstream_ask={} native_successor_runtime_status_code={} production_selected=false fail_closed=true blocker=\"{}\"",
        TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_SCHEMA,
        TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_SCHEMA_VERSION,
        TRUST_CG_PETRI_NATIVE_ADMISSION_KIND,
        TRUST_CG_PETRI_NATIVE_ADMISSION_SURFACE,
        reason_code,
        reason_code,
        cargo_dependency,
        TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_API,
        TRUST_CG_PETRI_NATIVE_EXECUTION_EXPECTED_API,
        TRUST_CG_PETRI_NATIVE_TRAMPOLINE_CONTRACT_API,
        TRUST_CG_PETRI_NATIVE_INSTALL_PACKET_API,
        TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_API,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_SCHEMA,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_SCHEMA_VERSION,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_TYPE,
        TRUST_CG_PETRI_NATIVE_CALLABLE_POINTER_TYPE,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_REQUIRED_TRUST_CG_REV,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV,
        call_packet_descriptor_available,
        call_packet_descriptor_source,
        call_packet_descriptor_status_code,
        call_packet_descriptor_authoritative,
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_DESCRIPTOR_DEPENDENCY,
        call_packet_descriptor_upstream_ask,
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_API,
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_SCHEMA,
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_SCHEMA_VERSION,
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_TYPE,
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_REQUIRED_TRUST_CG_REV,
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_CURRENT_TRUST_CG_REV,
        runtime_readiness_status_code,
        reason_code,
        TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_API,
        TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_SCHEMA,
        TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_SCHEMA_VERSION,
        TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_ROLE,
        cargo_dependency,
        call_packet_api_status.code(),
        cargo_dependency,
        cargo_dependency,
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
        state_bytes,
        state_bytes,
        TRUST_CG_PETRI_NATIVE_EXECUTION_STATE_ALIGNMENT_BYTES,
        execution_plan_status.code(),
        reason_code,
        install_packet_status.code(),
        reason_code,
        reason_code,
        callable_pointer_reason_code,
        concrete_callable_pointer_status.code(),
        concrete_callable_packet_status.code(),
        runtime_readiness_status_code,
        callable_handoff_blocker,
        reason_code,
        callable_handoff_available,
        callable_handoff_reason_code,
        callable_handoff_blocker,
        TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_UPSTREAM_ASK,
        runtime_readiness_status_code,
        blocker,
    ));
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_ir_transport_identity_available_evidence(
    report: &mut CapabilityReport,
    bundle: &trust_ir::NativeVerificationBundle,
    bundle_source: &'static str,
    bundle_validated: bool,
) {
    let identity = bundle.transport_identity();
    let transport_digest = identity.stable_digest();
    let source_digest = identity
        .source_digest
        .map_or_else(|| "none".to_owned(), |digest| digest.to_string());
    let target_abi_digest = identity.target_abi.as_ref().map_or_else(
        || "none".to_owned(),
        |target_abi| target_abi.digest.to_string(),
    );
    let producer = native_bundle_producer_code(identity.producer);
    let input = native_adapter_input_code(identity.input);

    report.add_evidence(format!(
        "Petri native_jit trust_ir_transport_identity available required_trust_ir_rev={} current_trust_ir_rev={} cargo_dependency=true api=NativeVerificationBundle::transport_identity validation_api=NativeVerificationBundle::validate bundle_source={} bundle_validated={} producer={} input={} schema={} schema_version={} bundle_schema_version={} transport_digest={} source_digest={} module_digest={} compiler_facts_digest={} lineage_digest={} bundle_digest={} target_abi_digest={} request_digests={} evidence_digests={} expected_fields={} production_selected=false fail_closed=true",
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_REQUIRED_REV,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV,
        bundle_source,
        bundle_validated,
        producer,
        input,
        identity.schema,
        identity.schema_version,
        identity.bundle_schema_version,
        transport_digest,
        source_digest,
        identity.trust_ir_module_digest,
        identity.compiler_facts_digest,
        identity.lineage_digest,
        identity.bundle_digest,
        target_abi_digest,
        identity.request_digests.len(),
        identity.evidence_digests.len(),
        TRUST_IR_NATIVE_VERIFICATION_EXPECTED_FIELDS,
    ));
    let module_digest = identity.trust_ir_module_digest.to_string();
    let transport_digest_string = transport_digest.to_string();
    add_trust_ir_transport_identity_producer_contract_evidence(
        report,
        PetriTrustIrTransportIdentityProducerContractEvidence {
            cargo_dependency: true,
            status_code: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_STATUS_AVAILABLE,
            reason_code: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_STATUS_AVAILABLE,
            bundle_source,
            bundle_validated,
            producer,
            input,
            transport_identity_available: true,
            module_digest: module_digest.as_str(),
            transport_digest: transport_digest_string.as_str(),
            blocker: TRUST_IR_NATIVE_TRANSPORT_IDENTITY_BLOCKER_NONE,
        },
    );
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_ir_native_verification_bundle_handoff_replay_evidence(report: &mut CapabilityReport) {
    let descriptor = trust_ir::petri_native_verification_bundle_handoff_descriptor();
    let manifest_identity = descriptor.manifest_identity();
    let contract_health = descriptor.contract_health_report();
    let diagnostic_manifest =
        trust_ir::petri_native_verification_bundle_handoff_diagnostic_fixture_manifest();
    let diagnostic_manifest_rows = diagnostic_manifest.key_value_rows();
    let diagnostic_round_trip = diagnostic_manifest.round_trip_report(&diagnostic_manifest_rows);
    let surface = trust_ir::petri_native_verification_bundle_handoff_replay_contract_surface();
    let surface_rows = surface.key_value_rows();
    let surface_round_trip = surface.round_trip_report(&surface_rows);
    let json_binding =
        surface_round_trip.compact_manifest_handoff_identity_report(&manifest_identity);

    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_TY_MCC_SHARED_PRIMITIVE_MANIFEST_COMPONENT,
        trust_ir_common_row_fields(
            trust_ir::TY_SHARED_PRIMITIVE_MANIFEST_SCHEMA,
            trust_ir::TY_SHARED_PRIMITIVE_MANIFEST_SCHEMA_VERSION,
        ),
        trust_ir::ty_shared_primitive_manifest_key_value_lines(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_HARDWARE_VECTOR_CONTRACT_SET_COMPONENT,
        trust_ir_common_row_fields(
            trust_ir::HARDWARE_VECTOR_CONTRACT_MANIFEST_SCHEMA,
            trust_ir::HARDWARE_VECTOR_CONTRACT_MANIFEST_SCHEMA_VERSION,
        ),
        trust_ir::chc_x86_hardware_vector_contract_manifest_key_value_lines(),
    );

    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_COMPONENT,
        trust_ir_common_row_fields(
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA,
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION,
        ),
        descriptor.manifest_key_value_lines(),
    );
    add_trust_ir_handoff_completeness_row(report, &descriptor);

    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_IDENTITY_COMPONENT,
        trust_ir_manifest_identity_row_fields(&descriptor, &manifest_identity),
        trust_ir_manifest_identity_key_value_lines(&descriptor, &manifest_identity),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_CONTRACT_HEALTH_COMPONENT,
        trust_ir_contract_health_row_fields(&descriptor, &manifest_identity),
        contract_health.key_value_lines(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_COMPONENT,
        trust_ir_diagnostic_fixture_manifest_row_fields(&descriptor, &manifest_identity),
        diagnostic_manifest.key_value_lines(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_ROUND_TRIP_COMPONENT,
        trust_ir_diagnostic_fixture_round_trip_row_fields(),
        trust_ir_diagnostic_fixture_round_trip_key_value_lines(&diagnostic_round_trip),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_COMPONENT,
        trust_ir_replay_contract_surface_row_fields(&descriptor),
        surface.key_value_lines(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_ROUND_TRIP_COMPONENT,
        trust_ir_replay_contract_surface_round_trip_row_fields(),
        trust_ir_replay_contract_surface_round_trip_key_value_lines(&surface_round_trip),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_REPORT_IDENTITY_COMPONENT,
        trust_ir_replay_contract_report_identity_row_fields(),
        surface_round_trip.key_value_lines(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_JSON_MANIFEST_BINDING_COMPONENT,
        trust_ir_replay_contract_json_binding_row_fields(&manifest_identity),
        json_binding.key_value_lines(),
    );
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_common_row_fields(
    schema: &'static str,
    schema_version: u32,
) -> Vec<(&'static str, String)> {
    vec![
        ("schema", schema.to_owned()),
        ("schema_version", schema_version.to_string()),
        (
            "source_package",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SOURCE_PACKAGE.to_owned(),
        ),
        ("source_project", TRUST_IR_PROJECT_CODE.to_owned()),
        ("project", TRUST_IR_PROJECT_CODE.to_owned()),
    ]
}

#[cfg(feature = "trust-cg-petri-native")]
// The bool flags map 1:1 onto distinct manifest rows; packing them into a
// struct would change every call site (some in other modules) without making
// this internal row-builder clearer.
#[allow(clippy::fn_params_excessive_bools)]
fn trust_ir_component_identity_row_fields(
    schema: &'static str,
    schema_version: u32,
    producer_api: &'static str,
    replay_api: &'static str,
    identity_digest: impl ToString,
    identity_row_count: usize,
    identity_text_available: bool,
    identity_replay_status_code: &'static str,
    identity_replayable: bool,
    identity_replay_fail_closed: bool,
    identity_replay_diagnostic_count: usize,
    identity_replay_component_health_api: &'static str,
    identity_replay_component_health_row_count: usize,
    identity_replay_component_health_text_available: bool,
    producer_status_code: &'static str,
    producer_reason_code: &'static str,
    producer_fail_closed: bool,
) -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(schema, schema_version);
    fields.extend([
        ("identity_schema", schema.to_owned()),
        ("identity_schema_version", schema_version.to_string()),
        ("producer_api", producer_api.to_owned()),
        ("replay_api", replay_api.to_owned()),
        ("identity_digest", identity_digest.to_string()),
        ("identity_row_count", identity_row_count.to_string()),
        (
            "identity_text_available",
            identity_text_available.to_string(),
        ),
        (
            "identity_replay_status_code",
            identity_replay_status_code.to_owned(),
        ),
        ("identity_replayable", identity_replayable.to_string()),
        (
            "identity_replay_fail_closed",
            identity_replay_fail_closed.to_string(),
        ),
        (
            "identity_replay_diagnostic_count",
            identity_replay_diagnostic_count.to_string(),
        ),
        (
            "identity_replay_component_health_api",
            identity_replay_component_health_api.to_owned(),
        ),
        (
            "identity_replay_component_health_row_count",
            identity_replay_component_health_row_count.to_string(),
        ),
        (
            "identity_replay_component_health_text_available",
            identity_replay_component_health_text_available.to_string(),
        ),
        ("producer_status_code", producer_status_code.to_owned()),
        ("producer_reason_code", producer_reason_code.to_owned()),
        ("producer_fail_closed", producer_fail_closed.to_string()),
        ("production_selected", "false".to_owned()),
        ("fail_closed", "true".to_owned()),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_manifest_identity_row_fields(
    _descriptor: &trust_ir::PetriNativeVerificationBundleHandoffDescriptor,
    manifest_identity: &trust_ir::PetriNativeVerificationBundleHandoffManifestIdentity,
) -> Vec<(&'static str, String)> {
    let mut fields =
        trust_ir_common_row_fields(manifest_identity.schema, manifest_identity.schema_version);
    fields.extend([
        (
            "linked_handoff_schema",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA.to_owned(),
        ),
        (
            "linked_handoff_schema_version",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION.to_string(),
        ),
        (
            "linked_handoff_manifest_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_COMPONENT.to_owned(),
        ),
        (
            "linked_handoff_completeness_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_COMPLETENESS_COMPONENT.to_owned(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_contract_health_row_fields(
    descriptor: &trust_ir::PetriNativeVerificationBundleHandoffDescriptor,
    manifest_identity: &trust_ir::PetriNativeVerificationBundleHandoffManifestIdentity,
) -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(descriptor.schema, descriptor.schema_version);
    fields.extend([
        (
            "linked_handoff_schema",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA.to_owned(),
        ),
        (
            "linked_handoff_schema_version",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION.to_string(),
        ),
        (
            "linked_manifest_identity_schema",
            manifest_identity.schema.to_owned(),
        ),
        (
            "linked_manifest_identity_schema_version",
            manifest_identity.schema_version.to_string(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_diagnostic_fixture_manifest_row_fields(
    _descriptor: &trust_ir::PetriNativeVerificationBundleHandoffDescriptor,
    manifest_identity: &trust_ir::PetriNativeVerificationBundleHandoffManifestIdentity,
) -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA,
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA_VERSION,
    );
    fields.extend([
        (
            "linked_handoff_schema",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA.to_owned(),
        ),
        (
            "linked_handoff_schema_version",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION.to_string(),
        ),
        (
            "linked_manifest_identity_schema",
            manifest_identity.schema.to_owned(),
        ),
        (
            "linked_manifest_identity_schema_version",
            manifest_identity.schema_version.to_string(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_diagnostic_fixture_round_trip_row_fields() -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_ROUND_TRIP_SCHEMA,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_ROUND_TRIP_SCHEMA_VERSION,
    );
    fields.extend([
        (
            "linked_fixture_manifest_schema",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA
                .to_owned(),
        ),
        (
            "linked_fixture_manifest_schema_version",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA_VERSION
                .to_string(),
        ),
        (
            "linked_fixture_manifest_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_COMPONENT
                .to_owned(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_replay_contract_surface_row_fields(
    _descriptor: &trust_ir::PetriNativeVerificationBundleHandoffDescriptor,
) -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA,
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA_VERSION,
    );
    fields.extend([
        (
            "linked_handoff_schema",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA.to_owned(),
        ),
        (
            "linked_handoff_schema_version",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION.to_string(),
        ),
        (
            "linked_fixture_manifest_schema",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA
                .to_owned(),
        ),
        (
            "linked_fixture_manifest_schema_version",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA_VERSION
                .to_string(),
        ),
        (
            "linked_fixture_manifest_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_COMPONENT
                .to_owned(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_replay_contract_surface_round_trip_row_fields() -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_ROUND_TRIP_REPORT_SCHEMA,
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_ROUND_TRIP_REPORT_SCHEMA_VERSION,
    );
    fields.extend([
        (
            "linked_replay_contract_surface_schema",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA
                .to_owned(),
        ),
        (
            "linked_replay_contract_surface_schema_version",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA_VERSION
                .to_string(),
        ),
        (
            "linked_replay_contract_surface_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_COMPONENT.to_owned(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_replay_contract_report_identity_row_fields() -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_ROUND_TRIP_REPORT_SCHEMA,
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_ROUND_TRIP_REPORT_SCHEMA_VERSION,
    );
    fields.extend([
        (
            "linked_replay_contract_surface_schema",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA
                .to_owned(),
        ),
        (
            "linked_replay_contract_surface_schema_version",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA_VERSION
                .to_string(),
        ),
        (
            "linked_replay_contract_surface_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_COMPONENT.to_owned(),
        ),
        (
            "linked_replay_contract_surface_round_trip_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_ROUND_TRIP_COMPONENT
                .to_owned(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_replay_contract_json_binding_row_fields(
    manifest_identity: &trust_ir::PetriNativeVerificationBundleHandoffManifestIdentity,
) -> Vec<(&'static str, String)> {
    let mut fields = trust_ir_common_row_fields(
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_JSON_MANIFEST_BINDING_SCHEMA,
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_JSON_MANIFEST_BINDING_SCHEMA_VERSION,
    );
    fields.extend([
        (
            "linked_replay_contract_surface_schema",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA
                .to_owned(),
        ),
        (
            "linked_replay_contract_surface_schema_version",
            trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA_VERSION
                .to_string(),
        ),
        (
            "linked_replay_contract_report_identity_component",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_REPORT_IDENTITY_COMPONENT
                .to_owned(),
        ),
        (
            "linked_manifest_identity_schema",
            manifest_identity.schema.to_owned(),
        ),
        (
            "linked_manifest_identity_schema_version",
            manifest_identity.schema_version.to_string(),
        ),
    ]);
    fields
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_ir_component_manifest_lines(
    report: &mut CapabilityReport,
    component: &'static str,
    fields: Vec<(&'static str, String)>,
    lines: Vec<String>,
) {
    let mut prefix = format!("trust-ir {component}");
    for (key, value) in fields {
        prefix.push(' ');
        prefix.push_str(key);
        prefix.push('=');
        prefix.push_str(&value);
    }
    for line in lines {
        let mut row = prefix.clone();
        if let Some((key, value)) = line.split_once('=') {
            row.push_str(" row_key=");
            row.push_str(key);
            row.push_str(" row_value=");
            row.push_str(value);
        }
        row.push_str(" manifest_line=");
        row.push_str(&line);
        report.add_evidence(row);
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_ir_component_readiness_row(
    report: &mut CapabilityReport,
    component: &'static str,
    fields: Vec<(&'static str, String)>,
) {
    let mut row = format!("trust-ir component_readiness component={component}");
    for (key, value) in fields {
        row.push(' ');
        row.push_str(key);
        row.push('=');
        row.push_str(&value);
    }
    report.add_evidence(row);
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_ir_handoff_completeness_row(
    report: &mut CapabilityReport,
    _descriptor: &trust_ir::PetriNativeVerificationBundleHandoffDescriptor,
) {
    report.add_evidence(format!(
        "trust-ir {} schema={} schema_version={} source_package={} source_project={} project={} manifest_schema={} manifest_schema_version={} bundle_identity_status=complete artifact_authority_status=complete ay_evidence_identity_status=complete downstream_responsibility_status=blocked handoff_complete=false status_code=blocked reason_code=missing_replay_transcript_artifact production_selected=false fail_closed=true",
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_COMPLETENESS_COMPONENT,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION,
        trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SOURCE_PACKAGE,
        TRUST_IR_PROJECT_CODE,
        TRUST_IR_PROJECT_CODE,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA,
        TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION,
    ));
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_cg_compile_artifact_cache_telemetry_evidence(report: &mut CapabilityReport) {
    let descriptor = tla_trust_cg::compile_artifact_cache_telemetry_descriptor();
    let mut descriptor_fields = Vec::new();
    push_trust_cg_native_admission_field(&mut descriptor_fields, "schema", descriptor.schema);
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "schema_version",
        descriptor.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "required_fields",
        join_ay_strs(descriptor.required_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "optional_fields",
        join_ay_strs(descriptor.optional_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "boundary_codes",
        join_ay_strs(descriptor.boundary_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "status_codes",
        join_ay_strs(descriptor.status_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "metric_fields",
        join_ay_strs(descriptor.metric_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "authorizes_useful_native",
        descriptor.authorizes_useful_native,
    );
    push_trust_cg_native_admission_field(&mut descriptor_fields, "production_selected", false);
    push_trust_cg_native_admission_field(&mut descriptor_fields, "fail_closed", true);
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "source",
        "CompileArtifactCacheTelemetryDescriptor",
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "producer_api_status",
        "available_tla_trust_cg_reexport",
    );
    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-cg compile_artifact_cache_telemetry_descriptor",
        &descriptor_fields,
    ));

    let telemetry = tla_trust_cg::CompileArtifactCacheTelemetry {
        boundary: tla_trust_cg::CompileArtifactCacheBoundary::Service,
        status: tla_trust_cg::CompileArtifactCacheStatus::Miss,
        key_sha256: TRUST_CG_COMPILE_ARTIFACT_CACHE_TELEMETRY_PROBE_KEY_SHA256.to_owned(),
        artifact_sha256: None,
        cache_path: std::path::PathBuf::from("/tmp/ty-mcc-trust_cg-cache"),
        reason: Some("cache_probe_only".to_owned()),
        elapsed_micros: 0,
    };
    let mut telemetry_fields = Vec::new();
    push_trust_cg_native_admission_field(&mut telemetry_fields, "schema", descriptor.schema);
    push_trust_cg_native_admission_field(
        &mut telemetry_fields,
        "schema_version",
        descriptor.schema_version,
    );
    for row in telemetry.to_key_value_rows() {
        push_trust_cg_native_admission_field(&mut telemetry_fields, &row.key, row.value);
    }
    push_trust_cg_native_admission_field(&mut telemetry_fields, "artifact_sha256", "none");
    push_trust_cg_native_admission_field(
        &mut telemetry_fields,
        "source",
        "CompileArtifactCacheTelemetry",
    );
    push_trust_cg_native_admission_field(
        &mut telemetry_fields,
        "producer_api_status",
        "available_tla_trust_cg_reexport",
    );
    push_trust_cg_native_admission_field(&mut telemetry_fields, "authorizes_useful_native", false);
    push_trust_cg_native_admission_field(&mut telemetry_fields, "production_selected", false);
    push_trust_cg_native_admission_field(&mut telemetry_fields, "fail_closed", true);
    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-cg compile_artifact_cache_telemetry",
        &telemetry_fields,
    ));
}

#[cfg(feature = "trust-cg-petri-native")]
fn fail_closed_pgo_profile_use_status() -> tla_trust_cg::pgo::ProfileUseStatus {
    tla_trust_cg::pgo::ProfileUseStatus {
        key: tla_trust_cg::pgo::ProfileReportKey {
            profile_key_digest: "0000000000000000000000000000000000000000000000000000000000000000"
                .to_owned(),
            module_hash: "0000000000000000000000000000000000000000000000000000000000000000"
                .to_owned(),
            target_triple: "unknown-target".to_owned(),
            target_cpu: "unknown-cpu".to_owned(),
            target_features: Vec::new(),
            opt_level: "O0".to_owned(),
            opt_level_num: 0,
            cache_key_version: 1,
        },
        profile: tla_trust_cg::pgo::ProfileArtifactReport {
            path: None,
            sha256: None,
        },
        counters: tla_trust_cg::pgo::ProfileCounterSummary {
            function_count: 0,
            block_count: 0,
            edge_count: 0,
            total_call_count: 0,
            total_block_hits: 0,
            max_block_hits: 0,
        },
        consumer: "ty-mcc".to_owned(),
        scheduled: false,
        pass: None,
        reason: "profile_use_not_scheduled".to_owned(),
        summary: None,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_cg_host_jit_pgo_provenance_evidence(report: &mut CapabilityReport) {
    let descriptor = tla_trust_cg::pgo::trust_cg_host_jit_pgo_provenance_descriptor();
    let mut descriptor_fields = Vec::new();
    push_trust_cg_native_admission_field(&mut descriptor_fields, "schema", descriptor.schema);
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "schema_version",
        descriptor.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_report_schema",
        descriptor.profile_report_schema,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_key_fields",
        join_ay_strs(descriptor.profile_key_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "capture_fields",
        join_ay_strs(descriptor.capture_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_use_fields",
        join_ay_strs(descriptor.profile_use_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_use_soundness_fields",
        join_ay_strs(descriptor.profile_use_soundness_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_evidence_schema",
        descriptor.profile_authority_evidence_schema,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_manifest_schema",
        descriptor.profile_authority_manifest_schema,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_manifest_schema_version",
        descriptor.profile_authority_manifest_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_fields",
        join_ay_strs(descriptor.profile_authority_fields),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_manifest_row_keys",
        join_ay_strs(descriptor.profile_authority_manifest_row_keys),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_status_codes",
        join_ay_strs(descriptor.profile_authority_status_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_reason_codes",
        join_ay_strs(descriptor.profile_authority_reason_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "runner_error_reason_codes",
        join_ay_strs(descriptor.runner_error_reason_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "entry_shape_codes",
        join_ay_strs(descriptor.entry_shape_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_use_reason_codes",
        join_ay_strs(descriptor.profile_use_reason_codes),
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_use_pass_code",
        descriptor.profile_use_pass_code,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "soundness_helper",
        descriptor.soundness_helper,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_helper",
        descriptor.profile_authority_helper,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "profile_authority_manifest_helper",
        descriptor.profile_authority_manifest_helper,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "target_compatibility_helper",
        descriptor.target_compatibility_helper,
    );
    push_trust_cg_native_admission_field(
        &mut descriptor_fields,
        "authorizes_useful_native",
        descriptor.authorizes_useful_native,
    );
    push_trust_cg_native_admission_field(&mut descriptor_fields, "production_selected", false);
    push_trust_cg_native_admission_field(&mut descriptor_fields, "fail_closed", true);
    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-cg host_jit_pgo_provenance_descriptor",
        &descriptor_fields,
    ));

    let manifest_lines = fail_closed_pgo_profile_use_status()
        .trust_cg_profile_authority_manifest_lines()
        .expect("static fail-closed PGO profile authority manifest should be valid");
    for line in manifest_lines {
        let mut fields = Vec::new();
        push_trust_cg_native_admission_field(
            &mut fields,
            "schema",
            descriptor.profile_authority_manifest_schema,
        );
        push_trust_cg_native_admission_field(
            &mut fields,
            "schema_version",
            descriptor.profile_authority_manifest_schema_version,
        );
        if let Some((key, value)) = line.split_once('=') {
            push_trust_cg_native_admission_field(&mut fields, "row_key", key);
            push_trust_cg_native_admission_field(&mut fields, "row_value", value);
        }
        push_trust_cg_native_admission_field(&mut fields, "manifest_line", line);
        report.add_evidence(render_trust_cg_native_admission_row(
            "trust-cg host_jit_pgo_profile_authority_manifest",
            &fields,
        ));
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_manifest_identity_key_value_lines(
    _descriptor: &trust_ir::PetriNativeVerificationBundleHandoffDescriptor,
    manifest_identity: &trust_ir::PetriNativeVerificationBundleHandoffManifestIdentity,
) -> Vec<String> {
    let mut lines = manifest_identity.key_value_lines();
    lines.extend([
        format!(
            "manifest_identity.linked_handoff.schema={}",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA
        ),
        format!(
            "manifest_identity.linked_handoff.schema_version={}",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA_VERSION
        ),
        format!(
            "manifest_identity.linked_handoff.manifest_component={}",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_COMPONENT
        ),
        format!(
            "manifest_identity.linked_handoff.completeness_component={}",
            TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_COMPLETENESS_COMPONENT
        ),
    ]);
    lines
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_diagnostic_fixture_round_trip_key_value_lines(
    report: &trust_ir::PetriNativeVerificationBundleHandoffDiagnosticFixtureManifestRoundTripReport,
) -> Vec<String> {
    let mut lines = vec![
        format!("round_trip.status_code={}", report.status_code),
        format!("round_trip.fail_closed={}", report.fail_closed),
        format!(
            "round_trip.expected_row_count={}",
            report.expected_row_count
        ),
        format!(
            "round_trip.observed_row_count={}",
            report.observed_row_count
        ),
        format!("round_trip.unique_key_count={}", report.unique_key_count),
        format!(
            "round_trip.duplicate_key_count={}",
            report.duplicate_keys.len()
        ),
        format!("round_trip.missing_key_count={}", report.missing_keys.len()),
        format!(
            "round_trip.unexpected_key_count={}",
            report.unexpected_keys.len()
        ),
        format!(
            "round_trip.mismatched_value_key_count={}",
            report.mismatched_value_keys.len()
        ),
        format!(
            "round_trip.invalid_bool_key_count={}",
            report.invalid_bool_keys.len()
        ),
        format!(
            "round_trip.reconstructed_fixture_name_count={}",
            report.reconstructed_fixture_names.len()
        ),
        format!(
            "round_trip.reconstructed_completeness_status_count={}",
            report.reconstructed_completeness_status_codes.len()
        ),
        format!(
            "round_trip.reconstructed_manifest_identity_status_count={}",
            report.reconstructed_manifest_identity_status_codes.len()
        ),
        format!(
            "round_trip.reconstructed_contract_health_status_count={}",
            report.reconstructed_contract_health_status_codes.len()
        ),
        format!(
            "round_trip.reconstructed_accepted_value_count={}",
            report.reconstructed_accepted_values.len()
        ),
        format!(
            "round_trip.reconstructed_fail_closed_value_count={}",
            report.reconstructed_fail_closed_values.len()
        ),
    ];
    lines.extend(
        report
            .reconstructed_fixture_names
            .iter()
            .enumerate()
            .map(|(index, value)| format!("round_trip.reconstructed_fixture_name.{index}={value}")),
    );
    lines.extend(
        report
            .reconstructed_completeness_status_codes
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("round_trip.reconstructed_completeness_status_code.{index}={value}")
            }),
    );
    lines.extend(
        report
            .reconstructed_manifest_identity_status_codes
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("round_trip.reconstructed_manifest_identity_status_code.{index}={value}")
            }),
    );
    lines.extend(
        report
            .reconstructed_contract_health_status_codes
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("round_trip.reconstructed_contract_health_status_code.{index}={value}")
            }),
    );
    lines.extend(
        report
            .reconstructed_accepted_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("round_trip.reconstructed_accepted_value.{index}={value}")
            }),
    );
    lines.extend(
        report
            .reconstructed_fail_closed_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("round_trip.reconstructed_fail_closed_value.{index}={value}")
            }),
    );
    lines
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_replay_contract_surface_round_trip_key_value_lines(
    report: &trust_ir::PetriNativeVerificationBundleHandoffReplayContractSurfaceRoundTripReport,
) -> Vec<String> {
    let reconstructed_fixture_count = report
        .reconstructed_fixture_count
        .unwrap_or(report.reconstructed_fixture_names.len());
    let mut lines = vec![
        format!("round_trip.status_code={}", report.status_code),
        format!("round_trip.fail_closed={}", report.fail_closed),
        format!(
            "round_trip.expected_row_count={}",
            report.expected_row_count
        ),
        format!(
            "round_trip.observed_row_count={}",
            report.observed_row_count
        ),
        format!("round_trip.unique_key_count={}", report.unique_key_count),
        format!(
            "round_trip.duplicate_key_count={}",
            report.duplicate_keys.len()
        ),
        format!("round_trip.missing_key_count={}", report.missing_keys.len()),
        format!(
            "round_trip.unexpected_key_count={}",
            report.unexpected_keys.len()
        ),
        format!(
            "round_trip.mismatched_value_key_count={}",
            report.mismatched_value_keys.len()
        ),
        format!(
            "round_trip.invalid_usize_key_count={}",
            report.invalid_usize_keys.len()
        ),
        format!(
            "round_trip.invalid_line_count={}",
            report.invalid_lines.len()
        ),
        format!(
            "round_trip.reconstructed_schema={}",
            report.reconstructed_schema.as_deref().unwrap_or("")
        ),
        format!(
            "round_trip.reconstructed_schema_version={}",
            report
                .reconstructed_schema_version
                .map(|version| version.to_string())
                .unwrap_or_default()
        ),
        format!(
            "round_trip.reconstructed_helper_name_count={}",
            report.reconstructed_helper_names.len()
        ),
        format!(
            "round_trip.reconstructed_schema_name_count={}",
            report.reconstructed_schema_names.len()
        ),
        format!(
            "round_trip.reconstructed_schema_value_count={}",
            report.reconstructed_schema_values.len()
        ),
        format!(
            "round_trip.reconstructed_fixture_count={}",
            reconstructed_fixture_count
        ),
        format!(
            "round_trip.reconstructed_fixture_name_count={}",
            report.reconstructed_fixture_names.len()
        ),
        format!(
            "round_trip.reconstructed_validator_name_count={}",
            report.reconstructed_validator_names.len()
        ),
        format!(
            "round_trip.schema_header_matches={}",
            report.schema_header_matches
        ),
        format!(
            "round_trip.schema_name_value_rows_agree={}",
            report.schema_name_value_rows_agree
        ),
        format!(
            "round_trip.helper_names_match={}",
            report.helper_names_match
        ),
        format!(
            "round_trip.fixture_count_matches={}",
            report.fixture_count_matches
        ),
        format!(
            "round_trip.fixture_names_match={}",
            report.fixture_names_match
        ),
        format!(
            "round_trip.validator_names_match={}",
            report.validator_names_match
        ),
    ];
    lines.extend(
        report
            .reconstructed_helper_names
            .iter()
            .enumerate()
            .map(|(index, value)| format!("round_trip.reconstructed_helper_name.{index}={value}")),
    );
    lines.extend(
        report
            .reconstructed_schema_names
            .iter()
            .enumerate()
            .map(|(index, value)| format!("round_trip.reconstructed_schema_name.{index}={value}")),
    );
    lines.extend(
        report
            .reconstructed_schema_values
            .iter()
            .enumerate()
            .map(|(index, value)| format!("round_trip.reconstructed_schema_value.{index}={value}")),
    );
    lines.extend(
        report
            .reconstructed_fixture_names
            .iter()
            .enumerate()
            .map(|(index, value)| format!("round_trip.reconstructed_fixture_name.{index}={value}")),
    );
    lines.extend(
        report
            .reconstructed_validator_names
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("round_trip.reconstructed_validator_name.{index}={value}")
            }),
    );
    lines
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_proof_status_code(status: trust_ir::ProofStatus) -> &'static str {
    match status {
        trust_ir::ProofStatus::Pending => "pending",
        trust_ir::ProofStatus::Discharged => "discharged",
        trust_ir::ProofStatus::Failed => "failed",
        trust_ir::ProofStatus::Trusted => "trusted",
        // Kernel-checkable CIC proof term — strictly stronger than `Trusted`.
        trust_ir::ProofStatus::Certified => "certified",
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn petri_native_successor_semantic_bridge(
    bundle: &trust_ir::NativeVerificationBundle,
) -> trust_ir::request::NativeSemanticBridge {
    let semantic_function = bundle
        .module
        .functions
        .iter()
        .find(|function| function.name == PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL)
        .map(|function| function.id)
        .unwrap_or_else(|| trust_ir::FuncId::new(u32::MAX));

    trust_ir::request::NativeSemanticBridge::petri_successor_plan_cache_equivalence(
        semantic_function,
    )
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone, Copy)]
struct PetriNativeSemanticBridgeAuthority {
    represented: bool,
    api: &'static str,
    formula_schema: &'static str,
    required_evidence: &'static str,
    status_code: &'static str,
    reason_code: &'static str,
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_petri_native_successor_semantic_bridge_evidence(
    report: &mut CapabilityReport,
    cache: &PetriKernelPlanCache,
    bundle: &trust_ir::NativeVerificationBundle,
    bundle_source: &'static str,
    bundle_validated: bool,
) -> PetriNativeSemanticBridgeAuthority {
    let identity = bundle.transport_identity();
    let transport_digest = identity.stable_digest();
    let plan_cache_digest = native::petri_kernel_plan_cache_digest(cache);
    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let semantic_bridge_surface = downstream_contract.semantic_bridge;
    let trust_ir_native_bundle_identity = downstream_contract.trust_ir_native_bundle_identity;
    let trust_cg_bridge =
        tla_trust_cg::petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle(
            bundle,
            tla_trust_cg::PetriNativeSuccessorSemanticBridgeExpected::new(
                PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
            )
            .with_formula_schema(PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA),
        );
    let trust_ir_bridge = petri_native_successor_semantic_bridge(bundle);
    let trust_ir_bridge_report =
        bundle.petri_successor_semantic_bridge_report(trust_ir_bridge.function);
    let semantic_successor_authority = trust_cg_bridge.is_ready()
        && trust_ir_bridge_report.represents_petri_successor_plan_cache_equivalence();
    let semantic_bridge_status_code = if semantic_successor_authority {
        "ready"
    } else {
        "blocked"
    };
    let trust_cg_reason_code = trust_cg_bridge.reason_code.unwrap_or("none");
    let reason_code = if semantic_successor_authority {
        "none"
    } else {
        trust_cg_reason_code
    };
    let trust_cg_required_field = trust_cg_bridge.required_field.unwrap_or("none");
    let trust_cg_required_evidence = trust_cg_bridge.required_evidence.unwrap_or("none");
    let trust_cg_target_abi_digest = trust_cg_bridge
        .target_abi_digest
        .as_deref()
        .unwrap_or("none");
    let downstream_semantic_bridge_required_fields =
        semantic_bridge_surface.required_fields.join(",");
    let downstream_semantic_bridge_status_codes = semantic_bridge_surface.status_codes.join(",");
    let downstream_semantic_bridge_blocker_codes = semantic_bridge_surface.blocker_codes.join(",");
    let trust_ir_proof_obligation = trust_ir_bridge_report
        .proof_obligation
        .map_or_else(|| "none".to_owned(), |proof| proof.to_string());
    let trust_ir_proof_digest = trust_ir_bridge_report
        .proof_digest
        .as_ref()
        .map_or_else(|| "none".to_owned(), |digest| digest.to_string());
    let trust_ir_proof_status = trust_ir_bridge_report
        .proof_status
        .map(trust_ir_proof_status_code)
        .unwrap_or("none");
    let trust_ir_evidence_digest = trust_ir_bridge_report
        .evidence_digest
        .as_ref()
        .map_or_else(|| "none".to_owned(), |digest| digest.to_string());
    let trust_ir_semantic_bridge_proof_identity_digest =
        trust_ir_bridge_report.proof_identity_digest().to_string();
    let trust_ir_semantic_bridge_proof_identity_text =
        trust_ir_bridge_report.proof_identity_key_value_text();
    let trust_ir_semantic_bridge_proof_identity_replay = trust_ir_bridge_report
        .proof_identity_replay_report_for_key_value_text(
            &trust_ir_semantic_bridge_proof_identity_text,
        );
    let trust_ir_semantic_bridge_proof_identity_replay_component_health_text =
        trust_ir_semantic_bridge_proof_identity_replay.component_health_summary_key_value_text();
    let trust_ir_semantic_bridge_proof_identity_replay_component_health_lines =
        trust_ir_semantic_bridge_proof_identity_replay.component_health_summary_key_value_lines();
    let trust_ir_semantic_bridge_proof_identity_lines =
        trust_ir_bridge_report.proof_identity_key_value_lines();
    let trust_ir_semantic_bridge_proof_admission_attachments:
        &[trust_ir::request::NativeEvidenceArtifactAttachment] = &[];
    let trust_ir_semantic_bridge_proof_admission = bundle
        .petri_successor_semantic_bridge_proof_admission_report(
            trust_ir_bridge.function,
            trust_ir_semantic_bridge_proof_admission_attachments,
        );
    let trust_ir_semantic_bridge_proof_admission_resolution_status_codes =
        join_trust_ir_attachment_resolution_status_codes(
            &trust_ir_semantic_bridge_proof_admission.artifact_resolutions,
        );
    let trust_ir_semantic_bridge_proof_admission_resolution_reason_codes =
        join_trust_ir_attachment_resolution_reason_codes(
            &trust_ir_semantic_bridge_proof_admission.artifact_resolutions,
        );
    let trust_ir_semantic_bridge_proof_admission_resolution_authority_codes =
        join_trust_ir_attachment_resolution_authority_codes(
            &trust_ir_semantic_bridge_proof_admission.artifact_resolutions,
        );
    let trust_ir_semantic_bridge_proof_admission_resolution_required_kinds =
        join_trust_ir_attachment_resolution_required_kind_codes(
            &trust_ir_semantic_bridge_proof_admission.artifact_resolutions,
        );
    let trust_ir_semantic_bridge_proof_admission_authoritative_bytes_count =
        count_trust_ir_attachment_resolution_authoritative_bytes(
            &trust_ir_semantic_bridge_proof_admission.artifact_resolutions,
        );

    let mut fields = Vec::new();
    push_trust_cg_native_admission_field(&mut fields, "schema", semantic_bridge_surface.schema);
    push_trust_cg_native_admission_field(
        &mut fields,
        "schema_version",
        semantic_bridge_surface.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "source",
        "PetriNativeSuccessorSemanticBridgeEvidence",
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "api",
        PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "required_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_SEMANTIC_BRIDGE_REQUIRED_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "current_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_SEMANTIC_BRIDGE_CURRENT_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeReport,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_petri_successor_report_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorSemanticBridgeReport,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_constructor_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorSemanticBridgeConstructor,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_acceptance_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::RepresentsPetriSuccessorPlanCacheEquivalence,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_text_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityKeyValueText,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_replay_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayReportForKeyValueText,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_replay_component_health_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayComponentHealthSummaryKeyValueText,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_api",
        "NativeVerificationBundle::petri_successor_semantic_bridge_proof_admission_report()",
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_schema",
        &trust_ir_semantic_bridge_proof_admission.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_schema_version",
        trust_ir_semantic_bridge_proof_admission.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_status_code",
        trust_ir_semantic_bridge_proof_admission.status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_reason_code",
        trust_ir_semantic_bridge_proof_admission.reason_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_fail_closed",
        trust_ir_semantic_bridge_proof_admission.fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_function",
        trust_ir_semantic_bridge_proof_admission.function,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_attachment_count",
        trust_ir_semantic_bridge_proof_admission_attachments.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_required_artifact_kinds",
        join_trust_ir_artifact_kind_codes(
            &trust_ir_semantic_bridge_proof_admission.required_artifact_kinds,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_resolution_count",
        trust_ir_semantic_bridge_proof_admission
            .artifact_resolutions
            .len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_resolution_required_kinds",
        &trust_ir_semantic_bridge_proof_admission_resolution_required_kinds,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_resolution_status_codes",
        &trust_ir_semantic_bridge_proof_admission_resolution_status_codes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_resolution_reason_codes",
        &trust_ir_semantic_bridge_proof_admission_resolution_reason_codes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_resolution_authority_codes",
        &trust_ir_semantic_bridge_proof_admission_resolution_authority_codes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_authoritative_bytes_count",
        trust_ir_semantic_bridge_proof_admission_authoritative_bytes_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_blocked_artifact_kind",
        optional_trust_ir_artifact_kind_code_from_option(
            trust_ir_semantic_bridge_proof_admission.blocked_artifact_kind,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_blocked_artifact_reason_code",
        trust_ir_semantic_bridge_proof_admission
            .blocked_artifact_reason_code()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_proof_handoff_status_code",
        trust_ir_semantic_bridge_proof_admission
            .proof_handoff_report
            .status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_admission_proof_handoff_reason_code",
        trust_ir_semantic_bridge_proof_admission
            .proof_handoff_report
            .reason_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "formula_schema",
        PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_semantic_bridge_surface",
        semantic_bridge_surface.name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_semantic_bridge_required_fields",
        downstream_semantic_bridge_required_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_semantic_bridge_status_codes",
        downstream_semantic_bridge_status_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_semantic_bridge_blocker_codes",
        downstream_semantic_bridge_blocker_codes.as_str(),
    );
    push_trust_cg_native_admission_field(&mut fields, "bundle_source", bundle_source);
    push_trust_cg_native_admission_field(&mut fields, "bundle_validated", bundle_validated);
    push_trust_cg_native_admission_field(
        &mut fields,
        "entry_function",
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
    );
    push_trust_cg_native_admission_field(&mut fields, "plan_cache_abi_version", cache.abi_version);
    push_trust_cg_native_admission_field(&mut fields, "place_count", cache.place_count);
    push_trust_cg_native_admission_field(&mut fields, "transition_count", cache.transition_count);
    push_trust_cg_native_admission_field(&mut fields, "plan_count", cache.plans().len());
    push_trust_cg_native_admission_field(&mut fields, "plan_cache_digest", plan_cache_digest);
    push_trust_cg_native_admission_field(&mut fields, "transport_digest", transport_digest);
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_module_digest",
        identity.trust_ir_module_digest,
    );
    push_trust_cg_native_admission_field(&mut fields, "bundle_digest", identity.bundle_digest);
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_successor_body_status",
        "lowered_all_transition_successors",
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "successor_relation_represented",
        semantic_successor_authority,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_successor_authority",
        semantic_successor_authority,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_status_code",
        semantic_bridge_status_code,
    );
    push_trust_cg_native_admission_field(&mut fields, "reason_code", reason_code);
    push_trust_cg_native_admission_field(&mut fields, "trust_cg_schema", trust_cg_bridge.schema);
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_schema_version",
        trust_cg_bridge.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_status_code",
        trust_cg_bridge.status.as_str(),
    );
    push_trust_cg_native_admission_field(&mut fields, "trust_cg_reason_code", trust_cg_reason_code);
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_required_field",
        trust_cg_required_field,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_required_evidence",
        trust_cg_required_evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_bundle_validated",
        trust_cg_bridge.bundle_validated,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_target_abi_digest",
        trust_cg_target_abi_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_native_evidence_report_entries",
        trust_cg_bridge.native_evidence_report_entries,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_semantic_obligation_count",
        trust_cg_bridge.semantic_obligation_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_semantic_evidence_entry_count",
        trust_cg_bridge.semantic_evidence_entry_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_consumed_certificate_count",
        trust_cg_bridge.consumed_certificate_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_artifact_count",
        trust_cg_bridge.artifact_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_successor_relation_represented",
        trust_cg_bridge.successor_relation_represented,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_semantic_successor_authority",
        trust_cg_bridge.semantic_successor_authority,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_cg_semantic_bridge_sha256",
        &trust_cg_bridge.semantic_bridge_sha256,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_schema",
        &trust_ir_bridge_report.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_schema_version",
        trust_ir_bridge_report.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_status_code",
        trust_ir_bridge_report.status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_reason_code",
        trust_ir_bridge_report.reason_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_evidence_status",
        trust_ir_bridge_report.evidence_status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_schema",
        trust_ir_bridge_report.proof_identity_schema(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_schema_version",
        trust_ir_bridge_report.proof_identity_schema_version(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_digest",
        &trust_ir_semantic_bridge_proof_identity_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_row_count",
        trust_ir_semantic_bridge_proof_identity_lines.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_text_available",
        !trust_ir_semantic_bridge_proof_identity_text.is_empty(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_replay_status_code",
        trust_ir_semantic_bridge_proof_identity_replay.status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_replayable",
        trust_ir_semantic_bridge_proof_identity_replay.is_replayable(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_replay_fail_closed",
        trust_ir_semantic_bridge_proof_identity_replay.fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_identity_replay_diagnostic_count",
        trust_ir_semantic_bridge_proof_identity_replay.diagnostic_count(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_fail_closed",
        trust_ir_bridge_report.fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_relation",
        trust_ir_bridge_report.bridge.relation.code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_function",
        trust_ir_bridge_report.bridge.function,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_digest",
        trust_ir_bridge_report.bridge_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_obligation",
        trust_ir_proof_obligation,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_digest",
        trust_ir_proof_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_proof_status",
        trust_ir_proof_status,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_semantic_bridge_evidence_digest",
        trust_ir_evidence_digest,
    );
    push_trust_cg_native_admission_field(&mut fields, "production_selected", false);
    push_trust_cg_native_admission_field(&mut fields, "fail_closed", true);
    report.add_evidence(render_trust_cg_native_admission_row(
        "Petri native_jit semantic_successor_bridge",
        &fields,
    ));
    let mut proof_admission_fields = Vec::new();
    push_trust_cg_native_admission_field(&mut proof_admission_fields, "source_package", "trust_ir");
    push_trust_cg_native_admission_field(&mut proof_admission_fields, "source_project", "trust-ir");
    push_trust_cg_native_admission_field(&mut proof_admission_fields, "project", "trust-ir");
    push_trust_cg_native_admission_field(
        &mut proof_admission_fields,
        "producer_api",
        "PetriSuccessorSemanticBridgeProofAdmissionReport::key_value_rows()",
    );
    push_trust_cg_native_admission_field(&mut proof_admission_fields, "production_selected", false);
    push_trust_cg_native_admission_field(
        &mut proof_admission_fields,
        "fail_closed",
        trust_ir_semantic_bridge_proof_admission.fail_closed(),
    );
    for row in trust_ir_semantic_bridge_proof_admission.key_value_rows() {
        push_trust_cg_native_admission_field(&mut proof_admission_fields, &row.key, row.value);
    }
    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-ir petri_successor_semantic_bridge_proof_admission",
        &proof_admission_fields,
    ));
    let trust_ir_semantic_bridge_proof_identity_fields = trust_ir_component_identity_row_fields(
        trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA,
        trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA_VERSION,
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityKeyValueText,
        ),
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayReportForKeyValueText,
        ),
        &trust_ir_semantic_bridge_proof_identity_digest,
        trust_ir_semantic_bridge_proof_identity_lines.len(),
        !trust_ir_semantic_bridge_proof_identity_text.is_empty(),
        trust_ir_semantic_bridge_proof_identity_replay.status_code,
        trust_ir_semantic_bridge_proof_identity_replay.is_replayable(),
        trust_ir_semantic_bridge_proof_identity_replay.fail_closed,
        trust_ir_semantic_bridge_proof_identity_replay.diagnostic_count(),
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayComponentHealthSummaryKeyValueText,
        ),
        trust_ir_semantic_bridge_proof_identity_replay_component_health_lines.len(),
        !trust_ir_semantic_bridge_proof_identity_replay_component_health_text.is_empty(),
        trust_ir_bridge_report.status_code(),
        trust_ir_bridge_report.reason_code(),
        trust_ir_bridge_report.fail_closed(),
    );
    add_trust_ir_component_readiness_row(
        report,
        TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT,
        trust_ir_semantic_bridge_proof_identity_fields.clone(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT,
        trust_ir_semantic_bridge_proof_identity_fields,
        trust_ir_semantic_bridge_proof_identity_lines,
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_REPLAY_HEALTH_COMPONENT,
        trust_ir_common_row_fields(
            trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA,
            trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA_VERSION,
        ),
        trust_ir_semantic_bridge_proof_identity_replay_component_health_lines,
    );
    PetriNativeSemanticBridgeAuthority {
        represented: semantic_successor_authority,
        api: PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_API,
        formula_schema: PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA,
        required_evidence: trust_cg_required_evidence,
        status_code: semantic_bridge_status_code,
        reason_code,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_ay_trust_mc_native_verification_bundle_facade_evidence(
    report: &mut CapabilityReport,
    bundle: &trust_ir::NativeVerificationBundle,
    bundle_source: &'static str,
    bundle_validated: bool,
) -> AYTrustMcNativeVerificationBundleFacadeEvidence {
    let identity = bundle.transport_identity();
    let transport_digest = identity.stable_digest();
    let bridge = petri_native_successor_semantic_bridge(bundle);
    let ay_report =
        ay_trust_mc_native_bundle::solve_trust_mc_petri_successor_native_verification_bundle(
            bundle,
            bridge.function,
        );
    let model_acceptance_report =
        ay_trust_mc_native_bundle::trust_mc_petri_successor_chc_model_acceptance_report(
            bundle,
            bridge.function,
        );
    let lowering_report = ay_trust_mc_native_bundle::trust_mc_petri_successor_chc_lowering_report(
        bundle,
        bridge.function,
    );
    let route_admission =
        ay_trust_mc_native_bundle::trust_mc_petri_successor_native_route_admission_decision_from_reports(
            &ay_report,
            &lowering_report,
            &model_acceptance_report,
        );
    let route_admission_rows = route_admission.to_key_value_rows();
    let validated_route_admission =
        ay_trust_mc_native_bundle::validate_trust_mc_petri_successor_native_route_admission_key_value_rows(
            &route_admission,
            &route_admission_rows,
        );
    let model_validation_readiness_report =
        &model_acceptance_report.trust_mc_chc_model_validation_readiness_report;
    let proof_handoff_report = &model_validation_readiness_report.proof_handoff_report;
    let proof_evidence_identity_text =
        proof_handoff_report.proof_evidence_identity_key_value_text();
    let proof_evidence_identity_replay = proof_handoff_report
        .proof_evidence_identity_replay_report_for_key_value_text(&proof_evidence_identity_text);
    let proof_evidence_identity_replay_component_health_text =
        proof_evidence_identity_replay.component_health_summary_key_value_text();
    let proof_evidence_identity_replay_component_health_lines =
        proof_evidence_identity_replay.component_health_summary_key_value_lines();
    let proof_evidence_identity_lines =
        proof_handoff_report.proof_evidence_identity_key_value_lines();
    let semantic_bridge_report = &ay_report.semantic_bridge_report;
    let semantic_bridge_proof_identity = ay_report.semantic_bridge_proof_identity();
    let consumer_decision = ay_report.consumer_decision();
    let consumer_acceptance = ay_report.accept_for_consumer();
    let accepted_for_consumer = consumer_acceptance.is_ok();
    let model_acceptance_rejection = model_acceptance_report.accept_for_consumer().err();
    let model_acceptance_accepted_for_consumer = model_acceptance_rejection.is_none();
    let model_acceptance_consumer_rejection_status_code = optional_ay_str(
        model_acceptance_rejection
            .as_ref()
            .map(|rejection| rejection.status_code),
    );
    let model_acceptance_consumer_rejection_reason_code = optional_ay_str(
        model_acceptance_rejection
            .as_ref()
            .map(|rejection| rejection.reason_code),
    );
    let model_acceptance_consumer_rejection_fail_closed = model_acceptance_rejection
        .as_ref()
        .map(|rejection| rejection.fail_closed)
        .unwrap_or(false);
    let consumer_rejection = match consumer_decision {
        ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleConsumerDecision::Accepted => {
            None
        }
        ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleConsumerDecision::Rejected(
            rejection,
        ) => Some(rejection),
    };
    let consumer_rejection_status_code =
        optional_ay_str(consumer_rejection.map(|rejection| rejection.status_code));
    let consumer_rejection_reason_code =
        optional_ay_str(consumer_rejection.map(|rejection| rejection.reason_code));
    let consumer_rejection_code =
        optional_ay_str(consumer_rejection.map(|rejection| rejection.consumer_rejection_code));
    let consumer_rejection_fail_closed = consumer_rejection
        .map(|rejection| rejection.fail_closed)
        .unwrap_or(false);
    let consumer_rejection_ready_for_trust_mc_chc_handoff = consumer_rejection
        .map(|rejection| rejection.ready_for_trust_mc_chc_handoff)
        .unwrap_or(false);
    let fail_closed = consumer_rejection_fail_closed;
    let status_code = ay_report.status_code;

    let mut fields = Vec::new();
    push_trust_cg_native_admission_field(&mut fields, "schema", ay_report.schema);
    push_trust_cg_native_admission_field(&mut fields, "schema_version", ay_report.schema_version);
    push_trust_cg_native_admission_field(
        &mut fields,
        "source",
        "trust_mcNativeVerificationBundleReport",
    );
    push_trust_cg_native_admission_field(&mut fields, "problem", ay_report.problem);
    push_trust_cg_native_admission_field(
        &mut fields,
        "preferred_backend_code",
        ay_report.preferred_backend_code,
    );
    push_trust_cg_native_admission_field(&mut fields, "domain", ay_report.domain);
    push_trust_cg_native_admission_field(&mut fields, "scope", ay_report.scope);
    push_trust_cg_native_admission_field(
        &mut fields,
        "api",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "required_ay_rev",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_REQUIRED_AY_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "current_ay_rev",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV,
    );
    push_trust_cg_native_admission_field(&mut fields, "bundle_source", bundle_source);
    push_trust_cg_native_admission_field(&mut fields, "bundle_validated", bundle_validated);
    push_trust_cg_native_admission_field(
        &mut fields,
        "formula_schema",
        PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "entry_function",
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
    );
    push_trust_cg_native_admission_field(&mut fields, "status_code", status_code);
    push_trust_cg_native_admission_field(&mut fields, "reason_code", ay_report.reason_code);
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumer_acceptance_api",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CONSUMER_ACCEPTANCE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumer_rejection_status_code",
        consumer_rejection_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumer_rejection_reason_code",
        consumer_rejection_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumer_rejection_code",
        consumer_rejection_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "accepted_for_consumer",
        accepted_for_consumer,
    );
    push_trust_cg_native_admission_field(&mut fields, "fail_closed", fail_closed);
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumer_rejection_fail_closed",
        consumer_rejection_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumer_rejection_ready_for_trust_mc_chc_handoff",
        consumer_rejection_ready_for_trust_mc_chc_handoff,
    );
    push_trust_cg_native_admission_field(&mut fields, "model_validated", ay_report.model_validated);
    push_trust_cg_native_admission_field(
        &mut fields,
        "verification_level_code",
        ay_report.verification_level_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "proof_replay_status_code",
        ay_report.proof_replay_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ready_for_trust_mc_chc_handoff",
        ay_report.ready_for_trust_mc_chc_handoff,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_mc_request_count",
        ay_report.trust_mc_request_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_mc_evidence_count",
        ay_report.trust_mc_evidence_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_evidence_entry_count",
        ay_report.native_evidence_entry_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_request_count",
        ay_report.matched_trust_mc_request_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_chc_request_count",
        ay_report.matched_trust_mc_chc_request_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_evidence_count",
        ay_report.matched_trust_mc_evidence_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_artifact_count",
        ay_report.matched_trust_mc_artifact_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_request_ids",
        join_ay_u32s(&ay_report.matched_trust_mc_request_ids),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_request_mode_codes",
        join_ay_strs(&ay_report.matched_trust_mc_request_mode_codes),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_request_digests",
        join_ay_strings(&ay_report.matched_trust_mc_request_digests),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_evidence_digests",
        join_ay_strings(&ay_report.matched_trust_mc_evidence_digests),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "matched_trust_mc_artifact_kind_codes",
        join_ay_strs(&ay_report.matched_trust_mc_artifact_kind_codes),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_status_code",
        ay_report.semantic_bridge_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_reason_code",
        ay_report.semantic_bridge_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_evidence_status_code",
        ay_report.semantic_bridge_evidence_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_proof_identity_schema",
        semantic_bridge_proof_identity.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_proof_identity_schema_version",
        semantic_bridge_proof_identity.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_proof_identity_digest",
        &semantic_bridge_proof_identity.digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_fail_closed",
        semantic_bridge_proof_identity.fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_schema",
        &semantic_bridge_report.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_schema_version",
        semantic_bridge_report.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_relation",
        ay_report.semantic_bridge_relation_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_function",
        ay_report.semantic_bridge_function_index,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_relation_code",
        ay_report.semantic_bridge_relation_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_function_index",
        ay_report.semantic_bridge_function_index,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_formula_schema",
        &ay_report.semantic_bridge_formula_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_digest",
        &ay_report.semantic_bridge_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_proof_obligation_index",
        optional_ay_u32(ay_report.semantic_bridge_proof_obligation_index),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_proof_status_code",
        optional_ay_str(ay_report.semantic_bridge_proof_status_code),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_proof_digest",
        optional_ay_string(&ay_report.semantic_bridge_proof_digest),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "semantic_bridge_evidence_digest",
        optional_ay_string(&ay_report.semantic_bridge_evidence_digest),
    );
    push_trust_cg_native_admission_field(&mut fields, "transport_digest", transport_digest);
    push_trust_cg_native_admission_field(&mut fields, "bundle_digest", identity.bundle_digest);
    push_trust_cg_native_admission_field(
        &mut fields,
        "request_digests",
        identity.request_digests.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "evidence_digests",
        identity.evidence_digests.len(),
    );
    push_trust_cg_native_admission_field(&mut fields, "production_selected", false);
    report.add_evidence(render_trust_cg_native_admission_row(
        "AY trust_mc_native_verification_bundle_facade",
        &fields,
    ));
    report.add_evidence(render_trust_cg_native_admission_row(
        "AY trust_mc_petri_successor_native_route_admission",
        &validated_route_admission.to_key_value_rows(),
    ));

    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let trust_ir_native_bundle_identity = downstream_contract.trust_ir_native_bundle_identity;
    let trust_mc_contract = downstream_contract.trust_ir_petri_trust_mc_chc_contract;
    let shared_primitive_contract =
        downstream_contract.trust_ir_petri_trust_mc_chc_shared_primitive_contract;
    let production_artifact_requirements =
        shared_primitive_contract.production_required_artifact_requirements();
    let artifact_byte_attachments: &[trust_ir::request::NativeEvidenceArtifactAttachment] = &[];
    add_trust_ir_native_evidence_artifact_resolution_evidence(
        report,
        bundle,
        shared_primitive_contract.verifier_suite,
        production_artifact_requirements,
        artifact_byte_attachments,
    );
    let artifact_authority_summary = summarize_trust_ir_artifact_authority(
        bundle,
        shared_primitive_contract.verifier_suite,
        production_artifact_requirements,
        artifact_byte_attachments,
    );
    let mut model_acceptance_fields = Vec::new();
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "schema",
        model_acceptance_report.schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "schema_version",
        model_acceptance_report.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "source",
        "trust_mcPetriSuccessorChcModelAcceptanceReport",
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "problem",
        ay_trust_mc_native_bundle::TRUST_MC_NATIVE_VERIFICATION_BUNDLE_PROBLEM,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "preferred_backend_code",
        ay_trust_mc_native_bundle::TRUST_MC_NATIVE_VERIFICATION_BUNDLE_BACKEND_CODE,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "domain",
        ay_trust_mc_native_bundle::TRUST_MC_NATIVE_VERIFICATION_BUNDLE_DOMAIN,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "scope",
        ay_trust_mc_native_bundle::TRUST_MC_NATIVE_VERIFICATION_BUNDLE_SCOPE,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "api",
        shared_primitive_contract.production_acceptance_report_api_name(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "consumer_acceptance_api",
        shared_primitive_contract.production_consumer_acceptance_api_name(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "required_ay_rev",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_REQUIRED_AY_REV,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "current_ay_rev",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "bundle_source",
        bundle_source,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "bundle_validated",
        bundle_validated,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "status_code",
        model_acceptance_report.status_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "reason_code",
        model_acceptance_report.reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "accepted_for_consumer",
        model_acceptance_accepted_for_consumer,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "fail_closed",
        model_acceptance_report.fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "consumer_rejection_status_code",
        model_acceptance_consumer_rejection_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "consumer_rejection_reason_code",
        model_acceptance_consumer_rejection_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "consumer_rejection_fail_closed",
        model_acceptance_consumer_rejection_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "proof_handoff_ready",
        model_acceptance_report.proof_handoff_ready,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "ready_for_solver_validation",
        model_acceptance_report.ready_for_solver_validation,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_model_validation_present",
        model_acceptance_report.solver_model_validation_present,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_model_validation_accepted",
        model_acceptance_report.solver_model_validation_accepted,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_status_code",
        model_acceptance_report.trust_mc_chc_proof_handoff_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_reason_code",
        model_acceptance_report.trust_mc_chc_proof_handoff_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_status_code",
        model_acceptance_report.trust_mc_chc_model_validation_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_reason_code",
        model_acceptance_report.trust_mc_chc_model_validation_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "model_artifact_digest",
        optional_ay_string(&model_acceptance_report.model_artifact_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "proof_identity_digest",
        optional_ay_string(&model_acceptance_report.proof_identity_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "replay_transcript_digest",
        optional_ay_string(&model_acceptance_report.replay_transcript_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_model_artifact_digest",
        optional_ay_string(&model_acceptance_report.solver_model_artifact_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_proof_identity_digest",
        optional_ay_string(&model_acceptance_report.solver_proof_identity_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_replay_transcript_digest",
        optional_ay_string(&model_acceptance_report.solver_replay_transcript_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_artifact_bytes_validated",
        model_acceptance_report.solver_artifact_bytes_validated,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_model_artifact_bytes_digest",
        optional_ay_string(&model_acceptance_report.solver_model_artifact_bytes_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_replay_transcript_artifact_bytes_digest",
        optional_ay_string(&model_acceptance_report.solver_replay_transcript_artifact_bytes_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_validation_digest",
        optional_ay_string(&model_acceptance_report.solver_validation_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "solver_identity_count",
        model_acceptance_report.solver_identity_count,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_schema",
        &proof_handoff_report.schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_schema_version",
        proof_handoff_report.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_fail_closed",
        proof_handoff_report.fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_proof_identity_digest",
        optional_trust_ir_proof_digest_string(&proof_handoff_report.proof_identity_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_text_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityKeyValueText,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_replay_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReportForKeyValueText,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_replay_component_health_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayComponentHealthSummaryKeyValueText,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_schema",
        trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_schema_version",
        trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_digest",
        proof_handoff_report.proof_evidence_identity_digest(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_row_count",
        proof_evidence_identity_lines.len(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_text_available",
        !proof_evidence_identity_text.is_empty(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_replay_status_code",
        proof_evidence_identity_replay.status_code,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_replayable",
        proof_evidence_identity_replay.is_replayable(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_replay_fail_closed",
        proof_evidence_identity_replay.fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_proof_evidence_identity_replay_diagnostic_count",
        proof_evidence_identity_replay.diagnostic_count(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_replay_transcript_digest",
        optional_trust_ir_proof_digest_string(&proof_handoff_report.replay_transcript_digest),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_replay_artifact_name",
        optional_trust_ir_artifact_name(&proof_handoff_report.replay_transcript_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_replay_artifact_kind_code",
        optional_trust_ir_artifact_kind_code(&proof_handoff_report.replay_transcript_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_replay_artifact_digest",
        optional_trust_ir_artifact_digest_string(&proof_handoff_report.replay_transcript_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_model_artifact_name",
        optional_trust_ir_artifact_name(&proof_handoff_report.model_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_model_artifact_kind_code",
        optional_trust_ir_artifact_kind_code(&proof_handoff_report.model_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_model_artifact_digest",
        optional_trust_ir_artifact_digest_string(&proof_handoff_report.model_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_proof_handoff_solver_identity_count",
        proof_handoff_report.solver_identities.len(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_schema",
        &model_validation_readiness_report.schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_schema_version",
        model_validation_readiness_report.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_fail_closed",
        model_validation_readiness_report.fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_model_artifact_name",
        optional_trust_ir_artifact_name(&model_validation_readiness_report.model_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_model_artifact_kind_code",
        optional_trust_ir_artifact_kind_code(&model_validation_readiness_report.model_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_model_artifact_digest",
        optional_trust_ir_artifact_digest_string(&model_validation_readiness_report.model_artifact),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_mc_chc_model_validation_solver_identity_count",
        model_validation_readiness_report.solver_identities.len(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::ContractDescriptor,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_schema",
        trust_mc_contract.schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_schema_version",
        trust_mc_contract.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_formula_schema",
        trust_mc_contract.formula_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_binding_report_schema",
        trust_mc_contract.binding_report_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_binding_report_schema_version",
        trust_mc_contract.binding_report_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_proof_handoff_report_schema",
        trust_mc_contract.proof_handoff_report_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_proof_handoff_report_schema_version",
        trust_mc_contract.proof_handoff_report_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_validation_readiness_report_schema",
        trust_mc_contract.model_validation_readiness_report_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_validation_readiness_report_schema_version",
        trust_mc_contract.model_validation_readiness_report_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_verifier_suite",
        trust_mc_contract.verifier_suite,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_verification_mode",
        trust_mc_verification_mode_code(trust_mc_contract.verification_mode),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_binding_required_artifact_kinds",
        join_trust_ir_artifact_kind_codes(trust_mc_contract.binding_required_artifact_kinds),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_proof_handoff_required_artifact_kinds",
        join_trust_ir_artifact_kind_codes(trust_mc_contract.proof_handoff_required_artifact_kinds),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_proof_handoff_optional_artifact_kinds",
        join_trust_ir_artifact_kind_codes(trust_mc_contract.proof_handoff_optional_artifact_kinds),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_validation_required_artifact_kinds",
        join_trust_ir_artifact_kind_codes(
            trust_mc_contract.model_validation_required_artifact_kinds,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_production_acceptance_required_artifact_kinds",
        join_trust_ir_artifact_kind_codes(
            trust_mc_contract.production_acceptance_required_artifact_kinds,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_validation_requires_solver_acceptance",
        trust_mc_contract.model_validation_requires_solver_acceptance,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_acceptance_report_api_name",
        trust_mc_contract.model_acceptance_report_api_name,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_consumer_acceptance_api_name",
        trust_mc_contract.consumer_acceptance_api_name,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_production_acceptance_owner_suite",
        trust_mc_contract.production_acceptance_owner_suite,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_production_requires_emitted_solver_artifacts",
        shared_primitive_contract.production_requires_emitted_solver_artifacts(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_schema",
        shared_primitive_contract.schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_schema_version",
        shared_primitive_contract.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_contract_schema",
        shared_primitive_contract.contract_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_contract_schema_version",
        shared_primitive_contract.contract_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_formula_schema",
        shared_primitive_contract.formula_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_readiness_report_schema",
        shared_primitive_contract.readiness_report_schema,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_readiness_report_schema_version",
        shared_primitive_contract.readiness_report_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_verifier_suite",
        shared_primitive_contract.verifier_suite,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_verification_mode",
        native_shared_primitive_verification_mode_code(shared_primitive_contract.verification_mode),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_kinds",
        join_trust_ir_artifact_kind_codes(shared_primitive_contract.required_artifact_kinds),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_roles",
        join_trust_ir_artifact_role_codes(
            shared_primitive_contract.production_required_artifact_roles(),
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_optional_artifact_kinds",
        join_trust_ir_artifact_kind_codes(shared_primitive_contract.optional_artifact_kinds),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_identity_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::ArtifactIdentity,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_byte_resolution_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::ArtifactByteResolution,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_authority_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::ArtifactAuthority,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_authoritative_bytes_api",
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::AuthoritativeBytes,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_byte_attachment_count",
        artifact_authority_summary.attachment_count,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_byte_resolution_status_codes",
        &artifact_authority_summary.resolution_status_codes,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_byte_resolution_reason_codes",
        &artifact_authority_summary.resolution_reason_codes,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_artifact_byte_resolution_authority_codes",
        &artifact_authority_summary.resolution_authority_codes,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_authoritative_artifact_requirement_count",
        artifact_authority_summary.authoritative_requirement_count,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_authoritative_artifact_requirement_roles",
        &artifact_authority_summary.authoritative_requirement_roles,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_unauthoritative_artifact_requirement_roles",
        &artifact_authority_summary.unauthoritative_requirement_roles,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_authoritative_artifact_requirement_kinds",
        &artifact_authority_summary.authoritative_requirement_kinds,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_unauthoritative_artifact_requirement_kinds",
        &artifact_authority_summary.unauthoritative_requirement_kinds,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_authoritative_artifact_bytes_count",
        artifact_authority_summary.authoritative_bytes_count,
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_requirement_kinds",
        join_trust_ir_artifact_requirement_kind_codes(production_artifact_requirements),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_requirement_roles",
        join_trust_ir_artifact_requirement_role_codes(production_artifact_requirements),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_requirement_digest_algorithms",
        join_trust_ir_artifact_requirement_digest_algorithm_codes(production_artifact_requirements),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_requirement_owner_suites",
        join_trust_ir_artifact_requirement_owner_suite_codes(production_artifact_requirements),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_required_artifact_requirement_requires_emitted_solver_artifacts",
        join_trust_ir_artifact_requirement_emitted_solver_artifact_codes(
            production_artifact_requirements,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_production_artifact_owner_suites",
        join_trust_ir_verifier_suite_codes(
            shared_primitive_contract.production_artifact_owner_suites(),
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_bound_artifact_requirement_count",
        count_trust_ir_bound_artifact_requirements(bundle, production_artifact_requirements),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_bound_artifact_requirement_roles",
        join_trust_ir_bound_artifact_requirement_role_codes(
            bundle,
            production_artifact_requirements,
            true,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_unbound_artifact_requirement_roles",
        join_trust_ir_bound_artifact_requirement_role_codes(
            bundle,
            production_artifact_requirements,
            false,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_bound_artifact_requirement_kinds",
        join_trust_ir_bound_artifact_requirement_kind_codes(
            bundle,
            production_artifact_requirements,
            true,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_unbound_artifact_requirement_kinds",
        join_trust_ir_bound_artifact_requirement_kind_codes(
            bundle,
            production_artifact_requirements,
            false,
        ),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_requires_solver_acceptance",
        shared_primitive_contract.production_acceptance_requires_solver(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_model_acceptance_report_api_name",
        shared_primitive_contract.production_acceptance_report_api_name(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_consumer_acceptance_api_name",
        shared_primitive_contract.production_consumer_acceptance_api_name(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_production_acceptance_owner_suite",
        shared_primitive_contract.production_acceptance_owner_suite(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_shared_primitive_production_requires_emitted_solver_artifacts",
        shared_primitive_contract.production_requires_emitted_solver_artifacts(),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_provided_fields",
        join_ay_strs(trust_mc_contract.provided_fields),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_binding_status_codes",
        join_ay_strs(trust_mc_contract.binding_status_codes),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_binding_reason_codes",
        join_ay_strs(trust_mc_contract.binding_reason_codes),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_proof_handoff_status_codes",
        join_ay_strs(trust_mc_contract.proof_handoff_status_codes),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_proof_handoff_reason_codes",
        join_ay_strs(trust_mc_contract.proof_handoff_reason_codes),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_validation_readiness_status_codes",
        join_ay_strs(trust_mc_contract.model_validation_readiness_status_codes),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "trust_ir_contract_model_validation_readiness_reason_codes",
        join_ay_strs(trust_mc_contract.model_validation_readiness_reason_codes),
    );
    push_trust_cg_native_admission_field(
        &mut model_acceptance_fields,
        "production_selected",
        false,
    );
    report.add_evidence(render_trust_cg_native_admission_row(
        "AY trust_mc_petri_successor_chc_model_acceptance",
        &model_acceptance_fields,
    ));
    let proof_evidence_identity_fields = trust_ir_component_identity_row_fields(
        trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA,
        trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA_VERSION,
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityKeyValueText,
        ),
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReportForKeyValueText,
        ),
        proof_handoff_report.proof_evidence_identity_digest(),
        proof_evidence_identity_lines.len(),
        !proof_evidence_identity_text.is_empty(),
        proof_evidence_identity_replay.status_code,
        proof_evidence_identity_replay.is_replayable(),
        proof_evidence_identity_replay.fail_closed,
        proof_evidence_identity_replay.diagnostic_count(),
        trust_ir_petri_trust_mc_provided_field(
            trust_ir_native_bundle_identity.provided_fields,
            TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayComponentHealthSummaryKeyValueText,
        ),
        proof_evidence_identity_replay_component_health_lines.len(),
        !proof_evidence_identity_replay_component_health_text.is_empty(),
        proof_handoff_report.status_code(),
        proof_handoff_report.reason_code(),
        proof_handoff_report.fail_closed(),
    );
    add_trust_ir_component_readiness_row(
        report,
        TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_COMPONENT,
        proof_evidence_identity_fields.clone(),
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_COMPONENT,
        proof_evidence_identity_fields,
        proof_evidence_identity_lines,
    );
    add_trust_ir_component_manifest_lines(
        report,
        TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_REPLAY_HEALTH_COMPONENT,
        trust_ir_common_row_fields(
            trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_REPLAY_REPORT_SCHEMA,
            trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_REPLAY_REPORT_SCHEMA_VERSION,
        ),
        proof_evidence_identity_replay_component_health_lines,
    );

    AYTrustMcNativeVerificationBundleFacadeEvidence {
        accepted_for_consumer,
        fail_closed,
        status_code,
        reason_code: ay_report.reason_code,
        consumer_acceptance_api:
            AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CONSUMER_ACCEPTANCE_API,
        consumer_rejection_status_code,
        consumer_rejection_reason_code,
        consumer_rejection_code,
        consumer_rejection_fail_closed,
        consumer_rejection_ready_for_trust_mc_chc_handoff,
        model_validated: ay_report.model_validated,
        verification_level_code: ay_report.verification_level_code,
        proof_replay_status_code: ay_report.proof_replay_status_code,
        ready_for_trust_mc_chc_handoff: ay_report.ready_for_trust_mc_chc_handoff,
        matched_trust_mc_request_count: ay_report.matched_trust_mc_request_count,
        matched_trust_mc_chc_request_count: ay_report.matched_trust_mc_chc_request_count,
        matched_trust_mc_evidence_count: ay_report.matched_trust_mc_evidence_count,
        matched_trust_mc_artifact_count: ay_report.matched_trust_mc_artifact_count,
        model_acceptance_accepted_for_consumer,
        model_acceptance_fail_closed: model_acceptance_report.fail_closed,
        model_acceptance_status_code: model_acceptance_report.status_code,
        model_acceptance_reason_code: model_acceptance_report.reason_code,
        model_acceptance_api: shared_primitive_contract.production_acceptance_report_api_name(),
        model_acceptance_consumer_acceptance_api: shared_primitive_contract
            .production_consumer_acceptance_api_name(),
        model_acceptance_consumer_rejection_status_code,
        model_acceptance_consumer_rejection_reason_code,
        model_acceptance_consumer_rejection_fail_closed,
        model_acceptance_proof_handoff_ready: model_acceptance_report.proof_handoff_ready,
        model_acceptance_ready_for_solver_validation: model_acceptance_report
            .ready_for_solver_validation,
        model_acceptance_solver_model_validation_present: model_acceptance_report
            .solver_model_validation_present,
        model_acceptance_solver_model_validation_accepted: model_acceptance_report
            .solver_model_validation_accepted,
        model_acceptance_solver_artifact_bytes_validated: model_acceptance_report
            .solver_artifact_bytes_validated,
        model_acceptance_solver_model_artifact_bytes_digest: optional_ay_string(
            &model_acceptance_report.solver_model_artifact_bytes_digest,
        )
        .to_owned(),
        model_acceptance_solver_replay_transcript_artifact_bytes_digest: optional_ay_string(
            &model_acceptance_report.solver_replay_transcript_artifact_bytes_digest,
        )
        .to_owned(),
        model_acceptance_trust_ir_artifact_byte_attachment_count: artifact_authority_summary
            .attachment_count,
        model_acceptance_trust_ir_artifact_byte_resolution_status_codes: artifact_authority_summary
            .resolution_status_codes,
        model_acceptance_trust_ir_artifact_byte_resolution_reason_codes: artifact_authority_summary
            .resolution_reason_codes,
        model_acceptance_trust_ir_artifact_byte_resolution_authority_codes:
            artifact_authority_summary.resolution_authority_codes,
        model_acceptance_trust_ir_authoritative_artifact_requirement_count:
            artifact_authority_summary.authoritative_requirement_count,
        model_acceptance_trust_ir_authoritative_artifact_requirement_roles:
            artifact_authority_summary.authoritative_requirement_roles,
        model_acceptance_trust_ir_unauthoritative_artifact_requirement_roles:
            artifact_authority_summary.unauthoritative_requirement_roles,
        model_acceptance_trust_ir_authoritative_artifact_requirement_kinds:
            artifact_authority_summary.authoritative_requirement_kinds,
        model_acceptance_trust_ir_unauthoritative_artifact_requirement_kinds:
            artifact_authority_summary.unauthoritative_requirement_kinds,
        model_acceptance_trust_ir_authoritative_artifact_bytes_count: artifact_authority_summary
            .authoritative_bytes_count,
        model_acceptance_trust_mc_chc_proof_handoff_status_code: model_acceptance_report
            .trust_mc_chc_proof_handoff_status_code,
        model_acceptance_trust_mc_chc_proof_handoff_reason_code: model_acceptance_report
            .trust_mc_chc_proof_handoff_reason_code,
        model_acceptance_trust_mc_chc_proof_handoff_schema: proof_handoff_report.schema.clone(),
        model_acceptance_trust_mc_chc_proof_handoff_schema_version: proof_handoff_report
            .schema_version,
        model_acceptance_trust_mc_chc_proof_handoff_fail_closed: proof_handoff_report.fail_closed(),
        model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_name:
            optional_trust_ir_artifact_name(&proof_handoff_report.replay_transcript_artifact)
                .to_owned(),
        model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_kind_code:
            optional_trust_ir_artifact_kind_code(&proof_handoff_report.replay_transcript_artifact)
                .to_owned(),
        model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_digest:
            optional_trust_ir_artifact_digest_string(
                &proof_handoff_report.replay_transcript_artifact,
            ),
        model_acceptance_trust_mc_chc_proof_handoff_model_artifact_name:
            optional_trust_ir_artifact_name(&proof_handoff_report.model_artifact).to_owned(),
        model_acceptance_trust_mc_chc_proof_handoff_model_artifact_kind_code:
            optional_trust_ir_artifact_kind_code(&proof_handoff_report.model_artifact).to_owned(),
        model_acceptance_trust_mc_chc_proof_handoff_model_artifact_digest:
            optional_trust_ir_artifact_digest_string(&proof_handoff_report.model_artifact),
        model_acceptance_trust_mc_chc_model_validation_status_code: model_acceptance_report
            .trust_mc_chc_model_validation_status_code,
        model_acceptance_trust_mc_chc_model_validation_reason_code: model_acceptance_report
            .trust_mc_chc_model_validation_reason_code,
        model_acceptance_trust_mc_chc_model_validation_schema: model_validation_readiness_report
            .schema
            .clone(),
        model_acceptance_trust_mc_chc_model_validation_schema_version:
            model_validation_readiness_report.schema_version,
        model_acceptance_trust_mc_chc_model_validation_fail_closed:
            model_validation_readiness_report.fail_closed(),
        model_acceptance_trust_mc_chc_model_validation_model_artifact_name:
            optional_trust_ir_artifact_name(&model_validation_readiness_report.model_artifact)
                .to_owned(),
        model_acceptance_trust_mc_chc_model_validation_model_artifact_kind_code:
            optional_trust_ir_artifact_kind_code(&model_validation_readiness_report.model_artifact)
                .to_owned(),
        model_acceptance_trust_mc_chc_model_validation_model_artifact_digest:
            optional_trust_ir_artifact_digest_string(
                &model_validation_readiness_report.model_artifact,
            ),
        semantic_bridge_proof_identity_schema: semantic_bridge_proof_identity.schema,
        semantic_bridge_proof_identity_schema_version: semantic_bridge_proof_identity
            .schema_version,
        semantic_bridge_proof_identity_digest: semantic_bridge_proof_identity.digest,
        semantic_bridge_fail_closed: semantic_bridge_proof_identity.fail_closed,
        semantic_bridge_status_code: semantic_bridge_proof_identity.status_code,
        semantic_bridge_reason_code: semantic_bridge_proof_identity.reason_code,
        semantic_bridge_evidence_status_code: semantic_bridge_proof_identity.evidence_status_code,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_cg_native_admission_blocker_for_bundle(
    report: &mut CapabilityReport,
    bundle: &trust_ir::NativeVerificationBundle,
    bundle_source: &'static str,
    bundle_validated: bool,
) {
    let identity = bundle.transport_identity();
    let transport_digest = identity.stable_digest();
    let summary = tla_trust_cg::petri_native_successor_admission_from_trust_ir_bundle(
        bundle,
        tla_trust_cg::PetriNativeSuccessorAdmissionExpected::validation_only(),
    );
    let reason_code = summary.reason_code.unwrap_or("none");
    let proof_report_sha256 = summary.proof_report_sha256.as_deref().unwrap_or("none");
    let telemetry_event_id = summary.telemetry_event_id.as_deref().unwrap_or("none");
    let telemetry_record_sha256 = summary.telemetry_record_sha256.as_deref().unwrap_or("none");
    let replay_root_sha256 = summary.replay_root_sha256.as_deref().unwrap_or("none");
    let install_consumer_verdict_sha256 = summary
        .install_consumer_verdict_sha256
        .as_deref()
        .unwrap_or("none");
    let admission_evidence_sha256 = summary
        .admission_evidence_sha256
        .as_deref()
        .unwrap_or("none");
    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let admission_surface = downstream_contract.install_gate_admission;
    let admission_required_fields = admission_surface.required_fields.join(",");
    let admission_status_codes = admission_surface.status_codes.join(",");
    let admission_blocker_codes = admission_surface.blocker_codes.join(",");
    let admission_status_in_downstream_contract = admission_surface
        .status_codes
        .contains(&summary.disposition);
    let admission_reason_in_downstream_contract =
        reason_code == "none" || admission_surface.blocker_codes.contains(&reason_code);
    let (
        trust_ir_bundle_consumed,
        trust_ir_consumption_status,
        trust_ir_consumption_entries,
        consumed_certificates,
        artifact_count,
        validation_errors,
    ) = match bundle.native_evidence_consumption_report() {
        Ok(consumption) => {
            let artifact_count = consumption
                .entries
                .iter()
                .map(|entry| entry.artifacts.len())
                .sum::<usize>();
            let status = if consumption.entries.is_empty() {
                "missing_native_evidence"
            } else {
                "available"
            };
            (
                true,
                status,
                consumption.entries.len(),
                consumption.consumed_certificate_count(),
                artifact_count,
                0,
            )
        }
        Err(errors) => (false, "validation_failed", 0, 0, 0, errors.len()),
    };
    let evidence_profile = trust_cg_petri_native_evidence_profile(bundle);

    let mut fields = Vec::new();
    push_trust_cg_native_admission_field(
        &mut fields,
        "source",
        "NativeInstallGateAdmissionSummary",
    );
    push_trust_cg_native_admission_field(&mut fields, "source_package", "trust-cg-codegen");
    push_trust_cg_native_admission_field(&mut fields, "package", "trust-cg-codegen");
    push_trust_cg_native_admission_field(&mut fields, "schema", summary.schema);
    push_trust_cg_native_admission_field(&mut fields, "schema_version", summary.schema_version);
    push_trust_cg_native_admission_field(&mut fields, "consumer", &summary.consumer);
    push_trust_cg_native_admission_field(&mut fields, "consumer_mode", "petri_successor");
    push_trust_cg_native_admission_field(&mut fields, "kind", "petri_native_successor");
    push_trust_cg_native_admission_field(&mut fields, "surface", "mcc_replay");
    push_trust_cg_native_admission_field(
        &mut fields,
        "summary_consumer_mode",
        &summary.consumer_mode,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "summary_kind",
        TRUST_CG_PETRI_NATIVE_ADMISSION_KIND,
    );
    push_trust_cg_native_admission_field(&mut fields, "summary_surface", summary.surface);
    push_trust_cg_native_admission_field(&mut fields, "disposition", summary.disposition);
    push_trust_cg_native_admission_field(&mut fields, "status_code", summary.disposition);
    push_trust_cg_native_admission_field(&mut fields, "rejection_code", reason_code);
    push_trust_cg_native_admission_field(&mut fields, "reason_code", reason_code);
    push_trust_cg_native_admission_field(&mut fields, "requested_authority", "active_callable");
    push_trust_cg_native_admission_field(
        &mut fields,
        "summary_requested_authority",
        summary.requested_authority,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_authority",
        summary.install_authority,
    );
    push_trust_cg_native_admission_field(&mut fields, "cargo_dependency", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "bundle_api",
        TRUST_CG_PETRI_NATIVE_ADMISSION_BUNDLE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_api",
        TRUST_CG_PETRI_NATIVE_ADMISSION_API,
    );
    push_trust_cg_native_admission_field(&mut fields, "admission_descriptor_available", true);
    push_trust_cg_native_admission_field(&mut fields, "admission_descriptor_authoritative", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_source",
        TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_name",
        admission_surface.name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_schema",
        admission_surface.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_schema_version",
        admission_surface.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_required_fields",
        admission_required_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_status_codes",
        admission_status_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_descriptor_blocker_codes",
        admission_blocker_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_status_in_downstream_contract",
        admission_status_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_reason_in_downstream_contract",
        admission_reason_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(&mut fields, "bundle_source", bundle_source);
    push_trust_cg_native_admission_field(&mut fields, "bundle_validated", bundle_validated);
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_transport_identity_available",
        true,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_bundle_consumed",
        trust_ir_bundle_consumed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_consumption_status",
        trust_ir_consumption_status,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_consumption_entries",
        trust_ir_consumption_entries,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumed_certificates",
        consumed_certificates,
    );
    push_trust_cg_native_admission_field(&mut fields, "artifact_count", artifact_count);
    push_trust_cg_native_admission_field(&mut fields, "validation_errors", validation_errors);
    push_trust_cg_native_evidence_profile_fields(&mut fields, &evidence_profile);
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_expose_callable",
        summary.actions.expose_callable,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_typed_symbol_lookup",
        summary.actions.typed_symbol_lookup,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_insert_installable_cache",
        summary.actions.insert_installable_cache,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_accept_installable_cache_hit",
        summary.actions.accept_installable_cache_hit,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_release_installable",
        summary.actions.release_installable,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_ay_registry_insert",
        summary.actions.ay_registry_insert,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_ty_native_activate",
        summary.actions.ty_native_activate,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_useful_native_eligible",
        summary.actions.useful_native_eligible,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "useful_native_delta",
        summary.useful_native_delta,
    );
    push_trust_cg_native_admission_field(&mut fields, "packet_hash", summary.packet_hash);
    push_trust_cg_native_admission_field(
        &mut fields,
        "persisted_packet_hash",
        summary.persisted_packet_hash,
    );
    push_trust_cg_native_admission_field(&mut fields, "artifact_id", &summary.artifact_id);
    push_trust_cg_native_admission_field(
        &mut fields,
        "manifest_checksum",
        summary.manifest_checksum,
    );
    push_trust_cg_native_admission_field(&mut fields, "source_sha256", &summary.source_sha256);
    push_trust_cg_native_admission_field(&mut fields, "trust_ir_sha256", &summary.trust_ir_sha256);
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_payload_sha256",
        &summary.native_payload_sha256,
    );
    push_trust_cg_native_admission_field(&mut fields, "target_checksum", summary.target_checksum);
    push_trust_cg_native_admission_field(&mut fields, "abi_checksum", summary.abi_checksum);
    push_trust_cg_native_admission_field(&mut fields, "layout_checksum", summary.layout_checksum);
    push_trust_cg_native_admission_field(
        &mut fields,
        "proof_policy_checksum",
        summary.proof_policy_checksum,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "invalidation_checksum",
        summary.invalidation_checksum,
    );
    push_trust_cg_native_admission_field(&mut fields, "proof_report_sha256", proof_report_sha256);
    push_trust_cg_native_admission_field(&mut fields, "counter_scope", &summary.counter_scope);
    push_trust_cg_native_admission_field(&mut fields, "telemetry_event_id", telemetry_event_id);
    push_trust_cg_native_admission_field(
        &mut fields,
        "telemetry_record_sha256",
        telemetry_record_sha256,
    );
    push_trust_cg_native_admission_field(&mut fields, "replay_root_sha256", replay_root_sha256);
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_consumer_verdict_sha256",
        install_consumer_verdict_sha256,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_evidence_sha256",
        admission_evidence_sha256,
    );
    push_trust_cg_native_admission_field(&mut fields, "transport_digest", transport_digest);
    push_trust_cg_native_admission_field(&mut fields, "bundle_digest", identity.bundle_digest);
    push_trust_cg_native_admission_field(
        &mut fields,
        "request_digests",
        identity.request_digests.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "evidence_digests",
        identity.evidence_digests.len(),
    );
    push_trust_cg_native_admission_field(&mut fields, "production_selected", false);
    push_trust_cg_native_admission_field(&mut fields, "fail_closed", true);

    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-cg trust_cg_admission_blocker",
        &fields,
    ));
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_cg_native_execution_plan_blocker_for_bundle(
    report: &mut CapabilityReport,
    bundle: &trust_ir::NativeVerificationBundle,
    semantic_bundle: &trust_ir::NativeVerificationBundle,
    bundle_source: &'static str,
    bundle_validated: bool,
    state_bytes: u64,
    cache: &PetriKernelPlanCache,
    installed_artifact: PetriNativeInstalledArtifactEvidenceRef<'_>,
    gate: NativeJitFailClosedGate,
) -> PetriNativeRouteSelection {
    let identity = bundle.transport_identity();
    let transport_digest = identity.stable_digest();
    let semantic_bridge_authority = add_petri_native_successor_semantic_bridge_evidence(
        report,
        cache,
        semantic_bundle,
        bundle_source,
        bundle_validated,
    );
    let ay_native_bundle_facade = add_ay_trust_mc_native_verification_bundle_facade_evidence(
        report,
        semantic_bundle,
        bundle_source,
        bundle_validated,
    );
    let mut expected = tla_trust_cg::PetriNativeSuccessorExecutionExpected::canary_callable(
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
        state_bytes,
    );
    let target_abi_proof_digest = identity
        .target_abi
        .as_ref()
        .map(|target_abi| target_abi.digest);
    if let Some(target_abi_digest) = target_abi_proof_digest {
        expected = expected.with_target_abi_digest(target_abi_digest);
    }
    let initial_plan =
        tla_trust_cg::petri_native_successor_execution_plan_from_trust_ir_bundle(bundle, expected);
    let initial_callable_contract = initial_plan.callable_contract.as_ref();
    let initial_trampoline_contract = initial_plan.trampoline_contract.clone();
    let (
        trust_ir_bundle_consumed,
        trust_ir_consumption_status,
        trust_ir_consumption_entries,
        consumed_certificates,
        artifact_count,
        validation_errors,
    ) = match bundle.native_evidence_consumption_report() {
        Ok(consumption) => {
            let artifact_count = consumption
                .entries
                .iter()
                .map(|entry| entry.artifacts.len())
                .sum::<usize>();
            let status = if consumption.entries.is_empty() {
                "missing_native_evidence"
            } else {
                "available"
            };
            (
                true,
                status,
                consumption.entries.len(),
                consumption.consumed_certificate_count(),
                artifact_count,
                0,
            )
        }
        Err(errors) => (false, "validation_failed", 0, 0, 0, errors.len()),
    };
    let evidence_profile = trust_cg_petri_native_evidence_profile(bundle);
    let target_abi_digest = identity.target_abi.as_ref().map_or_else(
        || "none".to_owned(),
        |target_abi| target_abi.digest.to_string(),
    );
    let call_packet_surface = trust_cg_petri_call_packet_surface();
    let compile_artifact_handoff_surface = trust_cg_petri_compile_artifact_handoff_surface();
    let runtime_readiness_surface = trust_cg_petri_runtime_readiness_surface();
    let TrustCgPetriCompileArtifactHandoffAttempt {
        evidence: compile_artifact_handoff,
        installed_artifact_available: compile_artifact_handoff_installed_artifact_available,
        real_artifact_source: compile_artifact_handoff_real_artifact_source,
        entry_symbol_source: compile_artifact_handoff_entry_symbol_source,
        native_payload_source: compile_artifact_handoff_native_payload_source,
        ty_wiring_status: compile_artifact_handoff_ty_wiring_status,
        ty_wiring_blocker: compile_artifact_handoff_ty_wiring_blocker,
        ty_required_field: compile_artifact_handoff_ty_required_field,
        missing_ty_artifact_field: compile_artifact_handoff_missing_ty_artifact_field,
        missing_trust_cg_artifact_field: compile_artifact_handoff_missing_trust_cg_artifact_field,
        missing_artifact_blocker: compile_artifact_handoff_missing_artifact_blocker,
        next_production_api: compile_artifact_handoff_next_production_api,
        next_production_input: compile_artifact_handoff_next_production_input,
        next_production_reason_code: compile_artifact_handoff_next_production_reason_code,
    } = trust_cg_petri_compile_artifact_handoff_attempt(
        initial_callable_contract,
        installed_artifact.artifact,
        installed_artifact.lookup_entry_symbol,
    );
    let runtime_inputs = trust_cg_petri_runtime_readiness_inputs(
        bundle,
        state_bytes,
        target_abi_proof_digest,
        &compile_artifact_handoff,
    );
    let trampoline_contract = runtime_inputs
        .trampoline_contract
        .as_ref()
        .or(initial_trampoline_contract.as_ref());
    let plan = match (
        compile_artifact_handoff.native_payload_sha256.as_deref(),
        trampoline_contract,
        runtime_inputs.install_packet.as_ref(),
    ) {
        (Some(native_payload_sha256), Some(trampoline_contract), Some(install_packet)) => {
            tla_trust_cg::petri_native_successor_execution_plan_from_trust_ir_bundle(
                bundle,
                expected
                    .with_native_payload_sha256(native_payload_sha256)
                    .with_trampoline_contract(trampoline_contract)
                    .with_native_install_gate_packet(install_packet),
            )
        }
        _ => initial_plan,
    };
    let summary = &plan.admission_summary;
    let reason_code = plan.reason_code.unwrap_or("none");
    let callable_contract = plan.callable_contract.as_ref();
    let callable_target_abi_digest = callable_contract
        .and_then(|contract| contract.target_abi_digest.as_deref())
        .unwrap_or("none");
    let runtime_entry_symbol = compile_artifact_handoff
        .entry_symbol
        .as_deref()
        .unwrap_or(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL);
    let runtime_generation = compile_artifact_handoff.current_generation.unwrap_or(0);
    let runtime_readiness = match installed_artifact.artifact {
        Some(artifact) => artifact.petri_native_successor_runtime_readiness_packet(
            installed_artifact
                .lookup_entry_symbol
                .or(Some(runtime_entry_symbol)),
            runtime_inputs.install_packet.as_ref(),
            trampoline_contract,
            runtime_inputs.call_packet.as_ref(),
            None,
        ),
        None => tla_trust_cg::petri_native_successor_runtime_readiness_packet(
            runtime_inputs.call_packet.as_ref(),
            runtime_inputs.install_packet.as_ref(),
            trampoline_contract,
            None,
            None,
            runtime_generation,
        ),
    };
    let runtime_readiness_source = if installed_artifact.artifact.is_some() {
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_INSTALLED_ARTIFACT_API
    } else {
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_API
    };
    let runtime_readiness_status_code = runtime_readiness.status.as_str();
    let runtime_readiness_reason_code = runtime_readiness.reason_code.unwrap_or("none");
    let runtime_readiness_blocker_stage = runtime_readiness.blocker_stage.unwrap_or("none");
    let runtime_readiness_required_evidence = runtime_readiness.required_evidence.unwrap_or("none");
    let trampoline_sha256 = trampoline_contract
        .map(|contract| contract.trampoline_sha256.as_str())
        .unwrap_or("none");
    let downstream_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
    let compile_artifact_handoff_status_code = compile_artifact_handoff.status.as_str();
    let compile_artifact_handoff_reason_code =
        compile_artifact_handoff.reason_code.unwrap_or("none");
    let compile_artifact_handoff_blocker_code = compile_artifact_handoff
        .blocker
        .map(|blocker| blocker.as_str())
        .unwrap_or("none");
    let compile_artifact_handoff_required_field =
        compile_artifact_handoff.required_field.unwrap_or("none");
    let compile_artifact_handoff_required_evidence =
        compile_artifact_handoff.required_evidence.unwrap_or("none");
    let compile_artifact_handoff_status_in_downstream_contract = downstream_contract
        .compile_artifact_handoff
        .status_codes
        .contains(&compile_artifact_handoff_status_code);
    let compile_artifact_handoff_blocker_in_downstream_contract =
        compile_artifact_handoff.blocker.map_or(true, |blocker| {
            downstream_contract
                .compile_artifact_handoff
                .blocker_codes
                .contains(&blocker.as_str())
        });
    let runtime_readiness_blocker_code = runtime_readiness
        .blocker
        .map(|blocker| blocker.as_str())
        .unwrap_or("none");
    let runtime_readiness_status_in_downstream_contract = downstream_contract
        .runtime_readiness
        .status_codes
        .contains(&runtime_readiness_status_code);
    let runtime_readiness_blocker_in_downstream_contract =
        runtime_readiness.blocker.map_or(true, |blocker| {
            downstream_contract
                .runtime_readiness
                .blocker_codes
                .contains(&blocker.as_str())
        });
    let execution_authority = tla_trust_cg::petri_native_successor_execution_authority_decision(
        tla_trust_cg::PetriNativeSuccessorExecutionAuthorityInput::default()
            .with_compile_artifact_handoff(&compile_artifact_handoff)
            .with_runtime_readiness(&runtime_readiness),
    );
    let execution_authority_status_code = execution_authority.status.as_str();
    let execution_authority_reason_code = execution_authority.reason_code.unwrap_or("none");
    let execution_authority_source_reason_code =
        execution_authority.source_reason_code.unwrap_or("none");
    let execution_authority_required_field = execution_authority.required_field.unwrap_or("none");
    let execution_authority_required_evidence =
        execution_authority.required_evidence.unwrap_or("none");
    let execution_authority_status_in_downstream_contract = downstream_contract
        .execution_authority
        .status_codes
        .contains(&execution_authority_status_code);
    let execution_authority_blocker_in_downstream_contract =
        execution_authority.reason_code.map_or(true, |reason_code| {
            downstream_contract
                .execution_authority
                .blocker_codes
                .contains(&reason_code)
        });
    let execution_authority_manifest_rows = execution_authority.manifest_rows();
    let execution_authority_manifest_validation = execution_authority.manifest_validation_report();
    let execution_authority_summary = execution_authority.compact_authority_summary();
    let execution_authority_summary_rows = execution_authority_summary.manifest_rows();
    let execution_authority_summary_validation =
        tla_trust_cg::validate_petri_native_successor_execution_authority_summary_rows(
            &execution_authority_summary_rows,
            &execution_authority_manifest_rows,
        );
    let production_selection = tla_trust_cg::petri_native_successor_production_selection_decision(
        &execution_authority,
        runtime_inputs.call_packet.as_ref(),
    );
    let production_selection_status_code = production_selection.status.as_str();
    let production_selection_reason_code = production_selection.reason_code.unwrap_or("none");
    let production_selection_source_reason_code =
        production_selection.source_reason_code.unwrap_or("none");
    let production_selection_required_evidence =
        production_selection.required_evidence.unwrap_or("none");
    let production_selection_selected = production_selection.is_selected_for_native_execution();
    let production_selection_rows = production_selection.manifest_rows();
    let runtime_readiness_blocked = !runtime_readiness.ready_for_runtime_call;
    let callable_receipt_available = compile_artifact_handoff.is_ready()
        && runtime_readiness.is_ready_for_runtime_call()
        && runtime_readiness.install_packet_hash.is_some()
        && runtime_readiness.call_packet_available
        && runtime_readiness.callable_pointer.is_some()
        && execution_authority.is_authorized_for_execution()
        && production_selection_selected;
    let callable_receipt_status_code = if callable_receipt_available {
        PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_ACCEPTED
    } else {
        PETRI_NATIVE_CALLABLE_RECEIPT_STATUS_MISSING
    };
    let callable_receipt_reason_code = if callable_receipt_available {
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    } else if !compile_artifact_handoff.is_ready() {
        compile_artifact_handoff_reason_code
    } else if !runtime_readiness.is_ready_for_runtime_call() {
        runtime_readiness_reason_code
    } else if runtime_readiness.install_packet_hash.is_none() {
        runtime_readiness_reason_code
    } else if !runtime_readiness.call_packet_available {
        runtime_readiness_reason_code
    } else if runtime_readiness.callable_pointer.is_none() {
        runtime_readiness_reason_code
    } else if !execution_authority.is_authorized_for_execution() {
        execution_authority_reason_code
    } else if !production_selection_selected {
        production_selection_reason_code
    } else {
        PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT
    };
    let ay_native_production_accepted = ay_native_bundle_facade.is_accepted_for_native_production();
    let native_successor_next_production = if compile_artifact_handoff.is_ready() {
        if !semantic_bridge_authority.represented {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_SEMANTIC_SUCCESSOR_BRIDGE,
                api: semantic_bridge_authority.api,
                input: semantic_bridge_authority.formula_schema,
                evidence: semantic_bridge_authority.required_evidence,
                reason_code: semantic_bridge_authority.reason_code,
                status_code: semantic_bridge_authority.status_code,
                blocker_stage:
                    TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_SEMANTIC_SUCCESSOR_BRIDGE,
                blocker_code: semantic_bridge_authority.reason_code,
            }
        } else if runtime_readiness.is_ready_for_runtime_call()
            && !execution_authority.is_authorized_for_execution()
        {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_EXECUTION_AUTHORITY,
                api: TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_API,
                input: execution_authority_required_field,
                evidence: execution_authority_required_evidence,
                reason_code: execution_authority_reason_code,
                status_code: execution_authority_status_code,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_EXECUTION_AUTHORITY,
                blocker_code: execution_authority_reason_code,
            }
        } else if runtime_readiness.is_ready_for_runtime_call() && !production_selection_selected {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PRODUCTION_SELECTION,
                api: TRUST_CG_PETRI_NATIVE_PRODUCTION_SELECTION_API,
                input: production_selection_required_evidence,
                evidence: production_selection_required_evidence,
                reason_code: production_selection_reason_code,
                status_code: production_selection_status_code,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PRODUCTION_SELECTION,
                blocker_code: production_selection_reason_code,
            }
        } else if runtime_readiness.is_ready_for_runtime_call() && !gate.parity_enabled {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PARITY_GATE,
                api: ENABLE_TRANSITION_PARITY_ENV,
                input: ENABLE_TRANSITION_PARITY_ENV,
                evidence: PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
                reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY,
                status_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PARITY_GATE,
                blocker_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY,
            }
        } else if runtime_readiness.is_ready_for_runtime_call() && !gate.parity_receipt_available {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PARITY_RECEIPT,
                api: PETRI_NATIVE_PARITY_RECEIPT_GATE_API,
                input: PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
                evidence: PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
                reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT,
                status_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_PARITY_RECEIPT,
                blocker_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT,
            }
        } else if runtime_readiness.is_ready_for_runtime_call()
            && !gate.validation_receipt_available
        {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_VALIDATION_RECEIPT,
                api: PETRI_NATIVE_VALIDATION_RECEIPT_GATE_API,
                input: VALIDATION_RECEIPT_SCHEMA,
                evidence: PETRI_NATIVE_VALIDATION_RECEIPT_REQUIRED_EVIDENCE,
                reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT,
                status_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_VALIDATION_RECEIPT,
                blocker_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT,
            }
        } else if runtime_readiness.is_ready_for_runtime_call() && !callable_receipt_available {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_CALLABLE_RECEIPT,
                api: PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API,
                input: PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA,
                evidence: PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE,
                reason_code: callable_receipt_reason_code,
                status_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_CALLABLE_RECEIPT,
                blocker_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
            }
        } else if runtime_readiness.is_ready_for_runtime_call()
            && !PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE
        {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_RUNTIME_READINESS,
                api: TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_API,
                input: TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_API,
                evidence: TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_BLOCKER,
                reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL,
                status_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_PRODUCTION_STATUS_BLOCKED,
                blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_RUNTIME_READINESS,
                blocker_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL,
            }
        } else if runtime_readiness.is_ready_for_runtime_call() {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_RUNTIME_READINESS,
                api: runtime_readiness_surface.api,
                input: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                evidence: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
                status_code: runtime_readiness_status_code,
                blocker_stage: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
                blocker_code: TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_SOURCE_NONE,
            }
        } else {
            TrustCgPetriNextProductionBlocker {
                source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_RUNTIME_READINESS,
                api: runtime_readiness_surface.api,
                input: runtime_readiness_required_evidence,
                evidence: runtime_readiness_required_evidence,
                reason_code: runtime_readiness_reason_code,
                status_code: runtime_readiness_status_code,
                blocker_stage: runtime_readiness_blocker_stage,
                blocker_code: runtime_readiness_blocker_code,
            }
        }
    } else {
        TrustCgPetriNextProductionBlocker {
            source: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_COMPILE_ARTIFACT_HANDOFF,
            api: compile_artifact_handoff_next_production_api,
            input: compile_artifact_handoff_next_production_input,
            evidence: compile_artifact_handoff_required_evidence,
            reason_code: compile_artifact_handoff_next_production_reason_code,
            status_code: compile_artifact_handoff_status_code,
            blocker_stage: TRUST_CG_PETRI_NATIVE_NEXT_PRODUCTION_SOURCE_COMPILE_ARTIFACT_HANDOFF,
            blocker_code: compile_artifact_handoff_blocker_code,
        }
    };
    let producer_production_selection = production_selection_selected
        && compile_artifact_handoff.is_ready()
        && semantic_bridge_authority.represented
        && runtime_readiness.is_ready_for_runtime_call()
        && execution_authority.is_authorized_for_execution();
    let production_selected = producer_production_selection
        && gate.parity_enabled
        && gate.parity_receipt_available
        && gate.validation_receipt_available
        && callable_receipt_available
        && PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE;
    let production_fail_closed = !production_selected;
    let route_producer_admission_reason_code = if summary.disposition == "installable" {
        reason_code
    } else if compile_artifact_handoff.is_ready()
        && native_successor_next_production.reason_code != TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    {
        native_successor_next_production.reason_code
    } else {
        reason_code
    };
    let route_production_selection_reason_code = if producer_production_selection {
        production_selection_reason_code
    } else {
        native_successor_next_production.reason_code
    };
    let downstream_runtime_required_fields = downstream_contract
        .runtime_readiness
        .required_fields
        .join(",");
    let downstream_runtime_status_codes =
        downstream_contract.runtime_readiness.status_codes.join(",");
    let downstream_execution_authority_required_fields = downstream_contract
        .execution_authority
        .required_fields
        .join(",");
    let downstream_execution_authority_status_codes = downstream_contract
        .execution_authority
        .status_codes
        .join(",");
    let downstream_compile_artifact_handoff_required_fields = downstream_contract
        .compile_artifact_handoff
        .required_fields
        .join(",");
    let downstream_compile_artifact_handoff_status_codes = downstream_contract
        .compile_artifact_handoff
        .status_codes
        .join(",");
    let downstream_mock_required_fields = downstream_contract
        .mock_executable_call
        .required_fields
        .join(",");
    let downstream_mock_status_codes = downstream_contract
        .mock_executable_call
        .status_codes
        .join(",");
    let downstream_trust_ir_bundle_identity_provided_fields = downstream_contract
        .trust_ir_native_bundle_identity
        .provided_fields
        .join(",");
    let downstream_trust_ir_bundle_identity_digest_contexts = downstream_contract
        .trust_ir_native_bundle_identity
        .digest_contexts
        .join(",");
    let downstream_trust_ir_bundle_identity_external_fields = downstream_contract
        .trust_ir_native_bundle_identity
        .external_fields
        .join(",");
    let install_packet_available = runtime_readiness.install_packet_hash.is_some();
    let install_packet_reason_code = if install_packet_available {
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    } else {
        runtime_readiness_reason_code
    };
    let callable_authorized = plan.callable_authorized || runtime_readiness.ready_for_runtime_call;
    let callable_authorized_reason_code = if callable_authorized {
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    } else {
        runtime_readiness_reason_code
    };
    let call_packet_available = runtime_readiness.call_packet_available;
    let call_packet_reason_code = if call_packet_available {
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    } else {
        runtime_readiness_reason_code
    };
    let callable_pointer_available = runtime_readiness.callable_pointer.is_some();
    let callable_pointer_reason_code = if callable_pointer_available {
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
    } else {
        runtime_readiness_reason_code
    };
    let callable_handoff_available = true;
    let callable_handoff_reason_code = TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE;
    let call_packet_api_status = TrustCgPetriNativeReadinessStatus::Available;
    let execution_plan_status = TrustCgPetriNativeReadinessStatus::Available;
    let install_packet_status = if install_packet_available {
        TrustCgPetriNativeReadinessStatus::Available
    } else {
        TrustCgPetriNativeReadinessStatus::Missing
    };
    let concrete_callable_pointer_status = if callable_pointer_available {
        TrustCgPetriNativeReadinessStatus::Available
    } else {
        TrustCgPetriNativeReadinessStatus::Missing
    };
    let concrete_callable_packet_status = if call_packet_available {
        TrustCgPetriNativeReadinessStatus::Available
    } else {
        TrustCgPetriNativeReadinessStatus::Missing
    };

    let mut fields = Vec::new();
    push_trust_cg_native_admission_field(
        &mut fields,
        "source",
        "PetriNativeSuccessorExecutionPlan",
    );
    push_trust_cg_native_admission_field(&mut fields, "schema", plan.schema);
    push_trust_cg_native_admission_field(&mut fields, "schema_version", plan.schema_version);
    push_trust_cg_native_admission_field(&mut fields, "consumer", &summary.consumer);
    push_trust_cg_native_admission_field(&mut fields, "consumer_mode", &summary.consumer_mode);
    push_trust_cg_native_admission_field(&mut fields, "kind", TRUST_CG_PETRI_NATIVE_ADMISSION_KIND);
    push_trust_cg_native_admission_field(&mut fields, "surface", summary.surface);
    push_trust_cg_native_admission_field(&mut fields, "disposition", summary.disposition);
    push_trust_cg_native_admission_field(&mut fields, "status_code", summary.disposition);
    push_trust_cg_native_admission_field(&mut fields, "rejection_code", reason_code);
    push_trust_cg_native_admission_field(&mut fields, "reason_code", reason_code);
    push_trust_cg_native_admission_field(
        &mut fields,
        "requested_authority",
        summary.requested_authority,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_authority",
        summary.install_authority,
    );
    push_trust_cg_native_admission_field(&mut fields, "cargo_dependency", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_plan_api",
        TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "expected_api",
        TRUST_CG_PETRI_NATIVE_EXECUTION_EXPECTED_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trampoline_contract_api",
        TRUST_CG_PETRI_NATIVE_TRAMPOLINE_CONTRACT_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_packet_api",
        TRUST_CG_PETRI_NATIVE_INSTALL_PACKET_API,
    );
    push_trust_cg_native_admission_field(&mut fields, "call_packet_api", call_packet_surface.api);
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_schema",
        call_packet_surface.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_schema_version",
        call_packet_surface.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_type",
        call_packet_surface.call_packet_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_pointer_type",
        call_packet_surface.callable_pointer_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_required_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_REQUIRED_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_current_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_descriptor_available",
        call_packet_surface.descriptor_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_descriptor_source",
        call_packet_surface.descriptor_source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_descriptor_status_code",
        call_packet_surface.descriptor_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_descriptor_authoritative",
        call_packet_surface.descriptor_authoritative,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_descriptor_dependency",
        call_packet_surface.descriptor_dependency,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_descriptor_upstream_ask",
        call_packet_surface.descriptor_upstream_ask,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_contract_api",
        TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_contract_schema",
        downstream_contract.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_contract_schema_version",
        downstream_contract.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_contract_required_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_REQUIRED_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_contract_current_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_api",
        TRUST_CG_PETRI_NATIVE_TRUST_IR_BUNDLE_IDENTITY_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_required_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_TRUST_IR_BUNDLE_IDENTITY_REQUIRED_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_schema",
        downstream_contract.trust_ir_native_bundle_identity.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_schema_version",
        downstream_contract
            .trust_ir_native_bundle_identity
            .schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_schema_version",
        downstream_contract
            .trust_ir_native_bundle_identity
            .bundle_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_transport_identity_schema",
        downstream_contract
            .trust_ir_native_bundle_identity
            .transport_identity_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_transport_identity_schema_version",
        downstream_contract
            .trust_ir_native_bundle_identity
            .transport_identity_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_provided_fields",
        downstream_trust_ir_bundle_identity_provided_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_digest_contexts",
        downstream_trust_ir_bundle_identity_digest_contexts.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_trust_ir_bundle_identity_external_fields",
        downstream_trust_ir_bundle_identity_external_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_compile_artifact_handoff_surface",
        downstream_contract.compile_artifact_handoff.name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_compile_artifact_handoff_required_fields",
        downstream_compile_artifact_handoff_required_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_compile_artifact_handoff_status_codes",
        downstream_compile_artifact_handoff_status_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_compile_artifact_handoff_blocker_codes_count",
        downstream_contract
            .compile_artifact_handoff
            .blocker_codes
            .len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_runtime_readiness_surface",
        downstream_contract.runtime_readiness.name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_runtime_readiness_required_fields",
        downstream_runtime_required_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_runtime_readiness_status_codes",
        downstream_runtime_status_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_runtime_readiness_blocker_codes_count",
        downstream_contract.runtime_readiness.blocker_codes.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_execution_authority_surface",
        downstream_contract.execution_authority.name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_execution_authority_required_fields",
        downstream_execution_authority_required_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_execution_authority_status_codes",
        downstream_execution_authority_status_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_execution_authority_blocker_codes_count",
        downstream_contract.execution_authority.blocker_codes.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_mock_executable_call_surface",
        downstream_contract.mock_executable_call.name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_mock_executable_call_required_fields",
        downstream_mock_required_fields.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_mock_executable_call_status_codes",
        downstream_mock_status_codes.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "downstream_mock_executable_call_blocker_codes_count",
        downstream_contract.mock_executable_call.blocker_codes.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_api",
        runtime_readiness_surface.api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_installed_artifact_api",
        runtime_readiness_surface.installed_artifact_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_installed_artifact_required_trust_cg_rev",
        runtime_readiness_surface.installed_artifact_required_trust_cg_rev,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_source",
        runtime_readiness_source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_installed_artifact_available",
        installed_artifact.artifact.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_schema",
        runtime_readiness_surface.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_schema_version",
        runtime_readiness_surface.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_packet_type",
        runtime_readiness_surface.packet_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_required_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_REQUIRED_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_current_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_CURRENT_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(&mut fields, "runtime_readiness_packet_available", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_packet_sha256",
        runtime_readiness.runtime_readiness_packet_sha256.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_status_code",
        runtime_readiness_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_status_in_downstream_contract",
        runtime_readiness_status_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_ready_for_runtime_call",
        runtime_readiness.ready_for_runtime_call,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_reason_code",
        runtime_readiness_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_blocker_code",
        runtime_readiness_blocker_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_blocker_in_downstream_contract",
        runtime_readiness_blocker_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_blocker_stage",
        runtime_readiness_blocker_stage,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_required_evidence",
        runtime_readiness_required_evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_manifest_identity_ready",
        runtime_readiness.manifest_identity_ready,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_install_binding_ready",
        runtime_readiness.install_binding_ready,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_executable_call_ready",
        runtime_readiness.executable_call_ready,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_call_packet_available",
        runtime_readiness.call_packet_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "runtime_readiness_callable_pointer_available",
        runtime_readiness.callable_pointer.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_api",
        TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_schema",
        execution_authority.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_schema_version",
        execution_authority.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_status_code",
        execution_authority_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_status_in_downstream_contract",
        execution_authority_status_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_authorized_for_execution",
        execution_authority.authorized_for_execution,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_is_authorized_for_execution",
        execution_authority.is_authorized_for_execution(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_reason_code",
        execution_authority_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_source_reason_code",
        execution_authority_source_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_blocker_in_downstream_contract",
        execution_authority_blocker_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_required_field",
        execution_authority_required_field,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_required_evidence",
        execution_authority_required_evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_compile_artifact_handoff_sha256",
        execution_authority
            .compile_artifact_handoff_sha256
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_runtime_readiness_packet_sha256",
        execution_authority
            .runtime_readiness_packet_sha256
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_compile_artifact_handoff_hash_current",
        execution_authority.compile_artifact_handoff_hash_current,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_runtime_readiness_packet_hash_current",
        execution_authority.runtime_readiness_packet_hash_current,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_callable_authorized",
        execution_authority.callable_authorized,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_ready_for_runtime_call",
        execution_authority.ready_for_runtime_call,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_runtime_authorizes_useful_native",
        execution_authority.runtime_authorizes_useful_native,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_sha256",
        execution_authority.execution_authority_sha256.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_api",
        TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_MANIFEST_VALIDATION_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_schema",
        execution_authority_manifest_validation.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_schema_version",
        execution_authority_manifest_validation.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_status_code",
        execution_authority_manifest_validation.status.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_accepted",
        execution_authority_manifest_validation.is_accepted(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_fail_closed",
        execution_authority_manifest_validation.is_fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_validation_reason_code",
        execution_authority_manifest_validation
            .reason_code
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_manifest_row_count",
        execution_authority_manifest_rows.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_api",
        TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_SUMMARY_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_schema",
        execution_authority_summary.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_schema_version",
        execution_authority_summary.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_api",
        TRUST_CG_PETRI_NATIVE_EXECUTION_AUTHORITY_SUMMARY_VALIDATION_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_schema",
        execution_authority_summary_validation.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_schema_version",
        execution_authority_summary_validation.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_status_code",
        execution_authority_summary_validation.status.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_accepted",
        execution_authority_summary_validation.is_accepted(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_fail_closed",
        execution_authority_summary_validation.is_fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_validation_reason_code",
        execution_authority_summary_validation
            .reason_code
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_status_code",
        execution_authority_summary.validation_status.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_reason_code",
        execution_authority_summary
            .validation_reason_code
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_fail_closed",
        execution_authority_summary.is_fail_closed(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_accepted",
        execution_authority_summary.is_accepted(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_row_count",
        execution_authority_summary_rows.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_summary_sha256",
        execution_authority_summary.summary_sha256.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_authority_replay_identity_sha256",
        execution_authority_summary.replay_identity_sha256.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_api",
        TRUST_CG_PETRI_NATIVE_PRODUCTION_SELECTION_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_schema",
        production_selection.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_schema_version",
        production_selection.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_status_code",
        production_selection_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_selected_for_native_execution",
        production_selection.selected_for_native_execution,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_is_selected_for_native_execution",
        production_selection_selected,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_fail_closed",
        production_selection.fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_reason_code",
        production_selection_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_source_reason_code",
        production_selection_source_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_required_evidence",
        production_selection_required_evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_execution_authority_sha256",
        production_selection.execution_authority_sha256.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_execution_authority_hash_current",
        production_selection.execution_authority_hash_current,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_call_packet_sha256",
        production_selection
            .call_packet_sha256
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_call_packet_hash_current",
        production_selection.call_packet_hash_current,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_callable_lane_admitted",
        production_selection.callable_lane_admitted,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_runtime_ready_for_call",
        production_selection.runtime_ready_for_call,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_runtime_authorizes_useful_native",
        production_selection.runtime_authorizes_useful_native,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_vector_constant_lowering_schema",
        production_selection.vector_constant_lowering_evidence_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_vector_constant_lowering_schema_version",
        production_selection.vector_constant_lowering_evidence_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_vector_constant_lowering_status_code",
        production_selection.vector_constant_lowering_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_vector_constant_lowering_supported",
        production_selection.vector_constant_lowering_supported,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_sha256",
        production_selection.production_selection_sha256.as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "production_selection_row_count",
        production_selection_rows.len(),
    );
    push_trust_cg_native_admission_field(&mut fields, "parity_env", ENABLE_TRANSITION_PARITY_ENV);
    push_trust_cg_native_admission_field(&mut fields, "parity_enabled", gate.parity_enabled);
    push_trust_cg_native_admission_field(&mut fields, "parity_receipt_required", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_schema",
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_schema_version",
        PETRI_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_gate_api",
        PETRI_NATIVE_PARITY_RECEIPT_GATE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_required_evidence",
        PETRI_NATIVE_PARITY_RECEIPT_REQUIRED_EVIDENCE,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_available",
        gate.parity_receipt_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_status_code",
        gate.parity_receipt_status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "parity_receipt_reason_code",
        gate.parity_receipt_reason_code(),
    );
    push_trust_cg_native_admission_field(&mut fields, "validation_receipt_required", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_schema",
        VALIDATION_RECEIPT_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_schema_version",
        VALIDATION_RECEIPT_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_gate_api",
        PETRI_NATIVE_VALIDATION_RECEIPT_GATE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_required_evidence",
        PETRI_NATIVE_VALIDATION_RECEIPT_REQUIRED_EVIDENCE,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_available",
        gate.validation_receipt_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_status_code",
        gate.validation_receipt_status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "validation_receipt_reason_code",
        gate.validation_receipt_reason_code(),
    );
    push_trust_cg_native_admission_field(&mut fields, "callable_receipt_required", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_schema",
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_schema_version",
        PETRI_NATIVE_CALLABLE_RECEIPT_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_gate_api",
        PETRI_NATIVE_CALLABLE_RECEIPT_GATE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_required_evidence",
        PETRI_NATIVE_CALLABLE_RECEIPT_REQUIRED_EVIDENCE,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_available",
        callable_receipt_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_status_code",
        callable_receipt_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_receipt_reason_code",
        callable_receipt_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_api",
        runtime_readiness_surface.mock_executable_call_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_schema",
        runtime_readiness_surface.mock_executable_call_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_schema_version",
        runtime_readiness_surface.mock_executable_call_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_role",
        runtime_readiness_surface.mock_executable_call_role,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_descriptor_available",
        runtime_readiness_surface.mock_executable_call_descriptor_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_descriptor_authoritative",
        runtime_readiness_surface.mock_executable_call_descriptor_authoritative,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_descriptor_source",
        runtime_readiness_surface.mock_executable_call_descriptor_source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_descriptor_name",
        runtime_readiness_surface.mock_executable_call_descriptor_name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_gate_kind",
        runtime_readiness_surface.mock_executable_call_gate_kind,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_gate_enabled",
        runtime_readiness_surface.mock_executable_call_gate_enabled,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "mock_executable_call_production_enabled",
        runtime_readiness_surface.mock_executable_call_gate_enabled,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_api",
        compile_artifact_handoff_surface.api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_schema",
        compile_artifact_handoff_surface.schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_schema_version",
        compile_artifact_handoff_surface.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_input_type",
        compile_artifact_handoff_surface.input_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_evidence_type",
        compile_artifact_handoff_surface.evidence_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_blocker_type",
        compile_artifact_handoff_surface.blocker_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_required_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_REQUIRED_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_current_trust_cg_rev",
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_CURRENT_TRUST_CG_REV,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_api",
        compile_artifact_handoff_surface.installed_artifact_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_type",
        compile_artifact_handoff_surface.installed_artifact_type,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_required_trust_cg_rev",
        compile_artifact_handoff_surface.installed_artifact_required_trust_cg_rev,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_native_library_bridge_api",
        TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_NATIVE_LIBRARY_BRIDGE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_production_status",
        installed_artifact.status,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_production_reason_code",
        installed_artifact.reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_production_path",
        installed_artifact.production_path,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_production_missing_api",
        installed_artifact.missing_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_production_blocker",
        installed_artifact.blocker,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_production_upstream_ask",
        installed_artifact.upstream_ask,
    );
    push_trust_cg_native_admission_field(&mut fields, "compile_artifact_handoff_available", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_installed_artifact_available",
        compile_artifact_handoff_installed_artifact_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_ty_wiring_status",
        compile_artifact_handoff_ty_wiring_status,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_ty_wiring_blocker",
        compile_artifact_handoff_ty_wiring_blocker,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_ty_required_field",
        compile_artifact_handoff_ty_required_field,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_ready",
        compile_artifact_handoff.is_ready(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_status_code",
        compile_artifact_handoff_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_status_in_downstream_contract",
        compile_artifact_handoff_status_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_reason_code",
        compile_artifact_handoff_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_blocker_code",
        compile_artifact_handoff_blocker_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_blocker_in_downstream_contract",
        compile_artifact_handoff_blocker_in_downstream_contract,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_required_field",
        compile_artifact_handoff_required_field,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_required_evidence",
        compile_artifact_handoff_required_evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_sha256",
        compile_artifact_handoff
            .compile_artifact_handoff_sha256
            .as_str(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_population_attempted",
        true,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_real_artifact_source",
        compile_artifact_handoff_real_artifact_source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_entry_symbol_present",
        compile_artifact_handoff.entry_symbol.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_entry_symbol",
        compile_artifact_handoff
            .entry_symbol
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_entry_symbol_source",
        compile_artifact_handoff_entry_symbol_source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_native_payload_present",
        compile_artifact_handoff.native_payload_sha256.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_native_payload_source",
        compile_artifact_handoff_native_payload_source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_callable_pointer_present",
        compile_artifact_handoff.callable_pointer.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_executable_region_present",
        compile_artifact_handoff.executable_region_sha256.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_lifetime_owner_present",
        compile_artifact_handoff.lifetime_owner.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_current_generation_present",
        compile_artifact_handoff.current_generation.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_missing_ty_artifact_field",
        compile_artifact_handoff_missing_ty_artifact_field,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_missing_trust_cg_artifact_field",
        compile_artifact_handoff_missing_trust_cg_artifact_field,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "compile_artifact_handoff_missing_artifact_blocker",
        compile_artifact_handoff_missing_artifact_blocker,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_api",
        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_schema",
        ay_trust_mc_native_bundle::TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_schema_version",
        ay_trust_mc_native_bundle::TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_status_code",
        ay_native_bundle_facade.status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_reason_code",
        ay_native_bundle_facade.reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_consumer_acceptance_api",
        ay_native_bundle_facade.consumer_acceptance_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_consumer_rejection_status_code",
        ay_native_bundle_facade.consumer_rejection_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_consumer_rejection_reason_code",
        ay_native_bundle_facade.consumer_rejection_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_consumer_rejection_code",
        ay_native_bundle_facade.consumer_rejection_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_accepted_for_consumer",
        ay_native_bundle_facade.accepted_for_consumer,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_fail_closed",
        ay_native_bundle_facade.fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_consumer_rejection_fail_closed",
        ay_native_bundle_facade.consumer_rejection_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_consumer_rejection_ready_for_trust_mc_chc_handoff",
        ay_native_bundle_facade.consumer_rejection_ready_for_trust_mc_chc_handoff,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_model_validated",
        ay_native_bundle_facade.model_validated,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_verification_level_code",
        ay_native_bundle_facade.verification_level_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_proof_replay_status_code",
        ay_native_bundle_facade.proof_replay_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_ready_for_trust_mc_chc_handoff",
        ay_native_bundle_facade.ready_for_trust_mc_chc_handoff,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_proof_identity_schema",
        ay_native_bundle_facade.semantic_bridge_proof_identity_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_proof_identity_schema_version",
        ay_native_bundle_facade.semantic_bridge_proof_identity_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_proof_identity_digest",
        &ay_native_bundle_facade.semantic_bridge_proof_identity_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_fail_closed",
        ay_native_bundle_facade.semantic_bridge_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_status_code",
        ay_native_bundle_facade.semantic_bridge_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_reason_code",
        ay_native_bundle_facade.semantic_bridge_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_semantic_bridge_evidence_status_code",
        ay_native_bundle_facade.semantic_bridge_evidence_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_matched_trust_mc_request_count",
        ay_native_bundle_facade.matched_trust_mc_request_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_matched_trust_mc_chc_request_count",
        ay_native_bundle_facade.matched_trust_mc_chc_request_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_matched_trust_mc_evidence_count",
        ay_native_bundle_facade.matched_trust_mc_evidence_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_matched_trust_mc_artifact_count",
        ay_native_bundle_facade.matched_trust_mc_artifact_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_api",
        ay_native_bundle_facade.model_acceptance_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_schema",
        ay_trust_mc_native_bundle::TRUST_MC_PETRI_SUCCESSOR_CHC_MODEL_ACCEPTANCE_SCHEMA,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_schema_version",
        ay_trust_mc_native_bundle::TRUST_MC_PETRI_SUCCESSOR_CHC_MODEL_ACCEPTANCE_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_consumer_acceptance_api",
        ay_native_bundle_facade.model_acceptance_consumer_acceptance_api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_status_code",
        ay_native_bundle_facade.model_acceptance_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_reason_code",
        ay_native_bundle_facade.model_acceptance_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_consumer_rejection_status_code",
        ay_native_bundle_facade.model_acceptance_consumer_rejection_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_consumer_rejection_reason_code",
        ay_native_bundle_facade.model_acceptance_consumer_rejection_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_consumer_rejection_fail_closed",
        ay_native_bundle_facade.model_acceptance_consumer_rejection_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_accepted_for_consumer",
        ay_native_bundle_facade.model_acceptance_accepted_for_consumer,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_fail_closed",
        ay_native_bundle_facade.model_acceptance_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_proof_handoff_ready",
        ay_native_bundle_facade.model_acceptance_proof_handoff_ready,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_ready_for_solver_validation",
        ay_native_bundle_facade.model_acceptance_ready_for_solver_validation,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_solver_model_validation_present",
        ay_native_bundle_facade.model_acceptance_solver_model_validation_present,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_solver_model_validation_accepted",
        ay_native_bundle_facade.model_acceptance_solver_model_validation_accepted,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_solver_artifact_bytes_validated",
        ay_native_bundle_facade.model_acceptance_solver_artifact_bytes_validated,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_solver_model_artifact_bytes_digest",
        &ay_native_bundle_facade.model_acceptance_solver_model_artifact_bytes_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_solver_replay_transcript_artifact_bytes_digest",
        &ay_native_bundle_facade.model_acceptance_solver_replay_transcript_artifact_bytes_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_artifact_byte_attachment_count",
        ay_native_bundle_facade.model_acceptance_trust_ir_artifact_byte_attachment_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_artifact_byte_resolution_status_codes",
        &ay_native_bundle_facade.model_acceptance_trust_ir_artifact_byte_resolution_status_codes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_artifact_byte_resolution_reason_codes",
        &ay_native_bundle_facade.model_acceptance_trust_ir_artifact_byte_resolution_reason_codes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_artifact_byte_resolution_authority_codes",
        &ay_native_bundle_facade.model_acceptance_trust_ir_artifact_byte_resolution_authority_codes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_authoritative_artifact_requirement_count",
        ay_native_bundle_facade.model_acceptance_trust_ir_authoritative_artifact_requirement_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_authoritative_artifact_requirement_roles",
        &ay_native_bundle_facade.model_acceptance_trust_ir_authoritative_artifact_requirement_roles,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_unauthoritative_artifact_requirement_roles",
        &ay_native_bundle_facade
            .model_acceptance_trust_ir_unauthoritative_artifact_requirement_roles,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_authoritative_artifact_requirement_kinds",
        &ay_native_bundle_facade.model_acceptance_trust_ir_authoritative_artifact_requirement_kinds,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_unauthoritative_artifact_requirement_kinds",
        &ay_native_bundle_facade
            .model_acceptance_trust_ir_unauthoritative_artifact_requirement_kinds,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_ir_authoritative_artifact_bytes_count",
        ay_native_bundle_facade.model_acceptance_trust_ir_authoritative_artifact_bytes_count,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_status_code",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_reason_code",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_schema",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_schema_version",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_fail_closed",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_name",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_kind_code",
        &ay_native_bundle_facade
            .model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_kind_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_digest",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_model_artifact_name",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_model_artifact_name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_model_artifact_kind_code",
        &ay_native_bundle_facade
            .model_acceptance_trust_mc_chc_proof_handoff_model_artifact_kind_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_proof_handoff_model_artifact_digest",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_proof_handoff_model_artifact_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_status_code",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_model_validation_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_reason_code",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_model_validation_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_schema",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_model_validation_schema,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_schema_version",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_model_validation_schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_fail_closed",
        ay_native_bundle_facade.model_acceptance_trust_mc_chc_model_validation_fail_closed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_model_artifact_name",
        &ay_native_bundle_facade.model_acceptance_trust_mc_chc_model_validation_model_artifact_name,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_model_artifact_kind_code",
        &ay_native_bundle_facade
            .model_acceptance_trust_mc_chc_model_validation_model_artifact_kind_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_model_acceptance_trust_mc_chc_model_validation_model_artifact_digest",
        &ay_native_bundle_facade
            .model_acceptance_trust_mc_chc_model_validation_model_artifact_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "ay_native_bundle_facade_accepted_for_native_production",
        ay_native_production_accepted,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_source",
        native_successor_next_production.source,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_api",
        native_successor_next_production.api,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_input",
        native_successor_next_production.input,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_evidence",
        native_successor_next_production.evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_reason_code",
        native_successor_next_production.reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_status_code",
        native_successor_next_production.status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_blocker_stage",
        native_successor_next_production.blocker_stage,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_next_production_blocker_code",
        native_successor_next_production.blocker_code,
    );
    push_trust_cg_native_admission_field(&mut fields, "call_packet_api_available", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_api_status_code",
        call_packet_api_status.code(),
    );
    push_trust_cg_native_admission_field(&mut fields, "call_packet_type_available", true);
    push_trust_cg_native_admission_field(&mut fields, "callable_pointer_type_available", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "admission_api",
        TRUST_CG_PETRI_NATIVE_ADMISSION_API,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "bundle_api",
        TRUST_CG_PETRI_NATIVE_ADMISSION_BUNDLE_API,
    );
    push_trust_cg_native_admission_field(&mut fields, "bundle_source", bundle_source);
    push_trust_cg_native_admission_field(&mut fields, "bundle_validated", bundle_validated);
    push_trust_cg_native_admission_field(
        &mut fields,
        "entry_function",
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "input_state_bytes",
        expected.input_state_bytes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "output_state_bytes",
        expected.output_state_bytes,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "state_alignment_bytes",
        expected.state_alignment_bytes,
    );
    push_trust_cg_native_admission_field(&mut fields, "execution_plan_available", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_plan_status_code",
        execution_plan_status.code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "execution_plan_reason_code",
        TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_transport_identity_available",
        true,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_bundle_consumed",
        trust_ir_bundle_consumed,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_consumption_status",
        trust_ir_consumption_status,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trust_ir_consumption_entries",
        trust_ir_consumption_entries,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "consumed_certificates",
        consumed_certificates,
    );
    push_trust_cg_native_admission_field(&mut fields, "artifact_count", artifact_count);
    push_trust_cg_native_admission_field(&mut fields, "validation_errors", validation_errors);
    push_trust_cg_native_evidence_profile_fields(&mut fields, &evidence_profile);
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_contract_available",
        callable_contract.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trampoline_contract_available",
        trampoline_contract.is_some(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_packet_available",
        install_packet_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_packet_status_code",
        install_packet_status.code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "install_packet_reason_code",
        install_packet_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_available",
        call_packet_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_reason_code",
        call_packet_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_pointer_available",
        callable_pointer_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_pointer_reason_code",
        callable_pointer_reason_code,
    );
    push_trust_cg_native_admission_field(&mut fields, "concrete_callable_pointer_required", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "concrete_callable_pointer_available",
        callable_pointer_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "concrete_callable_pointer_status_code",
        concrete_callable_pointer_status.code(),
    );
    push_trust_cg_native_admission_field(&mut fields, "concrete_callable_packet_required", true);
    push_trust_cg_native_admission_field(
        &mut fields,
        "concrete_callable_packet_available",
        call_packet_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "concrete_callable_packet_status_code",
        concrete_callable_packet_status.code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_readiness_status_code",
        runtime_readiness_status_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "call_packet_readiness_blocker",
        runtime_readiness_blocker_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_contract_schema",
        callable_contract.map_or("none", |contract| contract.schema),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_contract_schema_version",
        callable_contract.map_or(0, |contract| contract.schema_version),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "contract_entry_function",
        callable_contract.map_or("none", |contract| contract.entry_function.as_str()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "state_encoding",
        callable_contract.map_or("none", |contract| contract.state_encoding),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_contract_sha256",
        callable_contract.map_or("none", |contract| {
            contract.callable_contract_sha256.as_str()
        }),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "contract_artifact_id",
        callable_contract.map_or("none", |contract| contract.artifact_id.as_str()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "contract_source_sha256",
        callable_contract.map_or("none", |contract| contract.source_sha256.as_str()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "contract_trust_ir_sha256",
        callable_contract.map_or("none", |contract| contract.trust_ir_sha256.as_str()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "contract_native_payload_sha256",
        callable_contract.map_or("none", |contract| contract.native_payload_sha256.as_str()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "contract_target_abi_digest",
        callable_target_abi_digest,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trampoline_contract_schema",
        trampoline_contract.map_or("none", |contract| contract.schema),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trampoline_contract_schema_version",
        trampoline_contract.map_or(0, |contract| contract.schema_version),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "trampoline_entry_symbol",
        trampoline_contract.map_or("none", |contract| contract.entry_symbol.as_str()),
    );
    push_trust_cg_native_admission_field(&mut fields, "trampoline_sha256", trampoline_sha256);
    push_trust_cg_native_admission_field(&mut fields, "callable_authorized", callable_authorized);
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_authorized_reason_code",
        callable_authorized_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_handoff_available",
        callable_handoff_available,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_handoff_reason_code",
        callable_handoff_reason_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_handoff_blocker",
        runtime_readiness_blocker_code,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_handoff_required_evidence",
        runtime_readiness_required_evidence,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "callable_handoff_upstream_ask",
        TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_UPSTREAM_ASK,
    );
    push_trust_cg_native_admission_field(&mut fields, "plan_fail_closed", plan.fail_closed);
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_expose_callable",
        summary.actions.expose_callable,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_expose_callable_blocked_by_runtime_readiness",
        runtime_readiness_blocked,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_expose_callable_reason_code",
        if summary.actions.expose_callable {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            runtime_readiness_reason_code
        },
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_ty_native_activate",
        summary.actions.ty_native_activate,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_ty_native_activate_blocked_by_runtime_readiness",
        runtime_readiness_blocked,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actions_ty_native_activate_reason_code",
        if summary.actions.ty_native_activate {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            runtime_readiness_reason_code
        },
    );
    push_trust_cg_native_admission_field(&mut fields, "transport_digest", transport_digest);
    push_trust_cg_native_admission_field(&mut fields, "bundle_digest", identity.bundle_digest);
    push_trust_cg_native_admission_field(&mut fields, "target_abi_digest", target_abi_digest);
    push_trust_cg_native_admission_field(
        &mut fields,
        "request_digests",
        identity.request_digests.len(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "evidence_digests",
        identity.evidence_digests.len(),
    );
    push_trust_cg_native_admission_field(&mut fields, "production_selected", production_selected);
    push_trust_cg_native_admission_field(&mut fields, "fail_closed", production_fail_closed);
    push_trust_cg_native_admission_field(
        &mut fields,
        "native_successor_runtime_status_code",
        runtime_readiness_status_code,
    );

    for summary_row in &execution_authority_summary_rows {
        let mut summary_fields = Vec::new();
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "source",
            "PetriNativeSuccessorExecutionAuthoritySummary",
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "schema",
            execution_authority_summary.schema,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "schema_version",
            execution_authority_summary.schema_version,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_status_code",
            execution_authority_summary.validation_status.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_reason_code",
            execution_authority_summary
                .validation_reason_code
                .as_deref()
                .unwrap_or("none"),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_accepted",
            execution_authority_summary.is_accepted(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_fail_closed",
            execution_authority_summary.is_fail_closed(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "authority_schema",
            execution_authority.schema,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "authority_schema_version",
            execution_authority.schema_version,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "authority_status_code",
            execution_authority_status_code,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "authority_reason_code",
            execution_authority_reason_code,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "authority_authorized_for_execution",
            execution_authority.authorized_for_execution,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "authority_is_authorized_for_execution",
            execution_authority.is_authorized_for_execution(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "manifest_validation_status_code",
            execution_authority_manifest_validation.status.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "manifest_validation_reason_code",
            execution_authority_manifest_validation
                .reason_code
                .as_deref()
                .unwrap_or("none"),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_validation_status_code",
            execution_authority_summary_validation.status.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_validation_reason_code",
            execution_authority_summary_validation
                .reason_code
                .as_deref()
                .unwrap_or("none"),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "row_key",
            summary_row.key.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "row_value",
            summary_row.value.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "manifest_line",
            summary_row.to_key_value_line(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "summary_sha256",
            execution_authority_summary.summary_sha256.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "replay_identity_sha256",
            execution_authority_summary.replay_identity_sha256.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "production_selected",
            production_selected,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "production_selection_status_code",
            production_selection_status_code,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "production_selection_reason_code",
            production_selection_reason_code,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "production_selection_fail_closed",
            production_selection.fail_closed,
        );
        push_trust_cg_native_admission_field(
            &mut summary_fields,
            "fail_closed",
            execution_authority_summary.is_fail_closed(),
        );
        report.add_evidence(render_trust_cg_native_admission_row(
            "trust-cg petri_native_successor_execution_authority_summary",
            &summary_fields,
        ));
    }

    for selection_row in &production_selection_rows {
        let mut selection_fields = Vec::new();
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "source",
            "PetriNativeSuccessorProductionSelectionDecision",
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "schema",
            production_selection.schema,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "schema_version",
            production_selection.schema_version,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_status_code",
            production_selection_status_code,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_reason_code",
            production_selection_reason_code,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_source_reason_code",
            production_selection_source_reason_code,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_required_evidence",
            production_selection_required_evidence,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_selected_for_native_execution",
            production_selection.selected_for_native_execution,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_is_selected_for_native_execution",
            production_selection_selected,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "selection_fail_closed",
            production_selection.fail_closed,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "production_selected",
            production_selected,
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "row_key",
            selection_row.key.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "row_value",
            selection_row.value.as_str(),
        );
        push_trust_cg_native_admission_field(
            &mut selection_fields,
            "manifest_line",
            selection_row.to_key_value_line(),
        );
        report.add_evidence(render_trust_cg_native_admission_row(
            "trust-cg petri_native_successor_production_selection",
            &selection_fields,
        ));
    }

    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-cg petri_native_successor_execution_plan",
        &fields,
    ));

    PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
        transport_identity_available: true,
        producer_admission: summary.disposition == "installable",
        producer_execution_authority: execution_authority.is_authorized_for_execution(),
        producer_production_selection,
        parity_enabled: gate.parity_enabled,
        parity_receipt_available: gate.parity_receipt_available,
        validation_receipt_available: gate.validation_receipt_available,
        callable_receipt_available,
        native_runtime_callable_impl_available: PETRI_NATIVE_RUNTIME_CALLABLE_IMPL_AVAILABLE,
        producer_admission_reason_code: route_producer_admission_reason_code,
        producer_execution_authority_reason_code: execution_authority_reason_code,
        producer_production_selection_reason_code: route_production_selection_reason_code,
        parity_receipt_reason_code: gate.parity_receipt_reason_code(),
        validation_receipt_reason_code: gate.validation_receipt_reason_code(),
        callable_receipt_reason_code,
        runtime_readiness_reason_code,
    })
}

#[cfg(feature = "trust-cg-petri-native")]
fn push_trust_cg_native_admission_field(
    fields: &mut Vec<(String, String)>,
    key: &str,
    value: impl ToString,
) {
    fields.push((key.to_owned(), value.to_string()));
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_ay_strs(values: &[&str]) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_ay_strings(values: &[String]) -> String {
    join_strings_or_none(values)
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_strings_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_ay_u32s(values: &[u32]) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Clone, Copy)]
enum TrustIrPetriTrustMcProvidedField {
    NativeSemanticBridgeReport,
    PetriSuccessorSemanticBridgeReport,
    PetriSuccessorSemanticBridgeConstructor,
    RepresentsPetriSuccessorPlanCacheEquivalence,
    NativeSemanticBridgeProofIdentityKeyValueText,
    NativeSemanticBridgeProofIdentityReplayReportForKeyValueText,
    NativeSemanticBridgeProofIdentityReplayComponentHealthSummaryKeyValueText,
    PetriSuccessorTrustMcChcProofEvidenceIdentityKeyValueText,
    PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReportForKeyValueText,
    PetriSuccessorTrustMcChcProofEvidenceIdentityReplayComponentHealthSummaryKeyValueText,
    ContractDescriptor,
    ArtifactIdentity,
    ArtifactByteResolution,
    ArtifactAuthority,
    AuthoritativeBytes,
}

#[cfg(feature = "trust-cg-petri-native")]
impl TrustIrPetriTrustMcProvidedField {
    fn needle(self) -> &'static str {
        match self {
            Self::NativeSemanticBridgeReport => {
                "NativeVerificationBundle::native_semantic_bridge_report()"
            }
            Self::PetriSuccessorSemanticBridgeReport => {
                "NativeVerificationBundle::petri_successor_semantic_bridge_report()"
            }
            Self::PetriSuccessorSemanticBridgeConstructor => {
                "NativeSemanticBridge::petri_successor_plan_cache_equivalence()"
            }
            Self::RepresentsPetriSuccessorPlanCacheEquivalence => {
                "NativeSemanticBridgeReport::represents_petri_successor_plan_cache_equivalence()"
            }
            Self::NativeSemanticBridgeProofIdentityKeyValueText => {
                "NativeSemanticBridgeReport::proof_identity_key_value_text()"
            }
            Self::NativeSemanticBridgeProofIdentityReplayReportForKeyValueText => {
                "NativeSemanticBridgeReport::proof_identity_replay_report_for_key_value_text()"
            }
            Self::NativeSemanticBridgeProofIdentityReplayComponentHealthSummaryKeyValueText => {
                "NativeSemanticBridgeProofIdentityReplayReport::component_health_summary_key_value_text()"
            }
            Self::PetriSuccessorTrustMcChcProofEvidenceIdentityKeyValueText => {
                "PetriSuccessorTrustMcChcProofHandoffReport::proof_evidence_identity_key_value_text()"
            }
            Self::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReportForKeyValueText => {
                "PetriSuccessorTrustMcChcProofHandoffReport::proof_evidence_identity_replay_report_for_key_value_text()"
            }
            Self::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayComponentHealthSummaryKeyValueText => {
                "PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReport::component_health_summary_key_value_text()"
            }
            Self::ContractDescriptor => "petri_successor_trust_mc_chc_contract_descriptor()",
            Self::ArtifactIdentity => {
                "NativeSharedPrimitiveArtifactRequirement::accepts_artifact_identity()"
            }
            Self::ArtifactByteResolution => {
                "NativeVerificationBundle::resolve_evidence_artifact_attachment()"
            }
            Self::ArtifactAuthority => "NativeEvidenceArtifactResolution::is_authoritative()",
            Self::AuthoritativeBytes => "NativeEvidenceArtifactResolution::authoritative_bytes()",
        }
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_petri_trust_mc_provided_field(
    provided_fields: &'static [&'static str],
    field: TrustIrPetriTrustMcProvidedField,
) -> &'static str {
    provided_fields
        .iter()
        .copied()
        .find(|provided| *provided == field.needle())
        .unwrap_or("missing_trust_ir_petri_trust_mc_provided_field")
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_mc_verification_mode_code(
    mode: trust_ir::request::TrustMcVerificationMode,
) -> &'static str {
    match mode {
        trust_ir::request::TrustMcVerificationMode::BoundedModelCheck => "bounded_model_check",
        trust_ir::request::TrustMcVerificationMode::Chc => "chc",
        trust_ir::request::TrustMcVerificationMode::Pdr => "pdr",
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn native_shared_primitive_verification_mode_code(
    mode: trust_ir::request::NativeSharedPrimitiveVerificationMode,
) -> &'static str {
    match mode {
        trust_ir::request::NativeSharedPrimitiveVerificationMode::TrustVc(
            trust_ir::request::TrustVcVerificationMode::ImportProofCertificates,
        ) => "import_proof_certificates",
        trust_ir::request::NativeSharedPrimitiveVerificationMode::TrustVc(
            trust_ir::request::TrustVcVerificationMode::MergeProofCertificates,
        ) => "merge_proof_certificates",
        trust_ir::request::NativeSharedPrimitiveVerificationMode::TrustMc(mode) => {
            trust_mc_verification_mode_code(mode)
        }
        trust_ir::request::NativeSharedPrimitiveVerificationMode::TrustWp(
            trust_ir::request::TrustWpVerificationMode::WeakestPrecondition,
        ) => "weakest_precondition",
        trust_ir::request::NativeSharedPrimitiveVerificationMode::TrustWp(
            trust_ir::request::TrustWpVerificationMode::StrongestPostcondition,
        ) => "strongest_postcondition",
        trust_ir::request::NativeSharedPrimitiveVerificationMode::TrustWp(
            trust_ir::request::TrustWpVerificationMode::Abduction,
        ) => "abduction",
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn native_evidence_artifact_kind_code(
    kind: trust_ir::request::NativeEvidenceArtifactKind,
) -> &'static str {
    kind.code()
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_kind_codes(
    values: &[trust_ir::request::NativeEvidenceArtifactKind],
) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .copied()
            .map(native_evidence_artifact_kind_code)
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_trust_ir_artifact_kind_code_from_option(
    value: Option<trust_ir::request::NativeEvidenceArtifactKind>,
) -> &'static str {
    value
        .map(native_evidence_artifact_kind_code)
        .unwrap_or("none")
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_role_codes(
    values: impl IntoIterator<Item = trust_ir::request::NativeSharedPrimitiveArtifactRole>,
) -> String {
    let codes = values
        .into_iter()
        .map(|role| role.code())
        .collect::<Vec<_>>();
    if codes.is_empty() {
        "none".to_owned()
    } else {
        codes.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_verifier_suite_codes(
    values: impl IntoIterator<Item = trust_ir::request::NativeVerifierSuite>,
) -> String {
    let codes = values
        .into_iter()
        .map(|suite| suite.code())
        .collect::<Vec<_>>();
    if codes.is_empty() {
        "none".to_owned()
    } else {
        codes.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_requirement_kind_codes(
    values: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .map(|requirement| requirement.kind.code())
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_requirement_role_codes(
    values: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .copied()
            .map(trust_ir::request::NativeSharedPrimitiveArtifactRequirement::role_code)
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_requirement_digest_algorithm_codes(
    values: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .map(|requirement| requirement.digest_algorithm.to_string())
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_requirement_owner_suite_codes(
    values: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .map(|requirement| requirement.owner_suite.code())
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_artifact_requirement_emitted_solver_artifact_codes(
    values: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .map(|requirement| requirement.requires_emitted_solver_artifact.to_string())
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn add_trust_ir_native_evidence_artifact_resolution_evidence(
    report: &mut CapabilityReport,
    bundle: &trust_ir::NativeVerificationBundle,
    verifier_suite: trust_ir::request::NativeVerifierSuite,
    requirements: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
    attachments: &[trust_ir::request::NativeEvidenceArtifactAttachment],
) {
    let Some(requirement) = requirements
        .iter()
        .copied()
        .find(|requirement| requirement.role_code() == "replay_transcript")
        .or_else(|| requirements.first().copied())
    else {
        return;
    };

    let key = trust_ir_artifact_requirement_resolution_key(bundle, verifier_suite, requirement);
    let resolution = bundle.resolve_evidence_artifact_attachment(key, attachments);
    let resolution_report = &resolution.report;
    let mut fields = Vec::new();
    push_trust_cg_native_admission_field(&mut fields, "schema", &resolution_report.schema);
    push_trust_cg_native_admission_field(
        &mut fields,
        "schema_version",
        resolution_report.schema_version,
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "request",
        resolution_report.request.to_string(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "owner_suite",
        resolution_report
            .owner_suite
            .map(trust_ir::request::NativeVerifierSuite::code)
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "required_kind",
        resolution_report.required_kind.code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "digest_algorithm",
        resolution_report.digest_algorithm.to_string(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "digest",
        resolution_report.digest.to_string(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "artifact_name",
        resolution_report.artifact_name.as_deref().unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "byte_source_identity",
        resolution_report
            .byte_source_identity
            .as_deref()
            .unwrap_or("none"),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "byte_len",
        resolution_report
            .byte_len
            .map(|len| len.to_string())
            .unwrap_or_else(|| "none".to_owned()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "actual_digest",
        resolution_report
            .actual_digest
            .map(|digest| digest.to_string())
            .unwrap_or_else(|| "none".to_owned()),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "authority_code",
        resolution_report.authority_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "status_code",
        resolution_report.status_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "reason_code",
        resolution_report.reason_code(),
    );
    push_trust_cg_native_admission_field(
        &mut fields,
        "authority",
        resolution_report.authority_code(),
    );
    push_trust_cg_native_admission_field(&mut fields, "status", resolution_report.status_code());
    push_trust_cg_native_admission_field(&mut fields, "reason", resolution_report.reason_code());
    push_trust_cg_native_admission_field(&mut fields, "is_resolved", resolution.is_resolved());
    push_trust_cg_native_admission_field(
        &mut fields,
        "is_authoritative",
        resolution.is_authoritative(),
    );
    push_trust_cg_native_admission_field(&mut fields, "production_selected", false);
    push_trust_cg_native_admission_field(
        &mut fields,
        "fail_closed",
        !resolution.is_authoritative(),
    );
    for line in resolution.authority_evidence_key_value_lines() {
        if let Some((key, value)) = line.split_once('=') {
            push_trust_cg_native_admission_field(&mut fields, key, value);
        }
    }
    report.add_evidence(render_trust_cg_native_admission_row(
        "trust-ir native_evidence_artifact_resolution",
        &fields,
    ));
}

#[cfg(feature = "trust-cg-petri-native")]
struct TrustIrArtifactAuthoritySummary {
    attachment_count: usize,
    resolution_status_codes: String,
    resolution_reason_codes: String,
    resolution_authority_codes: String,
    authoritative_requirement_count: usize,
    authoritative_requirement_roles: String,
    unauthoritative_requirement_roles: String,
    authoritative_requirement_kinds: String,
    unauthoritative_requirement_kinds: String,
    authoritative_bytes_count: usize,
}

#[cfg(feature = "trust-cg-petri-native")]
fn summarize_trust_ir_artifact_authority(
    bundle: &trust_ir::NativeVerificationBundle,
    verifier_suite: trust_ir::request::NativeVerifierSuite,
    requirements: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
    attachments: &[trust_ir::request::NativeEvidenceArtifactAttachment],
) -> TrustIrArtifactAuthoritySummary {
    let mut resolution_status_codes = Vec::new();
    let mut resolution_reason_codes = Vec::new();
    let mut resolution_authority_codes = Vec::new();
    let mut authoritative_requirement_roles = Vec::new();
    let mut unauthoritative_requirement_roles = Vec::new();
    let mut authoritative_requirement_kinds = Vec::new();
    let mut unauthoritative_requirement_kinds = Vec::new();
    let mut authoritative_bytes_count = 0usize;

    for requirement in requirements.iter().copied() {
        let key = trust_ir_artifact_requirement_resolution_key(bundle, verifier_suite, requirement);
        let resolution = bundle.resolve_evidence_artifact_attachment(key, attachments);
        resolution_status_codes.push(resolution.report.status_code());
        resolution_reason_codes.push(resolution.report.reason_code());
        resolution_authority_codes.push(resolution.report.authority_code());

        if resolution.is_authoritative() {
            authoritative_requirement_roles.push(requirement.role_code());
            authoritative_requirement_kinds.push(requirement.kind.code());
        } else {
            unauthoritative_requirement_roles.push(requirement.role_code());
            unauthoritative_requirement_kinds.push(requirement.kind.code());
        }
        if resolution.authoritative_bytes().is_some() {
            authoritative_bytes_count += 1;
        }
    }

    TrustIrArtifactAuthoritySummary {
        attachment_count: attachments.len(),
        resolution_status_codes: join_static_codes(resolution_status_codes),
        resolution_reason_codes: join_static_codes(resolution_reason_codes),
        resolution_authority_codes: join_static_codes(resolution_authority_codes),
        authoritative_requirement_count: authoritative_requirement_roles.len(),
        authoritative_requirement_roles: join_static_codes(authoritative_requirement_roles),
        unauthoritative_requirement_roles: join_static_codes(unauthoritative_requirement_roles),
        authoritative_requirement_kinds: join_static_codes(authoritative_requirement_kinds),
        unauthoritative_requirement_kinds: join_static_codes(unauthoritative_requirement_kinds),
        authoritative_bytes_count,
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_static_codes(codes: Vec<&'static str>) -> String {
    if codes.is_empty() {
        "none".to_owned()
    } else {
        codes.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_attachment_resolution_required_kind_codes(
    values: &[trust_ir::request::NativeEvidenceArtifactAttachmentResolution<'_>],
) -> String {
    join_static_codes(
        values
            .iter()
            .map(|resolution| resolution.required_kind.code())
            .collect(),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_attachment_resolution_status_codes(
    values: &[trust_ir::request::NativeEvidenceArtifactAttachmentResolution<'_>],
) -> String {
    join_static_codes(
        values
            .iter()
            .map(trust_ir::request::NativeEvidenceArtifactAttachmentResolution::status_code)
            .collect(),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_attachment_resolution_reason_codes(
    values: &[trust_ir::request::NativeEvidenceArtifactAttachmentResolution<'_>],
) -> String {
    join_static_codes(
        values
            .iter()
            .map(trust_ir::request::NativeEvidenceArtifactAttachmentResolution::reason_code)
            .collect(),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_attachment_resolution_authority_codes(
    values: &[trust_ir::request::NativeEvidenceArtifactAttachmentResolution<'_>],
) -> String {
    join_static_codes(
        values
            .iter()
            .map(|resolution| {
                resolution
                    .resolution
                    .as_ref()
                    .map(|resolved| resolved.report.authority_code())
                    .unwrap_or("none")
            })
            .collect(),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn count_trust_ir_attachment_resolution_authoritative_bytes(
    values: &[trust_ir::request::NativeEvidenceArtifactAttachmentResolution<'_>],
) -> usize {
    values
        .iter()
        .filter(|resolution| resolution.authoritative_bytes().is_some())
        .count()
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_artifact_requirement_resolution_key(
    bundle: &trust_ir::NativeVerificationBundle,
    verifier_suite: trust_ir::request::NativeVerifierSuite,
    requirement: trust_ir::request::NativeSharedPrimitiveArtifactRequirement,
) -> trust_ir::request::NativeEvidenceArtifactAttachmentKey {
    if let Some((request, artifact)) = bundle.evidence_bundles.iter().find_map(|evidence| {
        if evidence.verifier_suite() != verifier_suite {
            return None;
        }
        evidence
            .artifacts()
            .iter()
            .find(|artifact| requirement.accepts_artifact_identity(artifact))
            .map(|artifact| (evidence.request(), artifact))
    }) {
        return trust_ir::request::NativeEvidenceArtifactAttachmentKey::for_artifact(
            request, artifact,
        );
    }

    let request = bundle
        .requests
        .iter()
        .find(|request| request.verifier_suite() == verifier_suite)
        .map(trust_ir::request::NativeVerificationRequest::id)
        .or_else(|| {
            bundle
                .evidence_bundles
                .iter()
                .find(|evidence| evidence.verifier_suite() == verifier_suite)
                .map(trust_ir::request::NativeEvidenceBundle::request)
        })
        .unwrap_or_else(|| trust_ir::request::NativeRequestId::new(0));

    trust_ir::request::NativeEvidenceArtifactAttachmentKey::new(
        request,
        requirement.kind,
        requirement.digest_algorithm,
        zero_trust_ir_proof_digest(requirement.digest_algorithm),
    )
}

#[cfg(feature = "trust-cg-petri-native")]
fn zero_trust_ir_proof_digest(algorithm: trust_ir::ProofDigestAlgorithm) -> trust_ir::ProofDigest {
    trust_ir::ProofDigest {
        algorithm,
        bytes: [0; 32],
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn trust_ir_artifact_requirement_is_bound(
    bundle: &trust_ir::NativeVerificationBundle,
    requirement: trust_ir::request::NativeSharedPrimitiveArtifactRequirement,
) -> bool {
    bundle
        .evidence_bundles
        .iter()
        .flat_map(trust_ir::request::NativeEvidenceBundle::artifacts)
        .any(|artifact| requirement.accepts_artifact_identity(artifact))
}

#[cfg(feature = "trust-cg-petri-native")]
fn count_trust_ir_bound_artifact_requirements(
    bundle: &trust_ir::NativeVerificationBundle,
    requirements: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
) -> usize {
    requirements
        .iter()
        .copied()
        .filter(|requirement| trust_ir_artifact_requirement_is_bound(bundle, *requirement))
        .count()
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_bound_artifact_requirement_role_codes(
    bundle: &trust_ir::NativeVerificationBundle,
    requirements: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
    bound: bool,
) -> String {
    let codes = requirements
        .iter()
        .copied()
        .filter(|requirement| trust_ir_artifact_requirement_is_bound(bundle, *requirement) == bound)
        .map(trust_ir::request::NativeSharedPrimitiveArtifactRequirement::role_code)
        .collect::<Vec<_>>();
    if codes.is_empty() {
        "none".to_owned()
    } else {
        codes.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn join_trust_ir_bound_artifact_requirement_kind_codes(
    bundle: &trust_ir::NativeVerificationBundle,
    requirements: &[trust_ir::request::NativeSharedPrimitiveArtifactRequirement],
    bound: bool,
) -> String {
    let codes = requirements
        .iter()
        .copied()
        .filter(|requirement| trust_ir_artifact_requirement_is_bound(bundle, *requirement) == bound)
        .map(|requirement| requirement.kind.code())
        .collect::<Vec<_>>();
    if codes.is_empty() {
        "none".to_owned()
    } else {
        codes.join("|")
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_ay_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_owned())
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_ay_str(value: Option<&'static str>) -> &'static str {
    value.unwrap_or("none")
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_ay_string(value: &Option<String>) -> &str {
    value.as_deref().unwrap_or("none")
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_trust_ir_proof_digest_string(value: &Option<trust_ir::ProofDigest>) -> String {
    value
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "none".to_owned())
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_trust_ir_artifact_name(
    value: &Option<trust_ir::request::NativeEvidenceArtifact>,
) -> &str {
    value
        .as_ref()
        .map(|artifact| artifact.name.as_str())
        .unwrap_or("none")
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_trust_ir_artifact_kind_code(
    value: &Option<trust_ir::request::NativeEvidenceArtifact>,
) -> &'static str {
    value
        .as_ref()
        .map(|artifact| artifact.kind.code())
        .unwrap_or("none")
}

#[cfg(feature = "trust-cg-petri-native")]
fn optional_trust_ir_artifact_digest_string(
    value: &Option<trust_ir::request::NativeEvidenceArtifact>,
) -> String {
    value
        .as_ref()
        .map(|artifact| artifact.digest.to_string())
        .unwrap_or_else(|| "none".to_owned())
}

#[cfg(feature = "trust-cg-petri-native")]
fn render_trust_cg_native_admission_row(prefix: &str, fields: &[(String, String)]) -> String {
    let mut row = String::from(prefix);
    for (key, value) in fields {
        row.push(' ');
        row.push_str(key);
        row.push('=');
        row.push_str(value);
    }
    row
}

#[cfg(feature = "trust-cg-petri-native")]
fn native_bundle_producer_code(producer: trust_ir::NativeBundleProducer) -> &'static str {
    match producer {
        trust_ir::NativeBundleProducer::TRust => "trust",
        trust_ir::NativeBundleProducer::TSwift => "tswift",
        trust_ir::NativeBundleProducer::TC => "tc",
        trust_ir::NativeBundleProducer::TrustIr => "trust_ir",
    }
}

#[cfg(feature = "trust-cg-petri-native")]
fn native_adapter_input_code(input: trust_ir::NativeAdapterInput) -> &'static str {
    match input {
        trust_ir::NativeAdapterInput::RustMir { .. } => "rust_mir",
        trust_ir::NativeAdapterInput::TrustIrModule => "trust_ir_module",
    }
}

fn reject_native_successor(
    report: &mut CapabilityReport,
    reason: UnsupportedReason,
    detail: String,
) {
    let capability = native_capability_with_evidence(
        report,
        "successor",
        "unavailable",
        false,
        BackendCapability::unsupported(BackendDomain::PetriMcc, BackendKind::NativeKernel, reason)
            .for_problem(ProblemKind::NativeSuccessor)
            .with_facets([SolverFacet::NativeCodegen])
            .with_role(CapabilityRole::Validation)
            .with_detail(detail),
    );
    report.reject(capability);
}

fn native_capability_with_evidence(
    report: &mut CapabilityReport,
    lane: &'static str,
    adoption: &'static str,
    deferred: bool,
    capability: BackendCapability,
) -> BackendCapability {
    let reason_code = capability.normalized_reason_code();
    report.add_evidence(format!(
        "Petri native_{lane} capability backend={:?} problem={:?} status={} role={:?} reason_code={reason_code} adoption={adoption} deferred={deferred}",
        capability.backend,
        capability.problem,
        capability.status.name(),
        capability.role,
    ));
    capability
}

fn petri_successor_kernel_artifact_adoption_evidence() -> KernelArtifactAdoptionEvidence {
    KernelArtifactAdoptionEvidence::successor_kernel(
        TY_KERNEL_ARTIFACT_CONSUMER,
        PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
        KernelSymbolSignature::native_successor_kernel(),
        deferred_kernel_artifact_checksums(),
        KernelStateDomain::Unknown,
    )
    .with_required_manifest_metadata(TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA)
}

fn petri_predicate_kernel_artifact_adoption_evidence() -> KernelArtifactAdoptionEvidence {
    KernelArtifactAdoptionEvidence::predicate_kernel(
        TY_KERNEL_ARTIFACT_CONSUMER,
        PETRI_NATIVE_PREDICATE_ENTRY_SYMBOL,
        KernelSymbolSignature::native_state_predicate_kernel(),
        deferred_kernel_artifact_checksums(),
        KernelStateDomain::Unknown,
    )
    .with_required_manifest_metadata(TY_PREDICATE_KERNEL_EVIDENCE_METADATA)
}

fn deferred_kernel_artifact_checksums() -> KernelArtifactChecksums {
    KernelArtifactChecksums::new(
        KernelArtifactChecksum::default(),
        KernelArtifactChecksum::default(),
        KernelArtifactChecksum::default(),
        KernelArtifactChecksum::default(),
        KernelArtifactChecksum::default(),
    )
}

fn add_kernel_artifact_contract_evidence(
    report: &mut CapabilityReport,
    lane: &'static str,
    adoption: &'static str,
    evidence: &KernelArtifactAdoptionEvidence,
) {
    let required_metadata = if evidence.required_manifest_metadata.is_empty() {
        "none".to_string()
    } else {
        evidence.required_manifest_metadata.join(",")
    };
    let artifact_checksums = if kernel_artifact_checksums_are_deferred(&evidence.checksums) {
        "deferred"
    } else {
        "present"
    };
    report.add_evidence(format!(
        "Petri native_{lane} JIT ABI artifact contract expected schema={} schema_version={} kind={} consumer={} entry_symbol={} signature_abi={} params={} returns={} required_manifest_metadata={} adopted=false adoption={adoption} artifact_checksums={artifact_checksums}",
        evidence.schema,
        evidence.schema_version,
        evidence.kind.as_str(),
        evidence.consumer,
        evidence.entry_symbol,
        evidence.signature.abi,
        evidence.signature.params.len(),
        evidence.signature.returns.len(),
        required_metadata,
    ));
}

fn kernel_artifact_checksums_are_deferred(checksums: &KernelArtifactChecksums) -> bool {
    checksums.target.is_zero()
        && checksums.abi.is_zero()
        && checksums.layout.is_zero()
        && checksums.proof_policy.is_zero()
        && checksums.semantic.is_zero()
}

fn unsupported_reason_for_kernel_error(error: &PetriKernelError) -> UnsupportedReason {
    match error {
        PetriKernelError::TokenExceedsI64 { .. } => {
            UnsupportedReason::TooLarge("token exceeds i64")
        }
        PetriKernelError::ArcWeightExceedsI64 { .. } => {
            UnsupportedReason::TooLarge("arc weight exceeds i64")
        }
        PetriKernelError::ConstantExceedsI64 { .. } => {
            UnsupportedReason::TooLarge("constant exceeds i64")
        }
        PetriKernelError::CountExceedsU32 { what, .. } => UnsupportedReason::TooLarge(what),
        PetriKernelError::NativeStatus { status, .. } => {
            unsupported_reason_for_native_status(*status)
        }
        PetriKernelError::NativeCompile { .. } | PetriKernelError::NativeSymbol { .. } => {
            UnsupportedReason::NativeKernelUnavailable
        }
        _ => UnsupportedReason::UnsupportedFragment("petri successor kernel plan"),
    }
}

fn unsupported_reason_for_native_status(
    status: PetriNativeAllSuccessorsStatus,
) -> UnsupportedReason {
    match status {
        PetriNativeAllSuccessorsStatus::Ok => UnsupportedReason::Other("native status ok"),
        PetriNativeAllSuccessorsStatus::InvalidAbi => {
            UnsupportedReason::UnsupportedFragment("petri native successor ABI")
        }
        PetriNativeAllSuccessorsStatus::BufferOverflow => {
            UnsupportedReason::TooLarge("native successor buffer capacity")
        }
        PetriNativeAllSuccessorsStatus::TokenOverflow => {
            UnsupportedReason::TooLarge("native successor token arithmetic")
        }
        PetriNativeAllSuccessorsStatus::Unsupported
        | PetriNativeAllSuccessorsStatus::Unknown(_) => UnsupportedReason::NativeKernelUnavailable,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FlatTransitionOutcome {
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CheckedTransitionOutcome {
    Disabled,
    Enabled { successor: Vec<u64> },
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct FlatAllTransitionCandidates {
    place_count: usize,
    transition_ids: Vec<TransitionIdx>,
    flat_successors: Vec<i64>,
}

impl FlatAllTransitionCandidates {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn clear(&mut self) {
        self.place_count = 0;
        self.transition_ids.clear();
        self.flat_successors.clear();
    }

    fn clear_for_layout(&mut self, layout: PetriKernelLayout) {
        self.clear();
        self.place_count = layout.state_len();
    }

    fn push(&mut self, transition: TransitionIdx, flat_successor: &[i64]) {
        debug_assert_eq!(flat_successor.len(), self.place_count);
        self.transition_ids.push(transition);
        self.flat_successors.extend_from_slice(flat_successor);
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.transition_ids.len()
    }

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.transition_ids.is_empty()
    }

    #[must_use]
    pub(crate) fn place_count(&self) -> usize {
        self.place_count
    }

    #[must_use]
    pub(crate) fn transition_ids(&self) -> &[TransitionIdx] {
        &self.transition_ids
    }

    #[must_use]
    pub(crate) fn flat_successors(&self) -> &[i64] {
        &self.flat_successors
    }

    #[must_use]
    pub(crate) fn flat_successor(&self, index: usize) -> Option<&[i64]> {
        if index >= self.len() {
            return None;
        }
        let start = index * self.place_count;
        let end = start + self.place_count;
        Some(&self.flat_successors[start..end])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PetriTransitionParityConfig {
    enabled: bool,
}

impl PetriTransitionParityConfig {
    #[must_use]
    pub(crate) fn from_env() -> Self {
        Self {
            enabled: env_flag_enabled(ENABLE_TRANSITION_PARITY_ENV),
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn enabled_for_tests(enabled: bool) -> Self {
        Self { enabled }
    }

    #[must_use]
    pub(crate) fn is_enabled(self) -> bool {
        self.enabled
    }

    pub(crate) fn check_transition_successor(
        self,
        net: &PetriNet,
        transition: TransitionIdx,
        marking: &[u64],
        successor: &[u64],
        scratch: &mut PetriKernelScratch,
    ) -> Result<(), PetriKernelError> {
        let Some(checked_successor) =
            self.checked_transition_successor(net, transition, marking, scratch)?
        else {
            return Ok(());
        };

        if checked_successor == successor {
            Ok(())
        } else {
            Err(PetriKernelError::ParityMismatch {
                transition,
                detail: format!(
                    "integration successor mismatch: checked={checked_successor:?}, integration={successor:?}",
                ),
            })
        }
    }

    pub(crate) fn checked_transition_successor(
        self,
        net: &PetriNet,
        transition: TransitionIdx,
        marking: &[u64],
        scratch: &mut PetriKernelScratch,
    ) -> Result<Option<Vec<u64>>, PetriKernelError> {
        if !self.enabled {
            return Ok(None);
        }

        let mut successor = Vec::with_capacity(net.num_places());
        match checked_fire_transition_into(net, transition, marking, scratch, &mut successor)? {
            FlatTransitionOutcome::Disabled => Err(PetriKernelError::ParityMismatch {
                transition,
                detail: "integration produced a successor for a disabled transition".to_string(),
            }),
            FlatTransitionOutcome::Enabled => Ok(Some(successor)),
        }
    }

    pub(crate) fn checked_transition_successor_cached_into(
        self,
        net: &PetriNet,
        cache: &PetriKernelPlanCache,
        transition: TransitionIdx,
        marking: &[u64],
        scratch: &mut PetriKernelScratch,
        successor_out: &mut Vec<u64>,
    ) -> Result<Option<()>, PetriKernelError> {
        successor_out.clear();
        if !self.enabled {
            return Ok(None);
        }

        let layout = cache.validate_for_net(net)?;
        let plan = cache.plan(transition)?;
        match checked_fire_transition_plan_into(net, layout, plan, marking, scratch, successor_out)?
        {
            FlatTransitionOutcome::Disabled => Err(PetriKernelError::ParityMismatch {
                transition,
                detail: "integration produced a successor for a disabled transition".to_string(),
            }),
            FlatTransitionOutcome::Enabled => Ok(Some(())),
        }
    }
}

fn env_flag_enabled(key: &str) -> bool {
    matches!(
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

pub(crate) fn checked_all_transition_successors_cached_into(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
    candidates: &mut FlatAllTransitionCandidates,
) -> Result<(), PetriKernelError> {
    let layout = cache.validate_for_net(net)?;
    layout.check_state_len(marking.len())?;
    candidates.clear_for_layout(layout);

    marking_to_flat_i64(marking, &mut scratch.flat_in)?;

    for plan in cache.plans() {
        let flat_outcome =
            fire_transition_plan_flat(layout, plan, &scratch.flat_in, &mut scratch.flat_out)?;
        let interpreter_enabled = net.is_enabled(marking, plan.transition);

        match (interpreter_enabled, flat_outcome) {
            (false, FlatTransitionOutcome::Disabled) => {}
            (true, FlatTransitionOutcome::Enabled) => {
                // Fail-closed (#22): map an interpreter token-count overflow to a
                // kernel decline so the cross-check declines rather than panics.
                net.fire_into(marking, plan.transition, &mut scratch.interpreter_out)
                    .map_err(|e| PetriKernelError::IntExprOverflow {
                        detail: format!("interpreter fire overflow: {e}"),
                    })?;
                validate_flat_successor_matches_interpreter(
                    layout,
                    plan.transition,
                    &scratch.flat_out,
                    &scratch.interpreter_out,
                )?;
                candidates.push(plan.transition, &scratch.flat_out);
            }
            (expected, actual) => {
                return Err(PetriKernelError::ParityMismatch {
                    transition: plan.transition,
                    detail: format!(
                        "enabled mismatch: kernel={}, interpreter={expected}",
                        matches!(actual, FlatTransitionOutcome::Enabled)
                    ),
                });
            }
        }
    }

    Ok(())
}

impl PetriNet {
    #[doc(hidden)]
    pub fn trust_cg_profile_all_transition_checked_successors<F>(
        &self,
        sample_markings: &[Vec<u64>],
        repeats: usize,
        mut observe_successor: F,
    ) -> Result<(), String>
    where
        F: FnMut(TransitionIdx, &[u64]),
    {
        let cache = PetriKernelPlanCache::for_net(self).map_err(|error| format!("{error:?}"))?;
        let mut scratch = PetriKernelScratch::new();
        let mut candidates = FlatAllTransitionCandidates::new();
        let mut successor = Vec::with_capacity(self.num_places());

        for _ in 0..repeats {
            for marking in sample_markings {
                checked_all_transition_successors_cached_into(
                    self,
                    &cache,
                    marking,
                    &mut scratch,
                    &mut candidates,
                )
                .map_err(|error| format!("{error:?}"))?;

                for (index, &transition) in candidates.transition_ids().iter().enumerate() {
                    let flat_successor = candidates
                        .flat_successor(index)
                        .expect("candidate transition index must have a flat successor row");
                    flat_i64_to_marking(flat_successor, &mut successor)
                        .map_err(|error| format!("{error:?}"))?;
                    observe_successor(transition, &successor);
                }
            }
        }

        Ok(())
    }
}

pub(crate) fn marking_to_flat_i64(
    marking: &[u64],
    out: &mut Vec<i64>,
) -> Result<(), PetriKernelError> {
    out.clear();

    for (place, &value) in marking.iter().enumerate() {
        if value > i64::MAX as u64 {
            return Err(PetriKernelError::TokenExceedsI64 { place, value });
        }
    }

    out.reserve(marking.len());
    for &value in marking {
        out.push(value as i64);
    }

    Ok(())
}

pub(crate) fn flat_i64_to_marking(
    flat: &[i64],
    out: &mut Vec<u64>,
) -> Result<(), PetriKernelError> {
    out.clear();

    for (place, &value) in flat.iter().enumerate() {
        if value < 0 {
            return Err(PetriKernelError::NegativeFlatToken { place, value });
        }
    }

    out.reserve(flat.len());
    for &value in flat {
        out.push(value as u64);
    }

    Ok(())
}

pub(crate) fn build_transition_plan(
    net: &PetriNet,
    transition: TransitionIdx,
) -> Result<TransitionKernelPlan, PetriKernelError> {
    let info = net.transitions.get(transition.0 as usize).ok_or(
        PetriKernelError::TransitionOutOfBounds {
            transition,
            transition_count: net.num_transitions(),
        },
    )?;

    let inputs = checked_arcs_to_i64(net, transition, &info.inputs)?;
    let outputs = checked_arcs_to_i64(net, transition, &info.outputs)?;

    Ok(TransitionKernelPlan {
        transition,
        inputs,
        outputs,
    })
}

pub(crate) fn fire_transition_plan_flat(
    layout: PetriKernelLayout,
    plan: &TransitionKernelPlan,
    state_in: &[i64],
    state_out: &mut Vec<i64>,
) -> Result<FlatTransitionOutcome, PetriKernelError> {
    layout.check_state_len(state_in.len())?;

    state_out.clear();
    state_out.extend_from_slice(state_in);

    for &(place, weight) in &plan.inputs {
        let current = state_in[place.0 as usize];
        if current < weight {
            return Ok(FlatTransitionOutcome::Disabled);
        }
    }

    for &(place, weight) in &plan.inputs {
        apply_checked_delta(state_out, place, -weight)?;
    }
    for &(place, weight) in &plan.outputs {
        apply_checked_delta(state_out, place, weight)?;
    }

    Ok(FlatTransitionOutcome::Enabled)
}

pub(crate) fn checked_fire_transition(
    net: &PetriNet,
    transition: TransitionIdx,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
) -> Result<CheckedTransitionOutcome, PetriKernelError> {
    let layout = PetriKernelLayout::for_net(net);
    let plan = build_transition_plan(net, transition)?;
    let mut successor = Vec::with_capacity(layout.state_len());
    match checked_fire_transition_plan_into(net, layout, &plan, marking, scratch, &mut successor)? {
        FlatTransitionOutcome::Disabled => Ok(CheckedTransitionOutcome::Disabled),
        FlatTransitionOutcome::Enabled => Ok(CheckedTransitionOutcome::Enabled { successor }),
    }
}

pub(crate) fn checked_fire_transition_into(
    net: &PetriNet,
    transition: TransitionIdx,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
    successor_out: &mut Vec<u64>,
) -> Result<FlatTransitionOutcome, PetriKernelError> {
    let layout = PetriKernelLayout::for_net(net);
    let plan = build_transition_plan(net, transition)?;
    checked_fire_transition_plan_into(net, layout, &plan, marking, scratch, successor_out)
}

fn checked_fire_transition_plan_into(
    net: &PetriNet,
    layout: PetriKernelLayout,
    plan: &TransitionKernelPlan,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
    successor_out: &mut Vec<u64>,
) -> Result<FlatTransitionOutcome, PetriKernelError> {
    layout.check_state_len(marking.len())?;
    marking_to_flat_i64(marking, &mut scratch.flat_in)?;
    let flat_outcome =
        fire_transition_plan_flat(layout, plan, &scratch.flat_in, &mut scratch.flat_out)?;

    let transition = plan.transition;
    let interpreter_enabled = net.is_enabled(marking, transition);
    if interpreter_enabled {
        // Fail-closed (#22): map a token-count overflow to a kernel decline.
        net.fire_into(marking, transition, &mut scratch.interpreter_out)
            .map_err(|e| PetriKernelError::IntExprOverflow {
                detail: format!("interpreter fire overflow: {e}"),
            })?;
    } else {
        scratch.interpreter_out.clear();
    }

    match (interpreter_enabled, flat_outcome) {
        (false, FlatTransitionOutcome::Disabled) => Ok(FlatTransitionOutcome::Disabled),
        (true, FlatTransitionOutcome::Enabled) => {
            flat_i64_to_marking(&scratch.flat_out, successor_out)?;
            if *successor_out == scratch.interpreter_out {
                Ok(FlatTransitionOutcome::Enabled)
            } else {
                Err(PetriKernelError::ParityMismatch {
                    transition,
                    detail: format!(
                        "successor mismatch: kernel={successor_out:?}, interpreter={:?}",
                        scratch.interpreter_out
                    ),
                })
            }
        }
        (expected, actual) => Err(PetriKernelError::ParityMismatch {
            transition,
            detail: format!(
                "enabled mismatch: kernel={}, interpreter={expected}",
                matches!(actual, FlatTransitionOutcome::Enabled)
            ),
        }),
    }
}

fn validate_flat_successor_matches_interpreter(
    layout: PetriKernelLayout,
    transition: TransitionIdx,
    flat: &[i64],
    interpreter: &[u64],
) -> Result<(), PetriKernelError> {
    layout.check_state_len(flat.len())?;
    layout.check_state_len(interpreter.len())?;

    for (place, (&kernel, &expected)) in flat.iter().zip(interpreter).enumerate() {
        if kernel < 0 {
            return Err(PetriKernelError::NegativeFlatToken {
                place,
                value: kernel,
            });
        }
        if kernel as u64 != expected {
            return Err(PetriKernelError::ParityMismatch {
                transition,
                detail: format!(
                    "successor mismatch at place {place}: kernel={kernel}, interpreter={expected}",
                ),
            });
        }
    }

    Ok(())
}

pub(crate) fn eval_int_expr_flat(
    layout: PetriKernelLayout,
    expr: &ResolvedIntExpr,
    flat: &[i64],
) -> Result<i64, PetriKernelError> {
    layout.check_state_len(flat.len())?;

    match expr {
        ResolvedIntExpr::Constant(value) => i64::try_from(*value)
            .map_err(|_| PetriKernelError::ConstantExceedsI64 { value: *value }),
        ResolvedIntExpr::TokensCount(places) => {
            let mut total = 0_i64;
            for &place in places {
                let tokens = flat_token_at(layout, flat, place)?;
                total =
                    total
                        .checked_add(tokens)
                        .ok_or_else(|| PetriKernelError::IntExprOverflow {
                            detail: format!("TokensCount overflow while adding place {place:?}"),
                        })?;
            }
            Ok(total)
        }
    }
}

pub(crate) fn eval_predicate_flat(
    layout: PetriKernelLayout,
    net: &PetriNet,
    pred: &ResolvedPredicate,
    flat: &[i64],
) -> Result<bool, PetriKernelError> {
    layout.check_state_len(flat.len())?;

    match pred {
        ResolvedPredicate::And(children) => {
            for child in children {
                if !eval_predicate_flat(layout, net, child, flat)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ResolvedPredicate::Or(children) => {
            for child in children {
                if eval_predicate_flat(layout, net, child, flat)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        ResolvedPredicate::Not(inner) => Ok(!eval_predicate_flat(layout, net, inner, flat)?),
        ResolvedPredicate::IntLe(left, right) => {
            Ok(eval_int_expr_flat(layout, left, flat)? <= eval_int_expr_flat(layout, right, flat)?)
        }
        ResolvedPredicate::IsFireable(transitions) => {
            for &transition in transitions {
                let plan = build_transition_plan(net, transition)?;
                if transition_plan_is_enabled_flat(layout, &plan, flat)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        ResolvedPredicate::True => Ok(true),
        ResolvedPredicate::False => Ok(false),
    }
}

pub(crate) fn checked_eval_predicate(
    net: &PetriNet,
    pred: &ResolvedPredicate,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
) -> Result<bool, PetriKernelError> {
    let layout = PetriKernelLayout::for_net(net);
    layout.check_state_len(marking.len())?;

    marking_to_flat_i64(marking, &mut scratch.flat_in)?;
    validate_predicate_flat_support(layout, net, pred, &scratch.flat_in)?;
    let flat_value = eval_predicate_flat(layout, net, pred, &scratch.flat_in)?;
    let interpreter_value = eval_predicate(pred, marking, net);

    if flat_value == interpreter_value {
        Ok(interpreter_value)
    } else {
        Err(PetriKernelError::PredicateParityMismatch {
            detail: format!("kernel={flat_value}, interpreter={interpreter_value}"),
        })
    }
}

fn validate_int_expr_flat_support(
    layout: PetriKernelLayout,
    expr: &ResolvedIntExpr,
    flat: &[i64],
) -> Result<(), PetriKernelError> {
    eval_int_expr_flat(layout, expr, flat).map(|_| ())
}

fn validate_predicate_flat_support(
    layout: PetriKernelLayout,
    net: &PetriNet,
    pred: &ResolvedPredicate,
    flat: &[i64],
) -> Result<(), PetriKernelError> {
    layout.check_state_len(flat.len())?;

    match pred {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => {
            for child in children {
                validate_predicate_flat_support(layout, net, child, flat)?;
            }
            Ok(())
        }
        ResolvedPredicate::Not(inner) => validate_predicate_flat_support(layout, net, inner, flat),
        ResolvedPredicate::IntLe(left, right) => {
            validate_int_expr_flat_support(layout, left, flat)?;
            validate_int_expr_flat_support(layout, right, flat)
        }
        ResolvedPredicate::IsFireable(transitions) => {
            for &transition in transitions {
                build_transition_plan(net, transition)?;
            }
            Ok(())
        }
        ResolvedPredicate::True | ResolvedPredicate::False => Ok(()),
    }
}

fn checked_arcs_to_i64(
    net: &PetriNet,
    transition: TransitionIdx,
    arcs: &[crate::petri_net::Arc],
) -> Result<Vec<(PlaceIdx, i64)>, PetriKernelError> {
    let place_count = net.num_places();
    arcs.iter()
        .map(|arc| {
            if arc.place.0 as usize >= place_count {
                return Err(PetriKernelError::PlaceOutOfBounds {
                    place: arc.place,
                    place_count,
                });
            }
            let weight =
                i64::try_from(arc.weight).map_err(|_| PetriKernelError::ArcWeightExceedsI64 {
                    transition,
                    place: arc.place,
                    weight: arc.weight,
                })?;
            Ok((arc.place, weight))
        })
        .collect()
}

fn transition_plan_is_enabled_flat(
    layout: PetriKernelLayout,
    plan: &TransitionKernelPlan,
    flat: &[i64],
) -> Result<bool, PetriKernelError> {
    layout.check_state_len(flat.len())?;
    for &(place, weight) in &plan.inputs {
        if flat_token_at(layout, flat, place)? < weight {
            return Ok(false);
        }
    }
    Ok(true)
}

fn flat_token_at(
    layout: PetriKernelLayout,
    flat: &[i64],
    place: PlaceIdx,
) -> Result<i64, PetriKernelError> {
    layout.check_state_len(flat.len())?;
    let place_idx = place.0 as usize;
    if place_idx >= layout.state_len() {
        return Err(PetriKernelError::PlaceOutOfBounds {
            place,
            place_count: layout.state_len(),
        });
    }
    let value = flat[place_idx];
    if value < 0 {
        return Err(PetriKernelError::NegativeFlatToken {
            place: place_idx,
            value,
        });
    }
    Ok(value)
}

fn apply_checked_delta(
    marking: &mut [i64],
    place: PlaceIdx,
    delta: i64,
) -> Result<(), PetriKernelError> {
    let slot = &mut marking[place.0 as usize];
    let next = slot
        .checked_add(delta)
        .ok_or(PetriKernelError::TokenOverflow {
            place,
            value: *slot,
            delta,
        })?;
    if next < 0 {
        return Err(PetriKernelError::NegativeFlatToken {
            place: place.0 as usize,
            value: next,
        });
    }
    *slot = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PlaceInfo, TransitionInfo};
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

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
            name: Some("simple".to_string()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1), arc(2, 3)])],
            initial_marking: vec![5, 0, 0],
        }
    }

    fn all_transition_net() -> PetriNet {
        PetriNet {
            name: Some("all-transition".to_string()),
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(2, 1)], vec![arc(0, 1)]),
                trans("t2", vec![arc(1, 1)], vec![arc(2, 2)]),
            ],
            initial_marking: vec![2, 1, 0],
        }
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn native_verification_bundle_fixture(net: &PetriNet) -> trust_ir::NativeVerificationBundle {
        let cache = PetriKernelPlanCache::for_net(net).expect("fixture plan cache should build");
        match native::petri_native_successor_verification_bundle(net, &cache) {
            native::PetriNativeVerificationBundleProduction::Available(bundle) => bundle,
            native::PetriNativeVerificationBundleProduction::Blocked(blocker) => {
                panic!("fixture native verification bundle should validate: {blocker:?}")
            }
        }
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn semantic_evidence_native_verification_bundle_fixture(
        net: &PetriNet,
    ) -> trust_ir::NativeVerificationBundle {
        let bundle = native_verification_bundle_fixture(net);
        let bridge = petri_native_successor_semantic_bridge(&bundle);
        let bridge_report = bundle.petri_successor_semantic_bridge_report(bridge.function);
        let proof_obligation = bridge_report
            .proof_obligation
            .expect("Petri native fixture should have a trust_mc proof obligation");
        let proof_obligation = bundle
            .module
            .proof_obligations
            .iter()
            .find(|obligation| obligation.id == proof_obligation)
            .expect("fixture proof obligation should be in the trust-ir module");
        assert_eq!(
            proof_obligation.status,
            trust_ir::ProofStatus::Discharged,
            "positive fixtures must finalize proof obligations before binding the module digest"
        );

        bundle
            .validate()
            .expect("semantic-evidence trust_mc native bundle fixture should validate");
        assert!(
            !bundle
                .petri_successor_semantic_bridge_report(bridge.function)
                .represents_petri_successor_plan_cache_equivalence(),
            "semantic artifacts and a bare Discharged status must remain fail-closed without a kernel-replayed certificate"
        );
        bundle
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
        let prefix = format!("{key}=");
        row.split_whitespace()
            .find_map(|field| field.strip_prefix(&prefix))
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn evidence_field_usize(row: &str, key: &str) -> usize {
        evidence_field(row, key)
            .unwrap_or_else(|| panic!("{key} should be present in evidence row: {row}"))
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("{key} should be a usize in {row}: {error}"))
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn trust_ir_component_readiness_row<'a>(
        report: &'a CapabilityReport,
        component: &str,
    ) -> &'a str {
        let marker = format!("trust-ir component_readiness component={component}");
        report
            .evidence
            .iter()
            .find(|evidence| evidence.contains(&marker))
            .map(String::as_str)
            .unwrap_or_else(|| panic!("missing trust-ir component readiness row: {marker}"))
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn trust_ir_component_manifest_rows<'a>(
        report: &'a CapabilityReport,
        component: &str,
    ) -> Vec<&'a str> {
        let prefix = format!("trust-ir {component} ");
        report
            .evidence
            .iter()
            .filter_map(|evidence| evidence.starts_with(&prefix).then_some(evidence.as_str()))
            .collect()
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_admission_blocker_sources_install_gate_descriptor_fields() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let admission_blocker = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg trust_cg_admission_blocker"))
            .expect("native JIT admission blocker evidence should be emitted");
        let contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let admission_surface = contract.install_gate_admission;
        let schema_version = admission_surface.schema_version.to_string();
        let required_fields = admission_surface.required_fields.join(",");
        let status_codes = admission_surface.status_codes.join(",");
        let blocker_codes = admission_surface.blocker_codes.join(",");

        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_available"),
            Some("true")
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_authoritative"),
            Some("true")
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_source"),
            Some(TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API)
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_name"),
            Some(admission_surface.name)
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_schema"),
            Some(admission_surface.schema)
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_schema_version"),
            Some(schema_version.as_str())
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_required_fields"),
            Some(required_fields.as_str())
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_status_codes"),
            Some(status_codes.as_str())
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_descriptor_blocker_codes"),
            Some(blocker_codes.as_str())
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_status_in_downstream_contract"),
            Some("true")
        );
        assert_eq!(
            evidence_field(admission_blocker, "admission_reason_in_downstream_contract"),
            Some("true")
        );
        let reason_code = evidence_field(admission_blocker, "reason_code")
            .expect("admission row should emit a reason code");
        assert!(
            admission_surface.blocker_codes.contains(&reason_code),
            "admission reason should come from the trust-codegen descriptor: {admission_blocker}"
        );
        assert_eq!(
            evidence_field(admission_blocker, "production_selected"),
            Some("false")
        );
        assert_eq!(
            evidence_field(admission_blocker, "fail_closed"),
            Some("true")
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn trust_cg_petri_call_packet_surface_reexports_are_linked() {
        let surface = trust_cg_petri_call_packet_surface();
        let descriptor =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor().call_packet;
        assert_eq!(
            surface.api,
            "trust-cg::petri_native_successor_call_packet_from_trust_ir_bundle"
        );
        assert_eq!(
            surface.schema,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA
        );
        assert_eq!(
            surface.schema_version,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA_VERSION
        );
        assert_eq!(surface.call_packet_type, "PetriNativeSuccessorCallPacket");
        assert_eq!(
            surface.callable_pointer_type,
            "PetriNativeSuccessorCallablePointer"
        );
        assert!(surface.descriptor_available);
        assert_eq!(
            surface.descriptor_source,
            "trust-cg::petri_native_successor_downstream_contract_descriptor.call_packet"
        );
        assert_eq!(surface.descriptor_status_code, descriptor.status_code);
        assert_eq!(surface.descriptor_status_code, "authoritative");
        assert!(surface.descriptor_authoritative);
        assert_eq!(
            surface.descriptor_dependency,
            "trust-cg::petri_native_successor_downstream_contract_descriptor.call_packet"
        );
        assert_eq!(surface.descriptor_upstream_ask, "none");
        assert!(descriptor.is_authoritative());
        assert!(descriptor.fails_closed_for_runtime_execution());

        let callable_pointer = tla_trust_cg::PetriNativeSuccessorCallablePointer::from_usize(1)
            .expect("non-zero callable address should produce a pointer identity");
        assert_eq!(callable_pointer.addr_usize(), 1);
        let _: TrustCgPetriCallPacketBuilder =
            tla_trust_cg::petri_native_successor_call_packet_from_trust_ir_bundle;
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_capability_report_emits_trust_ir_replay_json_binding_rows() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let descriptor = trust_ir::petri_native_verification_bundle_handoff_descriptor();
        let manifest_identity = descriptor.manifest_identity();
        let surface = trust_ir::petri_native_verification_bundle_handoff_replay_contract_surface();
        let surface_round_trip = surface.round_trip_report(&surface.key_value_rows());
        let json_binding =
            surface_round_trip.compact_manifest_handoff_identity_report(&manifest_identity);

        assert!(
            report.evidence.iter().any(|row| row
                .contains("trust-ir native_verification_bundle_handoff_manifest ")
                && row.contains(&format!(
                    "schema={}",
                    TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA
                ))
                && row.contains("row_key=handoff.schema")
                && row.contains(&format!("row_value={}", descriptor.schema))
                && row.contains(&format!(
                    "manifest_line=handoff.schema={}",
                    descriptor.schema
                ))),
            "Petri producer should emit the component schema while forwarding the trust-ir-owned handoff descriptor schema"
        );
        assert!(
            report.evidence.iter().any(|row| row
                .contains("trust-ir native_verification_bundle_handoff_completeness ")
                && row.contains(&format!(
                    "schema={}",
                    TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA
                ))
                && row.contains(&format!(
                    "manifest_schema={}",
                    TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA
                ))),
            "Petri producer should emit the normalized trust-ir handoff completeness schema"
        );
        assert!(
            report.evidence.iter().any(|row| row
                .contains("trust-ir native_verification_bundle_handoff_manifest_identity ")
                && row.contains(&format!(
                    "linked_handoff_schema={}",
                    TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA
                ))
                && row.contains(&format!(
                    "manifest_line=manifest_identity.linked_handoff.schema={}",
                    TRUST_IR_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA
                ))),
            "Petri producer should link manifest identity rows to the normalized handoff manifest"
        );
        assert!(
            report.evidence.iter().any(|row| row.contains(
                "trust-ir native_verification_bundle_handoff_replay_contract_surface "
            ) && row.contains(
                "manifest_line=replay_contract_surface.schema=trust_ir.native.petri_successor.bundle_solver_evidence_handoff.replay_contract_surface.v1"
            )),
            "Petri producer should emit the trust-ir replay contract surface rows"
        );
        assert!(
            report.evidence.iter().any(|row| row.contains(
                "trust-ir native_verification_bundle_handoff_replay_contract_surface_round_trip "
            ) && row.contains("manifest_line=round_trip.status_code=valid")
                && row.contains("linked_replay_contract_surface_component=native_verification_bundle_handoff_replay_contract_surface")),
            "Petri producer should emit the replay surface round-trip rows"
        );
        assert!(
            report.evidence.iter().any(|row| row.contains(
                "trust-ir native_verification_bundle_handoff_replay_contract_report_identity "
            ) && row.contains(&format!(
                "manifest_line=round_trip_report.digest={}",
                surface_round_trip.identity_digest()
            ))),
            "Petri producer should emit the trust-ir round-trip report identity digest"
        );
        assert!(
            report.evidence.iter().any(|row| {
                row.contains(
                "trust-ir native_verification_bundle_handoff_replay_contract_json_manifest_binding "
            ) && row.contains(&format!(
                "manifest_line=json_manifest_binding.status={}",
                json_binding.status_code
            ))
            }),
            "Petri producer should emit the trust-ir JSON manifest binding status"
        );
        assert!(
            report.evidence.iter().any(|row| {
                row.contains(
                "trust-ir native_verification_bundle_handoff_replay_contract_json_manifest_binding "
            ) && row.contains(&format!(
                "manifest_line=round_trip_report.identity_digest={}",
                json_binding.round_trip_report_identity_digest
            )) && row.contains(&format!(
                "linked_manifest_identity_schema={}",
                manifest_identity.schema
            ))
            }),
            "Petri producer should bind JSON replay identity back to the trust-ir handoff manifest"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_execution_plan_consumes_authoritative_call_packet_descriptor() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");

        assert_eq!(
            evidence_field(execution_plan, "call_packet_api"),
            Some("trust-cg::petri_native_successor_call_packet_from_trust_ir_bundle")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_schema"),
            Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA)
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_descriptor_available"),
            Some("true")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_descriptor_source"),
            Some("trust-cg::petri_native_successor_downstream_contract_descriptor.call_packet")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_descriptor_status_code"),
            Some("authoritative")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_descriptor_authoritative"),
            Some("true")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_descriptor_dependency"),
            Some("trust-cg::petri_native_successor_downstream_contract_descriptor.call_packet")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_descriptor_upstream_ask"),
            Some("none")
        );
        assert_eq!(
            evidence_field(execution_plan, "call_packet_api_status_code"),
            Some("available")
        );
        // With the Petri native producer now attaching semantic evidence, the call
        // packet, runtime readiness, and execution authority all reach their authoritative
        // / ready states. Production selection still emits `production_selected=false`
        // and the row remains `fail_closed=true` because final production activation is
        // gated downstream.
        assert_eq!(
            evidence_field(execution_plan, "call_packet_available"),
            Some("true")
        );
        assert_eq!(
            evidence_field(execution_plan, "runtime_readiness_ready_for_runtime_call"),
            Some("true")
        );
        assert_eq!(
            evidence_field(execution_plan, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(execution_plan, "fail_closed"), Some("true"));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn trust_cg_petri_compile_artifact_handoff_surface_reexports_are_linked() {
        let surface = trust_cg_petri_compile_artifact_handoff_surface();
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let compile_artifact_handoff_surface = downstream_contract.compile_artifact_handoff;
        assert_eq!(
            surface.api,
            "trust-cg::petri_native_successor_compile_artifact_handoff_evidence"
        );
        assert_eq!(
            surface.installed_artifact_api,
            "InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence"
        );
        assert_eq!(surface.installed_artifact_type, "InstalledArtifact");
        assert_eq!(surface.installed_artifact_required_trust_cg_rev, "00597478");
        assert_eq!(surface.schema, compile_artifact_handoff_surface.schema);
        assert_eq!(
            surface.schema_version,
            compile_artifact_handoff_surface.schema_version
        );
        assert_eq!(
            surface.input_type,
            "PetriNativeSuccessorCompileArtifactHandoffInput"
        );
        assert_eq!(
            surface.evidence_type,
            "PetriNativeSuccessorCompileArtifactHandoffEvidence"
        );
        assert_eq!(
            surface.blocker_type,
            "PetriNativeSuccessorCompileArtifactHandoffBlocker"
        );

        let _: TrustCgPetriCompileArtifactHandoffEvidenceBuilder =
            tla_trust_cg::petri_native_successor_compile_artifact_handoff_evidence;
        let _: TrustCgPetriInstalledArtifactHandoffEvidenceBuilder =
            tla_trust_cg::InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence;
        let _ = std::mem::size_of::<tla_trust_cg::InstalledArtifact>();
        let handoff = tla_trust_cg::petri_native_successor_compile_artifact_handoff_evidence(
            tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffInput::default(),
        );
        assert_eq!(
            handoff.schema,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA
        );
        assert_eq!(handoff.status.as_str(), "blocked");
        assert_eq!(handoff.reason_code, Some("missing_native_payload_sha256"));
        assert_eq!(
            handoff.required_field,
            Some("compiled_artifact.native_payload_sha256")
        );
        assert_eq!(
            handoff.required_evidence,
            Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA)
        );
        assert!(!handoff.is_ready());
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn trust_cg_petri_compile_artifact_handoff_attempt_populates_known_entry_symbol() {
        let attempt = trust_cg_petri_compile_artifact_handoff_attempt(None, None, None);

        assert_eq!(
            attempt.evidence.entry_symbol.as_deref(),
            Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL)
        );
        assert_eq!(
            attempt.evidence.reason_code,
            Some("missing_native_payload_sha256")
        );
        assert_eq!(
            attempt.evidence.required_field,
            Some("compiled_artifact.native_payload_sha256")
        );
        assert_eq!(attempt.entry_symbol_source, "petri_successor_entry_symbol");
        assert_eq!(attempt.native_payload_source, "unavailable");
        assert!(!attempt.installed_artifact_available);
        assert_eq!(attempt.ty_wiring_status, "missing_installed_artifact");
        assert_eq!(
            attempt.ty_wiring_blocker,
            "missing_ty_installed_artifact_wiring"
        );
        assert_eq!(
            attempt.ty_required_field,
            "petri_native_successor_capability_report.installed_artifact"
        );
        assert_eq!(
            attempt.missing_ty_artifact_field,
            "petri_native_successor_capability_report.installed_artifact"
        );
        assert_eq!(attempt.missing_trust_cg_artifact_field, "none");
        assert_eq!(
            attempt.missing_artifact_blocker,
            "missing_ty_installed_artifact_wiring"
        );
        assert_eq!(
            attempt.next_production_api,
            "InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence"
        );
        assert_eq!(
            attempt.next_production_input,
            "petri_native_successor_capability_report.installed_artifact"
        );
        assert_eq!(
            attempt.next_production_reason_code,
            "missing_ty_installed_artifact_wiring"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn trust_cg_petri_compile_artifact_handoff_attempt_uses_callable_contract_payload() {
        let contract = tla_trust_cg::PetriNativeSuccessorCallableContract {
            schema: tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALLABLE_CONTRACT_SCHEMA,
            schema_version: tla_trust_cg::PETRI_NATIVE_SUCCESSOR_CALLABLE_CONTRACT_SCHEMA_VERSION,
            consumer: "mcc".to_owned(),
            consumer_mode: "ty_petri_native_jit".to_owned(),
            kind: "petri_successor".to_owned(),
            surface: "native_successor",
            entry_function: PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL.to_owned(),
            state_encoding: tla_trust_cg::PETRI_NATIVE_SUCCESSOR_STATE_ENCODING_STABLE_BYTES_V1,
            input_state_bytes: 24,
            output_state_bytes: 24,
            state_alignment_bytes: TRUST_CG_PETRI_NATIVE_EXECUTION_STATE_ALIGNMENT_BYTES,
            artifact_id: "artifact:test".to_owned(),
            source_sha256: "sha256:source".to_owned(),
            trust_ir_sha256: "sha256:trust_ir".to_owned(),
            native_payload_sha256: "sha256:native-payload".to_owned(),
            transport_digest: "sha256:transport".to_owned(),
            bundle_digest: "sha256:bundle".to_owned(),
            target_abi_digest: None,
            callable_contract_sha256: "sha256:callable-contract".to_owned(),
        };

        let attempt = trust_cg_petri_compile_artifact_handoff_attempt(Some(&contract), None, None);

        assert_eq!(
            attempt.evidence.native_payload_sha256.as_deref(),
            Some("sha256:native-payload")
        );
        assert_eq!(
            attempt.evidence.entry_symbol.as_deref(),
            Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL)
        );
        assert_eq!(
            attempt.evidence.reason_code,
            Some("missing_callable_pointer")
        );
        assert_eq!(
            attempt.evidence.required_field,
            Some("compiled_artifact.callable_pointer")
        );
        assert_eq!(
            attempt.entry_symbol_source,
            "callable_contract.entry_function"
        );
        assert_eq!(
            attempt.native_payload_source,
            "callable_contract.native_payload_sha256"
        );
        assert_eq!(attempt.real_artifact_source, "callable_contract");
        assert!(!attempt.installed_artifact_available);
        assert_eq!(attempt.ty_wiring_status, "missing_installed_artifact");
        assert_eq!(
            attempt.ty_wiring_blocker,
            "missing_ty_installed_artifact_wiring"
        );
        assert_eq!(
            attempt.ty_required_field,
            "petri_native_successor_capability_report.installed_artifact"
        );
        assert_eq!(
            attempt.missing_ty_artifact_field,
            "petri_native_successor_capability_report.installed_artifact"
        );
        assert_eq!(attempt.missing_trust_cg_artifact_field, "none");
        assert_eq!(
            attempt.missing_artifact_blocker,
            "missing_ty_installed_artifact_wiring"
        );
        assert_eq!(
            attempt.next_production_input,
            "petri_native_successor_capability_report.installed_artifact"
        );
        assert_eq!(
            attempt.next_production_reason_code,
            "missing_ty_installed_artifact_wiring"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn trust_cg_petri_runtime_readiness_surface_reexports_are_linked() {
        let surface = trust_cg_petri_runtime_readiness_surface();
        let contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let runtime_readiness_surface = contract.runtime_readiness;
        let mock_executable_call_surface = contract.mock_executable_call;
        assert_eq!(
            surface.api,
            "trust-cg::petri_native_successor_runtime_readiness_packet"
        );
        assert_eq!(surface.schema, runtime_readiness_surface.schema);
        assert_eq!(
            surface.schema_version,
            runtime_readiness_surface.schema_version
        );
        assert_eq!(
            surface.packet_type,
            "PetriNativeSuccessorRuntimeReadinessPacket"
        );
        assert_eq!(
            surface.mock_executable_call_api,
            "trust-cg::petri_native_successor_mock_executable_call_dry_run"
        );
        assert_eq!(
            surface.mock_executable_call_schema,
            mock_executable_call_surface.schema
        );
        assert_eq!(
            surface.mock_executable_call_schema_version,
            mock_executable_call_surface.schema_version
        );
        assert_eq!(surface.mock_executable_call_role, "test_diagnostic_only");
        assert!(surface.mock_executable_call_descriptor_available);
        assert!(surface.mock_executable_call_descriptor_authoritative);
        assert_eq!(
            surface.mock_executable_call_descriptor_source,
            TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API
        );
        assert_eq!(
            surface.mock_executable_call_descriptor_name,
            mock_executable_call_surface.name
        );

        let readiness = tla_trust_cg::petri_native_successor_runtime_readiness_packet(
            None, None, None, None, None, 0,
        );
        assert_eq!(
            readiness.schema,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_PACKET_SCHEMA
        );
        assert_eq!(readiness.status.as_str(), "blocked");
        assert!(!readiness.is_ready_for_runtime_call());
        assert_eq!(
            readiness.reason_code,
            Some("missing_native_install_gate_packet")
        );
        assert_eq!(
            contract.schema,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_DOWNSTREAM_CONTRACT_SCHEMA
        );
        assert!(contract
            .runtime_readiness
            .required_fields
            .contains(&"native_install_gate_packet"));
        assert!(contract
            .compile_artifact_handoff
            .required_fields
            .contains(&"compiled_artifact.native_payload_sha256"));
        assert!(contract
            .compile_artifact_handoff
            .blocker_codes
            .contains(&"missing_callable_pointer"));
        assert!(contract
            .mock_executable_call
            .required_fields
            .contains(&"mock_executable_call_gate"));
        assert_eq!(
            contract.trust_ir_native_bundle_identity,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_TRUST_IR_BUNDLE_IDENTITY_DESCRIPTOR
        );
        assert_eq!(
            contract.trust_ir_native_bundle_identity,
            tla_trust_cg::petri_native_successor_trust_ir_bundle_identity_descriptor()
        );
        assert_eq!(
            contract.trust_ir_native_bundle_identity.schema,
            "trust_ir.native.bundle_identity_contract.v1"
        );
        let production_gate =
            tla_trust_cg::PetriNativeSuccessorMockExecutableCallGate::disabled_for_production();
        assert!(!production_gate.enabled);
        assert_eq!(production_gate.gate_kind, "production_fail_closed");
        assert_eq!(
            surface.mock_executable_call_gate_enabled,
            production_gate.enabled
        );
        assert_eq!(
            surface.mock_executable_call_gate_kind,
            production_gate.gate_kind
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_execution_plan_sources_runtime_readiness_descriptor_fields() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        let contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let runtime_readiness_surface = contract.runtime_readiness;
        let mock_executable_call_surface = contract.mock_executable_call;
        let runtime_readiness_schema_version = runtime_readiness_surface.schema_version.to_string();
        let runtime_readiness_required_fields = runtime_readiness_surface.required_fields.join(",");
        let runtime_readiness_status_codes = runtime_readiness_surface.status_codes.join(",");
        let mock_executable_call_schema_version =
            mock_executable_call_surface.schema_version.to_string();
        let mock_executable_call_required_fields =
            mock_executable_call_surface.required_fields.join(",");
        let mock_executable_call_status_codes = mock_executable_call_surface.status_codes.join(",");

        assert_eq!(
            evidence_field(execution_plan, "runtime_readiness_schema"),
            Some(runtime_readiness_surface.schema)
        );
        assert_eq!(
            evidence_field(execution_plan, "runtime_readiness_schema_version"),
            Some(runtime_readiness_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(execution_plan, "downstream_runtime_readiness_surface"),
            Some(runtime_readiness_surface.name)
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "downstream_runtime_readiness_required_fields"
            ),
            Some(runtime_readiness_required_fields.as_str())
        );
        assert_eq!(
            evidence_field(execution_plan, "downstream_runtime_readiness_status_codes"),
            Some(runtime_readiness_status_codes.as_str())
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_schema"),
            Some(mock_executable_call_surface.schema)
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_schema_version"),
            Some(mock_executable_call_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_descriptor_available"),
            Some("true")
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "mock_executable_call_descriptor_authoritative"
            ),
            Some("true")
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_descriptor_source"),
            Some(TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API)
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_descriptor_name"),
            Some(mock_executable_call_surface.name)
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_gate_kind"),
            Some("production_fail_closed")
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_gate_enabled"),
            Some("false")
        );
        assert_eq!(
            evidence_field(execution_plan, "mock_executable_call_production_enabled"),
            Some("false")
        );
        assert_eq!(
            evidence_field(execution_plan, "downstream_mock_executable_call_surface"),
            Some(mock_executable_call_surface.name)
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "downstream_mock_executable_call_required_fields"
            ),
            Some(mock_executable_call_required_fields.as_str())
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "downstream_mock_executable_call_status_codes"
            ),
            Some(mock_executable_call_status_codes.as_str())
        );
        let runtime_status_code = evidence_field(execution_plan, "runtime_readiness_status_code")
            .expect("runtime readiness status code should be emitted");
        assert!(
            runtime_readiness_surface
                .status_codes
                .contains(&runtime_status_code),
            "runtime readiness status should come from the trust-codegen descriptor: {execution_plan}"
        );
        let runtime_blocker_code = evidence_field(execution_plan, "runtime_readiness_blocker_code")
            .expect("runtime readiness blocker code should be emitted");
        // When the producer reaches `ready_for_runtime_call`, the blocker code is "none"
        // (not in the trust-codegen descriptor's blocker list). Otherwise it must be a known
        // blocker. Accept either to track both the producer's blocked and ready paths.
        assert!(
            runtime_blocker_code == "none"
                || runtime_readiness_surface
                    .blocker_codes
                    .contains(&runtime_blocker_code),
            "runtime readiness blocker should come from the trust-codegen descriptor: {execution_plan}"
        );
        let runtime_ready_for_runtime_call =
            evidence_field(execution_plan, "runtime_readiness_ready_for_runtime_call")
                .expect("runtime readiness ready flag should be emitted");
        assert!(
            runtime_ready_for_runtime_call == "true" || runtime_ready_for_runtime_call == "false",
            "runtime readiness ready flag should be boolean: {execution_plan}"
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "runtime_readiness_status_in_downstream_contract"
            ),
            Some("true")
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "runtime_readiness_blocker_in_downstream_contract"
            ),
            Some("true")
        );
        assert_eq!(
            evidence_field(execution_plan, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(execution_plan, "fail_closed"), Some("true"));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_execution_plan_sources_execution_authority_summary_rows() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        let contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let execution_authority_surface = contract.execution_authority;
        let execution_authority_schema_version =
            execution_authority_surface.schema_version.to_string();
        let execution_authority_required_fields =
            execution_authority_surface.required_fields.join(",");
        let execution_authority_status_codes = execution_authority_surface.status_codes.join(",");

        assert_eq!(
            evidence_field(execution_plan, "execution_authority_api"),
            Some("trust-cg::petri_native_successor_execution_authority_decision")
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "execution_authority_manifest_validation_api"
            ),
            Some("PetriNativeSuccessorExecutionAuthorityDecision::manifest_validation_report")
        );
        assert_eq!(
            evidence_field(execution_plan, "execution_authority_schema"),
            Some(execution_authority_surface.schema)
        );
        assert_eq!(
            evidence_field(execution_plan, "execution_authority_schema_version"),
            Some(execution_authority_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(execution_plan, "downstream_execution_authority_surface"),
            Some(execution_authority_surface.name)
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "downstream_execution_authority_required_fields"
            ),
            Some(execution_authority_required_fields.as_str())
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "downstream_execution_authority_status_codes"
            ),
            Some(execution_authority_status_codes.as_str())
        );
        let authority_status_code =
            evidence_field(execution_plan, "execution_authority_status_code")
                .expect("execution authority status should be emitted");
        assert!(
            execution_authority_surface
                .status_codes
                .contains(&authority_status_code),
            "execution authority status should come from the trust-codegen descriptor: {execution_plan}"
        );
        let authority_reason_code =
            evidence_field(execution_plan, "execution_authority_reason_code")
                .expect("execution authority reason should be emitted");
        assert!(
            authority_reason_code == "none"
                || execution_authority_surface
                    .blocker_codes
                    .contains(&authority_reason_code),
            "execution authority reason should come from the trust-codegen descriptor: {execution_plan}"
        );
        // With the Petri native producer attaching semantic evidence, the execution
        // authority decision now reaches `authorized`. The authority summary row also
        // promotes to `accepted` / non-fail-closed. Per-execution-plan production
        // remains `production_selected=false` and the overall row stays `fail_closed=true`
        // because the final production gate is held by downstream layers.
        let authority_authorized = evidence_field(
            execution_plan,
            "execution_authority_authorized_for_execution",
        )
        .expect("execution authority authorized flag should be emitted");
        assert!(
            authority_authorized == "true" || authority_authorized == "false",
            "execution authority authorized flag should be boolean: {execution_plan}"
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "execution_authority_is_authorized_for_execution"
            ),
            Some(authority_authorized)
        );
        assert_eq!(
            evidence_field(execution_plan, "execution_authority_summary_schema"),
            Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_SCHEMA)
        );
        assert_eq!(
            evidence_field(execution_plan, "execution_authority_summary_api"),
            Some("PetriNativeSuccessorExecutionAuthorityDecision::compact_authority_summary")
        );
        let summary_status_code =
            evidence_field(execution_plan, "execution_authority_summary_status_code")
                .expect("execution authority summary status should be emitted");
        assert!(
            summary_status_code == "fail_closed" || summary_status_code == "accepted",
            "summary status should be `fail_closed` or `accepted`: {execution_plan}"
        );
        assert_eq!(
            evidence_field(execution_plan, "execution_authority_summary_reason_code"),
            Some(authority_reason_code)
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "execution_authority_summary_validation_status_code"
            ),
            Some("accepted")
        );
        let summary_fail_closed =
            evidence_field(execution_plan, "execution_authority_summary_fail_closed")
                .expect("execution authority summary fail_closed should be emitted");
        let summary_accepted =
            evidence_field(execution_plan, "execution_authority_summary_accepted")
                .expect("execution authority summary accepted should be emitted");
        assert!(
            summary_fail_closed == "true" || summary_fail_closed == "false",
            "summary fail_closed should be boolean: {execution_plan}"
        );
        assert!(
            summary_accepted == "true" || summary_accepted == "false",
            "summary accepted should be boolean: {execution_plan}"
        );
        assert_eq!(
            evidence_field(execution_plan, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(execution_plan, "fail_closed"), Some("true"));

        let summary_row_count =
            evidence_field(execution_plan, "execution_authority_summary_row_count")
                .expect("execution authority summary row count should be emitted")
                .parse::<usize>()
                .expect("summary row count should parse");
        let summary_rows: Vec<_> = report
            .evidence
            .iter()
            .filter(|evidence| {
                evidence.contains("trust-cg petri_native_successor_execution_authority_summary")
            })
            .collect();
        assert_eq!(summary_rows.len(), summary_row_count);
        assert!(summary_rows.iter().all(|row| {
            evidence_field(row.as_str(), "summary_validation_status_code") == Some("accepted")
                && evidence_field(row.as_str(), "summary_status_code") == Some(summary_status_code)
                && evidence_field(row.as_str(), "summary_reason_code")
                    == Some(authority_reason_code)
                && evidence_field(row.as_str(), "summary_accepted") == Some(summary_accepted)
                && evidence_field(row.as_str(), "summary_fail_closed") == Some(summary_fail_closed)
                && evidence_field(row.as_str(), "production_selected") == Some("false")
        }));
        assert!(summary_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key") == Some("summary.schema")
                && evidence_field(row.as_str(), "row_value")
                    == Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_SCHEMA)
                && evidence_field(row.as_str(), "manifest_line").is_some_and(|line| {
                    line == format!(
                        "summary.schema={}",
                        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_SCHEMA
                    )
                })
        }));
        assert!(summary_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key") == Some("authority.authorized_for_execution")
                && evidence_field(row.as_str(), "row_value") == Some(authority_authorized)
        }));
        assert!(summary_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key") == Some("summary.fail_closed")
                && evidence_field(row.as_str(), "row_value") == Some(summary_fail_closed)
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_execution_plan_sources_production_selection_decision() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        let selection_schema_version =
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_SCHEMA_VERSION.to_string();
        let selection_status_code =
            evidence_field(execution_plan, "production_selection_status_code")
                .expect("production selection status should be emitted");
        let selection_reason_code =
            evidence_field(execution_plan, "production_selection_reason_code")
                .expect("production selection reason should be emitted");
        let execution_authority_reason_code =
            evidence_field(execution_plan, "execution_authority_reason_code")
                .expect("execution authority reason should be emitted");
        let production_selected = evidence_field(execution_plan, "production_selected")
            .expect("production selection should be emitted");

        assert_eq!(
            evidence_field(execution_plan, "production_selection_api"),
            Some("trust-cg::petri_native_successor_production_selection_decision")
        );
        assert_eq!(
            evidence_field(execution_plan, "production_selection_schema"),
            Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_SCHEMA)
        );
        assert_eq!(
            evidence_field(execution_plan, "production_selection_schema_version"),
            Some(selection_schema_version.as_str())
        );
        assert!(
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_STATUS_CODES
                .contains(&selection_status_code),
            "production selection status should come from trust-cg: {execution_plan}"
        );
        assert!(
            selection_reason_code == "none"
                || tla_trust_cg::PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_REASON_CODES
                    .contains(&selection_reason_code),
            "production selection reason should come from trust-cg: {execution_plan}"
        );
        // With the Petri native producer attaching semantic evidence, the production
        // selection decision now reaches `selected` (status) / `none` (reason) when
        // the call packet hashes and execution-authority gates all align. When any
        // upstream gate fails, the row falls back to `fail_closed` /
        // `execution_authority_not_authorized`. Both shapes are valid.
        assert!(
            selection_status_code == "fail_closed" || selection_status_code == "selected",
            "selection status should be `fail_closed` or `selected`: {execution_plan}"
        );
        assert!(
            selection_reason_code == "none"
                || selection_reason_code == "execution_authority_not_authorized",
            "selection reason should be `none` or `execution_authority_not_authorized`: {execution_plan}"
        );
        assert_eq!(
            evidence_field(execution_plan, "production_selection_source_reason_code"),
            Some(execution_authority_reason_code)
        );
        let selected_for_native_execution = evidence_field(
            execution_plan,
            "production_selection_selected_for_native_execution",
        )
        .expect("selected_for_native_execution should be emitted");
        assert!(
            selected_for_native_execution == "true" || selected_for_native_execution == "false",
            "selected_for_native_execution should be boolean: {execution_plan}"
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "production_selection_is_selected_for_native_execution"
            ),
            Some(selected_for_native_execution)
        );
        // The overall row remains gated downstream: `production_selected=false` and
        // `fail_closed=true` even when the per-execution-plan selection reaches
        // `selected_for_native_execution=true`.
        assert_eq!(production_selected, "false");
        let selection_fail_closed =
            evidence_field(execution_plan, "production_selection_fail_closed")
                .expect("selection fail_closed should be emitted");
        assert!(
            selection_fail_closed == "true" || selection_fail_closed == "false",
            "selection fail_closed should be boolean: {execution_plan}"
        );
        assert_eq!(evidence_field(execution_plan, "fail_closed"), Some("true"));
        assert_eq!(
            evidence_field(
                execution_plan,
                "production_selection_execution_authority_sha256"
            ),
            evidence_field(execution_plan, "execution_authority_sha256")
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "production_selection_execution_authority_hash_current"
            ),
            Some("true")
        );
        let call_packet_sha256 =
            evidence_field(execution_plan, "production_selection_call_packet_sha256")
                .expect("call_packet_sha256 should be emitted");
        assert!(
            call_packet_sha256 == "none" || call_packet_sha256.starts_with("sha256:"),
            "call_packet_sha256 should be `none` or `sha256:...`: {execution_plan}"
        );
        let call_packet_hash_current = evidence_field(
            execution_plan,
            "production_selection_call_packet_hash_current",
        )
        .expect("call_packet_hash_current should be emitted");
        assert!(
            call_packet_hash_current == "true" || call_packet_hash_current == "false",
            "call_packet_hash_current should be boolean: {execution_plan}"
        );
        let callable_lane_admitted = evidence_field(
            execution_plan,
            "production_selection_callable_lane_admitted",
        )
        .expect("callable_lane_admitted should be emitted");
        assert!(
            callable_lane_admitted == "true" || callable_lane_admitted == "false",
            "callable_lane_admitted should be boolean: {execution_plan}"
        );
        let runtime_ready_for_call = evidence_field(
            execution_plan,
            "production_selection_runtime_ready_for_call",
        )
        .expect("runtime_ready_for_call should be emitted");
        assert!(
            runtime_ready_for_call == "true" || runtime_ready_for_call == "false",
            "runtime_ready_for_call should be boolean: {execution_plan}"
        );
        let runtime_authorizes_useful_native = evidence_field(
            execution_plan,
            "production_selection_runtime_authorizes_useful_native",
        )
        .expect("runtime_authorizes_useful_native should be emitted");
        assert!(
            runtime_authorizes_useful_native == "true"
                || runtime_authorizes_useful_native == "false",
            "runtime_authorizes_useful_native should be boolean: {execution_plan}"
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "production_selection_vector_constant_lowering_schema"
            ),
            Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_VECTOR_CONSTANT_LOWERING_EVIDENCE_SCHEMA)
        );
        assert_eq!(
            evidence_field(
                execution_plan,
                "production_selection_vector_constant_lowering_supported"
            ),
            Some("true")
        );
        assert!(
            evidence_field(execution_plan, "production_selection_sha256")
                .is_some_and(|hash| hash.starts_with("sha256:"))
        );

        let selection_row_count = evidence_field(execution_plan, "production_selection_row_count")
            .expect("production selection row count should be emitted")
            .parse::<usize>()
            .expect("production selection row count should parse");
        let selection_rows: Vec<_> = report
            .evidence
            .iter()
            .filter(|evidence| {
                evidence.contains("trust-cg petri_native_successor_production_selection")
            })
            .collect();
        assert_eq!(selection_rows.len(), selection_row_count);
        assert!(selection_rows.iter().all(|row| {
            evidence_field(row.as_str(), "selection_status_code") == Some(selection_status_code)
                && evidence_field(row.as_str(), "selection_reason_code")
                    == Some(selection_reason_code)
                && evidence_field(row.as_str(), "production_selected") == Some(production_selected)
        }));
        assert!(selection_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key") == Some("selection.schema")
                && evidence_field(row.as_str(), "row_value")
                    == Some(tla_trust_cg::PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_SCHEMA)
                && evidence_field(row.as_str(), "manifest_line").is_some_and(|line| {
                    line == format!(
                        "selection.schema={}",
                        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_SCHEMA
                    )
                })
        }));
        // The compact field `selection_reason_code` uses the kernel-level "none"
        // sentinel when the trust-codegen selection emits `reason_code = None`; the
        // per-row `selection.reason_code` value (emitted via the trust-codegen manifest
        // row) is the raw producer string, which is empty when no reason is
        // present. Map the kernel sentinel to the empty row value before
        // comparing.
        let selection_reason_code_row_value = if selection_reason_code == "none" {
            ""
        } else {
            selection_reason_code
        };
        assert!(selection_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key") == Some("selection.reason_code")
                && evidence_field(row.as_str(), "row_value")
                    == Some(selection_reason_code_row_value)
        }));
        assert!(selection_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key")
                == Some("selection.selected_for_native_execution")
                && evidence_field(row.as_str(), "row_value") == Some(selected_for_native_execution)
        }));
        assert!(selection_rows.iter().any(|row| {
            evidence_field(row.as_str(), "row_key") == Some("trust-cg.vector_constant_lowering.schema")
                && evidence_field(row.as_str(), "row_value")
                    == Some(
                        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_VECTOR_CONSTANT_LOWERING_EVIDENCE_SCHEMA,
                    )
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn assert_execution_plan_exposes_compile_artifact_handoff(
        execution_plan: &str,
        installed_artifact_expected: bool,
    ) {
        assert!(
            execution_plan.contains(
                "downstream_compile_artifact_handoff_surface=petri_native_successor_compile_artifact_handoff"
            )
                && execution_plan.contains(
                    "downstream_compile_artifact_handoff_required_fields=compiled_artifact.native_payload_sha256,compiled_artifact.entry_symbol,compiled_artifact.callable_pointer,compiled_artifact.executable_region_sha256,compiled_artifact.lifetime_owner,compiled_artifact.current_generation"
                )
                && execution_plan.contains(
                    "downstream_compile_artifact_handoff_status_codes=ready,blocked"
                )
                && execution_plan.contains(
                    "downstream_compile_artifact_handoff_blocker_codes_count=6"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_api=trust-cg::petri_native_successor_compile_artifact_handoff_evidence"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_schema=trust-cg.petri.native_successor.compile_artifact_handoff.v1"
                )
                && execution_plan.contains("compile_artifact_handoff_schema_version=1")
                && execution_plan.contains(
                    "compile_artifact_handoff_input_type=PetriNativeSuccessorCompileArtifactHandoffInput"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_evidence_type=PetriNativeSuccessorCompileArtifactHandoffEvidence"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_blocker_type=PetriNativeSuccessorCompileArtifactHandoffBlocker"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_required_trust_cg_rev=df133c3f"
                )
                && execution_plan.contains(&format!(
                    "compile_artifact_handoff_current_trust_cg_rev={}",
                    TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_CURRENT_TRUST_CG_REV
                ))
                && execution_plan.contains(
                    "compile_artifact_handoff_installed_artifact_api=InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_installed_artifact_type=InstalledArtifact"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_installed_artifact_required_trust_cg_rev=00597478"
                )
                && execution_plan.contains(
                    "compile_artifact_handoff_native_library_bridge_api=NativeLibrary::petri_native_successor_installed_artifact"
                )
                && execution_plan.contains("compile_artifact_handoff_available=true")
                && execution_plan
                    .contains("compile_artifact_handoff_status_in_downstream_contract=true")
                && execution_plan.contains(
                    "compile_artifact_handoff_blocker_in_downstream_contract=true"
                )
                && execution_plan.contains("compile_artifact_handoff_sha256=sha256:")
                && execution_plan.contains("compile_artifact_handoff_population_attempted=true")
                && execution_plan.contains("compile_artifact_handoff_entry_symbol_present=true")
                && execution_plan.contains(&format!(
                    "compile_artifact_handoff_entry_symbol={}",
                    PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL
                )),
            "execution plan should expose trust-codegen compile-artifact handoff: {execution_plan}"
        );

        if installed_artifact_expected {
            assert!(
                execution_plan
                    .contains("compile_artifact_handoff_installed_artifact_available=true")
                    && execution_plan.contains(
                        "compile_artifact_handoff_installed_artifact_production_status=available"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_installed_artifact_production_reason_code=available"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_installed_artifact_production_path=NativeLibrary::petri_native_successor_installed_artifact"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_installed_artifact_production_blocker=none"
                    )
                    && execution_plan
                        .contains("compile_artifact_handoff_ty_wiring_status=available")
                    && execution_plan
                        .contains("compile_artifact_handoff_ty_wiring_blocker=none")
                    && execution_plan.contains("compile_artifact_handoff_ty_required_field=none")
                    && execution_plan.contains("compile_artifact_handoff_ready=true")
                    && execution_plan.contains("compile_artifact_handoff_status_code=ready")
                    && execution_plan.contains("compile_artifact_handoff_reason_code=none")
                    && execution_plan.contains("compile_artifact_handoff_blocker_code=none")
                    && execution_plan.contains("compile_artifact_handoff_required_field=none")
                    && execution_plan.contains("compile_artifact_handoff_required_evidence=none")
                    && execution_plan
                        .contains("compile_artifact_handoff_real_artifact_source=installed_artifact")
                    && execution_plan.contains(
                        "compile_artifact_handoff_entry_symbol_source=InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence"
                    )
                    && execution_plan.contains("compile_artifact_handoff_native_payload_present=true")
                    && execution_plan.contains(
                        "compile_artifact_handoff_native_payload_source=InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence"
                    )
                    && execution_plan
                        .contains("compile_artifact_handoff_callable_pointer_present=true")
                    && execution_plan
                        .contains("compile_artifact_handoff_executable_region_present=true")
                    && execution_plan
                        .contains("compile_artifact_handoff_lifetime_owner_present=true")
                    && execution_plan
                        .contains("compile_artifact_handoff_current_generation_present=true")
                    && execution_plan
                        .contains("compile_artifact_handoff_missing_ty_artifact_field=none")
                    && execution_plan
                        .contains("compile_artifact_handoff_missing_trust_cg_artifact_field=none")
                    && execution_plan.contains("compile_artifact_handoff_missing_artifact_blocker=none")
                    && execution_plan.contains(
                        "native_successor_next_production_source=semantic_successor_bridge"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_api=trust-cg::petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_input=ty.petri.native.successor.plan_cache_equivalence.v1"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_evidence=ty.petri.native.successor.plan_cache_equivalence.v1"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_reason_code=missing_semantic_successor_obligation"
                    )
                    && execution_plan
                        .contains("native_successor_next_production_status_code=blocked")
                    && execution_plan.contains(
                        "native_successor_next_production_blocker_stage=semantic_successor_bridge"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_blocker_code=missing_semantic_successor_obligation"
                    ),
                "produced Petri native execution plan must stop at the unadmitted semantic bridge: {execution_plan}"
            );
        } else {
            assert!(
                execution_plan
                    .contains("compile_artifact_handoff_installed_artifact_available=false")
                    && execution_plan.contains(
                        "compile_artifact_handoff_installed_artifact_production_status=not_attempted"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_installed_artifact_production_reason_code=none"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_ty_wiring_status=missing_installed_artifact"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_ty_wiring_blocker=missing_ty_installed_artifact_wiring"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_ty_required_field=petri_native_successor_capability_report.installed_artifact"
                    )
                    && execution_plan.contains("compile_artifact_handoff_ready=false")
                    && execution_plan.contains("compile_artifact_handoff_status_code=blocked")
                    && execution_plan.contains(
                        "compile_artifact_handoff_reason_code=missing_native_payload_sha256"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_blocker_code=missing_native_payload_sha256"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_required_field=compiled_artifact.native_payload_sha256"
                    )
                    && execution_plan.contains(
                        "compile_artifact_handoff_required_evidence=trust-cg.petri.native_successor.compile_artifact_handoff.v1"
                    )
                    && execution_plan
                        .contains("compile_artifact_handoff_real_artifact_source=none")
                    && execution_plan.contains(
                        "compile_artifact_handoff_entry_symbol_source=petri_successor_entry_symbol"
                    )
                    && execution_plan
                        .contains("compile_artifact_handoff_native_payload_present=false")
                    && execution_plan
                        .contains("compile_artifact_handoff_native_payload_source=unavailable")
                    && execution_plan
                        .contains("compile_artifact_handoff_callable_pointer_present=false")
                    && execution_plan
                        .contains("compile_artifact_handoff_executable_region_present=false")
                    && execution_plan
                        .contains("compile_artifact_handoff_lifetime_owner_present=false")
                    && execution_plan
                        .contains("compile_artifact_handoff_current_generation_present=false")
                    && execution_plan.contains(
                        "compile_artifact_handoff_missing_ty_artifact_field=petri_native_successor_capability_report.installed_artifact"
                    )
                    && execution_plan
                        .contains("compile_artifact_handoff_missing_trust_cg_artifact_field=none")
                    && execution_plan.contains(
                        "compile_artifact_handoff_missing_artifact_blocker=missing_ty_installed_artifact_wiring"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_source=compile_artifact_handoff"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_api=InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_input=petri_native_successor_capability_report.installed_artifact"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_evidence=trust-cg.petri.native_successor.compile_artifact_handoff.v1"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_reason_code=missing_ty_installed_artifact_wiring"
                    )
                    && execution_plan
                        .contains("native_successor_next_production_status_code=blocked")
                    && execution_plan.contains(
                        "native_successor_next_production_blocker_stage=compile_artifact_handoff"
                    )
                    && execution_plan.contains(
                        "native_successor_next_production_blocker_code=missing_native_payload_sha256"
                    ),
                "external bundle execution plan should remain fail-closed without a TY InstalledArtifact: {execution_plan}"
            );
        }
    }

    #[test]
    fn trust_cg_petri_native_readiness_status_codes_are_stable() {
        assert_eq!(
            TrustCgPetriNativeReadinessStatus::Available.code(),
            "available"
        );
        assert_eq!(
            TrustCgPetriNativeReadinessStatus::Unavailable.code(),
            "unavailable"
        );
        assert_eq!(TrustCgPetriNativeReadinessStatus::Missing.code(), "missing");
    }

    #[test]
    fn native_producer_rev_evidence_tracks_workspace_pins() {
        let workspace_cargo = include_str!("../../../Cargo.toml");
        let workspace_lock = include_str!("../../../Cargo.lock");
        let dockerfile = include_str!("../../../mcc/Dockerfile.mcc");
        #[cfg(feature = "trust-cg-petri-native")]
        let benchkit = include_str!("../../../mcc/BenchKit_head.sh");

        // Current-rev evidence is authority-bearing packaging metadata. Keep
        // every consumer on one immutable source identity; a branch source or
        // a stale constant must fail this test before an MCC image is built.
        let dep_line_has_rev = |dep_name: &str, rev: &str| -> bool {
            workspace_cargo.lines().any(|line| {
                let trimmed = line.trim_start();
                trimmed.starts_with(&format!("{dep_name} ="))
                    && trimmed.contains(&format!("rev = \"{rev}\""))
            })
        };
        let lock_has_exact_source = |repo: &str, rev: &str| -> bool {
            workspace_lock.contains(&format!(
                "git+https://github.com/alabsystems/{repo}.git?rev={rev}#{rev}"
            ))
        };

        for dep_name in ["trust-ir", "trust-ir-build"] {
            assert!(
                dep_line_has_rev(dep_name, TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV),
                "Petri trust-ir current-rev evidence ({}) must track workspace dependency `{dep_name}`",
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV,
            );
        }
        for dep_name in [
            "trust-cg-codegen",
            "trust-cg-ir",
            "trust-cg-lower",
            "trust-cg-opt",
            "trust-cg-jit-matrix",
        ] {
            assert!(
                dep_line_has_rev(
                    dep_name,
                    TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
                ),
                "Petri trust-cg current-rev evidence ({}) must track workspace dependency `{dep_name}`",
                TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV,
            );
        }
        assert!(
            lock_has_exact_source("trust-ir", TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV),
            "Cargo.lock must resolve the exact TrustIR source identity"
        );
        assert!(
            lock_has_exact_source(
                "trust-cg",
                TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
            ),
            "Cargo.lock must resolve the exact trust-cg source identity"
        );
        for repo in ["trust-ir", "trust-cg"] {
            assert!(
                !workspace_cargo
                    .contains(&format!("alabsystems/{repo}.git\", branch = \"main\"")),
                "workspace dependencies must not use a moving `{repo}` branch"
            );
            assert!(
                !workspace_lock.contains(&format!("alabsystems/{repo}.git?branch=main")),
                "Cargo.lock must not retain a moving `{repo}` source"
            );
        }
        assert!(
            dockerfile.contains(&format!(
                "ARG TRUST_IR_REV={}",
                TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV
            )),
            "Docker TrustIR pin must match native evidence"
        );
        assert!(
            dockerfile.contains(&format!(
                "ARG TRUST_CG_REV={}",
                TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
            )),
            "Docker trust-cg pin must match native evidence"
        );

        assert_eq!(
            TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV,
            TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
        );
        assert_eq!(
            TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_CURRENT_TRUST_CG_REV,
            TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
        );
        assert_eq!(
            TRUST_CG_PETRI_NATIVE_COMPILE_ARTIFACT_HANDOFF_CURRENT_TRUST_CG_REV,
            TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
        );
        #[cfg(feature = "trust-cg-petri-native")]
        assert_eq!(
            TRUST_CG_PETRI_NATIVE_SEMANTIC_BRIDGE_CURRENT_TRUST_CG_REV,
            TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_CURRENT_TRUST_CG_REV
        );
        #[cfg(feature = "trust-cg-petri-native")]
        {
            for dep_name in [
                "ay",
                "ay-dpll",
                "ay-core",
                "ay-proof",
                "ay-allsat",
                "ay-chc",
                "ay-sat",
                "ay-lrat-check",
                "ay-frontend",
                "ay-encode",
            ] {
                assert!(
                    dep_line_has_rev(
                        dep_name,
                        AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV
                    ),
                    "AY current-rev evidence ({}) must track workspace dependency `{dep_name}`",
                    AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV,
                );
            }
            assert!(
                lock_has_exact_source(
                    "ay",
                    AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV
                ),
                "Cargo.lock must resolve the exact AY source identity"
            );
            assert!(
                !workspace_cargo.contains("alabsystems/ay.git\", branch = \"main\""),
                "workspace dependencies must not use a moving AY branch"
            );
            assert!(
                !workspace_lock.contains("alabsystems/ay.git?branch=main"),
                "Cargo.lock must not retain a moving AY source"
            );
            assert!(
                dockerfile.contains(&format!(
                    "ARG AY_REV={}",
                    AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV
                )),
                "Docker AY pin must match native evidence"
            );
            assert!(
                benchkit.contains(&format!(
                    "TY_MCC_PACKAGED_AY_REV:={}",
                    AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV
                )),
                "BenchKit AY fallback must match native evidence"
            );
        }
    }

    fn native_jit_env_lock() -> std::sync::MutexGuard<'static, ()> {
        // Single crate-wide env lock: these tests mutate the native-candidate /
        // transition-parity feature-flag env vars, which production code reads at
        // runtime, so they must serialize against every other module's
        // env-touching test — not just each other.
        crate::env_test_lock()
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            crate::env_guard::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => crate::env_guard::set_var(self.key, value),
                None => crate::env_guard::remove_var(self.key),
            }
        }
    }

    #[test]
    fn marking_to_flat_i64_accepts_i64_range_tokens() {
        let mut flat = Vec::new();
        marking_to_flat_i64(&[0, i64::MAX as u64], &mut flat).unwrap();
        assert_eq!(flat, vec![0, i64::MAX]);
    }

    #[test]
    fn marking_to_flat_i64_rejects_wide_tokens() {
        let mut flat = Vec::new();
        let error = marking_to_flat_i64(&[i64::MAX as u64 + 1], &mut flat).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::TokenExceedsI64 {
                place: 0,
                value: i64::MAX as u64 + 1,
            }
        );
    }

    #[test]
    fn marking_to_flat_i64_clears_output_on_error() {
        let mut flat = vec![99];
        let error = marking_to_flat_i64(&[1, i64::MAX as u64 + 1], &mut flat).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::TokenExceedsI64 {
                place: 1,
                value: i64::MAX as u64 + 1,
            }
        );
        assert!(flat.is_empty());
    }

    #[test]
    fn flat_i64_to_marking_rejects_negative_tokens() {
        let mut marking = Vec::new();
        let error = flat_i64_to_marking(&[1, -1], &mut marking).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::NegativeFlatToken {
                place: 1,
                value: -1,
            }
        );
    }

    #[test]
    fn flat_i64_to_marking_clears_output_on_error() {
        let mut marking = vec![99];
        let error = flat_i64_to_marking(&[1, -1], &mut marking).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::NegativeFlatToken {
                place: 1,
                value: -1,
            }
        );
        assert!(marking.is_empty());
    }

    #[test]
    fn transition_plan_flat_enabled_matches_fire_into() {
        let net = simple_net();
        let mut scratch = PetriKernelScratch::new();
        let outcome =
            checked_fire_transition(&net, TransitionIdx(0), &[5, 0, 0], &mut scratch).unwrap();
        assert_eq!(
            outcome,
            CheckedTransitionOutcome::Enabled {
                successor: vec![3, 1, 3],
            }
        );
    }

    #[test]
    fn cached_transition_successor_into_matches_and_reuses_output_buffer() {
        let net = simple_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let parity = PetriTransitionParityConfig::enabled_for_tests(true);
        let mut scratch = PetriKernelScratch::new();
        let mut successor = Vec::with_capacity(8);
        successor.extend_from_slice(&[99, 99, 99]);
        let initial_capacity = successor.capacity();

        assert_eq!(
            parity
                .checked_transition_successor_cached_into(
                    &net,
                    &cache,
                    TransitionIdx(0),
                    &[5, 0, 0],
                    &mut scratch,
                    &mut successor,
                )
                .unwrap(),
            Some(())
        );

        assert_eq!(successor, vec![3, 1, 3]);
        assert_eq!(successor.capacity(), initial_capacity);
    }

    #[test]
    fn cached_transition_successor_into_rejects_mismatch() {
        let net = simple_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let parity = PetriTransitionParityConfig::enabled_for_tests(true);
        let mut scratch = PetriKernelScratch::new();
        let mut successor = Vec::new();

        let error = parity
            .checked_transition_successor_cached_into(
                &net,
                &cache,
                TransitionIdx(99),
                &[5, 0, 0],
                &mut scratch,
                &mut successor,
            )
            .unwrap_err();

        assert_eq!(
            error,
            PetriKernelError::TransitionOutOfBounds {
                transition: TransitionIdx(99),
                transition_count: 1,
            }
        );
    }

    #[test]
    fn plan_cache_validate_for_net_rejects_place_count_mismatch() {
        let net = simple_net();
        let mut cache = PetriKernelPlanCache::for_net(&net).unwrap();
        cache.place_count = 99;

        let error = cache.validate_for_net(&net).unwrap_err();

        assert_eq!(
            error,
            PetriKernelError::CachePlaceCountMismatch {
                expected: 3,
                actual: 99,
            }
        );
    }

    #[test]
    fn plan_cache_validate_for_net_rejects_transition_plan_order_mismatch() {
        let net = simple_net();
        let mut cache = PetriKernelPlanCache::for_net(&net).unwrap();
        cache.plans[0].transition = TransitionIdx(1);

        let error = cache.validate_for_net(&net).unwrap_err();

        assert_eq!(
            error,
            PetriKernelError::CachePlanTransitionMismatch {
                index: 0,
                transition: TransitionIdx(1),
            }
        );
    }

    #[test]
    fn checked_all_transition_candidates_returns_flat_rows_for_enabled_transitions() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let mut scratch = PetriKernelScratch::new();
        let mut candidates = FlatAllTransitionCandidates::new();

        checked_all_transition_successors_cached_into(
            &net,
            &cache,
            &[2, 1, 0],
            &mut scratch,
            &mut candidates,
        )
        .unwrap();

        assert_eq!(candidates.place_count(), 3);
        assert_eq!(candidates.len(), 2);
        assert_eq!(
            candidates.transition_ids(),
            &[TransitionIdx(0), TransitionIdx(2)]
        );
        assert_eq!(candidates.flat_successors(), &[1, 2, 0, 2, 0, 2]);
        assert_eq!(candidates.flat_successor(0), Some(&[1, 2, 0][..]));
        assert_eq!(candidates.flat_successor(1), Some(&[2, 0, 2][..]));
        assert_eq!(candidates.flat_successor(2), None);
    }

    #[test]
    fn flat_candidates_conform_to_shared_successor_kernel_abi_shape() {
        use tla_jit_abi::{
            FlatBfsStepOutputRef, SuccessorKernelDescriptor, SuccessorKernelOut,
            SuccessorKernelShape,
        };

        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let mut scratch = PetriKernelScratch::new();
        let mut candidates = FlatAllTransitionCandidates::new();

        checked_all_transition_successors_cached_into(
            &net,
            &cache,
            &[2, 1, 0],
            &mut scratch,
            &mut candidates,
        )
        .unwrap();

        let shape = SuccessorKernelShape::new(
            candidates.place_count() as u32,
            0,
            0,
            net.transitions.len() as u32,
        );
        let descriptor = SuccessorKernelDescriptor::new("petri-all-transitions", shape);
        let output = FlatBfsStepOutputRef::from_parts(
            candidates.flat_successors(),
            candidates.place_count(),
            candidates.len(),
            candidates.len() as u32,
            true,
            None,
            None,
        );
        let out_summary =
            SuccessorKernelOut::ok(candidates.len() as u32, candidates.place_count() as u32);

        assert_eq!(descriptor.shape.state_len, 3);
        assert_eq!(descriptor.shape.max_successors, 3);
        assert_eq!(descriptor.shape.successor_buffer_slots(), Some(9));
        assert!(descriptor.requires_parity);
        assert_eq!(output.successor_count(), 2);
        assert_eq!(out_summary.successor_count, output.successor_count() as u32);
        assert_eq!(out_summary.state_len as usize, candidates.place_count());
        assert_eq!(
            output.iter_successors().collect::<Vec<_>>(),
            vec![&[1, 2, 0][..], &[2, 0, 2][..]]
        );
    }

    #[test]
    fn petri_artifact_adoption_evidence_uses_shared_jit_abi_contract() {
        use tla_jit_abi::{
            KernelArtifactKind, KERNEL_ARTIFACT_CONTRACT_SCHEMA,
            KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION, KERNEL_SYMBOL_ABI_EXTERN_C,
            PREDICATE_KERNEL_ARTIFACT_KIND, SUCCESSOR_KERNEL_ARTIFACT_KIND,
        };

        let successor = petri_successor_kernel_artifact_adoption_evidence();
        assert_eq!(successor.schema, KERNEL_ARTIFACT_CONTRACT_SCHEMA);
        assert_eq!(
            successor.schema_version,
            KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION
        );
        assert_eq!(successor.consumer, TY_KERNEL_ARTIFACT_CONSUMER);
        assert_eq!(successor.kind, KernelArtifactKind::SuccessorKernel);
        assert_eq!(successor.kind.as_str(), SUCCESSOR_KERNEL_ARTIFACT_KIND);
        assert_eq!(successor.entry_symbol, PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL);
        assert_eq!(
            successor.signature,
            KernelSymbolSignature::native_successor_kernel()
        );
        assert_eq!(successor.signature.abi, KERNEL_SYMBOL_ABI_EXTERN_C);
        assert_eq!(
            successor.required_manifest_metadata,
            vec![TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA.to_owned()]
        );
        assert!(kernel_artifact_checksums_are_deferred(&successor.checksums));

        let predicate = petri_predicate_kernel_artifact_adoption_evidence();
        assert_eq!(predicate.schema, KERNEL_ARTIFACT_CONTRACT_SCHEMA);
        assert_eq!(
            predicate.schema_version,
            KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION
        );
        assert_eq!(predicate.consumer, TY_KERNEL_ARTIFACT_CONSUMER);
        assert_eq!(predicate.kind, KernelArtifactKind::PredicateKernel);
        assert_eq!(predicate.kind.as_str(), PREDICATE_KERNEL_ARTIFACT_KIND);
        assert_eq!(predicate.entry_symbol, PETRI_NATIVE_PREDICATE_ENTRY_SYMBOL);
        assert_eq!(
            predicate.signature,
            KernelSymbolSignature::native_state_predicate_kernel()
        );
        assert_eq!(
            predicate.required_manifest_metadata,
            vec![TY_PREDICATE_KERNEL_EVIDENCE_METADATA.to_owned()]
        );
        assert!(kernel_artifact_checksums_are_deferred(&predicate.checksums));
    }

    // The receipt/route-selection rows asserted below only become available
    // when the `trust-cg-petri-native` feature is compiled in (the no-feature
    // build emits `missing_trust_ir_transport_identity` instead). Keep the test
    // gated to that feature so the assertion set stays sound.
    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_capability_report_is_validation_disabled_by_default() {
        let net = all_transition_net();

        let report = petri_native_successor_capability_report(&net);

        assert_eq!(report.problem, Some(ProblemKind::NativeSuccessor));
        assert!(report.selected.is_empty());
        assert_eq!(report.rejected.len(), 2);
        let capability = report
            .rejected
            .iter()
            .find(|capability| capability.problem == Some(ProblemKind::NativeSuccessor))
            .expect("native successor capability should be rejected");
        assert_eq!(capability.domain, BackendDomain::PetriMcc);
        assert_eq!(capability.backend, BackendKind::NativeKernel);
        assert_eq!(capability.problem, Some(ProblemKind::NativeSuccessor));
        assert_eq!(capability.role, CapabilityRole::Validation);
        assert_eq!(capability.status, tla_mc_core::CapabilityStatus::Disabled);
        assert_eq!(
            capability.reason,
            Some(UnsupportedReason::DisabledByPolicy(
                "trust-cg-petri-native is not parity promoted"
            ))
        );
        assert_eq!(capability.reason_code(), Some("disabled_by_policy"));
        assert_eq!(
            capability.reason.as_ref().map(UnsupportedReason::code),
            Some("disabled_by_policy")
        );
        assert_eq!(
            report.rejection_reason_code(BackendKind::NativeKernel),
            Some("disabled_by_policy")
        );
        assert!(capability.facets.contains(&SolverFacet::NativeCodegen));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "successor kernel descriptor name=petri-all-transitions state_len=3 max_successors=3"
        )));
        let shared_candidate_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.starts_with("MCC prepared_native_candidate_shared_vocab "))
            .expect("native report should expose the shared prepared candidate vocabulary row");
        assert!(shared_candidate_row
            .contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(shared_candidate_row.contains("origin_frontend=mcc_petri"));
        assert!(shared_candidate_row.contains("shared_owner=shared_high_performance_engine"));
        assert!(shared_candidate_row.contains("first_beneficiary=mcc_petri_runtime_storage"));
        assert!(shared_candidate_row.contains(
            "second_beneficiary=trust_cg_batch_identity_contract,ay_analytical,witness_replay"
        ));
        assert!(shared_candidate_row
            .contains("extraction_status=frontend-local-with-tracked-extraction"));
        assert!(shared_candidate_row.contains("blocker_status=tracked-blockers"));
        assert!(shared_candidate_row.contains("adoption_matrix_fields=origin_frontend"));
        assert!(shared_candidate_row.contains("generic_prerequisites"));
        assert!(shared_candidate_row.contains("shared_engine_prerequisite"));
        assert!(shared_candidate_row.contains("native_adoption_blocker"));
        assert!(shared_candidate_row.contains("exact_or_unknown"));
        assert!(shared_candidate_row.contains("validation_receipt_status"));
        assert!(shared_candidate_row.contains("parity_receipt_status"));
        assert!(shared_candidate_row.contains("production_gate_status"));
        assert!(shared_candidate_row.contains("candidate_key=trust_cg_native"));
        assert!(shared_candidate_row.contains("lane_identity=shared_native_successor"));
        assert!(shared_candidate_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(shared_candidate_row
            .contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(shared_candidate_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(shared_candidate_row.contains("cache_digest=fnv1a64:"));
        assert!(
            shared_candidate_row.contains("transition_descriptor=shared_petri_transition_relation")
        );
        assert!(shared_candidate_row.contains("predicate_descriptor=shared_petri_state_predicate"));
        assert!(shared_candidate_row
            .contains("kernel_metadata_schema=tla_ir.whole_program_kernel_metadata.v1"));
        assert!(shared_candidate_row.contains("kernel_metadata_source=local_tla_ir_compatible"));
        assert!(shared_candidate_row.contains(
            "kernel_metadata_blocker=tla-ir_metadata_crate_not_a_default_tla-petri_dependency"
        ));
        assert!(shared_candidate_row.contains("kernel_layout_kind=petri_marking_i64_vector"));
        assert!(shared_candidate_row
            .contains("frontend_neutral_kernel_layout_fingerprint_algorithm=fnv1a64"));
        assert!(shared_candidate_row.contains("frontend_neutral_kernel_layout_fingerprint="));
        assert!(shared_candidate_row.contains("manifest_checksum=fnv1a64:"));
        assert!(shared_candidate_row.contains("layout_checksum=fnv1a64:"));
        assert!(shared_candidate_row.contains("semantic_checksum=fnv1a64:"));
        assert!(shared_candidate_row.contains("source_checksum=fnv1a64:"));
        assert!(shared_candidate_row.contains("payload_checksum=fnv1a64:"));
        assert!(shared_candidate_row.contains("validation_status=validation_unknown"));
        assert!(shared_candidate_row.contains("exact_or_unknown=unknown"));
        assert!(shared_candidate_row.contains("fail_closed=true"));
        let shared_native_contract_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.starts_with("MCC shared_native_contract "))
            .expect("native report should expose the shared native contract row");
        assert!(shared_native_contract_row.contains("schema=ty.shared.native_contract.v1"));
        assert!(shared_native_contract_row.contains("source_kind=mcc_petri"));
        assert!(shared_native_contract_row.contains("frontend_kind=mcc_petri"));
        assert!(shared_native_contract_row.contains("payload_kind=mcc_petri"));
        assert!(shared_native_contract_row.contains("storage_kind=petri_marking"));
        assert!(shared_native_contract_row.contains("contract_kind=successor_kernel"));
        assert!(shared_native_contract_row.contains("lane_kind=native"));
        assert!(
            shared_native_contract_row.contains("symbol=petri_marking_successor_predicate_batch")
        );
        assert!(shared_native_contract_row.contains("layout_kind=petri_marking"));
        assert!(shared_native_contract_row.contains("layout_identity=petri_marking_i64_vector"));
        assert!(shared_native_contract_row.contains("candidate_identity=trust_cg_native"));
        assert!(shared_native_contract_row.contains("lane_identity=shared_native_successor"));
        assert!(shared_native_contract_row.contains("semantic_digest=fnv1a64:"));
        assert!(shared_native_contract_row.contains("cache_digest=fnv1a64:"));
        assert!(shared_native_contract_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(shared_native_contract_row
            .contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(shared_native_contract_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(shared_native_contract_row.contains("storage_layout_fingerprint=fnv1a64:"));
        assert!(shared_native_contract_row.contains(
            "required_evidence=manifest_metadata,layout_checksum,semantic_checksum,validation_receipt"
        ));
        assert!(shared_native_contract_row.contains("install_authority=validation_only"));
        assert!(shared_native_contract_row.contains("admission_status=accepted"));
        assert!(shared_native_contract_row.contains("admission_disposition=profile_only"));
        assert!(shared_native_contract_row.contains("admission_authority=validation_only"));
        assert!(shared_native_contract_row.contains("production_selected=false"));
        let core_planning_identity_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.starts_with("MCC shared_native_planning_identity "))
            .expect("native report should expose the core shared native planning identity row");
        assert!(core_planning_identity_row.contains("schema=ty.shared.native_planning_identity.v1"));
        assert!(core_planning_identity_row.contains("source_kind=mcc_petri"));
        assert!(core_planning_identity_row.contains("frontend_kind=mcc_petri"));
        assert!(core_planning_identity_row.contains("source_fingerprint=fnv1a64_"));
        assert!(core_planning_identity_row.contains("plan_reuse_manifest_id=trust_cg_prepared_trust_ir_reuse_shared_engine_frontend_neutral_batch_petri_marking_successor_predicate_semantic_v1_fnv1a64_"));
        assert!(core_planning_identity_row.contains("plan_reuse_manifest_digest=fnv1a64_"));
        assert!(core_planning_identity_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key_canonical_bytes_sha256"));
        assert!(core_planning_identity_row
            .contains("cas_identity=accepted_fail_closed_fingerprint_domain"));
        assert!(core_planning_identity_row
            .contains("cache_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(core_planning_identity_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(core_planning_identity_row.contains("frontend_family_reusable=true"));
        let shared_native_contract_manifest_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.starts_with("MCC shared_native_contract_manifest "))
            .expect("native report should expose manifest/layout/semantic checksum evidence");
        assert!(shared_native_contract_manifest_row
            .contains("schema=ty.shared_engine.petri.native_contract_manifest.v1"));
        assert!(shared_native_contract_manifest_row.contains("source_kind=mcc_petri"));
        assert!(shared_native_contract_manifest_row.contains("payload_kind=mcc_petri"));
        assert!(shared_native_contract_manifest_row.contains("storage_kind=petri_marking"));
        assert!(shared_native_contract_manifest_row.contains("layout_kind=petri_marking"));
        assert!(shared_native_contract_manifest_row
            .contains("layout_identity=petri_marking_i64_vector"));
        assert!(shared_native_contract_manifest_row
            .contains("symbol=petri_marking_successor_predicate_batch"));
        assert!(shared_native_contract_manifest_row.contains("manifest_checksum=fnv1a64:"));
        assert!(shared_native_contract_manifest_row.contains("layout_checksum=fnv1a64:"));
        assert!(shared_native_contract_manifest_row.contains("semantic_checksum=fnv1a64:"));
        assert!(shared_native_contract_manifest_row.contains("source_checksum=fnv1a64:"));
        assert!(shared_native_contract_manifest_row.contains("payload_checksum=fnv1a64:"));
        assert!(shared_native_contract_manifest_row
            .contains("fingerprint_algorithm=canonical_bytes_sha256"));
        assert!(shared_native_contract_manifest_row.contains(
            "fingerprint_helper_symbol=crate::explorer::fingerprint::fingerprint_marking"
        ));
        assert!(shared_native_contract_manifest_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(shared_native_contract_manifest_row.contains(
            "fingerprint_domain_acceptance_identity=accepted_fail_closed_fingerprint_domain"
        ));
        assert!(shared_native_contract_manifest_row
            .contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(
            shared_native_contract_manifest_row.contains("cache_reuse_policy=frontend_local_only")
        );
        assert!(shared_native_contract_manifest_row.contains("manifest_metadata_status=present"));
        assert!(shared_native_contract_manifest_row.contains("layout_checksum_status=present"));
        assert!(shared_native_contract_manifest_row.contains("semantic_checksum_status=present"));
        assert!(shared_native_contract_manifest_row.contains("validation_receipt_status=missing"));
        assert!(shared_native_contract_manifest_row.contains("install_authority=validation_only"));
        assert!(shared_native_contract_manifest_row.contains("admission_disposition=profile_only"));
        assert!(shared_native_contract_manifest_row
            .contains("artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1"));
        assert!(shared_native_contract_manifest_row
            .contains("artifact_identity_kind=contract_template"));
        assert!(shared_native_contract_manifest_row
            .contains("artifact_identity_status=contract_template_only"));
        assert!(shared_native_contract_manifest_row
            .contains("artifact_digest_status=per_artifact_digest_missing"));
        assert!(shared_native_contract_manifest_row.contains("validation_only=true"));
        assert!(shared_native_contract_manifest_row
            .contains("shard_compatibility_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(shared_native_contract_manifest_row
            .contains("shard_compatibility_scope=marking_vector_batch_partitionable"));
        assert!(shared_native_contract_manifest_row
            .contains("shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(shared_native_contract_manifest_row
            .contains("shard_identity_provider=future_trust_ir_trust_cg_manifest"));
        assert!(shared_native_contract_manifest_row.contains("shard_identity_key=fnv1a64:"));
        assert!(shared_native_contract_manifest_row
            .contains("fingerprint_compatibility_status=validation_only_declared"));
        assert!(shared_native_contract_manifest_row
            .contains("fingerprint_compatibility=canonical_bytes_sha256"));
        assert!(shared_native_contract_manifest_row
            .contains("cache_compatibility_status=validation_only_frontend_local"));
        assert!(shared_native_contract_manifest_row
            .contains("cache_fingerprint_compatibility=frontend_local_only"));
        assert!(shared_native_contract_manifest_row.contains("parity_receipt_required=true"));
        assert!(shared_native_contract_manifest_row
            .contains("parity_receipt_schema=ty.petri.native_successor.parity_receipt.v1"));
        assert!(shared_native_contract_manifest_row.contains("parity_receipt_status=missing"));
        assert!(shared_native_contract_manifest_row.contains(
            "production_gate=native_install_validation_parity_and_callable_receipts_required"
        ));
        assert!(shared_native_contract_manifest_row.contains("production_selected=false"));
        assert!(shared_native_contract_manifest_row.contains("fail_closed=true"));
        let shared_engine_readiness_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.starts_with("MCC petri_native_shared_engine_readiness "))
            .expect("native report should expose Petri shared native/JIT readiness evidence");
        assert!(shared_engine_readiness_row
            .contains("schema=ty.shared_engine.petri.native_engine_readiness.v1"));
        assert!(shared_engine_readiness_row
            .contains("readiness_identity=mcc_petri_shared_native_readiness:"));
        assert!(shared_engine_readiness_row.contains("readiness_mode=validation_only"));
        assert!(shared_engine_readiness_row.contains("prepared_trust_ir_reuse_identity=trust_cg_prepared_trust_ir_reuse:shared_engine_frontend_neutral_batch:petri_marking_successor_predicate_semantic_v1:fnv1a64:"));
        assert!(shared_engine_readiness_row
            .contains("prepared_trust_ir_reuse_identity_status=deferred_until_trust_ir_manifest"));
        assert!(shared_engine_readiness_row.contains("origin_frontend=mcc_petri"));
        assert!(shared_engine_readiness_row.contains("diagnostic_module_family=mcc_petri"));
        assert!(shared_engine_readiness_row
            .contains("shared_engine_component=batch_native_artifact_identity"));
        assert!(
            shared_engine_readiness_row.contains("digest_source=petri_native_contract_manifest")
        );
        assert!(shared_engine_readiness_row.contains("prepared_semantic_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row.contains("artifact_link_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row.contains("artifact_cache_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row.contains(
            "batch_artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1"
        ));
        assert!(
            shared_engine_readiness_row.contains("batch_artifact_identity_kind=contract_template")
        );
        assert!(shared_engine_readiness_row
            .contains("batch_artifact_digest_status=per_artifact_digest_missing"));
        assert!(shared_engine_readiness_row
            .contains("export_set_identity_basis=petri_successor_predicate_symbol_set_v1"));
        assert!(shared_engine_readiness_row.contains("export_set_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row.contains(
            "alias_resolution_identity_basis=petri_marking_successor_predicate_alias_resolution_v1"
        ));
        assert!(shared_engine_readiness_row.contains("alias_resolution_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row.contains(
            "export_surface_identity_basis=petri_marking_successor_predicate_export_surface_v1"
        ));
        assert!(shared_engine_readiness_row.contains("export_surface_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row.contains(
            "native_requirements_identity_basis=petri_marking_successor_predicate_native_requirements_v1"
        ));
        assert!(shared_engine_readiness_row.contains("native_requirements_digest=fnv1a64:"));
        assert!(
            shared_engine_readiness_row.contains("readiness_owner=shared_high_performance_engine")
        );
        assert!(
            shared_engine_readiness_row.contains("primary_beneficiary=mcc_petri_runtime_storage")
        );
        assert!(shared_engine_readiness_row.contains(
            "secondary_beneficiary=trust_cg_batch_identity_contract,ay_analytical,witness_replay"
        ));
        assert!(shared_engine_readiness_row.contains("readiness_frontend_families=mcc_petri"));
        assert!(shared_engine_readiness_row.contains(
            "future_frontend_family_readiness=deferred_until_core_shared_adoption_schema"
        ));
        assert!(shared_engine_readiness_row
            .contains("checksum_scope=layout_semantic_source_payload_cache"));
        assert!(shared_engine_readiness_row.contains("frontend_fields_in_checksums=net_name,place_ids,place_names,transition_ids,transition_names,initial_marking,arcs"));
        assert!(shared_engine_readiness_row
            .contains("prepared_trust_ir_reuse=deferred_until_trust_ir_manifest"));
        assert!(shared_engine_readiness_row
            .contains("prepared_trust_ir_reuse_scope=shared_engine_frontend_neutral_batch"));
        assert!(shared_engine_readiness_row.contains("source_kind=mcc_petri"));
        assert!(shared_engine_readiness_row.contains("payload_kind=mcc_petri"));
        assert!(shared_engine_readiness_row.contains("storage_kind=petri_marking"));
        assert!(shared_engine_readiness_row.contains("layout_kind=petri_marking"));
        assert!(shared_engine_readiness_row.contains("layout_identity=petri_marking_i64_vector"));
        assert!(
            shared_engine_readiness_row.contains("symbol=petri_marking_successor_predicate_batch")
        );
        assert!(shared_engine_readiness_row.contains("validation_only=true"));
        assert!(shared_engine_readiness_row.contains("readiness_status=validation_only"));
        assert!(shared_engine_readiness_row.contains("install_authority=validation_only"));
        assert!(shared_engine_readiness_row.contains("admission_disposition=profile_only"));
        assert!(shared_engine_readiness_row.contains("validation_receipt_status=missing"));
        assert!(shared_engine_readiness_row
            .contains("shard_compatibility_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(shared_engine_readiness_row
            .contains("shard_compatibility_scope=marking_vector_batch_partitionable"));
        assert!(shared_engine_readiness_row
            .contains("shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(shared_engine_readiness_row
            .contains("shard_identity_provider=future_trust_ir_trust_cg_manifest"));
        assert!(shared_engine_readiness_row.contains("shard_identity_key=fnv1a64:"));
        assert!(shared_engine_readiness_row
            .contains("trust_ir_shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(shared_engine_readiness_row
            .contains("trust_cg_shard_identity_status=deferred_until_trust_ir_trust_cg_manifest"));
        assert!(shared_engine_readiness_row
            .contains("fingerprint_compatibility_status=validation_only_declared"));
        assert!(shared_engine_readiness_row
            .contains("fingerprint_compatibility=canonical_bytes_sha256"));
        assert!(shared_engine_readiness_row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(shared_engine_readiness_row.contains(
            "fingerprint_domain_acceptance_identity=accepted_fail_closed_fingerprint_domain"
        ));
        assert!(shared_engine_readiness_row
            .contains("cache_compatibility_status=validation_only_frontend_local"));
        assert!(shared_engine_readiness_row
            .contains("cache_fingerprint_compatibility=frontend_local_only"));
        assert!(shared_engine_readiness_row
            .contains("cache_namespace_identity=mcc_petri.shared_native.validation_cache.v1"));
        assert!(shared_engine_readiness_row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(shared_engine_readiness_row.contains("cache_digest=fnv1a64:"));
        assert!(shared_engine_readiness_row
            .contains("artifact_identity=mcc_petri.shared_native_contract.trust_cg_native.v1"));
        assert!(shared_engine_readiness_row.contains("artifact_identity_kind=contract_template"));
        assert!(
            shared_engine_readiness_row.contains("artifact_identity_status=contract_template_only")
        );
        assert!(shared_engine_readiness_row
            .contains("artifact_digest_status=per_artifact_digest_missing"));
        assert!(shared_engine_readiness_row.contains("parity_receipt_required=true"));
        assert!(shared_engine_readiness_row
            .contains("parity_receipt_schema=ty.petri.native_successor.parity_receipt.v1"));
        assert!(shared_engine_readiness_row.contains("parity_receipt_status=missing"));
        assert!(shared_engine_readiness_row.contains(
            "production_gate=native_install_validation_parity_and_callable_receipts_required"
        ));
        assert!(shared_engine_readiness_row.contains("production_selected=false"));
        assert!(shared_engine_readiness_row.contains("fail_closed=true"));
        let route_selection_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit route_selection"))
            .expect("native report should expose fail-closed route selection evidence");
        assert!(route_selection_row.contains("selected_lane=explicit_state"));
        assert!(route_selection_row.contains("status_code=fail_closed"));
        assert!(route_selection_row.contains("reason_code=producer_admission_not_installable"));
        assert!(route_selection_row.contains("parity_required=true"));
        assert!(route_selection_row.contains("parity_enabled=false"));
        assert!(route_selection_row.contains("parity_receipt_required=true"));
        assert!(route_selection_row.contains("parity_receipt_available=true"));
        assert!(route_selection_row.contains("parity_receipt_reason_code=available"));
        assert!(route_selection_row
            .contains("parity_receipt_schema=ty.petri.native_successor.parity_receipt.v1"));
        assert!(route_selection_row.contains(
            "parity_receipt_gate_api=tla_petri::petri_native_successor_parity_receipt_gate"
        ));
        assert!(route_selection_row.contains(
            "parity_receipt_required_evidence=exact_successor_parity_trace,native_vs_explicit_state_replay_receipt"
        ));
        assert!(route_selection_row.contains("producer_admission=false"));
        assert!(route_selection_row
            .contains("producer_admission_reason_code=missing_semantic_successor_obligation"));
        assert!(route_selection_row.contains("producer_execution_authority=false"));
        assert!(route_selection_row.contains(
            "producer_execution_authority_reason_code=missing_native_install_gate_packet"
        ));
        assert!(route_selection_row.contains("producer_production_selection=false"));
        assert!(route_selection_row.contains(
            "producer_production_selection_reason_code=missing_semantic_successor_obligation"
        ));
        assert!(route_selection_row.contains("validation_receipt_available=true"));
        assert!(route_selection_row.contains("validation_receipt_reason_code=available"));
        assert!(route_selection_row.contains("callable_receipt_available=false"));
        assert!(route_selection_row
            .contains("callable_receipt_reason_code=missing_native_install_gate_packet"));
        assert!(route_selection_row.contains("native_runtime_callable_impl_available=true"));
        assert!(route_selection_row
            .contains("runtime_readiness_reason_code=missing_native_install_gate_packet"));
        assert!(route_selection_row.contains("production_selected=false"));
        assert!(route_selection_row.contains("fail_closed=true"));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "Petri native_successor JIT ABI artifact contract expected schema=trust_cg.kernel_artifact_contract/v1 schema_version=1 kind=successor_kernel"
        )));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "consumer=ty entry_symbol=ty_petri_all_transition_successors signature_abi=extern_c params=9 returns=1 required_manifest_metadata=ty.successor_kernel.evidence adopted=false adoption=deferred artifact_checksums=deferred"
        )));
        let expected_fail_closed_gate_reason = if cfg!(feature = "trust-cg-petri-native") {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PRODUCER_ADMISSION
        } else {
            PETRI_NATIVE_ROUTE_SELECTION_REASON_MISSING_TRANSPORT
        };
        let expected_fail_closed_gate_reason =
            format!("reason_code={expected_fail_closed_gate_reason}");
        assert!(report.evidence.iter().any(|evidence| {
            evidence.contains("Petri native_jit fail_closed_gate")
                && evidence.contains("feature=trust-cg-petri-native")
                && evidence.contains("native_env=TY_MCC_TRUST_CG_PETRI_NATIVE")
                && evidence.contains("strict_env=TY_MCC_TRUST_CG_PETRI_NATIVE_STRICT")
                && evidence.contains("parity_env=TY_MCC_TRUST_CG_PETRI_PARITY")
                && evidence.contains("parity_receipt_required=true")
                && evidence.contains("parity_receipt_available=true")
                && evidence.contains("parity_receipt_status_code=accepted")
                && evidence.contains("parity_receipt_reason_code=available")
                && evidence.contains("validation_receipt_required=true")
                && evidence.contains("validation_receipt_available=true")
                && evidence.contains("validation_receipt_status_code=accepted")
                && evidence.contains("validation_receipt_reason_code=available")
                && evidence.contains("callable_receipt_available=false")
                && evidence
                    .contains("callable_receipt_reason_code=missing_native_install_gate_packet")
                && evidence.contains("production_selected=false")
                && evidence.contains("fail_closed=true")
                && evidence.contains(expected_fail_closed_gate_reason.as_str())
        }));
        let transport_identity = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit trust_ir_transport_identity"))
            .expect("native JIT transport identity evidence should be emitted");
        if cfg!(feature = "trust-cg-petri-native") {
            assert!(
                transport_identity.contains("available")
                    && transport_identity.contains("cargo_dependency=true")
                    && transport_identity
                        .contains("api=NativeVerificationBundle::transport_identity")
                    && transport_identity.contains("bundle_source=petri_native_production_path")
                    && transport_identity.contains("bundle_validated=true")
                    && transport_identity.contains("request_digests=1")
                    && transport_identity
                        .contains("expected_fields=transport,source,module,bundle,target_abi_digest")
                    && transport_identity.contains("production_selected=false")
                    && transport_identity.contains("fail_closed=true"),
                "trust-ir transport identity should use the produced bundle when linked: {transport_identity}"
            );
        } else {
            assert!(
                transport_identity.contains("unavailable")
                    && transport_identity
                        .contains("required_trust_ir_rev=222785e293636ac6c63b20525151aef2ccd586c1")
                    && transport_identity.contains(&format!(
                        "current_trust_ir_rev={TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV}"
                    ))
                    && transport_identity.contains("cargo_dependency=false")
                    && transport_identity
                        .contains("api=NativeVerificationBundle::transport_identity")
                    && transport_identity.contains(TRUST_IR_NATIVE_VERIFICATION_BUNDLE_DEPENDENCY_BLOCKER)
                    && transport_identity
                        .contains("expected_fields=transport,source,module,bundle,target_abi_digest")
                    && transport_identity.contains("production_selected=false")
                    && transport_identity.contains("fail_closed=true"),
                "trust-ir transport identity blocker should be explicit in evidence: {transport_identity}"
            );
        }
        let producer_contract = report
            .evidence
            .iter()
            .find(|evidence| {
                evidence.contains("Petri native_jit trust_ir_transport_identity_producer_contract")
            })
            .expect("native JIT transport identity producer contract evidence should be emitted");
        assert!(
            producer_contract.contains(&format!(
                "schema={}",
                TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SCHEMA
            )) && producer_contract.contains(&format!(
                "schema_version={}",
                TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_SCHEMA_VERSION
            )) && producer_contract.contains(&format!(
                "producer_api={}",
                TRUST_IR_NATIVE_TRANSPORT_IDENTITY_PRODUCER_CONTRACT_API
            )) && producer_contract.contains(&format!("consumer_api={TRUST_CG_PETRI_NATIVE_ADMISSION_API}"))
                && producer_contract.contains(&format!(
                    "required_output={}",
                    TRUST_IR_NATIVE_TRANSPORT_IDENTITY_REQUIRED_OUTPUT
                ))
                && producer_contract.contains(&format!(
                    "transport_identity_schema={}",
                    TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA
                ))
                && producer_contract.contains("requested_authority=validation_only")
                && producer_contract.contains("native_promotion_authorized=false")
                && producer_contract.contains("production_selected=false")
                && producer_contract.contains("fail_closed=true"),
            "producer contract should pin the trust-ir/native transport identity handoff without promotion: {producer_contract}"
        );
        if cfg!(feature = "trust-cg-petri-native") {
            assert!(
                producer_contract.contains("cargo_dependency=true")
                    && producer_contract.contains("status_code=available")
                    && producer_contract.contains("reason_code=available")
                    && producer_contract.contains("bundle_source=petri_native_production_path")
                    && producer_contract.contains("bundle_validated=true")
                    && producer_contract.contains("producer=trust_ir")
                    && producer_contract.contains("input=trust_ir_module")
                    && producer_contract.contains("transport_identity_available=true")
                    && producer_contract.contains("module_digest=sha256:")
                    && producer_contract.contains("transport_digest=sha256:")
                    && producer_contract.contains("blocker=\"none\""),
                "linked producer contract should identify the Petri-produced trust-ir bundle: {producer_contract}"
            );
        } else {
            assert!(
                producer_contract.contains("cargo_dependency=false")
                    && producer_contract.contains("status_code=blocked")
                    && producer_contract.contains(
                        "reason_code=missing_trust_ir_transport_identity"
                    )
                    && producer_contract.contains("bundle_source=none")
                    && producer_contract.contains("bundle_validated=false")
                    && producer_contract.contains("producer=none")
                    && producer_contract.contains("input=none")
                    && producer_contract.contains("transport_identity_available=false")
                    && producer_contract.contains("module_digest=none")
                    && producer_contract.contains("transport_digest=none")
                    && producer_contract
                        .contains(TRUST_IR_NATIVE_VERIFICATION_BUNDLE_DEPENDENCY_BLOCKER),
                "unlinked producer contract should pin the missing transport identity producer: {producer_contract}"
            );
        }
        let admission_blocker = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg trust_cg_admission_blocker"))
            .expect("native JIT admission blocker evidence should be emitted");
        let expected_admission_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            TRUST_CG_PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON
        };
        let expected_execution_plan_rejection_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            expected_admission_reason
        };
        let expected_execution_plan_available = cfg!(feature = "trust-cg-petri-native");
        let expected_execution_plan_reason = if expected_execution_plan_available {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            expected_admission_reason
        };
        let expected_call_packet_api_available = cfg!(feature = "trust-cg-petri-native");
        let expected_call_packet_api_status = if expected_call_packet_api_available {
            TrustCgPetriNativeReadinessStatus::Available
        } else {
            TrustCgPetriNativeReadinessStatus::Unavailable
        };
        let expected_execution_plan_status = if expected_execution_plan_available {
            TrustCgPetriNativeReadinessStatus::Available
        } else {
            TrustCgPetriNativeReadinessStatus::Missing
        };
        let expected_runtime_readiness_packet_available = cfg!(feature = "trust-cg-petri-native");
        let expected_runtime_readiness_status = if cfg!(feature = "trust-cg-petri-native") {
            "blocked"
        } else {
            "unavailable"
        };
        let expected_runtime_readiness_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            expected_admission_reason
        };
        let expected_runtime_readiness_blocker_stage = if cfg!(feature = "trust-cg-petri-native") {
            "manifest_identity"
        } else {
            "trust_ir_transport_identity"
        };
        let expected_call_packet_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            expected_admission_reason
        };
        let expected_callable_pointer_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON
        };
        let expected_callable_authorized_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            expected_admission_reason
        };
        let expected_callable_handoff_available = cfg!(feature = "trust-cg-petri-native");
        let expected_callable_handoff_reason = if expected_callable_handoff_available {
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        } else {
            TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON
        };
        let expected_callable_handoff_blocker = if expected_callable_handoff_available {
            "missing_native_install_gate_packet"
        } else {
            TRUST_CG_PETRI_NATIVE_MISSING_CALLABLE_POINTER_HANDOFF_REASON
        };
        let expected_install_packet_reason = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            expected_admission_reason
        };
        let expected_call_packet_readiness_blocker = if cfg!(feature = "trust-cg-petri-native") {
            "missing_native_install_gate_packet"
        } else {
            expected_callable_handoff_blocker
        };
        let expected_trampoline_contract_available = false;
        let expected_install_packet_available = false;
        let expected_install_packet_status = if expected_install_packet_available {
            TrustCgPetriNativeReadinessStatus::Available
        } else {
            TrustCgPetriNativeReadinessStatus::Missing
        };
        let expected_call_packet_available = false;
        let expected_callable_pointer_available = false;
        let expected_concrete_callable_status = TrustCgPetriNativeReadinessStatus::Missing;
        let expected_callable_authorized = false;
        if cfg!(feature = "trust-cg-petri-native") {
            assert!(
                admission_blocker.contains("source=NativeInstallGateAdmissionSummary")
                    && admission_blocker.contains("source_package=trust-cg-codegen")
                    && admission_blocker.contains("package=trust-cg-codegen")
                    && admission_blocker
                        .contains("schema=trust-cg.phase6.native_install_gate.admission_summary.v1")
                    && admission_blocker.contains("schema_version=1")
                    && admission_blocker.contains("consumer=mcc")
                    && admission_blocker.contains("consumer_mode=petri_successor")
                    && admission_blocker.contains("kind=petri_native_successor")
                    && admission_blocker.contains("surface=mcc_replay")
                    && admission_blocker.contains("summary_consumer_mode=ty_petri_native_jit")
                    && admission_blocker.contains("summary_kind=petri_successor")
                    && admission_blocker.contains("summary_surface=native_successor")
                    && admission_blocker
                        .contains(&format!("rejection_code={expected_admission_reason}"))
                    && admission_blocker.contains(&format!("reason_code={expected_admission_reason}"))
                    && admission_blocker.contains("requested_authority=active_callable")
                    && admission_blocker.contains("summary_requested_authority=validation_only")
                    && admission_blocker.contains("install_authority=none")
                    && admission_blocker.contains(
                        "bundle_api=NativeVerificationBundle::native_evidence_consumption_report"
                    )
                    && admission_blocker.contains(TRUST_CG_PETRI_NATIVE_ADMISSION_API)
                    && admission_blocker.contains("bundle_source=petri_native_production_path")
                    && admission_blocker.contains("bundle_validated=true")
                    && admission_blocker.contains("trust_ir_transport_identity_available=true")
                    && admission_blocker.contains("trust_ir_bundle_consumed=true")
                    && admission_blocker.contains("trust_ir_consumption_status=available")
                    && admission_blocker.contains("trust_ir_consumption_entries=1")
                    && admission_blocker.contains("artifact_count=4")
                    && admission_blocker
                        .contains("native_evidence_backend_metadata_artifacts=0")
                    && admission_blocker.contains("native_evidence_semantic_proof_artifacts=3")
                    && admission_blocker.contains("native_evidence_native_execution_artifacts=1")
                    && admission_blocker.contains("native_evidence_metadata_only=false")
                    && admission_blocker
                        .contains("native_evidence_semantic_proof_available=true")
                    && admission_blocker
                        .contains("native_evidence_native_execution_artifact_available=true")
                    && admission_blocker.contains("native_evidence_metadata_request_ids=none")
                    && admission_blocker
                        .contains("native_evidence_metadata_request_digests=none")
                    && admission_blocker
                        .contains("native_evidence_metadata_module_digest=sha256:")
                    && admission_blocker
                        .contains("native_evidence_metadata_artifact_digests=none")
                    && admission_blocker.contains("evidence_digests=1")
                    && admission_blocker.contains("production_selected=false")
                    && admission_blocker.contains("fail_closed=true"),
                "linked Petri native admission should use trust-cg's typed summary: {admission_blocker}"
            );
        } else {
            assert!(
                admission_blocker.contains("source=NativeInstallGateAdmissionSummary")
                    && admission_blocker.contains("source_package=trust-cg-codegen")
                    && admission_blocker
                        .contains("schema=trust-cg.phase6.native_install_gate.admission_summary.v1")
                    && admission_blocker.contains("schema_version=1")
                    && admission_blocker.contains("consumer=mcc")
                    && admission_blocker.contains("consumer_mode=petri_successor")
                    && admission_blocker.contains("kind=petri_native_successor")
                    && admission_blocker.contains("surface=mcc_replay")
                    && admission_blocker
                        .contains(&format!("rejection_code={expected_admission_reason}"))
                    && admission_blocker
                        .contains(&format!("reason_code={expected_admission_reason}"))
                    && admission_blocker.contains("requested_authority=active_callable")
                    && admission_blocker.contains("install_authority=none")
                    && admission_blocker.contains(TRUST_CG_PETRI_NATIVE_ADMISSION_API)
                    && admission_blocker.contains("production_selected=false")
                    && admission_blocker.contains("fail_closed=true"),
                "trust-cg admission blocker should be explicit and fail-closed: {admission_blocker}"
            );
        }
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        assert!(
            execution_plan.contains("consumer=mcc")
                && execution_plan.contains("kind=petri_successor")
                && execution_plan.contains("surface=native_successor")
                && execution_plan.contains(&format!(
                    "rejection_code={expected_execution_plan_rejection_reason}"
                ))
                && execution_plan.contains(&format!(
                    "reason_code={expected_execution_plan_rejection_reason}"
                ))
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_EXECUTION_PLAN_API)
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_EXECUTION_EXPECTED_API)
                && execution_plan.contains(&format!(
                    "entry_function={}",
                    PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL
                ))
                && execution_plan.contains("input_state_bytes=24")
                && execution_plan.contains("output_state_bytes=24")
                && execution_plan.contains("state_alignment_bytes=8")
                && execution_plan.contains(&format!(
                    "execution_plan_available={expected_execution_plan_available}"
                ))
                && execution_plan.contains(&format!(
                    "execution_plan_status_code={}",
                    expected_execution_plan_status.code()
                ))
                && execution_plan.contains(&format!(
                    "execution_plan_reason_code={expected_execution_plan_reason}"
                ))
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_TRAMPOLINE_CONTRACT_API)
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_INSTALL_PACKET_API)
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_API)
                && execution_plan.contains(&format!(
                    "call_packet_schema={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_SCHEMA
                ))
                && execution_plan.contains(&format!(
                    "call_packet_schema_version={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_SCHEMA_VERSION
                ))
                && execution_plan.contains(&format!(
                    "call_packet_type={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_TYPE
                ))
                && execution_plan.contains(&format!(
                    "callable_pointer_type={}",
                    TRUST_CG_PETRI_NATIVE_CALLABLE_POINTER_TYPE
                ))
                && execution_plan.contains(&format!(
                    "call_packet_required_trust_cg_rev={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_REQUIRED_TRUST_CG_REV
                ))
                && execution_plan.contains(&format!(
                    "call_packet_current_trust_cg_rev={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV
                ))
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_API)
                && execution_plan.contains(&format!(
                    "runtime_readiness_schema={}",
                    TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_SCHEMA
                ))
                && execution_plan.contains(&format!(
                    "runtime_readiness_schema_version={}",
                    TRUST_CG_PETRI_NATIVE_RUNTIME_READINESS_PACKET_SCHEMA_VERSION
                ))
                && execution_plan.contains(&format!(
                    "runtime_readiness_packet_available={expected_runtime_readiness_packet_available}"
                ))
                && execution_plan.contains(&format!(
                    "runtime_readiness_status_code={expected_runtime_readiness_status}"
                ))
                && execution_plan.contains(&format!(
                    "runtime_readiness_reason_code={expected_runtime_readiness_reason}"
                ))
                && execution_plan.contains(&format!(
                    "runtime_readiness_blocker_stage={expected_runtime_readiness_blocker_stage}"
                ))
                && execution_plan.contains(TRUST_CG_PETRI_NATIVE_MOCK_EXECUTABLE_CALL_API)
                && execution_plan.contains("mock_executable_call_production_enabled=false")
                && execution_plan.contains(&format!(
                    "call_packet_api_available={expected_call_packet_api_available}"
                ))
                && execution_plan.contains(&format!(
                    "call_packet_api_status_code={}",
                    expected_call_packet_api_status.code()
                ))
                && execution_plan.contains(&format!(
                    "call_packet_type_available={expected_call_packet_api_available}"
                ))
                && execution_plan.contains(&format!(
                    "callable_pointer_type_available={expected_call_packet_api_available}"
                ))
                && execution_plan.contains(&format!(
                    "trampoline_contract_available={expected_trampoline_contract_available}"
                ))
                && execution_plan.contains(&format!(
                    "install_packet_available={expected_install_packet_available}"
                ))
                && execution_plan.contains(&format!(
                    "install_packet_status_code={}",
                    expected_install_packet_status.code()
                ))
                && execution_plan.contains(&format!(
                    "install_packet_reason_code={expected_install_packet_reason}"
                ))
                && execution_plan.contains(&format!(
                    "call_packet_available={expected_call_packet_available} call_packet_reason_code={expected_call_packet_reason}"
                ))
                && execution_plan.contains(&format!(
                    "callable_pointer_available={expected_callable_pointer_available} callable_pointer_reason_code={expected_callable_pointer_reason}"
                ))
                && execution_plan.contains("concrete_callable_pointer_required=true")
                && execution_plan.contains(&format!(
                    "concrete_callable_pointer_available={expected_callable_pointer_available}"
                ))
                && execution_plan.contains(&format!(
                    "concrete_callable_pointer_status_code={}",
                    expected_concrete_callable_status.code()
                ))
                && execution_plan.contains("concrete_callable_packet_required=true")
                && execution_plan.contains(&format!(
                    "concrete_callable_packet_available={expected_call_packet_available}"
                ))
                && execution_plan.contains(&format!(
                    "concrete_callable_packet_status_code={}",
                    expected_concrete_callable_status.code()
                ))
                && execution_plan.contains(&format!(
                    "call_packet_readiness_status_code={expected_runtime_readiness_status}"
                ))
                && execution_plan.contains(&format!(
                    "call_packet_readiness_blocker={expected_call_packet_readiness_blocker}"
                ))
                && execution_plan
                    .contains(&format!("callable_authorized={expected_callable_authorized}"))
                && execution_plan.contains(&format!(
                    "callable_authorized_reason_code={expected_callable_authorized_reason}"
                ))
                && execution_plan.contains(&format!(
                    "callable_handoff_available={expected_callable_handoff_available}"
                ))
                && execution_plan.contains(&format!(
                    "callable_handoff_reason_code={expected_callable_handoff_reason}"
                ))
                && execution_plan.contains(&format!(
                    "callable_handoff_blocker={expected_callable_handoff_blocker}"
                ))
                && execution_plan.contains(&format!(
                    "callable_handoff_upstream_ask={}",
                    TRUST_CG_PETRI_NATIVE_CALLABLE_HANDOFF_UPSTREAM_ASK
                ))
                && execution_plan.contains("production_selected=false")
                && execution_plan.contains("fail_closed=true")
                && execution_plan.contains(&format!(
                    "native_successor_runtime_status_code={expected_runtime_readiness_status}"
                )),
            "trust-cg execution plan should be explicit and fail-closed: {execution_plan}"
        );
        #[cfg(feature = "trust-cg-petri-native")]
        {
            assert!(
                execution_plan.contains("source=PetriNativeSuccessorExecutionPlan")
                    && execution_plan
                        .contains("schema=trust-cg.petri.native_successor.execution_plan.v1")
                    && execution_plan.contains("trust_ir_transport_identity_available=true")
                    && execution_plan.contains("trust_ir_bundle_consumed=true")
                    && execution_plan.contains(TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_API)
                    && execution_plan.contains(&format!(
                        "downstream_contract_schema={}",
                        TRUST_CG_PETRI_NATIVE_DOWNSTREAM_CONTRACT_SCHEMA
                    ))
                    && execution_plan.contains(
                        "downstream_trust_ir_bundle_identity_schema=trust_ir.native.bundle_identity_contract.v1"
                    )
                    && execution_plan.contains(&format!(
                        "downstream_trust_ir_transport_identity_schema={}",
                        trust_ir::NATIVE_TRANSPORT_IDENTITY_SCHEMA
                    ))
                    && execution_plan.contains(
                        "downstream_compile_artifact_handoff_required_fields=compiled_artifact.native_payload_sha256,compiled_artifact.entry_symbol,compiled_artifact.callable_pointer,compiled_artifact.executable_region_sha256,compiled_artifact.lifetime_owner,compiled_artifact.current_generation"
                    )
                    && execution_plan.contains(
                        "downstream_runtime_readiness_required_fields=call_packet,native_install_gate_packet,trampoline_contract,callable_lifetime_proof,runtime_abi_proof,current_generation"
                    )
                    && execution_plan.contains(
                        "downstream_mock_executable_call_required_fields=runtime_readiness_packet,call_packet,mock_executable_call_gate,input_state,output_state"
                    )
                    && execution_plan.contains("runtime_readiness_status_in_downstream_contract=true")
                    && execution_plan.contains(
                        "runtime_readiness_blocker_code=missing_native_install_gate_packet"
                    )
                    && execution_plan.contains("runtime_readiness_blocker_in_downstream_contract=true")
                    && execution_plan.contains(
                        "callable_handoff_required_evidence=trust-cg.phase6.native_install_gate.v1"
                    )
                    && execution_plan
                        .contains("actions_expose_callable_blocked_by_runtime_readiness=true")
                    && execution_plan.contains(
                        "actions_expose_callable_reason_code=missing_native_install_gate_packet"
                    )
                    && execution_plan.contains(
                        "actions_ty_native_activate_blocked_by_runtime_readiness=true"
                    )
                    && execution_plan.contains(
                        "actions_ty_native_activate_reason_code=missing_native_install_gate_packet"
                    )
                    && execution_plan.contains("plan_fail_closed=true"),
                "linked Petri native execution plan should use trust-cg's typed plan: {execution_plan}"
            );
            assert_execution_plan_exposes_compile_artifact_handoff(execution_plan, true);
        }
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "Petri native_successor capability backend=NativeKernel problem=Some(NativeSuccessor) status=Disabled role=Validation reason_code=disabled_by_policy adoption=validation_only deferred=true"
        )));
        assert!(capability.detail.as_deref().is_some_and(|detail| {
            detail.contains("fail_closed=true")
                && detail.contains("feature=trust-cg-petri-native")
                && detail.contains("native_env=TY_MCC_TRUST_CG_PETRI_NATIVE")
                && detail.contains("production_selected=false")
                && detail.contains(&format!(
                    "trust_ir_transport_identity_available={}",
                    cfg!(feature = "trust-cg-petri-native")
                ))
                && detail.contains("trust_ir_required_rev=222785e293636ac6c63b20525151aef2ccd586c1")
                && detail.contains(&format!(
                    "trust_ir_current_rev={TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV}"
                ))
        }));
        let predicate = report
            .rejected
            .iter()
            .find(|capability| {
                capability.problem == Some(ProblemKind::Safety)
                    && capability.detail.as_deref() == Some(PETRI_NATIVE_PREDICATE_DETAIL)
            })
            .expect("native predicate capability should be rejected");
        assert_eq!(predicate.domain, BackendDomain::PetriMcc);
        assert_eq!(predicate.backend, BackendKind::NativeKernel);
        assert_eq!(predicate.role, CapabilityRole::Validation);
        assert_eq!(predicate.status, tla_mc_core::CapabilityStatus::Unsupported);
        assert_eq!(predicate.reason_code(), Some("native_kernel_unavailable"));
        assert!(predicate.facets.contains(&SolverFacet::NativeCodegen));
        assert!(predicate
            .facets
            .contains(&SolverFacet::LinearIntegerArithmetic));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "Petri native_predicate JIT ABI artifact contract expected schema=trust_cg.kernel_artifact_contract/v1 schema_version=1 kind=predicate_kernel"
        )));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "consumer=ty entry_symbol=ty_petri_state_predicate signature_abi=extern_c params=3 returns=1 required_manifest_metadata=ty.predicate_kernel.evidence adopted=false adoption=deferred artifact_checksums=deferred"
        )));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "Petri native_predicate capability backend=NativeKernel problem=Some(Safety) status=Unsupported role=Validation reason_code=native_kernel_unavailable adoption=deferred deferred=true"
        )));
        assert!(!report.has_unjustified_local_production());
    }

    #[test]
    fn native_shared_readiness_admission_contract_is_fail_closed_and_solver_generic() {
        let _lock = native_jit_env_lock();
        let _parity = EnvVarGuard::set(ENABLE_TRANSITION_PARITY_ENV, "0");
        let net = all_transition_net();

        let report = petri_native_successor_capability_report(&net);

        let row = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit shared_readiness_admission"))
            .expect("native report should expose shared readiness/admission evidence");
        assert!(row.contains("schema=ty.petri.native_successor.shared_readiness_admission.v1"));
        assert!(row.contains("api=tla_petri::petri_native_successor_shared_readiness_admission"));
        assert!(row.contains("shared_engine_owner=shared_high_performance_engine"));
        assert!(row.contains("origin_frontend=mcc_petri"));
        assert!(row.contains("solver_family_scope=explicit_state,native_successor,analytical_ay,witness_replay,hardware_transition_system,future_importer"));
        assert!(row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(row.contains("default_compatible_frontend_families=none"));
        assert!(row.contains("payload_identity_required=true"));
        assert!(row.contains("payload_identity_source=pnml_hlpnml_import_adapter"));
        assert!(row.contains("payload_identity_status=required_before_trusted_native"));
        assert!(row.contains("payload_identity_admission_status=blocked_for_trusted_native"));
        assert!(row.contains("layout_identity=petri_marking_i64_vector"));
        assert!(row.contains("layout_abi_version=1"));
        assert!(row.contains("layout_place_count=3"));
        assert!(row.contains("layout_transition_count=3"));
        assert!(row.contains("layout_plan_count=3"));
        assert!(row.contains("layout_matches_payload=true"));
        assert!(row.contains("layout_fingerprint_required=true"));
        assert!(row.contains("layout_fingerprint_admission_status=validation_only_declared"));
        assert!(row.contains("layout_fingerprint_exact_match_required=true"));
        assert!(row
            .contains("fingerprint_domain_identity=fingerprint_domain_key:canonical_bytes_sha256"));
        assert!(row.contains("fingerprint_admission_contract=prepared_fingerprint_admission"));
        assert!(row.contains("fingerprint_admission_status=blocked_for_trusted_native"));
        assert!(row.contains("parity_required=true"));
        assert!(row.contains("parity_enabled=false"));
        assert!(row.contains("parity_receipt_required=true"));
        assert!(row.contains("parity_receipt_status=missing"));
        assert!(row.contains("validation_receipt_required=true"));
        assert!(row.contains("validation_receipt_status=missing"));
        assert!(row.contains("callable_receipt_required=true"));
        assert!(row.contains("callable_receipt_status=missing"));
        assert!(row.contains("callable_receipt_reason_code=missing_callable_receipt"));
        assert!(row.contains("native_runtime_callable_impl_available=false"));
        assert!(row.contains("runtime_readiness_status_code=missing"));
        assert!(row.contains("runtime_readiness_reason_code=native_runtime_callable_impl_missing"));
        assert!(row.contains("exact_or_unknown=unknown"));
        assert!(row.contains("validation_status=validation_unknown"));
        assert!(row.contains(
            "exact_or_unknown_guard=native_output_unknown_until_explicit_replay_validation"
        ));
        assert!(row.contains("native_output_trusted=false"));
        assert!(row.contains("trusted_production_admitted=false"));
        assert!(row.contains("trusted_production_blockers=canonical_payload_identity,layout_fingerprint_admission,accepted_validation_receipt,accepted_parity_receipt,accepted_callable_receipt,native_runtime_callable_impl,history_suite_parity,end_to_end_speedup_evidence"));
        assert!(row.contains("performance_claim_status=not_claimed"));
        assert!(row.contains("production_selected=false"));
        assert!(row.contains("fail_closed=true"));
        assert!(
            report.selected.is_empty(),
            "readiness/admission evidence must not select native production"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_capability_report_records_produced_bundle_admission_blocker() {
        let net = all_transition_net();

        let report = petri_native_successor_capability_report(&net);

        let transport_identity = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit trust_ir_transport_identity"))
            .expect("native JIT transport identity evidence should be emitted");
        assert!(
            transport_identity.contains("available")
                && transport_identity.contains("cargo_dependency=true")
                && transport_identity.contains("bundle_source=petri_native_production_path")
                && transport_identity.contains("bundle_validated=true")
                && transport_identity.contains("request_digests=1")
                && transport_identity.contains("evidence_digests=1")
                && transport_identity.contains("production_selected=false")
                && transport_identity.contains("fail_closed=true"),
            "produced native bundle should expose typed trust-ir transport evidence: {transport_identity}"
        );

        let semantic_bridge = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit semantic_successor_bridge"))
            .expect("native JIT semantic successor bridge evidence should be emitted");
        assert!(
            semantic_bridge.contains("schema=trust-cg.petri.native_successor.semantic_bridge.v1")
                && semantic_bridge.contains("schema_version=1")
                && semantic_bridge.contains(
                    "api=trust-cg::petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle"
                )
                && semantic_bridge.contains(
                    "trust_ir_api=NativeVerificationBundle::native_semantic_bridge_report()"
                )
                && semantic_bridge.contains(
                    "trust_ir_petri_successor_report_api=NativeVerificationBundle::petri_successor_semantic_bridge_report()"
                )
                && semantic_bridge.contains(
                    "trust_ir_semantic_bridge_constructor_api=NativeSemanticBridge::petri_successor_plan_cache_equivalence()"
                )
                && semantic_bridge.contains(
                    "trust_ir_semantic_bridge_acceptance_api=NativeSemanticBridgeReport::represents_petri_successor_plan_cache_equivalence()"
                )
                && semantic_bridge.contains(
                    "formula_schema=ty.petri.native.successor.plan_cache_equivalence.v1"
                )
                && semantic_bridge.contains(
                    "downstream_semantic_bridge_surface=petri_native_successor_semantic_bridge"
                )
                && semantic_bridge.contains(
                    "downstream_semantic_bridge_required_fields=trust_ir_bundle,entry_function,semantic_successor_obligation,native_evidence_bundle"
                )
                && semantic_bridge.contains("downstream_semantic_bridge_status_codes=ready,blocked")
                && semantic_bridge.contains(
                    "downstream_semantic_bridge_blocker_codes=bundle_validation_failed,missing_entry_function,missing_semantic_successor_obligation,missing_semantic_successor_evidence"
                )
                && semantic_bridge.contains("bundle_source=petri_native_production_path")
                && semantic_bridge.contains("bundle_validated=true")
                && semantic_bridge.contains(&format!(
                    "entry_function={}",
                    PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL
                ))
                && semantic_bridge.contains("place_count=3")
                && semantic_bridge.contains("transition_count=3")
                && semantic_bridge.contains("plan_count=3")
                && semantic_bridge.contains("plan_cache_digest=")
                && semantic_bridge.contains("trust_ir_successor_body_status=lowered_all_transition_successors")
                && semantic_bridge.contains("successor_relation_represented=false")
                && semantic_bridge.contains("semantic_successor_authority=false")
                && semantic_bridge.contains("semantic_bridge_status_code=blocked")
                && semantic_bridge.contains("reason_code=missing_semantic_successor_obligation")
                && semantic_bridge
                    .contains("trust_cg_schema=trust-cg.petri.native_successor.semantic_bridge.v1")
                && semantic_bridge.contains("trust_cg_status_code=blocked")
                && semantic_bridge
                    .contains("trust_cg_reason_code=missing_semantic_successor_obligation")
                && semantic_bridge
                    .contains("trust_cg_required_field=semantic_successor_obligation")
                && semantic_bridge.contains(
                    "trust_cg_required_evidence=ty.petri.native.successor.plan_cache_equivalence.v1"
                )
                && semantic_bridge.contains("trust_cg_semantic_obligation_count=1")
                && semantic_bridge.contains("trust_cg_semantic_evidence_entry_count=1")
                && semantic_bridge.contains("trust_ir_semantic_bridge_schema=trust_ir.native.semantic_bridge.v2")
                && semantic_bridge.contains("trust_ir_semantic_bridge_status_code=blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_reason_code=trusted_proof_not_admitted")
                && semantic_bridge.contains("trust_ir_semantic_bridge_evidence_status=missing")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_identity_schema=trust_ir.native.semantic_bridge.proof_identity.v2")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_identity_schema_version=2")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_identity_digest=sha256:")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_api=NativeVerificationBundle::petri_successor_semantic_bridge_proof_admission_report()")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_schema=trust_ir.native.petri_successor.semantic_bridge_proof_admission.v1")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_schema_version=1")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_status_code=blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_reason_code=proof_handoff_blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_fail_closed=true")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_required_artifact_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_resolution_count=0")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_resolution_status_codes=none")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_resolution_reason_codes=none")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_resolution_authority_codes=none")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_authoritative_bytes_count=0")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_blocked_artifact_kind=none")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_blocked_artifact_reason_code=none")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_proof_handoff_status_code=blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_admission_proof_handoff_reason_code=binding_blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_fail_closed=true")
                && semantic_bridge.contains("trust_ir_semantic_bridge_proof_status=discharged")
                && semantic_bridge.contains("production_selected=false")
                && semantic_bridge.contains("fail_closed=true"),
            "produced bundle should expose a ready semantic successor bridge while production remains gated: {semantic_bridge}"
        );

        let ay_facade = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_native_verification_bundle_facade"))
            .expect("AY native bundle facade evidence should be emitted");
        assert!(
            ay_facade.contains("schema=ay.chc.trust_mc_native_verification_bundle_facade.v2")
                && ay_facade.contains("schema_version=2")
                && ay_facade.contains("source=trust_mcNativeVerificationBundleReport")
                && ay_facade.contains("problem=trust_mc_native_verification_bundle")
                && ay_facade.contains("preferred_backend_code=ay_chc_trust_mc_native_bundle")
                && ay_facade.contains("domain=native_bundle")
                && ay_facade.contains("scope=trust_mc_native_chc")
                && ay_facade
                    .contains("api=ay_trust_mc_native_bundle::solve_trust_mc_petri_successor_native_verification_bundle")
                && ay_facade.contains(&format!(
                    "required_ay_rev={AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_REQUIRED_AY_REV}"
                ))
                && ay_facade.contains(&format!(
                    "current_ay_rev={AY_TRUST_MC_NATIVE_VERIFICATION_BUNDLE_FACADE_CURRENT_AY_REV}"
                ))
                && ay_facade.contains("bundle_source=petri_native_production_path")
                && ay_facade.contains("bundle_validated=true")
                && ay_facade.contains(
                    "formula_schema=ty.petri.native.successor.plan_cache_equivalence.v1"
                )
                && ay_facade.contains("status_code=blocked")
                && ay_facade.contains("reason_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("consumer_acceptance_api=ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleReport::accept_for_consumer")
                && ay_facade.contains("consumer_rejection_status_code=blocked")
                && ay_facade.contains("consumer_rejection_reason_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("consumer_rejection_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("accepted_for_consumer=false")
                && ay_facade.contains("fail_closed=true")
                && ay_facade.contains("consumer_rejection_fail_closed=true")
                && ay_facade.contains("consumer_rejection_ready_for_trust_mc_chc_handoff=false")
                && ay_facade.contains("model_validated=false")
                && ay_facade.contains("verification_level_code=typed_handoff")
                && ay_facade.contains("proof_replay_status_code=blocked")
                && ay_facade.contains("ready_for_trust_mc_chc_handoff=false")
                && ay_facade.contains("trust_mc_request_count=0")
                && ay_facade.contains("trust_mc_evidence_count=0")
                && ay_facade.contains("native_evidence_entry_count=1")
                && ay_facade.contains("matched_trust_mc_request_count=0")
                && ay_facade.contains("matched_trust_mc_chc_request_count=0")
                && ay_facade.contains("matched_trust_mc_evidence_count=0")
                && ay_facade.contains("matched_trust_mc_artifact_count=0")
                && ay_facade.contains("matched_trust_mc_artifact_kind_codes=none")
                && ay_facade.contains("matched_trust_mc_request_ids=none")
                && ay_facade.contains("matched_trust_mc_request_mode_codes=none")
                && ay_facade.contains("semantic_bridge_status_code=blocked")
                && ay_facade.contains("semantic_bridge_reason_code=trusted_proof_not_admitted")
                && ay_facade.contains("semantic_bridge_evidence_status_code=missing")
                && ay_facade.contains("semantic_bridge_proof_identity_schema=trust_ir.native.semantic_bridge.proof_identity.v2")
                && ay_facade.contains("semantic_bridge_proof_identity_schema_version=2")
                && ay_facade.contains("semantic_bridge_proof_identity_digest=sha256:")
                && ay_facade.contains("semantic_bridge_fail_closed=true")
                && ay_facade.contains("semantic_bridge_relation_code=petri_successor")
                && ay_facade.contains("semantic_bridge_function_index=0")
                && ay_facade.contains(
                    "semantic_bridge_formula_schema=ty.petri.native.successor.plan_cache_equivalence.v1"
                )
                && ay_facade.contains("semantic_bridge_proof_obligation_index=0")
                && ay_facade.contains("semantic_bridge_proof_status_code=discharged")
                && ay_facade.contains("production_selected=false"),
            "AY facade row should see the represented semantic bridge and keep AY lowering advisory: {ay_facade}"
        );

        let admission_blocker = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg trust_cg_admission_blocker"))
            .expect("native JIT admission blocker evidence should be emitted");
        assert!(
            admission_blocker.contains("source=NativeInstallGateAdmissionSummary")
                && admission_blocker.contains("source_package=trust-cg-codegen")
                && admission_blocker.contains("package=trust-cg-codegen")
                && admission_blocker.contains("schema=trust-cg.phase6.native_install_gate.admission_summary.v1")
                && admission_blocker.contains("schema_version=1")
                && admission_blocker.contains("consumer=mcc")
                && admission_blocker.contains("consumer_mode=petri_successor")
                && admission_blocker.contains("kind=petri_native_successor")
                && admission_blocker.contains("surface=mcc_replay")
                && admission_blocker.contains("disposition=rejected")
                && admission_blocker.contains("status_code=rejected")
                && admission_blocker.contains("rejection_code=missing_native_install_gate_packet")
                && admission_blocker.contains("reason_code=missing_native_install_gate_packet")
                && admission_blocker.contains("requested_authority=active_callable")
                && admission_blocker.contains("install_authority=none")
                && admission_blocker
                    .contains("bundle_api=NativeVerificationBundle::native_evidence_consumption_report")
                && admission_blocker
                    .contains("admission_api=trust-cg::petri_native_successor_admission_from_trust_ir_bundle")
                && admission_blocker.contains("bundle_source=petri_native_production_path")
                && admission_blocker.contains("bundle_validated=true")
                && admission_blocker.contains("trust_ir_transport_identity_available=true")
                && admission_blocker.contains("trust_ir_bundle_consumed=true")
                && admission_blocker.contains("trust_ir_consumption_status=available")
                && admission_blocker.contains("trust_ir_consumption_entries=1")
                && admission_blocker.contains("consumed_certificates=0")
                && admission_blocker.contains("artifact_count=4")
                && admission_blocker.contains("native_evidence_backend_metadata_artifacts=0")
                && admission_blocker.contains("native_evidence_semantic_proof_artifacts=3")
                && admission_blocker.contains("native_evidence_native_execution_artifacts=1")
                && admission_blocker.contains("native_evidence_other_artifacts=0")
                && admission_blocker.contains("native_evidence_metadata_only=false")
                && admission_blocker.contains("native_evidence_semantic_proof_available=true")
                && admission_blocker
                    .contains("native_evidence_native_execution_artifact_available=true")
                && admission_blocker.contains("native_evidence_metadata_request_ids=none")
                && admission_blocker
                    .contains("native_evidence_metadata_request_digests=none")
                && admission_blocker.contains("native_evidence_metadata_module_digest=sha256:")
                && admission_blocker.contains("native_evidence_metadata_artifact_digests=none")
                && admission_blocker.contains("actions_ty_native_activate=false")
                && admission_blocker.contains("useful_native_delta=0")
                && admission_blocker.contains("packet_hash=")
                && admission_blocker.contains("artifact_id=petri_successor")
                && admission_blocker.contains("request_digests=1")
                && admission_blocker.contains("evidence_digests=1")
                && admission_blocker.contains("production_selected=false")
                && admission_blocker.contains("fail_closed=true"),
            "trust-cg admission blocker should come from the shared trust-codegen Petri summary and remain fail-closed: {admission_blocker}"
        );

        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        assert!(
            execution_plan.contains("source=PetriNativeSuccessorExecutionPlan")
                && execution_plan.contains("schema=trust-cg.petri.native_successor.execution_plan.v1")
                && execution_plan.contains("consumer=mcc")
                && execution_plan.contains("consumer_mode=ty_petri_native_jit")
                && execution_plan.contains("kind=petri_successor")
                && execution_plan.contains("surface=native_successor")
                && execution_plan.contains("status_code=rejected")
                && execution_plan.contains("rejection_code=missing_native_install_gate_packet")
                && execution_plan.contains("reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("requested_authority=canary_callable")
                && execution_plan.contains("install_authority=none")
                && execution_plan.contains(
                    "execution_plan_api=trust-cg::petri_native_successor_execution_plan_from_trust_ir_bundle"
                )
                && execution_plan.contains(
                    "expected_api=PetriNativeSuccessorExecutionExpected::canary_callable"
                )
                && execution_plan.contains(
                    "trampoline_contract_api=trust-cg::petri_native_successor_trampoline_contract"
                )
                && execution_plan.contains(
                    "install_packet_api=trust-cg::petri_native_successor_install_packet_from_trust_ir_bundle"
                )
                && execution_plan.contains(
                    "admission_api=trust-cg::petri_native_successor_admission_from_trust_ir_bundle"
                )
                && execution_plan.contains("bundle_source=petri_native_production_path")
                && execution_plan.contains("bundle_validated=true")
                && execution_plan.contains(&format!(
                    "entry_function={}",
                    PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL
                ))
                && execution_plan.contains("input_state_bytes=24")
                && execution_plan.contains("output_state_bytes=24")
                && execution_plan.contains("state_alignment_bytes=8")
                && execution_plan.contains("execution_plan_available=true")
                && execution_plan.contains("execution_plan_status_code=available")
                && execution_plan.contains("execution_plan_reason_code=available")
                && execution_plan.contains("trust_ir_transport_identity_available=true")
                && execution_plan.contains("trust_ir_bundle_consumed=true")
                && execution_plan.contains("trust_ir_consumption_status=available")
                && execution_plan.contains("native_evidence_backend_metadata_artifacts=0")
                && execution_plan.contains("native_evidence_semantic_proof_artifacts=3")
                && execution_plan.contains("native_evidence_native_execution_artifacts=0")
                && execution_plan.contains("native_evidence_metadata_only=false")
                && execution_plan.contains("native_evidence_semantic_proof_available=true")
                && execution_plan
                    .contains("native_evidence_native_execution_artifact_available=false")
                && execution_plan.contains("native_evidence_metadata_request_ids=none")
                && execution_plan.contains("native_evidence_metadata_request_digests=none")
                && execution_plan
                    .contains("native_evidence_metadata_module_digest=sha256:")
                && execution_plan.contains("native_evidence_metadata_artifact_digests=none")
                && execution_plan.contains("callable_contract_available=false")
                && execution_plan.contains("trampoline_contract_available=false")
                && execution_plan.contains("install_packet_available=false")
                && execution_plan.contains("install_packet_status_code=missing")
                && execution_plan
                    .contains("install_packet_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains(
                    "downstream_contract_api=trust-cg::petri_native_successor_downstream_contract_descriptor"
                )
                && execution_plan.contains(
                    "downstream_contract_schema=trust-cg.petri.native_successor.downstream_contract.v1"
                )
                && execution_plan.contains(
                    "downstream_trust_ir_bundle_identity_schema=trust_ir.native.bundle_identity_contract.v1"
                )
                && execution_plan.contains(&format!(
                    "downstream_trust_ir_transport_identity_schema={}",
                    trust_ir::NATIVE_TRANSPORT_IDENTITY_SCHEMA
                ))
                && execution_plan.contains(
                    "downstream_runtime_readiness_required_fields=call_packet,native_install_gate_packet,trampoline_contract,callable_lifetime_proof,runtime_abi_proof,current_generation"
                )
                && execution_plan.contains(
                    "runtime_readiness_api=trust-cg::petri_native_successor_runtime_readiness_packet"
                )
                && execution_plan.contains(
                    "runtime_readiness_installed_artifact_api=InstalledArtifact::petri_native_successor_runtime_readiness_packet"
                )
                && execution_plan
                    .contains("runtime_readiness_installed_artifact_required_trust_cg_rev=690f04d7")
                && execution_plan.contains(
                    "runtime_readiness_source=InstalledArtifact::petri_native_successor_runtime_readiness_packet"
                )
                && execution_plan.contains("runtime_readiness_installed_artifact_available=true")
                && execution_plan
                    .contains("runtime_readiness_schema=trust-cg.petri.native_successor.runtime_readiness_packet.v1")
                && execution_plan.contains("runtime_readiness_packet_available=true")
                && execution_plan.contains("runtime_readiness_status_code=blocked")
                && execution_plan
                    .contains("runtime_readiness_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("runtime_readiness_status_in_downstream_contract=true")
                && execution_plan
                    .contains("runtime_readiness_blocker_code=missing_native_install_gate_packet")
                && execution_plan.contains("runtime_readiness_blocker_in_downstream_contract=true")
                && execution_plan.contains("runtime_readiness_blocker_stage=manifest_identity")
                && execution_plan.contains("runtime_readiness_ready_for_runtime_call=false")
                && execution_plan.contains("mock_executable_call_role=test_diagnostic_only")
                && execution_plan.contains("mock_executable_call_production_enabled=false")
                && execution_plan.contains(
                    "call_packet_api=trust-cg::petri_native_successor_call_packet_from_trust_ir_bundle"
                )
                && execution_plan
                    .contains("call_packet_schema=trust-cg.petri.native_successor.call_packet.v1")
                && execution_plan.contains("call_packet_type=PetriNativeSuccessorCallPacket")
                && execution_plan
                    .contains("callable_pointer_type=PetriNativeSuccessorCallablePointer")
                && execution_plan.contains("call_packet_required_trust_cg_rev=2d31fd8b")
                && execution_plan.contains(&format!(
                    "call_packet_current_trust_cg_rev={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV
                ))
                && execution_plan.contains("call_packet_api_available=true")
                && execution_plan.contains("call_packet_api_status_code=available")
                && execution_plan.contains("call_packet_type_available=true")
                && execution_plan.contains("callable_pointer_type_available=true")
                && execution_plan.contains(
                    "call_packet_available=false call_packet_reason_code=missing_native_install_gate_packet"
                )
                && execution_plan.contains(
                    "callable_pointer_available=false callable_pointer_reason_code=missing_native_install_gate_packet"
                )
                && execution_plan.contains("concrete_callable_pointer_required=true")
                && execution_plan.contains("concrete_callable_pointer_available=false")
                && execution_plan.contains("concrete_callable_pointer_status_code=missing")
                && execution_plan.contains("concrete_callable_packet_required=true")
                && execution_plan.contains("concrete_callable_packet_available=false")
                && execution_plan.contains("concrete_callable_packet_status_code=missing")
                && execution_plan.contains("call_packet_readiness_status_code=blocked")
                && execution_plan
                    .contains("call_packet_readiness_blocker=missing_native_install_gate_packet")
                && execution_plan.contains("callable_authorized=false")
                && execution_plan
                    .contains("callable_authorized_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("callable_handoff_available=true")
                && execution_plan.contains("callable_handoff_reason_code=available")
                && execution_plan
                    .contains("callable_handoff_blocker=missing_native_install_gate_packet")
                && execution_plan
                    .contains("callable_handoff_required_evidence=trust-cg.phase6.native_install_gate.v1")
                && execution_plan.contains(
                    "callable_handoff_upstream_ask=provide_runtime_callable_pointer_and_accepted_install_packet"
                )
                && execution_plan.contains("plan_fail_closed=true")
                && execution_plan.contains("actions_expose_callable=false")
                && execution_plan.contains("actions_expose_callable_blocked_by_runtime_readiness=true")
                && execution_plan
                    .contains("actions_expose_callable_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("actions_ty_native_activate=false")
                && execution_plan
                    .contains("actions_ty_native_activate_blocked_by_runtime_readiness=true")
                && execution_plan
                    .contains("actions_ty_native_activate_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains(
                    "ay_native_bundle_facade_api=ay_trust_mc_native_bundle::solve_trust_mc_petri_successor_native_verification_bundle"
                )
                && execution_plan.contains(
                    "ay_native_bundle_facade_schema=ay.chc.trust_mc_native_verification_bundle_facade.v2"
                )
                && execution_plan.contains("ay_native_bundle_facade_status_code=blocked")
                && execution_plan
                    .contains("ay_native_bundle_facade_reason_code=chc_problem_lowering_unavailable")
                && execution_plan.contains("ay_native_bundle_facade_consumer_acceptance_api=ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleReport::accept_for_consumer")
                && execution_plan
                    .contains("ay_native_bundle_facade_consumer_rejection_status_code=blocked")
                && execution_plan.contains(
                    "ay_native_bundle_facade_consumer_rejection_reason_code=chc_problem_lowering_unavailable"
                )
                && execution_plan.contains(
                    "ay_native_bundle_facade_consumer_rejection_code=chc_problem_lowering_unavailable"
                )
                && execution_plan.contains("ay_native_bundle_facade_accepted_for_consumer=false")
                && execution_plan.contains("ay_native_bundle_facade_fail_closed=true")
                && execution_plan
                    .contains("ay_native_bundle_facade_consumer_rejection_fail_closed=true")
                && execution_plan.contains(
                    "ay_native_bundle_facade_consumer_rejection_ready_for_trust_mc_chc_handoff=false"
                )
                && execution_plan.contains("ay_native_bundle_facade_model_validated=false")
                && execution_plan
                    .contains("ay_native_bundle_facade_verification_level_code=typed_handoff")
                && execution_plan
                    .contains("ay_native_bundle_facade_proof_replay_status_code=blocked")
                && execution_plan
                    .contains("ay_native_bundle_facade_ready_for_trust_mc_chc_handoff=false")
                && execution_plan.contains(
                    "ay_native_bundle_facade_semantic_bridge_proof_identity_schema=trust_ir.native.semantic_bridge.proof_identity.v2"
                )
                && execution_plan.contains(
                    "ay_native_bundle_facade_semantic_bridge_proof_identity_schema_version=2"
                )
                && execution_plan.contains(
                    "ay_native_bundle_facade_semantic_bridge_proof_identity_digest=sha256:"
                )
                && execution_plan
                    .contains("ay_native_bundle_facade_semantic_bridge_fail_closed=true")
                && execution_plan
                    .contains("ay_native_bundle_facade_semantic_bridge_status_code=blocked")
                && execution_plan
                    .contains("ay_native_bundle_facade_semantic_bridge_reason_code=trusted_proof_not_admitted")
                && execution_plan.contains(
                    "ay_native_bundle_facade_semantic_bridge_evidence_status_code=missing"
                )
                && execution_plan
                    .contains("ay_native_bundle_facade_matched_trust_mc_request_count=0")
                && execution_plan
                    .contains("ay_native_bundle_facade_matched_trust_mc_chc_request_count=0")
                && execution_plan
                    .contains("ay_native_bundle_facade_matched_trust_mc_evidence_count=0")
                && execution_plan
                    .contains("ay_native_bundle_facade_matched_trust_mc_artifact_count=0")
                && execution_plan
                    .contains("ay_native_bundle_facade_accepted_for_native_production=false")
                && execution_plan.contains("production_selected=false")
                && execution_plan.contains("fail_closed=true")
                && execution_plan.contains("native_successor_runtime_status_code=blocked"),
            "trust-cg execution plan must preserve the native-install and semantic-proof blockers: {execution_plan}"
        );
        assert_execution_plan_exposes_compile_artifact_handoff(execution_plan, true);

        let successor = report
            .rejected
            .iter()
            .find(|capability| capability.problem == Some(ProblemKind::NativeSuccessor))
            .expect("successor native capability should remain rejected");
        assert!(successor.detail.as_deref().is_some_and(|detail| {
            detail.contains("production_selected=false")
                && detail.contains("trust_ir_transport_identity_available=true")
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn validation_only_native_evidence_artifact_binds_request_and_module_identity() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).expect("plan cache should build");
        let mut bundle = match native::petri_native_successor_verification_bundle(&net, &cache) {
            native::PetriNativeVerificationBundleProduction::Available(bundle) => bundle,
            native::PetriNativeVerificationBundleProduction::Blocked(blocker) => {
                panic!("native verification bundle should be available: {blocker:?}")
            }
        };
        bundle.evidence_bundles.clear();

        assert!(bundle.evidence_bundles.is_empty());
        let augmented =
            petri_native_successor_validation_only_native_evidence_bundle(bundle.clone());
        assert_eq!(augmented.evidence_bundles.len(), bundle.requests.len());

        let evidence = augmented
            .evidence_bundles
            .first()
            .expect("validation-only evidence bundle should be present");
        let request = bundle
            .requests
            .first()
            .expect("native bundle should carry a request");
        let artifact = evidence
            .artifacts()
            .first()
            .expect("validation-only evidence should carry a metadata artifact");
        let expected_artifact = petri_native_successor_validation_only_native_evidence_artifact(
            request,
            bundle.trust_ir_module_digest,
        );
        let wrong_module_artifact = petri_native_successor_validation_only_native_evidence_artifact(
            request,
            trust_ir::ProofDigest::sha256([0xA5; 32]),
        );
        let profile = trust_cg_petri_native_evidence_profile(&augmented);

        assert_eq!(evidence.request(), request.id());
        assert_eq!(
            artifact.kind,
            trust_ir::NativeEvidenceArtifactKind::BackendCapabilityMetadata
        );
        assert_eq!(artifact.digest, expected_artifact.digest);
        assert_ne!(artifact.digest, wrong_module_artifact.digest);
        assert_eq!(
            artifact.digest.algorithm,
            trust_ir::ProofDigestAlgorithm::Sha256
        );
        assert!(artifact.name.contains(&request.id().to_string()));
        assert_eq!(profile.backend_metadata_artifact_count, 1);
        assert_eq!(profile.semantic_proof_artifact_count, 0);
        assert_eq!(profile.native_execution_artifact_count, 0);
        assert!(profile.metadata_only());
        assert!(!profile.semantic_proof_available());
        assert_eq!(profile.metadata_request_ids, vec![request.id().to_string()]);
        assert_eq!(
            profile.metadata_request_digests,
            vec![request.stable_digest().to_string()]
        );
        assert_eq!(
            profile.metadata_artifact_digests,
            vec![artifact.digest.to_string()]
        );
        assert_eq!(
            profile.trust_ir_module_digest,
            bundle.trust_ir_module_digest.to_string()
        );

        let semantic_bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let semantic_profile = trust_cg_petri_native_evidence_profile(&semantic_bundle);
        assert_eq!(semantic_profile.backend_metadata_artifact_count, 0);
        assert_eq!(semantic_profile.semantic_proof_artifact_count, 3);
        assert_eq!(semantic_profile.native_execution_artifact_count, 0);
        assert!(!semantic_profile.metadata_only());
        assert!(semantic_profile.semantic_proof_available());

        let receipt_evidence = native_jit_receipt_evidence_for_bundle(
            &semantic_bundle,
            &semantic_bundle,
            PetriNativeInstalledArtifactEvidence::NotAttempted.as_ref(),
            &net,
            &cache,
        );
        assert!(receipt_evidence.validation_receipt_available);
        assert!(!receipt_evidence.parity_receipt_available);

        let accepted_gate =
            NativeJitFailClosedGate::from_env().with_receipt_evidence(receipt_evidence);
        assert_eq!(
            accepted_gate.validation_receipt_status_code(),
            PETRI_NATIVE_VALIDATION_RECEIPT_STATUS_ACCEPTED
        );
        assert_eq!(
            accepted_gate.validation_receipt_reason_code(),
            TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn petri_native_semantic_receipt_requires_discharged_exact_artifacts() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).expect("plan cache should build");
        let bundle = native_verification_bundle_fixture(&net);

        assert!(native::petri_native_successor_semantic_receipt_available(
            &bundle, &net, &cache
        ));

        let mut trusted = bundle.clone();
        let obligation = trusted
            .module
            .proof_obligations
            .iter_mut()
            .find(|obligation| obligation.id == native::PETRI_NATIVE_TRANSLATION_OBLIGATION)
            .expect("fixture should carry the Petri successor translation obligation");
        obligation.status = trust_ir::ProofStatus::Trusted;
        assert!(
            !native::petri_native_successor_semantic_receipt_available(&trusted, &net, &cache),
            "trusted local obligations must not clear the production semantic receipt"
        );

        let mut stale_artifact = bundle.clone();
        let mut mutated = false;
        for evidence in &mut stale_artifact.evidence_bundles {
            let artifacts = match evidence {
                trust_ir::NativeEvidenceBundle::TrustVc(evidence) => &mut evidence.artifacts,
                trust_ir::NativeEvidenceBundle::TrustMc(evidence) => &mut evidence.artifacts,
                trust_ir::NativeEvidenceBundle::TrustWp(evidence) => &mut evidence.artifacts,
            };
            if let Some(artifact) = artifacts.iter_mut().find(|artifact| {
                artifact.kind == trust_ir::NativeEvidenceArtifactKind::TrustMcModel
            }) {
                artifact.digest = trust_ir::ProofDigest::sha256([0xA5; 32]);
                mutated = true;
                break;
            }
        }
        assert!(mutated, "fixture should carry a trust_mc model artifact");
        assert!(
            !native::petri_native_successor_semantic_receipt_available(
                &stale_artifact,
                &net,
                &cache
            ),
            "stale semantic artifact digests must not clear the production semantic receipt"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn semantic_successor_bridge_row_sources_upstream_descriptor_fields() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let semantic_bridge = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit semantic_successor_bridge"))
            .expect("native JIT semantic successor bridge evidence should be emitted");
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let semantic_bridge_surface = downstream_contract.semantic_bridge;
        let trust_ir_identity = downstream_contract.trust_ir_native_bundle_identity;
        let expected_schema_version = semantic_bridge_surface.schema_version.to_string();
        let expected_required_fields = semantic_bridge_surface.required_fields.join(",");
        let expected_status_codes = semantic_bridge_surface.status_codes.join(",");
        let expected_blocker_codes = semantic_bridge_surface.blocker_codes.join(",");

        assert_eq!(
            evidence_field(semantic_bridge, "schema"),
            Some(semantic_bridge_surface.schema)
        );
        assert_eq!(
            evidence_field(semantic_bridge, "schema_version"),
            Some(expected_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(semantic_bridge, "downstream_semantic_bridge_surface"),
            Some(semantic_bridge_surface.name)
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "downstream_semantic_bridge_required_fields"
            ),
            Some(expected_required_fields.as_str())
        );
        assert_eq!(
            evidence_field(semantic_bridge, "downstream_semantic_bridge_status_codes"),
            Some(expected_status_codes.as_str())
        );
        assert_eq!(
            evidence_field(semantic_bridge, "downstream_semantic_bridge_blocker_codes"),
            Some(expected_blocker_codes.as_str())
        );
        for (key, field) in [
            (
                "trust_ir_api",
                TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeReport,
            ),
            (
                "trust_ir_petri_successor_report_api",
                TrustIrPetriTrustMcProvidedField::PetriSuccessorSemanticBridgeReport,
            ),
            (
                "trust_ir_semantic_bridge_constructor_api",
                TrustIrPetriTrustMcProvidedField::PetriSuccessorSemanticBridgeConstructor,
            ),
            (
                "trust_ir_semantic_bridge_acceptance_api",
                TrustIrPetriTrustMcProvidedField::RepresentsPetriSuccessorPlanCacheEquivalence,
            ),
        ] {
            let expected =
                trust_ir_petri_trust_mc_provided_field(trust_ir_identity.provided_fields, field);
            assert_ne!(
                expected, "missing_trust_ir_petri_trust_mc_provided_field",
                "trust-ir descriptor must expose the helper named by {key}"
            );
            assert_eq!(evidence_field(semantic_bridge, key), Some(expected));
        }
        assert_eq!(
            evidence_field(semantic_bridge, "reason_code"),
            evidence_field(semantic_bridge, "trust_cg_reason_code")
        );
        let reason_code = evidence_field(semantic_bridge, "reason_code")
            .expect("semantic bridge should emit a reason_code");
        // When the producer attaches a discharged proof obligation, the semantic
        // bridge represents the relation and emits `reason_code=none`. Otherwise it
        // must come from the trust-codegen descriptor's blocker list.
        assert!(
            reason_code == "none"
                || semantic_bridge_surface.blocker_codes.contains(&reason_code),
            "semantic bridge reason_code should be `none` or from trust-codegen descriptor: {semantic_bridge}"
        );
        let successor_relation_represented =
            evidence_field(semantic_bridge, "successor_relation_represented")
                .expect("successor_relation_represented should be emitted");
        assert!(
            successor_relation_represented == "true" || successor_relation_represented == "false",
            "successor_relation_represented should be boolean: {semantic_bridge}"
        );
        assert_eq!(
            evidence_field(semantic_bridge, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(semantic_bridge, "fail_closed"), Some("true"));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn semantic_successor_bridge_emits_trust_ir_proof_identity_component_rows() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let semantic_bridge = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit semantic_successor_bridge"))
            .expect("native JIT semantic successor bridge evidence should be emitted");
        let readiness = trust_ir_component_readiness_row(
            &report,
            TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT,
        );
        let manifest_rows = trust_ir_component_manifest_rows(
            &report,
            TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT,
        );
        let replay_health_rows = trust_ir_component_manifest_rows(
            &report,
            TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_REPLAY_HEALTH_COMPONENT,
        );
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let trust_ir_identity = downstream_contract.trust_ir_native_bundle_identity;
        let row_count = evidence_field_usize(readiness, "identity_row_count");
        let replay_health_row_count =
            evidence_field_usize(readiness, "identity_replay_component_health_row_count");
        let proof_identity_schema_version =
            trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA_VERSION.to_string();

        assert_eq!(manifest_rows.len(), row_count);
        assert_eq!(replay_health_rows.len(), replay_health_row_count);
        assert!(replay_health_row_count > 0);
        assert_eq!(
            evidence_field(readiness, "schema"),
            Some(trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA)
        );
        assert_eq!(
            evidence_field(readiness, "schema_version"),
            Some(proof_identity_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(readiness, "identity_schema"),
            Some(trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA)
        );
        assert_eq!(
            evidence_field(readiness, "identity_digest"),
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_digest"
            )
        );
        assert_eq!(
            evidence_field(readiness, "producer_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(readiness, "replay_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayReportForKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_component_health_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayComponentHealthSummaryKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_component_health_api"
            ),
            evidence_field(readiness, "identity_replay_component_health_api")
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_component_health_text_available"),
            Some("true")
        );
        assert_eq!(
            evidence_field(readiness, "producer_status_code"),
            evidence_field(semantic_bridge, "trust_ir_semantic_bridge_status_code")
        );
        assert_eq!(
            evidence_field(readiness, "producer_reason_code"),
            evidence_field(semantic_bridge, "trust_ir_semantic_bridge_reason_code")
        );
        assert_eq!(
            evidence_field(readiness, "producer_fail_closed"),
            evidence_field(semantic_bridge, "trust_ir_semantic_bridge_fail_closed")
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_status_code"),
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_status_code"
            )
        );
        assert_eq!(
            evidence_field(readiness, "identity_replayable"),
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replayable"
            )
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_fail_closed"),
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_fail_closed"
            )
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_diagnostic_count"),
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_diagnostic_count"
            )
        );
        assert_eq!(
            evidence_field(readiness, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(readiness, "fail_closed"), Some("true"));
        assert!(manifest_rows.iter().any(|row| {
            evidence_field(row, "row_key") == Some("semantic_bridge_proof_identity.digest")
                && evidence_field(row, "row_value") == evidence_field(readiness, "identity_digest")
        }));
        assert!(manifest_rows.iter().any(|row| {
            evidence_field(row, "row_key") == Some("semantic_bridge_proof_identity.report.reason")
                && evidence_field(row, "row_value")
                    == evidence_field(readiness, "producer_reason_code")
        }));
        assert!(replay_health_rows.iter().any(|row| {
            evidence_field(row, "row_key")
                == Some("semantic_bridge_proof_identity_replay_component_summary.status")
                && evidence_field(row, "row_value")
                    == evidence_field(readiness, "identity_replay_status_code")
        }));
        assert!(replay_health_rows.iter().any(|row| {
            evidence_field(row, "row_key")
                == Some("semantic_bridge_proof_identity_replay_component_summary.diagnostic_count")
                && evidence_field(row, "row_value")
                    == evidence_field(readiness, "identity_replay_diagnostic_count")
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn ay_facade_rejects_represented_chc_handoff_until_lowering_available() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        let bridge = petri_native_successor_semantic_bridge(&bundle);
        let ay_report =
            ay_trust_mc_native_bundle::solve_trust_mc_petri_successor_native_verification_bundle(
                &bundle,
                bridge.function,
            );
        let decision_accepts = matches!(
            ay_report.consumer_decision(),
            ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleConsumerDecision::Accepted
        );
        assert_eq!(
            ay_report.accept_for_consumer().is_ok(),
            decision_accepts,
            "AY consumer decision and acceptance accessors must stay consistent"
        );

        let ay_facade_evidence = add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );

        let ay_facade = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_native_verification_bundle_facade"))
            .expect("AY native bundle facade evidence should be emitted");
        assert!(
            !ay_facade_evidence.ready_for_trust_mc_chc_handoff
                && !ay_facade_evidence.accepted_for_consumer
                && !ay_facade_evidence.is_accepted_for_native_production()
                && ay_facade_evidence.consumer_rejection_code
                    == "chc_problem_lowering_unavailable"
                && ay_facade_evidence.model_acceptance_status_code == "rejected"
                && ay_facade_evidence.model_acceptance_reason_code == "proof_handoff_blocked"
                && ay_facade_evidence.model_acceptance_api
                    == ay_trust_mc_native_bundle::PETRI_SUCCESSOR_TRUST_MC_CHC_MODEL_ACCEPTANCE_REPORT_API_NAME
                && ay_facade_evidence.model_acceptance_consumer_acceptance_api
                    == ay_trust_mc_native_bundle::PETRI_SUCCESSOR_TRUST_MC_CHC_CONSUMER_ACCEPTANCE_API_NAME
                && ay_facade_evidence.model_acceptance_consumer_rejection_reason_code
                    == "proof_handoff_blocked"
                && ay_facade_evidence.model_acceptance_consumer_rejection_fail_closed
                && !ay_facade_evidence.model_acceptance_proof_handoff_ready
                && !ay_facade_evidence.model_acceptance_ready_for_solver_validation
                && !ay_facade_evidence.model_acceptance_solver_model_validation_present
                && !ay_facade_evidence.model_acceptance_solver_model_validation_accepted
                && ay_facade_evidence.consumer_rejection_fail_closed
                && ay_facade.contains("status_code=blocked")
                && ay_facade.contains("reason_code=chc_problem_lowering_unavailable")
                && ay_facade
                    .contains("consumer_rejection_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("accepted_for_consumer=false")
                && ay_facade.contains("consumer_rejection_fail_closed=true")
                && ay_facade.contains("ready_for_trust_mc_chc_handoff=false")
                && ay_facade.contains("consumer_rejection_ready_for_trust_mc_chc_handoff=false")
                && ay_facade.contains("semantic_bridge_status_code=blocked")
                && ay_facade.contains("semantic_bridge_reason_code=trusted_proof_not_admitted")
                && ay_facade.contains("semantic_bridge_evidence_status_code=missing")
                && ay_facade.contains("semantic_bridge_fail_closed=true")
                && ay_facade.contains("matched_trust_mc_chc_request_count=0")
                && ay_facade.contains("matched_trust_mc_evidence_count=0")
                && ay_facade.contains("matched_trust_mc_artifact_count=0"),
            "AY must preserve the fail-closed consumer rejection for an unadmitted semantic proof: {ay_facade}"
        );
        let ay_route_admission = report
            .evidence
            .iter()
            .find(|evidence| {
                evidence.contains("AY trust_mc_petri_successor_native_route_admission")
            })
            .expect("AY native route admission evidence should be emitted");
        let route_decision =
            ay_trust_mc_native_bundle::trust_mc_petri_successor_native_route_admission_decision(
                &bundle,
                bridge.function,
            );
        let route_schema_version = route_decision.schema_version.to_string();
        let route_accepted_for_consumer = route_decision.accepted_for_consumer.to_string();
        let route_fail_closed = route_decision.fail_closed.to_string();
        assert_eq!(
            evidence_field(ay_route_admission, "schema"),
            Some(route_decision.schema)
        );
        assert_eq!(
            evidence_field(ay_route_admission, "schema_version"),
            Some(route_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(ay_route_admission, "status_code"),
            Some(route_decision.status_code)
        );
        assert_eq!(
            evidence_field(ay_route_admission, "reason_code"),
            Some(route_decision.reason_code)
        );
        assert_eq!(
            evidence_field(ay_route_admission, "accepted_for_consumer"),
            Some(route_accepted_for_consumer.as_str())
        );
        assert_eq!(
            evidence_field(ay_route_admission, "fail_closed"),
            Some(route_fail_closed.as_str())
        );
        assert!(
            ay_route_admission.contains(
                "helpers=ay_trust_mc_native_bundle::trust_mc_petri_successor_native_route_admission_decision"
            ) && ay_route_admission.contains(
                "validators=ay_trust_mc_native_bundle::validate_trust_mc_petri_successor_native_route_admission_decision"
            ),
            "AY route admission row should forward producer helper and validator ownership: {ay_route_admission}"
        );
        let ay_model_acceptance = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance evidence should be emitted");
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let trust_ir_native_bundle_identity = downstream_contract.trust_ir_native_bundle_identity;
        assert!(
            ay_model_acceptance
                .contains("schema=ay.chc.trust_mc_petri_successor_model_acceptance.v1")
                && ay_model_acceptance.contains("schema_version=1")
                && ay_model_acceptance.contains(
                    "api=ay::chc::trust_mc_petri_successor_chc_model_acceptance_report"
                )
                && ay_model_acceptance.contains("consumer_acceptance_api=ay::chc::TrustMcPetriSuccessorChcModelAcceptanceReport::accept_for_consumer")
                && ay_model_acceptance.contains("status_code=rejected")
                && ay_model_acceptance.contains("reason_code=proof_handoff_blocked")
                && ay_model_acceptance.contains("accepted_for_consumer=false")
                && ay_model_acceptance.contains("fail_closed=true")
                && ay_model_acceptance.contains("consumer_rejection_status_code=rejected")
                && ay_model_acceptance.contains("consumer_rejection_reason_code=proof_handoff_blocked")
                && ay_model_acceptance.contains("consumer_rejection_fail_closed=true")
                && ay_model_acceptance.contains("proof_handoff_ready=false")
                && ay_model_acceptance.contains("ready_for_solver_validation=false")
                && ay_model_acceptance.contains("solver_model_validation_present=false")
                && ay_model_acceptance.contains("solver_model_validation_accepted=false")
                && ay_model_acceptance
                    .contains("trust_mc_chc_proof_handoff_status_code=blocked")
                && ay_model_acceptance.contains(
                    "trust_mc_chc_proof_handoff_reason_code=binding_blocked"
                )
                && ay_model_acceptance
                    .contains("trust_mc_chc_model_validation_status_code=blocked")
                && ay_model_acceptance
                    .contains("trust_mc_chc_model_validation_reason_code=proof_handoff_blocked")
                && evidence_field(ay_model_acceptance, "trust_ir_contract_api")
                    == Some(trust_ir_petri_trust_mc_provided_field(
                        trust_ir_native_bundle_identity.provided_fields,
                        TrustIrPetriTrustMcProvidedField::ContractDescriptor,
                    ))
                && ay_model_acceptance
                    .contains("trust_ir_contract_schema=trust_ir.native.petri_successor.trust_mc_chc_contract.v1")
                && ay_model_acceptance.contains("trust_ir_contract_schema_version=1")
                && ay_model_acceptance.contains("trust_ir_contract_verifier_suite=trust_mc")
                && ay_model_acceptance.contains("trust_ir_contract_verification_mode=chc")
                && ay_model_acceptance
                    .contains("trust_ir_contract_binding_required_artifact_kinds=trust_mc_horn_clauses")
                && ay_model_acceptance.contains(
                    "trust_ir_contract_proof_handoff_required_artifact_kinds=replay_transcript"
                )
                && ay_model_acceptance
                    .contains("trust_ir_contract_proof_handoff_optional_artifact_kinds=trust_mc_model")
                && ay_model_acceptance
                    .contains("trust_ir_contract_model_validation_required_artifact_kinds=trust_mc_model")
                && ay_model_acceptance
                    .contains("trust_ir_contract_model_validation_requires_solver_acceptance=true")
                && ay_model_acceptance.contains(
                    "trust_ir_contract_production_acceptance_required_artifact_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_contract_model_acceptance_report_api_name=ay::chc::trust_mc_petri_successor_chc_model_acceptance_report"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_contract_consumer_acceptance_api_name=ay::chc::TrustMcPetriSuccessorChcModelAcceptanceReport::accept_for_consumer"
                )
                && ay_model_acceptance.contains("trust_ir_contract_production_acceptance_owner_suite=ay")
                && ay_model_acceptance
                    .contains("trust_ir_contract_production_requires_emitted_solver_artifacts=true")
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_schema=trust_ir.native.shared_primitive_contract.v1")
                && ay_model_acceptance.contains("trust_ir_shared_primitive_schema_version=1")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_contract_schema=trust_ir.native.petri_successor.trust_mc_chc_contract.v1"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_contract_schema_version=1")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_formula_schema=ty.petri.native.successor.plan_cache_equivalence.v1"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_readiness_report_schema=trust_ir.native.petri_successor.trust_mc_chc_model_validation_readiness.v1"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_readiness_report_schema_version=1")
                && ay_model_acceptance.contains("trust_ir_shared_primitive_verifier_suite=trust_mc")
                && ay_model_acceptance.contains("trust_ir_shared_primitive_verification_mode=chc")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_roles=solver_input|replay_transcript|solver_witness"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_optional_artifact_kinds=none")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_artifact_identity_api=NativeSharedPrimitiveArtifactRequirement::accepts_artifact_identity()"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_roles=solver_input|replay_transcript|solver_witness"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_digest_algorithms=sha256|sha256|sha256"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_owner_suites=ay|ay|ay"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_requires_emitted_solver_artifacts=true|true|true"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_production_artifact_owner_suites=ay")
                // The Petri native producer now binds all three required artifact
                // descriptors to emitted artifacts on the bundle, so the bound count
                // is 3 and unbound lists are `none`.
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_bound_artifact_requirement_count=3")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_bound_artifact_requirement_roles=solver_input|replay_transcript|solver_witness"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_unbound_artifact_requirement_roles=none")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_bound_artifact_requirement_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_unbound_artifact_requirement_kinds=none")
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_requires_solver_acceptance=true")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_model_acceptance_report_api_name=ay::chc::trust_mc_petri_successor_chc_model_acceptance_report"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_consumer_acceptance_api_name=ay::chc::TrustMcPetriSuccessorChcModelAcceptanceReport::accept_for_consumer"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_production_acceptance_owner_suite=ay")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_production_requires_emitted_solver_artifacts=true"
                )
                && ay_model_acceptance
                    .contains("trust_ir_contract_binding_status_codes=bound|blocked")
                && ay_model_acceptance.contains(
                    "trust_ir_contract_model_validation_readiness_reason_codes=solver_validation_required|proof_handoff_blocked|missing_model_artifact"
                )
                && ay_model_acceptance.contains("production_selected=false"),
            "AY model acceptance evidence should expose AY-owned acceptance and trust-ir contract fields: {ay_model_acceptance}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn ay_model_acceptance_row_sources_authority_api_names_from_upstream_descriptors() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );

        let ay_model_acceptance = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance evidence should be emitted");
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let trust_ir_native_bundle_identity = downstream_contract.trust_ir_native_bundle_identity;
        let shared_primitive_contract =
            downstream_contract.trust_ir_petri_trust_mc_chc_shared_primitive_contract;

        for (key, provided_field) in [
            (
                "trust_ir_contract_api",
                TrustIrPetriTrustMcProvidedField::ContractDescriptor,
            ),
            (
                "trust_ir_shared_primitive_artifact_identity_api",
                TrustIrPetriTrustMcProvidedField::ArtifactIdentity,
            ),
            (
                "trust_ir_shared_primitive_artifact_byte_resolution_api",
                TrustIrPetriTrustMcProvidedField::ArtifactByteResolution,
            ),
            (
                "trust_ir_shared_primitive_artifact_authority_api",
                TrustIrPetriTrustMcProvidedField::ArtifactAuthority,
            ),
            (
                "trust_ir_shared_primitive_authoritative_bytes_api",
                TrustIrPetriTrustMcProvidedField::AuthoritativeBytes,
            ),
        ] {
            let expected = trust_ir_petri_trust_mc_provided_field(
                trust_ir_native_bundle_identity.provided_fields,
                provided_field,
            );
            assert_ne!(
                expected, "missing_trust_ir_petri_trust_mc_provided_field",
                "trust-ir descriptor must expose the helper named by {key}"
            );
            assert_eq!(
                evidence_field(ay_model_acceptance, key),
                Some(expected),
                "Petri evidence should use the trust-ir-provided helper name for {key}"
            );
        }

        assert_eq!(
            evidence_field(ay_model_acceptance, "api"),
            Some(ay_trust_mc_native_bundle::PETRI_SUCCESSOR_TRUST_MC_CHC_MODEL_ACCEPTANCE_REPORT_API_NAME)
        );
        assert_eq!(
            evidence_field(ay_model_acceptance, "consumer_acceptance_api"),
            Some(ay_trust_mc_native_bundle::PETRI_SUCCESSOR_TRUST_MC_CHC_CONSUMER_ACCEPTANCE_API_NAME)
        );
        assert_eq!(
            evidence_field(
                ay_model_acceptance,
                "trust_ir_shared_primitive_model_acceptance_report_api_name"
            ),
            Some(shared_primitive_contract.production_acceptance_report_api_name())
        );
        assert_eq!(
            evidence_field(
                ay_model_acceptance,
                "trust_ir_shared_primitive_consumer_acceptance_api_name"
            ),
            Some(shared_primitive_contract.production_consumer_acceptance_api_name())
        );
        assert_eq!(
            evidence_field(ay_model_acceptance, "accepted_for_consumer"),
            Some("false")
        );
        assert_eq!(
            evidence_field(ay_model_acceptance, "production_selected"),
            Some("false")
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn ay_model_acceptance_row_uses_trust_ir_artifact_authority_resolution() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );

        let ay_model_acceptance = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance evidence should be emitted");
        assert!(
            ay_model_acceptance.contains(
                "trust_ir_shared_primitive_artifact_byte_resolution_api=NativeVerificationBundle::resolve_evidence_artifact_attachment()"
            )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_artifact_authority_api=NativeEvidenceArtifactResolution::is_authoritative()"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_authoritative_bytes_api=NativeEvidenceArtifactResolution::authoritative_bytes()"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_artifact_byte_attachment_count=0")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_artifact_byte_resolution_status_codes=blocked|blocked|blocked"
                )
                // All three artifact-descriptor slots are now populated by the
                // producer (semantic-evidence attachment), so byte-resolution returns
                // `missing_attachment` for each (no byte attachments supplied).
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_artifact_byte_resolution_reason_codes=missing_attachment|missing_attachment|missing_attachment"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_artifact_byte_resolution_authority_codes=informational|informational|informational"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_authoritative_artifact_requirement_count=0")
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_authoritative_artifact_requirement_roles=none")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_unauthoritative_artifact_requirement_roles=solver_input|replay_transcript|solver_witness"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_authoritative_artifact_requirement_kinds=none")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_unauthoritative_artifact_requirement_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_authoritative_artifact_bytes_count=0")
                && ay_model_acceptance.contains("accepted_for_consumer=false")
                && ay_model_acceptance.contains("production_selected=false"),
            "AY model acceptance row should use trust-ir artifact byte authority APIs and remain fail-closed without authoritative bytes: {ay_model_acceptance}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn ay_model_acceptance_row_uses_trust_ir_artifact_identity_requirements() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );

        let ay_model_acceptance = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance evidence should be emitted");
        assert!(
            ay_model_acceptance.contains(
                "trust_ir_shared_primitive_artifact_identity_api=NativeSharedPrimitiveArtifactRequirement::accepts_artifact_identity()"
            )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_roles=solver_input|replay_transcript|solver_witness"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_digest_algorithms=sha256|sha256|sha256"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_required_artifact_requirement_owner_suites=ay|ay|ay"
                )
                // The Petri native producer now binds all three required artifact
                // descriptors (solver_input, replay_transcript, solver_witness) to
                // emitted artifacts on the bundle, so the bound count is 3 and the
                // unbound lists are empty (`none`).
                && ay_model_acceptance
                    .contains("trust_ir_shared_primitive_bound_artifact_requirement_count=3")
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_bound_artifact_requirement_roles=solver_input|replay_transcript|solver_witness"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_unbound_artifact_requirement_roles=none"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_bound_artifact_requirement_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model"
                )
                && ay_model_acceptance.contains(
                    "trust_ir_shared_primitive_unbound_artifact_requirement_kinds=none"
                )
                && ay_model_acceptance.contains("accepted_for_consumer=false")
                && ay_model_acceptance.contains("production_selected=false"),
            "AY model acceptance row should consume trust-ir artifact identity helpers and remain fail-closed: {ay_model_acceptance}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn ay_model_acceptance_row_exposes_proof_replay_bridge_fields() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );

        let ay_model_acceptance = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance evidence should be emitted");
        // The Petri native producer now binds replay/model artifact descriptors to
        // the proof-handoff report (so `trust_mc_chc_proof_handoff_replay_artifact_*`
        // and `trust_mc_chc_proof_handoff_model_artifact_*` carry real names/kinds/digests).
        // Solver-side bytes and model-validation-readiness artifacts remain `none`
        // because the solver has not yet produced a model artifact / no byte
        // attachments are supplied to the bundle.
        let replay_artifact_name = evidence_field(
            ay_model_acceptance,
            "trust_mc_chc_proof_handoff_replay_artifact_name",
        )
        .expect("replay artifact name should be emitted");
        let replay_artifact_kind_code = evidence_field(
            ay_model_acceptance,
            "trust_mc_chc_proof_handoff_replay_artifact_kind_code",
        )
        .expect("replay artifact kind should be emitted");
        let replay_artifact_digest = evidence_field(
            ay_model_acceptance,
            "trust_mc_chc_proof_handoff_replay_artifact_digest",
        )
        .expect("replay artifact digest should be emitted");
        let model_artifact_name = evidence_field(
            ay_model_acceptance,
            "trust_mc_chc_proof_handoff_model_artifact_name",
        )
        .expect("model artifact name should be emitted");
        let model_artifact_kind_code = evidence_field(
            ay_model_acceptance,
            "trust_mc_chc_proof_handoff_model_artifact_kind_code",
        )
        .expect("model artifact kind should be emitted");
        let model_artifact_digest = evidence_field(
            ay_model_acceptance,
            "trust_mc_chc_proof_handoff_model_artifact_digest",
        )
        .expect("model artifact digest should be emitted");
        assert!(
            replay_artifact_name == "none" || !replay_artifact_name.is_empty(),
            "replay artifact name should be `none` or non-empty: {ay_model_acceptance}"
        );
        assert!(
            replay_artifact_kind_code == "none" || replay_artifact_kind_code == "replay_transcript",
            "replay artifact kind should be `none` or `replay_transcript`: {ay_model_acceptance}"
        );
        assert!(
            replay_artifact_digest == "none" || replay_artifact_digest.starts_with("sha256:"),
            "replay artifact digest should be `none` or `sha256:...`: {ay_model_acceptance}"
        );
        assert!(
            model_artifact_name == "none" || !model_artifact_name.is_empty(),
            "model artifact name should be `none` or non-empty: {ay_model_acceptance}"
        );
        assert!(
            model_artifact_kind_code == "none" || model_artifact_kind_code == "trust_mc_model",
            "model artifact kind should be `none` or `trust_mc_model`: {ay_model_acceptance}"
        );
        assert!(
            model_artifact_digest == "none" || model_artifact_digest.starts_with("sha256:"),
            "model artifact digest should be `none` or `sha256:...`: {ay_model_acceptance}"
        );
        assert!(
            ay_model_acceptance.contains("solver_artifact_bytes_validated=false")
                && ay_model_acceptance.contains("solver_model_artifact_bytes_digest=none")
                && ay_model_acceptance
                    .contains("solver_replay_transcript_artifact_bytes_digest=none")
                && ay_model_acceptance.contains(
                    "trust_mc_chc_proof_handoff_schema=trust_ir.native.petri_successor.trust_mc_chc_proof_handoff.v1"
                )
                && ay_model_acceptance.contains("trust_mc_chc_proof_handoff_schema_version=1")
                && ay_model_acceptance.contains("trust_mc_chc_proof_handoff_fail_closed=true")
                && ay_model_acceptance.contains(
                    "trust_mc_chc_model_validation_schema=trust_ir.native.petri_successor.trust_mc_chc_model_validation_readiness.v1"
                )
                && ay_model_acceptance.contains("trust_mc_chc_model_validation_schema_version=1")
                && ay_model_acceptance.contains("trust_mc_chc_model_validation_fail_closed=true")
                && ay_model_acceptance
                    .contains("trust_mc_chc_model_validation_model_artifact_name=none")
                && ay_model_acceptance
                    .contains("trust_mc_chc_model_validation_model_artifact_kind_code=none")
                && ay_model_acceptance
                    .contains("trust_mc_chc_model_validation_model_artifact_digest=none")
                && ay_model_acceptance.contains("accepted_for_consumer=false")
                && ay_model_acceptance.contains("production_selected=false"),
            "AY model acceptance row should expose typed proof/replay bridge blockers from AY/trust-ir and remain fail-closed: {ay_model_acceptance}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn ay_model_acceptance_emits_trust_ir_proof_evidence_identity_component_rows() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );

        let model_acceptance = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance evidence should be emitted");
        let readiness = trust_ir_component_readiness_row(
            &report,
            TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_COMPONENT,
        );
        let manifest_rows = trust_ir_component_manifest_rows(
            &report,
            TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_COMPONENT,
        );
        let replay_health_rows = trust_ir_component_manifest_rows(
            &report,
            TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_REPLAY_HEALTH_COMPONENT,
        );
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let trust_ir_identity = downstream_contract.trust_ir_native_bundle_identity;
        let identity_schema_version =
            trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA_VERSION
                .to_string();
        let row_count = evidence_field_usize(readiness, "identity_row_count");
        let replay_health_row_count =
            evidence_field_usize(readiness, "identity_replay_component_health_row_count");

        assert_eq!(manifest_rows.len(), row_count);
        assert_eq!(replay_health_rows.len(), replay_health_row_count);
        assert!(replay_health_row_count > 0);
        assert_eq!(
            evidence_field(readiness, "schema"),
            Some(trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA)
        );
        assert_eq!(
            evidence_field(readiness, "schema_version"),
            Some(identity_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(readiness, "identity_schema"),
            Some(trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA)
        );
        assert_eq!(
            evidence_field(readiness, "identity_digest"),
            evidence_field(model_acceptance, "trust_ir_proof_evidence_identity_digest")
        );
        assert_eq!(
            evidence_field(readiness, "producer_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(readiness, "replay_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReportForKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_component_health_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayComponentHealthSummaryKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_component_health_api"
            ),
            evidence_field(readiness, "identity_replay_component_health_api")
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_component_health_text_available"),
            Some("true")
        );
        assert_eq!(
            evidence_field(readiness, "producer_status_code"),
            evidence_field(model_acceptance, "trust_mc_chc_proof_handoff_status_code")
        );
        assert_eq!(
            evidence_field(readiness, "producer_reason_code"),
            evidence_field(model_acceptance, "trust_mc_chc_proof_handoff_reason_code")
        );
        assert_eq!(
            evidence_field(readiness, "producer_fail_closed"),
            evidence_field(model_acceptance, "trust_mc_chc_proof_handoff_fail_closed")
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_status_code"),
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_status_code"
            )
        );
        assert_eq!(
            evidence_field(readiness, "identity_replayable"),
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replayable"
            )
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_fail_closed"),
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_fail_closed"
            )
        );
        assert_eq!(
            evidence_field(readiness, "identity_replay_diagnostic_count"),
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_diagnostic_count"
            )
        );
        assert_eq!(
            evidence_field(readiness, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(readiness, "fail_closed"), Some("true"));
        assert!(manifest_rows.iter().any(|row| {
            evidence_field(row, "row_key") == Some("proof_evidence_identity.digest")
                && evidence_field(row, "row_value") == evidence_field(readiness, "identity_digest")
        }));
        assert!(manifest_rows.iter().any(|row| {
            evidence_field(row, "row_key") == Some("proof_handoff.reason")
                && evidence_field(row, "row_value")
                    == evidence_field(readiness, "producer_reason_code")
        }));
        assert!(replay_health_rows.iter().any(|row| {
            evidence_field(row, "row_key")
                == Some("proof_evidence_identity_replay_component_summary.status")
                && evidence_field(row, "row_value")
                    == evidence_field(readiness, "identity_replay_status_code")
        }));
        assert!(replay_health_rows.iter().any(|row| {
            evidence_field(row, "row_key")
                == Some("proof_evidence_identity_replay_component_summary.diagnostic_count")
                && evidence_field(row, "row_value")
                    == evidence_field(readiness, "identity_replay_diagnostic_count")
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn execution_plan_routes_represented_chc_rejection_as_advisory_before_parity_gate() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let cache = PetriKernelPlanCache::for_net(&net).expect("fixture plan cache should build");
        let installed_artifact = match native::petri_native_successor_installed_artifact(&bundle) {
            native::PetriNativeInstalledArtifactProduction::Available(artifact) => {
                PetriNativeInstalledArtifactEvidence::Available(artifact)
            }
            native::PetriNativeInstalledArtifactProduction::Blocked(blocker) => {
                panic!("fixture installed artifact should be available: {blocker:?}")
            }
        };
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_trust_cg_native_execution_plan_blocker_for_bundle(
            &mut report,
            &bundle,
            &bundle,
            "represented_trust_mc_fixture",
            true,
            petri_native_successor_state_bytes(
                u32::try_from(net.num_places()).expect("fixture place count should fit ABI"),
            ),
            &cache,
            installed_artifact.as_ref(),
            NativeJitFailClosedGate::from_env(),
        );

        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        assert!(
            execution_plan.contains("bundle_source=represented_trust_mc_fixture")
                && execution_plan.contains("bundle_validated=true")
                && execution_plan.contains("status_code=rejected")
                && execution_plan.contains("rejection_code=missing_native_install_gate_packet")
                && execution_plan.contains("reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("compile_artifact_handoff_ready=true")
                && execution_plan.contains("callable_contract_available=false")
                && execution_plan.contains("trampoline_contract_available=false")
                && execution_plan.contains("install_packet_available=false")
                && execution_plan.contains("install_packet_status_code=missing")
                && execution_plan.contains("call_packet_available=false")
                && execution_plan
                    .contains("call_packet_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("callable_pointer_available=false")
                && execution_plan.contains("runtime_readiness_status_code=blocked")
                && execution_plan.contains("runtime_readiness_ready_for_runtime_call=false")
                && execution_plan
                    .contains("runtime_readiness_reason_code=missing_native_install_gate_packet")
                && execution_plan
                    .contains("runtime_readiness_blocker_code=missing_native_install_gate_packet")
                && execution_plan.contains("execution_authority_authorized_for_execution=false")
                && execution_plan.contains(
                    "production_selection_is_selected_for_native_execution=false"
                )
                && execution_plan.contains("callable_authorized=false")
                && execution_plan
                    .contains("callable_authorized_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("plan_fail_closed=true")
                && execution_plan.contains("actions_expose_callable=false")
                && execution_plan.contains("ay_native_bundle_facade_status_code=blocked")
                && execution_plan.contains(
                    "ay_native_bundle_facade_reason_code=chc_problem_lowering_unavailable"
                )
                && execution_plan.contains(
                    "ay_native_bundle_facade_consumer_rejection_code=chc_problem_lowering_unavailable"
                )
                && execution_plan.contains("ay_native_bundle_facade_ready_for_trust_mc_chc_handoff=false")
                && execution_plan.contains(
                    "ay_native_bundle_facade_semantic_bridge_status_code=blocked"
                )
                && execution_plan.contains(
                    "ay_native_bundle_facade_semantic_bridge_evidence_status_code=missing"
                )
                && execution_plan
                    .contains("ay_native_bundle_facade_accepted_for_native_production=false")
                && execution_plan.contains(
                    "ay_model_acceptance_api=ay::chc::trust_mc_petri_successor_chc_model_acceptance_report"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_consumer_acceptance_api=ay::chc::TrustMcPetriSuccessorChcModelAcceptanceReport::accept_for_consumer"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_schema=ay.chc.trust_mc_petri_successor_model_acceptance.v1"
                )
                && execution_plan.contains("ay_model_acceptance_status_code=rejected")
                && execution_plan.contains("ay_model_acceptance_reason_code=proof_handoff_blocked")
                && execution_plan.contains(
                    "ay_model_acceptance_consumer_rejection_reason_code=proof_handoff_blocked"
                )
                && execution_plan
                    .contains("ay_model_acceptance_accepted_for_consumer=false")
                && execution_plan.contains("ay_model_acceptance_fail_closed=true")
                && execution_plan
                    .contains("ay_model_acceptance_proof_handoff_ready=false")
                && execution_plan
                    .contains("ay_model_acceptance_ready_for_solver_validation=false")
                && execution_plan.contains(
                    "ay_model_acceptance_trust_mc_chc_proof_handoff_reason_code=binding_blocked"
                )
                && execution_plan
                    .contains("ay_model_acceptance_solver_artifact_bytes_validated=false")
                && execution_plan.contains(
                    "ay_model_acceptance_solver_replay_transcript_artifact_bytes_digest=none"
                )
                && execution_plan
                    .contains("ay_model_acceptance_trust_ir_artifact_byte_attachment_count=0")
                && execution_plan.contains(
                    "ay_model_acceptance_trust_ir_artifact_byte_resolution_status_codes=blocked|blocked|blocked"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_ir_artifact_byte_resolution_reason_codes=missing_attachment|missing_attachment|missing_attachment"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_ir_artifact_byte_resolution_authority_codes=informational|informational|informational"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_ir_authoritative_artifact_requirement_count=0"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_ir_unauthoritative_artifact_requirement_roles=solver_input|replay_transcript|solver_witness"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_ir_authoritative_artifact_bytes_count=0"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_mc_chc_proof_handoff_schema=trust_ir.native.petri_successor.trust_mc_chc_proof_handoff.v1"
                )
                && execution_plan
                    .contains("ay_model_acceptance_trust_mc_chc_proof_handoff_fail_closed=true")
                && execution_plan.contains(
                    "ay_model_acceptance_trust_mc_chc_proof_handoff_replay_artifact_kind_code=none"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_mc_chc_proof_handoff_model_artifact_kind_code=none"
                )
                && execution_plan.contains(
                    "ay_model_acceptance_trust_mc_chc_model_validation_schema=trust_ir.native.petri_successor.trust_mc_chc_model_validation_readiness.v1"
                )
                && execution_plan
                    .contains("ay_model_acceptance_trust_mc_chc_model_validation_fail_closed=true")
                && execution_plan.contains(
                    "ay_model_acceptance_trust_mc_chc_model_validation_model_artifact_kind_code=none"
                )
                && execution_plan
                    .contains("native_successor_next_production_source=semantic_successor_bridge")
                && execution_plan.contains(
                    "native_successor_next_production_api=trust-cg::petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle"
                )
                && execution_plan.contains(
                    "native_successor_next_production_input=ty.petri.native.successor.plan_cache_equivalence.v1"
                )
                && execution_plan.contains(
                    "native_successor_next_production_reason_code=missing_semantic_successor_obligation"
                )
                && execution_plan.contains(
                    "native_successor_next_production_status_code=blocked"
                )
                && execution_plan.contains(
                    "native_successor_next_production_blocker_code=missing_semantic_successor_obligation"
                )
                && execution_plan.contains("production_selected=false")
                && execution_plan.contains("fail_closed=true"),
            "unadmitted semantic proof must block before the parity gate: {execution_plan}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_successor_capability_report_consumes_trust_ir_transport_identity_bundle() {
        let net = all_transition_net();
        let bundle = native_verification_bundle_fixture(&net);
        let expected_identity = bundle.transport_identity();
        let expected_transport_digest = expected_identity.stable_digest();
        let expected_source_digest = expected_identity
            .source_digest
            .map_or_else(|| "none".to_owned(), |digest| digest.to_string());
        let expected_target_abi_digest = expected_identity
            .target_abi
            .as_ref()
            .expect("fixture should carry a target ABI identity")
            .digest;

        let report =
            petri_native_successor_capability_report_with_verification_bundle(&net, Some(&bundle));

        assert!(report.selected.is_empty());
        let transport_identity = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit trust_ir_transport_identity"))
            .expect("native JIT transport identity evidence should be emitted");
        assert!(
            transport_identity.contains("Petri native_jit trust_ir_transport_identity available")
                && transport_identity.contains("cargo_dependency=true")
                && transport_identity.contains("api=NativeVerificationBundle::transport_identity")
                && transport_identity.contains(&format!("schema={}", expected_identity.schema))
                && transport_identity.contains(&format!(
                    "schema_version={}",
                    expected_identity.schema_version
                ))
                && transport_identity.contains(&format!(
                    "bundle_schema_version={}",
                    expected_identity.bundle_schema_version
                ))
                && transport_identity
                    .contains(&format!("transport_digest={expected_transport_digest}"))
                && transport_identity.contains(&format!("source_digest={expected_source_digest}"))
                && transport_identity.contains(&format!(
                    "module_digest={}",
                    expected_identity.trust_ir_module_digest
                ))
                && transport_identity.contains(&format!(
                    "bundle_digest={}",
                    expected_identity.bundle_digest
                ))
                && transport_identity
                    .contains(&format!("target_abi_digest={expected_target_abi_digest}"))
                && transport_identity.contains("request_digests=1")
                && transport_identity.contains("evidence_digests=1")
                && transport_identity.contains("production_selected=false")
                && transport_identity.contains("fail_closed=true"),
            "typed trust-ir transport identity should be rendered from NativeVerificationBundle::transport_identity(): {transport_identity}"
        );
        let producer_contract = report
            .evidence
            .iter()
            .find(|evidence| {
                evidence.contains("Petri native_jit trust_ir_transport_identity_producer_contract")
            })
            .expect("native JIT transport identity producer contract evidence should be emitted");
        assert!(
            producer_contract.contains("status_code=available")
                && producer_contract.contains("reason_code=available")
                && producer_contract.contains("bundle_source=external_supplied")
                && producer_contract.contains("bundle_validated=false")
                && producer_contract.contains("producer=trust_ir")
                && producer_contract.contains("input=trust_ir_module")
                && producer_contract.contains("transport_identity_available=true")
                && producer_contract.contains(&format!(
                    "module_digest={}",
                    expected_identity.trust_ir_module_digest
                ))
                && producer_contract.contains(&format!(
                    "transport_digest={expected_transport_digest}"
                ))
                && producer_contract.contains(&format!("consumer_api={TRUST_CG_PETRI_NATIVE_ADMISSION_API}"))
                && producer_contract.contains("native_promotion_authorized=false")
                && producer_contract.contains("production_selected=false")
                && producer_contract.contains("fail_closed=true"),
            "external bundle should expose the same producer contract without promotion: {producer_contract}"
        );
        let semantic_bridge = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit semantic_successor_bridge"))
            .expect("native JIT semantic successor bridge evidence should be emitted");
        assert!(
            semantic_bridge.contains("bundle_source=external_supplied")
                && semantic_bridge.contains("bundle_validated=false")
                && semantic_bridge.contains(&format!(
                    "transport_digest={expected_transport_digest}"
                ))
                && semantic_bridge.contains(&format!(
                    "trust_ir_module_digest={}",
                    expected_identity.trust_ir_module_digest
                ))
                && semantic_bridge.contains(&format!(
                    "bundle_digest={}",
                    expected_identity.bundle_digest
                ))
                && semantic_bridge.contains("successor_relation_represented=false")
                && semantic_bridge.contains("semantic_successor_authority=false")
                && semantic_bridge.contains("semantic_bridge_status_code=blocked")
                && semantic_bridge.contains("reason_code=missing_semantic_successor_obligation")
                && semantic_bridge.contains("trust_cg_status_code=blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_status_code=blocked")
                && semantic_bridge.contains("trust_ir_semantic_bridge_evidence_status=missing")
                && semantic_bridge.contains("trust_ir_semantic_bridge_fail_closed=true")
                && semantic_bridge.contains("production_selected=false")
                && semantic_bridge.contains("fail_closed=true"),
            "external bundle should expose the semantic successor bridge without native promotion: {semantic_bridge}"
        );
        let ay_facade = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("AY trust_mc_native_verification_bundle_facade"))
            .expect("AY native bundle facade evidence should be emitted");
        assert!(
            ay_facade.contains("schema=ay.chc.trust_mc_native_verification_bundle_facade.v2")
                && ay_facade.contains("schema_version=2")
                && ay_facade.contains("api=ay_trust_mc_native_bundle::solve_trust_mc_petri_successor_native_verification_bundle")
                && ay_facade.contains("bundle_source=external_supplied")
                && ay_facade.contains("bundle_validated=false")
                && ay_facade.contains(&format!(
                    "transport_digest={expected_transport_digest}"
                ))
                && ay_facade.contains(&format!(
                    "bundle_digest={}",
                    expected_identity.bundle_digest
                ))
                && ay_facade.contains("status_code=blocked")
                && ay_facade.contains("reason_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("consumer_acceptance_api=ay_trust_mc_native_bundle::trust_mcNativeVerificationBundleReport::accept_for_consumer")
                && ay_facade.contains("consumer_rejection_status_code=blocked")
                && ay_facade.contains("consumer_rejection_reason_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("consumer_rejection_code=chc_problem_lowering_unavailable")
                && ay_facade.contains("accepted_for_consumer=false")
                && ay_facade.contains("fail_closed=true")
                && ay_facade.contains("consumer_rejection_fail_closed=true")
                && ay_facade.contains("consumer_rejection_ready_for_trust_mc_chc_handoff=false")
                && ay_facade.contains("model_validated=false")
                && ay_facade.contains("verification_level_code=typed_handoff")
                && ay_facade.contains("proof_replay_status_code=blocked")
                && ay_facade.contains("ready_for_trust_mc_chc_handoff=false")
                && ay_facade.contains("trust_mc_request_count=0")
                && ay_facade.contains("trust_mc_evidence_count=0")
                && ay_facade.contains("native_evidence_entry_count=1")
                && ay_facade.contains("matched_trust_mc_request_count=0")
                && ay_facade.contains("matched_trust_mc_chc_request_count=0")
                && ay_facade.contains("matched_trust_mc_evidence_count=0")
                && ay_facade.contains("matched_trust_mc_artifact_count=0")
                && ay_facade.contains("matched_trust_mc_artifact_kind_codes=none")
                && ay_facade.contains("matched_trust_mc_request_ids=none")
                && ay_facade.contains("matched_trust_mc_request_mode_codes=none")
                && ay_facade.contains("semantic_bridge_status_code=blocked")
                && ay_facade.contains("semantic_bridge_reason_code=trusted_proof_not_admitted")
                && ay_facade.contains("semantic_bridge_evidence_status_code=missing")
                && ay_facade.contains("semantic_bridge_fail_closed=true")
                && ay_facade.contains("semantic_bridge_proof_status_code=discharged")
                && ay_facade.contains("production_selected=false"),
            "external bundle must preserve unadmitted semantic proof evidence as fail-closed: {ay_facade}"
        );
        let admission_blocker = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg trust_cg_admission_blocker"))
            .expect("native JIT admission blocker evidence should be emitted");
        assert!(
            admission_blocker.contains("bundle_source=external_supplied")
                && admission_blocker.contains("bundle_validated=false")
                && admission_blocker.contains("trust_ir_bundle_consumed=true")
                && admission_blocker.contains("trust_ir_consumption_status=available")
                && admission_blocker.contains("native_evidence_semantic_proof_artifacts=3")
                && admission_blocker.contains("native_evidence_native_execution_artifacts=0")
                && admission_blocker.contains("native_evidence_semantic_proof_available=true")
                && admission_blocker.contains("native_evidence_native_execution_artifact_available=false")
                && admission_blocker.contains("rejection_code=missing_native_install_gate_packet")
                && admission_blocker.contains("reason_code=missing_native_install_gate_packet")
                && admission_blocker.contains("request_digests=1")
                && admission_blocker.contains("evidence_digests=1")
                && admission_blocker.contains("production_selected=false")
                && admission_blocker.contains("fail_closed=true"),
            "external supplied bundle should be typed-consumed by trust-codegen admission without native promotion: {admission_blocker}"
        );
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        assert!(
            execution_plan.contains("source=PetriNativeSuccessorExecutionPlan")
                && execution_plan.contains("bundle_source=external_supplied")
                && execution_plan.contains("bundle_validated=false")
                && execution_plan.contains("trust_ir_bundle_consumed=true")
                && execution_plan.contains("trust_ir_consumption_status=available")
                && execution_plan.contains("native_evidence_semantic_proof_artifacts=3")
                && execution_plan.contains("native_evidence_native_execution_artifacts=0")
                && execution_plan.contains("native_evidence_semantic_proof_available=true")
                && execution_plan.contains("native_evidence_native_execution_artifact_available=false")
                && execution_plan.contains("status_code=rejected")
                && execution_plan.contains("rejection_code=missing_native_install_gate_packet")
                && execution_plan.contains("reason_code=missing_native_install_gate_packet")
                && execution_plan.contains(
                    "execution_plan_api=trust-cg::petri_native_successor_execution_plan_from_trust_ir_bundle"
                )
                && execution_plan.contains(
                    "expected_api=PetriNativeSuccessorExecutionExpected::canary_callable"
                )
                && execution_plan.contains(
                    "trampoline_contract_api=trust-cg::petri_native_successor_trampoline_contract"
                )
                && execution_plan.contains(
                    "install_packet_api=trust-cg::petri_native_successor_install_packet_from_trust_ir_bundle"
                )
                && execution_plan.contains(&format!(
                    "entry_function={}",
                    PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL
                ))
                && execution_plan.contains("input_state_bytes=24")
                && execution_plan.contains("output_state_bytes=24")
                && execution_plan.contains("execution_plan_available=true")
                && execution_plan.contains("execution_plan_status_code=available")
                && execution_plan.contains("execution_plan_reason_code=available")
                && execution_plan.contains("callable_contract_available=false")
                && execution_plan.contains("trampoline_contract_available=false")
                && execution_plan.contains("install_packet_available=false")
                && execution_plan.contains("install_packet_status_code=missing")
                && execution_plan
                    .contains("install_packet_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains(
                    "downstream_contract_api=trust-cg::petri_native_successor_downstream_contract_descriptor"
                )
                && execution_plan.contains(
                    "downstream_contract_schema=trust-cg.petri.native_successor.downstream_contract.v1"
                )
                && execution_plan.contains(
                    "downstream_trust_ir_bundle_identity_schema=trust_ir.native.bundle_identity_contract.v1"
                )
                && execution_plan.contains(&format!(
                    "downstream_trust_ir_transport_identity_schema={}",
                    trust_ir::NATIVE_TRANSPORT_IDENTITY_SCHEMA
                ))
                && execution_plan.contains(
                    "downstream_runtime_readiness_required_fields=call_packet,native_install_gate_packet,trampoline_contract,callable_lifetime_proof,runtime_abi_proof,current_generation"
                )
                && execution_plan.contains(
                    "runtime_readiness_api=trust-cg::petri_native_successor_runtime_readiness_packet"
                )
                && execution_plan.contains(
                    "runtime_readiness_installed_artifact_api=InstalledArtifact::petri_native_successor_runtime_readiness_packet"
                )
                && execution_plan
                    .contains("runtime_readiness_installed_artifact_required_trust_cg_rev=690f04d7")
                && execution_plan.contains(
                    "runtime_readiness_source=trust-cg::petri_native_successor_runtime_readiness_packet"
                )
                && execution_plan.contains("runtime_readiness_installed_artifact_available=false")
                && execution_plan
                    .contains("runtime_readiness_schema=trust-cg.petri.native_successor.runtime_readiness_packet.v1")
                && execution_plan.contains("runtime_readiness_packet_available=true")
                && execution_plan.contains("runtime_readiness_status_code=blocked")
                && execution_plan
                    .contains("runtime_readiness_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("runtime_readiness_status_in_downstream_contract=true")
                && execution_plan
                    .contains("runtime_readiness_blocker_code=missing_native_install_gate_packet")
                && execution_plan.contains("runtime_readiness_blocker_in_downstream_contract=true")
                && execution_plan.contains("runtime_readiness_blocker_stage=manifest_identity")
                && execution_plan.contains("runtime_readiness_ready_for_runtime_call=false")
                && execution_plan.contains("mock_executable_call_role=test_diagnostic_only")
                && execution_plan.contains("mock_executable_call_production_enabled=false")
                && execution_plan.contains(
                    "call_packet_api=trust-cg::petri_native_successor_call_packet_from_trust_ir_bundle"
                )
                && execution_plan
                    .contains("call_packet_schema=trust-cg.petri.native_successor.call_packet.v1")
                && execution_plan.contains("call_packet_type=PetriNativeSuccessorCallPacket")
                && execution_plan
                    .contains("callable_pointer_type=PetriNativeSuccessorCallablePointer")
                && execution_plan.contains("call_packet_required_trust_cg_rev=2d31fd8b")
                && execution_plan.contains(&format!(
                    "call_packet_current_trust_cg_rev={}",
                    TRUST_CG_PETRI_NATIVE_CALL_PACKET_CURRENT_TRUST_CG_REV
                ))
                && execution_plan.contains("call_packet_api_available=true")
                && execution_plan.contains("call_packet_api_status_code=available")
                && execution_plan.contains("call_packet_type_available=true")
                && execution_plan.contains("callable_pointer_type_available=true")
                && execution_plan.contains(
                    "call_packet_available=false call_packet_reason_code=missing_native_install_gate_packet"
                )
                && execution_plan.contains(
                    "callable_pointer_available=false callable_pointer_reason_code=missing_native_install_gate_packet"
                )
                && execution_plan.contains("concrete_callable_pointer_required=true")
                && execution_plan.contains("concrete_callable_pointer_available=false")
                && execution_plan.contains("concrete_callable_pointer_status_code=missing")
                && execution_plan.contains("concrete_callable_packet_required=true")
                && execution_plan.contains("concrete_callable_packet_available=false")
                && execution_plan.contains("concrete_callable_packet_status_code=missing")
                && execution_plan.contains("call_packet_readiness_status_code=blocked")
                && execution_plan
                    .contains("call_packet_readiness_blocker=missing_native_install_gate_packet")
                && execution_plan.contains("callable_authorized=false")
                && execution_plan
                    .contains("callable_authorized_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("callable_handoff_available=true")
                && execution_plan.contains("callable_handoff_reason_code=available")
                && execution_plan
                    .contains("callable_handoff_blocker=missing_native_install_gate_packet")
                && execution_plan
                    .contains("callable_handoff_required_evidence=trust-cg.phase6.native_install_gate.v1")
                && execution_plan.contains(
                    "callable_handoff_upstream_ask=provide_runtime_callable_pointer_and_accepted_install_packet"
                )
                && execution_plan.contains("plan_fail_closed=true")
                && execution_plan.contains("actions_expose_callable=false")
                && execution_plan.contains("actions_expose_callable_blocked_by_runtime_readiness=true")
                && execution_plan
                    .contains("actions_expose_callable_reason_code=missing_native_install_gate_packet")
                && execution_plan
                    .contains("actions_ty_native_activate_blocked_by_runtime_readiness=true")
                && execution_plan
                    .contains("actions_ty_native_activate_reason_code=missing_native_install_gate_packet")
                && execution_plan.contains("production_selected=false")
                && execution_plan.contains("fail_closed=true")
                && execution_plan.contains("native_successor_runtime_status_code=blocked"),
            "external supplied bundle should expose the shared trust-codegen execution plan and remain fail-closed: {execution_plan}"
        );
        assert_execution_plan_exposes_compile_artifact_handoff(execution_plan, false);

        let successor = report
            .rejected
            .iter()
            .find(|capability| capability.problem == Some(ProblemKind::NativeSuccessor))
            .expect("successor native capability should remain rejected");
        assert_eq!(successor.status, tla_mc_core::CapabilityStatus::Disabled);
        assert_eq!(successor.reason_code(), Some("disabled_by_policy"));
        assert!(successor.detail.as_deref().is_some_and(|detail| {
            detail.contains("production_selected=false")
                && detail.contains("trust_ir_transport_identity_available=true")
        }));
    }

    // Production native promotion needs the verification bundle (and
    // therefore the `trust-cg-petri-native` feature); without it the gate stays
    // blocked on `missing_trust_ir_transport_identity`. Gate the test to match
    // production semantics.
    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_jit_env_requests_select_promoted_callable_backend() {
        let _lock = native_jit_env_lock();
        let _native = EnvVarGuard::set(ENABLE_NATIVE_CANDIDATE_ENV, "1");
        let _strict = EnvVarGuard::set(ENABLE_NATIVE_CANDIDATE_STRICT_ENV, "true");
        let _parity = EnvVarGuard::set(ENABLE_TRANSITION_PARITY_ENV, "yes");
        let net = all_transition_net();

        let report = petri_native_successor_capability_report(&net);

        let gate = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit fail_closed_gate"))
            .expect("native JIT fail-closed gate evidence should be emitted");
        assert!(
            gate.contains("native_requested=true"),
            "native env request should be visible in evidence: {gate}"
        );
        assert!(
            gate.contains("strict_requested=true"),
            "strict native env request should be visible in evidence: {gate}"
        );
        assert!(
            gate.contains("parity_enabled=true"),
            "parity env request should be visible in evidence: {gate}"
        );
        assert!(
            gate.contains("parity_receipt_required=true")
                && gate.contains("parity_receipt_available=true")
                && gate.contains("parity_receipt_status_code=accepted")
                && gate.contains("parity_receipt_reason_code=available")
                && gate
                    .contains("parity_receipt_schema=ty.petri.native_successor.parity_receipt.v1")
                && gate.contains(
                    "parity_receipt_gate_api=tla_petri::petri_native_successor_parity_receipt_gate"
                ),
            "parity env request should carry the accepted native parity receipt: {gate}"
        );
        assert!(
            gate.contains("validation_receipt_required=true")
                && gate.contains("validation_receipt_available=true")
                && gate.contains("validation_receipt_status_code=accepted")
                && gate.contains("validation_receipt_reason_code=available")
                && gate.contains("validation_receipt_schema=ty.shared.validation_receipt.v1")
                && gate.contains(
                    "validation_receipt_gate_api=tla_mc_core::validate_validation_receipt_evidence_row"
                ),
            "native env request should carry the accepted semantic validation receipt: {gate}"
        );
        assert!(
            gate.contains("callable_receipt_required=true")
                && gate.contains("callable_receipt_available=true")
                && gate.contains("callable_receipt_status_code=accepted")
                && gate.contains("callable_receipt_reason_code=available")
                && gate.contains(
                    "callable_receipt_gate_api=tla_petri::petri_native_successor_callable_receipt_gate"
                )
                && gate.contains("native_runtime_callable_impl_available=true")
                && gate.contains("runtime_readiness_status_code=accepted")
                && gate.contains("runtime_readiness_reason_code=none"),
            "native env request should carry install-gate-backed callable and runtime readiness evidence: {gate}"
        );
        assert!(
            gate.contains("production_gate_status=selected")
                && gate.contains("production_selected=true")
                && gate.contains("fail_closed=false")
                && gate.contains("reason_code=none"),
            "native candidate should be selected in evidence: {gate}"
        );
        assert!(
            report.selected.iter().any(|capability| capability.problem
                == Some(ProblemKind::NativeSuccessor)
                && capability.status == tla_mc_core::CapabilityStatus::Available),
            "native JIT env requests should select the production native successor backend"
        );
        let route_selection = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit route_selection"))
            .expect("native JIT route-selection evidence should be emitted");
        assert!(
            route_selection.contains("selected_lane=native_successor")
                && route_selection.contains("status_code=selected")
                && route_selection.contains("reason_code=none")
                && route_selection.contains("producer_production_selection=true")
                && route_selection.contains("callable_receipt_available=true")
                && route_selection.contains("production_gate_status=selected")
                && route_selection.contains("production_selected=true")
                && route_selection.contains("fail_closed=false")
                && route_selection.contains("mcc_petri:none")
                && route_selection.contains("blocker_status=mcc_petri-cleared")
                && route_selection.contains("blocker_issue_refs=none")
                && route_selection.contains("todo=none"),
            "route selection should expose native production selection: {route_selection}"
        );
        let transport_identity = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit trust_ir_transport_identity"))
            .expect("native JIT transport identity evidence should be emitted");
        assert!(
            transport_identity
                .contains("trust_ir_transport_identity available")
                && transport_identity
                    .contains("required_trust_ir_rev=222785e293636ac6c63b20525151aef2ccd586c1")
                && transport_identity.contains(&format!(
                    "current_trust_ir_rev={TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV}"
                ))
                && transport_identity
                    .contains("expected_fields=transport,source,module,bundle,target_abi_digest"),
            "transport identity evidence should remain present for the selected native route: {transport_identity}"
        );
        let execution_plan = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("trust-cg petri_native_successor_execution_plan"))
            .expect("native JIT execution-plan evidence should be emitted");
        assert!(
            execution_plan.contains("ay_native_bundle_facade_status_code=blocked")
                && execution_plan
                    .contains("ay_native_bundle_facade_accepted_for_native_production=false")
                && execution_plan.contains("production_selected=true")
                && execution_plan.contains("fail_closed=false"),
            "AY native bundle rejection should remain advisory after exact receipts and callable identity select production: {execution_plan}"
        );
        let successor = report
            .selected
            .iter()
            .find(|capability| capability.problem == Some(ProblemKind::NativeSuccessor))
            .expect("successor native capability should be selected");
        assert_eq!(successor.status, tla_mc_core::CapabilityStatus::Available);
        assert_eq!(successor.role, CapabilityRole::Production);
        assert!(successor.detail.as_deref().is_some_and(|detail| {
            detail.contains("native_requested=true")
                && detail.contains("strict_requested=true")
                && detail.contains("parity_enabled=true")
                && detail.contains("parity_receipt_required=true")
                && detail.contains("parity_receipt_available=true")
                && detail.contains("parity_receipt_reason_code=available")
                && detail.contains("validation_receipt_required=true")
                && detail.contains("validation_receipt_available=true")
                && detail.contains("validation_receipt_reason_code=available")
                && detail.contains("callable_receipt_required=true")
                && detail.contains("callable_receipt_available=true")
                && detail.contains("callable_receipt_status_code=accepted")
                && detail.contains("callable_receipt_reason_code=available")
                && detail.contains("native_runtime_callable_impl_available=true")
                && detail.contains("runtime_readiness_status_code=accepted")
                && detail.contains("runtime_readiness_reason_code=none")
                && detail.contains("production_selected=true")
                && detail.contains("fail_closed=false")
        }));
    }

    #[test]
    fn native_route_selection_requires_explicit_parity_receipt() {
        let route_selection = PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
            transport_identity_available: true,
            producer_admission: true,
            producer_execution_authority: true,
            producer_production_selection: true,
            parity_enabled: true,
            parity_receipt_available: false,
            validation_receipt_available: false,
            callable_receipt_available: false,
            native_runtime_callable_impl_available: true,
            producer_admission_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            producer_execution_authority_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            producer_production_selection_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            parity_receipt_reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT,
            validation_receipt_reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT,
            callable_receipt_reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
            runtime_readiness_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
        });

        assert!(!route_selection.selected_for_native_execution);
        assert!(route_selection.fail_closed);
        assert_eq!(
            route_selection.reason_code,
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT
        );
        assert_eq!(
            route_selection.selected_lane,
            PETRI_NATIVE_ROUTE_SELECTION_LANE_FALLBACK
        );
        assert!(!route_selection.parity_receipt_available);
        assert_eq!(
            route_selection.parity_receipt_reason_code.as_str(),
            PETRI_NATIVE_ROUTE_SELECTION_REASON_PARITY_RECEIPT
        );
        assert!(!route_selection.validation_receipt_available);
        assert_eq!(
            route_selection.validation_receipt_reason_code.as_str(),
            PETRI_NATIVE_ROUTE_SELECTION_REASON_VALIDATION_RECEIPT
        );
        assert!(!route_selection.callable_receipt_available);
        assert_eq!(
            route_selection.callable_receipt_reason_code.as_str(),
            PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT
        );

        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_petri_native_route_selection_evidence(&mut report, &route_selection);
        let row = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit route_selection"))
            .expect("route selection evidence should be emitted");
        assert!(row.contains("selected_lane=explicit_state"));
        assert!(row.contains("status_code=fail_closed"));
        assert!(row.contains("production_selected=false"));
        assert!(row.contains("fail_closed=true"));
        assert!(row.contains(
            "safe_class_criteria=producer_admission,producer_execution_authority,producer_production_selection,parity_enabled,parity_receipt_available,validation_receipt_available,callable_receipt_available,native_runtime_callable_impl"
        ));
        assert!(row.contains("parity_receipt_required=true"));
        assert!(row.contains("parity_receipt_available=false"));
        assert!(row.contains("parity_receipt_reason_code=missing_parity_receipt"));
        assert!(row.contains("validation_receipt_required=true"));
        assert!(row.contains("validation_receipt_available=false"));
        assert!(row.contains("validation_receipt_reason_code=missing_validation_receipt"));
        assert!(row.contains("validation_receipt_schema=ty.shared.validation_receipt.v1"));
        assert!(row.contains("callable_receipt_required=true"));
        assert!(row.contains("callable_receipt_available=false"));
        assert!(row.contains("callable_receipt_status_code=missing"));
        assert!(row.contains("callable_receipt_reason_code=missing_callable_receipt"));
        assert!(
            row.contains("callable_receipt_schema=ty.petri.native_successor.callable_receipt.v1")
        );
        assert!(row.contains(
            "shared_engine_component=tla_mc_core.prepared_checker_program,tla_ir.whole_program_kernel,trust_cg.batch_native_artifact_identity,tla_mc_core.validation_receipt"
        ));
        assert!(row.contains("origin_frontend=mcc_petri"));
        assert!(row.contains("shared_engine_owner=shared_high_performance_engine"));
        assert!(row.contains("adoption_level=level-0"));
        assert!(row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(row.contains("default_compatible_frontend_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"));
        assert!(row.contains("frontend_family_blockers=tla_plus:needs_state_vector_native_layout_manifest,quint:needs_source_identity_preserving_native_manifest,mcc_petri:missing_native_install_validation_parity_and_callable_receipts,aiger:needs_register_vector_native_layout_manifest,btor2:needs_bitvector_register_native_layout_manifest,vmt_transition_system:needs_transition_system_native_layout_manifest,ay_analytical:needs_native_helper_validation_receipt,witness_replay:needs_replay_validation_receipt_adapter,future_importer:awaiting_registered_importer_frontend"));
        assert!(row.contains("generic_prerequisites=prepared_checker_program_descriptor,marking_storage_identity,transition_relation_descriptor,state_predicate_descriptor,native_candidate_descriptor,validation_plan_descriptor,accepted_validation_receipt,accepted_parity_receipt,accepted_callable_receipt"));
        assert!(row.contains(
            "production_gate_status=blocked_missing_native_install_validation_parity_and_callable_receipts"
        ));
    }

    #[test]
    fn native_route_selection_requires_callable_receipt_after_validation_receipts() {
        let route_selection = PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
            transport_identity_available: true,
            producer_admission: true,
            producer_execution_authority: true,
            producer_production_selection: true,
            parity_enabled: true,
            parity_receipt_available: true,
            validation_receipt_available: true,
            callable_receipt_available: false,
            native_runtime_callable_impl_available: true,
            producer_admission_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            producer_execution_authority_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            producer_production_selection_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            parity_receipt_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            validation_receipt_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            callable_receipt_reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT,
            runtime_readiness_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
        });

        assert!(!route_selection.selected_for_native_execution);
        assert!(route_selection.fail_closed);
        assert_eq!(
            route_selection.reason_code,
            PETRI_NATIVE_ROUTE_SELECTION_REASON_CALLABLE_RECEIPT
        );
        assert_eq!(
            route_selection.selected_lane,
            PETRI_NATIVE_ROUTE_SELECTION_LANE_FALLBACK
        );

        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_petri_native_route_selection_evidence(&mut report, &route_selection);
        add_native_jit_fail_closed_gate_evidence(
            &mut report,
            NativeJitFailClosedGate {
                feature_enabled: true,
                native_requested: true,
                strict_requested: false,
                parity_enabled: true,
                parity_receipt_available: true,
                validation_receipt_available: true,
            },
            &route_selection,
        );

        let route_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit route_selection"))
            .expect("route selection evidence should be emitted");
        assert!(route_row.contains("reason_code=missing_callable_receipt"));
        assert!(route_row.contains("parity_receipt_available=true"));
        assert!(route_row.contains("validation_receipt_available=true"));
        assert!(route_row.contains("callable_receipt_required=true"));
        assert!(route_row.contains("callable_receipt_available=false"));
        assert!(route_row.contains("callable_receipt_status_code=missing"));
        assert!(route_row.contains("production_selected=false"));
        assert!(route_row.contains("fail_closed=true"));

        let gate_row = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit fail_closed_gate"))
            .expect("fail-closed gate evidence should be emitted");
        assert!(gate_row.contains("callable_receipt_required=true"));
        assert!(gate_row.contains("callable_receipt_available=false"));
        assert!(gate_row.contains("callable_receipt_status_code=missing"));
        assert!(gate_row.contains("callable_receipt_reason_code=missing_callable_receipt"));
        assert!(gate_row.contains("production_selected=false"));
        assert!(gate_row.contains("fail_closed=true"));
    }

    #[test]
    fn native_route_selection_requires_runtime_callable_impl_after_callable_receipt() {
        let route_selection = PetriNativeRouteSelection::evaluate(PetriNativeRouteSelectionInput {
            transport_identity_available: true,
            producer_admission: true,
            producer_execution_authority: true,
            producer_production_selection: true,
            parity_enabled: true,
            parity_receipt_available: true,
            validation_receipt_available: true,
            callable_receipt_available: true,
            native_runtime_callable_impl_available: false,
            producer_admission_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            producer_execution_authority_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            producer_production_selection_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            parity_receipt_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            validation_receipt_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            callable_receipt_reason_code: TRUST_CG_PETRI_NATIVE_REASON_AVAILABLE,
            runtime_readiness_reason_code: PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL,
        });

        assert!(!route_selection.selected_for_native_execution);
        assert!(route_selection.fail_closed);
        assert_eq!(
            route_selection.reason_code,
            PETRI_NATIVE_ROUTE_SELECTION_REASON_RUNTIME_IMPL
        );

        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_petri_native_route_selection_evidence(&mut report, &route_selection);
        let row = report
            .evidence
            .iter()
            .find(|evidence| evidence.contains("Petri native_jit route_selection"))
            .expect("route selection evidence should be emitted");
        assert!(row.contains("callable_receipt_available=true"));
        assert!(row.contains("callable_receipt_status_code=accepted"));
        assert!(row.contains("native_runtime_callable_impl_available=false"));
        assert!(row.contains("runtime_readiness_status_code=missing"));
        assert!(row.contains("runtime_readiness_reason_code=native_runtime_callable_impl_missing"));
        assert!(row.contains("reason_code=native_runtime_callable_impl_missing"));
        assert!(row.contains("production_selected=false"));
        assert!(row.contains("fail_closed=true"));
    }

    #[test]
    fn native_successor_capability_report_rejects_oversized_arc_weights() {
        let mut net = simple_net();
        net.transitions[0].inputs[0].weight = i64::MAX as u64 + 1;

        let report = petri_native_successor_capability_report(&net);

        assert_eq!(
            report.rejection_reason(BackendKind::NativeKernel),
            Some(&UnsupportedReason::TooLarge("arc weight exceeds i64"))
        );
        assert_eq!(
            report.rejection_reason_code(BackendKind::NativeKernel),
            Some("too_large")
        );
        assert_eq!(
            report.rejected[0].status,
            tla_mc_core::CapabilityStatus::Unsupported
        );
        assert_eq!(report.rejected[0].reason_code(), Some("too_large"));
        assert!(report.rejected[0]
            .detail
            .as_ref()
            .expect("rejection should include detail")
            .contains("ArcWeightExceedsI64"));
        assert!(report.evidence.iter().any(|evidence| evidence.contains(
            "Petri native_successor capability backend=NativeKernel problem=Some(NativeSuccessor) status=Unsupported role=Validation reason_code=too_large adoption=unavailable deferred=false"
        )));
    }

    #[test]
    fn native_status_errors_map_to_shared_reason_codes() {
        let unsupported = unsupported_reason_for_kernel_error(&PetriKernelError::NativeStatus {
            status: PetriNativeAllSuccessorsStatus::Unsupported,
            detail: "placeholder".to_string(),
        });
        assert_eq!(unsupported, UnsupportedReason::NativeKernelUnavailable);
        assert_eq!(unsupported.code(), "native_kernel_unavailable");

        let token_overflow = unsupported_reason_for_kernel_error(&PetriKernelError::NativeStatus {
            status: PetriNativeAllSuccessorsStatus::TokenOverflow,
            detail: "token overflow".to_string(),
        });
        assert_eq!(
            token_overflow,
            UnsupportedReason::TooLarge("native successor token arithmetic")
        );
        assert_eq!(token_overflow.code(), "too_large");
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_candidate_checked_path_executes_promoted_callable_artifact() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let kernel = PetriNativeAllTransitionKernel;
        let mut scratch = PetriKernelScratch::new();

        let candidate = native::petri_native_successor_batch_candidate(&net, &cache);
        let native::PetriNativeSuccessorBatchCandidate::CallableArtifact(batch) = candidate else {
            panic!("fixture should build a callable native artifact candidate: {candidate:?}");
        };
        assert_eq!(batch.readiness.status_code, "callable_artifact");
        assert_eq!(batch.readiness.reason_code, "available");
        assert_eq!(batch.readiness.blocker, "none");
        assert_eq!(batch.readiness.validation_receipt_status, "accepted");
        assert_eq!(batch.readiness.parity_receipt_status, "accepted");
        assert_eq!(batch.readiness.callable_receipt_status, "accepted");
        assert_eq!(batch.readiness.callable_receipt_reason_code, "available");
        assert_eq!(batch.readiness.native_missing_receipts, "none");
        assert_eq!(batch.readiness.production_gate_status, "selected");
        assert!(batch.readiness.production_selected);
        assert!(!batch.readiness.fail_closed);

        let count = checked_native_all_transition_successors_cached_into(
            &kernel,
            &cache,
            &net,
            &[2, 1, 0],
            &mut scratch,
            PetriNativeAllTransitionConfig { strict: true },
        )
        .unwrap();
        assert_eq!(count, 2);

        let disabled_count = checked_native_all_transition_successors_cached_into(
            &kernel,
            &cache,
            &net,
            &[0, 0, 0],
            &mut scratch,
            PetriNativeAllTransitionConfig { strict: true },
        )
        .unwrap();
        assert_eq!(disabled_count, 0);
    }

    /// End-to-end soundness check for the production install path: install the
    /// promoted native batch onto a `PetriNetSystem` and verify that exploring
    /// through `PetriNetSystem::successors` (the native fast-path) yields the
    /// byte-identical reachable set and the identical per-state successor sets
    /// as the scalar interpreter. This is the "native runs must give IDENTICAL
    /// state counts to the interpreter" guarantee, exercised on every marking in
    /// a depth-bounded frontier rather than a single fixture state. In debug
    /// builds the per-state parity `debug_assert` inside `successors` also fires
    /// on every native state visited below.
    ///
    /// The frontier is depth-bounded because `all_transition_net` is *unbounded*
    /// (transition `t2` produces two tokens for one consumed), so its reachable
    /// set is infinite; a fixed depth keeps the comparison finite while still
    /// covering many distinct markings and successor multiplicities.
    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_petrinet_system_successors_match_interpreter_over_frontier() {
        use crate::system::{CompactMarking, PetriNetSystem};
        use std::collections::{BTreeSet, VecDeque};
        use tla_mc_core::TransitionSystem;

        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();

        let candidate = native::petri_native_successor_batch_candidate(&net, &cache);
        let native::PetriNativeSuccessorBatchCandidate::CallableArtifact(batch) = candidate else {
            panic!("fixture should build a callable native artifact candidate: {candidate:?}");
        };
        assert!(
            batch.readiness.production_selected,
            "fixture native batch must be production-selected so the system installs it",
        );

        let interpreter = PetriNetSystem::new(net.clone());
        let native_system = PetriNetSystem::new(net.clone()).with_native_batch(batch);

        // Depth-bounded BFS over the interpreter to collect a frontier of
        // distinct reachable markings (the net is unbounded, so cap the depth).
        const MAX_DEPTH: usize = 6;
        let mut frontier: Vec<CompactMarking> = Vec::new();
        let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
        let mut queue: VecDeque<(CompactMarking, usize)> = VecDeque::new();
        for s in interpreter.initial_states() {
            if seen.insert(s.as_bytes().to_vec()) {
                frontier.push(s.clone());
                queue.push_back((s, 0));
            }
        }
        while let Some((state, depth)) = queue.pop_front() {
            if depth >= MAX_DEPTH {
                continue;
            }
            for (_t, next) in interpreter.successors(&state) {
                if seen.insert(next.as_bytes().to_vec()) {
                    frontier.push(next.clone());
                    queue.push_back((next, depth + 1));
                }
            }
        }

        assert!(
            frontier.len() >= 8,
            "frontier should cover a non-trivial set of markings, got {}",
            frontier.len(),
        );

        // For every marking in the frontier, the native fast-path must produce
        // an identical, identically-ordered (transition, successor) sequence as
        // the scalar interpreter. Empty successor sets (deadlocks/disabled) are
        // included and must also agree.
        let mut compared_with_successors = 0usize;
        for marking in &frontier {
            let native_succ = native_system.successors(marking);
            let interp_succ = interpreter.successors(marking);
            assert_eq!(
                native_succ.len(),
                interp_succ.len(),
                "native vs interpreter successor count mismatch at marking {:?}",
                interpreter.unpack_marking(marking),
            );
            for (idx, ((nt, nm), (it, im))) in
                native_succ.iter().zip(interp_succ.iter()).enumerate()
            {
                assert_eq!(
                    nt,
                    it,
                    "transition-id mismatch at marking {:?} row {idx}",
                    interpreter.unpack_marking(marking),
                );
                assert_eq!(
                    nm.as_bytes(),
                    im.as_bytes(),
                    "successor marking mismatch at marking {:?} row {idx} (transition {nt:?})",
                    interpreter.unpack_marking(marking),
                );
            }
            if !native_succ.is_empty() {
                compared_with_successors += 1;
            }
        }
        assert!(
            compared_with_successors >= 8,
            "expected many markings with native successors to compare, got {compared_with_successors}",
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_candidate_generated_successor_body_matches_checked_all_transition_fixture() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let mut scratch = PetriKernelScratch::new();

        let (out, native_successors) =
            native::unchecked_native_all_transition_successors_for_tests(
                &cache,
                &net,
                &[2, 1, 0],
                &mut scratch,
                cache.transition_count,
            )
            .unwrap();

        assert_eq!(
            out.status,
            tla_jit_abi::SuccessorKernelStatus::Ok,
            "native out={out:?} native_successors={native_successors:?}"
        );
        assert_eq!(
            out.successor_count, 2,
            "native out={out:?} native_successors={native_successors:?}"
        );
        assert_eq!(
            out.generated_count, 2,
            "native out={out:?} native_successors={native_successors:?}"
        );
        assert_eq!(out.state_len, 3);
        assert_eq!(out.overflow_count, 0);
        assert_eq!(
            out.runtime_error,
            tla_jit_abi::JitRuntimeErrorKind::DivisionByZero
        );
        assert_eq!(
            out.unsupported_reason,
            tla_jit_abi::SuccessorKernelUnsupportedReason::None
        );
        assert_eq!(out.metadata_bits, 0b101);
        assert_eq!(&native_successors[..6], &[1, 2, 0, 2, 0, 2]);

        let mut overflow_scratch = PetriKernelScratch::new();
        let (overflow, overflow_successors) =
            native::unchecked_native_all_transition_successors_for_tests(
                &cache,
                &net,
                &[2, 1, 0],
                &mut overflow_scratch,
                1,
            )
            .unwrap();

        assert_eq!(
            overflow.status,
            tla_jit_abi::SuccessorKernelStatus::BufferOverflow,
            "native overflow out={overflow:?} overflow_successors={overflow_successors:?}"
        );
        assert_eq!(
            overflow.successor_count, 1,
            "native overflow out={overflow:?} overflow_successors={overflow_successors:?}"
        );
        assert_eq!(
            overflow.generated_count, 2,
            "native overflow out={overflow:?} overflow_successors={overflow_successors:?}"
        );
        assert_eq!(
            overflow.state_len, 3,
            "native overflow out={overflow:?} overflow_successors={overflow_successors:?}"
        );
        assert_eq!(
            overflow.overflow_count, 1,
            "native overflow out={overflow:?} overflow_successors={overflow_successors:?}"
        );
        assert_eq!(
            overflow.runtime_error,
            tla_jit_abi::JitRuntimeErrorKind::DivisionByZero
        );
        assert_eq!(
            overflow.unsupported_reason,
            tla_jit_abi::SuccessorKernelUnsupportedReason::None
        );
        assert_eq!(overflow.metadata_bits, 0b101);
        assert_eq!(&overflow_successors[..3], &[1, 2, 0]);

        let mut zero_capacity_scratch = PetriKernelScratch::new();
        let (zero_capacity, zero_capacity_successors) =
            native::unchecked_native_all_transition_successors_for_tests(
                &cache,
                &net,
                &[2, 1, 0],
                &mut zero_capacity_scratch,
                0,
            )
            .unwrap();

        assert_eq!(
            zero_capacity.status,
            tla_jit_abi::SuccessorKernelStatus::BufferOverflow,
            "native zero-capacity out={zero_capacity:?} zero_capacity_successors={zero_capacity_successors:?}"
        );
        assert_eq!(zero_capacity.successor_count, 0);
        assert_eq!(zero_capacity.generated_count, 2);
        assert_eq!(zero_capacity.state_len, 3);
        assert_eq!(zero_capacity.overflow_count, 2);
        assert_eq!(
            zero_capacity.runtime_error,
            tla_jit_abi::JitRuntimeErrorKind::DivisionByZero
        );
        assert_eq!(
            zero_capacity.unsupported_reason,
            tla_jit_abi::SuccessorKernelUnsupportedReason::None
        );
        assert_eq!(zero_capacity.metadata_bits, 0b101);
        assert!(zero_capacity_successors.is_empty());

        let mut disabled_scratch = PetriKernelScratch::new();
        let (disabled, disabled_successors) =
            native::unchecked_native_all_transition_successors_for_tests(
                &cache,
                &net,
                &[0, 0, 0],
                &mut disabled_scratch,
                cache.transition_count,
            )
            .unwrap();

        assert_eq!(
            disabled.status,
            tla_jit_abi::SuccessorKernelStatus::Disabled,
            "native disabled out={disabled:?} disabled_successors={disabled_successors:?}"
        );
        assert_eq!(disabled.successor_count, 0);
        assert_eq!(disabled.generated_count, 0);
        assert_eq!(disabled.state_len, 3);
        assert_eq!(disabled.overflow_count, 0);
        assert_eq!(
            disabled.runtime_error,
            tla_jit_abi::JitRuntimeErrorKind::DivisionByZero
        );
        assert_eq!(
            disabled.unsupported_reason,
            tla_jit_abi::SuccessorKernelUnsupportedReason::None
        );
        assert_eq!(disabled.metadata_bits, 0);
        assert_eq!(disabled_successors, vec![0; cache.transition_count * 3]);
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_candidate_callable_artifact_carries_shared_engine_identity() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();

        let candidate = native::petri_native_successor_batch_candidate(&net, &cache);

        let native::PetriNativeSuccessorBatchCandidate::CallableArtifact(batch) = candidate else {
            panic!("fixture should build a callable native artifact candidate: {candidate:?}");
        };
        let readiness = &batch.readiness;
        assert_eq!(
            readiness.schema,
            "ty.petri.native_successor.candidate_batch.v1"
        );
        assert_eq!(readiness.status_code, "callable_artifact");
        assert_eq!(
            readiness.artifact_kind,
            tla_jit_abi::SUCCESSOR_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(readiness.entry_symbol, PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL);
        assert_eq!(readiness.shared_signature_abi, "extern_c");
        assert_eq!(readiness.shared_signature_params, 9);
        assert_eq!(readiness.shared_signature_returns, 1);
        assert!(readiness.trust_ir_entry_abi_matches_shared_successor_kernel);
        assert_eq!(readiness.state_len, 3);
        assert_eq!(readiness.max_successors, 3);
        assert_eq!(readiness.state_bytes, 24);
        assert!(readiness.compile_artifact_handoff_ready);
        assert!(readiness.callable_pointer_available);
        assert_ne!(readiness.native_payload_sha256, "none");
        assert_ne!(readiness.executable_region_sha256, "none");
        assert_ne!(readiness.lifetime_owner, "none");
        assert_ne!(readiness.transport_digest, "none");
        assert_ne!(readiness.bundle_digest, "none");
        assert_ne!(readiness.target_abi_digest, "none");
        assert_eq!(
            readiness.runtime_readiness_status_code,
            "ready_for_runtime_call"
        );
        assert_eq!(readiness.runtime_readiness_reason_code, "none");
        assert_eq!(readiness.runtime_readiness_required_evidence, "none");
        assert!(readiness.runtime_ready_for_call);
        assert_eq!(readiness.validation_receipt_status, "accepted");
        assert_eq!(readiness.parity_receipt_status, "accepted");
        assert_eq!(readiness.callable_receipt_status, "accepted");
        assert_eq!(readiness.callable_receipt_reason_code, "available");
        assert_eq!(readiness.native_missing_receipts, "none");
        assert_eq!(readiness.production_gate_status, "selected");
        assert!(readiness.production_selected);
        assert!(!readiness.fail_closed);
        assert_eq!(
            batch.compile_artifact_handoff.entry_symbol.as_deref(),
            Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL)
        );
        assert!(batch.compile_artifact_handoff.is_ready());
        assert!(batch.runtime_readiness.is_ready_for_runtime_call());
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn native_candidate_unsupported_plan_cache_path_is_explicit() {
        let net = all_transition_net();
        let mut cache = PetriKernelPlanCache::for_net(&net).unwrap();
        cache.place_count = 99;

        let candidate = native::petri_native_successor_batch_candidate(&net, &cache);

        let native::PetriNativeSuccessorBatchCandidate::Blocked(packet) = candidate else {
            panic!("invalid plan cache must not build a callable artifact: {candidate:?}");
        };
        assert_eq!(packet.status_code, "blocked");
        assert_eq!(packet.reason_code, "plan_cache_invalid");
        assert!(
            packet.blocker.contains("CachePlaceCountMismatch"),
            "blocked candidate should expose the precise plan-cache blocker: {packet:?}"
        );
        assert_eq!(
            packet.artifact_kind,
            tla_jit_abi::SUCCESSOR_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(packet.shared_signature_abi, "extern_c");
        assert_eq!(packet.shared_signature_params, 9);
        assert_eq!(packet.shared_signature_returns, 1);
        assert!(!packet.callable_pointer_available);
        assert!(!packet.production_selected);
        assert!(packet.fail_closed);
    }

    #[test]
    fn checked_all_transition_candidates_clears_stale_rows_when_none_enabled() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).unwrap();
        let mut scratch = PetriKernelScratch::new();
        let mut candidates = FlatAllTransitionCandidates {
            place_count: 99,
            transition_ids: vec![TransitionIdx(99)],
            flat_successors: vec![99],
        };

        checked_all_transition_successors_cached_into(
            &net,
            &cache,
            &[0, 0, 0],
            &mut scratch,
            &mut candidates,
        )
        .unwrap();

        assert!(candidates.is_empty());
        assert_eq!(candidates.place_count(), 3);
        assert!(candidates.flat_successors().is_empty());
    }

    #[test]
    fn checked_all_transition_candidates_rejects_enabled_decision_mismatch() {
        let stale_cache = PetriKernelPlanCache::for_net(&simple_net()).unwrap();
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(2, 1)], vec![arc(1, 1)])],
            initial_marking: vec![2, 0, 0],
        };
        let mut scratch = PetriKernelScratch::new();
        let mut candidates = FlatAllTransitionCandidates::new();

        let error = checked_all_transition_successors_cached_into(
            &net,
            &stale_cache,
            &[2, 0, 0],
            &mut scratch,
            &mut candidates,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PetriKernelError::ParityMismatch {
                transition: TransitionIdx(0),
                ..
            }
        ));
    }

    #[test]
    fn checked_all_transition_candidates_rejects_successor_mismatch() {
        let stale_cache = PetriKernelPlanCache::for_net(&simple_net()).unwrap();
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1), arc(2, 2)])],
            initial_marking: vec![5, 0, 0],
        };
        let mut scratch = PetriKernelScratch::new();
        let mut candidates = FlatAllTransitionCandidates::new();

        let error = checked_all_transition_successors_cached_into(
            &net,
            &stale_cache,
            &[5, 0, 0],
            &mut scratch,
            &mut candidates,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PetriKernelError::ParityMismatch {
                transition: TransitionIdx(0),
                ..
            }
        ));
    }

    #[test]
    fn transition_plan_flat_disabled_matches_is_enabled() {
        let net = simple_net();
        let mut scratch = PetriKernelScratch::new();
        let outcome =
            checked_fire_transition(&net, TransitionIdx(0), &[1, 0, 0], &mut scratch).unwrap();
        assert_eq!(outcome, CheckedTransitionOutcome::Disabled);
    }

    #[test]
    fn transition_plan_flat_requires_all_inputs() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2), arc(1, 1)], vec![arc(2, 1)])],
            initial_marking: vec![2, 1, 0],
        };
        let mut scratch = PetriKernelScratch::new();

        let enabled =
            checked_fire_transition(&net, TransitionIdx(0), &[2, 1, 0], &mut scratch).unwrap();
        assert_eq!(
            enabled,
            CheckedTransitionOutcome::Enabled {
                successor: vec![0, 0, 1],
            }
        );

        let disabled =
            checked_fire_transition(&net, TransitionIdx(0), &[2, 0, 0], &mut scratch).unwrap();
        assert_eq!(disabled, CheckedTransitionOutcome::Disabled);
    }

    #[test]
    fn transition_plan_flat_self_loop_matches_fire_into() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 1), arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        let mut scratch = PetriKernelScratch::new();
        let outcome =
            checked_fire_transition(&net, TransitionIdx(0), &[1, 0], &mut scratch).unwrap();
        assert_eq!(
            outcome,
            CheckedTransitionOutcome::Enabled {
                successor: vec![1, 1],
            }
        );
    }

    #[test]
    fn transition_plan_flat_reports_i64_successor_overflow() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![i64::MAX as u64],
        };
        let mut scratch = PetriKernelScratch::new();
        let error =
            checked_fire_transition(&net, TransitionIdx(0), &[i64::MAX as u64], &mut scratch)
                .unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::TokenOverflow {
                place: PlaceIdx(0),
                value: i64::MAX,
                delta: 1,
            }
        );
    }

    #[test]
    fn checked_fire_transition_rejects_wide_token_before_interpreter() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![u64::MAX],
        };
        let mut scratch = PetriKernelScratch::new();

        let error =
            checked_fire_transition(&net, TransitionIdx(0), &[u64::MAX], &mut scratch).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::TokenExceedsI64 {
                place: 0,
                value: u64::MAX,
            }
        );
    }

    #[test]
    fn transition_parity_config_accepts_matching_successor() {
        let net = simple_net();
        let parity = PetriTransitionParityConfig::enabled_for_tests(true);
        let mut scratch = PetriKernelScratch::new();

        parity
            .check_transition_successor(
                &net,
                TransitionIdx(0),
                &[5, 0, 0],
                &[3, 1, 3],
                &mut scratch,
            )
            .unwrap();
    }

    #[test]
    fn transition_parity_config_rejects_mismatched_successor() {
        let net = simple_net();
        let parity = PetriTransitionParityConfig::enabled_for_tests(true);
        let mut scratch = PetriKernelScratch::new();

        let error = parity
            .check_transition_successor(
                &net,
                TransitionIdx(0),
                &[5, 0, 0],
                &[3, 1, 2],
                &mut scratch,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            PetriKernelError::ParityMismatch {
                transition: TransitionIdx(0),
                ..
            }
        ));
    }

    #[test]
    fn transition_parity_config_rejects_disabled_transition_successor() {
        let net = simple_net();
        let parity = PetriTransitionParityConfig::enabled_for_tests(true);
        let mut scratch = PetriKernelScratch::new();

        let error = parity
            .check_transition_successor(
                &net,
                TransitionIdx(0),
                &[1, 0, 0],
                &[1, 0, 0],
                &mut scratch,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            PetriKernelError::ParityMismatch {
                transition: TransitionIdx(0),
                ..
            }
        ));
    }

    #[test]
    fn transition_parity_config_disabled_is_observation_only() {
        let net = simple_net();
        let parity = PetriTransitionParityConfig::enabled_for_tests(false);
        let mut scratch = PetriKernelScratch::new();

        parity
            .check_transition_successor(
                &net,
                TransitionIdx(0),
                &[1, 0, 0],
                &[99, 99, 99],
                &mut scratch,
            )
            .unwrap();
    }

    #[test]
    fn predicate_flat_tokens_count_matches_interpreter() {
        let net = simple_net();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ResolvedIntExpr::Constant(6),
        );
        let mut scratch = PetriKernelScratch::new();

        let value = checked_eval_predicate(&net, &pred, &[5, 0, 0], &mut scratch).unwrap();
        assert!(value);

        let value = checked_eval_predicate(&net, &pred, &[5, 2, 0], &mut scratch).unwrap();
        assert!(!value);
    }

    #[test]
    fn predicate_flat_intle_true_and_false_match_interpreter() {
        let net = simple_net();
        let true_pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
        );
        let false_pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(9),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
        );
        let mut scratch = PetriKernelScratch::new();

        assert!(checked_eval_predicate(&net, &true_pred, &[5, 1, 0], &mut scratch).unwrap());
        assert!(!checked_eval_predicate(&net, &false_pred, &[5, 1, 0], &mut scratch).unwrap());
    }

    #[test]
    fn predicate_flat_boolean_tree_matches_interpreter() {
        let net = simple_net();
        let pred = ResolvedPredicate::And(vec![
            ResolvedPredicate::True,
            ResolvedPredicate::Not(Box::new(ResolvedPredicate::False)),
            ResolvedPredicate::Or(vec![
                ResolvedPredicate::False,
                ResolvedPredicate::IntLe(
                    ResolvedIntExpr::TokensCount(vec![PlaceIdx(2)]),
                    ResolvedIntExpr::Constant(3),
                ),
            ]),
        ]);
        let mut scratch = PetriKernelScratch::new();

        assert!(checked_eval_predicate(&net, &pred, &[5, 0, 3], &mut scratch).unwrap());
        assert!(!checked_eval_predicate(&net, &pred, &[5, 0, 4], &mut scratch).unwrap());
    }

    #[test]
    fn predicate_flat_is_fireable_matches_interpreter() {
        let net = simple_net();
        let pred = ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]);
        let mut scratch = PetriKernelScratch::new();

        assert!(checked_eval_predicate(&net, &pred, &[2, 0, 0], &mut scratch).unwrap());
        assert!(!checked_eval_predicate(&net, &pred, &[1, 0, 0], &mut scratch).unwrap());
    }

    #[test]
    fn predicate_flat_is_fireable_checks_any_listed_transition() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![
                trans("t0", vec![arc(0, 2)], vec![arc(2, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![0, 1, 0],
        };
        let pred = ResolvedPredicate::IsFireable(vec![TransitionIdx(0), TransitionIdx(1)]);
        let mut scratch = PetriKernelScratch::new();

        assert!(checked_eval_predicate(&net, &pred, &[0, 1, 0], &mut scratch).unwrap());
        assert!(!checked_eval_predicate(&net, &pred, &[0, 0, 0], &mut scratch).unwrap());
    }

    #[test]
    fn predicate_flat_rejects_tokens_count_place_out_of_bounds() {
        let net = simple_net();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(99)]),
            ResolvedIntExpr::Constant(0),
        );
        let mut flat = Vec::new();
        marking_to_flat_i64(&[0, 0, 0], &mut flat).unwrap();

        let error =
            eval_predicate_flat(PetriKernelLayout::for_net(&net), &net, &pred, &flat).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::PlaceOutOfBounds {
                place: PlaceIdx(99),
                place_count: 3,
            }
        );
    }

    #[test]
    fn checked_predicate_rejects_tokens_count_place_out_of_bounds_before_interpreter() {
        let net = simple_net();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(99)]),
            ResolvedIntExpr::Constant(0),
        );
        let mut scratch = PetriKernelScratch::new();

        let error = checked_eval_predicate(&net, &pred, &[0, 0, 0], &mut scratch).unwrap_err();
        assert!(matches!(
            error,
            PetriKernelError::PlaceOutOfBounds {
                place: PlaceIdx(99),
                place_count: 3,
            }
        ));
    }

    #[test]
    fn checked_predicate_rejects_is_fireable_transition_out_of_bounds_before_interpreter() {
        let net = simple_net();
        let pred = ResolvedPredicate::IsFireable(vec![TransitionIdx(99)]);
        let mut scratch = PetriKernelScratch::new();

        let error = checked_eval_predicate(&net, &pred, &[0, 0, 0], &mut scratch).unwrap_err();
        assert!(matches!(
            error,
            PetriKernelError::TransitionOutOfBounds {
                transition: TransitionIdx(99),
                transition_count: 1,
            }
        ));
    }

    #[test]
    fn checked_predicate_validates_short_circuited_and_child() {
        let net = simple_net();
        let pred = ResolvedPredicate::And(vec![
            ResolvedPredicate::False,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(99)]),
                ResolvedIntExpr::Constant(0),
            ),
        ]);
        let mut scratch = PetriKernelScratch::new();

        let error = checked_eval_predicate(&net, &pred, &[0, 0, 0], &mut scratch).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::PlaceOutOfBounds {
                place: PlaceIdx(99),
                place_count: 3,
            }
        );
    }

    #[test]
    fn checked_predicate_validates_short_circuited_or_child() {
        let net = simple_net();
        let pred = ResolvedPredicate::Or(vec![
            ResolvedPredicate::True,
            ResolvedPredicate::IsFireable(vec![TransitionIdx(99)]),
        ]);
        let mut scratch = PetriKernelScratch::new();

        let error = checked_eval_predicate(&net, &pred, &[0, 0, 0], &mut scratch).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::TransitionOutOfBounds {
                transition: TransitionIdx(99),
                transition_count: 1,
            }
        );
    }

    #[test]
    fn checked_predicate_validates_short_circuited_constant_above_i64_max() {
        let net = simple_net();
        let pred = ResolvedPredicate::Or(vec![
            ResolvedPredicate::True,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(i64::MAX as u64 + 1),
                ResolvedIntExpr::Constant(0),
            ),
        ]);
        let mut scratch = PetriKernelScratch::new();

        let error = checked_eval_predicate(&net, &pred, &[0, 0, 0], &mut scratch).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::ConstantExceedsI64 {
                value: i64::MAX as u64 + 1,
            }
        );
    }

    #[test]
    fn predicate_flat_rejects_constant_above_i64_max() {
        let net = simple_net();
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(i64::MAX as u64 + 1),
            ResolvedIntExpr::Constant(0),
        );
        let mut flat = Vec::new();
        marking_to_flat_i64(&[0, 0, 0], &mut flat).unwrap();

        let error =
            eval_predicate_flat(PetriKernelLayout::for_net(&net), &net, &pred, &flat).unwrap_err();
        assert_eq!(
            error,
            PetriKernelError::ConstantExceedsI64 {
                value: i64::MAX as u64 + 1,
            }
        );
    }

    #[test]
    fn predicate_flat_rejects_tokens_count_i64_overflow() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![],
            initial_marking: vec![i64::MAX as u64, 1],
        };
        let pred = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ResolvedIntExpr::Constant(i64::MAX as u64),
        );
        let mut flat = Vec::new();
        marking_to_flat_i64(&[i64::MAX as u64, 1], &mut flat).unwrap();

        let error =
            eval_predicate_flat(PetriKernelLayout::for_net(&net), &net, &pred, &flat).unwrap_err();
        assert!(matches!(error, PetriKernelError::IntExprOverflow { .. }));
    }

    #[test]
    fn checked_predicate_validates_short_circuited_tokens_count_i64_overflow() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![],
            initial_marking: vec![i64::MAX as u64, 1],
        };
        let pred = ResolvedPredicate::And(vec![
            ResolvedPredicate::False,
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
                ResolvedIntExpr::Constant(i64::MAX as u64),
            ),
        ]);
        let mut scratch = PetriKernelScratch::new();

        let error =
            checked_eval_predicate(&net, &pred, &[i64::MAX as u64, 1], &mut scratch).unwrap_err();
        assert!(matches!(error, PetriKernelError::IntExprOverflow { .. }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn focused_trust_ir_component_rows<'a>(
        report: &'a CapabilityReport,
        component: &str,
    ) -> Vec<&'a str> {
        let prefix = format!("trust-ir {component} ");
        report
            .evidence
            .iter()
            .filter_map(|row| row.starts_with(&prefix).then_some(row.as_str()))
            .collect()
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn focused_manifest_line(row: &str) -> Option<&str> {
        row.split_once(" manifest_line=")
            .map(|(_prefix, line)| line)
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn assert_focused_trust_ir_component_lines(
        rows: &[&str],
        schema: &str,
        schema_version: u32,
        expected_lines: &[String],
    ) {
        let schema_version = schema_version.to_string();
        assert_eq!(rows.len(), expected_lines.len());
        for (row, expected_line) in rows.iter().zip(expected_lines) {
            assert_eq!(evidence_field(row, "schema"), Some(schema));
            assert_eq!(
                evidence_field(row, "schema_version"),
                Some(schema_version.as_str())
            );
            assert_eq!(
                evidence_field(row, "source_package"),
                Some(trust_ir::PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SOURCE_PACKAGE)
            );
            assert_eq!(evidence_field(row, "source_project"), Some("trust-ir"));
            assert_eq!(evidence_field(row, "project"), Some("trust-ir"));
            assert_eq!(focused_manifest_line(row), Some(expected_line.as_str()));
        }
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_ir_ty_mcc_shared_primitive_manifest_rows_match_producer() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let rows = focused_trust_ir_component_rows(
            &report,
            TRUST_IR_TY_MCC_SHARED_PRIMITIVE_MANIFEST_COMPONENT,
        );
        let expected_lines = trust_ir::ty_shared_primitive_manifest_key_value_lines();

        assert_focused_trust_ir_component_lines(
            &rows,
            trust_ir::TY_SHARED_PRIMITIVE_MANIFEST_SCHEMA,
            trust_ir::TY_SHARED_PRIMITIVE_MANIFEST_SCHEMA_VERSION,
            &expected_lines,
        );
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row) == Some("ty_shared_primitive_manifest.status=available")
        }));
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row).is_some_and(|line| {
                line == "ty_shared_primitive_manifest.component.2.rows_api=chc_x86_hardware_vector_contract_manifest_rows()"
            })
        }));
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row).is_some_and(|line| {
                line == "ty_shared_primitive_manifest.hardware_vector_contract_row.0.key=hardware_vector_contract_set.manifest.schema"
            })
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_ir_hardware_vector_contract_rows_match_producer() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let rows = focused_trust_ir_component_rows(
            &report,
            TRUST_IR_HARDWARE_VECTOR_CONTRACT_SET_COMPONENT,
        );
        let expected_lines = trust_ir::chc_x86_hardware_vector_contract_manifest_key_value_lines();

        assert_focused_trust_ir_component_lines(
            &rows,
            trust_ir::HARDWARE_VECTOR_CONTRACT_MANIFEST_SCHEMA,
            trust_ir::HARDWARE_VECTOR_CONTRACT_MANIFEST_SCHEMA_VERSION,
            &expected_lines,
        );
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row)
                == Some("hardware_vector_contract_set.source.package=trust-ir")
        }));
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row)
                == Some("hardware_vector_contract_set.contract.0.contract.name=chc_x86.v4_i32")
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_cg_compile_artifact_cache_rows_remain_fail_closed() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let descriptor = report
            .evidence
            .iter()
            .find(|row| row.starts_with("trust-cg compile_artifact_cache_telemetry_descriptor "))
            .expect("trust-cg cache telemetry descriptor should be emitted");
        let telemetry = report
            .evidence
            .iter()
            .find(|row| row.starts_with("trust-cg compile_artifact_cache_telemetry "))
            .expect("trust-cg cache telemetry row should be emitted");

        assert_eq!(
            evidence_field(descriptor, "schema"),
            Some(tla_trust_cg::COMPILE_ARTIFACT_CACHE_TELEMETRY_SCHEMA)
        );
        assert_eq!(
            evidence_field(descriptor, "required_fields"),
            Some(
                join_ay_strs(tla_trust_cg::COMPILE_ARTIFACT_CACHE_TELEMETRY_REQUIRED_FIELDS)
                    .as_str()
            )
        );
        assert_eq!(
            evidence_field(descriptor, "authorizes_useful_native"),
            Some("false")
        );
        assert_eq!(evidence_field(descriptor, "fail_closed"), Some("true"));
        assert_eq!(
            evidence_field(descriptor, "source"),
            Some("CompileArtifactCacheTelemetryDescriptor")
        );
        assert_eq!(
            evidence_field(descriptor, "producer_api_status"),
            Some("available_tla_trust_cg_reexport")
        );
        assert_eq!(
            evidence_field(telemetry, "key_sha256"),
            Some(TRUST_CG_COMPILE_ARTIFACT_CACHE_TELEMETRY_PROBE_KEY_SHA256)
        );
        assert_eq!(evidence_field(telemetry, "status"), Some("miss"));
        assert_eq!(
            evidence_field(telemetry, "reason"),
            Some("cache_probe_only")
        );
        assert_eq!(
            evidence_field(telemetry, "source"),
            Some("CompileArtifactCacheTelemetry")
        );
        assert_eq!(
            evidence_field(telemetry, "producer_api_status"),
            Some("available_tla_trust_cg_reexport")
        );
        assert_eq!(
            evidence_field(telemetry, "authorizes_useful_native"),
            Some("false")
        );
        assert_eq!(evidence_field(telemetry, "fail_closed"), Some("true"));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_cg_host_jit_pgo_rows_match_producer_descriptor() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let producer = tla_trust_cg::pgo::trust_cg_host_jit_pgo_provenance_descriptor();
        let descriptor = report
            .evidence
            .iter()
            .find(|row| row.starts_with("trust-cg host_jit_pgo_provenance_descriptor "))
            .expect("trust-cg PGO provenance descriptor should be emitted");
        let manifest_rows = report
            .evidence
            .iter()
            .filter_map(|row| {
                row.starts_with("trust-cg host_jit_pgo_profile_authority_manifest ")
                    .then_some(row.as_str())
            })
            .collect::<Vec<_>>();
        let expected_manifest_lines = fail_closed_pgo_profile_use_status()
            .trust_cg_profile_authority_manifest_lines()
            .expect("static PGO manifest should render");

        assert_eq!(evidence_field(descriptor, "schema"), Some(producer.schema));
        assert_eq!(
            evidence_field(descriptor, "profile_report_schema"),
            Some(producer.profile_report_schema)
        );
        assert_eq!(
            evidence_field(descriptor, "profile_key_fields"),
            Some(join_ay_strs(producer.profile_key_fields).as_str())
        );
        assert_eq!(
            evidence_field(descriptor, "profile_use_soundness_fields"),
            Some(join_ay_strs(producer.profile_use_soundness_fields).as_str())
        );
        assert_eq!(
            evidence_field(descriptor, "profile_authority_manifest_row_keys"),
            Some(join_ay_strs(producer.profile_authority_manifest_row_keys).as_str())
        );
        assert_eq!(
            evidence_field(descriptor, "authorizes_useful_native"),
            Some("false")
        );
        assert_eq!(evidence_field(descriptor, "fail_closed"), Some("true"));
        assert_eq!(manifest_rows.len(), expected_manifest_lines.len());
        for (row, expected_line) in manifest_rows.iter().zip(expected_manifest_lines.iter()) {
            assert_eq!(focused_manifest_line(row), Some(expected_line.as_str()));
        }
        assert!(manifest_rows.iter().any(|row| {
            focused_manifest_line(row)
                == Some("profile_authority.status=not_authoritative_for_compiled_function")
        }));
        assert!(manifest_rows.iter().any(|row| {
            focused_manifest_line(row) == Some("profile_authority.reason=profile_use_not_scheduled")
        }));
        assert!(manifest_rows.iter().any(|row| {
            focused_manifest_line(row) == Some("profile_authority.authorizes_useful_native=false")
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_ir_native_evidence_artifact_resolution_row_matches_producer() {
        let net = all_transition_net();
        let report = petri_native_successor_capability_report(&net);
        let row = report
            .evidence
            .iter()
            .find(|row| row.starts_with("trust-ir native_evidence_artifact_resolution "))
            .expect("trust-ir native evidence artifact resolution should be emitted");
        let bundle = native_verification_bundle_fixture(&net);
        let shared_contract = tla_trust_cg::petri_native_successor_downstream_contract_descriptor()
            .trust_ir_petri_trust_mc_chc_shared_primitive_contract;
        let requirement = shared_contract
            .production_required_artifact_requirements()
            .iter()
            .copied()
            .find(|requirement| requirement.role_code() == "replay_transcript")
            .expect("fixture should require a replay transcript artifact");
        let key = trust_ir_artifact_requirement_resolution_key(
            &bundle,
            shared_contract.verifier_suite,
            requirement,
        );
        let expected = bundle.resolve_evidence_artifact_attachment(key, &[]);

        assert_eq!(
            evidence_field(row, "schema"),
            Some(expected.report.schema.as_str())
        );
        assert_eq!(
            evidence_field(row, "request"),
            Some(expected.report.request.to_string().as_str())
        );
        assert_eq!(
            evidence_field(row, "owner_suite"),
            expected
                .report
                .owner_suite
                .map(trust_ir::request::NativeVerifierSuite::code)
        );
        assert_eq!(
            evidence_field(row, "required_kind"),
            Some(expected.report.required_kind.code())
        );
        assert_eq!(
            evidence_field(row, "status_code"),
            Some(expected.report.status_code())
        );
        assert_eq!(
            evidence_field(row, "reason_code"),
            Some(expected.report.reason_code())
        );
        assert_eq!(evidence_field(row, "status_code"), Some("blocked"));
        // The Petri native producer now attaches a semantic evidence bundle to the
        // verification bundle, so resolution proceeds past the bundle lookup and
        // returns `missing_attachment` when no attachments are supplied (instead of
        // the earlier `missing_evidence_bundle`). See `attach_petri_native_successor_semantic_evidence`.
        assert_eq!(
            evidence_field(row, "reason_code"),
            Some("missing_attachment")
        );
        assert_eq!(evidence_field(row, "authority_code"), Some("informational"));
        assert_eq!(evidence_field(row, "is_resolved"), Some("false"));
        assert_eq!(evidence_field(row, "is_authoritative"), Some("false"));
        assert_eq!(evidence_field(row, "fail_closed"), Some("true"));
        for line in expected.authority_evidence_key_value_lines() {
            let (key, value) = line
                .split_once('=')
                .expect("producer authority line should be key=value");
            assert_eq!(
                evidence_field(row, key),
                Some(value),
                "missing producer authority field {key} from {row}"
            );
        }
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_ir_semantic_bridge_proof_identity_rows_match_producer() {
        let net = all_transition_net();
        let bundle = native_verification_bundle_fixture(&net);
        let bridge = petri_native_successor_semantic_bridge(&bundle);
        let producer_report = bundle.petri_successor_semantic_bridge_report(bridge.function);
        let expected_lines = producer_report.proof_identity_key_value_lines();
        let proof_identity_text = producer_report.proof_identity_key_value_text();
        let replay =
            producer_report.proof_identity_replay_report_for_key_value_text(&proof_identity_text);
        let report = petri_native_successor_capability_report(&net);
        let semantic_bridge = report
            .evidence
            .iter()
            .find(|row| row.contains("Petri native_jit semantic_successor_bridge"))
            .expect("semantic bridge row should be emitted");
        let rows = focused_trust_ir_component_rows(
            &report,
            TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT,
        );
        let row_count = rows.len().to_string();
        let digest = producer_report.proof_identity_digest().to_string();
        let replayable = replay.is_replayable().to_string();
        let replay_fail_closed = replay.fail_closed.to_string();
        let diagnostic_count = replay.diagnostic_count().to_string();
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let trust_ir_identity = downstream_contract.trust_ir_native_bundle_identity;

        assert_focused_trust_ir_component_lines(
            &rows,
            trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA,
            trust_ir::NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA_VERSION,
            &expected_lines,
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_row_count"
            ),
            Some(row_count.as_str())
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_digest"
            ),
            Some(digest.as_str())
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_status_code"
            ),
            Some(replay.status_code)
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replayable"
            ),
            Some(replayable.as_str())
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_fail_closed"
            ),
            Some(replay_fail_closed.as_str())
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_diagnostic_count"
            ),
            Some(diagnostic_count.as_str())
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_text_api"
            ),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(
                semantic_bridge,
                "trust_ir_semantic_bridge_proof_identity_replay_api"
            ),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::NativeSemanticBridgeProofIdentityReplayReportForKeyValueText,
            ))
        );
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row)
                .is_some_and(|line| line.starts_with("semantic_bridge_proof_identity.digest="))
        }));
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row).is_some_and(|line| {
                line == format!(
                    "semantic_bridge_proof_identity.report.reason={}",
                    producer_report.reason_code()
                )
            })
        }));
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn focused_trust_ir_trust_mc_chc_proof_evidence_identity_rows_match_producer() {
        let net = all_transition_net();
        let bundle = semantic_evidence_native_verification_bundle_fixture(&net);
        let bridge = petri_native_successor_semantic_bridge(&bundle);
        let producer_report =
            ay_trust_mc_native_bundle::trust_mc_petri_successor_chc_model_acceptance_report(
                &bundle,
                bridge.function,
            );
        let proof_handoff_report = &producer_report
            .trust_mc_chc_model_validation_readiness_report
            .proof_handoff_report;
        let expected_lines = proof_handoff_report.proof_evidence_identity_key_value_lines();
        let proof_identity_text = proof_handoff_report.proof_evidence_identity_key_value_text();
        let replay = proof_handoff_report
            .proof_evidence_identity_replay_report_for_key_value_text(&proof_identity_text);
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        add_ay_trust_mc_native_verification_bundle_facade_evidence(
            &mut report,
            &bundle,
            "represented_trust_mc_fixture",
            true,
        );
        let model_acceptance = report
            .evidence
            .iter()
            .find(|row| row.contains("AY trust_mc_petri_successor_chc_model_acceptance"))
            .expect("AY model acceptance row should be emitted");
        let rows = focused_trust_ir_component_rows(
            &report,
            TRUST_IR_PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_COMPONENT,
        );
        let row_count = rows.len().to_string();
        let digest = proof_handoff_report
            .proof_evidence_identity_digest()
            .to_string();
        let replayable = replay.is_replayable().to_string();
        let replay_fail_closed = replay.fail_closed.to_string();
        let diagnostic_count = replay.diagnostic_count().to_string();
        let downstream_contract =
            tla_trust_cg::petri_native_successor_downstream_contract_descriptor();
        let trust_ir_identity = downstream_contract.trust_ir_native_bundle_identity;

        assert_focused_trust_ir_component_lines(
            &rows,
            trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA,
            trust_ir::PETRI_SUCCESSOR_TRUST_MC_CHC_PROOF_EVIDENCE_IDENTITY_SCHEMA_VERSION,
            &expected_lines,
        );
        assert_eq!(
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_row_count"
            ),
            Some(row_count.as_str())
        );
        assert_eq!(
            evidence_field(model_acceptance, "trust_ir_proof_evidence_identity_digest"),
            Some(digest.as_str())
        );
        assert_eq!(
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_status_code"
            ),
            Some(replay.status_code)
        );
        assert_eq!(
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replayable"
            ),
            Some(replayable.as_str())
        );
        assert_eq!(
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_fail_closed"
            ),
            Some(replay_fail_closed.as_str())
        );
        assert_eq!(
            evidence_field(
                model_acceptance,
                "trust_ir_proof_evidence_identity_replay_diagnostic_count"
            ),
            Some(diagnostic_count.as_str())
        );
        assert_eq!(
            evidence_field(model_acceptance, "trust_ir_proof_evidence_identity_text_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityKeyValueText,
            ))
        );
        assert_eq!(
            evidence_field(model_acceptance, "trust_ir_proof_evidence_identity_replay_api"),
            Some(trust_ir_petri_trust_mc_provided_field(
                trust_ir_identity.provided_fields,
                TrustIrPetriTrustMcProvidedField::PetriSuccessorTrustMcChcProofEvidenceIdentityReplayReportForKeyValueText,
            ))
        );
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row)
                .is_some_and(|line| line.starts_with("proof_evidence_identity.digest="))
        }));
        assert!(rows.iter().any(|row| {
            focused_manifest_line(row).is_some_and(|line| {
                line == format!(
                    "proof_handoff.reason={}",
                    proof_handoff_report.reason_code()
                )
            })
        }));
    }
}
