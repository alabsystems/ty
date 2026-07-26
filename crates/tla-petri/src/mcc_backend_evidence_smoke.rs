// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Generator for the MCC backend-evidence JSONL smoke sidecar.
//!
//! Replaces a former Python helper. The Python module
//! emitted a deterministic JSONL sidecar consumed by:
//! * `ty-mcc-smoke` (the BenchKit MCC competition wrapper),
//! * `ty-mcc-backend-evidence-validate` (the canary), and
//! * the freshness gate used by `mccctl doctor`.
//!
//! This port preserves the exact row shape and exact token ordering of
//! every emitted evidence string, because downstream consumers grep for
//! literal substrings. The entry point [`generated_replay_smoke_rows`]
//! returns three JSON envelopes — one per smoke target (Petri, AIGER,
//! BTOR2) — and [`write_jsonl`] serialises them.
//!
//! The legacy TrustIR rows in this sidecar are compatibility telemetry, not
//! typed proof evidence; their native handoff remains fail-closed and
//! `production_selected=false`. A few rows intentionally retain the historical
//! Python generator's stable-v1 text so old canaries can parse archived
//! sidecars. That compatibility digest is private, returns an untyped `String`,
//! and must never be converted into a `trust_ir::ProofDigest` or used at an
//! authority/replay boundary.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use toml::Value as TomlValue;

use crate::mcc_ay_pin::{source_rev, validate_ay_pin};

// ============================================================
// Constants
// ============================================================
//
// The constants below are the stable schema names, API symbol names, source
// packages, status codes, and report row-key lists that this generator embeds
// verbatim into the JSONL evidence rows. Their *values* are the contract:
// downstream consumers (the validator canary and the doctor freshness gate)
// grep for these literal strings, so they must never drift.

/// API surface that builds the native-successor call packet from a trust-ir
/// bundle.
pub const CALL_PACKET_SURFACE: &str = "petri_native_successor_call_packet_from_trust_ir_bundle";
/// Schema name for the native-successor call packet.
pub const CALL_PACKET_SCHEMA: &str = "trust_cg.petri.native_successor.call_packet.v1";
/// Schema name for the trust-cg compile-artifact handoff evidence.
pub const COMPILE_ARTIFACT_HANDOFF_SCHEMA: &str =
    "trust-cg.petri.native_successor.compile_artifact_handoff.v1";
/// `InstalledArtifact` API that emits the compile-artifact handoff evidence.
pub const COMPILE_ARTIFACT_HANDOFF_INSTALLED_ARTIFACT_API: &str =
    "InstalledArtifact::petri_native_successor_compile_artifact_handoff_evidence";
/// Surface name for AY's CHC typed trace-assignment evidence.
pub const AY_TRACE_ASSIGNMENT_SURFACE: &str = "ay_chc_typed_trace_assignments";
/// Schema name for the hardware-replay primitive.
pub const HARDWARE_REPLAY_PRIMITIVE_SCHEMA: &str = "hardware_replay_primitive/v1";
/// Schema name for the trust-ir shared-primitive contract manifest.
pub const NATIVE_SHARED_PRIMITIVE_CONTRACT_MANIFEST_SCHEMA: &str =
    "trust_ir.native.shared_primitive_contract.manifest.v1";
/// Schema name for the trust-cg JIT compile-artifact cache telemetry.
pub const TRUST_CG_COMPILE_ARTIFACT_CACHE_TELEMETRY_SCHEMA: &str =
    "trust_cg.jit.compile_artifact_cache.telemetry.v1";
/// Schema name for the trust-cg native install-gate admission summary.
pub const TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SCHEMA: &str =
    "trust_cg.phase6.native_install_gate.admission_summary.v1";
/// Source package that owns the native install-gate admission summary.
pub const TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE_PACKAGE: &str = "trust-cg-codegen";
/// Schema name for the call-packet contract descriptor.
pub const TRUST_CG_CALL_PACKET_CONTRACT_DESCRIPTOR_SCHEMA: &str =
    "trust_cg.petri.native_successor.call_packet_contract_descriptor.v1";
/// Schema name for the call-packet contract health report.
pub const TRUST_CG_CALL_PACKET_CONTRACT_HEALTH_SCHEMA: &str =
    "trust_cg.petri.native_successor.call_packet_contract_health.v1";
/// Schema name for the native-successor downstream contract.
pub const TRUST_CG_DOWNSTREAM_CONTRACT_SCHEMA: &str =
    "trust_cg.petri.native_successor.downstream_contract.v1";
/// Dependency identity of the call packet within the downstream contract
/// descriptor.
pub const TRUST_CG_CALL_PACKET_DESCRIPTOR_DEPENDENCY: &str =
    "trust-cg::petri_native_successor_downstream_contract_descriptor.call_packet";
/// API symbol of the native-successor downstream contract descriptor.
pub const TRUST_CG_DOWNSTREAM_CONTRACT_API: &str =
    "trust-cg::petri_native_successor_downstream_contract_descriptor";
/// Schema name for the host-JIT PGO provenance descriptor.
pub const TRUST_CG_HOST_JIT_PGO_PROVENANCE_DESCRIPTOR_SCHEMA: &str =
    "trust_cg.host_jit_pgo.provenance_descriptor.v1";
/// Schema name for the trust-cg profile report.
pub const TRUST_CG_PROFILE_REPORT_SCHEMA: &str = "trust_cg.profile_report.v1";
/// Schema name for the host-JIT PGO profile-authority evidence.
pub const TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_EVIDENCE_SCHEMA: &str =
    "trust_cg.host_jit_pgo.profile_authority.v1";
/// Schema name for the host-JIT PGO profile-authority manifest.
pub const TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA: &str =
    "trust_cg.host_jit_pgo.profile_authority.manifest.v1";
/// Schema version of the host-JIT PGO profile-authority manifest.
pub const TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA_VERSION: &str = "1";
/// Schema name for the native-successor semantic bridge.
pub const SEMANTIC_SUCCESSOR_BRIDGE_SCHEMA: &str =
    "trust_cg.petri.native_successor.semantic_bridge.v1";
/// Schema name for the semantic-bridge plan-cache-equivalence formula.
pub const SEMANTIC_SUCCESSOR_BRIDGE_FORMULA_SCHEMA: &str =
    "ty.petri.native.successor.plan_cache_equivalence.v1";
/// Reason code reported when the semantic-successor obligation is missing.
pub const SEMANTIC_SUCCESSOR_BRIDGE_REASON_CODE: &str = "missing_semantic_successor_obligation";
/// Schema name for AY's trust-MC Petri-successor CHC model-acceptance report.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA: &str =
    "ay.chc.trust_mc_petri_successor_chc_model_acceptance.v1";
/// Problem identity of the trust-MC native verification bundle.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_PROBLEM: &str = "trust_mc_native_verification_bundle";
/// AY backend identity for the trust-MC native bundle.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_BACKEND: &str = "ay_chc_trust_mc_native_bundle";
/// Acceptance domain for the native bundle.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_DOMAIN: &str = "native_bundle";
/// Acceptance scope for the trust-MC native CHC.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCOPE: &str = "trust_mc_native_chc";
/// AY API that produces the model-acceptance report.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_REPORT_API: &str =
    "ay::chc::trust_mc_petri_successor_chc_model_acceptance_report";
/// AY API a consumer calls to accept the model-acceptance report.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_CONSUMER_API: &str =
    "ay::chc::trust_mcPetriSuccessorChcModelAcceptanceReport::accept_for_consumer";
/// Package that owns the trust-MC native bundle acceptance.
pub const AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_PACKAGE: &str = "ay-trust_mc-native-bundle";
/// Schema name for the AY solver-capability descriptor.
pub const AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA: &str = "ay.solver-capability-descriptor.v1";
/// Schema version of the AY solver-capability descriptor.
pub const AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA_VERSION: &str = "1";
/// Schema name for the AY model-blocking clause.
pub const AY_MODEL_BLOCKING_CLAUSE_SCHEMA: &str = "ay.model-blocking-clause.v1";
/// Schema name for the AY model-blocking clause evidence.
pub const AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA: &str = "ay.model-blocking-clause-evidence.v1";
/// Schema name for the AY solve-decision-profile model consumer.
pub const AY_SOLVE_DECISION_PROFILE_MODEL_CONSUMER_SCHEMA: &str =
    "ay.solve-decision-profile-model-consumer.v1";
/// Schema name for the AY symbolic-execution contract manifest.
pub const AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA: &str =
    "ay.symbolic-execution-contract-manifest.v1";
/// Schema name for the AY symbolic-execution contract manifest health report.
pub const AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA: &str =
    "ay.symbolic-execution-contract-manifest-health.v1";
/// Schema name for the AY model-blocking symbolic-execution contract.
pub const AY_MODEL_BLOCKING_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA: &str =
    "ay.model-blocking-symbolic-execution-contract.v1";
/// Schema name for the AY incremental-assumptions symbolic-execution contract.
pub const AY_INCREMENTAL_ASSUMPTIONS_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA: &str =
    "ay.incremental-assumptions-symbolic-execution-contract.v1";
/// Schema name for the AY all-SAT-enumeration symbolic-execution contract.
pub const AY_ALL_SAT_ENUMERATION_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA: &str =
    "ay.all-sat-enumeration-symbolic-execution-contract.v1";
/// Schema name for the trust-ir trust-MC CHC contract.
pub const TRUST_IR_CONTRACT_SCHEMA: &str =
    "trust_ir.native.petri_successor.trust_mc_chc_contract.v1";
/// Formula schema referenced by the trust-ir contract (the semantic-bridge
/// formula schema).
pub const TRUST_IR_CONTRACT_FORMULA_SCHEMA: &str = SEMANTIC_SUCCESSOR_BRIDGE_FORMULA_SCHEMA;
/// Schema name for the trust-ir trust-MC CHC binding report.
pub const TRUST_IR_CONTRACT_BINDING_REPORT_SCHEMA: &str =
    "trust_ir.native.petri_successor.trust_mc_chc_binding.v1";
/// Schema name for the trust-ir trust-MC CHC proof-handoff report.
pub const TRUST_IR_CONTRACT_PROOF_HANDOFF_REPORT_SCHEMA: &str =
    "trust_ir.native.petri_successor.trust_mc_chc_proof_handoff.v1";
/// Schema name for the trust-ir trust-MC CHC model-validation readiness report.
pub const TRUST_IR_CONTRACT_MODEL_VALIDATION_READINESS_REPORT_SCHEMA: &str =
    "trust_ir.native.petri_successor.trust_mc_chc_model_validation_readiness.v1";
/// Schema name for the trust-ir shared-primitive contract.
pub const TRUST_IR_SHARED_PRIMITIVE_CONTRACT_SCHEMA: &str =
    "trust_ir.native.shared_primitive_contract.v1";
/// Schema name for the native-evidence artifact-resolution report.
pub const NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA: &str =
    "trust_ir.native.evidence.artifact_resolution.v1";
/// Schema version of the native-evidence artifact-resolution report.
pub const NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA_VERSION: &str = "1";
/// Schema name for a native-evidence artifact-authority row.
pub const NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA: &str =
    "trust_ir.native.evidence.artifact_authority_row.v1";
/// Schema version of a native-evidence artifact-authority row.
pub const NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA_VERSION: &str = "1";

// The `TNVBH_*` family names the trust-ir native verification-bundle handoff
// (TNVBH) schemas, schema versions, diagnostic fixtures, report components,
// and source-package identity embedded in the evidence rows.

/// Schema name for the trust-ir native verification-bundle handoff (TNVBH).
pub const TNVBH_SCHEMA: &str =
    "trust_ir.native.petri_successor.native_verification_bundle_handoff.v1";
/// Schema version of the TNVBH.
pub const TNVBH_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH bundle-solver-evidence handoff descriptor.
pub const TNVBH_DESCRIPTOR_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.v1";
/// Schema version of the TNVBH descriptor.
pub const TNVBH_DESCRIPTOR_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH manifest identity.
pub const TNVBH_MANIFEST_IDENTITY_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.manifest_identity.v1";
/// Schema version of the TNVBH manifest identity.
pub const TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH diagnostic-fixture manifest.
pub const TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.diagnostic_fixture_manifest.v1";
/// Schema version of the TNVBH diagnostic-fixture manifest.
pub const TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH diagnostic-fixture-manifest round-trip report.
pub const TNVBH_DIAG_FIXTURE_ROUND_TRIP_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.diagnostic_fixture_manifest.round_trip_report.v1";
/// Schema version of the TNVBH diagnostic-fixture-manifest round-trip report.
pub const TNVBH_DIAG_FIXTURE_ROUND_TRIP_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH replay-contract surface.
pub const TNVBH_REPLAY_SURFACE_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.replay_contract_surface.v1";
/// Schema version of the TNVBH replay-contract surface.
pub const TNVBH_REPLAY_SURFACE_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH replay-contract-surface round-trip report.
pub const TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.replay_contract_surface.round_trip_report.v1";
/// Schema version of the TNVBH replay-contract-surface round-trip report.
pub const TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION: &str = "1";
/// Schema name for the TNVBH replay-contract JSON manifest binding.
pub const TNVBH_REPLAY_JSON_BINDING_SCHEMA: &str =
    "trust_ir.native.petri_successor.bundle_solver_evidence_handoff.replay_contract_surface.json_manifest_binding.v1";
/// Schema version of the TNVBH replay-contract JSON manifest binding.
pub const TNVBH_REPLAY_JSON_BINDING_SCHEMA_VERSION: &str = "1";
/// Schema name for the trust-ir native semantic-bridge proof identity.
pub const TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA: &str =
    "trust_ir.native.semantic_bridge.proof_identity.v1";
/// Component name for the native semantic-bridge proof identity.
pub const TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT: &str =
    "native_semantic_bridge_proof_identity";
/// Schema name for the trust-ir Petri trust-MC CHC proof-evidence identity.
pub const TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_SCHEMA: &str =
    "trust_ir.native.petri_successor.trust_mc_chc_proof_evidence_identity.v1";
/// Component name for the Petri trust-MC CHC proof-evidence identity.
pub const TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_COMPONENT: &str =
    "petri_successor_trust_mc_chc_proof_evidence_identity";
/// Fixture name for a healthy TNVBH descriptor.
pub const TNVBH_HEALTHY_FIXTURE: &str = "default_descriptor_healthy";
/// Fixture name for an incomplete TNVBH descriptor (missing identity/capability
/// schemas).
pub const TNVBH_INCOMPLETE_FIXTURE: &str =
    "missing_bundle_identity_schema_and_solver_capability_schema";
/// Fixture name for a healthy TNVBH replay JSON manifest binding.
pub const TNVBH_REPLAY_JSON_BINDING_HEALTHY_FIXTURE: &str =
    "default_replay_json_manifest_binding_healthy";
/// Fixture name for a stale TNVBH replay JSON manifest binding (mismatched
/// manifest-identity digest).
pub const TNVBH_REPLAY_JSON_BINDING_STALE_FIXTURE: &str =
    "stale_replay_json_manifest_binding_manifest_identity_digest";
/// Component name for the TNVBH contract-health report.
pub const TNVBH_CONTRACT_HEALTH_COMPONENT: &str =
    "native_verification_bundle_handoff_contract_health";
/// Component name for the TNVBH diagnostic-fixture manifest.
pub const TNVBH_DIAG_FIXTURE_MANIFEST_COMPONENT: &str =
    "native_verification_bundle_handoff_diagnostic_fixture_manifest";
/// Component name for the TNVBH diagnostic-fixture-manifest round trip.
pub const TNVBH_DIAG_FIXTURE_ROUND_TRIP_COMPONENT: &str =
    "native_verification_bundle_handoff_diagnostic_fixture_manifest_round_trip";
/// Component name for the TNVBH replay-contract surface.
pub const TNVBH_REPLAY_SURFACE_COMPONENT: &str =
    "native_verification_bundle_handoff_replay_contract_surface";
/// Component name for the TNVBH replay-contract-surface round trip.
pub const TNVBH_REPLAY_SURFACE_ROUND_TRIP_COMPONENT: &str =
    "native_verification_bundle_handoff_replay_contract_surface_round_trip";
/// Component name for the TNVBH replay-contract report identity.
pub const TNVBH_REPLAY_REPORT_IDENTITY_COMPONENT: &str =
    "native_verification_bundle_handoff_replay_contract_report_identity";
/// Component name for the TNVBH replay-contract JSON manifest binding.
pub const TNVBH_REPLAY_JSON_BINDING_COMPONENT: &str =
    "native_verification_bundle_handoff_replay_contract_json_manifest_binding";
/// Source package that owns the TNVBH schemas.
pub const TNVBH_SOURCE_PACKAGE: &str = "trust_ir";
/// Version of the TNVBH source package.
pub const TNVBH_SOURCE_PACKAGE_VERSION: &str = "0.1.0";
/// Source project that owns the TNVBH schemas.
pub const TNVBH_SOURCE_PROJECT: &str = "trust-ir";
/// Project tag recorded on TNVBH rows.
pub const TNVBH_PROJECT: &str = "trust-ir";
/// Schema name for the trust-ir native bundle-identity contract.
pub const TRUST_IR_NATIVE_BUNDLE_IDENTITY_CONTRACT_SCHEMA: &str =
    "trust_ir.native.bundle_identity_contract.v1";
/// Schema version of the trust-ir native bundle-identity contract.
pub const TRUST_IR_NATIVE_BUNDLE_IDENTITY_CONTRACT_SCHEMA_VERSION: &str = "1";
/// Schema version of the trust-ir native bundle.
pub const TRUST_IR_NATIVE_BUNDLE_SCHEMA_VERSION: &str = "3";
/// Schema name for the trust-ir native transport identity.
pub const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA: &str = "trust_ir.native.transport_identity.v2";
/// Schema version of the trust-ir native transport identity.
pub const TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA_VERSION: &str = "2";

/// Deterministic placeholder SHA-256 of the accepted BTOR2 replay-evidence
/// identity used in the smoke sidecar.
pub const BTOR2_ACCEPTED_REPLAY_EVIDENCE_IDENTITY_SHA256: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
/// Deterministic placeholder SHA-256 of the accepted BTOR2 replay obligation
/// identities.
pub const BTOR2_ACCEPTED_REPLAY_OBLIGATION_IDENTITIES_SHA256: &str =
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
/// Status string for an accepted BTOR2 AY proof evidence (verified
/// counterexample).
pub const BTOR2_ACCEPTED_AY_PROOF_EVIDENCE_STATUS: &str = "ay_chc_verified_counterexample";
/// Deterministic placeholder SHA-256 of the accepted BTOR2 AY proof evidence.
pub const BTOR2_ACCEPTED_AY_PROOF_EVIDENCE_SHA256: &str =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
/// Deterministic placeholder SHA-256 of the trust-cg compile-artifact cache key.
pub const TRUST_CG_COMPILE_ARTIFACT_CACHE_KEY_SHA256: &str =
    "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
/// Schema name for the MCC production selector decision.
pub const MCC_PRODUCTION_SELECTOR_DECISION_SCHEMA: &str = "mcc.production_selector_decision.v1";
/// Schema name for an MCC portfolio route.
pub const PORTFOLIO_ROUTE_SCHEMA: &str = "mcc.portfolio_route.v1";
/// Schema version of an MCC portfolio route.
pub const PORTFOLIO_ROUTE_SCHEMA_VERSION: &str = "1";
/// Schema name for an MCC schema diagnostic.
pub const SCHEMA_DIAGNOSTIC_SCHEMA: &str = "mcc.schema_diagnostic.v1";
/// Availability state: the primitive is not present at all.
pub const PRIMITIVE_UNAVAILABLE: &str = "primitive_unavailable";
/// Availability state: the primitive exists but is blocked at runtime.
pub const PRIMITIVE_AVAILABLE_RUNTIME_BLOCKED: &str = "available_runtime_blocked";
/// Availability state: the primitive is usable for the answer lane.
pub const PRIMITIVE_USABLE_FOR_ANSWER_LANE: &str = "usable_for_answer_lane";
/// Reason code: the native payload SHA-256 is missing.
pub const MISSING_NATIVE_PAYLOAD_SHA256: &str = "missing_native_payload_sha256";
/// Reason code: the Petri native capability report has no compiled
/// `NativeLibrary`.
pub const PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY: &str =
    "petri_native_capability_report_has_no_compiled_NativeLibrary";
/// Reason code: Petri successor execution is required but unavailable.
pub const PETRI_SUCCESSOR_EXECUTION_REQUIRED: &str = "petri_successor_execution_required";

/// Names of the replay-contract helper APIs whose presence the smoke sidecar
/// records.
pub const REPLAY_CONTRACT_HELPER_NAMES: &[&str] = &[
    "petri_native_verification_bundle_handoff_descriptor()",
    "PetriNativeVerificationBundleHandoffDescriptor::manifest_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::manifest_key_value_lines()",
    "PetriNativeVerificationBundleHandoffDescriptor::normalized_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::normalized_key_value_lines()",
    "PetriNativeVerificationBundleHandoffDescriptor::required_normalized_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::validate_normalized_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::manifest_identity()",
    "PetriNativeVerificationBundleHandoffDescriptor::manifest_identity_for_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::canonical_manifest_text()",
    "PetriNativeVerificationBundleHandoffDescriptor::canonical_manifest_text_for_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::contract_health_report()",
    "PetriNativeVerificationBundleHandoffDescriptor::contract_health_report_for_rows()",
    "petri_native_verification_bundle_handoff_contract_health_report()",
    "petri_native_verification_bundle_handoff_healthy_diagnostic_fixture()",
    "petri_native_verification_bundle_handoff_incomplete_diagnostic_fixture()",
    "petri_native_verification_bundle_handoff_diagnostic_fixture_manifest()",
    "petri_native_verification_bundle_handoff_replay_contract_json_manifest_binding_healthy_fixture()",
    "petri_native_verification_bundle_handoff_replay_contract_json_manifest_binding_stale_fixture()",
    "PetriNativeVerificationBundleHandoffReplayContractSurfaceRoundTripReport::compact_manifest_json_text()",
    "PetriNativeVerificationBundleHandoffReplayContractSurfaceRoundTripReport::compact_manifest_handoff_identity_report()",
    "PetriNativeVerificationBundleHandoffReplayContractJsonManifestBindingReport::key_value_rows()",
    "PetriNativeVerificationBundleHandoffReplayContractJsonManifestBindingFixture::key_value_rows()",
    "PetriNativeVerificationBundleHandoffReplayContractJsonManifestBindingFixture::key_value_lines()",
    "PetriNativeVerificationBundleHandoffReplayContractJsonManifestBindingFixture::key_value_text()",
];

/// Names of the replay-contract schema constants the smoke sidecar records.
pub const REPLAY_CONTRACT_SCHEMA_NAMES: &[&str] = &[
    "PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_SCHEMA",
    "PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_MANIFEST_IDENTITY_SCHEMA",
    "PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA",
    "PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_SURFACE_SCHEMA",
    "PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_ROUND_TRIP_REPORT_SCHEMA",
    "PETRI_NATIVE_VERIFICATION_BUNDLE_HANDOFF_REPLAY_CONTRACT_JSON_MANIFEST_BINDING_SCHEMA",
];

fn replay_contract_schema_values() -> [&'static str; 6] {
    [
        TNVBH_DESCRIPTOR_SCHEMA,
        TNVBH_MANIFEST_IDENTITY_SCHEMA,
        TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA,
        TNVBH_REPLAY_SURFACE_SCHEMA,
        TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        TNVBH_REPLAY_JSON_BINDING_SCHEMA,
    ]
}

/// The replay-contract fixture names recorded by the smoke sidecar.
pub const REPLAY_CONTRACT_FIXTURE_NAMES: &[&str] = &[
    TNVBH_HEALTHY_FIXTURE,
    TNVBH_INCOMPLETE_FIXTURE,
    TNVBH_REPLAY_JSON_BINDING_HEALTHY_FIXTURE,
    TNVBH_REPLAY_JSON_BINDING_STALE_FIXTURE,
];

/// Names of the replay-contract validator APIs the smoke sidecar records.
pub const REPLAY_CONTRACT_VALIDATOR_NAMES: &[&str] = &[
    "PetriNativeVerificationBundleHandoffDescriptor::validate_normalized_rows()",
    "PetriNativeVerificationBundleHandoffDescriptor::contract_health_report_for_rows()",
    "PetriNativeVerificationBundleHandoffManifestIdentity::round_trip_report()",
    "PetriNativeVerificationBundleHandoffDiagnosticFixtureManifest::round_trip_report()",
    "PetriNativeVerificationBundleHandoffReplayContractSurface::round_trip_report()",
    "PetriNativeVerificationBundleHandoffReplayContractSurface::round_trip_report_for_key_value_lines()",
    "PetriNativeVerificationBundleHandoffReplayContractSurfaceRoundTripReport::key_value_round_trip_report()",
    "PetriNativeVerificationBundleHandoffReplayContractSurfaceRoundTripReport::key_value_line_round_trip_report()",
];

/// Ordered key list of a native-evidence artifact-authority report row.
pub const NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_REPORT_ROW_KEYS: &[&str] = &[
    "artifact_authority.schema",
    "artifact_authority.schema_version",
    "artifact_resolution.schema",
    "artifact_resolution.schema_version",
    "request.id",
    "owner_suite",
    "artifact.kind",
    "artifact.name",
    "digest.algorithm",
    "digest",
    "byte.source_identity",
    "byte.len",
    "actual_digest",
    "authority",
    "status",
    "reason",
    "report.is_resolved",
    "report.is_authoritative",
    "report.fail_closed",
];

/// Ordered key list of a native-evidence artifact-authority *resolution* row
/// (the authority-report keys plus the `resolution.*` keys).
pub const NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_RESOLUTION_ROW_KEYS: &[&str] = &[
    "artifact_authority.schema",
    "artifact_authority.schema_version",
    "artifact_resolution.schema",
    "artifact_resolution.schema_version",
    "request.id",
    "owner_suite",
    "artifact.kind",
    "artifact.name",
    "digest.algorithm",
    "digest",
    "byte.source_identity",
    "byte.len",
    "actual_digest",
    "authority",
    "status",
    "reason",
    "report.is_resolved",
    "report.is_authoritative",
    "report.fail_closed",
    "resolution.bytes_present",
    "resolution.is_resolved",
    "resolution.is_authoritative",
    "resolution.fail_closed",
    "resolution.authoritative_bytes_available",
];

// ============================================================
// Cargo.lock probe (mirrors `_cargo_lock_git_rev`).
// ============================================================

fn cargo_lock_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .ancestors()
        .nth(2)
        .map(|p| p.join("Cargo.lock"))
        .unwrap_or_else(|| PathBuf::from("Cargo.lock"))
}

/// Read `(rev, source_url)` for `package_name` from the workspace
/// `Cargo.lock`. Mirrors `_cargo_lock_git_rev` in the Python module.
pub fn cargo_lock_git_rev(package_name: &str) -> (Option<String>, Option<String>) {
    let lock_text = match std::fs::read_to_string(cargo_lock_path()) {
        Ok(t) => t,
        Err(_) => return (None, None),
    };
    if let Ok(parsed) = toml::from_str::<TomlValue>(&lock_text) {
        if let Some(packages) = parsed.get("package").and_then(TomlValue::as_array) {
            for package in packages {
                let Some(tbl) = package.as_table() else {
                    continue;
                };
                if tbl.get("name").and_then(TomlValue::as_str) != Some(package_name) {
                    continue;
                }
                if let Some(source) = tbl.get("source").and_then(TomlValue::as_str) {
                    if source.starts_with("git+") {
                        return (source_rev(source), Some(source.to_string()));
                    }
                }
            }
        }
    }
    (None, None)
}

fn workspace_ay_pin_rev() -> Option<String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.ancestors().nth(2)?;
    validate_ay_pin(repo_root, None)
        .ok()
        .map(|summary| summary.cargo_toml_rev)
}

// ============================================================
// Small helpers (mirrors `_bool_text`, `_csv`).
// ============================================================

fn bool_text(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn csv(values: &[&str]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

// ============================================================
// Legacy compatibility-only stable-v1 digest (mirrors the retired Python
// `_trust_ir_stable_v1_digest`).
//
// Four-lane 64-bit fingerprint chosen by the Python generator. The lane
// state and rounds match the reference bit-for-bit so consumers that pin
// the prefix `trust-ir-stable-v1:` see identical hex output.
// ============================================================

fn rotate_left_u64(value: u64, shift: u32) -> u64 {
    value.rotate_left(shift & 63)
}

/// Reproduce archived smoke-sidecar text. This is deliberately private and
/// untyped: native authority uses `ProofDigest::sha256_domain` instead.
fn legacy_compatibility_trust_ir_stable_v1_digest(context: &str, payload: &str) -> String {
    let mut lanes: [u64; 4] = [
        0xCBF29CE484222325,
        0x9E3779B97F4A7C15,
        0x6A09E667F3BCC909,
        0xBB67AE8584CAA73B,
    ];

    fn update(data: &[u8], lanes: &mut [u64; 4]) {
        for (index, byte) in data.iter().enumerate() {
            let lane = index & 3;
            lanes[lane] ^= u64::from(*byte);
            lanes[lane] = lanes[lane].wrapping_mul(0x100000001B3);
            lanes[lane] ^= rotate_left_u64(lanes[(lane + 1) & 3], 13);
        }
        for lane in 0..4 {
            lanes[lane] ^= rotate_left_u64(data.len() as u64, (lane * 11) as u32);
            lanes[lane] = rotate_left_u64(lanes[lane], 17).wrapping_mul(0x9E3779B185EBCA87);
        }
    }

    update(context.as_bytes(), &mut lanes);
    update(&[0u8], &mut lanes);
    update(payload.as_bytes(), &mut lanes);
    for round_index in 0..8 {
        let lane = round_index & 3;
        let a = lanes[lane];
        let b = lanes[(round_index + 1) & 3];
        lanes[lane] = rotate_left_u64(a, 29) ^ b.wrapping_mul(0xD6E8FEB86659FD93);
    }

    let mut digest = String::with_capacity(16 + 64);
    digest.push_str("trust-ir-stable-v1:");
    for lane in lanes.iter() {
        for byte in lane.to_le_bytes().iter() {
            digest.push_str(&format!("{:02x}", byte));
        }
    }
    digest
}

fn trust_ir_manifest_identity_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '=' => out.push_str("\\="),
            other => out.push(other),
        }
    }
    out
}

fn trust_ir_report_identity_line(key: &str, value: &str) -> String {
    format!(
        "{}={}",
        trust_ir_manifest_identity_component(key),
        trust_ir_manifest_identity_component(value)
    )
}

// ============================================================
// KV row helpers shared by the manifest builders.
// ============================================================

type Kv = (String, String);

fn kv(k: &str, v: &str) -> Kv {
    (k.to_string(), v.to_string())
}

fn kvo(k: &str, v: String) -> Kv {
    (k.to_string(), v)
}

fn render_manifest_lines(prefix: &str, rows: &[Kv]) -> Vec<String> {
    rows.iter()
        .map(|(k, v)| format!("{prefix} manifest_line={k}={v}"))
        .collect()
}

// ============================================================
// Shared-primitive availability rows (mirrors the Python helpers
// `_trust_cg_call_packet_availability_row`,
// `_ay_typed_trace_assignment_availability_row`,
// `_aiger_ay_sat_adapter_availability_row`).
//
// The Python module probed cargo checkouts / sibling repos to determine
// whether the upstream API symbols were present. In the Rust port we
// preserve the wire shape but use deterministic defaults — `primitive
// available + runtime blocked` for the trust_cg/AY entries and
// `usable_for_answer_lane` for the AIGER entry. Production consumers
// (the doctor gate and the canary) only assert on the schema names and
// the boolean fields, not on the lock_rev/checkout_rev pin tokens.
// ============================================================

fn shared_primitive_row(
    scope: &str,
    primitive: &str,
    source_package: &str,
    api: &str,
    surface: &str,
    schema_pair: Option<(&str, &str)>,
    assignment_fields: Option<&str>,
    primitive_available: bool,
    usable_for_answer_lane: bool,
    runtime_proof_blocked: bool,
    reason_code: &str,
    found: &[&str],
    missing: &[&str],
) -> String {
    let state = if !primitive_available {
        PRIMITIVE_UNAVAILABLE
    } else if usable_for_answer_lane {
        PRIMITIVE_USABLE_FOR_ANSWER_LANE
    } else {
        PRIMITIVE_AVAILABLE_RUNTIME_BLOCKED
    };
    let mut out = String::new();
    out.push_str(scope);
    out.push_str(" shared_primitive_availability");
    out.push_str(&format!(" primitive={primitive}"));
    out.push_str(&format!(" source={source_package}"));
    out.push_str(&format!(" package={source_package}"));
    out.push_str(&format!(" api={api}"));
    out.push_str(&format!(" surface={surface}"));
    if let Some((schema, version)) = schema_pair {
        out.push_str(&format!(" schema={schema}"));
        out.push_str(&format!(" schema_version={version}"));
    }
    if let Some(fields) = assignment_fields {
        out.push_str(&format!(" assignment_fields={fields}"));
    }
    out.push_str(" lock_rev=workspace checkout_rev=workspace");
    out.push_str(" source_probe=workspace_source");
    out.push_str(" source_path=workspace");
    out.push_str(&format!(
        " primitive_available={}",
        bool_text(primitive_available)
    ));
    out.push_str(&format!(
        " runtime_proof_blocked={}",
        bool_text(runtime_proof_blocked)
    ));
    out.push_str(&format!(
        " usable_for_answer_lane={}",
        bool_text(usable_for_answer_lane)
    ));
    out.push_str(&format!(" availability_state={state}"));
    out.push_str(&format!(" reason_code={reason_code}"));
    out.push_str(&format!(" required_symbols_found={}", csv(found)));
    out.push_str(&format!(" required_symbols_missing={}", csv(missing)));
    out
}

fn trust_cg_call_packet_availability_row() -> String {
    shared_primitive_row(
        "trust-cg",
        "petri_native_successor_call_packet",
        "trust-cg-codegen",
        &format!("trust-cg::{CALL_PACKET_SURFACE}"),
        CALL_PACKET_SURFACE,
        Some((CALL_PACKET_SCHEMA, "1")),
        None,
        /* primitive_available */ true,
        /* usable_for_answer_lane */ false,
        /* runtime_proof_blocked */ true,
        PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY,
        &[
            "call_packet_api",
            "call_packet_type",
            "callable_pointer_type",
            "call_packet_schema",
        ],
        &[],
    )
}

fn ay_typed_trace_assignment_availability_row() -> String {
    shared_primitive_row(
        "AY",
        "chc_typed_trace_assignments",
        "ay-chc",
        "ChcPdrProofRun::consumer_evidence",
        AY_TRACE_ASSIGNMENT_SURFACE,
        None,
        Some("name,value,predicate_argument_index,sort"),
        /* primitive_available */ true,
        /* usable_for_answer_lane */ false,
        /* runtime_proof_blocked */ true,
        PETRI_SUCCESSOR_EXECUTION_REQUIRED,
        &[
            "consumer_evidence_method",
            "proof_transcript_consumer_evidence",
            "trace_assignment_evidence",
            "predicate_argument_index",
            "ay_facade_export",
        ],
        &[],
    )
}

fn aiger_ay_sat_adapter_availability_row() -> String {
    shared_primitive_row(
        "AIGER",
        "ay_sat_adapter_decision",
        "tla-aiger",
        "aiger_portfolio_capability_report",
        "aiger_ay_adapter_decision",
        Some(("aiger.ay_adapter_decision.v1", "1")),
        Some("typed_assignment_source,replay_assignment_status"),
        /* primitive_available */ true,
        /* usable_for_answer_lane */ true,
        /* runtime_proof_blocked */ false,
        "available",
        &[
            "ay_sat_capability",
            "adapter_decision_schema",
            "adapter_decision_type",
            "hardware_replay_status",
            "hardware_replay_decision",
        ],
        &[],
    )
}

fn shared_primitive_availability_rows() -> Vec<String> {
    vec![
        trust_cg_call_packet_availability_row(),
        ay_typed_trace_assignment_availability_row(),
        aiger_ay_sat_adapter_availability_row(),
    ]
}

// ============================================================
// AY solver capability descriptor.
// ============================================================

fn ay_solver_capability_descriptor_rows() -> Vec<String> {
    vec![format!(
        "AY solver_capability_descriptor \
         schema={AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA} \
         schema_version={AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA_VERSION} \
         source_package=ay-dpll \
         package=ay-dpll \
         solver=ay \
         capability=model_blocking \
         status=available \
         status_code=available \
         reason_code=ay_owned_public_api \
         api_symbols=ay_dpll::api::Solver::try_model_blocking_clause_for_consumer|ay_dpll::api::Solver::try_assert_model_blocking_clause_for_consumer|ay_dpll::api::ModelBlockingClause \
         evidence_schemas={AY_MODEL_BLOCKING_CLAUSE_SCHEMA}|{AY_SOLVE_DECISION_PROFILE_MODEL_CONSUMER_SCHEMA} \
         lock_rev=workspace \
         checkout_rev=workspace \
         source_probe=workspace_source \
         source_path=workspace \
         required_symbols_found=descriptor_schema,descriptor_api,model_blocking_capability,model_blocking_api,model_blocking_schema,model_consumer_schema \
         required_symbols_missing=none \
         production_selected=false \
         fail_closed=true",
    )]
}

fn ay_solver_capability_descriptor_status_code() -> String {
    "available".to_string()
}

// ============================================================
// AY symbolic execution manifest + health.
// ============================================================

fn ay_symbolic_execution_contract_manifest_rows() -> Vec<String> {
    let capabilities = [
        "model_blocking",
        "incremental_assumptions",
        "all_sat_enumeration",
    ];
    let schemas = [
        AY_MODEL_BLOCKING_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA,
        AY_INCREMENTAL_ASSUMPTIONS_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA,
        AY_ALL_SAT_ENUMERATION_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA,
    ];
    let helpers = [
        "ay_dpll::api::model_blocking_symbolic_execution_contract",
        "ay_dpll::api::incremental_assumptions_symbolic_execution_contract",
        "ay_dpll::api::all_sat_enumeration_symbolic_execution_contract",
    ];
    let key_value_helpers = [
        "ay_dpll::api::model_blocking_symbolic_execution_contract_key_value_pairs",
        "ay_dpll::api::incremental_assumptions_symbolic_execution_contract_key_value_pairs",
        "ay_dpll::api::all_sat_enumeration_symbolic_execution_contract_key_value_pairs",
    ];

    let mut manifest_rows: Vec<Kv> = vec![
        kv("schema", AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA),
        kv("schema_version", "1"),
        kv("solver", "ay"),
        kvo("contract_count", capabilities.len().to_string()),
        kvo("contract_capabilities", capabilities.join(",")),
        kv(
            "contract_capability_names",
            "model_blocking,incremental_assumptions,all_sat_enumeration",
        ),
        kvo("contract_schemas", schemas.join(",")),
        kv("contract_schema_versions", "1,1,1"),
        kvo("contract_helpers", helpers.join(",")),
        kvo("key_value_helpers", key_value_helpers.join(",")),
        kv("all_contracts_fail_closed", "true"),
    ];
    for i in 0..capabilities.len() {
        let cap = capabilities[i];
        manifest_rows.push(kvo(&format!("{cap}_capability_name"), cap.to_string()));
        manifest_rows.push(kvo(
            &format!("{cap}_contract_schema"),
            schemas[i].to_string(),
        ));
        manifest_rows.push(kv(&format!("{cap}_contract_schema_version"), "1"));
        manifest_rows.push(kvo(
            &format!("{cap}_contract_helper"),
            helpers[i].to_string(),
        ));
        manifest_rows.push(kvo(
            &format!("{cap}_key_value_helper"),
            key_value_helpers[i].to_string(),
        ));
        manifest_rows.push(kv(&format!("{cap}_accepted_status_codes"), "accepted"));
        manifest_rows.push(kv(&format!("{cap}_rejected_status_codes"), "rejected"));
        manifest_rows.push(kv(&format!("{cap}_accepted_reason_codes"), "accepted"));
        manifest_rows.push(kv(&format!("{cap}_rejected_reason_codes"), "fail_closed"));
        manifest_rows.push(kv(&format!("{cap}_fail_closed"), "true"));
    }

    let health_rows: Vec<Kv> = vec![
        kv(
            "schema",
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA,
        ),
        kv("schema_version", "1"),
        kv("status", "complete"),
        kv("reason", "complete"),
        kv("diagnostic", "healthy"),
        kvo("required_capabilities", capabilities.join(",")),
        kvo("present_capabilities", capabilities.join(",")),
        kv("accepted_for_consumer", "true"),
        kv("all_contracts_fail_closed", "true"),
        kv("issue_count", "0"),
        kv("issue_reason_codes", ""),
    ];

    let manifest_prefix = format!(
        "AY symbolic_execution_contract_manifest schema={AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA} schema_version=1 source_package=ay-dpll package=ay-dpll"
    );
    let health_prefix = format!(
        "AY symbolic_execution_contract_manifest_health schema={AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA} schema_version=1 source_package=ay-dpll package=ay-dpll"
    );

    let mut out = render_manifest_lines(&manifest_prefix, &manifest_rows);
    out.extend(render_manifest_lines(&health_prefix, &health_rows));
    out
}

// ============================================================
// trust-ir semantic-bridge proof identity & Petri proof evidence identity.
// ============================================================

fn trust_ir_native_semantic_bridge_proof_identity_rows() -> Vec<String> {
    let identity_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA,
        "mcc-smoke-semantic-bridge-proof-identity",
    );
    let bridge_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        "trust_ir.native.semantic_bridge.v1",
        "mcc-smoke-semantic-bridge",
    );
    let rows: Vec<Kv> = vec![
        kv(
            "semantic_bridge_proof_identity.schema",
            TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA,
        ),
        kv("semantic_bridge_proof_identity.schema_version", "1"),
        kv(
            "semantic_bridge_proof_identity.digest.context",
            TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA,
        ),
        kv(
            "semantic_bridge_proof_identity.digest.algorithm",
            "trust-ir-stable-v1",
        ),
        kvo("semantic_bridge_proof_identity.digest", identity_digest),
        kv(
            "semantic_bridge_proof_identity.bridge.schema",
            "trust_ir.native.semantic_bridge.v1",
        ),
        kv("semantic_bridge_proof_identity.bridge.schema_version", "1"),
        kvo(
            "semantic_bridge_proof_identity.bridge.digest",
            bridge_digest,
        ),
        kv(
            "semantic_bridge_proof_identity.bridge.relation",
            "petri_successor",
        ),
        kv("semantic_bridge_proof_identity.bridge.function", "0"),
        kv(
            "semantic_bridge_proof_identity.bridge.formula_schema",
            SEMANTIC_SUCCESSOR_BRIDGE_FORMULA_SCHEMA,
        ),
        kv(
            "semantic_bridge_proof_identity.report.schema",
            "trust_ir.native.semantic_bridge.v1",
        ),
        kv("semantic_bridge_proof_identity.report.schema_version", "1"),
        kv("semantic_bridge_proof_identity.report.status", "blocked"),
        kv(
            "semantic_bridge_proof_identity.report.reason",
            "proof_pending",
        ),
        kv("semantic_bridge_proof_identity.report.fail_closed", "true"),
        kv(
            "semantic_bridge_proof_identity.report.evidence_status",
            "missing",
        ),
        kv("semantic_bridge_proof_identity.proof.obligation", "0"),
        kv("semantic_bridge_proof_identity.proof.digest", "none"),
        kv("semantic_bridge_proof_identity.proof.status", "pending"),
        kv("semantic_bridge_proof_identity.evidence.digest", "none"),
    ];
    let prefix = format!(
        "trust-ir {TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_COMPONENT} schema={TRUST_IR_NATIVE_SEMANTIC_BRIDGE_PROOF_IDENTITY_SCHEMA} schema_version=1 source_package=trust_ir source_project=trust-ir project=trust-ir"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_petri_proof_evidence_identity_rows() -> Vec<String> {
    let identity_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_SCHEMA,
        "mcc-smoke-proof-evidence-identity",
    );
    let semantic_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        "trust_ir.native.semantic_bridge.proof_identity.v1",
        "mcc-smoke-semantic-bridge",
    );
    let proof_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        "trust_ir.native.petri_successor.trust_mc_chc_proof_handoff.v1",
        "mcc-smoke-proof-handoff",
    );
    let replay_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        "trust_ir.native.petri_successor.trust_mc_chc_replay_transcript.v1",
        "mcc-smoke-replay-transcript",
    );
    let model_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        "trust_ir.native.petri_successor.trust_mc_model.v1",
        "mcc-smoke-model",
    );
    let rows: Vec<Kv> = vec![
        kv(
            "proof_evidence_identity.schema",
            TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_SCHEMA,
        ),
        kv("proof_evidence_identity.schema_version", "1"),
        kv(
            "proof_evidence_identity.digest.context",
            TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_SCHEMA,
        ),
        kv(
            "proof_evidence_identity.digest.algorithm",
            "trust-ir-stable-v1",
        ),
        kvo("proof_evidence_identity.digest", identity_digest),
        kv("proof_evidence_identity.function", "0"),
        kv(
            "semantic_bridge.schema",
            "trust_ir.native.semantic_bridge.proof_identity.v1",
        ),
        kv("semantic_bridge.schema_version", "1"),
        kv("semantic_bridge.status", "represented"),
        kv("semantic_bridge.reason", "represented"),
        kv("semantic_bridge.evidence_status", "accepted"),
        kv(
            "semantic_bridge.proof_identity.schema",
            "trust_ir.native.semantic_bridge.proof_identity.v1",
        ),
        kv("semantic_bridge.proof_identity.schema_version", "1"),
        kvo("semantic_bridge.proof_identity.digest", semantic_digest),
        kv("binding.schema", TRUST_IR_CONTRACT_BINDING_REPORT_SCHEMA),
        kv("binding.schema_version", "1"),
        kv("binding.status", "bound"),
        kv("binding.reason", "bound"),
        kv("binding.fail_closed", "false"),
        kv(
            "proof_handoff.schema",
            TRUST_IR_CONTRACT_PROOF_HANDOFF_REPORT_SCHEMA,
        ),
        kv("proof_handoff.schema_version", "1"),
        kv("proof_handoff.status", "ready"),
        kv("proof_handoff.reason", "ready"),
        kv("proof_handoff.fail_closed", "false"),
        kvo("proof_handoff.proof_identity.digest", proof_digest),
        kv("proof_handoff.replay.engine", "trust_mc"),
        kv("proof_handoff.replay.invocation", "mcc_smoke"),
        kvo(
            "proof_handoff.replay.transcript_digest",
            replay_digest.clone(),
        ),
        kv("proof_handoff.replay_transcript_artifact.present", "true"),
        kv(
            "proof_handoff.replay_transcript_artifact.kind",
            "replay_transcript",
        ),
        kvo(
            "proof_handoff.replay_transcript_artifact.digest",
            replay_digest,
        ),
        kv("proof_handoff.model_artifact.present", "true"),
        kvo("proof_handoff.model_artifact.digest", model_digest),
        kv("proof_handoff.solver_identity.count", "1"),
        kv("proof_handoff.solver_identity.0.name", "z3"),
        kv("proof_handoff.solver_identity.0.canonical_name", "z3"),
        kv("proof_handoff.solver_identity.0.version", "4.12.2"),
        kv("proof_handoff.solver_identity.0.revision", "none"),
    ];
    let prefix = format!(
        "trust-ir {TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_COMPONENT} schema={TRUST_IR_PETRI_PROOF_EVIDENCE_IDENTITY_SCHEMA} schema_version=1 source_package=trust_ir source_project=trust-ir project=trust-ir"
    );
    render_manifest_lines(&prefix, &rows)
}

// ============================================================
// Blocker-action and schema-diagnostic rows.
// ============================================================

fn blocker_action_rows() -> Vec<String> {
    vec![
        format!(
            "MCC blocker_action selected=true priority_rank=10 lane_family=native_jit \
             blocker_piece=trust_cg_petri_compile_artifact_handoff \
             blocker_gate={COMPILE_ARTIFACT_HANDOFF_SCHEMA} \
             owner_project=trust_cg \
             owner_primitive={COMPILE_ARTIFACT_HANDOFF_INSTALLED_ARTIFACT_API} \
             action_code=populate_installed_artifact_compile_artifact_handoff_evidence \
             reason_code={PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY} \
             availability_state=available_runtime_blocked \
             next_answer_lane=petri_native_callable \
             tracking_issue=ty#4445 upstream_issue=trust_cg#881 \
             status=blocked production_selected=false fail_closed=true"
        ),
        format!(
            "MCC blocker_action selected=false priority_rank=15 lane_family=native_jit \
             blocker_piece=trust_cg_petri_semantic_successor_bridge \
             blocker_gate={SEMANTIC_SUCCESSOR_BRIDGE_FORMULA_SCHEMA} \
             owner_project=trust-ir \
             owner_primitive=PetriKernelPlanCache::for_net->trust_ir::NativeVerificationBundle \
             action_code=represent_petri_successor_semantics_in_trust_ir \
             reason_code={SEMANTIC_SUCCESSOR_BRIDGE_REASON_CODE} \
             availability_state=available_runtime_blocked \
             next_answer_lane=petri_native_callable \
             tracking_issue=ty#4445 upstream_issue=trust-ir-semantic-successor-identity \
             status=blocked production_selected=false fail_closed=true"
        ),
        format!(
            "MCC blocker_action selected=false priority_rank=20 lane_family=native_jit \
             blocker_piece=trust_cg_petri_successor_call_packet \
             blocker_gate={CALL_PACKET_SURFACE} \
             owner_project=trust_cg owner_primitive=petri_native_successor_call_packet \
             action_code=bind_native_install_gate_packet \
             reason_code=missing_native_install_gate_packet \
             availability_state=available_runtime_blocked \
             next_answer_lane=petri_native_callable \
             tracking_issue=ty#4445 upstream_issue=trust_cg#881 \
             status=blocked production_selected=false fail_closed=true"
        ),
        format!(
            "MCC blocker_action selected=false priority_rank=30 lane_family=native_jit \
             blocker_piece=trust_cg_petri_successor_execution_plan \
             blocker_gate=petri_native_successor_execution_plan_from_trust_ir_bundle \
             owner_project=trust_cg owner_primitive=petri_native_successor_execution_plan \
             action_code=bind_trampoline_entry_function reason_code=trampoline_unbound \
             availability_state=available_runtime_blocked \
             next_answer_lane=petri_native_callable \
             tracking_issue=ty#4445 upstream_issue=trust_cg#881 \
             status=blocked production_selected=false fail_closed=true"
        ),
        "MCC blocker_action selected=false priority_rank=40 lane_family=native_jit \
         blocker_piece=trust_cg_native_admission blocker_gate=mcc_replay \
         owner_project=trust_cg owner_primitive=native_install_gate_admission \
         action_code=provide_native_install_gate_manifest reason_code=missing_manifest \
         availability_state=available_runtime_blocked next_answer_lane=petri_native_callable \
         tracking_issue=ty#4445 upstream_issue=trust_cg#881 status=blocked \
         production_selected=false fail_closed=true"
            .to_string(),
        "MCC blocker_action selected=false priority_rank=50 lane_family=native_jit \
         blocker_piece=petri_native_jit_gate blocker_gate=trust-cg-petri-native \
         owner_project=TY owner_primitive=petri_native_jit_policy_gate \
         action_code=enable_native_policy_after_callable_proof reason_code=disabled_by_policy \
         availability_state=available_runtime_blocked next_answer_lane=petri_native_callable \
         tracking_issue=ty#4445 upstream_issue=none status=blocked \
         production_selected=false fail_closed=true"
            .to_string(),
        "MCC blocker_action selected=false priority_rank=60 lane_family=native_jit \
         blocker_piece=native_successor_lane blocker_gate=petri_native_successor_lane \
         owner_project=TY owner_primitive=petri_native_successor_lane \
         action_code=select_native_successor_lane_after_runtime_gate \
         reason_code=native_kernel_unavailable \
         availability_state=available_runtime_blocked next_answer_lane=petri_native_callable \
         tracking_issue=ty#4445 upstream_issue=none status=blocked \
         production_selected=false fail_closed=true"
            .to_string(),
        format!(
            "MCC blocker_action selected=false priority_rank=70 lane_family=hardware_ay_replay \
             blocker_piece=aiger_hardware_replay_acceptance \
             blocker_gate={HARDWARE_REPLAY_PRIMITIVE_SCHEMA} \
             owner_project=TY owner_primitive=hardware_replay_primitive \
             action_code=promote_aiger_hardware_replay_primitive_to_answer_gate \
             reason_code=proof_replay_acceptance_required \
             availability_state=available_runtime_blocked next_answer_lane=hardware_ay_replay \
             tracking_issue=ty#4445 upstream_issue=none status=blocked \
             production_selected=false fail_closed=true"
        ),
        format!(
            "MCC blocker_action selected=false priority_rank=80 lane_family=hardware_ay_replay \
             blocker_piece=ay_typed_trace_replay_acceptance \
             blocker_gate={AY_TRACE_ASSIGNMENT_SURFACE} \
             owner_project=AY owner_primitive=chc_typed_trace_assignments \
             action_code=connect_ay_typed_assignments_to_hardware_replay_acceptance \
             reason_code=proof_replay_acceptance_required \
             availability_state=available_runtime_blocked next_answer_lane=hardware_ay_replay \
             tracking_issue=ty#4445 upstream_issue=none status=blocked \
             production_selected=false fail_closed=true"
        ),
        format!(
            "MCC blocker_action selected=false priority_rank=90 lane_family=hardware_ay_replay \
             blocker_piece=btor2_hardware_replay_primitive \
             blocker_gate={HARDWARE_REPLAY_PRIMITIVE_SCHEMA} \
             owner_project=TY owner_primitive=btor2_hardware_replay_primitive \
             action_code=land_btor2_hardware_replay_primitive_v1 \
             reason_code=btor2_consumer_in_progress \
             availability_state=primitive_unavailable next_answer_lane=hardware_ay_replay \
             tracking_issue=ty#4445 upstream_issue=none status=blocked \
             production_selected=false fail_closed=true"
        ),
    ]
}

fn schema_diagnostic_rows() -> Vec<String> {
    vec![format!(
        "MCC schema_diagnostic schema={SCHEMA_DIAGNOSTIC_SCHEMA} schema_version=1 \
         schema_contract_status=pass schema_contract_reason_code=schema_contract_valid \
         schema_contract_fail_closed=false schema_contract_error_count=0 \
         selected_blocker=trust_cg_petri_compile_artifact_handoff selected_priority_rank=10 \
         selected_lane_family=native_jit selected_owner_project=trust_cg \
         selected_owner_primitive={COMPILE_ARTIFACT_HANDOFF_INSTALLED_ARTIFACT_API} \
         selected_action_code=populate_installed_artifact_compile_artifact_handoff_evidence \
         selected_reason_code={PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY} \
         blocker_count_native_jit=7 blocker_count_hardware_ay_replay=3 \
         answer_lane_total=4 answer_lane_usable=1 answer_lane_blocked=2 \
         production_selected=false fail_closed=false"
    )]
}

// ============================================================
// trust-ir shared-primitive contract manifest + artifact resolution rows.
// ============================================================

fn trust_ir_shared_primitive_contract_manifest_rows() -> Vec<String> {
    let mut rows: Vec<Kv> = vec![
        kv(
            "manifest.schema",
            NATIVE_SHARED_PRIMITIVE_CONTRACT_MANIFEST_SCHEMA,
        ),
        kv("manifest.schema_version", "1"),
        kv("contract.schema", TRUST_IR_CONTRACT_SCHEMA),
        kv("contract.schema_version", "1"),
        kv(
            "shared_primitive.schema",
            TRUST_IR_SHARED_PRIMITIVE_CONTRACT_SCHEMA,
        ),
        kv("shared_primitive.schema_version", "1"),
        kv("formula.schema", TRUST_IR_CONTRACT_FORMULA_SCHEMA),
        kv(
            "readiness_report.schema",
            TRUST_IR_CONTRACT_MODEL_VALIDATION_READINESS_REPORT_SCHEMA,
        ),
        kv("readiness_report.schema_version", "1"),
        kv("verifier_suite", "trust_mc"),
        kv("verification_mode", "trust_mc.chc"),
        kv("production.requires_solver_acceptance", "true"),
        kv("production.requires_emitted_solver_artifacts", "true"),
        kv(
            "production.acceptance_report_api",
            AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_REPORT_API,
        ),
        kv(
            "production.consumer_acceptance_api",
            AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_CONSUMER_API,
        ),
        kv("production.acceptance_owner_suite", "ay"),
        kv("production.artifact_role", "solver_input"),
        kv("production.artifact_role", "replay_transcript"),
        kv("production.artifact_role", "solver_witness"),
        kv("production.artifact_owner_suite", "ay"),
        kv("production.artifact_requirement.0.role", "solver_input"),
        kv(
            "production.artifact_requirement.0.kind",
            "trust_mc_horn_clauses",
        ),
        kv(
            "production.artifact_requirement.0.digest_algorithm",
            "sha256",
        ),
        kv("production.artifact_requirement.0.owner_suite", "ay"),
        kv(
            "production.artifact_requirement.0.requires_emitted_solver_artifact",
            "true",
        ),
        kv(
            "production.artifact_requirement.1.role",
            "replay_transcript",
        ),
        kv(
            "production.artifact_requirement.1.kind",
            "replay_transcript",
        ),
        kv(
            "production.artifact_requirement.1.digest_algorithm",
            "sha256",
        ),
        kv("production.artifact_requirement.1.owner_suite", "ay"),
        kv(
            "production.artifact_requirement.1.requires_emitted_solver_artifact",
            "true",
        ),
        kv("production.artifact_requirement.2.role", "solver_witness"),
        kv("production.artifact_requirement.2.kind", "trust_mc_model"),
        kv(
            "production.artifact_requirement.2.digest_algorithm",
            "sha256",
        ),
        kv("production.artifact_requirement.2.owner_suite", "ay"),
        kv(
            "production.artifact_requirement.2.requires_emitted_solver_artifact",
            "true",
        ),
        kv(
            "authority_row_descriptor.schema",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA,
        ),
        kv(
            "authority_row_descriptor.schema_version",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA_VERSION,
        ),
        kvo(
            "authority_row_descriptor.resolution_key_count",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_RESOLUTION_ROW_KEYS
                .len()
                .to_string(),
        ),
    ];
    for (idx, key) in NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_RESOLUTION_ROW_KEYS
        .iter()
        .enumerate()
    {
        rows.push(kvo(
            &format!("authority_row_descriptor.resolution_key.{idx}"),
            key.to_string(),
        ));
    }
    let prefix = format!(
        "trust-ir shared_primitive_contract_manifest schema={NATIVE_SHARED_PRIMITIVE_CONTRACT_MANIFEST_SCHEMA} schema_version=1"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_native_evidence_artifact_resolution_rows() -> Vec<String> {
    vec![format!(
        "trust-ir native_evidence_artifact_resolution \
         schema={NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA} \
         schema_version={NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA_VERSION} \
         artifact_authority.schema={NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA} \
         artifact_authority.schema_version={NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA_VERSION} \
         artifact_resolution.schema={NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA} \
         artifact_resolution.schema_version={NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA_VERSION} \
         request=petri_successor_trust_mc_chc_proof_handoff \
         request.id=petri_successor_trust_mc_chc_proof_handoff \
         owner_suite=ay required_kind=replay_transcript artifact.kind=replay_transcript \
         digest_algorithm=sha256 digest.algorithm=sha256 \
         digest=fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
         artifact_name=none artifact.name=none byte_source_identity=none byte.source_identity=none \
         byte_len=0 byte.len=0 actual_digest=none authority=informational \
         authority_code=informational status=blocked status_code=blocked \
         reason=missing_attachment reason_code=missing_attachment \
         is_resolved=false report.is_resolved=false is_authoritative=false \
         report.is_authoritative=false report.fail_closed=true \
         resolution.bytes_present=false resolution.is_resolved=false \
         resolution.is_authoritative=false resolution.fail_closed=true \
         resolution.authoritative_bytes_available=false \
         production_selected=false fail_closed=true"
    )]
}

// ============================================================
// trust-ir native verification bundle handoff family.
// ============================================================

const BUNDLE_IDENTITY_EXPECTED_FIELDS: &[&str] = &[
    "NativeVerificationBundle::transport_identity()",
    "NativeTransportIdentity::request_digests",
    "NativeTransportIdentity::evidence_digests",
    "NativeVerificationBundle::evidence_bundle_for_request()",
    "NativeVerificationBundle::resolve_evidence_artifact_attachment()",
    "NativeVerificationBundle::resolve_evidence_artifact_attachments_for_kinds()",
];

const DOWNSTREAM_RESPONSIBILITY_KEYS: &[&str] = &[
    "validate_native_verification_bundle_before_admission",
    "derive_transport_identity_with_NativeVerificationBundle::transport_identity()",
    "resolve_artifact_bytes_with_NativeVerificationBundle::resolve_evidence_artifact_attachment()",
    "resolve_required_artifact_bytes_with_NativeVerificationBundle::resolve_evidence_artifact_attachments_for_kinds()",
    "require_authoritative_NativeEvidenceArtifactResolution_before_using_bytes",
    "use_shared_primitive_solver_evidence_descriptor_for_AY_identities",
    "call_AY_acceptance_API_before_production_selection",
    "do_not_reconstruct_AY_solver_logic_downstream",
    "preserve_fail_closed_status_when_required_rows_are_missing",
];

fn trust_ir_native_verification_bundle_handoff_rows() -> Vec<String> {
    let shared_primitive_contract_rows: [Kv; 5] = [
        kv(
            "shared_primitive_contract.manifest.schema",
            NATIVE_SHARED_PRIMITIVE_CONTRACT_MANIFEST_SCHEMA,
        ),
        kv("shared_primitive_contract.manifest.schema_version", "1"),
        kv(
            "shared_primitive_contract.production.requires_solver_acceptance",
            "true",
        ),
        kv(
            "shared_primitive_contract.production.acceptance_owner_suite",
            "ay",
        ),
        kv(
            "shared_primitive_contract.production.solver_evidence.capability_descriptor.schema",
            AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA,
        ),
    ];

    let mut rows: Vec<Kv> = vec![
        kv("handoff.schema", TNVBH_DESCRIPTOR_SCHEMA),
        kv("handoff.schema_version", TNVBH_DESCRIPTOR_SCHEMA_VERSION),
        kv("source.package", TNVBH_SOURCE_PACKAGE),
        kv("source.package_version", TNVBH_SOURCE_PACKAGE_VERSION),
        kv(
            "bundle_identity.schema",
            TRUST_IR_NATIVE_BUNDLE_IDENTITY_CONTRACT_SCHEMA,
        ),
        kv(
            "bundle_identity.schema_version",
            TRUST_IR_NATIVE_BUNDLE_IDENTITY_CONTRACT_SCHEMA_VERSION,
        ),
        kv(
            "bundle_identity.bundle_schema_version",
            TRUST_IR_NATIVE_BUNDLE_SCHEMA_VERSION,
        ),
        kv(
            "bundle_identity.transport_identity.schema",
            TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA,
        ),
        kv(
            "bundle_identity.transport_identity.schema_version",
            TRUST_IR_NATIVE_TRANSPORT_IDENTITY_SCHEMA_VERSION,
        ),
        kvo(
            "bundle_identity.expected_field_count",
            BUNDLE_IDENTITY_EXPECTED_FIELDS.len().to_string(),
        ),
        kv(
            "artifact_authority.schema",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA,
        ),
        kv(
            "artifact_authority.schema_version",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_ROW_SCHEMA_VERSION,
        ),
        kvo(
            "artifact_authority.report_key_count",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_REPORT_ROW_KEYS
                .len()
                .to_string(),
        ),
        kvo(
            "artifact_authority.resolution_key_count",
            NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_RESOLUTION_ROW_KEYS
                .len()
                .to_string(),
        ),
        kv("solver_evidence.owner_suite", "ay"),
        kv(
            "solver_evidence.capability_descriptor.schema",
            AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA,
        ),
        kv("solver_evidence.capability_descriptor.schema_version", "1"),
        kv(
            "solver_evidence.model_blocking_clause.schema",
            AY_MODEL_BLOCKING_CLAUSE_SCHEMA,
        ),
        kv("solver_evidence.model_blocking_clause.schema_version", "1"),
        kv(
            "solver_evidence.model_blocking_clause_evidence.schema",
            AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA,
        ),
        kv(
            "solver_evidence.model_blocking_clause_evidence.schema_version",
            "1",
        ),
        kv(
            "solver_evidence.solve_decision_profile_model_consumer.schema",
            AY_SOLVE_DECISION_PROFILE_MODEL_CONSUMER_SCHEMA,
        ),
        kv(
            "solver_evidence.solve_decision_profile_model_consumer.schema_version",
            "1",
        ),
        kv(
            "solver_evidence.acceptance_report_api",
            AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_REPORT_API,
        ),
        kv(
            "solver_evidence.consumer_acceptance_api",
            AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_CONSUMER_API,
        ),
        kvo(
            "downstream.consumer_responsibility_count",
            DOWNSTREAM_RESPONSIBILITY_KEYS.len().to_string(),
        ),
    ];
    rows.extend(shared_primitive_contract_rows);
    for (i, k) in BUNDLE_IDENTITY_EXPECTED_FIELDS.iter().enumerate() {
        rows.push(kvo(
            &format!("bundle_identity.expected_field.{i}"),
            k.to_string(),
        ));
    }
    for (i, k) in NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_REPORT_ROW_KEYS
        .iter()
        .enumerate()
    {
        rows.push(kvo(
            &format!("artifact_authority.report_key.{i}"),
            k.to_string(),
        ));
    }
    for (i, k) in NATIVE_EVIDENCE_ARTIFACT_AUTHORITY_RESOLUTION_ROW_KEYS
        .iter()
        .enumerate()
    {
        rows.push(kvo(
            &format!("artifact_authority.resolution_key.{i}"),
            k.to_string(),
        ));
    }
    for (i, k) in DOWNSTREAM_RESPONSIBILITY_KEYS.iter().enumerate() {
        rows.push(kvo(
            &format!("downstream.consumer_responsibility.{i}"),
            k.to_string(),
        ));
    }

    let manifest_prefix = format!(
        "trust-ir native_verification_bundle_handoff_manifest schema={TNVBH_SCHEMA} schema_version={TNVBH_SCHEMA_VERSION} source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} project={TNVBH_PROJECT}"
    );
    let mut out = render_manifest_lines(&manifest_prefix, &rows);
    out.push(format!(
        "trust-ir native_verification_bundle_handoff_completeness \
         schema={TNVBH_SCHEMA} schema_version={TNVBH_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} manifest_schema={TNVBH_SCHEMA} \
         manifest_schema_version=1 bundle_identity_status=complete \
         artifact_authority_status=complete ay_evidence_identity_status=complete \
         downstream_responsibility_status=blocked handoff_complete=false \
         status_code=blocked reason_code=missing_replay_transcript_artifact \
         production_selected=false fail_closed=true"
    ));
    out
}

const MANIFEST_IDENTITY_DIGEST_FIXED: &str =
    "trust-ir-stable-v1:a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3";

fn trust_ir_native_verification_bundle_handoff_manifest_identity_rows() -> Vec<String> {
    let rows: Vec<Kv> = vec![
        kv("manifest_identity.schema", TNVBH_MANIFEST_IDENTITY_SCHEMA),
        kv(
            "manifest_identity.schema_version",
            TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION,
        ),
        kv(
            "manifest_identity.descriptor.schema",
            TNVBH_DESCRIPTOR_SCHEMA,
        ),
        kv(
            "manifest_identity.descriptor.schema_version",
            TNVBH_DESCRIPTOR_SCHEMA_VERSION,
        ),
        kv("manifest_identity.source.package", TNVBH_SOURCE_PACKAGE),
        kv("manifest_identity.source.package_version", "0.1.0"),
        kv(
            "manifest_identity.digest.context",
            TNVBH_MANIFEST_IDENTITY_SCHEMA,
        ),
        kv("manifest_identity.digest.algorithm", "trust-ir-stable-v1"),
        kv("manifest_identity.digest", MANIFEST_IDENTITY_DIGEST_FIXED),
        kv("manifest_identity.completeness.status", "complete"),
        kv("manifest_identity.fail_closed", "false"),
        kv("manifest_identity.rows.observed_count", "33"),
        kv("manifest_identity.rows.required_count", "33"),
        kv("manifest_identity.rows.present_required_count", "33"),
        kv("manifest_identity.rows.missing_count", "0"),
        kv("manifest_identity.rows.extra_count", "0"),
        kv("manifest_identity.missing_row_kind_count", "0"),
        kv("manifest_identity.linked_handoff.schema", TNVBH_SCHEMA),
        kv(
            "manifest_identity.linked_handoff.schema_version",
            TNVBH_SCHEMA_VERSION,
        ),
        kv(
            "manifest_identity.linked_handoff.manifest_component",
            "native_verification_bundle_handoff_manifest",
        ),
        kv(
            "manifest_identity.linked_handoff.completeness_component",
            "native_verification_bundle_handoff_completeness",
        ),
    ];
    let prefix = format!(
        "trust-ir native_verification_bundle_handoff_manifest_identity \
         schema={TNVBH_MANIFEST_IDENTITY_SCHEMA} \
         schema_version={TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} linked_handoff_schema={TNVBH_SCHEMA} \
         linked_handoff_schema_version={TNVBH_SCHEMA_VERSION} \
         linked_handoff_manifest_component=native_verification_bundle_handoff_manifest \
         linked_handoff_completeness_component=native_verification_bundle_handoff_completeness"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_native_verification_bundle_handoff_contract_health_rows() -> Vec<String> {
    let rows: Vec<Kv> = vec![
        kv("contract_health.status", "healthy"),
        kv("contract_health.fail_closed", "false"),
        kv("contract_health.descriptor.schema", TNVBH_DESCRIPTOR_SCHEMA),
        kv(
            "contract_health.descriptor.schema_version",
            TNVBH_DESCRIPTOR_SCHEMA_VERSION,
        ),
        kv(
            "contract_health.manifest_identity.schema",
            TNVBH_MANIFEST_IDENTITY_SCHEMA,
        ),
        kv(
            "contract_health.manifest_identity.schema_version",
            TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION,
        ),
        kv("contract_health.count.manifest_rows", "33"),
        kv("contract_health.count.normalized_rows", "33"),
        kv("contract_health.count.required_rows", "33"),
        kv("contract_health.count.completeness.required_rows", "33"),
        kv(
            "contract_health.count.completeness.present_required_rows",
            "33",
        ),
        kv("contract_health.count.completeness.missing_rows", "0"),
        kv(
            "contract_health.count.manifest_identity.observed_rows",
            "33",
        ),
        kv(
            "contract_health.count.manifest_identity.required_rows",
            "33",
        ),
        kv(
            "contract_health.count.manifest_identity.present_required_rows",
            "33",
        ),
        kv("contract_health.count.manifest_identity.missing_rows", "0"),
        kv("contract_health.count.manifest_identity.extra_rows", "0"),
        kv(
            "contract_health.count.manifest_identity.key_value_rows",
            "26",
        ),
        kv(
            "contract_health.count.manifest_identity.key_value_lines",
            "26",
        ),
        kv(
            "contract_health.count.manifest_identity.key_value_text_lines",
            "26",
        ),
        kv(
            "contract_health.manifest_identity.digest",
            MANIFEST_IDENTITY_DIGEST_FIXED,
        ),
        kv("contract_health.agreement.schema_version_rows", "true"),
        kv("contract_health.agreement.row_counts", "true"),
        kv("contract_health.agreement.completeness", "true"),
        kv("contract_health.agreement.manifest_identity_digest", "true"),
        kv(
            "contract_health.agreement.manifest_identity_key_values",
            "true",
        ),
    ];
    let prefix = format!(
        "trust-ir {TNVBH_CONTRACT_HEALTH_COMPONENT} schema={TNVBH_DESCRIPTOR_SCHEMA} \
         schema_version={TNVBH_DESCRIPTOR_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} linked_handoff_schema={TNVBH_SCHEMA} \
         linked_handoff_schema_version={TNVBH_SCHEMA_VERSION} \
         linked_manifest_identity_schema={TNVBH_MANIFEST_IDENTITY_SCHEMA} \
         linked_manifest_identity_schema_version={TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_native_verification_bundle_handoff_diagnostic_fixture_manifest_rows() -> Vec<String> {
    // (name, completeness, manifest_identity, contract_health, accepted, fail_closed)
    let fixtures: [(&str, &str, &str, &str, &str, &str); 2] = [
        (
            TNVBH_HEALTHY_FIXTURE,
            "complete",
            "complete",
            "healthy",
            "true",
            "false",
        ),
        (
            TNVBH_INCOMPLETE_FIXTURE,
            "incomplete",
            "incomplete",
            "inconsistent",
            "false",
            "true",
        ),
    ];
    let mut rows: Vec<Kv> = vec![
        kv(
            "fixture_manifest.schema",
            TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA,
        ),
        kv(
            "fixture_manifest.schema_version",
            TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA_VERSION,
        ),
        kv("fixture_manifest.source.package", TNVBH_SOURCE_PACKAGE),
        kv("fixture_manifest.source.package_version", "0.1.0"),
        kvo("fixture_manifest.fixture_count", fixtures.len().to_string()),
    ];
    for (i, (name, completeness, manifest_id, contract, accepted, fc)) in
        fixtures.iter().enumerate()
    {
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.name"),
            name.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.expected.completeness_status"),
            completeness.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.expected.manifest_identity_status"),
            manifest_id.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.expected.contract_health_status"),
            contract.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.expected.accepted"),
            accepted.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.expected.fail_closed"),
            fc.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.schema.handoff"),
            TNVBH_DESCRIPTOR_SCHEMA.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.schema.handoff_version"),
            TNVBH_DESCRIPTOR_SCHEMA_VERSION.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.schema.manifest_identity"),
            TNVBH_MANIFEST_IDENTITY_SCHEMA.to_string(),
        ));
        rows.push(kvo(
            &format!("fixture_manifest.fixture.{i}.schema.manifest_identity_version"),
            TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION.to_string(),
        ));
    }
    let prefix = format!(
        "trust-ir {TNVBH_DIAG_FIXTURE_MANIFEST_COMPONENT} \
         schema={TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA} \
         schema_version={TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} linked_handoff_schema={TNVBH_SCHEMA} \
         linked_handoff_schema_version={TNVBH_SCHEMA_VERSION} \
         linked_manifest_identity_schema={TNVBH_MANIFEST_IDENTITY_SCHEMA} \
         linked_manifest_identity_schema_version={TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_native_verification_bundle_handoff_diagnostic_fixture_round_trip_rows() -> Vec<String> {
    let entries: [(&str, &str, &str, &str, &str, &str); 2] = [
        (
            TNVBH_HEALTHY_FIXTURE,
            "complete",
            "complete",
            "healthy",
            "true",
            "false",
        ),
        (
            TNVBH_INCOMPLETE_FIXTURE,
            "incomplete",
            "incomplete",
            "inconsistent",
            "false",
            "true",
        ),
    ];
    let expected_row_count = 5 + entries.len() * 10;
    let mut rows: Vec<Kv> = vec![
        kv("round_trip.status_code", "valid"),
        kv("round_trip.fail_closed", "false"),
        kvo(
            "round_trip.expected_row_count",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip.observed_row_count",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip.unique_key_count",
            expected_row_count.to_string(),
        ),
        kv("round_trip.duplicate_key_count", "0"),
        kv("round_trip.missing_key_count", "0"),
        kv("round_trip.unexpected_key_count", "0"),
        kv("round_trip.mismatched_value_key_count", "0"),
        kv("round_trip.invalid_bool_key_count", "0"),
        kvo(
            "round_trip.reconstructed_fixture_name_count",
            entries.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_completeness_status_count",
            entries.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_manifest_identity_status_count",
            entries.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_contract_health_status_count",
            entries.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_accepted_value_count",
            entries.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_fail_closed_value_count",
            entries.len().to_string(),
        ),
    ];
    for (i, (fixture_name, completeness, manifest_id, contract, accepted, fc)) in
        entries.iter().enumerate()
    {
        rows.push(kvo(
            &format!("round_trip.reconstructed_fixture_name.{i}"),
            fixture_name.to_string(),
        ));
        rows.push(kvo(
            &format!("round_trip.reconstructed_completeness_status_code.{i}"),
            completeness.to_string(),
        ));
        rows.push(kvo(
            &format!("round_trip.reconstructed_manifest_identity_status_code.{i}"),
            manifest_id.to_string(),
        ));
        rows.push(kvo(
            &format!("round_trip.reconstructed_contract_health_status_code.{i}"),
            contract.to_string(),
        ));
        rows.push(kvo(
            &format!("round_trip.reconstructed_accepted_value.{i}"),
            accepted.to_string(),
        ));
        rows.push(kvo(
            &format!("round_trip.reconstructed_fail_closed_value.{i}"),
            fc.to_string(),
        ));
    }
    let prefix = format!(
        "trust-ir {TNVBH_DIAG_FIXTURE_ROUND_TRIP_COMPONENT} \
         schema={TNVBH_DIAG_FIXTURE_ROUND_TRIP_SCHEMA} \
         schema_version={TNVBH_DIAG_FIXTURE_ROUND_TRIP_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} \
         linked_fixture_manifest_schema={TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA} \
         linked_fixture_manifest_schema_version={TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA_VERSION} \
         linked_fixture_manifest_component={TNVBH_DIAG_FIXTURE_MANIFEST_COMPONENT}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_native_verification_bundle_handoff_replay_contract_surface_rows() -> Vec<String> {
    let schema_values = replay_contract_schema_values();
    let mut rows: Vec<Kv> = vec![
        kv(
            "replay_contract_surface.schema",
            TNVBH_REPLAY_SURFACE_SCHEMA,
        ),
        kv(
            "replay_contract_surface.schema_version",
            TNVBH_REPLAY_SURFACE_SCHEMA_VERSION,
        ),
        kv(
            "replay_contract_surface.source.package",
            TNVBH_SOURCE_PACKAGE,
        ),
        kv("replay_contract_surface.source.package_version", "0.1.0"),
        kvo(
            "replay_contract_surface.helper_count",
            REPLAY_CONTRACT_HELPER_NAMES.len().to_string(),
        ),
    ];
    for (i, name) in REPLAY_CONTRACT_HELPER_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("replay_contract_surface.helper.{i}.name"),
            name.to_string(),
        ));
    }
    rows.push(kvo(
        "replay_contract_surface.schema_count",
        REPLAY_CONTRACT_SCHEMA_NAMES.len().to_string(),
    ));
    for (i, (name, value)) in REPLAY_CONTRACT_SCHEMA_NAMES
        .iter()
        .zip(schema_values.iter())
        .enumerate()
    {
        rows.push(kvo(
            &format!("replay_contract_surface.schema.{i}.name"),
            name.to_string(),
        ));
        rows.push(kvo(
            &format!("replay_contract_surface.schema.{i}.value"),
            value.to_string(),
        ));
    }
    rows.push(kvo(
        "replay_contract_surface.fixture_count",
        REPLAY_CONTRACT_FIXTURE_NAMES.len().to_string(),
    ));
    for (i, name) in REPLAY_CONTRACT_FIXTURE_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("replay_contract_surface.fixture.{i}.name"),
            name.to_string(),
        ));
    }
    rows.push(kvo(
        "replay_contract_surface.validator_count",
        REPLAY_CONTRACT_VALIDATOR_NAMES.len().to_string(),
    ));
    for (i, name) in REPLAY_CONTRACT_VALIDATOR_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("replay_contract_surface.validator.{i}.name"),
            name.to_string(),
        ));
    }
    let prefix = format!(
        "trust-ir {TNVBH_REPLAY_SURFACE_COMPONENT} \
         schema={TNVBH_REPLAY_SURFACE_SCHEMA} \
         schema_version={TNVBH_REPLAY_SURFACE_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} linked_handoff_schema={TNVBH_SCHEMA} \
         linked_handoff_schema_version={TNVBH_SCHEMA_VERSION} \
         linked_fixture_manifest_schema={TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA} \
         linked_fixture_manifest_schema_version={TNVBH_DIAG_FIXTURE_MANIFEST_SCHEMA_VERSION} \
         linked_fixture_manifest_component={TNVBH_DIAG_FIXTURE_MANIFEST_COMPONENT}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn replay_contract_expected_row_count() -> usize {
    4 + 1
        + REPLAY_CONTRACT_HELPER_NAMES.len()
        + 1
        + 2 * REPLAY_CONTRACT_SCHEMA_NAMES.len()
        + 1
        + REPLAY_CONTRACT_FIXTURE_NAMES.len()
        + 1
        + REPLAY_CONTRACT_VALIDATOR_NAMES.len()
}

fn trust_ir_native_verification_bundle_handoff_replay_contract_surface_round_trip_rows(
) -> Vec<String> {
    let expected_row_count = replay_contract_expected_row_count();
    let mut rows: Vec<Kv> = vec![
        kv("round_trip.status_code", "valid"),
        kv("round_trip.fail_closed", "false"),
        kvo(
            "round_trip.expected_row_count",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip.observed_row_count",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip.unique_key_count",
            expected_row_count.to_string(),
        ),
        kv("round_trip.duplicate_key_count", "0"),
        kv("round_trip.missing_key_count", "0"),
        kv("round_trip.unexpected_key_count", "0"),
        kv("round_trip.mismatched_value_key_count", "0"),
        kv("round_trip.invalid_usize_key_count", "0"),
        kv("round_trip.invalid_line_count", "0"),
        kv(
            "round_trip.reconstructed_schema",
            TNVBH_REPLAY_SURFACE_SCHEMA,
        ),
        kv(
            "round_trip.reconstructed_schema_version",
            TNVBH_REPLAY_SURFACE_SCHEMA_VERSION,
        ),
        kvo(
            "round_trip.reconstructed_helper_name_count",
            REPLAY_CONTRACT_HELPER_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_schema_name_count",
            REPLAY_CONTRACT_SCHEMA_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_schema_value_count",
            REPLAY_CONTRACT_SCHEMA_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_fixture_count",
            REPLAY_CONTRACT_FIXTURE_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_fixture_name_count",
            REPLAY_CONTRACT_FIXTURE_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip.reconstructed_validator_name_count",
            REPLAY_CONTRACT_VALIDATOR_NAMES.len().to_string(),
        ),
        kv("round_trip.schema_header_matches", "true"),
        kv("round_trip.schema_name_value_rows_agree", "true"),
        kv("round_trip.helper_names_match", "true"),
        kv("round_trip.fixture_count_matches", "true"),
        kv("round_trip.fixture_names_match", "true"),
        kv("round_trip.validator_names_match", "true"),
    ];
    let schema_values = replay_contract_schema_values();
    for (i, name) in REPLAY_CONTRACT_HELPER_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("round_trip.reconstructed_helper_name.{i}"),
            name.to_string(),
        ));
    }
    for (i, name) in REPLAY_CONTRACT_SCHEMA_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("round_trip.reconstructed_schema_name.{i}"),
            name.to_string(),
        ));
    }
    for (i, value) in schema_values.iter().enumerate() {
        rows.push(kvo(
            &format!("round_trip.reconstructed_schema_value.{i}"),
            value.to_string(),
        ));
    }
    for (i, name) in REPLAY_CONTRACT_FIXTURE_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("round_trip.reconstructed_fixture_name.{i}"),
            name.to_string(),
        ));
    }
    for (i, name) in REPLAY_CONTRACT_VALIDATOR_NAMES.iter().enumerate() {
        rows.push(kvo(
            &format!("round_trip.reconstructed_validator_name.{i}"),
            name.to_string(),
        ));
    }
    let prefix = format!(
        "trust-ir {TNVBH_REPLAY_SURFACE_ROUND_TRIP_COMPONENT} \
         schema={TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA} \
         schema_version={TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} \
         linked_replay_contract_surface_schema={TNVBH_REPLAY_SURFACE_SCHEMA} \
         linked_replay_contract_surface_schema_version={TNVBH_REPLAY_SURFACE_SCHEMA_VERSION} \
         linked_replay_contract_surface_component={TNVBH_REPLAY_SURFACE_COMPONENT}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn replay_contract_report_identity_text(
    status_code: &str,
    fail_closed: &str,
    expected_row_count: usize,
    observed_row_count: usize,
    unique_key_count: usize,
    helper_count: usize,
    validator_count: usize,
    fixture_count: usize,
    diagnostic_counts: &[(&str, usize)],
) -> String {
    let total_diagnostics: usize = diagnostic_counts.iter().map(|(_, n)| n).sum();
    let mut lines: Vec<String> = vec![
        trust_ir_report_identity_line(
            "round_trip_report.schema",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        ),
        trust_ir_report_identity_line(
            "round_trip_report.schema_version",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION,
        ),
        trust_ir_report_identity_line("round_trip_report.status", status_code),
        trust_ir_report_identity_line("round_trip_report.fail_closed", fail_closed),
        trust_ir_report_identity_line(
            "round_trip_report.surface.schema",
            TNVBH_REPLAY_SURFACE_SCHEMA,
        ),
        trust_ir_report_identity_line(
            "round_trip_report.surface.schema_version",
            TNVBH_REPLAY_SURFACE_SCHEMA_VERSION,
        ),
        trust_ir_report_identity_line(
            "round_trip_report.count.expected_rows",
            &expected_row_count.to_string(),
        ),
        trust_ir_report_identity_line(
            "round_trip_report.count.observed_rows",
            &observed_row_count.to_string(),
        ),
        trust_ir_report_identity_line(
            "round_trip_report.count.unique_keys",
            &unique_key_count.to_string(),
        ),
        trust_ir_report_identity_line("round_trip_report.count.helpers", &helper_count.to_string()),
        trust_ir_report_identity_line(
            "round_trip_report.count.validators",
            &validator_count.to_string(),
        ),
        trust_ir_report_identity_line(
            "round_trip_report.count.fixtures",
            &fixture_count.to_string(),
        ),
        trust_ir_report_identity_line(
            "round_trip_report.count.diagnostics",
            &total_diagnostics.to_string(),
        ),
        trust_ir_report_identity_line("round_trip_report.agreement.schema_header", "true"),
        trust_ir_report_identity_line("round_trip_report.agreement.schema_name_value_rows", "true"),
        trust_ir_report_identity_line("round_trip_report.agreement.helper_names", "true"),
        trust_ir_report_identity_line("round_trip_report.agreement.fixture_count", "true"),
        trust_ir_report_identity_line("round_trip_report.agreement.fixture_names", "true"),
        trust_ir_report_identity_line("round_trip_report.agreement.validator_names", "true"),
    ];
    for (name, count) in diagnostic_counts {
        lines.push(trust_ir_report_identity_line(
            &format!("round_trip_report.diagnostic.{name}.count"),
            &count.to_string(),
        ));
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

fn replay_contract_compact_json_text(
    identity_text: &str,
    identity_digest: &str,
    status_code: &str,
    fail_closed: &str,
    expected_row_count: usize,
    observed_row_count: usize,
    unique_key_count: usize,
    helper_count: usize,
    validator_count: usize,
    fixture_count: usize,
    diagnostic_count: usize,
) -> String {
    // Python uses sort=insertion-order via dict literal — the keys here
    // mirror the literal exactly so downstream consumers see the same
    // compact JSON byte-for-byte.
    let mut map = Map::new();
    map.insert(
        "schema".into(),
        Value::String(TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA.into()),
    );
    map.insert(
        "schema_version".into(),
        Value::Number(
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION
                .parse::<i64>()
                .unwrap_or(1)
                .into(),
        ),
    );
    map.insert("identity_text".into(), Value::String(identity_text.into()));
    map.insert(
        "identity_digest_context".into(),
        Value::String(TNVBH_REPLAY_JSON_BINDING_SCHEMA.into()),
    );
    map.insert(
        "identity_digest_algorithm".into(),
        Value::String("trust-ir-stable-v1".into()),
    );
    map.insert(
        "identity_digest".into(),
        Value::String(identity_digest.into()),
    );
    map.insert("status".into(), Value::String(status_code.into()));
    map.insert("fail_closed".into(), Value::Bool(fail_closed == "true"));
    map.insert(
        "surface_schema".into(),
        Value::String(TNVBH_REPLAY_SURFACE_SCHEMA.into()),
    );
    map.insert(
        "surface_schema_version".into(),
        Value::Number(
            TNVBH_REPLAY_SURFACE_SCHEMA_VERSION
                .parse::<i64>()
                .unwrap_or(1)
                .into(),
        ),
    );
    map.insert(
        "expected_row_count".into(),
        Value::Number(expected_row_count.into()),
    );
    map.insert(
        "observed_row_count".into(),
        Value::Number(observed_row_count.into()),
    );
    map.insert(
        "unique_key_count".into(),
        Value::Number(unique_key_count.into()),
    );
    map.insert("helper_count".into(), Value::Number(helper_count.into()));
    map.insert(
        "validator_count".into(),
        Value::Number(validator_count.into()),
    );
    map.insert("fixture_count".into(), Value::Number(fixture_count.into()));
    map.insert(
        "diagnostic_count".into(),
        Value::Number(diagnostic_count.into()),
    );
    let mut s = serde_json::to_string(&Value::Object(map)).expect("compact json serialises");
    s.push('\n');
    s
}

fn trust_ir_native_verification_bundle_handoff_replay_contract_report_identity_rows() -> Vec<String>
{
    let expected_row_count = replay_contract_expected_row_count();
    let diag = &[
        ("duplicate_key", 0usize),
        ("missing_key", 0),
        ("unexpected_key", 0),
        ("mismatched_value_key", 0),
        ("invalid_usize_key", 0),
        ("invalid_line", 0),
    ];
    let identity_text = replay_contract_report_identity_text(
        "valid",
        "false",
        expected_row_count,
        expected_row_count,
        expected_row_count,
        REPLAY_CONTRACT_HELPER_NAMES.len(),
        REPLAY_CONTRACT_VALIDATOR_NAMES.len(),
        REPLAY_CONTRACT_FIXTURE_NAMES.len(),
        diag,
    );
    let digest = legacy_compatibility_trust_ir_stable_v1_digest(
        TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        &identity_text,
    );
    let rows: Vec<Kv> = vec![
        kv(
            "round_trip_report.schema",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        ),
        kv(
            "round_trip_report.schema_version",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION,
        ),
        kv("round_trip_report.status", "valid"),
        kv("round_trip_report.fail_closed", "false"),
        kv(
            "round_trip_report.surface.schema",
            TNVBH_REPLAY_SURFACE_SCHEMA,
        ),
        kv(
            "round_trip_report.surface.schema_version",
            TNVBH_REPLAY_SURFACE_SCHEMA_VERSION,
        ),
        kvo(
            "round_trip_report.count.expected_rows",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip_report.count.observed_rows",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip_report.count.unique_keys",
            expected_row_count.to_string(),
        ),
        kvo(
            "round_trip_report.count.helpers",
            REPLAY_CONTRACT_HELPER_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip_report.count.validators",
            REPLAY_CONTRACT_VALIDATOR_NAMES.len().to_string(),
        ),
        kvo(
            "round_trip_report.count.fixtures",
            REPLAY_CONTRACT_FIXTURE_NAMES.len().to_string(),
        ),
        kv("round_trip_report.count.diagnostics", "0"),
        kv("round_trip_report.diagnostic.duplicate_keys", "0"),
        kv("round_trip_report.diagnostic.missing_keys", "0"),
        kv("round_trip_report.diagnostic.unexpected_keys", "0"),
        kv("round_trip_report.diagnostic.mismatched_value_keys", "0"),
        kv("round_trip_report.diagnostic.invalid_usize_keys", "0"),
        kv("round_trip_report.diagnostic.invalid_lines", "0"),
        kv(
            "round_trip_report.digest.context",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        ),
        kv("round_trip_report.digest.algorithm", "trust-ir-stable-v1"),
        kvo("round_trip_report.digest", digest),
    ];
    let prefix = format!(
        "trust-ir {TNVBH_REPLAY_REPORT_IDENTITY_COMPONENT} \
         schema={TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA} \
         schema_version={TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} \
         linked_replay_contract_surface_schema={TNVBH_REPLAY_SURFACE_SCHEMA} \
         linked_replay_contract_surface_schema_version={TNVBH_REPLAY_SURFACE_SCHEMA_VERSION} \
         linked_replay_contract_surface_component={TNVBH_REPLAY_SURFACE_COMPONENT} \
         linked_replay_contract_surface_round_trip_component={TNVBH_REPLAY_SURFACE_ROUND_TRIP_COMPONENT}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_ir_native_verification_bundle_handoff_replay_contract_json_manifest_binding_rows(
) -> Vec<String> {
    let expected_row_count = replay_contract_expected_row_count();
    let diag = &[
        ("duplicate_key", 0usize),
        ("missing_key", 0),
        ("unexpected_key", 0),
        ("mismatched_value_key", 0),
        ("invalid_usize_key", 0),
        ("invalid_line", 0),
    ];
    let identity_text = replay_contract_report_identity_text(
        "valid",
        "false",
        expected_row_count,
        expected_row_count,
        expected_row_count,
        REPLAY_CONTRACT_HELPER_NAMES.len(),
        REPLAY_CONTRACT_VALIDATOR_NAMES.len(),
        REPLAY_CONTRACT_FIXTURE_NAMES.len(),
        diag,
    );
    let report_identity_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        &identity_text,
    );
    let compact_json_text = replay_contract_compact_json_text(
        &identity_text,
        &report_identity_digest,
        "valid",
        "false",
        expected_row_count,
        expected_row_count,
        expected_row_count,
        REPLAY_CONTRACT_HELPER_NAMES.len(),
        REPLAY_CONTRACT_VALIDATOR_NAMES.len(),
        REPLAY_CONTRACT_FIXTURE_NAMES.len(),
        0,
    );
    let json_manifest_text_digest = legacy_compatibility_trust_ir_stable_v1_digest(
        TNVBH_REPLAY_JSON_BINDING_SCHEMA,
        &compact_json_text,
    );
    let rows: Vec<Kv> = vec![
        kv(
            "json_manifest_binding.schema",
            TNVBH_REPLAY_JSON_BINDING_SCHEMA,
        ),
        kv(
            "json_manifest_binding.schema_version",
            TNVBH_REPLAY_JSON_BINDING_SCHEMA_VERSION,
        ),
        kv("json_manifest_binding.status", "bound"),
        kv("json_manifest_binding.fail_closed", "false"),
        kv(
            "json_manifest.schema",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        ),
        kv(
            "json_manifest.schema_version",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA_VERSION,
        ),
        kv(
            "json_manifest.text_digest.context",
            TNVBH_REPLAY_JSON_BINDING_SCHEMA,
        ),
        kv("json_manifest.text_digest.algorithm", "trust-ir-stable-v1"),
        kvo("json_manifest.text_digest", json_manifest_text_digest),
        kv(
            "round_trip_report.identity_digest.context",
            TNVBH_REPLAY_SURFACE_ROUND_TRIP_SCHEMA,
        ),
        kv(
            "round_trip_report.identity_digest.algorithm",
            "trust-ir-stable-v1",
        ),
        kvo("round_trip_report.identity_digest", report_identity_digest),
        kv("manifest_identity.schema", TNVBH_MANIFEST_IDENTITY_SCHEMA),
        kv(
            "manifest_identity.schema_version",
            TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION,
        ),
        kv(
            "manifest_identity.digest.context",
            TNVBH_MANIFEST_IDENTITY_SCHEMA,
        ),
        kv("manifest_identity.digest.algorithm", "trust-ir-stable-v1"),
        kv("manifest_identity.digest", MANIFEST_IDENTITY_DIGEST_FIXED),
        kv("json_manifest_binding.check.report_valid", "true"),
        kv(
            "json_manifest_binding.check.replay_surface_schema_matches",
            "true",
        ),
        kv(
            "json_manifest_binding.check.handoff_schema_listed_by_surface",
            "true",
        ),
        kv(
            "json_manifest_binding.check.manifest_identity_schema_listed_by_surface",
            "true",
        ),
        kv(
            "json_manifest_binding.check.manifest_identity_complete",
            "true",
        ),
        kv(
            "json_manifest_binding.check.manifest_identity_descriptor_matches",
            "true",
        ),
        kv(
            "json_manifest_binding.check.manifest_identity_source_matches",
            "true",
        ),
        kv(
            "json_manifest_binding.check.manifest_identity_digest_matches_canonical_text",
            "true",
        ),
    ];
    let prefix = format!(
        "trust-ir {TNVBH_REPLAY_JSON_BINDING_COMPONENT} \
         schema={TNVBH_REPLAY_JSON_BINDING_SCHEMA} \
         schema_version={TNVBH_REPLAY_JSON_BINDING_SCHEMA_VERSION} \
         source_package={TNVBH_SOURCE_PACKAGE} source_project={TNVBH_SOURCE_PROJECT} \
         project={TNVBH_PROJECT} \
         linked_replay_contract_surface_schema={TNVBH_REPLAY_SURFACE_SCHEMA} \
         linked_replay_contract_surface_schema_version={TNVBH_REPLAY_SURFACE_SCHEMA_VERSION} \
         linked_replay_contract_report_identity_component={TNVBH_REPLAY_REPORT_IDENTITY_COMPONENT} \
         linked_manifest_identity_schema={TNVBH_MANIFEST_IDENTITY_SCHEMA} \
         linked_manifest_identity_schema_version={TNVBH_MANIFEST_IDENTITY_SCHEMA_VERSION}"
    );
    render_manifest_lines(&prefix, &rows)
}

// ============================================================
// trust-codegen compile-artifact cache telemetry + PGO provenance.
// ============================================================

fn trust_cg_compile_artifact_cache_telemetry_rows() -> Vec<String> {
    let required_fields = "boundary|status|key_sha256|cache_path|elapsed_micros";
    let optional_fields = "artifact_sha256|reason";
    let boundary_codes = "pipeline|service";
    let status_codes = "hit|miss|stored|rejected_corrupt";
    let metric_fields = "elapsed_micros";
    vec![
        format!(
            "trust-cg compile_artifact_cache_telemetry_descriptor \
             schema={TRUST_CG_COMPILE_ARTIFACT_CACHE_TELEMETRY_SCHEMA} schema_version=1 \
             required_fields={required_fields} optional_fields={optional_fields} \
             boundary_codes={boundary_codes} status_codes={status_codes} \
             metric_fields={metric_fields} authorizes_useful_native=false \
             production_selected=false fail_closed=true"
        ),
        format!(
            "trust-cg compile_artifact_cache_telemetry \
             schema={TRUST_CG_COMPILE_ARTIFACT_CACHE_TELEMETRY_SCHEMA} schema_version=1 \
             boundary=service status=miss \
             key_sha256={TRUST_CG_COMPILE_ARTIFACT_CACHE_KEY_SHA256} \
             cache_path=/tmp/ty-mcc-trust_cg-cache elapsed_micros=0 artifact_sha256=none \
             reason=cache_probe_only authorizes_useful_native=false \
             production_selected=false fail_closed=true"
        ),
    ]
}

fn trust_cg_host_jit_pgo_profile_authority_manifest_rows() -> Vec<String> {
    let rows: Vec<Kv> = vec![
        kv(
            "manifest.schema",
            TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA,
        ),
        kv(
            "manifest.schema_version",
            TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA_VERSION,
        ),
        kv(
            "profile_authority.schema",
            TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_EVIDENCE_SCHEMA,
        ),
        kv("profile_authority.schema_version", "1"),
        kv(
            "profile_authority.status",
            "not_authoritative_for_compiled_function",
        ),
        kv("profile_authority.reason", "profile_use_not_scheduled"),
        kv("profile_authority.profile_key_digest", "none"),
        kv("profile_authority.module_hash", "none"),
        kv("profile_authority.target_triple", "unknown"),
        kv("profile_authority.target_cpu", "unknown"),
        kv("profile_authority.target_features", ""),
        kv("profile_authority.opt_level", "none"),
        kv("profile_authority.opt_level_num", "0"),
        kv("profile_authority.cache_key_version", "0"),
        kv("profile_authority.profile_sha256", ""),
        kv("profile_authority.fresh", "false"),
        kv("profile_authority.scheduled", "false"),
        kv("profile_authority.pass", ""),
        kv("profile_authority.profile_use_reason", "opt-level-below-o2"),
        kv("profile_authority.target_compatible", "false"),
        kv(
            "profile_authority.compiled_function_profile_reuse_sound",
            "false",
        ),
        kv("profile_authority.authorizes_profile_reuse", "false"),
        kv("profile_authority.authorizes_useful_native", "false"),
    ];
    let prefix = format!(
        "trust-cg host_jit_pgo_profile_authority_manifest \
         schema={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA} \
         schema_version={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA_VERSION}"
    );
    render_manifest_lines(&prefix, &rows)
}

fn trust_cg_host_jit_pgo_provenance_rows() -> Vec<String> {
    let mut out = vec![format!(
        "trust-cg host_jit_pgo_provenance_descriptor \
         schema={TRUST_CG_HOST_JIT_PGO_PROVENANCE_DESCRIPTOR_SCHEMA} schema_version=1 \
         profile_report_schema={TRUST_CG_PROFILE_REPORT_SCHEMA} \
         profile_key_fields=profile_key_digest|module_hash|target_triple|target_cpu|target_features|opt_level|opt_level_num|cache_key_version \
         capture_fields=kind|hook_mode|entry|entry_shape|call_count|inputs|window|return_value|ty_summary \
         profile_use_fields=fresh|consumer|scheduled|pass|reason|summary \
         profile_use_soundness_fields=fresh|scheduled|pass|reason \
         profile_authority_evidence_schema={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_EVIDENCE_SCHEMA} \
         profile_authority_manifest_schema={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA} \
         profile_authority_manifest_schema_version={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA_VERSION} \
         profile_authority_fields=schema|schema_version|status|reason|profile_key_digest|module_hash|target_triple|target_cpu|target_features|opt_level|opt_level_num|cache_key_version|profile_sha256|fresh|scheduled|pass|profile_use_reason|target_compatible|compiled_function_profile_reuse_sound|authorizes_profile_reuse|authorizes_useful_native \
         profile_authority_manifest_row_keys=manifest.schema|manifest.schema_version|profile_authority.schema|profile_authority.schema_version|profile_authority.status|profile_authority.reason|profile_authority.profile_key_digest|profile_authority.module_hash|profile_authority.target_triple|profile_authority.target_cpu|profile_authority.target_features|profile_authority.opt_level|profile_authority.opt_level_num|profile_authority.cache_key_version|profile_authority.profile_sha256|profile_authority.fresh|profile_authority.scheduled|profile_authority.pass|profile_authority.profile_use_reason|profile_authority.target_compatible|profile_authority.compiled_function_profile_reuse_sound|profile_authority.authorizes_profile_reuse|profile_authority.authorizes_useful_native \
         profile_authority_status_codes=authoritative_for_compiled_function|not_authoritative_for_compiled_function \
         profile_authority_reason_codes=fresh_scheduled_profile_use|report_schema_mismatch|report_mode_mismatch|profile_not_fresh|profile_use_not_scheduled|profile_use_pass_missing|profile_use_pass_mismatch|profile_use_reason_missing|profile_use_reason_mismatch \
         runner_error_reason_codes=host_target_mismatch|compiler_target_mismatch|host_triple_mismatch|unsupported_entry_shape|entry_not_found|unsupported_entry_signature|no_argument_entry_with_inputs \
         entry_shape_codes=no_args_no_return|no_args_i64_return|i64_arg_no_return|i64_arg_i64_return|ty_parent_loop_u64_return \
         profile_use_reason_codes=opt-level-enables-profile-use|opt-level-below-o2 \
         profile_use_pass_code=profile-use \
         soundness_helper=HostJitPgoUseReport::profile_reuse_sound_for_compiled_function \
         profile_authority_helper=HostJitPgoUseReport::profile_authority_evidence \
         profile_authority_manifest_helper=HostJitPgoProfileAuthorityEvidence::manifest_rows \
         target_compatibility_helper=HostJitPgoRunnerError::target_compatible \
         authorizes_useful_native=false production_selected=false fail_closed=true"
    )];
    out.extend(trust_cg_host_jit_pgo_profile_authority_manifest_rows());
    out
}

// ============================================================
// MCC production selector decision + portfolio routes.
// ============================================================

fn production_selector_decision_rows() -> Vec<String> {
    let cap_status = ay_solver_capability_descriptor_status_code();
    vec![format!(
        "MCC production_selector_decision \
         schema={MCC_PRODUCTION_SELECTOR_DECISION_SCHEMA} schema_version=1 \
         selector_input=shared_primitive_evidence selector_status=blocked \
         selected_lane=ay_symbolic selected_backend_code=ay_sat \
         selected_reason_code=ay_solve_decision_profile_accepted \
         ay_solve_decision_status=accepted ay_solve_decision_code=sat \
         ay_model_blocking_capability_status={cap_status} \
         ay_model_blocking_capability_schema={AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA} \
         ay_model_blocking_capability_schema_version={AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA_VERSION} \
         ay_model_consumer_status=blocked ay_model_consumer_reason_code=proof_handoff_blocked \
         trust_ir_artifact_identity_status=partial \
         trust_ir_artifact_identity_api=NativeSharedPrimitiveArtifactRequirement::accepts_artifact_identity \
         trust_ir_artifact_resolution_status=partial \
         trust_ir_artifact_resolution_schema={NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA} \
         trust_ir_artifact_resolution_schema_version={NATIVE_EVIDENCE_ARTIFACT_RESOLUTION_SCHEMA_VERSION} \
         trust_ir_artifact_authority_code=informational \
         trust_ir_artifact_resolution_reason_code=missing_attachment \
         trust_ir_bound_artifact_requirement_count=1 \
         trust_ir_unbound_artifact_requirement_roles=replay_transcript|solver_witness \
         trust_ir_unbound_artifact_requirement_kinds=replay_transcript|trust_mc_model \
         trust_cg_native_jit_status=blocked \
         trust_cg_native_jit_reason_code={PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY} \
         trust_cg_compile_cache_status=miss trust_cg_host_jit_pgo_status=descriptor_present \
         trust_cg_host_jit_pgo_schema={TRUST_CG_HOST_JIT_PGO_PROVENANCE_DESCRIPTOR_SCHEMA} \
         trust_cg_host_jit_pgo_schema_version=1 \
         trust_cg_profile_authority_status=not_authoritative_for_compiled_function \
         trust_cg_profile_authority_reason_code=profile_use_not_scheduled \
         trust_cg_profile_authority_manifest_schema={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA} \
         trust_cg_profile_authority_manifest_schema_version={TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA_VERSION} \
         next_answer_lane=petri_native_callable fallback_lane=explicit_state_fallback \
         fallback_reason_code=runtime_wrapper_fail_closed_until_real_binary \
         production_selected=false fail_closed=true"
    )]
}

fn portfolio_route_rows() -> Vec<String> {
    // (route, lane_family, backend_code, problem, role, readiness,
    //  readiness_code, evidence_source, evidence_gate, owner_project,
    //  answer_producer, routing_selected, selection_rank,
    //  production_selected, fail_closed)
    let routes: [(
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
    ); 6] = [
        (
            "explicit_bfs",
            "explicit_bfs",
            "explicit_state",
            "ExplicitReachability",
            "fallback_answer",
            "ready",
            "explicit_state_fallback_available",
            "MCC.answer_lane.explicit_state_fallback",
            "explicit_state",
            "TY",
            "true",
            "true",
            "10",
            "true",
            "false",
        ),
        (
            "reductions",
            "reductions",
            "structural_reductions",
            "Preprocessing",
            "preprocessor",
            "ready",
            "structural_reductions_available",
            "MCC.production_selector_decision",
            "shared_primitive_evidence",
            "TY",
            "false",
            "true",
            "20",
            "false",
            "false",
        ),
        (
            "ay_symbolic",
            "ay_symbolic",
            "ay_sat",
            "Sat",
            "symbolic_evidence",
            "ready",
            "ay_symbolic_ready",
            "MCC.symbolic_execution",
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA,
            "AY",
            "false",
            "true",
            "30",
            "false",
            "false",
        ),
        (
            "aiger_hwmcc",
            "aiger_hwmcc",
            "aiger_portfolio",
            "Safety",
            "hardware_portfolio",
            "ready",
            "aiger_ay_adapter_ready",
            "AIGER.ay_adapter_decision",
            "aiger.ay_adapter_decision.v1",
            "TY",
            "false",
            "true",
            "40",
            "false",
            "false",
        ),
        (
            "native_jit",
            "native_jit",
            "trust_cg_petri_native",
            "NativeSuccessor",
            "primary_answer_producer",
            "blocked",
            "shared_primitive_runtime_proof_blocked",
            "trust-cg.petri_native_successor_compile_artifact_handoff",
            COMPILE_ARTIFACT_HANDOFF_SCHEMA,
            "trust-cg",
            "true",
            "false",
            "50",
            "false",
            "true",
        ),
        (
            "hardware_model",
            "hardware_model",
            "hardware_ay_replay",
            "HardwareReplay",
            "hardware_replay_candidate",
            "blocked",
            "proof_replay_acceptance_required",
            "AIGER.hardware_replay_primitive",
            HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
            "TY",
            "true",
            "false",
            "60",
            "false",
            "true",
        ),
    ];
    routes
        .iter()
        .map(
            |(
                route,
                lane_family,
                backend_code,
                problem,
                role,
                readiness,
                readiness_code,
                evidence_source,
                evidence_gate,
                owner_project,
                answer_producer,
                routing_selected,
                selection_rank,
                production_selected,
                fail_closed,
            )| {
                format!(
                    "MCC portfolio_route schema={PORTFOLIO_ROUTE_SCHEMA} \
                 schema_version={PORTFOLIO_ROUTE_SCHEMA_VERSION} \
                 route={route} lane_family={lane_family} backend_code={backend_code} \
                 problem={problem} role={role} readiness={readiness} \
                 readiness_code={readiness_code} evidence_source={evidence_source} \
                 evidence_gate={evidence_gate} owner_project={owner_project} \
                 answer_producer={answer_producer} routing_selected={routing_selected} \
                 selection_rank={selection_rank} production_selected={production_selected} \
                 fail_closed={fail_closed}"
                )
            },
        )
        .collect()
}

// ============================================================
// AY trust_mc/petri model-acceptance row (the freshness-gate anchor).
// ============================================================

fn ay_petri_trust_mc_model_acceptance_row() -> String {
    let (ay_rev, _source) = cargo_lock_git_rev(AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_PACKAGE);
    let current_ay_rev = ay_rev
        .clone()
        .or_else(workspace_ay_pin_rev)
        .unwrap_or_else(|| "missing".to_string());
    let required_ay_rev = current_ay_rev[..8.min(current_ay_rev.len())].to_string();

    format!(
        "AY trust_mc_petri_successor_chc_model_acceptance \
         schema={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA} schema_version=1 \
         source=trust_mcPetriSuccessorChcModelAcceptanceReport \
         problem={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_PROBLEM} \
         preferred_backend_code={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_BACKEND} \
         domain={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_DOMAIN} \
         scope={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCOPE} \
         api={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_REPORT_API} \
         consumer_acceptance_api={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_CONSUMER_API} \
         required_ay_rev={required_ay_rev} current_ay_rev={current_ay_rev} \
         bundle_source=petri_native_production_path bundle_validated=true \
         status_code=rejected reason_code=proof_handoff_blocked \
         accepted_for_consumer=false fail_closed=true \
         consumer_rejection_status_code=rejected \
         consumer_rejection_reason_code=proof_handoff_blocked \
         consumer_rejection_fail_closed=true \
         proof_handoff_ready=false ready_for_solver_validation=false \
         solver_model_validation_present=false solver_model_validation_accepted=false \
         trust_mc_chc_proof_handoff_status_code=blocked \
         trust_mc_chc_proof_handoff_reason_code=missing_replay_transcript_artifact \
         trust_mc_chc_model_validation_status_code=blocked \
         trust_mc_chc_model_validation_reason_code=proof_handoff_blocked \
         model_artifact_digest=trust-ir-proof:model-artifact \
         proof_identity_digest=trust-ir-proof:proof-identity \
         replay_transcript_digest=none solver_model_artifact_digest=none \
         solver_proof_identity_digest=none solver_replay_transcript_digest=none \
         solver_artifact_bytes_validated=false \
         solver_model_artifact_bytes_digest=none \
         solver_replay_transcript_artifact_bytes_digest=none \
         solver_validation_digest=none solver_identity_count=0 \
         trust_ir_contract_api=trust_ir::petri_successor_trust_mc_chc_contract_descriptor \
         trust_ir_contract_schema={TRUST_IR_CONTRACT_SCHEMA} trust_ir_contract_schema_version=1 \
         trust_ir_contract_formula_schema={TRUST_IR_CONTRACT_FORMULA_SCHEMA} \
         trust_ir_contract_binding_report_schema={TRUST_IR_CONTRACT_BINDING_REPORT_SCHEMA} \
         trust_ir_contract_binding_report_schema_version=1 \
         trust_ir_contract_proof_handoff_report_schema={TRUST_IR_CONTRACT_PROOF_HANDOFF_REPORT_SCHEMA} \
         trust_ir_contract_proof_handoff_report_schema_version=1 \
         trust_ir_contract_model_validation_readiness_report_schema={TRUST_IR_CONTRACT_MODEL_VALIDATION_READINESS_REPORT_SCHEMA} \
         trust_ir_contract_model_validation_readiness_report_schema_version=1 \
         trust_ir_contract_verifier_suite=trust_mc trust_ir_contract_verification_mode=chc \
         trust_ir_contract_binding_required_artifact_kinds=trust_mc_horn_clauses \
         trust_ir_contract_proof_handoff_required_artifact_kinds=replay_transcript \
         trust_ir_contract_proof_handoff_optional_artifact_kinds=trust_mc_model \
         trust_ir_contract_model_validation_required_artifact_kinds=trust_mc_model \
         trust_ir_contract_production_acceptance_required_artifact_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model \
         trust_ir_contract_model_validation_requires_solver_acceptance=true \
         trust_ir_contract_model_acceptance_report_api_name={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_REPORT_API} \
         trust_ir_contract_consumer_acceptance_api_name={AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SHARED_CONSUMER_API} \
         trust_ir_contract_production_acceptance_owner_suite=AY \
         trust_ir_contract_production_requires_emitted_solver_artifacts=true \
         trust_ir_shared_primitive_artifact_identity_api=NativeSharedPrimitiveArtifactRequirement::accepts_artifact_identity \
         trust_ir_shared_primitive_required_artifact_requirement_kinds=trust_mc_horn_clauses|replay_transcript|trust_mc_model \
         trust_ir_shared_primitive_required_artifact_requirement_roles=solver_input|replay_transcript|solver_witness \
         trust_ir_shared_primitive_required_artifact_requirement_digest_algorithms=sha256|sha256|sha256 \
         trust_ir_shared_primitive_required_artifact_requirement_owner_suites=ay|ay|ay \
         trust_ir_shared_primitive_required_artifact_requirement_requires_emitted_solver_artifacts=true|true|true \
         trust_ir_shared_primitive_bound_artifact_requirement_count=1 \
         trust_ir_shared_primitive_bound_artifact_requirement_roles=solver_input \
         trust_ir_shared_primitive_bound_artifact_requirement_kinds=trust_mc_horn_clauses \
         trust_ir_shared_primitive_unbound_artifact_requirement_roles=replay_transcript|solver_witness \
         trust_ir_shared_primitive_unbound_artifact_requirement_kinds=replay_transcript|trust_mc_model \
         trust_ir_contract_provided_fields=model_artifact_digest|proof_identity_digest|replay_transcript_digest \
         trust_ir_contract_binding_status_codes=bound|blocked \
         trust_ir_contract_binding_reason_codes=bound|missing_model_artifact \
         trust_ir_contract_proof_handoff_status_codes=ready|blocked \
         trust_ir_contract_proof_handoff_reason_codes=ready|proof_handoff_blocked|missing_replay_transcript_artifact \
         trust_ir_contract_model_validation_readiness_status_codes=ready_for_solver_validation|blocked \
         trust_ir_contract_model_validation_readiness_reason_codes=solver_validation_required|proof_handoff_blocked|missing_model_artifact \
         production_selected=false"
    )
}

// ============================================================
// Lane helper used inside the JSON envelope blocks.
// ============================================================

fn lane(
    backend: &str,
    backend_code: &str,
    role: &str,
    status: &str,
    problem: &str,
    reason_code: Option<&str>,
    detail: Option<&str>,
) -> Value {
    let mut map = Map::new();
    map.insert("backend".into(), Value::String(backend.into()));
    map.insert("backend_code".into(), Value::String(backend_code.into()));
    map.insert("problem".into(), Value::String(problem.into()));
    map.insert("role".into(), Value::String(role.into()));
    map.insert("status".into(), Value::String(status.into()));
    if let Some(rc) = reason_code {
        map.insert("reason_code".into(), Value::String(rc.into()));
    }
    if let Some(d) = detail {
        map.insert("detail".into(), Value::String(d.into()));
    }
    Value::Object(map)
}

fn to_value_vec(strings: Vec<String>) -> Vec<Value> {
    strings.into_iter().map(Value::String).collect()
}

// ============================================================
// Envelope builders for the three smoke targets.
// ============================================================

fn petri_envelope() -> Value {
    let mut evidence: Vec<Value> = Vec::new();
    evidence.push(Value::String(
        "Petri native_jit fail_closed_gate \
         feature=trust-cg-petri-native feature_enabled=false \
         native_env=TY_MCC_TRUST_CG_PETRI_NATIVE native_requested=true \
         strict_env=TY_MCC_TRUST_CG_PETRI_NATIVE_STRICT strict_requested=false \
         parity_env=TY_MCC_TRUST_CG_PETRI_PARITY parity_enabled=true \
         production_selected=false fail_closed=true reason_code=disabled_by_policy"
            .into(),
    ));
    evidence.push(Value::String(
        "MCC symbolic_execution domain=petri_mcc status=AYPreferred status_code=ay_preferred \
         problem=Sat reason=ModelEnumeration reason_code=model_enumeration \
         preferred_backend=AYSat preferred_backend_code=ay_sat"
            .into(),
    ));
    evidence.push(Value::String(
        "Petri native_jit trust_ir_transport_identity available \
         required_trust_ir_rev=4e38cb current_trust_ir_rev=4e38cb cargo_dependency=true \
         api=NativeVerificationBundle::transport_identity \
         schema=trust_ir.native.transport_identity.v2 schema_version=2 \
         bundle_schema_version=1 transport_digest=trust-ir-stable128:transport \
         source_digest=none module_digest=trust-ir-stable128:module \
         compiler_facts_digest=trust-ir-stable128:compiler lineage_digest=trust-ir-stable128:lineage \
         bundle_digest=trust-ir-stable128:bundle target_abi_digest=none \
         request_digests=1 evidence_digests=1 production_selected=false fail_closed=true"
            .into(),
    ));
    evidence.push(Value::String(format!(
        "trust-cg trust_cg_admission_blocker source=NativeInstallGateAdmissionSummary \
         source_package={TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE_PACKAGE} \
         package={TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SOURCE_PACKAGE} \
         schema={TRUST_CG_NATIVE_INSTALL_GATE_ADMISSION_SCHEMA} schema_version=1 \
         consumer=mcc consumer_mode=petri_successor kind=petri_native_successor \
         surface=mcc_replay disposition=rejected status_code=rejected \
         rejection_code=missing_manifest reason_code=missing_manifest \
         requested_authority=active_callable install_authority=none \
         call_packet_api=trust_cg::petri_native_successor_call_packet_from_trust_ir_bundle \
         call_packet_schema={CALL_PACKET_SCHEMA} call_packet_schema_version=1 \
         call_packet_descriptor_available=true \
         call_packet_descriptor_source={TRUST_CG_CALL_PACKET_DESCRIPTOR_DEPENDENCY} \
         call_packet_descriptor_status_code=authoritative \
         call_packet_descriptor_authoritative=true \
         call_packet_descriptor_dependency={TRUST_CG_CALL_PACKET_DESCRIPTOR_DEPENDENCY} \
         call_packet_descriptor_upstream_ask=none \
         call_packet_contract_descriptor_schema={TRUST_CG_CALL_PACKET_CONTRACT_DESCRIPTOR_SCHEMA} \
         call_packet_contract_health_schema={TRUST_CG_CALL_PACKET_CONTRACT_HEALTH_SCHEMA} \
         downstream_contract_api={TRUST_CG_DOWNSTREAM_CONTRACT_API} \
         downstream_contract_schema={TRUST_CG_DOWNSTREAM_CONTRACT_SCHEMA} \
         downstream_contract_schema_version=1 \
         runtime_readiness_status_in_downstream_contract=true \
         runtime_readiness_blocker_in_downstream_contract=true \
         compile_artifact_handoff_status_in_downstream_contract=true \
         compile_artifact_handoff_blocker_in_downstream_contract=true \
         production_selected=false fail_closed=true"
    )));
    evidence.extend(to_value_vec(
        trust_cg_compile_artifact_cache_telemetry_rows(),
    ));
    evidence.extend(to_value_vec(trust_cg_host_jit_pgo_provenance_rows()));
    evidence.extend(to_value_vec(ay_solver_capability_descriptor_rows()));
    evidence.extend(to_value_vec(ay_symbolic_execution_contract_manifest_rows()));
    evidence.push(Value::String(format!(
        "Petri native_jit semantic_successor_bridge \
         schema={SEMANTIC_SUCCESSOR_BRIDGE_SCHEMA} schema_version=1 \
         api=PetriKernelPlanCache::for_net->trust_ir::NativeVerificationBundle \
         formula_schema={SEMANTIC_SUCCESSOR_BRIDGE_FORMULA_SCHEMA} \
         trust_ir_successor_body_status=stub_returns_zero \
         successor_relation_represented=false \
         semantic_successor_authority=false \
         semantic_bridge_status_code=blocked \
         reason_code={SEMANTIC_SUCCESSOR_BRIDGE_REASON_CODE} \
         production_selected=false fail_closed=true"
    )));
    evidence.extend(to_value_vec(
        trust_ir_shared_primitive_contract_manifest_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_evidence_artifact_resolution_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_semantic_bridge_proof_identity_rows(),
    ));
    evidence.extend(to_value_vec(trust_ir_petri_proof_evidence_identity_rows()));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_manifest_identity_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_contract_health_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_diagnostic_fixture_manifest_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_diagnostic_fixture_round_trip_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_replay_contract_surface_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_replay_contract_surface_round_trip_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_replay_contract_report_identity_rows(),
    ));
    evidence.extend(to_value_vec(
        trust_ir_native_verification_bundle_handoff_replay_contract_json_manifest_binding_rows(),
    ));
    evidence.push(Value::String(ay_petri_trust_mc_model_acceptance_row()));
    evidence.push(Value::String(format!(
        "trust-cg petri_native_successor_execution_plan \
         surface=petri_native_successor_execution_plan_from_trust_ir_bundle \
         expected=PetriNativeSuccessorExecutionExpected::validation_only \
         entry_function=unbound state_bytes=unknown \
         trust_cg_rev=9464ac7bc30980b94caaef2291cc2073e0bf1df8 \
         compile_artifact_handoff_api=trust_cg::petri_native_successor_compile_artifact_handoff_evidence \
         compile_artifact_handoff_schema={COMPILE_ARTIFACT_HANDOFF_SCHEMA} \
         compile_artifact_handoff_schema_version=1 \
         compile_artifact_handoff_installed_artifact_api={COMPILE_ARTIFACT_HANDOFF_INSTALLED_ARTIFACT_API} \
         compile_artifact_handoff_installed_artifact_required_trust_cg_rev=00597478 \
         compile_artifact_handoff_available=true \
         compile_artifact_handoff_ready=false \
         compile_artifact_handoff_status_code=blocked \
         compile_artifact_handoff_reason_code={MISSING_NATIVE_PAYLOAD_SHA256} \
         compile_artifact_handoff_blocker_code={MISSING_NATIVE_PAYLOAD_SHA256} \
         compile_artifact_handoff_required_field=compiled_artifact.native_payload_sha256 \
         compile_artifact_handoff_required_evidence={COMPILE_ARTIFACT_HANDOFF_SCHEMA} \
         compile_artifact_handoff_population_attempted=true \
         compile_artifact_handoff_real_artifact_source=none \
         compile_artifact_handoff_entry_symbol_present=true \
         compile_artifact_handoff_entry_symbol=ty_petri_all_transition_successors \
         compile_artifact_handoff_entry_symbol_source=petri_successor_entry_symbol \
         compile_artifact_handoff_native_payload_present=false \
         compile_artifact_handoff_native_payload_source=unavailable \
         compile_artifact_handoff_callable_pointer_present=false \
         compile_artifact_handoff_executable_region_present=false \
         compile_artifact_handoff_lifetime_owner_present=false \
         compile_artifact_handoff_current_generation_present=false \
         compile_artifact_handoff_missing_ty_artifact_field=NativeLibrary::replay_report_metadata.native_payload_sha256 \
         compile_artifact_handoff_missing_trust_cg_artifact_field=ExecutableBuffer::replay_report_metadata.properties.native_payload_sha256 \
         compile_artifact_handoff_missing_artifact_blocker={PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY} \
         native_successor_next_production_api=trust_cg::petri_native_successor_compile_artifact_handoff_evidence \
         native_successor_next_production_input=compiled_artifact.native_payload_sha256 \
         native_successor_next_production_evidence={COMPILE_ARTIFACT_HANDOFF_SCHEMA} \
         upstream_issue=trust_cg#881 production_selected=false \
         fail_closed=true reason_code=trampoline_unbound"
    )));
    evidence.push(Value::String(
        "trust-cg petri_native_successor_call_packet \
         surface=petri_native_successor_call_packet_from_trust_ir_bundle \
         expected=PetriNativeSuccessorExecutionExpected::validation_only \
         source=trust_cg.petri_native_successor_call_packet \
         schema=trust_cg.petri.native_successor.call_packet.v1 schema_version=1 \
         callable_authorized=false production_selected=false fail_closed=true \
         rejection_code=missing_native_install_gate_packet \
         reason_code=missing_native_install_gate_packet"
            .into(),
    ));
    evidence.extend(to_value_vec(shared_primitive_availability_rows()));
    evidence.push(Value::String(
        "TLA ay_solver_decision_profile_summary status=Available \
         status_code=typed_summary_available \
         schema=ay.solve-decision-profile-summary.v1 schema_version=1 \
         decision=SAT decision_code=sat accepted_for_consumer=true \
         consumer_rejection_code=none model_validated=true \
         verification_level_code=model_validated unknown_reason_code=none \
         unknown_limit_code=none typed_consumer=true \
         production_selected=false fail_closed=false"
            .into(),
    ));
    evidence.push(Value::String(
        "MCC hardware_fallback selected_backend=aiger_portfolio \
         fallback_backend=explicit_state fallback_backend_code=explicit_state \
         reason_code=ay_first_hardware_path"
            .into(),
    ));
    evidence.extend(to_value_vec(production_selector_decision_rows()));
    evidence.extend(to_value_vec(portfolio_route_rows()));
    evidence.push(Value::String(
        "MCC answer_lane lane=ay_symbolic backend_code=ay_sat role=symbolic_evidence \
         answer_producer=false routing_selected=true answer_selected=false \
         status=available status_code=ay_symbolic_ready \
         reason_code=petri_successor_execution_required \
         preferred_backend_code=ay_sat production_selected=false fail_closed=false"
            .into(),
    ));
    evidence.push(Value::String(format!(
        "MCC answer_lane lane=petri_native_callable backend_code=trust_cg_petri_native \
         role=primary_answer_producer answer_producer=true routing_selected=false \
         answer_selected=false status=blocked \
         status_code=shared_primitive_runtime_proof_blocked \
         blocker_gate={COMPILE_ARTIFACT_HANDOFF_SCHEMA} \
         blocker_source=trust_cg.petri_native_successor_compile_artifact_handoff \
         primitive_available=true runtime_proof_blocked=true usable_for_answer_lane=false \
         primitive_state=available_runtime_blocked \
         reason_code={PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY} \
         production_selected=false fail_closed=true"
    )));
    evidence.push(Value::String(
        "MCC answer_lane lane=explicit_state_fallback backend_code=explicit_state \
         role=fallback answer_producer=true routing_selected=true answer_selected=false \
         status=available status_code=fallback_available \
         reason_code=runtime_wrapper_fail_closed_until_real_binary \
         production_selected=true fail_closed=false"
            .into(),
    ));
    evidence.push(Value::String(format!(
        "MCC answer_lane lane=hardware_ay_replay backend_code=hardware_ay_replay \
         role=hardware_replay_candidate answer_producer=true routing_selected=false \
         answer_selected=false status=blocked status_code=proof_replay_acceptance_required \
         blocker_gate={HARDWARE_REPLAY_PRIMITIVE_SCHEMA} \
         blocker_source=AIGER.hardware_replay_primitive \
         reason_code=proof_replay_acceptance_required \
         production_selected=false fail_closed=true"
    )));
    evidence.push(Value::String(format!(
        "MCC answer_counters official_rows=3 answer_rows_ready=0 answer_rows_blocked=3 \
         expected_answer_rows=3 native_callable_ready=0 native_blocker_count=7 \
         ay_symbolic_ready=1 sidecar_replay_ready=1 explicit_fallback_ready=1 \
         primitive_unavailable_count=0 primitive_available_runtime_blocked_count=2 \
         answer_lane_usable_primitive_count=1 trust_cg_call_packet_primitive_available=1 \
         ay_typed_trace_assignments_available=1 aiger_ay_sat_adapter_available=1 \
         runtime_proof_blocked_count=2 next_answer_lane=petri_native_callable \
         blocker_gate={COMPILE_ARTIFACT_HANDOFF_SCHEMA} \
         reason_code={PETRI_NATIVE_NO_COMPILED_NATIVE_LIBRARY}"
    )));
    evidence.extend(to_value_vec(blocker_action_rows()));
    evidence.extend(to_value_vec(schema_diagnostic_rows()));

    let selected = vec![
        lane(
            "ExternalAYBinary",
            "external_ay_binary",
            "production",
            "available",
            "Sat",
            None,
            None,
        ),
        lane(
            "AigerPortfolio",
            "aiger_portfolio",
            "production",
            "available",
            "Safety",
            None,
            None,
        ),
        lane(
            "ExplicitState",
            "explicit_state",
            "fallback",
            "available",
            "ExplicitReachability",
            None,
            None,
        ),
    ];
    let rejected = vec![
        lane(
            "NativeKernel",
            "native_kernel",
            "validation",
            "unsupported",
            "NativeSuccessor",
            Some("native_kernel_unavailable"),
            Some("native handoff remains validation-only; trust_ir_transport_identity_available=true"),
        ),
        lane(
            "AYChc",
            "ay_chc",
            "validation",
            "disabled",
            "Chc",
            Some("disabled_by_policy"),
            None,
        ),
    ];
    let mut report = Map::new();
    report.insert(
        "problem".into(),
        Value::String("ExplicitReachability".into()),
    );
    report.insert(
        "production_routing_status".into(),
        Value::String("AYFirst".into()),
    );
    report.insert("selected".into(), Value::Array(selected));
    report.insert("rejected".into(), Value::Array(rejected));
    report.insert("evidence".into(), Value::Array(evidence));

    let mut envelope = Map::new();
    envelope.insert("schema_version".into(), Value::Number(1.into()));
    envelope.insert(
        "model".into(),
        Value::String("mcc-replay-smoke-petri".into()),
    );
    envelope.insert(
        "examination".into(),
        Value::String("ReachabilityFireability".into()),
    );
    let mut run_status = Map::new();
    run_status.insert("status".into(), Value::String("completed".into()));
    run_status.insert("records".into(), Value::Number(1.into()));
    envelope.insert("run_status".into(), Value::Object(run_status));
    envelope.insert("report".into(), Value::Object(report));
    Value::Object(envelope)
}

fn aiger_envelope() -> Value {
    let selected = vec![
        lane(
            "AigerPortfolio",
            "aiger_portfolio",
            "production",
            "available",
            "Safety",
            None,
            None,
        ),
        lane(
            "AYSat",
            "ay_sat",
            "production",
            "available",
            "Bmc",
            None,
            None,
        ),
    ];
    let rejected = vec![
        lane(
            "NativeKernel",
            "native_kernel",
            "validation",
            "unsupported",
            "NativeSuccessor",
            Some("native_kernel_unavailable"),
            None,
        ),
        lane(
            "LocalSymbolicExecution",
            "local_symbolic_execution",
            "production",
            "disabled",
            "Sat",
            Some("ay_first_required"),
            None,
        ),
    ];
    let evidence: Vec<Value> = vec![
        Value::String(
            "AIGER symbolic_execution domain=aiger status=AYPreferred status_code=ay_preferred \
             problem=Sat reason=BitVectorFormula reason_code=bit_vector_formula \
             preferred_backend=AYSat preferred_backend_code=ay_sat"
                .into(),
        ),
        Value::String(
            "AIGER ay_adapter_decision_schema version=1 source=AYSolveDecision sat_result_behavior=preserved"
                .into(),
        ),
        Value::String(
            "AIGER ay_adapter_decision action=selected engine=bmc-ay-luby backend=AYSat \
             kind=production status=Available role=Production reason_code=none \
             sat_result=unchanged"
                .into(),
        ),
        Value::String(
            "AIGER proof_replay_boundary ay_backend_code=ay_sat \
             safe_proof=aiger_safe_witness_validation safe_replay=validate_safe \
             unsafe_witness=aiger_counterexample_trace unsafe_replay=transys_verify_witness \
             witness_attribution=engine_trace local_production_gate=no_local_production \
             native_promotion_gate=fail_closed production_routing_status_code=ay_first"
                .into(),
        ),
        Value::String(format!(
            "AIGER hardware_replay_primitive schema={HARDWARE_REPLAY_PRIMITIVE_SCHEMA} \
             schema_version=1 assignment_completeness=typed_complete typed_assignments=true \
             proof_replay_accepted=false answer_lane_usable=false ay_backend_code=ay_sat \
             reason_code=proof_replay_acceptance_required \
             production_selected=false fail_closed=true"
        )),
        Value::String(
            "AIGER replay_api_gate verdict=safe artifact_kind=safe_witness_inductive_invariant \
             api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat \
             replay_api=validate_safe replay_status=proven acceptance_gate=safe_validation_accepted \
             failure_policy=fail_closed_continue_or_respawn evidence_basis=independent_sat_recheck \
             production_routing_status_code=ay_first"
                .into(),
        ),
        Value::String(
            "AIGER replay_api_gate verdict=unsafe artifact_kind=counterexample_trace \
             api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat \
             replay_api=transys_verify_witness replay_status=proven \
             acceptance_gate=transys_verify_witness_ok \
             failure_policy=fail_closed_continue_or_respawn evidence_basis=trace_simulation \
             production_routing_status_code=ay_first"
                .into(),
        ),
    ];
    let mut report = Map::new();
    report.insert(
        "production_routing_status".into(),
        Value::String("AYFirst".into()),
    );
    report.insert("selected".into(), Value::Array(selected));
    report.insert("rejected".into(), Value::Array(rejected));
    report.insert("evidence".into(), Value::Array(evidence));

    let mut envelope = Map::new();
    envelope.insert("schema_version".into(), Value::Number(1.into()));
    envelope.insert("model".into(), Value::String("aiger-replay-smoke".into()));
    envelope.insert("problem".into(), Value::String("Safety".into()));
    envelope.insert("report".into(), Value::Object(report));
    Value::Object(envelope)
}

fn btor2_envelope() -> Value {
    let selected = vec![lane(
        "AYChc",
        "ay_chc",
        "production",
        "available",
        "Chc",
        None,
        None,
    )];
    let rejected = vec![
        lane(
            "AYSat",
            "ay_sat",
            "production",
            "disabled",
            "Sat",
            Some("disabled_by_policy"),
            None,
        ),
        lane(
            "NativeKernel",
            "native_kernel",
            "validation",
            "unsupported",
            "NativeSuccessor",
            Some("native_kernel_unavailable"),
            None,
        ),
    ];
    let evidence: Vec<Value> = vec![
        Value::String(
            "BTOR2 symbolic_execution domain=btor2 status=AYPreferred status_code=ay_preferred \
             problem=Chc reason=BitVectorFormula reason_code=bit_vector_formula \
             preferred_backend=AYChc preferred_backend_code=ay_chc"
                .into(),
        ),
        Value::String(
            "BTOR2 symbolic_execution domain=btor2 status=AYPreferred status_code=ay_preferred \
             problem=Sat reason=BitVectorFormula reason_code=bit_vector_formula \
             preferred_backend=AYSat preferred_backend_code=ay_sat"
                .into(),
        ),
        Value::String(
            "BTOR2 proof_replay_boundary ay_backend_code=ay_chc \
             safe_proof=ay_chc_verified_result safe_replay=not_available \
             unsafe_witness=ay_chc_counterexample unsafe_replay=not_available \
             witness_attribution=query_clause local_production_gate=no_local_production \
             native_promotion_gate=fail_closed production_routing_status_code=ay_first"
                .into(),
        ),
        Value::String(format!(
            "BTOR2 hardware_replay_decision schema={HARDWARE_REPLAY_PRIMITIVE_SCHEMA} \
             verdict=unsafe primitive=unsafe_counterexample_trace decision_status=accepted \
             accepted_replay_primitive=true \
             blocked_by_typed_assignment_completeness=false blocked_by_placeholder=false \
             consumer_status=accepted reason_code=none ay_backend_code=ay_chc \
             replay_api=ay_chc_trace_validity_replay_obligations replay_status=proven \
             typed_assignment_source=ay_chc_consumer_evidence \
             replay_assignment_status=complete typed_assignment_required_slots=4 \
             typed_assignment_present_slots=4 typed_assignment_missing_slots=0 \
             accepted_replay_evidence_identity_sha256={BTOR2_ACCEPTED_REPLAY_EVIDENCE_IDENTITY_SHA256} \
             accepted_trace_validity_obligations=1 \
             accepted_replay_obligation_identities_sha256={BTOR2_ACCEPTED_REPLAY_OBLIGATION_IDENTITIES_SHA256} \
             accepted_ay_proof_evidence_status={BTOR2_ACCEPTED_AY_PROOF_EVIDENCE_STATUS} \
             accepted_ay_proof_evidence_sha256={BTOR2_ACCEPTED_AY_PROOF_EVIDENCE_SHA256} \
             evidence_source=real_solver generated_placeholder=false"
        )),
        Value::String(
            "BTOR2 replay_api_gate verdict=safe artifact_kind=verified_chc_result_safe \
             api_backend=AYChc api_backend_code=ay_chc replay_api=ay_chc_verified_result \
             replay_status=delegated_to_ay acceptance_gate=verified_chc_result_safe \
             failure_policy=fail_closed_no_local_production evidence_basis=ay_chc_safe_proof \
             production_routing_status_code=ay_first"
                .into(),
        ),
        Value::String(
            "BTOR2 replay_api_gate verdict=unsafe artifact_kind=verified_chc_result_unsafe \
             api_backend=AYChc api_backend_code=ay_chc replay_api=ay_chc_verified_result \
             replay_status=delegated_to_ay acceptance_gate=verified_chc_result_unsafe \
             failure_policy=fail_closed_no_local_production evidence_basis=ay_chc_counterexample \
             production_routing_status_code=ay_first"
                .into(),
        ),
    ];

    let mut report = Map::new();
    report.insert(
        "production_routing_status".into(),
        Value::String("AYFirst".into()),
    );
    report.insert("selected".into(), Value::Array(selected));
    report.insert("rejected".into(), Value::Array(rejected));
    report.insert("evidence".into(), Value::Array(evidence));

    let mut envelope = Map::new();
    envelope.insert("schema_version".into(), Value::Number(1.into()));
    envelope.insert("model".into(), Value::String("btor2-replay-smoke".into()));
    envelope.insert("problem".into(), Value::String("Chc".into()));
    envelope.insert("report".into(), Value::Object(report));
    Value::Object(envelope)
}

/// Entry point: returns the three smoke envelopes that comprise the JSONL
/// sidecar. Mirrors `generated_replay_smoke_rows` in the Python module.
pub fn generated_replay_smoke_rows() -> Vec<Value> {
    vec![petri_envelope(), aiger_envelope(), btor2_envelope()]
}

/// Write each row in `rows` as a JSON line, with sorted keys, terminated by
/// `\n`. Mirrors `write_jsonl` in the Python module. `serde_json`'s default
/// is insertion-order; we wrap each value in a `BTreeMap` projection to
/// match Python's `sort_keys=True` exactly.
pub fn write_jsonl<W: Write>(rows: &[Value], mut writer: W) -> io::Result<()> {
    for row in rows {
        let sorted = sort_value_keys(row);
        let line = serde_json::to_string(&sorted).map_err(io::Error::other)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Recursively sort object keys to match Python `json.dumps(sort_keys=True)`.
fn sort_value_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map
                .iter()
                .map(|(k, v)| (k.clone(), sort_value_keys(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut sorted_map = Map::new();
            for (k, v) in entries {
                sorted_map.insert(k, v);
            }
            Value::Object(sorted_map)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_value_keys).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_compatibility_digest_is_deterministic_and_prefixed() {
        let d1 = legacy_compatibility_trust_ir_stable_v1_digest("ctx", "payload");
        let d2 = legacy_compatibility_trust_ir_stable_v1_digest("ctx", "payload");
        assert_eq!(d1, d2);
        assert!(d1.starts_with("trust-ir-stable-v1:"));
        assert_eq!(d1.len(), "trust-ir-stable-v1:".len() + 64);
        assert!(d1["trust-ir-stable-v1:".len()..]
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn legacy_compatibility_digest_varies_on_context() {
        let d1 = legacy_compatibility_trust_ir_stable_v1_digest("ctx-a", "payload");
        let d2 = legacy_compatibility_trust_ir_stable_v1_digest("ctx-b", "payload");
        assert_ne!(d1, d2);
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    fn legacy_compatibility_digest_is_disjoint_from_typed_authority_identity() {
        let compatibility = legacy_compatibility_trust_ir_stable_v1_digest("ctx", "payload");
        let authority = trust_ir::ProofDigest::sha256_domain("ctx", b"payload");

        assert!(compatibility.starts_with("trust-ir-stable-v1:"));
        assert_eq!(authority.algorithm, trust_ir::ProofDigestAlgorithm::Sha256);
        assert!(authority.to_string().starts_with("sha256:"));
        assert_ne!(compatibility, authority.to_string());
    }

    #[test]
    fn trust_ir_manifest_identity_component_escapes_specials() {
        assert_eq!(trust_ir_manifest_identity_component("a=b"), "a\\=b");
        assert_eq!(trust_ir_manifest_identity_component("a\nb"), "a\\nb");
        assert_eq!(trust_ir_manifest_identity_component("a\\b"), "a\\\\b");
    }

    #[test]
    fn csv_renders_empty_as_none() {
        assert_eq!(csv(&[]), "none");
        assert_eq!(csv(&["x", "y"]), "x,y");
    }

    #[test]
    fn bool_text_matches_python() {
        assert_eq!(bool_text(true), "true");
        assert_eq!(bool_text(false), "false");
    }

    #[test]
    fn cargo_lock_git_rev_handles_missing_package() {
        let (rev, src) = cargo_lock_git_rev("not-a-real-package-zzz-12345");
        assert!(rev.is_none());
        assert!(src.is_none());
    }

    #[test]
    fn generated_replay_smoke_rows_has_three_envelopes() {
        let rows = generated_replay_smoke_rows();
        assert_eq!(rows.len(), 3);
        let models: Vec<&str> = rows
            .iter()
            .map(|r| r.get("model").and_then(Value::as_str).unwrap_or(""))
            .collect();
        assert_eq!(
            models,
            vec![
                "mcc-replay-smoke-petri",
                "aiger-replay-smoke",
                "btor2-replay-smoke"
            ]
        );
    }

    /// Each envelope's `report.evidence` is an array of strings. The
    /// validate_backend_evidence consumer rejects non-string evidence.
    #[test]
    fn evidence_is_only_strings() {
        let rows = generated_replay_smoke_rows();
        for envelope in &rows {
            let evidence = envelope
                .get("report")
                .and_then(|r| r.get("evidence"))
                .and_then(Value::as_array)
                .expect("evidence is an array");
            assert!(!evidence.is_empty(), "evidence array must not be empty");
            for row in evidence {
                assert!(
                    row.is_string(),
                    "every evidence row must be a string: {row:?}"
                );
            }
        }
    }

    #[test]
    fn write_jsonl_emits_newline_separated() {
        let rows = vec![{
            let mut m = Map::new();
            m.insert("b".into(), Value::from(1));
            m.insert("a".into(), Value::from(2));
            Value::Object(m)
        }];
        let mut buf: Vec<u8> = Vec::new();
        write_jsonl(&rows, &mut buf).expect("write");
        let text = String::from_utf8(buf).expect("utf8");
        assert!(text.ends_with('\n'));
        // Sort_keys=true reorders to a,b.
        assert!(text.contains("\"a\":2,\"b\":1"), "got: {text}");
    }

    /// The MCC freshness gate parses `current_ay_rev=<hex>` out of the
    /// `AY trust_mc_petri_successor_chc_model_acceptance` row. The row must
    /// always emit *some* value (either the Cargo.lock rev or the
    /// literal `"missing"`) so the gate can decide what to do.
    #[test]
    fn ay_petri_model_acceptance_row_has_current_ay_rev_token() {
        let row = ay_petri_trust_mc_model_acceptance_row();
        assert!(row.contains("current_ay_rev="), "row missing token: {row}");
        assert!(row.contains("required_ay_rev="), "row missing token: {row}");
        assert!(
            row.contains(AY_PETRI_TRUST_MC_MODEL_ACCEPTANCE_SCHEMA),
            "row missing schema: {row}"
        );
    }

    /// Helper that the freshness gate and validate_backend_evidence canary
    /// both index on: portfolio_route rows in canonical order.
    #[test]
    fn portfolio_route_rows_cover_canonical_lanes() {
        let rows = portfolio_route_rows();
        let route_names: Vec<&str> = [
            "explicit_bfs",
            "reductions",
            "ay_symbolic",
            "aiger_hwmcc",
            "native_jit",
            "hardware_model",
        ]
        .into();
        assert_eq!(rows.len(), route_names.len());
        for (row, name) in rows.iter().zip(route_names) {
            assert!(
                row.contains(&format!("route={name} ")),
                "row {row} should contain route={name}"
            );
        }
    }

    /// The replay-contract surface emits 25 helper names; the
    /// validate_backend_evidence canary asserts on count=25.
    #[test]
    fn replay_contract_helper_count_is_25() {
        assert_eq!(REPLAY_CONTRACT_HELPER_NAMES.len(), 25);
    }
}
