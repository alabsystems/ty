// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TY trust-codegen compilation backend — turn TLA+ model-checking kernels into
//! native code via the pure-Rust `trust-cg` pipeline (zero C/LLVM dependency).
//!
//! # Pipeline
//!
//! Compilation runs in three stages:
//!
//! 1. Bytecode -> `TrustIr` (via [`tla-ir`](tla_ir))
//! 2. `TrustIr` -> trust-codegen `ISel` function (via `trust-cg-lower`)
//! 3. `ISel` -> machine code (via `trust-cg-codegen`)
//!
//! The codegen stages live behind the `native` Cargo feature; without it the
//! crate still exposes the frontend-neutral planning, identity, telemetry, and
//! ABI types but cannot emit machine code (see `is_native_available`, exported
//! with the `native` feature).
//!
//! # What this crate produces
//!
//! The model checker compiles a *fused BFS level* — a single native function
//! that walks a flat-state parent frontier and, for each parent, runs the
//! enabled actions, applies state constraints and action-property predicates to
//! each candidate successor, dedups, and checks invariants. The principal entry
//! points are the `compile_bfs_level_native*` family (e.g.
//! `compile_bfs_level_native` and
//! `compile_bfs_level_native_with_state_constraints_and_implied_actions`),
//! which return a [`bfs_level::TrustCgBfsLevelNative`]. Individual next-state and
//! invariant kernels can also be compiled via `compile_next_state_native` and
//! `compile_invariant_native_with_constants_and_layout`. These `compile_*`
//! entry points are exported only with the `native` feature.
//!
//! # Caching and identity
//!
//! Compiled native artifacts are cached on disk (see [`artifact_cache`]) and
//! loaded via `libloading` at runtime. Cache keys and reuse are driven by the
//! frontend-neutral [`compile::BatchJitArtifactIdentity`], which is derived from
//! canonical trust-ir plus stable compile options so that equivalent kernels
//! from any frontend share an artifact. [`compile::BatchJitStats`] and the
//! telemetry types record per-batch compile evidence.
//!
//! # Module map
//!
//! - [`compile`] — batch JIT compilation, identity, telemetry, and the native
//!   `compile_*` entry points.
//! - [`bfs_level`] — the fused BFS-level ABI, descriptors, and prototype.
//! - [`native_bfs`] — trust-ir generation for the fused parent loop.
//! - [`runtime_abi`] — stable ABI types shared with compiled code (flat state,
//!   fingerprints, layouts, TLA op helpers).
//! - [`runtime`] — host runtime helper symbols linked into JIT artifacts.
//! - [`artifact_cache`] — on-disk cache of compiled libraries plus metadata.
//! - [`error`] — the crate's [`TrustCgError`] type.
//!
//! Most error-returning entry points yield [`TrustCgError`]; the BFS-level ABI
//! layer additionally uses [`bfs_level::TrustCgBfsLevelError`].

// Drive the public-API surface to fully documented. `deny(missing_docs)` keeps
// it that way; it applies to every Cargo feature, so new public items (including
// `#[cfg(feature = "native")]` ones) must carry docs to build.
#![deny(missing_docs)]

// Other lints are inherited from `[lints] workspace = true` (see workspace
// Cargo.toml): clippy::pedantic at warn with the project's documented
// intentional-pattern allows. (Previously this crate re-asserted
// `#![warn(clippy::pedantic)]`, which overrode those workspace allows, and
// carried a self-canceling deny/allow(missing_docs) pair.)

pub mod artifact_cache;

/// Single blessed choke point for process-environment mutation (test/CLI
/// plumbing). Always compiled so in-crate `#[cfg(test)]` tests and the crate's
/// examples reach the same choke point. The one `env_mutation` allow lives on
/// `env_guard::raw_env_write`.
#[doc(hidden)]
pub mod env_guard;
pub mod bfs_level;
pub mod compile;
pub mod compiled_fingerprint;
pub mod compiled_liveness;
pub mod emit;
pub mod error;
pub mod lower;
pub mod native_bfs;
pub mod pgo;
pub mod prefetch;
pub mod runtime;
pub mod runtime_abi;
pub mod trust_ir_lower;
pub mod validate_ir;

pub use compile::{
    prepare_batch, BatchJitArtifactAdmissionInput, BatchJitArtifactIdentity, BatchJitOptions,
    BatchJitPreparedBatch, BatchJitStats, BatchJitSymbolContract, CompiledBfsLevel,
    CompiledBfsStep, CompiledModule, NativeExternSymbol, NativeExternSymbolOverlay, NativeLibrary,
    OptLevel, TrustCgBfsLevelNativeAction, TrustCgBfsLevelNativeImpliedAction,
    TrustCgBfsLevelNativeInvariant, TrustCgBfsLevelNativeStateConstraint, TrustCgCompilePhase,
    TrustCgCompilePhaseEvidence, TrustCgCompilePhaseMetadata, TrustCgCompilePhaseStatus,
    TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA,
    TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA_VERSION,
    TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA,
    TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION, TRUST_CG_COMPILE_PHASE_EVIDENCE_SCHEMA,
    TRUST_CG_ENTRY_COUNTER_DISPATCH_GATE_ENV,
};

pub use bfs_level::{
    ActionDescriptor, InvariantDescriptor, TrustCgBfsLevelError, TrustCgBfsLevelNative,
    TrustCgBfsLevelOutcome, TrustCgBfsLevelStatus, TrustCgBfsParentArenaAbi,
    TrustCgBfsSuccessorArenaAbi, TrustCgCompiledAction, TrustCgCompiledInvariant,
    TrustCgFusedLevelFn, TrustCgInvariantStatus, TrustCgSuccessorArena,
    TRUST_CG_BFS_LEVEL_ABI_VERSION, TRUST_CG_BFS_NO_INDEX,
};

pub use native_bfs::NativeBfsPreCallPcGuard;

#[cfg(feature = "native")]
pub use compile::{
    admit_batch_jit_artifact, batch_jit_compile_telemetry_descriptor, compile_bfs_level_native,
    compile_bfs_level_native_actions_only, compile_bfs_level_native_with_state_constraints,
    compile_bfs_level_native_with_state_constraints_and_implied_actions,
    compile_entry_invariant_native_with_chunk,
    compile_entry_invariant_native_with_chunk_and_layout,
    compile_entry_next_state_native_with_chunk,
    compile_entry_next_state_native_with_chunk_and_layout,
    compile_invariant_native_with_constants_and_layout, compile_module_native,
    compile_next_state_native, compile_next_state_native_with_constants_and_layout,
    extern_symbol_map_for_tests, is_native_available,
    petri_native_successor_runtime_readiness_from_installed_artifact,
    trust_cg_entry_counter_dispatch_gate_limit, PetriNativeSuccessorRuntimeReadinessEvidence,
};
pub use error::TrustCgError;
pub use runtime::{RuntimeHelper, RUNTIME_HELPERS};
pub use trust_ir_lower::lower_tir_to_llvm_ir;

/// Ensure that JIT execute mode is enabled.
#[cfg(feature = "native")]
pub fn ensure_jit_execute_mode() {
    trust_cg_codegen::jit::ensure_jit_execute_mode();
}

pub use lower::LoweringStats;
#[cfg(feature = "native")]
pub use trust_cg_codegen::jit_contract::ArtifactChecksum;
#[cfg(feature = "native")]
pub use trust_cg_codegen::{
    compile_artifact_cache_telemetry_descriptor,
    petri_native_successor_admission_from_trust_ir_bundle,
    petri_native_successor_call_packet_contract_descriptor,
    petri_native_successor_call_packet_from_trust_ir_bundle,
    petri_native_successor_compile_artifact_handoff_evidence,
    petri_native_successor_downstream_contract_descriptor,
    petri_native_successor_execution_authority_decision,
    petri_native_successor_execution_authority_diagnostic_fixture_manifest,
    petri_native_successor_execution_authority_healthy_diagnostic_fixture,
    petri_native_successor_execution_authority_incomplete_diagnostic_fixture,
    petri_native_successor_execution_authority_replay_identity_for_manifest_key_value_lines,
    petri_native_successor_execution_authority_replay_identity_for_manifest_rows,
    petri_native_successor_execution_authority_stale_diagnostic_fixture,
    petri_native_successor_execution_authority_summary_for_manifest_key_value_lines,
    petri_native_successor_execution_authority_summary_for_manifest_rows,
    petri_native_successor_execution_plan_from_trust_ir_bundle,
    petri_native_successor_install_packet_from_trust_ir_bundle,
    petri_native_successor_mock_executable_call_dry_run,
    petri_native_successor_production_selection_decision,
    petri_native_successor_runtime_readiness_packet,
    petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle,
    petri_native_successor_trampoline_contract,
    petri_native_successor_trust_ir_bundle_identity_descriptor,
    petri_native_successor_trust_mc_chc_shared_primitive_contract_descriptor,
    validate_native_install_gate,
    validate_petri_native_successor_call_packet_contract_descriptor_key_value_lines,
    validate_petri_native_successor_call_packet_contract_descriptor_rows,
    validate_petri_native_successor_execution_authority_diagnostic_fixture_manifest_key_value_lines,
    validate_petri_native_successor_execution_authority_diagnostic_fixture_manifest_rows,
    validate_petri_native_successor_execution_authority_manifest_key_value_lines,
    validate_petri_native_successor_execution_authority_manifest_rows,
    validate_petri_native_successor_execution_authority_summary_json_str,
    validate_petri_native_successor_execution_authority_summary_json_value,
    validate_petri_native_successor_execution_authority_summary_key_value_lines,
    validate_petri_native_successor_execution_authority_summary_rows,
    validate_petri_native_successor_execution_authority_summary_text,
    validate_petri_native_successor_trust_mc_admission_route_descriptor_json_str,
    validate_petri_native_successor_trust_mc_admission_route_descriptor_json_value,
    validate_petri_native_successor_trust_mc_admission_route_descriptor_key_value_lines,
    validate_petri_native_successor_trust_mc_admission_route_descriptor_rows,
    validate_petri_native_successor_trust_mc_admission_route_descriptor_text,
    CompileArtifactCacheBoundary, CompileArtifactCacheStatus, CompileArtifactCacheTelemetry,
    CompileArtifactCacheTelemetryDescriptor, CompileArtifactCacheTelemetryKeyValueRow,
    CompileArtifactCacheTelemetryManifestRow, CompileArtifactCacheTelemetryManifestRowKind,
    CompileArtifactCacheTelemetryRowKind, InstalledArtifact, NativeInstallGateActions,
    NativeInstallGateAdmissionSummary, NativeInstallGateAuthority, NativeInstallGateDisposition,
    NativeInstallGateExpectedBindings, NativeInstallGateInput, NativeInstallGatePacket,
    NativeInstallGatePayloadIdentity, NativeInstallGateRejectionCode, NativeInstallGateSurface,
    PetriNativeSuccessorAdmissionExpected, PetriNativeSuccessorCallPacket,
    PetriNativeSuccessorCallPacketContractDescriptor,
    PetriNativeSuccessorCallPacketContractDescriptorRow,
    PetriNativeSuccessorCallPacketContractHealthReport,
    PetriNativeSuccessorCallPacketContractHealthStatus, PetriNativeSuccessorCallableContract,
    PetriNativeSuccessorCallableLifetimeProof, PetriNativeSuccessorCallablePointer,
    PetriNativeSuccessorCompileArtifactHandoffBlocker,
    PetriNativeSuccessorCompileArtifactHandoffEvidence,
    PetriNativeSuccessorCompileArtifactHandoffInput,
    PetriNativeSuccessorDownstreamContractDescriptor,
    PetriNativeSuccessorEvidenceSurfaceDescriptor, PetriNativeSuccessorExecutableCallStatus,
    PetriNativeSuccessorExecutionAuthorityDecision,
    PetriNativeSuccessorExecutionAuthorityDiagnosticFixture,
    PetriNativeSuccessorExecutionAuthorityDiagnosticFixtureManifest,
    PetriNativeSuccessorExecutionAuthorityDiagnosticFixtureManifestEntry,
    PetriNativeSuccessorExecutionAuthorityDiagnosticFixtureManifestRow,
    PetriNativeSuccessorExecutionAuthorityDiagnosticFixtureManifestValidationReport,
    PetriNativeSuccessorExecutionAuthorityInput,
    PetriNativeSuccessorExecutionAuthorityManifestValidationReport,
    PetriNativeSuccessorExecutionAuthorityManifestValidationStatus,
    PetriNativeSuccessorExecutionAuthorityReplayIdentity,
    PetriNativeSuccessorExecutionAuthorityStatus, PetriNativeSuccessorExecutionAuthoritySummary,
    PetriNativeSuccessorExecutionAuthoritySummaryRow,
    PetriNativeSuccessorExecutionAuthoritySummaryValidationReport,
    PetriNativeSuccessorExecutionAuthoritySummaryValidationStatus,
    PetriNativeSuccessorExecutionExpected, PetriNativeSuccessorExecutionPlan,
    PetriNativeSuccessorHandoffManifestRow, PetriNativeSuccessorHandoffManifestRowKind,
    PetriNativeSuccessorMockExecutableCallBlocker, PetriNativeSuccessorMockExecutableCallGate,
    PetriNativeSuccessorMockExecutableCallReport, PetriNativeSuccessorMockExecutableCallStatus,
    PetriNativeSuccessorProductionSelectionDecision, PetriNativeSuccessorProductionSelectionStatus,
    PetriNativeSuccessorRuntimeAbiProof, PetriNativeSuccessorRuntimeReadinessBlocker,
    PetriNativeSuccessorRuntimeReadinessPacket, PetriNativeSuccessorRuntimeReadinessStatus,
    PetriNativeSuccessorSemanticBridgeBlocker, PetriNativeSuccessorSemanticBridgeEvidence,
    PetriNativeSuccessorSemanticBridgeExpected, PetriNativeSuccessorTrampolineContract,
    PetriNativeSuccessorTrustMcAdmissionRouteDescriptor,
    PetriNativeSuccessorTrustMcAdmissionRouteDescriptorRow,
    PetriNativeSuccessorTrustMcAdmissionRouteDescriptorValidationReport,
    PetriNativeSuccessorTrustMcAdmissionRouteDescriptorValidationStatus,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_ARTIFACT_REUSE_STATUS_CODES,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_BOUNDARY_CODES,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_DIGEST_FIELDS,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_IDENTITY_FIELDS,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_MANIFEST_SCHEMA,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_MANIFEST_SCHEMA_VERSION,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_METRIC_FIELDS,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_NON_REUSE_STATUS_CODES,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_OPTIONAL_FIELDS,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_OPTIONAL_IDENTITY_FIELDS,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_REQUIRED_FIELDS, COMPILE_ARTIFACT_CACHE_TELEMETRY_SCHEMA,
    COMPILE_ARTIFACT_CACHE_TELEMETRY_SCHEMA_VERSION, COMPILE_ARTIFACT_CACHE_TELEMETRY_STATUS_CODES,
    NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA,
    NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_CALLABLE_CONTRACT_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_CALLABLE_CONTRACT_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_CONTRACT_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_CONTRACT_DESCRIPTOR_ID,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_CONTRACT_DESCRIPTOR_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_CONTRACT_HEALTH_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_CONTRACT_HEALTH_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_REQUIRED_RUNTIME_EVIDENCE,
    PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA, PETRI_NATIVE_SUCCESSOR_CALL_PACKET_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_BLOCKER_CODES,
    PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_REQUIRED_FIELDS,
    PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_STATUS_CODES, PETRI_NATIVE_SUCCESSOR_CONSUMER,
    PETRI_NATIVE_SUCCESSOR_CONSUMER_MODE, PETRI_NATIVE_SUCCESSOR_DOWNSTREAM_CONTRACT_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_DOWNSTREAM_CONTRACT_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_DOWNSTREAM_CONTRACT_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_BLOCKER_CODES,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_DIAGNOSTIC_FIXTURE_MANIFEST_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_DIAGNOSTIC_FIXTURE_MANIFEST_VALIDATION_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_DIAGNOSTIC_FIXTURE_MANIFEST_VALIDATION_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_MANIFEST_ACCEPTED_REQUIRED_VALUE_KEYS,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_MANIFEST_REQUIRED_KEYS,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_MANIFEST_VALIDATION_REASON_CODES,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_MANIFEST_VALIDATION_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_MANIFEST_VALIDATION_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_MANIFEST_VALIDATION_STATUS_CODES,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_REPLAY_IDENTITY_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_REPLAY_IDENTITY_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_REQUIRED_FIELDS,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_STATUS_CODES,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_VALIDATION_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_AUTHORITY_SUMMARY_VALIDATION_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_PLAN_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_EXECUTION_PLAN_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_HANDOFF_EVIDENCE_MANIFEST_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_HANDOFF_EVIDENCE_MANIFEST_SCHEMA_VERSION, PETRI_NATIVE_SUCCESSOR_KIND,
    PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_BLOCKER_CODES,
    PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_REQUIRED_FIELDS,
    PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_MOCK_EXECUTABLE_CALL_STATUS_CODES,
    PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_REASON_CODES,
    PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_PRODUCTION_SELECTION_STATUS_CODES,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_BLOCKER_CODES,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_DESCRIPTOR as PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_TRIAL_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_PACKET_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_PACKET_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_REQUIRED_FIELDS,
    PETRI_NATIVE_SUCCESSOR_RUNTIME_READINESS_STATUS_CODES,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_BLOCKER_CODES,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_REQUIRED_FIELDS,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_BRIDGE_STATUS_CODES,
    PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_STATE_ENCODING_STABLE_BYTES_V1,
    PETRI_NATIVE_SUCCESSOR_TRAMPOLINE_ABI_STABLE_BYTES_V1,
    PETRI_NATIVE_SUCCESSOR_TRAMPOLINE_CONTRACT_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_TRAMPOLINE_CONTRACT_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_TRUST_IR_BUNDLE_IDENTITY_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_DESCRIPTOR,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_DESCRIPTOR_ID,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_DESCRIPTOR_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_DESCRIPTOR_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_DESCRIPTOR_VALIDATION_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_DESCRIPTOR_VALIDATION_SCHEMA_VERSION,
    PETRI_NATIVE_SUCCESSOR_TRUST_MC_ADMISSION_ROUTE_REQUIRED_SUMMARY_VALIDATORS,
    PETRI_NATIVE_SUCCESSOR_VECTOR_CONSTANT_LOWERING_EVIDENCE_SCHEMA,
    PETRI_NATIVE_SUCCESSOR_VECTOR_CONSTANT_LOWERING_EVIDENCE_SCHEMA_VERSION,
};
