// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Opt-in trust-codegen Petri native successor backend.
//!
//! The native bundle is derived from the validated Petri plan cache and stays
//! behind the shared install/readiness gates before any generated entrypoint is
//! called.

use crate::petri_net::PetriNet;
use tla_jit_abi::{
    KernelSymbolSignature, SuccessorKernelFn, SuccessorKernelOut, SuccessorKernelStatus,
    SUCCESSOR_KERNEL_ARTIFACT_KIND,
};
use trust_ir::inst::{BinOp, CastOp, ICmpOp};
use trust_ir::{
    Block, BlockId, Constant, FuncId, FuncTy, Function, Inst, InstrNode, NativeAdapterInput,
    NativeBundleProducer, NativeBundleProvenance, NativeCompilerFactRef, NativeDiagnosticLevel,
    NativeDiagnosticsPolicy, NativeEvidenceArtifact, NativeEvidenceArtifactKind,
    NativeObligationCause, NativeObligationSource, NativeRequestId, NativeRequestProvenance,
    NativeSourceLanguage, NativeToolIdentity, NativeVerificationBundle, NativeVerificationRequest,
    NativeVerifierSuite, ObligationKind, ProofDigest, ProofFormula, ProofId, ProofLineageId,
    ProofLineageManifest, ProofLineageNode, ProofObligation, ProofObligationSourceIdentity,
    ProofObligationSourceRange, ProofReplayIdentity, ProofStatus, ProofTransform,
    ProofTransformStage, PublicObligationIdentity, SourceSpan, TargetInfo, TrustMcArithmeticModel,
    TrustMcChcEngine, TrustMcChcOptions, TrustMcInvariantSource, TrustMcMemoryModel,
    TrustMcNativeRequest, TrustMcPdrGeneralization, TrustMcPdrOptions, TrustMcRequestOptions,
    TrustMcSlicingMode, TrustMcVerificationMode, Ty, ValueId,
};

use super::{
    checked_all_transition_successors_cached_into, marking_to_flat_i64,
    FlatAllTransitionCandidates, PetriKernelError, PetriKernelPlanCache, PetriKernelScratch,
    PetriNativeAllSuccessorsStatus, PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
};

const PETRI_NATIVE_BUNDLE_PRODUCTION_PATH: &str =
    "trust_cg_petri_native::petri_native_successor_verification_bundle";
const PETRI_NATIVE_BUNDLE_MISSING_API: &str =
    "validated PetriKernelPlanCache -> trust_ir::NativeVerificationBundle";
const PETRI_NATIVE_BUNDLE_UPSTREAM_ASK: &str = "Wire trust-codegen to consume the Petri-produced trust_ir::NativeVerificationBundle for native successor codegen once parity promotion gates are satisfied.";
const PETRI_NATIVE_BUNDLE_VALIDATION_BLOCKER_DETAIL: &str = "Petri native successor produced a trust-ir NativeVerificationBundle that failed trust-ir validation; native promotion must remain fail-closed.";
const PETRI_NATIVE_BUNDLE_SCHEMA_VERSION: &str = "ty.petri.native_bundle.v1";
const PETRI_NATIVE_BUNDLE_FUNCTION: &str = PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL;
const PETRI_NATIVE_INSTALLED_ARTIFACT_PRODUCTION_PATH: &str =
    "trust_cg_petri_native::petri_native_successor_installed_artifact";
const PETRI_NATIVE_INSTALLED_ARTIFACT_API: &str =
    "tla_trust_cg::compile_module_native + NativeLibrary::petri_native_successor_installed_artifact";
const PETRI_NATIVE_INSTALLED_ARTIFACT_COMPILE_BLOCKER: &str =
    "petri_trust_ir_successor_native_compile_failed";
const PETRI_NATIVE_INSTALLED_ARTIFACT_UPSTREAM_ASK: &str = "Complete the Petri successor trust-ir lowering and trust-codegen native compile contract so the generated NativeLibrary can be promoted beyond validation-only evidence.";
const PETRI_NATIVE_CANDIDATE_BATCH_SCHEMA: &str = "ty.petri.native_successor.candidate_batch.v1";
const PETRI_NATIVE_CANDIDATE_BATCH_SCHEMA_VERSION: u32 = 1;
const PETRI_NATIVE_CANDIDATE_BATCH_API: &str =
    "trust_cg_petri_native::petri_native_successor_batch_candidate";
const PETRI_NATIVE_CANDIDATE_STATUS_CALLABLE_ARTIFACT: &str = "callable_artifact";
const PETRI_NATIVE_CANDIDATE_STATUS_BLOCKED: &str = "blocked";
const PETRI_NATIVE_CANDIDATE_REASON_AVAILABLE: &str = "available";
const PETRI_NATIVE_CANDIDATE_REASON_PLAN_CACHE_INVALID: &str = "plan_cache_invalid";
const PETRI_NATIVE_CANDIDATE_REASON_BUNDLE_BLOCKED: &str = "native_verification_bundle_blocked";
const PETRI_NATIVE_CANDIDATE_REASON_INSTALLED_ARTIFACT_BLOCKED: &str = "installed_artifact_blocked";
const PETRI_NATIVE_CANDIDATE_REASON_COMPILE_HANDOFF_BLOCKED: &str =
    "compile_artifact_handoff_blocked";
const PETRI_NATIVE_CANDIDATE_REASON_ABI_MISMATCH: &str = "successor_kernel_abi_shape_mismatch";
const PETRI_NATIVE_CANDIDATE_REASON_RUNTIME_READINESS: &str = "runtime_readiness_blocked";
const PETRI_NATIVE_CANDIDATE_REASON_PARITY_REPLAY_GATE: &str = "parity_replay_gate_not_promoted";
const PETRI_NATIVE_CANDIDATE_REASON_VALIDATION_RECEIPT: &str = "missing_validation_receipt";
const PETRI_NATIVE_CANDIDATE_REASON_DUPLICATE_ARC_PLACE: &str = "duplicate_arc_place_wrap_guard";
const PETRI_NATIVE_CANDIDATE_REASON_CALLABLE_POINTER_MISMATCH: &str = "callable_pointer_mismatch";
const PETRI_NATIVE_CANDIDATE_BLOCKER_NONE: &str = "none";
const PETRI_NATIVE_CANDIDATE_GATE_DETAIL: &str =
    "Petri native candidate is executable evidence only; parity/replay gates are not promoted";
const PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_SCHEMA: &str =
    "ty.petri.native_successor.callable_receipt.v1";
const PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED: &str = "accepted";
const PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_MISSING: &str = "missing";
const PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_REASON_MISSING: &str = "missing_callable_receipt";
const PETRI_NATIVE_CANDIDATE_MISSING_RECEIPTS: &str =
    "native_install_receipt,validation_receipt,parity_receipt,callable_receipt";
const PETRI_NATIVE_CANDIDATE_MISSING_RECEIPTS_NONE: &str = "none";
const PETRI_NATIVE_CANDIDATE_PRODUCTION_GATE_STATUS_SELECTED: &str = "selected";
const PETRI_NATIVE_SUCCESSOR_TRUST_IR_PARAM_COUNT: usize = 9;
pub(crate) const PETRI_NATIVE_TRANSLATION_OBLIGATION: ProofId = ProofId::new(0);
const PETRI_NATIVE_TRANSLATION_PUBLIC_OBLIGATION_ID: &str = "vc:tla-petri:translation:0";
const PETRI_NATIVE_TRANSLATION_SOURCE_ID: &str = "ty:petri-native-successor";
const PETRI_NATIVE_TRANSLATION_ASSERTION_ID: &str = "assertion:translation:0";
const PETRI_NATIVE_TRANSLATION_SOURCE_FILE: &str = "ty_petri_native_successor.tir";
const PETRI_NATIVE_TRANSLATION_SEMANTIC_DIGEST_DOMAIN: &str =
    "ty.petri.native.successor.translation.public_obligation.v1";
const PETRI_NATIVE_LINEAGE_ROOT: ProofLineageId = ProofLineageId::new(0);
const SUCCESSOR_OUT_STATUS_OFFSET: u64 = 0;
const SUCCESSOR_OUT_SUCCESSOR_COUNT_OFFSET: u64 = 4;
const SUCCESSOR_OUT_GENERATED_COUNT_OFFSET: u64 = 8;
const SUCCESSOR_OUT_STATE_LEN_OFFSET: u64 = 12;
const SUCCESSOR_OUT_OVERFLOW_COUNT_OFFSET: u64 = 16;
const SUCCESSOR_OUT_RUNTIME_ERROR_OFFSET: u64 = 20;
const SUCCESSOR_OUT_UNSUPPORTED_REASON_OFFSET: u64 = 21;
const SUCCESSOR_OUT_METADATA_BITS_OFFSET: u64 = 24;
const SUCCESSOR_STATUS_OK: u8 = 0;
const SUCCESSOR_STATUS_DISABLED: u8 = 1;
const SUCCESSOR_STATUS_BUFFER_OVERFLOW: u8 = 2;
const SUCCESSOR_STATUS_UNSUPPORTED: u8 = 5;
const SUCCESSOR_RUNTIME_ERROR_DIVISION_BY_ZERO: u8 = 1;
const SUCCESSOR_UNSUPPORTED_REASON_NONE: u8 = 0;
const SUCCESSOR_UNSUPPORTED_REASON_UNSUPPORTED_STATE_LAYOUT: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PetriNativeAllTransitionConfig {
    pub(crate) strict: bool,
}

#[derive(Debug)]
pub(crate) struct PetriNativeAllTransitionKernel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PetriNativeVerificationBundleProductionBlocker {
    pub(crate) reason_code: &'static str,
    pub(crate) production_path: &'static str,
    pub(crate) missing_api: &'static str,
    pub(crate) detail: &'static str,
    pub(crate) upstream_ask: &'static str,
}

// Boxing the large `Available` payload would alter construction/match ergonomics
// at call sites; kept inline to preserve the existing value-passing API.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PetriNativeVerificationBundleProduction {
    Available(trust_ir::NativeVerificationBundle),
    Blocked(PetriNativeVerificationBundleProductionBlocker),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PetriNativeInstalledArtifactProductionBlocker {
    pub(crate) reason_code: &'static str,
    pub(crate) production_path: &'static str,
    pub(crate) missing_api: &'static str,
    pub(crate) detail: String,
    pub(crate) upstream_ask: &'static str,
}

// Boxing the large `Available` payload would alter construction/match ergonomics
// at call sites; kept inline to preserve the existing value-passing API.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(crate) enum PetriNativeInstalledArtifactProduction {
    Available(PetriNativeInstalledArtifact),
    Blocked(PetriNativeInstalledArtifactProductionBlocker),
}

#[derive(Debug, Clone)]
pub(crate) struct PetriNativeInstalledArtifact {
    pub(crate) artifact: tla_trust_cg::InstalledArtifact,
    lookup_entry_symbol: String,
}

impl PetriNativeInstalledArtifact {
    pub(crate) fn lookup_entry_symbol(&self) -> &str {
        self.lookup_entry_symbol.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PetriNativeSharedPlanningFingerprintIdentity {
    schema: String,
    schema_version: String,
    planning_identity_status: String,
    planning_identity_digest: String,
    planning_identity_required_fields: String,
    prepared_program_identity: String,
    candidate_identity: String,
    lane_identity: String,
    layout_checksum: String,
    semantic_checksum: String,
    source_checksum: String,
    payload_checksum: String,
    manifest_checksum: String,
    fingerprint_domain_identity: String,
    fingerprint_policy_identity: String,
    cache_namespace_identity: String,
    cache_reuse_policy: String,
    cache_digest: String,
    prepared_trust_ir_reuse_identity: String,
    trust_cg_batch_cache_reuse_status: String,
    trust_cg_batch_cache_reuse_blocker_code: String,
    validation_receipt_status: String,
    parity_receipt_status: String,
    callable_receipt_status: String,
    production_gate_status: String,
}

impl PetriNativeSharedPlanningFingerprintIdentity {
    fn for_net(net: &PetriNet) -> Self {
        let setup = crate::explorer::ExplorationSetup::analyze(net);
        let fields = setup.shared_native_planning_fingerprint_identity_fields_for_net(net);
        Self {
            schema: planning_identity_field(&fields, "schema"),
            schema_version: planning_identity_field(&fields, "schema_version"),
            planning_identity_status: planning_identity_field(&fields, "planning_identity_status"),
            planning_identity_digest: planning_identity_field(&fields, "planning_identity_digest"),
            planning_identity_required_fields: planning_identity_field(
                &fields,
                "planning_identity_required_fields",
            ),
            prepared_program_identity: planning_identity_field(
                &fields,
                "prepared_program_identity",
            ),
            candidate_identity: planning_identity_field(&fields, "candidate_identity"),
            lane_identity: planning_identity_field(&fields, "lane_identity"),
            layout_checksum: planning_identity_field(&fields, "layout_checksum"),
            semantic_checksum: planning_identity_field(&fields, "semantic_checksum"),
            source_checksum: planning_identity_field(&fields, "source_checksum"),
            payload_checksum: planning_identity_field(&fields, "payload_checksum"),
            manifest_checksum: planning_identity_field(&fields, "manifest_checksum"),
            fingerprint_domain_identity: planning_identity_field(
                &fields,
                "fingerprint_domain_identity",
            ),
            fingerprint_policy_identity: planning_identity_field(
                &fields,
                "fingerprint_domain_acceptance_identity",
            ),
            cache_namespace_identity: planning_identity_field(&fields, "cache_namespace_identity"),
            cache_reuse_policy: planning_identity_field(&fields, "cache_reuse_policy"),
            cache_digest: planning_identity_field(&fields, "cache_digest"),
            prepared_trust_ir_reuse_identity: planning_identity_field(
                &fields,
                "prepared_trust_ir_reuse_identity",
            ),
            trust_cg_batch_cache_reuse_status: planning_identity_field(
                &fields,
                "trust_cg_batch_cache_reuse_status",
            ),
            trust_cg_batch_cache_reuse_blocker_code: planning_identity_field(
                &fields,
                "trust_cg_batch_cache_reuse_blocker_code",
            ),
            validation_receipt_status: planning_identity_field(
                &fields,
                "validation_receipt_status",
            ),
            parity_receipt_status: planning_identity_field(&fields, "parity_receipt_status"),
            callable_receipt_status: planning_identity_field(&fields, "callable_receipt_status"),
            production_gate_status: planning_identity_field(&fields, "production_gate_status"),
        }
    }
}

fn planning_identity_field(fields: &[(&'static str, String)], key: &str) -> String {
    fields
        .iter()
        .find_map(|(field_key, value)| (*field_key == key).then(|| value.clone()))
        .unwrap_or_else(|| PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PetriNativeCandidatePromotionBlocker {
    reason_code: &'static str,
    detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PetriNativeSuccessorBatchReadinessPacket {
    pub(crate) schema: &'static str,
    pub(crate) schema_version: u32,
    pub(crate) api: &'static str,
    pub(crate) status_code: &'static str,
    pub(crate) reason_code: &'static str,
    pub(crate) blocker: String,
    pub(crate) entry_symbol: &'static str,
    pub(crate) artifact_kind: &'static str,
    pub(crate) shared_signature_abi: String,
    pub(crate) shared_signature_params: usize,
    pub(crate) shared_signature_returns: usize,
    pub(crate) trust_ir_entry_abi_matches_shared_successor_kernel: bool,
    pub(crate) state_len: u32,
    pub(crate) max_successors: u32,
    pub(crate) state_bytes: u64,
    pub(crate) runtime_readiness_status_code: &'static str,
    pub(crate) runtime_readiness_reason_code: &'static str,
    pub(crate) runtime_readiness_required_evidence: &'static str,
    pub(crate) runtime_readiness_packet_sha256: String,
    pub(crate) runtime_ready_for_call: bool,
    pub(crate) compile_artifact_handoff_ready: bool,
    pub(crate) callable_pointer_available: bool,
    pub(crate) native_payload_sha256: String,
    pub(crate) executable_region_sha256: String,
    pub(crate) lifetime_owner: String,
    pub(crate) current_generation: u64,
    pub(crate) transport_digest: String,
    pub(crate) bundle_digest: String,
    pub(crate) target_abi_digest: String,
    pub(crate) shared_planning_identity_schema: String,
    pub(crate) shared_planning_identity_schema_version: String,
    pub(crate) shared_planning_identity_status: String,
    pub(crate) shared_planning_identity_digest: String,
    pub(crate) shared_planning_identity_required_fields: String,
    pub(crate) shared_prepared_program_identity: String,
    pub(crate) shared_candidate_identity: String,
    pub(crate) shared_lane_identity: String,
    pub(crate) shared_layout_checksum: String,
    pub(crate) shared_semantic_checksum: String,
    pub(crate) shared_source_checksum: String,
    pub(crate) shared_payload_checksum: String,
    pub(crate) shared_manifest_checksum: String,
    pub(crate) shared_fingerprint_domain_identity: String,
    pub(crate) shared_fingerprint_policy_identity: String,
    pub(crate) shared_cache_namespace_identity: String,
    pub(crate) shared_cache_reuse_policy: String,
    pub(crate) shared_cache_digest: String,
    pub(crate) prepared_trust_ir_reuse_identity: String,
    pub(crate) trust_cg_batch_cache_reuse_status: String,
    pub(crate) trust_cg_batch_cache_reuse_blocker_code: String,
    pub(crate) validation_receipt_status: String,
    pub(crate) parity_receipt_status: String,
    pub(crate) callable_receipt_schema: &'static str,
    pub(crate) callable_receipt_status: String,
    pub(crate) callable_receipt_reason_code: String,
    pub(crate) native_missing_receipts: &'static str,
    pub(crate) production_gate_status: String,
    pub(crate) production_selected: bool,
    pub(crate) fail_closed: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct PetriNativeCallableSuccessorBatch {
    pub(crate) readiness: PetriNativeSuccessorBatchReadinessPacket,
    pub(crate) compile_artifact_handoff:
        tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
    pub(crate) runtime_readiness: tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
    pub(crate) installed_artifact: PetriNativeInstalledArtifact,
}

// Boxing the large `CallableArtifact` payload would alter construction/match
// ergonomics at call sites; kept inline to preserve the existing value-passing API.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(crate) enum PetriNativeSuccessorBatchCandidate {
    CallableArtifact(PetriNativeCallableSuccessorBatch),
    Blocked(PetriNativeSuccessorBatchReadinessPacket),
}

impl PetriNativeSuccessorBatchCandidate {
    fn readiness(&self) -> &PetriNativeSuccessorBatchReadinessPacket {
        match self {
            Self::CallableArtifact(batch) => &batch.readiness,
            Self::Blocked(packet) => packet,
        }
    }

    fn compile_artifact_handoff_ready(&self) -> bool {
        match self {
            Self::CallableArtifact(batch) => batch.compile_artifact_handoff.is_ready(),
            Self::Blocked(packet) => packet.compile_artifact_handoff_ready,
        }
    }

    fn runtime_ready_for_call(&self) -> bool {
        match self {
            Self::CallableArtifact(batch) => batch.runtime_readiness.is_ready_for_runtime_call(),
            Self::Blocked(packet) => packet.runtime_ready_for_call,
        }
    }

    fn into_fail_closed_error(self, config: PetriNativeAllTransitionConfig) -> PetriKernelError {
        let compile_artifact_handoff_ready = self.compile_artifact_handoff_ready();
        let runtime_ready_for_call = self.runtime_ready_for_call();
        let readiness = self.readiness();
        PetriKernelError::NativeStatus {
            status: PetriNativeAllSuccessorsStatus::Unsupported,
            detail: format!(
                "trust-cg Petri native successor candidate is fail-closed; \
                 candidate_schema={} candidate_schema_version={} candidate_api={} \
                 candidate_status_code={} reason_code={} blocker=\"{}\" \
                 strict={} artifact_kind={} entry_symbol={} shared_signature_abi={} \
                 shared_signature_params={} shared_signature_returns={} \
                 trust_ir_entry_abi_matches_shared_successor_kernel={} state_len={} \
                 max_successors={} state_bytes={} compile_artifact_handoff_ready={} \
                 callable_pointer_available={} native_payload_sha256={} \
                 executable_region_sha256={} lifetime_owner={} current_generation={} \
                 runtime_readiness_status_code={} runtime_readiness_reason_code={} \
                 runtime_readiness_required_evidence={} runtime_readiness_packet_sha256={} \
                 runtime_ready_for_call={} transport_digest={} bundle_digest={} \
                 target_abi_digest={} shared_planning_identity_schema={} \
                 shared_planning_identity_schema_version={} shared_planning_identity_status={} \
                 shared_planning_identity_digest={} shared_planning_identity_required_fields={} \
                 shared_prepared_program_identity={} shared_candidate_identity={} \
                 shared_lane_identity={} shared_layout_checksum={} shared_semantic_checksum={} \
                 shared_source_checksum={} shared_payload_checksum={} shared_manifest_checksum={} \
                 shared_fingerprint_domain_identity={} shared_fingerprint_policy_identity={} \
                 shared_cache_namespace_identity={} shared_cache_reuse_policy={} shared_cache_digest={} \
                 prepared_trust_ir_reuse_identity={} trust_cg_batch_cache_reuse_status={} \
                 trust_cg_batch_cache_reuse_blocker_code={} validation_receipt_status={} \
                 parity_receipt_status={} callable_receipt_schema={} callable_receipt_status={} \
                 callable_receipt_reason_code={} native_missing_receipts={} production_gate_status={} \
                 production_selected={} fail_closed={} \
                 parity_replay_gate_reason_code={}",
                readiness.schema,
                readiness.schema_version,
                readiness.api,
                readiness.status_code,
                readiness.reason_code,
                readiness.blocker,
                config.strict,
                readiness.artifact_kind,
                readiness.entry_symbol,
                readiness.shared_signature_abi,
                readiness.shared_signature_params,
                readiness.shared_signature_returns,
                readiness.trust_ir_entry_abi_matches_shared_successor_kernel,
                readiness.state_len,
                readiness.max_successors,
                readiness.state_bytes,
                compile_artifact_handoff_ready,
                readiness.callable_pointer_available,
                readiness.native_payload_sha256,
                readiness.executable_region_sha256,
                readiness.lifetime_owner,
                readiness.current_generation,
                readiness.runtime_readiness_status_code,
                readiness.runtime_readiness_reason_code,
                readiness.runtime_readiness_required_evidence,
                readiness.runtime_readiness_packet_sha256,
                runtime_ready_for_call,
                readiness.transport_digest,
                readiness.bundle_digest,
                readiness.target_abi_digest,
                readiness.shared_planning_identity_schema,
                readiness.shared_planning_identity_schema_version,
                readiness.shared_planning_identity_status,
                readiness.shared_planning_identity_digest,
                readiness.shared_planning_identity_required_fields,
                readiness.shared_prepared_program_identity,
                readiness.shared_candidate_identity,
                readiness.shared_lane_identity,
                readiness.shared_layout_checksum,
                readiness.shared_semantic_checksum,
                readiness.shared_source_checksum,
                readiness.shared_payload_checksum,
                readiness.shared_manifest_checksum,
                readiness.shared_fingerprint_domain_identity,
                readiness.shared_fingerprint_policy_identity,
                readiness.shared_cache_namespace_identity,
                readiness.shared_cache_reuse_policy,
                readiness.shared_cache_digest,
                readiness.prepared_trust_ir_reuse_identity,
                readiness.trust_cg_batch_cache_reuse_status,
                readiness.trust_cg_batch_cache_reuse_blocker_code,
                readiness.validation_receipt_status,
                readiness.parity_receipt_status,
                readiness.callable_receipt_schema,
                readiness.callable_receipt_status,
                readiness.callable_receipt_reason_code,
                readiness.native_missing_receipts,
                readiness.production_gate_status,
                readiness.production_selected,
                readiness.fail_closed,
                PETRI_NATIVE_CANDIDATE_REASON_PARITY_REPLAY_GATE,
            ),
        }
    }
}

pub(crate) fn petri_native_successor_verification_bundle(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
) -> PetriNativeVerificationBundleProduction {
    let bundle = build_petri_native_successor_verification_bundle(net, cache);
    match bundle.validate() {
        Ok(()) => PetriNativeVerificationBundleProduction::Available(bundle),
        Err(_) => PetriNativeVerificationBundleProduction::Blocked(
            PetriNativeVerificationBundleProductionBlocker {
                reason_code: "petri_trust_ir_bundle_validation_failed",
                production_path: PETRI_NATIVE_BUNDLE_PRODUCTION_PATH,
                missing_api: PETRI_NATIVE_BUNDLE_MISSING_API,
                detail: PETRI_NATIVE_BUNDLE_VALIDATION_BLOCKER_DETAIL,
                upstream_ask: PETRI_NATIVE_BUNDLE_UPSTREAM_ASK,
            },
        ),
    }
}

pub(crate) fn petri_native_successor_installed_artifact(
    bundle: &trust_ir::NativeVerificationBundle,
) -> PetriNativeInstalledArtifactProduction {
    // The Petri successor kernel is compiled ONCE per net but executed across the entire
    // state-space exploration, so optimization cost amortizes (same profile as the
    // explicit-state BFS, which pays O3). O0 ran zero passes. O2 (not O3) keeps JIT
    // compile latency bounded under the MCC wall-clock deadline while enabling DCE/CSE/
    // GVN/scheduling + (at O2) the trust-cg proof_opts guard-elimination passes.
    match tla_trust_cg::compile_module_native(&bundle.module, tla_trust_cg::OptLevel::O2) {
        Ok(library) => {
            let readiness =
                library.petri_native_successor_runtime_readiness(Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL));
            let lookup_entry_symbol = readiness
                .compile_artifact_handoff
                .entry_symbol
                .clone()
                .unwrap_or_else(|| PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL.to_string());
            PetriNativeInstalledArtifactProduction::Available(PetriNativeInstalledArtifact {
                artifact: library.petri_native_successor_installed_artifact(),
                lookup_entry_symbol,
            })
        }
        Err(error) => {
            PetriNativeInstalledArtifactProduction::Blocked(
                PetriNativeInstalledArtifactProductionBlocker {
                    reason_code: PETRI_NATIVE_INSTALLED_ARTIFACT_COMPILE_BLOCKER,
                    production_path: PETRI_NATIVE_INSTALLED_ARTIFACT_PRODUCTION_PATH,
                    missing_api: PETRI_NATIVE_INSTALLED_ARTIFACT_API,
                    detail: format!(
                        "Petri native successor trust-ir bundle did not compile into an trust-codegen NativeLibrary: {error}"
                    ),
                    upstream_ask: PETRI_NATIVE_INSTALLED_ARTIFACT_UPSTREAM_ASK,
                },
            )
        }
    }
}

/// Kernel arithmetic-exactness gate (trap-vs-wrap invariant, part iii).
///
/// The interpreter fires transitions in u64 with `overflow-checks = true`
/// (release profile), while the emitted kernel uses unchecked two's-complement
/// i64 add/sub. The two are bit-identical on the guarded domain — input tokens
/// in `[0, i64::MAX]` (per-state `marking_to_flat_i64` guard in
/// `PetriNetSystem::successors`) and arc weights in `[0, i64::MAX]`
/// (`checked_arcs_to_i64` at plan build) — PROVIDED each transition touches
/// each place through at most one input arc and at most one output arc. Then
/// the kernel's per-place sub-then-add sequence `m[p] - w_in + w_out`:
///   - never goes negative mid-computation (per-arc enabledness `m[p] >= w_in`
///     is exact when there is a single input arc on the place), and
///   - can wrap i64 at most once on the single add, landing strictly in the
///     negative range (max true value `(2^63-1) + (2^63-1) = 2^64-2 < 2^64`),
///     which the per-row negative-token guard in `PetriNetSystem::successors`
///     rejects with a sound scalar fallback.
///
/// With duplicate per-(transition, place) arcs neither bound holds: per-arc
/// enabledness no longer implies the SUM of input weights is covered, so the
/// kernel can compute a wrapped value at a marking where the interpreter
/// PANICS on u64 underflow — engine-dependent outcomes on the same input.
/// PNML P/T duplicates survive the parser unmerged, so refuse to promote such
/// plans (interpreter fallback; verdict-preserving) instead of trusting the
/// invariant.
fn petri_native_duplicate_arc_place_blocker(cache: &PetriKernelPlanCache) -> Option<String> {
    for plan in cache.plans() {
        for (kind, arcs) in [("input", &plan.inputs), ("output", &plan.outputs)] {
            for (index, &(place, _)) in arcs.iter().enumerate() {
                if arcs[..index].iter().any(|&(earlier, _)| earlier == place) {
                    return Some(format!(
                        "transition {:?} has duplicate {kind} arcs on place {:?}; per-arc \
                         enabledness with duplicate arcs breaks the i64-wrap-unreachable \
                         invariant of the native kernel (fail-closed to the interpreter)",
                        plan.transition, place,
                    ));
                }
            }
        }
    }
    None
}

pub(crate) fn petri_native_successor_batch_candidate(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
) -> PetriNativeSuccessorBatchCandidate {
    let shared_planning_identity = PetriNativeSharedPlanningFingerprintIdentity::for_net(net);
    let layout = match cache.validate_for_net(net) {
        Ok(layout) => layout,
        Err(error) => {
            return PetriNativeSuccessorBatchCandidate::Blocked(
                PetriNativeSuccessorBatchReadinessPacket::blocked_without_artifact(
                    net,
                    &shared_planning_identity,
                    PETRI_NATIVE_CANDIDATE_REASON_PLAN_CACHE_INVALID,
                    format!("{error:?}"),
                ),
            );
        }
    };

    if let Some(detail) = petri_native_duplicate_arc_place_blocker(cache) {
        return PetriNativeSuccessorBatchCandidate::Blocked(
            PetriNativeSuccessorBatchReadinessPacket::blocked_without_artifact(
                net,
                &shared_planning_identity,
                PETRI_NATIVE_CANDIDATE_REASON_DUPLICATE_ARC_PLACE,
                detail,
            ),
        );
    }

    let bundle = match petri_native_successor_verification_bundle(net, cache) {
        PetriNativeVerificationBundleProduction::Available(bundle) => bundle,
        PetriNativeVerificationBundleProduction::Blocked(blocker) => {
            return PetriNativeSuccessorBatchCandidate::Blocked(
                PetriNativeSuccessorBatchReadinessPacket::blocked_without_artifact(
                    net,
                    &shared_planning_identity,
                    PETRI_NATIVE_CANDIDATE_REASON_BUNDLE_BLOCKED,
                    format!("{}: {}", blocker.reason_code, blocker.detail),
                ),
            );
        }
    };

    let state_len = usize_to_u32_saturating(layout.state_len());
    let max_successors = usize_to_u32_saturating(cache.transition_count);
    let trust_ir_entry_abi_matches_shared_successor_kernel =
        trust_ir_entry_uses_shared_successor_kernel_abi(&bundle);
    if !trust_ir_entry_abi_matches_shared_successor_kernel {
        return PetriNativeSuccessorBatchCandidate::Blocked(
            PetriNativeSuccessorBatchReadinessPacket::blocked_for_bundle(
                &bundle,
                &shared_planning_identity,
                state_len,
                max_successors,
                PETRI_NATIVE_CANDIDATE_REASON_ABI_MISMATCH,
                "Petri native successor trust-ir entrypoint does not match SuccessorKernelFn ABI"
                    .to_string(),
            ),
        );
    }

    let installed_artifact = match petri_native_successor_installed_artifact(&bundle) {
        PetriNativeInstalledArtifactProduction::Available(artifact) => artifact,
        PetriNativeInstalledArtifactProduction::Blocked(blocker) => {
            return PetriNativeSuccessorBatchCandidate::Blocked(
                PetriNativeSuccessorBatchReadinessPacket::blocked_for_bundle(
                    &bundle,
                    &shared_planning_identity,
                    state_len,
                    max_successors,
                    PETRI_NATIVE_CANDIDATE_REASON_INSTALLED_ARTIFACT_BLOCKED,
                    format!("{}: {}", blocker.reason_code, blocker.detail),
                ),
            );
        }
    };

    let mut compile_artifact_handoff = installed_artifact
        .artifact
        .petri_native_successor_compile_artifact_handoff_evidence(Some(
            installed_artifact.lookup_entry_symbol(),
        ));
    normalize_frontend_entry_symbol(&mut compile_artifact_handoff);
    let target_abi_digest = bundle
        .transport_identity()
        .target_abi
        .as_ref()
        .map(|target_abi| target_abi.digest);
    let runtime_inputs = super::trust_cg_petri_runtime_readiness_inputs(
        &bundle,
        u64::from(state_len) * std::mem::size_of::<i64>() as u64,
        target_abi_digest,
        &compile_artifact_handoff,
    );
    let runtime_readiness = installed_artifact
        .artifact
        .petri_native_successor_runtime_readiness_packet(
            Some(installed_artifact.lookup_entry_symbol()),
            runtime_inputs.install_packet.as_ref(),
            runtime_inputs.trampoline_contract.as_ref(),
            runtime_inputs.call_packet.as_ref(),
            None,
        );

    if !compile_artifact_handoff.is_ready() {
        return PetriNativeSuccessorBatchCandidate::Blocked(
            PetriNativeSuccessorBatchReadinessPacket::blocked_for_handoff(
                net,
                cache,
                &bundle,
                &shared_planning_identity,
                state_len,
                max_successors,
                PETRI_NATIVE_CANDIDATE_REASON_COMPILE_HANDOFF_BLOCKED,
                &compile_artifact_handoff,
                &runtime_readiness,
            ),
        );
    }

    let readiness_packet = PetriNativeSuccessorBatchReadinessPacket::callable_artifact(
        net,
        cache,
        &bundle,
        &installed_artifact,
        &shared_planning_identity,
        state_len,
        max_successors,
        &compile_artifact_handoff,
        &runtime_readiness,
    );

    PetriNativeSuccessorBatchCandidate::CallableArtifact(PetriNativeCallableSuccessorBatch {
        readiness: readiness_packet,
        compile_artifact_handoff,
        runtime_readiness,
        installed_artifact,
    })
}

fn normalize_frontend_entry_symbol(
    handoff: &mut tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
) {
    if handoff.entry_symbol.as_deref() == Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL) {
        return;
    }

    handoff.entry_symbol = Some(PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL.to_string());
    handoff.compile_artifact_handoff_sha256 = handoff.canonical_compile_artifact_handoff_sha256();
}

pub(crate) fn checked_native_all_transition_successors_cached_into(
    _kernel: &PetriNativeAllTransitionKernel,
    cache: &PetriKernelPlanCache,
    net: &PetriNet,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
    config: PetriNativeAllTransitionConfig,
) -> Result<usize, PetriKernelError> {
    let layout = cache.validate_for_net(net)?;
    layout.check_state_len(marking.len())?;

    let batch = match petri_native_successor_batch_candidate(net, cache) {
        PetriNativeSuccessorBatchCandidate::CallableArtifact(batch)
            if batch.readiness.production_selected
                && batch.runtime_readiness.is_ready_for_runtime_call() =>
        {
            batch
        }
        candidate => return Err(candidate.into_fail_closed_error(config)),
    };

    let successor_capacity = cache.transition_count;
    let (out, native_successors) = invoke_native_all_transition_successor_artifact(
        &batch.installed_artifact,
        &batch.runtime_readiness,
        cache,
        net,
        marking,
        scratch,
        successor_capacity,
    )?;

    match out.status {
        SuccessorKernelStatus::Ok | SuccessorKernelStatus::Disabled => {}
        SuccessorKernelStatus::BufferOverflow => {
            return Err(PetriKernelError::NativeStatus {
                status: PetriNativeAllSuccessorsStatus::BufferOverflow,
                detail: format!(
                    "native successor buffer overflow: written={} generated={} state_len={} overflow_count={}",
                    out.successor_count, out.generated_count, out.state_len, out.overflow_count
                ),
            });
        }
        SuccessorKernelStatus::RuntimeError => {
            return Err(PetriKernelError::NativeStatus {
                status: PetriNativeAllSuccessorsStatus::TokenOverflow,
                detail: format!("native successor runtime error: {:?}", out.runtime_error),
            });
        }
        status => {
            return Err(PetriKernelError::NativeStatus {
                status: PetriNativeAllSuccessorsStatus::Unsupported,
                detail: format!(
                    "native successor returned status={status:?} unsupported_reason={:?}",
                    out.unsupported_reason
                ),
            });
        }
    }

    let native_count =
        usize::try_from(out.successor_count).map_err(|_| PetriKernelError::CountExceedsU32 {
            what: "native successor_count",
            count: out.successor_count as usize,
        })?;
    if native_count > successor_capacity {
        return Err(PetriKernelError::NativeOutputCountExceedsCapacity {
            count: native_count,
            capacity: successor_capacity,
        });
    }

    let mut checked = FlatAllTransitionCandidates::new();
    checked_all_transition_successors_cached_into(net, cache, marking, scratch, &mut checked)?;
    let native_len = native_count.checked_mul(layout.state_len()).ok_or(
        PetriKernelError::NativeOutputCountExceedsCapacity {
            count: native_count,
            capacity: successor_capacity,
        },
    )?;
    if native_count != checked.len()
        || native_successors[..native_len] != checked.flat_successors()[..native_len]
    {
        return Err(PetriKernelError::NativeCandidateMismatch {
            detail: format!(
                "native successor parity mismatch: native_count={} generated_count={} state_len={} overflow_count={} checked_count={} native={:?} checked={:?}",
                native_count,
                out.generated_count,
                out.state_len,
                out.overflow_count,
                checked.len(),
                &native_successors[..native_len],
                checked.flat_successors()
            ),
        });
    }

    Ok(native_count)
}

#[cfg(test)]
pub(crate) fn unchecked_native_all_transition_successors_for_tests(
    cache: &PetriKernelPlanCache,
    net: &PetriNet,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
    successor_capacity: usize,
) -> Result<(SuccessorKernelOut, Vec<i64>), PetriKernelError> {
    let candidate = petri_native_successor_batch_candidate(net, cache);
    let PetriNativeSuccessorBatchCandidate::CallableArtifact(batch) = candidate else {
        return Err(
            candidate.into_fail_closed_error(PetriNativeAllTransitionConfig { strict: true })
        );
    };
    invoke_native_all_transition_successor_artifact(
        &batch.installed_artifact,
        &batch.runtime_readiness,
        cache,
        net,
        marking,
        scratch,
        successor_capacity,
    )
}

fn invoke_native_all_transition_successor_artifact(
    installed_artifact: &PetriNativeInstalledArtifact,
    runtime_readiness: &tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
    cache: &PetriKernelPlanCache,
    net: &PetriNet,
    marking: &[u64],
    scratch: &mut PetriKernelScratch,
    successor_capacity: usize,
) -> Result<(SuccessorKernelOut, Vec<i64>), PetriKernelError> {
    let layout = cache.validate_for_net(net)?;
    layout.check_state_len(marking.len())?;

    marking_to_flat_i64(marking, &mut scratch.flat_in)?;
    let successor_slots = layout.state_len().checked_mul(successor_capacity).ok_or(
        PetriKernelError::NativeOutputCountExceedsCapacity {
            count: usize::MAX,
            capacity: successor_capacity,
        },
    )?;
    let mut native_successors = vec![0_i64; successor_slots];
    let mut out = SuccessorKernelOut::default();
    let entry_symbol = installed_artifact.lookup_entry_symbol();
    let entrypoint = installed_artifact
        .artifact
        .entrypoint_ptr(entry_symbol)
        .ok_or_else(|| PetriKernelError::NativeSymbol {
            detail: format!("native successor entrypoint {entry_symbol} was not exported"),
        })?;
    let entrypoint_ptr = entrypoint.as_ptr();
    let entrypoint_callable = tla_trust_cg::PetriNativeSuccessorCallablePointer::from_ptr(
        entrypoint_ptr,
    )
    .ok_or_else(|| PetriKernelError::NativeSymbol {
        detail: format!("native successor entrypoint {entry_symbol} had a null pointer"),
    })?;
    if Some(entrypoint_callable) != runtime_readiness.callable_pointer {
        return Err(PetriKernelError::NativeSymbol {
            detail: format!(
                "native successor entrypoint {entry_symbol} pointer did not match runtime readiness packet"
            ),
        });
    }
    installed_artifact
        .artifact
        .ensure_published_entrypoint_ptr(entry_symbol, entrypoint_ptr)
        .map_err(|error| PetriKernelError::NativeCompile {
            detail: format!("native successor executable publication failed: {error}"),
        })?;
    let native_fn = unsafe {
        installed_artifact
            .artifact
            .entrypoint::<SuccessorKernelFn>(entry_symbol)
    }
    .ok_or_else(|| PetriKernelError::NativeSymbol {
        detail: format!("native successor entrypoint {entry_symbol} was not exported"),
    })?;
    let native_fn = *native_fn.as_ref();
    unsafe {
        native_fn(
            &mut out,
            scratch.flat_in.as_ptr(),
            u32::try_from(layout.state_len()).map_err(|_| PetriKernelError::CountExceedsU32 {
                what: "native successor state_len",
                count: layout.state_len(),
            })?,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            native_successors.as_mut_ptr(),
            u32::try_from(successor_capacity).map_err(|_| PetriKernelError::CountExceedsU32 {
                what: "native successor capacity",
                count: successor_capacity,
            })?,
        );
    }

    Ok((out, native_successors))
}

fn build_petri_native_successor_verification_bundle(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
) -> NativeVerificationBundle {
    let plan_digest = petri_kernel_plan_cache_digest(cache);
    let shared_planning_identity = PetriNativeSharedPlanningFingerprintIdentity::for_net(net);
    let module = petri_native_successor_complete_trust_ir_module(
        net,
        cache,
        plan_digest,
        &shared_planning_identity,
    );
    let trust_ir_module_digest = module.stable_digest();
    let lineage = petri_native_successor_lineage(
        plan_digest,
        trust_ir_module_digest,
        &shared_planning_identity,
    );

    let mut bundle = NativeVerificationBundle::new(
        NativeBundleProducer::TrustIr,
        NativeAdapterInput::TrustIrModule,
        trust_ir_module_digest,
        module,
        lineage,
    );
    bundle.provenance = NativeBundleProvenance {
        producer_version: PETRI_NATIVE_BUNDLE_SCHEMA_VERSION.to_string(),
        source_language: NativeSourceLanguage::TrustIr,
        source_artifact: Some("PetriKernelPlanCache+SharedPlanningFingerprintIdentity".to_string()),
        source_digest: Some(plan_digest),
        toolchain: vec![
            NativeToolIdentity::new("tla-petri").with_version(PETRI_NATIVE_BUNDLE_SCHEMA_VERSION),
            NativeToolIdentity::new("trust_ir")
                .with_revision(super::TRUST_IR_NATIVE_VERIFICATION_BUNDLE_CURRENT_REV),
        ],
    };
    bundle.diagnostics = NativeDiagnosticsPolicy {
        level: NativeDiagnosticLevel::Trace,
        include_source_spans: true,
        include_lineage: true,
        emit_counterexamples: true,
        emit_unsat_cores: true,
        emit_proof_traces: true,
        max_counterexamples: 1,
    };
    bundle
        .compiler_facts
        .obligation_sources
        .push(NativeObligationSource {
            obligation: PETRI_NATIVE_TRANSLATION_OBLIGATION,
            public_obligation_id: PETRI_NATIVE_TRANSLATION_PUBLIC_OBLIGATION_ID.to_string(),
            function: Some(FuncId::new(0)),
            span: Some(SourceSpan {
                file: 0,
                line: 1,
                col: 1,
            }),
            assertion_id: Some(trust_ir::NativeAssertionId::new(0)),
            cause: NativeObligationCause::Translation,
            monomorphization: None,
            facts: Vec::<NativeCompilerFactRef>::new(),
        });
    bundle
        .requests
        .push(NativeVerificationRequest::TrustMc(TrustMcNativeRequest {
            id: NativeRequestId::new(0),
            mode: TrustMcVerificationMode::Chc,
            function: FuncId::new(0),
            obligations: vec![PETRI_NATIVE_TRANSLATION_OBLIGATION],
            lineage_roots: vec![PETRI_NATIVE_LINEAGE_ROOT],
            options: TrustMcRequestOptions {
                memory_model: TrustMcMemoryModel::TrustIrPlaces,
                arithmetic_model: TrustMcArithmeticModel::MathematicalIntegers,
                chc: TrustMcChcOptions {
                    engine: TrustMcChcEngine::Spacer,
                    invariant_source: TrustMcInvariantSource::TrustIrProofObligations,
                    pdr: TrustMcPdrOptions {
                        enabled: true,
                        max_frames: Some(16),
                        generalization: TrustMcPdrGeneralization::Cubes,
                    },
                    emit_horn_clauses: true,
                },
                slicing: TrustMcSlicingMode::ObligationBackwardSlice,
                ..TrustMcRequestOptions::default()
            },
            diagnostics: bundle.diagnostics,
            provenance: NativeRequestProvenance::new(
                NativeVerifierSuite::TrustMc,
                NativeToolIdentity::new("trust_mc").with_version("petri-chc-v1"),
            )
            .with_solver(NativeToolIdentity::new("ay").with_version("shared-chc"))
            .with_replay(
                ProofReplayIdentity::new(
                    "trust_mc",
                    "trust_mc chc --native-bundle petri_successor --function ty_petri_all_transition_successors",
                )
                .with_transcript_digest(ProofDigest::sha256_domain(
                    "ty.petri.native.successor.trust_mc.replay.v1",
                    &digest_payload(plan_digest),
                )),
            ),
        }));
    attach_petri_native_successor_semantic_evidence(
        &mut bundle,
        net,
        cache,
        plan_digest,
        trust_ir_module_digest,
        &shared_planning_identity,
    );
    bundle
}

fn petri_native_successor_trust_ir_module(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    translation_obligation: ProofObligation,
) -> trust_ir::Module {
    let mut module = trust_ir::Module::new("ty_petri_native_successor");
    let source_file = module.intern_file(PETRI_NATIVE_TRANSLATION_SOURCE_FILE);
    assert_eq!(
        source_file, 0,
        "the canonical Petri native translation source must be module file 0"
    );
    module.target_info = Some(TargetInfo {
        triple: native_target_triple(),
        pointer_size: std::mem::size_of::<usize>() as u32,
        endianness: native_endianness(),
        // ABI derived from the triple (the documented legacy state) and the
        // default NativeC struct-passing policy: the petri successor module
        // passes only scalars, so no aggregate ever crosses its call edge.
        abi: None,
        struct_passing: trust_ir::StructPassingPolicy::default(),
    });
    // Reserve the producer-owned proof id before `add_function` synthesizes
    // Pending obligations for claim-bearing annotations. Appending this after
    // `add_function` would collide with its first generated id and make
    // request/source lookup select a Pending claim instead of this translation
    // obligation.
    module.proof_obligations.push(translation_obligation);
    let function_type = module.add_func_type(FuncTy {
        params: petri_native_successor_trust_ir_param_types().to_vec(),
        returns: vec![],
        is_vararg: false,
    });
    let mut function = Function::new(
        FuncId::new(0),
        PETRI_NATIVE_BUNDLE_FUNCTION,
        function_type,
        BlockId::new(0),
    );
    let mut builder =
        PetriNativeTrustIrBuilder::new(PETRI_NATIVE_SUCCESSOR_TRUST_IR_PARAM_COUNT as u32);
    let mut entry = Block::new(BlockId::new(0));
    for (index, ty) in petri_native_successor_trust_ir_param_types()
        .into_iter()
        .enumerate()
    {
        entry = entry.with_param(ValueId::new(index as u32), ty);
    }
    let out = builder.copy(&mut entry, Ty::Ptr, ValueId::new(0));
    let state_in = builder.copy(&mut entry, Ty::Ptr, ValueId::new(1));
    let state_len = builder.copy(&mut entry, Ty::U32, ValueId::new(2));
    let successors = builder.copy(&mut entry, Ty::Ptr, ValueId::new(7));
    let successor_capacity = builder.copy(&mut entry, Ty::U32, ValueId::new(8));
    let unsupported_block = builder.block_id();
    let end_block = builder.block_id();
    let first_transition_block = if cache.plans().is_empty() {
        end_block
    } else {
        builder.block_id()
    };
    let expected_state_len = builder.const_value(
        &mut entry,
        Ty::U32,
        u32::try_from(net.num_places()).unwrap_or(u32::MAX) as i128,
    );
    let state_len_ok = builder.icmp(
        &mut entry,
        ICmpOp::Eq,
        Ty::U32,
        state_len,
        expected_state_len,
    );
    let zero_u32 = builder.const_value(&mut entry, Ty::U32, 0);
    let zero_u64 = builder.const_value(&mut entry, Ty::U64, 0);
    builder.cond_br(
        &mut entry,
        state_len_ok,
        first_transition_block,
        vec![zero_u32, zero_u32, zero_u64],
        unsupported_block,
        vec![],
    );
    function.blocks.push(entry);

    let mut current_block = first_transition_block;
    for (index, plan) in cache.plans().iter().enumerate() {
        let next_block = if index + 1 == cache.plans().len() {
            end_block
        } else {
            builder.block_id()
        };
        let enabled_block = builder.block_id();
        let capacity_block = builder.block_id();
        let write_block = builder.block_id();

        let (mut check, counters) =
            builder.block_with_params(current_block, &[Ty::U32, Ty::U32, Ty::U64]);
        let generated = counters[0];
        let written = counters[1];
        let metadata = counters[2];
        let enabled = builder.emit_transition_enabled(&mut check, plan, state_in);
        builder.cond_br(
            &mut check,
            enabled,
            enabled_block,
            vec![generated, written, metadata],
            next_block,
            vec![generated, written, metadata],
        );
        function.blocks.push(check);

        let (mut enabled_body, counters) =
            builder.block_with_params(enabled_block, &[Ty::U32, Ty::U32, Ty::U64]);
        let generated = counters[0];
        let written = counters[1];
        let metadata = counters[2];
        let one = builder.const_value(&mut enabled_body, Ty::U32, 1);
        let generated_next = builder.binop(&mut enabled_body, BinOp::Add, Ty::U32, generated, one);
        let metadata_next = if index < u64::BITS as usize {
            let bit = builder.const_value(&mut enabled_body, Ty::U64, i128::from(1_u64 << index));
            builder.binop(&mut enabled_body, BinOp::Add, Ty::U64, metadata, bit)
        } else {
            metadata
        };
        let capacity_available = builder.icmp(
            &mut enabled_body,
            ICmpOp::Ult,
            Ty::U32,
            written,
            successor_capacity,
        );
        builder.cond_br(
            &mut enabled_body,
            capacity_available,
            capacity_block,
            vec![generated_next, written, metadata_next],
            next_block,
            vec![generated_next, written, metadata_next],
        );
        function.blocks.push(enabled_body);

        let (mut capacity_body, counters) =
            builder.block_with_params(capacity_block, &[Ty::U32, Ty::U32, Ty::U64]);
        let generated = counters[0];
        let written = counters[1];
        let metadata = counters[2];
        let successors_present = builder.ptr_non_null(&mut capacity_body, successors);
        builder.cond_br(
            &mut capacity_body,
            successors_present,
            write_block,
            vec![generated, written, metadata],
            next_block,
            vec![generated, written, metadata],
        );
        function.blocks.push(capacity_body);

        let (mut write, counters) =
            builder.block_with_params(write_block, &[Ty::U32, Ty::U32, Ty::U64]);
        let generated = counters[0];
        let written = counters[1];
        let metadata = counters[2];
        builder.emit_transition_successor(
            &mut write,
            plan,
            state_in,
            successors,
            written,
            net.num_places(),
        );
        let one = builder.const_value(&mut write, Ty::U32, 1);
        let written_next = builder.binop(&mut write, BinOp::Add, Ty::U32, written, one);
        builder.br(
            &mut write,
            next_block,
            vec![generated, written_next, metadata],
        );
        function.blocks.push(write);

        current_block = next_block;
    }

    let (mut end, counters) = builder.block_with_params(end_block, &[Ty::U32, Ty::U32, Ty::U64]);
    let generated = counters[0];
    let written = counters[1];
    let metadata = counters[2];
    builder.emit_successor_out(&mut end, out, state_len, generated, written, metadata);
    builder.ret_void(&mut end);
    function.blocks.push(end);

    let mut unsupported = Block::new(unsupported_block);
    builder.emit_unsupported_out(&mut unsupported, out, state_len);
    builder.ret_void(&mut unsupported);
    function.blocks.push(unsupported);

    function
        .proofs
        .push(trust_ir::ProofAnnotation::Deterministic);
    function.proofs.push(trust_ir::ProofAnnotation::NoPanic);
    function.proofs.push(trust_ir::ProofAnnotation::BoundedLoop(
        cache.plans().len() as u64
    ));
    module.add_function(function);
    module
}

/// Construct the exact module whose digest is admitted by the native bundle.
///
/// Proof obligations are part of canonical TrustIR serialization, so they must
/// be attached before `Module::stable_digest()` is evaluated. Keeping that
/// ordering in one helper prevents bundle construction and receipt validation
/// from authenticating different module values.
fn petri_native_successor_complete_trust_ir_module(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> trust_ir::Module {
    petri_native_successor_trust_ir_module(
        net,
        cache,
        petri_native_translation_obligation(net, cache, plan_digest, shared_planning_identity),
    )
}

struct PetriNativeTrustIrBuilder {
    next_value: u32,
    next_block: u32,
}

impl PetriNativeTrustIrBuilder {
    fn new(next_value: u32) -> Self {
        Self {
            next_value,
            next_block: 1,
        }
    }

    fn value(&mut self) -> ValueId {
        let value = ValueId::new(self.next_value);
        self.next_value += 1;
        value
    }

    fn block_id(&mut self) -> BlockId {
        let block = BlockId::new(self.next_block);
        self.next_block += 1;
        block
    }

    fn block_with_params(&mut self, id: BlockId, params: &[Ty]) -> (Block, Vec<ValueId>) {
        let mut block = Block::new(id);
        let values = params
            .iter()
            .cloned()
            .map(|ty| {
                let value = self.value();
                block.params.push((value, ty));
                value
            })
            .collect();
        (block, values)
    }

    fn const_value(&mut self, block: &mut Block, ty: Ty, value: i128) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Const {
                ty,
                value: Constant::Int(value),
            })
            .with_result(result),
        );
        result
    }

    fn bool_const(&mut self, block: &mut Block, value: bool) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::Bool,
                value: Constant::Bool(value),
            })
            .with_result(result),
        );
        result
    }

    fn binop(
        &mut self,
        block: &mut Block,
        op: BinOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let result = self.value();
        block
            .body
            .push(InstrNode::new(Inst::BinOp { op, ty, lhs, rhs }).with_result(result));
        result
    }

    fn cast(
        &mut self,
        block: &mut Block,
        op: CastOp,
        src_ty: Ty,
        dst_ty: Ty,
        operand: ValueId,
    ) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Cast {
                op,
                src_ty,
                dst_ty,
                operand,
            })
            .with_result(result),
        );
        result
    }

    fn icmp(
        &mut self,
        block: &mut Block,
        op: ICmpOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let result = self.value();
        block
            .body
            .push(InstrNode::new(Inst::ICmp { op, ty, lhs, rhs }).with_result(result));
        result
    }

    fn select(
        &mut self,
        block: &mut Block,
        ty: Ty,
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
    ) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Select {
                ty,
                cond,
                then_val,
                else_val,
            })
            .with_result(result),
        );
        result
    }

    fn copy(&mut self, block: &mut Block, ty: Ty, operand: ValueId) -> ValueId {
        let result = self.value();
        block
            .body
            .push(InstrNode::new(Inst::Copy { ty, operand }).with_result(result));
        result
    }

    fn bool_and(&mut self, block: &mut Block, lhs: ValueId, rhs: ValueId) -> ValueId {
        let false_value = self.bool_const(block, false);
        self.select(block, Ty::Bool, lhs, rhs, false_value)
    }

    fn ptr_non_null(&mut self, block: &mut Block, ptr: ValueId) -> ValueId {
        let raw = self.cast(block, CastOp::PtrToInt, Ty::Ptr, Ty::U64, ptr);
        let zero = self.const_value(block, Ty::U64, 0);
        self.icmp(block, ICmpOp::Ne, Ty::U64, raw, zero)
    }

    fn gep(&mut self, block: &mut Block, pointee_ty: Ty, base: ValueId, index: ValueId) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty,
                base,
                indices: vec![index],
                inbounds: false,
            })
            .with_result(result),
        );
        result
    }

    fn byte_gep(&mut self, block: &mut Block, base: ValueId, offset: u64) -> ValueId {
        let offset = self.const_value(block, Ty::U64, i128::from(offset));
        self.gep(block, Ty::U8, base, offset)
    }

    fn load(&mut self, block: &mut Block, ty: Ty, ptr: ValueId) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Load {
                ty,
                ptr,
                volatile: false,
                align: None,
            })
            .with_result(result),
        );
        result
    }

    fn store(&mut self, block: &mut Block, ty: Ty, ptr: ValueId, value: ValueId) {
        block.body.push(InstrNode::new(Inst::Store {
            ty,
            ptr,
            value,
            volatile: false,
            align: None,
        }));
    }

    fn store_at_offset(
        &mut self,
        block: &mut Block,
        ty: Ty,
        base: ValueId,
        offset: u64,
        value: ValueId,
    ) {
        let ptr = self.byte_gep(block, base, offset);
        self.store(block, ty, ptr, value);
    }

    fn load_at_offset(&mut self, block: &mut Block, ty: Ty, base: ValueId, offset: u64) -> ValueId {
        let ptr = self.byte_gep(block, base, offset);
        self.load(block, ty, ptr)
    }

    fn br(&mut self, block: &mut Block, target: BlockId, args: Vec<ValueId>) {
        block.body.push(InstrNode::new(Inst::Br { target, args }));
    }

    fn cond_br(
        &mut self,
        block: &mut Block,
        cond: ValueId,
        then_target: BlockId,
        then_args: Vec<ValueId>,
        else_target: BlockId,
        else_args: Vec<ValueId>,
    ) {
        block.body.push(InstrNode::new(Inst::CondBr {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        }));
    }

    fn ret_void(&mut self, block: &mut Block) {
        block
            .body
            .push(InstrNode::new(Inst::Return { values: vec![] }));
    }

    fn state_load(&mut self, block: &mut Block, state_in: ValueId, place: u32) -> ValueId {
        let index = self.const_value(block, Ty::U32, i128::from(place));
        let ptr = self.gep(block, Ty::I64, state_in, index);
        self.load(block, Ty::I64, ptr)
    }

    fn emit_transition_enabled(
        &mut self,
        block: &mut Block,
        plan: &super::TransitionKernelPlan,
        state_in: ValueId,
    ) -> ValueId {
        let mut inputs = plan.inputs.iter();
        let Some(&(place, weight)) = inputs.next() else {
            return self.bool_const(block, true);
        };
        let token = self.state_load(block, state_in, place.0);
        let required = self.const_value(block, Ty::I64, i128::from(weight));
        let mut enabled = self.icmp(block, ICmpOp::Sge, Ty::I64, token, required);
        for &(place, weight) in inputs {
            let token = self.state_load(block, state_in, place.0);
            let required = self.const_value(block, Ty::I64, i128::from(weight));
            let enough = self.icmp(block, ICmpOp::Sge, Ty::I64, token, required);
            enabled = self.bool_and(block, enabled, enough);
        }
        enabled
    }

    fn emit_transition_successor(
        &mut self,
        block: &mut Block,
        plan: &super::TransitionKernelPlan,
        state_in: ValueId,
        successors: ValueId,
        written: ValueId,
        place_count: usize,
    ) {
        // The successor-row store index `written * place_count + place` is
        // computed in 64-bit arithmetic. Computing it in U32 (as an earlier
        // revision did) silently wraps once the product reaches 2^32, landing
        // stores inside EARLIER rows of the buffer — a silent wrong-marking
        // class no release-mode status/count gate can see. Bound proof for the
        // U64 computation: `written < successor_capacity <= u32::MAX` and
        // `place_count <= u32::MAX`, so
        //   slot <= (2^32-1)*(2^32-1) + (2^32-1) = 2^64 - 2^32 < 2^64
        // — no u64 wrap is reachable. The GEP's i64 index interpretation is
        // also exact for every buffer the host can actually allocate: a
        // `Vec<i64>` of `successor_capacity * place_count` slots caps the
        // largest valid slot far below 2^63 (a 2^63-slot buffer would need
        // 64 EiB). `extend_index_to_i64_if_needed` in trust-cg-lower passes
        // 64-bit indexes through unchanged (no post-wrap zero-extension).
        let written64 = self.cast(block, CastOp::ZExt, Ty::U32, Ty::U64, written);
        let state_len64 = self.const_value(block, Ty::U64, i128::from(place_count as u64));
        let row_base = self.binop(block, BinOp::Mul, Ty::U64, written64, state_len64);
        for place in 0..place_count {
            let place_u32 = u32::try_from(place).unwrap_or(u32::MAX);
            let mut value = self.state_load(block, state_in, place_u32);
            for &(input_place, weight) in &plan.inputs {
                if input_place.0 as usize == place {
                    let weight = self.const_value(block, Ty::I64, i128::from(weight));
                    value = self.binop(block, BinOp::Sub, Ty::I64, value, weight);
                }
            }
            for &(output_place, weight) in &plan.outputs {
                if output_place.0 as usize == place {
                    let weight = self.const_value(block, Ty::I64, i128::from(weight));
                    value = self.binop(block, BinOp::Add, Ty::I64, value, weight);
                }
            }
            let place_index = self.const_value(block, Ty::U64, i128::from(place as u64));
            let slot = self.binop(block, BinOp::Add, Ty::U64, row_base, place_index);
            let ptr = self.gep(block, Ty::I64, successors, slot);
            self.store(block, Ty::I64, ptr, value);
        }
    }

    fn emit_successor_out(
        &mut self,
        block: &mut Block,
        out: ValueId,
        state_len: ValueId,
        generated: ValueId,
        written: ValueId,
        metadata: ValueId,
    ) {
        let zero_u32 = self.const_value(block, Ty::U32, 0);
        let has_successor = self.icmp(block, ICmpOp::Ne, Ty::U32, generated, zero_u32);
        let full = self.icmp(block, ICmpOp::Eq, Ty::U32, written, generated);
        let ok_status = self.const_value(block, Ty::U8, i128::from(SUCCESSOR_STATUS_OK));
        let overflow_status =
            self.const_value(block, Ty::U8, i128::from(SUCCESSOR_STATUS_BUFFER_OVERFLOW));
        let disabled_status =
            self.const_value(block, Ty::U8, i128::from(SUCCESSOR_STATUS_DISABLED));
        let success_status = self.select(block, Ty::U8, full, ok_status, overflow_status);
        let status = self.select(
            block,
            Ty::U8,
            has_successor,
            success_status,
            disabled_status,
        );
        let raw_overflow = self.binop(block, BinOp::Sub, Ty::U32, generated, written);
        let overflow = self.select(block, Ty::U32, full, zero_u32, raw_overflow);

        self.store_at_offset(block, Ty::U8, out, SUCCESSOR_OUT_STATUS_OFFSET, status);
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_SUCCESSOR_COUNT_OFFSET,
            written,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_GENERATED_COUNT_OFFSET,
            generated,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_STATE_LEN_OFFSET,
            state_len,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_OVERFLOW_COUNT_OFFSET,
            overflow,
        );
        let runtime_error = self.const_value(
            block,
            Ty::U8,
            i128::from(SUCCESSOR_RUNTIME_ERROR_DIVISION_BY_ZERO),
        );
        self.store_at_offset(
            block,
            Ty::U8,
            out,
            SUCCESSOR_OUT_RUNTIME_ERROR_OFFSET,
            runtime_error,
        );
        let unsupported_reason =
            self.const_value(block, Ty::U8, i128::from(SUCCESSOR_UNSUPPORTED_REASON_NONE));
        self.store_at_offset(
            block,
            Ty::U8,
            out,
            SUCCESSOR_OUT_UNSUPPORTED_REASON_OFFSET,
            unsupported_reason,
        );
        self.store_at_offset(
            block,
            Ty::U64,
            out,
            SUCCESSOR_OUT_METADATA_BITS_OFFSET,
            metadata,
        );
    }

    fn emit_unsupported_out(&mut self, block: &mut Block, out: ValueId, state_len: ValueId) {
        let zero_u32 = self.const_value(block, Ty::U32, 0);
        let unsupported_status =
            self.const_value(block, Ty::U8, i128::from(SUCCESSOR_STATUS_UNSUPPORTED));
        self.store_at_offset(
            block,
            Ty::U8,
            out,
            SUCCESSOR_OUT_STATUS_OFFSET,
            unsupported_status,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_SUCCESSOR_COUNT_OFFSET,
            zero_u32,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_GENERATED_COUNT_OFFSET,
            zero_u32,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_STATE_LEN_OFFSET,
            state_len,
        );
        self.store_at_offset(
            block,
            Ty::U32,
            out,
            SUCCESSOR_OUT_OVERFLOW_COUNT_OFFSET,
            zero_u32,
        );
        let runtime_error = self.const_value(
            block,
            Ty::U8,
            i128::from(SUCCESSOR_RUNTIME_ERROR_DIVISION_BY_ZERO),
        );
        self.store_at_offset(
            block,
            Ty::U8,
            out,
            SUCCESSOR_OUT_RUNTIME_ERROR_OFFSET,
            runtime_error,
        );
        let unsupported_reason = self.const_value(
            block,
            Ty::U8,
            i128::from(SUCCESSOR_UNSUPPORTED_REASON_UNSUPPORTED_STATE_LAYOUT),
        );
        self.store_at_offset(
            block,
            Ty::U8,
            out,
            SUCCESSOR_OUT_UNSUPPORTED_REASON_OFFSET,
            unsupported_reason,
        );
        let metadata = self.const_value(block, Ty::U64, 0);
        self.store_at_offset(
            block,
            Ty::U64,
            out,
            SUCCESSOR_OUT_METADATA_BITS_OFFSET,
            metadata,
        );
    }
}

fn petri_native_translation_obligation(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> ProofObligation {
    let formula_body = format!(
        "places={} transitions={} abi_version={} plan_digest={} shared_planning_identity_digest={} fingerprint_domain_identity={} cache_digest={} trust_cg_batch_cache_reuse_status={} trust_ir_successor_body_status=lowered_all_transition_successors semantic_replay=explicit_plan_cache_parity",
        net.num_places(),
        net.num_transitions(),
        cache.abi_version,
        plan_digest,
        shared_planning_identity.planning_identity_digest,
        shared_planning_identity.fingerprint_domain_identity,
        shared_planning_identity.cache_digest,
        shared_planning_identity.trust_cg_batch_cache_reuse_status,
    );
    let public_semantic_digest = ProofDigest::sha256_domain(
        PETRI_NATIVE_TRANSLATION_SEMANTIC_DIGEST_DOMAIN,
        formula_body.as_bytes(),
    );
    ProofObligation::new(
        PETRI_NATIVE_TRANSLATION_OBLIGATION,
        ObligationKind::TranslationValidation,
        ProofStatus::Discharged,
        "PetriKernelPlanCache trust-ir successor bundle preserves Rust Petri successor semantics",
    )
    .with_formula(ProofFormula::new(
        tla_trust_cg::PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA,
        formula_body,
    ))
    .with_function(FuncId::new(0))
    .with_source(
        ProofObligationSourceIdentity::new(
            PETRI_NATIVE_TRANSLATION_SOURCE_ID,
            PETRI_NATIVE_TRANSLATION_ASSERTION_ID,
        )
        .with_range(ProofObligationSourceRange {
            file: 0,
            start_line: 1,
            start_col: 1,
            end_line: 1,
            end_col: 1,
        })
        .with_public(PublicObligationIdentity {
            obligation_id: PETRI_NATIVE_TRANSLATION_PUBLIC_OBLIGATION_ID.to_string(),
            semantic_digest: public_semantic_digest,
        }),
    )
}

fn attach_petri_native_successor_semantic_evidence(
    bundle: &mut NativeVerificationBundle,
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    trust_ir_module_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) {
    let Some(request) = bundle.requests.first().cloned() else {
        return;
    };
    let artifacts = petri_native_successor_semantic_evidence_artifacts(
        net,
        cache,
        plan_digest,
        trust_ir_module_digest,
        shared_planning_identity,
    );
    let Ok(evidence) = bundle.evidence_bundle_for_request(&request, artifacts) else {
        return;
    };
    bundle.evidence_bundles.push(evidence);
}

fn petri_native_successor_semantic_evidence_artifacts(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    trust_ir_module_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> Vec<NativeEvidenceArtifact> {
    let horn_clauses = petri_native_successor_horn_clause_replay(
        net,
        cache,
        plan_digest,
        shared_planning_identity,
    );
    let replay = petri_native_successor_replay_transcript(
        net,
        cache,
        plan_digest,
        trust_ir_module_digest,
        shared_planning_identity,
    );
    let model = petri_native_successor_model_artifact(
        net,
        cache,
        plan_digest,
        trust_ir_module_digest,
        shared_planning_identity,
    );

    vec![
        NativeEvidenceArtifact::new(
            "petri-successor.smt2",
            NativeEvidenceArtifactKind::TrustMcHornClauses,
            ProofDigest::sha256_domain(
                "ty.petri.native.successor.trust_mc.horn_clauses.v1",
                horn_clauses.as_bytes(),
            ),
        ),
        NativeEvidenceArtifact::new(
            "petri-successor.replay.json",
            NativeEvidenceArtifactKind::ReplayTranscript,
            ProofDigest::sha256_domain(
                "ty.petri.native.successor.trust_mc.replay_artifact.v1",
                replay.as_bytes(),
            ),
        ),
        NativeEvidenceArtifact::new(
            "petri-successor.model.json",
            NativeEvidenceArtifactKind::TrustMcModel,
            ProofDigest::sha256_domain(
                "ty.petri.native.successor.trust_mc.model.v1",
                model.as_bytes(),
            ),
        ),
    ]
}

fn petri_native_successor_horn_clause_replay(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> String {
    let mut text = String::new();
    text.push_str("(set-logic HORN)\n");
    text.push_str("; ty.petri.native.successor.semantic_replay.v1\n");
    text.push_str(&format!("; places {}\n", net.num_places()));
    text.push_str(&format!("; transitions {}\n", net.num_transitions()));
    text.push_str(&format!("; plan_digest {}\n", plan_digest));
    text.push_str(&format!(
        "; shared_planning_identity_digest {}\n",
        shared_planning_identity.planning_identity_digest
    ));
    let place_args = (0..net.num_places())
        .map(|place| format!("p{place} Int"))
        .collect::<Vec<_>>()
        .join(") (");
    let next_args = (0..net.num_places())
        .map(|place| format!("q{place} Int"))
        .collect::<Vec<_>>()
        .join(") (");
    text.push_str(&format!(
        "(declare-fun PetriSucc ({}) Bool)\n",
        (0..(net.num_places() * 2))
            .map(|_| "Int")
            .collect::<Vec<_>>()
            .join(" ")
    ));
    for plan in cache.plans() {
        text.push_str(&format!("; transition {}\n", plan.transition.0));
        text.push_str("(assert (forall (");
        if !place_args.is_empty() {
            text.push('(');
            text.push_str(&place_args);
            text.push(')');
        }
        if !next_args.is_empty() {
            if !place_args.is_empty() {
                text.push(' ');
            }
            text.push('(');
            text.push_str(&next_args);
            text.push(')');
        }
        text.push_str(") (=> (and");
        for &(place, weight) in &plan.inputs {
            text.push_str(&format!(" (>= p{} {})", place.0, weight));
        }
        for place in 0..net.num_places() {
            let mut delta = 0_i128;
            for &(input_place, weight) in &plan.inputs {
                if input_place.0 as usize == place {
                    delta -= i128::from(weight);
                }
            }
            for &(output_place, weight) in &plan.outputs {
                if output_place.0 as usize == place {
                    delta += i128::from(weight);
                }
            }
            if delta == 0 {
                text.push_str(&format!(" (= q{place} p{place})"));
            } else if delta > 0 {
                text.push_str(&format!(" (= q{place} (+ p{place} {delta}))"));
            } else {
                text.push_str(&format!(" (= q{place} (- p{place} {}))", -delta));
            }
        }
        text.push_str(") (PetriSucc");
        for place in 0..net.num_places() {
            text.push_str(&format!(" p{place}"));
        }
        for place in 0..net.num_places() {
            text.push_str(&format!(" q{place}"));
        }
        text.push_str("))))\n");
    }
    text.push_str("(check-sat)\n");
    text
}

fn petri_native_successor_replay_transcript(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    trust_ir_module_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> String {
    format!(
        "{{\"schema\":\"ty.petri.native.successor.replay.v1\",\"places\":{},\"transitions\":{},\"abi_version\":{},\"plan_digest\":\"{}\",\"trust_ir_module_digest\":\"{}\",\"shared_planning_identity_digest\":\"{}\",\"body\":\"lowered_all_transition_successors\",\"parity\":\"PetriKernelPlanCache::fire_transition_plan_flat\"}}",
        net.num_places(),
        net.num_transitions(),
        cache.abi_version,
        plan_digest,
        trust_ir_module_digest,
        shared_planning_identity.planning_identity_digest
    )
}

fn petri_native_successor_model_artifact(
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    plan_digest: ProofDigest,
    trust_ir_module_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> String {
    format!(
        "{{\"schema\":\"ty.petri.native.successor.model.v1\",\"successor_relation\":\"all_transition_plan_cache_equivalence\",\"places\":{},\"transitions\":{},\"abi_version\":{},\"plan_digest\":\"{}\",\"trust_ir_module_digest\":\"{}\",\"cache_digest\":\"{}\",\"fingerprint_domain_identity\":\"{}\"}}",
        net.num_places(),
        net.num_transitions(),
        cache.abi_version,
        plan_digest,
        trust_ir_module_digest,
        shared_planning_identity.cache_digest,
        shared_planning_identity.fingerprint_domain_identity
    )
}

fn petri_native_successor_lineage(
    plan_digest: ProofDigest,
    trust_ir_module_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> ProofLineageManifest {
    let mut node = ProofLineageNode::new(
        PETRI_NATIVE_LINEAGE_ROOT,
        ProofTransform::new(
            ProofTransformStage::TrustIrLowering,
            "petri-plan-cache-to-trust-ir-native-successor-bundle",
            "tla-petri",
            PETRI_NATIVE_BUNDLE_SCHEMA_VERSION,
        ),
        plan_digest,
        trust_ir_module_digest,
    );
    node.obligations.push(PETRI_NATIVE_TRANSLATION_OBLIGATION);
    node.replay = Some(
        ProofReplayIdentity::new(
            "tla-petri",
            "PetriKernelPlanCache::for_net -> trust_ir::NativeVerificationBundle",
        )
        .with_transcript_digest(ProofDigest::sha256_domain(
            "ty.petri.native.successor.bundle.replay.v1",
            &shared_planning_digest_payload(plan_digest, shared_planning_identity),
        )),
    );

    ProofLineageManifest {
        schema_version: ProofLineageManifest::SCHEMA_VERSION,
        nodes: vec![node],
        roots: vec![PETRI_NATIVE_LINEAGE_ROOT],
    }
}

pub(crate) fn petri_kernel_plan_cache_digest(cache: &PetriKernelPlanCache) -> ProofDigest {
    let bytes = petri_kernel_plan_cache_digest_material(cache);
    ProofDigest::sha256_domain("ty.petri.kernel_plan_cache.v1", &bytes)
}

fn petri_kernel_plan_cache_digest_material(cache: &PetriKernelPlanCache) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_str(&mut bytes, "ty.petri.kernel_plan_cache.v1");
    write_u32(&mut bytes, cache.abi_version);
    write_u32(&mut bytes, cache.place_count as u32);
    write_u32(&mut bytes, cache.transition_count as u32);
    write_u32(&mut bytes, cache.plans().len() as u32);
    for plan in cache.plans() {
        write_u32(&mut bytes, plan.transition.0);
        write_u32(&mut bytes, plan.inputs.len() as u32);
        for (place, weight) in &plan.inputs {
            write_u32(&mut bytes, place.0);
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        write_u32(&mut bytes, plan.outputs.len() as u32);
        for (place, weight) in &plan.outputs {
            write_u32(&mut bytes, place.0);
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
    }
    bytes
}

fn digest_payload(digest: ProofDigest) -> [u8; 32] {
    digest.bytes
}

fn shared_planning_digest_payload(
    plan_digest: ProofDigest,
    shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&digest_payload(plan_digest));
    write_str(
        &mut bytes,
        &shared_planning_identity.planning_identity_digest,
    );
    write_str(&mut bytes, &shared_planning_identity.cache_digest);
    bytes
}

impl PetriNativeSuccessorBatchReadinessPacket {
    fn blocked_without_artifact(
        net: &PetriNet,
        shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
        reason_code: &'static str,
        blocker: String,
    ) -> Self {
        let state_len = u32::try_from(net.num_places()).unwrap_or(u32::MAX);
        let max_successors = u32::try_from(net.num_transitions()).unwrap_or(u32::MAX);
        Self::base(state_len, max_successors, shared_planning_identity).with_status(
            PETRI_NATIVE_CANDIDATE_STATUS_BLOCKED,
            reason_code,
            blocker,
        )
    }

    fn blocked_for_bundle(
        bundle: &trust_ir::NativeVerificationBundle,
        shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
        state_len: u32,
        max_successors: u32,
        reason_code: &'static str,
        blocker: String,
    ) -> Self {
        let mut packet = Self::base(state_len, max_successors, shared_planning_identity)
            .with_bundle_identity(bundle)
            .with_status(PETRI_NATIVE_CANDIDATE_STATUS_BLOCKED, reason_code, blocker);
        packet.trust_ir_entry_abi_matches_shared_successor_kernel =
            trust_ir_entry_uses_shared_successor_kernel_abi(bundle);
        packet
    }

    fn blocked_for_handoff(
        net: &PetriNet,
        cache: &PetriKernelPlanCache,
        bundle: &trust_ir::NativeVerificationBundle,
        shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
        state_len: u32,
        max_successors: u32,
        reason_code: &'static str,
        handoff: &tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
        runtime_readiness: &tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
    ) -> Self {
        Self::for_artifact(
            net,
            cache,
            bundle,
            shared_planning_identity,
            state_len,
            max_successors,
            handoff,
            runtime_readiness,
        )
        .with_status(
            PETRI_NATIVE_CANDIDATE_STATUS_BLOCKED,
            reason_code,
            handoff
                .reason_code
                .unwrap_or(PETRI_NATIVE_CANDIDATE_REASON_COMPILE_HANDOFF_BLOCKED)
                .to_string(),
        )
    }

    fn callable_artifact(
        net: &PetriNet,
        cache: &PetriKernelPlanCache,
        bundle: &trust_ir::NativeVerificationBundle,
        installed_artifact: &PetriNativeInstalledArtifact,
        shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
        state_len: u32,
        max_successors: u32,
        handoff: &tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
        runtime_readiness: &tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
    ) -> Self {
        let packet = Self::for_artifact(
            net,
            cache,
            bundle,
            shared_planning_identity,
            state_len,
            max_successors,
            handoff,
            runtime_readiness,
        );
        if let Some(blocker) = petri_native_candidate_promotion_blocker(
            bundle,
            net,
            cache,
            handoff,
            runtime_readiness,
            installed_artifact,
        ) {
            return packet.with_status(
                PETRI_NATIVE_CANDIDATE_STATUS_CALLABLE_ARTIFACT,
                blocker.reason_code,
                blocker.detail,
            );
        }

        packet.with_production_selection()
    }

    fn for_artifact(
        net: &PetriNet,
        cache: &PetriKernelPlanCache,
        bundle: &trust_ir::NativeVerificationBundle,
        shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
        state_len: u32,
        max_successors: u32,
        handoff: &tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
        runtime_readiness: &tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
    ) -> Self {
        let mut packet = Self::base(state_len, max_successors, shared_planning_identity)
            .with_bundle_identity(bundle);
        packet.trust_ir_entry_abi_matches_shared_successor_kernel =
            trust_ir_entry_uses_shared_successor_kernel_abi(bundle);
        packet.runtime_readiness_status_code = runtime_readiness.status.as_str();
        packet.runtime_readiness_reason_code = runtime_readiness
            .reason_code
            .unwrap_or(PETRI_NATIVE_CANDIDATE_BLOCKER_NONE);
        packet.runtime_readiness_required_evidence = runtime_readiness
            .required_evidence
            .unwrap_or(PETRI_NATIVE_CANDIDATE_BLOCKER_NONE);
        packet.runtime_readiness_packet_sha256 =
            runtime_readiness.runtime_readiness_packet_sha256.clone();
        packet.runtime_ready_for_call = runtime_readiness.ready_for_runtime_call;
        packet.compile_artifact_handoff_ready = handoff.is_ready();
        packet.callable_pointer_available = handoff.callable_pointer.is_some();
        packet.native_payload_sha256 = handoff
            .native_payload_sha256
            .clone()
            .unwrap_or_else(|| PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string());
        packet.executable_region_sha256 = handoff
            .executable_region_sha256
            .clone()
            .unwrap_or_else(|| PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string());
        packet.lifetime_owner = handoff
            .lifetime_owner
            .clone()
            .unwrap_or_else(|| PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string());
        packet.current_generation = handoff.current_generation.unwrap_or(0);
        packet.validation_receipt_status =
            if petri_native_successor_semantic_receipt_available(bundle, net, cache) {
                PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED.to_string()
            } else {
                PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_MISSING.to_string()
            };
        packet.parity_receipt_status = if packet.validation_receipt_status
            == PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED
            && handoff.is_ready()
            && runtime_readiness.authorizes_useful_native()
        {
            PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED.to_string()
        } else {
            PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_MISSING.to_string()
        };
        packet.callable_receipt_status =
            if handoff.is_ready() && runtime_readiness.authorizes_useful_native() {
                PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED.to_string()
            } else {
                PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_MISSING.to_string()
            };
        packet.callable_receipt_reason_code = if packet.callable_receipt_status
            == PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        {
            PETRI_NATIVE_CANDIDATE_REASON_AVAILABLE.to_string()
        } else {
            runtime_readiness
                .reason_code
                .unwrap_or(PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_REASON_MISSING)
                .to_string()
        };
        packet
    }

    fn base(
        state_len: u32,
        max_successors: u32,
        shared_planning_identity: &PetriNativeSharedPlanningFingerprintIdentity,
    ) -> Self {
        let signature = KernelSymbolSignature::native_successor_kernel();
        Self {
            schema: PETRI_NATIVE_CANDIDATE_BATCH_SCHEMA,
            schema_version: PETRI_NATIVE_CANDIDATE_BATCH_SCHEMA_VERSION,
            api: PETRI_NATIVE_CANDIDATE_BATCH_API,
            status_code: PETRI_NATIVE_CANDIDATE_STATUS_BLOCKED,
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_AVAILABLE,
            blocker: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            entry_symbol: PETRI_NATIVE_SUCCESSOR_ENTRY_SYMBOL,
            artifact_kind: SUCCESSOR_KERNEL_ARTIFACT_KIND,
            shared_signature_abi: signature.abi,
            shared_signature_params: signature.params.len(),
            shared_signature_returns: signature.returns.len(),
            trust_ir_entry_abi_matches_shared_successor_kernel: false,
            state_len,
            max_successors,
            state_bytes: u64::from(state_len) * std::mem::size_of::<i64>() as u64,
            runtime_readiness_status_code: PETRI_NATIVE_CANDIDATE_STATUS_BLOCKED,
            runtime_readiness_reason_code: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE,
            runtime_readiness_required_evidence: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE,
            runtime_readiness_packet_sha256: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            runtime_ready_for_call: false,
            compile_artifact_handoff_ready: false,
            callable_pointer_available: false,
            native_payload_sha256: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            executable_region_sha256: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            lifetime_owner: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            current_generation: 0,
            transport_digest: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            bundle_digest: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            target_abi_digest: PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string(),
            shared_planning_identity_schema: shared_planning_identity.schema.clone(),
            shared_planning_identity_schema_version: shared_planning_identity
                .schema_version
                .clone(),
            shared_planning_identity_status: shared_planning_identity
                .planning_identity_status
                .clone(),
            shared_planning_identity_digest: shared_planning_identity
                .planning_identity_digest
                .clone(),
            shared_planning_identity_required_fields: shared_planning_identity
                .planning_identity_required_fields
                .clone(),
            shared_prepared_program_identity: shared_planning_identity
                .prepared_program_identity
                .clone(),
            shared_candidate_identity: shared_planning_identity.candidate_identity.clone(),
            shared_lane_identity: shared_planning_identity.lane_identity.clone(),
            shared_layout_checksum: shared_planning_identity.layout_checksum.clone(),
            shared_semantic_checksum: shared_planning_identity.semantic_checksum.clone(),
            shared_source_checksum: shared_planning_identity.source_checksum.clone(),
            shared_payload_checksum: shared_planning_identity.payload_checksum.clone(),
            shared_manifest_checksum: shared_planning_identity.manifest_checksum.clone(),
            shared_fingerprint_domain_identity: shared_planning_identity
                .fingerprint_domain_identity
                .clone(),
            shared_fingerprint_policy_identity: shared_planning_identity
                .fingerprint_policy_identity
                .clone(),
            shared_cache_namespace_identity: shared_planning_identity
                .cache_namespace_identity
                .clone(),
            shared_cache_reuse_policy: shared_planning_identity.cache_reuse_policy.clone(),
            shared_cache_digest: shared_planning_identity.cache_digest.clone(),
            prepared_trust_ir_reuse_identity: shared_planning_identity
                .prepared_trust_ir_reuse_identity
                .clone(),
            trust_cg_batch_cache_reuse_status: shared_planning_identity
                .trust_cg_batch_cache_reuse_status
                .clone(),
            trust_cg_batch_cache_reuse_blocker_code: shared_planning_identity
                .trust_cg_batch_cache_reuse_blocker_code
                .clone(),
            validation_receipt_status: shared_planning_identity.validation_receipt_status.clone(),
            parity_receipt_status: shared_planning_identity.parity_receipt_status.clone(),
            callable_receipt_schema: PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_SCHEMA,
            callable_receipt_status: shared_planning_identity.callable_receipt_status.clone(),
            callable_receipt_reason_code: PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_REASON_MISSING
                .to_string(),
            native_missing_receipts: PETRI_NATIVE_CANDIDATE_MISSING_RECEIPTS,
            production_gate_status: shared_planning_identity.production_gate_status.clone(),
            production_selected: false,
            fail_closed: true,
        }
    }

    fn with_status(
        mut self,
        status_code: &'static str,
        reason_code: &'static str,
        blocker: String,
    ) -> Self {
        self.status_code = status_code;
        self.reason_code = reason_code;
        self.blocker = blocker;
        self.production_selected = false;
        self.fail_closed = true;
        self
    }

    fn with_production_selection(mut self) -> Self {
        debug_assert_eq!(
            self.validation_receipt_status,
            PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        );
        debug_assert_eq!(
            self.parity_receipt_status,
            PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        );
        debug_assert_eq!(
            self.callable_receipt_status,
            PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        );
        self.status_code = PETRI_NATIVE_CANDIDATE_STATUS_CALLABLE_ARTIFACT;
        self.reason_code = PETRI_NATIVE_CANDIDATE_REASON_AVAILABLE;
        self.blocker = PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string();
        self.callable_receipt_reason_code = PETRI_NATIVE_CANDIDATE_REASON_AVAILABLE.to_string();
        self.native_missing_receipts = PETRI_NATIVE_CANDIDATE_MISSING_RECEIPTS_NONE;
        self.production_gate_status =
            PETRI_NATIVE_CANDIDATE_PRODUCTION_GATE_STATUS_SELECTED.to_string();
        self.production_selected = true;
        self.fail_closed = false;
        self
    }

    fn with_bundle_identity(mut self, bundle: &trust_ir::NativeVerificationBundle) -> Self {
        let identity = bundle.transport_identity();
        self.transport_digest = identity.stable_digest().to_string();
        self.bundle_digest = identity.bundle_digest.to_string();
        self.target_abi_digest = identity
            .target_abi
            .as_ref()
            .map(|target_abi| target_abi.digest.to_string())
            .unwrap_or_else(|| PETRI_NATIVE_CANDIDATE_BLOCKER_NONE.to_string());
        self
    }
}

fn petri_native_candidate_promotion_blocker(
    bundle: &trust_ir::NativeVerificationBundle,
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
    handoff: &tla_trust_cg::PetriNativeSuccessorCompileArtifactHandoffEvidence,
    runtime_readiness: &tla_trust_cg::PetriNativeSuccessorRuntimeReadinessPacket,
    installed_artifact: &PetriNativeInstalledArtifact,
) -> Option<PetriNativeCandidatePromotionBlocker> {
    if !trust_ir_entry_uses_shared_successor_kernel_abi(bundle) {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_ABI_MISMATCH,
            detail: "trust-ir entry function does not match the shared successor-kernel ABI"
                .to_string(),
        });
    }

    if bundle.validate().is_err() {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_BUNDLE_BLOCKED,
            detail: PETRI_NATIVE_BUNDLE_VALIDATION_BLOCKER_DETAIL.to_string(),
        });
    }

    if !petri_native_successor_semantic_receipt_available(bundle, net, cache) {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_VALIDATION_RECEIPT,
            detail: "validated trust-ir bundle is missing the discharged Petri successor semantic obligation and trust_mc/replay/model evidence artifacts".to_string(),
        });
    }

    if !handoff.is_ready() {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_COMPILE_HANDOFF_BLOCKED,
            detail: handoff
                .reason_code
                .unwrap_or(PETRI_NATIVE_CANDIDATE_REASON_COMPILE_HANDOFF_BLOCKED)
                .to_string(),
        });
    }

    if !runtime_readiness.authorizes_useful_native() {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_RUNTIME_READINESS,
            detail: runtime_readiness
                .reason_code
                .unwrap_or(PETRI_NATIVE_CANDIDATE_REASON_RUNTIME_READINESS)
                .to_string(),
        });
    }

    if handoff.callable_pointer != runtime_readiness.callable_pointer {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_CALLABLE_POINTER_MISMATCH,
            detail:
                "compile-artifact handoff callable pointer does not match runtime readiness packet"
                    .to_string(),
        });
    }

    let Some(entrypoint) = installed_artifact
        .artifact
        .entrypoint_ptr(installed_artifact.lookup_entry_symbol())
    else {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_RUNTIME_READINESS,
            detail: format!(
                "installed artifact does not export runtime entrypoint {}",
                installed_artifact.lookup_entry_symbol()
            ),
        });
    };
    let Some(entrypoint_pointer) =
        tla_trust_cg::PetriNativeSuccessorCallablePointer::from_ptr(entrypoint.as_ptr())
    else {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_CALLABLE_POINTER_MISMATCH,
            detail: "installed artifact runtime entrypoint pointer is null".to_string(),
        });
    };
    if Some(entrypoint_pointer) != runtime_readiness.callable_pointer {
        return Some(PetriNativeCandidatePromotionBlocker {
            reason_code: PETRI_NATIVE_CANDIDATE_REASON_CALLABLE_POINTER_MISMATCH,
            detail:
                "runtime readiness callable pointer does not match the installed artifact entrypoint"
                    .to_string(),
        });
    }

    None
}

pub(crate) fn petri_native_successor_semantic_receipt_available(
    bundle: &trust_ir::NativeVerificationBundle,
    net: &PetriNet,
    cache: &PetriKernelPlanCache,
) -> bool {
    if bundle.validate().is_err() {
        return false;
    }
    let Some(request) = bundle.requests.first() else {
        return false;
    };

    let plan_digest = petri_kernel_plan_cache_digest(cache);
    let shared_planning_identity = PetriNativeSharedPlanningFingerprintIdentity::for_net(net);
    let expected_module = petri_native_successor_complete_trust_ir_module(
        net,
        cache,
        plan_digest,
        &shared_planning_identity,
    );
    let trust_ir_module_digest = expected_module.stable_digest();
    // `validate()` above already binds the advertised digest to
    // `bundle.module`; compare that authenticated value to the independently
    // reconstructed canonical module without serializing the bundle a second
    // time.
    if bundle.trust_ir_module_digest != trust_ir_module_digest {
        return false;
    }
    let expected_obligation =
        petri_native_translation_obligation(net, cache, plan_digest, &shared_planning_identity);

    let discharged_translation = bundle.module.proof_obligations.iter().any(|obligation| {
        obligation.id == expected_obligation.id
            && obligation.kind == expected_obligation.kind
            && obligation.status == ProofStatus::Discharged
            && obligation.formula == expected_obligation.formula
    });
    if !discharged_translation {
        return false;
    }

    let expected_artifacts = petri_native_successor_semantic_evidence_artifacts(
        net,
        cache,
        plan_digest,
        trust_ir_module_digest,
        &shared_planning_identity,
    );

    expected_artifacts.iter().all(|expected| {
        bundle.evidence_bundles.iter().any(|evidence| {
            evidence.request() == request.id()
                && evidence.verifier_suite() == request.verifier_suite()
                && evidence
                    .artifacts()
                    .iter()
                    .any(|artifact| artifact == expected)
        })
    })
}

fn trust_ir_entry_uses_shared_successor_kernel_abi(
    bundle: &trust_ir::NativeVerificationBundle,
) -> bool {
    let Some(function) = bundle
        .module
        .functions
        .iter()
        .find(|function| function.name == PETRI_NATIVE_BUNDLE_FUNCTION)
    else {
        return false;
    };
    let Some(function_type) = bundle.module.func_types.get(function.ty.as_usize()) else {
        return false;
    };
    let expected_params = petri_native_successor_trust_ir_param_types();
    function_type.params.as_slice() == expected_params.as_slice()
        && function_type.returns.is_empty()
        && !function_type.is_vararg
}

fn petri_native_successor_trust_ir_param_types() -> [Ty; PETRI_NATIVE_SUCCESSOR_TRUST_IR_PARAM_COUNT]
{
    [
        Ty::Ptr,
        Ty::Ptr,
        Ty::U32,
        Ty::Ptr,
        Ty::U32,
        Ty::Ptr,
        Ty::U32,
        Ty::Ptr,
        Ty::U32,
    ]
}

fn usize_to_u32_saturating(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn native_target_triple() -> String {
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        "aarch64-apple-darwin".to_string()
    } else if cfg!(all(target_arch = "x86_64", target_os = "macos")) {
        "x86_64-apple-darwin".to_string()
    } else if cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        "x86_64-unknown-linux-gnu".to_string()
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        "aarch64-unknown-linux-gnu".to_string()
    } else {
        format!(
            "{}-unknown-{}",
            std::env::consts::ARCH,
            std::env::consts::OS
        )
    }
}

fn native_endianness() -> trust_ir::Endianness {
    if cfg!(target_endian = "little") {
        trust_ir::Endianness::Little
    } else {
        trust_ir::Endianness::Big
    }
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_str(bytes: &mut Vec<u8>, value: &str) {
    write_u32(bytes, value.len() as u32);
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(all(test, feature = "trust-cg-petri-native"))]
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

    #[test]
    fn native_bundle_binds_every_authority_digest_to_sha256_and_completed_module() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).expect("fixture plan cache should build");
        let plan_digest = petri_kernel_plan_cache_digest(&cache);
        assert_eq!(
            plan_digest,
            ProofDigest::sha256_domain(
                "ty.petri.kernel_plan_cache.v1",
                &petri_kernel_plan_cache_digest_material(&cache),
            )
        );
        assert_eq!(
            plan_digest.algorithm,
            trust_ir::ProofDigestAlgorithm::Sha256
        );

        let bundle = build_petri_native_successor_verification_bundle(&net, &cache);
        bundle
            .validate()
            .expect("completed native TrustIR bundle should validate");
        for (index, obligation) in bundle.module.proof_obligations.iter().enumerate() {
            assert!(
                !bundle.module.proof_obligations[..index]
                    .iter()
                    .any(|earlier| earlier.id == obligation.id),
                "completed native TrustIR module must not contain duplicate proof id {}",
                obligation.id
            );
        }
        let translation_obligation = bundle
            .module
            .proof_obligations
            .iter()
            .find(|obligation| obligation.id == PETRI_NATIVE_TRANSLATION_OBLIGATION)
            .expect("completed module should contain the producer translation obligation");
        assert_eq!(
            translation_obligation.kind,
            ObligationKind::TranslationValidation
        );
        assert_eq!(translation_obligation.status, ProofStatus::Discharged);
        assert_eq!(translation_obligation.function, Some(FuncId::new(0)));
        let embedded_source = translation_obligation
            .source
            .as_ref()
            .expect("requested translation obligation must embed its source identity");
        assert_eq!(
            embedded_source.source_id,
            PETRI_NATIVE_TRANSLATION_SOURCE_ID
        );
        assert_eq!(
            embedded_source.assertion_id,
            PETRI_NATIVE_TRANSLATION_ASSERTION_ID
        );
        let public_identity = embedded_source
            .public
            .as_ref()
            .expect("requested translation obligation must embed its public identity");
        assert_eq!(
            public_identity.obligation_id,
            PETRI_NATIVE_TRANSLATION_PUBLIC_OBLIGATION_ID
        );
        assert_eq!(
            public_identity.semantic_digest.algorithm,
            trust_ir::ProofDigestAlgorithm::Sha256
        );
        let formula = translation_obligation
            .formula
            .as_ref()
            .expect("translation obligation must carry its semantic formula");
        assert_eq!(
            formula.schema,
            tla_trust_cg::PETRI_NATIVE_SUCCESSOR_SEMANTIC_FORMULA_SCHEMA
        );
        assert_eq!(
            public_identity.semantic_digest,
            ProofDigest::sha256_domain(
                PETRI_NATIVE_TRANSLATION_SEMANTIC_DIGEST_DOMAIN,
                formula.payload.as_bytes(),
            ),
            "public obligation identity must bind the exact translation formula payload"
        );
        assert_eq!(bundle.trust_ir_module_digest, bundle.module.stable_digest());
        assert_eq!(
            bundle.trust_ir_module_digest.algorithm,
            trust_ir::ProofDigestAlgorithm::Sha256
        );
        assert_eq!(bundle.provenance.source_digest, Some(plan_digest));

        for node in &bundle.lineage.nodes {
            assert_eq!(
                node.source_module.algorithm,
                trust_ir::ProofDigestAlgorithm::Sha256
            );
            assert_eq!(
                node.target_module.algorithm,
                trust_ir::ProofDigestAlgorithm::Sha256
            );
            assert_eq!(node.target_module, bundle.trust_ir_module_digest);
            assert_eq!(
                node.replay
                    .as_ref()
                    .expect("native lineage should carry replay identity")
                    .transcript_digest
                    .expect("native lineage replay should be content-bound")
                    .algorithm,
                trust_ir::ProofDigestAlgorithm::Sha256
            );
        }
        for request in &bundle.requests {
            assert_eq!(
                request
                    .provenance()
                    .replay
                    .as_ref()
                    .expect("native request should carry replay identity")
                    .transcript_digest
                    .expect("native request replay should be content-bound")
                    .algorithm,
                trust_ir::ProofDigestAlgorithm::Sha256
            );
        }
        for evidence in &bundle.evidence_bundles {
            for artifact in evidence.artifacts() {
                assert_eq!(
                    artifact.digest.algorithm,
                    trust_ir::ProofDigestAlgorithm::Sha256
                );
            }
        }

        let mut stale = bundle.clone();
        stale.module.proof_obligations[0]
            .description
            .push_str(" changed");
        assert_ne!(stale.module.stable_digest(), stale.trust_ir_module_digest);
        assert!(
            stale.validate().is_err(),
            "changing a post-lowering obligation must invalidate the admitted module identity"
        );

        let mut stale_source = bundle.clone();
        stale_source.module.proof_obligations[0]
            .source
            .as_mut()
            .expect("translation source must be present")
            .assertion_id
            .push_str(":changed");
        assert_ne!(
            stale_source.module.stable_digest(),
            stale_source.trust_ir_module_digest
        );
        assert!(
            stale_source.validate().is_err(),
            "changing embedded source identity must invalidate the admitted module identity"
        );
    }

    #[test]
    fn native_candidate_promotion_rejects_runtime_callable_pointer_mismatch() {
        let net = all_transition_net();
        let cache = PetriKernelPlanCache::for_net(&net).expect("fixture plan cache should build");
        let layout = cache
            .validate_for_net(&net)
            .expect("fixture plan cache should validate");
        let bundle = match petri_native_successor_verification_bundle(&net, &cache) {
            PetriNativeVerificationBundleProduction::Available(bundle) => bundle,
            PetriNativeVerificationBundleProduction::Blocked(blocker) => {
                panic!("fixture native verification bundle should validate: {blocker:?}")
            }
        };
        let installed_artifact = match petri_native_successor_installed_artifact(&bundle) {
            PetriNativeInstalledArtifactProduction::Available(artifact) => artifact,
            PetriNativeInstalledArtifactProduction::Blocked(blocker) => {
                panic!("fixture installed artifact should compile: {blocker:?}")
            }
        };
        let mut handoff = installed_artifact
            .artifact
            .petri_native_successor_compile_artifact_handoff_evidence(Some(
                installed_artifact.lookup_entry_symbol(),
            ));
        normalize_frontend_entry_symbol(&mut handoff);
        let target_abi_digest = bundle
            .transport_identity()
            .target_abi
            .as_ref()
            .map(|target_abi| target_abi.digest);
        let runtime_inputs = super::super::trust_cg_petri_runtime_readiness_inputs(
            &bundle,
            u64::try_from(layout.state_len()).expect("state len should fit in u64")
                * std::mem::size_of::<i64>() as u64,
            target_abi_digest,
            &handoff,
        );
        let mut runtime_readiness = installed_artifact
            .artifact
            .petri_native_successor_runtime_readiness_packet(
                Some(installed_artifact.lookup_entry_symbol()),
                runtime_inputs.install_packet.as_ref(),
                runtime_inputs.trampoline_contract.as_ref(),
                runtime_inputs.call_packet.as_ref(),
                None,
            );
        assert!(handoff.is_ready());
        assert!(runtime_readiness.authorizes_useful_native());
        assert_eq!(handoff.callable_pointer, runtime_readiness.callable_pointer);
        runtime_readiness.callable_pointer =
            tla_trust_cg::PetriNativeSuccessorCallablePointer::from_usize(1);

        let shared_planning_identity = PetriNativeSharedPlanningFingerprintIdentity::for_net(&net);
        let packet = PetriNativeSuccessorBatchReadinessPacket::callable_artifact(
            &net,
            &cache,
            &bundle,
            &installed_artifact,
            &shared_planning_identity,
            usize_to_u32_saturating(layout.state_len()),
            usize_to_u32_saturating(cache.transition_count),
            &handoff,
            &runtime_readiness,
        );

        assert_eq!(
            packet.status_code,
            PETRI_NATIVE_CANDIDATE_STATUS_CALLABLE_ARTIFACT
        );
        assert_eq!(
            packet.reason_code,
            PETRI_NATIVE_CANDIDATE_REASON_CALLABLE_POINTER_MISMATCH
        );
        assert!(packet.blocker.contains("callable pointer"));
        assert_eq!(
            packet.validation_receipt_status,
            PETRI_NATIVE_CANDIDATE_CALLABLE_RECEIPT_STATUS_ACCEPTED
        );
        assert!(!packet.production_selected);
        assert!(packet.fail_closed);
    }
}
