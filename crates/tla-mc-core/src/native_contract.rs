// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral native checker artifact contracts.
//!
//! These records describe the shared native boundary as data: candidate kind,
//! ABI shape, storage/vector layout identity, compiler and bundle identities,
//! evidence policy, and install admission. The core crate intentionally keeps
//! these fields string/checksum based so it does not depend on native producer
//! crates such as `tla-jit-abi`, `tla-ir`, or `tla-trust_cg`.

use std::fmt;

use thiserror::Error;

use crate::{
    fingerprint_identity::{
        SharedNativeCacheReusePolicy, SharedNativePlanningIdentity,
        SHARED_NATIVE_CACHE_REUSE_DISABLED, SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY,
        SHARED_NATIVE_CACHE_REUSE_FRONTEND_REUSABLE,
    },
    prepared_program::{PreparedProgramPayloadKind, PreparedStorageKind},
    setup_trace::{CheckerSourceKind, SetupTraceLaneKind},
    shared_engine_adoption::SharedEngineFrontendFamily,
    validation_receipt::{
        ValidationReceipt, ValidationReceiptArtifactKind, ValidationReceiptStatus,
        ValidationReceiptValidatorKind,
    },
};

/// Stable row kind for shared native contract evidence.
pub const SHARED_NATIVE_CONTRACT_ROW_KIND: &str = "shared_native_contract";

/// Stable schema label for shared native contract evidence.
pub const SHARED_NATIVE_CONTRACT_SCHEMA: &str = "ty.shared.native_contract.v1";

/// Stable schema version for shared native contract evidence.
pub const SHARED_NATIVE_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Fields every shared native contract evidence row publishes.
pub const SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "source_kind",
    "frontend_kind",
    "payload_kind",
    "storage_kind",
    "contract_kind",
    "lane_kind",
    "compatible_frontend_families",
    "abi",
    "symbol",
    "abi_params",
    "abi_returns",
    "abi_variadic",
    "layout_kind",
    "layout_identity",
    "prepared_program_identity",
    "candidate_identity",
    "lane_identity",
    "native_planning_identity",
    "source_fingerprint",
    "frontend_payload_identity",
    "plan_reuse_manifest_id",
    "plan_reuse_manifest_digest",
    "trust_ir_identity",
    "trust_ir_module_digest",
    "compiler_facts_digest",
    "native_bundle_digest",
    "transport_identity",
    "semantic_digest",
    "link_digest",
    "cache_digest",
    "fingerprint_domain_identity",
    "cas_identity",
    "cache_identity",
    "cache_namespace_identity",
    "cache_reuse_policy",
    "frontend_family_scope_identity",
    "storage_layout_fingerprint",
    "artifact_identity",
    "target_abi_identity",
    "required_evidence",
    "required_validators",
    "required_artifacts",
    "install_authority",
    "evidence_fail_closed",
    "admission_status",
    "admission_disposition",
    "admission_authority",
    "admission_reason",
    "admission_fail_closed",
    "production_selected",
];

/// Current frontend families covered by the shared native contract.
pub const SHARED_NATIVE_CONTRACT_FRONTEND_FAMILIES: [SharedEngineFrontendFamily; 8] = [
    SharedEngineFrontendFamily::TlaPlus,
    SharedEngineFrontendFamily::Quint,
    SharedEngineFrontendFamily::MccPetri,
    SharedEngineFrontendFamily::Aiger,
    SharedEngineFrontendFamily::Btor2,
    SharedEngineFrontendFamily::VmtTransitionSystem,
    SharedEngineFrontendFamily::AYAnalytical,
    SharedEngineFrontendFamily::WitnessReplay,
];

/// Native artifact role at the shared checker boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeContractKind {
    /// Wire code `"successor_kernel"`.
    SuccessorKernel,
    /// Wire code `"predicate_kernel"`.
    PredicateKernel,
    /// Wire code `"analytical_kernel"`.
    AnalyticalKernel,
    /// Wire code `"ay_symbolic_kernel"`.
    AYSymbolicKernel,
    /// Wire code `"fingerprint_kernel"`.
    FingerprintKernel,
    /// Wire code `"replay_kernel"`.
    ReplayKernel,
    /// Wire code `"native_helper_kernel"`.
    NativeHelperKernel,
}

impl SharedNativeContractKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::SuccessorKernel => "successor_kernel",
            Self::PredicateKernel => "predicate_kernel",
            Self::AnalyticalKernel => "analytical_kernel",
            Self::AYSymbolicKernel => "ay_symbolic_kernel",
            Self::FingerprintKernel => "fingerprint_kernel",
            Self::ReplayKernel => "replay_kernel",
            Self::NativeHelperKernel => "native_helper_kernel",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "successor_kernel" => Some(Self::SuccessorKernel),
            "predicate_kernel" => Some(Self::PredicateKernel),
            "analytical_kernel" => Some(Self::AnalyticalKernel),
            "ay_symbolic_kernel" => Some(Self::AYSymbolicKernel),
            "fingerprint_kernel" => Some(Self::FingerprintKernel),
            "replay_kernel" => Some(Self::ReplayKernel),
            "native_helper_kernel" => Some(Self::NativeHelperKernel),
            _ => None,
        }
    }
}

impl fmt::Display for SharedNativeContractKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Scalar value kinds used to describe native ABI signatures without importing
/// a producer-specific ABI enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeAbiValueKind {
    /// Wire code `"void"`.
    Void,
    /// Wire code `"ptr"`.
    Ptr,
    /// Wire code `"bool"`.
    Bool,
    /// Wire code `"i8"`.
    I8,
    /// Wire code `"i16"`.
    I16,
    /// Wire code `"i32"`.
    I32,
    /// Wire code `"i64"`.
    I64,
    /// Wire code `"i128"`.
    I128,
    /// Wire code `"u8"`.
    U8,
    /// Wire code `"u16"`.
    U16,
    /// Wire code `"u32"`.
    U32,
    /// Wire code `"u64"`.
    U64,
    /// Wire code `"u128"`.
    U128,
    /// Wire code `"usize"`.
    Usize,
    /// Wire code `"f32"`.
    F32,
    /// Wire code `"f64"`.
    F64,
    /// Wire code `"bytes"`.
    Bytes,
    /// Wire code `"opaque"`.
    Opaque,
}

impl SharedNativeAbiValueKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Ptr => "ptr",
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::I128 => "i128",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
            Self::Usize => "usize",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bytes => "bytes",
            Self::Opaque => "opaque",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "void" => Some(Self::Void),
            "ptr" => Some(Self::Ptr),
            "bool" => Some(Self::Bool),
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "i128" => Some(Self::I128),
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "u128" => Some(Self::U128),
            "usize" => Some(Self::Usize),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            "bytes" => Some(Self::Bytes),
            "opaque" => Some(Self::Opaque),
            _ => None,
        }
    }
}

impl fmt::Display for SharedNativeAbiValueKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Storage/layout family at the native ABI boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeLayoutKind {
    /// Wire code `"tla_state_slots"`.
    TlaStateSlots,
    /// Wire code `"petri_marking"`.
    PetriMarking,
    /// Wire code `"hardware_registers"`.
    HardwareRegisters,
    /// Wire code `"smt_variables"`.
    SmtVariables,
    /// Wire code `"witness_steps"`.
    WitnessSteps,
    /// Wire code `"opaque"`.
    Opaque,
}

impl SharedNativeLayoutKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaStateSlots => "tla_state_slots",
            Self::PetriMarking => "petri_marking",
            Self::HardwareRegisters => "hardware_registers",
            Self::SmtVariables => "smt_variables",
            Self::WitnessSteps => "witness_steps",
            Self::Opaque => "opaque",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "tla_state_slots" => Some(Self::TlaStateSlots),
            "petri_marking" => Some(Self::PetriMarking),
            "hardware_registers" => Some(Self::HardwareRegisters),
            "smt_variables" => Some(Self::SmtVariables),
            "witness_steps" => Some(Self::WitnessSteps),
            "opaque" => Some(Self::Opaque),
            _ => None,
        }
    }
}

impl From<PreparedStorageKind> for SharedNativeLayoutKind {
    fn from(value: PreparedStorageKind) -> Self {
        match value {
            PreparedStorageKind::TlaStateSlots => Self::TlaStateSlots,
            PreparedStorageKind::PetriMarking => Self::PetriMarking,
            PreparedStorageKind::HardwareRegisters => Self::HardwareRegisters,
            PreparedStorageKind::SmtVariables => Self::SmtVariables,
            PreparedStorageKind::WitnessSteps => Self::WitnessSteps,
            PreparedStorageKind::Unknown => Self::Opaque,
        }
    }
}

impl fmt::Display for SharedNativeLayoutKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Evidence class a native install gate may require before publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeEvidenceKind {
    /// Wire code `"manifest_metadata"`.
    ManifestMetadata,
    /// Wire code `"target_abi_checksum"`.
    TargetAbiChecksum,
    /// Wire code `"layout_checksum"`.
    LayoutChecksum,
    /// Wire code `"proof_policy_checksum"`.
    ProofPolicyChecksum,
    /// Wire code `"semantic_checksum"`.
    SemanticChecksum,
    /// Wire code `"validation_receipt"`.
    ValidationReceipt,
    /// Wire code `"replay_identity"`.
    ReplayIdentity,
    /// Wire code `"native_payload_hash"`.
    NativePayloadHash,
    /// Wire code `"telemetry"`.
    Telemetry,
}

impl SharedNativeEvidenceKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::ManifestMetadata => "manifest_metadata",
            Self::TargetAbiChecksum => "target_abi_checksum",
            Self::LayoutChecksum => "layout_checksum",
            Self::ProofPolicyChecksum => "proof_policy_checksum",
            Self::SemanticChecksum => "semantic_checksum",
            Self::ValidationReceipt => "validation_receipt",
            Self::ReplayIdentity => "replay_identity",
            Self::NativePayloadHash => "native_payload_hash",
            Self::Telemetry => "telemetry",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "manifest_metadata" => Some(Self::ManifestMetadata),
            "target_abi_checksum" => Some(Self::TargetAbiChecksum),
            "layout_checksum" => Some(Self::LayoutChecksum),
            "proof_policy_checksum" => Some(Self::ProofPolicyChecksum),
            "semantic_checksum" => Some(Self::SemanticChecksum),
            "validation_receipt" => Some(Self::ValidationReceipt),
            "replay_identity" => Some(Self::ReplayIdentity),
            "native_payload_hash" => Some(Self::NativePayloadHash),
            "telemetry" => Some(Self::Telemetry),
            _ => None,
        }
    }
}

impl fmt::Display for SharedNativeEvidenceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Install authority granted after evidence validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SharedNativeInstallAuthority {
    /// No install authority granted (wire code `"none"`); the default.
    #[default]
    None,
    /// Wire code `"shadow_only"`.
    ShadowOnly,
    /// Wire code `"canary_callable"`.
    CanaryCallable,
    /// Wire code `"active_callable"`.
    ActiveCallable,
    /// Wire code `"validation_only"`.
    ValidationOnly,
}

impl SharedNativeInstallAuthority {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ShadowOnly => "shadow_only",
            Self::CanaryCallable => "canary_callable",
            Self::ActiveCallable => "active_callable",
            Self::ValidationOnly => "validation_only",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "none" => Some(Self::None),
            "shadow_only" => Some(Self::ShadowOnly),
            "canary_callable" => Some(Self::CanaryCallable),
            "active_callable" => Some(Self::ActiveCallable),
            "validation_only" => Some(Self::ValidationOnly),
            _ => None,
        }
    }

    /// Whether this authority permits calling the native artifact
    /// (`CanaryCallable` or `ActiveCallable`).
    pub fn is_callable(self) -> bool {
        matches!(self, Self::CanaryCallable | Self::ActiveCallable)
    }
}

impl fmt::Display for SharedNativeInstallAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Native install disposition after the fail-closed admission gate runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeAdmissionDisposition {
    /// Wire code `"installable"`.
    Installable,
    /// Wire code `"profile_only"`.
    ProfileOnly,
    /// Wire code `"replay_only"`.
    ReplayOnly,
    /// Wire code `"shadow_only"`.
    ShadowOnly,
    /// Wire code `"rejected"`.
    Rejected,
}

impl SharedNativeAdmissionDisposition {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Installable => "installable",
            Self::ProfileOnly => "profile_only",
            Self::ReplayOnly => "replay_only",
            Self::ShadowOnly => "shadow_only",
            Self::Rejected => "rejected",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "installable" => Some(Self::Installable),
            "profile_only" => Some(Self::ProfileOnly),
            "replay_only" => Some(Self::ReplayOnly),
            "shadow_only" => Some(Self::ShadowOnly),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

impl fmt::Display for SharedNativeAdmissionDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Native admission status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeAdmissionStatus {
    /// Wire code `"accepted"`.
    Accepted,
    /// Wire code `"rejected"`.
    Rejected,
}

impl SharedNativeAdmissionStatus {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }
}

impl fmt::Display for SharedNativeAdmissionStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Stable reason code for native admission decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeAdmissionReason {
    /// Wire code `"accepted_evidence"`.
    AcceptedEvidence,
    /// Wire code `"missing_evidence"`.
    MissingEvidence,
    /// Wire code `"rejected_evidence"`.
    RejectedEvidence,
    /// Wire code `"abi_mismatch"`.
    AbiMismatch,
    /// Wire code `"layout_mismatch"`.
    LayoutMismatch,
    /// Wire code `"proof_missing"`.
    ProofMissing,
    /// Wire code `"proof_rejected"`.
    ProofRejected,
    /// Wire code `"replay_missing"`.
    ReplayMissing,
    /// Wire code `"payload_digest_mismatch"`.
    PayloadDigestMismatch,
    /// Wire code `"freshness_rejected"`.
    FreshnessRejected,
    /// Wire code `"runtime_blocked"`.
    RuntimeBlocked,
    /// Wire code `"unsupported"`.
    Unsupported,
    /// Wire code `"policy_rejected"`.
    PolicyRejected,
}

impl SharedNativeAdmissionReason {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::AcceptedEvidence => "accepted_evidence",
            Self::MissingEvidence => "missing_evidence",
            Self::RejectedEvidence => "rejected_evidence",
            Self::AbiMismatch => "abi_mismatch",
            Self::LayoutMismatch => "layout_mismatch",
            Self::ProofMissing => "proof_missing",
            Self::ProofRejected => "proof_rejected",
            Self::ReplayMissing => "replay_missing",
            Self::PayloadDigestMismatch => "payload_digest_mismatch",
            Self::FreshnessRejected => "freshness_rejected",
            Self::RuntimeBlocked => "runtime_blocked",
            Self::Unsupported => "unsupported",
            Self::PolicyRejected => "policy_rejected",
        }
    }

    /// Parse a value from its wire [`code`](Self::code), or `None` if unrecognized.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "accepted_evidence" => Some(Self::AcceptedEvidence),
            "missing_evidence" => Some(Self::MissingEvidence),
            "rejected_evidence" => Some(Self::RejectedEvidence),
            "abi_mismatch" => Some(Self::AbiMismatch),
            "layout_mismatch" => Some(Self::LayoutMismatch),
            "proof_missing" => Some(Self::ProofMissing),
            "proof_rejected" => Some(Self::ProofRejected),
            "replay_missing" => Some(Self::ReplayMissing),
            "payload_digest_mismatch" => Some(Self::PayloadDigestMismatch),
            "freshness_rejected" => Some(Self::FreshnessRejected),
            "runtime_blocked" => Some(Self::RuntimeBlocked),
            "unsupported" => Some(Self::Unsupported),
            "policy_rejected" => Some(Self::PolicyRejected),
            _ => None,
        }
    }
}

impl fmt::Display for SharedNativeAdmissionReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// One ordered native ABI parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeAbiParam {
    /// Parameter name.
    pub name: String,
    /// Parameter scalar value kind.
    pub value_kind: SharedNativeAbiValueKind,
}

impl SharedNativeAbiParam {
    /// Create a named ABI parameter of the given value kind.
    pub fn new(name: impl Into<String>, value_kind: SharedNativeAbiValueKind) -> Self {
        Self {
            name: name.into(),
            value_kind,
        }
    }

    fn render(&self) -> String {
        format!("{}:{}", evidence_value(&self.name), self.value_kind.code())
    }
}

/// Frontend-neutral native symbol signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeAbiSignature {
    /// ABI name (calling convention).
    pub abi: String,
    /// Exported symbol name.
    pub symbol: String,
    /// Ordered parameters.
    pub params: Vec<SharedNativeAbiParam>,
    /// Return value kinds.
    pub returns: Vec<SharedNativeAbiValueKind>,
    /// Whether the signature is variadic.
    pub variadic: bool,
}

impl SharedNativeAbiSignature {
    /// Create a signature for `abi`/`symbol` with no params or returns.
    pub fn new(abi: impl Into<String>, symbol: impl Into<String>) -> Self {
        Self {
            abi: abi.into(),
            symbol: symbol.into(),
            params: Vec::new(),
            returns: Vec::new(),
            variadic: false,
        }
    }

    /// Append a named parameter.
    pub fn with_param(mut self, name: impl Into<String>, kind: SharedNativeAbiValueKind) -> Self {
        self.params.push(SharedNativeAbiParam::new(name, kind));
        self
    }

    /// Append a return value kind.
    pub fn with_return(mut self, kind: SharedNativeAbiValueKind) -> Self {
        self.returns.push(kind);
        self
    }

    /// Set the variadic flag.
    pub fn with_variadic(mut self, variadic: bool) -> Self {
        self.variadic = variadic;
        self
    }

    /// Render the parameters as a stable evidence string (`name:kind` joined).
    pub fn render_params(&self) -> String {
        join_rendered(self.params.iter().map(SharedNativeAbiParam::render))
    }

    /// Render the return kinds as a stable evidence string.
    pub fn render_returns(&self) -> String {
        join_codes(self.returns.iter().map(|value| value.code()))
    }
}

/// Vector layout facts that are part of native artifact identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeVectorContract {
    /// Vector-contract identity.
    pub identity: String,
    /// Element scalar value kind.
    pub value_kind: SharedNativeAbiValueKind,
    /// Logical lane count.
    pub logical_lanes: u32,
    /// Physical lane count.
    pub physical_lanes: u32,
    /// Bits per element.
    pub element_bits: u32,
    /// Bits per physical lane.
    pub lane_bits: u32,
    /// Identity of the mask layout, when present.
    pub mask_identity: Option<String>,
    /// Identity of the operations set, when present.
    pub operations_identity: Option<String>,
    /// CPU feature guards required to use this vector contract.
    pub feature_guards: Vec<String>,
    /// Whether unavailability of the contract fails closed (default `true`).
    pub fail_closed_unavailable: bool,
}

impl SharedNativeVectorContract {
    /// Create a vector contract from its required lane/bit-width facts; the
    /// optional identities/guards start empty and `fail_closed_unavailable` is
    /// `true`.
    pub fn new(
        identity: impl Into<String>,
        value_kind: SharedNativeAbiValueKind,
        logical_lanes: u32,
        physical_lanes: u32,
        element_bits: u32,
        lane_bits: u32,
    ) -> Self {
        Self {
            identity: identity.into(),
            value_kind,
            logical_lanes,
            physical_lanes,
            element_bits,
            lane_bits,
            mask_identity: None,
            operations_identity: None,
            feature_guards: Vec::new(),
            fail_closed_unavailable: true,
        }
    }

    /// Set [`mask_identity`](Self::mask_identity) (empty clears it).
    pub fn with_mask_identity(mut self, identity: impl Into<String>) -> Self {
        self.mask_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`operations_identity`](Self::operations_identity) (empty clears it).
    pub fn with_operations_identity(mut self, identity: impl Into<String>) -> Self {
        self.operations_identity = non_empty_string(identity.into());
        self
    }

    /// Append a CPU feature guard (empty input is ignored).
    pub fn with_feature_guard(mut self, guard: impl Into<String>) -> Self {
        push_non_empty(&mut self.feature_guards, guard.into());
        self
    }

    /// Set [`fail_closed_unavailable`](Self::fail_closed_unavailable).
    pub fn with_fail_closed_unavailable(mut self, fail_closed: bool) -> Self {
        self.fail_closed_unavailable = fail_closed;
        self
    }
}

/// Native storage/layout contract, including optional vector facts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeLayoutContract {
    /// Layout family.
    pub kind: SharedNativeLayoutKind,
    /// Layout identity.
    pub identity: String,
    /// Layout fingerprint, when known.
    pub fingerprint: Option<String>,
    /// State length in slots, when fixed.
    pub state_len: Option<u32>,
    /// Vector contracts embedded in the layout.
    pub vector_contracts: Vec<SharedNativeVectorContract>,
}

impl SharedNativeLayoutContract {
    /// Create a layout contract of `kind` with the given identity.
    pub fn new(kind: SharedNativeLayoutKind, identity: impl Into<String>) -> Self {
        Self {
            kind,
            identity: identity.into(),
            fingerprint: None,
            state_len: None,
            vector_contracts: Vec::new(),
        }
    }

    /// Create a layout contract whose kind is mapped from a prepared storage kind.
    pub fn from_storage_kind(
        storage_kind: PreparedStorageKind,
        identity: impl Into<String>,
    ) -> Self {
        Self::new(storage_kind.into(), identity)
    }

    /// Set [`fingerprint`](Self::fingerprint) (empty clears it).
    pub fn with_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`state_len`](Self::state_len).
    pub fn with_state_len(mut self, state_len: u32) -> Self {
        self.state_len = Some(state_len);
        self
    }

    /// Append a vector contract.
    pub fn with_vector_contract(mut self, vector: SharedNativeVectorContract) -> Self {
        self.vector_contracts.push(vector);
        self
    }
}

/// Stable identities and digests that bind a native artifact to prepared
/// program, trust-ir bundle, compiler facts, target ABI, and cache lineage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeContractIdentity {
    /// Identity of the prepared program.
    pub prepared_program_identity: String,
    /// Identity of the candidate lane.
    pub candidate_identity: String,
    /// Identity of the execution lane.
    pub lane_identity: String,
    /// Digest of the original source, when known.
    pub source_fingerprint: Option<String>,
    /// Identity of the frontend-produced payload, when known.
    pub frontend_payload_identity: Option<String>,
    /// Plan-reuse manifest id, when reusing a prior plan.
    pub plan_reuse_manifest_id: Option<String>,
    /// Plan-reuse manifest digest, when reusing a prior plan.
    pub plan_reuse_manifest_digest: Option<String>,
    /// trust-ir identity, when known.
    pub trust_ir_identity: Option<String>,
    /// trust-ir module digest, when known.
    pub trust_ir_module_digest: Option<String>,
    /// Compiler-facts digest, when known.
    pub compiler_facts_digest: Option<String>,
    /// Native bundle digest, when known.
    pub native_bundle_digest: Option<String>,
    /// Transport identity for the native artifact, when known.
    pub transport_identity: Option<String>,
    /// Semantic digest binding the payload, when known.
    pub semantic_digest: Option<String>,
    /// Link digest of the native artifact, when known.
    pub link_digest: Option<String>,
    /// Cache digest, when the artifact is cached.
    pub cache_digest: Option<String>,
    /// Fingerprint-domain identity, when known.
    pub fingerprint_domain_identity: Option<String>,
    /// Content-addressed-store identity, when known.
    pub cas_identity: Option<String>,
    /// Cache identity, when known.
    pub cache_identity: Option<String>,
    /// Cache namespace identity, when known.
    pub cache_namespace_identity: Option<String>,
    /// Cache reuse policy code (see the `SHARED_NATIVE_CACHE_REUSE_*` constants).
    pub cache_reuse_policy: String,
    /// Identity of the produced artifact, when known.
    pub artifact_identity: Option<String>,
    /// Digest of the produced artifact, when known.
    pub artifact_fingerprint: Option<String>,
    /// Target-ABI identity, when known.
    pub target_abi_identity: Option<String>,
    /// Storage-layout fingerprint, when known.
    pub storage_layout_fingerprint: Option<String>,
    /// Proof-policy identity, when known.
    pub proof_policy_identity: Option<String>,
    /// Replay identity, when known.
    pub replay_identity: Option<String>,
}

impl SharedNativeContractIdentity {
    /// Create an identity from the three required identities; all optional
    /// digests/identities start empty and the cache reuse policy defaults to
    /// frontend-local-only.
    pub fn new(
        prepared_program_identity: impl Into<String>,
        candidate_identity: impl Into<String>,
        lane_identity: impl Into<String>,
    ) -> Self {
        Self {
            prepared_program_identity: prepared_program_identity.into(),
            candidate_identity: candidate_identity.into(),
            lane_identity: lane_identity.into(),
            source_fingerprint: None,
            frontend_payload_identity: None,
            plan_reuse_manifest_id: None,
            plan_reuse_manifest_digest: None,
            trust_ir_identity: None,
            trust_ir_module_digest: None,
            compiler_facts_digest: None,
            native_bundle_digest: None,
            transport_identity: None,
            semantic_digest: None,
            link_digest: None,
            cache_digest: None,
            fingerprint_domain_identity: None,
            cas_identity: None,
            cache_identity: None,
            cache_namespace_identity: None,
            cache_reuse_policy: SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY.to_string(),
            artifact_identity: None,
            artifact_fingerprint: None,
            target_abi_identity: None,
            storage_layout_fingerprint: None,
            proof_policy_identity: None,
            replay_identity: None,
        }
    }

    /// Set [`source_fingerprint`](Self::source_fingerprint) (empty clears it to `None`).
    pub fn with_source_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.source_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`frontend_payload_identity`](Self::frontend_payload_identity) (empty clears it to `None`).
    pub fn with_frontend_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.frontend_payload_identity = non_empty_string(identity.into());
        self
    }

    /// Set the plan-reuse manifest id and digest (empty values clear them).
    pub fn with_plan_reuse_manifest(
        mut self,
        manifest_id: impl Into<String>,
        manifest_digest: impl Into<String>,
    ) -> Self {
        self.plan_reuse_manifest_id = non_empty_string(manifest_id.into());
        self.plan_reuse_manifest_digest = non_empty_string(manifest_digest.into());
        self
    }

    /// Set [`trust_ir_identity`](Self::trust_ir_identity) (empty clears it to `None`).
    pub fn with_trust_ir_identity(mut self, identity: impl Into<String>) -> Self {
        self.trust_ir_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`trust_ir_module_digest`](Self::trust_ir_module_digest) (empty clears it to `None`).
    pub fn with_trust_ir_module_digest(mut self, digest: impl Into<String>) -> Self {
        self.trust_ir_module_digest = non_empty_string(digest.into());
        self
    }

    /// Set [`compiler_facts_digest`](Self::compiler_facts_digest) (empty clears it to `None`).
    pub fn with_compiler_facts_digest(mut self, digest: impl Into<String>) -> Self {
        self.compiler_facts_digest = non_empty_string(digest.into());
        self
    }

    /// Set [`native_bundle_digest`](Self::native_bundle_digest) (empty clears it to `None`).
    pub fn with_native_bundle_digest(mut self, digest: impl Into<String>) -> Self {
        self.native_bundle_digest = non_empty_string(digest.into());
        self
    }

    /// Set [`transport_identity`](Self::transport_identity) (empty clears it to `None`).
    pub fn with_transport_identity(mut self, identity: impl Into<String>) -> Self {
        self.transport_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`semantic_digest`](Self::semantic_digest) (empty clears it to `None`).
    pub fn with_semantic_digest(mut self, digest: impl Into<String>) -> Self {
        self.semantic_digest = non_empty_string(digest.into());
        self
    }

    /// Set [`link_digest`](Self::link_digest) (empty clears it to `None`).
    pub fn with_link_digest(mut self, digest: impl Into<String>) -> Self {
        self.link_digest = non_empty_string(digest.into());
        self
    }

    /// Set [`cache_digest`](Self::cache_digest) (empty clears it to `None`).
    pub fn with_cache_digest(mut self, digest: impl Into<String>) -> Self {
        self.cache_digest = non_empty_string(digest.into());
        self
    }

    /// Set [`fingerprint_domain_identity`](Self::fingerprint_domain_identity) (empty clears it to `None`).
    pub fn with_fingerprint_domain_identity(mut self, identity: impl Into<String>) -> Self {
        self.fingerprint_domain_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`cas_identity`](Self::cas_identity) (empty clears it to `None`).
    pub fn with_cas_identity(mut self, identity: impl Into<String>) -> Self {
        self.cas_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`cache_identity`](Self::cache_identity) (empty clears it to `None`).
    pub fn with_cache_identity(mut self, identity: impl Into<String>) -> Self {
        self.cache_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`cache_namespace_identity`](Self::cache_namespace_identity) (empty clears it to `None`).
    pub fn with_cache_namespace_identity(mut self, identity: impl Into<String>) -> Self {
        self.cache_namespace_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`cache_reuse_policy`](Self::cache_reuse_policy).
    pub fn with_cache_reuse_policy(mut self, policy: impl Into<String>) -> Self {
        self.cache_reuse_policy = policy.into();
        self
    }

    /// Set [`cache_reuse_policy`](Self::cache_reuse_policy) from a typed policy.
    pub fn with_cache_reuse_policy_kind(mut self, policy: SharedNativeCacheReusePolicy) -> Self {
        self.cache_reuse_policy = policy.code().to_string();
        self
    }

    /// Parse the [`cache_reuse_policy`](Self::cache_reuse_policy) code into a
    /// typed policy, or `None` if unrecognized.
    pub fn cache_reuse_policy_kind(&self) -> Option<SharedNativeCacheReusePolicy> {
        SharedNativeCacheReusePolicy::from_code(&self.cache_reuse_policy)
    }

    /// Declare a frontend-reusable cache domain: set the fingerprint-domain and
    /// cache-namespace identities and mark the cache reuse policy reusable.
    pub fn with_frontend_reusable_cache_domain(
        mut self,
        fingerprint_domain_identity: impl Into<String>,
        cache_namespace_identity: impl Into<String>,
    ) -> Self {
        self.fingerprint_domain_identity = non_empty_string(fingerprint_domain_identity.into());
        self.cache_namespace_identity = non_empty_string(cache_namespace_identity.into());
        self.cache_reuse_policy = SharedNativeCacheReusePolicy::FrontendReusable
            .code()
            .to_string();
        self
    }

    /// Whether the cache reuse policy is the frontend-reusable kind.
    pub fn is_frontend_reusable_cache_domain(&self) -> bool {
        self.cache_reuse_policy_kind()
            .is_some_and(SharedNativeCacheReusePolicy::frontend_reusable)
    }

    /// Set [`artifact_identity`](Self::artifact_identity) (empty clears it to `None`).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.artifact_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`artifact_fingerprint`](Self::artifact_fingerprint) (empty clears it to `None`).
    pub fn with_artifact_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.artifact_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`target_abi_identity`](Self::target_abi_identity) (empty clears it to `None`).
    pub fn with_target_abi_identity(mut self, identity: impl Into<String>) -> Self {
        self.target_abi_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`storage_layout_fingerprint`](Self::storage_layout_fingerprint) (empty clears it to `None`).
    pub fn with_storage_layout_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.storage_layout_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`proof_policy_identity`](Self::proof_policy_identity) (empty clears it to `None`).
    pub fn with_proof_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.proof_policy_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`replay_identity`](Self::replay_identity) (empty clears it to `None`).
    pub fn with_replay_identity(mut self, identity: impl Into<String>) -> Self {
        self.replay_identity = non_empty_string(identity.into());
        self
    }
}

/// One fail-closed evidence requirement for native admission.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeEvidenceRequirement {
    /// Evidence class required.
    pub kind: SharedNativeEvidenceKind,
    /// Identity the evidence must carry.
    pub identity: String,
    /// Whether the evidence is required.
    pub required: bool,
    /// Whether absence fails closed.
    pub fail_closed: bool,
}

impl SharedNativeEvidenceRequirement {
    /// Create a required, fail-closed evidence requirement.
    pub fn fail_closed(kind: SharedNativeEvidenceKind, identity: impl Into<String>) -> Self {
        Self {
            kind,
            identity: identity.into(),
            required: true,
            fail_closed: true,
        }
    }
}

/// Evidence policy that must be satisfied before a native artifact is
/// installable or callable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeEvidencePolicy {
    /// Install authority granted once the policy is satisfied.
    pub install_authority: SharedNativeInstallAuthority,
    /// Whether the policy fails closed.
    pub fail_closed: bool,
    /// Evidence requirements that must be met.
    pub required_evidence: Vec<SharedNativeEvidenceRequirement>,
    /// Validator kinds whose accepted receipts are required.
    pub required_validator_kinds: Vec<ValidationReceiptValidatorKind>,
    /// Artifact kinds whose accepted receipts are required.
    pub required_artifact_kinds: Vec<ValidationReceiptArtifactKind>,
    /// Manifest-metadata keys that must be present.
    pub required_manifest_metadata: Vec<String>,
}

impl SharedNativeEvidencePolicy {
    /// Create a fail-closed policy granting `install_authority`, with no
    /// requirements.
    pub fn fail_closed(install_authority: SharedNativeInstallAuthority) -> Self {
        Self {
            install_authority,
            fail_closed: true,
            required_evidence: Vec::new(),
            required_validator_kinds: Vec::new(),
            required_artifact_kinds: Vec::new(),
            required_manifest_metadata: Vec::new(),
        }
    }

    /// Add an evidence requirement.
    pub fn with_required_evidence(mut self, requirement: SharedNativeEvidenceRequirement) -> Self {
        self.required_evidence.push(requirement);
        self
    }

    /// Require an accepted receipt from `kind` (de-duplicated).
    pub fn with_required_validator(mut self, kind: ValidationReceiptValidatorKind) -> Self {
        if !self.required_validator_kinds.contains(&kind) {
            self.required_validator_kinds.push(kind);
        }
        self
    }

    /// Require an accepted receipt covering artifact `kind` (de-duplicated).
    pub fn with_required_artifact(mut self, kind: ValidationReceiptArtifactKind) -> Self {
        if !self.required_artifact_kinds.contains(&kind) {
            self.required_artifact_kinds.push(kind);
        }
        self
    }

    /// Require a manifest-metadata key (empty input is ignored).
    pub fn with_required_manifest_metadata(mut self, key: impl Into<String>) -> Self {
        push_non_empty(&mut self.required_manifest_metadata, key.into());
        self
    }

    /// Wire codes of the required evidence kinds.
    pub fn required_evidence_codes(&self) -> Vec<&'static str> {
        self.required_evidence
            .iter()
            .map(|requirement| requirement.kind.code())
            .collect()
    }
}

/// Result of a fail-closed native admission decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeAdmission {
    /// Accept/reject status.
    pub status: SharedNativeAdmissionStatus,
    /// Resulting install disposition.
    pub disposition: SharedNativeAdmissionDisposition,
    /// Install authority granted.
    pub authority: SharedNativeInstallAuthority,
    /// Reason code for the admission decision.
    pub reason: SharedNativeAdmissionReason,
    /// Failure reason; required for a rejected admission, absent when accepted.
    pub failure_reason: Option<String>,
    /// Whether the admission was performed fail-closed.
    pub fail_closed: bool,
    /// Whether the artifact was selected for production.
    pub production_selected: bool,
    /// Accepted validation receipts.
    pub accepted_validation_receipts: Vec<ValidationReceipt>,
    /// Rejected validation receipts.
    pub rejected_validation_receipts: Vec<ValidationReceipt>,
    /// Hash of the install packet, when present.
    pub install_packet_hash: Option<String>,
    /// Replay identity, when present.
    pub replay_identity: Option<String>,
    /// SHA-256 of the native payload, when present.
    pub native_payload_sha256: Option<String>,
    /// Target checksum, when present.
    pub target_checksum: Option<String>,
    /// ABI checksum, when present.
    pub abi_checksum: Option<String>,
    /// Layout checksum, when present.
    pub layout_checksum: Option<String>,
    /// Proof-policy checksum, when present.
    pub proof_policy_checksum: Option<String>,
    /// Semantic checksum, when present.
    pub semantic_checksum: Option<String>,
}

impl SharedNativeAdmission {
    /// Build an accepted, fail-closed admission granting `authority` (installable
    /// disposition, no receipts/checksums yet).
    pub fn accepted_fail_closed(authority: SharedNativeInstallAuthority) -> Self {
        Self {
            status: SharedNativeAdmissionStatus::Accepted,
            disposition: SharedNativeAdmissionDisposition::Installable,
            authority,
            reason: SharedNativeAdmissionReason::AcceptedEvidence,
            failure_reason: None,
            fail_closed: true,
            production_selected: false,
            accepted_validation_receipts: Vec::new(),
            rejected_validation_receipts: Vec::new(),
            install_packet_hash: None,
            replay_identity: None,
            native_payload_sha256: None,
            target_checksum: None,
            abi_checksum: None,
            layout_checksum: None,
            proof_policy_checksum: None,
            semantic_checksum: None,
        }
    }

    /// Build a rejected, fail-closed admission with a reason and failure detail
    /// (no install authority granted).
    pub fn rejected_fail_closed(
        reason: SharedNativeAdmissionReason,
        failure_reason: impl Into<String>,
    ) -> Self {
        Self {
            status: SharedNativeAdmissionStatus::Rejected,
            disposition: SharedNativeAdmissionDisposition::Rejected,
            authority: SharedNativeInstallAuthority::None,
            reason,
            failure_reason: non_empty_string(failure_reason.into()),
            fail_closed: true,
            production_selected: false,
            accepted_validation_receipts: Vec::new(),
            rejected_validation_receipts: Vec::new(),
            install_packet_hash: None,
            replay_identity: None,
            native_payload_sha256: None,
            target_checksum: None,
            abi_checksum: None,
            layout_checksum: None,
            proof_policy_checksum: None,
            semantic_checksum: None,
        }
    }

    /// Set [`disposition`](Self::disposition).
    pub fn with_disposition(mut self, disposition: SharedNativeAdmissionDisposition) -> Self {
        self.disposition = disposition;
        self
    }

    /// Set [`production_selected`](Self::production_selected).
    pub fn with_production_selected(mut self, production_selected: bool) -> Self {
        self.production_selected = production_selected;
        self
    }

    /// Record a validation receipt, routing it to the accepted or rejected list
    /// by its status.
    pub fn with_validation_receipt(mut self, receipt: ValidationReceipt) -> Self {
        match receipt.status {
            ValidationReceiptStatus::Accepted => self.accepted_validation_receipts.push(receipt),
            ValidationReceiptStatus::Rejected => self.rejected_validation_receipts.push(receipt),
        }
        self
    }

    /// Set [`install_packet_hash`](Self::install_packet_hash) (empty clears it to `None`).
    pub fn with_install_packet_hash(mut self, hash: impl Into<String>) -> Self {
        self.install_packet_hash = non_empty_string(hash.into());
        self
    }

    /// Set [`replay_identity`](Self::replay_identity) (empty clears it to `None`).
    pub fn with_replay_identity(mut self, identity: impl Into<String>) -> Self {
        self.replay_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`native_payload_sha256`](Self::native_payload_sha256) (empty clears it to `None`).
    pub fn with_native_payload_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.native_payload_sha256 = non_empty_string(sha256.into());
        self
    }

    /// Set [`target_checksum`](Self::target_checksum) (empty clears it to `None`).
    pub fn with_target_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.target_checksum = non_empty_string(checksum.into());
        self
    }

    /// Set [`abi_checksum`](Self::abi_checksum) (empty clears it to `None`).
    pub fn with_abi_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.abi_checksum = non_empty_string(checksum.into());
        self
    }

    /// Set [`layout_checksum`](Self::layout_checksum) (empty clears it to `None`).
    pub fn with_layout_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.layout_checksum = non_empty_string(checksum.into());
        self
    }

    /// Set [`proof_policy_checksum`](Self::proof_policy_checksum) (empty clears it to `None`).
    pub fn with_proof_policy_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.proof_policy_checksum = non_empty_string(checksum.into());
        self
    }

    /// Set [`semantic_checksum`](Self::semantic_checksum) (empty clears it to `None`).
    pub fn with_semantic_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.semantic_checksum = non_empty_string(checksum.into());
        self
    }

    /// Whether the admission status is accepted.
    pub fn is_accepted(&self) -> bool {
        self.status == SharedNativeAdmissionStatus::Accepted
    }

    /// Whether the admission status is rejected.
    pub fn is_rejected(&self) -> bool {
        self.status == SharedNativeAdmissionStatus::Rejected
    }

    /// Whether the artifact may be published as a callable production lane.
    ///
    /// Requires acceptance, an installable disposition, callable authority,
    /// production selection, and no rejected receipts.
    pub fn can_publish_callable(&self) -> bool {
        self.is_accepted()
            && self.disposition == SharedNativeAdmissionDisposition::Installable
            && self.authority.is_callable()
            && self.production_selected
            && self.rejected_validation_receipts.is_empty()
    }
}

/// Complete data-only native contract shared by all frontend lanes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativeContract {
    /// Source/interchange family.
    pub source_kind: CheckerSourceKind,
    /// Prepared-program payload kind.
    pub payload_kind: PreparedProgramPayloadKind,
    /// Storage kind backing the native artifact.
    pub storage_kind: PreparedStorageKind,
    /// Native artifact role.
    pub contract_kind: SharedNativeContractKind,
    /// Execution lane (defaults to native).
    pub lane_kind: SetupTraceLaneKind,
    /// Frontend families this contract is compatible with.
    pub compatible_frontend_families: Vec<SharedEngineFrontendFamily>,
    /// ABI signature of the native symbol.
    pub abi: SharedNativeAbiSignature,
    /// Storage/layout contract.
    pub layout: SharedNativeLayoutContract,
    /// Identity/digest lineage of the artifact.
    pub identity: SharedNativeContractIdentity,
    /// Evidence policy gating installation.
    pub evidence_policy: SharedNativeEvidencePolicy,
    /// Admission decision for the artifact.
    pub admission: SharedNativeAdmission,
}

impl SharedNativeContract {
    /// Build a contract from its core parts, defaulting the lane to native, the
    /// compatible families to the current native set, and the layout to one
    /// derived from `storage_kind`.
    pub fn new(
        source_kind: CheckerSourceKind,
        payload_kind: PreparedProgramPayloadKind,
        storage_kind: PreparedStorageKind,
        contract_kind: SharedNativeContractKind,
        abi: SharedNativeAbiSignature,
        identity: SharedNativeContractIdentity,
    ) -> Self {
        Self {
            source_kind,
            payload_kind,
            storage_kind,
            contract_kind,
            lane_kind: SetupTraceLaneKind::Native,
            compatible_frontend_families: current_native_frontend_families().to_vec(),
            abi,
            layout: SharedNativeLayoutContract::from_storage_kind(
                storage_kind,
                storage_kind.code(),
            ),
            identity,
            evidence_policy: SharedNativeEvidencePolicy::fail_closed(
                SharedNativeInstallAuthority::ValidationOnly,
            ),
            admission: SharedNativeAdmission::rejected_fail_closed(
                SharedNativeAdmissionReason::MissingEvidence,
                "native admission evidence not supplied",
            ),
        }
    }

    /// Set [`lane_kind`](Self::lane_kind).
    pub fn with_lane_kind(mut self, lane_kind: SetupTraceLaneKind) -> Self {
        self.lane_kind = lane_kind;
        self
    }

    /// Set [`compatible_frontend_families`](Self::compatible_frontend_families).
    pub fn with_compatible_frontend_families(
        mut self,
        families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
    ) -> Self {
        self.compatible_frontend_families = families.into_iter().collect();
        self
    }

    /// Set [`layout`](Self::layout).
    pub fn with_layout(mut self, layout: SharedNativeLayoutContract) -> Self {
        self.layout = layout;
        self
    }

    /// Set [`evidence_policy`](Self::evidence_policy).
    pub fn with_evidence_policy(mut self, policy: SharedNativeEvidencePolicy) -> Self {
        self.evidence_policy = policy;
        self
    }

    /// Set [`admission`](Self::admission).
    pub fn with_admission(mut self, admission: SharedNativeAdmission) -> Self {
        self.admission = admission;
        self
    }

    fn planning_identity(&self) -> SharedNativePlanningIdentity {
        let mut identity =
            SharedNativePlanningIdentity::new(self.compatible_frontend_families.iter().copied())
                .with_cache_reuse_policy(self.identity.cache_reuse_policy.clone());

        identity.source_fingerprint = self.identity.source_fingerprint.clone();
        identity.plan_reuse_manifest_id = self.identity.plan_reuse_manifest_id.clone();
        identity.plan_reuse_manifest_digest = self.identity.plan_reuse_manifest_digest.clone();
        identity.fingerprint_domain_identity = self.identity.fingerprint_domain_identity.clone();
        identity.cas_identity = self.identity.cas_identity.clone();
        identity.cache_identity = self
            .identity
            .cache_identity
            .clone()
            .or_else(|| self.identity.cache_namespace_identity.clone())
            .or_else(|| self.identity.cache_digest.clone());

        identity
    }

    /// Validate the full native contract against the fail-closed shared rules.
    ///
    /// # Errors
    ///
    /// Returns a [`SharedNativeContractValidationError`] describing the first
    /// violation: a missing required identity/ABI/layout field, an empty or
    /// source-incompatible frontend-family set, an invalid ABI/layout, an
    /// inconsistent cache-reuse or planning identity, an evidence policy that is
    /// not fail-closed, or an admission that is inconsistent with itself or with
    /// the evidence policy.
    pub fn validate(&self) -> Result<(), SharedNativeContractValidationError> {
        require_identity(
            "prepared_program_identity",
            &self.identity.prepared_program_identity,
        )?;
        require_identity("candidate_identity", &self.identity.candidate_identity)?;
        require_identity("lane_identity", &self.identity.lane_identity)?;
        require_identity("abi", &self.abi.abi)?;
        require_identity("symbol", &self.abi.symbol)?;
        require_identity("layout_identity", &self.layout.identity)?;

        if self.compatible_frontend_families.is_empty() {
            return Err(SharedNativeContractValidationError::EmptyCompatibleFrontendFamilies);
        }

        if let Some(source_family) = self.source_kind.adoption_frontend_family() {
            if !self.compatible_frontend_families.contains(&source_family) {
                return Err(
                    SharedNativeContractValidationError::MissingSourceFrontendFamily {
                        family: source_family.code(),
                    },
                );
            }
        }

        validate_abi(&self.abi)?;
        validate_layout(&self.layout)?;
        validate_cache_reuse_contract(self)?;
        validate_planning_identity(self)?;
        validate_evidence_policy(&self.evidence_policy)?;
        validate_admission(&self.admission)?;
        validate_admission_against_policy(&self.admission, &self.evidence_policy)?;

        Ok(())
    }

    /// Render the full shared native contract as one stable evidence row,
    /// prefixed by `scope`.
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let planning_identity = self.planning_identity();
        format!(
            "{} {} schema={} schema_version={} source_kind={} frontend_kind={} payload_kind={} storage_kind={} contract_kind={} lane_kind={} compatible_frontend_families={} abi={} symbol={} abi_params={} abi_returns={} abi_variadic={} layout_kind={} layout_identity={} prepared_program_identity={} candidate_identity={} lane_identity={} native_planning_identity={} source_fingerprint={} frontend_payload_identity={} plan_reuse_manifest_id={} plan_reuse_manifest_digest={} trust_ir_identity={} trust_ir_module_digest={} compiler_facts_digest={} native_bundle_digest={} transport_identity={} semantic_digest={} link_digest={} cache_digest={} fingerprint_domain_identity={} cas_identity={} cache_identity={} cache_namespace_identity={} cache_reuse_policy={} frontend_family_scope_identity={} storage_layout_fingerprint={} artifact_identity={} target_abi_identity={} required_evidence={} required_validators={} required_artifacts={} install_authority={} evidence_fail_closed={} admission_status={} admission_disposition={} admission_authority={} admission_reason={} admission_fail_closed={} production_selected={}",
            scope,
            SHARED_NATIVE_CONTRACT_ROW_KIND,
            SHARED_NATIVE_CONTRACT_SCHEMA,
            SHARED_NATIVE_CONTRACT_SCHEMA_VERSION,
            self.source_kind.code(),
            self.source_kind.frontend_family_code(),
            self.payload_kind.code(),
            self.storage_kind.code(),
            self.contract_kind.code(),
            self.lane_kind.code(),
            join_codes(self.compatible_frontend_families.iter().map(|family| family.code())),
            evidence_value(&self.abi.abi),
            evidence_value(&self.abi.symbol),
            self.abi.render_params(),
            self.abi.render_returns(),
            self.abi.variadic,
            self.layout.kind.code(),
            evidence_value(&self.layout.identity),
            evidence_value(&self.identity.prepared_program_identity),
            evidence_value(&self.identity.candidate_identity),
            evidence_value(&self.identity.lane_identity),
            evidence_value(&planning_identity.stable_identity()),
            evidence_optional(self.identity.source_fingerprint.as_deref()),
            evidence_optional(self.identity.frontend_payload_identity.as_deref()),
            evidence_optional(self.identity.plan_reuse_manifest_id.as_deref()),
            evidence_optional(self.identity.plan_reuse_manifest_digest.as_deref()),
            evidence_optional(self.identity.trust_ir_identity.as_deref()),
            evidence_optional(self.identity.trust_ir_module_digest.as_deref()),
            evidence_optional(self.identity.compiler_facts_digest.as_deref()),
            evidence_optional(self.identity.native_bundle_digest.as_deref()),
            evidence_optional(self.identity.transport_identity.as_deref()),
            evidence_optional(self.identity.semantic_digest.as_deref()),
            evidence_optional(self.identity.link_digest.as_deref()),
            evidence_optional(self.identity.cache_digest.as_deref()),
            evidence_optional(self.identity.fingerprint_domain_identity.as_deref()),
            evidence_optional(self.identity.cas_identity.as_deref()),
            evidence_optional(self.identity.cache_identity.as_deref()),
            evidence_optional(self.identity.cache_namespace_identity.as_deref()),
            evidence_value(&self.identity.cache_reuse_policy),
            evidence_value(&planning_identity.frontend_family_scope_identity()),
            evidence_optional(self.identity.storage_layout_fingerprint.as_deref()),
            evidence_optional(self.identity.artifact_identity.as_deref()),
            evidence_optional(self.identity.target_abi_identity.as_deref()),
            join_codes(self.evidence_policy.required_evidence_codes()),
            join_codes(
                self.evidence_policy
                    .required_validator_kinds
                    .iter()
                    .map(|kind| kind.code())
            ),
            join_codes(
                self.evidence_policy
                    .required_artifact_kinds
                    .iter()
                    .map(|kind| kind.code())
            ),
            self.evidence_policy.install_authority.code(),
            self.evidence_policy.fail_closed,
            self.admission.status.code(),
            self.admission.disposition.code(),
            self.admission.authority.code(),
            self.admission.reason.code(),
            self.admission.fail_closed,
            self.admission.production_selected
        )
    }

    /// Render the native planning identity as a stable evidence row, prefixed by
    /// `scope`.
    pub fn render_planning_identity_evidence_row(&self, scope: &str) -> String {
        self.planning_identity()
            .render_evidence_row(scope, self.source_kind)
    }
}

/// Validation failures for malformed shared native contracts.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SharedNativeContractValidationError {
    /// A required contract field was missing (field name carried).
    #[error("missing shared native contract field: {0}")]
    MissingField(&'static str),
    /// The contract declared no compatible frontend families.
    #[error("shared native contract has no compatible frontend families")]
    EmptyCompatibleFrontendFamilies,
    /// The source frontend family is not in the compatible set.
    #[error("source frontend family {family} is not compatible with native contract")]
    MissingSourceFrontendFamily {
        /// Source frontend family code.
        family: &'static str,
    },
    /// The cache reuse policy code was not recognized.
    #[error("invalid native cache reuse policy: {policy}")]
    InvalidCacheReusePolicy {
        /// Offending policy code.
        policy: String,
    },
    /// Frontend-reusable cache needs at least two compatible families.
    #[error("frontend-reusable native cache requires at least two compatible frontend families")]
    FrontendReusableCacheRequiresCompatibleFamilies,
    /// A frontend-reusable cache's layout fingerprint did not match.
    #[error(
        "frontend-reusable native cache layout fingerprint mismatch: identity={identity} layout={layout}"
    )]
    CacheReuseLayoutFingerprintMismatch {
        /// Declared cache identity.
        identity: String,
        /// Declared layout fingerprint.
        layout: String,
    },
    /// Cache reuse is disabled yet a cache digest was present.
    #[error("native cache reuse is disabled but cache_digest is present")]
    CacheReuseDisabledHasCacheDigest,
    /// An ABI parameter lacked a name (parameter index carried).
    #[error("ABI parameter {index} is missing a name")]
    MissingAbiParameterName {
        /// Index of the unnamed parameter.
        index: usize,
    },
    /// A vector contract was invalid (index and reason carried).
    #[error("vector contract {index} is invalid: {reason}")]
    InvalidVectorContract {
        /// Index of the invalid vector contract.
        index: usize,
        /// Why it is invalid.
        reason: &'static str,
    },
    /// The evidence policy was not fail-closed.
    #[error("shared native evidence policy must be fail closed")]
    EvidencePolicyNotFailClosed,
    /// A required evidence kind lacked an identity (kind carried).
    #[error("required native evidence {kind} is missing an identity")]
    RequiredEvidenceMissingIdentity {
        /// Evidence kind code.
        kind: &'static str,
    },
    /// A required evidence kind was not fail-closed (kind carried).
    #[error("required native evidence {kind} is not fail closed")]
    RequiredEvidenceNotFailClosed {
        /// Evidence kind code.
        kind: &'static str,
    },
    /// The admission was not fail-closed.
    #[error("native admission must be fail closed")]
    AdmissionNotFailClosed,
    /// An accepted admission carried a failure reason.
    #[error("accepted native admission cannot carry a failure reason")]
    AcceptedAdmissionHasFailureReason,
    /// A rejected admission lacked a failure reason.
    #[error("rejected native admission requires a failure reason")]
    RejectedAdmissionMissingFailureReason,
    /// A rejected admission was marked production-selected.
    #[error("rejected native admission cannot be production selected")]
    RejectedAdmissionProductionSelected,
    /// An accepted admission used the rejected disposition.
    #[error("accepted native admission cannot use rejected disposition")]
    AcceptedAdmissionRejectedDisposition,
    /// A production-selected admission lacked callable install authority.
    #[error("production selected native admission requires callable authority")]
    ProductionWithoutCallableAuthority,
    /// An accepted admission contained a rejected receipt.
    #[error("accepted native admission contains a rejected receipt")]
    AcceptedAdmissionContainsRejectedReceipt,
    /// An admission carried an internally invalid validation receipt.
    #[error("invalid validation receipt in native admission: {0}")]
    InvalidValidationReceipt(String),
    /// An accepted admission was missing a required validator receipt (kind carried).
    #[error("accepted native admission is missing required validator receipt: {kind}")]
    MissingRequiredValidatorReceipt {
        /// Required validator kind code.
        kind: &'static str,
    },
    /// An accepted admission was missing a required artifact receipt (kind carried).
    #[error("accepted native admission is missing required artifact receipt: {kind}")]
    MissingRequiredArtifactReceipt {
        /// Required artifact kind code.
        kind: &'static str,
    },
    /// The native planning identity was invalid (reason code and detail carried).
    #[error("invalid native planning identity: {reason_code}: {detail}")]
    InvalidPlanningIdentity {
        /// Stable reason code for the planning-identity rejection.
        reason_code: &'static str,
        /// Human-readable detail.
        detail: String,
    },
}

/// Registry order for frontend families covered by native contracts.
pub fn current_native_frontend_families() -> &'static [SharedEngineFrontendFamily] {
    &SHARED_NATIVE_CONTRACT_FRONTEND_FAMILIES
}

fn validate_abi(abi: &SharedNativeAbiSignature) -> Result<(), SharedNativeContractValidationError> {
    for (index, param) in abi.params.iter().enumerate() {
        if param.name.trim().is_empty() {
            return Err(SharedNativeContractValidationError::MissingAbiParameterName { index });
        }
    }
    Ok(())
}

fn validate_planning_identity(
    contract: &SharedNativeContract,
) -> Result<(), SharedNativeContractValidationError> {
    contract
        .planning_identity()
        .validate(contract.source_kind)
        .map_err(
            |rejection| SharedNativeContractValidationError::InvalidPlanningIdentity {
                reason_code: rejection.reason_code,
                detail: rejection.detail,
            },
        )
}

fn validate_layout(
    layout: &SharedNativeLayoutContract,
) -> Result<(), SharedNativeContractValidationError> {
    for (index, vector) in layout.vector_contracts.iter().enumerate() {
        if vector.identity.trim().is_empty() {
            return Err(SharedNativeContractValidationError::InvalidVectorContract {
                index,
                reason: "missing_identity",
            });
        }
        if vector.logical_lanes == 0 {
            return Err(SharedNativeContractValidationError::InvalidVectorContract {
                index,
                reason: "zero_logical_lanes",
            });
        }
        if vector.physical_lanes == 0 {
            return Err(SharedNativeContractValidationError::InvalidVectorContract {
                index,
                reason: "zero_physical_lanes",
            });
        }
        if vector.element_bits == 0 {
            return Err(SharedNativeContractValidationError::InvalidVectorContract {
                index,
                reason: "zero_element_bits",
            });
        }
        if vector.lane_bits == 0 {
            return Err(SharedNativeContractValidationError::InvalidVectorContract {
                index,
                reason: "zero_lane_bits",
            });
        }
        if !vector.fail_closed_unavailable {
            return Err(SharedNativeContractValidationError::InvalidVectorContract {
                index,
                reason: "not_fail_closed_unavailable",
            });
        }
    }
    Ok(())
}

fn validate_cache_reuse_contract(
    contract: &SharedNativeContract,
) -> Result<(), SharedNativeContractValidationError> {
    let policy = contract.identity.cache_reuse_policy.trim();
    if policy != contract.identity.cache_reuse_policy {
        return Err(
            SharedNativeContractValidationError::InvalidCacheReusePolicy {
                policy: contract.identity.cache_reuse_policy.clone(),
            },
        );
    }
    match policy {
        "" | "none" => {
            return Err(SharedNativeContractValidationError::MissingField(
                "cache_reuse_policy",
            ));
        }
        _ => {
            if SharedNativeCacheReusePolicy::from_code(policy).is_none() {
                return Err(
                    SharedNativeContractValidationError::InvalidCacheReusePolicy {
                        policy: policy.to_string(),
                    },
                );
            }
        }
    }

    if policy == SHARED_NATIVE_CACHE_REUSE_DISABLED
        && has_identity(contract.identity.cache_digest.as_deref())
    {
        return Err(SharedNativeContractValidationError::CacheReuseDisabledHasCacheDigest);
    }

    if policy != SHARED_NATIVE_CACHE_REUSE_FRONTEND_REUSABLE {
        return Ok(());
    }

    if contract.compatible_frontend_families.len() < 2 {
        return Err(
            SharedNativeContractValidationError::FrontendReusableCacheRequiresCompatibleFamilies,
        );
    }

    require_optional_identity("cache_digest", &contract.identity.cache_digest)?;
    require_optional_identity(
        "fingerprint_domain_identity",
        &contract.identity.fingerprint_domain_identity,
    )?;
    require_optional_identity(
        "cache_namespace_identity",
        &contract.identity.cache_namespace_identity,
    )?;
    require_optional_identity("semantic_digest", &contract.identity.semantic_digest)?;
    require_optional_identity(
        "storage_layout_fingerprint",
        &contract.identity.storage_layout_fingerprint,
    )?;

    if let (Some(identity), Some(layout)) = (
        contract.identity.storage_layout_fingerprint.as_deref(),
        contract.layout.fingerprint.as_deref(),
    ) {
        if identity.trim() != layout.trim() {
            return Err(
                SharedNativeContractValidationError::CacheReuseLayoutFingerprintMismatch {
                    identity: identity.to_string(),
                    layout: layout.to_string(),
                },
            );
        }
    }

    Ok(())
}

fn validate_evidence_policy(
    policy: &SharedNativeEvidencePolicy,
) -> Result<(), SharedNativeContractValidationError> {
    if !policy.fail_closed {
        return Err(SharedNativeContractValidationError::EvidencePolicyNotFailClosed);
    }

    for requirement in &policy.required_evidence {
        if requirement.required && !has_identity(Some(&requirement.identity)) {
            return Err(
                SharedNativeContractValidationError::RequiredEvidenceMissingIdentity {
                    kind: requirement.kind.code(),
                },
            );
        }
        if requirement.required && !requirement.fail_closed {
            return Err(
                SharedNativeContractValidationError::RequiredEvidenceNotFailClosed {
                    kind: requirement.kind.code(),
                },
            );
        }
    }
    Ok(())
}

fn validate_admission(
    admission: &SharedNativeAdmission,
) -> Result<(), SharedNativeContractValidationError> {
    if !admission.fail_closed {
        return Err(SharedNativeContractValidationError::AdmissionNotFailClosed);
    }

    for receipt in admission
        .accepted_validation_receipts
        .iter()
        .chain(admission.rejected_validation_receipts.iter())
    {
        receipt.validate().map_err(|error| {
            SharedNativeContractValidationError::InvalidValidationReceipt(error.to_string())
        })?;
    }

    match admission.status {
        SharedNativeAdmissionStatus::Accepted => {
            if has_failure_reason(admission.failure_reason.as_deref()) {
                return Err(SharedNativeContractValidationError::AcceptedAdmissionHasFailureReason);
            }
            if admission.disposition == SharedNativeAdmissionDisposition::Rejected {
                return Err(
                    SharedNativeContractValidationError::AcceptedAdmissionRejectedDisposition,
                );
            }
            if !admission.rejected_validation_receipts.is_empty() {
                return Err(
                    SharedNativeContractValidationError::AcceptedAdmissionContainsRejectedReceipt,
                );
            }
        }
        SharedNativeAdmissionStatus::Rejected => {
            if !has_failure_reason(admission.failure_reason.as_deref()) {
                return Err(
                    SharedNativeContractValidationError::RejectedAdmissionMissingFailureReason,
                );
            }
            if admission.production_selected {
                return Err(
                    SharedNativeContractValidationError::RejectedAdmissionProductionSelected,
                );
            }
        }
    }

    if admission.production_selected && !admission.authority.is_callable() {
        return Err(SharedNativeContractValidationError::ProductionWithoutCallableAuthority);
    }

    Ok(())
}

fn validate_admission_against_policy(
    admission: &SharedNativeAdmission,
    policy: &SharedNativeEvidencePolicy,
) -> Result<(), SharedNativeContractValidationError> {
    if !admission.is_accepted() {
        return Ok(());
    }

    for kind in &policy.required_validator_kinds {
        if !admission
            .accepted_validation_receipts
            .iter()
            .any(|receipt| receipt.validator_kind == *kind)
        {
            return Err(
                SharedNativeContractValidationError::MissingRequiredValidatorReceipt {
                    kind: kind.code(),
                },
            );
        }
    }

    for kind in &policy.required_artifact_kinds {
        if !admission
            .accepted_validation_receipts
            .iter()
            .any(|receipt| receipt.validation_artifact_kind == *kind)
        {
            return Err(
                SharedNativeContractValidationError::MissingRequiredArtifactReceipt {
                    kind: kind.code(),
                },
            );
        }
    }

    Ok(())
}

fn require_identity(
    field: &'static str,
    value: &str,
) -> Result<(), SharedNativeContractValidationError> {
    if !has_identity(Some(value)) {
        Err(SharedNativeContractValidationError::MissingField(field))
    } else {
        Ok(())
    }
}

fn require_optional_identity(
    field: &'static str,
    value: &Option<String>,
) -> Result<(), SharedNativeContractValidationError> {
    if has_identity(value.as_deref()) {
        Ok(())
    } else {
        Err(SharedNativeContractValidationError::MissingField(field))
    }
}

fn has_identity(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|value| !value.is_empty() && value != "none")
}

fn has_failure_reason(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|value| !value.is_empty() && value != "none")
}

fn non_empty_string(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn push_non_empty(values: &mut Vec<String>, value: String) {
    if !value.trim().is_empty() {
        values.push(value);
    }
}

fn evidence_value(value: &str) -> String {
    if value.trim().is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn join_codes<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn join_rendered(values: impl IntoIterator<Item = String>) -> String {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_identity() -> SharedNativeContractIdentity {
        SharedNativeContractIdentity::new(
            "prepared mcc petri program",
            "native successor candidate",
            "native successor lane",
        )
        .with_source_fingerprint("source fingerprint")
        .with_frontend_payload_identity("mcc pnml payload")
        .with_plan_reuse_manifest("trust-ir-batch-manifest-v1-abc", "manifest digest")
        .with_trust_ir_identity("frontend neutral trust_ir")
        .with_trust_ir_module_digest("trust_ir digest")
        .with_compiler_facts_digest("compiler facts v2 digest")
        .with_native_bundle_digest("native bundle v4 digest")
        .with_transport_identity("native transport identity")
        .with_semantic_digest("semantic digest")
        .with_link_digest("link digest")
        .with_cache_digest("cache digest")
        .with_frontend_reusable_cache_domain(
            "fingerprint_domain_key:petri_marking:canonical:v1",
            "shared_native_cache:petri_marking_successor:v1",
        )
        .with_cas_identity("partitioned cas petri marking v1")
        .with_cache_identity("native plan cache petri marking v1")
        .with_artifact_identity("trust-cg artifact")
        .with_target_abi_identity("darwin arm64 abi")
        .with_storage_layout_fingerprint("marking layout digest")
        .with_proof_policy_identity("proof tv policy")
        .with_replay_identity("replay transcript")
    }

    fn petri_successor_abi() -> SharedNativeAbiSignature {
        SharedNativeAbiSignature::new("extern_c", "petri_successor_batch")
            .with_param("input_states", SharedNativeAbiValueKind::Ptr)
            .with_param("output_states", SharedNativeAbiValueKind::Ptr)
            .with_param("state_len", SharedNativeAbiValueKind::U32)
            .with_param("successor_counts", SharedNativeAbiValueKind::Ptr)
            .with_param("state_count", SharedNativeAbiValueKind::U32)
            .with_param("scratch", SharedNativeAbiValueKind::Ptr)
            .with_param("scratch_len", SharedNativeAbiValueKind::U32)
            .with_param("diagnostics", SharedNativeAbiValueKind::Ptr)
            .with_param("diagnostics_len", SharedNativeAbiValueKind::U32)
    }

    fn accepted_proof_receipt() -> ValidationReceipt {
        ValidationReceipt::accepted(
            ValidationReceiptValidatorKind::AYProof,
            "sha256",
            "receipt digest",
            "prepared mcc petri program",
            "native successor candidate",
            ValidationReceiptArtifactKind::Proof,
            "proof tv accepted evidence",
        )
    }

    fn base_contract() -> SharedNativeContract {
        let policy =
            SharedNativeEvidencePolicy::fail_closed(SharedNativeInstallAuthority::ActiveCallable)
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::ManifestMetadata,
                    "trust-cg manifest metadata",
                ))
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::LayoutChecksum,
                    "petri marking layout checksum",
                ))
                .with_required_evidence(SharedNativeEvidenceRequirement::fail_closed(
                    SharedNativeEvidenceKind::ProofPolicyChecksum,
                    "proof tv policy checksum",
                ))
                .with_required_validator(ValidationReceiptValidatorKind::AYProof)
                .with_required_artifact(ValidationReceiptArtifactKind::Proof)
                .with_required_manifest_metadata("target_triple")
                .with_required_manifest_metadata("kernel_artifact_kind");

        let admission = SharedNativeAdmission::accepted_fail_closed(
            SharedNativeInstallAuthority::ActiveCallable,
        )
        .with_production_selected(true)
        .with_validation_receipt(accepted_proof_receipt())
        .with_install_packet_hash("install packet hash")
        .with_replay_identity("replay transcript")
        .with_native_payload_sha256("payload sha256")
        .with_target_checksum("target checksum")
        .with_abi_checksum("abi checksum")
        .with_layout_checksum("layout checksum")
        .with_proof_policy_checksum("proof policy checksum")
        .with_semantic_checksum("semantic checksum");

        SharedNativeContract::new(
            CheckerSourceKind::MccPetri,
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
            SharedNativeContractKind::SuccessorKernel,
            petri_successor_abi(),
            base_identity(),
        )
        .with_layout(
            SharedNativeLayoutContract::from_storage_kind(
                PreparedStorageKind::PetriMarking,
                "petri marking layout",
            )
            .with_fingerprint("marking layout digest")
            .with_state_len(4)
            .with_vector_contract(
                SharedNativeVectorContract::new(
                    "u32 marking vector",
                    SharedNativeAbiValueKind::U32,
                    4,
                    4,
                    32,
                    32,
                )
                .with_mask_identity("all lanes live")
                .with_operations_identity("add/sub/compare")
                .with_feature_guard("scalar fallback"),
            ),
        )
        .with_evidence_policy(policy)
        .with_admission(admission)
    }

    #[test]
    fn shared_native_contract_schema_and_frontends_are_stable() {
        assert_eq!(SHARED_NATIVE_CONTRACT_ROW_KIND, "shared_native_contract");
        assert_eq!(
            SHARED_NATIVE_CONTRACT_SCHEMA,
            "ty.shared.native_contract.v1"
        );
        assert_eq!(SHARED_NATIVE_CONTRACT_SCHEMA_VERSION, 1);
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"compiler_facts_digest"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"native_bundle_digest"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"native_planning_identity"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"source_fingerprint"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"plan_reuse_manifest_id"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"plan_reuse_manifest_digest"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"fingerprint_domain_identity"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"cas_identity"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"cache_identity"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"cache_namespace_identity"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"cache_reuse_policy"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"frontend_family_scope_identity"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"storage_layout_fingerprint"));
        assert!(SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS.contains(&"admission_fail_closed"));

        let family_codes = current_native_frontend_families()
            .iter()
            .map(|family| family.code())
            .collect::<Vec<_>>();
        assert_eq!(
            family_codes,
            vec![
                "tla_plus",
                "quint",
                "mcc_petri",
                "aiger",
                "btor2",
                "vmt_transition_system",
                "ay_analytical",
                "witness_replay",
            ]
        );

        assert_eq!(
            SharedNativeContractKind::from_code("successor_kernel"),
            Some(SharedNativeContractKind::SuccessorKernel)
        );
        assert_eq!(
            SharedNativeLayoutKind::from(PreparedStorageKind::HardwareRegisters),
            SharedNativeLayoutKind::HardwareRegisters
        );
    }

    #[test]
    fn shared_native_admission_helpers_are_fail_closed() {
        let accepted = SharedNativeAdmission::accepted_fail_closed(
            SharedNativeInstallAuthority::CanaryCallable,
        )
        .with_production_selected(true)
        .with_validation_receipt(accepted_proof_receipt());

        assert!(accepted.is_accepted());
        assert!(accepted.fail_closed);
        assert!(accepted.can_publish_callable());
        assert_eq!(
            accepted.reason,
            SharedNativeAdmissionReason::AcceptedEvidence
        );

        let rejected = SharedNativeAdmission::rejected_fail_closed(
            SharedNativeAdmissionReason::RuntimeBlocked,
            "parity replay gate not promoted",
        );
        assert!(rejected.is_rejected());
        assert!(rejected.fail_closed);
        assert!(!rejected.can_publish_callable());
        assert_eq!(
            rejected.failure_reason.as_deref(),
            Some("parity replay gate not promoted")
        );
    }

    #[test]
    fn shared_native_contract_validates_and_renders_core_row() {
        let contract = base_contract();
        contract.validate().unwrap();

        let row = contract.render_evidence_row("CORE");
        assert!(row.starts_with("CORE shared_native_contract "));
        assert!(row.contains("schema=ty.shared.native_contract.v1"));
        assert!(row.contains("source_kind=mcc_petri"));
        assert!(row.contains("frontend_kind=mcc_petri"));
        assert!(row.contains("payload_kind=mcc_petri"));
        assert!(row.contains("storage_kind=petri_marking"));
        assert!(row.contains("contract_kind=successor_kernel"));
        assert!(row.contains("lane_kind=native"));
        assert!(row.contains("abi=extern_c"));
        assert!(row.contains("symbol=petri_successor_batch"));
        assert!(row.contains("abi_params=input_states:ptr"));
        assert!(row.contains("native_planning_identity=native_planning_identity"));
        assert!(row.contains("source_fingerprint=source_fingerprint"));
        assert!(row.contains("plan_reuse_manifest_id=trust-ir-batch-manifest-v1-abc"));
        assert!(row.contains("plan_reuse_manifest_digest=manifest_digest"));
        assert!(row.contains("compiler_facts_digest=compiler_facts_v2_digest"));
        assert!(row.contains("native_bundle_digest=native_bundle_v4_digest"));
        assert!(row.contains(
            "fingerprint_domain_identity=fingerprint_domain_key:petri_marking:canonical:v1"
        ));
        assert!(row.contains("cas_identity=partitioned_cas_petri_marking_v1"));
        assert!(row.contains("cache_identity=native_plan_cache_petri_marking_v1"));
        assert!(
            row.contains("cache_namespace_identity=shared_native_cache:petri_marking_successor:v1")
        );
        assert!(row.contains("cache_reuse_policy=frontend_reusable"));
        assert!(row.contains("frontend_family_scope_identity=native_frontend_family_scope"));
        assert!(row.contains("storage_layout_fingerprint=marking_layout_digest"));
        assert!(row.contains("required_validators=ay_proof"));
        assert!(row.contains("required_artifacts=proof"));
        assert!(row.contains("admission_status=accepted"));
        assert!(row.contains("admission_authority=active_callable"));
        assert!(row.contains("admission_fail_closed=true"));
        assert!(row.contains("production_selected=true"));
    }

    #[test]
    fn shared_native_contract_renders_planning_identity_row() {
        let contract = base_contract();
        contract.validate().unwrap();

        let row = contract.render_planning_identity_evidence_row("CORE");
        assert!(row.starts_with("CORE shared_native_planning_identity "));
        assert!(row.contains("schema=ty.shared.native_planning_identity.v1"));
        assert!(row.contains("source_kind=mcc_petri"));
        assert!(row.contains("frontend_kind=mcc_petri"));
        assert!(row.contains("native_planning_identity=native_planning_identity"));
        assert!(row.contains("source_fingerprint=source_fingerprint"));
        assert!(row.contains("plan_reuse_manifest_id=trust-ir-batch-manifest-v1-abc"));
        assert!(row.contains("plan_reuse_manifest_digest=manifest_digest"));
        assert!(row.contains(
            "fingerprint_domain_identity=fingerprint_domain_key_petri_marking_canonical_v1"
        ));
        assert!(row.contains("cas_identity=partitioned_cas_petri_marking_v1"));
        assert!(row.contains("cache_identity=native_plan_cache_petri_marking_v1"));
        assert!(row.contains("cache_reuse_policy=frontend_reusable"));
        assert!(row.contains("frontend_family_scope=tla_plus,quint,mcc_petri,aiger,btor2"));
        assert!(row.contains("frontend_family_scope_identity=native_frontend_family_scope"));
        assert!(row.contains("frontend_family_reusable=true"));
    }

    #[test]
    fn shared_native_contract_defaults_cache_reuse_to_frontend_local() {
        let contract = SharedNativeContract::new(
            CheckerSourceKind::AYOnly,
            PreparedProgramPayloadKind::AYOnly,
            PreparedStorageKind::SmtVariables,
            SharedNativeContractKind::AYSymbolicKernel,
            SharedNativeAbiSignature::new("extern_c", "ay_symbolic_helper"),
            SharedNativeContractIdentity::new("prepared", "candidate", "lane"),
        )
        .with_layout(SharedNativeLayoutContract::from_storage_kind(
            PreparedStorageKind::SmtVariables,
            "solver object layout",
        ))
        .with_admission(SharedNativeAdmission::rejected_fail_closed(
            SharedNativeAdmissionReason::RuntimeBlocked,
            "candidate only",
        ));

        contract.validate().unwrap();
        assert_eq!(
            contract.identity.cache_reuse_policy,
            SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY
        );

        let row = contract.render_evidence_row("CORE");
        assert!(row.contains("fingerprint_domain_identity=none"));
        assert!(row.contains("cache_namespace_identity=none"));
        assert!(row.contains("cache_reuse_policy=frontend_local_only"));
        assert!(row.contains("native_planning_identity=native_planning_identity"));
    }

    #[test]
    fn shared_native_contract_rejects_frontend_reusable_cache_without_domain_evidence() {
        let mut missing_domain = base_contract();
        missing_domain.identity.fingerprint_domain_identity = None;
        assert_eq!(
            missing_domain.validate(),
            Err(SharedNativeContractValidationError::MissingField(
                "fingerprint_domain_identity"
            ))
        );

        let mut missing_namespace = base_contract();
        missing_namespace.identity.cache_namespace_identity = None;
        assert_eq!(
            missing_namespace.validate(),
            Err(SharedNativeContractValidationError::MissingField(
                "cache_namespace_identity"
            ))
        );
    }

    #[test]
    fn shared_native_contract_rejects_frontend_reusable_cache_without_cache_digest() {
        let mut contract = base_contract();
        contract.identity.cache_digest = None;

        assert_eq!(
            contract.validate(),
            Err(SharedNativeContractValidationError::MissingField(
                "cache_digest"
            ))
        );
    }

    #[test]
    fn shared_native_contract_rejects_mismatched_frontend_reusable_layout_fingerprint() {
        let mut contract = base_contract();
        contract.layout.fingerprint = Some("different layout digest".to_string());

        assert_eq!(
            contract.validate(),
            Err(
                SharedNativeContractValidationError::CacheReuseLayoutFingerprintMismatch {
                    identity: "marking layout digest".to_string(),
                    layout: "different layout digest".to_string(),
                }
            )
        );
    }

    #[test]
    fn shared_native_contract_rejects_invalid_cache_reuse_policy() {
        let mut contract = base_contract();
        contract.identity.cache_reuse_policy = "frontend opportunistic".to_string();

        assert_eq!(
            contract.validate(),
            Err(
                SharedNativeContractValidationError::InvalidCacheReusePolicy {
                    policy: "frontend opportunistic".to_string()
                }
            )
        );
    }

    #[test]
    fn shared_native_contract_rejects_duplicate_planning_frontend_scope() {
        let contract = base_contract().with_compatible_frontend_families([
            SharedEngineFrontendFamily::MccPetri,
            SharedEngineFrontendFamily::MccPetri,
        ]);

        assert_eq!(
            contract.validate(),
            Err(
                SharedNativeContractValidationError::InvalidPlanningIdentity {
                    reason_code: "duplicate_frontend_family",
                    detail: "native planning frontend scope contains duplicate family mcc_petri"
                        .to_string(),
                }
            )
        );
    }

    #[test]
    fn shared_native_contract_rejects_partial_plan_reuse_manifest_identity() {
        let mut contract = base_contract();
        contract.identity.plan_reuse_manifest_digest = None;

        assert_eq!(
            contract.validate(),
            Err(
                SharedNativeContractValidationError::InvalidPlanningIdentity {
                    reason_code: "incomplete_plan_reuse_manifest",
                    detail: "native planning reuse manifest requires both id and digest"
                        .to_string(),
                }
            )
        );
    }

    #[test]
    fn shared_native_contract_covers_each_current_frontend_family() {
        let cases = [
            (
                CheckerSourceKind::Tla,
                PreparedProgramPayloadKind::Tla,
                PreparedStorageKind::TlaStateSlots,
            ),
            (
                CheckerSourceKind::Quint,
                PreparedProgramPayloadKind::Quint,
                PreparedStorageKind::TlaStateSlots,
            ),
            (
                CheckerSourceKind::MccPetri,
                PreparedProgramPayloadKind::MccPetri,
                PreparedStorageKind::PetriMarking,
            ),
            (
                CheckerSourceKind::Aiger,
                PreparedProgramPayloadKind::Aiger,
                PreparedStorageKind::HardwareRegisters,
            ),
            (
                CheckerSourceKind::Btor2,
                PreparedProgramPayloadKind::Btor2,
                PreparedStorageKind::HardwareRegisters,
            ),
            (
                CheckerSourceKind::VmtInterchange,
                PreparedProgramPayloadKind::VmtInterchange,
                PreparedStorageKind::SmtVariables,
            ),
            (
                CheckerSourceKind::AYOnly,
                PreparedProgramPayloadKind::AYOnly,
                PreparedStorageKind::SmtVariables,
            ),
            (
                CheckerSourceKind::WitnessReplay,
                PreparedProgramPayloadKind::WitnessReplay,
                PreparedStorageKind::WitnessSteps,
            ),
        ];

        for (source, payload, storage) in cases {
            let contract = SharedNativeContract::new(
                source,
                payload,
                storage,
                SharedNativeContractKind::NativeHelperKernel,
                SharedNativeAbiSignature::new("extern_c", "shared_helper"),
                SharedNativeContractIdentity::new("prepared", "candidate", "lane"),
            )
            .with_layout(SharedNativeLayoutContract::from_storage_kind(
                storage,
                storage.code(),
            ))
            .with_admission(SharedNativeAdmission::rejected_fail_closed(
                SharedNativeAdmissionReason::RuntimeBlocked,
                "candidate only",
            ));

            contract.validate().unwrap();
        }
    }

    #[test]
    fn shared_native_contract_rejects_missing_required_receipt() {
        let contract = base_contract().with_admission(
            SharedNativeAdmission::accepted_fail_closed(
                SharedNativeInstallAuthority::ActiveCallable,
            )
            .with_production_selected(true),
        );

        assert_eq!(
            contract.validate(),
            Err(
                SharedNativeContractValidationError::MissingRequiredValidatorReceipt {
                    kind: "ay_proof"
                }
            )
        );
    }

    #[test]
    fn shared_native_contract_rejects_identityless_required_evidence() {
        let mut contract = base_contract();
        contract.evidence_policy.required_evidence[0].identity = "none".to_string();

        assert_eq!(
            contract.validate(),
            Err(
                SharedNativeContractValidationError::RequiredEvidenceMissingIdentity {
                    kind: "manifest_metadata"
                }
            )
        );
    }

    #[test]
    fn shared_native_contract_keeps_fail_closed_policy_on_vectors_and_admission() {
        let invalid_vector = SharedNativeVectorContract::new(
            "hardware register vector",
            SharedNativeAbiValueKind::U64,
            8,
            8,
            64,
            64,
        )
        .with_fail_closed_unavailable(false);

        let contract = base_contract().with_layout(
            SharedNativeLayoutContract::from_storage_kind(
                PreparedStorageKind::HardwareRegisters,
                "hardware register layout",
            )
            .with_vector_contract(invalid_vector),
        );

        assert_eq!(
            contract.validate(),
            Err(SharedNativeContractValidationError::InvalidVectorContract {
                index: 0,
                reason: "not_fail_closed_unavailable"
            })
        );

        let rejected = base_contract().with_admission(SharedNativeAdmission {
            production_selected: true,
            ..SharedNativeAdmission::rejected_fail_closed(
                SharedNativeAdmissionReason::RuntimeBlocked,
                "candidate only",
            )
        });
        assert_eq!(
            rejected.validate(),
            Err(SharedNativeContractValidationError::RejectedAdmissionProductionSelected)
        );

        let mut not_fail_closed_admission = SharedNativeAdmission::accepted_fail_closed(
            SharedNativeInstallAuthority::ActiveCallable,
        )
        .with_production_selected(true)
        .with_validation_receipt(accepted_proof_receipt());
        not_fail_closed_admission.fail_closed = false;
        let not_fail_closed = base_contract().with_admission(not_fail_closed_admission);
        assert_eq!(
            not_fail_closed.validate(),
            Err(SharedNativeContractValidationError::AdmissionNotFailClosed)
        );
    }
}
