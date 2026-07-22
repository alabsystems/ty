// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral prepared checker program descriptors.
//!
//! These types intentionally carry identities, layouts, transition/property
//! descriptors, and validation obligations, not runtime state such as queues,
//! caches, compiled libraries, solver handles, or explored markings.

use crate::{
    backend_capability::{BackendKind, ProblemKind, SolverFacet},
    evidence_row::evidence_field as prepared_evidence_field,
    setup_trace::{
        CheckerArtifactIdentityFields, CheckerSourceKind, SetupTraceKey, SetupTraceLaneKind,
    },
};

/// Stable row kind for prepared-program summary evidence.
pub const PREPARED_CHECKER_PROGRAM_ROW_KIND: &str = "prepared_checker_program";

/// Stable row kind for frontend-extension evidence.
pub const PREPARED_FRONTEND_EXTENSION_ROW_KIND: &str = "prepared_frontend_extension";

/// Stable row kind for candidate-lane evidence.
pub const PREPARED_CANDIDATE_LANE_ROW_KIND: &str = "prepared_candidate_lane";

/// Stable row kind for validation-plan evidence.
pub const PREPARED_VALIDATION_PLAN_ROW_KIND: &str = "prepared_validation_plan";

/// Reserved payload-family code for importers that have not registered a
/// concrete prepared-program contract.
pub const PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE: &str = "future_importer";

/// Fields every prepared-program summary row publishes.
pub const PREPARED_CHECKER_PROGRAM_REQUIRED_FIELDS: &[&str] = &[
    "identity",
    "source_kind",
    "frontend_kind",
    "payload_kind",
    "storage_kind",
    "canonical_payload_identity",
    "source_identity",
    "config_identity",
    "examination_identity",
    "cache_key",
    "source_fingerprint",
    "frontend_payload_identity",
    "frontend_payload_fingerprint",
    "prepared_program_fingerprint",
    "artifact_identity",
    "artifact_fingerprint",
    "storage_policy_identity",
    "storage_layout_fingerprint",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "batch_artifact_identity",
    "candidate_identity",
    "lane_identity",
    "transition_descriptor_fingerprint",
    "property_descriptor_fingerprint",
    "validation_plan_fingerprint",
    "transitions",
    "properties",
    "analytical_solves",
    "symbolic_proofs",
    "backend_families",
    "fingerprint_id",
    "fingerprint_scheme",
    "canonical_identities",
    "frontend_extensions",
    "candidate_lanes",
    "validation_plans",
    "validations",
];

/// Fields every frontend-extension row publishes.
pub const PREPARED_FRONTEND_EXTENSION_REQUIRED_FIELDS: &[&str] = &[
    "identity",
    "source_kind",
    "frontend_kind",
    "payload_kind",
    "storage_kind",
    "extension_kind",
    "extension_source_kind",
    "extension_payload_kind",
    "extension_storage_kind",
    "problem",
    "canonical_payload_identity",
    "source_identity",
    "config_identity",
    "examination_identity",
    "cache_key",
    "source_fingerprint",
    "frontend_payload_identity",
    "frontend_payload_fingerprint",
    "prepared_program_fingerprint",
    "artifact_identity",
    "artifact_fingerprint",
    "storage_policy_identity",
    "storage_layout_fingerprint",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "batch_artifact_identity",
    "candidate_identity",
    "lane_identity",
    "transition_descriptor_fingerprint",
    "property_descriptor_fingerprint",
    "validation_plan_fingerprint",
];

/// Fields every prepared candidate-lane row publishes.
pub const PREPARED_CANDIDATE_LANE_REQUIRED_FIELDS: &[&str] = &[
    "identity",
    "source_kind",
    "frontend_kind",
    "payload_kind",
    "storage_kind",
    "lane_kind",
    "lane",
    "candidate_key",
    "candidate_identity",
    "lane_identity",
    "canonical_payload_identity",
    "source_identity",
    "config_identity",
    "examination_identity",
    "cache_key",
    "source_fingerprint",
    "frontend_payload_identity",
    "frontend_payload_fingerprint",
    "prepared_program_fingerprint",
    "artifact_identity",
    "artifact_fingerprint",
    "storage_policy_identity",
    "storage_layout_fingerprint",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "batch_artifact_identity",
    "transition_descriptor_fingerprint",
    "property_descriptor_fingerprint",
    "validation_plan_fingerprint",
    "fingerprint_id",
    "fingerprint_scheme",
];

/// Fields every prepared validation-plan row publishes.
pub const PREPARED_VALIDATION_PLAN_REQUIRED_FIELDS: &[&str] = &[
    "identity",
    "source_kind",
    "frontend_kind",
    "payload_kind",
    "storage_kind",
    "validation_kind",
    "problem",
    "required",
    "fail_closed",
    "fingerprint_id",
    "fingerprint_scheme",
    "fingerprint_canonicalization",
    "canonical_payload_identity",
    "source_identity",
    "config_identity",
    "examination_identity",
    "cache_key",
    "source_fingerprint",
    "frontend_payload_identity",
    "frontend_payload_fingerprint",
    "prepared_program_fingerprint",
    "artifact_identity",
    "artifact_fingerprint",
    "storage_policy_identity",
    "storage_layout_fingerprint",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "batch_artifact_identity",
    "candidate_identity",
    "lane_identity",
    "transition_descriptor_fingerprint",
    "property_descriptor_fingerprint",
    "validation_plan_fingerprint",
];

/// Frontend-specific payload carried behind the shared preparation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedProgramPayloadKind {
    /// Wire code `"tla"`.
    Tla,
    /// Wire code `"quint"`.
    Quint,
    /// Wire code `"mcc_petri"`.
    MccPetri,
    /// Wire code `"aiger"`.
    Aiger,
    /// Wire code `"btor2"`.
    Btor2,
    /// Wire code `"vmt_interchange"`.
    VmtInterchange,
    /// Wire code `"ay_only"`.
    AYOnly,
    /// Wire code `"witness_replay"`.
    WitnessReplay,
}

impl PreparedProgramPayloadKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Tla => "tla",
            Self::Quint => "quint",
            Self::MccPetri => "mcc_petri",
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::VmtInterchange => "vmt_interchange",
            Self::AYOnly => "ay_only",
            Self::WitnessReplay => "witness_replay",
        }
    }

    /// The checker source kind corresponding to this payload kind.
    pub fn source_kind(self) -> CheckerSourceKind {
        self.into()
    }

    /// All payload kinds that participate in the shared engine.
    pub fn shared_engine_payloads() -> &'static [Self] {
        &[
            Self::Tla,
            Self::Quint,
            Self::MccPetri,
            Self::Aiger,
            Self::Btor2,
            Self::VmtInterchange,
            Self::AYOnly,
            Self::WitnessReplay,
        ]
    }

    /// Payload codes reserved for future importers (not yet assignable).
    pub fn reserved_payload_codes() -> &'static [&'static str] {
        &[PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE]
    }

    /// Wire codes of the default (currently supported) payload kinds.
    pub fn default_payload_codes() -> &'static [&'static str] {
        &[
            "tla",
            "quint",
            "mcc_petri",
            "aiger",
            "btor2",
            "vmt_interchange",
            "ay_only",
            "witness_replay",
        ]
    }
}

impl From<PreparedProgramPayloadKind> for CheckerSourceKind {
    fn from(value: PreparedProgramPayloadKind) -> Self {
        match value {
            PreparedProgramPayloadKind::Tla => Self::Tla,
            PreparedProgramPayloadKind::Quint => Self::Quint,
            PreparedProgramPayloadKind::MccPetri => Self::MccPetri,
            PreparedProgramPayloadKind::Aiger => Self::Aiger,
            PreparedProgramPayloadKind::Btor2 => Self::Btor2,
            PreparedProgramPayloadKind::VmtInterchange => Self::VmtInterchange,
            PreparedProgramPayloadKind::AYOnly => Self::AYOnly,
            PreparedProgramPayloadKind::WitnessReplay => Self::WitnessReplay,
        }
    }
}

/// Storage ABI expected by transitions and property kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedStorageKind {
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
    /// Wire code `"unknown"`.
    Unknown,
}

impl PreparedStorageKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaStateSlots => "tla_state_slots",
            Self::PetriMarking => "petri_marking",
            Self::HardwareRegisters => "hardware_registers",
            Self::SmtVariables => "smt_variables",
            Self::WitnessSteps => "witness_steps",
            Self::Unknown => "unknown",
        }
    }
}

/// Frontend adapter families that extend the shared prepared-program boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedFrontendExtensionKind {
    /// Wire code `"aiger"`.
    Aiger,
    /// Wire code `"btor2"`.
    Btor2,
    /// Wire code `"vmt_interchange"`.
    VmtInterchange,
    /// Wire code `"ay_only"`.
    AYOnly,
    /// Wire code `"witness_replay"`.
    WitnessReplay,
}

impl PreparedFrontendExtensionKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::VmtInterchange => "vmt_interchange",
            Self::AYOnly => "ay_only",
            Self::WitnessReplay => "witness_replay",
        }
    }

    /// The prepared-program payload kind this extension produces.
    pub fn payload_kind(self) -> PreparedProgramPayloadKind {
        match self {
            Self::Aiger => PreparedProgramPayloadKind::Aiger,
            Self::Btor2 => PreparedProgramPayloadKind::Btor2,
            Self::VmtInterchange => PreparedProgramPayloadKind::VmtInterchange,
            Self::AYOnly => PreparedProgramPayloadKind::AYOnly,
            Self::WitnessReplay => PreparedProgramPayloadKind::WitnessReplay,
        }
    }

    /// The checker source kind this extension maps to.
    pub fn source_kind(self) -> CheckerSourceKind {
        self.payload_kind().source_kind()
    }

    /// The storage kind this extension's payloads use.
    pub fn storage_kind(self) -> PreparedStorageKind {
        match self {
            Self::Aiger | Self::Btor2 => PreparedStorageKind::HardwareRegisters,
            Self::VmtInterchange | Self::AYOnly => PreparedStorageKind::SmtVariables,
            Self::WitnessReplay => PreparedStorageKind::WitnessSteps,
        }
    }
}

/// Class of transition prepared by a frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedTransitionKind {
    /// Wire code `"tla_action"`.
    TlaAction,
    /// Wire code `"petri_transition"`.
    PetriTransition,
    /// Wire code `"hardware_next_state"`.
    HardwareNextState,
    /// Wire code `"symbolic_transition_relation"`.
    SymbolicTransitionRelation,
    /// Wire code `"replay_step"`.
    ReplayStep,
}

impl PreparedTransitionKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaAction => "tla_action",
            Self::PetriTransition => "petri_transition",
            Self::HardwareNextState => "hardware_next_state",
            Self::SymbolicTransitionRelation => "symbolic_transition_relation",
            Self::ReplayStep => "replay_step",
        }
    }
}

/// Class of property or output obligation prepared by a frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedPropertyKind {
    /// Wire code `"invariant"`.
    Invariant,
    /// Wire code `"state_constraint"`.
    StateConstraint,
    /// Wire code `"deadlock"`.
    Deadlock,
    /// Wire code `"reachability"`.
    Reachability,
    /// Wire code `"ctl"`.
    Ctl,
    /// Wire code `"ltl"`.
    Ltl,
    /// Wire code `"upper_bounds"`.
    UpperBounds,
    /// Wire code `"stable_marking"`.
    StableMarking,
    /// Wire code `"state_space"`.
    StateSpace,
    /// Wire code `"bad_state"`.
    BadState,
    /// Wire code `"proof_obligation"`.
    ProofObligation,
}

impl PreparedPropertyKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Invariant => "invariant",
            Self::StateConstraint => "state_constraint",
            Self::Deadlock => "deadlock",
            Self::Reachability => "reachability",
            Self::Ctl => "ctl",
            Self::Ltl => "ltl",
            Self::UpperBounds => "upper_bounds",
            Self::StableMarking => "stable_marking",
            Self::StateSpace => "state_space",
            Self::BadState => "bad_state",
            Self::ProofObligation => "proof_obligation",
        }
    }
}

/// Validation requirement before a candidate lane may publish an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedValidationKind {
    /// Wire code `"selftest"`.
    Selftest,
    /// Wire code `"trace_replay"`.
    TraceReplay,
    /// Wire code `"witness_replay"`.
    WitnessReplay,
    /// Wire code `"complete_graph"`.
    CompleteGraph,
    /// Wire code `"scc_certificate"`.
    SccCertificate,
    /// Wire code `"accepting_cycle_certificate"`.
    AcceptingCycleCertificate,
    /// Wire code `"structural_proof"`.
    StructuralProof,
    /// Wire code `"ay_proof"`.
    AYProof,
    /// Wire code `"output_format"`.
    OutputFormat,
}

impl PreparedValidationKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::Selftest => "selftest",
            Self::TraceReplay => "trace_replay",
            Self::WitnessReplay => "witness_replay",
            Self::CompleteGraph => "complete_graph",
            Self::SccCertificate => "scc_certificate",
            Self::AcceptingCycleCertificate => "accepting_cycle_certificate",
            Self::StructuralProof => "structural_proof",
            Self::AYProof => "ay_proof",
            Self::OutputFormat => "output_format",
        }
    }
}

/// Class of analytical solve obligation prepared by a frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedAnalyticalSolveKind {
    /// Wire code `"state_space_cardinality"`.
    StateSpaceCardinality,
    /// Wire code `"deadlock_freedom"`.
    DeadlockFreedom,
    /// Wire code `"reachability"`.
    Reachability,
    /// Wire code `"stable_marking"`.
    StableMarking,
    /// Wire code `"upper_bounds"`.
    UpperBounds,
    /// Wire code `"linear_invariant"`.
    LinearInvariant,
    /// Wire code `"bounded_model_check"`.
    BoundedModelCheck,
    /// Wire code `"pdr_safety"`.
    PdrSafety,
    /// Wire code `"k_induction"`.
    KInduction,
    /// Wire code `"smt_query"`.
    SmtQuery,
    /// Wire code `"sat_query"`.
    SatQuery,
}

impl PreparedAnalyticalSolveKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::StateSpaceCardinality => "state_space_cardinality",
            Self::DeadlockFreedom => "deadlock_freedom",
            Self::Reachability => "reachability",
            Self::StableMarking => "stable_marking",
            Self::UpperBounds => "upper_bounds",
            Self::LinearInvariant => "linear_invariant",
            Self::BoundedModelCheck => "bounded_model_check",
            Self::PdrSafety => "pdr_safety",
            Self::KInduction => "k_induction",
            Self::SmtQuery => "smt_query",
            Self::SatQuery => "sat_query",
        }
    }
}

/// Class of symbolic/proof obligation prepared by a frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedSymbolicProofKind {
    /// Wire code `"initial_condition"`.
    InitialCondition,
    /// Wire code `"transition_relation"`.
    TransitionRelation,
    /// Wire code `"state_predicate"`.
    StatePredicate,
    /// Wire code `"invariant_proof"`.
    InvariantProof,
    /// Wire code `"bounded_model_check"`.
    BoundedModelCheck,
    /// Wire code `"pdr_safety_proof"`.
    PdrSafetyProof,
    /// Wire code `"k_induction"`.
    KInduction,
    /// Wire code `"chc_query"`.
    ChcQuery,
    /// Wire code `"unsat_core"`.
    UnsatCore,
    /// Wire code `"proof_certificate"`.
    ProofCertificate,
    /// Wire code `"model_extraction"`.
    ModelExtraction,
}

impl PreparedSymbolicProofKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::InitialCondition => "initial_condition",
            Self::TransitionRelation => "transition_relation",
            Self::StatePredicate => "state_predicate",
            Self::InvariantProof => "invariant_proof",
            Self::BoundedModelCheck => "bounded_model_check",
            Self::PdrSafetyProof => "pdr_safety_proof",
            Self::KInduction => "k_induction",
            Self::ChcQuery => "chc_query",
            Self::UnsatCore => "unsat_core",
            Self::ProofCertificate => "proof_certificate",
            Self::ModelExtraction => "model_extraction",
        }
    }
}

/// Stable identity carried by prepared artifacts, proofs, witnesses, or models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedCanonicalIdentityKind {
    /// Wire code `"prepared_program"`.
    PreparedProgram,
    /// Wire code `"cache_key"`.
    CacheKey,
    /// Wire code `"frontend_payload"`.
    FrontendPayload,
    /// Wire code `"storage_policy"`.
    StoragePolicy,
    /// Wire code `"fingerprint_policy"`.
    FingerprintPolicy,
    /// Wire code `"batch_artifact"`.
    BatchArtifact,
    /// Wire code `"candidate_lane"`.
    CandidateLane,
    /// Wire code `"lane_artifact"`.
    LaneArtifact,
    /// Wire code `"state_fingerprint"`.
    StateFingerprint,
    /// Wire code `"proof_certificate"`.
    ProofCertificate,
    /// Wire code `"witness_trace"`.
    WitnessTrace,
    /// Wire code `"solver_model"`.
    SolverModel,
    /// Wire code `"canonical_payload"`.
    CanonicalPayload,
}

impl PreparedCanonicalIdentityKind {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::PreparedProgram => "prepared_program",
            Self::CacheKey => "cache_key",
            Self::FrontendPayload => "frontend_payload",
            Self::StoragePolicy => "storage_policy",
            Self::FingerprintPolicy => "fingerprint_policy",
            Self::BatchArtifact => "batch_artifact",
            Self::CandidateLane => "candidate_lane",
            Self::LaneArtifact => "lane_artifact",
            Self::StateFingerprint => "state_fingerprint",
            Self::ProofCertificate => "proof_certificate",
            Self::WitnessTrace => "witness_trace",
            Self::SolverModel => "solver_model",
            Self::CanonicalPayload => "canonical_payload",
        }
    }
}

/// Frontend-neutral state or artifact fingerprint scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparedFingerprintScheme {
    /// Wire code `"tla_fingerprint64"`.
    TlaFingerprint64,
    /// Wire code `"xxh3_u64"`.
    Xxh3U64,
    /// Wire code `"stable_u128"`.
    StableU128,
    /// Wire code `"canonical_bytes_sha256"`.
    CanonicalBytesSha256,
    /// Wire code `"solver_model_digest"`.
    SolverModelDigest,
}

impl PreparedFingerprintScheme {
    /// Stable lowercase wire code for this value.
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaFingerprint64 => "tla_fingerprint64",
            Self::Xxh3U64 => "xxh3_u64",
            Self::StableU128 => "stable_u128",
            Self::CanonicalBytesSha256 => "canonical_bytes_sha256",
            Self::SolverModelDigest => "solver_model_digest",
        }
    }
}

/// A canonical-identity descriptor: how an artifact's identity is canonicalized
/// and optionally digested.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedCanonicalIdentityDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Canonical-identity scheme kind.
    pub kind: PreparedCanonicalIdentityKind,
    /// Canonicalization version string.
    pub canonicalization_version: String,
    /// Digest algorithm, when a digest is attached.
    pub digest_algorithm: Option<String>,
    /// Digest value, when attached.
    pub digest: Option<String>,
}

impl PreparedCanonicalIdentityDescriptor {
    /// Create a descriptor with the given id, kind, and canonicalization version
    /// and no digest.
    pub fn new(
        id: impl Into<String>,
        kind: PreparedCanonicalIdentityKind,
        canonicalization_version: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            canonicalization_version: canonicalization_version.into(),
            digest_algorithm: None,
            digest: None,
        }
    }

    /// Attach a digest algorithm and value.
    pub fn with_digest(
        mut self,
        digest_algorithm: impl Into<String>,
        digest: impl Into<String>,
    ) -> Self {
        self.digest_algorithm = Some(digest_algorithm.into());
        self.digest = Some(digest.into());
        self
    }
}

/// The set of identities binding a prepared payload to its source, config, and
/// descriptor fingerprints.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PreparedPayloadIdentityDescriptor {
    /// Canonical payload identity.
    pub canonical_payload_identity: Option<String>,
    /// Identity of the original source.
    pub source_identity: Option<String>,
    /// Identity of the resolved configuration.
    pub config_identity: Option<String>,
    /// Identity of the examination/analysis input.
    pub examination_identity: Option<String>,
    /// Fingerprint of the frontend-produced payload.
    pub frontend_payload_fingerprint: Option<String>,
    /// Fingerprint of the transition descriptor.
    pub transition_descriptor_fingerprint: Option<String>,
    /// Fingerprint of the property descriptor.
    pub property_descriptor_fingerprint: Option<String>,
    /// Fingerprint of the validation plan.
    pub validation_plan_fingerprint: Option<String>,
}

impl PreparedPayloadIdentityDescriptor {
    /// Create a payload identity set with the canonical payload identity set
    /// (empty input is normalized to `None`) and all other fields empty.
    pub fn new(canonical_payload_identity: impl Into<String>) -> Self {
        Self {
            canonical_payload_identity: non_empty_string(canonical_payload_identity.into()),
            source_identity: None,
            config_identity: None,
            examination_identity: None,
            frontend_payload_fingerprint: None,
            transition_descriptor_fingerprint: None,
            property_descriptor_fingerprint: None,
            validation_plan_fingerprint: None,
        }
    }

    /// Set [`canonical_payload_identity`](Self::canonical_payload_identity) (empty clears it).
    pub fn with_canonical_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.canonical_payload_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`source_identity`](Self::source_identity) (empty clears it).
    pub fn with_source_identity(mut self, identity: impl Into<String>) -> Self {
        self.source_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`config_identity`](Self::config_identity) (empty clears it).
    pub fn with_config_identity(mut self, identity: impl Into<String>) -> Self {
        self.config_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`examination_identity`](Self::examination_identity) (empty clears it).
    pub fn with_examination_identity(mut self, identity: impl Into<String>) -> Self {
        self.examination_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`frontend_payload_fingerprint`](Self::frontend_payload_fingerprint) (empty clears it).
    pub fn with_frontend_payload_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.frontend_payload_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`transition_descriptor_fingerprint`](Self::transition_descriptor_fingerprint) (empty clears it).
    pub fn with_transition_descriptor_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
    ) -> Self {
        self.transition_descriptor_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`property_descriptor_fingerprint`](Self::property_descriptor_fingerprint) (empty clears it).
    pub fn with_property_descriptor_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.property_descriptor_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`validation_plan_fingerprint`](Self::validation_plan_fingerprint) (empty clears it).
    pub fn with_validation_plan_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.validation_plan_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Whether every payload-identity field is populated.
    pub fn has_required_payload_identity(&self) -> bool {
        self.canonical_payload_identity.is_some()
            && self.source_identity.is_some()
            && self.config_identity.is_some()
            && self.examination_identity.is_some()
            && self.frontend_payload_fingerprint.is_some()
            && self.transition_descriptor_fingerprint.is_some()
            && self.property_descriptor_fingerprint.is_some()
            && self.validation_plan_fingerprint.is_some()
    }
}

/// A fingerprint-scheme descriptor for a prepared transition or property.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedFingerprintDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Fingerprinting scheme.
    pub scheme: PreparedFingerprintScheme,
    /// Canonicalization version string.
    pub canonicalization_version: String,
    /// Artifact identity fields attached to the fingerprint.
    pub identities: CheckerArtifactIdentityFields,
}

impl PreparedFingerprintDescriptor {
    /// Create a descriptor with the given id, scheme, and canonicalization
    /// version and no identities.
    pub fn new(
        id: impl Into<String>,
        scheme: PreparedFingerprintScheme,
        canonicalization_version: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            scheme,
            canonicalization_version: canonicalization_version.into(),
            identities: CheckerArtifactIdentityFields::default(),
        }
    }

    /// Replace the whole [`identities`](Self::identities) set.
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = identities;
        self
    }

    /// Set the cache key on [`identities`](Self::identities).
    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.identities = self.identities.with_cache_key(cache_key);
        self
    }

    /// Set the source fingerprint on [`identities`](Self::identities).
    pub fn with_source_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_source_fingerprint(fingerprint);
        self
    }

    /// Set the frontend payload identity on [`identities`](Self::identities).
    pub fn with_frontend_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_frontend_payload_identity(identity);
        self
    }

    /// Set the prepared-program fingerprint on [`identities`](Self::identities).
    pub fn with_prepared_program_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_prepared_program_fingerprint(fingerprint);
        self
    }

    /// Set the artifact identity on [`identities`](Self::identities).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_identity(identity);
        self
    }

    /// Set the artifact fingerprint on [`identities`](Self::identities).
    pub fn with_artifact_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_fingerprint(fingerprint);
        self
    }

    /// Set the storage-policy identity on [`identities`](Self::identities).
    pub fn with_storage_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_policy_identity(identity);
        self
    }

    /// Set the storage-layout fingerprint on [`identities`](Self::identities).
    pub fn with_storage_layout_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_layout_fingerprint(fingerprint);
        self
    }

    /// Set the fingerprint-policy identity on [`identities`](Self::identities).
    pub fn with_fingerprint_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_policy_identity(identity);
        self
    }

    /// Set the fingerprint identity on [`identities`](Self::identities).
    pub fn with_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_identity(identity);
        self
    }
}

/// An analytical-solve obligation a prepared program may discharge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedAnalyticalSolveDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Analytical-solve kind.
    pub kind: PreparedAnalyticalSolveKind,
    /// Problem class addressed.
    pub problem: ProblemKind,
}

impl PreparedAnalyticalSolveDescriptor {
    /// Create a descriptor with the given id, kind, and problem.
    pub fn new(
        id: impl Into<String>,
        kind: PreparedAnalyticalSolveKind,
        problem: ProblemKind,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            problem,
        }
    }
}

/// A symbolic-proof obligation a prepared program may discharge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedSymbolicProofDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Symbolic-proof kind.
    pub kind: PreparedSymbolicProofKind,
    /// Problem class addressed.
    pub problem: ProblemKind,
}

impl PreparedSymbolicProofDescriptor {
    /// Create a descriptor with the given id, kind, and problem.
    pub fn new(
        id: impl Into<String>,
        kind: PreparedSymbolicProofKind,
        problem: ProblemKind,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            problem,
        }
    }
}

/// Solver/backend family that may discharge prepared analytical or proof work.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedBackendFamilyDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Backend family.
    pub backend: BackendKind,
    /// Problem class addressed.
    pub problem: ProblemKind,
    /// Solver facets the family exposes.
    pub facets: Vec<SolverFacet>,
}

impl PreparedBackendFamilyDescriptor {
    /// Create a descriptor with the given id, backend, and problem and no facets.
    pub fn new(id: impl Into<String>, backend: BackendKind, problem: ProblemKind) -> Self {
        Self {
            id: id.into(),
            backend,
            problem,
            facets: Vec::new(),
        }
    }

    /// Add a solver facet (de-duplicated).
    pub fn with_facet(mut self, facet: SolverFacet) -> Self {
        if !self.facets.contains(&facet) {
            self.facets.push(facet);
        }
        self
    }
}

/// A prepared transition (action) with an optional fingerprint scheme.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedTransitionDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Transition kind.
    pub kind: PreparedTransitionKind,
    /// Fingerprint scheme for states reached by this transition, if any.
    pub fingerprint: Option<PreparedFingerprintDescriptor>,
}

impl PreparedTransitionDescriptor {
    /// Create a descriptor with the given id and kind and no fingerprint.
    pub fn new(id: impl Into<String>, kind: PreparedTransitionKind) -> Self {
        Self {
            id: id.into(),
            kind,
            fingerprint: None,
        }
    }

    /// Attach a fingerprint descriptor.
    pub fn with_fingerprint(mut self, fingerprint: PreparedFingerprintDescriptor) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }
}

/// A prepared property (invariant/temporal) with an optional fingerprint scheme.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedPropertyDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Property kind.
    pub kind: PreparedPropertyKind,
    /// Fingerprint scheme for the property, if any.
    pub fingerprint: Option<PreparedFingerprintDescriptor>,
}

impl PreparedPropertyDescriptor {
    /// Create a descriptor with the given id and kind and no fingerprint.
    pub fn new(id: impl Into<String>, kind: PreparedPropertyKind) -> Self {
        Self {
            id: id.into(),
            kind,
            fingerprint: None,
        }
    }

    /// Attach a fingerprint descriptor.
    pub fn with_fingerprint(mut self, fingerprint: PreparedFingerprintDescriptor) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }
}

/// Optional frontend extension descriptor for non-native prepared payloads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedFrontendExtensionDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Frontend extension family.
    pub kind: PreparedFrontendExtensionKind,
    /// Payload kind produced (derived from `kind` by default).
    pub payload_kind: PreparedProgramPayloadKind,
    /// Source kind (derived from `kind` by default).
    pub source_kind: CheckerSourceKind,
    /// Storage kind (derived from `kind` by default).
    pub storage_kind: PreparedStorageKind,
    /// Problem class addressed.
    pub problem: ProblemKind,
    /// Artifact identity fields.
    pub identities: CheckerArtifactIdentityFields,
}

impl PreparedFrontendExtensionDescriptor {
    /// Create a descriptor for `kind`/`problem`, deriving the payload, source,
    /// and storage kinds from `kind`.
    pub fn new(
        id: impl Into<String>,
        kind: PreparedFrontendExtensionKind,
        problem: ProblemKind,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            payload_kind: kind.payload_kind(),
            source_kind: kind.source_kind(),
            storage_kind: kind.storage_kind(),
            problem,
            identities: CheckerArtifactIdentityFields::default(),
        }
    }

    /// Override [`payload_kind`](Self::payload_kind), re-deriving the source kind.
    pub fn with_payload_kind(mut self, payload_kind: PreparedProgramPayloadKind) -> Self {
        self.payload_kind = payload_kind;
        self.source_kind = payload_kind.source_kind();
        self
    }

    /// Override [`source_kind`](Self::source_kind).
    pub fn with_source_kind(mut self, source_kind: CheckerSourceKind) -> Self {
        self.source_kind = source_kind;
        self
    }

    /// Override [`storage_kind`](Self::storage_kind).
    pub fn with_storage_kind(mut self, storage_kind: PreparedStorageKind) -> Self {
        self.storage_kind = storage_kind;
        self
    }

    /// Replace the whole [`identities`](Self::identities) set.
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = identities;
        self
    }

    /// Set the cache key on [`identities`](Self::identities).
    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.identities = self.identities.with_cache_key(cache_key);
        self
    }

    /// Set the source fingerprint on [`identities`](Self::identities).
    pub fn with_source_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_source_fingerprint(fingerprint);
        self
    }

    /// Set the frontend payload identity on [`identities`](Self::identities).
    pub fn with_frontend_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_frontend_payload_identity(identity);
        self
    }

    /// Set the prepared-program fingerprint on [`identities`](Self::identities).
    pub fn with_prepared_program_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_prepared_program_fingerprint(fingerprint);
        self
    }

    /// Set the artifact identity on [`identities`](Self::identities).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_identity(identity);
        self
    }

    /// Set the artifact fingerprint on [`identities`](Self::identities).
    pub fn with_artifact_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_fingerprint(fingerprint);
        self
    }

    /// Set the storage-policy identity on [`identities`](Self::identities).
    pub fn with_storage_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_policy_identity(identity);
        self
    }

    /// Set the storage-layout fingerprint on [`identities`](Self::identities).
    pub fn with_storage_layout_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_layout_fingerprint(fingerprint);
        self
    }

    /// Set the fingerprint-policy identity on [`identities`](Self::identities).
    pub fn with_fingerprint_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_policy_identity(identity);
        self
    }

    /// Set the fingerprint identity on [`identities`](Self::identities).
    pub fn with_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_identity(identity);
        self
    }
}

/// Candidate execution lane offered by a prepared program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedCandidateLaneDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Execution lane this candidate runs in.
    pub lane: SetupTraceLaneKind,
    /// Candidate key distinguishing competing lanes, when set.
    pub candidate_key: Option<String>,
    /// Per-lane artifact identity fields.
    pub identities: CheckerArtifactIdentityFields,
}

impl PreparedCandidateLaneDescriptor {
    /// Create a candidate lane descriptor for `lane` with no candidate key.
    pub fn new(id: impl Into<String>, lane: SetupTraceLaneKind) -> Self {
        Self {
            id: id.into(),
            lane,
            candidate_key: None,
            identities: CheckerArtifactIdentityFields::default(),
        }
    }

    /// Set [`candidate_key`](Self::candidate_key) (empty clears it to `None`).
    pub fn with_candidate_key(mut self, candidate_key: impl Into<String>) -> Self {
        self.candidate_key = non_empty_string(candidate_key.into());
        self
    }

    /// Replace the whole [`identities`](Self::identities) set.
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = identities;
        self
    }

    /// Set the cache key on [`identities`](Self::identities).
    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.identities = self.identities.with_cache_key(cache_key);
        self
    }

    /// Set the frontend payload identity on [`identities`](Self::identities).
    pub fn with_frontend_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_frontend_payload_identity(identity);
        self
    }

    /// Set the artifact identity on [`identities`](Self::identities).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_identity(identity);
        self
    }

    /// Set the storage-policy identity on [`identities`](Self::identities).
    pub fn with_storage_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_policy_identity(identity);
        self
    }

    /// Set the fingerprint-policy identity on [`identities`](Self::identities).
    pub fn with_fingerprint_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_policy_identity(identity);
        self
    }

    /// Set the fingerprint identity on [`identities`](Self::identities).
    pub fn with_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_identity(identity);
        self
    }

    /// Set the batch-artifact identity on [`identities`](Self::identities).
    pub fn with_batch_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_batch_artifact_identity(identity);
        self
    }

    /// Set the candidate identity on [`identities`](Self::identities).
    pub fn with_candidate_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_candidate_identity(identity);
        self
    }

    /// Set the lane identity on [`identities`](Self::identities).
    pub fn with_lane_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_lane_identity(identity);
        self
    }
}

/// Validation requirement plus optional fingerprint identity for one plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedValidationPlanDescriptor {
    /// Stable descriptor id.
    pub id: String,
    /// Validation kind required.
    pub kind: PreparedValidationKind,
    /// Problem class addressed.
    pub problem: ProblemKind,
    /// Whether the validation is required (default `true`).
    pub required: bool,
    /// Whether a missing/failed validation fails closed (default `true`).
    pub fail_closed: bool,
    /// Fingerprint scheme for the validation artifact, if any.
    pub fingerprint: Option<PreparedFingerprintDescriptor>,
    /// Artifact identity fields.
    pub identities: CheckerArtifactIdentityFields,
}

impl PreparedValidationPlanDescriptor {
    /// Create a required, fail-closed validation plan for `kind`/`problem`.
    pub fn new(id: impl Into<String>, kind: PreparedValidationKind, problem: ProblemKind) -> Self {
        Self {
            id: id.into(),
            kind,
            problem,
            required: true,
            fail_closed: true,
            fingerprint: None,
            identities: CheckerArtifactIdentityFields::default(),
        }
    }

    /// Mark the plan optional (clears both `required` and `fail_closed`).
    pub fn optional(mut self) -> Self {
        self.required = false;
        self.fail_closed = false;
        self
    }

    /// Set [`required`](Self::required).
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set [`fail_closed`](Self::fail_closed).
    pub fn with_fail_closed(mut self, fail_closed: bool) -> Self {
        self.fail_closed = fail_closed;
        self
    }

    /// Attach a fingerprint descriptor.
    pub fn with_fingerprint(mut self, fingerprint: PreparedFingerprintDescriptor) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }

    /// Replace the whole [`identities`](Self::identities) set.
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = identities;
        self
    }

    /// Set the cache key on [`identities`](Self::identities).
    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.identities = self.identities.with_cache_key(cache_key);
        self
    }

    /// Set the artifact identity on [`identities`](Self::identities).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_identity(identity);
        self
    }

    /// Set the fingerprint-policy identity on [`identities`](Self::identities).
    pub fn with_fingerprint_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_policy_identity(identity);
        self
    }

    /// Set the fingerprint identity on [`identities`](Self::identities).
    pub fn with_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_identity(identity);
        self
    }
}

/// One prepared checker run. This is the shared identity shape consumed by
/// native, symbolic, proof, replay, and output lanes.
///
/// A program is built incrementally with the `with_*`/`add_*` builders and then
/// handed to a lane. It carries the program identity, payload/storage kinds, the
/// payload-identity bindings, and the descriptors (transitions, properties,
/// analytical solves, symbolic proofs, backend families, candidate lanes, and
/// validation plans) that lanes need to execute and to attribute evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCheckerProgram {
    /// Stable program identity.
    pub identity: String,
    /// Source/interchange family (derived from `payload_kind`).
    pub source_kind: CheckerSourceKind,
    /// Prepared-program payload kind.
    pub payload_kind: PreparedProgramPayloadKind,
    /// Storage kind backing the program.
    pub storage_kind: PreparedStorageKind,
    /// Payload-identity bindings (source, config, descriptor fingerprints).
    pub payload_identity: PreparedPayloadIdentityDescriptor,
    /// Program-level artifact identity fields.
    pub identities: CheckerArtifactIdentityFields,
    /// Prepared transitions (actions).
    pub transitions: Vec<PreparedTransitionDescriptor>,
    /// Prepared properties (invariants/temporal).
    pub properties: Vec<PreparedPropertyDescriptor>,
    /// Prepared analytical-solve obligations.
    pub analytical_solves: Vec<PreparedAnalyticalSolveDescriptor>,
    /// Prepared symbolic-proof obligations.
    pub symbolic_proofs: Vec<PreparedSymbolicProofDescriptor>,
    /// Backend families that may discharge the obligations.
    pub backend_families: Vec<PreparedBackendFamilyDescriptor>,
    /// Program-level fingerprint scheme, if any.
    pub fingerprint: Option<PreparedFingerprintDescriptor>,
    /// Canonical-identity descriptors.
    pub canonical_identities: Vec<PreparedCanonicalIdentityDescriptor>,
    /// Frontend-extension descriptors for non-native payloads.
    pub frontend_extensions: Vec<PreparedFrontendExtensionDescriptor>,
    /// Candidate execution lanes.
    pub candidate_lanes: Vec<PreparedCandidateLaneDescriptor>,
    /// Validation plans.
    pub validation_plans: Vec<PreparedValidationPlanDescriptor>,
    /// Applied validation kinds.
    pub validations: Vec<PreparedValidationKind>,
}

impl PreparedCheckerProgram {
    /// Create an empty program with the given identity, payload kind, and
    /// storage kind (the source kind is derived from `payload_kind`).
    pub fn new(
        identity: impl Into<String>,
        payload_kind: PreparedProgramPayloadKind,
        storage_kind: PreparedStorageKind,
    ) -> Self {
        Self {
            identity: identity.into(),
            source_kind: payload_kind.into(),
            payload_kind,
            storage_kind,
            payload_identity: PreparedPayloadIdentityDescriptor::default(),
            identities: CheckerArtifactIdentityFields::default(),
            transitions: Vec::new(),
            properties: Vec::new(),
            analytical_solves: Vec::new(),
            symbolic_proofs: Vec::new(),
            backend_families: Vec::new(),
            fingerprint: None,
            canonical_identities: Vec::new(),
            frontend_extensions: Vec::new(),
            candidate_lanes: Vec::new(),
            validation_plans: Vec::new(),
            validations: Vec::new(),
        }
    }

    /// Replace the whole program-level [`identities`](Self::identities) set.
    pub fn with_identity_fields(mut self, identities: CheckerArtifactIdentityFields) -> Self {
        self.identities = identities;
        self
    }

    /// Replace the whole [`payload_identity`](Self::payload_identity) descriptor.
    pub fn with_payload_identity(mut self, identity: PreparedPayloadIdentityDescriptor) -> Self {
        self.payload_identity = identity;
        self
    }

    /// Set the canonical payload identity on [`payload_identity`](Self::payload_identity).
    pub fn with_canonical_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.payload_identity = self
            .payload_identity
            .with_canonical_payload_identity(identity);
        self
    }

    /// Set the source identity on [`payload_identity`](Self::payload_identity).
    pub fn with_source_identity(mut self, identity: impl Into<String>) -> Self {
        self.payload_identity = self.payload_identity.with_source_identity(identity);
        self
    }

    /// Set the config identity on [`payload_identity`](Self::payload_identity).
    pub fn with_config_identity(mut self, identity: impl Into<String>) -> Self {
        self.payload_identity = self.payload_identity.with_config_identity(identity);
        self
    }

    /// Set the examination identity on [`payload_identity`](Self::payload_identity).
    pub fn with_examination_identity(mut self, identity: impl Into<String>) -> Self {
        self.payload_identity = self.payload_identity.with_examination_identity(identity);
        self
    }

    /// Set the cache key on [`identities`](Self::identities).
    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.identities = self.identities.with_cache_key(cache_key);
        self
    }

    /// Set the source fingerprint on [`identities`](Self::identities).
    pub fn with_source_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_source_fingerprint(fingerprint);
        self
    }

    /// Set the frontend payload identity on [`identities`](Self::identities).
    pub fn with_frontend_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_frontend_payload_identity(identity);
        self
    }

    /// Set the frontend payload fingerprint on [`payload_identity`](Self::payload_identity).
    pub fn with_frontend_payload_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.payload_identity = self
            .payload_identity
            .with_frontend_payload_fingerprint(fingerprint);
        self
    }

    /// Set the prepared-program fingerprint on [`identities`](Self::identities).
    pub fn with_prepared_program_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_prepared_program_fingerprint(fingerprint);
        self
    }

    /// Set the artifact identity on [`identities`](Self::identities).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_identity(identity);
        self
    }

    /// Set the artifact fingerprint on [`identities`](Self::identities).
    pub fn with_artifact_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_artifact_fingerprint(fingerprint);
        self
    }

    /// Set the storage-policy identity on [`identities`](Self::identities).
    pub fn with_storage_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_policy_identity(identity);
        self
    }

    /// Set the storage-layout fingerprint on [`identities`](Self::identities).
    pub fn with_storage_layout_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_storage_layout_fingerprint(fingerprint);
        self
    }

    /// Set the fingerprint-policy identity on [`identities`](Self::identities).
    pub fn with_fingerprint_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_policy_identity(identity);
        self
    }

    /// Set the fingerprint identity on [`identities`](Self::identities).
    pub fn with_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_fingerprint_identity(identity);
        self
    }

    /// Set the batch-artifact identity on [`identities`](Self::identities).
    pub fn with_batch_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_batch_artifact_identity(identity);
        self
    }

    /// Set the candidate identity on [`identities`](Self::identities).
    pub fn with_candidate_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_candidate_identity(identity);
        self
    }

    /// Set the lane identity on [`identities`](Self::identities).
    pub fn with_lane_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_lane_identity(identity);
        self
    }

    /// Set the transition-descriptor fingerprint on [`payload_identity`](Self::payload_identity).
    pub fn with_transition_descriptor_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
    ) -> Self {
        self.payload_identity = self
            .payload_identity
            .with_transition_descriptor_fingerprint(fingerprint);
        self
    }

    /// Set the property-descriptor fingerprint on [`payload_identity`](Self::payload_identity).
    pub fn with_property_descriptor_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.payload_identity = self
            .payload_identity
            .with_property_descriptor_fingerprint(fingerprint);
        self
    }

    /// Set the validation-plan fingerprint on [`payload_identity`](Self::payload_identity).
    pub fn with_validation_plan_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.payload_identity = self
            .payload_identity
            .with_validation_plan_fingerprint(fingerprint);
        self
    }

    /// Add a transition by id and kind.
    pub fn add_transition(mut self, id: impl Into<String>, kind: PreparedTransitionKind) -> Self {
        self.transitions
            .push(PreparedTransitionDescriptor::new(id, kind));
        self
    }

    /// Add a prepared transition descriptor.
    pub fn add_transition_descriptor(mut self, transition: PreparedTransitionDescriptor) -> Self {
        self.transitions.push(transition);
        self
    }

    /// Add a property by id and kind.
    pub fn add_property(mut self, id: impl Into<String>, kind: PreparedPropertyKind) -> Self {
        self.properties
            .push(PreparedPropertyDescriptor::new(id, kind));
        self
    }

    /// Add a prepared property descriptor.
    pub fn add_property_descriptor(mut self, property: PreparedPropertyDescriptor) -> Self {
        self.properties.push(property);
        self
    }

    /// Add an analytical-solve obligation.
    pub fn add_analytical_solve(
        mut self,
        id: impl Into<String>,
        kind: PreparedAnalyticalSolveKind,
        problem: ProblemKind,
    ) -> Self {
        self.analytical_solves
            .push(PreparedAnalyticalSolveDescriptor::new(id, kind, problem));
        self
    }

    /// Add a symbolic-proof obligation.
    pub fn add_symbolic_proof(
        mut self,
        id: impl Into<String>,
        kind: PreparedSymbolicProofKind,
        problem: ProblemKind,
    ) -> Self {
        self.symbolic_proofs
            .push(PreparedSymbolicProofDescriptor::new(id, kind, problem));
        self
    }

    /// Add a backend-family descriptor.
    pub fn add_backend_family(mut self, family: PreparedBackendFamilyDescriptor) -> Self {
        self.backend_families.push(family);
        self
    }

    /// Set the program-level [`fingerprint`](Self::fingerprint) scheme.
    pub fn with_fingerprint(mut self, fingerprint: PreparedFingerprintDescriptor) -> Self {
        self.fingerprint = Some(fingerprint);
        self
    }

    /// Add a canonical-identity descriptor.
    pub fn add_canonical_identity(mut self, identity: PreparedCanonicalIdentityDescriptor) -> Self {
        self.canonical_identities.push(identity);
        self
    }

    /// Add a frontend-extension descriptor.
    pub fn add_frontend_extension(
        mut self,
        extension: PreparedFrontendExtensionDescriptor,
    ) -> Self {
        self.frontend_extensions.push(extension);
        self
    }

    /// Add a candidate execution lane.
    pub fn add_candidate_lane(mut self, lane: PreparedCandidateLaneDescriptor) -> Self {
        self.candidate_lanes.push(lane);
        self
    }

    /// Add a validation plan, also registering its kind in
    /// [`validations`](Self::validations) (de-duplicated).
    pub fn add_validation_plan(mut self, plan: PreparedValidationPlanDescriptor) -> Self {
        if !self.validations.contains(&plan.kind) {
            self.validations.push(plan.kind);
        }
        self.validation_plans.push(plan);
        self
    }

    /// Register a required validation kind (de-duplicated) without a full plan.
    pub fn require_validation(mut self, kind: PreparedValidationKind) -> Self {
        if !self.validations.contains(&kind) {
            self.validations.push(kind);
        }
        self
    }

    /// Program identities after filling missing fingerprint fields.
    ///
    /// The prepared program owns cache, payload, storage, batch, candidate, and
    /// lane identities. The fingerprint descriptor only supplies missing
    /// fingerprint policy/namespace fields.
    pub fn effective_identity_fields(&self) -> CheckerArtifactIdentityFields {
        let mut identities = self.identities.clone();
        if let Some(fingerprint) = &self.fingerprint {
            if identities.fingerprint_policy_identity.is_none() {
                identities.fingerprint_policy_identity =
                    fingerprint.identities.fingerprint_policy_identity.clone();
            }
            if identities.fingerprint_identity.is_none() {
                identities.fingerprint_identity =
                    fingerprint.identities.fingerprint_identity.clone();
            }
        }
        identities
    }

    /// Candidate lane identities with lane fields taking precedence.
    ///
    /// This is the deterministic bridge from one prepared program to many
    /// candidate lanes in the shared engine.
    pub fn effective_candidate_lane_identity_fields(
        &self,
        lane: &PreparedCandidateLaneDescriptor,
    ) -> CheckerArtifactIdentityFields {
        lane.identities
            .merged_with_fallback(&self.effective_identity_fields())
    }

    /// Builds the setup trace key for one prepared candidate lane.
    pub fn setup_trace_key_for_candidate_lane(
        &self,
        lane: &PreparedCandidateLaneDescriptor,
    ) -> SetupTraceKey {
        let mut key = SetupTraceKey::new(self.source_kind, lane.lane)
            .with_identity_fields(self.effective_candidate_lane_identity_fields(lane));
        key.candidate_key.clone_from(&lane.candidate_key);
        key
    }

    /// Builds setup trace keys for all candidate lanes in declaration order.
    pub fn setup_trace_keys_for_candidate_lanes(&self) -> Vec<SetupTraceKey> {
        self.candidate_lanes
            .iter()
            .map(|lane| self.setup_trace_key_for_candidate_lane(lane))
            .collect()
    }

    /// Identities for one validation plan: plan identities, then the plan's
    /// fingerprint identities, then the program's effective identities, each as
    /// fallback for missing fields.
    pub fn effective_validation_plan_identity_fields(
        &self,
        plan: &PreparedValidationPlanDescriptor,
    ) -> CheckerArtifactIdentityFields {
        let mut identities = plan.identities.clone();
        if let Some(fingerprint) = &plan.fingerprint {
            identities = identities.merged_with_fallback(&fingerprint.identities);
        }
        identities.merged_with_fallback(&self.effective_identity_fields())
    }

    /// Whether the program carries every required payload-identity field plus a
    /// frontend payload identity and a storage-layout fingerprint.
    pub fn has_required_payload_identity(&self) -> bool {
        let identities = self.effective_identity_fields();
        self.payload_identity.has_required_payload_identity()
            && identities.frontend_payload_identity.is_some()
            && identities.storage_layout_fingerprint.is_some()
    }

    /// Validate the program's payload-identity contract.
    ///
    /// # Errors
    ///
    /// Returns the name of the first missing required payload-identity or
    /// program-identity field.
    pub fn validate_payload_identity_contract(&self) -> Result<(), String> {
        let identities = self.effective_identity_fields();
        require_option_field(
            &self.payload_identity.canonical_payload_identity,
            "canonical_payload_identity",
        )?;
        require_option_field(&self.payload_identity.source_identity, "source_identity")?;
        require_option_field(&self.payload_identity.config_identity, "config_identity")?;
        require_option_field(
            &self.payload_identity.examination_identity,
            "examination_identity",
        )?;
        require_option_field(
            &identities.frontend_payload_identity,
            "frontend_payload_identity",
        )?;
        require_option_field(
            &self.payload_identity.frontend_payload_fingerprint,
            "frontend_payload_fingerprint",
        )?;
        require_option_field(
            &identities.storage_layout_fingerprint,
            "storage_layout_fingerprint",
        )?;
        require_option_field(
            &self.payload_identity.transition_descriptor_fingerprint,
            "transition_descriptor_fingerprint",
        )?;
        require_option_field(
            &self.payload_identity.property_descriptor_fingerprint,
            "property_descriptor_fingerprint",
        )?;
        require_option_field(
            &self.payload_identity.validation_plan_fingerprint,
            "validation_plan_fingerprint",
        )?;
        Ok(())
    }

    /// Renders one prepared-program summary row for shared-engine evidence.
    ///
    /// The row uses `source_kind` and `frontend_kind` consistently with setup
    /// trace and candidate-lane rows.
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let fingerprint_scheme = self
            .fingerprint
            .as_ref()
            .map(|fingerprint| fingerprint.scheme.code())
            .unwrap_or("none");
        let fingerprint_id = self
            .fingerprint
            .as_ref()
            .map(|fingerprint| evidence_value(&fingerprint.id))
            .unwrap_or_else(|| "none".to_string());
        let identities = self.effective_identity_fields();
        let payload_identity = &self.payload_identity;
        format!(
            "{} prepared_checker_program identity={} source_kind={} frontend_kind={} payload_kind={} storage_kind={} canonical_payload_identity={} source_identity={} config_identity={} examination_identity={} cache_key={} source_fingerprint={} frontend_payload_identity={} frontend_payload_fingerprint={} prepared_program_fingerprint={} artifact_identity={} artifact_fingerprint={} storage_policy_identity={} storage_layout_fingerprint={} fingerprint_policy_identity={} fingerprint_identity={} batch_artifact_identity={} candidate_identity={} lane_identity={} transition_descriptor_fingerprint={} property_descriptor_fingerprint={} validation_plan_fingerprint={} transitions={} properties={} analytical_solves={} symbolic_proofs={} backend_families={} fingerprint_id={} fingerprint_scheme={} canonical_identities={} frontend_extensions={} candidate_lanes={} validation_plans={} validations={}",
            scope,
            evidence_value(&self.identity),
            self.source_kind.code(),
            self.source_kind.code(),
            self.payload_kind.code(),
            self.storage_kind.code(),
            evidence_optional(payload_identity.canonical_payload_identity.as_deref()),
            evidence_optional(payload_identity.source_identity.as_deref()),
            evidence_optional(payload_identity.config_identity.as_deref()),
            evidence_optional(payload_identity.examination_identity.as_deref()),
            evidence_optional(identities.cache_key.as_deref()),
            evidence_optional(identities.source_fingerprint.as_deref()),
            evidence_optional(identities.frontend_payload_identity.as_deref()),
            evidence_optional(payload_identity.frontend_payload_fingerprint.as_deref()),
            evidence_optional(identities.prepared_program_fingerprint.as_deref()),
            evidence_optional(identities.artifact_identity.as_deref()),
            evidence_optional(identities.artifact_fingerprint.as_deref()),
            evidence_optional(identities.storage_policy_identity.as_deref()),
            evidence_optional(identities.storage_layout_fingerprint.as_deref()),
            evidence_optional(identities.fingerprint_policy_identity.as_deref()),
            evidence_optional(identities.fingerprint_identity.as_deref()),
            evidence_optional(identities.batch_artifact_identity.as_deref()),
            evidence_optional(identities.candidate_identity.as_deref()),
            evidence_optional(identities.lane_identity.as_deref()),
            evidence_optional(
                payload_identity
                    .transition_descriptor_fingerprint
                    .as_deref()
            ),
            evidence_optional(
                payload_identity
                    .property_descriptor_fingerprint
                    .as_deref()
            ),
            evidence_optional(payload_identity.validation_plan_fingerprint.as_deref()),
            self.transitions.len(),
            self.properties.len(),
            self.analytical_solves.len(),
            self.symbolic_proofs.len(),
            self.backend_families.len(),
            fingerprint_id,
            fingerprint_scheme,
            self.canonical_identities.len(),
            self.frontend_extensions.len(),
            self.candidate_lanes.len(),
            self.validation_plans.len(),
            self.validations.len()
        )
    }

    /// Renders one row per frontend extension descriptor.
    pub fn render_frontend_extension_evidence_rows(&self, scope: &str) -> Vec<String> {
        self.frontend_extensions
            .iter()
            .map(|extension| {
                let identities = extension
                    .identities
                    .merged_with_fallback(&self.effective_identity_fields());
                let payload_identity = &self.payload_identity;
                format!(
                    "{} prepared_frontend_extension identity={} source_kind={} frontend_kind={} payload_kind={} storage_kind={} extension_kind={} extension_source_kind={} extension_payload_kind={} extension_storage_kind={} problem={} canonical_payload_identity={} source_identity={} config_identity={} examination_identity={} cache_key={} source_fingerprint={} frontend_payload_identity={} frontend_payload_fingerprint={} prepared_program_fingerprint={} artifact_identity={} artifact_fingerprint={} storage_policy_identity={} storage_layout_fingerprint={} fingerprint_policy_identity={} fingerprint_identity={} batch_artifact_identity={} candidate_identity={} lane_identity={} transition_descriptor_fingerprint={} property_descriptor_fingerprint={} validation_plan_fingerprint={}",
                    scope,
                    evidence_value(&extension.id),
                    self.source_kind.code(),
                    self.source_kind.code(),
                    self.payload_kind.code(),
                    self.storage_kind.code(),
                    extension.kind.code(),
                    extension.source_kind.code(),
                    extension.payload_kind.code(),
                    extension.storage_kind.code(),
                    extension.problem.code(),
                    evidence_optional(payload_identity.canonical_payload_identity.as_deref()),
                    evidence_optional(payload_identity.source_identity.as_deref()),
                    evidence_optional(payload_identity.config_identity.as_deref()),
                    evidence_optional(payload_identity.examination_identity.as_deref()),
                    evidence_optional(identities.cache_key.as_deref()),
                    evidence_optional(identities.source_fingerprint.as_deref()),
                    evidence_optional(identities.frontend_payload_identity.as_deref()),
                    evidence_optional(payload_identity.frontend_payload_fingerprint.as_deref()),
                    evidence_optional(identities.prepared_program_fingerprint.as_deref()),
                    evidence_optional(identities.artifact_identity.as_deref()),
                    evidence_optional(identities.artifact_fingerprint.as_deref()),
                    evidence_optional(identities.storage_policy_identity.as_deref()),
                    evidence_optional(identities.storage_layout_fingerprint.as_deref()),
                    evidence_optional(identities.fingerprint_policy_identity.as_deref()),
                    evidence_optional(identities.fingerprint_identity.as_deref()),
                    evidence_optional(identities.batch_artifact_identity.as_deref()),
                    evidence_optional(identities.candidate_identity.as_deref()),
                    evidence_optional(identities.lane_identity.as_deref()),
                    evidence_optional(
                        payload_identity
                            .transition_descriptor_fingerprint
                            .as_deref()
                    ),
                    evidence_optional(
                        payload_identity
                            .property_descriptor_fingerprint
                            .as_deref()
                    ),
                    evidence_optional(payload_identity.validation_plan_fingerprint.as_deref())
                )
            })
            .collect()
    }

    /// Renders one row per prepared candidate lane.
    ///
    /// TLA, MCC, AY, replay, and future adapters can consume these rows without
    /// frontend-specific payload types; lane identity fields are already merged
    /// with program/fingerprint identities using the same precedence as setup
    /// trace keys.
    pub fn render_candidate_lane_evidence_rows(&self, scope: &str) -> Vec<String> {
        let fingerprint_scheme = self
            .fingerprint
            .as_ref()
            .map(|fingerprint| fingerprint.scheme.code())
            .unwrap_or("none");
        let fingerprint_id = self
            .fingerprint
            .as_ref()
            .map(|fingerprint| evidence_value(&fingerprint.id))
            .unwrap_or_else(|| "none".to_string());

        self.candidate_lanes
            .iter()
            .map(|lane| {
                let identities = self.effective_candidate_lane_identity_fields(lane);
                let payload_identity = &self.payload_identity;
                format!(
                    "{} prepared_candidate_lane identity={} source_kind={} frontend_kind={} payload_kind={} storage_kind={} lane_kind={} lane={} candidate_key={} candidate_identity={} lane_identity={} canonical_payload_identity={} source_identity={} config_identity={} examination_identity={} cache_key={} source_fingerprint={} frontend_payload_identity={} frontend_payload_fingerprint={} prepared_program_fingerprint={} artifact_identity={} artifact_fingerprint={} storage_policy_identity={} storage_layout_fingerprint={} fingerprint_policy_identity={} fingerprint_identity={} batch_artifact_identity={} transition_descriptor_fingerprint={} property_descriptor_fingerprint={} validation_plan_fingerprint={} fingerprint_id={} fingerprint_scheme={}",
                    scope,
                    evidence_value(&lane.id),
                    self.source_kind.code(),
                    self.source_kind.code(),
                    self.payload_kind.code(),
                    self.storage_kind.code(),
                    lane.lane.code(),
                    lane.lane.code(),
                    evidence_optional(lane.candidate_key.as_deref()),
                    evidence_optional(identities.candidate_identity.as_deref()),
                    evidence_optional(identities.lane_identity.as_deref()),
                    evidence_optional(payload_identity.canonical_payload_identity.as_deref()),
                    evidence_optional(payload_identity.source_identity.as_deref()),
                    evidence_optional(payload_identity.config_identity.as_deref()),
                    evidence_optional(payload_identity.examination_identity.as_deref()),
                    evidence_optional(identities.cache_key.as_deref()),
                    evidence_optional(identities.source_fingerprint.as_deref()),
                    evidence_optional(identities.frontend_payload_identity.as_deref()),
                    evidence_optional(payload_identity.frontend_payload_fingerprint.as_deref()),
                    evidence_optional(identities.prepared_program_fingerprint.as_deref()),
                    evidence_optional(identities.artifact_identity.as_deref()),
                    evidence_optional(identities.artifact_fingerprint.as_deref()),
                    evidence_optional(identities.storage_policy_identity.as_deref()),
                    evidence_optional(identities.storage_layout_fingerprint.as_deref()),
                    evidence_optional(identities.fingerprint_policy_identity.as_deref()),
                    evidence_optional(identities.fingerprint_identity.as_deref()),
                    evidence_optional(identities.batch_artifact_identity.as_deref()),
                    evidence_optional(
                        payload_identity
                            .transition_descriptor_fingerprint
                            .as_deref()
                    ),
                    evidence_optional(
                        payload_identity
                            .property_descriptor_fingerprint
                            .as_deref()
                    ),
                    evidence_optional(payload_identity.validation_plan_fingerprint.as_deref()),
                    fingerprint_id,
                    fingerprint_scheme
                )
            })
            .collect()
    }

    /// Renders one row per validation plan descriptor.
    pub fn render_validation_plan_evidence_rows(&self, scope: &str) -> Vec<String> {
        self.validation_plans
            .iter()
            .map(|plan| {
                let identities = self.effective_validation_plan_identity_fields(plan);
                let payload_identity = &self.payload_identity;
                let fingerprint_id = plan
                    .fingerprint
                    .as_ref()
                    .map(|fingerprint| evidence_value(&fingerprint.id))
                    .unwrap_or_else(|| "none".to_string());
                let fingerprint_scheme = plan
                    .fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.scheme.code())
                    .unwrap_or("none");
                let fingerprint_canonicalization = plan
                    .fingerprint
                    .as_ref()
                    .map(|fingerprint| evidence_value(&fingerprint.canonicalization_version))
                    .unwrap_or_else(|| "none".to_string());
                format!(
                    "{} prepared_validation_plan identity={} source_kind={} frontend_kind={} payload_kind={} storage_kind={} validation_kind={} problem={} required={} fail_closed={} fingerprint_id={} fingerprint_scheme={} fingerprint_canonicalization={} canonical_payload_identity={} source_identity={} config_identity={} examination_identity={} cache_key={} source_fingerprint={} frontend_payload_identity={} frontend_payload_fingerprint={} prepared_program_fingerprint={} artifact_identity={} artifact_fingerprint={} storage_policy_identity={} storage_layout_fingerprint={} fingerprint_policy_identity={} fingerprint_identity={} batch_artifact_identity={} candidate_identity={} lane_identity={} transition_descriptor_fingerprint={} property_descriptor_fingerprint={} validation_plan_fingerprint={}",
                    scope,
                    evidence_value(&plan.id),
                    self.source_kind.code(),
                    self.source_kind.code(),
                    self.payload_kind.code(),
                    self.storage_kind.code(),
                    plan.kind.code(),
                    plan.problem.code(),
                    plan.required,
                    plan.fail_closed,
                    fingerprint_id,
                    fingerprint_scheme,
                    fingerprint_canonicalization,
                    evidence_optional(payload_identity.canonical_payload_identity.as_deref()),
                    evidence_optional(payload_identity.source_identity.as_deref()),
                    evidence_optional(payload_identity.config_identity.as_deref()),
                    evidence_optional(payload_identity.examination_identity.as_deref()),
                    evidence_optional(identities.cache_key.as_deref()),
                    evidence_optional(identities.source_fingerprint.as_deref()),
                    evidence_optional(identities.frontend_payload_identity.as_deref()),
                    evidence_optional(payload_identity.frontend_payload_fingerprint.as_deref()),
                    evidence_optional(identities.prepared_program_fingerprint.as_deref()),
                    evidence_optional(identities.artifact_identity.as_deref()),
                    evidence_optional(identities.artifact_fingerprint.as_deref()),
                    evidence_optional(identities.storage_policy_identity.as_deref()),
                    evidence_optional(identities.storage_layout_fingerprint.as_deref()),
                    evidence_optional(identities.fingerprint_policy_identity.as_deref()),
                    evidence_optional(identities.fingerprint_identity.as_deref()),
                    evidence_optional(identities.batch_artifact_identity.as_deref()),
                    evidence_optional(identities.candidate_identity.as_deref()),
                    evidence_optional(identities.lane_identity.as_deref()),
                    evidence_optional(
                        payload_identity
                            .transition_descriptor_fingerprint
                            .as_deref()
                    ),
                    evidence_optional(
                        payload_identity
                            .property_descriptor_fingerprint
                            .as_deref()
                    ),
                    evidence_optional(payload_identity.validation_plan_fingerprint.as_deref())
                )
            })
            .collect()
    }
}

fn non_empty_string(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn evidence_value(value: &str) -> String {
    if value.is_empty() {
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

/// Validate one prepared-program evidence row against shared vocabulary.
pub fn validate_prepared_checker_program_evidence_row(row: &str) -> Result<(), String> {
    validate_prepared_row_kind(row, PREPARED_CHECKER_PROGRAM_ROW_KIND)?;
    require_prepared_fields(row, PREPARED_CHECKER_PROGRAM_REQUIRED_FIELDS)?;
    validate_common_prepared_vocab(row)?;
    validate_fingerprint_scheme_field(row, "fingerprint_scheme")?;
    for field in [
        "transitions",
        "properties",
        "analytical_solves",
        "symbolic_proofs",
        "backend_families",
        "canonical_identities",
        "frontend_extensions",
        "candidate_lanes",
        "validation_plans",
        "validations",
    ] {
        validate_usize_field(row, field)?;
    }
    require_non_none_field(row, "identity")?;
    Ok(())
}

/// Validate that a payload family is allowed for default prepared-program use.
pub fn validate_prepared_payload_default_use(payload_kind_code: &str) -> Result<(), String> {
    if payload_kind_code == PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE {
        return Err(
            "future_importer is reserved until a registered importer supplies payload identity, layout fingerprints, and validation receipts"
                .to_string(),
        );
    }
    if is_payload_kind_code(payload_kind_code) {
        Ok(())
    } else {
        Err(format!(
            "unknown prepared-program payload kind: {payload_kind_code}"
        ))
    }
}

/// Validate one prepared frontend-extension evidence row against shared vocabulary.
pub fn validate_prepared_frontend_extension_evidence_row(row: &str) -> Result<(), String> {
    validate_prepared_row_kind(row, PREPARED_FRONTEND_EXTENSION_ROW_KIND)?;
    require_prepared_fields(row, PREPARED_FRONTEND_EXTENSION_REQUIRED_FIELDS)?;
    validate_common_prepared_vocab(row)?;
    validate_source_kind_field(row, "extension_source_kind")?;
    validate_payload_kind_field(row, "extension_payload_kind")?;
    validate_storage_kind_field(row, "extension_storage_kind")?;
    validate_frontend_extension_kind_field(row, "extension_kind")?;
    validate_problem_kind_field(row, "problem")?;
    require_non_none_field(row, "identity")?;
    Ok(())
}

/// Validate one prepared candidate-lane evidence row against shared vocabulary.
pub fn validate_prepared_candidate_lane_evidence_row(row: &str) -> Result<(), String> {
    validate_prepared_row_kind(row, PREPARED_CANDIDATE_LANE_ROW_KIND)?;
    require_prepared_fields(row, PREPARED_CANDIDATE_LANE_REQUIRED_FIELDS)?;
    validate_common_prepared_vocab(row)?;
    validate_lane_kind_field(row, "lane_kind")?;
    validate_lane_kind_field(row, "lane")?;
    if prepared_evidence_field(row, "lane_kind") != prepared_evidence_field(row, "lane") {
        return Err("lane_kind must match lane".to_string());
    }
    validate_fingerprint_scheme_field(row, "fingerprint_scheme")?;
    require_non_none_field(row, "identity")?;
    Ok(())
}

/// Validate one prepared validation-plan evidence row against shared vocabulary.
pub fn validate_prepared_validation_plan_evidence_row(row: &str) -> Result<(), String> {
    validate_prepared_row_kind(row, PREPARED_VALIDATION_PLAN_ROW_KIND)?;
    require_prepared_fields(row, PREPARED_VALIDATION_PLAN_REQUIRED_FIELDS)?;
    validate_common_prepared_vocab(row)?;
    validate_validation_kind_field(row, "validation_kind")?;
    validate_problem_kind_field(row, "problem")?;
    validate_bool_field(row, "required")?;
    validate_bool_field(row, "fail_closed")?;
    validate_fingerprint_scheme_field(row, "fingerprint_scheme")?;
    require_non_none_field(row, "identity")?;
    Ok(())
}

fn validate_prepared_row_kind(row: &str, expected_kind: &str) -> Result<(), String> {
    let mut tokens = row.split_whitespace();
    tokens
        .next()
        .ok_or_else(|| "missing evidence scope".to_string())?;
    let kind = tokens
        .next()
        .ok_or_else(|| "missing prepared-program row kind".to_string())?;
    if kind != expected_kind {
        return Err(format!(
            "wrong prepared-program row kind: expected {expected_kind}, got {kind}"
        ));
    }
    Ok(())
}

fn require_prepared_fields(row: &str, fields: &[&'static str]) -> Result<(), String> {
    for field in fields {
        require_prepared_field(row, field)?;
    }
    Ok(())
}

fn validate_common_prepared_vocab(row: &str) -> Result<(), String> {
    validate_source_kind_field(row, "source_kind")?;
    validate_source_kind_field(row, "frontend_kind")?;
    validate_payload_kind_field(row, "payload_kind")?;
    validate_storage_kind_field(row, "storage_kind")?;
    validate_prepared_payload_default_use(require_prepared_field(row, "payload_kind")?)?;
    validate_payload_source_kind_match(row)?;
    require_prepared_payload_identity_fields(row)?;
    Ok(())
}

fn require_prepared_field<'a>(row: &'a str, field: &'static str) -> Result<&'a str, String> {
    prepared_evidence_field(row, field).ok_or_else(|| format!("missing field {field}"))
}

fn require_non_none_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if value == "none" {
        return Err(format!("field {field} must not be none"));
    }
    Ok(())
}

fn require_option_field(value: &Option<String>, field: &'static str) -> Result<(), String> {
    if value.as_deref().is_some_and(|value| !value.is_empty()) {
        Ok(())
    } else {
        Err(format!("field {field} must not be none"))
    }
}

fn require_prepared_payload_identity_fields(row: &str) -> Result<(), String> {
    for field in [
        "canonical_payload_identity",
        "source_identity",
        "config_identity",
        "examination_identity",
        "frontend_payload_identity",
        "frontend_payload_fingerprint",
        "storage_layout_fingerprint",
        "transition_descriptor_fingerprint",
        "property_descriptor_fingerprint",
        "validation_plan_fingerprint",
    ] {
        require_non_none_field(row, field)?;
    }
    Ok(())
}

fn validate_payload_source_kind_match(row: &str) -> Result<(), String> {
    let payload_kind = require_prepared_field(row, "payload_kind")?;
    let source_kind = require_prepared_field(row, "source_kind")?;
    let frontend_kind = require_prepared_field(row, "frontend_kind")?;
    let expected_source_kind = match payload_kind {
        "tla" => "tla",
        "quint" => "quint",
        "mcc_petri" => "mcc_petri",
        "aiger" => "aiger",
        "btor2" => "btor2",
        "vmt_interchange" => "vmt_interchange",
        "ay_only" => "ay_only",
        "witness_replay" => "witness_replay",
        PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE => {
            PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE
        }
        other => {
            return Err(format!(
                "field payload_kind has unknown payload kind: {other}"
            ))
        }
    };
    if source_kind != expected_source_kind {
        return Err(format!(
            "payload_kind {payload_kind} requires source_kind {expected_source_kind}, got {source_kind}"
        ));
    }
    if frontend_kind != expected_source_kind {
        return Err(format!(
            "payload_kind {payload_kind} requires frontend_kind {expected_source_kind}, got {frontend_kind}"
        ));
    }
    Ok(())
}

fn validate_usize_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    value
        .parse::<usize>()
        .map(|_| ())
        .map_err(|_| format!("field {field} is not a nonnegative integer: {value}"))
}

fn validate_bool_field(row: &str, field: &'static str) -> Result<(), String> {
    match require_prepared_field(row, field)? {
        "true" | "false" => Ok(()),
        value => Err(format!("field {field} is not boolean: {value}")),
    }
}

fn validate_source_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_source_kind_code(value) {
        Ok(())
    } else {
        Err(format!("field {field} has unknown source kind: {value}"))
    }
}

fn validate_payload_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_payload_kind_code(value) {
        Ok(())
    } else {
        Err(format!("field {field} has unknown payload kind: {value}"))
    }
}

fn validate_storage_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_storage_kind_code(value) {
        Ok(())
    } else {
        Err(format!("field {field} has unknown storage kind: {value}"))
    }
}

fn validate_frontend_extension_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_frontend_extension_kind_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field {field} has unknown frontend extension kind: {value}"
        ))
    }
}

fn validate_lane_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_lane_kind_code(value) {
        Ok(())
    } else {
        Err(format!("field {field} has unknown lane kind: {value}"))
    }
}

fn validate_problem_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_problem_kind_code(value) {
        Ok(())
    } else {
        Err(format!("field {field} has unknown problem kind: {value}"))
    }
}

fn validate_validation_kind_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if is_validation_kind_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field {field} has unknown validation kind: {value}"
        ))
    }
}

fn validate_fingerprint_scheme_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_prepared_field(row, field)?;
    if value == "none" || is_fingerprint_scheme_code(value) {
        Ok(())
    } else {
        Err(format!(
            "field {field} has unknown fingerprint scheme: {value}"
        ))
    }
}

fn is_source_kind_code(value: &str) -> bool {
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
            | PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE
            | "unknown"
    )
}

fn is_payload_kind_code(value: &str) -> bool {
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
            | PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE
    )
}

fn is_storage_kind_code(value: &str) -> bool {
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

fn is_frontend_extension_kind_code(value: &str) -> bool {
    matches!(
        value,
        "aiger" | "btor2" | "vmt_interchange" | "ay_only" | "witness_replay"
    )
}

fn is_lane_kind_code(value: &str) -> bool {
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

fn is_problem_kind_code(value: &str) -> bool {
    matches!(
        value,
        "explicit_reachability"
            | "safety"
            | "liveness"
            | "deadlock"
            | "state_space"
            | "symbolic_execution"
            | "invariant"
            | "bmc"
            | "k_induction"
            | "chc"
            | "sat"
            | "smt"
            | "native_successor"
    )
}

fn is_validation_kind_code(value: &str) -> bool {
    matches!(
        value,
        "selftest"
            | "trace_replay"
            | "witness_replay"
            | "complete_graph"
            | "scc_certificate"
            | "accepting_cycle_certificate"
            | "structural_proof"
            | "ay_proof"
            | "output_format"
    )
}

fn is_fingerprint_scheme_code(value: &str) -> bool {
    matches!(
        value,
        "tla_fingerprint64"
            | "xxh3_u64"
            | "stable_u128"
            | "canonical_bytes_sha256"
            | "solver_model_digest"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_trace::{SetupTrace, SetupTracePhase};

    fn contract_program(
        identity: &'static str,
        payload_kind: PreparedProgramPayloadKind,
        storage_kind: PreparedStorageKind,
    ) -> PreparedCheckerProgram {
        let payload = payload_kind.code();
        let storage = storage_kind.code();
        PreparedCheckerProgram::new(identity, payload_kind, storage_kind)
            .with_canonical_payload_identity(format!("canonical:{payload}:payload"))
            .with_source_identity(format!("source:{payload}"))
            .with_config_identity(format!("config:{payload}"))
            .with_examination_identity(format!("examination:{payload}"))
            .with_cache_key(format!("cache:{payload}"))
            .with_source_fingerprint(format!("source-fp:{payload}"))
            .with_frontend_payload_identity(format!("frontend-payload:{payload}"))
            .with_frontend_payload_fingerprint(format!("frontend-payload-fp:{payload}"))
            .with_prepared_program_fingerprint(format!("prepared-fp:{payload}"))
            .with_artifact_identity(format!("artifact:{payload}"))
            .with_artifact_fingerprint(format!("artifact-fp:{payload}"))
            .with_storage_policy_identity(format!("storage-policy:{storage}"))
            .with_storage_layout_fingerprint(format!("storage-layout-fp:{storage}"))
            .with_fingerprint_policy_identity(format!("fingerprint-policy:{payload}"))
            .with_fingerprint_identity(format!("fingerprint:{payload}"))
            .with_transition_descriptor_fingerprint(format!("transition-fp:{payload}"))
            .with_property_descriptor_fingerprint(format!("property-fp:{payload}"))
            .with_validation_plan_fingerprint(format!("validation-plan-fp:{payload}"))
    }

    #[test]
    fn prepared_checker_program_records_frontend_neutral_shape() {
        let program = PreparedCheckerProgram::new(
            "MCL / TypeOK",
            PreparedProgramPayloadKind::Tla,
            PreparedStorageKind::TlaStateSlots,
        )
        .add_transition("Next", PreparedTransitionKind::TlaAction)
        .add_property("TypeOK", PreparedPropertyKind::Invariant)
        .require_validation(PreparedValidationKind::Selftest)
        .require_validation(PreparedValidationKind::TraceReplay)
        .require_validation(PreparedValidationKind::TraceReplay);

        assert_eq!(program.source_kind, CheckerSourceKind::Tla);
        assert_eq!(program.transitions.len(), 1);
        assert_eq!(program.properties.len(), 1);
        assert_eq!(program.validations.len(), 2);

        let row = program.render_evidence_row("TY");
        assert!(row.contains("identity=MCL_/_TypeOK"));
        assert!(row.contains("source_kind=tla"));
        assert!(row.contains("frontend_kind=tla"));
        assert!(row.contains("storage_kind=tla_state_slots"));
        assert!(row.contains("transitions=1"));
    }

    #[test]
    fn prepared_checker_program_records_solver_neutral_obligations() {
        let ay_family =
            PreparedBackendFamilyDescriptor::new("ay-chc", BackendKind::AYChc, ProblemKind::Chc)
                .with_facet(SolverFacet::Chc)
                .with_facet(SolverFacet::Pdr)
                .with_facet(SolverFacet::Pdr);

        let program = PreparedCheckerProgram::new(
            "Petri stable marking",
            PreparedProgramPayloadKind::MccPetri,
            PreparedStorageKind::PetriMarking,
        )
        .add_analytical_solve(
            "stable",
            PreparedAnalyticalSolveKind::StableMarking,
            ProblemKind::Smt,
        )
        .add_symbolic_proof(
            "inductive",
            PreparedSymbolicProofKind::InvariantProof,
            ProblemKind::Chc,
        )
        .add_backend_family(ay_family)
        .require_validation(PreparedValidationKind::AYProof);

        assert_eq!(
            PreparedAnalyticalSolveKind::StableMarking.code(),
            "stable_marking"
        );
        assert_eq!(
            PreparedSymbolicProofKind::InvariantProof.code(),
            "invariant_proof"
        );
        assert_eq!(program.analytical_solves.len(), 1);
        assert_eq!(program.symbolic_proofs.len(), 1);
        assert_eq!(program.backend_families.len(), 1);
        assert_eq!(program.backend_families[0].backend, BackendKind::AYChc);
        assert_eq!(
            program.backend_families[0].facets,
            vec![SolverFacet::Chc, SolverFacet::Pdr]
        );

        let row = program.render_evidence_row("MCC");
        assert!(row.contains("analytical_solves=1"));
        assert!(row.contains("symbolic_proofs=1"));
        assert!(row.contains("backend_families=1"));
    }

    #[test]
    fn prepared_checker_program_records_canonical_identity_metadata() {
        let program = PreparedCheckerProgram::new(
            "canonical hardware proof",
            PreparedProgramPayloadKind::Btor2,
            PreparedStorageKind::HardwareRegisters,
        )
        .with_fingerprint(PreparedFingerprintDescriptor::new(
            "state-vector",
            PreparedFingerprintScheme::CanonicalBytesSha256,
            "btor2-register-layout-v1",
        ))
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "prepared",
            PreparedCanonicalIdentityKind::PreparedProgram,
            "btor2-register-layout-v1",
        ))
        .add_canonical_identity(
            PreparedCanonicalIdentityDescriptor::new(
                "proof",
                PreparedCanonicalIdentityKind::ProofCertificate,
                "ay-proof-v1",
            )
            .with_digest("sha256", "abcdef"),
        );

        assert_eq!(
            PreparedFingerprintScheme::CanonicalBytesSha256.code(),
            "canonical_bytes_sha256"
        );
        assert_eq!(
            PreparedCanonicalIdentityKind::ProofCertificate.code(),
            "proof_certificate"
        );
        assert_eq!(
            program
                .fingerprint
                .as_ref()
                .map(|fingerprint| fingerprint.scheme),
            Some(PreparedFingerprintScheme::CanonicalBytesSha256)
        );
        assert_eq!(program.canonical_identities.len(), 2);
        assert_eq!(
            program.canonical_identities[1].digest_algorithm.as_deref(),
            Some("sha256")
        );

        let row = program.render_evidence_row("HW");
        assert!(row.contains("fingerprint_scheme=canonical_bytes_sha256"));
        assert!(row.contains("canonical_identities=2"));
    }

    #[test]
    fn prepared_checker_program_explains_shared_identity_contracts() {
        let fingerprint = PreparedFingerprintDescriptor::new(
            "state vector fp",
            PreparedFingerprintScheme::Xxh3U64,
            "slot-layout-v3",
        )
        .with_fingerprint_policy_identity("fingerprint descriptor policy")
        .with_fingerprint_identity("state fingerprint namespace");

        let native_lane =
            PreparedCandidateLaneDescriptor::new("trust-cg-native", SetupTraceLaneKind::Native)
                .with_candidate_key("native candidate")
                .with_candidate_identity("candidate native")
                .with_lane_identity("lane native")
                .with_cache_key("native cache key")
                .with_artifact_identity("native object artifact")
                .with_batch_artifact_identity("batch artifact 42");

        let program = PreparedCheckerProgram::new(
            "shared engine program",
            PreparedProgramPayloadKind::Quint,
            PreparedStorageKind::TlaStateSlots,
        )
        .with_cache_key("prepared cache key")
        .with_frontend_payload_identity("quint lowered tla payload")
        .with_artifact_identity("prepared artifact")
        .with_storage_policy_identity("slot layout v3")
        .with_batch_artifact_identity("batch artifact 42")
        .with_fingerprint(fingerprint)
        .add_candidate_lane(native_lane)
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "payload",
            PreparedCanonicalIdentityKind::FrontendPayload,
            "quint-lowered-v1",
        ))
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "batch",
            PreparedCanonicalIdentityKind::BatchArtifact,
            "batch-v1",
        ));

        assert_eq!(program.source_kind, CheckerSourceKind::Quint);
        assert_eq!(program.candidate_lanes.len(), 1);
        assert_eq!(
            program.candidate_lanes[0].candidate_key.as_deref(),
            Some("native candidate")
        );
        assert_eq!(
            program.candidate_lanes[0]
                .identities
                .artifact_identity
                .as_deref(),
            Some("native object artifact")
        );
        assert_eq!(
            PreparedCanonicalIdentityKind::BatchArtifact.code(),
            "batch_artifact"
        );

        let row = program.render_evidence_row("CORE");
        assert!(row.contains("frontend_kind=quint"));
        assert!(row.contains("cache_key=prepared_cache_key"));
        assert!(row.contains("frontend_payload_identity=quint_lowered_tla_payload"));
        assert!(row.contains("artifact_identity=prepared_artifact"));
        assert!(row.contains("storage_policy_identity=slot_layout_v3"));
        assert!(row.contains("fingerprint_policy_identity=fingerprint_descriptor_policy"));
        assert!(row.contains("fingerprint_identity=state_fingerprint_namespace"));
        assert!(row.contains("batch_artifact_identity=batch_artifact_42"));
        assert!(row.contains("fingerprint_id=state_vector_fp"));
        assert!(row.contains("candidate_lanes=1"));

        let lane_rows = program.render_candidate_lane_evidence_rows("CORE");
        assert_eq!(lane_rows.len(), 1);
        assert!(lane_rows[0].contains("prepared_candidate_lane"));
        assert!(lane_rows[0].contains("frontend_kind=quint"));
        assert!(lane_rows[0].contains("lane_kind=native"));
        assert!(lane_rows[0].contains("candidate_key=native_candidate"));
        assert!(lane_rows[0].contains("candidate_identity=candidate_native"));
        assert!(lane_rows[0].contains("lane_identity=lane_native"));
        assert!(lane_rows[0].contains("cache_key=native_cache_key"));
        assert!(lane_rows[0].contains("frontend_payload_identity=quint_lowered_tla_payload"));
        assert!(lane_rows[0].contains("artifact_identity=native_object_artifact"));
        assert!(lane_rows[0].contains("storage_policy_identity=slot_layout_v3"));
        assert!(lane_rows[0].contains("fingerprint_policy_identity=fingerprint_descriptor_policy"));
        assert!(lane_rows[0].contains("fingerprint_identity=state_fingerprint_namespace"));
        assert!(lane_rows[0].contains("batch_artifact_identity=batch_artifact_42"));

        let trace_keys = program.setup_trace_keys_for_candidate_lanes();
        assert_eq!(trace_keys.len(), 1);
        assert_eq!(trace_keys[0].frontend, CheckerSourceKind::Quint);
        assert_eq!(trace_keys[0].lane, SetupTraceLaneKind::Native);
        assert_eq!(
            trace_keys[0].candidate_key.as_deref(),
            Some("native candidate")
        );
        assert_eq!(
            trace_keys[0].identities.cache_key.as_deref(),
            Some("native cache key")
        );
        assert_eq!(
            trace_keys[0].identities.storage_policy_identity.as_deref(),
            Some("slot layout v3")
        );
        assert_eq!(
            trace_keys[0]
                .identities
                .fingerprint_policy_identity
                .as_deref(),
            Some("fingerprint descriptor policy")
        );

        let mut trace = SetupTrace::new(program.source_kind)
            .with_identity_fields(program.effective_identity_fields());
        trace.record_duration_for_key(
            trace_keys[0].clone(),
            SetupTracePhase::HotExecution,
            std::time::Duration::from_millis(2),
        );
        let trace_rows = trace.render_evidence_rows("CORE");
        assert!(trace_rows[0].contains("frontend_kind=quint"));
        assert!(trace_rows[0].contains("lane_kind=native"));
        assert!(trace_rows[0].contains("cache_key=native_cache_key"));
        assert!(trace_rows[0].contains("storage_policy_identity=slot_layout_v3"));
        assert!(trace_rows[0].contains("fingerprint_policy_identity=fingerprint_descriptor_policy"));
    }

    #[test]
    fn prepared_program_codes_cover_ay_shared_engine_lanes() {
        assert_eq!(
            PreparedProgramPayloadKind::shared_engine_payloads(),
            &[
                PreparedProgramPayloadKind::Tla,
                PreparedProgramPayloadKind::Quint,
                PreparedProgramPayloadKind::MccPetri,
                PreparedProgramPayloadKind::Aiger,
                PreparedProgramPayloadKind::Btor2,
                PreparedProgramPayloadKind::VmtInterchange,
                PreparedProgramPayloadKind::AYOnly,
                PreparedProgramPayloadKind::WitnessReplay,
            ]
        );
        assert_eq!(PreparedProgramPayloadKind::AYOnly.code(), "ay_only");
        assert_eq!(
            CheckerSourceKind::from(PreparedProgramPayloadKind::AYOnly),
            CheckerSourceKind::AYOnly
        );
        assert_eq!(
            PreparedAnalyticalSolveKind::BoundedModelCheck.code(),
            "bounded_model_check"
        );
        assert_eq!(PreparedAnalyticalSolveKind::PdrSafety.code(), "pdr_safety");
        assert_eq!(
            PreparedAnalyticalSolveKind::KInduction.code(),
            "k_induction"
        );
        assert_eq!(
            PreparedSymbolicProofKind::PdrSafetyProof.code(),
            "pdr_safety_proof"
        );
    }

    #[test]
    fn prepared_program_records_frontend_extensions_and_validation_plans() {
        let fingerprint = PreparedFingerprintDescriptor::new(
            "btor2 witness fp",
            PreparedFingerprintScheme::CanonicalBytesSha256,
            "btor2-witness-v1",
        )
        .with_fingerprint_policy_identity("witness fp policy")
        .with_fingerprint_identity("witness fp namespace");
        let extension = PreparedFrontendExtensionDescriptor::new(
            "btor2 payload",
            PreparedFrontendExtensionKind::Btor2,
            ProblemKind::Safety,
        )
        .with_cache_key("btor2 import cache")
        .with_artifact_identity("btor2 prepared payload");
        let validation = PreparedValidationPlanDescriptor::new(
            "typed witness replay",
            PreparedValidationKind::WitnessReplay,
            ProblemKind::Safety,
        )
        .with_fingerprint(fingerprint)
        .with_artifact_identity("witness replay plan");

        let program = contract_program(
            "hardware shared program",
            PreparedProgramPayloadKind::Btor2,
            PreparedStorageKind::HardwareRegisters,
        )
        .with_cache_key("prepared cache")
        .with_frontend_payload_identity("btor2 payload identity")
        .with_storage_policy_identity("register layout v1")
        .with_storage_layout_fingerprint("register layout fingerprint v1")
        .with_validation_plan_fingerprint("btor2 validation plan")
        .add_frontend_extension(extension)
        .add_validation_plan(validation);

        assert_eq!(program.frontend_extensions.len(), 1);
        assert_eq!(program.validation_plans.len(), 1);
        assert_eq!(
            program.validations,
            vec![PreparedValidationKind::WitnessReplay]
        );
        assert_eq!(
            PreparedFrontendExtensionKind::Btor2.storage_kind(),
            PreparedStorageKind::HardwareRegisters
        );

        let row = program.render_evidence_row("CORE");
        validate_prepared_checker_program_evidence_row(&row).unwrap();
        assert!(row.contains("frontend_extensions=1"));
        assert!(row.contains("validation_plans=1"));
        assert!(row.contains("validations=1"));

        let extension_rows = program.render_frontend_extension_evidence_rows("CORE");
        assert_eq!(extension_rows.len(), 1);
        validate_prepared_frontend_extension_evidence_row(&extension_rows[0]).unwrap();
        for field in PREPARED_FRONTEND_EXTENSION_REQUIRED_FIELDS {
            assert!(
                prepared_evidence_field(&extension_rows[0], field).is_some(),
                "missing extension field {field}"
            );
        }
        assert!(extension_rows[0].contains("prepared_frontend_extension"));
        assert!(extension_rows[0].contains("extension_kind=btor2"));
        assert!(extension_rows[0].contains("extension_storage_kind=hardware_registers"));
        assert!(extension_rows[0].contains("cache_key=btor2_import_cache"));
        assert!(extension_rows[0].contains("frontend_payload_identity=btor2_payload_identity"));
        assert!(extension_rows[0].contains("artifact_identity=btor2_prepared_payload"));

        let validation_rows = program.render_validation_plan_evidence_rows("CORE");
        assert_eq!(validation_rows.len(), 1);
        validate_prepared_validation_plan_evidence_row(&validation_rows[0]).unwrap();
        assert!(validation_rows[0].contains("prepared_validation_plan"));
        assert!(validation_rows[0].contains("validation_kind=witness_replay"));
        assert!(validation_rows[0].contains("problem=safety"));
        assert!(validation_rows[0].contains("required=true"));
        assert!(validation_rows[0].contains("fail_closed=true"));
        assert!(validation_rows[0].contains("fingerprint_id=btor2_witness_fp"));
        assert!(validation_rows[0].contains("fingerprint_scheme=canonical_bytes_sha256"));
        assert!(validation_rows[0].contains("fingerprint_policy_identity=witness_fp_policy"));
        assert!(validation_rows[0].contains("fingerprint_identity=witness_fp_namespace"));
        assert!(validation_rows[0].contains("storage_policy_identity=register_layout_v1"));
        assert!(validation_rows[0].contains("artifact_identity=witness_replay_plan"));
    }

    #[test]
    fn prepared_evidence_validators_reject_local_or_incomplete_adapter_rows() {
        let program = contract_program(
            "hardware shared program",
            PreparedProgramPayloadKind::Aiger,
            PreparedStorageKind::HardwareRegisters,
        )
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new("aiger.ay", SetupTraceLaneKind::AY)
                .with_candidate_key("ay_sat_safety"),
        )
        .add_validation_plan(PreparedValidationPlanDescriptor::new(
            "aiger.ay.proof",
            PreparedValidationKind::AYProof,
            ProblemKind::Sat,
        ));

        let row = program.render_evidence_row("CORE");
        validate_prepared_checker_program_evidence_row(&row).unwrap();

        let lane_row = program.render_candidate_lane_evidence_rows("CORE")[0].clone();
        validate_prepared_candidate_lane_evidence_row(&lane_row).unwrap();
        let bad_lane_row = lane_row.replace(" lane=ay ", " lane=frontend ");
        assert_eq!(
            validate_prepared_candidate_lane_evidence_row(&bad_lane_row),
            Err("lane_kind must match lane".to_string())
        );

        let validation_row = program.render_validation_plan_evidence_rows("CORE")[0].clone();
        validate_prepared_validation_plan_evidence_row(&validation_row).unwrap();
        let bad_validation_row = validation_row.replace(
            " validation_kind=ay_proof ",
            " validation_kind=local_check ",
        );
        assert_eq!(
            validate_prepared_validation_plan_evidence_row(&bad_validation_row),
            Err("field validation_kind has unknown validation kind: local_check".to_string())
        );
    }

    #[test]
    fn quint_payload_must_preserve_quint_identity_not_only_tla_lowering() {
        let program = contract_program(
            "quint preserved contract",
            PreparedProgramPayloadKind::Quint,
            PreparedStorageKind::TlaStateSlots,
        )
        .add_transition("Next", PreparedTransitionKind::TlaAction)
        .add_property("TypeOK", PreparedPropertyKind::Invariant)
        .add_validation_plan(PreparedValidationPlanDescriptor::new(
            "quint replay",
            PreparedValidationKind::TraceReplay,
            ProblemKind::Safety,
        ));

        assert!(program.has_required_payload_identity());
        program.validate_payload_identity_contract().unwrap();

        let row = program.render_evidence_row("QUINT");
        validate_prepared_checker_program_evidence_row(&row).unwrap();
        assert!(row.contains("source_kind=quint"));
        assert!(row.contains("payload_kind=quint"));
        assert!(row.contains("canonical_payload_identity=canonical:quint:payload"));
        assert!(row.contains("frontend_payload_identity=frontend-payload:quint"));
        assert!(row.contains("frontend_payload_fingerprint=frontend-payload-fp:quint"));

        let lowered_only_row = row.replace(
            " source_kind=quint frontend_kind=quint ",
            " source_kind=tla frontend_kind=tla ",
        );
        assert_eq!(
            validate_prepared_checker_program_evidence_row(&lowered_only_row),
            Err("payload_kind quint requires source_kind quint, got tla".to_string())
        );

        let missing_payload_row = row.replace(
            " canonical_payload_identity=canonical:quint:payload ",
            " canonical_payload_identity=none ",
        );
        assert_eq!(
            validate_prepared_checker_program_evidence_row(&missing_payload_row),
            Err("field canonical_payload_identity must not be none".to_string())
        );
    }

    #[test]
    fn future_importer_payload_family_is_reserved_not_default() {
        assert_eq!(
            PreparedProgramPayloadKind::reserved_payload_codes(),
            &[PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE]
        );
        assert!(!PreparedProgramPayloadKind::default_payload_codes()
            .contains(&PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE));
        assert_eq!(
            validate_prepared_payload_default_use(PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE),
            Err("future_importer is reserved until a registered importer supplies payload identity, layout fingerprints, and validation receipts".to_string())
        );

        let row = "CORE prepared_checker_program identity=future_importer_reserved source_kind=future_importer frontend_kind=future_importer payload_kind=future_importer storage_kind=unknown canonical_payload_identity=canonical:future_importer:payload source_identity=source:future_importer config_identity=config:future_importer examination_identity=examination:future_importer cache_key=cache:future_importer source_fingerprint=source-fp:future_importer frontend_payload_identity=frontend-payload:future_importer frontend_payload_fingerprint=frontend-payload-fp:future_importer prepared_program_fingerprint=prepared-fp:future_importer artifact_identity=artifact:future_importer artifact_fingerprint=artifact-fp:future_importer storage_policy_identity=storage-policy:future_importer storage_layout_fingerprint=storage-layout-fp:future_importer fingerprint_policy_identity=fingerprint-policy:future_importer fingerprint_identity=fingerprint:future_importer batch_artifact_identity=none candidate_identity=none lane_identity=none transition_descriptor_fingerprint=transition-fp:future_importer property_descriptor_fingerprint=property-fp:future_importer validation_plan_fingerprint=validation-plan-fp:future_importer transitions=0 properties=0 analytical_solves=0 symbolic_proofs=0 backend_families=0 fingerprint_id=none fingerprint_scheme=none canonical_identities=0 frontend_extensions=0 candidate_lanes=0 validation_plans=0 validations=0";
        assert_eq!(
            validate_prepared_checker_program_evidence_row(row),
            Err("future_importer is reserved until a registered importer supplies payload identity, layout fingerprints, and validation receipts".to_string())
        );
    }

    #[test]
    fn operational_payload_families_share_contract_shape() {
        let cases = [
            (
                PreparedProgramPayloadKind::Tla,
                PreparedStorageKind::TlaStateSlots,
                PreparedTransitionKind::TlaAction,
                PreparedPropertyKind::Invariant,
            ),
            (
                PreparedProgramPayloadKind::MccPetri,
                PreparedStorageKind::PetriMarking,
                PreparedTransitionKind::PetriTransition,
                PreparedPropertyKind::StableMarking,
            ),
            (
                PreparedProgramPayloadKind::Aiger,
                PreparedStorageKind::HardwareRegisters,
                PreparedTransitionKind::HardwareNextState,
                PreparedPropertyKind::BadState,
            ),
            (
                PreparedProgramPayloadKind::Btor2,
                PreparedStorageKind::HardwareRegisters,
                PreparedTransitionKind::HardwareNextState,
                PreparedPropertyKind::BadState,
            ),
            (
                PreparedProgramPayloadKind::VmtInterchange,
                PreparedStorageKind::SmtVariables,
                PreparedTransitionKind::SymbolicTransitionRelation,
                PreparedPropertyKind::ProofObligation,
            ),
        ];

        for (payload, storage, transition, property) in cases {
            let program = contract_program("shared contract", payload, storage)
                .add_transition("transition", transition)
                .add_property("property", property)
                .add_candidate_lane(
                    PreparedCandidateLaneDescriptor::new(
                        "candidate",
                        SetupTraceLaneKind::Analytical,
                    )
                    .with_candidate_key("candidate-key"),
                )
                .add_validation_plan(PreparedValidationPlanDescriptor::new(
                    "selftest",
                    PreparedValidationKind::Selftest,
                    ProblemKind::Safety,
                ));

            program.validate_payload_identity_contract().unwrap();
            let row = program.render_evidence_row("CORE");
            validate_prepared_checker_program_evidence_row(&row).unwrap();
            for field in PREPARED_CHECKER_PROGRAM_REQUIRED_FIELDS {
                assert!(
                    prepared_evidence_field(&row, field).is_some(),
                    "missing field {field} for payload {}",
                    payload.code()
                );
            }
            assert!(row.contains(&format!("payload_kind={}", payload.code())));
            assert!(row.contains(&format!("storage_kind={}", storage.code())));

            let lane_row = &program.render_candidate_lane_evidence_rows("CORE")[0];
            validate_prepared_candidate_lane_evidence_row(lane_row).unwrap();
            for field in PREPARED_CANDIDATE_LANE_REQUIRED_FIELDS {
                assert!(
                    prepared_evidence_field(lane_row, field).is_some(),
                    "missing lane field {field} for payload {}",
                    payload.code()
                );
            }

            let validation_row = &program.render_validation_plan_evidence_rows("CORE")[0];
            validate_prepared_validation_plan_evidence_row(validation_row).unwrap();
            for field in PREPARED_VALIDATION_PLAN_REQUIRED_FIELDS {
                assert!(
                    prepared_evidence_field(validation_row, field).is_some(),
                    "missing validation field {field} for payload {}",
                    payload.code()
                );
            }
        }
    }
}
