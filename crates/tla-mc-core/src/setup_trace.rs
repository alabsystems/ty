// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Domain-neutral setup and execution timing records.
//!
//! Frontends own semantic lowering, but cold-start policy needs one timing
//! vocabulary across TLA+, MCC/Petri, hardware, symbolic, and replay lanes.

use crate::shared_engine_adoption::SharedEngineFrontendFamily;

use std::time::Duration;

/// Stable row kind for shared setup timing evidence.
pub const SETUP_TRACE_ROW_KIND: &str = "setup_trace";

/// Stable schema label for shared setup timing evidence.
pub const SETUP_TRACE_SCHEMA: &str = "ty.shared.setup_trace.v1";

/// Stable schema version for shared setup timing evidence.
pub const SETUP_TRACE_SCHEMA_VERSION: u32 = 1;

/// Fields every setup timing row publishes for frontend-neutral consumers.
///
/// `source_kind` names the concrete payload or interchange family that entered
/// setup. `frontend_kind` and the legacy `frontend` alias publish the canonical
/// shared-engine adoption family, so payload names such as `vmt_interchange`
/// stay distinct from adoption families such as `vmt_transition_system`.
pub const SETUP_TRACE_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "source_kind",
    "frontend_kind",
    "frontend",
    "lane_kind",
    "lane",
    "candidate_key",
    "candidate_identity",
    "lane_identity",
    "cache_key",
    "source_fingerprint",
    "frontend_payload_identity",
    "prepared_program_fingerprint",
    "artifact_identity",
    "artifact_fingerprint",
    "storage_policy_identity",
    "storage_layout_fingerprint",
    "fingerprint_policy_identity",
    "fingerprint_identity",
    "batch_artifact_identity",
    "proof_or_witness_fingerprint",
    "replay_transcript_fingerprint",
    "publication_fingerprint",
    "source_identity",
    "property_identity",
    "origin_frontend",
    "shared_engine_component",
    "first_beneficiary",
    "second_beneficiary",
    "compatible_frontend_families",
    "frontend_family_compatibility",
    "extraction_status",
    "blocker_status",
    "validation_status",
    "phase",
    "nanos",
];

/// Source or interchange family that produced a checker program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckerSourceKind {
    /// TLA+ specification source.
    Tla,
    /// Quint specification source.
    Quint,
    /// MCC / Petri-net model source.
    MccPetri,
    /// AIGER hardware netlist.
    Aiger,
    /// BTOR2 word-level hardware model.
    Btor2,
    /// VMT (verification modulo theories) interchange payload.
    VmtInterchange,
    /// A model expressed directly to the AY solver, with no upstream frontend.
    AYOnly,
    /// A previously recorded witness/counterexample being replayed.
    WitnessReplay,
    /// Source family not known to this build.
    Unknown,
}

impl CheckerSourceKind {
    /// Stable lowercase wire code for this source kind (for example `"tla"`).
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
            Self::Unknown => "unknown",
        }
    }

    /// Map this source kind to its canonical shared-engine adoption family, or
    /// `None` for [`CheckerSourceKind::Unknown`].
    pub fn adoption_frontend_family(self) -> Option<SharedEngineFrontendFamily> {
        match self {
            Self::Tla => Some(SharedEngineFrontendFamily::TlaPlus),
            Self::Quint => Some(SharedEngineFrontendFamily::Quint),
            Self::MccPetri => Some(SharedEngineFrontendFamily::MccPetri),
            Self::Aiger => Some(SharedEngineFrontendFamily::Aiger),
            Self::Btor2 => Some(SharedEngineFrontendFamily::Btor2),
            Self::VmtInterchange => Some(SharedEngineFrontendFamily::VmtTransitionSystem),
            Self::AYOnly => Some(SharedEngineFrontendFamily::AYAnalytical),
            Self::WitnessReplay => Some(SharedEngineFrontendFamily::WitnessReplay),
            Self::Unknown => None,
        }
    }

    /// Wire code of the canonical adoption family, or `"unknown"` when none.
    pub fn frontend_family_code(self) -> &'static str {
        self.adoption_frontend_family()
            .map_or("unknown", SharedEngineFrontendFamily::code)
    }

    /// Rust-style variant name (for example `"Tla"`), for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Tla => "Tla",
            Self::Quint => "Quint",
            Self::MccPetri => "MccPetri",
            Self::Aiger => "Aiger",
            Self::Btor2 => "Btor2",
            Self::VmtInterchange => "VmtInterchange",
            Self::AYOnly => "AYOnly",
            Self::WitnessReplay => "WitnessReplay",
            Self::Unknown => "Unknown",
        }
    }
}

/// Broad execution lane that owns a setup or run timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SetupTraceLaneKind {
    /// Frontend import / semantic-lowering lane.
    Frontend,
    /// Explicit-state model-checking lane.
    ExplicitState,
    /// Native (compiled trust-cg) execution lane.
    Native,
    /// AY solver lane (SAT/SMT/CHC/symbolic).
    AY,
    /// Analytical (closed-form / structural) solve lane.
    Analytical,
    /// Witness/counterexample replay lane.
    Replay,
    /// Fingerprint policy/storage/compute lane.
    Fingerprint,
    /// Lane not known to this build.
    #[default]
    Unknown,
}

impl SetupTraceLaneKind {
    /// Stable lowercase wire code for this lane (for example `"native"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Frontend => "frontend",
            Self::ExplicitState => "explicit_state",
            Self::Native => "native",
            Self::AY => "ay",
            Self::Analytical => "analytical",
            Self::Replay => "replay",
            Self::Fingerprint => "fingerprint",
            Self::Unknown => "unknown",
        }
    }

    /// Rust-style variant name (for example `"Native"`), for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Frontend => "Frontend",
            Self::ExplicitState => "ExplicitState",
            Self::Native => "Native",
            Self::AY => "AY",
            Self::Analytical => "Analytical",
            Self::Replay => "Replay",
            Self::Fingerprint => "Fingerprint",
            Self::Unknown => "Unknown",
        }
    }
}

/// Frontend-neutral identity fields shared by setup traces, prepared programs,
/// batch artifacts, and candidate lanes.
///
/// Every field is optional; an absent (`None`) field is rendered as `none` in
/// evidence rows. The matching `with_*` builders normalize the empty string to
/// `None`. Field names mirror the corresponding entries in
/// [`SETUP_TRACE_REQUIRED_FIELDS`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct CheckerArtifactIdentityFields {
    /// Cache key under which the prepared artifact is stored/looked up.
    pub cache_key: Option<String>,
    /// Stable digest of the original source text/model.
    pub source_fingerprint: Option<String>,
    /// Identity of the frontend-produced payload fed into preparation.
    pub frontend_payload_identity: Option<String>,
    /// Digest of the prepared checker program.
    pub prepared_program_fingerprint: Option<String>,
    /// Identity of the produced executable/checker artifact.
    pub artifact_identity: Option<String>,
    /// Digest of the produced executable/checker artifact.
    pub artifact_fingerprint: Option<String>,
    /// Identity of the fingerprint-storage policy in effect.
    pub storage_policy_identity: Option<String>,
    /// Digest of the storage layout used for fingerprint state.
    pub storage_layout_fingerprint: Option<String>,
    /// Identity of the fingerprint policy (algorithm/canonicalization).
    pub fingerprint_policy_identity: Option<String>,
    /// Canonical fingerprint identity for this artifact.
    pub fingerprint_identity: Option<String>,
    /// Identity of the batch artifact this lane belongs to.
    pub batch_artifact_identity: Option<String>,
    /// Digest of an emitted proof or witness, when one exists.
    pub proof_or_witness_fingerprint: Option<String>,
    /// Digest of a replay transcript, when one exists.
    pub replay_transcript_fingerprint: Option<String>,
    /// Digest of the published evidence/result bundle.
    pub publication_fingerprint: Option<String>,
    /// Per-candidate identity (precedence over program-level identity).
    pub candidate_identity: Option<String>,
    /// Per-lane identity (precedence over program-level identity).
    pub lane_identity: Option<String>,
}

impl CheckerArtifactIdentityFields {
    /// Create an all-empty identity set (equivalent to [`Default::default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`cache_key`](Self::cache_key) (empty string clears it to `None`).
    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.cache_key = non_empty_string(cache_key.into());
        self
    }

    /// Set [`source_fingerprint`](Self::source_fingerprint) (empty clears it).
    pub fn with_source_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.source_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`frontend_payload_identity`](Self::frontend_payload_identity).
    pub fn with_frontend_payload_identity(mut self, identity: impl Into<String>) -> Self {
        self.frontend_payload_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`prepared_program_fingerprint`](Self::prepared_program_fingerprint).
    pub fn with_prepared_program_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.prepared_program_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`artifact_identity`](Self::artifact_identity) (empty clears it).
    pub fn with_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.artifact_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`artifact_fingerprint`](Self::artifact_fingerprint) (empty clears it).
    pub fn with_artifact_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.artifact_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`storage_policy_identity`](Self::storage_policy_identity).
    pub fn with_storage_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.storage_policy_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`storage_layout_fingerprint`](Self::storage_layout_fingerprint).
    pub fn with_storage_layout_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.storage_layout_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`fingerprint_policy_identity`](Self::fingerprint_policy_identity).
    pub fn with_fingerprint_policy_identity(mut self, identity: impl Into<String>) -> Self {
        self.fingerprint_policy_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`fingerprint_identity`](Self::fingerprint_identity) (empty clears it).
    pub fn with_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.fingerprint_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`batch_artifact_identity`](Self::batch_artifact_identity).
    pub fn with_batch_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.batch_artifact_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`proof_or_witness_fingerprint`](Self::proof_or_witness_fingerprint).
    pub fn with_proof_or_witness_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.proof_or_witness_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`replay_transcript_fingerprint`](Self::replay_transcript_fingerprint).
    pub fn with_replay_transcript_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.replay_transcript_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`publication_fingerprint`](Self::publication_fingerprint).
    pub fn with_publication_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.publication_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Set [`candidate_identity`](Self::candidate_identity) (empty clears it).
    pub fn with_candidate_identity(mut self, identity: impl Into<String>) -> Self {
        self.candidate_identity = non_empty_string(identity.into());
        self
    }

    /// Set [`lane_identity`](Self::lane_identity) (empty clears it).
    pub fn with_lane_identity(mut self, identity: impl Into<String>) -> Self {
        self.lane_identity = non_empty_string(identity.into());
        self
    }

    /// Returns these identities with each missing field filled from `fallback`.
    ///
    /// Candidate/lane identities should call this with program-level identities
    /// as fallback so lane-specific artifacts keep precedence.
    pub fn merged_with_fallback(&self, fallback: &Self) -> Self {
        Self {
            cache_key: self
                .cache_key
                .clone()
                .or_else(|| fallback.cache_key.clone()),
            source_fingerprint: self
                .source_fingerprint
                .clone()
                .or_else(|| fallback.source_fingerprint.clone()),
            frontend_payload_identity: self
                .frontend_payload_identity
                .clone()
                .or_else(|| fallback.frontend_payload_identity.clone()),
            prepared_program_fingerprint: self
                .prepared_program_fingerprint
                .clone()
                .or_else(|| fallback.prepared_program_fingerprint.clone()),
            artifact_identity: self
                .artifact_identity
                .clone()
                .or_else(|| fallback.artifact_identity.clone()),
            artifact_fingerprint: self
                .artifact_fingerprint
                .clone()
                .or_else(|| fallback.artifact_fingerprint.clone()),
            storage_policy_identity: self
                .storage_policy_identity
                .clone()
                .or_else(|| fallback.storage_policy_identity.clone()),
            storage_layout_fingerprint: self
                .storage_layout_fingerprint
                .clone()
                .or_else(|| fallback.storage_layout_fingerprint.clone()),
            fingerprint_policy_identity: self
                .fingerprint_policy_identity
                .clone()
                .or_else(|| fallback.fingerprint_policy_identity.clone()),
            fingerprint_identity: self
                .fingerprint_identity
                .clone()
                .or_else(|| fallback.fingerprint_identity.clone()),
            batch_artifact_identity: self
                .batch_artifact_identity
                .clone()
                .or_else(|| fallback.batch_artifact_identity.clone()),
            proof_or_witness_fingerprint: self
                .proof_or_witness_fingerprint
                .clone()
                .or_else(|| fallback.proof_or_witness_fingerprint.clone()),
            replay_transcript_fingerprint: self
                .replay_transcript_fingerprint
                .clone()
                .or_else(|| fallback.replay_transcript_fingerprint.clone()),
            publication_fingerprint: self
                .publication_fingerprint
                .clone()
                .or_else(|| fallback.publication_fingerprint.clone()),
            candidate_identity: self
                .candidate_identity
                .clone()
                .or_else(|| fallback.candidate_identity.clone()),
            lane_identity: self
                .lane_identity
                .clone()
                .or_else(|| fallback.lane_identity.clone()),
        }
    }
}

/// Stable key for attributing setup and run evidence to one candidate lane.
///
/// Two timings with the same key are considered the same measurement slot:
/// [`SetupTrace::record_duration_for_key`] overwrites rather than appends when
/// keys (and phase) match.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetupTraceKey {
    /// Source/interchange family the measured lane belongs to.
    pub frontend: CheckerSourceKind,
    /// Broad execution lane that owns the timing.
    pub lane: SetupTraceLaneKind,
    /// Optional candidate key distinguishing competing candidate lanes.
    pub candidate_key: Option<String>,
    /// Artifact identity fields attributing the timing to a concrete artifact.
    pub identities: CheckerArtifactIdentityFields,
}

impl SetupTraceKey {
    /// Create a key for the given source family and lane, with no identities.
    pub fn new(frontend: CheckerSourceKind, lane: SetupTraceLaneKind) -> Self {
        Self {
            frontend,
            lane,
            candidate_key: None,
            identities: CheckerArtifactIdentityFields::default(),
        }
    }

    /// Create a fully-unknown key (unknown source family and unknown lane).
    pub fn unknown() -> Self {
        Self::new(CheckerSourceKind::Unknown, SetupTraceLaneKind::Unknown)
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

    /// Set the batch-artifact identity on [`identities`](Self::identities).
    pub fn with_batch_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_batch_artifact_identity(identity);
        self
    }

    /// Set the proof/witness fingerprint on [`identities`](Self::identities).
    pub fn with_proof_or_witness_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_proof_or_witness_fingerprint(fingerprint);
        self
    }

    /// Set the replay-transcript fingerprint on [`identities`](Self::identities).
    pub fn with_replay_transcript_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_replay_transcript_fingerprint(fingerprint);
        self
    }

    /// Set the publication fingerprint on [`identities`](Self::identities).
    pub fn with_publication_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_publication_fingerprint(fingerprint);
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

impl Default for SetupTraceKey {
    fn default() -> Self {
        Self::unknown()
    }
}

/// Shared cold-start phase vocabulary.
///
/// Phases are ordered roughly by their place in the cold-start pipeline, from
/// reading the source through hot execution and result validation, ending with
/// [`SetupTracePhase::TotalWall`] for the end-to-end wall time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetupTracePhase {
    /// Reading the source text/model from disk or input.
    SourceLoad,
    /// Importing the source into the frontend representation.
    FrontendImport,
    /// Resolving configuration (constants, overrides, options).
    ConfigResolution,
    /// Lowering source semantics into the engine's representation.
    SemanticLowering,
    /// Lowering properties/invariants to be checked.
    PropertyLowering,
    /// Validating any applied reductions (for example partial-order).
    ReductionValidation,
    /// Extracting a proof obligation or proof scaffold.
    ProofExtraction,
    /// Building the prepared checker program.
    PreparedProgramBuild,
    /// Building trust-ir.
    TrustIrBuild,
    /// Verifying trust-ir.
    TrustIrVerify,
    /// Lowering trust-ir into trust-cg.
    TrustCgLower,
    /// Optimizing trust-cg.
    TrustCgOpt,
    /// Generating native code from trust-cg.
    TrustCgCodegen,
    /// Publishing the produced native artifact.
    NativePublish,
    /// Running the engine's self-test before hot execution.
    Selftest,
    /// Setting up the external solver.
    SolverSetup,
    /// Resolving the fingerprint policy.
    FingerprintPolicyResolution,
    /// Setting up fingerprint storage.
    FingerprintStorageSetup,
    /// Canonicalizing states prior to fingerprinting.
    FingerprintCanonicalization,
    /// Computing state fingerprints.
    FingerprintCompute,
    /// Deduplicating states by fingerprint.
    FingerprintDedup,
    /// Hot model-checking / solving execution.
    HotExecution,
    /// Replaying a witness/counterexample.
    WitnessReplay,
    /// Validating an emitted certificate.
    CertificateValidation,
    /// Replaying an emitted proof.
    ProofReplay,
    /// Validating the output format of the result.
    OutputFormatValidation,
    /// End-to-end wall-clock time for the whole run.
    TotalWall,
}

impl SetupTracePhase {
    /// Stable lowercase wire code for this phase (for example `"hot_execution"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::SourceLoad => "source_load",
            Self::FrontendImport => "frontend_import",
            Self::ConfigResolution => "config_resolution",
            Self::SemanticLowering => "semantic_lowering",
            Self::PropertyLowering => "property_lowering",
            Self::ReductionValidation => "reduction_validation",
            Self::ProofExtraction => "proof_extraction",
            Self::PreparedProgramBuild => "prepared_program_build",
            Self::TrustIrBuild => "trust_ir_build",
            Self::TrustIrVerify => "trust_ir_verify",
            Self::TrustCgLower => "trust_cg_lower",
            Self::TrustCgOpt => "trust_cg_opt",
            Self::TrustCgCodegen => "trust_cg_codegen",
            Self::NativePublish => "native_publish",
            Self::Selftest => "selftest",
            Self::SolverSetup => "solver_setup",
            Self::FingerprintPolicyResolution => "fingerprint_policy_resolution",
            Self::FingerprintStorageSetup => "fingerprint_storage_setup",
            Self::FingerprintCanonicalization => "fingerprint_canonicalization",
            Self::FingerprintCompute => "fingerprint_compute",
            Self::FingerprintDedup => "fingerprint_dedup",
            Self::HotExecution => "hot_execution",
            Self::WitnessReplay => "witness_replay",
            Self::CertificateValidation => "certificate_validation",
            Self::ProofReplay => "proof_replay",
            Self::OutputFormatValidation => "output_format_validation",
            Self::TotalWall => "total_wall",
        }
    }
}

/// Fingerprint setup/run phases shared by explicit-state, native, analytical,
/// replay, and future lanes.
pub const SETUP_TRACE_FINGERPRINT_PHASES: &[SetupTracePhase] = &[
    SetupTracePhase::FingerprintPolicyResolution,
    SetupTracePhase::FingerprintStorageSetup,
    SetupTracePhase::FingerprintCanonicalization,
    SetupTracePhase::FingerprintCompute,
    SetupTracePhase::FingerprintDedup,
];

/// One phase duration in nanoseconds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetupTraceTiming {
    /// Key attributing this timing to a lane/candidate/artifact.
    pub key: SetupTraceKey,
    /// Pipeline phase being measured.
    pub phase: SetupTracePhase,
    /// Measured duration in nanoseconds (saturated to `u64::MAX`).
    pub nanos: u64,
}

impl SetupTraceTiming {
    /// Create a timing for `phase` with a fully-unknown key.
    pub fn new(phase: SetupTracePhase, duration: Duration) -> Self {
        Self::new_for_key(SetupTraceKey::unknown(), phase, duration)
    }

    /// Create a timing for `phase` attributed to `key`.
    ///
    /// The duration is converted to nanoseconds and saturated at `u64::MAX`,
    /// so absurdly long durations never overflow.
    pub fn new_for_key(key: SetupTraceKey, phase: SetupTracePhase, duration: Duration) -> Self {
        Self {
            key,
            phase,
            nanos: duration.as_nanos().min(u128::from(u64::MAX)) as u64,
        }
    }
}

/// Fail-closed validation status attached to shared setup/publication evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SetupTraceValidationStatus {
    /// Validation outcome not yet determined (the fail-closed default).
    #[default]
    Unknown,
    /// No validation is required for this lane.
    NotRequired,
    /// Validation is required and still in progress.
    Pending,
    /// Validation completed and accepted the artifact.
    Accepted,
    /// Validation completed and rejected the artifact.
    Rejected,
}

impl SetupTraceValidationStatus {
    /// Stable lowercase wire code for this status (for example `"accepted"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::NotRequired => "not_required",
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }
}

/// Shared setup/execution record used by all checker lanes.
///
/// Accumulates per-phase timings (via [`SetupTrace::record_duration`]) together
/// with the shared-engine attribution fields needed to render frontend-neutral
/// evidence rows. Use [`SetupTrace::render_evidence_rows`] to emit rows and
/// [`SetupTrace::is_shared_engine_contract_valid`] to gate them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupTrace {
    /// Concrete source/interchange family that entered setup.
    pub source_kind: CheckerSourceKind,
    /// Broad execution lane this record describes.
    pub lane: SetupTraceLaneKind,
    /// Optional candidate key when several candidate lanes compete.
    pub candidate_key: Option<String>,
    /// Identity of the original source/model.
    pub source_identity: Option<String>,
    /// Identity of the property/invariant being checked.
    pub property_identity: Option<String>,
    /// Canonical family that originally owned this run (shared-engine attribution).
    pub origin_frontend: Option<String>,
    /// Name of the shared-engine component doing the work.
    pub shared_engine_component: Option<String>,
    /// First frontend family that benefits from the shared-engine work.
    pub first_beneficiary: Option<String>,
    /// Second, distinct frontend family that benefits from the shared work.
    pub second_beneficiary: Option<String>,
    /// Sorted, de-duplicated set of frontend families this work is compatible with.
    pub compatible_frontend_families: Vec<String>,
    /// Shared-engine extraction status string, if recorded.
    pub extraction_status: Option<String>,
    /// Shared-engine blocker status string, if recorded.
    pub blocker_status: Option<String>,
    /// Fail-closed validation status for emitted evidence.
    pub validation_status: SetupTraceValidationStatus,
    /// Artifact identity fields attached to every rendered row.
    pub identities: CheckerArtifactIdentityFields,
    /// Recorded per-phase timings (keyed; latest write per key+phase wins).
    timings: Vec<SetupTraceTiming>,
}

impl SetupTrace {
    /// Create an empty trace for the given source family (lane unknown).
    pub fn new(source_kind: CheckerSourceKind) -> Self {
        Self {
            source_kind,
            lane: SetupTraceLaneKind::Unknown,
            candidate_key: None,
            source_identity: None,
            property_identity: None,
            origin_frontend: None,
            shared_engine_component: None,
            first_beneficiary: None,
            second_beneficiary: None,
            compatible_frontend_families: Vec::new(),
            extraction_status: None,
            blocker_status: None,
            validation_status: SetupTraceValidationStatus::Unknown,
            identities: CheckerArtifactIdentityFields::default(),
            timings: Vec::new(),
        }
    }

    /// Set [`lane`](Self::lane).
    pub fn with_lane(mut self, lane: SetupTraceLaneKind) -> Self {
        self.lane = lane;
        self
    }

    /// Set [`candidate_key`](Self::candidate_key) (empty clears it to `None`).
    pub fn with_candidate_key(mut self, candidate_key: impl Into<String>) -> Self {
        self.candidate_key = non_empty_string(candidate_key.into());
        self
    }

    /// Set [`source_identity`](Self::source_identity).
    pub fn with_source_identity(mut self, identity: impl Into<String>) -> Self {
        self.source_identity = Some(identity.into());
        self
    }

    /// Set [`property_identity`](Self::property_identity).
    pub fn with_property_identity(mut self, identity: impl Into<String>) -> Self {
        self.property_identity = Some(identity.into());
        self
    }

    /// Set [`origin_frontend`](Self::origin_frontend), canonicalizing the family
    /// name when it is a recognized frontend reference.
    pub fn with_origin_frontend(mut self, origin_frontend: impl Into<String>) -> Self {
        self.origin_frontend = non_empty_string(origin_frontend.into())
            .map(|origin| canonical_frontend_family(&origin).unwrap_or(origin));
        self
    }

    /// Set [`shared_engine_component`](Self::shared_engine_component).
    pub fn with_shared_engine_component(mut self, component: impl Into<String>) -> Self {
        self.shared_engine_component = non_empty_string(component.into());
        self
    }

    /// Set [`first_beneficiary`](Self::first_beneficiary), canonicalizing a
    /// recognized frontend family name.
    pub fn with_first_beneficiary(mut self, beneficiary: impl Into<String>) -> Self {
        self.first_beneficiary = non_empty_string(beneficiary.into())
            .map(|beneficiary| canonical_frontend_family(&beneficiary).unwrap_or(beneficiary));
        self
    }

    /// Set [`second_beneficiary`](Self::second_beneficiary), canonicalizing a
    /// recognized frontend family name.
    pub fn with_second_beneficiary(mut self, beneficiary: impl Into<String>) -> Self {
        self.second_beneficiary = non_empty_string(beneficiary.into())
            .map(|beneficiary| canonical_frontend_family(&beneficiary).unwrap_or(beneficiary));
        self
    }

    /// Add one entry to [`compatible_frontend_families`](Self::compatible_frontend_families),
    /// canonicalizing it then keeping the set sorted and de-duplicated.
    pub fn with_compatible_frontend_family(mut self, family: impl Into<String>) -> Self {
        if let Some(family) = non_empty_string(family.into()) {
            self.compatible_frontend_families
                .push(canonical_frontend_family(&family).unwrap_or(family));
            self.compatible_frontend_families.sort();
            self.compatible_frontend_families.dedup();
        }
        self
    }

    /// Add several entries to
    /// [`compatible_frontend_families`](Self::compatible_frontend_families),
    /// canonicalizing each then sorting and de-duplicating the set.
    pub fn with_compatible_frontend_families<I, S>(mut self, families: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for family in families {
            if let Some(family) = non_empty_string(family.into()) {
                self.compatible_frontend_families
                    .push(canonical_frontend_family(&family).unwrap_or(family));
            }
        }
        self.compatible_frontend_families.sort();
        self.compatible_frontend_families.dedup();
        self
    }

    /// Set [`extraction_status`](Self::extraction_status) (empty clears it).
    pub fn with_shared_engine_extraction_status(mut self, status: impl Into<String>) -> Self {
        self.extraction_status = non_empty_string(status.into());
        self
    }

    /// Set [`blocker_status`](Self::blocker_status) (empty clears it).
    pub fn with_shared_engine_blocker_status(mut self, status: impl Into<String>) -> Self {
        self.blocker_status = non_empty_string(status.into());
        self
    }

    /// Set [`validation_status`](Self::validation_status).
    pub fn with_validation_status(mut self, status: SetupTraceValidationStatus) -> Self {
        self.validation_status = status;
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

    /// Set the batch-artifact identity on [`identities`](Self::identities).
    pub fn with_batch_artifact_identity(mut self, identity: impl Into<String>) -> Self {
        self.identities = self.identities.with_batch_artifact_identity(identity);
        self
    }

    /// Set the proof/witness fingerprint on [`identities`](Self::identities).
    pub fn with_proof_or_witness_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_proof_or_witness_fingerprint(fingerprint);
        self
    }

    /// Set the replay-transcript fingerprint on [`identities`](Self::identities).
    pub fn with_replay_transcript_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self
            .identities
            .with_replay_transcript_fingerprint(fingerprint);
        self
    }

    /// Set the publication fingerprint on [`identities`](Self::identities).
    pub fn with_publication_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.identities = self.identities.with_publication_fingerprint(fingerprint);
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

    /// Derive the key for this trace from its source kind, lane, candidate key,
    /// and identities.
    pub fn key(&self) -> SetupTraceKey {
        let mut key = SetupTraceKey::new(self.source_kind, self.lane);
        key.candidate_key.clone_from(&self.candidate_key);
        key.identities.clone_from(&self.identities);
        key
    }

    /// Record a phase duration under this trace's own [`key`](Self::key).
    ///
    /// Replaces any prior timing for the same key and phase.
    pub fn record_duration(&mut self, phase: SetupTracePhase, duration: Duration) {
        self.record_duration_for_key(self.key(), phase, duration);
    }

    /// Record a phase duration under an explicit `key`.
    ///
    /// Replaces any prior timing for the same `key` and `phase`; otherwise
    /// appends a new timing.
    pub fn record_duration_for_key(
        &mut self,
        key: SetupTraceKey,
        phase: SetupTracePhase,
        duration: Duration,
    ) {
        let timing = SetupTraceTiming::new_for_key(key, phase, duration);
        if let Some(existing) = self
            .timings
            .iter_mut()
            .find(|item| item.key == timing.key && item.phase == phase)
        {
            *existing = timing;
        } else {
            self.timings.push(timing);
        }
    }

    /// Nanoseconds recorded for `phase` under this trace's own key, falling back
    /// to the first timing for that phase under any key.
    pub fn phase_nanos(&self, phase: SetupTracePhase) -> Option<u64> {
        let key = self.key();
        self.phase_nanos_for_key(&key, phase)
            .or_else(|| self.first_phase_nanos(phase))
    }

    /// Nanoseconds recorded for `phase` under the exact `key`, if any.
    pub fn phase_nanos_for_key(&self, key: &SetupTraceKey, phase: SetupTracePhase) -> Option<u64> {
        self.timings
            .iter()
            .find(|item| item.key == *key && item.phase == phase)
            .map(|item| item.nanos)
    }

    /// All recorded timings, in insertion order.
    pub fn timings(&self) -> &[SetupTraceTiming] {
        &self.timings
    }

    /// List the shared-engine contract fields that are missing or invalid.
    ///
    /// Returns stable field codes for every unmet requirement: missing
    /// attribution fields, fewer than three compatible families, non-canonical
    /// or placeholder family references, frontends not listed as compatible,
    /// and a second beneficiary that is not distinct from the origin. An empty
    /// result means the contract is satisfied (see
    /// [`is_shared_engine_contract_valid`](Self::is_shared_engine_contract_valid)).
    pub fn shared_engine_contract_issues(&self) -> Vec<&'static str> {
        let mut issues = Vec::new();
        if self
            .origin_frontend
            .as_deref()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            issues.push("origin_frontend");
        }
        if self
            .shared_engine_component
            .as_deref()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            issues.push("shared_engine_component");
        }
        if self
            .first_beneficiary
            .as_deref()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            issues.push("first_beneficiary");
        }
        if self
            .second_beneficiary
            .as_deref()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            issues.push("second_beneficiary");
        }
        if self.compatible_frontend_families.len() < 3 {
            issues.push("compatible_frontend_families");
        }
        if self
            .compatible_frontend_families
            .iter()
            .any(|family| SharedEngineFrontendFamily::from_code(family).is_none())
        {
            issues.push("compatible_frontend_family_codes");
        }
        if !self
            .timings
            .iter()
            .map(|timing| timing.key.frontend)
            .chain(std::iter::once(self.source_kind))
            .all(|frontend| {
                frontend_family_is_compatible(frontend, &self.compatible_frontend_families)
            })
        {
            issues.push("frontend_kind_compatible");
        }
        if self
            .extraction_status
            .as_deref()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            issues.push("extraction_status");
        }
        if self
            .blocker_status
            .as_deref()
            .filter(|value| !value.is_empty())
            .is_none()
        {
            issues.push("blocker_status");
        }
        if self
            .first_beneficiary
            .as_deref()
            .is_some_and(is_placeholder_beneficiary)
        {
            issues.push("first_beneficiary_concrete");
        }
        if self
            .second_beneficiary
            .as_deref()
            .is_some_and(is_placeholder_beneficiary)
        {
            issues.push("second_beneficiary_concrete");
        }
        if self
            .origin_frontend
            .as_deref()
            .is_some_and(is_noncanonical_frontend_reference)
        {
            issues.push("origin_frontend_family_canonical");
        }
        if self
            .first_beneficiary
            .as_deref()
            .is_some_and(is_noncanonical_frontend_reference)
        {
            issues.push("first_beneficiary_family_canonical");
        }
        if self
            .second_beneficiary
            .as_deref()
            .is_some_and(is_noncanonical_frontend_reference)
        {
            issues.push("second_beneficiary_family_canonical");
        }
        if let (Some(origin), Some(second)) = (
            self.origin_frontend.as_deref(),
            self.second_beneficiary.as_deref(),
        ) {
            if normalized_token(origin) == normalized_token(second)
                || canonical_frontend_family(origin)
                    .zip(canonical_frontend_family(second))
                    .is_some_and(|(origin, second)| origin == second)
            {
                issues.push("second_beneficiary_distinct");
            }
        }
        issues
    }

    /// Whether every shared-engine contract requirement is met (no issues).
    pub fn is_shared_engine_contract_valid(&self) -> bool {
        self.shared_engine_contract_issues().is_empty()
    }

    /// Renders shared setup timing rows.
    ///
    /// Rows include both `source_kind` and `frontend_kind`: `source_kind`
    /// identifies the concrete payload/interchange family, while
    /// `frontend_kind` identifies the canonical shared-engine adoption family.
    /// `frontend_family_compatibility` records whether that canonical family is
    /// present in `compatible_frontend_families`, which keeps engine-owned rows
    /// distinct from adapter-local source-kind labels.
    /// The older `frontend` and `lane` aliases remain present for existing
    /// evidence consumers.
    pub fn render_evidence_rows(&self, scope: &str) -> Vec<String> {
        self.timings
            .iter()
            .map(|timing| {
                let identities = timing.key.identities.merged_with_fallback(&self.identities);
                let frontend_family_code = timing.key.frontend.frontend_family_code();
                let frontend_family_compatibility = frontend_family_compatibility_status(
                    timing.key.frontend,
                    &self.compatible_frontend_families,
                );
                format!(
                    "{} {} schema={} schema_version={} source_kind={} frontend_kind={} frontend={} lane_kind={} lane={} candidate_key={} candidate_identity={} lane_identity={} cache_key={} source_fingerprint={} frontend_payload_identity={} prepared_program_fingerprint={} artifact_identity={} artifact_fingerprint={} storage_policy_identity={} storage_layout_fingerprint={} fingerprint_policy_identity={} fingerprint_identity={} batch_artifact_identity={} proof_or_witness_fingerprint={} replay_transcript_fingerprint={} publication_fingerprint={} source_identity={} property_identity={} origin_frontend={} shared_engine_component={} first_beneficiary={} second_beneficiary={} compatible_frontend_families={} frontend_family_compatibility={} extraction_status={} blocker_status={} validation_status={} phase={} nanos={}",
                    scope,
                    SETUP_TRACE_ROW_KIND,
                    SETUP_TRACE_SCHEMA,
                    SETUP_TRACE_SCHEMA_VERSION,
                    timing.key.frontend.code(),
                    frontend_family_code,
                    frontend_family_code,
                    timing.key.lane.code(),
                    timing.key.lane.code(),
                    evidence_value(timing.key.candidate_key.as_deref()),
                    evidence_value(identities.candidate_identity.as_deref()),
                    evidence_value(identities.lane_identity.as_deref()),
                    evidence_value(identities.cache_key.as_deref()),
                    evidence_value(identities.source_fingerprint.as_deref()),
                    evidence_value(identities.frontend_payload_identity.as_deref()),
                    evidence_value(identities.prepared_program_fingerprint.as_deref()),
                    evidence_value(identities.artifact_identity.as_deref()),
                    evidence_value(identities.artifact_fingerprint.as_deref()),
                    evidence_value(identities.storage_policy_identity.as_deref()),
                    evidence_value(identities.storage_layout_fingerprint.as_deref()),
                    evidence_value(identities.fingerprint_policy_identity.as_deref()),
                    evidence_value(identities.fingerprint_identity.as_deref()),
                    evidence_value(identities.batch_artifact_identity.as_deref()),
                    evidence_value(identities.proof_or_witness_fingerprint.as_deref()),
                    evidence_value(identities.replay_transcript_fingerprint.as_deref()),
                    evidence_value(identities.publication_fingerprint.as_deref()),
                    evidence_value(self.source_identity.as_deref()),
                    evidence_value(self.property_identity.as_deref()),
                    evidence_value(self.origin_frontend.as_deref()),
                    evidence_value(self.shared_engine_component.as_deref()),
                    evidence_value(self.first_beneficiary.as_deref()),
                    evidence_value(self.second_beneficiary.as_deref()),
                    evidence_list_value(&self.compatible_frontend_families),
                    frontend_family_compatibility,
                    evidence_value(self.extraction_status.as_deref()),
                    evidence_value(self.blocker_status.as_deref()),
                    self.validation_status.code(),
                    timing.phase.code(),
                    timing.nanos
                )
            })
            .collect()
    }

    fn first_phase_nanos(&self, phase: SetupTracePhase) -> Option<u64> {
        self.timings
            .iter()
            .find(|item| item.phase == phase)
            .map(|item| item.nanos)
    }
}

fn non_empty_string(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn evidence_value(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .map(|value| value.replace(char::is_whitespace, "_"))
        .unwrap_or_else(|| "none".to_string())
}

fn evidence_list_value(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values
            .iter()
            .map(|value| evidence_value(Some(value.as_str())))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn normalized_token(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn canonical_frontend_family(value: &str) -> Option<String> {
    let normalized = normalized_token(value);
    let family = match normalized.as_str() {
        "tla" | "tla_plus" | "ty" => SharedEngineFrontendFamily::TlaPlus,
        "quint" => SharedEngineFrontendFamily::Quint,
        "mcc" | "petri" | "mcc_petri" | "pnml" | "hlpnml" => SharedEngineFrontendFamily::MccPetri,
        "aiger" => SharedEngineFrontendFamily::Aiger,
        "btor" | "btor2" => SharedEngineFrontendFamily::Btor2,
        "vmt" | "vmt_interchange" | "vmt_transition_system" => {
            SharedEngineFrontendFamily::VmtTransitionSystem
        }
        "ay" | "ay_only" | "ay_analytical" | "analytical" | "symbolic" => {
            SharedEngineFrontendFamily::AYAnalytical
        }
        "witness" | "replay" | "witness_replay" | "certificate" => {
            SharedEngineFrontendFamily::WitnessReplay
        }
        "future" | "future_importer" | "importer" => SharedEngineFrontendFamily::FutureImporter,
        _ => return None,
    };
    Some(family.code().to_string())
}

fn frontend_family_compatibility_status(
    frontend: CheckerSourceKind,
    compatible_frontend_families: &[String],
) -> &'static str {
    match frontend.adoption_frontend_family() {
        Some(family)
            if compatible_frontend_families
                .iter()
                .any(|compatible| compatible == family.code()) =>
        {
            "compatible"
        }
        Some(_) => "missing_compatible_family",
        None => "unknown_frontend_family",
    }
}

fn frontend_family_is_compatible(
    frontend: CheckerSourceKind,
    compatible_frontend_families: &[String],
) -> bool {
    matches!(
        frontend_family_compatibility_status(frontend, compatible_frontend_families),
        "compatible" | "unknown_frontend_family"
    )
}

fn is_placeholder_beneficiary(value: &str) -> bool {
    matches!(
        normalized_token(value).as_str(),
        "none"
            | "unknown"
            | "origin_frontend"
            | "frontend_family"
            | "compatible_frontend_family"
            | "compatible_frontend_families"
            | "first_beneficiary"
            | "second_beneficiary"
    )
}

fn is_noncanonical_frontend_reference(value: &str) -> bool {
    canonical_frontend_family(value).is_some_and(|family| family != value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_trace_records_and_replaces_phase_duration() {
        let mut trace = SetupTrace::new(CheckerSourceKind::Tla)
            .with_source_identity("MCL.tla")
            .with_property_identity("TypeOK");

        trace.record_duration(SetupTracePhase::SemanticLowering, Duration::from_millis(2));
        trace.record_duration(SetupTracePhase::SemanticLowering, Duration::from_millis(3));
        trace.record_duration(SetupTracePhase::TrustCgCodegen, Duration::from_nanos(9));

        assert_eq!(
            trace.phase_nanos(SetupTracePhase::SemanticLowering),
            Some(3_000_000)
        );
        assert_eq!(trace.timings().len(), 2);

        let rows = trace.render_evidence_rows("TY");
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains("source_kind=tla"));
        assert!(rows[0].contains("frontend_kind=tla_plus"));
        assert!(rows[0].contains("frontend=tla_plus"));
        assert!(rows[0].contains("lane_kind=unknown"));
        assert!(rows[0].contains("lane=unknown"));
        assert!(rows[0].contains("candidate_key=none"));
        assert!(rows[0].contains("source_identity=MCL.tla"));
        assert!(rows[0].contains("phase=semantic_lowering"));
    }

    #[test]
    fn setup_trace_records_same_phase_for_distinct_lane_candidates() {
        let mut trace = SetupTrace::new(CheckerSourceKind::Aiger)
            .with_lane(SetupTraceLaneKind::Native)
            .with_candidate_key("trust-cg native");
        let native_key = trace.key();
        let ay_key = SetupTraceKey::new(CheckerSourceKind::Aiger, SetupTraceLaneKind::AY)
            .with_candidate_key("ay chc");

        trace.record_duration(SetupTracePhase::TotalWall, Duration::from_millis(5));
        trace.record_duration_for_key(
            ay_key.clone(),
            SetupTracePhase::TotalWall,
            Duration::from_millis(7),
        );
        trace.record_duration_for_key(
            ay_key.clone(),
            SetupTracePhase::TotalWall,
            Duration::from_millis(9),
        );

        assert_eq!(
            trace.phase_nanos(SetupTracePhase::TotalWall),
            Some(5_000_000)
        );
        assert_eq!(
            trace.phase_nanos_for_key(&native_key, SetupTracePhase::TotalWall),
            Some(5_000_000)
        );
        assert_eq!(
            trace.phase_nanos_for_key(&ay_key, SetupTracePhase::TotalWall),
            Some(9_000_000)
        );
        assert_eq!(trace.timings().len(), 2);

        let rows = trace.render_evidence_rows("AIGER");
        assert!(rows[0].contains("frontend_kind=aiger"));
        assert!(rows[0].contains("lane_kind=native"));
        assert!(rows[0].contains("frontend=aiger"));
        assert!(rows[0].contains("lane=native"));
        assert!(rows[0].contains("candidate_key=trust-cg_native"));
        assert!(rows[1].contains("frontend_kind=aiger"));
        assert!(rows[1].contains("lane_kind=ay"));
        assert!(rows[1].contains("frontend=aiger"));
        assert!(rows[1].contains("lane=ay"));
        assert!(rows[1].contains("candidate_key=ay_chc"));
    }

    #[test]
    fn setup_trace_lane_codes_cover_shared_candidate_lanes() {
        assert_eq!(SetupTraceLaneKind::Frontend.code(), "frontend");
        assert_eq!(SetupTraceLaneKind::Native.code(), "native");
        assert_eq!(SetupTraceLaneKind::AY.code(), "ay");
        assert_eq!(SetupTraceLaneKind::Analytical.code(), "analytical");
        assert_eq!(SetupTraceLaneKind::Replay.code(), "replay");
        assert_eq!(SetupTraceLaneKind::Fingerprint.code(), "fingerprint");
    }

    #[test]
    fn setup_trace_schema_and_fingerprint_phase_contract_are_stable() {
        assert_eq!(SETUP_TRACE_ROW_KIND, "setup_trace");
        assert_eq!(SETUP_TRACE_SCHEMA, "ty.shared.setup_trace.v1");
        assert_eq!(SETUP_TRACE_SCHEMA_VERSION, 1);
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"source_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"prepared_program_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"artifact_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"storage_layout_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"fingerprint_policy_identity"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"batch_artifact_identity"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"proof_or_witness_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"replay_transcript_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"publication_fingerprint"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"origin_frontend"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"shared_engine_component"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"first_beneficiary"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"second_beneficiary"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"compatible_frontend_families"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"frontend_family_compatibility"));
        assert!(SETUP_TRACE_REQUIRED_FIELDS.contains(&"validation_status"));

        assert_eq!(
            SETUP_TRACE_FINGERPRINT_PHASES
                .iter()
                .map(|phase| phase.code())
                .collect::<Vec<_>>(),
            vec![
                "fingerprint_policy_resolution",
                "fingerprint_storage_setup",
                "fingerprint_canonicalization",
                "fingerprint_compute",
                "fingerprint_dedup",
            ]
        );

        let mut trace = SetupTrace::new(CheckerSourceKind::Btor2)
            .with_lane(SetupTraceLaneKind::Fingerprint)
            .with_fingerprint_policy_identity("register fp policy")
            .with_fingerprint_identity("register vector fp");
        trace.record_duration(
            SetupTracePhase::FingerprintPolicyResolution,
            Duration::from_nanos(17),
        );

        let rows = trace.render_evidence_rows("CORE");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("setup_trace"));
        assert!(rows[0].contains("schema=ty.shared.setup_trace.v1"));
        assert!(rows[0].contains("schema_version=1"));
        assert!(rows[0].contains("source_kind=btor2"));
        assert!(rows[0].contains("lane_kind=fingerprint"));
        assert!(rows[0].contains("phase=fingerprint_policy_resolution"));
        assert!(rows[0].contains("fingerprint_policy_identity=register_fp_policy"));
        assert!(rows[0].contains("fingerprint_identity=register_vector_fp"));
    }

    #[test]
    fn setup_trace_source_kind_codes_cover_shared_frontend_families() {
        assert_eq!(CheckerSourceKind::Tla.code(), "tla");
        assert_eq!(CheckerSourceKind::Tla.frontend_family_code(), "tla_plus");
        assert_eq!(CheckerSourceKind::Quint.code(), "quint");
        assert_eq!(CheckerSourceKind::Quint.frontend_family_code(), "quint");
        assert_eq!(CheckerSourceKind::MccPetri.code(), "mcc_petri");
        assert_eq!(
            CheckerSourceKind::MccPetri.frontend_family_code(),
            "mcc_petri"
        );
        assert_eq!(CheckerSourceKind::Aiger.code(), "aiger");
        assert_eq!(CheckerSourceKind::Aiger.frontend_family_code(), "aiger");
        assert_eq!(CheckerSourceKind::Btor2.code(), "btor2");
        assert_eq!(CheckerSourceKind::Btor2.frontend_family_code(), "btor2");
        assert_eq!(CheckerSourceKind::VmtInterchange.code(), "vmt_interchange");
        assert_eq!(
            CheckerSourceKind::VmtInterchange.frontend_family_code(),
            "vmt_transition_system"
        );
        assert_eq!(CheckerSourceKind::AYOnly.code(), "ay_only");
        assert_eq!(
            CheckerSourceKind::AYOnly.frontend_family_code(),
            "ay_analytical"
        );
        assert_eq!(CheckerSourceKind::WitnessReplay.code(), "witness_replay");
        assert_eq!(
            CheckerSourceKind::WitnessReplay.frontend_family_code(),
            "witness_replay"
        );
        assert_eq!(CheckerSourceKind::Unknown.code(), "unknown");
        assert_eq!(CheckerSourceKind::Unknown.frontend_family_code(), "unknown");
    }

    #[test]
    fn setup_trace_evidence_explains_shared_identity_fields() {
        let mut trace = SetupTrace::new(CheckerSourceKind::Quint)
            .with_lane(SetupTraceLaneKind::ExplicitState)
            .with_candidate_key("explicit bfs")
            .with_cache_key("prepared cache key")
            .with_frontend_payload_identity("quint lowered tla payload")
            .with_artifact_identity("prepared artifact")
            .with_storage_policy_identity("slot layout v3")
            .with_fingerprint_policy_identity("stable fp policy")
            .with_fingerprint_identity("state fingerprint namespace")
            .with_batch_artifact_identity("batch artifact 42")
            .with_candidate_identity("candidate explicit")
            .with_lane_identity("lane explicit state");

        trace.record_duration(
            SetupTracePhase::PreparedProgramBuild,
            Duration::from_millis(11),
        );

        let key = trace.key();
        assert_eq!(
            key.identities.cache_key.as_deref(),
            Some("prepared cache key")
        );
        assert_eq!(
            trace.phase_nanos_for_key(&key, SetupTracePhase::PreparedProgramBuild),
            Some(11_000_000)
        );

        let rows = trace.render_evidence_rows("QUINT");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("frontend_kind=quint"));
        assert!(rows[0].contains("lane_kind=explicit_state"));
        assert!(rows[0].contains("candidate_key=explicit_bfs"));
        assert!(rows[0].contains("candidate_identity=candidate_explicit"));
        assert!(rows[0].contains("lane_identity=lane_explicit_state"));
        assert!(rows[0].contains("cache_key=prepared_cache_key"));
        assert!(rows[0].contains("frontend_payload_identity=quint_lowered_tla_payload"));
        assert!(rows[0].contains("artifact_identity=prepared_artifact"));
        assert!(rows[0].contains("storage_policy_identity=slot_layout_v3"));
        assert!(rows[0].contains("fingerprint_policy_identity=stable_fp_policy"));
        assert!(rows[0].contains("fingerprint_identity=state_fingerprint_namespace"));
        assert!(rows[0].contains("batch_artifact_identity=batch_artifact_42"));
    }

    #[test]
    fn setup_trace_evidence_explains_shared_fingerprint_chain_and_validation() {
        let mut trace = SetupTrace::new(CheckerSourceKind::VmtInterchange)
            .with_lane(SetupTraceLaneKind::Replay)
            .with_source_identity("input.vmt")
            .with_property_identity("safety")
            .with_origin_frontend("vmt")
            .with_shared_engine_component("tla_mc_core.setup_trace")
            .with_first_beneficiary("vmt")
            .with_second_beneficiary("btor2")
            .with_compatible_frontend_families(["aiger", "btor2", "vmt"])
            .with_shared_engine_extraction_status("already-shared")
            .with_shared_engine_blocker_status("no-blockers")
            .with_validation_status(SetupTraceValidationStatus::Accepted)
            .with_source_fingerprint("source:vmt:abc")
            .with_prepared_program_fingerprint("prepared:v1:def")
            .with_storage_layout_fingerprint("storage:v1:ghi")
            .with_artifact_fingerprint("artifact:ay:jkl")
            .with_proof_or_witness_fingerprint("witness:v1:mno")
            .with_replay_transcript_fingerprint("replay:v1:pqr")
            .with_publication_fingerprint("publication:v1:stu");

        trace.record_duration(SetupTracePhase::WitnessReplay, Duration::from_micros(4));

        assert!(trace.is_shared_engine_contract_valid());
        assert_eq!(
            trace.phase_nanos(SetupTracePhase::WitnessReplay),
            Some(4_000)
        );

        let rows = trace.render_evidence_rows("VMT");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("source_kind=vmt_interchange"));
        assert!(rows[0].contains("frontend_kind=vmt_transition_system"));
        assert!(rows[0].contains("frontend=vmt_transition_system"));
        assert!(rows[0].contains("source_fingerprint=source:vmt:abc"));
        assert!(rows[0].contains("prepared_program_fingerprint=prepared:v1:def"));
        assert!(rows[0].contains("storage_layout_fingerprint=storage:v1:ghi"));
        assert!(rows[0].contains("artifact_fingerprint=artifact:ay:jkl"));
        assert!(rows[0].contains("proof_or_witness_fingerprint=witness:v1:mno"));
        assert!(rows[0].contains("replay_transcript_fingerprint=replay:v1:pqr"));
        assert!(rows[0].contains("publication_fingerprint=publication:v1:stu"));
        assert!(rows[0].contains("origin_frontend=vmt_transition_system"));
        assert!(rows[0].contains("shared_engine_component=tla_mc_core.setup_trace"));
        assert!(rows[0].contains("first_beneficiary=vmt_transition_system"));
        assert!(rows[0].contains("second_beneficiary=btor2"));
        assert!(rows[0].contains("compatible_frontend_families=aiger,btor2,vmt_transition_system"));
        assert!(rows[0].contains("frontend_family_compatibility=compatible"));
        assert!(rows[0].contains("extraction_status=already-shared"));
        assert!(rows[0].contains("blocker_status=no-blockers"));
        assert!(rows[0].contains("validation_status=accepted"));
    }

    #[test]
    fn setup_trace_cross_frontend_contract_is_not_tla_specific() {
        let mut trace = SetupTrace::new(CheckerSourceKind::Aiger)
            .with_lane(SetupTraceLaneKind::Native)
            .with_origin_frontend("aiger")
            .with_shared_engine_component("tla_mc_core.native_shared_engine")
            .with_first_beneficiary("btor2")
            .with_second_beneficiary("vmt")
            .with_compatible_frontend_families(["btor2", "vmt", "aiger", "ay_only"])
            .with_shared_engine_extraction_status("shared-core-extracted")
            .with_shared_engine_blocker_status("tracked-blockers")
            .with_validation_status(SetupTraceValidationStatus::Accepted);

        trace.record_duration(SetupTracePhase::NativePublish, Duration::from_nanos(5));

        assert!(trace.is_shared_engine_contract_valid());
        assert_eq!(trace.origin_frontend.as_deref(), Some("aiger"));
        assert_eq!(trace.first_beneficiary.as_deref(), Some("btor2"));
        assert_eq!(
            trace.second_beneficiary.as_deref(),
            Some("vmt_transition_system")
        );
        assert_eq!(
            trace.compatible_frontend_families,
            vec!["aiger", "ay_analytical", "btor2", "vmt_transition_system"]
        );

        let rows = trace.render_evidence_rows("AIGER");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("source_kind=aiger"));
        assert!(rows[0].contains("origin_frontend=aiger"));
        assert!(rows[0].contains("second_beneficiary=vmt_transition_system"));
        assert!(rows[0].contains(
            "compatible_frontend_families=aiger,ay_analytical,btor2,vmt_transition_system"
        ));
        assert!(rows[0].contains("frontend_family_compatibility=compatible"));
        assert!(!rows[0].contains("tla_plus"));
    }

    #[test]
    fn setup_trace_contract_rejects_timing_frontend_missing_from_compatible_families() {
        let mut trace = SetupTrace::new(CheckerSourceKind::Aiger)
            .with_lane(SetupTraceLaneKind::Native)
            .with_origin_frontend("aiger")
            .with_shared_engine_component("tla_mc_core.native_shared_engine")
            .with_first_beneficiary("btor2")
            .with_second_beneficiary("vmt")
            .with_compatible_frontend_families(["aiger", "vmt", "ay_only"])
            .with_shared_engine_extraction_status("shared-core-extracted")
            .with_shared_engine_blocker_status("tracked-blockers")
            .with_validation_status(SetupTraceValidationStatus::Rejected);

        trace.record_duration(SetupTracePhase::NativePublish, Duration::from_nanos(5));
        trace.record_duration_for_key(
            SetupTraceKey::new(CheckerSourceKind::Btor2, SetupTraceLaneKind::Native),
            SetupTracePhase::NativePublish,
            Duration::from_nanos(8),
        );

        let issues = trace.shared_engine_contract_issues();

        assert!(issues.contains(&"frontend_kind_compatible"));
        assert!(!trace.is_shared_engine_contract_valid());
        let rows = trace.render_evidence_rows("AIGER");
        assert!(rows[0].contains("frontend_kind=aiger"));
        assert!(rows[0].contains("frontend_family_compatibility=compatible"));
        assert!(rows[1].contains("frontend_kind=btor2"));
        assert!(rows[1].contains("frontend_family_compatibility=missing_compatible_family"));
    }

    #[test]
    fn setup_trace_preserves_source_kind_codes_while_publishing_canonical_families() {
        let mut trace = SetupTrace::new(CheckerSourceKind::Tla)
            .with_lane(SetupTraceLaneKind::ExplicitState)
            .with_origin_frontend("tla")
            .with_shared_engine_component("tla_mc_core.setup_trace")
            .with_first_beneficiary("tla cli")
            .with_second_beneficiary("quint cli")
            .with_compatible_frontend_families(["tla", "quint", "mcc_petri"])
            .with_shared_engine_extraction_status("already-shared")
            .with_shared_engine_blocker_status("tracked-blockers")
            .with_validation_status(SetupTraceValidationStatus::Accepted);

        trace.record_duration(
            SetupTracePhase::PreparedProgramBuild,
            Duration::from_nanos(1),
        );

        assert!(trace.is_shared_engine_contract_valid());
        assert_eq!(CheckerSourceKind::Tla.code(), "tla");
        assert_eq!(
            trace.compatible_frontend_families,
            vec!["mcc_petri", "quint", "tla_plus"]
        );

        let rows = trace.render_evidence_rows("TY");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("origin_frontend=tla_plus"));
        assert!(rows[0].contains("source_kind=tla"));
        assert!(rows[0].contains("frontend_kind=tla_plus"));
        assert!(rows[0].contains("frontend=tla_plus"));
        assert!(rows[0].contains("compatible_frontend_families=mcc_petri,quint,tla_plus"));
        assert!(rows[0].contains("frontend_family_compatibility=compatible"));
    }

    #[test]
    fn setup_trace_shared_engine_contract_rejects_frontend_local_claims() {
        let trace = SetupTrace::new(CheckerSourceKind::MccPetri)
            .with_origin_frontend("mcc_petri")
            .with_shared_engine_component("tla_mc_core.prepared_checker_program")
            .with_first_beneficiary("mcc_petri")
            .with_second_beneficiary("mcc petri")
            .with_compatible_frontend_family("mcc_petri")
            .with_validation_status(SetupTraceValidationStatus::Rejected);

        let issues = trace.shared_engine_contract_issues();

        assert!(issues.contains(&"compatible_frontend_families"));
        assert!(issues.contains(&"second_beneficiary_distinct"));
        assert!(!trace.is_shared_engine_contract_valid());
    }

    #[test]
    fn setup_trace_shared_engine_contract_rejects_placeholder_beneficiaries() {
        let trace = SetupTrace::new(CheckerSourceKind::Quint)
            .with_origin_frontend("quint")
            .with_shared_engine_component("tla_mc_core.prepared_checker_program")
            .with_first_beneficiary("origin_frontend")
            .with_second_beneficiary("compatible_frontend_family")
            .with_compatible_frontend_families(["tla_plus", "quint", "aiger"])
            .with_shared_engine_extraction_status("already-shared")
            .with_shared_engine_blocker_status("tracked-blockers")
            .with_validation_status(SetupTraceValidationStatus::Rejected);

        let issues = trace.shared_engine_contract_issues();

        assert!(issues.contains(&"first_beneficiary_concrete"));
        assert!(issues.contains(&"second_beneficiary_concrete"));
        assert!(!trace.is_shared_engine_contract_valid());
    }

    #[test]
    fn setup_trace_contract_rejects_publicly_mutated_noncanonical_family_aliases() {
        let mut trace = SetupTrace::new(CheckerSourceKind::VmtInterchange)
            .with_origin_frontend("vmt")
            .with_shared_engine_component("tla_mc_core.setup_trace")
            .with_first_beneficiary("btor2")
            .with_second_beneficiary("aiger")
            .with_compatible_frontend_families(["vmt", "btor2", "aiger"])
            .with_shared_engine_extraction_status("shared-core-extracted")
            .with_shared_engine_blocker_status("tracked-blockers")
            .with_validation_status(SetupTraceValidationStatus::Rejected);

        assert!(trace.is_shared_engine_contract_valid());
        trace.origin_frontend = Some("vmt".to_string());
        trace.second_beneficiary = Some("tla".to_string());
        trace
            .compatible_frontend_families
            .push("ay_only".to_string());

        let issues = trace.shared_engine_contract_issues();

        assert!(issues.contains(&"origin_frontend_family_canonical"));
        assert!(issues.contains(&"second_beneficiary_family_canonical"));
        assert!(issues.contains(&"compatible_frontend_family_codes"));
        assert!(!trace.is_shared_engine_contract_valid());
    }

    #[test]
    fn setup_trace_key_identity_distinguishes_same_lane_artifacts() {
        let mut trace = SetupTrace::new(CheckerSourceKind::MccPetri)
            .with_lane(SetupTraceLaneKind::Analytical)
            .with_cache_key("shared payload")
            .with_frontend_payload_identity("mcc canonical payload");

        let ay_key = SetupTraceKey::new(CheckerSourceKind::MccPetri, SetupTraceLaneKind::AY)
            .with_candidate_key("ay")
            .with_cache_key("ay artifact")
            .with_frontend_payload_identity("ay canonical payload")
            .with_artifact_identity("ay chc artifact")
            .with_lane_identity("lane ay");
        let native_key =
            SetupTraceKey::new(CheckerSourceKind::MccPetri, SetupTraceLaneKind::Native)
                .with_candidate_key("native")
                .with_cache_key("native artifact")
                .with_artifact_identity("trust-cg artifact")
                .with_lane_identity("lane native");

        trace.record_duration_for_key(
            ay_key.clone(),
            SetupTracePhase::HotExecution,
            Duration::from_millis(13),
        );
        trace.record_duration_for_key(
            native_key.clone(),
            SetupTracePhase::HotExecution,
            Duration::from_millis(17),
        );

        assert_eq!(
            trace.phase_nanos_for_key(&ay_key, SetupTracePhase::HotExecution),
            Some(13_000_000)
        );
        assert_eq!(
            trace.phase_nanos_for_key(&native_key, SetupTracePhase::HotExecution),
            Some(17_000_000)
        );

        let rows = trace.render_evidence_rows("MCC");
        assert!(rows[0].contains("cache_key=ay_artifact"));
        assert!(rows[0].contains("frontend_payload_identity=ay_canonical_payload"));
        assert!(rows[0].contains("artifact_identity=ay_chc_artifact"));
        assert!(rows[1].contains("cache_key=native_artifact"));
        assert!(rows[1].contains("frontend_payload_identity=mcc_canonical_payload"));
        assert!(rows[1].contains("artifact_identity=trust-cg_artifact"));
    }

    #[test]
    fn identity_fields_merge_deterministically_with_primary_precedence() {
        let primary = CheckerArtifactIdentityFields::new()
            .with_cache_key("lane cache")
            .with_lane_identity("lane id");
        let fallback = CheckerArtifactIdentityFields::new()
            .with_cache_key("program cache")
            .with_storage_policy_identity("slot layout")
            .with_fingerprint_identity("state fp");

        let merged = primary.merged_with_fallback(&fallback);
        assert_eq!(merged.cache_key.as_deref(), Some("lane cache"));
        assert_eq!(merged.lane_identity.as_deref(), Some("lane id"));
        assert_eq!(
            merged.storage_policy_identity.as_deref(),
            Some("slot layout")
        );
        assert_eq!(merged.fingerprint_identity.as_deref(), Some("state fp"));
    }
}
