// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral whole-program kernel metadata.
//!
//! This module is intentionally metadata-only. It gives TLA state slots, Petri
//! marking vectors, and hardware register vectors one small representation for
//! backend planning, replay evidence, and fingerprint correlation without
//! requiring any frontend crate to depend on another frontend's vocabulary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use thiserror::Error;

/// Stable schema label for [`WholeProgramKernelMetadata`].
pub const WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA: &str = "tla_ir.whole_program_kernel_metadata.v2";

/// Stable identity basis for [`WholeProgramKernelMetadata::stable_identity_fingerprint_hex`].
pub const WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS: &str =
    "tla_ir.whole_program_kernel_metadata.canonical_identity.v2";

/// Stable schema version for [`WholeProgramKernelMetadata`].
pub const WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION: u32 = 2;

/// Shared owner for the frontend-neutral kernel contract.
pub const WHOLE_PROGRAM_KERNEL_SHARED_OWNER: &str = "shared_high_performance_engine";

/// Shared-engine component name for whole-program kernel metadata.
pub const WHOLE_PROGRAM_KERNEL_SHARED_ENGINE_COMPONENT: &str =
    "tla_ir.whole_program_kernel_metadata";

/// Adoption extraction status for the shared whole-program kernel vocabulary.
pub const WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS: &str = "already-shared";

/// Blocker status for the shared whole-program kernel vocabulary.
pub const WHOLE_PROGRAM_KERNEL_BLOCKER_STATUS: &str = "no-blockers";

/// Shared evidence vocabulary version for [`WholeProgramKernelMetadata::render_evidence_row`].
pub const WHOLE_PROGRAM_KERNEL_EVIDENCE_ROW_KIND: &str = "trust_ir_whole_program_kernel";

/// Frontend families expected to consume the whole-program kernel vocabulary.
pub const WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer";

/// Shared admission surface for frontend-neutral fingerprint/state-vector storage.
pub const WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SURFACE: &str =
    "shared_fingerprint_state_vector_admission";

/// Stable admission states published by the shared fingerprint/state-vector contract.
pub const WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SEMANTICS: &str =
    "default_consumer,compatible_consumer,blocked";

/// Concrete frontend families admitted by the shared fingerprint/state-vector contract.
pub const WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";

/// Reserved future-importer blocker until a canonical importer mapping exists.
pub const WHOLE_PROGRAM_KERNEL_FINGERPRINT_FUTURE_IMPORTER_BLOCKER: &str =
    "future_importer:awaiting_registered_importer_frontend";

/// Frontend families blocked by the shared fingerprint/state-vector contract.
pub const WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES: &str =
    WHOLE_PROGRAM_KERNEL_FINGERPRINT_FUTURE_IMPORTER_BLOCKER;

/// Runtime/compile layer where kernel optimizations are meant to be shared.
pub const WHOLE_PROGRAM_KERNEL_OPTIMIZATION_LAYER: &str = "below_frontend_adapters";

/// Canonical identity scope for frontend-neutral kernel/layout correlation.
pub const WHOLE_PROGRAM_KERNEL_IDENTITY_SCOPE: &str = "frontend_neutral_kernel_layout";

const COMPATIBLE_FRONTEND_FAMILY_CODES: &[&str] = &[
    "tla_plus",
    "quint",
    "mcc_petri",
    "aiger",
    "btor2",
    "vmt_transition_system",
    "ay_analytical",
    "witness_replay",
    "future_importer",
];

/// Frontend family that supplied a whole-program kernel.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelFrontend {
    /// TLA+ state-machine frontend.
    Tla,
    /// Quint source lowered through the TLA/trust-ir pipeline.
    Quint,
    /// MCC/PNML Petri-net frontend.
    MccPetri,
    /// AIGER hardware-transition frontend.
    Aiger,
    /// BTOR2 hardware-transition frontend.
    Btor2,
    /// VMT transition-system frontend.
    VmtReplay,
    /// AY analytical symbolic frontend.
    AYOnlyHelper,
    /// Witness replay frontend.
    WitnessReplay,
    /// Future importer family reserved before a dedicated variant exists.
    FutureImporter,
    /// Frontend not yet modeled by a dedicated variant; published as `future_importer`.
    Other(String),
}

impl KernelFrontend {
    /// Stable frontend-family code (e.g. `tla_plus`, `mcc_petri`). Both
    /// [`FutureImporter`](Self::FutureImporter) and [`Other`](Self::Other)
    /// publish as `future_importer`.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::Tla => "tla_plus",
            Self::Quint => "quint",
            Self::MccPetri => "mcc_petri",
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::VmtReplay => "vmt_transition_system",
            Self::AYOnlyHelper => "ay_analytical",
            Self::WitnessReplay => "witness_replay",
            Self::FutureImporter => "future_importer",
            Self::Other(_) => "future_importer",
        }
    }

    /// Alias for [`code`](Self::code): the stable frontend-family string.
    #[must_use]
    pub fn as_stable_str(&self) -> &str {
        self.code()
    }

    /// First beneficiary of shared kernel work: the frontend's own family code.
    #[must_use]
    pub fn first_beneficiary(&self) -> &str {
        self.code()
    }

    /// A second frontend family that demonstrably reuses this frontend's kernel
    /// work (e.g. TLA pairs with Quint, AIGER pairs with BTOR2).
    #[must_use]
    pub fn second_beneficiary(&self) -> &'static str {
        match self {
            Self::Tla => "quint",
            Self::Quint => "tla_plus",
            Self::MccPetri => "tla_plus",
            Self::Aiger => "btor2",
            Self::Btor2 => "aiger",
            Self::VmtReplay => "ay_analytical",
            Self::AYOnlyHelper => "vmt_transition_system",
            Self::WitnessReplay => "tla_plus",
            Self::FutureImporter | Self::Other(_) => "tla_plus",
        }
    }
}

/// Admission relationship for a frontend family on the shared fingerprint surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelFingerprintAdmission {
    /// Consumes the shared fingerprint/state-vector contract without adapter-specific gating.
    DefaultConsumer,
    /// Can consume the contract once wired through its frontend adapter.
    CompatibleConsumer,
    /// Reserved or blocked until a named importer/adapter condition is satisfied.
    Blocked,
}

impl KernelFingerprintAdmission {
    /// Stable admission-state string (`default_consumer`, `compatible_consumer`,
    /// or `blocked`).
    #[must_use]
    pub fn as_stable_str(self) -> &'static str {
        match self {
            Self::DefaultConsumer => "default_consumer",
            Self::CompatibleConsumer => "compatible_consumer",
            Self::Blocked => "blocked",
        }
    }
}

/// Logical storage family represented by a kernel slot/vector element.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelStorageKind {
    /// Flattened TLA state slots.
    TlaStateSlot,
    /// Petri marking vector entries.
    PetriMarking,
    /// Hardware register-vector entries.
    HardwareRegister,
    /// Storage not yet modeled by a dedicated variant.
    Other(String),
}

impl KernelStorageKind {
    /// Stable storage-family string. [`Other`](Self::Other) returns its inner
    /// caller-supplied label verbatim.
    #[must_use]
    pub fn as_stable_str(&self) -> &str {
        match self {
            Self::TlaStateSlot => "tla_state_slot",
            Self::PetriMarking => "petri_marking",
            Self::HardwareRegister => "hardware_register",
            Self::Other(value) => value.as_str(),
        }
    }
}

/// One logical storage element consumed or produced by the kernel.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelStorageMetadata {
    /// Storage family.
    pub kind: KernelStorageKind,
    /// Frontend-neutral stable identifier for this slot/vector entry.
    pub id: String,
    /// Diagnostic name retained for reports; excluded from stable identity.
    pub diagnostic_name: String,
    /// Zero-based position in the frontend's storage vector.
    pub index: u32,
    /// Number of scalar lanes occupied by this element.
    pub lane_count: u32,
    /// Canonical bit width validated for this storage element.
    pub storage_width_bits: u32,
    /// Stable value-domain label, such as `i64`, `bool`, `token_count`, or `bv32`.
    pub value_domain: String,
}

impl KernelStorageMetadata {
    /// Build a storage element, inferring [`storage_width_bits`](Self::storage_width_bits)
    /// from `value_domain` (e.g. `i64` -> 64, `bool` -> 1, `bv32` -> 32), falling
    /// back to `lane_count` for unrecognized domains. Use
    /// [`with_storage_width_bits`](Self::with_storage_width_bits) to override.
    #[must_use]
    pub fn new(
        kind: KernelStorageKind,
        id: impl Into<String>,
        diagnostic_name: impl Into<String>,
        index: u32,
        lane_count: u32,
        value_domain: impl Into<String>,
    ) -> Self {
        let value_domain = value_domain.into();
        Self {
            kind,
            id: id.into(),
            diagnostic_name: diagnostic_name.into(),
            index,
            lane_count,
            storage_width_bits: inferred_storage_width_bits(&value_domain, lane_count),
            value_domain,
        }
    }

    /// Override the canonical bit width for this storage element, replacing the
    /// width inferred by [`new`](Self::new).
    #[must_use]
    pub const fn with_storage_width_bits(mut self, storage_width_bits: u32) -> Self {
        self.storage_width_bits = storage_width_bits;
        self
    }
}

/// Transition entry point or transition relation covered by the kernel.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelTransitionMetadata {
    /// Frontend-neutral stable identifier.
    pub id: String,
    /// Diagnostic name retained for reports; excluded from stable identity.
    pub diagnostic_name: String,
    /// Stable transition family label, such as `next`, `fire`, or `posedge`.
    pub kind: String,
    /// Stable storage IDs read by the transition.
    pub reads: Vec<String>,
    /// Stable storage IDs written by the transition.
    pub writes: Vec<String>,
}

impl KernelTransitionMetadata {
    /// Build a transition with empty read/write sets. Populate them with
    /// [`with_reads`](Self::with_reads) / [`with_writes`](Self::with_writes).
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        diagnostic_name: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            diagnostic_name: diagnostic_name.into(),
            kind: kind.into(),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Set the storage IDs read by this transition (sorted for stable identity).
    #[must_use]
    pub fn with_reads<I, S>(mut self, reads: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.reads = sorted_strings(reads);
        self
    }

    /// Set the storage IDs written by this transition (sorted for stable identity).
    #[must_use]
    pub fn with_writes<I, S>(mut self, writes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.writes = sorted_strings(writes);
        self
    }
}

/// Safety/liveness/property entry point covered by the kernel.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelPropertyMetadata {
    /// Frontend-neutral stable identifier.
    pub id: String,
    /// Diagnostic name retained for reports; excluded from stable identity.
    pub diagnostic_name: String,
    /// Stable property family label, such as `invariant`, `deadlock`, or `assert`.
    pub kind: String,
    /// Stable storage IDs observed by this property.
    pub observes: Vec<String>,
}

impl KernelPropertyMetadata {
    /// Build a property with an empty observed-storage set. Populate it with
    /// [`with_observes`](Self::with_observes).
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        diagnostic_name: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            diagnostic_name: diagnostic_name.into(),
            kind: kind.into(),
            observes: Vec::new(),
        }
    }

    /// Set the storage IDs observed by this property (sorted for stable identity).
    #[must_use]
    pub fn with_observes<I, S>(mut self, observes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.observes = sorted_strings(observes);
        self
    }
}

/// Proof, validation, analytical, or helper obligation covered by the kernel.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelObligationMetadata {
    /// Frontend-neutral stable obligation identifier.
    pub id: String,
    /// Diagnostic name retained for reports; excluded from stable identity.
    pub diagnostic_name: String,
    /// Stable obligation kind, such as `proof`, `translation_validation`, or `bmc`.
    pub kind: String,
    /// Shared solver/helper family, such as `ay_helper`, `analytical`, or `native_replay`.
    pub solver_family: String,
    /// Stable storage IDs observed by this obligation.
    pub observes: Vec<String>,
}

impl KernelObligationMetadata {
    /// Build an obligation with an empty observed-storage set. Populate it with
    /// [`with_observes`](Self::with_observes).
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        diagnostic_name: impl Into<String>,
        kind: impl Into<String>,
        solver_family: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            diagnostic_name: diagnostic_name.into(),
            kind: kind.into(),
            solver_family: solver_family.into(),
            observes: Vec::new(),
        }
    }

    /// Set the storage IDs observed by this obligation (sorted for stable identity).
    #[must_use]
    pub fn with_observes<I, S>(mut self, observes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.observes = sorted_strings(observes);
        self
    }
}

/// Fingerprint scheme available for states, markings, traces, or proof facts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelFingerprintMetadata {
    /// Stable scheme identifier.
    pub id: String,
    /// Fingerprint domain, such as `state`, `marking`, `register_vector`, or `trace`.
    pub domain: String,
    /// Algorithm label, such as `xxh3`, `siphash`, or `canonical_sha256`.
    pub algorithm: String,
    /// Number of digest bits exposed to callers.
    pub digest_bits: u16,
    /// Stable seed/policy identity. Use `none` when no seed is used.
    pub seed_identity: String,
}

/// Canonical admission fields for shared fingerprint/state-vector evidence rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFingerprintAdmissionContract {
    /// Shared admission surface identifier.
    pub surface: &'static str,
    /// Stable admission-state vocabulary.
    pub semantics: &'static str,
    /// Concrete compatible frontend families.
    pub compatible_frontend_families: &'static str,
    /// Frontend families admitted by default for this producer/consumer row.
    pub default_frontend_families: String,
    /// Frontend families blocked or reserved until a contract exists.
    pub blocked_frontend_families: &'static str,
}

impl KernelFingerprintAdmissionContract {
    /// Build the admission contract for `frontend`: shared surface/semantics
    /// constants plus the default-admitted families derived from this frontend's
    /// own code and second beneficiary (the reserved `future_importer` family is
    /// excluded from the default set).
    #[must_use]
    pub fn for_frontend(frontend: &KernelFrontend) -> Self {
        Self {
            surface: WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SURFACE,
            semantics: WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SEMANTICS,
            compatible_frontend_families:
                WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES,
            default_frontend_families: join_family_codes(
                fingerprint_admission_default_frontend_families(frontend),
            ),
            blocked_frontend_families:
                WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES,
        }
    }
}

impl KernelFingerprintMetadata {
    /// Build a fingerprint scheme descriptor from its stable id, domain,
    /// algorithm, digest width, and seed identity.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        domain: impl Into<String>,
        algorithm: impl Into<String>,
        digest_bits: u16,
        seed_identity: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            domain: domain.into(),
            algorithm: algorithm.into(),
            digest_bits,
            seed_identity: seed_identity.into(),
        }
    }
}

/// Width committed by the validation plan for one storage element.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelValidationStorageWidth {
    /// Storage ID covered by this validation row.
    pub storage_id: String,
    /// Canonical bit width accepted by the validator.
    pub width_bits: u32,
}

impl KernelValidationStorageWidth {
    /// Build a validation-plan width commitment for one storage element.
    #[must_use]
    pub fn new(storage_id: impl Into<String>, width_bits: u32) -> Self {
        Self {
            storage_id: storage_id.into(),
            width_bits,
        }
    }
}

/// Shared validation/fingerprint identity required before adopting a kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelValidationPlanIdentity {
    /// Stable validation-plan identity.
    pub id: String,
    /// Stable fingerprint policy identity paired with this validation plan.
    pub fingerprint_identity: String,
    /// Storage widths validated by this plan.
    pub storage_widths: Vec<KernelValidationStorageWidth>,
}

impl KernelValidationPlanIdentity {
    /// Build a validation plan with no storage widths yet. Add them with
    /// [`with_storage_widths`](Self::with_storage_widths).
    #[must_use]
    pub fn new(id: impl Into<String>, fingerprint_identity: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fingerprint_identity: fingerprint_identity.into(),
            storage_widths: Vec::new(),
        }
    }

    /// Set the per-storage width commitments (sorted for stable identity).
    #[must_use]
    pub fn with_storage_widths<I>(mut self, storage_widths: I) -> Self
    where
        I: IntoIterator<Item = KernelValidationStorageWidth>,
    {
        self.storage_widths = sorted_items(storage_widths);
        self
    }
}

/// Structural reuse evidence for one kernel family or batch shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelStructuralReuseMetadata {
    /// Number of stable trust-ir bodies represented before frontend specialization.
    pub unique_body_count: u32,
    /// Number of frontend/runtime specializations represented by this row.
    pub specialization_count: u32,
    /// Stable trust-ir body digest supplied by the lowering/batching layer.
    pub trust_ir_stable_digest: String,
    /// Process-local link digest; excluded from semantic stable identity.
    pub process_local_link_digest: String,
    /// Canonical frontend families that can reuse this structure.
    pub compatible_frontend_families: Vec<String>,
}

impl KernelStructuralReuseMetadata {
    /// Build structural-reuse evidence with a caller-supplied set of compatible
    /// frontend families (sorted for stable identity).
    #[must_use]
    pub fn new<I, S>(
        unique_body_count: u32,
        specialization_count: u32,
        trust_ir_stable_digest: impl Into<String>,
        process_local_link_digest: impl Into<String>,
        compatible_frontend_families: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            unique_body_count,
            specialization_count,
            trust_ir_stable_digest: trust_ir_stable_digest.into(),
            process_local_link_digest: process_local_link_digest.into(),
            compatible_frontend_families: sorted_strings(compatible_frontend_families),
        }
    }

    /// Build structural-reuse evidence compatible with every known frontend
    /// family (the full [`WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES`] set).
    #[must_use]
    pub fn frontend_neutral(
        unique_body_count: u32,
        specialization_count: u32,
        trust_ir_stable_digest: impl Into<String>,
        process_local_link_digest: impl Into<String>,
    ) -> Self {
        Self::new(
            unique_body_count,
            specialization_count,
            trust_ir_stable_digest,
            process_local_link_digest,
            COMPATIBLE_FRONTEND_FAMILY_CODES.iter().copied(),
        )
    }

    /// The "no reuse evidence yet" sentinel: zero counts and `missing` digests.
    #[must_use]
    pub fn missing() -> Self {
        Self::frontend_neutral(0, 0, "missing", "process-local:missing")
    }
}

/// Validation failures for malformed whole-program kernel metadata.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KernelMetadataValidationError {
    /// No validation plan was attached, or its identity was empty/`missing`/`none`.
    #[error("whole-program kernel metadata is missing validation-plan identity")]
    MissingValidationPlanIdentity,

    /// The validation plan's fingerprint identity was empty/`missing`/`none`.
    #[error("whole-program kernel metadata is missing fingerprint identity")]
    MissingFingerprintIdentity,

    /// No fingerprint schemes were declared.
    #[error("whole-program kernel metadata has no fingerprint metadata")]
    MissingFingerprintMetadata,

    /// A fingerprint scheme had an empty/duplicate id or an empty/zero field.
    #[error(
        "whole-program kernel metadata has invalid fingerprint '{fingerprint_id}' field {field}"
    )]
    InvalidFingerprintMetadata {
        /// Stable id of the offending fingerprint scheme.
        fingerprint_id: String,
        /// The field that failed validation (e.g. `id`, `duplicate_id`, `domain`).
        field: &'static str,
    },

    /// Two storage elements declared the same stable id.
    #[error("whole-program kernel metadata declares duplicate storage id '{storage_id}'")]
    DuplicateStorageId {
        /// The storage id declared more than once.
        storage_id: String,
    },

    /// A storage element declared a zero canonical bit width.
    #[error(
        "whole-program kernel metadata declares invalid storage width {storage_width_bits} for '{storage_id}'"
    )]
    InvalidStorageWidth {
        /// The offending storage id.
        storage_id: String,
        /// The rejected (zero) width.
        storage_width_bits: u32,
    },

    /// The validation plan committed a width for the same storage id twice.
    #[error("whole-program kernel validation plan duplicates storage width for '{storage_id}'")]
    DuplicateValidationStorageWidth {
        /// The storage id with duplicate width commitments.
        storage_id: String,
    },

    /// A declared storage element had no matching width in the validation plan.
    #[error("whole-program kernel validation plan is missing storage width for '{storage_id}'")]
    MissingValidationStorageWidth {
        /// The storage id lacking a validation-plan width.
        storage_id: String,
    },

    /// The validation plan named a storage id absent from the metadata.
    #[error("whole-program kernel validation plan names unknown storage id '{storage_id}'")]
    UnknownValidationStorageWidth {
        /// The unknown storage id referenced by the validation plan.
        storage_id: String,
    },

    /// A storage element's width disagreed with the validation-plan commitment.
    #[error(
        "whole-program kernel validation plan width mismatch for '{storage_id}': metadata {metadata_width_bits}, validation {validation_width_bits}"
    )]
    StorageWidthMismatch {
        /// The storage id whose widths disagreed.
        storage_id: String,
        /// Width declared on the storage metadata.
        metadata_width_bits: u32,
        /// Width committed by the validation plan.
        validation_width_bits: u32,
    },
}

/// Whole-program kernel metadata shared by TLA, Petri, and hardware frontends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WholeProgramKernelMetadata {
    /// Stable schema label.
    pub schema: &'static str,
    /// Stable schema version.
    pub schema_version: u32,
    /// Frontend family that emitted the metadata.
    pub frontend: KernelFrontend,
    /// Diagnostic kernel name retained for reports; excluded from stable identity.
    pub diagnostic_name: String,
    /// Logical storage elements.
    pub storage: Vec<KernelStorageMetadata>,
    /// Transition metadata.
    pub transitions: Vec<KernelTransitionMetadata>,
    /// Property metadata.
    pub properties: Vec<KernelPropertyMetadata>,
    /// Solver/helper obligations covered by this kernel.
    pub obligations: Vec<KernelObligationMetadata>,
    /// Fingerprint metadata.
    pub fingerprints: Vec<KernelFingerprintMetadata>,
    /// Validation and fingerprint identity required before adoption.
    pub validation_plan: Option<KernelValidationPlanIdentity>,
    /// Structural reuse evidence separating stable digests from process-local links.
    pub structural_reuse: KernelStructuralReuseMetadata,
}

impl WholeProgramKernelMetadata {
    /// Build empty kernel metadata for `frontend`, stamped with the current
    /// schema/version and a [`missing`](KernelStructuralReuseMetadata::missing)
    /// structural-reuse row. Populate the collections via the `with_*` builders.
    #[must_use]
    pub fn new(frontend: KernelFrontend, diagnostic_name: impl Into<String>) -> Self {
        Self {
            schema: WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA,
            schema_version: WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION,
            frontend,
            diagnostic_name: diagnostic_name.into(),
            storage: Vec::new(),
            transitions: Vec::new(),
            properties: Vec::new(),
            obligations: Vec::new(),
            fingerprints: Vec::new(),
            validation_plan: None,
            structural_reuse: KernelStructuralReuseMetadata::missing(),
        }
    }

    /// Set the storage elements (sorted for stable identity).
    #[must_use]
    pub fn with_storage<I>(mut self, storage: I) -> Self
    where
        I: IntoIterator<Item = KernelStorageMetadata>,
    {
        self.storage = sorted_items(storage);
        self
    }

    /// Set the transitions (sorted for stable identity).
    #[must_use]
    pub fn with_transitions<I>(mut self, transitions: I) -> Self
    where
        I: IntoIterator<Item = KernelTransitionMetadata>,
    {
        self.transitions = sorted_items(transitions);
        self
    }

    /// Set the properties (sorted for stable identity).
    #[must_use]
    pub fn with_properties<I>(mut self, properties: I) -> Self
    where
        I: IntoIterator<Item = KernelPropertyMetadata>,
    {
        self.properties = sorted_items(properties);
        self
    }

    /// Set the solver/helper obligations (sorted for stable identity).
    #[must_use]
    pub fn with_obligations<I>(mut self, obligations: I) -> Self
    where
        I: IntoIterator<Item = KernelObligationMetadata>,
    {
        self.obligations = sorted_items(obligations);
        self
    }

    /// Set the fingerprint schemes (sorted for stable identity).
    #[must_use]
    pub fn with_fingerprints<I>(mut self, fingerprints: I) -> Self
    where
        I: IntoIterator<Item = KernelFingerprintMetadata>,
    {
        self.fingerprints = sorted_items(fingerprints);
        self
    }

    /// Attach the validation/fingerprint identity required before adoption.
    #[must_use]
    pub fn with_validation_plan(mut self, validation_plan: KernelValidationPlanIdentity) -> Self {
        self.validation_plan = Some(validation_plan);
        self
    }

    /// Attach structural-reuse evidence, replacing the default `missing` row.
    #[must_use]
    pub fn with_structural_reuse(
        mut self,
        structural_reuse: KernelStructuralReuseMetadata,
    ) -> Self {
        self.structural_reuse = structural_reuse;
        self
    }

    /// Canonical bytes for stable identity. Diagnostic names and frontend
    /// origin are intentionally omitted so equivalent kernels from different
    /// frontends can correlate when their neutral layout metadata matches.
    #[must_use]
    pub fn stable_identity_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_field(
            &mut bytes,
            "identity_basis",
            WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
        );
        write_field(&mut bytes, "schema", self.schema);
        write_field(
            &mut bytes,
            "schema_version",
            &self.schema_version.to_string(),
        );

        let mut storage = self.storage.clone();
        storage.sort();
        write_count(&mut bytes, "storage", storage.len());
        for item in &storage {
            write_field(&mut bytes, "storage.kind", item.kind.as_stable_str());
            write_field(&mut bytes, "storage.id", &item.id);
            write_field(&mut bytes, "storage.index", &item.index.to_string());
            write_field(
                &mut bytes,
                "storage.lane_count",
                &item.lane_count.to_string(),
            );
            write_field(
                &mut bytes,
                "storage.storage_width_bits",
                &item.storage_width_bits.to_string(),
            );
            write_field(&mut bytes, "storage.value_domain", &item.value_domain);
        }

        let mut transitions = self.transitions.clone();
        transitions.sort();
        write_count(&mut bytes, "transitions", transitions.len());
        for item in &transitions {
            write_field(&mut bytes, "transition.id", &item.id);
            write_field(&mut bytes, "transition.kind", &item.kind);
            write_string_list(&mut bytes, "transition.reads", &item.reads);
            write_string_list(&mut bytes, "transition.writes", &item.writes);
        }

        let mut properties = self.properties.clone();
        properties.sort();
        write_count(&mut bytes, "properties", properties.len());
        for item in &properties {
            write_field(&mut bytes, "property.id", &item.id);
            write_field(&mut bytes, "property.kind", &item.kind);
            write_string_list(&mut bytes, "property.observes", &item.observes);
        }

        let mut obligations = self.obligations.clone();
        obligations.sort();
        write_count(&mut bytes, "obligations", obligations.len());
        for item in &obligations {
            write_field(&mut bytes, "obligation.id", &item.id);
            write_field(&mut bytes, "obligation.kind", &item.kind);
            write_field(&mut bytes, "obligation.solver_family", &item.solver_family);
            write_string_list(&mut bytes, "obligation.observes", &item.observes);
        }

        let mut fingerprints = self.fingerprints.clone();
        fingerprints.sort();
        write_count(&mut bytes, "fingerprints", fingerprints.len());
        for item in &fingerprints {
            write_field(&mut bytes, "fingerprint.id", &item.id);
            write_field(&mut bytes, "fingerprint.domain", &item.domain);
            write_field(&mut bytes, "fingerprint.algorithm", &item.algorithm);
            write_field(
                &mut bytes,
                "fingerprint.digest_bits",
                &item.digest_bits.to_string(),
            );
            write_field(&mut bytes, "fingerprint.seed_identity", &item.seed_identity);
        }

        if let Some(validation_plan) = &self.validation_plan {
            write_field(&mut bytes, "validation_plan.id", &validation_plan.id);
            write_field(
                &mut bytes,
                "validation_plan.fingerprint_identity",
                &validation_plan.fingerprint_identity,
            );
            let mut storage_widths = validation_plan.storage_widths.clone();
            storage_widths.sort();
            write_count(
                &mut bytes,
                "validation_plan.storage_widths",
                storage_widths.len(),
            );
            for width in &storage_widths {
                write_field(
                    &mut bytes,
                    "validation_plan.storage_width.storage_id",
                    &width.storage_id,
                );
                write_field(
                    &mut bytes,
                    "validation_plan.storage_width.width_bits",
                    &width.width_bits.to_string(),
                );
            }
        } else {
            write_field(&mut bytes, "validation_plan.id", "missing");
            write_field(
                &mut bytes,
                "validation_plan.fingerprint_identity",
                "missing",
            );
            write_count(&mut bytes, "validation_plan.storage_widths", 0);
        }

        bytes
    }

    /// Stable 64-bit FNV-1a hex fingerprint over [`Self::stable_identity_bytes`].
    #[must_use]
    pub fn stable_identity_fingerprint_hex(&self) -> String {
        fnv1a64_hex(&self.stable_identity_bytes())
    }

    /// Stable semantic digest used in evidence rows.
    #[must_use]
    pub fn semantic_stable_digest_hex(&self) -> String {
        self.stable_identity_fingerprint_hex()
    }

    /// Validate metadata before a caller admits the kernel into a shared engine plan.
    ///
    /// # Errors
    ///
    /// Returns a [`KernelMetadataValidationError`] when:
    /// - the validation plan is absent or its identity is empty/`missing`/`none`
    ///   ([`MissingValidationPlanIdentity`](KernelMetadataValidationError::MissingValidationPlanIdentity));
    /// - the plan's fingerprint identity is empty/`missing`/`none`
    ///   ([`MissingFingerprintIdentity`](KernelMetadataValidationError::MissingFingerprintIdentity));
    /// - no fingerprint schemes are declared, or one is malformed
    ///   ([`MissingFingerprintMetadata`](KernelMetadataValidationError::MissingFingerprintMetadata),
    ///   [`InvalidFingerprintMetadata`](KernelMetadataValidationError::InvalidFingerprintMetadata));
    /// - storage ids are duplicated, have zero width, or the validation plan's
    ///   widths are duplicated, unknown, missing, or disagree with the metadata
    ///   (the remaining `*StorageWidth*` / `DuplicateStorageId` variants).
    pub fn validate(&self) -> Result<(), KernelMetadataValidationError> {
        let Some(validation_plan) = &self.validation_plan else {
            return Err(KernelMetadataValidationError::MissingValidationPlanIdentity);
        };
        if missing_identity(&validation_plan.id) {
            return Err(KernelMetadataValidationError::MissingValidationPlanIdentity);
        }
        if missing_identity(&validation_plan.fingerprint_identity) {
            return Err(KernelMetadataValidationError::MissingFingerprintIdentity);
        }
        validate_fingerprints(&self.fingerprints)?;
        validate_storage_widths(&self.storage, validation_plan)
    }

    /// Render evidence only after fail-closed metadata validation succeeds.
    ///
    /// # Errors
    ///
    /// Returns the same [`KernelMetadataValidationError`] as [`validate`](Self::validate)
    /// when the metadata is malformed; the evidence row is rendered only on success.
    pub fn render_validated_evidence_row(
        &self,
        scope: &str,
    ) -> Result<String, KernelMetadataValidationError> {
        self.validate()?;
        Ok(self.render_evidence_row(scope))
    }

    /// Render one frontend-neutral evidence row for setup traces and adoption reports.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let fingerprint_admission =
            KernelFingerprintAdmissionContract::for_frontend(&self.frontend);
        let (validation_plan_identity, fingerprint_identity) = self
            .validation_plan
            .as_ref()
            .map(|plan| (plan.id.as_str(), plan.fingerprint_identity.as_str()))
            .unwrap_or(("missing", "missing"));
        format!(
            "{} {} schema={} schema_version={} frontend={} origin_frontend={} shared_engine_component={} optimization_layer={} identity_scope={} identity_basis={} stable_fingerprint={} semantic_stable_digest={} trust_ir_stable_digest={} process_local_link_digest={} unique_body_count={} specialization_count={} structural_reuse_compatible_frontend_families={} validation_plan_identity={} fingerprint_identity={} storage_count={} transition_count={} property_count={} obligation_count={} fingerprint_count={} compatible_frontend_families={} default_compatible_frontend_families={} downstream_beneficiary_families={} remaining_compatible_frontend_families={} fingerprint_admission_surface={} fingerprint_admission_semantics={} fingerprint_admission_compatible_frontend_families={} fingerprint_admission_default_frontend_families={} fingerprint_admission_blocked_frontend_families={} first_beneficiary={} second_beneficiary={} extraction_status={} blocker_status={} shared_owner={}",
            evidence_value(scope),
            WHOLE_PROGRAM_KERNEL_EVIDENCE_ROW_KIND,
            self.schema,
            self.schema_version,
            evidence_value(self.frontend.code()),
            evidence_value(self.frontend.code()),
            WHOLE_PROGRAM_KERNEL_SHARED_ENGINE_COMPONENT,
            WHOLE_PROGRAM_KERNEL_OPTIMIZATION_LAYER,
            WHOLE_PROGRAM_KERNEL_IDENTITY_SCOPE,
            WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
            self.stable_identity_fingerprint_hex(),
            self.semantic_stable_digest_hex(),
            evidence_value(&self.structural_reuse.trust_ir_stable_digest),
            evidence_value(&self.structural_reuse.process_local_link_digest),
            self.structural_reuse.unique_body_count,
            self.structural_reuse.specialization_count,
            join_strings(&self.structural_reuse.compatible_frontend_families),
            evidence_value(validation_plan_identity),
            evidence_value(fingerprint_identity),
            self.storage.len(),
            self.transitions.len(),
            self.properties.len(),
            self.obligations.len(),
            self.fingerprints.len(),
            WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES,
            join_family_codes(default_compatible_frontend_families(&self.frontend)),
            join_family_codes(downstream_beneficiary_families(&self.frontend)),
            join_family_codes(remaining_compatible_frontend_families(&self.frontend)),
            fingerprint_admission.surface,
            fingerprint_admission.semantics,
            fingerprint_admission.compatible_frontend_families,
            fingerprint_admission.default_frontend_families,
            fingerprint_admission.blocked_frontend_families,
            evidence_value(self.frontend.first_beneficiary()),
            self.frontend.second_beneficiary(),
            WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
            WHOLE_PROGRAM_KERNEL_BLOCKER_STATUS,
            WHOLE_PROGRAM_KERNEL_SHARED_OWNER,
        )
    }
}

fn validate_fingerprints(
    fingerprints: &[KernelFingerprintMetadata],
) -> Result<(), KernelMetadataValidationError> {
    if fingerprints.is_empty() {
        return Err(KernelMetadataValidationError::MissingFingerprintMetadata);
    }

    let mut seen = BTreeSet::new();
    for fingerprint in fingerprints {
        if missing_identity(&fingerprint.id) {
            return Err(KernelMetadataValidationError::InvalidFingerprintMetadata {
                fingerprint_id: fingerprint.id.clone(),
                field: "id",
            });
        }
        if !seen.insert(fingerprint.id.clone()) {
            return Err(KernelMetadataValidationError::InvalidFingerprintMetadata {
                fingerprint_id: fingerprint.id.clone(),
                field: "duplicate_id",
            });
        }
        if fingerprint.domain.is_empty() {
            return Err(KernelMetadataValidationError::InvalidFingerprintMetadata {
                fingerprint_id: fingerprint.id.clone(),
                field: "domain",
            });
        }
        if fingerprint.algorithm.is_empty() {
            return Err(KernelMetadataValidationError::InvalidFingerprintMetadata {
                fingerprint_id: fingerprint.id.clone(),
                field: "algorithm",
            });
        }
        if fingerprint.digest_bits == 0 {
            return Err(KernelMetadataValidationError::InvalidFingerprintMetadata {
                fingerprint_id: fingerprint.id.clone(),
                field: "digest_bits",
            });
        }
        if fingerprint.seed_identity.is_empty() {
            return Err(KernelMetadataValidationError::InvalidFingerprintMetadata {
                fingerprint_id: fingerprint.id.clone(),
                field: "seed_identity",
            });
        }
    }

    Ok(())
}

fn validate_storage_widths(
    storage: &[KernelStorageMetadata],
    validation_plan: &KernelValidationPlanIdentity,
) -> Result<(), KernelMetadataValidationError> {
    let mut storage_width_by_id = BTreeMap::new();
    for item in storage {
        if storage_width_by_id
            .insert(item.id.clone(), item.storage_width_bits)
            .is_some()
        {
            return Err(KernelMetadataValidationError::DuplicateStorageId {
                storage_id: item.id.clone(),
            });
        }
        if item.storage_width_bits == 0 {
            return Err(KernelMetadataValidationError::InvalidStorageWidth {
                storage_id: item.id.clone(),
                storage_width_bits: item.storage_width_bits,
            });
        }
    }

    let mut validation_width_by_id = BTreeMap::new();
    for width in &validation_plan.storage_widths {
        if validation_width_by_id
            .insert(width.storage_id.clone(), width.width_bits)
            .is_some()
        {
            return Err(
                KernelMetadataValidationError::DuplicateValidationStorageWidth {
                    storage_id: width.storage_id.clone(),
                },
            );
        }
        if !storage_width_by_id.contains_key(&width.storage_id) {
            return Err(
                KernelMetadataValidationError::UnknownValidationStorageWidth {
                    storage_id: width.storage_id.clone(),
                },
            );
        }
    }

    for (storage_id, metadata_width_bits) in storage_width_by_id {
        let Some(validation_width_bits) = validation_width_by_id.get(&storage_id).copied() else {
            return Err(
                KernelMetadataValidationError::MissingValidationStorageWidth { storage_id },
            );
        };
        if metadata_width_bits != validation_width_bits {
            return Err(KernelMetadataValidationError::StorageWidthMismatch {
                storage_id,
                metadata_width_bits,
                validation_width_bits,
            });
        }
    }

    Ok(())
}

fn missing_identity(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("missing")
        || trimmed.eq_ignore_ascii_case("none")
}

fn fingerprint_admission_default_frontend_families(frontend: &KernelFrontend) -> Vec<&'static str> {
    default_compatible_frontend_families(frontend)
        .into_iter()
        .filter(|family| *family != "future_importer")
        .collect()
}

fn default_compatible_frontend_families(frontend: &KernelFrontend) -> Vec<&'static str> {
    let origin = frontend.code();
    let second = frontend.second_beneficiary();
    COMPATIBLE_FRONTEND_FAMILY_CODES
        .iter()
        .copied()
        .filter(|family| *family == origin || *family == second)
        .collect()
}

fn downstream_beneficiary_families(frontend: &KernelFrontend) -> Vec<&'static str> {
    let defaults = default_compatible_frontend_families(frontend);
    COMPATIBLE_FRONTEND_FAMILY_CODES
        .iter()
        .copied()
        .filter(|family| !defaults.contains(family) && *family != "future_importer")
        .collect()
}

fn remaining_compatible_frontend_families(frontend: &KernelFrontend) -> Vec<&'static str> {
    let defaults = default_compatible_frontend_families(frontend);
    COMPATIBLE_FRONTEND_FAMILY_CODES
        .iter()
        .copied()
        .filter(|family| !defaults.contains(family) && *family == "future_importer")
        .collect()
}

fn join_family_codes(families: Vec<&'static str>) -> String {
    if families.is_empty() {
        "none".to_string()
    } else {
        families.join(",")
    }
}

fn join_strings(families: &[String]) -> String {
    if families.is_empty() {
        "none".to_string()
    } else {
        families.join(",")
    }
}

fn inferred_storage_width_bits(value_domain: &str, lane_count: u32) -> u32 {
    match value_domain {
        "bool" => 1,
        "i8" | "u8" => 8,
        "i16" | "u16" => 16,
        "i32" | "u32" => 32,
        "i64" | "u64" | "token_count" => 64,
        "i128" | "u128" => 128,
        _ => value_domain
            .strip_prefix("bv")
            .and_then(|bits| bits.parse::<u32>().ok())
            .filter(|bits| *bits > 0)
            .unwrap_or(lane_count),
    }
}

fn sorted_items<T: Ord>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut items: Vec<_> = items.into_iter().collect();
    items.sort();
    items
}

fn sorted_strings<I, S>(items: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    sorted_items(items.into_iter().map(Into::into))
}

fn write_count(bytes: &mut Vec<u8>, key: &str, count: usize) {
    write_field(bytes, key, &count.to_string());
}

fn write_string_list(bytes: &mut Vec<u8>, key: &str, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    write_count(bytes, key, values.len());
    for value in values {
        write_field(bytes, key, &value);
    }
}

fn write_field(bytes: &mut Vec<u8>, key: &str, value: &str) {
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key.as_bytes());
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut hex = String::with_capacity(16);
    write!(&mut hex, "{hash:016x}").expect("format stable fingerprint hex");
    hex
}

fn evidence_value(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_fingerprint() -> KernelFingerprintMetadata {
        KernelFingerprintMetadata::new("state_fp", "state", "canonical_sha256", 256, "none")
    }

    fn validation_plan(
        id: &str,
        fingerprint_identity: &str,
        widths: &[(&str, u32)],
    ) -> KernelValidationPlanIdentity {
        KernelValidationPlanIdentity::new(id, fingerprint_identity).with_storage_widths(
            widths
                .iter()
                .map(|(storage_id, width_bits)| {
                    KernelValidationStorageWidth::new(*storage_id, *width_bits)
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn whole_program_kernel_metadata_represents_tla_storage() {
        let kernel = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "SpecA")
            .with_storage([
                KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.account_balance",
                    "balance",
                    0,
                    1,
                    "i64",
                ),
                KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.locked",
                    "locked",
                    1,
                    1,
                    "bool",
                ),
            ])
            .with_transitions([KernelTransitionMetadata::new("next", "Next", "next")
                .with_reads(["slot.account_balance", "slot.locked"])
                .with_writes(["slot.account_balance"])])
            .with_properties([KernelPropertyMetadata::new(
                "inv.nonnegative",
                "NonNegative",
                "invariant",
            )
            .with_observes(["slot.account_balance"])])
            .with_obligations([KernelObligationMetadata::new(
                "obligation.ay.inv.nonnegative",
                "AY NonNegative",
                "proof",
                "ay_helper",
            )
            .with_observes(["slot.account_balance"])])
            .with_fingerprints([state_fingerprint()])
            .with_validation_plan(validation_plan(
                "validation:tla:SpecA:v1",
                "fingerprint:state:SpecA:v1",
                &[("slot.account_balance", 64), ("slot.locked", 1)],
            ))
            .with_structural_reuse(KernelStructuralReuseMetadata::frontend_neutral(
                1,
                2,
                "trust-ir-stable:tla-next:v1",
                "link:pid-local:tla-next",
            ));

        assert_eq!(kernel.storage.len(), 2);
        assert_eq!(kernel.storage[0].storage_width_bits, 64);
        assert_eq!(kernel.storage[1].storage_width_bits, 1);
        assert_eq!(kernel.transitions[0].reads.len(), 2);
        assert_eq!(kernel.properties[0].kind, "invariant");
        assert_eq!(kernel.obligations[0].solver_family, "ay_helper");
        assert_eq!(kernel.fingerprints[0].digest_bits, 256);
        assert_eq!(kernel.stable_identity_fingerprint_hex().len(), 16);
        kernel.validate().expect("TLA kernel metadata validates");
    }

    #[test]
    fn whole_program_kernel_metadata_represents_petri_and_hardware_vectors() {
        let petri = WholeProgramKernelMetadata::new(KernelFrontend::MccPetri, "net_a")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::PetriMarking,
                "place.ready",
                "Ready",
                0,
                1,
                "token_count",
            )])
            .with_transitions(
                [KernelTransitionMetadata::new("fire.start", "Start", "fire")
                    .with_reads(["place.ready"])
                    .with_writes(["place.ready"])],
            )
            .with_fingerprints([KernelFingerprintMetadata::new(
                "marking_fp",
                "marking",
                "canonical_sha256",
                256,
                "none",
            )])
            .with_validation_plan(validation_plan(
                "validation:petri:net_a:v1",
                "fingerprint:marking:net_a:v1",
                &[("place.ready", 64)],
            ));

        let hardware = WholeProgramKernelMetadata::new(KernelFrontend::Btor2, "counter")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::HardwareRegister,
                "reg.pc",
                "pc",
                0,
                32,
                "bv32",
            )])
            .with_transitions(
                [KernelTransitionMetadata::new("posedge", "posedge", "clock")
                    .with_reads(["reg.pc"])
                    .with_writes(["reg.pc"])],
            )
            .with_properties([
                KernelPropertyMetadata::new("assert.pc_safe", "pc_safe", "assert")
                    .with_observes(["reg.pc"]),
            ])
            .with_obligations([KernelObligationMetadata::new(
                "obligation.bmc.pc_safe",
                "pc_safe bmc",
                "bmc",
                "ay_helper",
            )
            .with_observes(["reg.pc"])])
            .with_fingerprints([KernelFingerprintMetadata::new(
                "register_vector_fp",
                "register_vector",
                "canonical_sha256",
                256,
                "none",
            )])
            .with_validation_plan(validation_plan(
                "validation:hardware:counter:v1",
                "fingerprint:register_vector:counter:v1",
                &[("reg.pc", 32)],
            ));

        assert_eq!(
            petri.storage[0].kind.as_stable_str(),
            KernelStorageKind::PetriMarking.as_stable_str()
        );
        assert_eq!(
            hardware.storage[0].kind.as_stable_str(),
            KernelStorageKind::HardwareRegister.as_stable_str()
        );
        assert_ne!(
            petri.stable_identity_fingerprint_hex(),
            hardware.stable_identity_fingerprint_hex()
        );
        petri.validate().expect("Petri kernel metadata validates");
        hardware
            .validate()
            .expect("hardware kernel metadata validates");
    }

    #[test]
    fn validated_kernel_metadata_fails_closed_on_width_mismatch() {
        let kernel = WholeProgramKernelMetadata::new(KernelFrontend::Btor2, "counter")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::HardwareRegister,
                "reg.pc",
                "pc",
                0,
                32,
                "bv32",
            )])
            .with_fingerprints([KernelFingerprintMetadata::new(
                "register_vector_fp",
                "register_vector",
                "canonical_sha256",
                256,
                "none",
            )])
            .with_validation_plan(validation_plan(
                "validation:hardware:counter:v1",
                "fingerprint:register_vector:counter:v1",
                &[("reg.pc", 64)],
            ));

        assert!(matches!(
            kernel.validate(),
            Err(KernelMetadataValidationError::StorageWidthMismatch {
                storage_id,
                metadata_width_bits: 32,
                validation_width_bits: 64,
            }) if storage_id == "reg.pc"
        ));
        assert!(matches!(
            kernel.render_validated_evidence_row("TRUST_IR"),
            Err(KernelMetadataValidationError::StorageWidthMismatch { .. })
        ));
    }

    #[test]
    fn validated_kernel_metadata_fails_closed_on_missing_validation_or_fingerprint_identity() {
        let missing_validation = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "Spec")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::TlaStateSlot,
                "slot.x",
                "x",
                0,
                1,
                "i64",
            )])
            .with_fingerprints([state_fingerprint()]);

        assert_eq!(
            missing_validation.validate(),
            Err(KernelMetadataValidationError::MissingValidationPlanIdentity)
        );

        let missing_fingerprint_identity =
            WholeProgramKernelMetadata::new(KernelFrontend::Tla, "Spec")
                .with_storage([KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.x",
                    "x",
                    0,
                    1,
                    "i64",
                )])
                .with_fingerprints([state_fingerprint()])
                .with_validation_plan(validation_plan(
                    "validation:tla:Spec:v1",
                    "missing",
                    &[("slot.x", 64)],
                ));

        assert_eq!(
            missing_fingerprint_identity.validate(),
            Err(KernelMetadataValidationError::MissingFingerprintIdentity)
        );

        let missing_fingerprint_metadata =
            WholeProgramKernelMetadata::new(KernelFrontend::Tla, "Spec")
                .with_storage([KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.x",
                    "x",
                    0,
                    1,
                    "i64",
                )])
                .with_validation_plan(validation_plan(
                    "validation:tla:Spec:v1",
                    "fingerprint:state:Spec:v1",
                    &[("slot.x", 64)],
                ));

        assert_eq!(
            missing_fingerprint_metadata.validate(),
            Err(KernelMetadataValidationError::MissingFingerprintMetadata)
        );
    }

    #[test]
    fn process_local_link_digest_does_not_change_semantic_stable_digest() {
        let kernel = |link_digest| {
            WholeProgramKernelMetadata::new(KernelFrontend::Aiger, "counter")
                .with_storage([KernelStorageMetadata::new(
                    KernelStorageKind::HardwareRegister,
                    "reg.counter",
                    "counter",
                    0,
                    32,
                    "bv32",
                )])
                .with_transitions([KernelTransitionMetadata::new(
                    "transition.tick",
                    "tick",
                    "clock",
                )
                .with_reads(["reg.counter"])
                .with_writes(["reg.counter"])])
                .with_fingerprints([KernelFingerprintMetadata::new(
                    "register_vector_fp",
                    "register_vector",
                    "canonical_sha256",
                    256,
                    "none",
                )])
                .with_validation_plan(validation_plan(
                    "validation:hardware:counter:v1",
                    "fingerprint:register_vector:counter:v1",
                    &[("reg.counter", 32)],
                ))
                .with_structural_reuse(KernelStructuralReuseMetadata::new(
                    1,
                    3,
                    "trust-ir-stable:counter-kernel:v1",
                    link_digest,
                    ["aiger", "btor2", "vmt_transition_system", "ay_analytical"],
                ))
        };

        let first = kernel("link:process-a");
        let second = kernel("link:process-b");

        assert_eq!(
            first.semantic_stable_digest_hex(),
            second.semantic_stable_digest_hex()
        );
        assert_eq!(
            first.stable_identity_bytes(),
            second.stable_identity_bytes()
        );
        assert_ne!(
            first.structural_reuse.process_local_link_digest,
            second.structural_reuse.process_local_link_digest
        );

        let row = first
            .render_validated_evidence_row("TRUST_IR")
            .expect("structural reuse row validates");
        assert!(row.contains("unique_body_count=1"));
        assert!(row.contains("specialization_count=3"));
        assert!(row.contains("semantic_stable_digest="));
        assert!(row.contains("trust_ir_stable_digest=trust-ir-stable:counter-kernel:v1"));
        assert!(row.contains("process_local_link_digest=link:process-a"));
        // `KernelStructuralReuseMetadata::new` canonicalizes the family list via
        // `sorted_strings` (so the semantic_stable_digest is order-independent —
        // the equal-digest assertions above rely on this). The rendered row
        // therefore lists the families alphabetically, not in insertion order.
        // The previous expectation used the unsorted insertion order and is
        // stale; the canonical sorted order is aiger, ay_analytical, btor2,
        // vmt_transition_system.
        assert!(row.contains(
            "structural_reuse_compatible_frontend_families=aiger,ay_analytical,btor2,vmt_transition_system"
        ));
    }

    #[test]
    fn stable_identity_ignores_diagnostic_names_and_input_order() {
        let left = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "SpecA_ModelA")
            .with_storage([
                KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.b",
                    "SpecA_b",
                    1,
                    1,
                    "i64",
                ),
                KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.a",
                    "SpecA_a",
                    0,
                    1,
                    "i64",
                ),
            ])
            .with_transitions([KernelTransitionMetadata::new("next", "SpecA_Next", "next")
                .with_reads(["slot.b", "slot.a"])
                .with_writes(["slot.b"])])
            .with_properties([
                KernelPropertyMetadata::new("inv.safe", "SpecA_Safe", "invariant")
                    .with_observes(["slot.b", "slot.a"]),
            ])
            .with_fingerprints([state_fingerprint()]);

        let right = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "SpecB_ModelB")
            .with_storage([
                KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.a",
                    "SpecB_a",
                    0,
                    1,
                    "i64",
                ),
                KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.b",
                    "SpecB_b",
                    1,
                    1,
                    "i64",
                ),
            ])
            .with_transitions([KernelTransitionMetadata::new("next", "SpecB_Next", "next")
                .with_reads(["slot.a", "slot.b"])
                .with_writes(["slot.b"])])
            .with_properties([
                KernelPropertyMetadata::new("inv.safe", "SpecB_Safe", "invariant")
                    .with_observes(["slot.a", "slot.b"]),
            ])
            .with_fingerprints([state_fingerprint()]);

        assert_ne!(left.diagnostic_name, right.diagnostic_name);
        assert_eq!(left.stable_identity_bytes(), right.stable_identity_bytes());
        assert_eq!(
            left.stable_identity_fingerprint_hex(),
            right.stable_identity_fingerprint_hex()
        );
    }

    #[test]
    fn stable_identity_changes_when_semantic_metadata_changes() {
        let base = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "Spec")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::TlaStateSlot,
                "slot.a",
                "a",
                0,
                1,
                "i64",
            )])
            .with_fingerprints([state_fingerprint()]);
        let changed = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "Spec")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::TlaStateSlot,
                "slot.a",
                "a",
                0,
                2,
                "i64",
            )])
            .with_fingerprints([state_fingerprint()]);

        assert_ne!(
            base.stable_identity_fingerprint_hex(),
            changed.stable_identity_fingerprint_hex()
        );
    }

    #[test]
    fn stable_identity_is_frontend_neutral_for_equivalent_kernel_layouts() {
        let kernel = |frontend| {
            WholeProgramKernelMetadata::new(frontend, "lowered_counter")
                .with_storage([KernelStorageMetadata::new(
                    KernelStorageKind::HardwareRegister,
                    "reg.counter",
                    "counter",
                    0,
                    32,
                    "bv32",
                )])
                .with_transitions([KernelTransitionMetadata::new(
                    "transition.tick",
                    "tick",
                    "clock",
                )
                .with_reads(["reg.counter"])
                .with_writes(["reg.counter"])])
                .with_properties([
                    KernelPropertyMetadata::new("property.safe", "safe", "assert")
                        .with_observes(["reg.counter"]),
                ])
                .with_fingerprints([KernelFingerprintMetadata::new(
                    "register_vector_fp",
                    "register_vector",
                    "canonical_sha256",
                    256,
                    "none",
                )])
        };

        let aiger = kernel(KernelFrontend::Aiger);
        let btor2 = kernel(KernelFrontend::Btor2);
        let vmt = kernel(KernelFrontend::VmtReplay);

        assert_ne!(aiger.frontend, btor2.frontend);
        assert_eq!(aiger.stable_identity_bytes(), btor2.stable_identity_bytes());
        assert_eq!(
            btor2.stable_identity_fingerprint_hex(),
            vmt.stable_identity_fingerprint_hex()
        );
    }

    #[test]
    fn equivalent_kernels_from_different_diagnostic_names_share_identity() {
        let kernel = |name, slot_name, transition_name, property_name| {
            WholeProgramKernelMetadata::new(KernelFrontend::Quint, name)
                .with_storage([KernelStorageMetadata::new(
                    KernelStorageKind::TlaStateSlot,
                    "slot.counter",
                    slot_name,
                    0,
                    1,
                    "i64",
                )])
                .with_transitions([KernelTransitionMetadata::new(
                    "transition.step",
                    transition_name,
                    "next",
                )
                .with_reads(["slot.counter"])
                .with_writes(["slot.counter"])])
                .with_properties([KernelPropertyMetadata::new(
                    "property.safe",
                    property_name,
                    "invariant",
                )
                .with_observes(["slot.counter"])])
                .with_fingerprints([state_fingerprint()])
        };

        let left = kernel("CounterFromQuint", "counter", "step", "safe");
        let right = kernel(
            "RenamedModel",
            "renamedCounter",
            "renamedStep",
            "renamedSafe",
        );

        assert_ne!(left.diagnostic_name, right.diagnostic_name);
        assert_eq!(left.stable_identity_bytes(), right.stable_identity_bytes());
        assert_eq!(
            left.stable_identity_fingerprint_hex(),
            right.stable_identity_fingerprint_hex()
        );
    }

    #[test]
    fn different_frontend_families_publish_same_kernel_evidence_vocabulary() {
        let tla = WholeProgramKernelMetadata::new(KernelFrontend::Tla, "Spec")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::TlaStateSlot,
                "slot.flag",
                "flag",
                0,
                1,
                "bool",
            )])
            .with_transitions([KernelTransitionMetadata::new("next", "Next", "next")])
            .with_properties([KernelPropertyMetadata::new("inv.safe", "Safe", "invariant")])
            .with_fingerprints([state_fingerprint()]);
        let petri = WholeProgramKernelMetadata::new(KernelFrontend::MccPetri, "Net")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::PetriMarking,
                "place.flag",
                "Flag",
                0,
                1,
                "token_count",
            )])
            .with_transitions([KernelTransitionMetadata::new("fire.step", "step", "fire")])
            .with_properties([KernelPropertyMetadata::new(
                "property.deadlock",
                "deadlock",
                "deadlock",
            )])
            .with_fingerprints([KernelFingerprintMetadata::new(
                "marking_fp",
                "marking",
                "canonical_sha256",
                256,
                "none",
            )]);
        let ay = WholeProgramKernelMetadata::new(KernelFrontend::AYOnlyHelper, "AY helper")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::Other("smt_variable".to_string()),
                "smt.flag",
                "flag",
                0,
                1,
                "bool",
            )])
            .with_transitions([KernelTransitionMetadata::new(
                "relation.step",
                "step",
                "symbolic_relation",
            )]);

        for (kernel, frontend) in [
            (&tla, "frontend=tla_plus"),
            (&petri, "frontend=mcc_petri"),
            (&ay, "frontend=ay_analytical"),
        ] {
            let row = kernel.render_evidence_row("TRUST_IR");
            assert!(row.starts_with("TRUST_IR trust_ir_whole_program_kernel "));
            assert!(row.contains("schema=tla_ir.whole_program_kernel_metadata.v2"));
            assert!(row.contains("schema_version=2"));
            assert!(row.contains(frontend));
            assert!(row.contains("origin_frontend="));
            assert!(row.contains("shared_engine_component=tla_ir.whole_program_kernel_metadata"));
            assert!(row.contains("optimization_layer=below_frontend_adapters"));
            assert!(row.contains("identity_scope=frontend_neutral_kernel_layout"));
            assert!(row.contains(
                "identity_basis=tla_ir.whole_program_kernel_metadata.canonical_identity.v2"
            ));
            assert!(row.contains("stable_fingerprint="));
            assert!(row.contains("semantic_stable_digest="));
            assert!(row.contains("trust_ir_stable_digest="));
            assert!(row.contains("process_local_link_digest="));
            assert!(row.contains("unique_body_count="));
            assert!(row.contains("specialization_count="));
            assert!(row.contains("structural_reuse_compatible_frontend_families="));
            assert!(row.contains("validation_plan_identity="));
            assert!(row.contains("fingerprint_identity="));
            assert!(row.contains("storage_count=1"));
            assert!(row.contains("transition_count=1"));
            assert!(row.contains("property_count="));
            assert!(row.contains("obligation_count="));
            assert!(row.contains("fingerprint_count="));
            assert!(row.contains(
                "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer"
            ));
            assert!(row.contains("default_compatible_frontend_families="));
            assert!(row.contains("downstream_beneficiary_families="));
            assert!(row.contains("remaining_compatible_frontend_families="));
            assert!(!row.contains("vmt_replay"));
            assert!(!row.contains("ay_only_helper"));
            assert!(!row.contains("other_importer"));
            assert!(row.contains("extraction_status=already-shared"));
            assert!(row.contains("blocker_status=no-blockers"));
            assert!(row.contains("shared_owner=shared_high_performance_engine"));
        }
    }

    #[test]
    fn kernel_evidence_roles_cover_all_compatible_families_without_frontend_local_aliases() {
        let frontends = [
            KernelFrontend::Tla,
            KernelFrontend::MccPetri,
            KernelFrontend::Aiger,
            KernelFrontend::Btor2,
            KernelFrontend::AYOnlyHelper,
            KernelFrontend::WitnessReplay,
        ];

        for frontend in frontends {
            let row =
                WholeProgramKernelMetadata::new(frontend, "kernel").render_evidence_row("TRUST_IR");
            let compatible = split_family_field(&row, "compatible_frontend_families");
            let defaults = split_family_field(&row, "default_compatible_frontend_families");
            let downstream = split_family_field(&row, "downstream_beneficiary_families");
            let remaining = split_family_field(&row, "remaining_compatible_frontend_families");
            let mut covered = defaults
                .iter()
                .chain(downstream.iter())
                .chain(remaining.iter())
                .copied()
                .collect::<Vec<_>>();

            covered.sort_unstable();
            covered.dedup();

            let mut compatible_sorted = compatible.clone();
            compatible_sorted.sort_unstable();
            compatible_sorted.dedup();

            assert_eq!(covered, compatible_sorted, "{row}");
            assert!(
                defaults.iter().all(|family| compatible.contains(family)),
                "{row}"
            );
            assert!(
                downstream.iter().all(|family| compatible.contains(family)),
                "{row}"
            );
            assert!(
                remaining.iter().all(|family| compatible.contains(family)),
                "{row}"
            );
            assert!(!row.contains("vmt_replay"));
            assert!(!row.contains("ay_only_helper"));
        }
    }

    #[test]
    fn fingerprint_admission_contract_covers_concrete_frontends_and_reserves_future_importer() {
        let expected_concrete = vec![
            "tla_plus",
            "quint",
            "mcc_petri",
            "aiger",
            "btor2",
            "vmt_transition_system",
            "ay_analytical",
            "witness_replay",
        ];

        for frontend in [
            KernelFrontend::Tla,
            KernelFrontend::Quint,
            KernelFrontend::MccPetri,
            KernelFrontend::Aiger,
            KernelFrontend::Btor2,
            KernelFrontend::VmtReplay,
            KernelFrontend::AYOnlyHelper,
            KernelFrontend::WitnessReplay,
            KernelFrontend::FutureImporter,
            KernelFrontend::Other("unmodeled_importer".to_string()),
        ] {
            let row =
                WholeProgramKernelMetadata::new(frontend, "kernel").render_evidence_row("TRUST_IR");
            let compatible =
                split_family_field(&row, "fingerprint_admission_compatible_frontend_families");
            let defaults =
                split_family_field(&row, "fingerprint_admission_default_frontend_families");
            let blocked =
                split_family_field(&row, "fingerprint_admission_blocked_frontend_families");

            assert_eq!(
                evidence_field(&row, "fingerprint_admission_surface"),
                Some(WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SURFACE),
                "{row}"
            );
            assert_eq!(
                split_family_field(&row, "fingerprint_admission_semantics"),
                vec!["default_consumer", "compatible_consumer", "blocked"],
                "{row}"
            );
            assert_eq!(compatible, expected_concrete, "{row}");
            assert!(
                defaults.iter().all(|family| compatible.contains(family)),
                "{row}"
            );
            assert!(
                !defaults.contains(&"future_importer"),
                "future importer is reserved, not default-admitted: {row}"
            );
            assert_eq!(
                blocked,
                vec![WHOLE_PROGRAM_KERNEL_FINGERPRINT_FUTURE_IMPORTER_BLOCKER],
                "{row}"
            );
        }
    }

    #[test]
    fn shared_family_evidence_rejects_legacy_frontend_alias_tokens() {
        let frontends = [
            KernelFrontend::Tla,
            KernelFrontend::Quint,
            KernelFrontend::MccPetri,
            KernelFrontend::Aiger,
            KernelFrontend::Btor2,
            KernelFrontend::VmtReplay,
            KernelFrontend::AYOnlyHelper,
            KernelFrontend::WitnessReplay,
            KernelFrontend::FutureImporter,
            KernelFrontend::Other("unmodeled_importer".to_string()),
        ];
        let fields = [
            "compatible_frontend_families",
            "default_compatible_frontend_families",
            "downstream_beneficiary_families",
            "remaining_compatible_frontend_families",
            "structural_reuse_compatible_frontend_families",
            "fingerprint_admission_compatible_frontend_families",
            "fingerprint_admission_default_frontend_families",
            "fingerprint_admission_blocked_frontend_families",
        ];
        let single_family_fields = [
            "frontend",
            "origin_frontend",
            "first_beneficiary",
            "second_beneficiary",
        ];

        for frontend in frontends {
            let row =
                WholeProgramKernelMetadata::new(frontend, "kernel").render_evidence_row("TRUST_IR");
            for field in fields {
                for family in split_family_field(&row, field) {
                    assert_not_legacy_family(field, family, &row);
                }
            }
            for field in single_family_fields {
                let family = evidence_field(&row, field).unwrap_or_else(|| {
                    panic!("missing {field} in {row}");
                });
                assert_not_legacy_family(field, family, &row);
            }
        }
    }

    #[test]
    fn kernel_evidence_row_names_non_origin_beneficiary() {
        for frontend in [
            KernelFrontend::Tla,
            KernelFrontend::Quint,
            KernelFrontend::MccPetri,
            KernelFrontend::Aiger,
            KernelFrontend::Btor2,
            KernelFrontend::VmtReplay,
            KernelFrontend::AYOnlyHelper,
            KernelFrontend::WitnessReplay,
            KernelFrontend::FutureImporter,
            KernelFrontend::Other("unmodeled_importer".to_string()),
        ] {
            let kernel = WholeProgramKernelMetadata::new(frontend, "kernel");
            let row = kernel.render_evidence_row("TRUST_IR");
            let first = evidence_field(&row, "first_beneficiary").expect("first beneficiary");
            let second = evidence_field(&row, "second_beneficiary").expect("second beneficiary");
            assert_ne!(first, second, "{row}");
        }
    }

    fn evidence_field<'a>(row: &'a str, field: &str) -> Option<&'a str> {
        row.split_whitespace().find_map(|token| {
            let (key, value) = token.split_once('=')?;
            (key == field).then_some(value)
        })
    }

    fn split_family_field<'a>(row: &'a str, field: &str) -> Vec<&'a str> {
        evidence_field(row, field)
            .unwrap_or_else(|| panic!("missing {field} in {row}"))
            .split(',')
            .filter(|family| *family != "none")
            .collect()
    }

    fn assert_not_legacy_family(field: &str, family: &str, row: &str) {
        let family = family.split_once(':').map_or(family, |(family, _)| family);
        assert_ne!(family, "vmt", "{field} leaked legacy VMT alias in {row}");
        assert_ne!(family, "ay", "{field} leaked legacy AY alias in {row}");
        assert_ne!(
            family, "replay",
            "{field} leaked legacy replay alias in {row}"
        );
    }
}
