// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral fingerprint and deduplication identities.
//!
//! These descriptors separate a fingerprint namespace from the frontend that
//! produced states or artifacts. TLA, Quint-lowered TLA, Petri/MCC, AY, trust_cg,
//! replay, and future lanes can all report the same fingerprint/dedup contract
//! without embedding adapter names in the identity itself.

use crate::{
    prepared_program::{PreparedFingerprintDescriptor, PreparedFingerprintScheme},
    setup_trace::{CheckerArtifactIdentityFields, CheckerSourceKind, SetupTraceLaneKind},
    SharedEngineFrontendFamily,
};

/// Stable schema label for shared fingerprint/dedup identity evidence.
pub const SHARED_FINGERPRINT_IDENTITY_SCHEMA: &str = "ty.shared.fingerprint_identity.v1";

/// Stable schema version for shared fingerprint/dedup identity evidence.
pub const SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION: u32 = 1;

/// Stable row kind for shared native planning identity evidence.
pub const SHARED_NATIVE_PLANNING_IDENTITY_ROW_KIND: &str = "shared_native_planning_identity";

/// Stable schema label for shared native planning identity evidence.
pub const SHARED_NATIVE_PLANNING_IDENTITY_SCHEMA: &str = "ty.shared.native_planning_identity.v1";

/// Stable schema version for shared native planning identity evidence.
pub const SHARED_NATIVE_PLANNING_IDENTITY_SCHEMA_VERSION: u32 = 1;

/// Fingerprint/dedup rejection: descriptor id is empty.
pub const SHARED_FINGERPRINT_REJECTION_EMPTY_ID: &str = "empty_id";
/// Fingerprint rejection: namespace is empty.
pub const SHARED_FINGERPRINT_REJECTION_EMPTY_NAMESPACE: &str = "empty_namespace";
/// Fingerprint rejection: canonicalization version is empty.
pub const SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICALIZATION_VERSION: &str =
    "empty_canonicalization_version";
/// Fingerprint rejection: canonical domain id or version is empty.
pub const SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICAL_DOMAIN: &str = "empty_canonical_domain";
/// Fingerprint rejection: digest bit count is zero.
pub const SHARED_FINGERPRINT_REJECTION_INVALID_DIGEST_BITS: &str = "invalid_digest_bits";
/// Fingerprint rejection: digest bit count exceeds the algorithm width.
pub const SHARED_FINGERPRINT_REJECTION_DIGEST_BITS_EXCEED_ALGORITHM: &str =
    "digest_bits_exceed_algorithm";
/// Dedup rejection: collision policy is not fail-closed.
pub const SHARED_FINGERPRINT_REJECTION_NON_FAIL_CLOSED_COLLISION_POLICY: &str =
    "non_fail_closed_collision_policy";
/// Fingerprint rejection: a required frontend family is missing from the
/// reusable contract.
pub const SHARED_FINGERPRINT_REJECTION_MISSING_REUSABLE_FRONTEND_FAMILY: &str =
    "missing_reusable_frontend_family";
/// Dedup rejection: proof/witness admission requires validation-receipt
/// fingerprints.
pub const SHARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REQUIRES_VALIDATION_RECEIPT: &str =
    "proof_witness_requires_validation_receipt";
/// Fingerprint rejection: a frontend-local domain cannot be advertised as
/// reusable shared infrastructure.
pub const SHARED_FINGERPRINT_REJECTION_FRONTEND_LOCAL_DOMAIN: &str = "frontend_local_domain";
/// Fingerprint rejection: a supplied canonical alias does not match the
/// canonical fingerprint identity.
pub const SHARED_FINGERPRINT_REJECTION_MALFORMED_CANONICAL_ALIAS: &str =
    "malformed_canonical_alias";
/// Native planning rejection: frontend-family scope is empty.
pub const SHARED_NATIVE_PLANNING_REJECTION_EMPTY_FRONTEND_FAMILY_SCOPE: &str =
    "empty_frontend_family_scope";
/// Native planning rejection: frontend-family scope contains a duplicate family.
pub const SHARED_NATIVE_PLANNING_REJECTION_DUPLICATE_FRONTEND_FAMILY: &str =
    "duplicate_frontend_family";
/// Native planning rejection: source frontend family is outside the declared scope.
pub const SHARED_NATIVE_PLANNING_REJECTION_MISSING_SOURCE_FRONTEND_FAMILY: &str =
    "missing_source_frontend_family";
/// Native planning rejection: plan reuse manifest id/digest evidence is partial.
pub const SHARED_NATIVE_PLANNING_REJECTION_INCOMPLETE_PLAN_REUSE_MANIFEST: &str =
    "incomplete_plan_reuse_manifest";
/// Native planning rejection: cache reuse policy is not registered.
pub const SHARED_NATIVE_PLANNING_REJECTION_INVALID_CACHE_REUSE_POLICY: &str =
    "invalid_cache_reuse_policy";
/// Native planning rejection: frontend-reusable cache planning needs a
/// multi-family scope.
pub const SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_COMPATIBLE_FAMILIES: &str =
    "frontend_reusable_requires_compatible_families";
/// Native planning rejection: frontend-reusable cache planning needs a trust-ir/trust_cg
/// plan reuse manifest.
pub const SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_PLAN_REUSE_MANIFEST: &str =
    "frontend_reusable_requires_plan_reuse_manifest";
/// Native planning rejection: frontend-reusable cache planning needs a canonical
/// fingerprint-domain identity.
pub const SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_FINGERPRINT_DOMAIN: &str =
    "frontend_reusable_requires_fingerprint_domain";
/// Native planning rejection: frontend-reusable cache planning needs a cache
/// namespace/domain identity.
pub const SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_CACHE_IDENTITY: &str =
    "frontend_reusable_requires_cache_identity";

/// Cache/fingerprint reuse is not advertised outside the source frontend.
pub const SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY: &str = "frontend_local_only";

/// Cache/fingerprint reuse may cross compatible frontend families after validation.
pub const SHARED_NATIVE_CACHE_REUSE_FRONTEND_REUSABLE: &str = "frontend_reusable";

/// Cache/fingerprint reuse is disabled even if artifact cache digests are present.
pub const SHARED_NATIVE_CACHE_REUSE_DISABLED: &str = "disabled";

const SHARED_FINGERPRINT_REJECTION_INCOMPLETE_DOMAIN_KEY: &str =
    "incomplete_fingerprint_domain_key";
const SHARED_FINGERPRINT_REJECTION_EMPTY_LAYOUT_DIGEST: &str = "empty_layout_digest";
const SHARED_FINGERPRINT_REJECTION_EMPTY_PROJECTION: &str = "empty_projection";

const SHARED_FINGERPRINT_REUSABLE_FRONTEND_FAMILIES: &[SharedEngineFrontendFamily] = &[
    SharedEngineFrontendFamily::TlaPlus,
    SharedEngineFrontendFamily::Quint,
    SharedEngineFrontendFamily::MccPetri,
    SharedEngineFrontendFamily::Aiger,
    SharedEngineFrontendFamily::Btor2,
    SharedEngineFrontendFamily::VmtTransitionSystem,
    SharedEngineFrontendFamily::AYAnalytical,
    SharedEngineFrontendFamily::WitnessReplay,
];

/// Canonical algorithm used to compute a frontend-neutral fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedFingerprintAlgorithm {
    /// TLC-compatible 64-bit state fingerprint.
    TlaFingerprint64,
    /// xxHash3 64-bit fingerprint.
    Xxh3U64,
    /// Stable 128-bit value or artifact fingerprint.
    StableU128,
    /// SHA-256 over canonical bytes.
    CanonicalBytesSha256,
    /// Digest of a solver model or proof-side symbolic object.
    SolverModelDigest,
}

impl SharedFingerprintAlgorithm {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaFingerprint64 => "tla_fingerprint64",
            Self::Xxh3U64 => "xxh3_u64",
            Self::StableU128 => "stable_u128",
            Self::CanonicalBytesSha256 => "canonical_bytes_sha256",
            Self::SolverModelDigest => "solver_model_digest",
        }
    }

    /// Matching prepared-program fingerprint scheme.
    #[must_use]
    pub fn prepared_scheme(self) -> PreparedFingerprintScheme {
        match self {
            Self::TlaFingerprint64 => PreparedFingerprintScheme::TlaFingerprint64,
            Self::Xxh3U64 => PreparedFingerprintScheme::Xxh3U64,
            Self::StableU128 => PreparedFingerprintScheme::StableU128,
            Self::CanonicalBytesSha256 => PreparedFingerprintScheme::CanonicalBytesSha256,
            Self::SolverModelDigest => PreparedFingerprintScheme::SolverModelDigest,
        }
    }

    /// Maximum significant digest width for this algorithm.
    #[must_use]
    pub fn max_digest_bits(self) -> u16 {
        match self {
            Self::TlaFingerprint64 | Self::Xxh3U64 => 64,
            Self::StableU128 => 128,
            Self::CanonicalBytesSha256 | Self::SolverModelDigest => 256,
        }
    }
}

/// Semantic value class covered by a fingerprint namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedFingerprintValueKind {
    /// Concrete explicit-state value.
    State,
    /// Canonical explicit-state vector.
    StateVector,
    /// Petri/MCC place-token marking vector.
    MarkingVector,
    /// Hardware, trust-ir, or solver register vector.
    RegisterVector,
    /// Ordered transition edge or state-pair value.
    StatePair,
    /// Whole transition/action relation artifact.
    TransitionArtifact,
    /// Whole batch/native artifact identity.
    BatchArtifact,
    /// Analytical or symbolic solver model/proof object.
    SolverObject,
    /// Replay or witness step value.
    WitnessStep,
    /// Replay/proof/certificate validation receipt.
    ValidationReceipt,
}

impl SharedFingerprintValueKind {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::StateVector => "state_vector",
            Self::MarkingVector => "marking_vector",
            Self::RegisterVector => "register_vector",
            Self::StatePair => "state_pair",
            Self::TransitionArtifact => "transition_artifact",
            Self::BatchArtifact => "batch_artifact",
            Self::SolverObject => "solver_object",
            Self::WitnessStep => "witness_step",
            Self::ValidationReceipt => "validation_receipt",
        }
    }
}

/// Evidence a caller has for suppressing a duplicate fingerprint.
///
/// This deliberately separates "the set says duplicate" from why that
/// duplicate is safe to suppress. State vectors, Petri marking vectors, and
/// register vectors normally use canonical payload equality. Replay/proof
/// lanes can instead use an accepted proof, witness, or certificate receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedDuplicateAuthorization {
    /// No reusable evidence proves that the duplicate is safe.
    Unconfirmed,
    /// Resident and candidate canonical payloads compare equal.
    CanonicalPayloadEquality,
    /// An accepted proof/witness/certificate validates the duplicate.
    ProofWitness,
}

impl SharedDuplicateAuthorization {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Unconfirmed => "unconfirmed",
            Self::CanonicalPayloadEquality => "canonical_payload_equality",
            Self::ProofWitness => "proof_witness",
        }
    }

    /// Convert a canonical-payload equality check into an authorization.
    #[must_use]
    pub fn canonical_payload_equality(confirmed: bool) -> Self {
        if confirmed {
            Self::CanonicalPayloadEquality
        } else {
            Self::Unconfirmed
        }
    }

    /// Convert a proof/witness/certificate validation result into an authorization.
    #[must_use]
    pub fn proof_witness(accepted: bool) -> Self {
        if accepted {
            Self::ProofWitness
        } else {
            Self::Unconfirmed
        }
    }
}

/// Canonical domain whose bytes are fingerprinted before lane-specific storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedFingerprintCanonicalDomain {
    /// Domain id, for example `flat-state` or `solver-object`.
    pub id: String,
    /// Domain/layout version.
    pub version: String,
}

impl SharedFingerprintCanonicalDomain {
    /// Create a canonical domain descriptor.
    #[must_use]
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
        }
    }

    /// Stable identity for domain separation.
    #[must_use]
    pub fn identity(&self) -> String {
        identity_join(["canonical_domain", &self.id, &self.version])
    }
}

/// Scope where fingerprint equality means "duplicate".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedDedupScope {
    /// State-space visited set.
    StateSpace,
    /// One frontier or level batch.
    FrontierBatch,
    /// Candidate native/symbolic batch artifact set.
    BatchArtifact,
    /// Candidate-lane admission set.
    CandidateLane,
    /// Solver/proof query cache.
    ProofQuery,
    /// Replay or witness trace cache.
    ReplayTrace,
}

impl SharedDedupScope {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::StateSpace => "state_space",
            Self::FrontierBatch => "frontier_batch",
            Self::BatchArtifact => "batch_artifact",
            Self::CandidateLane => "candidate_lane",
            Self::ProofQuery => "proof_query",
            Self::ReplayTrace => "replay_trace",
        }
    }
}

/// Storage policy used by a dedup lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedDedupStorageKind {
    /// Single in-memory set.
    InMemory,
    /// Sharded in-memory set.
    ShardedInMemory,
    /// Compare-and-swap/open-addressed fingerprint set.
    Cas,
    /// External or persisted dedup store.
    External,
    /// No storage is attached; identity is evidence-only.
    EvidenceOnly,
}

/// Typed cache/fingerprint reuse policy for shared native planning and native
/// contract identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedNativeCacheReusePolicy {
    /// Reuse evidence is valid only for the producing frontend family.
    FrontendLocalOnly,
    /// Reuse evidence may be consumed by compatible frontend families.
    FrontendReusable,
    /// Reuse is disabled even if cache identities are present.
    Disabled,
}

impl SharedNativeCacheReusePolicy {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::FrontendLocalOnly => SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY,
            Self::FrontendReusable => SHARED_NATIVE_CACHE_REUSE_FRONTEND_REUSABLE,
            Self::Disabled => SHARED_NATIVE_CACHE_REUSE_DISABLED,
        }
    }

    /// Parse a stable evidence/identity code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY => Some(Self::FrontendLocalOnly),
            SHARED_NATIVE_CACHE_REUSE_FRONTEND_REUSABLE => Some(Self::FrontendReusable),
            SHARED_NATIVE_CACHE_REUSE_DISABLED => Some(Self::Disabled),
            _ => None,
        }
    }

    /// Whether this policy advertises cross-frontend reuse.
    #[must_use]
    pub fn frontend_reusable(self) -> bool {
        matches!(self, Self::FrontendReusable)
    }
}

impl SharedDedupStorageKind {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::InMemory => "in_memory",
            Self::ShardedInMemory => "sharded_in_memory",
            Self::Cas => "cas",
            Self::External => "external",
            Self::EvidenceOnly => "evidence_only",
        }
    }
}

/// Collision policy for interpreting equal fingerprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedCollisionPolicy {
    /// Equality of fingerprints alone is trusted. Not fail-closed.
    Unchecked,
    /// Equal fingerprints must be confirmed against canonical payload bytes.
    CanonicalPayloadEquality,
    /// Equal fingerprints are rejected unless an external proof/witness authorizes them.
    ProofWitnessRequired,
    /// Any detected collision rejects the candidate/state.
    RejectOnCollision,
}

impl SharedCollisionPolicy {
    /// Stable evidence/identity code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Unchecked => "unchecked",
            Self::CanonicalPayloadEquality => "canonical_payload_equality",
            Self::ProofWitnessRequired => "proof_witness_required",
            Self::RejectOnCollision => "reject_on_collision",
        }
    }

    /// Whether this policy prevents silently accepting an ambiguous duplicate.
    #[must_use]
    pub fn is_fail_closed(self) -> bool {
        !matches!(self, Self::Unchecked)
    }

    /// Whether duplicate evidence satisfies this collision policy.
    ///
    /// `RejectOnCollision` and `CanonicalPayloadEquality` both require the
    /// caller to prove the resident and candidate canonical payloads are equal.
    /// `ProofWitnessRequired` is intentionally stricter: a payload comparison
    /// alone does not satisfy a policy that asked for proof/witness evidence.
    #[must_use]
    pub fn authorizes_duplicate(self, authorization: SharedDuplicateAuthorization) -> bool {
        match self {
            Self::Unchecked => false,
            Self::CanonicalPayloadEquality | Self::RejectOnCollision => {
                matches!(
                    authorization,
                    SharedDuplicateAuthorization::CanonicalPayloadEquality
                )
            }
            Self::ProofWitnessRequired => {
                matches!(authorization, SharedDuplicateAuthorization::ProofWitness)
            }
        }
    }

    /// Whether this collision policy requires a validation receipt rather than
    /// a direct canonical-payload equality check.
    #[must_use]
    pub fn requires_validation_receipt(self) -> bool {
        matches!(self, Self::ProofWitnessRequired)
    }
}

/// Canonical payload described by a shared fingerprint domain key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FingerprintCanonicalPayload {
    /// Semantic value class represented by the canonical payload bytes.
    pub value_kind: SharedFingerprintValueKind,
    /// Frontend-neutral domain/version for the canonical payload bytes.
    pub canonical_domain: SharedFingerprintCanonicalDomain,
    /// Canonicalization version applied before hashing.
    pub canonicalization_version: String,
    /// Number of digest bits produced or retained for the canonical payload.
    pub digest_bits: u16,
}

impl FingerprintCanonicalPayload {
    /// Create a canonical payload descriptor.
    #[must_use]
    pub fn new(
        value_kind: SharedFingerprintValueKind,
        canonical_domain: SharedFingerprintCanonicalDomain,
        canonicalization_version: impl Into<String>,
        digest_bits: u16,
    ) -> Self {
        Self {
            value_kind,
            canonical_domain,
            canonicalization_version: canonicalization_version.into(),
            digest_bits,
        }
    }

    /// Stable identity for the canonical payload bytes.
    #[must_use]
    pub fn identity(&self) -> String {
        let domain_identity = self.canonical_domain.identity();
        let digest_bits = self.digest_bits.to_string();
        identity_join([
            "canonical_payload",
            self.value_kind.code(),
            &domain_identity,
            &self.canonicalization_version,
            &digest_bits,
        ])
    }

    fn validate(
        &self,
        algorithm: SharedFingerprintAlgorithm,
    ) -> Result<(), SharedFingerprintIdentityRejection> {
        if self.canonical_domain.id.trim().is_empty()
            || self.canonical_domain.version.trim().is_empty()
        {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICAL_DOMAIN,
                "canonical payload domain id and version must not be empty",
            ));
        }
        if self.canonicalization_version.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICALIZATION_VERSION,
                "canonical payload canonicalization version must not be empty",
            ));
        }
        if self.digest_bits == 0 {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_INVALID_DIGEST_BITS,
                "canonical payload digest_bits must be greater than zero",
            ));
        }
        if self.digest_bits > algorithm.max_digest_bits() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_DIGEST_BITS_EXCEED_ALGORITHM,
                format!(
                    "canonical payload digest_bits {} exceeds {} width {}",
                    self.digest_bits,
                    algorithm.code(),
                    algorithm.max_digest_bits()
                ),
            ));
        }
        Ok(())
    }
}

/// Projection applied to a fingerprint before admission/storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FingerprintDomainProjection {
    /// Use the payload digest width declared by the canonical payload.
    Full,
    /// Keep the low `bits` of the canonical digest.
    LowBits(u16),
    /// XOR-fold the canonical digest to `bits`.
    XorFoldBits(u16),
    /// Frontend-neutral custom projection identity.
    Custom(String),
}

impl FingerprintDomainProjection {
    /// Stable identity for the projection.
    #[must_use]
    pub fn identity(&self) -> String {
        match self {
            Self::Full => identity_join(["projection", "full"]),
            Self::LowBits(bits) => {
                let bits = bits.to_string();
                identity_join(["projection", "low_bits", &bits])
            }
            Self::XorFoldBits(bits) => {
                let bits = bits.to_string();
                identity_join(["projection", "xor_fold_bits", &bits])
            }
            Self::Custom(identity) => identity_join(["projection", "custom", identity]),
        }
    }

    fn validate(&self, payload_bits: u16) -> Result<(), SharedFingerprintIdentityRejection> {
        match self {
            Self::Full => Ok(()),
            Self::LowBits(bits) | Self::XorFoldBits(bits) => {
                if *bits == 0 {
                    return Err(SharedFingerprintIdentityRejection::new(
                        SHARED_FINGERPRINT_REJECTION_EMPTY_PROJECTION,
                        "projection bit width must be greater than zero",
                    ));
                }
                if *bits > payload_bits {
                    return Err(SharedFingerprintIdentityRejection::new(
                        SHARED_FINGERPRINT_REJECTION_DIGEST_BITS_EXCEED_ALGORITHM,
                        format!(
                            "projection bit width {} exceeds canonical payload width {}",
                            bits, payload_bits
                        ),
                    ));
                }
                Ok(())
            }
            Self::Custom(identity) if identity.trim().is_empty() => {
                Err(SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_EMPTY_PROJECTION,
                    "custom projection identity must not be empty",
                ))
            }
            Self::Custom(_) => Ok(()),
        }
    }
}

/// Storage policy component of a [`FingerprintDomainKey`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FingerprintDomainStoragePolicy {
    /// Scope where fingerprint equality implies duplication.
    pub scope: SharedDedupScope,
    /// Storage mechanism used for the fingerprint domain.
    pub storage: SharedDedupStorageKind,
    /// Optional storage configuration/version identity.
    pub storage_config_identity: Option<String>,
}

impl FingerprintDomainStoragePolicy {
    /// Create a storage-policy descriptor.
    #[must_use]
    pub fn new(scope: SharedDedupScope, storage: SharedDedupStorageKind) -> Self {
        Self {
            scope,
            storage,
            storage_config_identity: None,
        }
    }

    /// Attach a storage configuration/version identity.
    #[must_use]
    pub fn with_storage_config_identity(
        mut self,
        storage_config_identity: impl Into<String>,
    ) -> Self {
        self.storage_config_identity = non_empty_string(storage_config_identity.into());
        self
    }

    /// Stable identity for the storage policy.
    #[must_use]
    pub fn identity(&self) -> String {
        identity_join([
            "storage_policy",
            self.storage.code(),
            self.scope.code(),
            self.storage_config_identity
                .as_deref()
                .unwrap_or("default_config"),
        ])
    }
}

/// Frontend-neutral fingerprint domain key for shared engine admission.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FingerprintDomainKey {
    /// Fingerprint algorithm.
    pub algorithm: SharedFingerprintAlgorithm,
    /// Optional runtime/helper symbol that computes this fingerprint domain.
    pub helper_symbol: Option<String>,
    /// Optional seed/domain-separation identity.
    pub seed_identity: Option<String>,
    /// Canonical payload bytes being fingerprinted.
    pub canonical_payload: FingerprintCanonicalPayload,
    /// Stable digest of the state/marking/register layout.
    pub layout_digest: String,
    /// Projection applied before admission/storage.
    pub projection: FingerprintDomainProjection,
    /// Storage policy for the admitted fingerprints.
    pub storage_policy: FingerprintDomainStoragePolicy,
    /// Collision policy for equal fingerprints.
    pub collision_policy: SharedCollisionPolicy,
}

impl FingerprintDomainKey {
    /// Start building a frontend-neutral fingerprint domain key.
    #[must_use]
    pub fn builder(algorithm: SharedFingerprintAlgorithm) -> FingerprintDomainKeyBuilder {
        FingerprintDomainKeyBuilder::new(algorithm)
    }

    /// Validate the data-only domain key.
    pub fn validate(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        if self
            .helper_symbol
            .as_deref()
            .is_some_and(|symbol| symbol.trim().is_empty())
        {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
                "helper symbol identity must not be empty when present",
            ));
        }
        if self
            .seed_identity
            .as_deref()
            .is_some_and(|seed| seed.trim().is_empty())
        {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
                "seed identity must not be empty when present",
            ));
        }
        self.canonical_payload.validate(self.algorithm)?;
        if self.layout_digest.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_LAYOUT_DIGEST,
                "layout digest must not be empty",
            ));
        }
        self.projection
            .validate(self.canonical_payload.digest_bits)?;
        Ok(())
    }

    /// Require a fail-closed collision policy for an accepted admission domain.
    pub fn require_fail_closed(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        self.validate()?;
        if !self.collision_policy.is_fail_closed() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_NON_FAIL_CLOSED_COLLISION_POLICY,
                "fingerprint domain collision policy must fail closed",
            ));
        }
        Ok(())
    }

    /// Stable data identity for the domain key.
    #[must_use]
    pub fn stable_identity(&self) -> String {
        let payload_identity = self.canonical_payload.identity();
        let projection_identity = self.projection.identity();
        let storage_identity = self.storage_policy.identity();
        let fail_closed = if self.collision_policy.is_fail_closed() {
            "fail_closed"
        } else {
            "not_fail_closed"
        };
        identity_join([
            "fingerprint_domain_key",
            self.algorithm.code(),
            self.helper_symbol.as_deref().unwrap_or("helperless"),
            self.seed_identity.as_deref().unwrap_or("seedless"),
            &payload_identity,
            &self.layout_digest,
            &projection_identity,
            &storage_identity,
            self.collision_policy.code(),
            fail_closed,
        ])
    }

    /// Stable identity for an accepted fail-closed admission policy.
    pub fn accepted_fail_closed_policy_identity(
        &self,
    ) -> Result<String, SharedFingerprintIdentityRejection> {
        self.require_fail_closed()?;
        let domain_identity = self.stable_identity();
        Ok(identity_join([
            "accepted_fail_closed_fingerprint_domain",
            &domain_identity,
            self.collision_policy.code(),
        ]))
    }
}

/// Builder for [`FingerprintDomainKey`].
#[derive(Debug, Clone)]
pub struct FingerprintDomainKeyBuilder {
    algorithm: SharedFingerprintAlgorithm,
    helper_symbol: Option<String>,
    seed_identity: Option<String>,
    canonical_payload: Option<FingerprintCanonicalPayload>,
    layout_digest: Option<String>,
    projection: Option<FingerprintDomainProjection>,
    storage_policy: Option<FingerprintDomainStoragePolicy>,
    collision_policy: Option<SharedCollisionPolicy>,
}

impl FingerprintDomainKeyBuilder {
    /// Create a builder for the requested algorithm.
    #[must_use]
    pub fn new(algorithm: SharedFingerprintAlgorithm) -> Self {
        Self {
            algorithm,
            helper_symbol: None,
            seed_identity: None,
            canonical_payload: None,
            layout_digest: None,
            projection: None,
            storage_policy: None,
            collision_policy: None,
        }
    }

    /// Set the runtime/helper symbol identity.
    #[must_use]
    pub fn helper_symbol(mut self, helper_symbol: impl Into<String>) -> Self {
        self.helper_symbol = non_empty_string(helper_symbol.into());
        self
    }

    /// Set the seed/domain-separation identity.
    #[must_use]
    pub fn seed_identity(mut self, seed_identity: impl Into<String>) -> Self {
        self.seed_identity = non_empty_string(seed_identity.into());
        self
    }

    /// Set the canonical payload descriptor.
    #[must_use]
    pub fn canonical_payload(mut self, canonical_payload: FingerprintCanonicalPayload) -> Self {
        self.canonical_payload = Some(canonical_payload);
        self
    }

    /// Set the state/marking/register layout digest.
    #[must_use]
    pub fn layout_digest(mut self, layout_digest: impl Into<String>) -> Self {
        self.layout_digest = non_empty_string(layout_digest.into());
        self
    }

    /// Set the projection applied before admission/storage.
    #[must_use]
    pub fn projection(mut self, projection: FingerprintDomainProjection) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Set the storage policy.
    #[must_use]
    pub fn storage_policy(mut self, storage_policy: FingerprintDomainStoragePolicy) -> Self {
        self.storage_policy = Some(storage_policy);
        self
    }

    /// Set the collision policy.
    #[must_use]
    pub fn collision_policy(mut self, collision_policy: SharedCollisionPolicy) -> Self {
        self.collision_policy = Some(collision_policy);
        self
    }

    /// Build and validate the data-only domain key.
    pub fn build(self) -> Result<FingerprintDomainKey, SharedFingerprintIdentityRejection> {
        let key = FingerprintDomainKey {
            algorithm: self.algorithm,
            helper_symbol: self.helper_symbol,
            seed_identity: self.seed_identity,
            canonical_payload: self.canonical_payload.ok_or_else(|| {
                SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_INCOMPLETE_DOMAIN_KEY,
                    "fingerprint domain key requires canonical payload",
                )
            })?,
            layout_digest: self.layout_digest.ok_or_else(|| {
                SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_INCOMPLETE_DOMAIN_KEY,
                    "fingerprint domain key requires layout digest",
                )
            })?,
            projection: self.projection.ok_or_else(|| {
                SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_INCOMPLETE_DOMAIN_KEY,
                    "fingerprint domain key requires projection",
                )
            })?,
            storage_policy: self.storage_policy.ok_or_else(|| {
                SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_INCOMPLETE_DOMAIN_KEY,
                    "fingerprint domain key requires storage policy",
                )
            })?,
            collision_policy: self.collision_policy.ok_or_else(|| {
                SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_INCOMPLETE_DOMAIN_KEY,
                    "fingerprint domain key requires collision policy",
                )
            })?,
        };
        key.validate()?;
        Ok(key)
    }
}

/// Structured rejection for shared fingerprint/dedup identity admission.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason_code}: {detail}")]
pub struct SharedFingerprintIdentityRejection {
    /// Stable reason code for evidence and fail-closed routing.
    pub reason_code: &'static str,
    /// Human-readable detail.
    pub detail: String,
}

impl SharedFingerprintIdentityRejection {
    /// Create a structured rejection.
    #[must_use]
    pub fn new(reason_code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            reason_code,
            detail: detail.into(),
        }
    }
}

/// Frontend-family scope and runtime-domain identities used to decide whether
/// native planning artifacts may be reused across frontend producers.
///
/// The fields are deliberately small strings. Producers can attach trust-ir batch
/// plan manifests, fingerprint-domain keys, CAS domains, and cache domains
/// without depending on a producer-specific manifest type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedNativePlanningIdentity {
    /// Stable source/payload fingerprint that the plan was derived from.
    pub source_fingerprint: Option<String>,
    /// Stable id of a reusable native planning manifest.
    pub plan_reuse_manifest_id: Option<String>,
    /// Stable digest/checksum for the reusable native planning manifest.
    pub plan_reuse_manifest_digest: Option<String>,
    /// Fingerprint-domain identity required for reuse.
    pub fingerprint_domain_identity: Option<String>,
    /// CAS/dedup domain identity required for reuse.
    pub cas_identity: Option<String>,
    /// Native cache domain identity required for reuse.
    pub cache_identity: Option<String>,
    /// Cache/fingerprint reuse policy associated with this planning identity.
    pub cache_reuse_policy: String,
    /// Frontend families allowed to consume this planning identity.
    pub frontend_family_scope: Vec<SharedEngineFrontendFamily>,
}

impl SharedNativePlanningIdentity {
    /// Create a planning identity with an explicit frontend-family scope.
    #[must_use]
    pub fn new(
        frontend_family_scope: impl IntoIterator<Item = SharedEngineFrontendFamily>,
    ) -> Self {
        Self {
            source_fingerprint: None,
            plan_reuse_manifest_id: None,
            plan_reuse_manifest_digest: None,
            fingerprint_domain_identity: None,
            cas_identity: None,
            cache_identity: None,
            cache_reuse_policy: SharedNativeCacheReusePolicy::FrontendLocalOnly
                .code()
                .to_string(),
            frontend_family_scope: frontend_family_scope.into_iter().collect(),
        }
    }

    /// Attach the stable source/payload fingerprint.
    #[must_use]
    pub fn with_source_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.source_fingerprint = non_empty_string(fingerprint.into());
        self
    }

    /// Attach the stable plan reuse manifest id and digest.
    #[must_use]
    pub fn with_plan_reuse_manifest(
        mut self,
        manifest_id: impl Into<String>,
        manifest_digest: impl Into<String>,
    ) -> Self {
        self.plan_reuse_manifest_id = non_empty_string(manifest_id.into());
        self.plan_reuse_manifest_digest = non_empty_string(manifest_digest.into());
        self
    }

    /// Attach the frontend-neutral fingerprint-domain identity.
    #[must_use]
    pub fn with_fingerprint_domain_identity(mut self, identity: impl Into<String>) -> Self {
        self.fingerprint_domain_identity = non_empty_string(identity.into());
        self
    }

    /// Attach the CAS/dedup compatibility identity.
    #[must_use]
    pub fn with_cas_identity(mut self, identity: impl Into<String>) -> Self {
        self.cas_identity = non_empty_string(identity.into());
        self
    }

    /// Attach the native cache compatibility identity.
    #[must_use]
    pub fn with_cache_identity(mut self, identity: impl Into<String>) -> Self {
        self.cache_identity = non_empty_string(identity.into());
        self
    }

    /// Attach the cache/fingerprint reuse policy.
    #[must_use]
    pub fn with_cache_reuse_policy(mut self, policy: impl Into<String>) -> Self {
        self.cache_reuse_policy = policy.into();
        self
    }

    /// Attach a typed cache/fingerprint reuse policy.
    #[must_use]
    pub fn with_cache_reuse_policy_kind(mut self, policy: SharedNativeCacheReusePolicy) -> Self {
        self.cache_reuse_policy = policy.code().to_string();
        self
    }

    /// Parse the typed cache/fingerprint reuse policy, if registered.
    #[must_use]
    pub fn cache_reuse_policy_kind(&self) -> Option<SharedNativeCacheReusePolicy> {
        SharedNativeCacheReusePolicy::from_code(&self.cache_reuse_policy)
    }

    /// Stable identity for the frontend-family reuse scope.
    #[must_use]
    pub fn frontend_family_scope_identity(&self) -> String {
        identity_join([
            "native_frontend_family_scope",
            &evidence_frontend_families(&self.frontend_family_scope),
        ])
    }

    /// Stable identity for native planning reuse compatibility.
    #[must_use]
    pub fn stable_identity(&self) -> String {
        let frontend_scope = self.frontend_family_scope_identity();
        identity_join([
            "native_planning_identity",
            self.source_fingerprint
                .as_deref()
                .unwrap_or("source_unknown"),
            self.plan_reuse_manifest_id
                .as_deref()
                .unwrap_or("manifest_id_unknown"),
            self.plan_reuse_manifest_digest
                .as_deref()
                .unwrap_or("manifest_digest_unknown"),
            self.fingerprint_domain_identity
                .as_deref()
                .unwrap_or("fingerprint_domain_unknown"),
            self.cas_identity.as_deref().unwrap_or("cas_unknown"),
            self.cache_identity.as_deref().unwrap_or("cache_unknown"),
            &self.cache_reuse_policy,
            &frontend_scope,
        ])
    }

    /// Whether this identity declares a multi-frontend reuse scope.
    #[must_use]
    pub fn frontend_family_reusable(&self) -> bool {
        unique_frontend_family_count(&self.frontend_family_scope) > 1
    }

    /// Validate the data-only planning identity against the producing source.
    pub fn validate(
        &self,
        source_kind: CheckerSourceKind,
    ) -> Result<(), SharedFingerprintIdentityRejection> {
        if self.frontend_family_scope.is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_NATIVE_PLANNING_REJECTION_EMPTY_FRONTEND_FAMILY_SCOPE,
                "native planning identity requires at least one frontend family",
            ));
        }
        if let Some(duplicate) = duplicate_frontend_family(&self.frontend_family_scope) {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_NATIVE_PLANNING_REJECTION_DUPLICATE_FRONTEND_FAMILY,
                format!(
                    "native planning frontend scope contains duplicate family {}",
                    duplicate.code()
                ),
            ));
        }

        if let Some(source_family) = source_kind.adoption_frontend_family() {
            if !self.frontend_family_scope.contains(&source_family) {
                return Err(SharedFingerprintIdentityRejection::new(
                    SHARED_NATIVE_PLANNING_REJECTION_MISSING_SOURCE_FRONTEND_FAMILY,
                    format!(
                        "native planning frontend scope does not include source family {}",
                        source_family.code()
                    ),
                ));
            }
        }

        validate_optional_identity("source_fingerprint", self.source_fingerprint.as_deref())?;
        validate_optional_identity(
            "plan_reuse_manifest_id",
            self.plan_reuse_manifest_id.as_deref(),
        )?;
        validate_optional_identity(
            "plan_reuse_manifest_digest",
            self.plan_reuse_manifest_digest.as_deref(),
        )?;
        validate_optional_identity(
            "fingerprint_domain_identity",
            self.fingerprint_domain_identity.as_deref(),
        )?;
        validate_optional_identity("cas_identity", self.cas_identity.as_deref())?;
        validate_optional_identity("cache_identity", self.cache_identity.as_deref())?;

        if self.plan_reuse_manifest_id.is_some() != self.plan_reuse_manifest_digest.is_some() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_NATIVE_PLANNING_REJECTION_INCOMPLETE_PLAN_REUSE_MANIFEST,
                "native planning reuse manifest requires both id and digest",
            ));
        }

        if self.cache_reuse_policy.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
                "native planning cache reuse policy must not be empty",
            ));
        }
        let Some(cache_reuse_policy) = self.cache_reuse_policy_kind() else {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_NATIVE_PLANNING_REJECTION_INVALID_CACHE_REUSE_POLICY,
                format!(
                    "native planning cache reuse policy {} is not registered",
                    self.cache_reuse_policy
                ),
            ));
        };

        if cache_reuse_policy.frontend_reusable() {
            if unique_frontend_family_count(&self.frontend_family_scope) < 2 {
                return Err(SharedFingerprintIdentityRejection::new(
                    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_COMPATIBLE_FAMILIES,
                    "frontend-reusable native planning identity requires at least two compatible frontend families",
                ));
            }
            if !has_optional_identity(self.plan_reuse_manifest_id.as_deref())
                || !has_optional_identity(self.plan_reuse_manifest_digest.as_deref())
            {
                return Err(SharedFingerprintIdentityRejection::new(
                    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_PLAN_REUSE_MANIFEST,
                    "frontend-reusable native planning identity requires a plan reuse manifest id and digest",
                ));
            }
            if !has_optional_identity(self.fingerprint_domain_identity.as_deref()) {
                return Err(SharedFingerprintIdentityRejection::new(
                    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_FINGERPRINT_DOMAIN,
                    "frontend-reusable native planning identity requires a fingerprint-domain identity",
                ));
            }
            if !has_optional_identity(self.cache_identity.as_deref()) {
                return Err(SharedFingerprintIdentityRejection::new(
                    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_CACHE_IDENTITY,
                    "frontend-reusable native planning identity requires a cache identity",
                ));
            }
        }

        Ok(())
    }

    /// Render one frontend-neutral native planning evidence row.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str, source_kind: CheckerSourceKind) -> String {
        format!(
            "{} {} schema={} schema_version={} source_kind={} frontend_kind={} native_planning_identity={} source_fingerprint={} plan_reuse_manifest_id={} plan_reuse_manifest_digest={} fingerprint_domain_identity={} cas_identity={} cache_identity={} cache_reuse_policy={} frontend_family_scope={} frontend_family_scope_identity={} frontend_family_reusable={}",
            scope,
            SHARED_NATIVE_PLANNING_IDENTITY_ROW_KIND,
            SHARED_NATIVE_PLANNING_IDENTITY_SCHEMA,
            SHARED_NATIVE_PLANNING_IDENTITY_SCHEMA_VERSION,
            source_kind.code(),
            source_kind.frontend_family_code(),
            evidence_value(&self.stable_identity()),
            evidence_optional(self.source_fingerprint.as_deref()),
            evidence_optional(self.plan_reuse_manifest_id.as_deref()),
            evidence_optional(self.plan_reuse_manifest_digest.as_deref()),
            evidence_optional(self.fingerprint_domain_identity.as_deref()),
            evidence_optional(self.cas_identity.as_deref()),
            evidence_optional(self.cache_identity.as_deref()),
            evidence_value(&self.cache_reuse_policy),
            evidence_frontend_families(&self.frontend_family_scope),
            evidence_value(&self.frontend_family_scope_identity()),
            self.frontend_family_reusable(),
        )
    }
}

/// Frontend-independent fingerprint namespace and canonicalization contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedFingerprintIdentity {
    /// Human-readable descriptor id.
    pub id: String,
    /// Fingerprint algorithm.
    pub algorithm: SharedFingerprintAlgorithm,
    /// Semantic value class covered by this fingerprint.
    pub value_kind: SharedFingerprintValueKind,
    /// Canonicalization/layout version before hashing.
    pub canonicalization_version: String,
    /// Frontend-neutral namespace, for example `flat-state-v1`.
    pub namespace: String,
    /// Canonical domain/version separated from frontend payload identity.
    pub canonical_domain: SharedFingerprintCanonicalDomain,
    /// Number of digest bits considered significant.
    pub digest_bits: u16,
    /// Optional seed/domain-separation id.
    pub seed_identity: Option<String>,
}

impl SharedFingerprintIdentity {
    /// Create a fingerprint identity. Empty optional strings are normalized
    /// away by builder methods so rendered evidence remains stable.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        algorithm: SharedFingerprintAlgorithm,
        value_kind: SharedFingerprintValueKind,
        canonicalization_version: impl Into<String>,
        namespace: impl Into<String>,
        digest_bits: u16,
    ) -> Self {
        let canonicalization_version = canonicalization_version.into();
        let namespace = namespace.into();
        Self {
            id: id.into(),
            algorithm,
            value_kind,
            canonicalization_version: canonicalization_version.clone(),
            canonical_domain: SharedFingerprintCanonicalDomain::new(
                namespace.clone(),
                canonicalization_version.clone(),
            ),
            namespace,
            digest_bits,
            seed_identity: None,
        }
    }

    /// Attach a seed/domain-separation identity.
    #[must_use]
    pub fn with_seed_identity(mut self, seed_identity: impl Into<String>) -> Self {
        self.seed_identity = non_empty_string(seed_identity.into());
        self
    }

    /// Attach an explicit canonical domain/version. This keeps frontend
    /// payload names out of the reusable fingerprint namespace.
    #[must_use]
    pub fn with_canonical_domain(
        mut self,
        id: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        self.canonical_domain = SharedFingerprintCanonicalDomain::new(id, version);
        self
    }

    /// Validate this fingerprint identity before using it for dedup.
    pub fn validate(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        if self.id.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
                "fingerprint identity id must not be empty",
            ));
        }
        if self.namespace.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_NAMESPACE,
                "fingerprint namespace must not be empty",
            ));
        }
        if self.canonicalization_version.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICALIZATION_VERSION,
                "canonicalization version must not be empty",
            ));
        }
        if self.canonical_domain.id.trim().is_empty()
            || self.canonical_domain.version.trim().is_empty()
        {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICAL_DOMAIN,
                "canonical domain id and version must not be empty",
            ));
        }
        if self.digest_bits == 0 {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_INVALID_DIGEST_BITS,
                "digest_bits must be greater than zero",
            ));
        }
        if self.digest_bits > self.algorithm.max_digest_bits() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_DIGEST_BITS_EXCEED_ALGORITHM,
                format!(
                    "digest_bits {} exceeds {} width {}",
                    self.digest_bits,
                    self.algorithm.code(),
                    self.algorithm.max_digest_bits()
                ),
            ));
        }
        Ok(())
    }

    /// Stable policy identity shared across frontends and execution lanes.
    #[must_use]
    pub fn fingerprint_policy_identity(&self) -> String {
        let digest_bits = self.digest_bits.to_string();
        let domain_identity = self.canonical_domain.identity();
        identity_join([
            "fingerprint_policy",
            self.algorithm.code(),
            self.value_kind.code(),
            &domain_identity,
            &self.canonicalization_version,
            &digest_bits,
            self.seed_identity.as_deref().unwrap_or("seedless"),
        ])
    }

    /// Stable namespace identity for fingerprint values produced under this policy.
    #[must_use]
    pub fn fingerprint_identity(&self) -> String {
        let domain_identity = self.canonical_domain.identity();
        identity_join([
            "fingerprint",
            &self.namespace,
            &domain_identity,
            self.algorithm.code(),
            self.value_kind.code(),
            &self.canonicalization_version,
        ])
    }

    /// Canonical identity for the fingerprint namespace. This is intentionally
    /// the same value as [`Self::fingerprint_identity`], named for evidence rows
    /// that need to make the canonical, frontend-family reusable contract clear.
    #[must_use]
    pub fn canonical_fingerprint_identity(&self) -> String {
        self.fingerprint_identity()
    }

    /// Canonical identity for the frontend-family set that may consume this
    /// fingerprint. Future importers intentionally stay outside this identity
    /// until their canonical layout contract is registered.
    #[must_use]
    pub fn canonical_frontend_family_identity(&self) -> String {
        identity_join([
            "frontend_family_set",
            &evidence_frontend_families(self.reusable_frontend_families()),
        ])
    }

    /// Frontend families that can consume this fingerprint identity when they
    /// provide the same canonical domain/layout bytes.
    #[must_use]
    pub fn reusable_frontend_families(&self) -> &'static [SharedEngineFrontendFamily] {
        SHARED_FINGERPRINT_REUSABLE_FRONTEND_FAMILIES
    }

    /// Whether this identity marks its bytes as frontend-local only.
    #[must_use]
    pub fn is_frontend_local_domain(&self) -> bool {
        is_frontend_local_component(&self.namespace)
            || is_frontend_local_component(&self.canonical_domain.id)
            || is_frontend_local_component(&self.canonical_domain.version)
    }

    /// Stable evidence code for the reuse domain.
    #[must_use]
    pub fn reuse_domain_code(&self) -> &'static str {
        if self.is_frontend_local_domain() {
            "frontend_local_only"
        } else {
            "frontend_reusable"
        }
    }

    /// Validate that a caller supplied the canonical fingerprint alias, not a
    /// frontend-local or stale alias.
    pub fn validate_canonical_fingerprint_alias(
        &self,
        alias: &str,
    ) -> Result<(), SharedFingerprintIdentityRejection> {
        if alias == self.canonical_fingerprint_identity() {
            Ok(())
        } else {
            Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_MALFORMED_CANONICAL_ALIAS,
                format!(
                    "canonical fingerprint alias {} does not match {}",
                    alias,
                    self.canonical_fingerprint_identity()
                ),
            ))
        }
    }

    /// Validate the frontend-family reuse contract for this fingerprint.
    pub fn validate_reuse_contract(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        self.validate()?;
        if self.is_frontend_local_domain() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_FRONTEND_LOCAL_DOMAIN,
                "frontend-local fingerprint domains cannot be reused across frontend families",
            ));
        }
        for required_family in SHARED_FINGERPRINT_REUSABLE_FRONTEND_FAMILIES {
            if !self.reusable_frontend_families().contains(required_family) {
                return Err(SharedFingerprintIdentityRejection::new(
                    SHARED_FINGERPRINT_REJECTION_MISSING_REUSABLE_FRONTEND_FAMILY,
                    format!(
                        "fingerprint identity must declare reusable frontend family {}",
                        required_family.code()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Identity fields that can be merged into setup, prepared-program, or lane evidence.
    #[must_use]
    pub fn identity_fields(&self) -> CheckerArtifactIdentityFields {
        CheckerArtifactIdentityFields::new()
            .with_fingerprint_policy_identity(self.fingerprint_policy_identity())
            .with_fingerprint_identity(self.fingerprint_identity())
    }

    /// Prepared-program descriptor using the same shared identity strings.
    #[must_use]
    pub fn prepared_descriptor(&self) -> PreparedFingerprintDescriptor {
        PreparedFingerprintDescriptor::new(
            self.id.clone(),
            self.algorithm.prepared_scheme(),
            self.canonicalization_version.clone(),
        )
        .with_identity_fields(self.identity_fields())
    }

    /// Render one frontend-neutral evidence row. `source_kind` identifies the
    /// producer of the current row, not the fingerprint identity.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str, source_kind: CheckerSourceKind) -> String {
        format!(
            "{} shared_fingerprint_identity schema={} schema_version={} source_kind={} frontend_kind={} id={} algorithm={} value_kind={} canonicalization_version={} canonical_domain={} canonical_domain_version={} canonical_domain_identity={} namespace={} digest_bits={} seed_identity={} fingerprint_policy_identity={} fingerprint_identity={} canonical_fingerprint_identity={} canonical_frontend_family_identity={} frontend_reuse_domain={} frontend_family_reusable={} compatible_frontend_families={}",
            scope,
            SHARED_FINGERPRINT_IDENTITY_SCHEMA,
            SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
            source_kind.code(),
            source_kind.code(),
            evidence_value(&self.id),
            self.algorithm.code(),
            self.value_kind.code(),
            evidence_value(&self.canonicalization_version),
            evidence_value(&self.canonical_domain.id),
            evidence_value(&self.canonical_domain.version),
            evidence_value(&self.canonical_domain.identity()),
            evidence_value(&self.namespace),
            self.digest_bits,
            evidence_optional(self.seed_identity.as_deref()),
            evidence_value(&self.fingerprint_policy_identity()),
            evidence_value(&self.fingerprint_identity()),
            evidence_value(&self.canonical_fingerprint_identity()),
            evidence_value(&self.canonical_frontend_family_identity()),
            self.reuse_domain_code(),
            !self.is_frontend_local_domain(),
            evidence_frontend_families(self.reusable_frontend_families()),
        )
    }

    /// Render one fail-closed admission row for this identity.
    #[must_use]
    pub fn render_validation_evidence_row(
        &self,
        scope: &str,
        source_kind: CheckerSourceKind,
    ) -> String {
        match self.validate_reuse_contract() {
            Ok(()) => format!(
                "{} shared_fingerprint_identity_validation schema={} schema_version={} source_kind={} frontend_kind={} id={} status_code=accepted reason_code=accepted fail_closed=true fingerprint_policy_identity={} fingerprint_identity={} canonical_fingerprint_identity={} canonical_frontend_family_identity={} frontend_reuse_domain={} frontend_family_reusable=true compatible_frontend_families={}",
                scope,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
                source_kind.code(),
                source_kind.code(),
                evidence_value(&self.id),
                evidence_value(&self.fingerprint_policy_identity()),
                evidence_value(&self.fingerprint_identity()),
                evidence_value(&self.canonical_fingerprint_identity()),
                evidence_value(&self.canonical_frontend_family_identity()),
                self.reuse_domain_code(),
                evidence_frontend_families(self.reusable_frontend_families()),
            ),
            Err(rejection) => format!(
                "{} shared_fingerprint_identity_validation schema={} schema_version={} source_kind={} frontend_kind={} id={} status_code=rejected reason_code={} fail_closed=true detail={}",
                scope,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
                source_kind.code(),
                source_kind.code(),
                evidence_value(&self.id),
                rejection.reason_code,
                evidence_value(&rejection.detail),
            ),
        }
    }
}

/// Complete dedup contract for a lane or artifact family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedDedupIdentity {
    /// Human-readable dedup descriptor id.
    pub id: String,
    /// Fingerprint namespace used for equality.
    pub fingerprint: SharedFingerprintIdentity,
    /// Scope where equality is interpreted as duplication.
    pub scope: SharedDedupScope,
    /// Storage policy for dedup state.
    pub storage: SharedDedupStorageKind,
    /// Lane that consumes the dedup contract.
    pub lane: SetupTraceLaneKind,
    /// Policy for handling equal fingerprints.
    pub collision_policy: SharedCollisionPolicy,
    /// Optional storage configuration/version identity.
    pub storage_config_identity: Option<String>,
}

impl SharedDedupIdentity {
    /// Create a shared dedup identity.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        fingerprint: SharedFingerprintIdentity,
        scope: SharedDedupScope,
        storage: SharedDedupStorageKind,
        lane: SetupTraceLaneKind,
    ) -> Self {
        Self {
            id: id.into(),
            fingerprint,
            scope,
            storage,
            lane,
            collision_policy: SharedCollisionPolicy::RejectOnCollision,
            storage_config_identity: None,
        }
    }

    /// Attach a storage configuration/version identity.
    #[must_use]
    pub fn with_storage_config_identity(
        mut self,
        storage_config_identity: impl Into<String>,
    ) -> Self {
        self.storage_config_identity = non_empty_string(storage_config_identity.into());
        self
    }

    /// Set the collision policy used by this dedup contract.
    #[must_use]
    pub fn with_collision_policy(mut self, collision_policy: SharedCollisionPolicy) -> Self {
        self.collision_policy = collision_policy;
        self
    }

    /// Validate this dedup identity before using it for an active lane.
    pub fn validate(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        if self.id.trim().is_empty() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
                "dedup identity id must not be empty",
            ));
        }
        self.fingerprint.validate()
    }

    /// Require a fail-closed collision policy before active dedup admission.
    pub fn require_fail_closed(&self) -> Result<(), SharedFingerprintIdentityRejection> {
        self.validate()?;
        if !self.collision_policy.is_fail_closed() {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_NON_FAIL_CLOSED_COLLISION_POLICY,
                "dedup collision policy must fail closed",
            ));
        }
        Ok(())
    }

    /// Require fail-closed, frontend-family reusable admission before a dedup
    /// contract can be advertised as shared infrastructure.
    pub fn require_frontend_reusable_admission(
        &self,
    ) -> Result<(), SharedFingerprintIdentityRejection> {
        self.require_fail_closed()?;
        self.fingerprint.validate_reuse_contract()?;
        self.fingerprint.validate_canonical_fingerprint_alias(
            &self.fingerprint.canonical_fingerprint_identity(),
        )?;
        if self.collision_policy.requires_validation_receipt()
            && self.fingerprint.value_kind != SharedFingerprintValueKind::ValidationReceipt
        {
            return Err(SharedFingerprintIdentityRejection::new(
                SHARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REQUIRES_VALIDATION_RECEIPT,
                "proof/witness dedup admission requires validation-receipt fingerprints",
            ));
        }
        Ok(())
    }

    /// Stable dedup policy identity.
    #[must_use]
    pub fn dedup_identity(&self) -> String {
        identity_join([
            "dedup",
            self.scope.code(),
            self.storage.code(),
            self.lane.code(),
            self.collision_policy.code(),
            &self.fingerprint.fingerprint_identity(),
        ])
    }

    /// Stable storage-policy identity for this dedup contract.
    #[must_use]
    pub fn storage_policy_identity(&self) -> String {
        identity_join([
            "dedup_storage",
            self.storage.code(),
            self.scope.code(),
            self.storage_config_identity
                .as_deref()
                .unwrap_or("default_config"),
        ])
    }

    /// Identity fields that can be merged into setup, prepared-program, or lane evidence.
    #[must_use]
    pub fn identity_fields(&self) -> CheckerArtifactIdentityFields {
        self.fingerprint
            .identity_fields()
            .with_storage_policy_identity(self.storage_policy_identity())
            .with_artifact_identity(self.dedup_identity())
    }

    /// Prepared-program fingerprint descriptor with dedup storage identity attached.
    #[must_use]
    pub fn prepared_fingerprint_descriptor(&self) -> PreparedFingerprintDescriptor {
        self.fingerprint
            .prepared_descriptor()
            .with_storage_policy_identity(self.storage_policy_identity())
    }

    /// Render one frontend-neutral evidence row. `source_kind` identifies the
    /// producer of the current row, not the dedup identity.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str, source_kind: CheckerSourceKind) -> String {
        format!(
            "{} shared_dedup_identity schema={} schema_version={} source_kind={} frontend_kind={} id={} lane_kind={} lane={} dedup_scope={} storage_kind={} collision_policy={} collision_fail_closed={} validation_receipt_required={} validation_receipt_admission_policy={} missing_validation_receipt_policy=reject storage_config_identity={} dedup_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} canonical_fingerprint_identity={} canonical_frontend_family_identity={} fingerprint_value_kind={} frontend_reuse_domain={} frontend_family_reusable={} compatible_frontend_families={} dedup_admission_policy={}",
            scope,
            SHARED_FINGERPRINT_IDENTITY_SCHEMA,
            SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
            source_kind.code(),
            source_kind.code(),
            evidence_value(&self.id),
            self.lane.code(),
            self.lane.code(),
            self.scope.code(),
            self.storage.code(),
            self.collision_policy.code(),
            self.collision_policy.is_fail_closed(),
            self.collision_policy.requires_validation_receipt(),
            validation_receipt_admission_policy(self.collision_policy),
            evidence_optional(self.storage_config_identity.as_deref()),
            evidence_value(&self.dedup_identity()),
            evidence_value(&self.storage_policy_identity()),
            evidence_value(&self.fingerprint.fingerprint_policy_identity()),
            evidence_value(&self.fingerprint.fingerprint_identity()),
            evidence_value(&self.fingerprint.canonical_fingerprint_identity()),
            evidence_value(&self.fingerprint.canonical_frontend_family_identity()),
            self.fingerprint.value_kind.code(),
            self.fingerprint.reuse_domain_code(),
            !self.fingerprint.is_frontend_local_domain(),
            evidence_frontend_families(self.fingerprint.reusable_frontend_families()),
            self.collision_policy.code(),
        )
    }

    /// Render one fail-closed admission row for this dedup identity.
    #[must_use]
    pub fn render_validation_evidence_row(
        &self,
        scope: &str,
        source_kind: CheckerSourceKind,
    ) -> String {
        match self.require_frontend_reusable_admission() {
            Ok(()) => format!(
                "{} shared_dedup_identity_validation schema={} schema_version={} source_kind={} frontend_kind={} id={} status_code=accepted reason_code=accepted fail_closed=true collision_policy={} validation_receipt_required={} validation_receipt_admission_policy={} missing_validation_receipt_policy=reject dedup_identity={} fingerprint_identity={} canonical_fingerprint_identity={} canonical_frontend_family_identity={} frontend_reuse_domain={} frontend_family_reusable=true compatible_frontend_families={}",
                scope,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
                source_kind.code(),
                source_kind.code(),
                evidence_value(&self.id),
                self.collision_policy.code(),
                self.collision_policy.requires_validation_receipt(),
                validation_receipt_admission_policy(self.collision_policy),
                evidence_value(&self.dedup_identity()),
                evidence_value(&self.fingerprint.fingerprint_identity()),
                evidence_value(&self.fingerprint.canonical_fingerprint_identity()),
                evidence_value(&self.fingerprint.canonical_frontend_family_identity()),
                self.fingerprint.reuse_domain_code(),
                evidence_frontend_families(self.fingerprint.reusable_frontend_families()),
            ),
            Err(rejection) => format!(
                "{} shared_dedup_identity_validation schema={} schema_version={} source_kind={} frontend_kind={} id={} status_code=rejected reason_code={} fail_closed=true collision_policy={} detail={}",
                scope,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA,
                SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
                source_kind.code(),
                source_kind.code(),
                evidence_value(&self.id),
                rejection.reason_code,
                self.collision_policy.code(),
                evidence_value(&rejection.detail),
            ),
        }
    }
}

fn non_empty_string(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn identity_join<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    parts
        .into_iter()
        .map(identity_component)
        .collect::<Vec<_>>()
        .join(":")
}

fn identity_component(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
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
}

fn evidence_value(value: &str) -> String {
    identity_component(value)
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn is_frontend_local_component(value: &str) -> bool {
    let value = identity_component(value);
    value.starts_with("frontend-local")
        || value.starts_with("frontend_local")
        || value.contains(".frontend-local")
        || value.contains(".frontend_local")
}

fn validation_receipt_admission_policy(collision_policy: SharedCollisionPolicy) -> &'static str {
    if collision_policy.requires_validation_receipt() {
        "required"
    } else {
        "not_required"
    }
}

fn validate_optional_identity(
    field: &'static str,
    value: Option<&str>,
) -> Result<(), SharedFingerprintIdentityRejection> {
    if value.is_some_and(|identity| identity.trim().is_empty() || identity.trim() == "none") {
        return Err(SharedFingerprintIdentityRejection::new(
            SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
            format!("{field} must not be empty when present"),
        ));
    }
    Ok(())
}

fn has_optional_identity(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|identity| !identity.is_empty() && identity != "none")
}

fn duplicate_frontend_family(
    frontend_families: &[SharedEngineFrontendFamily],
) -> Option<SharedEngineFrontendFamily> {
    for (index, family) in frontend_families.iter().copied().enumerate() {
        if frontend_families[..index].contains(&family) {
            return Some(family);
        }
    }
    None
}

fn unique_frontend_family_count(frontend_families: &[SharedEngineFrontendFamily]) -> usize {
    let mut count = 0;
    for (index, family) in frontend_families.iter().enumerate() {
        if !frontend_families[..index].contains(family) {
            count += 1;
        }
    }
    count
}

fn evidence_frontend_families(frontend_families: &[SharedEngineFrontendFamily]) -> String {
    if frontend_families.is_empty() {
        "none".to_string()
    } else {
        frontend_families
            .iter()
            .map(|family| family.code())
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared_state_fingerprint() -> SharedFingerprintIdentity {
        SharedFingerprintIdentity::new(
            "flat state",
            SharedFingerprintAlgorithm::Xxh3U64,
            SharedFingerprintValueKind::State,
            "flat-state-layout-v1",
            "flat-state",
            64,
        )
        .with_canonical_domain("flat-state-domain", "v1")
        .with_seed_identity("explicit-state-domain")
    }

    fn shared_domain_key() -> FingerprintDomainKey {
        FingerprintDomainKey::builder(SharedFingerprintAlgorithm::Xxh3U64)
            .helper_symbol("ty_compiled_fp_u64")
            .seed_identity("flat-compiled-domain-seed-v1")
            .canonical_payload(FingerprintCanonicalPayload::new(
                SharedFingerprintValueKind::State,
                SharedFingerprintCanonicalDomain::new("compiled-flat-state", "i64-v1"),
                "flat-i64-state-v1",
                64,
            ))
            .layout_digest("sha256:layout-a")
            .projection(FingerprintDomainProjection::Full)
            .storage_policy(
                FingerprintDomainStoragePolicy::new(
                    SharedDedupScope::StateSpace,
                    SharedDedupStorageKind::Cas,
                )
                .with_storage_config_identity("partitioned-cas-v1"),
            )
            .collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality)
            .build()
            .expect("shared domain key should be valid")
    }

    #[test]
    fn fingerprint_domain_key_records_frontend_neutral_components() {
        let key = shared_domain_key();
        let identity = key.stable_identity();

        assert!(identity.starts_with("fingerprint_domain_key:xxh3_u64"));
        assert!(identity.contains("ty_compiled_fp_u64"));
        assert!(identity.contains("flat-compiled-domain-seed-v1"));
        assert!(identity.contains("compiled-flat-state"));
        assert!(identity.contains("sha256_layout-a"));
        assert!(identity.contains("projection_full"));
        assert!(identity.contains("storage_policy_cas_state_space_partitioned-cas-v1"));
        assert!(identity.contains("canonical_payload_equality"));
        assert!(identity.contains("fail_closed"));
    }

    #[test]
    fn fingerprint_domain_key_layout_projection_and_seed_change_stable_identity() {
        let base = shared_domain_key();
        let base_identity = base.stable_identity();

        let layout_changed = FingerprintDomainKey {
            layout_digest: "sha256:layout-b".to_string(),
            ..base.clone()
        };
        assert_ne!(base_identity, layout_changed.stable_identity());

        let projection_changed = FingerprintDomainKey {
            projection: FingerprintDomainProjection::LowBits(32),
            ..base.clone()
        };
        assert_ne!(base_identity, projection_changed.stable_identity());

        let seed_changed = FingerprintDomainKey {
            seed_identity: Some("flat-compiled-domain-seed-v2".to_string()),
            ..base
        };
        assert_ne!(base_identity, seed_changed.stable_identity());
    }

    #[test]
    fn fingerprint_domain_key_requires_explicit_fail_closed_acceptance() {
        let key = shared_domain_key();
        let accepted = key
            .accepted_fail_closed_policy_identity()
            .expect("canonical payload equality is fail-closed");

        assert!(accepted.starts_with("accepted_fail_closed_fingerprint_domain"));
        assert!(accepted.contains("canonical_payload_equality"));
        assert!(accepted.contains("fail_closed"));
        assert!(accepted.contains(&evidence_value(&key.stable_identity())));

        let unchecked = FingerprintDomainKey {
            collision_policy: SharedCollisionPolicy::Unchecked,
            ..key
        };
        assert!(unchecked.stable_identity().contains("unchecked"));
        assert!(unchecked.stable_identity().contains("not_fail_closed"));
        let err = unchecked.accepted_fail_closed_policy_identity().expect_err(
            "unchecked collision policy must not produce accepted fail-closed identity",
        );
        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_NON_FAIL_CLOSED_COLLISION_POLICY
        );
    }

    #[test]
    fn shared_native_planning_identity_records_reuse_domains_and_scope() {
        let planning = SharedNativePlanningIdentity::new([
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::MccPetri,
            SharedEngineFrontendFamily::Aiger,
            SharedEngineFrontendFamily::Btor2,
        ])
        .with_source_fingerprint("sha256:source")
        .with_plan_reuse_manifest("trust-ir-batch-manifest-v1-abc", "sha256:manifest")
        .with_fingerprint_domain_identity("fingerprint_domain_key:canonical_bytes_sha256")
        .with_cas_identity("trust-ir-batch-cas-compat-v1-123")
        .with_cache_identity("trust-ir-batch-cache-compat-v1-456")
        .with_cache_reuse_policy_kind(SharedNativeCacheReusePolicy::FrontendReusable);

        planning.validate(CheckerSourceKind::Aiger).unwrap();
        assert_eq!(
            planning.cache_reuse_policy_kind(),
            Some(SharedNativeCacheReusePolicy::FrontendReusable)
        );
        assert!(planning
            .stable_identity()
            .starts_with("native_planning_identity"));
        assert!(planning.stable_identity().contains("sha256_source"));
        assert!(planning
            .stable_identity()
            .contains("trust-ir-batch-manifest-v1-abc"));
        assert!(planning
            .stable_identity()
            .contains("fingerprint_domain_key_canonical_bytes_sha256"));
        assert!(planning.frontend_family_reusable());
        assert!(planning
            .frontend_family_scope_identity()
            .contains("native_frontend_family_scope"));

        let row = planning.render_evidence_row("CORE", CheckerSourceKind::Aiger);
        assert!(row.starts_with("CORE shared_native_planning_identity "));
        assert!(row.contains("schema=ty.shared.native_planning_identity.v1"));
        assert!(row.contains("source_kind=aiger"));
        assert!(row.contains("frontend_kind=aiger"));
        assert!(row.contains("source_fingerprint=sha256_source"));
        assert!(row.contains("plan_reuse_manifest_id=trust-ir-batch-manifest-v1-abc"));
        assert!(row.contains("plan_reuse_manifest_digest=sha256_manifest"));
        assert!(row
            .contains("fingerprint_domain_identity=fingerprint_domain_key_canonical_bytes_sha256"));
        assert!(row.contains("cas_identity=trust-ir-batch-cas-compat-v1-123"));
        assert!(row.contains("cache_identity=trust-ir-batch-cache-compat-v1-456"));
        assert!(row.contains("cache_reuse_policy=frontend_reusable"));
        assert!(row.contains("frontend_family_scope=tla_plus,mcc_petri,aiger,btor2"));
        assert!(row.contains("frontend_family_scope_identity=native_frontend_family_scope"));
        assert!(row.contains("frontend_family_reusable=true"));
    }

    #[test]
    fn shared_native_planning_identity_rejects_partial_manifest_and_scope_mismatch() {
        let partial_manifest =
            SharedNativePlanningIdentity::new([SharedEngineFrontendFamily::TlaPlus])
                .with_plan_reuse_manifest("trust-ir-batch-manifest-v1-abc", "");
        let err = partial_manifest
            .validate(CheckerSourceKind::Tla)
            .expect_err("partial plan reuse manifest must reject");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_INCOMPLETE_PLAN_REUSE_MANIFEST
        );

        let wrong_scope = SharedNativePlanningIdentity::new([SharedEngineFrontendFamily::MccPetri]);
        let err = wrong_scope
            .validate(CheckerSourceKind::Tla)
            .expect_err("source frontend family must be inside scope");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_MISSING_SOURCE_FRONTEND_FAMILY
        );

        let empty_scope =
            SharedNativePlanningIdentity::new(std::iter::empty::<SharedEngineFrontendFamily>());
        let err = empty_scope
            .validate(CheckerSourceKind::Unknown)
            .expect_err("empty frontend-family scope must reject");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_EMPTY_FRONTEND_FAMILY_SCOPE
        );
    }

    #[test]
    fn shared_native_planning_identity_rejects_untyped_or_unbacked_reuse_claims() {
        let invalid_policy =
            SharedNativePlanningIdentity::new([SharedEngineFrontendFamily::TlaPlus])
                .with_cache_reuse_policy("opportunistic");
        let err = invalid_policy
            .validate(CheckerSourceKind::Tla)
            .expect_err("unregistered cache reuse policy must reject");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_INVALID_CACHE_REUSE_POLICY
        );

        let duplicate_scope = SharedNativePlanningIdentity::new([
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::TlaPlus,
        ]);
        let err = duplicate_scope
            .validate(CheckerSourceKind::Tla)
            .expect_err("duplicate frontend-family scopes must reject");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_DUPLICATE_FRONTEND_FAMILY
        );
        assert!(!duplicate_scope.frontend_family_reusable());

        let single_family_reuse =
            SharedNativePlanningIdentity::new([SharedEngineFrontendFamily::TlaPlus])
                .with_cache_reuse_policy_kind(SharedNativeCacheReusePolicy::FrontendReusable)
                .with_plan_reuse_manifest("trust-ir-batch-plan", "sha256:plan")
                .with_fingerprint_domain_identity("fingerprint-domain")
                .with_cache_identity("cache-namespace");
        let err = single_family_reuse
            .validate(CheckerSourceKind::Tla)
            .expect_err("frontend-reusable planning needs a multi-family scope");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_COMPATIBLE_FAMILIES
        );

        let missing_manifest = SharedNativePlanningIdentity::new([
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::MccPetri,
        ])
        .with_cache_reuse_policy_kind(SharedNativeCacheReusePolicy::FrontendReusable)
        .with_fingerprint_domain_identity("fingerprint-domain")
        .with_cache_identity("cache-namespace");
        let err = missing_manifest
            .validate(CheckerSourceKind::Tla)
            .expect_err("frontend-reusable planning needs a plan manifest");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_PLAN_REUSE_MANIFEST
        );

        let missing_fingerprint_domain = SharedNativePlanningIdentity::new([
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::MccPetri,
        ])
        .with_cache_reuse_policy_kind(SharedNativeCacheReusePolicy::FrontendReusable)
        .with_plan_reuse_manifest("trust-ir-batch-plan", "sha256:plan")
        .with_cache_identity("cache-namespace");
        let err = missing_fingerprint_domain
            .validate(CheckerSourceKind::Tla)
            .expect_err("frontend-reusable planning needs a fingerprint-domain identity");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_FINGERPRINT_DOMAIN
        );

        let missing_cache_identity = SharedNativePlanningIdentity::new([
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::MccPetri,
        ])
        .with_cache_reuse_policy_kind(SharedNativeCacheReusePolicy::FrontendReusable)
        .with_plan_reuse_manifest("trust-ir-batch-plan", "sha256:plan")
        .with_fingerprint_domain_identity("fingerprint-domain");
        let err = missing_cache_identity
            .validate(CheckerSourceKind::Tla)
            .expect_err("frontend-reusable planning needs a cache identity");
        assert_eq!(
            err.reason_code,
            SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_CACHE_IDENTITY
        );
    }

    #[test]
    fn shared_fingerprint_identity_is_frontend_independent() {
        let fingerprint = shared_state_fingerprint();
        let policy = fingerprint.fingerprint_policy_identity();
        let namespace = fingerprint.fingerprint_identity();

        for source in [
            CheckerSourceKind::Tla,
            CheckerSourceKind::Quint,
            CheckerSourceKind::MccPetri,
            CheckerSourceKind::Aiger,
            CheckerSourceKind::Btor2,
            CheckerSourceKind::VmtInterchange,
            CheckerSourceKind::AYOnly,
            CheckerSourceKind::WitnessReplay,
        ] {
            let row = fingerprint.render_evidence_row("CORE", source);
            assert!(row.contains("shared_fingerprint_identity"));
            assert!(row.contains(&format!("source_kind={}", source.code())));
            assert!(row.contains("canonical_domain=flat-state-domain"));
            assert!(row.contains("canonical_domain_version=v1"));
            assert!(row.contains("frontend_family_reusable=true"));
            assert!(row.contains("frontend_reuse_domain=frontend_reusable"));
            assert!(row.contains("canonical_frontend_family_identity=frontend_family_set"));
            assert!(row.contains(
                "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
            ));
            assert!(row.contains(&format!(
                "fingerprint_policy_identity={}",
                evidence_value(&policy)
            )));
            assert!(row.contains(&format!(
                "fingerprint_identity={}",
                evidence_value(&namespace)
            )));
            assert!(row.contains(&format!(
                "canonical_fingerprint_identity={}",
                evidence_value(&namespace)
            )));
        }
    }

    #[test]
    fn shared_dedup_identity_materializes_identity_fields() {
        let dedup = SharedDedupIdentity::new(
            "visited states",
            shared_state_fingerprint(),
            SharedDedupScope::StateSpace,
            SharedDedupStorageKind::ShardedInMemory,
            SetupTraceLaneKind::ExplicitState,
        )
        .with_storage_config_identity("shards=64");

        let fields = dedup.identity_fields();
        assert_eq!(
            fields.fingerprint_policy_identity.as_deref(),
            Some(dedup.fingerprint.fingerprint_policy_identity().as_str())
        );
        assert_eq!(
            fields.fingerprint_identity.as_deref(),
            Some(dedup.fingerprint.fingerprint_identity().as_str())
        );
        assert_eq!(
            fields.storage_policy_identity.as_deref(),
            Some(dedup.storage_policy_identity().as_str())
        );
        assert_eq!(
            fields.artifact_identity.as_deref(),
            Some(dedup.dedup_identity().as_str())
        );

        let row = dedup.render_evidence_row("CORE", CheckerSourceKind::MccPetri);
        assert!(row.contains("shared_dedup_identity"));
        assert!(row.contains("source_kind=mcc_petri"));
        assert!(row.contains("lane_kind=explicit_state"));
        assert!(row.contains("dedup_scope=state_space"));
        assert!(row.contains("storage_kind=sharded_in_memory"));
        assert!(row.contains("collision_policy=reject_on_collision"));
        assert!(row.contains("collision_fail_closed=true"));
        assert!(row.contains("validation_receipt_required=false"));
        assert!(row.contains("validation_receipt_admission_policy=not_required"));
        assert!(row.contains("missing_validation_receipt_policy=reject"));
        assert!(row.contains("storage_config_identity=shards_64"));
        assert!(row.contains("dedup_admission_policy=reject_on_collision"));
        assert!(row.contains("frontend_reuse_domain=frontend_reusable"));
        assert!(row.contains("frontend_family_reusable=true"));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));

        let validation = dedup.render_validation_evidence_row("CORE", CheckerSourceKind::MccPetri);
        assert!(validation.contains("shared_dedup_identity_validation"));
        assert!(validation.contains("status_code=accepted"));
        assert!(validation.contains("reason_code=accepted"));
        assert!(validation.contains("fail_closed=true"));
        assert!(validation.contains("validation_receipt_required=false"));
        assert!(validation.contains("validation_receipt_admission_policy=not_required"));
        assert!(validation.contains("canonical_fingerprint_identity="));
        assert!(validation.contains("canonical_frontend_family_identity=frontend_family_set"));
    }

    #[test]
    fn shared_fingerprint_identity_feeds_prepared_program_descriptor() {
        let dedup = SharedDedupIdentity::new(
            "trust-cg local dedup",
            shared_state_fingerprint(),
            SharedDedupScope::FrontierBatch,
            SharedDedupStorageKind::Cas,
            SetupTraceLaneKind::Native,
        );
        let descriptor = dedup.prepared_fingerprint_descriptor();

        assert_eq!(descriptor.id, "flat state");
        assert_eq!(descriptor.scheme, PreparedFingerprintScheme::Xxh3U64);
        assert_eq!(descriptor.canonicalization_version, "flat-state-layout-v1");
        assert_eq!(
            descriptor.identities.fingerprint_policy_identity.as_deref(),
            Some(dedup.fingerprint.fingerprint_policy_identity().as_str())
        );
        assert_eq!(
            descriptor.identities.fingerprint_identity.as_deref(),
            Some(dedup.fingerprint.fingerprint_identity().as_str())
        );
        assert_eq!(
            descriptor.identities.storage_policy_identity.as_deref(),
            Some(dedup.storage_policy_identity().as_str())
        );
    }

    #[test]
    fn fingerprint_policy_changes_when_canonicalization_changes() {
        let base = shared_state_fingerprint();
        let changed = SharedFingerprintIdentity::new(
            "flat state",
            SharedFingerprintAlgorithm::Xxh3U64,
            SharedFingerprintValueKind::State,
            "flat-state-layout-v2",
            "flat-state",
            64,
        )
        .with_seed_identity("explicit-state-domain");

        assert_ne!(
            base.fingerprint_policy_identity(),
            changed.fingerprint_policy_identity()
        );
        assert_ne!(base.fingerprint_identity(), changed.fingerprint_identity());
    }

    #[test]
    fn canonical_domain_version_participates_in_identity() {
        let base = shared_state_fingerprint();
        let changed_domain = SharedFingerprintIdentity::new(
            "flat state",
            SharedFingerprintAlgorithm::Xxh3U64,
            SharedFingerprintValueKind::State,
            "flat-state-layout-v1",
            "flat-state",
            64,
        )
        .with_canonical_domain("flat-state-domain", "v2")
        .with_seed_identity("explicit-state-domain");

        assert_ne!(
            base.fingerprint_policy_identity(),
            changed_domain.fingerprint_policy_identity()
        );
        assert_ne!(
            base.fingerprint_identity(),
            changed_domain.fingerprint_identity()
        );
        assert_eq!(
            base.canonical_domain.identity(),
            "canonical_domain:flat-state-domain:v1"
        );
    }

    #[test]
    fn invalid_fingerprint_identity_rejects_fail_closed() {
        let invalid = SharedFingerprintIdentity::new(
            "bad",
            SharedFingerprintAlgorithm::Xxh3U64,
            SharedFingerprintValueKind::State,
            "flat-state-layout-v1",
            "flat-state",
            128,
        );
        let err = invalid
            .validate()
            .expect_err("xxh3_u64 cannot report 128 significant bits");
        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_DIGEST_BITS_EXCEED_ALGORITHM
        );

        let row = invalid.render_validation_evidence_row("CORE", CheckerSourceKind::Quint);
        assert!(row.contains("shared_fingerprint_identity_validation"));
        assert!(row.contains("status_code=rejected"));
        assert!(row.contains("reason_code=digest_bits_exceed_algorithm"));
        assert!(row.contains("fail_closed=true"));
    }

    #[test]
    fn empty_canonical_domain_rejects_fail_closed() {
        let invalid = shared_state_fingerprint().with_canonical_domain("", "v1");
        let err = invalid
            .validate()
            .expect_err("empty canonical domain must reject");
        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICAL_DOMAIN
        );
    }

    #[test]
    fn unchecked_collision_policy_is_rejected_for_active_dedup() {
        let dedup = SharedDedupIdentity::new(
            "unsafe dedup",
            shared_state_fingerprint(),
            SharedDedupScope::StateSpace,
            SharedDedupStorageKind::InMemory,
            SetupTraceLaneKind::ExplicitState,
        )
        .with_collision_policy(SharedCollisionPolicy::Unchecked);

        let err = dedup
            .require_fail_closed()
            .expect_err("unchecked collision policy must not admit active dedup");
        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_NON_FAIL_CLOSED_COLLISION_POLICY
        );

        let row = dedup.render_validation_evidence_row("CORE", CheckerSourceKind::Tla);
        assert!(row.contains("shared_dedup_identity_validation"));
        assert!(row.contains("status_code=rejected"));
        assert!(row.contains("reason_code=non_fail_closed_collision_policy"));
        assert!(row.contains("collision_policy=unchecked"));
        assert!(row.contains("fail_closed=true"));
    }

    #[test]
    fn collision_policy_participates_in_dedup_identity() {
        let reject_on_collision = SharedDedupIdentity::new(
            "batch dedup",
            shared_state_fingerprint(),
            SharedDedupScope::BatchArtifact,
            SharedDedupStorageKind::EvidenceOnly,
            SetupTraceLaneKind::Native,
        );
        let canonical_payload_equality = reject_on_collision
            .clone()
            .with_collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality);

        assert_ne!(
            reject_on_collision.dedup_identity(),
            canonical_payload_equality.dedup_identity()
        );
        assert!(canonical_payload_equality.require_fail_closed().is_ok());
        assert!(canonical_payload_equality
            .render_evidence_row("CORE", CheckerSourceKind::AYOnly)
            .contains("collision_policy=canonical_payload_equality"));
    }

    #[test]
    fn shared_fingerprint_identity_declares_reusable_frontend_family_contract() {
        let fingerprint = shared_state_fingerprint();
        let families = fingerprint.reusable_frontend_families();

        assert!(families.contains(&SharedEngineFrontendFamily::TlaPlus));
        assert!(families.contains(&SharedEngineFrontendFamily::MccPetri));
        assert!(families.contains(&SharedEngineFrontendFamily::Aiger));
        assert!(families.contains(&SharedEngineFrontendFamily::Btor2));
        assert!(families.contains(&SharedEngineFrontendFamily::VmtTransitionSystem));
        assert!(families.contains(&SharedEngineFrontendFamily::AYAnalytical));
        assert!(families.contains(&SharedEngineFrontendFamily::WitnessReplay));
        assert!(!families.contains(&SharedEngineFrontendFamily::FutureImporter));
        assert!(fingerprint.validate_reuse_contract().is_ok());
        assert_eq!(fingerprint.reuse_domain_code(), "frontend_reusable");
        assert_eq!(
            fingerprint.canonical_fingerprint_identity(),
            fingerprint.fingerprint_identity()
        );
        assert!(fingerprint
            .validate_canonical_fingerprint_alias(&fingerprint.canonical_fingerprint_identity())
            .is_ok());

        let row = fingerprint.render_validation_evidence_row("CORE", CheckerSourceKind::Aiger);
        assert!(row.contains("status_code=accepted"));
        assert!(row.contains("frontend_family_reusable=true"));
        assert!(row.contains("frontend_reuse_domain=frontend_reusable"));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains("canonical_fingerprint_identity="));
        assert!(row.contains("canonical_frontend_family_identity=frontend_family_set"));
    }

    #[test]
    fn proof_witness_dedup_policy_declares_validation_receipt_requirement() {
        let dedup = SharedDedupIdentity::new(
            "proof replay cache",
            SharedFingerprintIdentity::new(
                "validation receipt",
                SharedFingerprintAlgorithm::CanonicalBytesSha256,
                SharedFingerprintValueKind::ValidationReceipt,
                "validation-receipt-v1",
                "validation-receipt",
                256,
            )
            .with_canonical_domain("validation-receipt", "v1"),
            SharedDedupScope::ReplayTrace,
            SharedDedupStorageKind::EvidenceOnly,
            SetupTraceLaneKind::Replay,
        )
        .with_collision_policy(SharedCollisionPolicy::ProofWitnessRequired);

        assert!(dedup.require_fail_closed().is_ok());
        assert!(dedup.require_frontend_reusable_admission().is_ok());
        assert!(dedup.collision_policy.requires_validation_receipt());
        let row = dedup.render_evidence_row("CORE", CheckerSourceKind::Btor2);
        assert!(row.contains("collision_policy=proof_witness_required"));
        assert!(row.contains("validation_receipt_required=true"));
        assert!(row.contains("validation_receipt_admission_policy=required"));
        assert!(row.contains("missing_validation_receipt_policy=reject"));
        assert!(row.contains("fingerprint_value_kind=validation_receipt"));
        assert!(row.contains("frontend_reuse_domain=frontend_reusable"));
        assert!(row.contains("frontend_family_reusable=true"));
        assert!(row.contains("dedup_admission_policy=proof_witness_required"));

        let validation = dedup.render_validation_evidence_row("CORE", CheckerSourceKind::Btor2);
        assert!(validation.contains("status_code=accepted"));
        assert!(validation.contains("validation_receipt_required=true"));
        assert!(validation.contains("validation_receipt_admission_policy=required"));
        assert!(validation.contains("canonical_fingerprint_identity="));
    }

    #[test]
    fn frontend_local_domains_fail_closed_for_shared_reuse() {
        let local = SharedFingerprintIdentity::new(
            "mcc local cache",
            SharedFingerprintAlgorithm::CanonicalBytesSha256,
            SharedFingerprintValueKind::StateVector,
            "mcc-local-layout-v1",
            "frontend-local.mcc_petri",
            256,
        )
        .with_canonical_domain("frontend-local.mcc_petri", "layout-v1");

        assert!(local.is_frontend_local_domain());
        assert_eq!(local.reuse_domain_code(), "frontend_local_only");
        let err = local
            .validate_reuse_contract()
            .expect_err("frontend-local domains cannot be shared");
        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_FRONTEND_LOCAL_DOMAIN
        );

        let row = local.render_evidence_row("CORE", CheckerSourceKind::MccPetri);
        assert!(row.contains("frontend_reuse_domain=frontend_local_only"));
        assert!(row.contains("frontend_family_reusable=false"));
        let validation = local.render_validation_evidence_row("CORE", CheckerSourceKind::MccPetri);
        assert!(validation.contains("status_code=rejected"));
        assert!(validation.contains("reason_code=frontend_local_domain"));
    }

    #[test]
    fn malformed_canonical_fingerprint_alias_rejects() {
        let fingerprint = shared_state_fingerprint();
        let err = fingerprint
            .validate_canonical_fingerprint_alias("frontend_local_alias")
            .expect_err("stale or frontend-local aliases must reject");

        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_MALFORMED_CANONICAL_ALIAS
        );
        assert!(err.detail.contains("frontend_local_alias"));
        assert!(err
            .detail
            .contains(&fingerprint.canonical_fingerprint_identity()));
    }

    #[test]
    fn compatible_frontend_family_identity_excludes_future_importer_until_registered() {
        let fingerprint = shared_state_fingerprint();
        let family_identity = fingerprint.canonical_frontend_family_identity();

        assert!(family_identity.contains("mcc_petri"));
        assert!(family_identity.contains("aiger"));
        assert!(family_identity.contains("btor2"));
        assert!(family_identity.contains("vmt_transition_system"));
        assert!(family_identity.contains("ay_analytical"));
        assert!(family_identity.contains("witness_replay"));
        assert!(!family_identity.contains("future_importer"));

        let row = fingerprint.render_evidence_row("CORE", CheckerSourceKind::VmtInterchange);
        assert!(row.contains("source_kind=vmt_interchange"));
        assert!(row.contains("canonical_frontend_family_identity=frontend_family_set"));
        assert!(!row.contains("future_importer"));

        let ay_row = fingerprint.render_evidence_row("CORE", CheckerSourceKind::AYOnly);
        assert!(ay_row.contains("source_kind=ay_only"));
        assert!(ay_row.contains("ay_analytical"));
    }

    #[test]
    fn proof_witness_dedup_policy_rejects_non_receipt_fingerprint() {
        let dedup = SharedDedupIdentity::new(
            "unsafe proof replay cache",
            shared_state_fingerprint(),
            SharedDedupScope::ReplayTrace,
            SharedDedupStorageKind::EvidenceOnly,
            SetupTraceLaneKind::Replay,
        )
        .with_collision_policy(SharedCollisionPolicy::ProofWitnessRequired);

        let err = dedup
            .require_frontend_reusable_admission()
            .expect_err("proof/witness admission must use validation-receipt fingerprints");
        assert_eq!(
            err.reason_code,
            SHARED_FINGERPRINT_REJECTION_PROOF_WITNESS_REQUIRES_VALIDATION_RECEIPT
        );

        let row = dedup.render_validation_evidence_row("CORE", CheckerSourceKind::Aiger);
        assert!(row.contains("shared_dedup_identity_validation"));
        assert!(row.contains("status_code=rejected"));
        assert!(row.contains("reason_code=proof_witness_requires_validation_receipt"));
        assert!(row.contains("fail_closed=true"));
        assert!(row.contains("collision_policy=proof_witness_required"));
    }

    #[test]
    fn shared_value_kinds_cover_vector_and_receipt_domains() {
        assert_eq!(
            SharedFingerprintValueKind::StateVector.code(),
            "state_vector"
        );
        assert_eq!(
            SharedFingerprintValueKind::MarkingVector.code(),
            "marking_vector"
        );
        assert_eq!(
            SharedFingerprintValueKind::RegisterVector.code(),
            "register_vector"
        );
        assert_eq!(
            SharedFingerprintValueKind::ValidationReceipt.code(),
            "validation_receipt"
        );
    }

    #[test]
    fn duplicate_authorization_is_collision_policy_aware() {
        assert!(SharedCollisionPolicy::CanonicalPayloadEquality
            .authorizes_duplicate(SharedDuplicateAuthorization::CanonicalPayloadEquality));
        assert!(SharedCollisionPolicy::RejectOnCollision
            .authorizes_duplicate(SharedDuplicateAuthorization::CanonicalPayloadEquality));
        assert!(SharedCollisionPolicy::ProofWitnessRequired
            .authorizes_duplicate(SharedDuplicateAuthorization::ProofWitness));

        assert!(!SharedCollisionPolicy::ProofWitnessRequired
            .authorizes_duplicate(SharedDuplicateAuthorization::CanonicalPayloadEquality));
        assert!(!SharedCollisionPolicy::RejectOnCollision
            .authorizes_duplicate(SharedDuplicateAuthorization::ProofWitness));
        assert!(!SharedCollisionPolicy::Unchecked
            .authorizes_duplicate(SharedDuplicateAuthorization::CanonicalPayloadEquality));
    }
}
