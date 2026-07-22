// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::fingerprint_identity::{
    SharedCollisionPolicy, SharedDedupIdentity, SharedDuplicateAuthorization,
    SharedFingerprintIdentityRejection,
};
use parking_lot::RwLock;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};

const SHARED_DEDUP_ADMISSION_BACKEND: &str = "shared_dedup_identity";
const SHARED_DEDUP_ADMISSION_OPERATION: &str = "admit";
const SHARED_DEDUP_COLLISION_REJECTION: &str = "fingerprint_collision_rejected";

/// Structured storage fault surfaced by checked fingerprint operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{backend} {operation} fault: {detail}")]
#[non_exhaustive]
pub struct StorageFault {
    /// Backend name (for example: `mmap`, `disk`).
    pub backend: &'static str,
    /// Operation name (for example: `insert`, `contains`).
    pub operation: &'static str,
    /// Backend-specific detail.
    pub detail: String,
}

impl StorageFault {
    /// Create a structured storage fault.
    pub fn new(backend: &'static str, operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            backend,
            operation,
            detail: detail.into(),
        }
    }
}

/// Typed result for fingerprint insertion.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertOutcome {
    /// Fingerprint was newly inserted.
    Inserted,
    /// Fingerprint already existed.
    AlreadyPresent,
    /// Storage subsystem fault occurred.
    StorageFault(StorageFault),
}

impl InsertOutcome {
    /// Convert an insert result into the frontend-neutral dedup admission
    /// decision used by shared exploration engines.
    pub fn into_admission(self) -> Result<FingerprintAdmission, StorageFault> {
        match self {
            Self::Inserted => Ok(FingerprintAdmission::New),
            Self::AlreadyPresent => Ok(FingerprintAdmission::Duplicate),
            Self::StorageFault(fault) => Err(fault),
        }
    }
}

/// Frontend-neutral dedup admission decision after a fingerprint insert.
///
/// This is the shared runtime contract that tells a caller whether the state
/// behind a fingerprint should be explored. It deliberately avoids any
/// TLA/Petri/TLC-specific policy so storage optimizations can be reused by all
/// frontends that implement [`FingerprintSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FingerprintAdmission {
    /// The fingerprint was not resident and has now been admitted.
    New,
    /// The fingerprint was already resident and should be treated as deduped.
    Duplicate,
}

impl FingerprintAdmission {
    /// Return true when the caller should process the newly admitted state.
    pub fn is_new(self) -> bool {
        matches!(self, Self::New)
    }

    /// Return true when the caller should suppress the duplicate state.
    pub fn is_duplicate(self) -> bool {
        matches!(self, Self::Duplicate)
    }

    /// Enforce a shared dedup collision policy on an already computed admission.
    ///
    /// Duplicate admissions must be authorized by the caller after comparing
    /// the candidate with the resident canonical payload or validating the
    /// required proof/witness for the policy. Returning `false` rejects the
    /// duplicate as a suspected collision.
    pub fn enforce_shared_collision_policy(
        self,
        dedup_identity: &SharedDedupIdentity,
        authorize_duplicate: &mut dyn FnMut(SharedCollisionPolicy) -> Result<bool, StorageFault>,
    ) -> Result<Self, StorageFault> {
        require_shared_dedup_admission(dedup_identity)?;
        enforce_duplicate_authorization(self, dedup_identity, authorize_duplicate)
    }

    /// Enforce a shared dedup collision policy with typed duplicate evidence.
    ///
    /// This keeps policy interpretation in the shared runtime instead of
    /// requiring each frontend to duplicate `match collision_policy` logic.
    pub fn enforce_shared_duplicate_authorization(
        self,
        dedup_identity: &SharedDedupIdentity,
        authorize_duplicate: &mut dyn FnMut(
            SharedCollisionPolicy,
        )
            -> Result<SharedDuplicateAuthorization, StorageFault>,
    ) -> Result<Self, StorageFault> {
        require_shared_dedup_admission(dedup_identity)?;
        enforce_duplicate_authorization_evidence(self, dedup_identity, authorize_duplicate)
    }
}

/// Ordered admission decisions for a batch of fingerprints.
///
/// The order of [`Self::admissions`] matches the order of fingerprints passed
/// to the storage backend. This keeps the shared contract useful for callers
/// that need to pair admissions back to successor candidates while still
/// exposing aggregate counts for prepared batch evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FingerprintBatchAdmission {
    admissions: Vec<FingerprintAdmission>,
}

impl FingerprintBatchAdmission {
    /// Build a batch admission summary from ordered scalar admissions.
    #[must_use]
    pub fn from_admissions(admissions: Vec<FingerprintAdmission>) -> Self {
        Self { admissions }
    }

    /// Ordered scalar admissions for the batch.
    #[must_use]
    pub fn admissions(&self) -> &[FingerprintAdmission] {
        &self.admissions
    }

    /// Consume the summary and return ordered scalar admissions.
    #[must_use]
    pub fn into_admissions(self) -> Vec<FingerprintAdmission> {
        self.admissions
    }

    /// Number of fingerprints admitted or suppressed by this batch.
    #[must_use]
    pub fn attempted_count(&self) -> usize {
        self.admissions.len()
    }

    /// Number of fingerprints newly inserted by this batch.
    #[must_use]
    pub fn inserted_count(&self) -> usize {
        self.admissions
            .iter()
            .filter(|admission| admission.is_new())
            .count()
    }

    /// Number of fingerprints suppressed as duplicates by this batch.
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.admissions
            .iter()
            .filter(|admission| admission.is_duplicate())
            .count()
    }

    /// Whether this batch contains no admissions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.admissions.is_empty()
    }
}

fn require_shared_dedup_admission(
    dedup_identity: &SharedDedupIdentity,
) -> Result<(), StorageFault> {
    dedup_identity
        .require_fail_closed()
        .map_err(shared_dedup_rejection_fault)
}

fn enforce_duplicate_authorization(
    admission: FingerprintAdmission,
    dedup_identity: &SharedDedupIdentity,
    authorize_duplicate: &mut dyn FnMut(SharedCollisionPolicy) -> Result<bool, StorageFault>,
) -> Result<FingerprintAdmission, StorageFault> {
    if admission.is_duplicate() && !authorize_duplicate(dedup_identity.collision_policy)? {
        return Err(shared_collision_rejection_fault(
            dedup_identity,
            SharedDuplicateAuthorization::Unconfirmed,
        ));
    }
    Ok(admission)
}

fn enforce_duplicate_authorization_evidence(
    admission: FingerprintAdmission,
    dedup_identity: &SharedDedupIdentity,
    authorize_duplicate: &mut dyn FnMut(
        SharedCollisionPolicy,
    ) -> Result<SharedDuplicateAuthorization, StorageFault>,
) -> Result<FingerprintAdmission, StorageFault> {
    if admission.is_duplicate() {
        let authorization = authorize_duplicate(dedup_identity.collision_policy)?;
        if !dedup_identity
            .collision_policy
            .authorizes_duplicate(authorization)
        {
            return Err(shared_collision_rejection_fault(
                dedup_identity,
                authorization,
            ));
        }
    }
    Ok(admission)
}

fn shared_dedup_rejection_fault(rejection: SharedFingerprintIdentityRejection) -> StorageFault {
    StorageFault::new(
        SHARED_DEDUP_ADMISSION_BACKEND,
        SHARED_DEDUP_ADMISSION_OPERATION,
        format!("{}: {}", rejection.reason_code, rejection.detail),
    )
}

fn shared_collision_rejection_fault(
    dedup_identity: &SharedDedupIdentity,
    authorization: SharedDuplicateAuthorization,
) -> StorageFault {
    StorageFault::new(
        SHARED_DEDUP_ADMISSION_BACKEND,
        SHARED_DEDUP_ADMISSION_OPERATION,
        format!(
            "{}: collision_policy={} duplicate_authorization={} dedup_identity={}",
            SHARED_DEDUP_COLLISION_REJECTION,
            dedup_identity.collision_policy.code(),
            authorization.code(),
            dedup_identity.dedup_identity()
        ),
    )
}

/// Typed result for fingerprint lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LookupOutcome {
    /// Fingerprint is present.
    Present,
    /// Fingerprint is absent.
    Absent,
    /// Storage subsystem fault occurred.
    StorageFault(StorageFault),
}

/// Capacity status for fingerprint storage.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CapacityStatus {
    /// Normal operation.
    Normal,
    /// Approaching capacity.
    Warning {
        /// Current resident count.
        count: usize,
        /// Maximum capacity.
        capacity: usize,
        /// Usage fraction in the range `[0.0, 1.0]`.
        usage: f64,
    },
    /// Near or at capacity.
    Critical {
        /// Current resident count.
        count: usize,
        /// Maximum capacity.
        capacity: usize,
        /// Usage fraction in the range `[0.0, 1.0]`.
        usage: f64,
    },
}

/// Storage counters exposed through [`FingerprintSet`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StorageStats {
    /// Fingerprints currently resident in the in-memory tier.
    pub memory_count: usize,
    /// Bytes reserved by the in-memory storage tier.
    pub memory_bytes: usize,
}

/// Trait for deduplication fingerprint sets.
pub trait FingerprintSet<F>: Send + Sync {
    /// Insert a fingerprint with typed outcome.
    fn insert_checked(&self, fingerprint: F) -> InsertOutcome;

    /// Insert a fingerprint that a caller has already observed absent.
    ///
    /// This is still an authoritative test-and-set operation: implementations
    /// must return [`InsertOutcome::AlreadyPresent`] if another writer inserted
    /// the fingerprint after the caller's precheck. Backends that cannot exploit
    /// the precheck can safely use the default scalar insert path.
    fn insert_prechecked_absent_checked(&self, fingerprint: F) -> InsertOutcome {
        self.insert_checked(fingerprint)
    }

    /// Insert a fingerprint and return the shared dedup admission decision.
    fn admit_fingerprint(&self, fingerprint: F) -> Result<FingerprintAdmission, StorageFault> {
        self.insert_checked(fingerprint).into_admission()
    }

    /// Admit an ordered batch of fingerprints.
    ///
    /// The default implementation preserves scalar semantics by admitting each
    /// fingerprint in order and stopping at the first storage fault.
    fn admit_fingerprint_batch(
        &self,
        fingerprints: &[F],
    ) -> Result<FingerprintBatchAdmission, StorageFault>
    where
        F: Copy,
    {
        let mut admissions = Vec::with_capacity(fingerprints.len());
        for &fingerprint in fingerprints {
            admissions.push(self.admit_fingerprint(fingerprint)?);
        }
        Ok(FingerprintBatchAdmission::from_admissions(admissions))
    }

    /// Admit a fingerprint under a validated shared dedup collision policy.
    ///
    /// The dedup identity is validated before storage mutation. On duplicate
    /// admission, `authorize_duplicate` must confirm the resident and candidate
    /// canonical payloads are equal or that the policy-required proof/witness is
    /// present. Returning `false` fails closed with a structured storage fault
    /// instead of silently suppressing a possible collision.
    fn admit_fingerprint_with_collision_check(
        &self,
        fingerprint: F,
        dedup_identity: &SharedDedupIdentity,
        authorize_duplicate: &mut dyn FnMut(SharedCollisionPolicy) -> Result<bool, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault> {
        require_shared_dedup_admission(dedup_identity)?;
        let admission = self.admit_fingerprint(fingerprint)?;
        enforce_duplicate_authorization(admission, dedup_identity, authorize_duplicate)
    }

    /// Admit a fingerprint with typed duplicate authorization evidence.
    ///
    /// State vectors, marking vectors, and register vectors should return
    /// [`SharedDuplicateAuthorization::CanonicalPayloadEquality`] only after
    /// comparing canonical resident/candidate payload bytes. Replay/proof lanes
    /// should return [`SharedDuplicateAuthorization::ProofWitness`] only after
    /// validating the policy-required proof, witness, certificate, or receipt.
    fn admit_fingerprint_with_duplicate_authorization(
        &self,
        fingerprint: F,
        dedup_identity: &SharedDedupIdentity,
        authorize_duplicate: &mut dyn FnMut(
            SharedCollisionPolicy,
        )
            -> Result<SharedDuplicateAuthorization, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault> {
        require_shared_dedup_admission(dedup_identity)?;
        let admission = self.admit_fingerprint(fingerprint)?;
        enforce_duplicate_authorization_evidence(admission, dedup_identity, authorize_duplicate)
    }

    /// Check whether a fingerprint is present with typed outcome.
    fn contains_checked(&self, fingerprint: F) -> LookupOutcome;

    /// Return the number of fingerprints.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if any insert errors have occurred (e.g., table overflow).
    ///
    /// When this returns true, some fingerprints may not have been stored
    /// and the exploration may be incomplete.
    ///
    /// Default implementation returns false (no errors possible).
    fn has_errors(&self) -> bool {
        false
    }

    /// Get the count of dropped fingerprints due to errors.
    ///
    /// If this is non-zero, exploration results are unreliable.
    ///
    /// Default implementation returns 0 (no errors possible).
    fn dropped_count(&self) -> usize {
        0
    }

    /// Check the current capacity status.
    fn capacity_status(&self) -> CapacityStatus {
        CapacityStatus::Normal
    }

    /// Return backend storage counters for observability.
    fn stats(&self) -> StorageStats {
        StorageStats::default()
    }
}

/// Single-threaded fingerprint storage backed by `FxHashSet`.
///
/// This gives sequential engines the same typed admission/lookup contract used
/// by shared parallel storage without paying an interior-locking cost on the
/// hot path.
pub struct LocalFingerprintSet<F> {
    inner: FxHashSet<F>,
}

impl<F> LocalFingerprintSet<F>
where
    F: Copy + Eq + Hash,
{
    /// Create an empty local fingerprint set.
    pub fn new() -> Self {
        Self {
            inner: FxHashSet::default(),
        }
    }

    /// Create an empty local fingerprint set with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Insert a fingerprint with typed outcome.
    pub fn insert_checked(&mut self, fingerprint: F) -> InsertOutcome {
        if self.inner.insert(fingerprint) {
            InsertOutcome::Inserted
        } else {
            InsertOutcome::AlreadyPresent
        }
    }

    /// Insert a fingerprint and return the shared dedup admission decision.
    pub fn admit_fingerprint(&mut self, fingerprint: F) -> FingerprintAdmission {
        match self.insert_checked(fingerprint) {
            InsertOutcome::Inserted => FingerprintAdmission::New,
            InsertOutcome::AlreadyPresent => FingerprintAdmission::Duplicate,
            InsertOutcome::StorageFault(_) => unreachable!("local fingerprint set cannot fault"),
        }
    }

    /// Admit an ordered batch of fingerprints.
    pub fn admit_fingerprint_batch(&mut self, fingerprints: &[F]) -> FingerprintBatchAdmission {
        let mut admissions = Vec::with_capacity(fingerprints.len());
        for &fingerprint in fingerprints {
            admissions.push(self.admit_fingerprint(fingerprint));
        }
        FingerprintBatchAdmission::from_admissions(admissions)
    }

    /// Admit a fingerprint under a validated shared dedup collision policy.
    ///
    /// This is the single-threaded counterpart to
    /// [`FingerprintSet::admit_fingerprint_with_collision_check`].
    pub fn admit_fingerprint_with_collision_check(
        &mut self,
        fingerprint: F,
        dedup_identity: &SharedDedupIdentity,
        mut authorize_duplicate: impl FnMut(SharedCollisionPolicy) -> Result<bool, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault> {
        require_shared_dedup_admission(dedup_identity)?;
        let admission = self.admit_fingerprint(fingerprint);
        enforce_duplicate_authorization(admission, dedup_identity, &mut authorize_duplicate)
    }

    /// Admit a fingerprint with typed duplicate authorization evidence.
    ///
    /// This is the single-threaded counterpart to
    /// [`FingerprintSet::admit_fingerprint_with_duplicate_authorization`].
    pub fn admit_fingerprint_with_duplicate_authorization(
        &mut self,
        fingerprint: F,
        dedup_identity: &SharedDedupIdentity,
        mut authorize_duplicate: impl FnMut(
            SharedCollisionPolicy,
        )
            -> Result<SharedDuplicateAuthorization, StorageFault>,
    ) -> Result<FingerprintAdmission, StorageFault> {
        require_shared_dedup_admission(dedup_identity)?;
        let admission = self.admit_fingerprint(fingerprint);
        enforce_duplicate_authorization_evidence(
            admission,
            dedup_identity,
            &mut authorize_duplicate,
        )
    }

    /// Check whether a fingerprint is present with typed outcome.
    pub fn contains_checked(&self, fingerprint: &F) -> LookupOutcome {
        if self.inner.contains(fingerprint) {
            LookupOutcome::Present
        } else {
            LookupOutcome::Absent
        }
    }

    /// Return the number of resident fingerprints.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return a copy of resident fingerprints for checkpoint persistence.
    pub fn collect_fingerprints(&self) -> Vec<F> {
        self.inner.iter().copied().collect()
    }

    /// Return backend storage counters for observability.
    pub fn stats(&self) -> StorageStats {
        StorageStats {
            memory_count: self.inner.len(),
            memory_bytes: self
                .inner
                .capacity()
                .saturating_mul(std::mem::size_of::<F>()),
        }
    }
}

impl<F> Default for LocalFingerprintSet<F>
where
    F: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory fingerprint storage backed by `FxHashSet`.
pub struct InMemoryFingerprintSet<F> {
    inner: RwLock<FxHashSet<F>>,
}

impl<F> InMemoryFingerprintSet<F>
where
    F: Copy + Eq + Hash,
{
    /// Create an empty set.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(FxHashSet::default()),
        }
    }

    /// Create an empty set with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(FxHashSet::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
        }
    }
}

impl<F> Default for InMemoryFingerprintSet<F>
where
    F: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> FingerprintSet<F> for InMemoryFingerprintSet<F>
where
    F: Copy + Eq + Hash + Send + Sync,
{
    fn insert_checked(&self, fingerprint: F) -> InsertOutcome {
        if self.inner.write().insert(fingerprint) {
            InsertOutcome::Inserted
        } else {
            InsertOutcome::AlreadyPresent
        }
    }

    fn admit_fingerprint_batch(
        &self,
        fingerprints: &[F],
    ) -> Result<FingerprintBatchAdmission, StorageFault> {
        let mut guard = self.inner.write();
        let mut admissions = Vec::with_capacity(fingerprints.len());
        for &fingerprint in fingerprints {
            let admission = if guard.insert(fingerprint) {
                FingerprintAdmission::New
            } else {
                FingerprintAdmission::Duplicate
            };
            admissions.push(admission);
        }
        Ok(FingerprintBatchAdmission::from_admissions(admissions))
    }

    fn contains_checked(&self, fingerprint: F) -> LookupOutcome {
        if self.inner.read().contains(&fingerprint) {
            LookupOutcome::Present
        } else {
            LookupOutcome::Absent
        }
    }

    fn len(&self) -> usize {
        self.inner.read().len()
    }

    fn stats(&self) -> StorageStats {
        let guard = self.inner.read();
        StorageStats {
            memory_count: guard.len(),
            memory_bytes: guard.capacity() * std::mem::size_of::<F>(),
        }
    }
}

/// Sharded in-memory fingerprint storage tuned for parallel insert-heavy workloads.
pub struct ShardedFingerprintSet<F> {
    shards: Vec<RwLock<FxHashSet<F>>>,
}

impl<F> ShardedFingerprintSet<F>
where
    F: Copy + Eq + Hash,
{
    /// Create a sharded set with an explicit shard count.
    #[must_use]
    pub fn with_shard_count(shard_count: usize) -> Self {
        let shard_count = shard_count.max(1).next_power_of_two();
        let mut shards = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shards.push(RwLock::new(FxHashSet::default()));
        }
        Self { shards }
    }

    fn shard_index(&self, fingerprint: &F) -> usize {
        let mut hasher = FxHasher::default();
        fingerprint.hash(&mut hasher);
        (hasher.finish() as usize) & (self.shards.len() - 1)
    }
}

impl<F> Default for ShardedFingerprintSet<F>
where
    F: Copy + Eq + Hash,
{
    fn default() -> Self {
        let parallelism = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(4);
        Self::with_shard_count((parallelism * 4).clamp(4, 256))
    }
}

impl<F> FingerprintSet<F> for ShardedFingerprintSet<F>
where
    F: Copy + Eq + Hash + Send + Sync,
{
    fn insert_checked(&self, fingerprint: F) -> InsertOutcome {
        let shard = &self.shards[self.shard_index(&fingerprint)];
        if shard.write().insert(fingerprint) {
            InsertOutcome::Inserted
        } else {
            InsertOutcome::AlreadyPresent
        }
    }

    fn admit_fingerprint_batch(
        &self,
        fingerprints: &[F],
    ) -> Result<FingerprintBatchAdmission, StorageFault> {
        let mut indices_by_shard = vec![Vec::new(); self.shards.len()];
        for (index, fingerprint) in fingerprints.iter().enumerate() {
            indices_by_shard[self.shard_index(fingerprint)].push(index);
        }

        let mut admissions = vec![FingerprintAdmission::Duplicate; fingerprints.len()];
        for (shard_index, indices) in indices_by_shard.into_iter().enumerate() {
            if indices.is_empty() {
                continue;
            }

            let mut shard = self.shards[shard_index].write();
            for index in indices {
                admissions[index] = if shard.insert(fingerprints[index]) {
                    FingerprintAdmission::New
                } else {
                    FingerprintAdmission::Duplicate
                };
            }
        }

        Ok(FingerprintBatchAdmission::from_admissions(admissions))
    }

    fn contains_checked(&self, fingerprint: F) -> LookupOutcome {
        let shard = &self.shards[self.shard_index(&fingerprint)];
        if shard.read().contains(&fingerprint) {
            LookupOutcome::Present
        } else {
            LookupOutcome::Absent
        }
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().len()).sum()
    }

    fn stats(&self) -> StorageStats {
        let mut stats = StorageStats::default();
        for shard in &self.shards {
            let guard = shard.read();
            stats.memory_count += guard.len();
            stats.memory_bytes += guard.capacity() * std::mem::size_of::<F>();
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fingerprint_identity::{
            SharedFingerprintAlgorithm, SharedFingerprintIdentity, SharedFingerprintValueKind,
        },
        setup_trace::SetupTraceLaneKind,
    };

    fn shared_state_fingerprint() -> SharedFingerprintIdentity {
        SharedFingerprintIdentity::new(
            "runtime state",
            SharedFingerprintAlgorithm::Xxh3U64,
            SharedFingerprintValueKind::State,
            "runtime-state-v1",
            "runtime-state",
            64,
        )
        .with_canonical_domain("runtime-state-domain", "v1")
    }

    fn shared_dedup(collision_policy: SharedCollisionPolicy) -> SharedDedupIdentity {
        SharedDedupIdentity::new(
            "runtime dedup",
            shared_state_fingerprint(),
            crate::fingerprint_identity::SharedDedupScope::StateSpace,
            crate::fingerprint_identity::SharedDedupStorageKind::InMemory,
            SetupTraceLaneKind::ExplicitState,
        )
        .with_collision_policy(collision_policy)
    }

    struct FaultyFingerprintSet;

    impl FingerprintSet<u8> for FaultyFingerprintSet {
        fn insert_checked(&self, _fingerprint: u8) -> InsertOutcome {
            InsertOutcome::StorageFault(StorageFault::new("test", "insert", "synthetic fault"))
        }

        fn contains_checked(&self, _fingerprint: u8) -> LookupOutcome {
            LookupOutcome::Absent
        }

        fn len(&self) -> usize {
            0
        }
    }

    #[test]
    fn insert_outcome_into_admission_is_frontend_neutral() {
        assert_eq!(
            InsertOutcome::Inserted.into_admission(),
            Ok(FingerprintAdmission::New)
        );
        assert_eq!(
            InsertOutcome::AlreadyPresent.into_admission(),
            Ok(FingerprintAdmission::Duplicate)
        );

        let fault = StorageFault::new("test", "insert", "synthetic fault");
        assert_eq!(
            InsertOutcome::StorageFault(fault.clone()).into_admission(),
            Err(fault)
        );
    }

    #[test]
    fn fingerprint_set_admit_fingerprint_tracks_new_and_duplicate() {
        let set = InMemoryFingerprintSet::default();

        let first = set
            .admit_fingerprint(7u8)
            .expect("first admission should succeed");
        let second = set
            .admit_fingerprint(7u8)
            .expect("duplicate admission should succeed");

        assert!(first.is_new());
        assert!(second.is_duplicate());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn fingerprint_set_batch_admission_preserves_order_and_counts() {
        let set = InMemoryFingerprintSet::default();

        let batch = set
            .admit_fingerprint_batch(&[7u8, 8, 7, 9])
            .expect("batch admission should succeed");

        assert_eq!(
            batch.admissions(),
            &[
                FingerprintAdmission::New,
                FingerprintAdmission::New,
                FingerprintAdmission::Duplicate,
                FingerprintAdmission::New,
            ]
        );
        assert_eq!(batch.attempted_count(), 4);
        assert_eq!(batch.inserted_count(), 3);
        assert_eq!(batch.duplicate_count(), 1);
        assert_eq!(set.len(), 3);

        let empty = set
            .admit_fingerprint_batch(&[])
            .expect("empty batch admission should succeed");
        assert!(empty.is_empty());
        assert_eq!(empty.inserted_count(), 0);
    }

    #[test]
    fn sharded_fingerprint_set_batch_admission_preserves_order_and_counts() {
        let set = ShardedFingerprintSet::with_shard_count(4);
        assert_eq!(
            set.admit_fingerprint(9u8),
            Ok(FingerprintAdmission::New),
            "pre-existing fingerprint setup should succeed"
        );

        let batch = set
            .admit_fingerprint_batch(&[7u8, 8, 7, 9, 8, 10])
            .expect("sharded batch admission should succeed");

        assert_eq!(
            batch.admissions(),
            &[
                FingerprintAdmission::New,
                FingerprintAdmission::New,
                FingerprintAdmission::Duplicate,
                FingerprintAdmission::Duplicate,
                FingerprintAdmission::Duplicate,
                FingerprintAdmission::New,
            ]
        );
        assert_eq!(batch.attempted_count(), 6);
        assert_eq!(batch.inserted_count(), 3);
        assert_eq!(batch.duplicate_count(), 3);
        assert_eq!(set.len(), 4);

        let empty = set
            .admit_fingerprint_batch(&[])
            .expect("empty sharded batch admission should succeed");
        assert!(empty.is_empty());
    }

    #[test]
    fn fingerprint_set_batch_admission_preserves_storage_fault() {
        let error = FaultyFingerprintSet
            .admit_fingerprint_batch(&[7u8, 8])
            .expect_err("storage fault should be preserved");

        assert_eq!(error.backend, "test");
        assert_eq!(error.operation, "insert");
        assert_eq!(error.detail, "synthetic fault");
    }

    #[test]
    fn fingerprint_set_admit_fingerprint_preserves_storage_fault() {
        let error = FaultyFingerprintSet
            .admit_fingerprint(7u8)
            .expect_err("storage fault should be preserved");

        assert_eq!(error.backend, "test");
        assert_eq!(error.operation, "insert");
        assert_eq!(error.detail, "synthetic fault");
    }

    #[test]
    fn fingerprint_set_collision_checked_admission_requires_duplicate_authorization() {
        let dedup = shared_dedup(SharedCollisionPolicy::CanonicalPayloadEquality);
        let set = InMemoryFingerprintSet::default();
        let mut new_authorizer = |_policy| -> Result<bool, StorageFault> {
            panic!("new admission should not require duplicate authorization")
        };

        let first = set
            .admit_fingerprint_with_collision_check(7u8, &dedup, &mut new_authorizer)
            .expect("first admission should succeed");
        assert_eq!(first, FingerprintAdmission::New);

        let mut duplicate_authorizer = |policy| {
            assert_eq!(policy, SharedCollisionPolicy::CanonicalPayloadEquality);
            Ok(true)
        };
        let second = set
            .admit_fingerprint_with_collision_check(7u8, &dedup, &mut duplicate_authorizer)
            .expect("payload-confirmed duplicate should succeed");
        assert_eq!(second, FingerprintAdmission::Duplicate);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn fingerprint_set_collision_checked_admission_rejects_unchecked_before_insert() {
        let dedup = shared_dedup(SharedCollisionPolicy::Unchecked);
        let set = InMemoryFingerprintSet::default();
        let mut authorizer = |_policy| -> Result<bool, StorageFault> {
            panic!("unchecked policy should reject before duplicate authorization")
        };

        let error = set
            .admit_fingerprint_with_collision_check(7u8, &dedup, &mut authorizer)
            .expect_err("unchecked collision policy must fail closed");

        assert_eq!(error.backend, SHARED_DEDUP_ADMISSION_BACKEND);
        assert_eq!(error.operation, SHARED_DEDUP_ADMISSION_OPERATION);
        assert!(error.detail.contains("non_fail_closed_collision_policy"));
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn fingerprint_set_collision_checked_admission_rejects_unauthorized_duplicate() {
        let dedup = shared_dedup(SharedCollisionPolicy::RejectOnCollision);
        let set = InMemoryFingerprintSet::default();
        let mut first_authorizer = |_policy| -> Result<bool, StorageFault> {
            panic!("new admission should not require duplicate authorization")
        };
        assert_eq!(
            set.admit_fingerprint_with_collision_check(7u8, &dedup, &mut first_authorizer),
            Ok(FingerprintAdmission::New)
        );

        let mut reject_duplicate = |policy| {
            assert_eq!(policy, SharedCollisionPolicy::RejectOnCollision);
            Ok(false)
        };
        let error = set
            .admit_fingerprint_with_collision_check(7u8, &dedup, &mut reject_duplicate)
            .expect_err("suspected collision must fail closed");

        assert_eq!(error.backend, SHARED_DEDUP_ADMISSION_BACKEND);
        assert_eq!(error.operation, SHARED_DEDUP_ADMISSION_OPERATION);
        assert!(error.detail.contains(SHARED_DEDUP_COLLISION_REJECTION));
        assert!(error
            .detail
            .contains("collision_policy=reject_on_collision"));
        assert!(error.detail.contains("duplicate_authorization=unconfirmed"));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn fingerprint_set_duplicate_authorization_is_policy_specific() {
        let proof_required = shared_dedup(SharedCollisionPolicy::ProofWitnessRequired);
        let set = InMemoryFingerprintSet::default();
        let mut first_authorizer = |_policy| -> Result<SharedDuplicateAuthorization, StorageFault> {
            panic!("new admission should not require duplicate authorization")
        };
        assert_eq!(
            set.admit_fingerprint_with_duplicate_authorization(
                7u8,
                &proof_required,
                &mut first_authorizer
            ),
            Ok(FingerprintAdmission::New)
        );

        let mut payload_authorizer = |policy| {
            assert_eq!(policy, SharedCollisionPolicy::ProofWitnessRequired);
            Ok(SharedDuplicateAuthorization::CanonicalPayloadEquality)
        };
        let error = set
            .admit_fingerprint_with_duplicate_authorization(
                7u8,
                &proof_required,
                &mut payload_authorizer,
            )
            .expect_err("payload equality alone must not satisfy proof/witness policy");
        assert!(error
            .detail
            .contains("collision_policy=proof_witness_required"));
        assert!(error
            .detail
            .contains("duplicate_authorization=canonical_payload_equality"));

        let mut proof_authorizer = |policy| {
            assert_eq!(policy, SharedCollisionPolicy::ProofWitnessRequired);
            Ok(SharedDuplicateAuthorization::ProofWitness)
        };
        assert_eq!(
            set.admit_fingerprint_with_duplicate_authorization(
                7u8,
                &proof_required,
                &mut proof_authorizer
            ),
            Ok(FingerprintAdmission::Duplicate)
        );
    }

    #[test]
    fn local_fingerprint_set_admit_fingerprint_tracks_new_and_duplicate() {
        let mut set = LocalFingerprintSet::new();

        assert_eq!(set.admit_fingerprint(7u8), FingerprintAdmission::New);
        assert_eq!(set.admit_fingerprint(7u8), FingerprintAdmission::Duplicate);
        assert_eq!(set.len(), 1);
        assert_eq!(set.contains_checked(&7u8), LookupOutcome::Present);
        assert_eq!(set.contains_checked(&8u8), LookupOutcome::Absent);
    }

    #[test]
    fn local_fingerprint_set_batch_admission_preserves_order_and_counts() {
        let mut set = LocalFingerprintSet::new();

        let batch = set.admit_fingerprint_batch(&[2u8, 3, 2, 4]);

        assert_eq!(
            batch.into_admissions(),
            vec![
                FingerprintAdmission::New,
                FingerprintAdmission::New,
                FingerprintAdmission::Duplicate,
                FingerprintAdmission::New,
            ]
        );
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn local_fingerprint_set_collects_resident_fingerprints() {
        let mut set = LocalFingerprintSet::with_capacity(4);
        assert!(set.is_empty());
        assert_eq!(set.stats().memory_count, 0);

        assert_eq!(set.admit_fingerprint(2u8), FingerprintAdmission::New);
        assert_eq!(set.admit_fingerprint(3u8), FingerprintAdmission::New);

        let mut fingerprints = set.collect_fingerprints();
        fingerprints.sort_unstable();
        assert_eq!(fingerprints, vec![2, 3]);
        assert_eq!(set.stats().memory_count, 2);
        assert!(set.stats().memory_bytes >= 2);
    }

    #[test]
    fn local_fingerprint_set_collision_checked_admission_rejects_unauthorized_duplicate() {
        let dedup = shared_dedup(SharedCollisionPolicy::RejectOnCollision);
        let mut set = LocalFingerprintSet::new();

        assert_eq!(
            set.admit_fingerprint_with_collision_check(7u8, &dedup, |_policy| {
                panic!("new admission should not require duplicate authorization")
            }),
            Ok(FingerprintAdmission::New)
        );

        let error = set
            .admit_fingerprint_with_collision_check(7u8, &dedup, |policy| {
                assert_eq!(policy, SharedCollisionPolicy::RejectOnCollision);
                Ok(false)
            })
            .expect_err("suspected local collision must fail closed");

        assert!(error.detail.contains(SHARED_DEDUP_COLLISION_REJECTION));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn local_fingerprint_set_uses_shared_duplicate_authorization() {
        let dedup = shared_dedup(SharedCollisionPolicy::CanonicalPayloadEquality);
        let mut set = LocalFingerprintSet::new();

        assert_eq!(
            set.admit_fingerprint_with_duplicate_authorization(7u8, &dedup, |_policy| {
                panic!("new admission should not require duplicate authorization")
            }),
            Ok(FingerprintAdmission::New)
        );
        assert_eq!(
            set.admit_fingerprint_with_duplicate_authorization(7u8, &dedup, |_policy| {
                Ok(SharedDuplicateAuthorization::canonical_payload_equality(
                    true,
                ))
            }),
            Ok(FingerprintAdmission::Duplicate)
        );
    }
}
