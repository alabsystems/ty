// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Storage-mode abstractions for BFS exploration.
//!
//! Part of #2351: extracted from `run_bfs_common.rs`.
//! Part of #3436: split into submodules (contract, full-state, fp-only, tests).
//!
//! Defines the `BfsStorage` trait and its two implementations:
//! - `FullStateStorage`: HashMap-backed with trace reconstruction.
//! - `FingerprintOnlyStorage`: fingerprint-set-backed with disk trace.
//!
//! The BFS loop in `bfs::engine` is generic over this trait, enabling a single
//! loop implementation for both storage modes (Part of #2133).

mod fingerprint_only;
mod full_state;
#[cfg(test)]
mod tests;

use super::super::frontier::BfsFrontier;
use super::super::{ArrayState, CheckResult, Fingerprint, ModelChecker, State, VecDeque};

// Re-export concrete backends for sibling modules.
pub(in crate::check::model_checker) use self::fingerprint_only::{
    FingerprintOnlyStorage, NoTraceQueueEntry,
};
pub(in crate::check::model_checker) use self::full_state::FullStateStorage;

/// Abstracts storage-mode differences between full-state and fingerprint-only BFS.
///
/// This trait enables a single generic BFS loop to work with both storage modes,
/// eliminating duplicated loop logic that has already drifted in `use_diffs` conditions,
/// `print_symmetry_stats()` calls, and state-limit handling patterns.
///
/// Implementations are zero-cost: the generic function is monomorphized for each
/// storage type, producing the same code as the hand-written variants.
pub(in crate::check::model_checker) trait BfsStorage {
    /// The type held in the BFS queue.
    type QueueEntry;

    /// Release backend allocations used only while BFS is running.
    ///
    /// The default is appropriate for backends whose state lives entirely on
    /// [`ModelChecker`]. Fingerprint-only storage overrides this to discard its
    /// local collision witnesses and batching scratch before post-BFS liveness.
    /// Call only after the worker explicitly reports frontier exhaustion and
    /// the run was not truncated by a depth limit.
    fn release_after_complete_bfs(&mut self) {}

    /// Estimated bytes retained by exact payload witnesses owned by this
    /// storage mode.
    ///
    /// The default covers full-state storage, which has no separate payload
    /// witness container. Fingerprint-only storage overrides this so periodic
    /// memory accounting and hard limits include its collision witnesses. The
    /// estimate may be a lower bound for recursively heap-owned values.
    fn payload_witness_memory_bytes(&self) -> usize {
        0
    }

    /// Extract the current state for processing from a queue entry.
    ///
    /// Returns `Ok(Some((fingerprint, owned_state, depth)))` on success,
    /// `Ok(None)` for phantom dequeues (full-state: fp not found in HashMap),
    /// or `Err(CheckResult)` on error (materialization/fingerprint failure).
    #[allow(clippy::result_large_err)]
    fn dequeue(
        &mut self,
        entry: Self::QueueEntry,
        mc: &mut ModelChecker,
    ) -> Result<Option<(Fingerprint, ArrayState, usize)>, CheckResult>;

    /// Return/restore the current state after processing.
    ///
    /// Full-state: re-inserts the ArrayState into the seen HashMap.
    /// No-trace: no-op (state is not stored in a HashMap).
    fn return_current(&mut self, fp: Fingerprint, state: ArrayState, mc: &mut ModelChecker);

    /// Atomic test-and-set admission: check if seen, insert if new, and create queue entry.
    ///
    /// Part of #2881 Step 2: combines the dedup check with state admission in a
    /// single operation, matching TLC's `FPSet.put()` pattern. Returns `Ok(None)`
    /// if the state was already seen (no bookkeeping performed), or
    /// `Ok(Some(entry))` if newly admitted.
    ///
    /// This eliminates the need for a separate `is_state_seen_checked` call before
    /// admission, reducing lock acquisitions from 3 to 2 per new state.
    ///
    /// Ownership semantics differ by mode:
    /// - Full-state: moves state into HashMap, returns fingerprint as queue entry.
    /// - No-trace: records fingerprint in FPSet, returns Owned{state, fp} queue entry.
    #[allow(clippy::result_large_err)]
    fn admit_successor(
        &mut self,
        fp: Fingerprint,
        state: ArrayState,
        parent_fp: Option<Fingerprint>,
        current: Option<(Fingerprint, &ArrayState)>,
        depth: usize,
        mc: &mut ModelChecker,
    ) -> Result<Option<Self::QueueEntry>, CheckResult>;

    /// Admit an ordered batch into caller-owned result scratch.
    ///
    /// Implementations must preserve scalar side-effect order for the attempted
    /// prefix: entries are aligned with input candidates, and a fault is reported
    /// only after entries for earlier successful admissions have been returned.
    /// Clears and reuses the result vector instead of allocating per flush.
    #[allow(clippy::result_large_err)]
    fn admit_successor_batch_into(
        &mut self,
        candidates: &mut Vec<BfsAdmissionCandidate>,
        current: Option<(Fingerprint, &ArrayState)>,
        result: &mut BfsBatchAdmissionResult<Self::QueueEntry>,
        mc: &mut ModelChecker,
    ) {
        result.clear_for_capacity(candidates.len());
        for candidate in candidates.drain(..) {
            let BfsAdmissionCandidate {
                fp,
                state,
                parent_fp,
                depth,
            } = candidate;
            let current = current.filter(|(current_fp, _)| Some(*current_fp) == parent_fp);
            let entry = match self.admit_successor(fp, state, parent_fp, current, depth, mc) {
                Ok(entry) => entry,
                Err(error) => {
                    result.fault = Some(error);
                    break;
                }
            };
            result.entries.push(BfsAdmittedEntry { depth, entry });
        }
    }

    /// Validate that a read-only prefilter hit is a real duplicate, not an
    /// unproven fingerprint collision.
    #[allow(clippy::result_large_err)]
    fn enforce_seen_successor_duplicate(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &mut ModelChecker,
    ) -> Result<(), CheckResult>;

    /// Confirm an already-seen virtual `base + changes` payload without
    /// materializing a complete successor state.
    ///
    /// `Some(true)` is an exact duplicate proof and may be used to skip full
    /// ArrayState construction. `Some(false)` and `None` are both non-proofs;
    /// callers must preserve the existing materialize-and-authorize path. The
    /// default keeps full-state and unsupported storage modes unchanged.
    fn confirm_seen_array_diff_duplicate(
        &mut self,
        _fp: Fingerprint,
        _base: &ArrayState,
        _changes: &[(crate::var_index::VarIndex, crate::Value)],
    ) -> Option<bool> {
        None
    }

    /// Whether diff-based successor generation is available in this mode.
    ///
    /// Full-state: true when no VIEW and no symmetry.
    /// No-trace: additionally requires no liveness caching (diff path doesn't
    /// record successor witnesses needed for liveness).
    fn use_diffs(&self, mc: &ModelChecker) -> bool;

    /// Build a checkpoint frontier from the current state and queue contents.
    fn checkpoint_frontier(
        &self,
        current: &ArrayState,
        queue: &impl BfsFrontier<Entry = Self::QueueEntry>,
        registry: &crate::var_index::VarRegistry,
        mc: &mut ModelChecker,
    ) -> VecDeque<State>;

    /// Cache successor fingerprints for liveness checking (diff path).
    ///
    /// Called after processing all diffs. Full-state: caches fps.
    /// No-trace: no-op (liveness excluded from diff path via `use_diffs`).
    fn cache_diff_liveness(
        &self,
        parent_fp: Fingerprint,
        succ_fps: Option<Vec<Fingerprint>>,
        mc: &mut ModelChecker,
    ) -> Result<(), crate::check::CheckError>;

    /// Cache successor info for liveness checking (full successor path).
    ///
    /// Full-state: caches fps + symmetry witness states if applicable.
    /// No-trace: caches fps only.
    fn cache_full_liveness(
        &self,
        parent_fp: Fingerprint,
        successors: &[(ArrayState, Fingerprint)],
        mc: &mut ModelChecker,
    ) -> Result<(), crate::check::CheckError>;
}

/// Successor staged for authoritative BFS admission.
pub(in crate::check::model_checker) struct BfsAdmissionCandidate {
    pub(super) fp: Fingerprint,
    pub(super) state: ArrayState,
    pub(super) parent_fp: Option<Fingerprint>,
    pub(super) depth: usize,
}

/// Queue entry produced for one attempted batch candidate.
pub(in crate::check::model_checker) struct BfsAdmittedEntry<Q> {
    pub(super) depth: usize,
    pub(super) entry: Option<Q>,
}

/// Ordered batch result with scalar-compatible prefix fault semantics.
pub(in crate::check::model_checker) struct BfsBatchAdmissionResult<Q> {
    pub(super) entries: Vec<BfsAdmittedEntry<Q>>,
    pub(super) fault: Option<CheckResult>,
}

impl<Q> BfsBatchAdmissionResult<Q> {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            fault: None,
        }
    }

    pub(super) fn clear_for_capacity(&mut self, capacity: usize) {
        self.entries.clear();
        self.entries.reserve(capacity);
        self.fault = None;
    }
}
