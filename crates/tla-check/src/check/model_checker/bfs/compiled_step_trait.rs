// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Backend-agnostic abstraction over compiled BFS step + fused BFS level.
//!
//! `compiled_bfs_loop.rs` is the single-thread supremacy path: it runs the
//! BFS level loop against a contiguous `FlatBfsFrontier` arena using a
//! compiled action dispatcher + invariant checker + fingerprinter. trust-codegen
//! provides the active native step and fused-level implementations when the
//! `trust_cg` feature is enabled; retired compiled-backend surfaces fail closed.
//!
//! The loop is coded against these traits rather than concrete backend
//! types so that:
//!
//! 1. The loop remains independent of retired backend crates.
//! 2. The trust-codegen backend can plug in compiled BFS step and level implementations
//!    without rewriting the level loop.
//! 3. Future backends (e.g., AOT `.dylib`) implement one trait surface
//!    instead of reimplementing the BFS integration logic.
//!
//! Part of #4171 (end-to-end compiled BFS wiring) and #4267 (retired backend
//! deletion). This file is the boundary between the level loop and whichever
//! backend generated the compiled action/invariant functions.

use crate::compiled_backend_unavailable::{
    CompiledBfsLevel as UnavailableCompiledBfsLevel, CompiledBfsStep as UnavailableCompiledBfsStep,
};

pub(in crate::check::model_checker) use tla_jit_abi::{
    BfsStepError, FlatBfsStepOutput, FlatBfsStepOutputRef,
};

enum CompiledSuccessorArena {
    Owned {
        arena: Vec<i64>,
        parent_indices: Option<Vec<usize>>,
    },
    TrustCg(tla_trust_cg::TrustCgSuccessorArena),
    ReusableTrustCg(ReusableTrustCgSuccessorArena),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::check::model_checker) enum CompiledSuccessorParentIndices<'a> {
    Usize(&'a [usize]),
    U32(&'a [u32]),
}

impl CompiledSuccessorParentIndices<'_> {
    #[must_use]
    pub(in crate::check::model_checker) fn len(&self) -> usize {
        match self {
            Self::Usize(indices) => indices.len(),
            Self::U32(indices) => indices.len(),
        }
    }

    #[must_use]
    pub(in crate::check::model_checker) fn get(&self, idx: usize) -> Option<usize> {
        match self {
            Self::Usize(indices) => indices.get(idx).copied(),
            Self::U32(indices) => indices
                .get(idx)
                .and_then(|parent_idx| usize::try_from(*parent_idx).ok()),
        }
    }

    #[must_use]
    pub(in crate::check::model_checker) fn first_out_of_bounds(
        &self,
        parent_count: usize,
    ) -> Option<(usize, usize)> {
        match self {
            Self::Usize(indices) => indices
                .iter()
                .copied()
                .enumerate()
                .find(|(_, parent_idx)| *parent_idx >= parent_count),
            Self::U32(indices) => {
                indices
                    .iter()
                    .copied()
                    .enumerate()
                    .find_map(|(successor_idx, parent_idx)| {
                        let parent_idx = usize::try_from(parent_idx).ok()?;
                        (parent_idx >= parent_count).then_some((successor_idx, parent_idx))
                    })
            }
        }
    }
}

impl CompiledSuccessorArena {
    fn as_flat_slice(&self) -> &[i64] {
        match self {
            Self::Owned { arena, .. } => arena,
            Self::TrustCg(arena) => arena.states_flat(),
            Self::ReusableTrustCg(arena) => arena.as_arena().states_flat(),
        }
    }

    fn as_flat_slice_mut(&mut self) -> &mut [i64] {
        match self {
            Self::Owned { arena, .. } => arena,
            Self::TrustCg(arena) => arena.states_flat_mut(),
            Self::ReusableTrustCg(arena) => arena.as_arena_mut().states_flat_mut(),
        }
    }

    fn parent_index_at(&self, _successor_idx: usize) -> Option<usize> {
        match self {
            Self::Owned { parent_indices, .. } => {
                parent_indices.as_ref()?.get(_successor_idx).copied()
            }
            Self::TrustCg(arena) => arena
                .parent_indices()
                .get(_successor_idx)
                .and_then(|parent_idx| usize::try_from(*parent_idx).ok()),
            Self::ReusableTrustCg(arena) => arena
                .as_arena()
                .parent_indices()
                .get(_successor_idx)
                .and_then(|parent_idx| usize::try_from(*parent_idx).ok()),
        }
    }

    fn parent_indices_complete(&self, successor_count: usize) -> bool {
        self.parent_indices(successor_count).is_some()
    }

    fn parent_indices(&self, successor_count: usize) -> Option<CompiledSuccessorParentIndices<'_>> {
        if successor_count == 0 {
            return Some(CompiledSuccessorParentIndices::Usize(&[]));
        }
        let indices = match self {
            Self::Owned { parent_indices, .. } => parent_indices
                .as_deref()
                .map(CompiledSuccessorParentIndices::Usize),
            Self::TrustCg(arena) => {
                Some(CompiledSuccessorParentIndices::U32(arena.parent_indices()))
            }
            Self::ReusableTrustCg(arena) => Some(CompiledSuccessorParentIndices::U32(
                arena.as_arena().parent_indices(),
            )),
        }?;
        (indices.len() == successor_count).then_some(indices)
    }

    fn fingerprint_values(&self, successor_count: usize) -> Option<&[u64]> {
        if successor_count == 0 {
            return Some(&[]);
        }
        let fingerprints: &[u64] = match self {
            Self::Owned { .. } => None,
            Self::TrustCg(arena) => Some(arena.successor_fingerprints()),
            Self::ReusableTrustCg(arena) => Some(arena.as_arena().successor_fingerprints()),
        }?;
        (fingerprints.len() == successor_count).then_some(fingerprints)
    }

    fn fingerprint_at(&self, _successor_idx: usize) -> Option<crate::Fingerprint> {
        match self {
            Self::Owned { .. } => None,
            Self::TrustCg(arena) => arena
                .successor_fingerprints()
                .get(_successor_idx)
                .copied()
                .map(crate::Fingerprint),
            Self::ReusableTrustCg(arena) => arena
                .as_arena()
                .successor_fingerprints()
                .get(_successor_idx)
                .copied()
                .map(crate::Fingerprint),
        }
    }

    fn clear_fingerprints(&mut self) {
        match self {
            Self::Owned { .. } => {}
            Self::TrustCg(arena) => arena.clear_successor_fingerprints(),
            Self::ReusableTrustCg(arena) => arena.as_arena_mut().clear_successor_fingerprints(),
        }
    }

    fn parent_indices_slice(&self) -> &[u32] {
        match self {
            Self::Owned { .. } => &[],
            Self::TrustCg(arena) => arena.parent_indices(),
            Self::ReusableTrustCg(arena) => arena.as_arena().parent_indices(),
        }
    }

    fn fingerprints_slice(&self) -> &[u64] {
        match self {
            Self::Owned { .. } => &[],
            Self::TrustCg(arena) => arena.successor_fingerprints(),
            Self::ReusableTrustCg(arena) => arena.as_arena().successor_fingerprints(),
        }
    }
}

impl std::fmt::Debug for CompiledSuccessorArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned {
                arena,
                parent_indices,
            } => f
                .debug_struct("Owned")
                .field("arena", arena)
                .field("parent_indices", parent_indices)
                .finish(),
            Self::TrustCg(arena) => f.debug_tuple("TrustCg").field(arena).finish(),
            Self::ReusableTrustCg(arena) => f.debug_tuple("ReusableTrustCg").field(arena).finish(),
        }
    }
}

impl Clone for CompiledSuccessorArena {
    fn clone(&self) -> Self {
        match self {
            Self::Owned {
                arena,
                parent_indices,
            } => Self::Owned {
                arena: arena.clone(),
                parent_indices: parent_indices.clone(),
            },
            Self::TrustCg(arena) => Self::TrustCg(arena.clone()),
            Self::ReusableTrustCg(arena) => Self::ReusableTrustCg(arena.clone()),
        }
    }
}

impl PartialEq for CompiledSuccessorArena {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Owned {
                    arena: left_arena,
                    parent_indices: left_parent_indices,
                },
                Self::Owned {
                    arena: right_arena,
                    parent_indices: right_parent_indices,
                },
            ) => left_arena == right_arena && left_parent_indices == right_parent_indices,
            (Self::TrustCg(left), Self::TrustCg(right)) => left == right,
            (Self::ReusableTrustCg(left), Self::ReusableTrustCg(right)) => left == right,
            (Self::TrustCg(_), Self::ReusableTrustCg(_))
            | (Self::ReusableTrustCg(_), Self::TrustCg(_)) => {
                self.as_flat_slice() == other.as_flat_slice()
                    && self.parent_indices_slice() == other.parent_indices_slice()
                    && self.fingerprints_slice() == other.fingerprints_slice()
            }
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

impl Eq for CompiledSuccessorArena {}

pub(in crate::check::model_checker) type TrustCgSuccessorArenaPool =
    std::sync::Arc<std::sync::Mutex<Option<tla_trust_cg::TrustCgSuccessorArena>>>;

struct ReusableTrustCgSuccessorArena {
    arena: Option<tla_trust_cg::TrustCgSuccessorArena>,
    pool: TrustCgSuccessorArenaPool,
}

impl ReusableTrustCgSuccessorArena {
    fn new(arena: tla_trust_cg::TrustCgSuccessorArena, pool: TrustCgSuccessorArenaPool) -> Self {
        Self {
            arena: Some(arena),
            pool,
        }
    }

    fn as_arena(&self) -> &tla_trust_cg::TrustCgSuccessorArena {
        self.arena
            .as_ref()
            .expect("reusable trust-codegen successor arena missing before drop")
    }

    fn as_arena_mut(&mut self) -> &mut tla_trust_cg::TrustCgSuccessorArena {
        self.arena
            .as_mut()
            .expect("reusable trust-codegen successor arena missing before drop")
    }
}

impl Drop for ReusableTrustCgSuccessorArena {
    fn drop(&mut self) {
        let Some(mut arena) = self.arena.take() else {
            return;
        };
        arena.clear();
        if let Ok(mut pooled) = self.pool.lock() {
            if pooled.is_none() {
                *pooled = Some(arena);
            }
        }
    }
}

impl std::fmt::Debug for ReusableTrustCgSuccessorArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_arena(), f)
    }
}

impl Clone for ReusableTrustCgSuccessorArena {
    fn clone(&self) -> Self {
        Self {
            arena: Some(self.as_arena().clone()),
            pool: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

impl PartialEq for ReusableTrustCgSuccessorArena {
    fn eq(&self, other: &Self) -> bool {
        self.as_arena() == other.as_arena()
    }
}

impl Eq for ReusableTrustCgSuccessorArena {}

/// Result of processing one BFS frontier level via a compiled, fused level
/// function.
///
/// The hot successor output is a single flat arena rather than `Vec<Vec<i64>>`:
/// `successor_count` states are packed consecutively, each with `state_len`
/// i64 slots. This lets the Rust BFS loop borrow successor slices directly for
/// fingerprinting, global dedup, and frontier enqueue.
///
/// Native fused backends return this arena shape directly,
/// so the model-checker loop can avoid per-successor materialization on the
/// fused path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::check::model_checker) struct CompiledLevelResult {
    /// New successor states discovered in this level, packed as
    /// `successor_count * state_len` i64 slots.
    successor_arena: CompiledSuccessorArena,
    /// Number of packed successor states in `successor_arena`.
    successor_count: usize,
    /// Number of i64 slots per successor state.
    state_len: usize,
    /// Number of parent states fully processed before early exit (or the
    /// full frontier count if the level completed without violation).
    pub parents_processed: usize,
    /// Total successors generated across the whole level (before dedup).
    pub total_generated: u64,
    /// Total new successors after the backend's first-level dedup.
    pub total_new: u64,
    /// `true` if every compiled invariant passed on every generated
    /// successor; `false` if any invariant failed.
    pub invariant_ok: bool,
    /// Index of the parent that produced the first invariant violation
    /// (only set when `invariant_ok == false`).
    pub failed_parent_idx: Option<usize>,
    /// Index of the invariant that failed (only set when
    /// `invariant_ok == false`).
    pub failed_invariant_idx: Option<u32>,
    /// The failing successor state.
    ///
    /// Backends may instead set `failed_successor_idx` when the failing state
    /// is already present in the packed successor arena. Invariant failures
    /// without either form of failed-successor metadata are treated as backend
    /// contract errors by the level loop.
    pub failed_successor: Option<Vec<i64>>,
    /// Index of the failing successor inside `successor_arena`.
    ///
    /// Backends that already return a packed successor arena can set this
    /// instead of copying the failing state into `failed_successor`.
    failed_successor_idx: Option<usize>,
    /// Whether this backend checked every configured regular invariant.
    ///
    /// The BFS loop may skip the Rust invariant path only when this is true.
    /// Action-only fused backends must leave it false so normal invariant
    /// evaluation still runs after global dedup.
    regular_invariants_checked_by_backend: bool,
    /// First parent in this level that generated zero raw successors, if the
    /// backend reports complete per-parent raw-successor metadata.
    first_parent_without_raw_successors: Option<usize>,
    /// Whether `first_parent_without_raw_successors` is complete for the
    /// processed parent prefix represented by this level result.
    raw_successor_metadata_complete: bool,
    /// Whether the successor arena contains every raw successor edge needed
    /// for liveness state-graph capture.
    ///
    /// This flag is only one part of the contract: callers also require a
    /// complete parent-index sidecar and no backend-level edge collapsing.
    state_graph_successors_complete: bool,
}

impl CompiledLevelResult {
    fn required_arena_slots(successor_count: usize, state_len: usize) -> usize {
        successor_count
            .checked_mul(state_len)
            .expect("successor_count * state_len overflow")
    }

    /// Construct a fused-level result from an already contiguous successor
    /// arena.
    ///
    /// `successor_arena` must contain at least
    /// `successor_count * state_len` slots. Slack capacity/slots are ignored
    /// by [`Self::iter_successors`].
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn from_successor_arena(
        successor_arena: Vec<i64>,
        successor_count: usize,
        state_len: usize,
        parents_processed: usize,
        total_generated: u64,
        total_new: u64,
        invariant_ok: bool,
        failed_parent_idx: Option<usize>,
        failed_invariant_idx: Option<u32>,
        failed_successor: Option<Vec<i64>>,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        Self::from_successor_arena_with_parent_indices(
            successor_arena,
            None,
            successor_count,
            state_len,
            parents_processed,
            total_generated,
            total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            failed_successor,
            regular_invariants_checked_by_backend,
        )
    }

    /// Construct a fused-level result from a contiguous successor arena plus
    /// optional per-successor parent provenance.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn from_successor_arena_with_parent_indices(
        successor_arena: Vec<i64>,
        parent_indices: Option<Vec<usize>>,
        successor_count: usize,
        state_len: usize,
        parents_processed: usize,
        total_generated: u64,
        total_new: u64,
        invariant_ok: bool,
        failed_parent_idx: Option<usize>,
        failed_invariant_idx: Option<u32>,
        failed_successor: Option<Vec<i64>>,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        let required_slots = Self::required_arena_slots(successor_count, state_len);
        assert!(
            successor_arena.len() >= required_slots,
            "successor arena shorter than successor_count * state_len",
        );
        assert_eq!(
            u64::try_from(successor_count).expect("successor_count exceeds u64"),
            total_new,
            "successor_count must match backend new count",
        );
        assert!(
            total_generated >= total_new,
            "total_generated must be at least total_new",
        );
        if let Some(parent_indices) = parent_indices.as_ref() {
            assert_eq!(
                parent_indices.len(),
                successor_count,
                "successor parent index count must match successor_count",
            );
        }
        Self {
            successor_arena: CompiledSuccessorArena::Owned {
                arena: successor_arena,
                parent_indices,
            },
            successor_count,
            state_len,
            parents_processed,
            total_generated,
            total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            failed_successor,
            failed_successor_idx: None,
            regular_invariants_checked_by_backend,
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: false,
            state_graph_successors_complete: false,
        }
    }

    /// Construct a fused-level result that owns an trust-codegen successor arena.
    ///
    /// This keeps the packed trust-codegen successor buffer in its original allocation
    /// so the model-checker loop can iterate it directly instead of copying
    /// `states_flat()` into another `Vec<i64>`.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn from_trust_cg_successor_arena(
        successor_arena: tla_trust_cg::TrustCgSuccessorArena,
        parents_processed: usize,
        total_generated: u64,
        total_new: u64,
        invariant_ok: bool,
        failed_parent_idx: Option<usize>,
        failed_invariant_idx: Option<u32>,
        failed_successor: Option<Vec<i64>>,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        Self::from_trust_cg_successor_arena_with_failed_successor_idx(
            successor_arena,
            parents_processed,
            total_generated,
            total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            failed_successor,
            None,
            regular_invariants_checked_by_backend,
        )
    }

    /// Construct a fused-level result that owns an trust-codegen successor arena and
    /// points at the failed successor inside that arena when possible.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn from_trust_cg_successor_arena_with_failed_successor_idx(
        successor_arena: tla_trust_cg::TrustCgSuccessorArena,
        parents_processed: usize,
        total_generated: u64,
        total_new: u64,
        invariant_ok: bool,
        failed_parent_idx: Option<usize>,
        failed_invariant_idx: Option<u32>,
        failed_successor: Option<Vec<i64>>,
        failed_successor_idx: Option<usize>,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        assert_eq!(
            successor_arena.successor_count() as u64,
            total_new,
            "trust-cg successor arena count must match backend new count",
        );
        assert!(
            total_generated >= total_new,
            "total_generated must be at least total_new",
        );
        let first_parent_without_raw_successors =
            successor_arena.first_parent_without_raw_successors();
        let raw_successor_metadata_complete = successor_arena.raw_successor_metadata_complete();
        Self {
            successor_count: successor_arena.successor_count(),
            state_len: successor_arena.state_len(),
            successor_arena: CompiledSuccessorArena::TrustCg(successor_arena),
            parents_processed,
            total_generated,
            total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            failed_successor,
            failed_successor_idx,
            regular_invariants_checked_by_backend,
            first_parent_without_raw_successors,
            raw_successor_metadata_complete,
            state_graph_successors_complete: false,
        }
    }

    /// Construct a fused-level result that returns its trust-codegen arena allocation
    /// to `pool` when the result is dropped.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn from_reusable_trust_cg_successor_arena(
        successor_arena: tla_trust_cg::TrustCgSuccessorArena,
        pool: TrustCgSuccessorArenaPool,
        parents_processed: usize,
        total_generated: u64,
        total_new: u64,
        invariant_ok: bool,
        failed_parent_idx: Option<usize>,
        failed_invariant_idx: Option<u32>,
        failed_successor: Option<Vec<i64>>,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        Self::from_reusable_trust_cg_successor_arena_with_failed_successor_idx(
            successor_arena,
            pool,
            parents_processed,
            total_generated,
            total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            failed_successor,
            None,
            regular_invariants_checked_by_backend,
        )
    }

    /// Construct a reusable trust-codegen result and keep failed-successor provenance
    /// as an arena index instead of cloning the state.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn from_reusable_trust_cg_successor_arena_with_failed_successor_idx(
        successor_arena: tla_trust_cg::TrustCgSuccessorArena,
        pool: TrustCgSuccessorArenaPool,
        parents_processed: usize,
        total_generated: u64,
        total_new: u64,
        invariant_ok: bool,
        failed_parent_idx: Option<usize>,
        failed_invariant_idx: Option<u32>,
        failed_successor: Option<Vec<i64>>,
        failed_successor_idx: Option<usize>,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        assert_eq!(
            successor_arena.successor_count() as u64,
            total_new,
            "trust-cg successor arena count must match backend new count",
        );
        assert!(
            total_generated >= total_new,
            "total_generated must be at least total_new",
        );
        let first_parent_without_raw_successors =
            successor_arena.first_parent_without_raw_successors();
        let raw_successor_metadata_complete = successor_arena.raw_successor_metadata_complete();
        let successor_count = successor_arena.successor_count();
        let state_len = successor_arena.state_len();
        Self {
            successor_count,
            state_len,
            successor_arena: CompiledSuccessorArena::ReusableTrustCg(
                ReusableTrustCgSuccessorArena::new(successor_arena, pool),
            ),
            parents_processed,
            total_generated,
            total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            failed_successor,
            failed_successor_idx,
            regular_invariants_checked_by_backend,
            first_parent_without_raw_successors,
            raw_successor_metadata_complete,
            state_graph_successors_complete: false,
        }
    }

    /// Number of new successors packed in this result.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_count(&self) -> usize {
        self.successor_count
    }

    /// Number of i64 slots per successor state in this result.
    #[must_use]
    pub(in crate::check::model_checker) fn state_len(&self) -> usize {
        self.state_len
    }

    /// First parent with zero raw successors, when the backend supplied
    /// complete per-parent raw-successor metadata.
    #[must_use]
    pub(in crate::check::model_checker) fn first_parent_without_raw_successors(
        &self,
    ) -> Option<usize> {
        self.first_parent_without_raw_successors
    }

    /// Whether deadlock metadata is available for the processed parent prefix.
    #[must_use]
    pub(in crate::check::model_checker) fn raw_successor_metadata_complete(&self) -> bool {
        self.raw_successor_metadata_complete
    }

    /// Whether every packed successor has originating-parent metadata.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_parent_indices_complete(&self) -> bool {
        self.successor_arena
            .parent_indices_complete(self.successor_count)
    }

    /// Borrow the complete originating-parent sidecar, when the backend
    /// supplied one entry for every successor.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_parent_indices(
        &self,
    ) -> Option<CompiledSuccessorParentIndices<'_>> {
        self.successor_arena.parent_indices(self.successor_count)
    }

    /// Mark whether this result carries the complete raw successor edge set
    /// needed by liveness state-graph capture.
    #[must_use]
    pub(in crate::check::model_checker) fn with_state_graph_successors_complete(
        mut self,
        complete: bool,
    ) -> Self {
        self.state_graph_successors_complete = complete;
        self
    }

    /// Whether this result is safe for liveness state-graph capture.
    ///
    /// Liveness captures a parent -> successor graph. A fused backend cannot
    /// satisfy that graph contract after deduplicating successors across
    /// parents, even if it can still report the right transition count. Require
    /// the packed arena to have one successor entry for every generated edge so
    /// the level loop falls back before losing a duplicate child edge from a
    /// different parent.
    #[must_use]
    pub(in crate::check::model_checker) fn state_graph_successors_complete(&self) -> bool {
        self.state_graph_successors_complete
            && self.successor_parent_indices_complete()
            && self.total_generated == self.successor_count as u64
    }

    /// Whether the fused backend already checked all regular invariants.
    #[must_use]
    pub(in crate::check::model_checker) fn regular_invariants_checked_by_backend(&self) -> bool {
        self.regular_invariants_checked_by_backend
    }

    /// Borrow the failed successor from the packed arena, falling back to the
    /// legacy owned copy when that is all the backend provided.
    #[must_use]
    pub(in crate::check::model_checker) fn failed_successor(&self) -> Option<&[i64]> {
        if let Some(successor) = self.failed_successor.as_deref() {
            return Some(successor);
        }
        self.failed_successor_idx
            .and_then(|idx| self.successor_at(idx))
    }

    /// Borrow one successor by index from the packed arena.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_at(&self, idx: usize) -> Option<&[i64]> {
        if idx >= self.successor_count {
            return None;
        }
        if self.state_len == 0 {
            return Some(&[]);
        }
        let start = idx.checked_mul(self.state_len)?;
        let end = start.checked_add(self.state_len)?;
        self.successor_arena.as_flat_slice().get(start..end)
    }

    /// Return the backend-provided compiled-flat fingerprint for one successor.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_fingerprint_at(
        &self,
        idx: usize,
    ) -> Option<crate::Fingerprint> {
        if idx >= self.successor_count {
            return None;
        }
        self.successor_arena.fingerprint_at(idx)
    }

    /// Borrow the complete compiled-flat fingerprint sidecar, when available.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_fingerprint_values(&self) -> Option<&[u64]> {
        self.successor_arena
            .fingerprint_values(self.successor_count)
    }

    /// Borrow the used prefix of the packed successor arena.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_arena_slice(&self) -> &[i64] {
        if self.state_len == 0 {
            return &[];
        }
        let slots = self
            .successor_count
            .checked_mul(self.state_len)
            .expect("successor_count * state_len overflow");
        assert!(
            self.successor_arena.as_flat_slice().len() >= slots,
            "successor arena shorter than successor_count * state_len",
        );
        &self.successor_arena.as_flat_slice()[..slots]
    }

    /// Iterate over flat successor slices in the arena.
    pub(in crate::check::model_checker) fn iter_successors(
        &self,
    ) -> impl Iterator<Item = &[i64]> + '_ {
        if self.state_len == 0 {
            return CompiledLevelSuccessorIter::Empty(self.successor_count);
        }

        let slots = self
            .successor_count
            .checked_mul(self.state_len)
            .expect("successor_count * state_len overflow");
        assert!(
            self.successor_arena.as_flat_slice().len() >= slots,
            "successor arena shorter than successor_count * state_len",
        );
        CompiledLevelSuccessorIter::Chunked(
            self.successor_arena.as_flat_slice()[..slots].chunks_exact(self.state_len),
        )
    }

    /// Iterate over successors with the originating parent index when the
    /// backend exposes that provenance.
    ///
    /// Owned arenas yield parent indices only when their backend supplied the
    /// optional sidecar; otherwise they preserve historical parent-less traces.
    pub(in crate::check::model_checker) fn iter_successors_with_parent_indices(
        &self,
    ) -> impl Iterator<Item = (Option<usize>, &[i64])> + '_ {
        self.iter_successors()
            .enumerate()
            .map(|(successor_idx, successor)| {
                (
                    self.successor_arena.parent_index_at(successor_idx),
                    successor,
                )
            })
    }

    /// Mutate every flat successor slice in place.
    pub(in crate::check::model_checker) fn for_each_successor_mut<E>(
        &mut self,
        mut f: impl FnMut(&mut [i64]) -> Result<(), E>,
    ) -> Result<(), E> {
        self.successor_arena.clear_fingerprints();
        if self.state_len == 0 {
            for _ in 0..self.successor_count {
                f(&mut [])?;
            }
            return Ok(());
        }

        let slots = self
            .successor_count
            .checked_mul(self.state_len)
            .expect("successor_count * state_len overflow");
        assert!(
            self.successor_arena.as_flat_slice_mut().len() >= slots,
            "successor arena shorter than successor_count * state_len",
        );
        for successor in
            self.successor_arena.as_flat_slice_mut()[..slots].chunks_exact_mut(self.state_len)
        {
            f(successor)?;
        }
        Ok(())
    }
}

/// Internal iterator for [`CompiledLevelResult::iter_successors`].
enum CompiledLevelSuccessorIter<'a> {
    Chunked(std::slice::ChunksExact<'a, i64>),
    Empty(usize),
}

impl<'a> Iterator for CompiledLevelSuccessorIter<'a> {
    type Item = &'a [i64];

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompiledLevelSuccessorIter::Chunked(chunks) => chunks.next(),
            CompiledLevelSuccessorIter::Empty(remaining) => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(&[])
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompiledLevelSuccessorIter::Chunked(chunks) => chunks.size_hint(),
            CompiledLevelSuccessorIter::Empty(remaining) => (*remaining, Some(*remaining)),
        }
    }
}

impl<'a> ExactSizeIterator for CompiledLevelSuccessorIter<'a> {}

/// Reusable caller-owned scratch for scoped compiled BFS stepping.
///
/// Backends that can return borrowed step results write successors into this
/// buffer and return a [`FlatBfsStepOutputRef`]. Backends without a scoped path
/// can ignore it and return an owned [`FlatBfsStepOutput`] through the default
/// trait method.
#[derive(Debug, Clone)]
pub(in crate::check::model_checker) struct CompiledBfsStepScratch {
    successor_arena: Vec<i64>,
}

impl CompiledBfsStepScratch {
    /// Create empty reusable successor scratch.
    #[must_use]
    pub(in crate::check::model_checker) fn new(state_len: usize) -> Self {
        Self {
            successor_arena: Vec::with_capacity(state_len),
        }
    }

    /// Clear written successors while retaining allocation.
    pub(in crate::check::model_checker) fn clear(&mut self) {
        self.successor_arena.clear();
    }

    /// Current arena capacity in i64 slots.
    #[cfg(test)]
    #[must_use]
    pub(in crate::check::model_checker) fn slot_capacity(&self) -> usize {
        self.successor_arena.capacity()
    }

    /// Append one successor slot prefilled with the parent state.
    ///
    /// trust-codegen next-state functions may perform partial writes, so the output
    /// slot starts as a copy of the parent and the native action mutates it in
    /// place. The returned start slot can be used with [`Self::successor_mut`]
    /// and [`Self::successor`].
    pub(in crate::check::model_checker) fn append_successor_template(
        &mut self,
        state: &[i64],
    ) -> Result<usize, BfsStepError> {
        let start = self.successor_arena.len();
        let end = start
            .checked_add(state.len())
            .ok_or(BfsStepError::RuntimeError)?;
        self.successor_arena.reserve(state.len());
        self.successor_arena.extend_from_slice(state);
        debug_assert_eq!(self.successor_arena.len(), end);
        Ok(start)
    }

    /// Truncate the scratch arena back to `slot_len`.
    pub(in crate::check::model_checker) fn truncate_slots(&mut self, slot_len: usize) {
        self.successor_arena.truncate(slot_len);
    }

    /// Borrow a mutable successor by start slot and state length.
    pub(in crate::check::model_checker) fn successor_mut(
        &mut self,
        start: usize,
        state_len: usize,
    ) -> Result<&mut [i64], BfsStepError> {
        let end = start
            .checked_add(state_len)
            .ok_or(BfsStepError::RuntimeError)?;
        self.successor_arena
            .get_mut(start..end)
            .ok_or(BfsStepError::RuntimeError)
    }

    /// Borrow a successor by start slot and state length.
    pub(in crate::check::model_checker) fn successor(
        &self,
        start: usize,
        state_len: usize,
    ) -> Result<&[i64], BfsStepError> {
        let end = start
            .checked_add(state_len)
            .ok_or(BfsStepError::RuntimeError)?;
        self.successor_arena
            .get(start..end)
            .ok_or(BfsStepError::RuntimeError)
    }

    fn successor_slice(
        &self,
        state_len: usize,
        successor_count: usize,
    ) -> Result<&[i64], BfsStepError> {
        let slots = successor_count
            .checked_mul(state_len)
            .ok_or(BfsStepError::RuntimeError)?;
        self.successor_arena
            .get(..slots)
            .ok_or(BfsStepError::RuntimeError)
    }

    /// Build a borrowed flat step output over the currently written arena.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check::model_checker) fn output_ref(
        &self,
        state_len: usize,
        successor_count: usize,
        generated_count: u32,
        invariant_ok: bool,
        failed_invariant_idx: Option<u32>,
        failed_successor_idx: Option<u32>,
    ) -> Result<FlatBfsStepOutputRef<'_>, BfsStepError> {
        Ok(FlatBfsStepOutputRef::from_parts(
            self.successor_slice(state_len, successor_count)?,
            state_len,
            successor_count,
            generated_count,
            invariant_ok,
            failed_invariant_idx,
            failed_successor_idx,
        ))
    }
}

/// Compiled BFS step output that is either owned or borrowed from caller
/// scratch.
pub(in crate::check::model_checker) enum CompiledStepOutput<'a> {
    Owned(FlatBfsStepOutput),
    Borrowed {
        output: FlatBfsStepOutputRef<'a>,
        state_len: usize,
    },
}

impl<'a> CompiledStepOutput<'a> {
    /// Wrap an owned runtime output.
    #[must_use]
    pub(in crate::check::model_checker) fn from_owned(output: FlatBfsStepOutput) -> Self {
        Self::Owned(output)
    }

    /// Wrap a borrowed runtime output.
    #[must_use]
    pub(in crate::check::model_checker) fn from_borrowed(
        output: FlatBfsStepOutputRef<'a>,
        state_len: usize,
    ) -> Self {
        Self::Borrowed { output, state_len }
    }

    /// Whether this output borrows a caller-owned contiguous successor arena.
    #[must_use]
    pub(in crate::check::model_checker) fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed { .. })
    }

    /// Total generated successors before local filtering.
    #[must_use]
    pub(in crate::check::model_checker) fn generated_count(&self) -> u32 {
        match self {
            Self::Owned(output) => output.generated_count,
            Self::Borrowed { output, .. } => output.generated_count,
        }
    }

    /// Whether every invariant passed.
    #[must_use]
    pub(in crate::check::model_checker) fn invariant_ok(&self) -> bool {
        match self {
            Self::Owned(output) => output.invariant_ok,
            Self::Borrowed { output, .. } => output.invariant_ok,
        }
    }

    /// Failed invariant index, if any.
    #[must_use]
    pub(in crate::check::model_checker) fn failed_invariant_idx(&self) -> Option<u32> {
        match self {
            Self::Owned(output) => output.failed_invariant_idx,
            Self::Borrowed { output, .. } => output.failed_invariant_idx,
        }
    }

    /// Failed successor index, if any.
    #[must_use]
    pub(in crate::check::model_checker) fn failed_successor_idx(&self) -> Option<u32> {
        match self {
            Self::Owned(output) => output.failed_successor_idx,
            Self::Borrowed { output, .. } => output.failed_successor_idx,
        }
    }

    /// Number of returned successors.
    #[must_use]
    pub(in crate::check::model_checker) fn successor_count(&self) -> usize {
        match self {
            Self::Owned(output) => output.successor_count(),
            Self::Borrowed { output, .. } => output.successor_count(),
        }
    }

    /// i64 slots per successor state.
    #[must_use]
    pub(in crate::check::model_checker) fn state_len(&self) -> usize {
        match self {
            Self::Owned(output) => output.state_len(),
            Self::Borrowed { state_len, .. } => *state_len,
        }
    }

    /// Borrow one successor by index.
    pub(in crate::check::model_checker) fn successor_at(&self, idx: usize) -> Option<&[i64]> {
        match self {
            Self::Owned(output) => output.iter_successors().nth(idx),
            Self::Borrowed { output, .. } => output.iter_successors().nth(idx),
        }
    }

    /// Visit every successor without exposing backend-specific iterator types.
    pub(in crate::check::model_checker) fn for_each_successor<E>(
        &self,
        mut f: impl FnMut(&[i64]) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Self::Owned(output) => {
                for successor in output.iter_successors() {
                    f(successor)?;
                }
            }
            Self::Borrowed { output, .. } => {
                for successor in output.iter_successors() {
                    f(successor)?;
                }
            }
        }
        Ok(())
    }

    /// Materialize this result as an owned runtime output.
    pub(in crate::check::model_checker) fn to_owned_flat(&self) -> FlatBfsStepOutput {
        let state_len = self.state_len();
        let successor_count = self.successor_count();
        let mut succ_buf = Vec::with_capacity(successor_count.saturating_mul(state_len));
        self.for_each_successor(|successor| {
            succ_buf.extend_from_slice(successor);
            Ok::<(), std::convert::Infallible>(())
        })
        .expect("infallible successor copy");

        FlatBfsStepOutput::from_parts(
            succ_buf,
            state_len,
            successor_count,
            self.generated_count(),
            self.invariant_ok(),
            self.failed_invariant_idx(),
            self.failed_successor_idx(),
        )
    }
}

/// Per-parent compiled BFS step.
///
/// Implementers take a parent flat-i64 state and return a `FlatBfsStepOutput`
/// describing the new successors, invariant status, and any violation.
/// Action dispatch, invariant checking, fingerprinting, and first-level
/// (backend-local) dedup all run in compiled code inside `step_flat`.
///
/// Currently implemented by the fail-closed legacy compatibility shim and,
/// when the `trust_cg` feature is enabled, `TrustCgCompiledBfsStep`.
pub(in crate::check::model_checker) trait CompiledBfsStep:
    Send
{
    /// Number of i64 slots per flat state.
    fn state_len(&self) -> usize;

    /// Whether this step emits every raw successor edge needed for liveness
    /// state-graph capture.
    fn preserves_state_graph_successor_edges(&self) -> bool {
        false
    }

    /// Execute one compiled BFS step and return flat successor slices.
    fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError>;

    /// Execute one compiled BFS step, borrowing successor storage from caller
    /// scratch when the backend supports it.
    fn step_flat_scoped<'a>(
        &self,
        state: &[i64],
        _scratch: &'a mut CompiledBfsStepScratch,
    ) -> Result<CompiledStepOutput<'a>, BfsStepError> {
        self.step_flat(state).map(CompiledStepOutput::from_owned)
    }
}

/// Fused compiled BFS level.
///
/// When available (`has_fused_level() == true`), processes an entire frontier
/// in a single native call by walking the arena in compiled code — no
/// per-parent Rust-to-native boundary crossing. The level loop prefers this
/// path when available and falls back to the per-parent `CompiledBfsStep`
/// otherwise.
///
/// Currently implemented by the fail-closed legacy compatibility shim and,
/// when the `trust_cg` feature is enabled, `TrustCgCompiledBfsLevel`.
pub(in crate::check::model_checker) trait CompiledBfsLevel:
    Send
{
    /// Whether the fused native level function is actually available.
    /// Implementers may return `false` when only the per-parent path was
    /// compiled.
    fn has_fused_level(&self) -> bool;

    /// Whether this level is backed by a native generated parent loop.
    ///
    /// Prototype fused adapters still batch through the Rust parent loop and
    /// must not satisfy the strict trust-codegen native-fused launch gate.
    fn has_native_fused_level(&self) -> bool {
        false
    }

    /// State width used by the fused level's flat parent/successor ABI.
    ///
    /// State-constrained native-fused admission requires this to match the
    /// active flat frontier width. Implementations that cannot report the
    /// width fail closed for that constrained path.
    fn fused_level_state_len(&self) -> Option<usize> {
        None
    }

    /// Number of configured state constraints checked inside the native fused
    /// parent loop.
    fn native_fused_state_constraint_count(&self) -> usize {
        0
    }

    /// Number of invariant/action-property predicates checked inside the
    /// native fused parent loop.
    // Default impl unused: overriding impls (trust_cg_dispatch) supply the real
    // count; kept as a trait default for backends that check no invariants.
    #[allow(dead_code)]
    fn native_fused_invariant_count(&self) -> usize {
        0
    }

    /// Whether the backend checked exactly the requested state constraints.
    fn native_fused_state_constraints_checked_by_backend(&self, expected_count: usize) -> bool {
        expected_count == 0
            || (self.has_native_fused_level()
                && self.native_fused_state_constraint_count() == expected_count)
    }

    /// Whether regular invariants are checked inside the native fused backend
    /// instead of falling back to Rust after successor admission.
    fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
        false
    }

    /// Whether this level contains a multi-successor record-set loop kernel.
    ///
    /// The loop-kernel path already sizes its transient successor arena
    /// adaptively (previous level's committed yield × 2), so it does not exhibit
    /// the fixed `parents × action_count` over-reservation that parent
    /// sub-batching targets. The compiled BFS loop consults this to leave the
    /// loop-kernel path (e.g. the PaxosCommit flagship) on the whole-level call.
    fn native_fused_has_loop_actions(&self) -> bool {
        false
    }

    /// Whether the compiled BFS loop may skip its pre-insert global seen
    /// lookup for fused successors.
    ///
    /// The default stays conservative because action-only/prototype fused
    /// paths may still need Rust-side invariant work before admission.
    fn skip_global_pre_seen_lookup(&self) -> bool {
        false
    }

    /// Run any backend-owned preflight checks against the current fused
    /// frontier arena before the compiled BFS loop commits to the full native
    /// level call.
    fn preflight_fused_arena(
        &self,
        _arena: &[i64],
        _parent_count: usize,
    ) -> Result<(), BfsStepError> {
        Ok(())
    }

    /// Process one frontier level in native code.
    ///
    /// Returns `None` when the fused function is unavailable. Returns
    /// `Some(Ok(_))` with the level result on success.
    ///
    /// Runtime errors are not a blanket "retry and continue" signal:
    /// `FatalRuntimeError` is terminal, state-constrained fused execution
    /// fails closed, and liveness capture may fall back only to a path that
    /// preserves every parent -> successor edge. Nonfatal errors in ordinary
    /// safety-only runs may still fall back to the per-parent step or standard
    /// BFS loop.
    ///
    /// `arena` holds `parent_count * state_len` i64 slots, laid out as
    /// `parent_count` contiguous parent states in row-major order.
    fn run_level_fused_arena(
        &self,
        arena: &[i64],
        parent_count: usize,
    ) -> Option<Result<CompiledLevelResult, BfsStepError>>;
}

// ---------------------------------------------------------------------------
// Legacy compatibility-shim blanket impls.
//
// These adapt the fail-closed compatibility types to the backend-agnostic
// traits above. trust-codegen implements the same traits in `trust_cg_dispatch` behind
// the `trust_cg` feature.
// ---------------------------------------------------------------------------

impl CompiledBfsStep for UnavailableCompiledBfsStep {
    fn state_len(&self) -> usize {
        UnavailableCompiledBfsStep::state_len(self)
    }

    fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
        UnavailableCompiledBfsStep::step_flat(self, state)
    }
}

impl CompiledBfsLevel for UnavailableCompiledBfsLevel {
    fn has_fused_level(&self) -> bool {
        UnavailableCompiledBfsLevel::has_fused_level(self)
    }

    fn fused_level_state_len(&self) -> Option<usize> {
        Some(UnavailableCompiledBfsLevel::state_len(self))
    }

    fn run_level_fused_arena(
        &self,
        arena: &[i64],
        parent_count: usize,
    ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
        UnavailableCompiledBfsLevel::run_level_fused_arena(self, arena, parent_count).map(
            |result| {
                result.map(|r| {
                    let successor_count = r.successor_count();
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        r.successor_arena,
                        r.successor_parent_indices,
                        successor_count,
                        r.state_len,
                        r.parents_processed,
                        r.total_generated,
                        r.total_new,
                        r.invariant_ok,
                        r.failed_parent_idx,
                        r.failed_invariant_idx,
                        r.failed_successor,
                        r.regular_invariants_checked_by_backend,
                    )
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successor_parent_indices_find_first_out_of_bounds() {
        let usize_indices = CompiledSuccessorParentIndices::Usize(&[0, 2, 4, 1]);
        assert_eq!(usize_indices.first_out_of_bounds(3), Some((2, 4)));
        assert_eq!(usize_indices.first_out_of_bounds(5), None);

        let u32_indices = CompiledSuccessorParentIndices::U32(&[0, 2, 4, 1]);
        assert_eq!(u32_indices.first_out_of_bounds(3), Some((2, 4)));
        assert_eq!(u32_indices.first_out_of_bounds(5), None);
    }

    #[test]
    fn owned_level_result_has_no_successor_parent_indices() {
        let result = CompiledLevelResult::from_successor_arena(
            vec![1, 2, 3, 4],
            2,
            2,
            1,
            2,
            2,
            true,
            None,
            None,
            None,
            false,
        );

        let successors = result
            .iter_successors_with_parent_indices()
            .collect::<Vec<_>>();
        assert_eq!(successors, vec![(None, &[1, 2][..]), (None, &[3, 4][..])]);
        assert_eq!(result.state_len(), 2);
        assert_eq!(result.first_parent_without_raw_successors(), None);
        assert!(!result.raw_successor_metadata_complete());
        assert!(!result.successor_parent_indices_complete());
        assert!(!result.state_graph_successors_complete());
        assert!(!result.regular_invariants_checked_by_backend());
    }

    #[test]
    fn owned_level_result_exposes_successor_parent_indices_when_supplied() {
        let result = CompiledLevelResult::from_successor_arena_with_parent_indices(
            vec![1, 2, 3, 4],
            Some(vec![5, 9]),
            2,
            2,
            10,
            2,
            2,
            true,
            None,
            None,
            None,
            true,
        );

        let successors = result
            .iter_successors_with_parent_indices()
            .collect::<Vec<_>>();
        assert_eq!(
            successors,
            vec![(Some(5), &[1, 2][..]), (Some(9), &[3, 4][..])]
        );
        assert!(result.successor_parent_indices_complete());
        let parent_indices = result.successor_parent_indices().unwrap();
        assert_eq!(parent_indices.get(0), Some(5));
        assert_eq!(parent_indices.get(1), Some(9));
        assert_eq!(result.successor_arena_slice(), &[1, 2, 3, 4]);
        assert_eq!(result.successor_fingerprint_values(), None);
        assert!(!result.state_graph_successors_complete());
        assert!(result.regular_invariants_checked_by_backend());
    }

    #[test]
    fn owned_zero_successor_level_result_has_complete_parent_indices() {
        let result = CompiledLevelResult::from_successor_arena(
            vec![],
            0,
            2,
            1,
            0,
            0,
            true,
            None,
            None,
            None,
            false,
        );

        assert!(result.successor_parent_indices_complete());
    }

    #[test]
    fn owned_level_result_state_graph_completeness_requires_parent_indices() {
        let without_parent_indices = CompiledLevelResult::from_successor_arena(
            vec![1, 2],
            1,
            2,
            1,
            1,
            1,
            true,
            None,
            None,
            None,
            false,
        )
        .with_state_graph_successors_complete(true);
        assert!(!without_parent_indices.state_graph_successors_complete());

        let with_parent_indices = CompiledLevelResult::from_successor_arena_with_parent_indices(
            vec![1, 2],
            Some(vec![0]),
            1,
            2,
            1,
            1,
            1,
            true,
            None,
            None,
            None,
            false,
        )
        .with_state_graph_successors_complete(true);
        assert!(with_parent_indices.state_graph_successors_complete());
    }

    #[test]
    fn owned_level_result_state_graph_completeness_rejects_edge_collapsing() {
        let result = CompiledLevelResult::from_successor_arena_with_parent_indices(
            vec![2],
            Some(vec![0]),
            1,
            1,
            2,
            2,
            1,
            true,
            None,
            None,
            None,
            false,
        )
        .with_state_graph_successors_complete(true);

        assert!(!result.state_graph_successors_complete());
    }

    #[test]
    #[should_panic(expected = "successor arena shorter than successor_count * state_len")]
    fn owned_level_result_rejects_short_successor_arena() {
        let _ = CompiledLevelResult::from_successor_arena(
            vec![1, 2, 3],
            2,
            2,
            1,
            2,
            2,
            true,
            None,
            None,
            None,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "successor parent index count must match successor_count")]
    fn owned_level_result_rejects_mismatched_parent_indices() {
        let _ = CompiledLevelResult::from_successor_arena_with_parent_indices(
            vec![1, 2, 3, 4],
            Some(vec![5]),
            2,
            2,
            10,
            2,
            2,
            true,
            None,
            None,
            None,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "total_generated must be at least total_new")]
    fn owned_level_result_rejects_generated_total_below_new_total() {
        let _ = CompiledLevelResult::from_successor_arena(
            vec![1, 2, 3, 4],
            2,
            2,
            1,
            1,
            2,
            true,
            None,
            None,
            None,
            false,
        );
    }

    #[test]
    fn trust_cg_level_result_exposes_successor_parent_indices() {
        let mut arena = tla_trust_cg::TrustCgSuccessorArena::new(2);
        arena.push_successor(3, &[1, 2]).unwrap();
        arena.push_successor(7, &[3, 4]).unwrap();
        let expected_fingerprints = arena.successor_fingerprints().to_vec();

        let result = CompiledLevelResult::from_trust_cg_successor_arena(
            arena, 8, 2, 2, true, None, None, None, true,
        );

        let successors = result
            .iter_successors_with_parent_indices()
            .collect::<Vec<_>>();
        assert_eq!(
            successors,
            vec![(Some(3), &[1, 2][..]), (Some(7), &[3, 4][..])]
        );
        assert_eq!(
            result.successor_fingerprint_at(0),
            Some(crate::Fingerprint(expected_fingerprints[0]))
        );
        assert_eq!(
            result.successor_fingerprint_at(1),
            Some(crate::Fingerprint(expected_fingerprints[1]))
        );
        let parent_indices = result.successor_parent_indices().unwrap();
        assert_eq!(parent_indices.get(0), Some(3));
        assert_eq!(parent_indices.get(1), Some(7));
        assert_eq!(
            result.successor_fingerprint_values(),
            Some(&expected_fingerprints[..])
        );
        assert_eq!(result.successor_arena_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn trust_cg_level_result_borrows_failed_successor_by_index() {
        let mut arena = tla_trust_cg::TrustCgSuccessorArena::new(2);
        arena.push_successor(3, &[1, 2]).unwrap();
        arena.push_successor(7, &[3, 4]).unwrap();

        let result = CompiledLevelResult::from_trust_cg_successor_arena_with_failed_successor_idx(
            arena,
            8,
            2,
            2,
            false,
            Some(7),
            Some(1),
            None,
            Some(1),
            true,
        );

        assert_eq!(result.failed_successor(), Some(&[3, 4][..]));
    }

    #[test]
    #[should_panic(expected = "total_generated must be at least total_new")]
    fn trust_cg_level_result_rejects_generated_total_below_new_total() {
        let mut arena = tla_trust_cg::TrustCgSuccessorArena::new(2);
        arena.push_successor(3, &[1, 2]).unwrap();
        arena.push_successor(7, &[3, 4]).unwrap();

        let _ = CompiledLevelResult::from_trust_cg_successor_arena(
            arena, 8, 1, 2, true, None, None, None, true,
        );
    }

    #[test]
    fn reusable_trust_cg_level_result_returns_successor_arena_on_drop() {
        let pool: TrustCgSuccessorArenaPool = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut arena = tla_trust_cg::TrustCgSuccessorArena::with_capacity(2, 4);
        arena.push_successor(3, &[1, 2]).unwrap();
        arena.push_successor(7, &[3, 4]).unwrap();

        {
            let result = CompiledLevelResult::from_reusable_trust_cg_successor_arena(
                arena,
                pool.clone(),
                8,
                2,
                2,
                true,
                None,
                None,
                None,
                true,
            );
            assert!(
                pool.lock().unwrap().is_none(),
                "arena must stay owned by the live level result"
            );
            let successors = result
                .iter_successors_with_parent_indices()
                .collect::<Vec<_>>();
            assert_eq!(
                successors,
                vec![(Some(3), &[1, 2][..]), (Some(7), &[3, 4][..])]
            );
        }

        let pooled = pool.lock().unwrap();
        let returned = pooled
            .as_ref()
            .expect("dropping the result should return its arena to the pool");
        assert_eq!(returned.state_len(), 2);
        assert_eq!(returned.successor_count(), 0);
        assert!(returned.states_flat().is_empty());
        assert!(returned.parent_indices().is_empty());
        assert!(returned.successor_fingerprints().is_empty());
    }

    #[test]
    fn compiled_step_output_reports_borrowed_storage() {
        let owned = CompiledStepOutput::from_owned(FlatBfsStepOutput::from_parts(
            vec![1, 2, 3, 4],
            2,
            2,
            2,
            true,
            None,
            None,
        ));
        assert!(!owned.is_borrowed());

        let arena = [5, 6, 7, 8];
        let borrowed_ref = FlatBfsStepOutputRef::from_parts(&arena, 2, 2, 2, true, None, None);
        let borrowed = CompiledStepOutput::from_borrowed(borrowed_ref, 2);
        assert!(borrowed.is_borrowed());
        assert_eq!(
            borrowed.for_each_successor(|successor| {
                assert_eq!(successor.len(), 2);
                Ok::<(), std::convert::Infallible>(())
            }),
            Ok(())
        );
    }
}
