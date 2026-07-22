// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! trust-codegen fused BFS level ABI and prototype runner.
//!
//! This module is the trust_cg-owned foundation for the single-thread compiled
//! BFS level path. It defines the flat parent arena ABI and the contiguous
//! successor arena shape that `tla-check` can wire through the stable ABI
//! crate without depending on retired backend types.
//!
//! The first implementation here is intentionally a Rust prototype over
//! already-compiled trust-codegen action/invariant function pointers. It processes an
//! entire flat parent arena and emits successors into one contiguous arena,
//! exercising the ABI and data movement contract before the parent loop itself
//! is generated as a native trust-codegen function.

use crate::runtime_abi::{
    ty_compiled_fp_u64, AtomicFpSet, InsertResult, JitCallOut, JitInvariantFn, JitNextStateFn,
    JitStatus, SingleThreadFpSet,
};
use crate::{compile::NativeLibrary, error::TrustCgError};
use std::fmt;

pub use tla_jit_abi::{ActionDescriptor, InvariantDescriptor};

/// Version tag written into every FFI-facing BFS level structure.
pub const TRUST_CG_BFS_LEVEL_ABI_VERSION: u32 = 2;

/// Sentinel used for optional u32 indexes in FFI structs.
pub const TRUST_CG_BFS_NO_INDEX: u32 = u32::MAX;

/// Status code returned by a fused trust-codegen BFS level entry point.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustCgBfsLevelStatus {
    /// The level completed successfully.
    Ok = 0,
    /// Compiled code reported a runtime error.
    RuntimeError = 1,
    /// The caller-provided successor arena was too small.
    BufferOverflow = 2,
    /// The caller and callee disagreed on ABI version, lengths, or pointers.
    InvalidAbi = 3,
    /// Compiled code requested an interpreter fallback.
    FallbackNeeded = 4,
}

impl TrustCgBfsLevelStatus {
    /// Convert a raw ABI status code into the typed enum.
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Ok),
            1 => Some(Self::RuntimeError),
            2 => Some(Self::BufferOverflow),
            3 => Some(Self::InvalidAbi),
            4 => Some(Self::FallbackNeeded),
            _ => None,
        }
    }

    /// Return this status as the raw ABI integer.
    #[must_use]
    pub fn as_raw(self) -> u32 {
        self as u32
    }
}

pub(crate) const TRUST_CG_BFS_NATIVE_CALLOUT_SCRATCH_SLOTS: usize = 5;
const TRUST_CG_BFS_NATIVE_COUNTER_SCRATCH_SLOTS: usize = 5;
/// Route B: scratch slots reserved for the caller-owned
/// `tla_jit_abi::NextStateLoopSink` header the fused parent loop seeds before
/// invoking a multi-successor record-set kernel. The `#[repr(C)]` header is 5
/// pointer-width fields (`succ_buf`/`capacity`/`state_len`/`count` at byte
/// offsets 0/8/16/24 plus the `u32 overflowed` at 32, padded to 40 bytes), so
/// 5 i64 slots hold it exactly. `next_state_loop_sink_header_fits_scratch`
/// (native_bfs.rs) asserts the size in lockstep with the real struct.
pub(crate) const TRUST_CG_BFS_NATIVE_LOOP_SINK_SCRATCH_SLOTS: usize = 5;
/// Route B: one loop-carried scratch slot for the per-successor commit loop's
/// record index (the no-alloca parent loop keeps ALL loop state in scratch).
const TRUST_CG_BFS_NATIVE_LOOP_COMMIT_IDX_SCRATCH_SLOTS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrustCgBfsNativeParentScratchLayout {
    pub callout_offset: usize,
    pub candidate_offset: usize,
    pub parent_idx_offset: usize,
    pub generated_offset: usize,
    pub state_count_offset: usize,
    pub parent_generated_offset: usize,
    pub parent_count_offset: usize,
    /// Route B: base slot of the `NextStateLoopSink` header region
    /// ([`TRUST_CG_BFS_NATIVE_LOOP_SINK_SCRATCH_SLOTS`] slots).
    pub loop_sink_offset: usize,
    /// Route B: slot holding the commit loop's record index.
    pub loop_commit_idx_offset: usize,
    pub total_slots: usize,
}

pub(crate) fn trust_cg_bfs_native_parent_scratch_layout(
    state_len: usize,
) -> Result<TrustCgBfsNativeParentScratchLayout, TrustCgBfsLevelError> {
    let callout_offset: usize = 0;
    let candidate_offset = callout_offset
        .checked_add(TRUST_CG_BFS_NATIVE_CALLOUT_SCRATCH_SLOTS)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let candidate_slots = state_len.max(1);
    let parent_idx_offset = candidate_offset
        .checked_add(candidate_slots)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let generated_offset = parent_idx_offset
        .checked_add(1)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let state_count_offset = generated_offset
        .checked_add(1)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let parent_generated_offset = state_count_offset
        .checked_add(1)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let parent_count_offset = parent_generated_offset
        .checked_add(1)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    // Route B sink header + commit-loop index live AFTER the counters so every
    // pre-existing offset is unchanged. Both the host allocation
    // (`TrustCgBfsLevelNative` scratch sizing / `TrustCgBfsParentArenaAbi`
    // validation) and the generated runtime `scratch_len_ok` check read
    // `total_slots` from this one function, so extending it here keeps both
    // sides of the ABI in lockstep.
    let loop_sink_offset = parent_idx_offset
        .checked_add(TRUST_CG_BFS_NATIVE_COUNTER_SCRATCH_SLOTS)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let loop_commit_idx_offset = loop_sink_offset
        .checked_add(TRUST_CG_BFS_NATIVE_LOOP_SINK_SCRATCH_SLOTS)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    let total_slots = loop_commit_idx_offset
        .checked_add(TRUST_CG_BFS_NATIVE_LOOP_COMMIT_IDX_SCRATCH_SLOTS)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;

    Ok(TrustCgBfsNativeParentScratchLayout {
        callout_offset,
        candidate_offset,
        parent_idx_offset,
        generated_offset,
        state_count_offset,
        parent_generated_offset,
        parent_count_offset,
        loop_sink_offset,
        loop_commit_idx_offset,
        total_slots,
    })
}

/// FFI view of the parent arena passed to a native fused-level function.
///
/// `parents` is a row-major `parent_count * state_len` flat i64 buffer.
/// `scratch` is caller-provided temporary storage sized by
/// [`trust_cg_bfs_native_parent_scratch_layout`]. Native fused loops use this
/// region for callout status, one successor candidate, and parent-loop
/// counters without allocating in the generated parent loop.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrustCgBfsParentArenaAbi {
    /// ABI version. Must equal [`TRUST_CG_BFS_LEVEL_ABI_VERSION`].
    pub abi_version: u32,
    /// Number of i64 slots per flat state.
    pub state_len: u32,
    /// Number of parent states in `parents`.
    pub parent_count: u32,
    /// Pointer to `parent_count * state_len` i64 values.
    pub parents: *const i64,
    /// Pointer to scratch i64 slots. May be null when `scratch_len == 0`.
    pub scratch: *mut i64,
    /// Scratch capacity in i64 slots.
    pub scratch_len: usize,
    /// Optional local dedup set pointer. Null disables native local dedup.
    pub local_fp_set: *const u8,
}

impl TrustCgBfsParentArenaAbi {
    /// Build an ABI view over a borrowed flat parent arena.
    ///
    /// The caller keeps ownership of both slices. This function validates that
    /// the parent slice length matches `parent_count * state_len`.
    pub fn new(
        parents: &[i64],
        parent_count: usize,
        state_len: usize,
        scratch: &mut [i64],
    ) -> Result<Self, TrustCgBfsLevelError> {
        Self::new_with_local_fp_set(parents, parent_count, state_len, scratch, std::ptr::null())
    }

    /// Build an ABI view over a borrowed flat parent arena with an optional
    /// local-dedup fingerprint set pointer.
    pub fn new_with_local_fp_set(
        parents: &[i64],
        parent_count: usize,
        state_len: usize,
        scratch: &mut [i64],
        local_fp_set: *const u8,
    ) -> Result<Self, TrustCgBfsLevelError> {
        validate_arena_len(parents.len(), parent_count, state_len)?;
        let scratch_needed = trust_cg_bfs_native_parent_scratch_layout(state_len)?.total_slots;
        if scratch.len() < scratch_needed {
            return Err(TrustCgBfsLevelError::ScratchTooSmall {
                needed: scratch_needed,
                actual: scratch.len(),
            });
        }
        Ok(Self {
            abi_version: TRUST_CG_BFS_LEVEL_ABI_VERSION,
            state_len: usize_to_u32(state_len, "state_len")?,
            parent_count: usize_to_u32(parent_count, "parent_count")?,
            parents: if parents.is_empty() {
                std::ptr::null()
            } else {
                parents.as_ptr()
            },
            scratch: if scratch.is_empty() {
                std::ptr::null_mut()
            } else {
                scratch.as_mut_ptr()
            },
            scratch_len: scratch.len(),
            local_fp_set,
        })
    }
}

/// FFI view of the contiguous successor arena written by a fused level.
///
/// Native code writes up to `state_capacity` states into `states`, each
/// occupying `state_len` i64 slots. `parent_index` must point to
/// `state_capacity` writable parent-index slots when `state_capacity` is
/// nonzero.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrustCgBfsSuccessorArenaAbi {
    /// ABI version. Must equal [`TRUST_CG_BFS_LEVEL_ABI_VERSION`].
    pub abi_version: u32,
    /// Number of i64 slots per flat state.
    pub state_len: u32,
    /// Number of successor states that fit in `states`.
    pub state_capacity: u32,
    /// Number of successor states actually written by the callee.
    pub state_count: u32,
    /// Pointer to `state_capacity * state_len` writable i64 slots.
    pub states: *mut i64,
    /// Pointer to `state_capacity` writable parent-index slots when capacity is nonzero.
    pub parent_index: *mut u32,
    /// Pointer to `state_capacity` writable compiled-fingerprint slots when capacity is nonzero.
    pub fingerprints: *mut u64,
    /// Total successors accepted by compiled constraints before local dedup.
    pub generated: u64,
    /// Number of parents fully processed.
    pub parents_processed: u32,
    /// 1 when all invariants passed, 0 when a violation was found.
    pub invariant_ok: u8,
    /// Status code as [`TrustCgBfsLevelStatus::as_raw`].
    pub status: u32,
    /// Parent index for the first invariant violation, or
    /// [`TRUST_CG_BFS_NO_INDEX`].
    pub failed_parent_idx: u32,
    /// Invariant index for the first invariant violation, or
    /// [`TRUST_CG_BFS_NO_INDEX`].
    pub failed_invariant_idx: u32,
    /// Successor arena index for the first invariant violation, or
    /// [`TRUST_CG_BFS_NO_INDEX`].
    pub failed_successor_idx: u32,
    /// First parent index that generated zero raw successors, or
    /// [`TRUST_CG_BFS_NO_INDEX`].
    pub first_zero_generated_parent_idx: u32,
    /// 1 when `first_zero_generated_parent_idx` is complete for all processed
    /// parents, 0 when the callee did not report deadlock metadata.
    pub raw_successor_metadata_complete: u8,
}

impl TrustCgBfsSuccessorArenaAbi {
    /// Construct an empty/null ABI value.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            abi_version: TRUST_CG_BFS_LEVEL_ABI_VERSION,
            state_len: 0,
            state_capacity: 0,
            state_count: 0,
            states: std::ptr::null_mut(),
            parent_index: std::ptr::null_mut(),
            fingerprints: std::ptr::null_mut(),
            generated: 0,
            parents_processed: 0,
            invariant_ok: 1,
            status: TrustCgBfsLevelStatus::Ok.as_raw(),
            failed_parent_idx: TRUST_CG_BFS_NO_INDEX,
            failed_invariant_idx: TRUST_CG_BFS_NO_INDEX,
            failed_successor_idx: TRUST_CG_BFS_NO_INDEX,
            first_zero_generated_parent_idx: TRUST_CG_BFS_NO_INDEX,
            raw_successor_metadata_complete: 0,
        }
    }
}

impl Default for TrustCgBfsSuccessorArenaAbi {
    fn default() -> Self {
        Self::empty()
    }
}

/// Native fused level entry point shape.
///
/// The function consumes a parent arena and writes a successor arena. It
/// returns the same status code it writes to `successors.status` so callers
/// can fast-path success while still reading detailed counters from the arena.
pub type TrustCgFusedLevelFn = unsafe extern "C" fn(
    parents: *const TrustCgBfsParentArenaAbi,
    successors: *mut TrustCgBfsSuccessorArenaAbi,
) -> u32;

/// Stable symbol name emitted by [`crate::compile::compile_bfs_level_native`].
pub const TRUST_CG_BFS_LEVEL_SYMBOL: &str = "trust_cg_bfs_level";

const BFS_LEVEL_TRACE_ENV: &str = "TY_TRUST_CG_BFS_LEVEL_TRACE";

/// Invariant status for a completed level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustCgInvariantStatus {
    /// Every compiled invariant passed for every emitted successor.
    Passed,
    /// A compiled invariant failed.
    Failed {
        /// Parent index in the input arena.
        parent_index: u32,
        /// Stable spec invariant index.
        invariant_index: u32,
        /// Index in the successor arena.
        successor_index: u32,
    },
}

impl TrustCgInvariantStatus {
    /// Whether all invariants passed.
    #[must_use]
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Passed)
    }
}

/// Contiguous flat successor arena owned by the trust-codegen BFS level wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustCgSuccessorArena {
    state_len: usize,
    states: Vec<i64>,
    parent_index: Vec<u32>,
    fingerprints: Vec<u64>,
    generated: u64,
    parents_processed: usize,
    invariant: TrustCgInvariantStatus,
    first_parent_without_raw_successors: Option<u32>,
    raw_successor_metadata_complete: bool,
}

impl TrustCgSuccessorArena {
    /// Create an empty successor arena for states with `state_len` i64 slots.
    #[must_use]
    pub fn new(state_len: usize) -> Self {
        Self::with_capacity(state_len, 0)
    }

    /// Create an empty successor arena with space for `state_capacity` states.
    ///
    /// The state and parent-index buffers are reserved up front, but the
    /// fingerprint sidecar is allocated lazily (see [`Self::reserve_fingerprints`]):
    /// it is reserved by [`Self::prepare_abi`] only when an arena is actually
    /// driven through a fingerprint-writing native level, so a constructed-but-
    /// unused (or pooled) arena never carries a dead `u64`-per-successor sidecar.
    /// `push_successor*` still works on a lazily-allocated sidecar because `Vec`
    /// grows on first push.
    #[must_use]
    pub fn with_capacity(state_len: usize, state_capacity: usize) -> Self {
        let slot_capacity = state_len.saturating_mul(state_capacity);
        Self {
            state_len,
            states: Vec::with_capacity(slot_capacity),
            parent_index: Vec::with_capacity(state_capacity),
            fingerprints: Vec::new(),
            generated: 0,
            parents_processed: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: false,
        }
    }

    /// Number of i64 slots per state.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Number of successors currently stored.
    #[must_use]
    pub fn successor_count(&self) -> usize {
        self.parent_index.len()
    }

    /// Total generated successors before local dedup.
    #[must_use]
    pub fn generated(&self) -> u64 {
        self.generated
    }

    /// Number of parent states fully processed.
    #[must_use]
    pub fn parents_processed(&self) -> usize {
        self.parents_processed
    }

    /// Invariant status for this arena.
    #[must_use]
    pub fn invariant_status(&self) -> TrustCgInvariantStatus {
        self.invariant
    }

    /// Whether this arena carries complete per-parent raw-successor metadata.
    #[must_use]
    pub fn raw_successor_metadata_complete(&self) -> bool {
        self.raw_successor_metadata_complete
    }

    /// First parent that generated no raw successors, if metadata is complete.
    #[must_use]
    pub fn first_parent_without_raw_successors(&self) -> Option<usize> {
        if !self.raw_successor_metadata_complete {
            return None;
        }
        self.first_parent_without_raw_successors
            .and_then(|parent_idx| usize::try_from(parent_idx).ok())
    }

    /// Raw flat successor state buffer.
    #[must_use]
    pub fn states_flat(&self) -> &[i64] {
        &self.states
    }

    /// Mutable raw flat successor state buffer.
    #[must_use]
    pub fn states_flat_mut(&mut self) -> &mut [i64] {
        self.clear_successor_fingerprints();
        &mut self.states
    }

    /// Parent index for every successor in this arena.
    #[must_use]
    pub fn parent_indices(&self) -> &[u32] {
        &self.parent_index
    }

    /// Compiled-flat fingerprint sidecar for each successor, when available.
    #[must_use]
    pub fn successor_fingerprints(&self) -> &[u64] {
        if self.fingerprints.len() == self.successor_count() {
            &self.fingerprints
        } else {
            &[]
        }
    }

    /// Clear the compiled-flat fingerprint sidecar without touching successor states.
    ///
    /// Callers that mutate raw successor buffers after native generation must
    /// invalidate these fingerprints before admitting the states.
    pub fn clear_successor_fingerprints(&mut self) {
        self.fingerprints.clear();
    }

    /// Reserve fingerprint-sidecar capacity for up to `state_capacity` successors.
    ///
    /// Lazy-fingerprint allocation lifecycle: the sidecar is reserved on demand
    /// from [`Self::prepare_abi`] rather than eagerly at construction, so an
    /// arena that is allocated but never used by a fingerprint-writing native
    /// level (e.g. one parked in a successor-arena pool, or a zero-capacity
    /// shape) never carries a dead `u64`-per-successor allocation.
    ///
    /// On the native fused path the sidecar IS load-bearing — the native parent
    /// loop writes a fingerprint per emitted successor through the ABI pointer
    /// and [`Self::commit_abi`] rejects a null/short sidecar — so we must
    /// materialize the reservation before exposing the pointer. After this call
    /// the sidecar is materialized (its backing allocation exists) and
    /// [`Self::fingerprints_materialized`] holds for any subsequent
    /// [`Self::successor_fingerprints`]/liveness read.
    fn reserve_fingerprints(&mut self, state_capacity: usize) {
        self.fingerprints.reserve(state_capacity);
    }

    /// Whether the fingerprint sidecar's backing allocation has been
    /// materialized (capacity reserved or entries already pushed).
    ///
    /// Used by debug-time lazy-materialize guards: any liveness /
    /// `fingerprint_values()` read must observe a materialized sidecar so the
    /// deferred allocation can never leave a fingerprint read reading from a
    /// not-yet-allocated buffer.
    #[must_use]
    fn fingerprints_materialized(&self, state_capacity: usize) -> bool {
        state_capacity == 0 || self.fingerprints.capacity() >= state_capacity
    }

    /// Reset the arena without releasing allocations.
    pub fn clear(&mut self) {
        self.states.clear();
        self.parent_index.clear();
        self.fingerprints.clear();
        self.generated = 0;
        self.parents_processed = 0;
        self.invariant = TrustCgInvariantStatus::Passed;
        self.first_parent_without_raw_successors = None;
        self.raw_successor_metadata_complete = false;
    }

    /// Return one successor by index.
    #[must_use]
    pub fn successor(&self, idx: usize) -> Option<&[i64]> {
        if idx >= self.successor_count() {
            return None;
        }
        if self.state_len == 0 {
            return Some(&[]);
        }
        let start = idx.checked_mul(self.state_len)?;
        let end = start.checked_add(self.state_len)?;
        self.states.get(start..end)
    }

    /// Iterate over successor state slices.
    #[must_use]
    pub fn iter_successors(&self) -> TrustCgSuccessorIter<'_> {
        if self.state_len == 0 {
            TrustCgSuccessorIter::Empty(self.successor_count())
        } else {
            TrustCgSuccessorIter::Chunked(self.states.chunks_exact(self.state_len))
        }
    }

    /// Append one successor state.
    pub fn push_successor(
        &mut self,
        parent_idx: u32,
        state: &[i64],
    ) -> Result<u32, TrustCgBfsLevelError> {
        if state.len() != self.state_len {
            return Err(TrustCgBfsLevelError::StateLenMismatch {
                expected: self.state_len,
                actual: state.len(),
            });
        }
        let state_len_u32 = usize_to_u32(self.state_len, "state_len")?;
        let fingerprint = fingerprint_state(state, state_len_u32);
        self.push_successor_with_fingerprint(parent_idx, state, fingerprint)
    }

    /// Append one successor state with its already-computed compiled-flat fingerprint.
    pub fn push_successor_with_fingerprint(
        &mut self,
        parent_idx: u32,
        state: &[i64],
        fingerprint: u64,
    ) -> Result<u32, TrustCgBfsLevelError> {
        if state.len() != self.state_len {
            return Err(TrustCgBfsLevelError::StateLenMismatch {
                expected: self.state_len,
                actual: state.len(),
            });
        }
        let idx = usize_to_u32(self.successor_count(), "successor_count")?;
        self.states.extend_from_slice(state);
        self.parent_index.push(parent_idx);
        self.fingerprints.push(fingerprint);
        Ok(idx)
    }

    /// Build an FFI arena view with room for `state_capacity` successors.
    ///
    /// The returned struct borrows this arena's allocations by raw pointer.
    /// Call [`Self::commit_abi`] after the native function returns.
    pub fn prepare_abi(
        &mut self,
        state_capacity: usize,
    ) -> Result<TrustCgBfsSuccessorArenaAbi, TrustCgBfsLevelError> {
        self.clear();
        // Clear-before-reuse soundness guard. This arena may have been recycled
        // from a successor-arena pool across BFS levels/actions. Before we hand
        // its raw allocations to native code through the ABI below, prove that
        // `clear()` left no stale successor states, parent indexes, or
        // fingerprints behind, so a reused arena can NEVER alias prior-level
        // successor/handle data into a fresh native call. This is exactly the
        // aliasing class behind the MCLamportMutex native SIGSEGV: stale data
        // surfacing through a pointer the native loop trusts to be fresh.
        debug_assert_eq!(
            self.states.len(),
            0,
            "trust-cg successor arena reused without a clear: stale successor states present"
        );
        debug_assert_eq!(
            self.parent_index.len(),
            0,
            "trust-cg successor arena reused without a clear: stale parent indexes present"
        );
        debug_assert_eq!(
            self.fingerprints.len(),
            0,
            "trust-cg successor arena reused without a clear: stale fingerprints present"
        );
        let slot_capacity = state_capacity
            .checked_mul(self.state_len)
            .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
        let state_capacity_u32 = usize_to_u32(state_capacity, "state_capacity")?;
        let state_len_u32 = usize_to_u32(self.state_len, "state_len")?;

        // Pool-reuse allocation lifecycle: `reserve` only grows the backing
        // allocation when the (cleared, capacity-retaining) Vec is too small.
        // For a pooled arena warmed by a prior level this is a no-op, so
        // recycling avoids the per-level dealloc+realloc churn a fresh arena
        // would pay. The fingerprint sidecar is reserved lazily by
        // `reserve_fingerprints` so a fingerprint-free arena shape never
        // allocates a u64-per-successor sidecar it will not use.
        self.states.reserve(slot_capacity);
        self.parent_index.reserve(state_capacity);
        self.reserve_fingerprints(state_capacity);

        // Lazy-materialize guard: the ABI struct below publishes a raw
        // `fingerprints` pointer that the native level dereferences to store one
        // fingerprint per successor. The deferred sidecar reservation MUST be
        // materialized before that pointer is exposed, otherwise a native write
        // (and the later `successor_fingerprints()`/liveness read) would touch a
        // not-yet-allocated buffer.
        debug_assert!(
            self.fingerprints_materialized(state_capacity),
            "fingerprint sidecar must be materialized before its ABI pointer is exposed"
        );

        Ok(TrustCgBfsSuccessorArenaAbi {
            abi_version: TRUST_CG_BFS_LEVEL_ABI_VERSION,
            state_len: state_len_u32,
            state_capacity: state_capacity_u32,
            state_count: 0,
            states: if slot_capacity == 0 {
                std::ptr::null_mut()
            } else {
                self.states.as_mut_ptr()
            },
            parent_index: if state_capacity == 0 {
                std::ptr::null_mut()
            } else {
                self.parent_index.as_mut_ptr()
            },
            fingerprints: if state_capacity == 0 {
                std::ptr::null_mut()
            } else {
                self.fingerprints.as_mut_ptr()
            },
            generated: 0,
            parents_processed: 0,
            invariant_ok: 1,
            status: TrustCgBfsLevelStatus::Ok.as_raw(),
            failed_parent_idx: TRUST_CG_BFS_NO_INDEX,
            failed_invariant_idx: TRUST_CG_BFS_NO_INDEX,
            failed_successor_idx: TRUST_CG_BFS_NO_INDEX,
            first_zero_generated_parent_idx: TRUST_CG_BFS_NO_INDEX,
            raw_successor_metadata_complete: 0,
        })
    }

    /// Commit counts and status written through an ABI arena view.
    ///
    /// # Safety
    ///
    /// `abi` must be the value returned by the last call to
    /// [`Self::prepare_abi`] for this arena, and native code must have
    /// initialized exactly `abi.state_count * self.state_len` i64 slots,
    /// `abi.state_count` parent-index slots, and `abi.state_count`
    /// fingerprint slots.
    pub unsafe fn commit_abi(
        &mut self,
        abi: &TrustCgBfsSuccessorArenaAbi,
    ) -> Result<TrustCgBfsLevelOutcome, TrustCgBfsLevelError> {
        if abi.abi_version != TRUST_CG_BFS_LEVEL_ABI_VERSION {
            return Err(TrustCgBfsLevelError::InvalidAbiVersion {
                expected: TRUST_CG_BFS_LEVEL_ABI_VERSION,
                actual: abi.abi_version,
            });
        }
        let abi_state_len = abi.state_len as usize;
        if abi_state_len != self.state_len {
            return Err(TrustCgBfsLevelError::StateLenMismatch {
                expected: self.state_len,
                actual: abi_state_len,
            });
        }
        let status = match TrustCgBfsLevelStatus::from_raw(abi.status) {
            Some(status) => status,
            None => return Err(status_error(abi.status, abi.state_count)),
        };
        if matches!(
            status,
            TrustCgBfsLevelStatus::RuntimeError
                | TrustCgBfsLevelStatus::FallbackNeeded
                | TrustCgBfsLevelStatus::InvalidAbi
        ) {
            return Err(status_error(abi.status, abi.state_count));
        }
        if abi.state_count > abi.state_capacity {
            return Err(TrustCgBfsLevelError::BufferOverflow {
                partial_count: abi.state_capacity,
            });
        }
        if abi.generated < u64::from(abi.state_count) {
            return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                "generated count {} is below state_count {}",
                abi.generated, abi.state_count
            )));
        }
        let count = abi.state_count as usize;
        let slots = count
            .checked_mul(self.state_len)
            .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
        if slots > self.states.capacity()
            || count > self.parent_index.capacity()
            || count > self.fingerprints.capacity()
        {
            return Err(TrustCgBfsLevelError::InvalidAbi(
                "native result exceeds arena allocation".to_string(),
            ));
        }
        if slots > 0 && abi.states.is_null() {
            return Err(TrustCgBfsLevelError::InvalidAbi(
                "native result omitted successor state arena".to_string(),
            ));
        }
        if count > 0 && abi.fingerprints.is_null() {
            return Err(TrustCgBfsLevelError::InvalidAbi(
                "native result omitted successor fingerprint sidecar".to_string(),
            ));
        }
        if count > 0 && abi.parent_index.is_null() {
            return Err(TrustCgBfsLevelError::InvalidAbi(
                "native result omitted successor parent-index sidecar".to_string(),
            ));
        }
        let invariant_ok = decode_native_result_bool_field(abi.invariant_ok, "invariant_ok")?;
        let raw_successor_metadata_complete = decode_native_result_bool_field(
            abi.raw_successor_metadata_complete,
            "raw_successor_metadata_complete",
        )?;

        // SAFETY: guaranteed by this method's safety contract and validated
        // against the Vec capacities above.
        unsafe {
            self.states.set_len(slots);
            self.parent_index.set_len(count);
            self.fingerprints.set_len(count);
        }
        self.generated = abi.generated;
        self.parents_processed = abi.parents_processed as usize;
        self.invariant = invariant_from_abi(abi, invariant_ok);
        self.first_parent_without_raw_successors =
            if abi.first_zero_generated_parent_idx == TRUST_CG_BFS_NO_INDEX {
                None
            } else {
                Some(abi.first_zero_generated_parent_idx)
            };
        self.raw_successor_metadata_complete = raw_successor_metadata_complete;

        status_to_result(status.as_raw(), abi.state_count)?;
        Ok(self.outcome())
    }

    fn set_generated(&mut self, generated: u64) {
        self.generated = generated;
    }

    fn set_parents_processed(&mut self, parents_processed: usize) {
        self.parents_processed = parents_processed;
    }

    fn fail_invariant(&mut self, parent_idx: u32, invariant_idx: u32, successor_idx: u32) {
        self.invariant = TrustCgInvariantStatus::Failed {
            parent_index: parent_idx,
            invariant_index: invariant_idx,
            successor_index: successor_idx,
        };
    }

    fn note_parent_without_raw_successors(&mut self, parent_idx: u32) {
        if self.first_parent_without_raw_successors.is_none() {
            self.first_parent_without_raw_successors = Some(parent_idx);
        }
    }

    fn set_raw_successor_metadata_complete(&mut self) {
        self.raw_successor_metadata_complete = true;
    }

    fn outcome(&self) -> TrustCgBfsLevelOutcome {
        TrustCgBfsLevelOutcome {
            parents_processed: self.parents_processed,
            total_generated: self.generated,
            total_new: self.successor_count() as u64,
            invariant: self.invariant,
            first_parent_without_raw_successors: self.first_parent_without_raw_successors(),
            raw_successor_metadata_complete: self.raw_successor_metadata_complete,
        }
    }
}

/// Iterator over flat successor slices.
pub enum TrustCgSuccessorIter<'a> {
    /// Walks a populated arena as fixed-width `state_len`-slot chunks.
    Chunked(std::slice::ChunksExact<'a, i64>),
    /// Yields the empty slice the given number of times (zero-width states).
    Empty(usize),
}

impl<'a> Iterator for TrustCgSuccessorIter<'a> {
    type Item = &'a [i64];

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Chunked(chunks) => chunks.next(),
            Self::Empty(remaining) => {
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
            Self::Chunked(chunks) => chunks.size_hint(),
            Self::Empty(remaining) => (*remaining, Some(*remaining)),
        }
    }
}

impl ExactSizeIterator for TrustCgSuccessorIter<'_> {}

/// Aggregate result of one fused/prototype BFS level run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustCgBfsLevelOutcome {
    /// Number of parents fully processed.
    pub parents_processed: usize,
    /// Total successors accepted by compiled constraints before local dedup.
    pub total_generated: u64,
    /// Successors emitted into the contiguous arena after local dedup.
    pub total_new: u64,
    /// Invariant status for this level.
    pub invariant: TrustCgInvariantStatus,
    /// First parent that generated no raw successors, when reported.
    pub first_parent_without_raw_successors: Option<usize>,
    /// Whether `first_parent_without_raw_successors` is complete for the
    /// processed parent prefix represented by this outcome.
    pub raw_successor_metadata_complete: bool,
}

/// Capability bits for the current trust-codegen BFS level foundation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustCgBfsLevelCapabilities {
    /// Accepts row-major flat parent arenas.
    pub flat_parent_arena_abi: bool,
    /// Emits row-major flat successor arenas.
    pub contiguous_successor_arena: bool,
    /// Emits optional source parent indexes with successors.
    pub parent_index_metadata: bool,
    /// Reports invariant failures with stable invariant indexes.
    pub invariant_status: bool,
    /// Performs local dedup before successor arena insertion.
    pub local_dedup: bool,
    /// Applies compiled state-constraint predicates before local dedup and
    /// successor arena admission.
    pub state_constraints: bool,
    /// Parent loop is compiled into a native trust-codegen fused function.
    pub native_fused_loop: bool,
}

impl TrustCgBfsLevelCapabilities {
    /// Capabilities for the Rust prototype runner in this module.
    pub const PROTOTYPE: Self = Self {
        flat_parent_arena_abi: true,
        contiguous_successor_arena: true,
        parent_index_metadata: true,
        invariant_status: true,
        local_dedup: true,
        state_constraints: false,
        native_fused_loop: false,
    };

    /// Capabilities for a resolved native trust-codegen fused-level entry point.
    pub const NATIVE_FUSED: Self = Self {
        flat_parent_arena_abi: true,
        contiguous_successor_arena: true,
        parent_index_metadata: true,
        invariant_status: true,
        local_dedup: true,
        state_constraints: false,
        native_fused_loop: true,
    };

    /// Capabilities for the first native trust-codegen fused-level parent loop.
    ///
    /// The loop is native and uses the flat arena ABI, but local fingerprint
    /// dedup is not fused into that generated loop yet.
    pub const NATIVE_FUSED_NO_LOCAL_DEDUP: Self = Self {
        flat_parent_arena_abi: true,
        contiguous_successor_arena: true,
        parent_index_metadata: true,
        invariant_status: true,
        local_dedup: false,
        state_constraints: false,
        native_fused_loop: true,
    };
}

/// Capacity and telemetry metadata for a native trust-codegen fused BFS level.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrustCgBfsLevelNativeMetadata {
    /// Number of action functions compiled into the native parent loop.
    pub action_count: usize,
    /// Number of state-constraint predicates compiled into the native parent loop.
    pub state_constraint_count: usize,
    /// Number of invariant functions compiled into the native parent loop.
    pub invariant_count: usize,
    /// Maximum successors one parent can emit before global/frontier dedup.
    pub max_successors_per_parent: usize,
    /// Whether the native parent loop performs local fingerprint dedup before
    /// successor arena insertion.
    pub local_dedup: bool,
    /// Route B: true when at least one compiled action is a multi-successor
    /// record-set kernel ([`tla_jit_abi::NextStateLoopFn`] sink ABI). Such a
    /// kernel emits a RUNTIME-dependent number of successors per parent, so
    /// the single-successor telemetry bound (`generated <= parents_processed *
    /// action_count`) and the static `max_successors_per_parent` capacity are
    /// no longer upper bounds; callers grow the successor arena adaptively on
    /// [`TrustCgBfsLevelError::BufferOverflow`] instead.
    pub has_loop_actions: bool,
}

impl TrustCgBfsLevelNativeMetadata {
    /// Build native fused-level metadata.
    #[must_use]
    pub fn new(
        action_count: usize,
        invariant_count: usize,
        max_successors_per_parent: usize,
        local_dedup: bool,
    ) -> Self {
        Self {
            action_count,
            state_constraint_count: 0,
            invariant_count,
            max_successors_per_parent,
            local_dedup,
            has_loop_actions: false,
        }
    }

    /// Build native fused-level metadata for a loop that applies compiled
    /// state constraints before local dedup and successor arena admission.
    #[must_use]
    pub fn new_with_state_constraints(
        action_count: usize,
        state_constraint_count: usize,
        invariant_count: usize,
        max_successors_per_parent: usize,
        local_dedup: bool,
    ) -> Self {
        Self {
            action_count,
            state_constraint_count,
            invariant_count,
            max_successors_per_parent,
            local_dedup,
            has_loop_actions: false,
        }
    }

    /// Mark this level as containing Route B multi-successor record-set
    /// kernels (see [`Self::has_loop_actions`]).
    #[must_use]
    pub fn with_loop_actions(mut self, has_loop_actions: bool) -> Self {
        self.has_loop_actions = has_loop_actions;
        self
    }

    /// Capability bits represented by this metadata.
    #[must_use]
    pub fn capabilities(&self) -> TrustCgBfsLevelCapabilities {
        TrustCgBfsLevelCapabilities {
            local_dedup: self.local_dedup,
            state_constraints: self.state_constraint_count > 0,
            ..TrustCgBfsLevelCapabilities::NATIVE_FUSED_NO_LOCAL_DEDUP
        }
    }

    /// Compute the arena capacity required for `parent_count`.
    pub fn successor_capacity_for(
        &self,
        parent_count: usize,
    ) -> Result<usize, TrustCgBfsLevelError> {
        parent_count
            .checked_mul(self.max_successors_per_parent)
            .ok_or(TrustCgBfsLevelError::CapacityOverflow)
    }

    /// Validate native-reported counters against wrapper-owned execution bounds.
    ///
    /// `TrustCgSuccessorArena::commit_abi` only knows ABI-local facts. The native
    /// fused wrapper knows the parent batch size and compiled action count, so
    /// it must reject impossible telemetry before callers turn it into model
    /// checker transition statistics.
    pub fn validate_reported_counters(
        &self,
        parent_count: usize,
        outcome: &TrustCgBfsLevelOutcome,
    ) -> Result<(), TrustCgBfsLevelError> {
        if outcome.parents_processed > parent_count {
            return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                "native parents_processed {} exceeds parent_count {}",
                outcome.parents_processed, parent_count
            )));
        }
        match outcome.invariant {
            TrustCgInvariantStatus::Passed => {
                if outcome.parents_processed != parent_count {
                    return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                        "native parents_processed {} must equal parent_count {} when invariants pass",
                        outcome.parents_processed, parent_count
                    )));
                }
            }
            TrustCgInvariantStatus::Failed { parent_index, .. } => {
                let parent_index = usize::try_from(parent_index)
                    .map_err(|_| TrustCgBfsLevelError::CapacityOverflow)?;
                let expected = parent_index
                    .checked_add(1)
                    .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
                if expected > parent_count {
                    return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                        "native invariant failure parent index {parent_index} is outside parent_count {parent_count}",
                    )));
                }
                if outcome.parents_processed != expected {
                    return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                        "native parents_processed {} must equal failed parent prefix {}",
                        outcome.parents_processed, expected
                    )));
                }
            }
        }
        if outcome.first_parent_without_raw_successors.is_some()
            && !outcome.raw_successor_metadata_complete
        {
            return Err(TrustCgBfsLevelError::InvalidAbi(
                "native first_zero_generated_parent_idx was reported without complete raw-successor metadata"
                    .to_string(),
            ));
        }
        if let Some(first_zero_parent) = outcome.first_parent_without_raw_successors {
            if first_zero_parent >= outcome.parents_processed {
                return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                    "native first_zero_generated_parent_idx {} is outside processed parent prefix {}",
                    first_zero_parent, outcome.parents_processed
                )));
            }
        }
        // Route B: a record-set loop kernel legitimately emits a
        // runtime-dependent number of successors per parent, so the
        // single-successor `parents * actions` telemetry ceiling does not
        // apply. Committed successors are still bounded by the arena capacity
        // (`commit_abi` rejects `state_count > state_capacity`), so skipping
        // only this generated-count plausibility check loses no soundness.
        if !self.has_loop_actions {
            let max_generated = outcome
                .parents_processed
                .checked_mul(self.action_count)
                .and_then(|count| u64::try_from(count).ok())
                .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
            if outcome.total_generated > max_generated {
                return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                    "native generated count {} exceeds single-successor action upper bound {} \
                     (processed_parents={}, actions={})",
                    outcome.total_generated,
                    max_generated,
                    outcome.parents_processed,
                    self.action_count
                )));
            }
        }
        Ok(())
    }
}

/// A pre-compiled trust-codegen next-state action and its descriptor.
///
/// `func` is a cached raw JIT pointer. Direct callers must keep the owning
/// native library alive, publish the exact owner, and call
/// [`crate::ensure_jit_execute_mode`] on the invoking thread immediately
/// before invoking it.
#[derive(Clone)]
pub struct TrustCgCompiledAction {
    /// Descriptor for this specialized action instance.
    pub descriptor: ActionDescriptor,
    /// Native trust-codegen next-state function pointer.
    ///
    /// For a Route B multi-successor action (`loop_fn.is_some()`) this holds
    /// the SAME address as `loop_fn`, reinterpreted only for Debug/publication
    /// bookkeeping. It MUST NEVER be invoked through the single-successor call
    /// convention in that case: the kernel would treat the `state_out` buffer
    /// as a `*mut NextStateLoopSink` header and scribble sink-header stores
    /// over live arena memory. [`TrustCgBfsLevelPrototype::run_level_arena`]
    /// enforces this by branching on [`Self::loop_fn`] BEFORE any call.
    pub func: JitNextStateFn,
    /// Route B multi-successor kernel entry point. `Some` when this action
    /// implements the [`tla_jit_abi::NextStateLoopFn`] sink ABI (arg#2 is a
    /// `*mut NextStateLoopSink` instead of a single `state_out` buffer).
    loop_fn: Option<tla_jit_abi::NextStateLoopFn>,
    owner_library: Option<NativeLibrary>,
    symbol_name: Option<String>,
}

impl TrustCgCompiledAction {
    /// Create a compiled action wrapper.
    #[must_use]
    pub fn new(descriptor: ActionDescriptor, func: JitNextStateFn) -> Self {
        Self {
            descriptor,
            func,
            loop_fn: None,
            owner_library: None,
            symbol_name: None,
        }
    }

    /// Create a compiled Route B multi-successor action wrapper.
    ///
    /// `loop_fn` implements the [`tla_jit_abi::NextStateLoopFn`] sink ABI.
    /// `func` is set to the same address (fn pointers of identical size)
    /// purely so Debug output and owner-library publication cover the loop
    /// kernel; every invocation is routed through the sink convention because
    /// `loop_fn` is `Some`.
    #[must_use]
    pub fn new_loop(descriptor: ActionDescriptor, loop_fn: tla_jit_abi::NextStateLoopFn) -> Self {
        // SAFETY: both are `unsafe extern "C" fn` pointers of identical size;
        // the transmuted copy is only used as an address for Debug formatting
        // and exact-owner symbol publication, never called through this type.
        let func =
            unsafe { std::mem::transmute::<tla_jit_abi::NextStateLoopFn, JitNextStateFn>(loop_fn) };
        Self {
            descriptor,
            func,
            loop_fn: Some(loop_fn),
            owner_library: None,
            symbol_name: None,
        }
    }

    /// The Route B multi-successor kernel entry point, when this action uses
    /// the [`tla_jit_abi::NextStateLoopFn`] sink ABI.
    #[must_use]
    pub fn loop_fn(&self) -> Option<tla_jit_abi::NextStateLoopFn> {
        self.loop_fn
    }

    /// True when this action must be dispatched via the sink call convention.
    #[must_use]
    pub fn is_loop(&self) -> bool {
        self.loop_fn.is_some()
    }

    /// Attach the native library that owns `func` so cached calls can
    /// re-publish the exact JIT buffer before invocation.
    #[must_use]
    pub fn with_owner_library(mut self, library: NativeLibrary) -> Self {
        self.owner_library = Some(library);
        self
    }

    /// Attach the native library and exact symbol name that own `func`.
    #[must_use]
    pub fn with_owner_library_and_symbol(
        mut self,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Self {
        self.owner_library = Some(library);
        self.symbol_name = Some(symbol_name.into());
        self
    }

    fn ensure_published_executable(&self) -> Result<(), TrustCgError> {
        if let Some(library) = &self.owner_library {
            if let Some(symbol_name) = &self.symbol_name {
                library.ensure_published_symbol_ptr(
                    symbol_name,
                    self.func as *const () as *mut std::ffi::c_void,
                )
            } else {
                library.ensure_published_executable()
            }
        } else {
            #[cfg(feature = "native")]
            crate::ensure_jit_execute_mode();
            Ok(())
        }
    }
}

impl fmt::Debug for TrustCgCompiledAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrustCgCompiledAction")
            .field("descriptor", &self.descriptor)
            .field("func", &(self.func as usize))
            .field("is_loop", &self.loop_fn.is_some())
            .finish()
    }
}

/// A pre-compiled trust-codegen invariant and its descriptor.
///
/// `func` is a cached raw JIT pointer. Direct callers must keep the owning
/// native library alive, publish the exact owner, and call
/// [`crate::ensure_jit_execute_mode`] on the invoking thread immediately
/// before invoking it.
#[derive(Clone)]
pub struct TrustCgCompiledInvariant {
    /// Descriptor for this invariant.
    pub descriptor: InvariantDescriptor,
    /// Native trust-codegen invariant function pointer.
    pub func: JitInvariantFn,
    owner_library: Option<NativeLibrary>,
    symbol_name: Option<String>,
}

impl TrustCgCompiledInvariant {
    /// Create a compiled invariant wrapper.
    #[must_use]
    pub fn new(descriptor: InvariantDescriptor, func: JitInvariantFn) -> Self {
        Self {
            descriptor,
            func,
            owner_library: None,
            symbol_name: None,
        }
    }

    /// Attach the native library that owns `func` so cached calls can
    /// re-publish the exact JIT buffer before invocation.
    #[must_use]
    pub fn with_owner_library(mut self, library: NativeLibrary) -> Self {
        self.owner_library = Some(library);
        self
    }

    /// Attach the native library and exact symbol name that own `func`.
    #[must_use]
    pub fn with_owner_library_and_symbol(
        mut self,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Self {
        self.owner_library = Some(library);
        self.symbol_name = Some(symbol_name.into());
        self
    }

    fn ensure_published_executable(&self) -> Result<(), TrustCgError> {
        if let Some(library) = &self.owner_library {
            if let Some(symbol_name) = &self.symbol_name {
                library.ensure_published_symbol_ptr(
                    symbol_name,
                    self.func as *const () as *mut std::ffi::c_void,
                )
            } else {
                library.ensure_published_executable()
            }
        } else {
            #[cfg(feature = "native")]
            crate::ensure_jit_execute_mode();
            Ok(())
        }
    }
}

impl fmt::Debug for TrustCgCompiledInvariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrustCgCompiledInvariant")
            .field("descriptor", &self.descriptor)
            .field("func", &(self.func as usize))
            .finish()
    }
}

/// Initial host-side Route B sink capacity in SUCCESSORS (not slots). The
/// buffer starts at `state_len * this` i64 slots and doubles on overflow.
const NEXT_STATE_LOOP_HOST_INITIAL_SUCCESSORS: usize = 64;

/// Hard cap (in i64 slots) for the host-side Route B sink buffer. A kernel
/// that still overflows at this capacity fails closed to the interpreter
/// (`TrustCgBfsLevelError::FallbackNeeded`) — a SOUND fallback, never a
/// truncated successor set.
const NEXT_STATE_LOOP_HOST_MAX_SLOTS: usize = 1 << 22;

/// Rust prototype for trust-codegen fused BFS level execution over flat arenas.
///
/// This owns the same inputs that a final native fused level will specialize:
/// state length, compiled action pointers, compiled invariant pointers, and a
/// local fingerprint set. The hot parent loop is still Rust here; the data ABI
/// and successor arena contract are the pieces this issue needs in place for
/// `tla-check` wiring.
pub struct TrustCgBfsLevelPrototype {
    state_len: usize,
    actions: Vec<TrustCgCompiledAction>,
    invariants: Vec<TrustCgCompiledInvariant>,
    local_seen: AtomicFpSet,
    scratch: Vec<i64>,
    /// Reusable flat successor buffer for Route B multi-successor
    /// ([`tla_jit_abi::NextStateLoopFn`]) actions. Grows on sink overflow and
    /// is retained across parents/levels; empty until the first loop action
    /// call. See [`Self::run_level_arena`] for the overflow/regrow protocol.
    loop_succ_buf: Vec<i64>,
}

impl fmt::Debug for TrustCgBfsLevelPrototype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrustCgBfsLevelPrototype")
            .field("state_len", &self.state_len)
            .field("actions", &self.actions)
            .field("invariants", &self.invariants)
            .finish_non_exhaustive()
    }
}

impl TrustCgBfsLevelPrototype {
    /// Build a prototype fused-level runner.
    pub fn new(
        state_len: usize,
        actions: Vec<TrustCgCompiledAction>,
        invariants: Vec<TrustCgCompiledInvariant>,
        expected_states: usize,
    ) -> Result<Self, TrustCgBfsLevelError> {
        let _ = usize_to_u32(state_len, "state_len")?;
        Ok(Self {
            state_len,
            actions,
            invariants,
            local_seen: AtomicFpSet::new(expected_states.saturating_mul(2).max(1)),
            scratch: vec![0; state_len],
            loop_succ_buf: Vec::new(),
        })
    }

    /// Return the current implementation capability bits.
    #[must_use]
    pub fn capabilities(&self) -> TrustCgBfsLevelCapabilities {
        TrustCgBfsLevelCapabilities::PROTOTYPE
    }

    /// Number of i64 slots per flat state.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Number of compiled action instances.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Number of compiled invariants.
    #[must_use]
    pub fn invariant_count(&self) -> usize {
        self.invariants.len()
    }

    /// Pre-seed the local dedup set with fingerprints from already-known
    /// states. Returns the number of newly inserted fingerprints.
    pub fn preseed_fingerprints<I>(&self, fingerprints: I) -> usize
    where
        I: IntoIterator<Item = u64>,
    {
        self.local_seen.preseed(fingerprints)
    }

    /// Process one flat parent frontier level.
    ///
    /// `parents` must contain `parent_count * state_len` i64 values in
    /// row-major order. Successors are appended to `out` as borrowed-state
    /// copies in one contiguous arena. Local duplicates are filtered before
    /// insertion, so duplicate successors allocate no per-successor `Vec`.
    pub fn run_level_arena(
        &mut self,
        parents: &[i64],
        parent_count: usize,
        out: &mut TrustCgSuccessorArena,
    ) -> Result<TrustCgBfsLevelOutcome, TrustCgBfsLevelError> {
        validate_arena_len(parents.len(), parent_count, self.state_len)?;
        if out.state_len() != self.state_len {
            return Err(TrustCgBfsLevelError::StateLenMismatch {
                expected: self.state_len,
                actual: out.state_len(),
            });
        }

        let state_len_u32 = usize_to_u32(self.state_len, "state_len")?;
        out.clear();
        // Per-invocation local dedup, mirroring the native wrapper's
        // `reset_local_seen_for_native_invocation`: a stale set from a prior
        // call (e.g. the sampled native-vs-interpreter crosscheck PEEK-run of
        // this same level) would misclassify this run's successors as local
        // duplicates and silently drop them from the arena.
        if !self.local_seen.is_empty() {
            self.local_seen = AtomicFpSet::new(self.local_seen.capacity());
        }

        if self.state_len == 0 {
            if parent_count > 0 {
                out.note_parent_without_raw_successors(0);
            }
            out.set_parents_processed(parent_count);
            out.set_raw_successor_metadata_complete();
            return Ok(out.outcome());
        }

        let mut generated = 0u64;
        let mut parents_processed = 0usize;

        for parent_idx in 0..parent_count {
            let parent_idx_u32 = usize_to_u32(parent_idx, "parent_idx")?;
            let start = parent_idx
                .checked_mul(self.state_len)
                .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
            let end = start
                .checked_add(self.state_len)
                .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
            let parent = &parents[start..end];
            let mut parent_generated = 0u64;

            for action in &self.actions {
                if let Some(loop_fn) = action.loop_fn {
                    // ---- Route B host-side sink dispatch ----
                    //
                    // The kernel is a pure function of `state_in`, so it is
                    // safe to re-call with a larger buffer after an overflow
                    // (the truncated pushes of the overflowed attempt are
                    // discarded wholesale — a partial successor set is
                    // unsound). `sink.count == 0` with `JitStatus::Ok` simply
                    // means the action is disabled for this parent; the loop
                    // ABI does NOT return the single-successor enabled
                    // boolean in `call_out.value` (the kernel only writes
                    // `out.status`), so no boolean decode happens here.
                    action
                        .ensure_published_executable()
                        .map_err(|err| TrustCgBfsLevelError::InvalidAbi(err.to_string()))?;
                    #[cfg(feature = "native")]
                    crate::ensure_jit_execute_mode();

                    // Largest slot count <= the hard cap that still holds an
                    // integral number of successor records.
                    let max_slots =
                        (NEXT_STATE_LOOP_HOST_MAX_SLOTS / self.state_len) * self.state_len;
                    if max_slots == 0 {
                        // One successor record alone exceeds the cap: fail
                        // closed to the interpreter.
                        return Err(TrustCgBfsLevelError::FallbackNeeded);
                    }
                    let initial_slots = self
                        .state_len
                        .saturating_mul(NEXT_STATE_LOOP_HOST_INITIAL_SUCCESSORS)
                        .min(max_slots);
                    if self.loop_succ_buf.len() < initial_slots {
                        self.loop_succ_buf.resize(initial_slots, 0);
                    }

                    let successor_count = loop {
                        let mut call_out = JitCallOut::default();
                        let mut sink = tla_jit_abi::NextStateLoopSink::new(
                            &mut self.loop_succ_buf,
                            self.state_len,
                        );
                        unsafe {
                            // SAFETY: `parent` is valid for `state_len_u32`
                            // i64 slots, `sink` describes a caller-owned
                            // buffer of `capacity` valid i64 slots, and the
                            // function pointer uses the stable
                            // `NextStateLoopFn` sink ABI (constructed only
                            // via `TrustCgCompiledAction::new_loop`).
                            (loop_fn)(&mut call_out, parent.as_ptr(), &mut sink, state_len_u32);
                        }
                        match call_out.status {
                            JitStatus::Ok => {}
                            JitStatus::FallbackNeeded | JitStatus::PartialPass => {
                                return Err(TrustCgBfsLevelError::FallbackNeeded);
                            }
                            JitStatus::RuntimeError => {
                                return Err(TrustCgBfsLevelError::RuntimeError);
                            }
                        }
                        if sink.overflowed() {
                            // Discard every push from this attempt and re-call
                            // with a doubled buffer; beyond the cap, fail
                            // closed to the interpreter (sound).
                            let current = self.loop_succ_buf.len();
                            if current >= max_slots {
                                return Err(TrustCgBfsLevelError::FallbackNeeded);
                            }
                            let grown = current.saturating_mul(2).min(max_slots);
                            self.loop_succ_buf.resize(grown, 0);
                            continue;
                        }
                        let used = sink
                            .count
                            .checked_mul(self.state_len)
                            .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
                        if used > self.loop_succ_buf.len() {
                            return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                                "native loop action callout name={} index={} parent_idx={parent_idx_u32} reported {} successors ({used} slots) exceeding sink capacity {} slots without flagging overflow",
                                action.descriptor.name,
                                action.descriptor.action_idx,
                                sink.count,
                                self.loop_succ_buf.len(),
                            )));
                        }
                        break sink.count;
                    };

                    for record_idx in 0..successor_count {
                        // In-bounds: `successor_count * state_len <= len` was
                        // proven above.
                        let start = record_idx * self.state_len;
                        let succ = &self.loop_succ_buf[start..start + self.state_len];

                        generated = generated
                            .checked_add(1)
                            .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
                        parent_generated = parent_generated
                            .checked_add(1)
                            .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;

                        let fp = fingerprint_state(succ, state_len_u32);
                        match self.local_seen.insert_if_absent(fp) {
                            InsertResult::Inserted => {}
                            InsertResult::AlreadyPresent => continue,
                            InsertResult::TableFull => {
                                return Err(TrustCgBfsLevelError::BufferOverflow {
                                    partial_count: out.successor_count() as u32,
                                });
                            }
                        }

                        let successor_idx =
                            out.push_successor_with_fingerprint(parent_idx_u32, succ, fp)?;

                        if let Some(invariant_idx) = first_failed_invariant(
                            &self.invariants,
                            succ,
                            state_len_u32,
                            parent_idx_u32,
                            successor_idx,
                        )? {
                            out.set_generated(generated);
                            out.set_parents_processed(
                                parents_processed
                                    .checked_add(1)
                                    .ok_or(TrustCgBfsLevelError::CapacityOverflow)?,
                            );
                            out.fail_invariant(parent_idx_u32, invariant_idx, successor_idx);
                            out.set_raw_successor_metadata_complete();
                            return Ok(out.outcome());
                        }
                    }
                    continue;
                }

                self.scratch.copy_from_slice(parent);
                let mut call_out = JitCallOut::default();
                action
                    .ensure_published_executable()
                    .map_err(|err| TrustCgBfsLevelError::InvalidAbi(err.to_string()))?;
                #[cfg(feature = "native")]
                crate::ensure_jit_execute_mode();
                unsafe {
                    // SAFETY: `parent` and `scratch` are valid for
                    // `state_len_u32` i64 slots, the function pointer uses
                    // the stable trust-codegen next-state ABI, and
                    // `action.loop_fn` is `None` (checked above), so the
                    // single-successor call convention is the correct one.
                    (action.func)(
                        &mut call_out,
                        parent.as_ptr(),
                        self.scratch.as_mut_ptr(),
                        state_len_u32,
                    );
                }

                match call_out.status {
                    JitStatus::Ok => {
                        let context = format!(
                            "action callout name={} index={} parent_idx={parent_idx_u32}",
                            action.descriptor.name, action.descriptor.action_idx,
                        );
                        if !decode_native_bool_callout_value(call_out.value, &context)? {
                            continue;
                        }
                    }
                    JitStatus::FallbackNeeded | JitStatus::PartialPass => {
                        return Err(TrustCgBfsLevelError::FallbackNeeded);
                    }
                    JitStatus::RuntimeError => return Err(TrustCgBfsLevelError::RuntimeError),
                }

                generated = generated
                    .checked_add(1)
                    .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
                parent_generated = parent_generated
                    .checked_add(1)
                    .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;

                let fp = fingerprint_state(&self.scratch, state_len_u32);
                match self.local_seen.insert_if_absent(fp) {
                    InsertResult::Inserted => {}
                    InsertResult::AlreadyPresent => continue,
                    InsertResult::TableFull => {
                        return Err(TrustCgBfsLevelError::BufferOverflow {
                            partial_count: out.successor_count() as u32,
                        });
                    }
                }

                let successor_idx =
                    out.push_successor_with_fingerprint(parent_idx_u32, &self.scratch, fp)?;

                if let Some(invariant_idx) = first_failed_invariant(
                    &self.invariants,
                    &self.scratch,
                    state_len_u32,
                    parent_idx_u32,
                    successor_idx,
                )? {
                    out.set_generated(generated);
                    out.set_parents_processed(
                        parents_processed
                            .checked_add(1)
                            .ok_or(TrustCgBfsLevelError::CapacityOverflow)?,
                    );
                    out.fail_invariant(parent_idx_u32, invariant_idx, successor_idx);
                    out.set_raw_successor_metadata_complete();
                    return Ok(out.outcome());
                }
            }

            if parent_generated == 0 {
                out.note_parent_without_raw_successors(parent_idx_u32);
            }
            parents_processed = parents_processed
                .checked_add(1)
                .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
        }

        out.set_generated(generated);
        out.set_parents_processed(parents_processed);
        out.set_raw_successor_metadata_complete();
        Ok(out.outcome())
    }
}

#[derive(Clone)]
pub(crate) struct NativeCalloutPublicationTarget {
    library: NativeLibrary,
    symbol_name: String,
    address: usize,
}

impl NativeCalloutPublicationTarget {
    #[must_use]
    pub(crate) fn new(
        library: NativeLibrary,
        symbol_name: impl Into<String>,
        address: usize,
    ) -> Self {
        Self {
            library,
            symbol_name: symbol_name.into(),
            address,
        }
    }

    fn ensure_published_executable(&self) -> Result<(), TrustCgError> {
        self.library
            .ensure_published_symbol_ptr(&self.symbol_name, self.address as *mut std::ffi::c_void)
    }
}

/// Native trust-codegen fused BFS level wrapper.
///
/// Construction performs symbol lookup up front and owns the
/// [`NativeLibrary`] that backs the resolved function pointer. Therefore
/// `capabilities().native_fused_loop` can only become true after native lookup
/// has succeeded.
pub struct TrustCgBfsLevelNative {
    state_len: usize,
    symbol_name: String,
    entrypoint: TrustCgFusedLevelFn,
    library: NativeLibrary,
    metadata: TrustCgBfsLevelNativeMetadata,
    extern_libraries: Vec<NativeLibrary>,
    extern_callouts: Vec<NativeCalloutPublicationTarget>,
    scratch: Vec<i64>,
    // Single-thread, non-atomic dedup set. The native fused BFS level is only
    // ever driven by the sequential ModelChecker (ParallelChecker never
    // constructs or runs it, and every call is Mutex-serialized), so the local
    // fingerprint dedup needs no atomics or locks. See `SingleThreadFpSet`.
    local_seen: SingleThreadFpSet,
}

impl fmt::Debug for TrustCgBfsLevelNative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrustCgBfsLevelNative")
            .field("state_len", &self.state_len)
            .field("symbol_name", &self.symbol_name)
            .field("metadata", &self.metadata)
            .field("extern_library_count", &self.extern_libraries.len())
            .finish_non_exhaustive()
    }
}

impl TrustCgBfsLevelNative {
    /// Resolve `symbol_name` from a compiled native library and build a fused
    /// BFS level wrapper around it.
    pub fn from_library(
        state_len: usize,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Result<Self, TrustCgError> {
        Self::from_library_with_metadata(
            state_len,
            library,
            symbol_name,
            TrustCgBfsLevelNativeMetadata::default(),
        )
    }

    /// Resolve `symbol_name` and attach native fused-level metadata.
    pub fn from_library_with_metadata(
        state_len: usize,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
        metadata: TrustCgBfsLevelNativeMetadata,
    ) -> Result<Self, TrustCgError> {
        Self::from_library_with_metadata_and_dependencies(
            state_len,
            library,
            symbol_name,
            metadata,
            Vec::new(),
        )
    }

    /// Resolve `symbol_name`, attach metadata, and keep dependent native
    /// libraries alive for extern calls made by the fused parent loop.
    pub(crate) fn from_library_with_metadata_and_dependencies(
        state_len: usize,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
        metadata: TrustCgBfsLevelNativeMetadata,
        extern_libraries: Vec<NativeLibrary>,
    ) -> Result<Self, TrustCgError> {
        Self::from_library_with_metadata_dependencies_and_callouts(
            state_len,
            library,
            symbol_name,
            metadata,
            extern_libraries,
            Vec::new(),
        )
    }

    /// Resolve `symbol_name`, attach metadata, keep dependent native libraries
    /// alive, and exact-publish their cached callout pointers before invocation.
    pub(crate) fn from_library_with_metadata_dependencies_and_callouts(
        state_len: usize,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
        metadata: TrustCgBfsLevelNativeMetadata,
        extern_libraries: Vec<NativeLibrary>,
        extern_callouts: Vec<NativeCalloutPublicationTarget>,
    ) -> Result<Self, TrustCgError> {
        usize_to_u32(state_len, "state_len")
            .map_err(|err| TrustCgError::InvalidModule(err.to_string()))?;
        let symbol_name = symbol_name.into();
        if symbol_name.is_empty() {
            return Err(TrustCgError::Loading(
                "trust-cg BFS native symbol name must not be empty".to_string(),
            ));
        }
        // SAFETY: symbol lookup only returns a raw address; this constructor
        // immediately casts it to the stable fused-level ABI and keeps
        // `library` owned by the wrapper for the pointer's executable lifetime.
        let raw = unsafe { library.get_symbol(&symbol_name)? };
        // SAFETY: callers must provide `symbol_name` for a function matching
        // `TrustCgFusedLevelFn`; construction fails before this point when the
        // symbol is absent.
        let entrypoint =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, TrustCgFusedLevelFn>(raw) };
        Ok(Self {
            state_len,
            symbol_name,
            entrypoint,
            library,
            metadata,
            extern_libraries,
            extern_callouts,
            scratch: vec![
                0;
                trust_cg_bfs_native_parent_scratch_layout(state_len)
                    .map_err(|err| TrustCgError::InvalidModule(err.to_string()))?
                    .total_slots
            ],
            local_seen: SingleThreadFpSet::new(1024),
        })
    }

    /// Return capability bits for a resolved native fused-loop wrapper.
    #[must_use]
    pub fn capabilities(&self) -> TrustCgBfsLevelCapabilities {
        self.metadata.capabilities()
    }

    /// Number of i64 slots per flat state.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Resolved native symbol name.
    #[must_use]
    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    /// Native fused-level capacity and telemetry metadata.
    #[must_use]
    pub fn metadata(&self) -> &TrustCgBfsLevelNativeMetadata {
        &self.metadata
    }

    /// Number of action functions compiled into the native parent loop.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.metadata.action_count
    }

    /// Number of invariant functions compiled into the native parent loop.
    #[must_use]
    pub fn invariant_count(&self) -> usize {
        self.metadata.invariant_count
    }

    /// Number of state-constraint predicates compiled into the native parent loop.
    #[must_use]
    pub fn state_constraint_count(&self) -> usize {
        self.metadata.state_constraint_count
    }

    /// Maximum successors one parent can emit before global/frontier dedup.
    #[must_use]
    pub fn max_successors_per_parent(&self) -> usize {
        self.metadata.max_successors_per_parent
    }

    /// Compute the native successor arena capacity for `parent_count`.
    pub fn successor_capacity_for(
        &self,
        parent_count: usize,
    ) -> Result<usize, TrustCgBfsLevelError> {
        self.metadata.successor_capacity_for(parent_count)
    }

    /// Keep the owning native library observable for diagnostics/tests.
    #[must_use]
    pub fn native_library(&self) -> &NativeLibrary {
        &self.library
    }

    /// Re-publish the fused loop and every dependent callout library
    /// executable before invoking cached raw entry points.
    pub fn ensure_published_executable(&self) -> Result<(), TrustCgError> {
        self.library.ensure_published_symbol_ptr(
            &self.symbol_name,
            self.entrypoint as *const () as *mut std::ffi::c_void,
        )?;
        for library in &self.extern_libraries {
            library.ensure_published_executable()?;
        }
        for callout in &self.extern_callouts {
            callout.ensure_published_executable()?;
        }
        Ok(())
    }

    /// Execute the resolved native fused level using metadata-derived capacity.
    pub fn run_level_arena(
        &mut self,
        parents: &[i64],
        parent_count: usize,
        out: &mut TrustCgSuccessorArena,
    ) -> Result<TrustCgBfsLevelOutcome, TrustCgBfsLevelError> {
        out.clear();
        let successor_capacity = match self.successor_capacity_for(parent_count) {
            Ok(successor_capacity) => successor_capacity,
            Err(error) => {
                self.reset_local_seen_after_native_error();
                return Err(error);
            }
        };
        self.run_level_arena_with_capacity(parents, parent_count, successor_capacity, out)
    }

    /// Execute the resolved native fused level against flat parent/successor
    /// arenas.
    ///
    /// The generated parent loop will eventually compute its exact maximum
    /// successor count. Until then the caller supplies `successor_capacity`,
    /// and native code reports [`TrustCgBfsLevelError::BufferOverflow`] if the
    /// arena is too small.
    pub fn run_level_arena_with_capacity(
        &mut self,
        parents: &[i64],
        parent_count: usize,
        successor_capacity: usize,
        out: &mut TrustCgSuccessorArena,
    ) -> Result<TrustCgBfsLevelOutcome, TrustCgBfsLevelError> {
        out.clear();
        if let Err(error) = validate_arena_len(parents.len(), parent_count, self.state_len) {
            self.reset_local_seen_after_native_error();
            return Err(error);
        }
        if out.state_len() != self.state_len {
            self.reset_local_seen_after_native_error();
            return Err(TrustCgBfsLevelError::StateLenMismatch {
                expected: self.state_len,
                actual: out.state_len(),
            });
        }

        let parent_abi = match TrustCgBfsParentArenaAbi::new(
            parents,
            parent_count,
            self.state_len,
            &mut self.scratch,
        ) {
            Ok(parent_abi) => parent_abi,
            Err(error) => {
                self.reset_local_seen_after_native_error();
                out.clear();
                return Err(error);
            }
        };
        let mut successor_abi = match out.prepare_abi(successor_capacity) {
            Ok(successor_abi) => successor_abi,
            Err(error) => {
                self.reset_local_seen_after_native_error();
                out.clear();
                return Err(error);
            }
        };
        self.reset_local_seen_for_native_invocation(parent_count, successor_capacity);
        // The native fused loop mutates the dedup set through the baked
        // `single_thread_fp_set_probe`. The ABI field is typed `*const u8`
        // (the native code only PtrToInt-casts it), but we derive the address
        // from a `&mut` so the runtime probe's `&mut SingleThreadFpSet` is the
        // unique reference to this set for the duration of the single-threaded
        // native call.
        let local_fp_set = if self.metadata.local_dedup {
            std::ptr::from_mut(&mut self.local_seen)
                .cast::<u8>()
                .cast_const()
        } else {
            std::ptr::null()
        };
        let parent_abi = TrustCgBfsParentArenaAbi {
            local_fp_set,
            ..parent_abi
        };
        trace_native_bfs_level_entrypoint(
            "before_entrypoint",
            &self.symbol_name,
            &self.metadata,
            None,
            &parent_abi,
            &successor_abi,
            successor_capacity,
        );
        // SAFETY: `entrypoint` was resolved by `from_library` as an
        // `TrustCgFusedLevelFn`; both ABI structs borrow valid arenas for the
        // duration of this call.
        self.ensure_published_executable()
            .map_err(|err| TrustCgBfsLevelError::InvalidAbi(err.to_string()))?;
        #[cfg(feature = "native")]
        crate::ensure_jit_execute_mode();
        let returned_status = unsafe { (self.entrypoint)(&parent_abi, &mut successor_abi) };
        trace_native_bfs_level_entrypoint(
            "after_entrypoint",
            &self.symbol_name,
            &self.metadata,
            Some(returned_status),
            &parent_abi,
            &successor_abi,
            successor_capacity,
        );
        if returned_status != successor_abi.status {
            trace_native_bfs_level_failure(
                "status_mismatch",
                returned_status,
                &parent_abi,
                &successor_abi,
                successor_capacity,
            );
            self.reset_local_seen_after_native_error();
            out.clear();
            return Err(TrustCgBfsLevelError::InvalidAbi(format!(
                "native status mismatch: returned {returned_status}, wrote {}",
                successor_abi.status
            )));
        }
        if successor_abi.status != TrustCgBfsLevelStatus::Ok.as_raw() {
            trace_native_bfs_level_failure(
                "non_ok_status",
                returned_status,
                &parent_abi,
                &successor_abi,
                successor_capacity,
            );
        }
        // SAFETY: `successor_abi` was produced by `out.prepare_abi` above and
        // the native function reported matching status after writing through it.
        match unsafe { out.commit_abi(&successor_abi) } {
            Ok(outcome) => {
                if let Err(error) = self
                    .metadata
                    .validate_reported_counters(parent_count, &outcome)
                {
                    trace_native_bfs_level_failure(
                        "invalid_reported_counters",
                        returned_status,
                        &parent_abi,
                        &successor_abi,
                        successor_capacity,
                    );
                    self.reset_local_seen_after_native_error();
                    out.clear();
                    return Err(error);
                }
                Ok(outcome)
            }
            Err(error) => {
                self.reset_local_seen_after_native_error();
                out.clear();
                Err(error)
            }
        }
    }

    fn reset_local_seen_for_native_invocation(
        &mut self,
        parent_count: usize,
        successor_capacity: usize,
    ) {
        if self.metadata.local_dedup {
            let capacity_hint = parent_count
                .saturating_add(successor_capacity)
                .saturating_mul(2)
                .max(1);
            self.local_seen = SingleThreadFpSet::new(capacity_hint);
        }
    }

    fn reset_local_seen_after_native_error(&mut self) {
        if self.metadata.local_dedup {
            self.local_seen = SingleThreadFpSet::new(1024);
        }
    }
}

/// Errors from the trust-codegen BFS level ABI/prototype layer.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TrustCgBfsLevelError {
    /// Flat arena length does not match `parent_count * state_len`.
    #[error("flat parent arena length mismatch: expected {expected} slots, got {actual} slots")]
    ParentArenaLenMismatch {
        /// Slot count required by `parent_count * state_len`.
        expected: usize,
        /// Slot count the caller actually supplied.
        actual: usize,
    },
    /// State slice length mismatch.
    #[error("state length mismatch: expected {expected} slots, got {actual} slots")]
    StateLenMismatch {
        /// Slot count the ABI expects per state.
        expected: usize,
        /// Slot count the caller actually supplied.
        actual: usize,
    },
    /// Caller-provided scratch was too small.
    #[error("BFS level scratch too small: needed {needed} slots, got {actual}")]
    ScratchTooSmall {
        /// Scratch slot count the level requires.
        needed: usize,
        /// Scratch slot count the caller supplied.
        actual: usize,
    },
    /// Integer conversion or allocation capacity overflow.
    #[error("BFS level arena capacity overflow")]
    CapacityOverflow,
    /// Native/prototype execution hit a runtime error.
    #[error("trust-cg BFS level runtime error")]
    RuntimeError,
    /// Native/prototype execution requested interpreter fallback.
    #[error("trust-cg BFS level requested interpreter fallback")]
    FallbackNeeded,
    /// Successor arena capacity was exhausted.
    #[error("trust-cg BFS successor arena overflow after {partial_count} successors")]
    BufferOverflow {
        /// Number of successors written before the arena filled.
        partial_count: u32,
    },
    /// ABI version mismatch.
    #[error("trust-cg BFS level ABI version mismatch: expected {expected}, got {actual}")]
    InvalidAbiVersion {
        /// ABI version this build of the crate speaks ([`TRUST_CG_BFS_LEVEL_ABI_VERSION`]).
        expected: u32,
        /// ABI version reported by the compiled artifact.
        actual: u32,
    },
    /// Other ABI validation failure.
    #[error("trust-cg BFS level ABI error: {0}")]
    InvalidAbi(String),
}

fn validate_arena_len(
    actual_len: usize,
    parent_count: usize,
    state_len: usize,
) -> Result<(), TrustCgBfsLevelError> {
    let expected = parent_count
        .checked_mul(state_len)
        .ok_or(TrustCgBfsLevelError::CapacityOverflow)?;
    if actual_len == expected {
        Ok(())
    } else {
        Err(TrustCgBfsLevelError::ParentArenaLenMismatch {
            expected,
            actual: actual_len,
        })
    }
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, TrustCgBfsLevelError> {
    u32::try_from(value)
        .map_err(|_| TrustCgBfsLevelError::InvalidAbi(format!("{field} exceeds u32::MAX")))
}

fn fingerprint_state(state: &[i64], state_len_u32: u32) -> u64 {
    let byte_len = state_len_u32 as usize * std::mem::size_of::<i64>();
    // SAFETY: `state` is valid for `state_len_u32 * size_of::<i64>()` bytes.
    unsafe { ty_compiled_fp_u64(state.as_ptr().cast::<u8>(), byte_len) }
}

fn decode_native_bool_callout_value(
    value: i64,
    context: &str,
) -> Result<bool, TrustCgBfsLevelError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(TrustCgBfsLevelError::InvalidAbi(format!(
            "native {context} returned noncanonical boolean value {value}"
        ))),
    }
}

fn decode_native_result_bool_field(value: u8, field: &str) -> Result<bool, TrustCgBfsLevelError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(TrustCgBfsLevelError::InvalidAbi(format!(
            "native result field {field} returned noncanonical boolean value {value}"
        ))),
    }
}

fn first_failed_invariant(
    invariants: &[TrustCgCompiledInvariant],
    state: &[i64],
    state_len_u32: u32,
    parent_idx: u32,
    successor_idx: u32,
) -> Result<Option<u32>, TrustCgBfsLevelError> {
    for invariant in invariants {
        let mut call_out = JitCallOut::default();
        invariant
            .ensure_published_executable()
            .map_err(|err| TrustCgBfsLevelError::InvalidAbi(err.to_string()))?;
        #[cfg(feature = "native")]
        crate::ensure_jit_execute_mode();
        unsafe {
            // SAFETY: `state` is valid for `state_len_u32` i64 slots and the
            // function pointer uses the stable trust-codegen invariant ABI.
            (invariant.func)(&mut call_out, state.as_ptr(), state_len_u32);
        }
        match call_out.status {
            JitStatus::Ok => {
                let context = format!(
                    "invariant callout name={} index={} parent_idx={parent_idx} successor_idx={successor_idx}",
                    invariant.descriptor.name, invariant.descriptor.invariant_idx,
                );
                if !decode_native_bool_callout_value(call_out.value, &context)? {
                    return Ok(Some(invariant.descriptor.invariant_idx));
                }
            }
            JitStatus::FallbackNeeded | JitStatus::PartialPass => {
                return Err(TrustCgBfsLevelError::FallbackNeeded);
            }
            JitStatus::RuntimeError => return Err(TrustCgBfsLevelError::RuntimeError),
        }
    }
    Ok(None)
}

fn invariant_from_abi(
    abi: &TrustCgBfsSuccessorArenaAbi,
    invariant_ok: bool,
) -> TrustCgInvariantStatus {
    if invariant_ok {
        TrustCgInvariantStatus::Passed
    } else {
        TrustCgInvariantStatus::Failed {
            parent_index: abi.failed_parent_idx,
            invariant_index: abi.failed_invariant_idx,
            successor_index: abi.failed_successor_idx,
        }
    }
}

fn trace_native_bfs_level_entrypoint(
    phase: &str,
    symbol_name: &str,
    metadata: &TrustCgBfsLevelNativeMetadata,
    returned_status: Option<u32>,
    parents: &TrustCgBfsParentArenaAbi,
    successors: &TrustCgBfsSuccessorArenaAbi,
    requested_successor_capacity: usize,
) {
    if std::env::var_os(BFS_LEVEL_TRACE_ENV).is_none() {
        return;
    }
    use std::io::Write as _;

    let returned_status_text =
        returned_status.map_or_else(|| "none".to_string(), |status| status.to_string());
    let returned_status_kind_text = returned_status
        .and_then(TrustCgBfsLevelStatus::from_raw)
        .map_or_else(|| "none".to_string(), |status| format!("{status:?}"));
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(
        stderr,
        concat!(
            "[trust-cg-native-bfs] phase={phase} symbol={symbol_name} ",
            "returned_status={returned_status_text} returned_status_kind={returned_status_kind_text} ",
            "written_status={written_status} written_status_kind={written_status_kind:?} ",
            "state_len={state_len} parent_count={parent_count} scratch_len={scratch_len} ",
            "actions={actions} state_constraints={state_constraints} invariants={invariants} ",
            "local_dedup={local_dedup} local_fp_set={local_fp_set} ",
            "requested_successor_capacity={requested_successor_capacity} ",
            "state_capacity={state_capacity} state_count={state_count} generated={generated} ",
            "parents_processed={parents_processed}"
        ),
        phase = phase,
        symbol_name = symbol_name,
        returned_status_text = returned_status_text,
        returned_status_kind_text = returned_status_kind_text,
        written_status = successors.status,
        written_status_kind = TrustCgBfsLevelStatus::from_raw(successors.status),
        state_len = successors.state_len,
        parent_count = parents.parent_count,
        scratch_len = parents.scratch_len,
        actions = metadata.action_count,
        state_constraints = metadata.state_constraint_count,
        invariants = metadata.invariant_count,
        local_dedup = metadata.local_dedup,
        local_fp_set = !parents.local_fp_set.is_null(),
        requested_successor_capacity = requested_successor_capacity,
        state_capacity = successors.state_capacity,
        state_count = successors.state_count,
        generated = successors.generated,
        parents_processed = successors.parents_processed,
    );
    let _ = stderr.flush();
}

fn trace_native_bfs_level_failure(
    reason: &str,
    returned_status: u32,
    parents: &TrustCgBfsParentArenaAbi,
    successors: &TrustCgBfsSuccessorArenaAbi,
    requested_successor_capacity: usize,
) {
    if std::env::var_os(BFS_LEVEL_TRACE_ENV).is_none() {
        return;
    }
    eprintln!(
        concat!(
            "trust-cg bfs level trace: reason={reason} ",
            "returned_status={returned_status} returned_status_kind={returned_status_kind:?} ",
            "written_status={written_status} written_status_kind={written_status_kind:?} ",
            "parent_abi_version={parent_abi_version} successor_abi_version={successor_abi_version} ",
            "state_len={state_len} parent_count={parent_count} scratch_len={scratch_len} ",
            "local_fp_set={local_fp_set} requested_successor_capacity={requested_successor_capacity} ",
            "state_capacity={state_capacity} state_count={state_count} generated={generated} ",
            "parents_processed={parents_processed} invariant_ok={invariant_ok} ",
            "failed_parent_idx={failed_parent_idx} failed_invariant_idx={failed_invariant_idx} ",
            "failed_successor_idx={failed_successor_idx} ",
            "first_zero_generated_parent_idx={first_zero_generated_parent_idx} ",
            "raw_successor_metadata_complete={raw_successor_metadata_complete}"
        ),
        reason = reason,
        returned_status = returned_status,
        returned_status_kind = TrustCgBfsLevelStatus::from_raw(returned_status),
        written_status = successors.status,
        written_status_kind = TrustCgBfsLevelStatus::from_raw(successors.status),
        parent_abi_version = parents.abi_version,
        successor_abi_version = successors.abi_version,
        state_len = successors.state_len,
        parent_count = parents.parent_count,
        scratch_len = parents.scratch_len,
        local_fp_set = !parents.local_fp_set.is_null(),
        requested_successor_capacity = requested_successor_capacity,
        state_capacity = successors.state_capacity,
        state_count = successors.state_count,
        generated = successors.generated,
        parents_processed = successors.parents_processed,
        invariant_ok = successors.invariant_ok,
        failed_parent_idx = successors.failed_parent_idx,
        failed_invariant_idx = successors.failed_invariant_idx,
        failed_successor_idx = successors.failed_successor_idx,
        first_zero_generated_parent_idx = successors.first_zero_generated_parent_idx,
        raw_successor_metadata_complete = successors.raw_successor_metadata_complete,
    );
}

fn status_to_result(status: u32, partial_count: u32) -> Result<(), TrustCgBfsLevelError> {
    match TrustCgBfsLevelStatus::from_raw(status) {
        Some(TrustCgBfsLevelStatus::Ok) => Ok(()),
        Some(TrustCgBfsLevelStatus::RuntimeError) => Err(TrustCgBfsLevelError::RuntimeError),
        Some(TrustCgBfsLevelStatus::BufferOverflow) => {
            Err(TrustCgBfsLevelError::BufferOverflow { partial_count })
        }
        Some(TrustCgBfsLevelStatus::InvalidAbi) => Err(TrustCgBfsLevelError::InvalidAbi(
            "native function reported invalid ABI".to_string(),
        )),
        Some(TrustCgBfsLevelStatus::FallbackNeeded) => Err(TrustCgBfsLevelError::FallbackNeeded),
        None => Err(TrustCgBfsLevelError::InvalidAbi(format!(
            "unknown status code {status}"
        ))),
    }
}

fn status_error(status: u32, partial_count: u32) -> TrustCgBfsLevelError {
    status_to_result(status, partial_count)
        .expect_err("non-Ok or unknown trust-codegen BFS level status must map to an error")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    fn make_native_bfs_status_module() -> trust_ir::Module {
        use trust_ir::constant::Constant;
        use trust_ir::inst::Inst;
        use trust_ir::ty::{FuncTy, Ty};
        use trust_ir::value::{BlockId, FuncId, ValueId};
        use trust_ir::{Block, Function, InstrNode, Module};

        let mut module = Module::new("native_bfs_status");
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr],
            returns: vec![Ty::U32],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let parent_arg = ValueId::new(0);
        let successor_arg = ValueId::new(1);
        let mut func = Function::new(FuncId::new(0), "trust_cg_bfs_level", ft, entry);
        let mut block = Block::new(entry)
            .with_param(parent_arg, Ty::Ptr)
            .with_param(successor_arg, Ty::Ptr);
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::U32,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(2)),
        );
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(2)],
        }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    #[cfg(feature = "native")]
    fn push_trust_ir_const(
        block: &mut trust_ir::Block,
        next_value: &mut u32,
        ty: trust_ir::ty::Ty,
        value: i128,
    ) -> trust_ir::value::ValueId {
        use trust_ir::constant::Constant;
        use trust_ir::inst::Inst;
        use trust_ir::value::ValueId;

        let result = ValueId::new(*next_value);
        *next_value += 1;
        block.body.push(
            trust_ir::InstrNode::new(Inst::Const {
                ty,
                value: Constant::Int(value),
            })
            .with_result(result),
        );
        result
    }

    #[cfg(feature = "native")]
    fn push_successor_abi_store(
        block: &mut trust_ir::Block,
        next_value: &mut u32,
        successor_arg: trust_ir::value::ValueId,
        offset: usize,
        ty: trust_ir::ty::Ty,
        value: i128,
    ) {
        use trust_ir::inst::Inst;
        use trust_ir::ty::Ty;
        use trust_ir::value::ValueId;

        let offset = push_trust_ir_const(block, next_value, Ty::U64, offset as i128);
        let ptr = ValueId::new(*next_value);
        *next_value += 1;
        block.body.push(
            trust_ir::InstrNode::new(Inst::GEP {
                pointee_ty: Ty::U8,
                base: successor_arg,
                indices: vec![offset],
                inbounds: false,
            })
            .with_result(ptr),
        );
        let value = push_trust_ir_const(block, next_value, ty.clone(), value);
        block.body.push(trust_ir::InstrNode::new(Inst::Store {
            ty,
            ptr,
            value,
            volatile: false,
            align: None,
        }));
    }

    #[cfg(feature = "native")]
    fn make_native_bfs_reported_counter_module(
        parents_processed: u32,
        generated: u64,
        state_count: u32,
    ) -> trust_ir::Module {
        make_native_bfs_reported_counter_status_module(
            parents_processed,
            generated,
            state_count,
            TrustCgBfsLevelStatus::Ok,
            TrustCgBfsLevelStatus::Ok,
        )
    }

    #[cfg(feature = "native")]
    fn make_native_bfs_reported_counter_status_module(
        parents_processed: u32,
        generated: u64,
        state_count: u32,
        written_status: TrustCgBfsLevelStatus,
        returned_status: TrustCgBfsLevelStatus,
    ) -> trust_ir::Module {
        use trust_ir::inst::Inst;
        use trust_ir::ty::{FuncTy, Ty};
        use trust_ir::value::{BlockId, FuncId, ValueId};
        use trust_ir::{Block, Function, Module};

        let mut module = Module::new("native_bfs_reported_counter");
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr],
            returns: vec![Ty::U32],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let parent_arg = ValueId::new(0);
        let successor_arg = ValueId::new(1);
        let mut func = Function::new(FuncId::new(0), "trust_cg_bfs_level", ft, entry);
        let mut block = Block::new(entry)
            .with_param(parent_arg, Ty::Ptr)
            .with_param(successor_arg, Ty::Ptr);
        let mut next_value = 2;

        push_successor_abi_store(
            &mut block,
            &mut next_value,
            successor_arg,
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, state_count),
            Ty::U32,
            i128::from(state_count),
        );
        push_successor_abi_store(
            &mut block,
            &mut next_value,
            successor_arg,
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, generated),
            Ty::U64,
            i128::from(generated),
        );
        push_successor_abi_store(
            &mut block,
            &mut next_value,
            successor_arg,
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, parents_processed),
            Ty::U32,
            i128::from(parents_processed),
        );
        push_successor_abi_store(
            &mut block,
            &mut next_value,
            successor_arg,
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, invariant_ok),
            Ty::U8,
            1,
        );
        push_successor_abi_store(
            &mut block,
            &mut next_value,
            successor_arg,
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, status),
            Ty::U32,
            i128::from(written_status.as_raw()),
        );
        push_successor_abi_store(
            &mut block,
            &mut next_value,
            successor_arg,
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, raw_successor_metadata_complete),
            Ty::U8,
            1,
        );
        let ok = push_trust_ir_const(
            &mut block,
            &mut next_value,
            Ty::U32,
            i128::from(returned_status.as_raw()),
        );
        block
            .body
            .push(trust_ir::InstrNode::new(Inst::Return { values: vec![ok] }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    fn action_descriptor(name: &str, idx: u32) -> ActionDescriptor {
        ActionDescriptor {
            name: name.to_string(),
            action_idx: idx,
            binding_values: Vec::new(),
            formal_values: Vec::new(),
            read_vars: vec![0],
            write_vars: vec![0],
            compound_read_vars: Vec::new(),
        }
    }

    fn invariant_descriptor(name: &str, idx: u32) -> InvariantDescriptor {
        InvariantDescriptor {
            name: name.to_string(),
            invariant_idx: idx,
        }
    }

    unsafe extern "C" fn increment_slot0(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        let len = state_len as usize;
        let src = unsafe { std::slice::from_raw_parts(state_in, len) };
        let dst = unsafe { std::slice::from_raw_parts_mut(state_out, len) };
        dst.copy_from_slice(src);
        if let Some(first) = dst.first_mut() {
            *first += 1;
        }
        unsafe {
            *out = JitCallOut::ok(1);
        }
    }

    unsafe extern "C" fn disabled_action(
        out: *mut JitCallOut,
        _state_in: *const i64,
        _state_out: *mut i64,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    unsafe extern "C" fn invariant_slot0_lt_3(
        out: *mut JitCallOut,
        state: *const i64,
        state_len: u32,
    ) {
        let len = state_len as usize;
        let state = unsafe { std::slice::from_raw_parts(state, len) };
        let ok = state.first().copied().unwrap_or_default() < 3;
        unsafe {
            *out = JitCallOut::ok(i64::from(ok));
        }
    }

    unsafe extern "C" fn invariant_always_ok(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut::ok(1);
        }
    }

    unsafe extern "C" fn noncanonical_true_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        let len = state_len as usize;
        let src = unsafe { std::slice::from_raw_parts(state_in, len) };
        let dst = unsafe { std::slice::from_raw_parts_mut(state_out, len) };
        dst.copy_from_slice(src);
        unsafe {
            *out = JitCallOut::ok(4_294_967_297);
        }
    }

    unsafe extern "C" fn noncanonical_true_invariant(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut::ok(4_294_967_297);
        }
    }

    /// Route B loop kernel: pushes two successors (`slot0 + 1`, `slot0 + 2`).
    /// Mirrors the compiled sink protocol: overflow sets the flag (via
    /// `try_push_succ`) and still returns `JitStatus::Ok`; `call_out.value`
    /// is never a meaningful boolean for the loop ABI.
    unsafe extern "C" fn loop_push_two(
        out: *mut JitCallOut,
        state_in: *const i64,
        sink: *mut tla_jit_abi::NextStateLoopSink,
        state_len: u32,
    ) {
        let len = state_len as usize;
        let src = unsafe { std::slice::from_raw_parts(state_in, len) };
        let sink = unsafe { &mut *sink };
        let base = src.first().copied().unwrap_or_default();
        let mut succ = src.to_vec();
        for delta in 1..=2i64 {
            succ.copy_from_slice(src);
            if let Some(first) = succ.first_mut() {
                *first = base + delta;
            }
            if !unsafe { sink.try_push_succ(&succ) } {
                break;
            }
        }
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Route B loop kernel: pushes 100 distinct successors (`1000..1100` in
    /// slot0), more than the 64-successor initial host sink, forcing at least
    /// one discard+regrow cycle.
    unsafe extern "C" fn loop_push_hundred(
        out: *mut JitCallOut,
        state_in: *const i64,
        sink: *mut tla_jit_abi::NextStateLoopSink,
        state_len: u32,
    ) {
        let len = state_len as usize;
        let src = unsafe { std::slice::from_raw_parts(state_in, len) };
        let sink = unsafe { &mut *sink };
        let mut succ = src.to_vec();
        for idx in 0..100i64 {
            succ.copy_from_slice(src);
            if let Some(first) = succ.first_mut() {
                *first = 1000 + idx;
            }
            if !unsafe { sink.try_push_succ(&succ) } {
                break;
            }
        }
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Route B loop kernel: disabled action (no pushes, clean Ok return).
    unsafe extern "C" fn loop_disabled(
        out: *mut JitCallOut,
        _state_in: *const i64,
        _sink: *mut tla_jit_abi::NextStateLoopSink,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Route B loop kernel: always flags overflow regardless of capacity, so
    /// the host regrow loop must eventually fail closed at the slot cap.
    unsafe extern "C" fn loop_always_overflow(
        out: *mut JitCallOut,
        _state_in: *const i64,
        sink: *mut tla_jit_abi::NextStateLoopSink,
        _state_len: u32,
    ) {
        let sink = unsafe { &mut *sink };
        sink.overflowed = 1;
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Route B loop kernel: reports a runtime error.
    unsafe extern "C" fn loop_runtime_error(
        out: *mut JitCallOut,
        _state_in: *const i64,
        _sink: *mut tla_jit_abi::NextStateLoopSink,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut {
                status: JitStatus::RuntimeError,
                ..Default::default()
            };
        }
    }

    /// Route B loop kernel: requests interpreter fallback.
    unsafe extern "C" fn loop_fallback_needed(
        out: *mut JitCallOut,
        _state_in: *const i64,
        _sink: *mut tla_jit_abi::NextStateLoopSink,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut {
                status: JitStatus::FallbackNeeded,
                ..Default::default()
            };
        }
    }

    #[test]
    fn native_bool_callout_decode_accepts_only_canonical_values() {
        assert!(!decode_native_bool_callout_value(0, "test callout").expect("decode false"));
        assert!(decode_native_bool_callout_value(1, "test callout").expect("decode true"));
        for value in [2, -1, 4_294_967_296, 4_294_967_297, i64::MAX] {
            let err = decode_native_bool_callout_value(value, "test callout")
                .expect_err("noncanonical boolean must be rejected");
            let TrustCgBfsLevelError::InvalidAbi(message) = err else {
                panic!("unexpected noncanonical boolean error for {value}: {err:?}");
            };
            assert!(
                message.contains(&format!(
                    "native test callout returned noncanonical boolean value {value}"
                )),
                "unexpected noncanonical boolean error for {value}: {message}"
            );
        }
    }

    #[test]
    fn successor_arena_stores_flat_states_and_parent_indexes() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 2);

        let first = arena.push_successor(0, &[1, 2]).expect("push first");
        let second = arena.push_successor(3, &[4, 5]).expect("push second");

        assert_eq!(first, 0);
        assert_eq!(second, 1);
        assert_eq!(arena.successor_count(), 2);
        assert_eq!(arena.states_flat(), &[1, 2, 4, 5]);
        assert_eq!(arena.parent_indices(), &[0, 3]);
        assert_eq!(
            arena.successor_fingerprints(),
            &[fingerprint_state(&[1, 2], 2), fingerprint_state(&[4, 5], 2),]
        );
        assert_eq!(arena.successor(1), Some(&[4, 5][..]));
        let collected: Vec<&[i64]> = arena.iter_successors().collect();
        assert_eq!(collected, vec![&[1, 2][..], &[4, 5][..]]);
    }

    #[test]
    fn successor_arena_mutable_state_access_invalidates_fingerprint_sidecar() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 1);
        arena.push_successor(0, &[1, 2]).expect("push successor");
        assert_eq!(arena.successor_fingerprints().len(), 1);

        arena.states_flat_mut()[0] = 7;

        assert_eq!(arena.states_flat(), &[7, 2]);
        assert!(arena.successor_fingerprints().is_empty());

        arena
            .push_successor(1, &[3, 4])
            .expect("append after mutable access");

        assert_eq!(arena.successor_count(), 2);
        assert_eq!(arena.states_flat(), &[7, 2, 3, 4]);
        assert!(
            arena.successor_fingerprints().is_empty(),
            "a partial fingerprint sidecar must not be exposed by index"
        );
    }

    #[test]
    fn parent_arena_abi_validates_lengths_and_scratch() {
        let parents = [1, 2, 3, 4];
        let scratch_len = trust_cg_bfs_native_parent_scratch_layout(2)
            .expect("scratch layout")
            .total_slots;
        let mut scratch = vec![0; scratch_len];
        let abi =
            TrustCgBfsParentArenaAbi::new(&parents, 2, 2, &mut scratch).expect("valid parent ABI");

        assert_eq!(abi.abi_version, TRUST_CG_BFS_LEVEL_ABI_VERSION);
        assert_eq!(abi.parent_count, 2);
        assert_eq!(abi.state_len, 2);
        assert!(!abi.parents.is_null());
        assert!(!abi.scratch.is_null());

        let err = TrustCgBfsParentArenaAbi::new(&parents, 3, 2, &mut scratch)
            .expect_err("length mismatch");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::ParentArenaLenMismatch {
                expected: 6,
                actual: 4
            }
        ));

        let mut short_scratch = vec![0; scratch_len - 1];
        let err = TrustCgBfsParentArenaAbi::new(&parents, 2, 2, &mut short_scratch)
            .expect_err("scratch mismatch");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::ScratchTooSmall {
                needed,
                actual
            } if needed == scratch_len && actual == scratch_len - 1
        ));
    }

    #[test]
    fn parent_arena_abi_allows_null_zero_width_parent_states() {
        let parents: [i64; 0] = [];
        let scratch_len = trust_cg_bfs_native_parent_scratch_layout(0)
            .expect("scratch layout")
            .total_slots;
        let mut scratch = vec![0; scratch_len];

        let abi = TrustCgBfsParentArenaAbi::new(&parents, 2, 0, &mut scratch)
            .expect("zero-width parent states should not require a state pointer");

        assert_eq!(abi.parent_count, 2);
        assert_eq!(abi.state_len, 0);
        assert!(abi.parents.is_null());
        assert!(!abi.scratch.is_null());
    }

    #[test]
    fn native_parent_scratch_layout_reserves_candidate_callout_and_counters() {
        let layout = trust_cg_bfs_native_parent_scratch_layout(4).expect("scratch layout");

        assert_eq!(layout.callout_offset, 0);
        assert_eq!(
            layout.candidate_offset,
            TRUST_CG_BFS_NATIVE_CALLOUT_SCRATCH_SLOTS
        );
        assert_eq!(layout.parent_idx_offset, layout.candidate_offset + 4);
        assert_eq!(layout.generated_offset, layout.parent_idx_offset + 1);
        assert_eq!(layout.state_count_offset, layout.generated_offset + 1);
        assert_eq!(
            layout.parent_generated_offset,
            layout.state_count_offset + 1
        );
        assert_eq!(
            layout.parent_count_offset,
            layout.parent_generated_offset + 1
        );
        assert_eq!(
            layout.loop_sink_offset,
            layout.parent_idx_offset + TRUST_CG_BFS_NATIVE_COUNTER_SCRATCH_SLOTS,
            "Route B sink header must start right after the counter slots so \
             every pre-existing offset is unchanged"
        );
        assert_eq!(
            layout.loop_commit_idx_offset,
            layout.loop_sink_offset + TRUST_CG_BFS_NATIVE_LOOP_SINK_SCRATCH_SLOTS
        );
        assert_eq!(
            layout.total_slots,
            layout.loop_commit_idx_offset + TRUST_CG_BFS_NATIVE_LOOP_COMMIT_IDX_SCRATCH_SLOTS
        );
        assert_eq!(
            trust_cg_bfs_native_parent_scratch_layout(0)
                .expect("zero-len scratch layout")
                .total_slots,
            TRUST_CG_BFS_NATIVE_CALLOUT_SCRATCH_SLOTS
                + 1
                + TRUST_CG_BFS_NATIVE_COUNTER_SCRATCH_SLOTS
                + TRUST_CG_BFS_NATIVE_LOOP_SINK_SCRATCH_SLOTS
                + TRUST_CG_BFS_NATIVE_LOOP_COMMIT_IDX_SCRATCH_SLOTS
        );
        assert!(matches!(
            trust_cg_bfs_native_parent_scratch_layout(usize::MAX),
            Err(TrustCgBfsLevelError::CapacityOverflow)
        ));
    }

    #[test]
    fn successor_arena_abi_round_trip_commits_native_counts() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 0);
        let mut abi = arena.prepare_abi(2).expect("prepare ABI");

        unsafe {
            *abi.states.add(0) = 10;
            *abi.states.add(1) = 11;
            *abi.states.add(2) = 20;
            *abi.states.add(3) = 21;
            *abi.parent_index.add(0) = 0;
            *abi.parent_index.add(1) = 7;
            *abi.fingerprints.add(0) = 100;
            *abi.fingerprints.add(1) = 200;
        }
        abi.state_count = 2;
        abi.generated = 3;
        abi.parents_processed = 8;
        abi.first_zero_generated_parent_idx = 7;
        abi.raw_successor_metadata_complete = 1;

        let outcome = unsafe { arena.commit_abi(&abi) }.expect("commit ABI");

        assert_eq!(outcome.parents_processed, 8);
        assert_eq!(outcome.total_generated, 3);
        assert_eq!(outcome.total_new, 2);
        assert!(outcome.raw_successor_metadata_complete);
        assert_eq!(outcome.first_parent_without_raw_successors, Some(7));
        assert_eq!(arena.states_flat(), &[10, 11, 20, 21]);
        assert_eq!(arena.parent_indices(), &[0, 7]);
        assert_eq!(arena.successor_fingerprints(), &[100, 200]);
        assert!(arena.raw_successor_metadata_complete());
        assert_eq!(arena.first_parent_without_raw_successors(), Some(7));
    }

    #[test]
    fn successor_arena_abi_round_trip_allows_null_zero_width_states() {
        let mut arena = TrustCgSuccessorArena::with_capacity(0, 0);
        let mut abi = arena.prepare_abi(2).expect("prepare zero-width ABI");

        assert_eq!(abi.state_len, 0);
        assert_eq!(abi.state_capacity, 2);
        assert!(abi.states.is_null());
        assert!(!abi.parent_index.is_null());
        assert!(!abi.fingerprints.is_null());

        unsafe {
            *abi.parent_index.add(0) = 0;
            *abi.parent_index.add(1) = 1;
            *abi.fingerprints.add(0) = 100;
            *abi.fingerprints.add(1) = 200;
        }
        abi.state_count = 2;
        abi.generated = 2;
        abi.parents_processed = 2;
        abi.raw_successor_metadata_complete = 1;

        let outcome = unsafe { arena.commit_abi(&abi) }.expect("commit zero-width ABI");

        assert_eq!(outcome.parents_processed, 2);
        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(arena.states_flat(), &[] as &[i64]);
        assert_eq!(arena.parent_indices(), &[0, 1]);
        assert_eq!(arena.successor_fingerprints(), &[100, 200]);
        assert_eq!(arena.successor(0), Some(&[][..]));
        assert_eq!(arena.successor(1), Some(&[][..]));
        let successors: Vec<&[i64]> = arena.iter_successors().collect();
        let empty: &[i64] = &[];
        assert_eq!(successors, vec![empty, empty]);
    }

    #[test]
    fn successor_arena_abi_rejects_missing_fingerprint_sidecar() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 2);
        let mut abi = arena.prepare_abi(2).expect("prepare ABI");

        unsafe {
            *abi.states.add(0) = 10;
            *abi.states.add(1) = 11;
            *abi.parent_index.add(0) = 0;
        }
        abi.state_count = 1;
        abi.generated = 1;
        abi.fingerprints = std::ptr::null_mut();

        let err = unsafe { arena.commit_abi(&abi) }.expect_err("missing sidecar must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("omitted successor fingerprint sidecar")
        ));
        assert_eq!(arena.successor_count(), 0);
        assert_eq!(arena.states_flat(), &[] as &[i64]);
        assert_eq!(arena.parent_indices(), &[] as &[u32]);
        assert_eq!(arena.successor_fingerprints(), &[] as &[u64]);
    }

    #[test]
    fn successor_arena_abi_rejects_missing_nonempty_state_arena() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 2);
        let mut abi = arena.prepare_abi(2).expect("prepare ABI");

        unsafe {
            *abi.parent_index.add(0) = 0;
            *abi.fingerprints.add(0) = 100;
        }
        abi.state_count = 1;
        abi.generated = 1;
        abi.states = std::ptr::null_mut();

        let err = unsafe { arena.commit_abi(&abi) }.expect_err("missing states must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("omitted successor state arena")
        ));
        assert_eq!(arena.successor_count(), 0);
        assert_eq!(arena.states_flat(), &[] as &[i64]);
        assert_eq!(arena.parent_indices(), &[] as &[u32]);
        assert_eq!(arena.successor_fingerprints(), &[] as &[u64]);
    }

    #[test]
    fn successor_arena_abi_rejects_missing_parent_index_sidecar() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 2);
        let mut abi = arena.prepare_abi(2).expect("prepare ABI");

        unsafe {
            *abi.states.add(0) = 10;
            *abi.states.add(1) = 11;
            *abi.fingerprints.add(0) = 100;
        }
        abi.state_count = 1;
        abi.generated = 1;
        abi.parent_index = std::ptr::null_mut();

        let err = unsafe { arena.commit_abi(&abi) }.expect_err("missing sidecar must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("omitted successor parent-index sidecar")
        ));
        assert_eq!(arena.successor_count(), 0);
        assert_eq!(arena.states_flat(), &[] as &[i64]);
        assert_eq!(arena.parent_indices(), &[] as &[u32]);
        assert_eq!(arena.successor_fingerprints(), &[] as &[u64]);
    }

    #[test]
    fn successor_arena_abi_terminal_errors_do_not_commit_native_counts() {
        for status in [
            TrustCgBfsLevelStatus::RuntimeError,
            TrustCgBfsLevelStatus::FallbackNeeded,
            TrustCgBfsLevelStatus::InvalidAbi,
        ] {
            let mut arena = TrustCgSuccessorArena::with_capacity(2, 2);
            let mut abi = arena.prepare_abi(2).expect("prepare ABI");
            arena.states.extend_from_slice(&[1, 2]);
            arena.parent_index.push(4);
            arena.fingerprints.push(8);
            arena.generated = 5;
            arena.parents_processed = 6;
            arena.invariant = TrustCgInvariantStatus::Failed {
                parent_index: 1,
                invariant_index: 2,
                successor_index: 0,
            };
            arena.first_parent_without_raw_successors = Some(3);
            arena.raw_successor_metadata_complete = true;
            let before = (
                arena.states.len(),
                arena.parent_index.len(),
                arena.fingerprints.len(),
                arena.generated,
                arena.parents_processed,
                arena.invariant,
                arena.first_parent_without_raw_successors,
                arena.raw_successor_metadata_complete,
            );

            abi.status = status.as_raw();
            abi.state_count = 2;
            abi.generated = 99;
            abi.parents_processed = 11;
            abi.invariant_ok = 1;
            abi.first_zero_generated_parent_idx = 9;
            abi.raw_successor_metadata_complete = 0;

            let err = unsafe { arena.commit_abi(&abi) }.expect_err("terminal status rejects");
            match status {
                TrustCgBfsLevelStatus::RuntimeError => {
                    assert_eq!(err, TrustCgBfsLevelError::RuntimeError);
                }
                TrustCgBfsLevelStatus::FallbackNeeded => {
                    assert_eq!(err, TrustCgBfsLevelError::FallbackNeeded);
                }
                TrustCgBfsLevelStatus::InvalidAbi => {
                    assert!(matches!(err, TrustCgBfsLevelError::InvalidAbi(_)));
                }
                _ => unreachable!("test only covers terminal errors"),
            }

            assert_eq!(
                (
                    arena.states.len(),
                    arena.parent_index.len(),
                    arena.fingerprints.len(),
                    arena.generated,
                    arena.parents_processed,
                    arena.invariant,
                    arena.first_parent_without_raw_successors,
                    arena.raw_successor_metadata_complete,
                ),
                before
            );
        }
    }

    #[test]
    fn successor_arena_abi_rejects_noncanonical_result_boolean_flags_before_commit() {
        for field in ["invariant_ok", "raw_successor_metadata_complete"] {
            let mut arena = TrustCgSuccessorArena::with_capacity(1, 1);
            let mut abi = arena.prepare_abi(1).expect("prepare ABI");
            unsafe {
                *abi.states = 7;
                *abi.parent_index = 0;
                *abi.fingerprints = 77;
            }
            abi.state_count = 1;
            abi.generated = 1;
            abi.parents_processed = 1;
            abi.invariant_ok = 1;
            abi.raw_successor_metadata_complete = 1;
            match field {
                "invariant_ok" => abi.invariant_ok = 2,
                "raw_successor_metadata_complete" => abi.raw_successor_metadata_complete = 2,
                _ => unreachable!("covered fields"),
            }
            let before = (
                arena.states.len(),
                arena.parent_index.len(),
                arena.fingerprints.len(),
                arena.generated,
                arena.parents_processed,
                arena.invariant,
                arena.first_parent_without_raw_successors,
                arena.raw_successor_metadata_complete,
            );

            let err = unsafe { arena.commit_abi(&abi) }
                .expect_err("noncanonical result boolean must fail closed");
            let TrustCgBfsLevelError::InvalidAbi(message) = err else {
                panic!("unexpected noncanonical result boolean error: {err:?}");
            };
            assert!(
                message.contains(field) && message.contains("noncanonical boolean value 2"),
                "unexpected noncanonical result boolean message: {message}"
            );
            assert_eq!(
                (
                    arena.states.len(),
                    arena.parent_index.len(),
                    arena.fingerprints.len(),
                    arena.generated,
                    arena.parents_processed,
                    arena.invariant,
                    arena.first_parent_without_raw_successors,
                    arena.raw_successor_metadata_complete,
                ),
                before,
                "noncanonical {field} must not partially commit arena state"
            );
        }
    }

    #[test]
    fn successor_arena_abi_rejects_generated_below_state_count_before_commit() {
        let mut arena = TrustCgSuccessorArena::with_capacity(2, 2);
        let mut abi = arena.prepare_abi(2).expect("prepare ABI");

        unsafe {
            *abi.states.add(0) = 10;
            *abi.states.add(1) = 11;
            *abi.states.add(2) = 20;
            *abi.states.add(3) = 21;
            *abi.parent_index.add(0) = 0;
            *abi.parent_index.add(1) = 7;
        }
        abi.state_count = 2;
        abi.generated = 1;

        let err = unsafe { arena.commit_abi(&abi) }.expect_err("impossible count must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("generated count 1 is below state_count 2")
        ));
        assert_eq!(arena.successor_count(), 0);
        assert_eq!(arena.states_flat(), &[] as &[i64]);
        assert_eq!(arena.parent_indices(), &[] as &[u32]);
        assert_eq!(arena.generated, 0);
    }

    #[test]
    fn native_metadata_rejects_generated_above_parent_action_bound() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 3,
            total_generated: 7,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: true,
        };

        let err = metadata
            .validate_reported_counters(3, &outcome)
            .expect_err("generated count above parent*actions must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native generated count 7 exceeds single-successor action upper bound 6")
        ));
    }

    #[test]
    fn native_metadata_rejects_parents_processed_above_parent_count() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 4,
            total_generated: 6,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: true,
        };

        let err = metadata
            .validate_reported_counters(3, &outcome)
            .expect_err("parents_processed above parent_count must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native parents_processed 4 exceeds parent_count 3")
        ));
    }

    #[test]
    fn native_metadata_rejects_passed_with_short_processed_prefix() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 2,
            total_generated: 4,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: true,
        };

        let err = metadata
            .validate_reported_counters(3, &outcome)
            .expect_err("passed native level must process the whole parent batch");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native parents_processed 2 must equal parent_count 3")
        ));
    }

    #[test]
    fn native_metadata_rejects_failed_with_processed_not_failed_parent_plus_one() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 1,
            total_generated: 2,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Failed {
                parent_index: 1,
                invariant_index: 0,
                successor_index: 0,
            },
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: true,
        };

        let err = metadata
            .validate_reported_counters(3, &outcome)
            .expect_err("failed native level must report the failed parent prefix");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native parents_processed 1 must equal failed parent prefix 2")
        ));
    }

    #[test]
    fn native_metadata_rejects_generated_above_processed_parent_action_bound() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 1,
            total_generated: 3,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: None,
            raw_successor_metadata_complete: true,
        };

        let err = metadata
            .validate_reported_counters(1, &outcome)
            .expect_err("generated count above processed_parents*actions must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native generated count 3 exceeds single-successor action upper bound 2")
        ));
    }

    #[test]
    fn native_metadata_rejects_first_zero_outside_processed_prefix() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 1,
            total_generated: 0,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: Some(1),
            raw_successor_metadata_complete: true,
        };

        let err = metadata
            .validate_reported_counters(1, &outcome)
            .expect_err("first-zero parent outside processed prefix must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native first_zero_generated_parent_idx 1 is outside processed parent prefix 1")
        ));
    }

    #[test]
    fn native_metadata_rejects_first_zero_without_complete_metadata() {
        let metadata = TrustCgBfsLevelNativeMetadata::new(2, 0, 2, true);
        let outcome = TrustCgBfsLevelOutcome {
            parents_processed: 1,
            total_generated: 0,
            total_new: 0,
            invariant: TrustCgInvariantStatus::Passed,
            first_parent_without_raw_successors: Some(0),
            raw_successor_metadata_complete: false,
        };

        let err = metadata
            .validate_reported_counters(1, &outcome)
            .expect_err("first-zero parent without complete metadata must reject");
        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("first_zero_generated_parent_idx was reported without complete raw-successor metadata")
        ));
    }

    #[test]
    fn prototype_runs_parent_arena_and_filters_local_duplicates() {
        let actions = vec![
            TrustCgCompiledAction::new(action_descriptor("inc", 0), increment_slot0),
            TrustCgCompiledAction::new(action_descriptor("inc_again", 1), increment_slot0),
            TrustCgCompiledAction::new(action_descriptor("disabled", 2), disabled_action),
        ];
        let invariants = vec![TrustCgCompiledInvariant::new(
            invariant_descriptor("ok", 0),
            invariant_always_ok,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, invariants, 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[0, 1, 1], 3, &mut out)
            .expect("run level");

        assert_eq!(level.capabilities(), TrustCgBfsLevelCapabilities::PROTOTYPE);
        assert_eq!(outcome.parents_processed, 3);
        assert_eq!(outcome.total_generated, 6);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(out.parent_indices(), &[0, 1]);
        let successors: Vec<&[i64]> = out.iter_successors().collect();
        assert_eq!(successors, vec![&[1][..], &[2][..]]);
    }

    #[test]
    fn prototype_rejects_noncanonical_action_boolean_callout_value() {
        let actions = vec![TrustCgCompiledAction::new(
            action_descriptor("noncanonical", 0),
            noncanonical_true_action,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, Vec::new(), 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[10], 1, &mut out)
            .expect_err("noncanonical action boolean must fail closed");

        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native action callout name=noncanonical index=0 parent_idx=0 returned noncanonical boolean value 4294967297")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[test]
    fn prototype_rejects_noncanonical_invariant_boolean_callout_value() {
        let actions = vec![TrustCgCompiledAction::new(
            action_descriptor("inc", 0),
            increment_slot0,
        )];
        let invariants = vec![TrustCgCompiledInvariant::new(
            invariant_descriptor("noncanonical", 9),
            noncanonical_true_invariant,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, invariants, 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[10], 1, &mut out)
            .expect_err("noncanonical invariant boolean must fail closed");

        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native invariant callout name=noncanonical index=9 parent_idx=0 successor_idx=0 returned noncanonical boolean value 4294967297")
        ));
        assert_eq!(out.successor_count(), 1);
    }

    #[test]
    fn prototype_reports_first_parent_without_raw_successors() {
        let actions = vec![TrustCgCompiledAction::new(
            action_descriptor("disabled", 0),
            disabled_action,
        )];
        let invariants = vec![TrustCgCompiledInvariant::new(
            invariant_descriptor("ok", 0),
            invariant_always_ok,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, invariants, 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run level");

        assert_eq!(outcome.total_generated, 0);
        assert_eq!(outcome.total_new, 0);
        assert!(outcome.raw_successor_metadata_complete);
        assert_eq!(outcome.first_parent_without_raw_successors, Some(0));
        assert!(out.raw_successor_metadata_complete());
        assert_eq!(out.first_parent_without_raw_successors(), Some(0));
    }

    #[test]
    fn prototype_reports_stable_invariant_failure_indexes() {
        let actions = vec![TrustCgCompiledAction::new(
            action_descriptor("inc", 0),
            increment_slot0,
        )];
        let invariants = vec![TrustCgCompiledInvariant::new(
            invariant_descriptor("slot0_lt_3", 9),
            invariant_slot0_lt_3,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, invariants, 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[0, 2], 2, &mut out)
            .expect("run level");

        assert_eq!(outcome.parents_processed, 2);
        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(
            outcome.invariant,
            TrustCgInvariantStatus::Failed {
                parent_index: 1,
                invariant_index: 9,
                successor_index: 1,
            }
        );
        assert_eq!(out.successor(1), Some(&[3][..]));
    }

    #[test]
    fn prototype_runs_loop_action_via_host_sink_dispatch() {
        // Mixed ABI level: the Route B loop kernel pushes slot0+1 and slot0+2,
        // the single-successor action pushes slot0+1 again (a local
        // duplicate). generated counts every raw successor; dedup keeps two.
        let actions = vec![
            TrustCgCompiledAction::new_loop(action_descriptor("recv_loop", 0), loop_push_two),
            TrustCgCompiledAction::new(action_descriptor("inc", 1), increment_slot0),
        ];
        let invariants = vec![TrustCgCompiledInvariant::new(
            invariant_descriptor("ok", 0),
            invariant_always_ok,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, invariants, 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level.run_level_arena(&[0], 1, &mut out).expect("run level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 3);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(outcome.invariant, TrustCgInvariantStatus::Passed);
        assert_eq!(out.parent_indices(), &[0, 0]);
        let successors: Vec<&[i64]> = out.iter_successors().collect();
        assert_eq!(successors, vec![&[1][..], &[2][..]]);
    }

    #[test]
    fn prototype_loop_action_disabled_reports_zero_successor_parent() {
        // A loop kernel with count == 0 and JitStatus::Ok is simply disabled:
        // no boolean decode of call_out.value happens for the sink ABI.
        let actions = vec![TrustCgCompiledAction::new_loop(
            action_descriptor("recv_loop", 0),
            loop_disabled,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, Vec::new(), 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level.run_level_arena(&[7], 1, &mut out).expect("run level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 0);
        assert_eq!(outcome.total_new, 0);
        assert_eq!(outcome.first_parent_without_raw_successors, Some(0));
        assert!(outcome.raw_successor_metadata_complete);
    }

    #[test]
    fn prototype_loop_action_regrows_sink_and_keeps_exact_successor_set() {
        // 100 successors exceed the 64-successor initial sink: the host must
        // discard the truncated attempt, double the buffer, and re-call. The
        // final set must be EXACTLY the 100 distinct successors — no
        // truncation, no duplication of the discarded partial pushes.
        let actions = vec![TrustCgCompiledAction::new_loop(
            action_descriptor("recv_loop", 0),
            loop_push_hundred,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, Vec::new(), 256).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level.run_level_arena(&[0], 1, &mut out).expect("run level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 100);
        assert_eq!(outcome.total_new, 100);
        let successors: Vec<i64> = out.iter_successors().map(|s| s[0]).collect();
        assert_eq!(successors, (1000..1100).collect::<Vec<i64>>());
    }

    #[test]
    fn prototype_loop_action_persistent_overflow_fails_closed() {
        // A kernel that flags overflow at every capacity must hit the slot cap
        // and fail closed to the interpreter — never a truncated set.
        let actions = vec![TrustCgCompiledAction::new_loop(
            action_descriptor("recv_loop", 0),
            loop_always_overflow,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, Vec::new(), 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[0], 1, &mut out)
            .expect_err("persistent sink overflow must fail closed");

        assert_eq!(err, TrustCgBfsLevelError::FallbackNeeded);
        assert_eq!(out.successor_count(), 0);
    }

    #[test]
    fn prototype_loop_action_error_statuses_propagate() {
        for (kernel, expected) in [
            (
                loop_runtime_error as tla_jit_abi::NextStateLoopFn,
                TrustCgBfsLevelError::RuntimeError,
            ),
            (
                loop_fallback_needed as tla_jit_abi::NextStateLoopFn,
                TrustCgBfsLevelError::FallbackNeeded,
            ),
        ] {
            let actions = vec![TrustCgCompiledAction::new_loop(
                action_descriptor("recv_loop", 0),
                kernel,
            )];
            let mut level =
                TrustCgBfsLevelPrototype::new(1, actions, Vec::new(), 16).expect("build prototype");
            let mut out = TrustCgSuccessorArena::new(1);

            let err = level
                .run_level_arena(&[0], 1, &mut out)
                .expect_err("loop kernel error status must propagate");
            assert_eq!(err, expected);
            assert_eq!(out.successor_count(), 0);
        }
    }

    #[test]
    fn prototype_loop_action_invariant_failure_reports_indexes() {
        // Parent [2]: the loop kernel pushes 3 then 4; slot0 < 3 fails on the
        // FIRST loop-emitted successor, so the level stops there with stable
        // parent/invariant/successor indexes (mirrors the single-ABI path).
        let actions = vec![TrustCgCompiledAction::new_loop(
            action_descriptor("recv_loop", 0),
            loop_push_two,
        )];
        let invariants = vec![TrustCgCompiledInvariant::new(
            invariant_descriptor("slot0_lt_3", 9),
            invariant_slot0_lt_3,
        )];
        let mut level =
            TrustCgBfsLevelPrototype::new(1, actions, invariants, 16).expect("build prototype");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level.run_level_arena(&[2], 1, &mut out).expect("run level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 1);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(
            outcome.invariant,
            TrustCgInvariantStatus::Failed {
                parent_index: 0,
                invariant_index: 9,
                successor_index: 0,
            }
        );
        assert_eq!(out.successor(0), Some(&[3][..]));
    }

    #[test]
    fn compiled_action_loop_constructor_tags_and_exposes_loop_fn() {
        let single = TrustCgCompiledAction::new(action_descriptor("inc", 0), increment_slot0);
        assert!(!single.is_loop());
        assert!(single.loop_fn().is_none());

        let looped =
            TrustCgCompiledAction::new_loop(action_descriptor("recv_loop", 1), loop_push_two);
        assert!(looped.is_loop());
        let loop_fn = looped.loop_fn().expect("loop fn present");
        assert_eq!(
            loop_fn as usize, looped.func as usize,
            "bookkeeping func pointer must alias the loop kernel address",
        );
        let debug = format!("{looped:?}");
        assert!(debug.contains("is_loop: true"), "debug output: {debug}");
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_reports_fused_loop_after_symbol_lookup() {
        let module = make_native_bfs_reported_counter_module(2, 0, 0);
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile native BFS status stub");

        let mut level = TrustCgBfsLevelNative::from_library(1, lib, "trust_cg_bfs_level")
            .expect("native symbol lookup should succeed");
        assert_eq!(
            level.capabilities(),
            TrustCgBfsLevelCapabilities::NATIVE_FUSED_NO_LOCAL_DEDUP
        );
        assert!(level.capabilities().native_fused_loop);
        assert!(!level.capabilities().local_dedup);
        assert_eq!(level.action_count(), 0);
        assert_eq!(level.successor_capacity_for(2).expect("capacity"), 0);
        assert_eq!(level.symbol_name(), "trust_cg_bfs_level");

        let mut out = TrustCgSuccessorArena::new(1);
        let outcome = level
            .run_level_arena_with_capacity(&[7, 8], 2, 0, &mut out)
            .expect("native stub should return ok");
        assert_eq!(outcome.total_new, 0);
        assert_eq!(outcome.total_generated, 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_missing_symbol_does_not_report_native_capability() {
        let module = make_native_bfs_status_module();
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile native BFS status stub");

        let err = TrustCgBfsLevelNative::from_library(1, lib, "missing_bfs_level")
            .expect_err("missing native symbol should fail construction");
        assert!(
            matches!(err, TrustCgError::Loading(_)),
            "missing symbol should surface as loading error, got {err}"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_after_invalid_reported_counters() {
        let module = make_native_bfs_reported_counter_module(1, 2, 0);
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile invalid-counter native BFS stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            1,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena_with_capacity(&[0], 1, 2_000, &mut out)
            .expect_err("invalid reported counter must fail before success");

        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native generated count 2 exceeds single-successor action upper bound 1")
        ));
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "invalid native telemetry must reset local dedup state before returning"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_clears_out_on_metadata_capacity_pre_call_error() {
        let module = make_native_bfs_status_module();
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile native BFS status stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(usize::MAX, 0, usize::MAX, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            1,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        level.local_seen = SingleThreadFpSet::new(4096);
        let mut out = TrustCgSuccessorArena::new(1);
        out.push_successor(0, &[99]).expect("seed stale successor");
        out.set_generated(1);

        let err = level
            .run_level_arena(&[1, 2], 2, &mut out)
            .expect_err("capacity overflow should fail before native call");

        assert_eq!(err, TrustCgBfsLevelError::CapacityOverflow);
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "metadata pre-call failures must reset local dedup state"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_on_parent_prevalidation_error() {
        let module = make_native_bfs_status_module();
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile native BFS status stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            1,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        level.local_seen = SingleThreadFpSet::new(4096);
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena_with_capacity(&[1, 2], 3, 3, &mut out)
            .expect_err("parent arena length mismatch should fail before native call");

        assert!(matches!(
            err,
            TrustCgBfsLevelError::ParentArenaLenMismatch {
                expected: 3,
                actual: 2
            }
        ));
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "parent pre-validation failures must reset stale local dedup state"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_on_output_state_len_error() {
        let module = make_native_bfs_status_module();
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile native BFS status stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            1,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        level.local_seen = SingleThreadFpSet::new(4096);
        let mut out = TrustCgSuccessorArena::new(2);

        let err = level
            .run_level_arena_with_capacity(&[1], 1, 1, &mut out)
            .expect_err("output state_len mismatch should fail before native call");

        assert_eq!(
            err,
            TrustCgBfsLevelError::StateLenMismatch {
                expected: 1,
                actual: 2,
            }
        );
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "output validation failures must reset stale local dedup state"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_on_prepare_abi_error() {
        let module = make_native_bfs_status_module();
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile native BFS status stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            2,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        level.local_seen = SingleThreadFpSet::new(4096);
        let mut out = TrustCgSuccessorArena::new(2);

        let err = level
            .run_level_arena_with_capacity(&[1, 2], 1, usize::MAX, &mut out)
            .expect_err("successor ABI capacity overflow should fail before native call");

        assert_eq!(err, TrustCgBfsLevelError::CapacityOverflow);
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "prepare_abi failures must reset local dedup state"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_and_clears_out_after_status_mismatch() {
        let module = make_native_bfs_reported_counter_status_module(
            0,
            0,
            0,
            TrustCgBfsLevelStatus::Ok,
            TrustCgBfsLevelStatus::RuntimeError,
        );
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile status-mismatch native BFS stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            1,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        let mut out = TrustCgSuccessorArena::new(1);
        out.push_successor(0, &[99]).expect("seed stale successor");
        out.set_generated(1);
        let parents = vec![0; 2_000];

        let err = level
            .run_level_arena_with_capacity(&parents, 2_000, 2_000, &mut out)
            .expect_err("status mismatch must fail before committing arena");

        assert!(matches!(
            err,
            TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native status mismatch")
        ));
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "status mismatches must reset local dedup state"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_and_clears_out_after_buffer_overflow() {
        let module = make_native_bfs_reported_counter_status_module(
            0,
            9,
            1,
            TrustCgBfsLevelStatus::BufferOverflow,
            TrustCgBfsLevelStatus::BufferOverflow,
        );
        let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
            .expect("compile buffer-overflow native BFS stub");
        let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
            1,
            lib,
            "trust_cg_bfs_level",
            metadata,
        )
        .expect("native symbol lookup should succeed");
        let mut out = TrustCgSuccessorArena::new(1);
        out.push_successor(0, &[99]).expect("seed stale successor");
        out.set_generated(1);
        let parents = vec![0; 2_000];

        let err = level
            .run_level_arena_with_capacity(&parents, 2_000, 2_000, &mut out)
            .expect_err("buffer overflow status must fail");

        assert_eq!(
            err,
            TrustCgBfsLevelError::BufferOverflow { partial_count: 1 }
        );
        assert_eq!(
            level.local_seen.capacity(),
            1024,
            "buffer-overflow statuses must reset local dedup state"
        );
        assert_eq!(out.successor_count(), 0);
        assert_eq!(out.generated(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_wrapper_resets_local_seen_after_terminal_error_statuses() {
        for status in [
            TrustCgBfsLevelStatus::RuntimeError,
            TrustCgBfsLevelStatus::FallbackNeeded,
            TrustCgBfsLevelStatus::InvalidAbi,
        ] {
            let module = make_native_bfs_reported_counter_status_module(0, 0, 0, status, status);
            let lib = crate::compile::compile_module_native(&module, crate::compile::OptLevel::O1)
                .expect("compile terminal-status native BFS stub");
            let metadata = TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true);
            let mut level = TrustCgBfsLevelNative::from_library_with_metadata(
                1,
                lib,
                "trust_cg_bfs_level",
                metadata,
            )
            .expect("native symbol lookup should succeed");
            let mut out = TrustCgSuccessorArena::new(1);
            out.push_successor(0, &[99]).expect("seed stale successor");
            out.set_generated(1);
            let parents = vec![0; 2_000];

            let err = level
                .run_level_arena_with_capacity(&parents, 2_000, 2_000, &mut out)
                .expect_err("terminal status must fail");

            match status {
                TrustCgBfsLevelStatus::RuntimeError => {
                    assert_eq!(err, TrustCgBfsLevelError::RuntimeError);
                }
                TrustCgBfsLevelStatus::FallbackNeeded => {
                    assert_eq!(err, TrustCgBfsLevelError::FallbackNeeded);
                }
                TrustCgBfsLevelStatus::InvalidAbi => {
                    assert!(matches!(
                        err,
                        TrustCgBfsLevelError::InvalidAbi(message)
                            if message.contains("native function reported invalid ABI")
                    ));
                }
                _ => unreachable!("test covers terminal error statuses only"),
            }
            assert_eq!(
                level.local_seen.capacity(),
                1024,
                "terminal status {status:?} must reset local dedup state"
            );
            assert_eq!(out.successor_count(), 0);
            assert_eq!(out.generated(), 0);
        }
    }
}
