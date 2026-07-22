// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Arena-backed BFS frontier for FlatState buffers.
//!
//! `FlatBfsFrontier` implements [`BfsFrontier`] using [`FlatStateStore`] as a
//! contiguous `Vec<i64>` arena instead of individual `Box<[i64]>` per state.
//! This eliminates per-state heap allocation on the BFS hot path and provides
//! cache-friendly sequential access during frontier iteration.
//!
//! # Design
//!
//! The frontier wraps a single `FlatStateStore` with a parallel metadata `Vec`
//! for per-state `(Fingerprint, depth, trace_loc)`. States are appended via
//! `push()` and consumed FIFO via `pop()` (tracked by a read cursor).
//!
//! When a `NoTraceQueueEntry::Flat` is pushed, only the raw `&[i64]` buffer
//! is copied into the arena (zero Box allocation). `Owned` ArrayState entries
//! are written directly into the arena when they fit the fixed layout, avoiding
//! a temporary boxed `FlatState`; otherwise non-flat entries fall back to a
//! `VecDeque`.
//!
//! On `pop()`, flat entries are reconstructed from the arena slice into a
//! `FlatState` (one `Box<[i64]>` clone per pop — but this state will be
//! immediately consumed by the interpreter, so the allocation is on the
//! cold path between dequeue and eval, not on the hot dedup path).
//!
//! # Memory savings
//!
//! For EWD998 N=3 (15 slots = 120 bytes/state):
//! - **Before**: `Box<[i64]>` + `FlatState` wrapper per state = 120 + 24 = 144 bytes
//! - **After**: 120 bytes contiguous in arena + 24 bytes metadata = 144 bytes
//!
//! The savings come from eliminating allocator overhead (no per-Box malloc
//! header, no fragmentation) and from sequential access patterns that maximize
//! L1/L2 cache utilization during frontier iteration.
//!
//! Part of #4126: FlatState as native BFS representation.

use std::collections::VecDeque;
use std::sync::Arc;

use super::super::frontier::BfsFrontier;
use super::storage_modes::NoTraceQueueEntry;
use crate::state::Fingerprint;
use crate::state::FlatState;
use crate::state::FlatStateStore;
use crate::state::StateLayout;

/// Minimum number of consumed states before a partial compaction pays off.
///
/// Below this, the arena is small enough that shifting is pure overhead;
/// above it, the `consumed >= remaining` trigger keeps the shift amortized
/// O(1) per pushed state.
const PARTIAL_COMPACT_MIN_CONSUMED: usize = 1024;

/// Per-state metadata stored alongside the flat buffer in the arena.
///
/// Each entry corresponds to one state in the `FlatStateStore` at the same
/// index. This parallel-arrays layout keeps the i64 arena tightly packed
/// for SIMD/cache-line-friendly fingerprinting while storing variable-size
/// metadata separately.
#[derive(Debug, Clone)]
struct FlatFrontierMeta {
    /// 64-bit fingerprint for dedup (traditional FP64 space).
    fp: Fingerprint,
    /// BFS depth of this state.
    depth: usize,
    /// Trace file location for trace reconstruction.
    trace_loc: u64,
}

/// Arena-backed BFS frontier that stores flat i64 state buffers contiguously.
///
/// Implements `BfsFrontier<Entry = (NoTraceQueueEntry, usize, u64)>` so it
/// is a drop-in replacement for `VecDeque` in the fingerprint-only BFS path.
///
/// States pushed as `NoTraceQueueEntry::Flat` are stored in the contiguous
/// arena. `Owned` entries are also stored there when the fixed layout can
/// represent them. `Bulk` and non-flattenable `Owned` entries are stored in a
/// fallback `VecDeque` (these typically only appear for initial states and
/// specs with Dynamic variables that cannot be flattened).
///
/// Part of #4126.
pub(in crate::check::model_checker) struct FlatBfsFrontier {
    /// Contiguous arena for flat state buffers.
    store: FlatStateStore,
    /// Per-state metadata (parallel to store indices).
    meta: Vec<FlatFrontierMeta>,
    /// Read cursor: next state index to pop from the arena.
    read_idx: usize,
    /// Fallback queue for non-flat entries (Bulk, Owned).
    fallback: VecDeque<(NoTraceQueueEntry, usize, u64)>,
    /// Shared layout for constructing FlatState from arena slices.
    layout: Arc<StateLayout>,
    /// Total states pushed (flat + fallback), for statistics.
    total_pushed: u64,
    /// States pushed to the flat arena (subset of total_pushed).
    flat_pushed: u64,
}

impl FlatBfsFrontier {
    /// Create a new arena-backed frontier for the given layout.
    ///
    /// The arena starts empty; capacity grows on demand.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(in crate::check::model_checker) fn new(layout: Arc<StateLayout>) -> Self {
        FlatBfsFrontier {
            store: FlatStateStore::new(Arc::clone(&layout)),
            meta: Vec::new(),
            read_idx: 0,
            fallback: VecDeque::new(),
            layout,
            total_pushed: 0,
            flat_pushed: 0,
        }
    }

    /// Create a new frontier with pre-allocated capacity for `capacity` states.
    #[must_use]
    pub(in crate::check::model_checker) fn with_capacity(
        layout: Arc<StateLayout>,
        capacity: usize,
    ) -> Self {
        FlatBfsFrontier {
            store: FlatStateStore::with_capacity(Arc::clone(&layout), capacity),
            meta: Vec::with_capacity(capacity),
            read_idx: 0,
            fallback: VecDeque::new(),
            layout,
            total_pushed: 0,
            flat_pushed: 0,
        }
    }

    /// Number of flat states remaining in the arena (not yet popped).
    #[must_use]
    fn flat_remaining(&self) -> usize {
        self.store.len().saturating_sub(self.read_idx)
    }

    /// Compact the arena by discarding already-popped states.
    ///
    /// Called when the read cursor has advanced past a significant fraction
    /// of the arena, freeing memory from already-consumed states. This is
    /// a bulk operation — O(remaining) to shift data.
    ///
    /// BFS pushes the next level while the current level is still being
    /// consumed, so the arena is never fully drained mid-run: without
    /// partial compaction it would retain every state ever pushed (for
    /// PaxosCommit-class runs, the entire reachable state space, ~184
    /// bytes/state). The partial branch discards the consumed prefix once
    /// it is at least as large as the remainder: each consumed state pays
    /// for at most one shifted state, keeping the memmove amortized O(1)
    /// per push while bounding the arena to roughly the current + next
    /// level. All external addressing is relative to the read cursor, so
    /// shifting the remainder to the front is observationally transparent.
    fn compact_if_needed(&mut self) {
        if self.read_idx == 0 {
            return;
        }
        let remaining = self.flat_remaining();
        if remaining == 0 {
            self.store.clear();
            self.meta.clear();
            self.read_idx = 0;
        } else if self.read_idx >= PARTIAL_COMPACT_MIN_CONSUMED && self.read_idx >= remaining {
            self.store.discard_prefix_states(self.read_idx);
            self.meta.drain(..self.read_idx);
            self.read_idx = 0;
        }
    }

    /// Number of states pushed to the flat arena.
    #[must_use]
    pub(in crate::check::model_checker) fn flat_pushed(&self) -> u64 {
        self.flat_pushed
    }

    /// Number of flat states remaining from the current read cursor.
    #[must_use]
    pub(in crate::check::model_checker) fn remaining_flat_count(&self) -> usize {
        self.flat_remaining()
    }

    /// Whether any non-flat entries are queued in the fallback path.
    #[must_use]
    pub(in crate::check::model_checker) fn has_fallback_entries(&self) -> bool {
        !self.fallback.is_empty()
    }

    /// Total states pushed (flat + fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(in crate::check::model_checker) fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Fraction of states stored in the flat arena (0.0 to 1.0).
    #[must_use]
    pub(in crate::check::model_checker) fn flat_ratio(&self) -> f64 {
        if self.total_pushed == 0 {
            return 0.0;
        }
        self.flat_pushed as f64 / self.total_pushed as f64
    }

    /// Bytes per state in the arena.
    #[must_use]
    pub(in crate::check::model_checker) fn bytes_per_state(&self) -> usize {
        self.store.bytes_per_state()
    }

    /// Number of i64 slots per state in the arena.
    #[must_use]
    pub(in crate::check::model_checker) fn slots_per_state(&self) -> usize {
        self.store.slots_per_state()
    }

    /// Access the raw contiguous i64 arena for the remaining (not-yet-popped) states.
    ///
    /// Returns `(arena_slice, state_count)` where `arena_slice` contains
    /// `state_count * slots_per_state` consecutive i64 values. The slice starts
    /// at the read cursor (skipping already-popped states).
    ///
    /// This is used by the compiled BFS level loop to process the entire
    /// frontier in a single pass through the arena without per-state pop/push
    /// overhead.
    ///
    /// Returns `None` if there are no flat states remaining (only fallback entries).
    ///
    /// Part of #3988: Wire compiled BFS level into model checker.
    #[must_use]
    pub(in crate::check::model_checker) fn remaining_arena(&self) -> Option<(&[i64], usize)> {
        let remaining = self.flat_remaining();
        if remaining == 0 {
            return None;
        }
        let slots_per_state = self.store.slots_per_state();
        let start_slot = self.read_idx * slots_per_state;
        let end_slot = start_slot + remaining * slots_per_state;
        let arena = self.store.raw_arena();
        if end_slot > arena.len() {
            return None;
        }
        Some((&arena[start_slot..end_slot], remaining))
    }

    /// Borrow one remaining flat state by offset from the current read cursor.
    ///
    /// This is the per-parent compiled BFS hot path. The caller borrows only
    /// the current parent, runs the native step, then releases the borrow
    /// before appending successors to this same frontier arena.
    #[must_use]
    pub(in crate::check::model_checker) fn remaining_state_at_offset(
        &self,
        offset: usize,
    ) -> Option<&[i64]> {
        if offset >= self.flat_remaining() {
            return None;
        }

        let idx = self.read_idx + offset;
        self.store.get(idx)
    }

    /// Advance the read cursor by `count` states without popping them.
    ///
    /// Used by the compiled BFS level loop after processing a batch of
    /// states directly from the arena via `remaining_arena()`. The caller
    /// has already processed these states and does not need them popped
    /// individually.
    ///
    /// # Panics
    ///
    /// Panics if `count > flat_remaining()`.
    ///
    /// Part of #3988: Wire compiled BFS level into model checker.
    pub(in crate::check::model_checker) fn advance_read_cursor(&mut self, count: usize) {
        assert!(
            count <= self.flat_remaining(),
            "advance_read_cursor({count}) exceeds remaining {} states",
            self.flat_remaining(),
        );
        self.read_idx += count;
        // Reclaim consumed states (full clear when drained, partial shift
        // when the consumed prefix dominates the remainder).
        self.compact_if_needed();
    }

    /// Get the metadata (fingerprint, depth, trace_loc) for a state at
    /// the given offset from the current read cursor.
    ///
    /// `offset` is relative to the read cursor: 0 = first unpopped state.
    ///
    /// Part of #3988: Wire compiled BFS level into model checker.
    #[must_use]
    pub(in crate::check::model_checker) fn meta_at_offset(
        &self,
        offset: usize,
    ) -> Option<(Fingerprint, usize, u64)> {
        let idx = self.read_idx + offset;
        if idx >= self.meta.len() {
            return None;
        }
        let m = &self.meta[idx];
        Some((m.fp, m.depth, m.trace_loc))
    }

    /// Push multiple raw fixed-width buffers directly into the arena with metadata.
    ///
    /// `buffers` contains the appended states back-to-back. `metadata` supplies
    /// the corresponding `(fingerprint, depth, trace_loc)` tuple for each state,
    /// and therefore defines the number of states in the batch. This keeps
    /// zero-slot layouts representable while preserving arena/metadata alignment.
    ///
    /// Returns the absolute arena index range of the newly appended states.
    ///
    /// # Panics
    ///
    /// Panics if `metadata.len() * slots_per_state` overflows, if `buffers`
    /// does not contain exactly that many slots, or if the push counters
    /// overflow.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::check::model_checker) fn push_raw_buffer_batch(
        &mut self,
        buffers: &[i64],
        metadata: &[(Fingerprint, usize, u64)],
    ) -> std::ops::Range<usize> {
        let state_count = metadata.len();
        let expected_slots = state_count
            .checked_mul(self.store.slots_per_state())
            .expect("FlatBfsFrontier::push_raw_buffer_batch: slot count overflow");
        assert_eq!(
            buffers.len(),
            expected_slots,
            "FlatBfsFrontier::push_raw_buffer_batch: buffers have {} slots for {} metadata entries, expected {} slots",
            buffers.len(),
            state_count,
            expected_slots
        );

        let batch_count = u64::try_from(state_count)
            .expect("FlatBfsFrontier::push_raw_buffer_batch: state count exceeds u64");
        let total_pushed = self
            .total_pushed
            .checked_add(batch_count)
            .expect("FlatBfsFrontier::push_raw_buffer_batch: total_pushed overflow");
        let flat_pushed = self
            .flat_pushed
            .checked_add(batch_count)
            .expect("FlatBfsFrontier::push_raw_buffer_batch: flat_pushed overflow");

        self.store.reserve(state_count);
        self.meta.reserve(state_count);

        let range = self.store.push_buffer_batch(buffers, state_count);
        self.meta.extend(
            metadata
                .iter()
                .map(|(fp, depth, trace_loc)| FlatFrontierMeta {
                    fp: *fp,
                    depth: *depth,
                    trace_loc: *trace_loc,
                }),
        );
        self.total_pushed = total_pushed;
        self.flat_pushed = flat_pushed;

        debug_assert_eq!(
            self.store.len(),
            self.meta.len(),
            "FlatBfsFrontier::push_raw_buffer_batch: arena/metadata alignment drifted"
        );
        range
    }

    /// Push selected raw fixed-width buffers directly into the arena with metadata.
    ///
    /// This variant accepts non-contiguous source buffers and copies each
    /// selected successor straight into the frontier arena. It avoids building
    /// a temporary contiguous batch when the caller has already filtered a
    /// native successor list down to newly inserted states.
    ///
    /// Returns the absolute arena index range of the newly appended states.
    ///
    /// # Panics
    ///
    /// Panics if any selected buffer has the wrong slot count or if the push
    /// counters overflow.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::check::model_checker) fn push_selected_raw_buffers<'a, I>(
        &mut self,
        selected: I,
    ) -> std::ops::Range<usize>
    where
        I: IntoIterator<Item = (&'a [i64], Fingerprint, usize, u64)>,
        I::IntoIter: ExactSizeIterator,
    {
        let selected = selected.into_iter();
        let state_count = selected.len();
        let batch_count = u64::try_from(state_count)
            .expect("FlatBfsFrontier::push_selected_raw_buffers: state count exceeds u64");
        let total_pushed = self
            .total_pushed
            .checked_add(batch_count)
            .expect("FlatBfsFrontier::push_selected_raw_buffers: total_pushed overflow");
        let flat_pushed = self
            .flat_pushed
            .checked_add(batch_count)
            .expect("FlatBfsFrontier::push_selected_raw_buffers: flat_pushed overflow");

        let start = self.store.len();
        let end = start
            .checked_add(state_count)
            .expect("FlatBfsFrontier::push_selected_raw_buffers: state count overflow");
        self.store.reserve(state_count);
        self.meta.reserve(state_count);
        for (buffer, fp, depth, trace_loc) in selected {
            assert_eq!(
                buffer.len(),
                self.store.slots_per_state(),
                "FlatBfsFrontier::push_selected_raw_buffers: buffer has {} slots, expected {}",
                buffer.len(),
                self.store.slots_per_state()
            );
            self.store.push_buffer(buffer);
            self.meta.push(FlatFrontierMeta {
                fp,
                depth,
                trace_loc,
            });
        }

        self.total_pushed = total_pushed;
        self.flat_pushed = flat_pushed;

        debug_assert_eq!(
            self.store.len(),
            self.meta.len(),
            "FlatBfsFrontier::push_selected_raw_buffers: arena/metadata alignment drifted"
        );
        start..end
    }

    /// Push one raw fixed-width buffer selected by state index from a packed arena.
    ///
    /// The source arena remains owned by the fused backend result. This helper
    /// lets the compiled BFS loop enqueue inserted successors directly by
    /// index after batch admission, without building a temporary vector of
    /// selected successor slices.
    ///
    /// # Panics
    ///
    /// Panics if `source_state_len` does not match this frontier layout, if
    /// `source_idx >= source_count`, if the source slot arithmetic overflows,
    /// or if `source_arena` does not contain the indexed state.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::check::model_checker) fn push_raw_buffer_from_arena_index(
        &mut self,
        source_arena: &[i64],
        source_state_len: usize,
        source_count: usize,
        source_idx: usize,
        fp: Fingerprint,
        depth: usize,
        trace_loc: u64,
    ) {
        assert_eq!(
            source_state_len,
            self.store.slots_per_state(),
            "FlatBfsFrontier::push_raw_buffer_from_arena_index: source state has {} slots, expected {}",
            source_state_len,
            self.store.slots_per_state()
        );
        assert!(
            source_idx < source_count,
            "FlatBfsFrontier::push_raw_buffer_from_arena_index: source index {source_idx} out of {source_count}"
        );
        let expected_slots = source_count.checked_mul(source_state_len).expect(
            "FlatBfsFrontier::push_raw_buffer_from_arena_index: source slot count overflow",
        );
        assert!(
            source_arena.len() >= expected_slots,
            "FlatBfsFrontier::push_raw_buffer_from_arena_index: source arena has {} slots, expected at least {}",
            source_arena.len(),
            expected_slots
        );
        let start = source_idx.checked_mul(source_state_len).expect(
            "FlatBfsFrontier::push_raw_buffer_from_arena_index: source start slot overflow",
        );
        let end = start
            .checked_add(source_state_len)
            .expect("FlatBfsFrontier::push_raw_buffer_from_arena_index: source end slot overflow");
        let buffer = source_arena.get(start..end).expect(
            "FlatBfsFrontier::push_raw_buffer_from_arena_index: source buffer index validated",
        );
        self.push_raw_buffer(buffer, fp, depth, trace_loc);
    }

    /// Push one raw fixed-width buffer selected by state index from a packed arena.
    ///
    /// The caller must have already validated the source arena layout, the
    /// selected index, and destination capacity. This is for compiled BFS batch
    /// admission hot paths that need to append already-admitted successors
    /// without materializing a selected-successor tuple vector.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::check::model_checker) fn push_prevalidated_raw_buffer_from_arena_index(
        &mut self,
        source_arena: &[i64],
        source_state_len: usize,
        source_idx: usize,
        fp: Fingerprint,
        depth: usize,
        trace_loc: u64,
    ) {
        debug_assert_eq!(
            source_state_len,
            self.store.slots_per_state(),
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: source state has {} slots, expected {}",
            source_state_len,
            self.store.slots_per_state()
        );
        let source_start = source_idx.checked_mul(source_state_len).expect(
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: source start slot overflow",
        );
        let source_end = source_start.checked_add(source_state_len).expect(
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: source end slot overflow",
        );
        debug_assert!(
            source_end <= source_arena.len(),
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: source buffer index validated"
        );
        let buffer = source_arena.get(source_start..source_end).expect(
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: source buffer index validated",
        );

        self.total_pushed = self.total_pushed.checked_add(1).expect(
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: total_pushed overflow",
        );
        self.store.push_buffer(buffer);
        self.meta.push(FlatFrontierMeta {
            fp,
            depth,
            trace_loc,
        });
        self.flat_pushed = self.flat_pushed.checked_add(1).expect(
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: flat_pushed overflow",
        );

        debug_assert_eq!(
            self.store.len(),
            self.meta.len(),
            "FlatBfsFrontier::push_prevalidated_raw_buffer_from_arena_index: arena/metadata alignment drifted"
        );
    }

    /// Push prevalidated raw fixed-width buffers selected by state index.
    ///
    /// The caller must have already validated the source arena layout, selected
    /// indices, and destination capacity. This keeps the native-fused hot path
    /// from materializing a tuple vector while preserving the contiguous-prefix
    /// bulk-copy fast path.
    pub(in crate::check::model_checker) fn push_prevalidated_raw_buffers_from_arena_indices(
        &mut self,
        source_arena: &[i64],
        source_state_len: usize,
        source_count: usize,
        selected_indices: &[usize],
        fingerprint_values: &[u64],
        depth: usize,
        trace_loc: u64,
    ) -> std::ops::Range<usize> {
        debug_assert_eq!(
            source_state_len,
            self.store.slots_per_state(),
            "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source state has {} slots, expected {}",
            source_state_len,
            self.store.slots_per_state()
        );
        debug_assert!(
            source_arena.len() >= source_count.saturating_mul(source_state_len),
            "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source arena length validated",
        );
        debug_assert!(
            selected_indices
                .iter()
                .all(|&source_idx| source_idx < source_count && source_idx < fingerprint_values.len()),
            "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: selected indices validated",
        );

        let state_count = selected_indices.len();
        let batch_count = u64::try_from(state_count).expect(
            "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: state count exceeds u64",
        );
        let total_pushed = self
            .total_pushed
            .checked_add(batch_count)
            .expect(
                "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: total_pushed overflow",
            );
        let flat_pushed = self
            .flat_pushed
            .checked_add(batch_count)
            .expect(
                "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: flat_pushed overflow",
            );

        let start = self.store.len();
        let end = start
            .checked_add(state_count)
            .expect(
                "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: state count overflow",
            );
        let is_contiguous_prefix = selected_indices
            .iter()
            .enumerate()
            .all(|(expected_idx, &source_idx)| source_idx == expected_idx);

        if is_contiguous_prefix {
            let source_end = state_count.checked_mul(source_state_len).expect(
                "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source prefix slot count overflow",
            );
            let source_prefix = source_arena.get(..source_end).expect(
                "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source prefix validated",
            );
            let range = self.store.push_buffer_batch(source_prefix, state_count);
            debug_assert_eq!(
                range,
                start..end,
                "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: unexpected prefix append range",
            );
            self.meta
                .extend(selected_indices.iter().map(|&source_idx| FlatFrontierMeta {
                    fp: Fingerprint(fingerprint_values[source_idx]),
                    depth,
                    trace_loc,
                }));
        } else {
            let mut cursor = 0;
            while cursor < selected_indices.len() {
                let run_start_idx = selected_indices[cursor];
                let mut run_len = 1;
                while cursor + run_len < selected_indices.len()
                    && selected_indices[cursor + run_len] == run_start_idx + run_len
                {
                    run_len += 1;
                }

                let source_start = run_start_idx.checked_mul(source_state_len).expect(
                    "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source start slot overflow",
                );
                let source_end_idx = run_start_idx.checked_add(run_len).expect(
                    "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source run end overflow",
                );
                let source_end = source_end_idx.checked_mul(source_state_len).expect(
                    "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source end slot overflow",
                );
                let buffers = source_arena.get(source_start..source_end).expect(
                    "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: source buffer index validated",
                );
                self.store.push_buffer_batch(buffers, run_len);
                self.meta
                    .extend(
                        selected_indices[cursor..cursor + run_len]
                            .iter()
                            .map(|&source_idx| FlatFrontierMeta {
                                fp: Fingerprint(fingerprint_values[source_idx]),
                                depth,
                                trace_loc,
                            }),
                    );
                cursor += run_len;
            }
        }

        self.total_pushed = total_pushed;
        self.flat_pushed = flat_pushed;

        debug_assert_eq!(
            self.store.len(),
            self.meta.len(),
            "FlatBfsFrontier::push_prevalidated_raw_buffers_from_arena_indices: arena/metadata alignment drifted"
        );
        start..end
    }

    /// Push selected raw fixed-width buffers by state index from a packed arena.
    ///
    /// This is the native-fused companion to
    /// [`Self::push_raw_buffer_from_arena_index`]: it validates the source arena
    /// once, reserves destination capacity once, updates counters once, then
    /// copies each selected successor into the frontier arena.
    pub(in crate::check::model_checker) fn push_raw_buffers_from_arena_indices<'a, I>(
        &mut self,
        source_arena: &[i64],
        source_state_len: usize,
        source_count: usize,
        selected: I,
    ) -> std::ops::Range<usize>
    where
        I: IntoIterator<Item = &'a (usize, Fingerprint, usize, u64)>,
        I::IntoIter: Clone + ExactSizeIterator,
    {
        assert_eq!(
            source_state_len,
            self.store.slots_per_state(),
            "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source state has {} slots, expected {}",
            source_state_len,
            self.store.slots_per_state()
        );
        let expected_slots = source_count.checked_mul(source_state_len).expect(
            "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source slot count overflow",
        );
        assert!(
            source_arena.len() >= expected_slots,
            "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source arena has {} slots, expected at least {}",
            source_arena.len(),
            expected_slots
        );

        let selected = selected.into_iter();
        let state_count = selected.len();
        let batch_count = u64::try_from(state_count).expect(
            "FlatBfsFrontier::push_raw_buffers_from_arena_indices: state count exceeds u64",
        );
        let total_pushed = self
            .total_pushed
            .checked_add(batch_count)
            .expect("FlatBfsFrontier::push_raw_buffers_from_arena_indices: total_pushed overflow");
        let flat_pushed = self
            .flat_pushed
            .checked_add(batch_count)
            .expect("FlatBfsFrontier::push_raw_buffers_from_arena_indices: flat_pushed overflow");

        let start = self.store.len();
        let end = start
            .checked_add(state_count)
            .expect("FlatBfsFrontier::push_raw_buffers_from_arena_indices: state count overflow");

        for (source_idx, _, _, _) in selected.clone() {
            assert!(
                *source_idx < source_count,
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source index {} out of {}",
                *source_idx,
                source_count
            );
            let source_start = source_idx.checked_mul(source_state_len).expect(
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source start slot overflow",
            );
            let source_end = source_start.checked_add(source_state_len).expect(
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source end slot overflow",
            );
            assert!(
                source_end <= expected_slots,
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source buffer index validated",
            );
        }

        let is_contiguous_prefix = selected
            .clone()
            .enumerate()
            .all(|(expected_idx, (source_idx, _, _, _))| *source_idx == expected_idx);

        self.store.reserve(state_count);
        self.meta.reserve(state_count);
        if is_contiguous_prefix {
            let source_end = state_count.checked_mul(source_state_len).expect(
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source prefix slot count overflow",
            );
            let source_prefix = source_arena.get(..source_end).expect(
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source prefix validated",
            );
            let range = self.store.push_buffer_batch(source_prefix, state_count);
            debug_assert_eq!(
                range,
                start..end,
                "FlatBfsFrontier::push_raw_buffers_from_arena_indices: unexpected prefix append range"
            );
            self.meta
                .extend(selected.map(|(_, fp, depth, trace_loc)| FlatFrontierMeta {
                    fp: *fp,
                    depth: *depth,
                    trace_loc: *trace_loc,
                }));
        } else {
            for (source_idx, fp, depth, trace_loc) in selected {
                let source_start = source_idx.checked_mul(source_state_len).expect(
                    "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source start slot overflow",
                );
                let source_end = source_start.checked_add(source_state_len).expect(
                    "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source end slot overflow",
                );
                let buffer = source_arena.get(source_start..source_end).expect(
                    "FlatBfsFrontier::push_raw_buffers_from_arena_indices: source buffer index validated",
                );
                self.store.push_buffer(buffer);
                self.meta.push(FlatFrontierMeta {
                    fp: *fp,
                    depth: *depth,
                    trace_loc: *trace_loc,
                });
            }
        }

        self.total_pushed = total_pushed;
        self.flat_pushed = flat_pushed;

        debug_assert_eq!(
            self.store.len(),
            self.meta.len(),
            "FlatBfsFrontier::push_raw_buffers_from_arena_indices: arena/metadata alignment drifted"
        );
        start..end
    }

    /// Try to reserve capacity for additional raw-buffer frontier entries.
    pub(in crate::check::model_checker) fn try_reserve_raw_buffers(
        &mut self,
        additional: usize,
    ) -> Result<(), String> {
        self.store.try_reserve(additional)?;
        self.meta
            .try_reserve(additional)
            .map_err(|err| format!("FlatBfsFrontier metadata reserve failed: {err}"))
    }

    /// Push a raw `&[i64]` buffer directly into the arena with metadata.
    ///
    /// This is the zero-allocation enqueue path for the compiled BFS loop:
    /// successor buffers produced by the JIT are raw `&[i64]` slices, and
    /// constructing a `FlatState` (which requires `Box<[i64]>`) just to push
    /// the buffer into the arena is pure overhead. This method copies the
    /// buffer directly into the contiguous arena.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len()` does not match the layout's slot count.
    ///
    /// Part of #3986: Phase 3 zero-alloc compiled BFS enqueue.
    pub(in crate::check::model_checker) fn push_raw_buffer(
        &mut self,
        buffer: &[i64],
        fp: Fingerprint,
        depth: usize,
        trace_loc: u64,
    ) {
        assert_eq!(
            buffer.len(),
            self.store.slots_per_state(),
            "FlatBfsFrontier::push_raw_buffer: buffer has {} slots, expected {}",
            buffer.len(),
            self.store.slots_per_state()
        );
        self.total_pushed = self
            .total_pushed
            .checked_add(1)
            .expect("FlatBfsFrontier::push_raw_buffer: total_pushed overflow");
        self.store.push_buffer(buffer);
        self.meta.push(FlatFrontierMeta {
            fp,
            depth,
            trace_loc,
        });
        self.flat_pushed = self
            .flat_pushed
            .checked_add(1)
            .expect("FlatBfsFrontier::push_raw_buffer: flat_pushed overflow");

        debug_assert_eq!(
            self.store.len(),
            self.meta.len(),
            "FlatBfsFrontier::push_raw_buffer: arena/metadata alignment drifted"
        );
    }

    /// Log frontier statistics to stderr.
    pub(in crate::check::model_checker) fn report_stats(&self) {
        telemetry_eprintln!(
            "[flat-frontier] flat_bfs_frontier_active={}: {} total pushed, {} flat ({:.1}%), {} fallback, {} bytes/state",
            self.flat_pushed > 0 && !self.has_fallback_entries(),
            self.total_pushed,
            self.flat_pushed,
            self.flat_ratio() * 100.0,
            self.total_pushed - self.flat_pushed,
            self.bytes_per_state(),
        );
    }
}

impl BfsFrontier for FlatBfsFrontier {
    type Entry = (NoTraceQueueEntry, usize, u64);

    fn push(&mut self, (entry, depth, trace_loc): (NoTraceQueueEntry, usize, u64)) {
        self.total_pushed += 1;
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                // Hot path: store raw buffer in contiguous arena.
                self.store.push_buffer(flat.buffer());
                self.meta.push(FlatFrontierMeta {
                    fp,
                    depth,
                    trace_loc,
                });
                self.flat_pushed += 1;
            }
            NoTraceQueueEntry::Owned { state, fp } => {
                if self.store.try_push_array_state(&state) {
                    self.meta.push(FlatFrontierMeta {
                        fp,
                        depth,
                        trace_loc,
                    });
                    self.flat_pushed += 1;
                } else {
                    self.fallback.push_back((
                        NoTraceQueueEntry::Owned { state, fp },
                        depth,
                        trace_loc,
                    ));
                }
            }
            _ => {
                // Cold path: Bulk and other non-flat entries go to fallback queue.
                self.fallback.push_back((entry, depth, trace_loc));
            }
        }
    }

    fn pop(&mut self) -> Option<(NoTraceQueueEntry, usize, u64)> {
        // Drain fallback first (initial states, non-flat entries).
        if let Some(entry) = self.fallback.pop_front() {
            return Some(entry);
        }

        // Then drain the flat arena.
        if self.read_idx < self.store.len() {
            let idx = self.read_idx;
            self.read_idx += 1;

            let buffer_slice = self
                .store
                .get(idx)
                .expect("invariant: read_idx < store.len()");
            // Copy metadata fields before potential compaction (avoids borrow conflict).
            let fp = self.meta[idx].fp;
            let depth = self.meta[idx].depth;
            let trace_loc = self.meta[idx].trace_loc;

            // Construct FlatState from arena slice (one Box allocation).
            let buffer: Box<[i64]> = buffer_slice.to_vec().into_boxed_slice();
            let flat = FlatState::from_buffer(buffer, Arc::clone(&self.layout));

            let entry = NoTraceQueueEntry::Flat { flat, fp };

            // Reclaim consumed states (full clear when drained, partial
            // shift when the consumed prefix dominates the remainder).
            self.compact_if_needed();

            return Some((entry, depth, trace_loc));
        }

        None
    }

    fn len(&self) -> usize {
        self.fallback.len() + self.flat_remaining()
    }

    fn release_after_complete_bfs(&mut self) {
        // `clear` retains the largest level's arena. Replace all queue backing
        // stores so post-BFS liveness does not carry that BFS-only capacity.
        // Keep cumulative counters: report_stats() runs after the checker
        // returns and must continue to describe the completed exploration.
        self.store = FlatStateStore::new(Arc::clone(&self.layout));
        self.meta = Vec::new();
        self.read_idx = 0;
        self.fallback = VecDeque::new();
    }

    fn checkpoint_entries(&self) -> Vec<(NoTraceQueueEntry, usize, u64)>
    where
        (NoTraceQueueEntry, usize, u64): Clone,
    {
        let mut entries: Vec<(NoTraceQueueEntry, usize, u64)> =
            self.fallback.iter().cloned().collect();
        for idx in self.read_idx..self.store.len() {
            let buffer_slice = self.store.get(idx).expect("invariant: idx < store.len()");
            let buffer: Box<[i64]> = buffer_slice.to_vec().into_boxed_slice();
            let meta = &self.meta[idx];
            let flat = FlatState::from_buffer(buffer, Arc::clone(&self.layout));
            entries.push((
                NoTraceQueueEntry::Flat { flat, fp: meta.fp },
                meta.depth,
                meta.trace_loc,
            ));
        }
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{StateLayout, VarLayoutKind};
    use crate::var_index::VarRegistry;

    fn scalar_layout_3() -> Arc<StateLayout> {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        Arc::new(StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::Scalar,
                VarLayoutKind::Scalar,
                VarLayoutKind::Scalar,
            ],
        ))
    }

    fn dynamic_layout_1() -> Arc<StateLayout> {
        let registry = VarRegistry::from_names(["x"]);
        Arc::new(StateLayout::new(&registry, vec![VarLayoutKind::Dynamic]))
    }

    #[test]
    fn release_after_complete_bfs_empties_frontier_and_preserves_counters() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::with_capacity(Arc::clone(&layout), 128);
        frontier.push(make_flat_entry(&[1, 2, 3], 11, 0, 7, &layout));
        frontier.push(make_flat_entry(&[4, 5, 6], 12, 1, 8, &layout));

        let total_pushed = frontier.total_pushed();
        let flat_pushed = frontier.flat_pushed();
        frontier.release_after_complete_bfs();

        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.remaining_flat_count(), 0);
        assert!(!frontier.has_fallback_entries());
        assert!(frontier.remaining_arena().is_none());
        assert_eq!(frontier.meta.capacity(), 0);
        assert_eq!(frontier.total_pushed(), total_pushed);
        assert_eq!(frontier.flat_pushed(), flat_pushed);
    }

    fn make_flat_entry(
        vals: &[i64],
        fp: u64,
        depth: usize,
        trace_loc: u64,
        layout: &Arc<StateLayout>,
    ) -> (NoTraceQueueEntry, usize, u64) {
        let buffer: Box<[i64]> = vals.to_vec().into_boxed_slice();
        let flat = FlatState::from_buffer(buffer, Arc::clone(layout));
        (
            NoTraceQueueEntry::Flat {
                flat,
                fp: Fingerprint(fp),
            },
            depth,
            trace_loc,
        )
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_empty() {
        let layout = scalar_layout_3();
        let frontier = FlatBfsFrontier::new(layout);

        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.flat_pushed(), 0);
        assert_eq!(frontier.total_pushed(), 0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_pop_flat() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push(make_flat_entry(&[1, 2, 3], 100, 0, 0, &layout));
        frontier.push(make_flat_entry(&[4, 5, 6], 200, 0, 1, &layout));
        frontier.push(make_flat_entry(&[7, 8, 9], 300, 1, 2, &layout));

        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 3);
        assert_eq!(frontier.total_pushed(), 3);

        // Pop in FIFO order.
        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[1, 2, 3]);
                assert_eq!(fp, Fingerprint(100));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 0);
        assert_eq!(trace_loc, 0);

        let (entry, depth, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[4, 5, 6]);
                assert_eq!(fp, Fingerprint(200));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 0);

        let (entry, depth, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[7, 8, 9]);
                assert_eq!(fp, Fingerprint(300));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 1);

        assert!(frontier.pop().is_none());
        assert_eq!(frontier.len(), 0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_checkpoint_entries_materialize_arena_states() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push(make_flat_entry(&[1, 2, 3], 10, 0, 100, &layout));
        frontier.push(make_flat_entry(&[4, 5, 6], 20, 1, 200, &layout));

        let entries = frontier.checkpoint_entries();
        assert_eq!(entries.len(), 2);

        match &entries[0].0 {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(*fp, Fingerprint(10));
                assert_eq!(flat.buffer(), &[1, 2, 3]);
            }
            _ => panic!("expected flat entry"),
        }
        assert_eq!(entries[0].1, 0);
        assert_eq!(entries[0].2, 100);

        match &entries[1].0 {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(*fp, Fingerprint(20));
                assert_eq!(flat.buffer(), &[4, 5, 6]);
            }
            _ => panic!("expected flat entry"),
        }
        assert_eq!(entries[1].1, 1);
        assert_eq!(entries[1].2, 200);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_owned_array_state_directly_to_arena() {
        use crate::state::ArrayState;
        use crate::Value;

        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        let owned = ArrayState::from_values(vec![
            Value::SmallInt(10),
            Value::SmallInt(20),
            Value::SmallInt(30),
        ]);
        frontier.push((
            NoTraceQueueEntry::Owned {
                state: owned,
                fp: Fingerprint(999),
            },
            7,
            42,
        ));

        assert_eq!(frontier.remaining_flat_count(), 1);
        assert_eq!(frontier.flat_pushed(), 1);
        assert!(!frontier.has_fallback_entries());

        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[10, 20, 30]);
                assert_eq!(fp, Fingerprint(999));
            }
            _ => panic!("expected Owned entry to be flattened into arena"),
        }
        assert_eq!(depth, 7);
        assert_eq!(trace_loc, 42);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_fallback_drained_first() {
        use crate::state::ArrayState;
        use crate::Value;

        let layout = dynamic_layout_1();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        // Push an Owned entry that cannot be represented by this fixed layout
        // (goes to fallback).
        let owned = ArrayState::from_values(vec![Value::SmallInt(10)]);
        frontier.push((
            NoTraceQueueEntry::Owned {
                state: owned,
                fp: Fingerprint(999),
            },
            0,
            0,
        ));

        // Push a Flat entry (goes to arena).
        frontier.push(make_flat_entry(&[1], 100, 1, 1, &layout));

        assert_eq!(frontier.len(), 2);
        assert_eq!(frontier.flat_pushed(), 1);
        assert_eq!(frontier.total_pushed(), 2);

        // Fallback is drained first.
        let (entry, _, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Owned { fp, .. } => {
                assert_eq!(fp, Fingerprint(999));
            }
            _ => panic!("expected Owned entry from fallback"),
        }

        // Then flat arena.
        let (entry, depth, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[1]);
                assert_eq!(fp, Fingerprint(100));
            }
            _ => panic!("expected Flat entry from arena"),
        }
        assert_eq!(depth, 1);

        assert!(frontier.pop().is_none());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_reports_fallback_when_no_flat_entries_remain() {
        use crate::state::ArrayState;
        use crate::Value;

        let layout = dynamic_layout_1();
        let mut frontier = FlatBfsFrontier::new(layout);

        let owned = ArrayState::from_values(vec![Value::SmallInt(10)]);
        frontier.push((
            NoTraceQueueEntry::Owned {
                state: owned,
                fp: Fingerprint(999),
            },
            0,
            0,
        ));

        assert_eq!(frontier.remaining_flat_count(), 0);
        assert_eq!(frontier.len(), 1);
        assert!(frontier.has_fallback_entries());
        assert_eq!(frontier.flat_pushed(), 0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_compaction() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        // Push 100 states.
        for i in 0..100i64 {
            frontier.push(make_flat_entry(
                &[i, i + 1, i + 2],
                i as u64,
                0,
                i as u64,
                &layout,
            ));
        }

        // Pop all 100 — should trigger compaction.
        for _ in 0..100 {
            assert!(frontier.pop().is_some());
        }

        assert!(frontier.pop().is_none());
        assert_eq!(frontier.len(), 0);
        // After compaction, the arena should be empty.
        assert_eq!(frontier.store.len(), 0);
        assert_eq!(frontier.read_idx, 0);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_stats() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        for i in 0..50i64 {
            frontier.push(make_flat_entry(&[i, 0, 0], i as u64, 0, 0, &layout));
        }

        assert_eq!(frontier.flat_pushed(), 50);
        assert_eq!(frontier.total_pushed(), 50);
        assert!((frontier.flat_ratio() - 1.0).abs() < 1e-10);
        assert_eq!(frontier.bytes_per_state(), 24);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_with_capacity() {
        let layout = scalar_layout_3();
        let frontier = FlatBfsFrontier::with_capacity(Arc::clone(&layout), 1000);

        assert_eq!(frontier.len(), 0);
        assert!(frontier.store.capacity() >= 1000);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffer() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        // Push raw buffers (no FlatState construction).
        frontier.push_raw_buffer(&[10, 20, 30], Fingerprint(100), 0, 0);
        frontier.push_raw_buffer(&[40, 50, 60], Fingerprint(200), 1, 1);
        frontier.push_raw_buffer(&[70, 80, 90], Fingerprint(300), 2, 2);

        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 3);
        assert_eq!(frontier.total_pushed(), 3);

        // Pop in FIFO order — produces the same FlatState as if we had
        // pushed via make_flat_entry.
        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[10, 20, 30]);
                assert_eq!(fp, Fingerprint(100));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 0);
        assert_eq!(trace_loc, 0);

        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[40, 50, 60]);
                assert_eq!(fp, Fingerprint(200));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 1);
        assert_eq!(trace_loc, 1);

        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[70, 80, 90]);
                assert_eq!(fp, Fingerprint(300));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 2);
        assert_eq!(trace_loc, 2);

        assert!(frontier.pop().is_none());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffer_batch_preserves_alignment() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 200);
        frontier.advance_read_cursor(1);

        let range = frontier.push_raw_buffer_batch(
            &[7, 8, 9, 10, 11, 12],
            &[(Fingerprint(30), 1, 300), (Fingerprint(40), 1, 400)],
        );

        assert_eq!(range, 2..4);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 4);
        assert_eq!(frontier.total_pushed(), 4);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 3);
        assert_eq!(arena, &[4, 5, 6, 7, 8, 9, 10, 11, 12]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(20), 0, 200)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(30), 1, 300)));
        assert_eq!(frontier.meta_at_offset(2), Some((Fingerprint(40), 1, 400)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    #[should_panic(expected = "FlatBfsFrontier::push_raw_buffer_batch: buffers have")]
    fn test_flat_frontier_push_raw_buffer_batch_rejects_wrong_width() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(layout);

        frontier.push_raw_buffer_batch(
            &[1, 2, 3, 4, 5],
            &[(Fingerprint(10), 0, 100), (Fingerprint(20), 0, 200)],
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_selected_raw_buffers_preserves_alignment() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 200);
        frontier.advance_read_cursor(1);

        let source = [[7, 8, 9], [10, 11, 12], [13, 14, 15]];
        let selected = [0usize, 2usize];
        let range = frontier.push_selected_raw_buffers(selected.iter().map(|idx| {
            let fp = if *idx == 0 {
                Fingerprint(30)
            } else {
                Fingerprint(50)
            };
            (&source[*idx][..], fp, 1, 300 + u64::try_from(*idx).unwrap())
        }));

        assert_eq!(range, 2..4);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 4);
        assert_eq!(frontier.total_pushed(), 4);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 3);
        assert_eq!(arena, &[4, 5, 6, 7, 8, 9, 13, 14, 15]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(20), 0, 200)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(30), 1, 300)));
        assert_eq!(frontier.meta_at_offset(2), Some((Fingerprint(50), 1, 302)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffer_from_arena_index_preserves_alignment() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.advance_read_cursor(1);

        let source = [7, 8, 9, 10, 11, 12, 13, 14, 15];
        frontier.push_raw_buffer_from_arena_index(&source, 3, 3, 1, Fingerprint(40), 1, 400);

        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.flat_pushed(), 2);
        assert_eq!(frontier.total_pushed(), 2);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 1);
        assert_eq!(arena, &[10, 11, 12]);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(40), 1, 400)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffer_from_arena_index_zero_slots() {
        let registry = VarRegistry::from_names(std::iter::empty::<&str>());
        let layout = Arc::new(StateLayout::new(&registry, vec![]));
        let mut frontier = FlatBfsFrontier::new(layout);

        frontier.push_raw_buffer_from_arena_index(&[], 0, 3, 2, Fingerprint(40), 1, 400);

        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.flat_pushed(), 1);
        assert_eq!(frontier.total_pushed(), 1);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(40), 1, 400)));

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 1);
        assert_eq!(arena, &[] as &[i64]);

        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[] as &[i64]);
                assert_eq!(fp, Fingerprint(40));
            }
            _ => panic!("expected Flat entry"),
        }
        assert_eq!(depth, 1);
        assert_eq!(trace_loc, 400);
        assert!(frontier.pop().is_none());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_prevalidated_raw_buffer_from_arena_index_zero_slots() {
        let registry = VarRegistry::from_names(std::iter::empty::<&str>());
        let layout = Arc::new(StateLayout::new(&registry, vec![]));
        let mut frontier = FlatBfsFrontier::new(layout);

        frontier.push_prevalidated_raw_buffer_from_arena_index(&[], 0, 2, Fingerprint(40), 1, 400);

        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.flat_pushed(), 1);
        assert_eq!(frontier.total_pushed(), 1);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(40), 1, 400)));

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 1);
        assert_eq!(arena, &[] as &[i64]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_prevalidated_raw_buffers_from_arena_indices_contiguous_prefix() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 200);
        frontier.advance_read_cursor(1);

        let source = [7, 8, 9, 10, 11, 12, 13, 14, 15];
        let selected = [0usize, 1usize];
        let fps = [30u64, 40, 50];
        let range = frontier.push_prevalidated_raw_buffers_from_arena_indices(
            &source, 3, 3, &selected, &fps, 1, 300,
        );

        assert_eq!(range, 2..4);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 4);
        assert_eq!(frontier.total_pushed(), 4);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 3);
        assert_eq!(arena, &[4, 5, 6, 7, 8, 9, 10, 11, 12]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(20), 0, 200)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(30), 1, 300)));
        assert_eq!(frontier.meta_at_offset(2), Some((Fingerprint(40), 1, 300)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_prevalidated_raw_buffers_from_arena_indices_sparse() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.advance_read_cursor(1);

        let source = [7, 8, 9, 10, 11, 12, 13, 14, 15];
        let selected = [2usize, 0usize];
        let fps = [30u64, 40, 50];
        let range = frontier.push_prevalidated_raw_buffers_from_arena_indices(
            &source, 3, 3, &selected, &fps, 1, 300,
        );

        assert_eq!(range, 0..2);
        assert_eq!(frontier.len(), 2);
        assert_eq!(frontier.flat_pushed(), 3);
        assert_eq!(frontier.total_pushed(), 3);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 2);
        assert_eq!(arena, &[13, 14, 15, 7, 8, 9]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(50), 1, 300)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(30), 1, 300)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_prevalidated_raw_buffers_from_arena_indices_non_prefix_runs() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        let source = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let selected = [1usize, 2usize, 4usize];
        let fps = [30u64, 40, 50, 60, 70];
        let range = frontier.push_prevalidated_raw_buffers_from_arena_indices(
            &source, 3, 5, &selected, &fps, 1, 300,
        );

        assert_eq!(range, 0..3);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 3);
        assert_eq!(frontier.total_pushed(), 3);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 3);
        assert_eq!(arena, &[10, 11, 12, 13, 14, 15, 19, 20, 21]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(40), 1, 300)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(50), 1, 300)));
        assert_eq!(frontier.meta_at_offset(2), Some((Fingerprint(70), 1, 300)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffers_from_arena_indices_preserves_alignment() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 200);
        frontier.advance_read_cursor(1);

        let source = [7, 8, 9, 10, 11, 12, 13, 14, 15];
        let selected = [
            (2usize, Fingerprint(50), 1usize, 500u64),
            (0usize, Fingerprint(30), 1usize, 300u64),
        ];
        let range = frontier.push_raw_buffers_from_arena_indices(&source, 3, 3, selected.iter());

        assert_eq!(range, 2..4);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 4);
        assert_eq!(frontier.total_pushed(), 4);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 3);
        assert_eq!(arena, &[4, 5, 6, 13, 14, 15, 7, 8, 9]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(20), 0, 200)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(50), 1, 500)));
        assert_eq!(frontier.meta_at_offset(2), Some((Fingerprint(30), 1, 300)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffers_from_arena_indices_contiguous_prefix() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 200);
        frontier.advance_read_cursor(1);

        let source = [7, 8, 9, 10, 11, 12, 13, 14, 15];
        let selected = [
            (0usize, Fingerprint(30), 1usize, 300u64),
            (1usize, Fingerprint(40), 1usize, 400u64),
        ];
        let range = frontier.push_raw_buffers_from_arena_indices(&source, 3, 3, selected.iter());

        assert_eq!(range, 2..4);
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.flat_pushed(), 4);
        assert_eq!(frontier.total_pushed(), 4);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 3);
        assert_eq!(arena, &[4, 5, 6, 7, 8, 9, 10, 11, 12]);

        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(20), 0, 200)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(30), 1, 300)));
        assert_eq!(frontier.meta_at_offset(2), Some((Fingerprint(40), 1, 400)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffers_from_arena_indices_zero_slots() {
        let registry = VarRegistry::from_names(std::iter::empty::<&str>());
        let layout = Arc::new(StateLayout::new(&registry, vec![]));
        let mut frontier = FlatBfsFrontier::new(layout);

        let selected = [
            (2usize, Fingerprint(40), 1usize, 400u64),
            (0usize, Fingerprint(20), 1usize, 200u64),
        ];
        let range = frontier.push_raw_buffers_from_arena_indices(&[], 0, 3, selected.iter());

        assert_eq!(range, 0..2);
        assert_eq!(frontier.len(), 2);
        assert_eq!(frontier.flat_pushed(), 2);
        assert_eq!(frontier.total_pushed(), 2);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(40), 1, 400)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(20), 1, 200)));

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 2);
        assert_eq!(arena, &[] as &[i64]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffers_from_arena_indices_zero_slot_prefix() {
        let registry = VarRegistry::from_names(std::iter::empty::<&str>());
        let layout = Arc::new(StateLayout::new(&registry, vec![]));
        let mut frontier = FlatBfsFrontier::new(layout);

        let selected = [
            (0usize, Fingerprint(20), 1usize, 200u64),
            (1usize, Fingerprint(40), 1usize, 400u64),
        ];
        let range = frontier.push_raw_buffers_from_arena_indices(&[], 0, 3, selected.iter());

        assert_eq!(range, 0..2);
        assert_eq!(frontier.len(), 2);
        assert_eq!(frontier.flat_pushed(), 2);
        assert_eq!(frontier.total_pushed(), 2);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(20), 1, 200)));
        assert_eq!(frontier.meta_at_offset(1), Some((Fingerprint(40), 1, 400)));

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 2);
        assert_eq!(arena, &[] as &[i64]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffers_from_arena_indices_prevalidates_selection() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(layout);

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);

        let source = [7, 8, 9, 10, 11, 12];
        let selected = [
            (1usize, Fingerprint(30), 1usize, 300u64),
            (2usize, Fingerprint(40), 1usize, 400u64),
        ];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            frontier.push_raw_buffers_from_arena_indices(&source, 3, 2, selected.iter());
        }));

        assert!(result.is_err());
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.flat_pushed(), 1);
        assert_eq!(frontier.total_pushed(), 1);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(10), 0, 100)));

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 1);
        assert_eq!(arena, &[1, 2, 3]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_buffers_from_arena_indices_prevalidates_prefix() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(layout);

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 100);

        let source = [7, 8, 9, 10, 11, 12];
        let selected = [
            (0usize, Fingerprint(30), 1usize, 300u64),
            (1usize, Fingerprint(40), 1usize, 400u64),
            (2usize, Fingerprint(50), 1usize, 500u64),
        ];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            frontier.push_raw_buffers_from_arena_indices(&source, 3, 2, selected.iter());
        }));

        assert!(result.is_err());
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.flat_pushed(), 1);
        assert_eq!(frontier.total_pushed(), 1);
        assert_eq!(frontier.meta_at_offset(0), Some((Fingerprint(10), 0, 100)));

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 1);
        assert_eq!(arena, &[1, 2, 3]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    #[should_panic(expected = "FlatBfsFrontier::push_selected_raw_buffers: buffer has")]
    fn test_flat_frontier_push_selected_raw_buffers_rejects_wrong_width() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(layout);

        let wrong_width = [1, 2];
        frontier.push_selected_raw_buffers(std::iter::once((
            &wrong_width[..],
            Fingerprint(10),
            0,
            100,
        )));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_matches_flat_entry() {
        // Verify that push_raw_buffer produces identical results to pushing
        // via NoTraceQueueEntry::Flat (the FlatState wrapper path).
        let layout = scalar_layout_3();
        let vals = [42i64, -7, 100];
        let fp = Fingerprint(12345);

        // Path 1: push via FlatState wrapper.
        let mut frontier1 = FlatBfsFrontier::new(Arc::clone(&layout));
        let buffer: Box<[i64]> = vals.to_vec().into_boxed_slice();
        let flat = FlatState::from_buffer(buffer, Arc::clone(&layout));
        frontier1.push((NoTraceQueueEntry::Flat { flat, fp }, 3, 99));

        // Path 2: push raw buffer directly.
        let mut frontier2 = FlatBfsFrontier::new(Arc::clone(&layout));
        frontier2.push_raw_buffer(&vals, fp, 3, 99);

        // Both must produce identical pop results.
        let (e1, d1, t1) = frontier1.pop().unwrap();
        let (e2, d2, t2) = frontier2.pop().unwrap();
        assert_eq!(d1, d2);
        assert_eq!(t1, t2);
        match (e1, e2) {
            (
                NoTraceQueueEntry::Flat { flat: f1, fp: fp1 },
                NoTraceQueueEntry::Flat { flat: f2, fp: fp2 },
            ) => {
                assert_eq!(f1.buffer(), f2.buffer());
                assert_eq!(fp1, fp2);
            }
            _ => panic!("expected Flat entries from both paths"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_push_raw_remaining_arena() {
        // Verify that push_raw_buffer states are accessible via remaining_arena.
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 0);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 1);

        let (arena, count) = frontier.remaining_arena().unwrap();
        assert_eq!(count, 2);
        assert_eq!(arena, &[1, 2, 3, 4, 5, 6]);

        // Metadata should also be accessible.
        let (fp, depth, trace_loc) = frontier.meta_at_offset(0).unwrap();
        assert_eq!(fp, Fingerprint(10));
        assert_eq!(depth, 0);
        assert_eq!(trace_loc, 0);

        let (fp, depth, trace_loc) = frontier.meta_at_offset(1).unwrap();
        assert_eq!(fp, Fingerprint(20));
        assert_eq!(depth, 0);
        assert_eq!(trace_loc, 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_remaining_state_at_offset_after_append() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        frontier.push_raw_buffer(&[1, 2, 3], Fingerprint(10), 0, 0);
        frontier.push_raw_buffer(&[4, 5, 6], Fingerprint(20), 0, 1);

        assert_eq!(frontier.remaining_flat_count(), 2);
        assert_eq!(frontier.remaining_state_at_offset(0).unwrap(), &[1, 2, 3]);
        assert_eq!(frontier.remaining_state_at_offset(1).unwrap(), &[4, 5, 6]);
        assert!(frontier.remaining_state_at_offset(2).is_none());

        // Appending the next level must not shift the current-level parent
        // offsets while the compiled BFS loop is still processing them.
        frontier.push_raw_buffer(&[7, 8, 9], Fingerprint(30), 1, 2);

        assert_eq!(frontier.remaining_flat_count(), 3);
        assert_eq!(frontier.remaining_state_at_offset(0).unwrap(), &[1, 2, 3]);
        assert_eq!(frontier.remaining_state_at_offset(1).unwrap(), &[4, 5, 6]);
        assert_eq!(frontier.remaining_state_at_offset(2).unwrap(), &[7, 8, 9]);

        frontier.advance_read_cursor(2);
        assert_eq!(frontier.remaining_flat_count(), 1);
        assert_eq!(frontier.remaining_state_at_offset(0).unwrap(), &[7, 8, 9]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_interleaved_push_pop() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        // Push 2, pop 1, push 2, pop 1 — tests cursor management.
        frontier.push(make_flat_entry(&[1, 0, 0], 1, 0, 0, &layout));
        frontier.push(make_flat_entry(&[2, 0, 0], 2, 0, 0, &layout));

        let (entry, _, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, .. } => assert_eq!(flat.buffer()[0], 1),
            _ => panic!("expected Flat"),
        }

        frontier.push(make_flat_entry(&[3, 0, 0], 3, 0, 0, &layout));
        frontier.push(make_flat_entry(&[4, 0, 0], 4, 0, 0, &layout));

        // Remaining: [2, 3, 4]
        assert_eq!(frontier.len(), 3);

        let (entry, _, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, .. } => assert_eq!(flat.buffer()[0], 2),
            _ => panic!("expected Flat"),
        }

        let (entry, _, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, .. } => assert_eq!(flat.buffer()[0], 3),
            _ => panic!("expected Flat"),
        }

        let (entry, _, _) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, .. } => assert_eq!(flat.buffer()[0], 4),
            _ => panic!("expected Flat"),
        }

        assert!(frontier.pop().is_none());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_partial_compaction_preserves_relative_addressing() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        // Simulate the BFS level pattern that previously kept the arena
        // monotonic: push level N, push level N+1 while N is unconsumed,
        // then advance past level N. The consumed prefix must be reclaimed
        // while offsets stay relative to the read cursor.
        let level_n = PARTIAL_COMPACT_MIN_CONSUMED + 500;
        let level_n1 = 700usize; // fewer remaining than consumed → compacts
        for i in 0..level_n {
            frontier.push_raw_buffer(&[i as i64, 0, 0], Fingerprint(i as u64), 3, 30);
        }
        for i in 0..level_n1 {
            frontier.push_raw_buffer(
                &[1_000_000 + i as i64, 1, 1],
                Fingerprint(9_000 + i as u64),
                4,
                40,
            );
        }

        frontier.advance_read_cursor(level_n);

        // Partial compaction fired: cursor reset, only level N+1 retained.
        assert_eq!(frontier.read_idx, 0);
        assert_eq!(frontier.remaining_flat_count(), level_n1);
        assert_eq!(frontier.store.len(), level_n1);
        assert_eq!(frontier.meta.len(), level_n1);

        // Relative addressing yields level N+1 states unchanged.
        for i in 0..level_n1 {
            assert_eq!(
                frontier.remaining_state_at_offset(i).unwrap(),
                &[1_000_000 + i as i64, 1, 1]
            );
            assert_eq!(
                frontier.meta_at_offset(i),
                Some((Fingerprint(9_000 + i as u64), 4, 40))
            );
        }

        // The frontier stays usable: push level N+2 and drain in order.
        frontier.push_raw_buffer(&[-7, 2, 2], Fingerprint(77), 5, 50);
        assert_eq!(frontier.remaining_flat_count(), level_n1 + 1);
        let (entry, depth, trace_loc) = frontier.pop().unwrap();
        match entry {
            NoTraceQueueEntry::Flat { flat, fp } => {
                assert_eq!(flat.buffer(), &[1_000_000, 1, 1]);
                assert_eq!(fp, Fingerprint(9_000));
            }
            _ => panic!("expected Flat"),
        }
        assert_eq!((depth, trace_loc), (4, 40));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_frontier_no_partial_compaction_below_threshold() {
        let layout = scalar_layout_3();
        let mut frontier = FlatBfsFrontier::new(Arc::clone(&layout));

        // Consumed prefix below PARTIAL_COMPACT_MIN_CONSUMED: no shift.
        for i in 0..10i64 {
            frontier.push_raw_buffer(&[i, 0, 0], Fingerprint(i as u64), 0, 0);
        }
        frontier.advance_read_cursor(6);
        assert_eq!(frontier.read_idx, 6);
        assert_eq!(frontier.store.len(), 10);
        assert_eq!(frontier.remaining_flat_count(), 4);
        assert_eq!(frontier.remaining_state_at_offset(0).unwrap(), &[6, 0, 0]);

        // Full drain still clears entirely (capacity-retaining reset).
        frontier.advance_read_cursor(4);
        assert_eq!(frontier.read_idx, 0);
        assert_eq!(frontier.store.len(), 0);
        assert!(frontier.meta.is_empty());
    }
}
