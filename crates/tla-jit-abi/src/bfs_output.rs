// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pure-data output types for the compiled BFS step pipeline.
//!
//! Consumers of compiled BFS output only need stable result shapes, so these
//! types live in the stable ABI crate.
//!
//! Part of #4395.

use thiserror::Error;

/// Owned result of executing one compiled BFS step.
///
/// Produced when one parent state is expanded by a compiled `bfs_step`
/// function. Each entry in `successors` is one flat state vector that passed
/// every checked invariant. When an invariant violation is found, the
/// `failed_*` fields locate it and expansion stops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfsStepOutput {
    /// New successor states, each a flat `state_len`-slot i64 vector. Only
    /// states that satisfied all checked invariants appear here.
    pub successors: Vec<Vec<i64>>,
    /// Total successors the compiled action relation produced before local
    /// dedup; always `>= successors.len()` (used for the generated-states stat).
    pub generated_count: u32,
    /// `true` if every checked invariant held on every successor; `false` if a
    /// violation was found (see the `failed_*` fields).
    pub invariant_ok: bool,
    /// Index into the spec's invariant list of the first invariant that failed,
    /// or `None` if all passed.
    pub failed_invariant_idx: Option<u32>,
    /// Index (within this step's generated successors) of the successor that
    /// violated the invariant, or `None` if all passed.
    pub failed_successor_idx: Option<u32>,
    /// The flat state vector that violated the invariant, captured for trace
    /// reconstruction, or `None` if all passed.
    pub failed_successor: Option<Vec<i64>>,
}

/// Zero-copy result of executing one compiled BFS step.
///
/// Like [`BfsStepOutput`] but keeps the successors packed contiguously in a
/// single owned `i64` buffer (`successor_count` chunks of `state_len` slots)
/// rather than a `Vec<Vec<i64>>`, avoiding a per-successor heap allocation.
/// Iterate the successors with [`FlatBfsStepOutput::iter_successors`].
#[derive(Debug, Clone)]
pub struct FlatBfsStepOutput {
    succ_buf: Vec<i64>,
    state_len: usize,
    successor_count: usize,
    /// Total successors produced before local dedup; `>= successor_count`.
    pub generated_count: u32,
    /// `true` if every checked invariant held; `false` on a violation.
    pub invariant_ok: bool,
    /// Index of the first failing invariant, or `None` if all passed.
    pub failed_invariant_idx: Option<u32>,
    /// Index of the successor that violated the invariant, or `None`.
    pub failed_successor_idx: Option<u32>,
}

impl FlatBfsStepOutput {
    /// Construct a `FlatBfsStepOutput` from its constituent parts.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        succ_buf: Vec<i64>,
        state_len: usize,
        successor_count: usize,
        generated_count: u32,
        invariant_ok: bool,
        failed_invariant_idx: Option<u32>,
        failed_successor_idx: Option<u32>,
    ) -> Self {
        assert!(
            state_len == 0 || succ_buf.len() >= successor_count.saturating_mul(state_len),
            "successor buffer shorter than successor_count * state_len",
        );
        assert!(
            u64::from(generated_count)
                >= u64::try_from(successor_count).expect("successor_count exceeds u64"),
            "generated_count must be at least successor_count",
        );
        Self {
            succ_buf,
            state_len,
            successor_count,
            generated_count,
            invariant_ok,
            failed_invariant_idx,
            failed_successor_idx,
        }
    }

    /// Number of new successors in this output.
    #[must_use]
    pub fn successor_count(&self) -> usize {
        self.successor_count
    }

    /// i64 slots per successor in this output.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Iterate over the flat successor slices.
    pub fn iter_successors(&self) -> impl Iterator<Item = &[i64]> + '_ {
        let len = self.state_len;
        if len == 0 {
            return FlatSuccessorIter::Empty(self.successor_count);
        }
        FlatSuccessorIter::Chunked(self.succ_buf[..self.successor_count * len].chunks_exact(len))
    }
}

/// Borrowed zero-copy view over a compiled BFS step result.
///
/// The borrowing counterpart of [`FlatBfsStepOutput`]: the successors are
/// borrowed from a caller-owned buffer (e.g. the trust-codegen arena) rather
/// than owned, so the view itself is `Copy` and allocation-free.
#[derive(Debug, Clone, Copy)]
pub struct FlatBfsStepOutputRef<'a> {
    succ_buf: &'a [i64],
    state_len: usize,
    successor_count: usize,
    /// Total successors produced before local dedup; `>= successor_count`.
    pub generated_count: u32,
    /// `true` if every checked invariant held; `false` on a violation.
    pub invariant_ok: bool,
    /// Index of the first failing invariant, or `None` if all passed.
    pub failed_invariant_idx: Option<u32>,
    /// Index of the successor that violated the invariant, or `None`.
    pub failed_successor_idx: Option<u32>,
}

impl<'a> FlatBfsStepOutputRef<'a> {
    /// Construct a `FlatBfsStepOutputRef` from its constituent parts.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        succ_buf: &'a [i64],
        state_len: usize,
        successor_count: usize,
        generated_count: u32,
        invariant_ok: bool,
        failed_invariant_idx: Option<u32>,
        failed_successor_idx: Option<u32>,
    ) -> Self {
        assert!(
            state_len == 0 || succ_buf.len() >= successor_count.saturating_mul(state_len),
            "successor buffer shorter than successor_count * state_len",
        );
        assert!(
            u64::from(generated_count)
                >= u64::try_from(successor_count).expect("successor_count exceeds u64"),
            "generated_count must be at least successor_count",
        );
        Self {
            succ_buf,
            state_len,
            successor_count,
            generated_count,
            invariant_ok,
            failed_invariant_idx,
            failed_successor_idx,
        }
    }

    /// Number of new successors in this output.
    #[must_use]
    pub fn successor_count(&self) -> usize {
        self.successor_count
    }

    /// Iterate over the flat successor slices.
    pub fn iter_successors(&self) -> impl Iterator<Item = &[i64]> + '_ {
        if self.state_len == 0 {
            return FlatSuccessorIter::Empty(self.successor_count);
        }
        let visible_len = self.successor_count * self.state_len;
        FlatSuccessorIter::Chunked(self.succ_buf[..visible_len].chunks_exact(self.state_len))
    }

    /// Pointer to the underlying successor buffer.
    #[must_use]
    pub fn succ_buf_as_ptr(&self) -> *const i64 {
        self.succ_buf.as_ptr()
    }
}

enum FlatSuccessorIter<'a> {
    Chunked(std::slice::ChunksExact<'a, i64>),
    Empty(usize),
}

impl<'a> Iterator for FlatSuccessorIter<'a> {
    type Item = &'a [i64];

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            FlatSuccessorIter::Chunked(chunks) => chunks.next(),
            FlatSuccessorIter::Empty(remaining) => {
                if *remaining > 0 {
                    *remaining -= 1;
                    Some(&[])
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            FlatSuccessorIter::Chunked(chunks) => chunks.size_hint(),
            FlatSuccessorIter::Empty(remaining) => (*remaining, Some(*remaining)),
        }
    }
}

impl<'a> ExactSizeIterator for FlatSuccessorIter<'a> {}

/// Aggregate result of executing a compiled BFS step over a batch of parent states.
///
/// Returned when a whole frontier (many parents) is expanded in one compiled
/// call. The counters are summed across all parents; on an invariant violation
/// `failed_parent_idx` additionally identifies which parent produced the
/// offending successor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfsBatchResult {
    /// All accepted successor states across every processed parent, each a flat
    /// `state_len`-slot i64 vector.
    pub successors: Vec<Vec<i64>>,
    /// Number of parent states actually expanded (may be fewer than supplied if
    /// expansion stopped early on a violation).
    pub parents_processed: usize,
    /// Total successors produced across all parents before dedup.
    pub generated_count: u64,
    /// Number of successors that survived local dedup (the size of `successors`
    /// before any cross-batch dedup the caller applies).
    pub new_count: u64,
    /// `true` if every checked invariant held across the batch; `false` on a
    /// violation.
    pub invariant_ok: bool,
    /// Index (into the supplied parent batch) of the parent whose successor
    /// violated an invariant, or `None` if all passed.
    pub failed_parent_idx: Option<usize>,
    /// Index into the spec's invariant list of the first failing invariant, or
    /// `None` if all passed.
    pub failed_invariant_idx: Option<u32>,
    /// Index of the violating successor within its parent's expansion, or
    /// `None` if all passed.
    pub failed_successor_idx: Option<u32>,
    /// The flat state vector that violated the invariant, captured for trace
    /// reconstruction, or `None` if all passed.
    pub failed_successor: Option<Vec<i64>>,
}

/// Execution errors from a compiled BFS step.
///
/// Distinguishes faults the caller may recover from (by re-expanding the
/// offending state with the interpreter) from buffer-capacity faults that
/// require a larger arena and one fatal class.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BfsStepError {
    /// The compiled step hit a recoverable runtime fault for one parent (e.g.
    /// a compound-value opcode the native path could not evaluate). The caller
    /// should fall back to the interpreter for the affected state.
    #[error("compiled BFS step runtime error")]
    RuntimeError,
    /// The compiled step hit an unrecoverable runtime fault: the result cannot
    /// be trusted and the caller must abort the compiled path rather than fall
    /// back per state.
    #[error("compiled BFS step fatal runtime error")]
    FatalRuntimeError,
    /// The successor output buffer filled before the step finished.
    /// `partial_count` successors were written before the overflow; the caller
    /// must retry with a larger arena.
    #[error("compiled BFS step successor buffer overflow after {partial_count} successors")]
    BufferOverflow {
        /// Number of successors written into the buffer before it overflowed.
        partial_count: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_bfs_step_output_empty_state_len() {
        let out = FlatBfsStepOutput::from_parts(vec![], 0, 3, 3, true, None, None);
        assert_eq!(out.successor_count(), 3);
        assert_eq!(out.state_len(), 0);
        let slices: Vec<&[i64]> = out.iter_successors().collect();
        assert_eq!(slices.len(), 3);
        assert!(slices.iter().all(|s| s.is_empty()));
    }

    #[test]
    fn flat_bfs_step_output_chunked() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let out = FlatBfsStepOutput::from_parts(buf, 2, 3, 3, true, None, None);
        let slices: Vec<&[i64]> = out.iter_successors().collect();
        assert_eq!(slices, vec![&[1, 2][..], &[3, 4][..], &[5, 6][..]]);
    }

    #[test]
    fn flat_bfs_step_output_ref_chunked() {
        let buf = [10, 20, 30, 40];
        let out = FlatBfsStepOutputRef::from_parts(&buf, 2, 2, 2, true, None, None);
        let slices: Vec<&[i64]> = out.iter_successors().collect();
        assert_eq!(slices, vec![&[10, 20][..], &[30, 40][..]]);
    }

    #[test]
    #[should_panic(expected = "successor buffer shorter than successor_count * state_len")]
    fn flat_bfs_step_output_rejects_short_buffer() {
        let _ = FlatBfsStepOutput::from_parts(vec![1, 2, 3], 2, 2, 2, true, None, None);
    }

    #[test]
    #[should_panic(expected = "generated_count must be at least successor_count")]
    fn flat_bfs_step_output_rejects_generated_count_below_successor_count() {
        let _ = FlatBfsStepOutput::from_parts(vec![1, 2, 3, 4], 2, 2, 1, true, None, None);
    }

    #[test]
    fn bfs_step_error_display() {
        let e = BfsStepError::BufferOverflow { partial_count: 7 };
        assert!(format!("{e}").contains("7"));
    }
}
