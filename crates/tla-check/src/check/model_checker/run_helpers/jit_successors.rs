// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! JIT flat-successor value types.
//!
//! These types carry JIT-compiled successors as flat i64 buffers so the BFS
//! loop can defer the expensive unflatten/Value materialization until after the
//! dedup check. The `impl ModelChecker` methods that produce them live in the
//! parent module (`run_helpers`).

use crate::state::{ArrayState, Fingerprint};

/// A JIT-compiled successor stored as flat i64 buffers.
///
/// Defers the expensive `unflatten_i64_to_array_state_with_input` conversion
/// until after the BFS dedup check. Most successors are duplicates — by keeping
/// them flat, we avoid allocating Value objects and ArrayState for states that
/// will be immediately discarded.
///
/// Part of #4032: Eliminate per-action unflatten.
pub(in crate::check) struct JitFlatSuccessor {
    /// The raw i64 output from the JIT-compiled action function.
    /// Each slot corresponds to a state variable (same layout as flatten).
    pub(in crate::check) jit_output: Vec<i64>,
    /// Snapshot of the JIT input buffer at the time this action was evaluated.
    /// Needed by `unflatten_i64_to_array_state_with_input` to deserialize
    /// compound values that were modified in-place by native FuncExcept.
    pub(in crate::check) jit_input: Vec<i64>,
    /// Number of state variables (may be less than buffer length for compact layouts).
    pub(in crate::check) state_var_count: usize,
}

/// A flat successor that survived the read-only seen prefilter.
///
/// The hot flat-primary path computes the fingerprint directly from the raw
/// JIT output buffer, checks the seen set, and only then boxes the buffer into
/// a `FlatState`. Storage admission remains authoritative later in the BFS
/// pipeline.
pub(in crate::check) struct FlatPrefilteredSuccessor {
    pub(in crate::check) flat: crate::state::FlatState,
    pub(in crate::check) fingerprint: Fingerprint,
}

pub(in crate::check) struct FlatPrefilteredActionSuccessors {
    pub(in crate::check) successors: Vec<FlatPrefilteredSuccessor>,
    pub(in crate::check) raw_successor_count: usize,
}

pub(in crate::check) struct FlatPrefilteredSuccessorResult {
    pub(in crate::check) successors: Vec<FlatPrefilteredSuccessor>,
    pub(in crate::check) raw_successor_count: usize,
    pub(in crate::check) had_raw_successors: bool,
}

impl JitFlatSuccessor {
    /// Convert this flat successor to an ArrayState by unflattening.
    ///
    /// This is the cold-path materialization: only called for NEW states that
    /// pass the dedup check and need invariant checking + queue insertion.
    #[inline]
    pub(in crate::check) fn to_array_state(&self, parent: &ArrayState) -> ArrayState {
        super::super::invariants::unflatten_i64_to_array_state_with_input(
            parent,
            &self.jit_output,
            self.state_var_count,
            Some(&self.jit_input),
        )
    }

    /// Try to compute a fingerprint directly from the flat buffer.
    ///
    /// Returns `Some(Fingerprint)` for all-scalar states where the fingerprint
    /// can be computed without materializing Value objects. Returns `None` when
    /// compound variables were modified, requiring full unflatten.
    ///
    /// Part of #4032: Hot-path fingerprinting without ArrayState allocation.
    #[inline]
    pub(in crate::check) fn try_flat_fingerprint(
        &self,
        parent: &ArrayState,
        registry: &crate::var_index::VarRegistry,
    ) -> Option<Fingerprint> {
        super::super::invariants::fingerprint_jit_flat_successor(
            parent,
            &self.jit_output,
            self.state_var_count,
            Some(&self.jit_input),
            registry,
        )
        .map(|(fp, _xor)| fp)
    }

    /// Compute fingerprint using compiled xxh3 SIMD on the raw i64 output buffer.
    ///
    /// This is the fast path when `jit_compiled_fp_active` is true: a single
    /// xxh3-64 call on the raw byte representation of the successor state.
    /// Only valid when ALL variables are scalar (Int/Bool).
    ///
    /// Part of #3987: Compiled xxh3 fingerprinting for the BFS hot path.
    #[inline]
    pub(in crate::check) fn compiled_xxh3_fingerprint(&self) -> Fingerprint {
        super::super::invariants::fingerprint_flat_compiled(
            &self.jit_output[..self.state_var_count],
        )
    }
}
