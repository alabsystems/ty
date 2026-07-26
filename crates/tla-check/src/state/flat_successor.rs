// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Flat diff-based successor representation for `FlatState`.
//!
//! `FlatSuccessor` stores only the changed flat-buffer slots relative to a
//! parent `FlatState`, along with the successor's 128-bit flat fingerprint.
//! This avoids allocating or hashing an entire successor buffer when only a
//! handful of slots change.

use std::sync::Arc;
#[cfg(test)]
use tla_value::Rp;

use smallvec::SmallVec;

use super::flat_fingerprint::{FlatFingerprintStrategy, FlatFingerprinter};
use super::flat_state::FlatState;
use super::state_layout::StateLayout;

/// A successor encoded as flat-slot changes against a parent `FlatState`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlatSuccessor {
    /// 128-bit flat fingerprint of the successor buffer.
    pub(crate) fingerprint: u128,
    /// Changed flat slots as `(slot_index, old_value, new_value)`.
    pub(crate) changes: SmallVec<[(usize, i64, i64); 8]>,
}

impl FlatSuccessor {
    /// Compute a flat-slot diff between a parent and successor `FlatState`.
    ///
    /// Only changed slots are stored. The returned fingerprint is computed via
    /// the incremental `FlatFingerprinter::diff` path and therefore lives in
    /// the same 128-bit fingerprint space as a direct full fingerprint.
    #[must_use]
    #[inline]
    pub(crate) fn from_flat_diff(
        parent: &FlatState,
        successor: &FlatState,
        fingerprinter: &FlatFingerprinter,
    ) -> Self {
        assert_eq!(
            parent.buffer().len(),
            successor.buffer().len(),
            "FlatSuccessor::from_flat_diff requires matching flat buffer lengths"
        );
        assert_eq!(
            fingerprinter.num_slots(),
            parent.buffer().len(),
            "FlatSuccessor::from_flat_diff requires fingerprinter slot count to match the layout"
        );

        let mut changes = SmallVec::<[(usize, i64, i64); 8]>::new();
        for (slot_index, (&old_value, &new_value)) in parent
            .buffer()
            .iter()
            .zip(successor.buffer().iter())
            .enumerate()
        {
            if old_value != new_value {
                changes.push((slot_index, old_value, new_value));
            }
        }

        let parent_fingerprint = fingerprinter.fingerprint(parent.buffer());
        let fingerprint = fingerprinter.diff(parent.buffer(), parent_fingerprint, &changes);

        debug_assert_eq!(
            fingerprint,
            fingerprinter.fingerprint(successor.buffer()),
            "FlatSuccessor::from_flat_diff produced an incremental fingerprint that does not \
             match direct fingerprinting"
        );

        Self {
            fingerprint,
            changes,
        }
    }

    /// Compute a flat-slot diff using a [`FlatFingerprintStrategy`] backend.
    ///
    /// Like [`from_flat_diff`](Self::from_flat_diff) but dispatches through the
    /// strategy enum, allowing either XOR-accumulator or xxh3-128 backends.
    /// The `scratch` buffer is reused across calls to avoid allocation.
    #[must_use]
    #[inline]
    pub(crate) fn from_flat_diff_with_strategy(
        parent: &FlatState,
        successor: &FlatState,
        strategy: &FlatFingerprintStrategy,
        scratch: &mut Vec<i64>,
    ) -> Self {
        assert_eq!(
            parent.buffer().len(),
            successor.buffer().len(),
            "from_flat_diff_with_strategy requires matching flat buffer lengths"
        );
        assert_eq!(
            strategy.num_slots(),
            parent.buffer().len(),
            "from_flat_diff_with_strategy requires strategy slot count to match layout"
        );

        let mut changes = SmallVec::<[(usize, i64, i64); 8]>::new();
        for (slot_index, (&old_value, &new_value)) in parent
            .buffer()
            .iter()
            .zip(successor.buffer().iter())
            .enumerate()
        {
            if old_value != new_value {
                changes.push((slot_index, old_value, new_value));
            }
        }

        let parent_fingerprint = strategy.fingerprint(parent.buffer());
        let fingerprint = strategy.diff(parent.buffer(), parent_fingerprint, &changes, scratch);

        Self {
            fingerprint,
            changes,
        }
    }

    /// Construct a flat successor directly from a fingerprint and slot changes.
    #[must_use]
    #[inline]
    pub(crate) fn from_changes<I>(fingerprint: u128, changes: I) -> Self
    where
        I: IntoIterator<Item = (usize, i64, i64)>,
    {
        let changes: SmallVec<[(usize, i64, i64); 8]> = changes.into_iter().collect();
        Self {
            fingerprint,
            changes,
        }
    }

    /// Materialize the successor state by applying this diff to `parent`.
    #[must_use]
    pub(crate) fn apply_to(&self, parent: &FlatState) -> FlatState {
        let mut buffer = parent.buffer().to_vec();
        for &(slot_index, _old_value, new_value) in &self.changes {
            buffer[slot_index] = new_value;
        }
        FlatState::from_buffer(buffer.into_boxed_slice(), Arc::clone(parent.layout_arc()))
    }

    /// Apply this successor diff into a caller-provided output buffer.
    ///
    /// `output` must have the same length as `parent_buffer`.
    #[inline]
    pub(crate) fn apply_to_buffer(&self, parent_buffer: &[i64], output: &mut [i64]) {
        assert_eq!(
            parent_buffer.len(),
            output.len(),
            "FlatSuccessor::apply_to_buffer requires equal parent/output lengths"
        );

        output.copy_from_slice(parent_buffer);
        for &(slot_index, old_value, new_value) in &self.changes {
            debug_assert_eq!(
                parent_buffer[slot_index], old_value,
                "FlatSuccessor::apply_to_buffer was applied to a different parent buffer"
            );
            output[slot_index] = new_value;
        }
    }

    /// Number of changed flat slots stored in this successor.
    #[must_use]
    #[inline]
    pub(crate) fn num_changes(&self) -> usize {
        self.changes.len()
    }
}

/// Compute the flat buffer size in bytes for a given layout.
///
/// This is the size of the i64 buffer only (the hot-path per-state cost).
/// Does not include the `StateLayout` overhead (shared across all states)
/// or `Box`/`Arc` metadata.
///
/// Part of #3986: acceptance criterion "bytes per state < 200".
#[must_use]
pub(crate) fn flat_state_bytes(layout: &StateLayout) -> usize {
    layout.total_slots() * std::mem::size_of::<i64>()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tla_value::IntIntervalFunc;

    use super::*;
    use crate::state::flat_fingerprint::FlatFingerprintStrategy;
    use crate::state::state_layout::VarLayoutKind;
    use crate::state::ArrayState;
    use crate::var_index::VarRegistry;
    use crate::Value;

    fn scalar_layout(registry: &VarRegistry) -> Arc<StateLayout> {
        Arc::new(StateLayout::new(
            registry,
            vec![
                VarLayoutKind::Scalar,
                VarLayoutKind::Scalar,
                VarLayoutKind::Scalar,
            ],
        ))
    }

    fn mixed_layout(registry: &VarRegistry) -> Arc<StateLayout> {
        Arc::new(StateLayout::new(
            registry,
            vec![
                VarLayoutKind::Scalar,
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 1,
                    len: 3,
                    elements_are_bool: false,
                    element_types: None,
                },
                VarLayoutKind::ScalarBool,
                VarLayoutKind::Dynamic,
            ],
        ))
    }

    fn int_func(lo: i64, values: [i64; 3]) -> Value {
        Value::IntFunc(Rp::new(IntIntervalFunc::new(
            lo,
            lo + values.len() as i64 - 1,
            values.into_iter().map(Value::SmallInt).collect(),
        )))
    }

    fn flat_state_from_values(values: Vec<Value>, layout: Arc<StateLayout>) -> FlatState {
        FlatState::from_array_state(&ArrayState::from_values(values), layout)
    }

    fn mixed_parent_values() -> Vec<Value> {
        vec![
            Value::SmallInt(1),
            int_func(1, [10, 20, 30]),
            Value::Bool(false),
            Value::String(Rp::from("parent")),
        ]
    }

    fn mixed_successor_values() -> Vec<Value> {
        vec![
            Value::SmallInt(2),
            int_func(1, [10, 99, 30]),
            Value::Bool(true),
            Value::String(Rp::from("child")),
        ]
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_successor_from_diff_single_change() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let layout = scalar_layout(&registry);
        let parent = flat_state_from_values(
            vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)],
            Arc::clone(&layout),
        );
        let successor = flat_state_from_values(
            vec![Value::SmallInt(1), Value::SmallInt(7), Value::SmallInt(3)],
            layout,
        );
        let fingerprinter = FlatFingerprinter::new(parent.buffer().len());

        let diff = FlatSuccessor::from_flat_diff(&parent, &successor, &fingerprinter);

        assert_eq!(diff.num_changes(), 1);
        assert_eq!(diff.changes.as_slice(), &[(1, 2, 7)]);
        assert_eq!(
            diff.fingerprint,
            fingerprinter.fingerprint(successor.buffer())
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_successor_from_diff_multiple_changes() {
        let registry = VarRegistry::from_names(["pc", "arr", "flag", "blob"]);
        let layout = mixed_layout(&registry);
        let parent = flat_state_from_values(mixed_parent_values(), Arc::clone(&layout));
        let successor = flat_state_from_values(mixed_successor_values(), layout);
        let fingerprinter = FlatFingerprinter::new(parent.buffer().len());

        let diff = FlatSuccessor::from_flat_diff(&parent, &successor, &fingerprinter);

        assert_eq!(diff.num_changes(), 3);
        assert_eq!(
            diff.changes.as_slice(),
            &[(0, 1, 2), (2, 20, 99), (4, 0, 1)]
        );
        assert_eq!(
            diff.fingerprint,
            fingerprinter.fingerprint(successor.buffer())
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_successor_from_diff_no_changes() {
        let registry = VarRegistry::from_names(["pc", "arr", "flag", "blob"]);
        let layout = mixed_layout(&registry);
        let parent = flat_state_from_values(mixed_parent_values(), Arc::clone(&layout));
        let successor = flat_state_from_values(mixed_parent_values(), layout);
        let fingerprinter = FlatFingerprinter::new(parent.buffer().len());

        let diff = FlatSuccessor::from_flat_diff(&parent, &successor, &fingerprinter);

        assert_eq!(diff.num_changes(), 0);
        assert!(diff.changes.is_empty());
        assert_eq!(diff.fingerprint, fingerprinter.fingerprint(parent.buffer()));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_successor_apply_to_roundtrip() {
        let registry = VarRegistry::from_names(["pc", "arr", "flag", "blob"]);
        let layout = mixed_layout(&registry);
        let parent = flat_state_from_values(mixed_parent_values(), Arc::clone(&layout));
        let successor = flat_state_from_values(mixed_successor_values(), layout);
        let fingerprinter = FlatFingerprinter::new(parent.buffer().len());
        let diff = FlatSuccessor::from_flat_diff(&parent, &successor, &fingerprinter);

        let materialized = diff.apply_to(&parent);

        assert_eq!(materialized.buffer(), successor.buffer());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_successor_apply_to_buffer() {
        let registry = VarRegistry::from_names(["pc", "arr", "flag", "blob"]);
        let layout = mixed_layout(&registry);
        let parent = flat_state_from_values(mixed_parent_values(), Arc::clone(&layout));
        let successor = flat_state_from_values(mixed_successor_values(), layout);
        let fingerprinter = FlatFingerprinter::new(parent.buffer().len());
        let diff = FlatSuccessor::from_flat_diff(&parent, &successor, &fingerprinter);
        let mut output = vec![0i64; parent.buffer().len()];

        diff.apply_to_buffer(parent.buffer(), &mut output);

        assert_eq!(output.as_slice(), successor.buffer());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_successor_fingerprint_matches_direct() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let layout = scalar_layout(&registry);
        let parent = flat_state_from_values(
            vec![Value::SmallInt(4), Value::SmallInt(5), Value::SmallInt(6)],
            layout,
        );
        let fingerprinter = FlatFingerprinter::new(parent.buffer().len());
        let base_fingerprint = fingerprinter.fingerprint(parent.buffer());
        let changes: SmallVec<[(usize, i64, i64); 8]> =
            SmallVec::from_vec(vec![(0, 4, 40), (2, 6, 60)]);
        let fingerprint = fingerprinter.diff(parent.buffer(), base_fingerprint, &changes);
        let successor = FlatSuccessor::from_changes(fingerprint, changes.clone());

        let materialized = successor.apply_to(&parent);

        assert_eq!(successor.num_changes(), 2);
        assert_eq!(
            successor.fingerprint,
            fingerprinter.fingerprint(materialized.buffer())
        );
        assert_eq!(materialized.buffer(), &[40, 5, 60]);
    }

    // ====================================================================
    // FlatFingerprintStrategy-based diff tests (Part of #3987)
    // ====================================================================

    #[test]
    fn test_flat_successor_from_diff_with_strategy_xor() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let layout = scalar_layout(&registry);
        let parent = flat_state_from_values(
            vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)],
            Arc::clone(&layout),
        );
        let successor = flat_state_from_values(
            vec![Value::SmallInt(1), Value::SmallInt(7), Value::SmallInt(3)],
            layout,
        );
        let strategy = FlatFingerprintStrategy::new_xor(parent.buffer().len());
        let mut scratch = Vec::new();

        let diff = FlatSuccessor::from_flat_diff_with_strategy(
            &parent,
            &successor,
            &strategy,
            &mut scratch,
        );

        assert_eq!(diff.num_changes(), 1);
        assert_eq!(diff.changes.as_slice(), &[(1, 2, 7)]);
        assert_eq!(diff.fingerprint, strategy.fingerprint(successor.buffer()));
    }

    #[test]
    fn test_flat_successor_from_diff_with_strategy_xxh3() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let layout = scalar_layout(&registry);
        let parent = flat_state_from_values(
            vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)],
            Arc::clone(&layout),
        );
        let successor = flat_state_from_values(
            vec![Value::SmallInt(1), Value::SmallInt(7), Value::SmallInt(3)],
            layout,
        );
        let strategy = FlatFingerprintStrategy::new_xxh3(parent.buffer().len());
        let mut scratch = Vec::new();

        let diff = FlatSuccessor::from_flat_diff_with_strategy(
            &parent,
            &successor,
            &strategy,
            &mut scratch,
        );

        assert_eq!(diff.num_changes(), 1);
        assert_eq!(diff.changes.as_slice(), &[(1, 2, 7)]);
        assert_eq!(diff.fingerprint, strategy.fingerprint(successor.buffer()));
    }

    #[test]
    fn test_flat_successor_from_diff_with_strategy_mixed_layout() {
        let registry = VarRegistry::from_names(["pc", "arr", "flag", "blob"]);
        let layout = mixed_layout(&registry);
        let parent = flat_state_from_values(mixed_parent_values(), Arc::clone(&layout));
        let successor = flat_state_from_values(mixed_successor_values(), layout);

        for strategy in [
            FlatFingerprintStrategy::new_xor(parent.buffer().len()),
            FlatFingerprintStrategy::new_xxh3(parent.buffer().len()),
        ] {
            let mut scratch = Vec::new();
            let diff = FlatSuccessor::from_flat_diff_with_strategy(
                &parent,
                &successor,
                &strategy,
                &mut scratch,
            );
            assert_eq!(diff.num_changes(), 3);
            assert_eq!(diff.fingerprint, strategy.fingerprint(successor.buffer()));
        }
    }
}
