// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compact canonical payload witnesses for collision-checked admission.
//!
//! These witnesses are intentionally frontend-neutral: they describe the typed
//! canonical payload slots used to authorize a duplicate fingerprint, not a
//! TLA-specific `ArrayState` owner. TLA's interpreter can use compact value
//! slots, while compiled, Petri/MCC, and hardware/register-vector lanes can use
//! flat `i64` slots.

use tla_value::CompactValue;

use super::ArrayState;

/// Typed storage form carried by a [`StatePayloadWitness`].
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatePayloadWitnessKind {
    /// Exact compact TLA value slots in canonical variable order.
    CompactValueSlots,
    /// Exact flat/register `i64` slots in canonical layout order.
    FlatI64Slots,
}

/// Compact witness used to authorize duplicate fingerprint admission.
///
/// This deliberately excludes `ArrayState`'s fingerprint cache and other
/// evaluator-facing state. The witness only keeps the canonical payload needed
/// to decide whether a duplicate fingerprint is the same payload in the same
/// typed domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StatePayloadWitness {
    CompactValueSlots(Box<[CompactValue]>),
    FlatI64Slots(Box<[i64]>),
}

impl StatePayloadWitness {
    /// Build an exact witness from an interpreter `ArrayState`.
    #[must_use]
    pub(crate) fn from_array_state(state: &ArrayState) -> Self {
        Self::CompactValueSlots(state.values().to_vec().into_boxed_slice())
    }

    /// Build an exact witness from compact value slots.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn from_compact_values(values: &[CompactValue]) -> Self {
        Self::CompactValueSlots(values.to_vec().into_boxed_slice())
    }

    /// Build an exact witness from flat/register slots.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn from_flat_i64_slots(slots: &[i64]) -> Self {
        Self::FlatI64Slots(slots.to_vec().into_boxed_slice())
    }

    /// Return the typed witness representation.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) const fn kind(&self) -> StatePayloadWitnessKind {
        match self {
            Self::CompactValueSlots(_) => StatePayloadWitnessKind::CompactValueSlots,
            Self::FlatI64Slots(_) => StatePayloadWitnessKind::FlatI64Slots,
        }
    }

    /// Confirm an `ArrayState` candidate against this witness.
    #[must_use]
    pub(crate) fn matches_array_state(&self, candidate: &ArrayState) -> bool {
        match self {
            Self::CompactValueSlots(values) => values.as_ref() == candidate.values(),
            Self::FlatI64Slots(_) => false,
        }
    }

    /// Confirm a flat/register candidate against this witness.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn matches_flat_i64_slots(&self, candidate: &[i64]) -> bool {
        match self {
            Self::CompactValueSlots(_) => false,
            Self::FlatI64Slots(slots) => slots.as_ref() == candidate,
        }
    }

    /// Payload bytes retained by the witness, excluding map/enum overhead.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn payload_bytes(&self) -> usize {
        match self {
            Self::CompactValueSlots(values) => values.len() * std::mem::size_of::<CompactValue>(),
            Self::FlatI64Slots(slots) => slots.len() * std::mem::size_of::<i64>(),
        }
    }
}
