// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! State layout for flat i64 state representation.
//!
//! Maps each state variable to a contiguous region of slots in a flat `[i64]`
//! buffer. Unlike the native ABI `compound_layout::StateLayout` (which uses
//! self-describing type tags), this layout is a fixed-offset
//! scheme for the model checker's state storage and comparison.
//!
//! # Slot mapping
//!
//! ```text
//! Variable 0 (Scalar):       [slot 0]
//! Variable 1 (IntArray(3)):  [slot 1, slot 2, slot 3]
//! Variable 2 (Bitmask):      [slot 4]
//! Variable 3 (Dynamic):      [slot 5]  (pointer/index to side table)
//! ```
//!
//! Part of #3986.

use std::sync::Arc;
#[cfg(test)]
use tla_value::Rp;

use crate::var_index::VarRegistry;

/// Describes how a single state variable maps onto i64 slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VarLayout {
    /// Human-readable variable name (for debugging).
    pub(crate) name: Arc<str>,
    /// Starting offset in the flat i64 buffer.
    pub(crate) offset: usize,
    /// Number of i64 slots this variable occupies.
    pub(crate) slot_count: usize,
    /// What kind of mapping is used.
    pub(crate) kind: VarLayoutKind,
}

/// Per-element type tag for flat state encoding.
///
/// Tracks whether each slot in an IntArray or Record field is an integer,
/// boolean, or interned string/model-value. This enables correct roundtrip
/// reconstruction from i64 slots.
///
/// Part of #3908: compound type flat state roundtrip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotType {
    /// Integer value — stored as raw i64.
    Int,
    /// Boolean value — stored as 0/1.
    Bool,
    /// Interned string — stored as NameId (u32 as i64).
    String,
    /// Model value — stored as NameId (u32 as i64), reconstructed as ModelValue.
    ModelValue,
}

/// Scalar value stored as fixed layout metadata.
///
/// Recursive aggregate layouts store function domains and set universes as
/// metadata so the flat buffer only needs to contain range values/bitmasks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FlatScalarValue {
    /// Integer key or set element.
    Int(i64),
    /// Boolean key or set element.
    Bool(bool),
    /// String key or set element.
    String(Arc<str>),
    /// Model value key or set element.
    ModelValue(Arc<str>),
}

/// Decoded value for a compact scalar-or-set flat slot.
///
/// This is a representation contract for Dijkstra-style `temp[self]` slots
/// whose value can be either an interned scalar sentinel or a finite set over a
/// fixed universe. Scalars remain nonnegative raw ids; set masks use a negative
/// sign tag via `-1 - mask`, making the one-slot encoding injective without
/// relying on accidental disjointness between intern ids and bitmasks.
///
/// This is intentionally not admitted into native-fused flat BFS by itself.
/// Callers still need a layout/proof token showing that every producer and
/// consumer uses this encoding for the same finite universe.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaggedScalarSetSlot {
    Scalar(i64),
    SetMask(i64),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaggedScalarSetSlotError {
    NegativeScalar(i64),
    InvalidUniverseLen(usize),
    DuplicateUniverseValue,
    NonCanonicalSetMask { mask: i64, universe_len: usize },
    NonCanonicalTaggedSet { raw: i64, universe_len: usize },
}

/// Error for a typed finite scalar-union slot.
///
/// A scalar-union slot is intentionally domain-indexed, not raw-value encoded:
/// raw slot `i` means `universe[i]`. That keeps `Int(1)`, `String(name_id=1)`,
/// and `ModelValue(name_id=1)` distinct even though all three legacy scalar
/// lanes can otherwise collapse to the same compact `i64` payload.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TaggedScalarUnionSlotError {
    InvalidUniverseLen(usize),
    DuplicateUniverseValue,
    ValueOutsideUniverse,
    NonCanonicalTaggedUnion { raw: i64, universe_len: usize },
}

/// Proof metadata for a one-slot finite scalar union such as `{None} \cup Procs`.
///
/// The compact slot stores the typed universe index, not the untagged scalar
/// payload. This is the minimal injective representation for mixed scalar lanes.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarUnionProof {
    universe: Vec<FlatScalarValue>,
    source: Arc<str>,
}

impl TaggedScalarUnionProof {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(
        universe: Vec<FlatScalarValue>,
        source: Arc<str>,
    ) -> Result<Self, TaggedScalarUnionSlotError> {
        validate_tagged_scalar_union_universe(&universe)?;
        Ok(Self { universe, source })
    }

    #[must_use]
    pub(crate) fn universe(&self) -> &[FlatScalarValue] {
        &self.universe
    }

    #[must_use]
    pub(crate) fn source(&self) -> &Arc<str> {
        &self.source
    }
}

/// Proof metadata for a one-slot `scalar | finite-set` function range.
///
/// This describes the Dijkstra `temp[self]` shape: the function is keyed by a
/// finite model-value domain, and each range slot is either a scalar sentinel or
/// a finite set over the declared universe. The encoding is injective only when
/// every writer uses [`encode_tagged_scalar_set_scalar`] for scalar values and
/// [`encode_tagged_scalar_set_mask`] for set values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarSetRangeProof {
    scalar_type: SlotType,
    set_universe: Vec<FlatScalarValue>,
    source: Arc<str>,
}

impl TaggedScalarSetRangeProof {
    pub(crate) fn new(
        scalar_type: SlotType,
        set_universe: Vec<FlatScalarValue>,
        source: Arc<str>,
    ) -> Result<Self, TaggedScalarSetSlotError> {
        valid_tagged_set_mask(set_universe.len()).ok_or(
            TaggedScalarSetSlotError::InvalidUniverseLen(set_universe.len()),
        )?;
        for (index, value) in set_universe.iter().enumerate() {
            if set_universe[index + 1..]
                .iter()
                .any(|candidate| candidate == value)
            {
                return Err(TaggedScalarSetSlotError::DuplicateUniverseValue);
            }
        }
        Ok(Self {
            scalar_type,
            set_universe,
            source,
        })
    }

    #[must_use]
    pub(crate) fn scalar_type(&self) -> SlotType {
        self.scalar_type
    }

    #[must_use]
    pub(crate) fn set_universe(&self) -> &[FlatScalarValue] {
        &self.set_universe
    }

    #[must_use]
    pub(crate) fn source(&self) -> &Arc<str> {
        &self.source
    }
}

/// Proof metadata for a scalar-only fixed finite-function range.
///
/// This is intentionally weaker than a primary-storage proof. It records that a
/// model-value-keyed function range is a bounded scalar label set, which is
/// enough for strict native-fused flat-frontier admission once the backend also
/// proves invariant coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FixedScalarRangeProof {
    scalar_type: SlotType,
    scalar_universe: Vec<FlatScalarValue>,
    source: Arc<str>,
}

impl FixedScalarRangeProof {
    pub(crate) fn new(
        scalar_type: SlotType,
        scalar_universe: Vec<FlatScalarValue>,
        source: Arc<str>,
    ) -> Result<Self, TaggedScalarSetSlotError> {
        if scalar_universe.is_empty() {
            return Err(TaggedScalarSetSlotError::InvalidUniverseLen(0));
        }
        for (index, value) in scalar_universe.iter().enumerate() {
            if value.slot_type() != scalar_type {
                return Err(TaggedScalarSetSlotError::NonCanonicalSetMask {
                    mask: index as i64,
                    universe_len: scalar_universe.len(),
                });
            }
            if scalar_universe[index + 1..]
                .iter()
                .any(|candidate| candidate == value)
            {
                return Err(TaggedScalarSetSlotError::DuplicateUniverseValue);
            }
        }
        Ok(Self {
            scalar_type,
            scalar_universe,
            source,
        })
    }

    #[must_use]
    pub(crate) fn scalar_type(&self) -> SlotType {
        self.scalar_type
    }

    #[must_use]
    pub(crate) fn scalar_universe(&self) -> &[FlatScalarValue] {
        &self.scalar_universe
    }

    #[must_use]
    pub(crate) fn source(&self) -> &Arc<str> {
        &self.source
    }
}

fn tagged_scalar_set_scalar_type_supports_flat_primary(slot_type: SlotType) -> bool {
    // The tagged set branch uses negative payloads (`-1 - mask`). Scalar slots
    // are therefore primary-safe only when their flat encoding is never
    // negative. Interned values and bools satisfy that; arbitrary Int values do
    // not carry a non-negative proof here.
    matches!(
        slot_type,
        SlotType::Bool | SlotType::String | SlotType::ModelValue
    )
}

/// Range-slot encoding for a tuple/cross-product-keyed function.
///
/// The tuple analogue of the union arm of [`StringKeyedArrayRangeEncoding`].
/// A `TupleKeyedArray` maps each canonical tuple key to one contiguous i64 slot;
/// this enum says how the RANGE value at that slot is encoded. `ScalarSlots`
/// (default) is the historical plain raw-i64 encoding. `TaggedScalarUnion`
/// stores the injective universe index of a heterogeneous finite scalar union
/// range `[D1 x D2 -> s1 \cup s2]` (e.g. btree `childOf \in [Nodes x Keys ->
/// Nodes \cup {NIL}]`), so mixed int/model-value ranges round-trip without the
/// unsound raw int==model-value collision. Only ever built under
/// `TY_TAGGED_SCALAR_UNION` from a checked whole-variable type invariant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum TupleKeyedArrayRangeEncoding {
    /// Legacy plain raw-i64 scalar slots. `value_types` reconstruct each slot.
    #[default]
    ScalarSlots,
    /// One-slot injective universe index of a heterogeneous finite scalar union
    /// range. The slot stores `encode_tagged_scalar_union_value(cell, universe)`.
    TaggedScalarUnion(TaggedScalarUnionProof),
    /// A HOMOGENEOUS proven-finite model-value/string/bool range. The tuple
    /// analogue of the `FixedScalar` arm of [`StringKeyedArrayRangeEncoding`].
    /// The slot stores the raw interned-`NameId` (identical to `ScalarSlots`);
    /// the proof adds nothing to the encoding — it certifies that every reachable
    /// cell is drawn from a closed finite homogeneous non-int universe (proven by
    /// a checked whole-var `TypeOK` invariant), so the `NameId` slot is injective
    /// and can never alias a plain integer, making it flat-primary safe. Resolves
    /// btree `valOf \in [Nodes \X Keys -> Vals \cup {NIL}]` where `Vals \cup {NIL}`
    /// is the homogeneous model-value set `{x,y,z,nil}` (NOT a heterogeneous
    /// union, so `TaggedScalarUnion` correctly does not apply).
    FixedScalar(FixedScalarRangeProof),
}

/// Range-slot encoding for legacy fixed string/model-value keyed functions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum StringKeyedArrayRangeEncoding {
    /// Legacy scalar-only slots. `value_types` reconstruct each slot.
    #[default]
    ScalarSlots,
    /// Scalar-only slots backed by a fixed finite scalar range proof.
    FixedScalar(FixedScalarRangeProof),
    /// One-slot tagged `scalar | finite-set` range proof.
    TaggedScalarOrSet(TaggedScalarSetRangeProof),
}

#[allow(dead_code)]
fn valid_tagged_set_mask(universe_len: usize) -> Option<i64> {
    match universe_len {
        0 => Some(0),
        1..=62 => Some((1_i64 << universe_len) - 1),
        63 => Some(i64::MAX),
        _ => None,
    }
}

#[allow(dead_code)]
pub(crate) fn encode_tagged_scalar_set_scalar(
    scalar: i64,
) -> Result<i64, TaggedScalarSetSlotError> {
    if scalar < 0 {
        return Err(TaggedScalarSetSlotError::NegativeScalar(scalar));
    }
    Ok(scalar)
}

#[allow(dead_code)]
pub(crate) fn encode_tagged_scalar_set_mask(
    mask: i64,
    universe_len: usize,
) -> Result<i64, TaggedScalarSetSlotError> {
    let valid_mask = valid_tagged_set_mask(universe_len)
        .ok_or(TaggedScalarSetSlotError::InvalidUniverseLen(universe_len))?;
    if mask < 0 || (mask & !valid_mask) != 0 {
        return Err(TaggedScalarSetSlotError::NonCanonicalSetMask { mask, universe_len });
    }
    Ok(-1 - mask)
}

#[allow(dead_code)]
pub(crate) fn decode_tagged_scalar_set_slot(
    raw: i64,
    universe_len: usize,
) -> Result<TaggedScalarSetSlot, TaggedScalarSetSlotError> {
    let valid_mask = valid_tagged_set_mask(universe_len)
        .ok_or(TaggedScalarSetSlotError::InvalidUniverseLen(universe_len))?;
    if raw >= 0 {
        return Ok(TaggedScalarSetSlot::Scalar(raw));
    }

    let mask = raw
        .checked_add(1)
        .and_then(i64::checked_neg)
        .ok_or(TaggedScalarSetSlotError::NonCanonicalTaggedSet { raw, universe_len })?;
    if (mask & !valid_mask) != 0 {
        return Err(TaggedScalarSetSlotError::NonCanonicalTaggedSet { raw, universe_len });
    }
    Ok(TaggedScalarSetSlot::SetMask(mask))
}

#[allow(dead_code)]
fn validate_tagged_scalar_union_universe(
    universe: &[FlatScalarValue],
) -> Result<(), TaggedScalarUnionSlotError> {
    if universe.is_empty() || universe.len() > i64::MAX as usize {
        return Err(TaggedScalarUnionSlotError::InvalidUniverseLen(
            universe.len(),
        ));
    }
    for (index, value) in universe.iter().enumerate() {
        if universe[index + 1..]
            .iter()
            .any(|candidate| candidate == value)
        {
            return Err(TaggedScalarUnionSlotError::DuplicateUniverseValue);
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn encode_tagged_scalar_union_value(
    value: &FlatScalarValue,
    universe: &[FlatScalarValue],
) -> Result<i64, TaggedScalarUnionSlotError> {
    validate_tagged_scalar_union_universe(universe)?;
    let index = universe
        .iter()
        .position(|candidate| candidate == value)
        .ok_or(TaggedScalarUnionSlotError::ValueOutsideUniverse)?;
    i64::try_from(index).map_err(|_| TaggedScalarUnionSlotError::InvalidUniverseLen(universe.len()))
}

#[allow(dead_code)]
pub(crate) fn decode_tagged_scalar_union_slot(
    raw: i64,
    universe: &[FlatScalarValue],
) -> Result<FlatScalarValue, TaggedScalarUnionSlotError> {
    validate_tagged_scalar_union_universe(universe)?;
    if raw < 0 {
        return Err(TaggedScalarUnionSlotError::NonCanonicalTaggedUnion {
            raw,
            universe_len: universe.len(),
        });
    }
    let index =
        usize::try_from(raw).map_err(|_| TaggedScalarUnionSlotError::NonCanonicalTaggedUnion {
            raw,
            universe_len: universe.len(),
        })?;
    universe
        .get(index)
        .cloned()
        .ok_or(TaggedScalarUnionSlotError::NonCanonicalTaggedUnion {
            raw,
            universe_len: universe.len(),
        })
}

/// Error raised while building or validating a [`TaggedUnionProof`].
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TaggedUnionSlotError {
    /// Fewer than two variants — a "union" with a single shape is just that
    /// shape and must not be encoded as a tagged union.
    TooFewVariants(usize),
    /// More than [`i64::MAX`] variants (tag overflow) or otherwise unencodable.
    TooManyVariants(usize),
    /// Two variants share an identical layout. Tags must be injective onto
    /// distinct shapes, otherwise the tag carries no information and two logical
    /// values could map to different tags for the same shape (non-canonical).
    DuplicateVariant,
    /// A variant layout is not finite/closed (e.g. a sampled-bound sequence or
    /// a function range that is not provably closed), so the union universe is
    /// not statically finite and MUST fail closed to the interpreter.
    NonFiniteVariant,
    /// Decoded tag is out of range for the recorded variant list.
    NonCanonicalTag { tag: i64, variant_count: usize },
}

/// Proof metadata for a finite tagged-union state variable.
///
/// A tagged union encodes a state variable whose value ranges over a finite,
/// statically-known set of *heterogeneous* shapes (a TLA+ sum type), e.g.
/// btree/kvstore's `args \in {NIL} \cup {<<k>>: k \in Keys} \cup {<<k,v>>: ...}`.
///
/// # Encoding
///
/// The flat representation is `1 + max_payload_slots` slots:
///   * slot 0 — the **tag**: the index of the active variant in `variants`.
///   * slots `1..=max_payload` — the active variant's own flat encoding,
///     written by [`FlatValueLayout::slot_count`]-many slots and **zero** for
///     every trailing payload slot the active variant does not use.
///
/// # Soundness (fingerprint distinctness)
///
/// The flat fingerprint hashes the raw slots, so two states with the same
/// `args` produce identical slots and two states with *different* `args`
/// produce different slots **provided**:
///   1. distinct logical values map to distinct `(tag, payload)` — guaranteed
///      because each variant is itself a canonical injective flat encoding
///      (every variant's encoder zero-fills then writes; see
///      `try_write_flat_value_slots`), and the tag distinguishes variants of
///      different shape (so `NIL` (tag 0) can never collide with `<<k>>`
///      (tag 1) or `<<k,v>>` (tag 2) regardless of payload bits);
///   2. trailing payload slots a variant does not use are canonically zero —
///      guaranteed because the parent encoder zero-fills the whole slice before
///      writing the active variant into the leading payload slots only.
///
/// # Finiteness (fail-closed)
///
/// Every variant must be finite and closed (`FlatValueLayout::supports_flat_primary`
/// for the variant's own shape, which already rejects sampled-bound sequences
/// and non-proven-closed function ranges). If ANY variant is infinite or merely
/// sampled, construction returns an error and the caller falls back to the
/// interpreter — never an approximation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedUnionProof {
    /// Distinct variant layouts in canonical (tag) order. Tag `i` selects
    /// `variants[i]`.
    variants: Vec<FlatValueLayout>,
    /// Source invariant / writer-relation that justified the finite universe.
    source: Arc<str>,
}

impl TaggedUnionProof {
    /// Build a proof from a list of distinct, finite variant layouts.
    ///
    /// Fails closed (returns `Err`) when there are fewer than two variants, a
    /// duplicate variant, or a variant that is not provably finite/closed.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(
        variants: Vec<FlatValueLayout>,
        source: Arc<str>,
    ) -> Result<Self, TaggedUnionSlotError> {
        if variants.len() < 2 {
            return Err(TaggedUnionSlotError::TooFewVariants(variants.len()));
        }
        if i64::try_from(variants.len()).is_err() {
            return Err(TaggedUnionSlotError::TooManyVariants(variants.len()));
        }
        for (index, variant) in variants.iter().enumerate() {
            // A tagged union of tagged unions, or one whose variant carries an
            // unproven (sampled) capacity, is not a statically-finite universe.
            // `supports_flat_primary` is exactly the per-variant finiteness gate
            // (it rejects sampled sequences and non-proven-closed ranges) and it
            // also already returns false for a nested `TaggedUnion`/`TaggedScalarUnion`.
            if !variant.supports_flat_primary() {
                return Err(TaggedUnionSlotError::NonFiniteVariant);
            }
            if variants[index + 1..]
                .iter()
                .any(|candidate| candidate == variant)
            {
                return Err(TaggedUnionSlotError::DuplicateVariant);
            }
        }
        Ok(Self { variants, source })
    }

    #[must_use]
    pub(crate) fn variants(&self) -> &[FlatValueLayout] {
        &self.variants
    }

    #[must_use]
    pub(crate) fn source(&self) -> &Arc<str> {
        &self.source
    }

    /// Number of payload slots (excludes the tag slot): the widest variant.
    #[must_use]
    pub(crate) fn max_payload_slots(&self) -> usize {
        self.variants
            .iter()
            .map(FlatValueLayout::slot_count)
            .max()
            .unwrap_or(0)
    }

    /// Total slot count: 1 tag slot + the widest variant's payload.
    #[must_use]
    pub(crate) fn slot_count(&self) -> usize {
        1 + self.max_payload_slots()
    }
}

/// Evidence attached to a recursive sequence capacity.
///
/// `max_len` is just storage width. This marker records whether that width was
/// only observed from sampled states or is backed by a global proof/invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SequenceBoundEvidence {
    /// Capacity was inferred from concrete initial/wavefront values.
    Observed,
    /// Capacity and element layout are backed by a fixed finite function-domain
    /// type proof, e.g. `x \in [1..N -> T]`, for a runtime representation that
    /// is stored as a TLA sequence.
    FixedDomainTypeLayout { invariant: Arc<str> },
    /// Capacity is backed by a checked source-level bound.
    ProvenInvariant { invariant: Arc<str> },
    /// Capacity and sequence element layout are both backed by checked
    /// source-level invariants.
    ProvenInvariantWithElementLayout {
        invariant: Arc<str>,
        element_invariant: Arc<str>,
    },
    /// HEURISTIC capacity = element universe cardinality `|U|` from the checked
    /// `v \in Seq(U)` invariant, admitted WITHOUT a proven length bound. This is
    /// NOT a certified upper bound on `Len(v)`; it is a best guess whose
    /// soundness rests ENTIRELY on the fail-closed overflow backstop (a reachable
    /// state whose length exceeds `max_len` fails flat serialization with
    /// `SequenceLengthExceedsCapacity`, so the CLI re-runs WITHOUT flat storage —
    /// the interpreter path is authoritative and never a silent undercount). Only
    /// constructed under the `TY_SEQ_HEURISTIC_CAPACITY` opt-in, and
    /// [`Self::is_proven`] is deliberately `false` so the native lowering bridges
    /// it to `capacity_proven=false` (fail closed, exactly like `Observed`). See
    /// [`seq_heuristic_capacity_enabled`].
    HeuristicUniverseCapacity {
        /// Name of the checked `v \in Seq(U)` invariant the universe was drawn
        /// from (diagnostic + provenance).
        universe_invariant: Arc<str>,
    },
}

impl SequenceBoundEvidence {
    #[must_use]
    pub(crate) fn is_proven(&self) -> bool {
        // HeuristicUniverseCapacity is intentionally EXCLUDED: it is not a
        // certified bound, so it must bridge to `capacity_proven=false` and stay
        // fail-closed in every native capacity-driven path. Only the flat STORAGE
        // path (with its own `SequenceLengthExceedsCapacity` overflow backstop)
        // relies on it.
        matches!(
            self,
            SequenceBoundEvidence::FixedDomainTypeLayout { .. }
                | SequenceBoundEvidence::ProvenInvariant { .. }
                | SequenceBoundEvidence::ProvenInvariantWithElementLayout { .. }
        )
    }

    #[must_use]
    pub(crate) fn supports_flat_primary(&self) -> bool {
        match self {
            SequenceBoundEvidence::FixedDomainTypeLayout { .. }
            | SequenceBoundEvidence::ProvenInvariantWithElementLayout { .. } => true,
            // A heuristic capacity admits flat-primary STORAGE only under the
            // opt-in. Sound because the write path fails closed on overflow
            // (retry-without-flat backstop). Off by default => byte-identical.
            SequenceBoundEvidence::HeuristicUniverseCapacity { .. } => {
                seq_heuristic_capacity_enabled()
            }
            SequenceBoundEvidence::Observed | SequenceBoundEvidence::ProvenInvariant { .. } => {
                false
            }
        }
    }
}

impl FlatScalarValue {
    /// Slot type used when this scalar is encoded as a flat i64 value.
    #[must_use]
    pub(crate) fn slot_type(&self) -> SlotType {
        match self {
            FlatScalarValue::Int(_) => SlotType::Int,
            FlatScalarValue::Bool(_) => SlotType::Bool,
            FlatScalarValue::String(_) => SlotType::String,
            FlatScalarValue::ModelValue(_) => SlotType::ModelValue,
        }
    }
}

fn slot_type_is_plain_i64(slot_type: SlotType) -> bool {
    matches!(slot_type, SlotType::Int | SlotType::Bool)
}

fn var_layout_kind_supports_flat_bfs_auto_admission(kind: &VarLayoutKind) -> bool {
    match kind {
        VarLayoutKind::Scalar | VarLayoutKind::ScalarBool => true,
        // A top-level string/model-value scalar slot only proves the sampled
        // initial value type. The one-word representation overlaps ordinary
        // integer slots, so a later heterogeneous successor can be corrupted by
        // flat-primary/native-flat paths unless a richer aggregate proof exists.
        VarLayoutKind::ScalarString | VarLayoutKind::ScalarModelValue => false,
        // A finite-universe scalar string/model-value enum is safe for default
        // flat BFS once the source-level proof shows the universe is total (G2).
        VarLayoutKind::FixedScalar { .. } => kind.fixed_scalar_var_proof().is_some(),
        VarLayoutKind::IntArray {
            elements_are_bool,
            element_types,
            ..
        } => {
            if *elements_are_bool {
                return true;
            }
            if element_types
                .as_ref()
                .is_some_and(|types| types.iter().all(|ty| slot_type_is_plain_i64(*ty)))
            {
                return true;
            }
            // G2-extension: a finite-universe `TypeOK` proof upgrades non-`i64`
            // (string/model-value) elements into safe default flat-BFS storage.
            kind.int_array_element_range_proof().is_some()
        }
        // G2-extension: string/model-value record fields are admissible for
        // default flat BFS only when a finite-universe `TypeOK` proof covers
        // them; plain `i64` fields stay admissible unconditionally.
        VarLayoutKind::Record { .. } => kind.record_fields_flat_admissible(),
        // String/model-value-keyed scalar functions are only safe for default
        // flat BFS when a source type proof selects the tagged scalar/set
        // encoding, or a validated `FixedScalar(String|ModelValue)` range proof
        // confirms the range is a finite interned scalar universe. Plain-i64
        // ranges are intentionally excluded: a `StringKeyedArray` range is
        // inferred from sampled values (unlike a structurally scalar-only
        // `TupleKeyedArray`), so an Int/Bool-sampled slot can later hold a finite
        // set and silently collide in the native-fused fingerprint. Fail closed.
        VarLayoutKind::StringKeyedArray { .. } => {
            kind.tagged_scalar_set_range_primary_proof().is_some()
                || kind.fixed_scalar_range_primary_proof().is_some()
        }
        // A static, fully-enumerated tuple/cross-product domain with plain-i64
        // (Int/Bool) range slots is safe for default flat BFS: the key set is
        // fixed across all successors and every slot is a plain integer, so
        // there is no aliasing or growth hazard (mirrors the `IntArray` rule).
        // G2-extension: a homogeneous string/model-value range is upgraded by a
        // validated finite-universe `TypeOK` proof, mirroring the
        // `StringKeyedArray` `FixedScalar` route (btree `valOf`).
        // WP-09/Part A: a heterogeneous finite scalar-union range (btree
        // `childOf`, Int ∪ model value) is upgraded by a validated
        // `TaggedScalarUnion` `TypeOK` proof under the `TY_TAGGED_SCALAR_UNION`
        // opt-in — the slot stores the injective universe index, so the mixed
        // sorts can never alias.
        VarLayoutKind::TupleKeyedArray { value_types, .. } => {
            value_types.iter().all(|ty| slot_type_is_plain_i64(*ty))
                || kind
                    .tuple_keyed_fixed_scalar_range_primary_proof()
                    .is_some()
                || kind
                    .tuple_keyed_tagged_scalar_union_range_primary_proof()
                    .is_some()
        }
        // A heterogeneous finite scalar-union function range
        // (`[Nodes -> Nodes \cup {NIL}]`, `Int \cup {sentinel}`, ...) is carried
        // here as `Recursive { IntFunction | Function | Sequence { value_layout:
        // TaggedScalarUnion } }` (a mixed Int/model-value/string element cannot
        // collapse into the scalar-typed `IntArray`/`StringKeyedArray`/
        // `TupleKeyedArray` `value_types`, so inference falls to the recursive
        // aggregate). This arm delegates straight to `supports_flat_primary`,
        // which admits the nested `TaggedScalarUnion` only under the
        // `TY_TAGGED_SCALAR_UNION` opt-in and only when the function-range guard
        // (`direct_function_range_blocks_flat_primary`) also clears it — the
        // union index encoding is injective, so `Int(k)` and `ModelValue(nil)`
        // never alias in the fixed-width fingerprint. A top-level scalar-union
        // variable is likewise carried as `Recursive { TaggedScalarUnion }` and
        // admitted through the same delegation.
        VarLayoutKind::Recursive { layout } => layout.supports_flat_primary(),
        VarLayoutKind::Bitmask { .. } | VarLayoutKind::Dynamic => false,
    }
}

fn direct_function_range_blocks_flat_primary(value_layout: &FlatValueLayout) -> bool {
    match value_layout {
        // A function-range `SetBitmask` writes a fixed-width bitmask into the
        // flat i64 buffer. Two independent facts must both hold for that to be
        // sound as a native flat-primary slot:
        //
        //   1. The universe is provably closed under every successor write (so
        //      no successor stores an out-of-universe element). The
        //      `universe_closure` provenance below captures this: a proven-closed
        //      static range (e.g. a TypeOK `SUBSET {0,1}` range) versus a merely
        //      sampled universe.
        //   2. The native action lowering must read AND write the bitmask slot in
        //      the SAME canonical i64 encoding the flat dedup layer fingerprints.
        //
        // GAP B: both facts now hold for a *proven-closed* universe, so such a
        // slot is admitted (`false` below). The native EXCEPT-set lowering
        // materializes the replacement value through the canonical
        // `static_set_bitmask_materialization_mask` / runtime bit-fold path
        // (bit `i` = position of the element in the range universe), which is
        // bit-identical to the interpreter's `set_bitmask_value_to_slot` — fact
        // (2) is satisfied. (The earlier `canonical_payload_mismatch` abort on
        // SimpleRegular — 9 states + a dedup fault versus the 277726 oracle —
        // was NOT a lowering bug: the masks were always canonical. The real
        // defect was in the compiled-BFS per-successor pre-seen dedup, whose
        // fail-closed rejection did not consult the recorded compiled-flat
        // payload witness before declaring a `canonical_payload_mismatch`; that
        // path is reached only when regular invariants are checked Rust-side, as
        // here, because TypeOK's `SUBSET {0,1} \ {{}}` range does not compile
        // natively. With the witness consulted, distinct logical states no longer
        // collide and SimpleRegular reports 277726 natively.)
        //
        // A *sampled* universe MUST stay fail-closed: a successor could write an
        // element outside the sampled universe and silently corrupt the slot, so
        // fact (1) is not established and we keep blocking.
        FlatValueLayout::SetBitmask {
            universe_closure, ..
        } if !universe_closure.is_proven_closed() => true,
        FlatValueLayout::SetBitmask { .. } => false,
        // A record-set function range is the record analogue: only a
        // proven-closed record universe is sound to write into a fixed-width
        // bitmask slot. A sampled universe stays fail-closed.
        FlatValueLayout::RecordSetBitmask {
            universe_closure, ..
        } if !universe_closure.is_proven_closed() => true,
        FlatValueLayout::RecordSetBitmask { .. } => false,
        // A `TaggedScalarUnion` function-range slot is domain-index encoded: a
        // range value maps to its position in the fixed universe, so `Int(k)`
        // and `ModelValue(nil)` occupy DISTINCT indices and can never alias in
        // the fixed-width fingerprint (unlike a bare `Scalar(String|ModelValue)`
        // range, whose raw `NameId` payload overlaps ordinary integer slots and
        // stays blocked below). It is therefore sound as a function-range slot
        // once the universe covers every successor write (established by the
        // `TypeOK`-derived proof that built the union). Gated behind the
        // `TY_TAGGED_SCALAR_UNION` opt-in so the default surface is unchanged.
        FlatValueLayout::TaggedScalarUnion { .. } => {
            !tagged_scalar_union_native_flat_primary_enabled()
        }
        FlatValueLayout::Scalar(SlotType::String | SlotType::ModelValue) => true,
        _ => false,
    }
}

/// Whether Track B increment 1's record-set native flat-primary admission is
/// enabled (env `TY_RECORD_SET_NATIVE=1`, default OFF).
///
/// Default OFF keeps every existing corpus spec with a proven record-set var
/// (e.g. PaxosCommit's 144-record `Message`) BYTE-IDENTICAL and avoids changing
/// their BFS storage path. The carrier + byte-exact lowering are wired
/// regardless; this flag only controls whether a record-set var is *admitted*
/// as a native flat-primary slot. Mirrors the `TY_NESTED_NATIVE_PROTO` probe.
#[must_use]
pub(crate) fn record_set_native_flat_primary_enabled() -> bool {
    std::env::var_os("TY_RECORD_SET_NATIVE").is_some_and(|v| v == "1")
}

/// Whether the heterogeneous finite scalar-union (`TaggedScalarUnion`) native
/// flat-primary admission is enabled (env `TY_TAGGED_SCALAR_UNION=1`, default
/// OFF).
///
/// Default OFF keeps every existing corpus spec BYTE-IDENTICAL: with the flag
/// off the union shape is never *constructed* by layout inference (a
/// heterogeneous `Int \cup {NIL}`-style range keeps returning no flat layout, as
/// today) and is never *admitted* as a flat-primary slot even if some other path
/// produced it. When the flag is on, layout inference assembles the concrete
/// deduplicated universe for such a union and admits the resulting
/// `FlatValueLayout::TaggedScalarUnion` slot as native flat-primary storage.
///
/// The flat-state round-trip (`tagged_scalar_union_value_to_slot` /
/// `reconstruct_tagged_scalar_union_slot`) is domain-index encoded and bijective
/// over its universe regardless of this flag; the flag only controls whether the
/// shape is *constructed and admitted* as the primary BFS representation, so no
/// non-opted spec changes its fingerprint. Mirrors the `TY_RECORD_SET_NATIVE`
/// and `TY_NESTED_NATIVE_PROTO` probes.
#[must_use]
pub(crate) fn tagged_scalar_union_native_flat_primary_enabled() -> bool {
    std::env::var_os("TY_TAGGED_SCALAR_UNION").is_some_and(|v| v == "1")
}

/// Whether the scalar-or-tuple union (`TaggedUnion`) native flat-primary
/// admission is enabled (env `TY_SCALAR_TUPLE_UNION=1`, default OFF).
///
/// This is the sum-type sibling of [`tagged_scalar_union_native_flat_primary_enabled`]:
/// it covers a variable whose value is a scalar sentinel OR one of several
/// fixed-arity tuples — btree's
/// `args \in {NIL} \cup {<<k>>: k \in Keys} \cup {<<k,v>>: k \in Keys, v \in Vals}`.
/// Init-sampling alone sees only `NIL` and infers a one-slot `Scalar(String)`,
/// so every tuple write and every `args[i]` read fails closed; the union layout
/// is what makes the variable genuinely flattenable.
///
/// Default OFF keeps every existing corpus spec BYTE-IDENTICAL: with the flag
/// off no writer-evidence union is *constructed* and the shape is never
/// *admitted* as flat-primary even if some other path produced it, which
/// reproduces the historical `false` veto exactly. The flat-state round-trip
/// (`try_write_flat_value_slots` / `try_reconstruct_flat_value` `TaggedUnion`
/// arms) is tag-dispatched and bijective regardless of this flag.
#[must_use]
pub(crate) fn scalar_tuple_union_native_flat_primary_enabled() -> bool {
    std::env::var_os("TY_SCALAR_TUPLE_UNION").is_some_and(|v| v == "1")
}

/// Whether WP-15's write-side flat-admission extension is enabled
/// (env `TY_FLAT_WRITE_ADMIT=1`, default OFF).
///
/// This gate currently controls exactly one tolerance: when SEVERAL checked
/// invariants prove the SAME sequence/fixed-domain type fact (e.g. MCBakery
/// checks both `TypeOK` and `Inv == TypeOK /\ IInv`, so every
/// `v \in [Procs -> T]` clause is collected twice with two different
/// proving-invariant labels), the per-path uniqueness lookups
/// (`unique_sequence_fixed_domain_type_proof`, `unique_sequence_proof`) judge
/// the duplicates STRUCTURALLY — on `(domain, element layout)` / `max_len`,
/// with the proving-invariant label normalized away — instead of by full
/// equality, which treats a label-only duplicate as an ambiguity and fails
/// closed. Structurally identical proofs are interchangeable for flat-state
/// encoding (the encoding depends only on the domain and element layout, never
/// on WHICH invariant proved them), so accepting one of them is sound; hints
/// that genuinely disagree on the encoding still fail closed under the gate.
/// This mirrors the structural-fingerprint rule `unique_sequence_element_proof`
/// has always used.
///
/// Default OFF keeps every existing corpus spec BYTE-IDENTICAL: a spec that
/// today fails the label-sensitive uniqueness check keeps its historical
/// fail-closed layout (and BFS storage path) exactly. Mirrors the
/// `TY_RECORD_SET_NATIVE` / `TY_TAGGED_SCALAR_UNION` opt-in probes.
#[must_use]
pub(crate) fn flat_write_admission_enabled() -> bool {
    std::env::var_os("TY_FLAT_WRITE_ADMIT").is_some_and(|v| v == "1")
}

/// WP-33: whether a model-value/string-keyed function whose range is a PROVEN
/// FINITE `Int` universe (`v \in [D -> a..b]`) is admitted to flat-primary
/// storage (env `TY_FLAT_INT_RANGE_ADMIT=1`, default OFF).
///
/// This is a narrow widening of `FixedScalarRangeProof` admission from
/// `{String, ModelValue, Bool}` to `{.., Int}`; the full soundness argument
/// lives on [`VarLayoutKind::fixed_scalar_range_primary_proof`]. It reaches
/// ONLY the `StringKeyedArrayRangeEncoding::FixedScalar` route, so an unproven
/// `ScalarSlots` Int range (a sampled `[D -> Int]`) is untouched and keeps
/// failing closed.
///
/// Default OFF keeps every existing corpus spec BYTE-IDENTICAL: with the flag
/// off, an Int-ranged `FixedScalar` proof returns `None` here exactly as
/// before, so the variable stays off flat-primary storage. Mirrors the
/// `TY_RECORD_SET_NATIVE` / `TY_TAGGED_SCALAR_UNION` / `TY_FLAT_WRITE_ADMIT`
/// opt-in probes.
#[must_use]
pub(crate) fn flat_int_range_admission_enabled() -> bool {
    std::env::var_os("TY_FLAT_INT_RANGE_ADMIT").is_some_and(|v| v == "1")
}

/// Whether the finite tagged-union (sum-type) native flat-primary admission is
/// enabled (env `TY_TAGGED_UNION=1`, default OFF).
///
/// This is the SAME opt-in that gates the writer-analysis construction of a
/// [`FlatValueLayout::TaggedUnion`] layout (sub-pieces 1+2), so with the flag off
/// no tagged-union layout is ever built and this predicate is unreachable; every
/// non-opted spec stays BYTE-IDENTICAL. When the flag is on, the layout is
/// constructed and this admits the whole-var `TaggedUnion` slot as the primary
/// (flat) BFS representation.
///
/// SOUNDNESS: promoting the var to flat-primary changes its storage to the flat
/// tag+payload encoding. That encoding is byte-exact and round-trip-verified by
/// `flat_state.rs` (`try_write_flat_value_slots` / `try_reconstruct_flat_value`),
/// and any reachable value that fits no variant raises
/// `TaggedUnionValueOutsideUniverse`, which the CLI catches and transparently
/// retries WITHOUT flat storage (never a silent undercount). The native
/// tag-dispatch store/read ABI is NOT yet complete, so every native context that
/// touches a tagged-union slot fails closed (`UnsupportedOpcode`) and the fused
/// build falls back to the interpreter — whose flat codec (with the retry
/// backstop above) is authoritative. Mirrors the `TY_TAGGED_SCALAR_UNION`,
/// `TY_RECORD_SET_NATIVE`, and `TY_NESTED_NATIVE_PROTO` probes.
#[must_use]
pub(crate) fn tagged_union_native_flat_primary_enabled() -> bool {
    std::env::var_os("TY_TAGGED_UNION").is_some_and(|v| v == "1")
}

/// Whether the HEURISTIC (unproven) sequence-capacity flat-primary admission is
/// enabled (env `TY_SEQ_HEURISTIC_CAPACITY=1`, default OFF).
///
/// A growing sequence like btree's `toSplit' = <<parent>> \o toSplit`
/// (`toSplit \in Seq(Nodes)`, Init `<<>>`) has NO checked length bound, so the
/// duplicate-free `Len(v) <= |U|` capacity proof does not fire (a prepend is not
/// one of the DF-preserving write forms) and the sequence stays
/// [`SequenceBoundEvidence::Observed`] — a flat-primary blocker.
///
/// With this flag on, layout inference derives a HEURISTIC capacity equal to the
/// element universe cardinality `|U|` from the checked `v \in Seq(U)` invariant
/// (btree: `|Nodes| = |1..MaxNode| = 8`) and admits the sequence to flat-primary
/// storage with [`SequenceBoundEvidence::HeuristicUniverseCapacity`]. Unlike a
/// proven bound this is NOT a certified upper bound on `Len(v)`.
///
/// SOUNDNESS rests ENTIRELY on the fail-closed overflow backstop, NOT on the
/// heuristic being correct: the flat write path
/// ([`super::flat_state::try_write_flat_value_slots`]) checks `len > max_len` and
/// returns [`super::flat_state::FlatSerializationError::SequenceLengthExceedsCapacity`]
/// on any overflow — which surfaces as `CheckError::flat_layout_unsupported_value`
/// and makes the CLI transparently re-run WITHOUT flat storage (the interpreter
/// path, authoritative), never a silent truncation or undercount. The bound is
/// non-proven ([`SequenceBoundEvidence::is_proven`] is `false`), so the native
/// lowering bridges it to `capacity_proven=false` exactly like `Observed`: no
/// native capacity-driven enumeration or write fires on it (fail closed). Thus
/// the promotion only changes the flat STORAGE path (dropping the var from
/// `flat_primary_blockers`), and every path that could get a count wrong instead
/// fails closed. Default OFF keeps every non-opted spec BYTE-IDENTICAL: with the
/// flag off the heuristic capacity is never CONSTRUCTED by layout inference and
/// never ADMITTED here, so the sequence stays `Observed`. Mirrors the
/// `TY_TAGGED_UNION` / `TY_TAGGED_SCALAR_UNION` / `TY_RECORD_SET_NATIVE` probes.
#[must_use]
pub(crate) fn seq_heuristic_capacity_enabled() -> bool {
    std::env::var_os("TY_SEQ_HEURISTIC_CAPACITY").is_some_and(|v| v == "1")
}

/// True iff a single record-set-bitmask universe record FIELD value is natively
/// representable by the ABI carrier (`tla_jit_abi::SetBitmaskElement`).
///
/// Accepts the four scalar leaves — `SmallInt` / `Bool` / `String` /
/// `ModelValue` — directly, and additionally a `Value::Set` whose every element
/// is one of those scalar leaves. A set-valued field (e.g. the `rsrc : SUBSET
/// Resources` field of an `AllocatorImplementation` message record) is folded to
/// a single scalar carrier slot by the bridge; the fold value is NOT load-bearing
/// for soundness (every native record-set op — enum-fold union/diff and
/// membership — fails closed on a set-shaped runtime element field before the
/// carried constant is ever compared, so the action falls back to the
/// interpreter, and flat-state STORAGE bit assignment goes through the
/// interpreter's full-`Value` `record_set_bitmask_value_to_slots`, never the ABI
/// fold). The gate here only certifies that the bridge can produce SOME faithful
/// scalar carrier for the field, so the two never disagree and flat-primary is
/// never admitted over a field the bridge would map to `CompoundLayout::Dynamic`
/// (the generic-pointer rc=139 path).
///
/// `Value::Int` (arbitrary-precision BigInt) is deliberately rejected: the native
/// bitmask element is `i64`-only. A nested set of non-scalars (set-of-sets,
/// set-of-records) is rejected too — only a flat set of scalar leaves folds.
#[must_use]
pub(crate) fn record_set_bitmask_field_native_representable(field: &crate::Value) -> bool {
    match field {
        crate::Value::SmallInt(_)
        | crate::Value::Bool(_)
        | crate::Value::String(_)
        | crate::Value::ModelValue(_) => true,
        crate::Value::Set(set) => set.iter().all(|elem| {
            matches!(
                elem,
                crate::Value::SmallInt(_)
                    | crate::Value::Bool(_)
                    | crate::Value::String(_)
                    | crate::Value::ModelValue(_)
            )
        }),
        _ => false,
    }
}

/// True iff every record in a record-set-bitmask universe is natively
/// representable: each element is a `Value::Record` whose every field value is
/// natively representable (see [`record_set_bitmask_field_native_representable`]).
///
/// This predicate is the flat-primary admission gate's mirror of the bridge's
/// `record_to_jit_bitmask_fields` fail-closed check, so the two never disagree:
/// a universe this accepts is one the bridge can faithfully carry, and one it
/// rejects keeps flat-primary declined rather than re-exposing the generic
/// set-pointer (rc=139) path.
#[must_use]
pub(crate) fn record_set_bitmask_universe_native_representable(universe: &[crate::Value]) -> bool {
    universe.iter().all(|value| {
        let crate::Value::Record(record) = value else {
            return false;
        };
        record
            .iter()
            .all(|(_name, field)| record_set_bitmask_field_native_representable(field))
    })
}

#[must_use]
pub(crate) fn ordered_dense_int_domain(domain: &[FlatScalarValue]) -> Option<(i64, usize)> {
    let Some(FlatScalarValue::Int(lo)) = domain.first() else {
        return None;
    };

    for (index, value) in domain.iter().enumerate() {
        let index = i64::try_from(index).ok()?;
        let expected = lo.checked_add(index)?;
        if !matches!(value, FlatScalarValue::Int(actual) if *actual == expected) {
            return None;
        }
    }

    Some((*lo, domain.len()))
}

/// Provenance of a [`FlatValueLayout::SetBitmask`] universe.
///
/// The bitmask encoding maps each universe element to a fixed bit index. The
/// encoding is only sound for as long as every value that can occupy the slot
/// is a subset of the recorded universe. A *sampled* universe (grown from
/// observed states) carries no such guarantee — a later successor could write
/// an element outside it — whereas a *proven-closed* universe is backed by a
/// static type fact (e.g. a TypeOK `SUBSET {0,1}` range) that the model checker
/// enforces on every reachable state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum SetBitmaskUniverseClosure {
    /// Universe was inferred/grown from sampled state values. Sound for
    /// flat-state roundtrip and as a top-level slot, but NOT proven closed
    /// under arbitrary successor writes — fails closed for function-range
    /// flat-primary native dispatch.
    #[default]
    Sampled,
    /// Universe is a static, finite set proven closed under all successor
    /// writes by a source-level type invariant (the `invariant` source that
    /// justified it). Safe to use as a function-range flat-primary slot.
    ProvenClosed { invariant: Arc<str> },
    /// Universe was grown from sampled state values but is *monitored* at
    /// runtime: the model checker tracks every successor write and would catch
    /// (rather than silently corrupt) an out-of-universe element. This is the
    /// future provenance for dynamic universe discovery of set-of-sets state
    /// (the nested-set / `SlidingPuzzles` board track).
    ///
    /// INERT for now: nothing constructs this variant yet (the discovery
    /// monitor is a later step), and it deliberately reports
    /// [`Self::is_proven_closed`] `== false`, so everywhere it is matched it
    /// behaves EXACTLY like [`Self::Sampled`] — it never unblocks a
    /// function-range flat-primary admission. `monitor_enforced` records
    /// whether the runtime out-of-universe monitor is installed; even when
    /// `true` the universe is not statically proven closed, so flat-primary
    /// native dispatch stays fail-closed.
    #[cfg_attr(not(test), allow(dead_code))]
    DynamicallyDiscovered { monitor_enforced: bool },
}

impl SetBitmaskUniverseClosure {
    /// True when this universe is proven closed under all successor writes.
    #[must_use]
    pub(crate) fn is_proven_closed(&self) -> bool {
        matches!(self, SetBitmaskUniverseClosure::ProvenClosed { .. })
    }

    /// Merge two closure facts. Closure is only preserved when both sides are
    /// proven closed by the *same* invariant source; any sampled or mismatched
    /// merge degrades to `Sampled` (fail-closed).
    #[must_use]
    pub(crate) fn merge(&self, other: &Self) -> Self {
        match (self, other) {
            (
                SetBitmaskUniverseClosure::ProvenClosed { invariant: a },
                SetBitmaskUniverseClosure::ProvenClosed { invariant: b },
            ) if a == b => SetBitmaskUniverseClosure::ProvenClosed {
                invariant: Arc::clone(a),
            },
            _ => SetBitmaskUniverseClosure::Sampled,
        }
    }
}

/// Recursive fixed-size value layout for aggregate flat-state encoding.
///
/// This is intentionally compact: function keys, record fields, and set
/// universes are metadata; only mutable range values and sequence lengths are
/// serialized into the flat i64 buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FlatValueLayout {
    /// Single scalar slot.
    Scalar(SlotType),
    /// Function with contiguous integer domain `lo..lo+len-1`.
    IntFunction {
        lo: i64,
        len: usize,
        value_layout: Box<FlatValueLayout>,
    },
    /// Function with a known finite scalar domain in canonical order.
    Function {
        domain: Vec<FlatScalarValue>,
        value_layout: Box<FlatValueLayout>,
    },
    /// Record with known field names and recursive field layouts.
    Record {
        field_names: Vec<Arc<str>>,
        field_layouts: Vec<FlatValueLayout>,
    },
    /// Finite scalar set encoded as one bitmask slot.
    ///
    /// `universe_closure` records whether the universe is a static, provably
    /// closed set (e.g. a TypeOK `SUBSET {0,1}` range) under which every
    /// reachable successor write stays inside the universe, versus a universe
    /// that was merely sampled/grown from observed states. Only a proven-closed
    /// universe is safe to use as a *function-range* slot in the native
    /// flat-primary i64 buffer ABI (see `direct_function_range_blocks_flat_primary`):
    /// a successor write outside a sampled universe would silently corrupt the
    /// fixed-width bitmask slot.
    SetBitmask {
        universe: Vec<FlatScalarValue>,
        universe_closure: SetBitmaskUniverseClosure,
    },
    /// Finite *record* set encoded as one bitmask slot over a finite, statically
    /// enumerable record universe.
    ///
    /// This is the record analogue of [`FlatValueLayout::SetBitmask`]: a set
    /// variable (or function range) whose elements are records drawn from a
    /// finite universe `[f1: T1, ...]` / a union of such record-sets. Each
    /// universe record is flattened to a fixed bit index in canonical order, and
    /// the set value is a fixed-width bitmask (bit `i` = the `i`-th universe
    /// record is present).
    ///
    /// Soundness mirrors `SetBitmask`: the bitmask slot is only injective when
    /// the universe is provably closed under all successor writes. The universe
    /// is sorted+deduped in canonical `Value` order and every member is a
    /// concrete `Value::Record`, so the per-record encoding is canonical. A
    /// merely *sampled* universe stays fail-closed for function-range
    /// flat-primary dispatch via `direct_function_range_blocks_flat_primary`.
    RecordSetBitmask {
        /// Canonical, sorted, deduped record universe. Every element is a
        /// `Value::Record`.
        universe: Vec<crate::Value>,
        universe_closure: SetBitmaskUniverseClosure,
    },
    /// Finite *set-of-sets* (nested set) encoded as one bitmask slot per outer
    /// element word over a finite, statically enumerable universe of inner
    /// sets.
    ///
    /// This is the set-of-sets analogue of [`FlatValueLayout::RecordSetBitmask`]
    /// for nested-set state such as the `SlidingPuzzles` board (a set whose
    /// elements are themselves sets/positions). The `inner_universe` is the flat
    /// scalar universe shared by every inner set; each inner set is encoded as a
    /// `u64` bitmask over `inner_universe`. The `outer_universe` then enumerates
    /// the admissible inner-set bitmasks, and the nested-set value is itself a
    /// fixed-width bitmask (bit `i` = the `i`-th `outer_universe` inner-set is
    /// present), spanning `ceil(|outer_universe| / 64)` i64 slots.
    ///
    /// INERT scaffolding (nested-set discovery A2): nothing constructs this
    /// variant yet — inference does NOT produce it (that is A3/A4), there is no
    /// native compound ABI for it (Step B), and it deliberately reports
    /// [`Self::supports_flat_primary`] `== false`, so it cannot be admitted as a
    /// native flat-primary slot. The `#[allow(dead_code)]` is load-bearing: it
    /// proves the variant is genuinely unconstructed.
    #[cfg_attr(not(test), allow(dead_code))]
    NestedSetBitmask {
        /// Enumerated outer universe: each `u64` is an inner-set bitmask over
        /// `inner_universe` (bit `j` = `inner_universe[j]` present in that inner
        /// set). Canonical (sorted, deduped) order.
        outer_universe: Vec<u64>,
        /// Flat scalar universe shared by every inner set, in canonical order.
        inner_universe: Vec<FlatScalarValue>,
        /// Closure provenance of the *outer* universe (the set of admissible
        /// inner sets) under successor writes.
        outer_closure: SetBitmaskUniverseClosure,
        /// Closure provenance of the *inner* scalar universe under successor
        /// writes.
        inner_closure: SetBitmaskUniverseClosure,
    },
    /// Finite scalar union encoded as one typed universe index slot.
    // kept for shape/proof parity
    #[cfg_attr(not(test), allow(dead_code))]
    TaggedScalarUnion { proof: TaggedScalarUnionProof },
    /// Finite tagged union of *heterogeneous* shapes (a TLA+ sum type) encoded
    /// as `1 + max_payload` slots: slot 0 is the variant tag, the remaining
    /// slots hold the active variant's flat encoding (zero-filled otherwise).
    ///
    /// Admitted only when every variant is itself finite/closed; see
    /// [`TaggedUnionProof`] for the soundness and fingerprint-distinctness
    /// argument.
    // kept for shape/proof parity
    #[cfg_attr(not(test), allow(dead_code))]
    TaggedUnion { proof: TaggedUnionProof },
    /// Fixed-arity tuple with **per-position** (heterogeneous) element layouts.
    ///
    /// Unlike [`FlatValueLayout::Sequence`] (a homogeneous `Sequence(element)`
    /// with a runtime length slot), this is a positionally-typed tuple of a
    /// STATICALLY FIXED arity: position `i` is stored with `element_layouts[i]`'s
    /// own encoding, contiguously, with NO length slot (the arity is part of the
    /// layout). This is the tuple analogue of [`FlatValueLayout::Record`] for a
    /// `Value::Tuple`/`Value::Seq` value whose positions have different scalar
    /// types (e.g. btree `args = <<int-key, model-value-val>>`: position 1 is
    /// `Int`, position 2 is a `ModelValue`).
    ///
    /// # Non-overlap / canonicity
    /// Two distinct arities are two distinct layouts (a 1-tuple never fits a
    /// 2-tuple layout and vice versa), so as a set of TaggedUnion variants they
    /// are unambiguous. Within one arity, `value_fits_flat_value_layout` requires
    /// every position to fit its own layout, and each position encoder is itself
    /// canonical/injective, so the whole tuple encoding is injective.
    // kept for shape/proof parity
    #[cfg_attr(not(test), allow(dead_code))]
    HeterogeneousTuple {
        element_layouts: Vec<FlatValueLayout>,
    },
    /// Sequence with a fixed capacity. Slot 0 stores the current length.
    Sequence {
        bound: SequenceBoundEvidence,
        max_len: usize,
        element_layout: Box<FlatValueLayout>,
    },
}

impl FlatValueLayout {
    /// Number of compact i64 slots occupied by this fixed value layout.
    #[must_use]
    pub(crate) fn slot_count(&self) -> usize {
        match self {
            FlatValueLayout::Scalar(_) => 1,
            FlatValueLayout::IntFunction {
                len, value_layout, ..
            } => len * value_layout.slot_count(),
            FlatValueLayout::Function {
                domain,
                value_layout,
            } => domain.len() * value_layout.slot_count(),
            FlatValueLayout::Record { field_layouts, .. } => {
                field_layouts.iter().map(FlatValueLayout::slot_count).sum()
            }
            FlatValueLayout::SetBitmask { .. } | FlatValueLayout::TaggedScalarUnion { .. } => 1,
            // A record-set bitmask spans `ceil(|universe| / 64)` i64 slots so a
            // universe larger than a single 63-bit word still flat-encodes. The
            // universe is admission-capped at `MAX_RECORD_SET_BITMASK_UNIVERSE`
            // before any RecordSetBitmask layout is constructed, so this always
            // resolves; surface a regression loudly rather than masking it as 1.
            FlatValueLayout::RecordSetBitmask { universe, .. } => {
                super::flat_state::record_set_bitmask_slot_count(universe.len())
                    .expect("RecordSetBitmask universe must be admission-capped at MAX_RECORD_SET_BITMASK_UNIVERSE")
            }
            // A nested-set bitmask spans `ceil(|outer_universe| / 64)` i64 slots,
            // mirroring `RecordSetBitmask` (the outer universe is the bitmask
            // domain). INERT: never constructed, so the admission cap is upheld
            // by A3/A4 when a construction site lands; surface a regression
            // loudly rather than masking it as 1.
            FlatValueLayout::NestedSetBitmask { outer_universe, .. } => {
                super::flat_state::record_set_bitmask_slot_count(outer_universe.len())
                    .expect("NestedSetBitmask outer universe must be admission-capped at MAX_RECORD_SET_BITMASK_UNIVERSE")
            }
            FlatValueLayout::TaggedUnion { proof } => proof.slot_count(),
            // Fixed-arity: sum of the per-position widths, NO length slot.
            FlatValueLayout::HeterogeneousTuple { element_layouts } => {
                element_layouts.iter().map(FlatValueLayout::slot_count).sum()
            }
            FlatValueLayout::Sequence {
                max_len,
                element_layout,
                ..
            } => 1 + max_len * element_layout.slot_count(),
        }
    }

    /// True when every recursive sequence in this layout has both proven
    /// capacity and proven element layout.
    #[must_use]
    pub(crate) fn supports_flat_primary(&self) -> bool {
        match self {
            // A *proven* capacity of zero is a degenerate (and almost always
            // under-approximated) bound: it reserves no element slots, so a
            // single non-empty successor cannot be encoded and would panic on
            // serialization. INIT-only roundtrip verification cannot catch this
            // because the init sequence is empty. A genuinely always-empty
            // sequence gains nothing from flat encoding, so refusing `max_len ==
            // 0` here is strictly conservative and closes the
            // `sched' = sched \circ sq`-style growth hazard where a writer proof
            // missed the concatenation write and proved capacity 0.
            FlatValueLayout::Sequence { max_len: 0, .. } => false,
            FlatValueLayout::Sequence {
                bound,
                element_layout,
                ..
            } => bound.supports_flat_primary() && element_layout.supports_flat_primary(),
            FlatValueLayout::IntFunction { value_layout, .. }
            | FlatValueLayout::Function { value_layout, .. } => {
                !direct_function_range_blocks_flat_primary(value_layout)
                    && value_layout.supports_flat_primary()
            }
            FlatValueLayout::Record { field_layouts, .. } => field_layouts
                .iter()
                .all(FlatValueLayout::supports_flat_primary),
            // The compact scalar-union representation is flat-state
            // roundtrip-safe (the domain-index encode/decode round-trip is
            // bijective over its universe), but promoting it to the *primary*
            // native-fused representation changes a heterogeneous scalar var's
            // BFS storage path (compound / fail-closed → one union-index slot),
            // so it is admitted only under the `TY_TAGGED_SCALAR_UNION` opt-in.
            // With the flag off this matches the historical veto (`false`)
            // exactly, keeping every non-opted spec byte-identical. The local
            // trust-cg backend cannot yet lower the typed universe index, so an
            // admitted union var still falls back to the interpreter for its
            // actions; admission here only establishes the sound, state-exact
            // flat storage so it composes with the parallel trust-cg lowering.
            FlatValueLayout::TaggedScalarUnion { .. } => {
                tagged_scalar_union_native_flat_primary_enabled()
            }
            // SOUNDNESS (record-set-bitmask `\cup` SIGSEGV history): a top-level
            // record-set var packed as a fixed-width bitmask i64 is flat-state
            // roundtrip-safe (the `flat_state.rs` encode/reconstruct paths are
            // intact and used by the secondary/dedup flat path). Historically it
            // was NOT admitted as the *primary* native-fused representation
            // because the native ABI had no faithful record-set-bitmask compound
            // layout: the bridge mapped the slot to `CompoundLayout::Dynamic`, a
            // compiled `v' = v \cup {rec}` write went through the generic
            // set-pointer path, and the JIT `IntToPtr`-cast the packed bitmask
            // i64 and dereferenced it as a heap pointer — a NULL/garbage deref
            // (rc=139).
            //
            // Track B increment 1 wired the missing pieces: the bridge now
            // carries a faithful `CompoundLayout::RecordSetBitmask { universe,
            // slot_count, .. }` (NOT `Dynamic`), `tracked_shape_from_compound_layout`
            // derives `AggregateShape::RecordSetBitmask`, and the byte-exact
            // `set_ops` membership / union / diff lowering fires off that shape,
            // loading the multi-slot mask from its pointer-backed compact region
            // (never `IntToPtr`-dereferencing it). Every OTHER native context
            // that touches the mask operand fails closed (`UnsupportedOpcode`)
            // and falls back to the interpreter for that action, so the rc=139
            // trap is structurally closed.
            //
            // Admit flat-primary ONLY when (1) the universe is PROVEN closed (a
            // sampled universe could see an out-of-universe successor write and
            // silently corrupt the slot) AND (2) every universe record is
            // natively representable — each field is a scalar leaf the carrier's
            // `SetBitmaskElement` can encode. If the universe is NOT representable
            // the bridge falls back to `CompoundLayout::Dynamic`, which would
            // re-expose the generic-pointer path; declining flat-primary there
            // keeps the interpreter authoritative and fail-closed.
            //
            // TRACK B INCREMENT 1 ADMISSION GATE (env `TY_RECORD_SET_NATIVE=1`,
            // default OFF): promoting a record-set var to flat-primary changes
            // its BFS *storage* path (it becomes a pointer-backed multi-slot
            // bitmask the native lowering reads), which is a behavioural change
            // for every existing corpus spec that has a proven record-set var
            // (e.g. PaxosCommit's 144-record `Message`). To keep every non-opted
            // spec BYTE-IDENTICAL and avoid a perf regression on those large
            // specs, the admission is OFF by default and enabled only under the
            // probe flag. When OFF this arm matches the historical veto (`false`)
            // exactly; the carrier + lowering wiring is still fully in place, so
            // the env-on path validates the record-set native flat-primary slot
            // end-to-end without disturbing the default verdict surface. This
            // mirrors the `TY_NESTED_NATIVE_PROTO` probe below.
            FlatValueLayout::RecordSetBitmask { .. } => {
                record_set_native_flat_primary_enabled()
                    && self.record_set_bitmask_flat_primary_sound()
            }
            // SOUNDNESS (nested-set): a set-of-sets var packed as a fixed-width
            // bitmask has no native compound ABI yet (Step B), so it can never
            // be the primary native-fused representation. Fail closed exactly
            // like `RecordSetBitmask`: the bridge maps the slot to
            // `CompoundLayout::Dynamic` and the interpreter successor path stays
            // authoritative. INERT: never constructed yet.
            FlatValueLayout::NestedSetBitmask { .. } => false,
            // The tagged-union (sum-type) encoding is finite & flat-state
            // roundtrip-safe (every variant is finite by construction:
            // `TaggedUnionProof::new` rejects any variant that is not itself
            // `supports_flat_primary`, the tag slot distinguishes variants of
            // different shape, and the parent encoder zero-fills the trailing
            // payload slots, so the `(tag, payload)` encoding is injective and
            // fingerprint-distinct).
            //
            // TWO independent opt-ins CONSTRUCT this layout, and admission must
            // cover BOTH or a constructed union silently degrades to `Dynamic`:
            //   * `TY_TAGGED_UNION` — the invariant/writer sum-type scan
            //     (`collect_tagged_union_var_writer_proofs_with_ops`, gated in
            //     `run_prepare::configured_tagged_union_var_type_proofs`);
            //   * `TY_SCALAR_TUPLE_UNION` — the scalar-or-tuple writer-coverage
            //     scan (`collect_scalar_tuple_union_var_writer_proofs`, gated in
            //     `layout_inference`), which is what promotes btree's `args`.
            // Testing only one gate here was a merge artifact: the two features
            // landed on separate branches and the conflict resolution kept the
            // `TY_TAGGED_UNION` producer's admit line while the
            // `TY_SCALAR_TUPLE_UNION` producer stayed live, so `args` was built
            // as a union and then vetoed back to `Dynamic`.
            //
            // Both gates default OFF, so with neither set this is `false` —
            // the historical veto, byte-identical for every non-opted spec.
            //
            // Soundness does not rest on WHICH gate is on: it rests on the
            // `TaggedUnionProof` object, whose constructor enforces >= 2
            // pairwise-distinct, finite, non-overlapping variants. The native/JIT
            // tag-dispatch store/read ABI is NOT yet complete: the bridge carries
            // a faithful `CompoundLayout::TaggedUnion` (so the compact buffer
            // offsets agree with the flat layout), but every native context that
            // cannot prove the live tag fails closed (`UnsupportedOpcode`) and
            // falls back to the interpreter, whose flat codec — with the
            // `TaggedUnionValueOutsideUniverse` retry-without-flat backstop — is
            // authoritative. So flat-primary here is sound EVEN without native
            // execution: the promotion drops the var from `flat_primary_blockers`,
            // and no native loop can read/write it incorrectly (it can only fail
            // closed). See `tagged_union_native_flat_primary_enabled` and
            // `scalar_tuple_union_native_flat_primary_enabled`.
            FlatValueLayout::TaggedUnion { .. } => {
                tagged_union_native_flat_primary_enabled()
                    || scalar_tuple_union_native_flat_primary_enabled()
            }
            // FINITENESS gate (per-variant admission): a fixed-arity tuple is a
            // statically-finite, closed shape iff every position is. This is the
            // predicate `TaggedUnionProof::new` uses to admit a variant, so it
            // MUST report the finiteness of the positions (scalars → true), not
            // whether native tag-dispatch can lower it. A `HeterogeneousTuple` is
            // only ever constructed as a TaggedUnion variant (whose top-level
            // `supports_flat_primary` is `false` until the native tag-dispatch ABI
            // lands), so returning the finiteness result here never admits a bare
            // heterogeneous-tuple var into a native loop.
            FlatValueLayout::HeterogeneousTuple { element_layouts } => element_layouts
                .iter()
                .all(FlatValueLayout::supports_flat_primary),
            FlatValueLayout::Scalar(_) | FlatValueLayout::SetBitmask { .. } => true,
        }
    }

    /// Soundness predicate for admitting a `RecordSetBitmask` slot as a native
    /// flat-primary representation, independent of the
    /// [`record_set_native_flat_primary_enabled`] opt-in gate.
    ///
    /// Returns `true` iff the universe is PROVEN closed (a sampled universe
    /// could see an out-of-universe successor write and silently corrupt the
    /// fixed-width slot) AND every universe record is natively representable
    /// (scalar fields only — see [`record_set_bitmask_universe_native_representable`]).
    /// A non-record-set layout returns `false`. Factored out of
    /// [`Self::supports_flat_primary`] so the soundness condition is unit-testable
    /// without racing the process-global env flag.
    #[must_use]
    pub(crate) fn record_set_bitmask_flat_primary_sound(&self) -> bool {
        match self {
            FlatValueLayout::RecordSetBitmask {
                universe,
                universe_closure,
            } => {
                universe_closure.is_proven_closed()
                    && record_set_bitmask_universe_native_representable(universe)
            }
            _ => false,
        }
    }

    /// True when this layout contains a sequence component that is not safe as
    /// a flat-primary representation — an unproven (sampled) capacity bound or
    /// an unproven element layout.
    ///
    /// Such sequences are lossy on growth: the fixed flat buffer reserves
    /// `max_len` element slots from sampled states, but a later successor longer
    /// than that capacity cannot be encoded without truncation or panic. When
    /// flat BFS is force-enabled, that corruption silently breaks dedup (the
    /// same logical state encodes differently across paths, inflating the state
    /// count). This predicate lets the force-enable path fail closed for those
    /// shapes while still admitting sequence-free aggregate ranges such as the
    /// fixed-domain model-value `StringKeyedArray` sandbox.
    ///
    /// Sequence-safety mirrors [`Self::supports_flat_primary`] for the sequence
    /// node itself (proven capacity *and* proven element layout) but, unlike
    /// `supports_flat_primary`, tolerates the tagged-scalar/set and bitmask
    /// function-range encodings that have an explicit ArrayState reconstruction
    /// path.
    #[must_use]
    pub(crate) fn has_flat_primary_unsafe_sequence(&self) -> bool {
        match self {
            // A proven `max_len == 0` capacity is degenerate / under-approximated
            // (see `supports_flat_primary`): treat it as unsafe so the forced
            // flat path also fails closed on the `\circ`-growth hazard.
            FlatValueLayout::Sequence { max_len: 0, .. } => true,
            FlatValueLayout::Sequence {
                bound,
                element_layout,
                ..
            } => {
                !(bound.supports_flat_primary() && element_layout.supports_flat_primary())
                    || element_layout.has_flat_primary_unsafe_sequence()
            }
            FlatValueLayout::IntFunction { value_layout, .. }
            | FlatValueLayout::Function { value_layout, .. } => {
                value_layout.has_flat_primary_unsafe_sequence()
            }
            FlatValueLayout::Record { field_layouts, .. } => field_layouts
                .iter()
                .any(FlatValueLayout::has_flat_primary_unsafe_sequence),
            // A tagged-union's sequence variants carry a proven finite capacity
            // by construction (`TaggedUnionProof::new` rejects any variant that
            // is not `supports_flat_primary`), so they are never the lossy
            // sampled-bound sequences this predicate guards against. Recurse to
            // honor a deeper unproven sequence should one ever slip through.
            FlatValueLayout::TaggedUnion { proof } => proof
                .variants()
                .iter()
                .any(FlatValueLayout::has_flat_primary_unsafe_sequence),
            // A fixed-arity tuple has no runtime length slot, so it is never the
            // lossy sampled-bound sequence this predicate guards; recurse to honor
            // any deeper unproven sequence that might sit at a position.
            FlatValueLayout::HeterogeneousTuple { element_layouts } => element_layouts
                .iter()
                .any(FlatValueLayout::has_flat_primary_unsafe_sequence),
            FlatValueLayout::Scalar(_)
            | FlatValueLayout::SetBitmask { .. }
            | FlatValueLayout::RecordSetBitmask { .. }
            // A nested-set bitmask contains no sequence component, so it is
            // never the lossy sampled-bound sequence this predicate guards.
            // INERT: never constructed yet.
            | FlatValueLayout::NestedSetBitmask { .. }
            | FlatValueLayout::TaggedScalarUnion { .. } => false,
        }
    }
}

/// Classification of how a variable's value maps to i64 slots.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum VarLayoutKind {
    /// Int — 1 slot. Raw i64 value.
    Scalar,
    /// Bool — 1 slot. 0 = false, 1 = true.
    /// Distinct from `Scalar` so that roundtrip conversion preserves
    /// `Value::Bool` instead of returning `Value::SmallInt`.
    ScalarBool,
    /// String — 1 slot. Interned NameId stored as i64.
    /// Distinct from `Scalar` so roundtrip produces `Value::String`.
    /// Part of #3908.
    ScalarString,
    /// ModelValue — 1 slot. Interned NameId stored as i64.
    /// Distinct from `Scalar` so roundtrip produces `Value::ModelValue`.
    /// Part of #3908.
    ScalarModelValue,
    /// Top-level scalar string/model-value variable whose value is drawn from a
    /// provably finite, total universe (a string enum). Encoded identically to
    /// `ScalarString`/`ScalarModelValue` (interned NameId stored as i64 in a
    /// single slot), but admitted as primary flat-state storage because a
    /// source-level proof shows every reachable value stays within the universe.
    ///
    /// The flat encoding is total and bijective over *all* interned strings, so
    /// it never aliases regardless of the universe — the proof only authorizes
    /// the slot as the canonical BFS representation (G2). `base` records whether
    /// the variable is a `String` or `ModelValue` so roundtrip restores the
    /// correct `Value` variant.
    FixedScalar {
        /// Underlying scalar slot type: `SlotType::String` or
        /// `SlotType::ModelValue`.
        base: SlotType,
        /// Proof that the variable's value-set is a finite homogeneous universe.
        proof: FixedScalarRangeProof,
    },
    /// Integer-indexed array `[lo..hi -> Int/Bool/String]` — `len` contiguous slots.
    /// Each element is a scalar i64 (int value, 0/1 for bool, or NameId for string).
    IntArray {
        /// Inclusive lower bound of the domain interval.
        lo: i64,
        /// Number of elements (= hi - lo + 1).
        len: usize,
        /// When true, all array elements are Bool-typed. Enables
        /// `reconstruct_int_array` to emit `Value::Bool(slot != 0)`
        /// instead of `Value::SmallInt(slot)`. Fixes #4014.
        elements_are_bool: bool,
        /// Per-element type tag. When `Some`, enables correct reconstruction
        /// of string/model-value elements. `None` means all elements are
        /// either Int or Bool (determined by `elements_are_bool`).
        /// Part of #3908.
        element_types: Option<Vec<SlotType>>,
        /// Optional proof that every string/model-value element is drawn from a
        /// finite, homogeneous scalar universe (from a `[Dom -> FiniteScalarSet]`
        /// `TypeOK` clause, e.g. EWD998 `color \in [Node -> Color]`). Populated
        /// at inference time via `unique_fixed_scalar_range_proof`. When present
        /// and validated by [`Self::int_array_element_range_proof`], non-`i64`
        /// elements are admissible for *default* flat BFS, because the proof
        /// guarantees no successor writes a plain integer into the interned slot.
        /// `None` for plain Int/Bool arrays (G2-extension).
        element_range_proof: Option<FixedScalarRangeProof>,
    },
    /// Record with known field names — one scalar slot per field.
    /// Only applicable when ALL fields are scalar (Int/Bool/String/ModelValue).
    Record {
        /// Field names in canonical (sorted NameId) order.
        field_names: Vec<Arc<str>>,
        /// Per-field Bool type tracking. `field_is_bool[i]` is true when
        /// field `field_names[i]` holds a `Value::Bool`. This enables
        /// `reconstruct_record` to emit `Value::Bool(slot != 0)` instead
        /// of `Value::SmallInt(slot)` for Bool-typed fields. Fixes #4014.
        field_is_bool: Vec<bool>,
        /// Per-field type tag. Enables correct reconstruction of
        /// string/model-value fields. Part of #3908.
        field_types: Vec<SlotType>,
        /// Optional per-field proofs that each string/model-value field is drawn
        /// from a finite, homogeneous scalar universe (from a `[f:
        /// FiniteScalarSet, ...]` `TypeOK` record-set clause, e.g. EWD998 `token
        /// \in [pos: Node, q: Int, color: Color]`). When present, this vector is
        /// parallel to `field_names`/`field_types`: entry `i` covers
        /// `field_names[i]`, and `None` entries mean the field has no universe
        /// proof (so it must be plain Int/Bool to be flat-admissible). A `None`
        /// outer vector means no field has a proof. Validated by
        /// [`Self::record_fields_flat_admissible`] (G2-extension).
        field_range_proofs: Option<Vec<Option<FixedScalarRangeProof>>>,
    },
    /// String-keyed function — `len` contiguous slots for values, domain keys
    /// stored as metadata. Used for `[{"a", "b"} -> Int]` patterns.
    /// Part of #3908: compound type flat state roundtrip.
    StringKeyedArray {
        /// Domain keys as interned NameId values, in canonical sorted order.
        domain_keys: Vec<Arc<str>>,
        /// Per-key type tag indicating whether each `domain_keys[i]` is a
        /// `Value::String` or `Value::ModelValue`. Without this, ModelValue
        /// domains (e.g. `RM = {rm1, rm2, rm3}` from a CONSTANTS config) are
        /// silently reconstructed as `Value::String` and fail the flat
        /// roundtrip equality check. Fixes #4277.
        domain_types: Vec<SlotType>,
        /// Per-element type tag for the range values.
        value_types: Vec<SlotType>,
        /// Optional range encoding proof. Kept explicit so scalar-only compact
        /// functions cannot be mistaken for Dijkstra-style scalar/set slots.
        range_encoding: StringKeyedArrayRangeEncoding,
    },
    /// Tuple/cross-product-keyed function — `len` contiguous slots for values,
    /// the canonical sorted tuple-key table stored as metadata. Used for
    /// `[{<<x, y>> : x, y \in 1..N} -> BOOLEAN]` patterns (e.g. GameOfLife's
    /// `grid`), where the domain is a fully-enumerated, static finite set of
    /// scalar tuples.
    ///
    /// This is the tuple analogue of [`Self::StringKeyedArray`]. A statically
    /// enumerated tuple domain has an obvious canonical flat encoding: sort the
    /// tuple keys in `Value` (lexicographic) order, assign one contiguous i64
    /// slot per key, exactly like `IntArray`/`StringKeyedArray`. Because the
    /// domain is fixed, there is no successor-write fingerprint-collision hazard
    /// — every reachable successor function shares the identical key set.
    TupleKeyedArray {
        /// Canonical sorted tuple domain keys (each a `Value::Tuple` of
        /// scalars), in ascending `Value` order. Used both to map a key to its
        /// slot index and to reconstruct the function domain.
        domain_keys: Vec<tla_value::Value>,
        /// Per-element type tag for the range values, parallel to `domain_keys`.
        /// Under `range_encoding = TaggedScalarUnion`, the slot holds a universe
        /// index rather than a raw scalar, so `value_types` is not consulted for
        /// reconstruction (the union universe drives it).
        value_types: Vec<SlotType>,
        /// How each range slot is encoded. `ScalarSlots` (default) preserves the
        /// historical raw-i64 encoding; `TaggedScalarUnion` stores the injective
        /// universe index of a heterogeneous finite scalar union range.
        range_encoding: TupleKeyedArrayRangeEncoding,
    },
    /// Recursive fixed-size aggregate layout.
    Recursive { layout: FlatValueLayout },
    /// Small finite set encoded as a bitmask in a single i64.
    /// Bit i is set iff element i (from a canonical enumeration) is in the set.
    /// Only applicable when the universe has <= 63 elements.
    /// Scaffolding for JIT V2 flat state pipeline (#3986).
    #[allow(dead_code)]
    Bitmask {
        /// Number of elements in the universe.
        universe_size: usize,
    },
    /// Fallback: the variable cannot be flattened statically. The single i64
    /// slot holds 0 as a placeholder, and the actual value must be retrieved
    /// from the originating ArrayState. This allows the layout to always
    /// produce a fixed-size buffer even for heterogeneous states.
    Dynamic,
}

impl VarLayoutKind {
    /// Number of i64 slots this kind occupies.
    #[must_use]
    pub(crate) fn slot_count(&self) -> usize {
        match self {
            VarLayoutKind::Scalar
            | VarLayoutKind::ScalarBool
            | VarLayoutKind::ScalarString
            | VarLayoutKind::ScalarModelValue
            | VarLayoutKind::FixedScalar { .. } => 1,
            VarLayoutKind::IntArray { len, .. } => *len,
            VarLayoutKind::Record { field_names, .. } => field_names.len(),
            VarLayoutKind::StringKeyedArray { domain_keys, .. } => domain_keys.len(),
            VarLayoutKind::TupleKeyedArray { domain_keys, .. } => domain_keys.len(),
            VarLayoutKind::Recursive { layout } => layout.slot_count(),
            VarLayoutKind::Bitmask { .. } => 1,
            VarLayoutKind::Dynamic => 1,
        }
    }

    #[must_use]
    pub(crate) fn tagged_scalar_set_range_proof(&self) -> Option<&TaggedScalarSetRangeProof> {
        let VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
        } = self
        else {
            return None;
        };

        if domain_keys.is_empty()
            || domain_keys.len() != domain_types.len()
            || domain_keys.len() != value_types.len()
            || !value_types.iter().all(|ty| *ty == proof.scalar_type())
        {
            return None;
        }

        Some(proof)
    }

    #[must_use]
    pub(crate) fn fixed_scalar_range_proof(&self) -> Option<&FixedScalarRangeProof> {
        let VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            range_encoding: StringKeyedArrayRangeEncoding::FixedScalar(proof),
        } = self
        else {
            return None;
        };

        if domain_keys.is_empty()
            || domain_keys.len() != domain_types.len()
            || domain_keys.len() != value_types.len()
            || !value_types.iter().all(|ty| *ty == proof.scalar_type())
        {
            return None;
        }

        Some(proof)
    }

    /// Shape-validated `FixedScalar` range proof for a tuple-keyed function.
    ///
    /// The tuple analogue of [`Self::fixed_scalar_range_proof`]: returns
    /// `Some(proof)` only when the kind is a `TupleKeyedArray` whose
    /// `range_encoding` carries a `FixedScalar` proof, the domain is non-empty
    /// and parallel to `value_types`, and every per-slot `value_type` equals
    /// the proof's scalar type. Anything else fails closed.
    #[must_use]
    pub(crate) fn tuple_keyed_fixed_scalar_range_proof(&self) -> Option<&FixedScalarRangeProof> {
        let VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding: TupleKeyedArrayRangeEncoding::FixedScalar(proof),
        } = self
        else {
            return None;
        };

        if domain_keys.is_empty()
            || domain_keys.len() != value_types.len()
            || !value_types.iter().all(|ty| *ty == proof.scalar_type())
        {
            return None;
        }

        Some(proof)
    }

    /// Primary-storage proof for a tuple-keyed function whose range is a finite
    /// scalar string/model-value universe (a `FixedScalar` range proof derived
    /// from `TypeOK`).
    ///
    /// Soundness mirrors [`Self::fixed_scalar_range_primary_proof`]: the
    /// interned-`NameId` slot encoding (`u32 as i64`) is fixed-width,
    /// non-negative and injective over all strings/model-values, the static
    /// tuple domain fixes the key set across all successors, and the checked
    /// `TypeOK` proof certifies the range universe is closed under every
    /// successor write. `Bool`/`Int` ranges are excluded here because plain-i64
    /// ranges are already structurally admitted by the `TupleKeyedArray` arm;
    /// a *mixed* i64/interned range can never earn a homogeneous proof and thus
    /// stays fail-closed (the WP-05 union-carrier seam).
    #[must_use]
    fn tuple_keyed_fixed_scalar_range_primary_proof(&self) -> Option<&FixedScalarRangeProof> {
        let proof = self.tuple_keyed_fixed_scalar_range_proof()?;
        if !matches!(proof.scalar_type(), SlotType::String | SlotType::ModelValue) {
            return None;
        }
        if proof.scalar_universe().is_empty()
            || proof
                .scalar_universe()
                .iter()
                .any(|value| value.slot_type() != proof.scalar_type())
        {
            return None;
        }
        Some(proof)
    }

    /// Shape-validated `TaggedScalarUnion` range proof for a tuple-keyed
    /// function (WP-09/Part A).
    ///
    /// Returns `Some(proof)` only when the kind is a `TupleKeyedArray` whose
    /// `range_encoding` carries a `TaggedScalarUnion` proof, the domain is
    /// non-empty and parallel to `value_types`, and every *sampled* per-slot
    /// type is a sort the proof universe actually contains (so an
    /// init-sampled value the universe cannot express is impossible). The
    /// sampled `value_types` deliberately do NOT constrain later states — a
    /// union slot legally alternates between its Int and model-value arms;
    /// runtime fit is enforced value-by-value against the universe
    /// (`value_fits_tuple_keyed_range_slot`). Anything malformed fails closed.
    #[must_use]
    pub(crate) fn tuple_keyed_tagged_scalar_union_range_proof(
        &self,
    ) -> Option<&TaggedScalarUnionProof> {
        let VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof),
        } = self
        else {
            return None;
        };

        if domain_keys.is_empty()
            || domain_keys.len() != value_types.len()
            || proof.universe().is_empty()
            || value_types.iter().any(|ty| {
                !proof
                    .universe()
                    .iter()
                    .any(|member| member.slot_type() == *ty)
            })
        {
            return None;
        }

        Some(proof)
    }

    /// Primary-storage variant of
    /// [`Self::tuple_keyed_tagged_scalar_union_range_proof`], gated behind the
    /// `TY_TAGGED_SCALAR_UNION` opt-in so the default surface is unchanged
    /// (mirrors the `direct_function_range_blocks_flat_primary` gating of the
    /// nested `FlatValueLayout::TaggedScalarUnion` carrier).
    ///
    /// Soundness: each range slot stores the injective index of its value in
    /// the fixed, deduplicated union universe
    /// (`encode_tagged_scalar_union_value`), so distinct values — including an
    /// `Int(k)` / `ModelValue(nil)` pair whose legacy raw payloads could
    /// collide — occupy distinct slot values, and the static tuple domain
    /// fixes the key set across all successors. The CHECKED `TypeOK`-derived
    /// proof certifies the universe is closed under every successor write, and
    /// — unlike the infallible raw `FixedScalar` encode the #43 writer veto
    /// protects — the union encode/fit is fail-closed by construction: a
    /// non-scalar, out-of-universe, missing-key, or domain-drifted runtime
    /// value is a hard encode/fit error that keeps the state on the
    /// compound/interpreter path, never a mis-encoded slot (mirrors the WP-05
    /// whole-variable union override's soundness stance).
    #[must_use]
    fn tuple_keyed_tagged_scalar_union_range_primary_proof(
        &self,
    ) -> Option<&TaggedScalarUnionProof> {
        if !tagged_scalar_union_native_flat_primary_enabled() {
            return None;
        }
        self.tuple_keyed_tagged_scalar_union_range_proof()
    }

    /// Proof for a top-level finite-universe scalar string/model-value variable.
    ///
    /// Returns `Some(proof)` only when the variable is a `FixedScalar` whose
    /// `base` slot type is `String` or `ModelValue`, the proof universe is
    /// non-empty, and the proof's scalar type agrees with `base`. Anything else
    /// fails closed.
    #[must_use]
    pub(crate) fn fixed_scalar_var_proof(&self) -> Option<&FixedScalarRangeProof> {
        let VarLayoutKind::FixedScalar { base, proof } = self else {
            return None;
        };
        if !matches!(base, SlotType::String | SlotType::ModelValue) {
            return None;
        }
        if *base != proof.scalar_type() || proof.scalar_universe().is_empty() {
            return None;
        }
        if proof
            .scalar_universe()
            .iter()
            .any(|value| value.slot_type() != *base)
        {
            return None;
        }
        Some(proof)
    }

    #[must_use]
    fn tagged_scalar_set_range_primary_proof(&self) -> Option<&TaggedScalarSetRangeProof> {
        let proof = self.tagged_scalar_set_range_proof()?;
        tagged_scalar_set_scalar_type_supports_flat_primary(proof.scalar_type()).then_some(proof)
    }

    /// Primary-storage proof for a fixed string/model-value keyed function whose
    /// range is a finite scalar string/model-value universe (a `FixedScalar`
    /// range proof derived from `TypeOK`).
    ///
    /// Returns `Some(proof)` only when the validated `FixedScalar` range proof
    /// (already shape-checked by [`Self::fixed_scalar_range_proof`]) has a
    /// `scalar_type` of `String` or `ModelValue`, with a non-empty, homogeneous
    /// universe. Anything else fails closed.
    ///
    /// Soundness: each range slot is encoded with the identical one-slot interned
    /// `NameId` scheme used by bare `FixedScalar` scalar vars
    /// (`value_to_scalar_i64` -> `intern_name(s).0 as u32 as i64`, inverted by
    /// `resolve_name_id`). That encoding is fixed-width (one i64 per key),
    /// non-negative (`u32 as i64` is always >= 0, so it can never collide with the
    /// tagged-set sign convention), and injective over all strings/model-values
    /// (the global intern table is a bijection). The domain key set is fixed and
    /// each per-slot `value_type` is constrained to equal `scalar_type`, so the
    /// whole packed buffer is a bijection on the reachable value set — exactly the
    /// safety property `fixed_scalar_var_proof` already relies on.
    ///
    /// `Bool` ranges are admitted by the same argument: every payload is `0`/`1`
    /// (`value_to_scalar_i64` for `Bool`), which is fixed-width, non-negative
    /// (never collides with the tagged-set sign convention), and injective over
    /// the `{FALSE, TRUE}` universe; the validated range proof (writer-
    /// corroborated per #43, so no Init/Next writer can store a set or other
    /// non-scalar) certifies closure, which is what the general
    /// `StringKeyedArray` fail-closed comment guards against for *sampled*
    /// Bool slots. This admits the ubiquitous PlusCal shape
    /// `flags \in [Proc -> BOOLEAN]` (e.g. DijkstraMutex `b`/`c`) to
    /// flat-primary storage.
    ///
    /// WP-33 (`TY_FLAT_INT_RANGE_ADMIT=1`, default OFF): `Int` ranges are
    /// admitted under the SAME argument once — and only once — a validated
    /// finite-universe `FixedScalar` range proof covers them, i.e. the checked
    /// `TypeOK` clause is `v \in [D -> a..b]` (or any other fully enumerated
    /// finite Int set) rather than the unbounded `v \in [D -> Int]`.
    ///
    /// The historical rejection reason — "arbitrary i64 payloads are not proven
    /// non-negative, so they could alias the tagged-set sign convention" — does
    /// not apply on this route: the tagged-set sign convention (`-1 - mask`)
    /// exists ONLY in `StringKeyedArrayRangeEncoding::TaggedScalarOrSet`, and
    /// `fixed_scalar_range_proof()` above matches exclusively on
    /// `StringKeyedArrayRangeEncoding::FixedScalar`. A `FixedScalar` range slot
    /// is written as a bare `value_to_scalar_i64` payload and read back as
    /// `reconstruct_slot_value(slot, SlotType::Int)`, with no sign-tagged
    /// branch anywhere, so a negative Int payload cannot alias anything.
    ///
    /// Losslessness: every universe member is a `FlatScalarValue::Int(i64)`, so
    /// `value_to_scalar_i64` -> `Value::SmallInt` is an exact bijection on the
    /// universe (no `to_i64()` truncation is reachable — an out-of-i64 integer
    /// can never be a universe member in the first place). Closure is enforced
    /// twice and fails CLOSED both times: `value_fits_string_keyed_range_slot`
    /// rejects any value outside `proof.scalar_universe()` before the state
    /// takes the flat path, and `string_keyed_range_value_to_slot` re-checks at
    /// write time and returns `FixedScalarRangeValueOutsideUniverse` rather
    /// than writing an uncovered slot. That is exactly the guarantee the
    /// general `StringKeyedArray` fail-closed comment demands and that a bare
    /// `ScalarSlots` Int range (sampled, unproven — Disruptor's
    /// `claimed_sequence`/`read`, which are `[D -> Int]`) still cannot offer:
    /// those keep failing closed here, since they carry no proof at all.
    #[must_use]
    fn fixed_scalar_range_primary_proof(&self) -> Option<&FixedScalarRangeProof> {
        let proof = self.fixed_scalar_range_proof()?;
        let int_admitted =
            proof.scalar_type() == SlotType::Int && flat_int_range_admission_enabled();
        if !int_admitted
            && !matches!(
                proof.scalar_type(),
                SlotType::String | SlotType::ModelValue | SlotType::Bool
            )
        {
            return None;
        }
        if proof.scalar_universe().is_empty() {
            return None;
        }
        if proof
            .scalar_universe()
            .iter()
            .any(|value| value.slot_type() != proof.scalar_type())
        {
            return None;
        }
        Some(proof)
    }

    /// Validated proof that every string/model-value element of an `IntArray`
    /// is drawn from a finite, homogeneous scalar universe, making the array
    /// safe for *default* flat BFS despite non-`i64` elements (G2-extension).
    ///
    /// Returns `Some` only when: the kind is `IntArray` carrying an
    /// `element_range_proof`; the array is not a pure-bool array; the proof's
    /// `scalar_type` is `String` or `ModelValue` with a non-empty, homogeneous
    /// universe; and *every* element type equals the proof's `scalar_type`
    /// (a fully homogeneous string/model-value array such as `[Node ->
    /// Color]`). Anything else fails closed.
    ///
    /// Soundness mirrors [`Self::fixed_scalar_range_primary_proof`]: the
    /// interned-`NameId` slot encoding (`u32 as i64`) is fixed-width,
    /// non-negative and injective over all strings/model-values, and the
    /// `TypeOK` proof certifies the element universe is closed under every
    /// successor, so no transition can write a colliding plain integer.
    #[must_use]
    fn int_array_element_range_proof(&self) -> Option<&FixedScalarRangeProof> {
        let VarLayoutKind::IntArray {
            elements_are_bool,
            element_types,
            element_range_proof,
            ..
        } = self
        else {
            return None;
        };
        if *elements_are_bool {
            return None;
        }
        let proof = element_range_proof.as_ref()?;
        if !matches!(proof.scalar_type(), SlotType::String | SlotType::ModelValue) {
            return None;
        }
        if proof.scalar_universe().is_empty()
            || proof
                .scalar_universe()
                .iter()
                .any(|value| value.slot_type() != proof.scalar_type())
        {
            return None;
        }
        let element_types = element_types.as_ref()?;
        if element_types.is_empty() || element_types.iter().any(|ty| *ty != proof.scalar_type()) {
            return None;
        }
        Some(proof)
    }

    /// Whether every `Record` field is flat-admissible for *default* flat BFS:
    /// each field is either plain-`i64` (Int/Bool) or a string/model-value field
    /// covered by a validated finite-universe proof in `field_range_proofs`
    /// (G2-extension). Admits records like EWD998 `token \in [pos: Node, q: Int,
    /// color: Color]` whose only non-`i64` field (`color`) is proven finite.
    ///
    /// Fails closed if the proof vector is present but not parallel to
    /// `field_types`, or if any non-`i64` field lacks a homogeneous
    /// string/model-value universe proof whose `scalar_type` matches the field.
    /// Soundness is identical to [`Self::int_array_element_range_proof`].
    #[must_use]
    fn record_fields_flat_admissible(&self) -> bool {
        let VarLayoutKind::Record {
            field_types,
            field_range_proofs,
            ..
        } = self
        else {
            return false;
        };
        if let Some(proofs) = field_range_proofs.as_ref() {
            if proofs.len() != field_types.len() {
                return false;
            }
        }
        field_types.iter().enumerate().all(|(i, ty)| {
            if slot_type_is_plain_i64(*ty) {
                return true;
            }
            let Some(Some(proof)) = field_range_proofs.as_ref().map(|p| &p[i]) else {
                return false;
            };
            matches!(proof.scalar_type(), SlotType::String | SlotType::ModelValue)
                && *ty == proof.scalar_type()
                && !proof.scalar_universe().is_empty()
                && proof
                    .scalar_universe()
                    .iter()
                    .all(|value| value.slot_type() == proof.scalar_type())
        })
    }

    /// Per-variable flat-primary admissibility: `true` when this variable's
    /// layout can be encoded losslessly as its own contiguous flat `[i64]`
    /// slots (scalars, proven `FixedScalar`, `IntArray`, proven records/keyed
    /// functions, capacity-proven recursive sequences). This is the individual
    /// counterpart of [`StateLayout::supports_flat_primary`]: the whole-state
    /// gate is the conjunction of this predicate over every variable. The
    /// hybrid per-action dispatch path (`hybrid_flat_view`) uses it to select
    /// the flat-admissible variable subset while leaving un-flattenable vars
    /// (`Dynamic`, `Bitmask`, un-proven string/model-value scalars) compound.
    #[must_use]
    pub(crate) fn supports_flat_primary(&self) -> bool {
        match self {
            // Top-level string/model-value scalar slots are init-sampled only.
            // Keep them on ArrayState storage unless a proven aggregate layout
            // supplies a stronger type contract.
            VarLayoutKind::ScalarString | VarLayoutKind::ScalarModelValue => false,
            // A top-level scalar string/model-value enum whose finite universe
            // is proven total (G2). The encoding is identical to ScalarString,
            // so promoting it to primary storage cannot alias or truncate; the
            // proof only confirms every reachable value stays in the universe.
            VarLayoutKind::FixedScalar { .. } => self.fixed_scalar_var_proof().is_some(),
            // Simple records are inferred from sampled values, so a string/
            // model-value field that lacks a proof can later be assigned an
            // integer in the same slot shape (for example Apalache
            // Variant("None", UNIT) -> Variant("Some", 1)) and alias. A plain
            // i64 field is always primary-safe; a string/model-value field is
            // primary-safe exactly when `record_fields_flat_admissible` finds a
            // validated finite-universe `TypeOK` proof for it. That proof is the
            // identical per-field artifact (`unique_fixed_scalar_var_proof`)
            // that upgrades a top-level `FixedScalar` var to primary above:
            // homogeneous, non-empty, with a `scalar_type` matching the field,
            // certifying the universe is closed under every successor. The
            // interned-`NameId` encoding is fixed-width, non-negative and
            // injective, so a proven field can never alias a plain integer. The
            // heterogeneous Variant case fails closed because it cannot earn a
            // homogeneous proof. This mirrors the auto-admission Record arm, so
            // the two gates stay symmetric.
            VarLayoutKind::Record { .. } => self.record_fields_flat_admissible(),
            // A legacy fixed string/model-value keyed function only proves the
            // sampled range values. Either a tagged scalar/set proof or a
            // validated `FixedScalar(String|ModelValue)` range proof upgrades that
            // sampled shape into an injective, canonical one-slot range contract;
            // malformed or scalar-encoding-unsafe proof metadata still fails
            // closed through `tagged_scalar_set_range_primary_proof()` /
            // `fixed_scalar_range_primary_proof()`.
            //
            // A plain-i64 (Int/Bool) range is intentionally NOT admitted to
            // flat-PRIMARY storage here, even though every value slot is a plain
            // integer: unlike `TupleKeyedArray` (whose static tuple domain makes
            // the range structurally scalar-only) a `StringKeyedArray` range is
            // inferred from sampled values, so a slot sampled as Int/Bool can be
            // a `TaggedScalarOrSet` slot or a heterogeneous range that a later
            // successor overwrites with a finite set (e.g. DijkstraMutex `temp`,
            // whose `[Proc -> ...]` range holds a model value in some states and
            // `Proc \ {self}` — a set — in others). Such ranges must stay on the
            // strict flat-frontier admission path, never flat-primary, or the
            // native-fused BFS silently collapses distinct set-valued states into
            // one fingerprint and undercounts. Fail closed.
            VarLayoutKind::StringKeyedArray { .. } => {
                self.tagged_scalar_set_range_primary_proof().is_some()
                    || self.fixed_scalar_range_primary_proof().is_some()
            }
            // A tuple/cross-product-keyed function over a STATIC, fully
            // enumerated domain. The key set is fixed across all reachable
            // successors (no key can appear or vanish), so the canonical sorted
            // slot mapping is stable and collision-free. Under `ScalarSlots` the
            // range values are stored as plain i64 scalar slots, identical to
            // `IntArray`; this is only primary-safe when every range slot is a
            // plain i64 (Int/Bool), matching the record-field rule — a
            // string/model-value slot could later be overwritten by an integer of
            // the same width and alias.
            //
            // Under `TaggedScalarUnion` each slot instead stores a non-negative
            // INJECTIVE universe index (a canonical, fixed-width, collision-free
            // encoding whose closure is proven by a checked whole-var `TypeOK`
            // invariant), so it is flat-primary safe exactly like the
            // `StringKeyedArray` tagged/fixed range proofs — the sampled
            // `value_types` are not the slot encoding in this case.
            VarLayoutKind::TupleKeyedArray {
                value_types,
                range_encoding,
                ..
            } => match range_encoding {
                TupleKeyedArrayRangeEncoding::ScalarSlots => {
                    value_types.iter().all(|ty| slot_type_is_plain_i64(*ty))
                }
                // WP-09/Part A (campaign): the union-index range is admitted only
                // under the same `TY_TAGGED_SCALAR_UNION` opt-in that authorizes
                // the carrier. Upstream relies on the collectors never BUILDING
                // this encoding with the gate off; keeping the admission-side
                // check as well makes a directly-constructed layout fail closed
                // too (byte-identical default surface).
                TupleKeyedArrayRangeEncoding::TaggedScalarUnion(_) => self
                    .tuple_keyed_tagged_scalar_union_range_primary_proof()
                    .is_some(),
                // A homogeneous proven-finite model-value/string/bool range: the
                // interned-`NameId` slot encoding is fixed-width, non-negative and
                // injective over all strings/model-values, and the `TypeOK` proof
                // certifies the universe is closed (non-int) under every
                // successor, so no transition can write a colliding plain integer.
                // Mirror `fixed_scalar_range_primary_proof`: admit only a
                // non-empty homogeneous String/ModelValue/Bool universe whose
                // scalar type matches every sampled value slot.
                TupleKeyedArrayRangeEncoding::FixedScalar(proof) => {
                    matches!(
                        proof.scalar_type(),
                        SlotType::String | SlotType::ModelValue | SlotType::Bool
                    ) && !proof.scalar_universe().is_empty()
                        && proof
                            .scalar_universe()
                            .iter()
                            .all(|value| value.slot_type() == proof.scalar_type())
                        && value_types.iter().all(|ty| *ty == proof.scalar_type())
                }
            },
            VarLayoutKind::Recursive { layout } => layout.supports_flat_primary(),
            VarLayoutKind::Bitmask { .. } | VarLayoutKind::Dynamic => false,
            VarLayoutKind::Scalar | VarLayoutKind::ScalarBool | VarLayoutKind::IntArray { .. } => {
                true
            }
        }
    }
}

/// Fixed-offset mapping from variable indices to i64 slots.
///
/// Created once from an initial state (or spec metadata) and shared across
/// all `FlatState` instances for a given model-checking run.
#[derive(Debug, Clone)]
pub(crate) struct StateLayout {
    /// Per-variable layouts in VarIndex order.
    vars: Vec<VarLayout>,
    /// Total number of i64 slots in the flat buffer.
    total_slots: usize,
}

impl StateLayout {
    /// Build a layout from variable descriptors.
    ///
    /// Computes offsets by packing variables contiguously in index order.
    #[must_use]
    pub(crate) fn new(registry: &VarRegistry, kinds: Vec<VarLayoutKind>) -> Self {
        assert_eq!(
            registry.len(),
            kinds.len(),
            "StateLayout::new: registry has {} vars but {} kinds provided",
            registry.len(),
            kinds.len()
        );

        let mut vars = Vec::with_capacity(kinds.len());
        let mut offset = 0;
        for (i, kind) in kinds.into_iter().enumerate() {
            let idx = crate::var_index::VarIndex::new(i);
            let name = Arc::from(registry.name(idx));
            let slot_count = kind.slot_count();
            vars.push(VarLayout {
                name,
                offset,
                slot_count,
                kind,
            });
            offset += slot_count;
        }

        StateLayout {
            vars,
            total_slots: offset,
        }
    }

    /// Total number of i64 slots in the flat buffer.
    #[must_use]
    pub(crate) fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Number of state variables.
    #[must_use]
    pub(crate) fn var_count(&self) -> usize {
        self.vars.len()
    }

    /// Get the layout for a specific variable by index.
    /// Scaffolding for JIT V2 flat state pipeline (#3986).
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn var_layout(&self, idx: usize) -> Option<&VarLayout> {
        self.vars.get(idx)
    }

    /// Iterate over all variable layouts.
    /// Scaffolding for JIT V2 flat state pipeline (#3986).
    #[allow(dead_code)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &VarLayout> {
        self.vars.iter()
    }

    /// Swap variable `var_idx`'s kind in place, preserving its slot span.
    ///
    /// Returns `false` (no-op) when the variable is absent or the replacement's
    /// `slot_count` differs from the current one — swapping a different-width
    /// kind would invalidate every following variable's precomputed `offset`.
    /// Used by the `TaggedScalarUnion` whole-variable override, which only ever
    /// replaces a one-slot scalar kind with a one-slot union kind.
    pub(crate) fn replace_var_kind_same_slots(
        &mut self,
        var_idx: usize,
        new_kind: VarLayoutKind,
    ) -> bool {
        let Some(var) = self.vars.get_mut(var_idx) else {
            return false;
        };
        if var.slot_count != new_kind.slot_count() {
            return false;
        }
        var.kind = new_kind;
        true
    }

    /// Replace a variable's kind with a possibly DIFFERENT-width kind and
    /// recompute every following variable's slot offset + the total slot count.
    ///
    /// Unlike [`Self::replace_var_kind_same_slots`], this admits a width change
    /// (e.g. promoting a 1-slot `ScalarModelValue` to a `1 + max_payload`-slot
    /// `Recursive { TaggedUnion }` sum-type var), so the layout stays internally
    /// consistent (offsets/total). Returns `false` if `var_idx` is out of range.
    pub(crate) fn replace_var_kind_recompute(
        &mut self,
        var_idx: usize,
        new_kind: VarLayoutKind,
    ) -> bool {
        if self.vars.get(var_idx).is_none() {
            return false;
        }
        self.vars[var_idx].kind = new_kind;
        let mut offset = 0;
        for var in &mut self.vars {
            var.slot_count = var.kind.slot_count();
            var.offset = offset;
            offset += var.slot_count;
        }
        self.total_slots = offset;
        true
    }

    /// True when every variable is `Scalar` (the buffer is 1:1 with ArrayState).
    #[must_use]
    pub(crate) fn is_all_scalar(&self) -> bool {
        self.vars.iter().all(|v| {
            matches!(
                v.kind,
                VarLayoutKind::Scalar
                    | VarLayoutKind::ScalarBool
                    | VarLayoutKind::ScalarString
                    | VarLayoutKind::ScalarModelValue
                    | VarLayoutKind::FixedScalar { .. }
            )
        })
    }

    /// True when every variable is either `Scalar`/`ScalarBool` or `Dynamic`.
    /// In this case the flat buffer has the same number of slots as variables.
    #[must_use]
    pub(crate) fn is_trivial(&self) -> bool {
        self.vars.iter().all(|v| {
            matches!(
                v.kind,
                VarLayoutKind::Scalar
                    | VarLayoutKind::ScalarBool
                    | VarLayoutKind::ScalarString
                    | VarLayoutKind::ScalarModelValue
                    | VarLayoutKind::FixedScalar { .. }
                    | VarLayoutKind::Dynamic
            )
        })
    }

    /// True when at least one variable has `Dynamic` layout.
    ///
    /// When true, the flat buffer alone is not sufficient for exact state
    /// reconstruction — the original `ArrayState` must be consulted for
    /// dynamic variables. This determines whether the fast-path (flat-only)
    /// or fallback-path (flat + ArrayState) is needed.
    ///
    /// Part of #3986.
    #[must_use]
    pub(crate) fn has_dynamic_vars(&self) -> bool {
        self.vars
            .iter()
            .any(|v| matches!(v.kind, VarLayoutKind::Dynamic))
    }

    /// True when every variable is fully flattenable (no Dynamic vars).
    ///
    /// When true, the flat buffer is a complete representation of the state
    /// and no fallback to ArrayState is needed. This is the condition for
    /// enabling the pure flat-state BFS path.
    ///
    /// Part of #3986.
    #[must_use]
    pub(crate) fn is_fully_flat(&self) -> bool {
        !self.has_dynamic_vars()
    }

    /// True when the flat buffer is safe as the primary BFS representation.
    ///
    /// `is_fully_flat()` only means every current variable has a fixed slot
    /// layout. Recursive sequences are primary-safe only when source-level
    /// invariants prove both the sequence capacity and the element layout.
    /// String/model-value keyed functions are primary-safe only with a
    /// validated tagged scalar/set range proof; legacy scalar slots stay
    /// fail-closed because sampled scalar ranges can later produce sets.
    #[must_use]
    pub(crate) fn supports_flat_primary(&self) -> bool {
        self.is_fully_flat() && self.vars.iter().all(|var| var.kind.supports_flat_primary())
    }

    /// Fail-closed veto (#43 extension): demote any variable in `vetoed` whose
    /// inferred layout is a **single scalar slot** kind to [`VarLayoutKind::Dynamic`].
    ///
    /// `vetoed` is the set of state-var indices (in [`VarIndex`] order) whose
    /// Init/Next writers can assign a SET (or any other non-scalar) value — see
    /// `vars_with_nonscalar_writers`. The #43 writer veto only gated *TypeOK*-
    /// derived `FixedScalar` type-proofs; this method closes the parallel hole
    /// where a slot was admitted to flat-primary **by init-sampling alone**, with
    /// no type-proof at all (e.g. `x = 0` in Init makes `x` a plain `Scalar`,
    /// even though a successor writes `x' = {1, 2}`). Encoding that set into the
    /// same flat i64 scalar slot aliases the set-valued state against a distinct
    /// scalar state in the flat fingerprint; the visited-set dedup then drops one
    /// of them and the BFS silently undercounts — a missed-violation soundness bug.
    ///
    /// Only the kinds that store a *scalar value in a scalar i64 slot* are
    /// demoted: `Scalar`, `ScalarBool`, `ScalarString`, `ScalarModelValue`,
    /// `FixedScalar`, and `IntArray` (whose elements are scalar slots). Genuinely
    /// non-scalar layouts (`Record`, `StringKeyedArray`, `TupleKeyedArray`,
    /// `Bitmask`, `Recursive`) are NOT demoted: those are the value's actual
    /// representation (a set-bearing var never infers a scalar-slot layout for the
    /// set itself — only this collision path does), they carry their own
    /// roundtrip-verified encoding, and demoting them would over-reject sound
    /// flat-primary specs. This matches the writer scanner's classification:
    /// a genuine scalar var (`x' = x + 1`), scalar-range `IntArray`
    /// (`[v EXCEPT ![i] = scalar]`), or scalar-field `Record` is never in
    /// `vetoed`, so it is never demoted (no over-rejection).
    ///
    /// Returns the indices actually demoted (for diagnostics). Idempotent and
    /// recomputes `total_slots` so the buffer stays consistent (each demoted var
    /// keeps its 1-slot `Dynamic` placeholder; multi-slot scalar arrays shrink to
    /// a single Dynamic slot).
    pub(crate) fn veto_flat_primary_scalar_slot_vars(
        &mut self,
        vetoed: &std::collections::BTreeSet<usize>,
    ) -> Vec<usize> {
        let mut demoted = Vec::new();
        if vetoed.is_empty() {
            return demoted;
        }
        for (idx, var) in self.vars.iter_mut().enumerate() {
            if !vetoed.contains(&idx) {
                continue;
            }
            let is_scalar_slot_kind = matches!(
                var.kind,
                VarLayoutKind::Scalar
                    | VarLayoutKind::ScalarBool
                    | VarLayoutKind::ScalarString
                    | VarLayoutKind::ScalarModelValue
                    | VarLayoutKind::FixedScalar { .. }
                    | VarLayoutKind::IntArray { .. }
            );
            if is_scalar_slot_kind {
                var.kind = VarLayoutKind::Dynamic;
                demoted.push(idx);
            }
        }
        if !demoted.is_empty() {
            // Recompute slot offsets/total: a demoted multi-slot `IntArray`
            // collapses to a single `Dynamic` slot, so offsets shift.
            let mut offset = 0;
            for var in &mut self.vars {
                var.slot_count = var.kind.slot_count();
                var.offset = offset;
                offset += var.slot_count;
            }
            self.total_slots = offset;
        }
        demoted
    }

    /// Human-readable reasons why this layout cannot be used as primary flat
    /// state storage.
    #[must_use]
    pub(crate) fn flat_primary_blockers(&self) -> Vec<String> {
        let mut blockers = Vec::new();
        if !self.is_fully_flat() {
            blockers.push("layout has dynamic variables".to_string());
        }
        for (idx, var) in self.vars.iter().enumerate() {
            if !var.kind.supports_flat_primary() {
                blockers.push(format!(
                    "var {idx} `{}` at slot {} ({} slots): {:?}",
                    var.name, var.offset, var.slot_count, var.kind
                ));
            }
        }
        blockers
    }

    /// True when the layout is safe for default flat-BFS auto-admission.
    ///
    /// This is intentionally narrower than `is_fully_flat()`: fully-flat only
    /// says sampled values fit a fixed slot layout. It does not prove that
    /// future successors will keep the same shape. Fixed model-value keyed
    /// function layouts are especially risky because init states often use a
    /// scalar sentinel while later transitions store sets in the range.
    #[must_use]
    pub(crate) fn supports_flat_bfs_auto_admission(&self) -> bool {
        self.is_fully_flat()
            && self
                .vars
                .iter()
                .all(|var| var_layout_kind_supports_flat_bfs_auto_admission(&var.kind))
    }

    /// True when force-enabling flat BFS (`use_flat_state=Some(true)`) is sound
    /// for this layout.
    ///
    /// Force-enable intentionally bypasses the conservative auto-admission
    /// heuristic so callers can opt scalar/aggregate specs into the flat path.
    /// The one shape it must still refuse is a sequence whose flat capacity was
    /// only sampled (`SequenceBoundEvidence::Observed`) or whose element layout
    /// is unproven: those are lossy on growth and silently corrupt dedup,
    /// inflating the reported state count (a recursive `Dom -> Seq` variable
    /// whose sampled capacity is exceeded by a later successor).
    ///
    /// Everything else stays admitted, including the fixed-domain model-value
    /// `StringKeyedArray` sandbox, which carries no sequence and has an explicit
    /// ArrayState reconstruction path. Roundtrip verification is still required
    /// separately by the caller; this predicate only adds the growth-safety
    /// floor.
    #[must_use]
    pub(crate) fn supports_forced_flat_bfs(&self) -> bool {
        self.vars.iter().all(|var| match &var.kind {
            VarLayoutKind::Recursive { layout } => !layout.has_flat_primary_unsafe_sequence(),
            _ => true,
        })
    }

    /// True when at least one variable carries a tagged scalar/set range proof.
    /// This proof is what distinguishes the one-slot tagged encoding from
    /// legacy scalar-only fixed-function slots.
    #[must_use]
    pub(crate) fn has_model_value_keyed_tagged_scalar_set_range(&self) -> bool {
        self.vars
            .iter()
            .any(|var| var.kind.tagged_scalar_set_range_proof().is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::var_index::VarRegistry;

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_state_layout_all_scalar() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let kinds = vec![
            VarLayoutKind::Scalar,
            VarLayoutKind::Scalar,
            VarLayoutKind::Scalar,
        ];
        let layout = StateLayout::new(&registry, kinds);

        assert_eq!(layout.var_count(), 3);
        assert_eq!(layout.total_slots(), 3);
        assert!(layout.is_all_scalar());
        assert!(layout.is_trivial());

        // Check offsets
        assert_eq!(layout.var_layout(0).unwrap().offset, 0);
        assert_eq!(layout.var_layout(1).unwrap().offset, 1);
        assert_eq!(layout.var_layout(2).unwrap().offset, 2);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_top_level_named_scalar_layouts_fail_closed_for_flat_primary() {
        let registry = VarRegistry::from_names(["owner", "label"]);
        let layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue, VarLayoutKind::ScalarString],
        );

        assert!(layout.is_fully_flat());
        assert!(
            !layout.supports_flat_primary(),
            "init-sampled named scalar slots can collide with later integer successors"
        );
        assert!(
            !layout.supports_flat_bfs_auto_admission(),
            "native flat-frontier admission must not consume unproved named scalar slots"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_record_model_value_field_flat_primary_requires_finite_universe_proof() {
        // A record like EWD998 `token \in [pos: Node, color: Color]`: the
        // `color` field is a model value whose `TypeOK` universe is finite and
        // homogeneous. The proof is the identical per-field artifact that
        // upgrades a top-level `FixedScalar` var to primary, so the record is
        // primary-safe (and the two gates stay symmetric).
        let color_proof = FixedScalarRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("red")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("green")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("blue")),
            ],
            Arc::from("TokenTypeOK"),
        )
        .unwrap();
        let registry = VarRegistry::from_names(["token"]);
        let proven = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Record {
                field_names: vec![Arc::from("pos"), Arc::from("color")],
                field_is_bool: vec![false, false],
                field_types: vec![SlotType::Int, SlotType::ModelValue],
                field_range_proofs: Some(vec![None, Some(color_proof)]),
            }],
        );
        assert!(proven.is_fully_flat());
        assert!(
            proven.supports_flat_primary(),
            "a model-value field with a finite-universe TypeOK proof is primary-safe"
        );
        assert!(
            proven.supports_flat_bfs_auto_admission(),
            "the same proof admits the record to flat BFS — the two gates are symmetric"
        );

        // Without the proof the model-value field could later alias a plain
        // integer in the same slot (Apalache Variant("None") -> Variant("Some",
        // 1)), so both gates must fail closed.
        let unproven = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Record {
                field_names: vec![Arc::from("pos"), Arc::from("color")],
                field_is_bool: vec![false, false],
                field_types: vec![SlotType::Int, SlotType::ModelValue],
                field_range_proofs: None,
            }],
        );
        assert!(
            !unproven.supports_flat_primary(),
            "an unproven model-value field can alias a later integer successor"
        );
        assert!(
            !unproven.supports_flat_bfs_auto_admission(),
            "flat BFS admission must also reject the unproven model-value field"
        );

        // An all-i64 record stays primary-safe with no proof (baseline).
        let plain = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Record {
                field_names: vec![Arc::from("pos"), Arc::from("rdy")],
                field_is_bool: vec![false, true],
                field_types: vec![SlotType::Int, SlotType::Bool],
                field_range_proofs: None,
            }],
        );
        assert!(
            plain.supports_flat_primary(),
            "a record of plain i64/bool fields is unconditionally primary-safe"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bitmask_layout_is_fully_flat_but_rejects_flat_primary() {
        let registry = VarRegistry::from_names(["flags"]);
        let layout = StateLayout::new(&registry, vec![VarLayoutKind::Bitmask { universe_size: 5 }]);

        assert!(layout.is_fully_flat());
        assert_eq!(layout.total_slots(), 1);
        assert!(
            !layout.supports_flat_primary(),
            "top-level bitmask slots are fully flat but lack a flat-primary safety proof"
        );
        assert!(
            !layout.supports_flat_bfs_auto_admission(),
            "flat BFS auto-admission must inherit the same fail-closed bitmask guard"
        );
    }

    /// Track B increment 1: a top-level record-set var packed as a fixed-width
    /// bitmask i64 IS admitted to flat-primary when (1) its universe is
    /// PROVEN closed and (2) every universe record is natively representable
    /// (scalar fields only). The native ABI now carries a faithful
    /// `CompoundLayout::RecordSetBitmask` (NOT `Dynamic`), the byte-exact
    /// `set_ops` membership/union/diff lowering fires off
    /// `AggregateShape::RecordSetBitmask`, and every other native context
    /// fail-closes — so the historical `\cup` `IntToPtr` rc=139 trap is
    /// structurally closed.
    ///
    /// A SAMPLED universe still fails the soundness check (a successor write
    /// outside the sampled universe would corrupt the slot), and a universe with
    /// a non-scalar field also fails it (the carrier cannot represent it, so the
    /// bridge would fall back to `Dynamic`).
    ///
    /// This asserts the env-independent soundness predicate
    /// `record_set_bitmask_flat_primary_sound`. The full
    /// `supports_flat_primary` additionally requires the
    /// `TY_RECORD_SET_NATIVE=1` opt-in gate (default OFF), which is tested at the
    /// integration level rather than here (the env flag is process-global and
    /// would race the parallel unit-test runner).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_record_set_bitmask_layout_flat_primary_soundness() {
        // This test asserts on the process-global env var TY_RECORD_SET_NATIVE (default OFF).
        // Hold the SHARED process-wide env lock so the read below cannot race the
        // `TY_RECORD_SET_NATIVE` set/unset performed by the `trust_cg_dispatch` test module,
        // which serializes on the same `crate::process_env_lock()` mutex.
        let _env_guard = crate::process_env_lock();

        let rec = |ty: &str| {
            crate::Value::Record(tla_value::value::RecordValue::from_sorted_str_entries(
                vec![(Arc::from("type"), crate::Value::String(Rp::from(ty)))],
            ))
        };
        let mut universe = vec![rec("Commit"), rec("Abort")];
        universe.sort();
        universe.dedup();

        // A `ProvenClosed` record-set bitmask over scalar-field records is
        // sound for flat-PRIMARY native dispatch (carrier + lowering wired).
        let proven = FlatValueLayout::RecordSetBitmask {
            universe: universe.clone(),
            universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                invariant: Arc::from("TypeOK"),
            },
        };
        assert!(
            proven.record_set_bitmask_flat_primary_sound(),
            "a proven-closed, natively-representable record-set-bitmask slot is flat-primary-sound (Track B increment 1)"
        );
        // With the opt-in gate OFF (default), `supports_flat_primary` stays
        // fail-closed so every existing corpus spec is byte-identical.
        assert!(
            !record_set_native_flat_primary_enabled(),
            "TY_RECORD_SET_NATIVE must default OFF in the unit-test environment"
        );
        assert!(
            !proven.supports_flat_primary(),
            "with the opt-in gate OFF, even a sound record-set-bitmask slot is not admitted to flat-primary (byte-identical default)"
        );

        // A SAMPLED universe must fail the soundness check: an out-of-universe
        // successor write would silently corrupt the fixed-width slot.
        let sampled = FlatValueLayout::RecordSetBitmask {
            universe: universe.clone(),
            universe_closure: SetBitmaskUniverseClosure::Sampled,
        };
        assert!(
            !sampled.record_set_bitmask_flat_primary_sound(),
            "a sampled record-set-bitmask slot must fail the flat-primary soundness check"
        );

        // A SET-OF-SCALARS field (e.g. AllocatorImplementation's
        // `rsrc : SUBSET Resources` message field) IS representable: the bridge
        // folds it to a single deterministic scalar carrier slot, every native
        // record-set op fails closed on a set-shaped runtime element before the
        // carried constant is compared, and flat-state STORAGE bit assignment
        // goes through the interpreter's full-Value path — so the fold is not
        // load-bearing (see record_set_bitmask_field_native_representable).
        let scalar_set_field_rec = crate::Value::Record(
            tla_value::value::RecordValue::from_sorted_str_entries(vec![(
                Arc::from("payload"),
                crate::Value::Set(Rp::new(tla_value::value::SortedSet::from_sorted_vec(vec![
                    crate::Value::SmallInt(1),
                ]))),
            )]),
        );
        let scalar_set_field = FlatValueLayout::RecordSetBitmask {
            universe: vec![scalar_set_field_rec],
            universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                invariant: Arc::from("TypeOK"),
            },
        };
        assert!(
            scalar_set_field.record_set_bitmask_flat_primary_sound(),
            "a record-set-bitmask whose records carry a set-of-scalars field is natively \
             representable (the bridge folds it; the fold is not load-bearing)"
        );

        // A set field with NON-scalar elements (here a set-of-sets) remains
        // unrepresentable, so the soundness check must still fail even when
        // proven-closed.
        let nested_set_field_rec = crate::Value::Record(
            tla_value::value::RecordValue::from_sorted_str_entries(vec![(
                Arc::from("payload"),
                crate::Value::Set(Rp::new(tla_value::value::SortedSet::from_sorted_vec(vec![
                    crate::Value::Set(Rp::new(tla_value::value::SortedSet::from_sorted_vec(vec![
                        crate::Value::SmallInt(1),
                    ]))),
                ]))),
            )]),
        );
        let non_representable = FlatValueLayout::RecordSetBitmask {
            universe: vec![nested_set_field_rec],
            universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                invariant: Arc::from("TypeOK"),
            },
        };
        assert!(
            !non_representable.record_set_bitmask_flat_primary_sound(),
            "a record-set-bitmask whose records have a set field with NON-scalar elements is \
             not natively representable"
        );

        // The scalar `SetBitmask` analogue MUST stay flat-primary (unaffected by
        // the gate).
        let scalar_set = FlatValueLayout::SetBitmask {
            universe: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
            universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                invariant: Arc::from("TypeOK"),
            },
        };
        assert!(
            scalar_set.supports_flat_primary(),
            "scalar SetBitmask stays flat-primary"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_state_layout_mixed() {
        let registry = VarRegistry::from_names(["pc", "network", "flags"]);
        let kinds = vec![
            VarLayoutKind::Scalar,
            VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 0,
                len: 3,
                elements_are_bool: false,
                element_types: None,
            },
            VarLayoutKind::Bitmask { universe_size: 5 },
        ];
        let layout = StateLayout::new(&registry, kinds);

        assert_eq!(layout.var_count(), 3);
        assert_eq!(layout.total_slots(), 5); // 1 + 3 + 1
        assert!(!layout.is_all_scalar());
        assert!(!layout.is_trivial());

        // pc at offset 0, 1 slot
        let pc = layout.var_layout(0).unwrap();
        assert_eq!(pc.offset, 0);
        assert_eq!(pc.slot_count, 1);

        // network at offset 1, 3 slots
        let net = layout.var_layout(1).unwrap();
        assert_eq!(net.offset, 1);
        assert_eq!(net.slot_count, 3);

        // flags at offset 4, 1 slot
        let flags = layout.var_layout(2).unwrap();
        assert_eq!(flags.offset, 4);
        assert_eq!(flags.slot_count, 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_var_layout_kind_slot_count() {
        assert_eq!(VarLayoutKind::Scalar.slot_count(), 1);
        assert_eq!(VarLayoutKind::ScalarBool.slot_count(), 1);
        assert_eq!(
            VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 1,
                len: 5,
                elements_are_bool: false,
                element_types: None,
            }
            .slot_count(),
            5
        );
        assert_eq!(
            VarLayoutKind::Record {
                field_range_proofs: None,
                field_names: vec![Arc::from("a"), Arc::from("b")],
                field_is_bool: vec![false, false],
                field_types: vec![SlotType::Int, SlotType::Int],
            }
            .slot_count(),
            2
        );
        assert_eq!(
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 1,
                    len: 2,
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }
            .slot_count(),
            2
        );
        assert_eq!(VarLayoutKind::Bitmask { universe_size: 8 }.slot_count(), 1);
        assert_eq!(VarLayoutKind::Dynamic.slot_count(), 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_state_layout_with_dynamic() {
        let registry = VarRegistry::from_names(["a", "b"]);
        let kinds = vec![VarLayoutKind::Scalar, VarLayoutKind::Dynamic];
        let layout = StateLayout::new(&registry, kinds);

        assert!(!layout.is_all_scalar());
        assert!(layout.is_trivial());
        assert_eq!(layout.total_slots(), 2);
        assert!(layout.has_dynamic_vars());
        assert!(!layout.is_fully_flat());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_state_layout_fully_flat() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let kinds = vec![
            VarLayoutKind::Scalar,
            VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 0,
                len: 3,
                elements_are_bool: false,
                element_types: None,
            },
            VarLayoutKind::ScalarBool,
        ];
        let layout = StateLayout::new(&registry, kinds);

        assert!(!layout.has_dynamic_vars());
        assert!(layout.is_fully_flat());
        assert!(!layout.is_all_scalar());
        assert!(!layout.is_trivial());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_state_layout_has_dynamic_vars_all_scalar() {
        let registry = VarRegistry::from_names(["a", "b"]);
        let kinds = vec![VarLayoutKind::Scalar, VarLayoutKind::ScalarBool];
        let layout = StateLayout::new(&registry, kinds);

        assert!(!layout.has_dynamic_vars());
        assert!(layout.is_fully_flat());
        assert!(layout.is_all_scalar());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_primary_rejects_recursive_sequences_even_when_bound_proven() {
        let registry = VarRegistry::from_names(["network"]);
        let observed = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::Observed,
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );
        assert!(observed.is_fully_flat());
        assert!(!observed.supports_flat_primary());

        let proven = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedNetwork"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );
        assert!(proven.is_fully_flat());
        assert!(proven.vars[0].kind.slot_count() > 0);
        assert!(!proven.supports_flat_primary());

        let fixed_domain_type_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::FixedDomainTypeLayout {
                        invariant: Arc::from("TypeOK"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );
        assert!(fixed_domain_type_layout.is_fully_flat());
        assert!(fixed_domain_type_layout.supports_flat_primary());

        let proven_with_element_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                        invariant: Arc::from("BoundedNetwork"),
                        element_invariant: Arc::from("TypeOK"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Record {
                        field_names: vec![Arc::from("clock"), Arc::from("type")],
                        field_layouts: vec![
                            FlatValueLayout::Scalar(SlotType::Int),
                            FlatValueLayout::Scalar(SlotType::String),
                        ],
                    }),
                },
            }],
        );
        assert!(proven_with_element_layout.is_fully_flat());
        assert!(proven_with_element_layout.supports_flat_primary());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_supports_forced_flat_bfs_rejects_sampled_capacity_sequence() {
        let registry = VarRegistry::from_names(["network"]);

        // A recursive `Dom -> Seq` shape: Recursive { Function { domain ->
        // Sequence { Observed } } }. The sampled (Observed) capacity is lossy
        // once a successor grows past the inferred max_len, so force-enable must
        // refuse it (regression for the force-flat dedup over-count).
        let observed = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                    value_layout: Box::new(FlatValueLayout::Sequence {
                        bound: SequenceBoundEvidence::Observed,
                        max_len: 0,
                        element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                    }),
                },
            }],
        );
        assert!(observed.is_fully_flat());
        assert!(
            !observed.supports_forced_flat_bfs(),
            "force-enable must refuse a sampled-capacity (Observed) sequence"
        );

        // A capacity proof alone does not prove the element layout is
        // growth-safe, so it stays refused under force.
        let proven_capacity_only = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedNetwork"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );
        assert!(
            !proven_capacity_only.supports_forced_flat_bfs(),
            "capacity proof alone does not prove the element layout is growth-safe"
        );

        // A fully proven capacity + element layout is admitted under force.
        let fully_proven = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::FixedDomainTypeLayout {
                        invariant: Arc::from("TypeOK"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );
        assert!(
            fully_proven.supports_forced_flat_bfs(),
            "a fully proven sequence layout stays admissible under force"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_supports_forced_flat_bfs_admits_sequence_free_layouts() {
        // Scalars, int arrays, and the fixed-domain model-value
        // StringKeyedArray sandbox carry no sequence, so force-enable still
        // admits them (preserves the intentional sandbox asserted by
        // run_prepare_tests 1124/2429).
        let registry = VarRegistry::from_names(["x", "arr", "tcb"]);
        let layout = StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::Scalar,
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 0,
                    len: 3,
                    elements_are_bool: false,
                    element_types: None,
                },
                VarLayoutKind::StringKeyedArray {
                    domain_keys: vec![Arc::from("p1"), Arc::from("p2")],
                    domain_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                    value_types: vec![SlotType::Int, SlotType::Int],
                    range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
                },
            ],
        );
        assert!(
            layout.supports_forced_flat_bfs(),
            "sequence-free layouts (incl. the model-value sandbox) stay admissible under force"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_untagged_scalar_set_slot_collision_requires_tagged_encoding() {
        let scalar_payload = 7;
        let proc_subset_mask = 0b0111;

        assert_eq!(
            scalar_payload, proc_subset_mask,
            "a one-word untagged slot cannot distinguish a scalar intern id \
             from an equal finite-set bitmask"
        );

        assert_eq!(
            encode_tagged_scalar_set_scalar(scalar_payload).unwrap(),
            scalar_payload
        );
        assert_eq!(
            encode_tagged_scalar_set_mask(proc_subset_mask, 4).unwrap(),
            -8
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_slot_encoding_separates_scalar_ids_from_proc_masks() {
        let universe_len = 4;
        let scalar_ids = [0, 7, 14, 15, 30, i64::from(u32::MAX)];
        let mut encoded_scalars = std::collections::BTreeSet::new();

        for scalar in scalar_ids {
            let raw = encode_tagged_scalar_set_scalar(scalar).unwrap();
            assert_eq!(raw, scalar);
            assert!(raw >= 0);
            assert_eq!(
                decode_tagged_scalar_set_slot(raw, universe_len).unwrap(),
                TaggedScalarSetSlot::Scalar(scalar)
            );
            encoded_scalars.insert(raw);
        }

        for mask in 0..=0b1111 {
            let raw = encode_tagged_scalar_set_mask(mask, universe_len).unwrap();
            assert!(raw < 0);
            assert!(!encoded_scalars.contains(&raw));
            assert_eq!(
                decode_tagged_scalar_set_slot(raw, universe_len).unwrap(),
                TaggedScalarSetSlot::SetMask(mask)
            );
        }

        assert_eq!(
            encode_tagged_scalar_set_scalar(-1),
            Err(TaggedScalarSetSlotError::NegativeScalar(-1))
        );
        assert_eq!(
            encode_tagged_scalar_set_mask(0b1_0000, universe_len),
            Err(TaggedScalarSetSlotError::NonCanonicalSetMask {
                mask: 0b1_0000,
                universe_len,
            })
        );
        assert_eq!(
            decode_tagged_scalar_set_slot(-17, universe_len),
            Err(TaggedScalarSetSlotError::NonCanonicalTaggedSet {
                raw: -17,
                universe_len,
            })
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_slot_encoding_supports_sixty_three_bit_universe() {
        let raw = encode_tagged_scalar_set_mask(i64::MAX, 63).unwrap();
        assert_eq!(raw, i64::MIN);
        assert_eq!(
            decode_tagged_scalar_set_slot(raw, 63).unwrap(),
            TaggedScalarSetSlot::SetMask(i64::MAX)
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_union_slot_indexes_typed_universe_without_collisions() {
        let universe = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("None")),
            FlatScalarValue::Int(1),
            FlatScalarValue::String(std::sync::Arc::from("1")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("1")),
        ];

        for (index, value) in universe.iter().enumerate() {
            let raw = encode_tagged_scalar_union_value(value, &universe).unwrap();
            assert_eq!(raw, index as i64);
            assert_eq!(
                decode_tagged_scalar_union_slot(raw, &universe).unwrap(),
                value.clone()
            );
        }

        assert_ne!(
            encode_tagged_scalar_union_value(&FlatScalarValue::Int(1), &universe).unwrap(),
            encode_tagged_scalar_union_value(
                &FlatScalarValue::String(std::sync::Arc::from("1")),
                &universe
            )
            .unwrap()
        );
        assert_ne!(
            encode_tagged_scalar_union_value(
                &FlatScalarValue::String(std::sync::Arc::from("1")),
                &universe
            )
            .unwrap(),
            encode_tagged_scalar_union_value(
                &FlatScalarValue::ModelValue(std::sync::Arc::from("1")),
                &universe,
            )
            .unwrap()
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_union_slot_rejects_noncanonical_raw_values() {
        let universe = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("None")),
            FlatScalarValue::Int(1),
            FlatScalarValue::Int(2),
        ];

        assert_eq!(
            decode_tagged_scalar_union_slot(-1, &universe),
            Err(TaggedScalarUnionSlotError::NonCanonicalTaggedUnion {
                raw: -1,
                universe_len: universe.len(),
            })
        );
        assert_eq!(
            decode_tagged_scalar_union_slot(3, &universe),
            Err(TaggedScalarUnionSlotError::NonCanonicalTaggedUnion {
                raw: 3,
                universe_len: universe.len(),
            })
        );
        assert_eq!(
            encode_tagged_scalar_union_value(&FlatScalarValue::Int(3), &universe),
            Err(TaggedScalarUnionSlotError::ValueOutsideUniverse)
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_union_proof_rejects_duplicate_typed_values() {
        let duplicate = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("None")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("None")),
        ];
        assert_eq!(
            TaggedScalarUnionProof::new(duplicate, Arc::from("TypeOK")),
            Err(TaggedScalarUnionSlotError::DuplicateUniverseValue)
        );

        let typed_distinct = vec![
            FlatScalarValue::String(std::sync::Arc::from("None")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("None")),
        ];
        let proof = TaggedScalarUnionProof::new(typed_distinct.clone(), Arc::from("TypeOK"))
            .expect("string and model-value lanes with the same name are distinct");
        assert_eq!(proof.universe(), typed_distinct.as_slice());
        assert_eq!(proof.source().as_ref(), "TypeOK");
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_range_proof_promotes_flat_primary() {
        let registry = VarRegistry::from_names(["temp"]);
        let proc_universe = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("p3")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("p4")),
        ];
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::ModelValue,
            proc_universe.clone(),
            Arc::from("DijkstraTempTypeOK"),
        )
        .unwrap();
        assert_eq!(proof.source().as_ref(), "DijkstraTempTypeOK");
        let layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![
                    Arc::from("p1"),
                    Arc::from("p2"),
                    Arc::from("p3"),
                    Arc::from("p4"),
                ],
                domain_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                value_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
            }],
        );

        assert!(layout.is_fully_flat());
        assert!(layout.has_model_value_keyed_tagged_scalar_set_range());
        assert!(
            layout.supports_flat_primary(),
            "canonical tagged scalar/set proof metadata makes the fixed function primary-safe"
        );
        assert!(
            layout.supports_flat_bfs_auto_admission(),
            "the tagged proof distinguishes scalar slots from scalar/set slots for flat BFS"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_range_proof_rejects_int_scalar_primary_safety() {
        let registry = VarRegistry::from_names(["temp"]);
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::Int,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            ],
            Arc::from("IntOrProcSetTypeOK"),
        )
        .unwrap();
        let layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2")],
                domain_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                value_types: vec![SlotType::Int, SlotType::Int],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
            }],
        );

        assert!(layout.is_fully_flat());
        assert!(
            layout.has_model_value_keyed_tagged_scalar_set_range(),
            "the structural tagged proof is still present"
        );
        assert!(
            !layout.supports_flat_primary(),
            "int scalar arms lack a non-negative scalar proof for primary tagged encoding"
        );
        assert!(!layout.supports_flat_bfs_auto_admission());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_scalar_model_value_function_without_range_proof_is_not_tagged() {
        let registry = VarRegistry::from_names(["temp"]);
        let layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2")],
                domain_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                value_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
            }],
        );

        assert!(!layout.has_model_value_keyed_tagged_scalar_set_range());
        assert!(!layout.supports_flat_primary());
        assert!(!layout.supports_flat_bfs_auto_admission());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_range_proof_rejects_duplicate_universe() {
        assert_eq!(
            TaggedScalarSetRangeProof::new(
                SlotType::ModelValue,
                vec![
                    FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                    FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                ],
                Arc::from("DijkstraTempTypeOK"),
            ),
            Err(TaggedScalarSetSlotError::DuplicateUniverseValue)
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_range_proof_accepts_string_keyed_domain() {
        let registry = VarRegistry::from_names(["temp"]);
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::String,
            vec![FlatScalarValue::String(std::sync::Arc::from("p1"))],
            Arc::from("DijkstraTempTypeOK"),
        )
        .unwrap();
        let layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1")],
                domain_types: vec![SlotType::String],
                value_types: vec![SlotType::String],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
            }],
        );

        assert!(layout.has_model_value_keyed_tagged_scalar_set_range());
        assert!(layout.supports_flat_primary());
        assert!(layout.supports_flat_bfs_auto_admission());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_primary_rejects_direct_function_sampled_set_slots() {
        let registry = VarRegistry::from_names(["temp"]);
        let set_range = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: vec![
                        FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                        FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                    ],
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![
                            FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                            FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                        ],
                        // Sampled universe: a successor could write outside it.
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );

        assert!(set_range.is_fully_flat());
        assert!(
            !set_range.supports_flat_primary(),
            "a sampled function-range bitmask universe is not proven closed under successor writes"
        );
        assert!(!set_range.supports_flat_bfs_auto_admission());

        let scalar_model_value_range = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: vec![
                        FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                        FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                    ],
                    value_layout: Box::new(FlatValueLayout::Scalar(SlotType::ModelValue)),
                },
            }],
        );

        assert!(scalar_model_value_range.is_fully_flat());
        assert!(!scalar_model_value_range.supports_flat_primary());
        assert!(!scalar_model_value_range.supports_flat_bfs_auto_admission());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_flat_primary_admits_direct_function_proven_closed_set_slots() {
        // GAP B: a function-range bitmask whose universe is proven closed by a
        // type invariant (e.g. TypeOK `x \in [Domain -> SUBSET {0,1}]`) is now
        // admitted as a native flat-primary slot. Both soundness facts hold:
        //   (1) closure — no successor can introduce an out-of-universe element;
        //   (2) canonical encoding — the native EXCEPT-set lowering writes the
        //       slot through `static_set_bitmask_materialization_mask` (bit `i` =
        //       element position in the range universe), bit-identical to the
        //       interpreter's `set_bitmask_value_to_slot`.
        // (The earlier `canonical_payload_mismatch` on SimpleRegular — 9 states
        // versus the 277726 oracle — was a compiled-BFS per-successor pre-seen
        // dedup defect, not a lowering bug; that path now consults the recorded
        // compiled-flat payload witness before failing closed.)
        let registry = VarRegistry::from_names(["x"]);
        let proven_range = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 0,
                    len: 8,
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![FlatScalarValue::Int(0), FlatScalarValue::Int(1)],
                        universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                            invariant: Arc::from("TypeOK"),
                        },
                    }),
                },
            }],
        );

        assert!(proven_range.is_fully_flat());
        assert!(
            proven_range.supports_flat_primary(),
            "a proven-closed function-range bitmask is admitted to flat-primary: the universe is \
             closed under successor writes and the native lowering writes the canonical bitmask"
        );

        // A *sampled* universe stays fenced: a successor could write an element
        // outside the sampled universe and silently corrupt the slot.
        let sampled_range = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 0,
                    len: 8,
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![FlatScalarValue::Int(0), FlatScalarValue::Int(1)],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );

        assert!(sampled_range.is_fully_flat());
        assert!(
            !sampled_range.supports_flat_primary(),
            "a sampled function-range bitmask universe is not proven closed and must stay fenced"
        );
    }
}
