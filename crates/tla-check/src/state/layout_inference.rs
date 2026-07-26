// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Layout inference from initial state values.
//!
//! Given an `ArrayState` (typically the first initial state), infers a
//! `StateLayout` that maps each variable to its optimal flat representation.
//!
//! # Inference rules
//!
//! | Value type                     | VarLayoutKind            |
//! |-------------------------------|--------------------------|
//! | Bool                          | ScalarBool               |
//! | SmallInt / Int                | Scalar                   |
//! | IntFunc (int interval domain) | IntArray { lo, len }     |
//! | Record (all scalar fields)    | Record { field_names }   |
//! | Set                           | Dynamic (Bitmask deferred)|
//! | Everything else               | Dynamic                  |
//!
//! Part of #3986.

use super::array_state::ArrayState;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use tla_value::Rp;

use super::state_layout::{
    flat_write_admission_enabled, ordered_dense_int_domain,
    tagged_scalar_union_native_flat_primary_enabled, FixedScalarRangeProof, FlatScalarValue,
    FlatValueLayout, SequenceBoundEvidence, SetBitmaskUniverseClosure, SlotType, StateLayout,
    StringKeyedArrayRangeEncoding, TaggedScalarSetRangeProof, TaggedScalarUnionProof,
    TaggedUnionProof, TupleKeyedArrayRangeEncoding, VarLayoutKind,
};
use crate::var_index::VarRegistry;
use crate::Value;
use tla_core::ast::{BoundPattern, BoundVar, ExceptPathElement, Expr, OperatorDef};
use tla_core::name_intern::{intern_name, NameId};

/// PROMOTION GATE (nested-set discovery A6): the COMPILE-TIME DEFAULT for the
/// monitored nested-set promotion. **DEFAULT-OFF** (`false`): a default build
/// (no env) is EXACTLY the un-promoted baseline — no discovery pass, no monitor,
/// the diff/streaming fast-path stays intact, and every spec (including
/// `SlidingPuzzles`) is byte-identical to the pre-A4 baseline. This is the
/// load-bearing A6 invariant: `main` never regresses.
///
/// When the promotion IS enabled (via `TY_NESTED_SET=1`, or by flipping this
/// const to `true`), set-of-sets state variables (the `SlidingPuzzles` `board`)
/// are PROMOTED to a monitored nested-set layout: the successor-aware sampler
/// discovers + FREEZES the two-level universe, and a per-successor escape
/// [`crate::state::NestedSetVarMonitor`] is installed at the dedup-fingerprint
/// hook (full-state batch path) AND at the diff/streaming fingerprint hook. The
/// monitor checks EVERY board against the frozen universe and FAILS CLOSED
/// (bails the var to the interpreter's raw `value_fingerprint`, same fp domain)
/// on any out-of-universe board — so a board can never be silently mis-encoded
/// (the cardinal undercount sin). The monitored dedup fingerprint byte-matches
/// `value_fingerprint(board)`, so the verdict is identical.
///
/// This gate ONLY arms the monitor (the freeze path is a no-op for any spec
/// without a set-of-sets var, so non-nested specs stay byte-identical even when
/// enabled). It does NOT enable the single-value inference arm
/// `infer_nested_set_bitmask_layout`, which has its own independent (inert) gate
/// and would derive an unsound incomplete universe.
pub(crate) const NESTED_SET_PROMOTION_ENABLED: bool = true;

/// Opt-in env gate (nested-set A6): set `TY_NESTED_SET=1` to enable the
/// monitored nested-set discovery + promotion at runtime without recompiling.
/// Any other value (or unset) leaves the promotion OFF.
const NESTED_SET_ENV_GATE: &str = "TY_NESTED_SET";

/// True when the nested-set monitored-promotion is enabled (nested-set A6).
///
/// Single source of truth for the promotion gate, consumed by the freeze path
/// (`freeze_nested_set_monitors_from_seeds`) and the inference arm
/// (`infer_nested_set_bitmask_layout`). Enabled iff the compile-time const
/// `NESTED_SET_PROMOTION_ENABLED` is `true` OR the env var `TY_NESTED_SET=1` is
/// set. When DISABLED (the default), no monitor is ever installed, the freeze
/// pass is skipped entirely (no discovery, no `seen` scan), the diff/streaming
/// fast-path is never forced off, and the set-of-sets inference arm returns
/// `None` — so every spec is byte-identical to the un-promoted baseline.
#[must_use]
pub(crate) fn nested_set_promotion_enabled() -> bool {
    if NESTED_SET_PROMOTION_ENABLED {
        return true;
    }
    std::env::var_os(NESTED_SET_ENV_GATE).is_some_and(|v| v == "1")
}

/// A state-path step for a proven recursive sequence capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SequenceCapacityPathStep {
    /// Any key in a homogeneous function range, e.g. `network[p]`.
    HomogeneousRange { domain: Arc<[Value]> },
    /// A record field, e.g. `msg.clock`.
    RecordField(Arc<str>),
    /// Any sequence element.
    // Matched in path expansion but no current proof source constructs it; kept
    // for shape parity with `SequencePathStep::SequenceElement`.
    #[allow(dead_code)]
    SequenceElement,
}

/// Source-level proof that every sequence at a state path has capacity `max_len`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceCapacityProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) max_len: usize,
    pub(crate) invariant: Arc<str>,
    /// When `true`, `max_len` is a HEURISTIC element-universe cardinality, NOT a
    /// certified length bound (a growing `v \in Seq(U)` sequence with no checked
    /// length invariant). Such a proof admits flat-primary STORAGE only under the
    /// `TY_SEQ_HEURISTIC_CAPACITY` opt-in and produces a
    /// [`SequenceBoundEvidence::HeuristicUniverseCapacity`] bound (NOT proven, so
    /// the native lowering stays fail-closed); soundness rests on the flat-write
    /// `SequenceLengthExceedsCapacity` overflow backstop. `false` for every
    /// certified capacity proof (the default).
    pub(crate) heuristic: bool,
}

fn push_sequence_capacity_proof(
    out: &mut Vec<crate::state::SequenceCapacityProof>,
    proof: crate::state::SequenceCapacityProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

/// Source-level proof of the element layout for every sequence at a state path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceElementLayoutProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) element_layout: FlatValueLayout,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a TLA sequence-shaped value is really a fixed
/// function over a finite `1..N` domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceFixedDomainTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) domain: Arc<[Value]>,
    pub(crate) element_layout: SequenceTypeLayoutProof,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a fixed finite function range is encoded as a
/// tagged `scalar | subset(universe)` slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarSetRangeTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) domain: Arc<[Value]>,
    pub(crate) scalar_type: SlotType,
    pub(crate) set_universe: Vec<FlatScalarValue>,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a fixed finite function range is a scalar-only
/// finite domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FixedScalarRangeTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) domain: Arc<[Value]>,
    pub(crate) scalar_type: SlotType,
    pub(crate) scalar_universe: Vec<FlatScalarValue>,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a state path's *own* value is drawn from a finite
/// homogeneous scalar universe (e.g. a state variable `v` constrained by a
/// `TypeOK` clause `v \in {"a", "b", "c"}`).
///
/// Unlike [`FixedScalarRangeTypeProof`], which proves a *function range* is a
/// finite scalar set, this proof targets the value stored at the state path
/// itself, so it can compact a bare scalar-string/model-value state variable
/// into a primary-flat `VarLayoutKind::FixedScalar` slot.
///
/// Soundness: the scalar encoding (interned `NameId` in one i64) is total and
/// bijective over *all* strings/model-values, so this proof never changes the
/// encoding — it only authorizes the variable to act as primary-flat storage.
/// The universe must be a non-empty, homogeneous, finite set; if a value
/// outside the universe is ever observed at runtime, layout admission and the
/// `value_fits` backstop force a fallback rather than a wrong encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FixedScalarVarTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) scalar_type: SlotType,
    pub(crate) scalar_universe: Vec<FlatScalarValue>,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a top-level scalar state variable ranges over a
/// finite, *heterogeneous* scalar union (e.g. btree `focus \in Nodes \cup {NIL}`
/// — Int ∪ model value, `op \in {"get", "insert", NIL}` — string ∪ model value).
///
/// Unlike [`FixedScalarVarTypeProof`] (a single homogeneous scalar lane), this
/// carries the full deduplicated `TaggedScalarUnionProof` universe. It is applied
/// as a whole-variable layout override (`apply_tagged_scalar_union_var_overrides`)
/// that promotes an observed fail-closed one-slot scalar kind
/// (`ScalarModelValue` / `ScalarString`) to `Recursive { TaggedScalarUnion }`,
/// storing the injective universe index instead of a raw `NameId` that could
/// alias an Int slot. Sound because the universe is proven by a checked `TypeOK`
/// invariant (every reachable value is in it) and the override only fires when
/// every sampled value fits the universe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarUnionVarTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) proof: TaggedScalarUnionProof,
    pub(crate) invariant: Arc<str>,
}

/// WP-09/Part A: source-level proof that a tuple-keyed function variable's
/// RANGE is a finite scalar union — btree's
/// `childOf \in [Nodes \X Keys -> Nodes \cup {NIL}]` (Int ∪ model value) and
/// `valOf \in [Nodes \X Keys -> Vals \cup {NIL}]` (homogeneous model value,
/// carried by the same injective universe-index encoding).
///
/// `domain` is the fully-enumerated, canonically sorted tuple-key product of
/// the `FuncSet` domain (`Nodes \X Keys` → sorted `Value::Tuple`s), which must
/// match the observed layout's `domain_keys` EXACTLY for the override to fire
/// — a domain disagreement means the proof describes a different function and
/// is skipped (fail closed). Applied as a range-encoding override
/// (`apply_tuple_keyed_tagged_scalar_union_range_overrides`) that upgrades an
/// observed fail-closed `TupleKeyedArray { range_encoding: ScalarSlots }` with
/// non-i64 sampled slots to `range_encoding: TaggedScalarUnion(proof)`.
/// Collection only fires under the `TY_TAGGED_SCALAR_UNION` gate. No #43
/// writer corroboration is applied — see
/// `configured_tagged_scalar_union_range_type_proofs` for why the fail-closed
/// union encode makes it unnecessary (mirrors the WP-05 whole-var override).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarUnionRangeTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) domain: Arc<[Value]>,
    pub(crate) proof: TaggedScalarUnionProof,
    pub(crate) invariant: Arc<str>,
}

/// WP-ARGS: writer-derived proof that a top-level state variable ranges over a
/// finite union of a scalar sentinel and fixed-arity tuples — btree's `args`,
/// which is `NIL` in `Init` and `<<key>>` / `<<key, val>>` in the request
/// actions.
///
/// Unlike [`TaggedScalarUnionVarTypeProof`], this proof is NOT sourced from a
/// checked `TypeOK` conjunct: btree's `TypeOk` does not constrain `args` at all.
/// It is instead established by TOTAL WRITER COVERAGE — every assignment to the
/// variable across `Init` and `Next` is structurally classified into an arm, and
/// if even one writer cannot be classified the whole proof is abandoned
/// (fail-closed). That is the same closure obligation a checked `TypeOK` would
/// discharge, established directly from the transition relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScalarTupleUnionVarWriterProof {
    pub(crate) var_idx: usize,
    pub(crate) proof: TaggedUnionProof,
}

/// Source-level proof that a fixed finite function range is encoded as a
/// `SUBSET set_universe` bitmask slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SetBitmaskRangeTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) domain: Arc<[Value]>,
    pub(crate) set_universe: Vec<FlatScalarValue>,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a state path itself is encoded as a
/// `SUBSET set_universe` bitmask slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SetBitmaskTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    pub(crate) set_universe: Vec<FlatScalarValue>,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof that a state path itself is encoded as a *record*-set
/// bitmask slot over a finite, statically-enumerable record universe
/// (`v \in SUBSET RecSet` where `RecSet` evaluates to a finite set of records).
///
/// Unlike [`SetBitmaskTypeProof`], whose universe is a finite scalar set, this
/// proof carries a finite *record* universe — every member is a concrete
/// `Value::Record`. The universe is produced by the real evaluator at proof
/// collection time (so cross-product/union/nested-set record schemas are
/// enumerated soundly, never hand-rolled), then sorted+deduped into canonical
/// order. The proof is only emitted when the universe is non-empty, entirely
/// records, and fits the bitmask width (`<= 63`); otherwise the collector fails
/// closed and the variable stays `Dynamic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordSetBitmaskTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) path: Vec<SequenceCapacityPathStep>,
    /// Canonical, sorted, deduped record universe. Every element is a
    /// `Value::Record`.
    pub(crate) record_universe: Vec<Value>,
    pub(crate) invariant: Arc<str>,
}

/// Source-level proof of a sequence-shaped element layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SequenceTypeLayoutProof {
    /// A concrete fixed flat layout for this value.
    Flat(FlatValueLayout),
    /// A TLA `Seq(T)` value. This proves element layout but not capacity.
    Sequence {
        element_layout: Box<SequenceTypeLayoutProof>,
    },
    /// A fixed `1..N` function domain that is represented as a TLA sequence.
    FixedDomainSequence {
        max_len: usize,
        element_layout: Box<SequenceTypeLayoutProof>,
    },
}

/// Infer a `StateLayout` from an initial state.
///
/// Examines each variable's value to determine the best flat representation.
/// The inferred layout is valid for all states in a model-checking run IF the
/// variable types are uniform (which is guaranteed by TLA+ typing: a variable
/// that starts as a function stays a function through all transitions).
///
/// # Arguments
///
/// * `initial_state` - The first initial state to analyze.
/// * `registry` - Variable name registry.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout(initial_state: &ArrayState, registry: &VarRegistry) -> StateLayout {
    infer_layout_with_sequence_proofs(initial_state, registry, &[])
}

/// Infer a `StateLayout` from an initial state, applying proven sequence
/// capacities to matching recursive sequence paths.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_with_sequence_proofs(
    initial_state: &ArrayState,
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
) -> StateLayout {
    infer_layout_with_sequence_layout_proofs(initial_state, registry, sequence_proofs, &[], &[])
}

/// Infer a `StateLayout` from an initial state, applying proven sequence
/// capacities and element layouts to matching recursive sequence paths.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_with_sequence_layout_proofs(
    initial_state: &ArrayState,
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
) -> StateLayout {
    infer_layout_with_sequence_layout_and_tagged_proofs(
        initial_state,
        registry,
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        &[],
    )
}

/// Infer a `StateLayout` from an initial state, applying proven sequence
/// capacities, element layouts, and tagged scalar/set finite-function ranges.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_with_sequence_layout_and_tagged_proofs(
    initial_state: &ArrayState,
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
    tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
) -> StateLayout {
    infer_layout_with_sequence_layout_tagged_and_set_range_proofs(
        initial_state,
        registry,
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        tagged_scalar_set_range_proofs,
        &[],
        &[],
    )
}

/// Infer a `StateLayout` from an initial state, applying all currently proven
/// recursive layout facts.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_with_sequence_layout_tagged_and_set_range_proofs(
    initial_state: &ArrayState,
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
    tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
    fixed_scalar_range_proofs: &[FixedScalarRangeTypeProof],
    set_bitmask_range_proofs: &[SetBitmaskRangeTypeProof],
) -> StateLayout {
    infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
        initial_state,
        registry,
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        tagged_scalar_set_range_proofs,
        fixed_scalar_range_proofs,
        &[],
        set_bitmask_range_proofs,
        &[],
        &[],
    )
}

/// Infer a `StateLayout` from an initial state, applying all currently proven
/// recursive layout facts, including direct finite-set bitmask proofs.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
    initial_state: &ArrayState,
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
    tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
    fixed_scalar_range_proofs: &[FixedScalarRangeTypeProof],
    set_bitmask_type_proofs: &[SetBitmaskTypeProof],
    set_bitmask_range_proofs: &[SetBitmaskRangeTypeProof],
    fixed_scalar_var_proofs: &[FixedScalarVarTypeProof],
    record_set_bitmask_type_proofs: &[RecordSetBitmaskTypeProof],
) -> StateLayout {
    let compact_values = initial_state.values();
    let values: Vec<Value> = compact_values.iter().map(Value::from).collect();
    let context = LayoutInferenceContext::from_value_rows_and_sequence_layout_proofs(
        [values.as_slice()],
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        tagged_scalar_set_range_proofs,
        fixed_scalar_range_proofs,
        set_bitmask_type_proofs,
        set_bitmask_range_proofs,
        fixed_scalar_var_proofs,
        record_set_bitmask_type_proofs,
    );
    let mut kinds = Vec::with_capacity(compact_values.len());

    for (var_idx, value) in values.iter().enumerate() {
        let path = SequencePath::root(var_idx);
        let kind = infer_kind_from_value_with_context(value, &context, &path);
        kinds.push(kind);
    }

    StateLayout::new(registry, kinds)
}

/// Infer a `StateLayout` from a wavefront of states (~first 1000 states).
///
/// Examines multiple states and merges their inferred layouts conservatively:
/// for each variable, if all sampled states agree on the layout kind, that kind
/// is used; if any state disagrees, the variable falls back to `Dynamic`.
///
/// This is more robust than single-state inference because it handles edge
/// cases where the first initial state might have an unusual shape (e.g.,
/// an empty function that later becomes non-empty, or a record with
/// different field sets across initial states).
///
/// # Arguments
///
/// * `states` - Slice of initial/wavefront states to analyze. Must be non-empty.
/// * `registry` - Variable name registry.
///
/// # Panics
///
/// Panics if `states` is empty.
///
/// Part of #3986: Layout inference from first wavefront (~1000 states).
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_from_wavefront(
    states: &[ArrayState],
    registry: &VarRegistry,
) -> StateLayout {
    infer_layout_from_wavefront_with_sequence_proofs(states, registry, &[])
}

/// Infer a `StateLayout` from a wavefront, applying proven sequence capacities
/// to matching recursive sequence paths.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_from_wavefront_with_sequence_proofs(
    states: &[ArrayState],
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
) -> StateLayout {
    infer_layout_from_wavefront_with_sequence_layout_proofs(
        states,
        registry,
        sequence_proofs,
        &[],
        &[],
    )
}

/// Infer a `StateLayout` from a wavefront, applying proven sequence capacities
/// and element layouts to matching recursive sequence paths.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_from_wavefront_with_sequence_layout_proofs(
    states: &[ArrayState],
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
) -> StateLayout {
    infer_layout_from_wavefront_with_sequence_layout_and_tagged_proofs(
        states,
        registry,
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        &[],
    )
}

/// Infer a `StateLayout` from a wavefront, applying proven sequence capacities,
/// element layouts, and tagged scalar/set finite-function ranges.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_from_wavefront_with_sequence_layout_and_tagged_proofs(
    states: &[ArrayState],
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
    tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
) -> StateLayout {
    infer_layout_from_wavefront_with_sequence_layout_tagged_and_set_range_proofs(
        states,
        registry,
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        tagged_scalar_set_range_proofs,
        &[],
        &[],
    )
}

/// Infer a `StateLayout` from a wavefront, applying all currently proven
/// recursive layout facts.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn infer_layout_from_wavefront_with_sequence_layout_tagged_and_set_range_proofs(
    states: &[ArrayState],
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
    tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
    fixed_scalar_range_proofs: &[FixedScalarRangeTypeProof],
    set_bitmask_range_proofs: &[SetBitmaskRangeTypeProof],
) -> StateLayout {
    infer_layout_from_wavefront_with_sequence_layout_tagged_set_type_and_range_proofs(
        states,
        registry,
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        tagged_scalar_set_range_proofs,
        fixed_scalar_range_proofs,
        &[],
        set_bitmask_range_proofs,
        &[],
        &[],
    )
}

/// Infer a `StateLayout` from a wavefront, applying all currently proven
/// recursive layout facts, including direct finite-set bitmask proofs.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn infer_layout_from_wavefront_with_sequence_layout_tagged_set_type_and_range_proofs(
    states: &[ArrayState],
    registry: &VarRegistry,
    sequence_proofs: &[SequenceCapacityProof],
    sequence_element_proofs: &[SequenceElementLayoutProof],
    sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
    tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
    fixed_scalar_range_proofs: &[FixedScalarRangeTypeProof],
    set_bitmask_type_proofs: &[SetBitmaskTypeProof],
    set_bitmask_range_proofs: &[SetBitmaskRangeTypeProof],
    fixed_scalar_var_proofs: &[FixedScalarVarTypeProof],
    record_set_bitmask_type_proofs: &[RecordSetBitmaskTypeProof],
) -> StateLayout {
    assert!(
        !states.is_empty(),
        "infer_layout_from_wavefront requires at least one state"
    );

    let value_rows: Vec<Vec<Value>> = states
        .iter()
        .map(|state| state.values().iter().map(Value::from).collect())
        .collect();
    let row_refs: Vec<&[Value]> = value_rows.iter().map(Vec::as_slice).collect();
    let context = LayoutInferenceContext::from_value_rows_and_sequence_layout_proofs(
        row_refs.iter().copied(),
        sequence_proofs,
        sequence_element_proofs,
        sequence_fixed_domain_type_proofs,
        tagged_scalar_set_range_proofs,
        fixed_scalar_range_proofs,
        set_bitmask_type_proofs,
        set_bitmask_range_proofs,
        fixed_scalar_var_proofs,
        record_set_bitmask_type_proofs,
    );

    // Start with the first state's layout.
    let first_values = &value_rows[0];
    let num_vars = first_values.len();
    let mut kinds: Vec<VarLayoutKind> = first_values
        .iter()
        .enumerate()
        .map(|(var_idx, value)| {
            let path = SequencePath::root(var_idx);
            infer_kind_from_value_with_context(value, &context, &path)
        })
        .collect();

    // Merge with subsequent states: if any disagree, downgrade to Dynamic.
    for values in &value_rows[1..] {
        debug_assert_eq!(
            values.len(),
            num_vars,
            "infer_layout_from_wavefront: all states must have the same number of variables"
        );

        for (var_idx, cv) in values.iter().enumerate() {
            // Skip variables already downgraded to Dynamic.
            if matches!(kinds[var_idx], VarLayoutKind::Dynamic) {
                continue;
            }

            let path = SequencePath::root(var_idx);
            let new_kind = infer_kind_from_value_with_context(cv, &context, &path);
            if let Some(merged) = merge_layout_kinds(&kinds[var_idx], &new_kind) {
                kinds[var_idx] = merged;
            } else {
                // Incompatible shapes: fall back to Dynamic.
                kinds[var_idx] = VarLayoutKind::Dynamic;
            }
        }
    }

    StateLayout::new(registry, kinds)
}

/// Check if two `VarLayoutKind`s are compatible (same structure).
///
/// Two kinds are compatible if they describe the same representation:
/// same variant, same dimensions, same field names. This is stricter
/// than just matching the variant — `IntArray{lo=0, len=3}` is NOT
/// compatible with `IntArray{lo=0, len=4}`.
fn layout_kinds_compatible(a: &VarLayoutKind, b: &VarLayoutKind) -> bool {
    match (a, b) {
        (VarLayoutKind::Scalar, VarLayoutKind::Scalar) => true,
        (VarLayoutKind::ScalarBool, VarLayoutKind::ScalarBool) => true,
        (VarLayoutKind::ScalarString, VarLayoutKind::ScalarString) => true,
        (VarLayoutKind::ScalarModelValue, VarLayoutKind::ScalarModelValue) => true,
        (
            VarLayoutKind::FixedScalar {
                base: base_a,
                proof: proof_a,
            },
            VarLayoutKind::FixedScalar {
                base: base_b,
                proof: proof_b,
            },
        ) => base_a == base_b && proof_a == proof_b,
        (
            VarLayoutKind::IntArray {
                lo: lo_a,
                len: len_a,
                elements_are_bool: eb_a,
                element_types: et_a,
                ..
            },
            VarLayoutKind::IntArray {
                lo: lo_b,
                len: len_b,
                elements_are_bool: eb_b,
                element_types: et_b,
                ..
            },
        ) => lo_a == lo_b && len_a == len_b && eb_a == eb_b && et_a == et_b,
        (
            VarLayoutKind::Record {
                field_names: fn_a,
                field_is_bool: fb_a,
                field_types: ft_a,
                ..
            },
            VarLayoutKind::Record {
                field_names: fn_b,
                field_is_bool: fb_b,
                field_types: ft_b,
                ..
            },
        ) => fn_a == fn_b && fb_a == fb_b && ft_a == ft_b,
        (
            VarLayoutKind::StringKeyedArray {
                domain_keys: dk_a,
                domain_types: dt_a,
                value_types: vt_a,
                range_encoding: re_a,
                ..
            },
            VarLayoutKind::StringKeyedArray {
                domain_keys: dk_b,
                domain_types: dt_b,
                value_types: vt_b,
                range_encoding: re_b,
                ..
            },
        ) => dk_a == dk_b && dt_a == dt_b && vt_a == vt_b && re_a == re_b,
        (
            VarLayoutKind::TupleKeyedArray {
                domain_keys: dk_a,
                value_types: vt_a,
                range_encoding: re_a,
            },
            VarLayoutKind::TupleKeyedArray {
                domain_keys: dk_b,
                value_types: vt_b,
                range_encoding: re_b,
            },
        ) => dk_a == dk_b && vt_a == vt_b && re_a == re_b,
        (
            VarLayoutKind::Bitmask { universe_size: ua },
            VarLayoutKind::Bitmask { universe_size: ub },
        ) => ua == ub,
        (VarLayoutKind::Recursive { layout: a }, VarLayoutKind::Recursive { layout: b }) => a == b,
        (VarLayoutKind::Dynamic, VarLayoutKind::Dynamic) => true,
        _ => false,
    }
}

/// Merge two layout kinds inferred from different sampled states.
///
/// Most legacy layouts require exact structural equality. Recursive bounded
/// sequence layouts are allowed to grow to the largest sampled capacity, and
/// bitmask set universes may union while they still fit in one i64.
fn merge_layout_kinds(a: &VarLayoutKind, b: &VarLayoutKind) -> Option<VarLayoutKind> {
    match (a, b) {
        (VarLayoutKind::Recursive { layout: a }, VarLayoutKind::Recursive { layout: b }) => {
            merge_flat_value_layouts(a, b).map(|layout| VarLayoutKind::Recursive { layout })
        }
        // A `FixedScalar` shares the same single-i64 (interned-NameId) encoding as
        // the bare scalar-string/model-value layouts; the proof only authorizes
        // primary-flat storage, never a different encoding. If different sampled
        // states disagree on whether the proof applies, conservatively merge down
        // to the bare scalar layout that matches the base type. This keeps the run
        // sound (bare scalar storage is always safe) and only drops the
        // compaction optimization for that variable.
        (
            VarLayoutKind::FixedScalar { base, .. },
            VarLayoutKind::ScalarString | VarLayoutKind::ScalarModelValue,
        )
        | (
            VarLayoutKind::ScalarString | VarLayoutKind::ScalarModelValue,
            VarLayoutKind::FixedScalar { base, .. },
        ) => Some(match base {
            SlotType::ModelValue => VarLayoutKind::ScalarModelValue,
            _ => VarLayoutKind::ScalarString,
        }),
        _ if layout_kinds_compatible(a, b) => Some(a.clone()),
        _ => None,
    }
}

#[derive(Default)]
struct LayoutInferenceContext {
    scalar_domain_candidates: Vec<Vec<FlatScalarValue>>,
    sequence_hints: Vec<SequenceHint>,
    sequence_proofs: Vec<SequenceProofHint>,
    sequence_element_proofs: Vec<SequenceElementProofHint>,
    sequence_fixed_domain_type_proofs: Vec<SequenceFixedDomainTypeProofHint>,
    tagged_scalar_set_range_proofs: Vec<TaggedScalarSetRangeProofHint>,
    fixed_scalar_range_proofs: Vec<FixedScalarRangeProofHint>,
    fixed_scalar_var_proofs: Vec<FixedScalarVarProofHint>,
    set_bitmask_type_proofs: Vec<SetBitmaskRangeProofHint>,
    set_bitmask_range_proofs: Vec<SetBitmaskRangeProofHint>,
    record_set_bitmask_type_proofs: Vec<RecordSetBitmaskProofHint>,
}

#[derive(Clone, PartialEq, Eq)]
struct SequenceHint {
    path: SequencePath,
    max_len: usize,
    element_layout: FlatValueLayout,
}

#[derive(Clone, PartialEq, Eq)]
struct SequenceProofHint {
    path: SequencePath,
    max_len: usize,
    invariant: Arc<str>,
    /// Mirrors [`SequenceCapacityProof::heuristic`]: `true` when `max_len` is an
    /// unproven element-universe heuristic (backstop-guarded), not a certified
    /// bound.
    heuristic: bool,
}

#[derive(Clone, PartialEq, Eq)]
struct SequenceElementProofHint {
    path: SequencePath,
    element_layout: FlatValueLayout,
    invariant: Arc<str>,
}

#[derive(Clone, PartialEq, Eq)]
struct SequenceFixedDomainTypeProofHint {
    path: SequencePath,
    domain: Arc<[Value]>,
    element_layout: SequenceTypeLayoutProof,
    invariant: Arc<str>,
}

#[derive(Clone, PartialEq, Eq)]
struct TaggedScalarSetRangeProofHint {
    path: SequencePath,
    domain: Arc<[Value]>,
    proof: TaggedScalarSetRangeProof,
}

#[derive(Clone, PartialEq, Eq)]
struct FixedScalarRangeProofHint {
    path: SequencePath,
    domain: Arc<[Value]>,
    proof: FixedScalarRangeProof,
}

#[derive(Clone, PartialEq, Eq)]
struct FixedScalarVarProofHint {
    path: SequencePath,
    proof: FixedScalarRangeProof,
}

#[derive(Clone, PartialEq, Eq)]
struct SetBitmaskRangeProofHint {
    path: SequencePath,
    set_universe: Vec<FlatScalarValue>,
    /// Source-level invariant (e.g. TypeOK) that proves this universe is closed
    /// under every successor write. Carried so the inferred layout can mark the
    /// resulting bitmask universe as provably closed for flat-primary dispatch.
    invariant: Arc<str>,
}

#[derive(Clone, PartialEq, Eq)]
struct RecordSetBitmaskProofHint {
    path: SequencePath,
    /// Canonical finite record universe (every element a `Value::Record`).
    record_universe: Vec<Value>,
    /// Source-level invariant proving the record universe is closed under every
    /// successor write (e.g. `v \in SUBSET Messages`). Carried so the inferred
    /// `RecordSetBitmask` is marked `ProvenClosed`.
    invariant: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SequencePath(Vec<SequencePathStep>);

#[derive(Debug, Clone, PartialEq, Eq)]
enum SequencePathStep {
    Var(usize),
    HomogeneousRange(Arc<[Value]>),
    RecordField(Arc<str>),
    SequenceElement,
    SetElement,
}

impl SequencePath {
    fn root(var_idx: usize) -> Self {
        Self(vec![SequencePathStep::Var(var_idx)])
    }

    fn child(&self, step: SequencePathStep) -> Self {
        let mut path = self.0.clone();
        path.push(step);
        Self(path)
    }
}

impl LayoutInferenceContext {
    #[allow(clippy::too_many_arguments)]
    fn from_value_rows_and_sequence_layout_proofs<'a, I>(
        rows: I,
        sequence_proofs: &[SequenceCapacityProof],
        sequence_element_proofs: &[SequenceElementLayoutProof],
        sequence_fixed_domain_type_proofs: &[SequenceFixedDomainTypeProof],
        tagged_scalar_set_range_proofs: &[TaggedScalarSetRangeTypeProof],
        fixed_scalar_range_proofs: &[FixedScalarRangeTypeProof],
        set_bitmask_type_proofs: &[SetBitmaskTypeProof],
        set_bitmask_range_proofs: &[SetBitmaskRangeTypeProof],
        fixed_scalar_var_proofs: &[FixedScalarVarTypeProof],
        record_set_bitmask_type_proofs: &[RecordSetBitmaskTypeProof],
    ) -> Self
    where
        I: IntoIterator<Item = &'a [Value]>,
    {
        let rows: Vec<&[Value]> = rows.into_iter().collect();
        let mut context = LayoutInferenceContext {
            sequence_proofs: dedup_exact(
                sequence_proofs
                    .iter()
                    .flat_map(sequence_proof_to_hints)
                    .collect(),
            ),
            sequence_element_proofs: dedup_exact(
                sequence_element_proofs
                    .iter()
                    .flat_map(sequence_element_proof_to_hints)
                    .collect(),
            ),
            sequence_fixed_domain_type_proofs: dedup_exact(
                sequence_fixed_domain_type_proofs
                    .iter()
                    .flat_map(sequence_fixed_domain_type_proof_to_hints)
                    .collect(),
            ),
            tagged_scalar_set_range_proofs: dedup_exact(
                tagged_scalar_set_range_proofs
                    .iter()
                    .flat_map(tagged_scalar_set_range_type_proof_to_hints)
                    .collect(),
            ),
            fixed_scalar_range_proofs: dedup_exact(
                fixed_scalar_range_proofs
                    .iter()
                    .flat_map(fixed_scalar_range_type_proof_to_hints)
                    .collect(),
            ),
            fixed_scalar_var_proofs: dedup_exact(
                fixed_scalar_var_proofs
                    .iter()
                    .flat_map(fixed_scalar_var_type_proof_to_hints)
                    .collect(),
            ),
            set_bitmask_type_proofs: dedup_exact(
                set_bitmask_type_proofs
                    .iter()
                    .flat_map(set_bitmask_type_proof_to_hints)
                    .collect(),
            ),
            set_bitmask_range_proofs: dedup_exact(
                set_bitmask_range_proofs
                    .iter()
                    .flat_map(set_bitmask_range_type_proof_to_hints)
                    .collect(),
            ),
            record_set_bitmask_type_proofs: dedup_exact(
                record_set_bitmask_type_proofs
                    .iter()
                    .flat_map(record_set_bitmask_type_proof_to_hints)
                    .collect(),
            ),
            ..LayoutInferenceContext::default()
        };

        for row in &rows {
            for value in *row {
                collect_scalar_domain_candidates(value, &mut context.scalar_domain_candidates);
            }
        }

        loop {
            let before = context.sequence_hints.clone();
            let mut sequence_hints = before.clone();
            for row in &rows {
                for (var_idx, value) in row.iter().enumerate() {
                    let path = SequencePath::root(var_idx);
                    collect_sequence_hints(value, &context, &mut sequence_hints, &path);
                }
            }
            if sequence_hints == before {
                break;
            }
            context.sequence_hints = sequence_hints;
        }
        // WP-15 diagnosis surface (`TY_LAYOUT_PROOF_DEBUG=1`, debug-only): dump
        // the assembled per-path proof-hint tables so a missing/ambiguous hint
        // behind a `bound=ProvenInvariant`-only or `Observed` sequence is
        // attributable to the exact conflicting entries rather than guessed at.
        if std::env::var_os("TY_LAYOUT_PROOF_DEBUG").is_some_and(|v| v == "1") {
            for hint in &context.sequence_proofs {
                eprintln!(
                    "[layout-proof] capacity path={:?} max_len={} invariant={}",
                    hint.path.0, hint.max_len, hint.invariant
                );
            }
            for hint in &context.sequence_element_proofs {
                eprintln!(
                    "[layout-proof] element path={:?} layout={:?} invariant={}",
                    hint.path.0, hint.element_layout, hint.invariant
                );
            }
            for hint in &context.sequence_fixed_domain_type_proofs {
                eprintln!(
                    "[layout-proof] fixed-domain path={:?} |domain|={} layout={:?} invariant={}",
                    hint.path.0,
                    hint.domain.len(),
                    hint.element_layout,
                    hint.invariant
                );
            }
        }
        context
    }

    fn unique_scalar_domain_covering(
        &self,
        elements: &[FlatScalarValue],
    ) -> Option<Vec<FlatScalarValue>> {
        let matches: Vec<&Vec<FlatScalarValue>> = self
            .scalar_domain_candidates
            .iter()
            .filter(|candidate| {
                candidate.len() <= 63 && elements.iter().all(|elem| candidate.contains(elem))
            })
            .collect();
        let mut dominant = matches.iter().copied().filter(|candidate| {
            matches
                .iter()
                .all(|other| other.iter().all(|elem| candidate.contains(elem)))
        });
        let first = dominant.next()?;
        dominant.next().is_none().then(|| first.clone())
    }

    fn unique_sequence_hint(&self, path: &SequencePath) -> Option<&SequenceHint> {
        let mut hints = self.sequence_hints.iter().filter(|hint| &hint.path == path);
        let first = hints.next()?;
        hints.all(|hint| hint == first).then_some(first)
    }

    fn unique_sequence_proof(
        &self,
        path: &SequencePath,
        observed_len: usize,
    ) -> Option<&SequenceProofHint> {
        let mut hints = self
            .sequence_proofs
            .iter()
            .filter(|hint| &hint.path == path && hint.max_len >= observed_len);
        let first = hints.next()?;
        // WP-15 (`TY_FLAT_WRITE_ADMIT=1`): several checked invariants can prove
        // the SAME capacity at the same path (e.g. `TypeOK` and
        // `Inv == TypeOK /\ ...` both containing the identical clause), each
        // carrying a distinct proving-invariant label. Capacity proofs are
        // interchangeable when their `max_len` agrees — the flat encoding
        // depends only on the bound, never on which invariant proved it — so
        // under the opt-in the duplicates are judged on `max_len` alone.
        // Proofs that genuinely disagree on the bound still fail closed. With
        // the gate OFF this matches the historical full-equality veto exactly.
        if flat_write_admission_enabled() {
            hints
                .all(|hint| hint.max_len == first.max_len)
                .then_some(first)
        } else {
            hints.all(|hint| hint == first).then_some(first)
        }
    }

    fn unique_sequence_element_proof(
        &self,
        path: &SequencePath,
    ) -> Option<&SequenceElementProofHint> {
        // The same element layout at a path can be proven by several independent
        // sources — the writer relation, a SUBSET/scalar-range type invariant,
        // multiple configured invariants — each carrying a distinct `invariant`
        // diagnostic label (and, for set bitmasks, a distinct `ProvenClosed`
        // closure-invariant tag). These are interchangeable for flat-state
        // encoding, so uniqueness is judged on the *structural* element layout
        // (proving-invariant labels normalized away), not the full hint. Hints
        // that genuinely disagree on the encoding still fail closed (None).
        let mut hints = self
            .sequence_element_proofs
            .iter()
            .filter(|hint| &hint.path == path);
        let first = hints.next()?;
        let first_fingerprint = flat_value_layout_structural_fingerprint(&first.element_layout);
        hints
            .all(|hint| {
                flat_value_layout_structural_fingerprint(&hint.element_layout) == first_fingerprint
            })
            .then_some(first)
    }

    fn unique_sequence_fixed_domain_type_proof(
        &self,
        path: &SequencePath,
    ) -> Option<&SequenceFixedDomainTypeProofHint> {
        let mut hints = self
            .sequence_fixed_domain_type_proofs
            .iter()
            .filter(|hint| &hint.path == path);
        let first = hints.next()?;
        // WP-15 (`TY_FLAT_WRITE_ADMIT=1`): a spec that checks several
        // invariants proving the SAME `v \in [1..N -> T]` fixed-domain type
        // fact (MCBakery checks `TypeOK` AND `Inv == TypeOK /\ IInv`, so every
        // clause is collected once per invariant) produces hints that are
        // identical except for the proving-invariant label. Those are
        // interchangeable for flat-state encoding — the encoding depends only
        // on `(domain, element layout)` — so under the opt-in the duplicates
        // are judged structurally, with the label (and any closure-invariant
        // tag inside the element layout) normalized away, mirroring the rule
        // `unique_sequence_element_proof` has always used. Hints that
        // genuinely disagree on domain or element layout still fail closed.
        // With the gate OFF this matches the historical full-equality veto
        // exactly (label-only duplicates stay ambiguous and the caller falls
        // back to the weaker `ProvenInvariant`/`Observed` evidence).
        if flat_write_admission_enabled() {
            let first_fingerprint =
                sequence_type_layout_proof_structural_fingerprint(&first.element_layout);
            hints
                .all(|hint| {
                    hint.domain == first.domain
                        && sequence_type_layout_proof_structural_fingerprint(&hint.element_layout)
                            == first_fingerprint
                })
                .then_some(first)
        } else {
            hints.all(|hint| hint == first).then_some(first)
        }
    }

    fn unique_tagged_scalar_set_range_proof(
        &self,
        path: &SequencePath,
        domain: &[Value],
        value_types: &[SlotType],
    ) -> Option<&TaggedScalarSetRangeProofHint> {
        let mut hints = self.tagged_scalar_set_range_proofs.iter().filter(|hint| {
            &hint.path == path
                && hint.domain.as_ref() == domain
                && value_types
                    .iter()
                    .all(|value_type| *value_type == hint.proof.scalar_type())
        });
        let first = hints.next()?;
        hints
            .all(|hint| tagged_scalar_set_range_proof_compatible(hint, first))
            .then_some(first)
    }

    fn unique_tagged_scalar_set_range_proof_for_values(
        &self,
        path: &SequencePath,
        domain: &[Value],
        values: &[&Value],
    ) -> Option<&TaggedScalarSetRangeProofHint> {
        let mut hints = self.tagged_scalar_set_range_proofs.iter().filter(|hint| {
            &hint.path == path
                && hint.domain.as_ref() == domain
                && values
                    .iter()
                    .all(|value| value_fits_tagged_scalar_set_range_proof(value, &hint.proof))
        });
        let first = hints.next()?;
        hints
            .all(|hint| tagged_scalar_set_range_proof_compatible(hint, first))
            .then_some(first)
    }

    fn unique_fixed_scalar_range_proof(
        &self,
        path: &SequencePath,
        domain: &[Value],
        value_types: &[SlotType],
    ) -> Option<&FixedScalarRangeProofHint> {
        let hints: Vec<&FixedScalarRangeProofHint> = self
            .fixed_scalar_range_proofs
            .iter()
            .filter(|hint| {
                &hint.path == path
                    && hint.domain.as_ref() == domain
                    && !value_types.is_empty()
                    && value_types
                        .iter()
                        .all(|value_type| *value_type == hint.proof.scalar_type())
            })
            .collect();
        let (&first, rest) = hints.split_first()?;
        if rest
            .iter()
            .all(|hint| fixed_scalar_range_proof_compatible(hint, first))
        {
            return Some(first);
        }
        // Universe-only disagreement (H6 mutation robustness): a TypeOK-derived
        // proof carries the CHECKED invariant's universe, while the Init/Next
        // writer-closure proof carries every value a writer can actually store.
        // A buggy/mutated spec whose writer stores a value outside TypeOK makes
        // the two universes diverge — the writer-closure universe is then a
        // strict SUPERSET and is the unconditionally sound "reachable values"
        // set. Pick the hint whose universe contains every other hint's
        // universe: the one-slot interned-NameId encoding is total and
        // injective over the whole sort, so a WIDER universe can never
        // mis-encode, and the out-of-TypeOK value is caught by the checked
        // invariant itself (semantic wall, not an encode-time wall). If the
        // universes are incomparable (no superset exists), fail closed as
        // before. All hints here already agree on path, domain, and scalar
        // type via the filter above.
        hints.iter().copied().find(|candidate| {
            hints.iter().all(|other| {
                other
                    .proof
                    .scalar_universe()
                    .iter()
                    .all(|value| candidate.proof.scalar_universe().contains(value))
            })
        })
    }

    /// Find a unique proof that the value at `path` is itself drawn from a
    /// finite homogeneous scalar universe of `value_type` (e.g. a state variable
    /// `v \in {"a", "b"}`). Used to upgrade a bare scalar-string/model-value var
    /// into a primary-flat `FixedScalar` slot. Returns `None` unless exactly one
    /// matching proof (with the right scalar type and the observed value present
    /// in its universe) covers the path.
    fn unique_fixed_scalar_var_proof(
        &self,
        path: &SequencePath,
        value_type: SlotType,
        observed: &FlatScalarValue,
    ) -> Option<&FixedScalarVarProofHint> {
        let mut hints = self.fixed_scalar_var_proofs.iter().filter(|hint| {
            &hint.path == path
                && hint.proof.scalar_type() == value_type
                && hint
                    .proof
                    .scalar_universe()
                    .iter()
                    .any(|candidate| candidate == observed)
        });
        let first = hints.next()?;
        hints
            .all(|hint| fixed_scalar_var_proof_compatible(hint, first))
            .then_some(first)
    }

    fn unique_set_bitmask_range_proof(
        &self,
        path: &SequencePath,
    ) -> Option<&SetBitmaskRangeProofHint> {
        let mut hints = self
            .set_bitmask_range_proofs
            .iter()
            .filter(|hint| &hint.path == path);
        let first = hints.next()?;
        hints
            .all(|hint| hint.set_universe == first.set_universe)
            .then_some(first)
    }

    fn unique_set_bitmask_type_proof(
        &self,
        path: &SequencePath,
    ) -> Option<&SetBitmaskRangeProofHint> {
        let mut hints = self
            .set_bitmask_type_proofs
            .iter()
            .filter(|hint| &hint.path == path);
        let first = hints.next()?;
        hints
            .all(|hint| hint.set_universe == first.set_universe)
            .then_some(first)
    }

    fn unique_record_set_bitmask_type_proof(
        &self,
        path: &SequencePath,
    ) -> Option<&RecordSetBitmaskProofHint> {
        let mut hints = self
            .record_set_bitmask_type_proofs
            .iter()
            .filter(|hint| &hint.path == path);
        let first = hints.next()?;
        hints
            .all(|hint| hint.record_universe == first.record_universe)
            .then_some(first)
    }
}

fn tagged_scalar_set_range_proof_compatible(
    left: &TaggedScalarSetRangeProofHint,
    right: &TaggedScalarSetRangeProofHint,
) -> bool {
    left.path == right.path
        && left.domain == right.domain
        && left.proof.scalar_type() == right.proof.scalar_type()
        && left.proof.set_universe() == right.proof.set_universe()
}

fn fixed_scalar_range_proof_compatible(
    left: &FixedScalarRangeProofHint,
    right: &FixedScalarRangeProofHint,
) -> bool {
    left.path == right.path
        && left.domain == right.domain
        && left.proof.scalar_type() == right.proof.scalar_type()
        && left.proof.scalar_universe() == right.proof.scalar_universe()
}

fn fixed_scalar_var_proof_compatible(
    left: &FixedScalarVarProofHint,
    right: &FixedScalarVarProofHint,
) -> bool {
    left.path == right.path
        && left.proof.scalar_type() == right.proof.scalar_type()
        && left.proof.scalar_universe() == right.proof.scalar_universe()
}

fn value_fits_tagged_scalar_set_range_proof(
    value: &Value,
    proof: &TaggedScalarSetRangeProof,
) -> bool {
    if is_scalar_value(value) {
        return slot_type_from_value(value) == proof.scalar_type();
    }

    let Value::Set(set) = value else {
        return false;
    };
    set.iter().all(|elem| {
        proof
            .set_universe()
            .iter()
            .any(|candidate| flat_scalar_to_value(candidate) == *elem)
    })
}

fn dedup_exact<T: PartialEq>(hints: Vec<T>) -> Vec<T> {
    let mut deduped = Vec::with_capacity(hints.len());
    for hint in hints {
        if !deduped.iter().any(|existing| existing == &hint) {
            deduped.push(hint);
        }
    }
    deduped
}

fn sequence_proof_to_hints(proof: &SequenceCapacityProof) -> Vec<SequenceProofHint> {
    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| SequenceProofHint {
            path,
            max_len: proof.max_len,
            invariant: Arc::clone(&proof.invariant),
            heuristic: proof.heuristic,
        })
        .collect()
}

fn sequence_element_proof_to_hints(
    proof: &SequenceElementLayoutProof,
) -> Vec<SequenceElementProofHint> {
    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| SequenceElementProofHint {
            path,
            element_layout: proof.element_layout.clone(),
            invariant: Arc::clone(&proof.invariant),
        })
        .collect()
}

fn sequence_fixed_domain_type_proof_to_hints(
    proof: &SequenceFixedDomainTypeProof,
) -> Vec<SequenceFixedDomainTypeProofHint> {
    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| SequenceFixedDomainTypeProofHint {
            path,
            domain: Arc::clone(&proof.domain),
            element_layout: proof.element_layout.clone(),
            invariant: Arc::clone(&proof.invariant),
        })
        .collect()
}

fn tagged_scalar_set_range_type_proof_to_hints(
    proof: &TaggedScalarSetRangeTypeProof,
) -> Vec<TaggedScalarSetRangeProofHint> {
    let Ok(range_proof) = TaggedScalarSetRangeProof::new(
        proof.scalar_type,
        proof.set_universe.clone(),
        Arc::clone(&proof.invariant),
    ) else {
        return Vec::new();
    };

    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| TaggedScalarSetRangeProofHint {
            path,
            domain: Arc::clone(&proof.domain),
            proof: range_proof.clone(),
        })
        .collect()
}

fn fixed_scalar_range_type_proof_to_hints(
    proof: &FixedScalarRangeTypeProof,
) -> Vec<FixedScalarRangeProofHint> {
    let Ok(range_proof) = FixedScalarRangeProof::new(
        proof.scalar_type,
        proof.scalar_universe.clone(),
        Arc::clone(&proof.invariant),
    ) else {
        return Vec::new();
    };

    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| FixedScalarRangeProofHint {
            path,
            domain: Arc::clone(&proof.domain),
            proof: range_proof.clone(),
        })
        .collect()
}

fn fixed_scalar_var_type_proof_to_hints(
    proof: &FixedScalarVarTypeProof,
) -> Vec<FixedScalarVarProofHint> {
    let Ok(range_proof) = FixedScalarRangeProof::new(
        proof.scalar_type,
        proof.scalar_universe.clone(),
        Arc::clone(&proof.invariant),
    ) else {
        return Vec::new();
    };

    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| FixedScalarVarProofHint {
            path,
            proof: range_proof.clone(),
        })
        .collect()
}

fn set_bitmask_range_type_proof_to_hints(
    proof: &SetBitmaskRangeTypeProof,
) -> Vec<SetBitmaskRangeProofHint> {
    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .flat_map(|parent_path| {
            let mut paths =
                vec![
                    parent_path.child(SequencePathStep::HomogeneousRange(Arc::clone(
                        &proof.domain,
                    ))),
                ];
            if contiguous_int_value_domain(&proof.domain)
                .is_some_and(|(lo, len)| domain_is_one_based_int_interval(&proof.domain, lo, len))
            {
                paths.push(parent_path.child(SequencePathStep::SequenceElement));
            }
            paths.into_iter().map(|path| SetBitmaskRangeProofHint {
                path,
                set_universe: proof.set_universe.clone(),
                invariant: Arc::clone(&proof.invariant),
            })
        })
        .collect()
}

fn set_bitmask_type_proof_to_hints(proof: &SetBitmaskTypeProof) -> Vec<SetBitmaskRangeProofHint> {
    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| SetBitmaskRangeProofHint {
            path,
            set_universe: proof.set_universe.clone(),
            invariant: Arc::clone(&proof.invariant),
        })
        .collect()
}

fn record_set_bitmask_type_proof_to_hints(
    proof: &RecordSetBitmaskTypeProof,
) -> Vec<RecordSetBitmaskProofHint> {
    sequence_capacity_path_aliases(proof.var_idx, &proof.path)
        .into_iter()
        .map(|path| RecordSetBitmaskProofHint {
            path,
            record_universe: proof.record_universe.clone(),
            invariant: Arc::clone(&proof.invariant),
        })
        .collect()
}

fn domain_is_one_based_int_interval(domain: &[Value], lo: i64, len: usize) -> bool {
    if domain.is_empty() || len == 0 || lo != 1 || domain.len() != len {
        return false;
    }
    domain
        .iter()
        .enumerate()
        .all(|(index, value)| matches!(value, Value::SmallInt(n) if *n == index as i64 + 1))
}

fn contiguous_int_value_domain(domain: &[Value]) -> Option<(i64, usize)> {
    let mut ints: Vec<i64> = domain
        .iter()
        .map(|value| match value {
            Value::SmallInt(n) => Some(*n),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if ints.is_empty() {
        return None;
    }
    ints.sort_unstable();
    ints.dedup();
    if ints.len() != domain.len() {
        return None;
    }
    let lo = ints[0];
    let hi = *ints.last()?;
    let len = usize::try_from(hi.checked_sub(lo)? + 1).ok()?;
    (len == ints.len()).then_some((lo, len))
}

fn sequence_capacity_path_aliases(
    var_idx: usize,
    proof_path: &[SequenceCapacityPathStep],
) -> Vec<SequencePath> {
    let mut paths = vec![vec![SequencePathStep::Var(var_idx)]];
    for step in proof_path {
        let mut next_paths = Vec::with_capacity(paths.len() * 2);
        match step {
            SequenceCapacityPathStep::HomogeneousRange { domain } => {
                let sequence_shaped_domain = contiguous_int_value_domain(domain)
                    .is_some_and(|(lo, len)| domain_is_one_based_int_interval(domain, lo, len));
                for path in paths {
                    let mut function_path = path.clone();
                    function_path.push(SequencePathStep::HomogeneousRange(Arc::clone(domain)));
                    next_paths.push(function_path);

                    if sequence_shaped_domain {
                        let mut sequence_path = path;
                        sequence_path.push(SequencePathStep::SequenceElement);
                        next_paths.push(sequence_path);
                    }
                }
            }
            SequenceCapacityPathStep::RecordField(name) => {
                for mut path in paths {
                    path.push(SequencePathStep::RecordField(Arc::clone(name)));
                    next_paths.push(path);
                }
            }
            SequenceCapacityPathStep::SequenceElement => {
                for mut path in paths {
                    path.push(SequencePathStep::SequenceElement);
                    next_paths.push(path);
                }
            }
        }
        paths = next_paths;
    }
    dedup_exact(paths.into_iter().map(SequencePath).collect())
}

fn sequence_bound_evidence_for_path(
    context: &LayoutInferenceContext,
    path: &SequencePath,
    observed_len: usize,
    element_invariant: Option<&Arc<str>>,
) -> (SequenceBoundEvidence, usize) {
    if let Some(proof) = context.unique_sequence_proof(path, observed_len) {
        let bound = if proof.heuristic {
            // Unproven element-universe heuristic (backstop-guarded): produce the
            // HeuristicUniverseCapacity bound regardless of whether the element
            // layout is separately proven. It stays non-proven (`is_proven()` is
            // false), so the native lowering bridges it to `capacity_proven=false`
            // and only the flat STORAGE path — with the SequenceLengthExceedsCapacity
            // overflow backstop — relies on `proof.max_len`.
            SequenceBoundEvidence::HeuristicUniverseCapacity {
                universe_invariant: Arc::clone(&proof.invariant),
            }
        } else if let Some(element_invariant) = element_invariant {
            SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                invariant: Arc::clone(&proof.invariant),
                element_invariant: Arc::clone(element_invariant),
            }
        } else {
            SequenceBoundEvidence::ProvenInvariant {
                invariant: Arc::clone(&proof.invariant),
            }
        };
        (bound, proof.max_len)
    } else {
        (SequenceBoundEvidence::Observed, observed_len)
    }
}

fn fixed_domain_sequence_layout_for_path(
    context: &LayoutInferenceContext,
    path: &SequencePath,
    observed_len: usize,
    element_layout: &FlatValueLayout,
) -> Option<(SequenceBoundEvidence, usize, FlatValueLayout)> {
    let proof = context.unique_sequence_fixed_domain_type_proof(path)?;
    if proof.domain.is_empty() {
        return None;
    }
    let element_layout =
        sequence_type_layout_proof_apply_flat_layout(&proof.element_layout, element_layout)?;
    (observed_len == proof.domain.len()).then(|| {
        (
            SequenceBoundEvidence::FixedDomainTypeLayout {
                invariant: Arc::clone(&proof.invariant),
            },
            proof.domain.len(),
            element_layout,
        )
    })
}

pub(crate) fn collect_sequence_element_layout_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<SequenceElementLayoutProof>,
) {
    collect_sequence_element_layout_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        Some(op_defs),
        Some(op_replacements),
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

/// Infer top-level sequence element layout proofs from the closed `Init`/`Next`
/// writer relation. This is intentionally conservative: a variable is proven
/// only when every collected write is an empty sequence, preserves a known
/// sequence, or constructs a sequence from elements with one compatible flat
/// layout. Unsupported writes fail closed for that variable.
pub(crate) fn collect_sequence_element_layout_writer_proofs_with_ops(
    init_expr: &Expr,
    next_expr: &Expr,
    source: &str,
    registry: &VarRegistry,
    seed_values: &[Value],
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    seeded_proofs: &[SequenceElementLayoutProof],
    out: &mut Vec<SequenceElementLayoutProof>,
) {
    let seed_slot_types: BTreeMap<usize, SlotType> = seed_values
        .iter()
        .enumerate()
        .filter(|&(_idx, value)| is_scalar_value(value))
        .map(|(idx, value)| (idx, slot_type_from_value(value)))
        .collect();
    let mut assignments = SequenceWriterAssignments::default();
    let mut scope = WriterExprScope::default();
    collect_sequence_writer_assignments(
        init_expr,
        registry,
        constants,
        op_defs,
        Some(op_replacements),
        &mut scope,
        &mut BTreeSet::new(),
        &mut assignments,
    );
    collect_sequence_writer_assignments(
        next_expr,
        registry,
        constants,
        op_defs,
        Some(op_replacements),
        &mut scope,
        &mut BTreeSet::new(),
        &mut assignments,
    );

    let mut candidates: BTreeMap<usize, FlatValueLayout> = seed_values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| sequence_value_element_layout(value).map(|layout| (idx, layout)))
        .collect();
    let mut seeded_root_vars = BTreeSet::new();
    for proof in seeded_proofs.iter().filter(|proof| proof.path.is_empty()) {
        seeded_root_vars.insert(proof.var_idx);
        merge_writer_candidate(&mut candidates, proof.var_idx, proof.element_layout.clone());
    }

    let mut invalid = assignments.invalid.clone();
    let max_iterations = assignments.writes.len().saturating_mul(4).saturating_add(8);
    let mut converged = false;
    for _ in 0..max_iterations {
        let mut changed = false;
        let mut newly_invalid = BTreeSet::new();
        for (&var_idx, writes) in &assignments.writes {
            if invalid.contains(&var_idx) {
                continue;
            }
            let mut var_layout = candidates.get(&var_idx).cloned();
            for write in writes {
                match sequence_writer_expr_element_layout(
                    &write.expr,
                    registry,
                    constants,
                    op_defs,
                    Some(op_replacements),
                    &write.scope,
                    &seed_slot_types,
                    &candidates,
                    &mut BTreeSet::new(),
                ) {
                    Some(None) => {}
                    Some(Some(layout)) => {
                        var_layout = Some(if let Some(existing) = var_layout.as_ref() {
                            match merge_flat_value_layouts(existing, &layout) {
                                Some(merged) => merged,
                                None => {
                                    newly_invalid.insert(var_idx);
                                    break;
                                }
                            }
                        } else {
                            layout
                        });
                    }
                    None => {
                        continue;
                    }
                }
            }
            if let Some(layout) = var_layout {
                let before = candidates.get(&var_idx).cloned();
                merge_writer_candidate(&mut candidates, var_idx, layout);
                if candidates.get(&var_idx).cloned() != before {
                    changed = true;
                }
            }
        }
        if !newly_invalid.is_empty() {
            for var_idx in newly_invalid {
                invalid.insert(var_idx);
                candidates.remove(&var_idx);
            }
            changed = true;
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return;
    }

    for (&var_idx, element_layout) in &candidates {
        if invalid.contains(&var_idx) || seeded_root_vars.contains(&var_idx) {
            continue;
        }
        let Some(writes) = assignments.writes.get(&var_idx) else {
            continue;
        };
        if !sequence_writer_element_writes_validate(
            writes,
            registry,
            constants,
            op_defs,
            Some(op_replacements),
            &seed_slot_types,
            &candidates,
            element_layout,
        ) {
            continue;
        }
        push_sequence_element_layout_proof(
            out,
            SequenceElementLayoutProof {
                var_idx,
                path: Vec::new(),
                element_layout: element_layout.clone(),
                invariant: Arc::from(source),
            },
        );
    }
}

/// Derive sequence *element*-layout proofs for function ranges proven by a
/// finite-type invariant, from already-collected `SetBitmaskRangeTypeProof`s
/// (set-valued ranges, e.g. `v \in [1..N -> SUBSET universe]`) and
/// `FixedScalarRangeTypeProof`s (scalar ranges, e.g. `v \in [1..N -> 1..N]`).
///
/// When a state variable's function *range* is proven by a checked type
/// invariant, and the function *domain* is a one-based integer interval `1..N`
/// (so the function is stored as a TLA sequence whose elements are exactly the
/// per-key range values), every sequence element is — in every reachable state —
/// drawn from the proven range type:
///
/// * `SUBSET universe`  →  `SetBitmask { ProvenClosed }` element layout, and
/// * a finite scalar set →  `Scalar(slot_type)` element layout.
///
/// Soundness: unlike a sampled element layout (which only describes the elements
/// seen so far, and could collide a later out-of-range element), each proof is
/// derived from a *checked* type invariant on the function range, enforced on
/// every reachable state — including the empty-at-INIT case (`{} \subseteq
/// universe`, and `nxt = [Procs |-> 1]`). It is a function-*range* fact applied
/// to function-*range* elements, so the flat serializer sees the expected kind
/// (`Value::Set` for SetBitmask, a scalar for `Scalar`) on every reachable state
/// and never panics. Set proofs are only emitted when the universe fits the
/// 63-bit bitmask width; scalar proofs only for plain `i64` slot types (`Int`,
/// `Bool`) whose one-word encoding cannot collide. Any other shape fails closed.
pub(crate) fn derive_set_valued_sequence_element_proofs(
    set_bitmask_range_proofs: &[SetBitmaskRangeTypeProof],
    fixed_scalar_range_proofs: &[FixedScalarRangeTypeProof],
    out: &mut Vec<SequenceElementLayoutProof>,
) {
    // The same range shape can be proven by several configured invariants (e.g.
    // both `TypeOK` and `Inv == TypeOK /\ ...`). Each carries a distinct
    // `invariant` label (and, for sets, a distinct `ProvenClosed { invariant }`
    // closure tag), which would otherwise emit element proofs differing only in
    // the proving-invariant string — and the downstream
    // `unique_sequence_element_proof` uniqueness check would then reject both as
    // ambiguous. Collect at most one proof per `(var, path)` location, keyed by
    // the *structural* element layout (ignoring the proving-invariant label).
    //
    // If two proofs at the same location disagree *structurally* (different
    // universe or slot type — which would be a genuinely conflicting type fact),
    // fail closed: drop the location entirely rather than pick one arbitrarily.
    struct Candidate {
        var_idx: usize,
        path: Vec<SequenceCapacityPathStep>,
        element_layout: FlatValueLayout,
        invariant: Arc<str>,
        // Structural fingerprint used for conflict detection (closure-invariant
        // label stripped).
        fingerprint: FlatValueLayout,
        conflicting: bool,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut emit = |var_idx: usize,
                    path: &[SequenceCapacityPathStep],
                    element_layout: FlatValueLayout,
                    invariant: &Arc<str>| {
        let fingerprint = flat_value_layout_structural_fingerprint(&element_layout);
        if let Some(existing) = candidates
            .iter_mut()
            .find(|c| c.var_idx == var_idx && c.path == path)
        {
            if existing.fingerprint != fingerprint {
                existing.conflicting = true;
            }
            return;
        }
        candidates.push(Candidate {
            var_idx,
            path: path.to_vec(),
            element_layout,
            invariant: Arc::clone(invariant),
            fingerprint,
            conflicting: false,
        });
    };

    for proof in set_bitmask_range_proofs {
        // The function range must be a finite scalar `SUBSET universe` that fits
        // the fixed-width bitmask slot.
        if proof.set_universe.is_empty() || proof.set_universe.len() > 63 {
            continue;
        }
        // The function domain must be a one-based integer interval `1..N`, so the
        // function is stored as a TLA sequence whose elements are exactly the
        // per-key SUBSET range values. (This mirrors the sequence-element alias
        // condition in `set_bitmask_range_type_proof_to_hints`.)
        if !range_proof_domain_is_sequence_shaped(&proof.domain) {
            continue;
        }
        let mut universe = proof.set_universe.clone();
        universe.sort();
        universe.dedup();
        let element_layout = FlatValueLayout::SetBitmask {
            universe,
            universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                invariant: Arc::clone(&proof.invariant),
            },
        };
        emit(proof.var_idx, &proof.path, element_layout, &proof.invariant);
    }

    for proof in fixed_scalar_range_proofs {
        // Only plain `i64` slot types (Int/Bool) are safe as a bare scalar
        // sequence-element slot: their one-word encoding cannot collide with a
        // later differently-typed value (string/model-value scalars overlap
        // ordinary integer slots, so they stay fail-closed here).
        if !matches!(proof.scalar_type, SlotType::Int | SlotType::Bool) {
            continue;
        }
        if !range_proof_domain_is_sequence_shaped(&proof.domain) {
            continue;
        }
        let element_layout = FlatValueLayout::Scalar(proof.scalar_type);
        emit(proof.var_idx, &proof.path, element_layout, &proof.invariant);
    }

    for candidate in candidates {
        if candidate.conflicting {
            continue;
        }
        push_sequence_element_layout_proof(
            out,
            SequenceElementLayoutProof {
                var_idx: candidate.var_idx,
                path: candidate.path,
                element_layout: candidate.element_layout,
                invariant: candidate.invariant,
            },
        );
    }
}

/// Structural fingerprint of a `FlatValueLayout` used to detect *conflicting*
/// proven element layouts independent of which invariant proved them: the
/// `ProvenClosed { invariant }` closure label is normalized away so that the
/// same universe proven by `TypeOK` and by `Inv` compares equal. Two layouts
/// with the same fingerprint are interchangeable for flat-state encoding.
fn flat_value_layout_structural_fingerprint(layout: &FlatValueLayout) -> FlatValueLayout {
    match layout {
        FlatValueLayout::SetBitmask { universe, .. } => FlatValueLayout::SetBitmask {
            universe: universe.clone(),
            universe_closure: SetBitmaskUniverseClosure::Sampled,
        },
        FlatValueLayout::RecordSetBitmask { universe, .. } => FlatValueLayout::RecordSetBitmask {
            universe: universe.clone(),
            universe_closure: SetBitmaskUniverseClosure::Sampled,
        },
        other => other.clone(),
    }
}

/// Structural fingerprint of a [`SequenceTypeLayoutProof`], the
/// fixed-domain-proof analogue of [`flat_value_layout_structural_fingerprint`]:
/// `Flat` leaves have their closure-invariant labels normalized away so two
/// proofs of the same shape proven by different invariants compare equal, while
/// genuinely different domains/element layouts stay distinct. Used by the
/// `TY_FLAT_WRITE_ADMIT` structural-uniqueness rule in
/// `unique_sequence_fixed_domain_type_proof`. Normalization is deliberately the
/// SAME shallow rule the element-proof lookup uses (nested closure labels are
/// not normalized), so a deeper label mismatch keeps failing closed rather than
/// being papered over.
fn sequence_type_layout_proof_structural_fingerprint(
    proof: &SequenceTypeLayoutProof,
) -> SequenceTypeLayoutProof {
    match proof {
        SequenceTypeLayoutProof::Flat(layout) => {
            SequenceTypeLayoutProof::Flat(flat_value_layout_structural_fingerprint(layout))
        }
        SequenceTypeLayoutProof::Sequence { element_layout } => SequenceTypeLayoutProof::Sequence {
            element_layout: Box::new(sequence_type_layout_proof_structural_fingerprint(
                element_layout,
            )),
        },
        SequenceTypeLayoutProof::FixedDomainSequence {
            max_len,
            element_layout,
        } => SequenceTypeLayoutProof::FixedDomainSequence {
            max_len: *max_len,
            element_layout: Box::new(sequence_type_layout_proof_structural_fingerprint(
                element_layout,
            )),
        },
    }
}

/// True when a function-range type proof's domain is a one-based integer
/// interval `1..N`, so the function is stored as a TLA sequence whose elements
/// are exactly the per-key range values.
fn range_proof_domain_is_sequence_shaped(domain: &[Value]) -> bool {
    contiguous_int_value_domain(domain)
        .is_some_and(|(lo, len)| domain_is_one_based_int_interval(domain, lo, len))
}

#[allow(clippy::too_many_arguments)]
fn sequence_writer_element_writes_validate(
    writes: &[SequenceWriterWrite],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    seed_slot_types: &BTreeMap<usize, SlotType>,
    candidates: &BTreeMap<usize, FlatValueLayout>,
    expected: &FlatValueLayout,
) -> bool {
    writes.iter().all(|write| {
        match sequence_writer_expr_element_layout(
            &write.expr,
            registry,
            constants,
            op_defs,
            op_replacements,
            &write.scope,
            seed_slot_types,
            candidates,
            &mut BTreeSet::new(),
        ) {
            Some(None) => true,
            Some(Some(layout)) => merge_flat_value_layouts(expected, &layout).is_some(),
            None => false,
        }
    })
}

/// Infer top-level sequence capacity proofs from the closed `Init`/`Next`
/// writer relation. This complements configured invariant proofs for specs
/// that encode a bounded sequence as an inductive writer discipline rather
/// than as a checked type invariant.
pub(crate) fn collect_sequence_capacity_writer_proofs_with_ops(
    init_expr: &Expr,
    next_expr: &Expr,
    source: &str,
    registry: &VarRegistry,
    seed_values: &[Value],
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<SequenceCapacityProof>,
) {
    let mut assignments = SequenceWriterAssignments::default();
    let mut scope = WriterExprScope::default();
    collect_sequence_writer_assignments(
        init_expr,
        registry,
        constants,
        op_defs,
        Some(op_replacements),
        &mut scope,
        &mut BTreeSet::new(),
        &mut assignments,
    );
    collect_sequence_writer_assignments(
        next_expr,
        registry,
        constants,
        op_defs,
        Some(op_replacements),
        &mut scope,
        &mut BTreeSet::new(),
        &mut assignments,
    );

    let mut candidates: BTreeMap<usize, usize> = seed_values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| sequence_value_len(value).map(|len| (idx, len)))
        .collect();
    let invalid = assignments.invalid.clone();
    let max_iterations = assignments.writes.len().saturating_mul(4).saturating_add(8);
    let mut converged = false;
    for _ in 0..max_iterations {
        let mut changed = false;
        for (&var_idx, writes) in &assignments.writes {
            if invalid.contains(&var_idx) {
                continue;
            }
            let mut var_capacity = candidates.get(&var_idx).copied();
            for write in writes {
                match sequence_writer_expr_capacity(
                    &write.expr,
                    registry,
                    constants,
                    op_defs,
                    Some(op_replacements),
                    &write.scope,
                    &candidates,
                    &mut BTreeSet::new(),
                ) {
                    Some(capacity) => {
                        var_capacity = Some(var_capacity.map_or(capacity, |max| max.max(capacity)));
                    }
                    None => {
                        continue;
                    }
                }
            }
            if let Some(capacity) = var_capacity {
                if candidates.insert(var_idx, capacity) != Some(capacity) {
                    changed = true;
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return;
    }

    for (&var_idx, &max_len) in &candidates {
        if invalid.contains(&var_idx) || !assignments.writes.contains_key(&var_idx) {
            continue;
        }
        let Some(writes) = assignments.writes.get(&var_idx) else {
            continue;
        };
        if !sequence_writer_capacity_writes_validate(
            writes,
            registry,
            constants,
            op_defs,
            Some(op_replacements),
            &candidates,
            max_len,
        ) {
            continue;
        }
        if out
            .iter()
            .any(|proof| proof.var_idx == var_idx && proof.path.is_empty())
        {
            continue;
        }
        push_sequence_capacity_proof(
            out,
            SequenceCapacityProof {
                var_idx,
                path: Vec::new(),
                max_len,
                invariant: Arc::from(source),
                heuristic: false,
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn sequence_writer_capacity_writes_validate(
    writes: &[SequenceWriterWrite],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    candidates: &BTreeMap<usize, usize>,
    expected_max_len: usize,
) -> bool {
    writes.iter().all(|write| {
        sequence_writer_expr_capacity(
            &write.expr,
            registry,
            constants,
            op_defs,
            op_replacements,
            &write.scope,
            candidates,
            &mut BTreeSet::new(),
        )
        .is_some_and(|capacity| capacity <= expected_max_len)
    })
}

fn sequence_value_len(value: &Value) -> Option<usize> {
    match value {
        Value::Seq(seq) => Some(seq.len()),
        Value::Tuple(elems) => Some(elems.len()),
        _ => None,
    }
}

fn merge_writer_candidate(
    candidates: &mut BTreeMap<usize, FlatValueLayout>,
    var_idx: usize,
    layout: FlatValueLayout,
) {
    if let Some(existing) = candidates.get(&var_idx).cloned() {
        if let Some(merged) = merge_flat_value_layouts(&existing, &layout) {
            candidates.insert(var_idx, merged);
        }
    } else {
        candidates.insert(var_idx, layout);
    }
}

fn sequence_value_element_layout(value: &Value) -> Option<FlatValueLayout> {
    match value {
        Value::Seq(seq) if !seq.is_empty() => {
            let mut layout = None;
            for value in seq.iter() {
                let value_layout = scalar_or_record_value_layout(value)?;
                layout = Some(if let Some(existing) = layout {
                    merge_flat_value_layouts(&existing, &value_layout)?
                } else {
                    value_layout
                });
            }
            layout
        }
        Value::Tuple(elems) if !elems.is_empty() => {
            let mut layout = None;
            for value in elems.iter() {
                let value_layout = scalar_or_record_value_layout(value)?;
                layout = Some(if let Some(existing) = layout {
                    merge_flat_value_layouts(&existing, &value_layout)?
                } else {
                    value_layout
                });
            }
            layout
        }
        _ => None,
    }
}

fn scalar_or_record_value_layout(value: &Value) -> Option<FlatValueLayout> {
    if is_scalar_value(value) {
        return Some(FlatValueLayout::Scalar(slot_type_from_value(value)));
    }
    match value {
        Value::Record(record) => {
            let mut fields = Vec::with_capacity(record.len());
            for (name_id, value) in record.iter() {
                let field_name = tla_core::resolve_name_id(name_id);
                fields.push((
                    intern_name(&field_name),
                    field_name,
                    scalar_or_record_value_layout(value)?,
                ));
            }
            // Canonical record field order: field-name STRING, matching
            // RecordValue's storage order (NameId order is run-dependent).
            fields.sort_by(|a, b| a.1.cmp(&b.1));
            Some(FlatValueLayout::Record {
                field_names: fields.iter().map(|(_, name, _)| Arc::clone(name)).collect(),
                field_layouts: fields.into_iter().map(|(_, _, layout)| layout).collect(),
            })
        }
        _ => None,
    }
}

#[derive(Default)]
struct SequenceWriterAssignments {
    writes: BTreeMap<usize, Vec<SequenceWriterWrite>>,
    invalid: BTreeSet<usize>,
}

#[derive(Clone)]
struct SequenceWriterWrite {
    expr: Expr,
    scope: WriterExprScope,
}

#[derive(Default, Clone)]
struct WriterExprScope {
    bindings: BTreeMap<String, Vec<Expr>>,
    int_bindings: BTreeMap<String, usize>,
    sequence_bindings: BTreeMap<String, Vec<SequenceWriterBinding>>,
    value_layout_bindings: BTreeMap<String, Vec<FlatValueLayout>>,
    len_upper_bounds: BTreeMap<usize, Vec<usize>>,
}

#[derive(Clone)]
struct SequenceWriterBinding {
    capacity: usize,
    element_layout: FlatValueLayout,
}

impl WriterExprScope {
    fn push(&mut self, name: String, expr: Expr) {
        self.bindings.entry(name).or_default().push(expr);
    }

    fn pop(&mut self, name: &str) {
        if let Some(stack) = self.bindings.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.bindings.remove(name);
            }
        }
    }

    fn get(&self, name: &str) -> Option<&Expr> {
        self.bindings.get(name).and_then(|stack| stack.last())
    }

    fn push_int(&mut self, name: String) {
        *self.int_bindings.entry(name).or_default() += 1;
    }

    fn pop_int(&mut self, name: &str) {
        if let Some(count) = self.int_bindings.get_mut(name) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.int_bindings.remove(name);
            }
        }
    }

    fn is_int(&self, name: &str) -> bool {
        self.int_bindings.get(name).copied().unwrap_or_default() > 0
    }

    fn push_sequence(&mut self, name: String, binding: SequenceWriterBinding) {
        self.sequence_bindings
            .entry(name)
            .or_default()
            .push(binding);
    }

    fn pop_sequence(&mut self, name: &str) {
        if let Some(stack) = self.sequence_bindings.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.sequence_bindings.remove(name);
            }
        }
    }

    fn sequence(&self, name: &str) -> Option<&SequenceWriterBinding> {
        self.sequence_bindings
            .get(name)
            .and_then(|stack| stack.last())
    }

    fn push_value_layout(&mut self, name: String, layout: FlatValueLayout) {
        self.value_layout_bindings
            .entry(name)
            .or_default()
            .push(layout);
    }

    fn pop_value_layout(&mut self, name: &str) {
        if let Some(stack) = self.value_layout_bindings.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.value_layout_bindings.remove(name);
            }
        }
    }

    fn value_layout(&self, name: &str) -> Option<&FlatValueLayout> {
        self.value_layout_bindings
            .get(name)
            .and_then(|stack| stack.last())
    }

    fn push_len_upper_bound(&mut self, var_idx: usize, bound: usize) {
        self.len_upper_bounds
            .entry(var_idx)
            .or_default()
            .push(bound);
    }

    fn len_upper_bound(&self, var_idx: usize) -> Option<usize> {
        self.len_upper_bounds
            .get(&var_idx)
            .and_then(|bounds| bounds.iter().copied().min())
    }
}

fn collect_sequence_writer_assignments(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &mut WriterExprScope,
    visiting: &mut BTreeSet<String>,
    out: &mut SequenceWriterAssignments,
) {
    match expr {
        Expr::And(left, right) => {
            let left_bounds = collect_writer_len_upper_bounds(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
            );
            let right_bounds = collect_writer_len_upper_bounds(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
            );
            let mut left_scope = scope.clone();
            for (var_idx, bound) in right_bounds {
                left_scope.push_len_upper_bound(var_idx, bound);
            }
            collect_sequence_writer_assignments(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                &mut left_scope,
                visiting,
                out,
            );
            let mut right_scope = scope.clone();
            for (var_idx, bound) in left_bounds {
                right_scope.push_len_upper_bound(var_idx, bound);
            }
            collect_sequence_writer_assignments(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                &mut right_scope,
                visiting,
                out,
            );
        }
        Expr::Or(left, right) => {
            collect_sequence_writer_assignments(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_sequence_writer_assignments(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::If(_, then_expr, else_expr) => {
            collect_sequence_writer_assignments(
                &then_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_sequence_writer_assignments(
                &else_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Exists(bounds, body) | Expr::Forall(bounds, body) => {
            let mut pushed_ints = Vec::new();
            let mut pushed_sequences = Vec::new();
            let int_domains = integer_bound_domains(bounds, constants, op_replacements);
            for bound in bounds {
                let Some(domain) = bound.domain.as_ref() else {
                    continue;
                };
                if bound.pattern.is_none()
                    && int_range_domain(&domain.node, constants, op_defs, op_replacements)
                {
                    scope.push_int(bound.name.node.clone());
                    pushed_ints.push(bound.name.node.clone());
                } else if bound.pattern.is_none() {
                    let binding = sequence_binding_from_writer_domain(
                        &domain.node,
                        constants,
                        op_replacements,
                        &int_domains,
                    );
                    if let Some(binding) = binding {
                        scope.push_sequence(bound.name.node.clone(), binding);
                        pushed_sequences.push(bound.name.node.clone());
                    }
                }
            }
            collect_sequence_writer_assignments(
                &body.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            for name in pushed_ints.into_iter().rev() {
                scope.pop_int(&name);
            }
            for name in pushed_sequences.into_iter().rev() {
                scope.pop_sequence(&name);
            }
        }
        Expr::Let(defs, body) => {
            let mut pushed = Vec::new();
            for def in defs {
                if def.params.is_empty() {
                    scope.push(def.name.node.clone(), def.body.node.clone());
                    pushed.push(def.name.node.clone());
                }
            }
            collect_sequence_writer_assignments(
                &body.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            for name in pushed.into_iter().rev() {
                scope.pop(&name);
            }
        }
        Expr::Eq(left, right) => {
            if let Some(var_idx) = primed_state_var_idx(&left.node, registry, scope) {
                let write = resolve_writer_expr_deep(&right.node, scope);
                out.writes
                    .entry(var_idx)
                    .or_default()
                    .push(SequenceWriterWrite {
                        expr: write,
                        scope: scope.clone(),
                    });
            }
        }
        Expr::Unchanged(target) => collect_unchanged_writer_targets(&target.node, registry, out),
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            if scope.get(name).is_none() {
                collect_sequence_writer_assignments_from_op(
                    name,
                    &[],
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
            }
        }
        Expr::Apply(op, args) => {
            if let Some(name) = operator_ident_name(&op.node) {
                collect_sequence_writer_assignments_from_op(
                    name,
                    args,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
            }
        }
        _ => {}
    }
}

fn collect_sequence_writer_assignments_from_op(
    name: &str,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &mut WriterExprScope,
    visiting: &mut BTreeSet<String>,
    out: &mut SequenceWriterAssignments,
) {
    let Some((resolved_name, def)) = writer_safe_op_def(name, op_defs, op_replacements) else {
        return;
    };
    if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    let mut pushed = Vec::new();
    for (param, arg) in def.params.iter().zip(args) {
        if param.arity != 0 {
            visiting.remove(resolved_name);
            return;
        }
        scope.push(
            param.name.node.clone(),
            resolve_writer_arg_expr(&arg.node, scope),
        );
        pushed.push(param.name.node.clone());
    }
    collect_sequence_writer_assignments(
        &def.body.node,
        registry,
        constants,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    for name in pushed.into_iter().rev() {
        scope.pop(&name);
    }
    visiting.remove(resolved_name);
}

fn collect_writer_len_upper_bounds(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    collect_writer_len_upper_bounds_inner(
        expr,
        registry,
        constants,
        op_defs,
        op_replacements,
        scope,
        &mut BTreeSet::new(),
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn collect_writer_len_upper_bounds_inner(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<(usize, usize)>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_writer_len_upper_bounds_inner(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_writer_len_upper_bounds_inner(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Ident(name, _) => {
            if let Some(bound) = scope.get(name) {
                if !matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                    collect_writer_len_upper_bounds_inner(
                        bound,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        visiting,
                        out,
                    );
                }
                return;
            }
            let Some((resolved_name, def)) = writer_safe_op_def(name, op_defs, op_replacements)
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_writer_len_upper_bounds_inner(
                &def.body.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        Expr::Lt(left, right) => {
            if let (Some(var_idx), Some(limit)) = (
                writer_len_expr_var_idx(
                    &left.node,
                    registry,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                ),
                const_usize_expr(&right.node, constants, op_replacements),
            ) {
                if let Some(bound) = limit.checked_sub(1) {
                    out.push((var_idx, bound));
                }
            }
        }
        Expr::Leq(left, right) => {
            if let (Some(var_idx), Some(limit)) = (
                writer_len_expr_var_idx(
                    &left.node,
                    registry,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                ),
                const_usize_expr(&right.node, constants, op_replacements),
            ) {
                out.push((var_idx, limit));
            }
        }
        Expr::Gt(left, right) => {
            if let (Some(limit), Some(var_idx)) = (
                const_usize_expr(&left.node, constants, op_replacements),
                writer_len_expr_var_idx(
                    &right.node,
                    registry,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                ),
            ) {
                if let Some(bound) = limit.checked_sub(1) {
                    out.push((var_idx, bound));
                }
            }
        }
        Expr::Geq(left, right) => {
            if let (Some(limit), Some(var_idx)) = (
                const_usize_expr(&left.node, constants, op_replacements),
                writer_len_expr_var_idx(
                    &right.node,
                    registry,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                ),
            ) {
                out.push((var_idx, limit));
            }
        }
        Expr::If(cond, then_expr, else_expr) => {
            collect_writer_len_upper_bounds_inner(
                &cond.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_writer_len_upper_bounds_inner(
                &then_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_writer_len_upper_bounds_inner(
                &else_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        _ => {}
    }
}

fn writer_len_expr_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    visiting: &mut BTreeSet<String>,
) -> Option<usize> {
    match expr {
        Expr::Apply(op, args)
            if args.len() == 1 && is_seq_len_operator(&op.node, op_replacements) =>
        {
            sequence_identity_var_idx(&args[0].node, registry, scope)
        }
        Expr::Ident(name, _) => {
            if let Some(bound) = scope.get(name) {
                if matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                    return None;
                }
                return writer_len_expr_var_idx(
                    bound,
                    registry,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                );
            }
            let (resolved_name, def) = writer_safe_op_def(name, op_defs, op_replacements)?;
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return None;
            }
            let result = writer_len_expr_var_idx(
                &def.body.node,
                registry,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
            visiting.remove(resolved_name);
            result
        }
        _ => None,
    }
}

fn primed_state_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    scope: &WriterExprScope,
) -> Option<usize> {
    match expr {
        Expr::Prime(inner) => state_var_idx_for_writer(&inner.node, registry, scope),
        _ => None,
    }
}

fn state_var_idx_for_writer(
    expr: &Expr,
    registry: &VarRegistry,
    scope: &WriterExprScope,
) -> Option<usize> {
    match expr {
        Expr::StateVar(_, idx, _) => Some(*idx as usize),
        Expr::Ident(name, _) => scope
            .get(name)
            .and_then(|expr| {
                if matches!(expr, Expr::Ident(bound_name, _) if bound_name == name) {
                    None
                } else {
                    state_var_idx_for_writer(expr, registry, scope)
                }
            })
            .or_else(|| registry.get(name).map(|idx| idx.as_usize())),
        _ => None,
    }
}

fn collect_unchanged_writer_targets(
    expr: &Expr,
    registry: &VarRegistry,
    out: &mut SequenceWriterAssignments,
) {
    match expr {
        Expr::StateVar(_, idx, _) => {
            out.writes.entry(*idx as usize).or_default();
        }
        Expr::Ident(name, _) => {
            if let Some(idx) = registry.get(name) {
                out.writes.entry(idx.as_usize()).or_default();
            }
        }
        Expr::Tuple(elems) => {
            for elem in elems {
                collect_unchanged_writer_targets(&elem.node, registry, out);
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn sequence_writer_expr_capacity(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    candidates: &BTreeMap<usize, usize>,
    visiting: &mut BTreeSet<String>,
) -> Option<usize> {
    match expr {
        Expr::Tuple(elems) => Some(elems.len()),
        Expr::FuncDef(bounds, _) if bounds.len() == 1 && bounds[0].pattern.is_none() => {
            writer_int_range_len_upper(
                &bounds[0].domain.as_ref()?.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )
        }
        Expr::Except(base, _) => sequence_writer_expr_capacity(
            &base.node,
            registry,
            constants,
            op_defs,
            op_replacements,
            scope,
            candidates,
            visiting,
        ),
        Expr::StateVar(_, idx, _) => candidates.get(&(*idx as usize)).copied(),
        Expr::Ident(name, _) => {
            if let Some(binding) = scope.sequence(name) {
                return Some(binding.capacity);
            }
            if let Some(bound) = scope.get(name) {
                if matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                    return None;
                }
                return sequence_writer_expr_capacity(
                    bound,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    candidates,
                    visiting,
                );
            }
            registry
                .get(name)
                .and_then(|idx| candidates.get(&idx.as_usize()).copied())
        }
        Expr::If(cond, then_expr, else_expr) => guarded_trim_append_capacity(
            &cond.node,
            &then_expr.node,
            &else_expr.node,
            registry,
            constants,
            op_replacements,
            scope,
        )
        .or_else(|| {
            let then_capacity = sequence_writer_expr_capacity(
                &then_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )?;
            let else_capacity = sequence_writer_expr_capacity(
                &else_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )?;
            Some(then_capacity.max(else_capacity))
        }),
        Expr::Apply(op, args) => {
            let name = operator_ident_name(&op.node)?;
            let resolved = resolve_layout_op_name(name, op_replacements)?;
            match (resolved, args.as_slice()) {
                ("Append", [seq, _]) => sequence_writer_expr_capacity(
                    &seq.node,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    candidates,
                    visiting,
                )?
                .checked_add(1),
                ("Tail", [seq]) => sequence_writer_expr_capacity(
                    &seq.node,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    candidates,
                    visiting,
                ),
                // WP-15 (`TY_FLAT_WRITE_ADMIT=1`): `SubSeq(seq, m, n)` is a
                // contiguous sub-window of `seq`, so its length can never
                // exceed `seq`'s capacity: TLA semantics require
                // `1 <= m /\ n <= Len(seq)` (an out-of-range window is a
                // runtime evaluation error that aborts checking before any
                // state is stored), hence `Len(SubSeq(seq, m, n)) =
                // max(n - m + 1, 0) <= Len(seq)`. Gated so the default
                // surface keeps its historical fail-closed shape.
                ("SubSeq", [seq, _, _]) if flat_write_admission_enabled() => {
                    sequence_writer_expr_capacity(
                        &seq.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        candidates,
                        visiting,
                    )
                }
                // WP-15 (`TY_FLAT_WRITE_ADMIT=1`): the element-removal idiom
                // `SubSeq(v, 1, i-1) \o SubSeq(v, i+1, Len(v))` (AllocateTest's
                // `Drop(sched, i)`). The two windows are index-DISJOINT
                // sub-windows of the SAME base, so the concatenation is a
                // sub-multiset of `v` and `capacity(v)` bounds it. This arm
                // must come before the general `\o` sum arm: the sum bound
                // `capacity(v) + capacity(v)` is also sound but is NOT a fixed
                // point of the candidates iteration (2 -> 4 -> 8 -> ...), so
                // the collector would fail to converge and emit NO proofs.
                // `capacity(v)` is fixpoint-stable (f(c) = c).
                ("\\o" | "\\circ", [left, right])
                    if flat_write_admission_enabled()
                        && concat_of_disjoint_subseq_windows(
                            &left.node,
                            &right.node,
                            constants,
                            op_replacements,
                        )
                        .is_some() =>
                {
                    let base = concat_of_disjoint_subseq_windows(
                        &left.node,
                        &right.node,
                        constants,
                        op_replacements,
                    )?;
                    sequence_writer_expr_capacity(
                        base,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        candidates,
                        visiting,
                    )
                }
                ("\\o" | "\\circ", [left, right]) => {
                    let left_capacity = sequence_writer_expr_capacity(
                        &left.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        candidates,
                        visiting,
                    )?;
                    let right_capacity = sequence_writer_expr_capacity(
                        &right.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        candidates,
                        visiting,
                    )?;
                    left_capacity.checked_add(right_capacity)
                }
                _ => sequence_writer_expr_capacity_from_op(
                    name,
                    args,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    candidates,
                    visiting,
                ),
            }
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn sequence_writer_expr_capacity_from_op(
    name: &str,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    candidates: &BTreeMap<usize, usize>,
    visiting: &mut BTreeSet<String>,
) -> Option<usize> {
    let (resolved_name, def) = writer_safe_op_def(name, op_defs, op_replacements)?;
    if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
        return None;
    }
    let mut child_scope = scope.clone();
    for (param, arg) in def.params.iter().zip(args) {
        child_scope.push(
            param.name.node.clone(),
            resolve_writer_arg_expr(&arg.node, scope),
        );
    }
    let result = sequence_writer_expr_capacity(
        &def.body.node,
        registry,
        constants,
        op_defs,
        op_replacements,
        &child_scope,
        candidates,
        visiting,
    );
    visiting.remove(resolved_name);
    result
}

fn guarded_trim_append_capacity(
    cond: &Expr,
    then_expr: &Expr,
    else_expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Option<usize> {
    let guard = guarded_len_lower_bound(cond, registry, constants, op_replacements, scope);
    let else_var = append_base_var_idx(else_expr, registry, op_replacements, scope);
    let then_var = trim_append_base_var_idx(then_expr, registry, op_replacements, scope);
    let (guarded_var, limit) = guard?;
    let else_var = else_var?;
    let then_var = then_var?;
    (guarded_var == else_var && guarded_var == then_var).then_some(limit)
}

fn guarded_len_lower_bound(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Option<(usize, usize)> {
    match expr {
        Expr::Ident(name, _) => scope.get(name).and_then(|bound| {
            if matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                None
            } else {
                guarded_len_lower_bound(bound, registry, constants, op_replacements, scope)
            }
        }),
        Expr::And(left, right) => {
            guarded_len_lower_bound(&left.node, registry, constants, op_replacements, scope)
                .or_else(|| {
                    guarded_len_lower_bound(
                        &right.node,
                        registry,
                        constants,
                        op_replacements,
                        scope,
                    )
                })
        }
        Expr::Geq(left, right) => {
            let var_idx = len_operand_var_idx(&left.node, registry, op_replacements, scope)?;
            let limit = const_usize_expr(&right.node, constants, op_replacements)?;
            Some((var_idx, limit))
        }
        Expr::Leq(left, right) => {
            let limit = const_usize_expr(&left.node, constants, op_replacements)?;
            let var_idx = len_operand_var_idx(&right.node, registry, op_replacements, scope)?;
            Some((var_idx, limit))
        }
        _ => None,
    }
}

fn len_operand_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Option<usize> {
    let Expr::Apply(op, args) = expr else {
        return None;
    };
    if args.len() == 1 && is_seq_len_operator(&op.node, op_replacements) {
        return sequence_identity_var_idx(&args[0].node, registry, scope);
    }
    None
}

fn is_seq_len_operator(expr: &Expr, op_replacements: Option<&OpReplacements>) -> bool {
    matches!(
        expr,
        Expr::Ident(name, _) | Expr::OpRef(name)
            if matches!(resolve_layout_op_name(name, op_replacements), Some("Len"))
    )
}

fn append_base_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Option<usize> {
    let Expr::Apply(op, args) = expr else {
        return None;
    };
    let name = operator_ident_name(&op.node)?;
    (resolve_layout_op_name(name, op_replacements)? == "Append")
        .then(|| sequence_identity_var_idx(&args.first()?.node, registry, scope))?
}

/// `SubSeq(base, lo, hi)` decomposition for the WP-15 writer-capacity
/// disjoint-window rule: returns `(base, lo, hi)` when `expr` is a `SubSeq`
/// application with exactly three arguments, `None` otherwise.
fn wp15_subseq_parts<'e>(
    expr: &'e Expr,
    op_replacements: Option<&OpReplacements>,
) -> Option<(&'e Expr, &'e Expr, &'e Expr)> {
    let Expr::Apply(op, args) = expr else {
        return None;
    };
    let name = operator_ident_name(&op.node)?;
    if resolve_layout_op_name(name, op_replacements)? != "SubSeq" {
        return None;
    }
    let [base, lo, hi] = args.as_slice() else {
        return None;
    };
    Some((&base.node, &lo.node, &hi.node))
}

/// Structural base-equality for the disjoint-window rule: the two `SubSeq`
/// bases must be the SAME simple reference — the same identifier (an operator
/// formal like `Drop`'s `seq`, or an unresolved state-var name) or the same
/// resolved state variable. Anything more complex fails closed.
fn wp15_same_subseq_base(left: &Expr, right: &Expr) -> bool {
    match (left, right) {
        (Expr::Ident(a, _), Expr::Ident(b, _)) => a == b,
        (Expr::StateVar(_, a, _), Expr::StateVar(_, b, _)) => a == b,
        _ => false,
    }
}

/// WP-15 (`TY_FLAT_WRITE_ADMIT=1`): recognize `SubSeq(v, a, b) \o
/// SubSeq(v, c, d)` where the two windows are provably index-disjoint
/// (`b < c`), so the concatenation is a sub-multiset of `v` and `capacity(v)`
/// is a sound, fixpoint-stable bound. Two disjointness certificates are
/// accepted, both structural and fail-closed:
///
///   * constant windows: `b` and `c` both evaluate to constants with `b < c`
///     (e.g. `SubSeq(v, 1, 0) \o SubSeq(v, 2, Len(v))`), or
///   * the removal idiom: `b = i - 1` and `c = i + 1` for the SAME simple
///     index expression `i` (`Drop(seq, i) == SubSeq(seq, 1, i-1) \o
///     SubSeq(seq, i+1, Len(seq))`).
///
/// Returns the shared base expression on success.
fn concat_of_disjoint_subseq_windows<'e>(
    left: &'e Expr,
    right: &'e Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<&'e Expr> {
    let (lbase, _llo, lhi) = wp15_subseq_parts(left, op_replacements)?;
    let (rbase, rlo, _rhi) = wp15_subseq_parts(right, op_replacements)?;
    if !wp15_same_subseq_base(lbase, rbase) {
        return None;
    }
    // Certificate 1: constant window ends with `hi < lo`.
    if let (Some(hi), Some(lo)) = (
        const_usize_expr(lhi, constants, op_replacements),
        const_usize_expr(rlo, constants, op_replacements),
    ) {
        return (hi < lo).then_some(lbase);
    }
    // Certificate 2: the removal idiom `(i - 1, i + 1)` over the same `i`.
    let Expr::Sub(l_idx, l_one) = lhi else {
        return None;
    };
    let Expr::Add(r_idx, r_one) = rlo else {
        return None;
    };
    if const_usize_expr(&l_one.node, constants, op_replacements) != Some(1)
        || const_usize_expr(&r_one.node, constants, op_replacements) != Some(1)
    {
        return None;
    }
    df_simple_exprs_match(&l_idx.node, &r_idx.node).then_some(lbase)
}

fn trim_append_base_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Option<usize> {
    let Expr::Apply(op, args) = expr else {
        return None;
    };
    let name = operator_ident_name(&op.node)?;
    if !matches!(
        resolve_layout_op_name(name, op_replacements),
        Some("\\o" | "\\circ")
    ) || args.len() != 2
    {
        return None;
    }
    let left = tail_base_var_idx(&args[0].node, registry, op_replacements, scope)?;
    sequence_writer_literal_len(&args[1].node)
        .is_some_and(|len| len == 1)
        .then_some(left)
}

fn tail_base_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
) -> Option<usize> {
    let Expr::Apply(op, args) = expr else {
        return None;
    };
    let name = operator_ident_name(&op.node)?;
    (resolve_layout_op_name(name, op_replacements)? == "Tail")
        .then(|| sequence_identity_var_idx(&args.first()?.node, registry, scope))?
}

fn sequence_writer_literal_len(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Tuple(elems) => Some(elems.len()),
        _ => None,
    }
}

fn sequence_identity_var_idx(
    expr: &Expr,
    registry: &VarRegistry,
    scope: &WriterExprScope,
) -> Option<usize> {
    match expr {
        Expr::StateVar(_, idx, _) => Some(*idx as usize),
        Expr::Ident(name, _) => scope
            .get(name)
            .and_then(|expr| {
                if matches!(expr, Expr::Ident(bound_name, _) if bound_name == name) {
                    None
                } else {
                    sequence_identity_var_idx(expr, registry, scope)
                }
            })
            .or_else(|| registry.get(name).map(|idx| idx.as_usize())),
        _ => None,
    }
}

fn const_usize_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<usize> {
    use num_traits::ToPrimitive;

    let value = const_expr_to_value_with_replacements(expr, constants, op_replacements)?;
    match &value {
        Value::SmallInt(n) => usize::try_from(*n).ok(),
        Value::Int(n) => n.to_usize(),
        _ => None,
    }
}

/// True when `expr` is (or resolves to) a *finite integer* binder domain: a
/// literal `lo..hi` range with constant integer bounds, a finite set whose
/// elements are all integers, or a `SetFilter` whose base domain is itself a
/// finite integer domain. The domain expression is resolved through
/// `op_replacements` (e.g. a `Nat <- NatOverride` config substitution) and
/// zero-arity operator definitions (e.g. `NatOverride == 0..MaxNat`) before
/// matching.
///
/// Soundness: this only authorizes treating a binder as `is_int` (so writes
/// from it are `Scalar(Int)`). A binder ranging over a finite integer domain is
/// always an integer in *every* reachable state, because the domain depends
/// only on model constants — never on the mutable state — so the conclusion is
/// universal, not sampled. A `SetFilter` `{j \in D : P(j)}` is a subset of `D`,
/// so when `D` is a finite integer domain every filtered element is also an
/// integer regardless of the predicate. Anything not provably integer-typed
/// fails closed (returns `false`).
fn int_range_domain(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
) -> bool {
    int_range_domain_inner(
        expr,
        constants,
        op_defs,
        op_replacements,
        &mut BTreeSet::new(),
    )
}

fn int_range_domain_inner(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    match expr {
        // A `lo..hi` range is a set of integers regardless of whether the bounds
        // fold to constants (they may be state-dependent, e.g. `1..(Len(s)+1)`).
        // It is unconditionally an integer binder domain (matching the original
        // structural recognition this helper replaced).
        Expr::Range(_, _) => true,
        // `{j \in D : P}` — subset of `D`, integer-typed iff `D` is.
        Expr::SetFilter(bound, _) => bound.domain.as_ref().is_some_and(|domain| {
            int_range_domain_inner(&domain.node, constants, op_defs, op_replacements, visiting)
        }),
        // Resolve a bare name through config replacement chains and zero-arity
        // operator definitions (e.g. `Nat` -> `NatOverride` -> `0..MaxNat`).
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            // First try resolving to a constant value (e.g. `Nat` bound to a
            // finite `Value::Interval`/int set via `CONSTANT` substitution).
            if let Some(value) =
                precomputed_constant_value_with_replacements(expr, constants, op_replacements)
            {
                return value_is_finite_int_domain(value);
            }
            // Otherwise inline a zero-arity, non-recursive operator body.
            let Some((resolved, def)) = layout_safe_op_def(name, op_defs, op_replacements) else {
                return false;
            };
            if !def.params.is_empty() || !visiting.insert(resolved.to_owned()) {
                return false;
            }
            let result = int_range_domain_inner(
                &def.body.node,
                constants,
                op_defs,
                op_replacements,
                visiting,
            );
            visiting.remove(resolved);
            result
        }
        // Any other shape that folds to a finite integer set value.
        _ => const_expr_to_value_with_replacements(expr, constants, op_replacements)
            .as_ref()
            .is_some_and(value_is_finite_int_domain),
    }
}

/// True when `value` is a finite, non-empty set whose every element is an
/// integer (a `Value::Interval`, or a `Value::Set` of integers).
fn value_is_finite_int_domain(value: &Value) -> bool {
    match value {
        Value::Interval(interval) => !interval.is_empty(),
        Value::Set(set) => {
            !set.is_empty()
                && set
                    .iter()
                    .all(|elem| matches!(elem, Value::SmallInt(_) | Value::Int(_)))
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_int_range_len_upper(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    candidates: &BTreeMap<usize, usize>,
    visiting: &mut BTreeSet<String>,
) -> Option<usize> {
    let Expr::Range(lo, hi) = expr else {
        return None;
    };
    let lo = const_i64_expr(&lo.node, constants, op_replacements)?;
    if lo != 1 {
        return None;
    }
    writer_numeric_upper(
        &hi.node,
        registry,
        constants,
        op_defs,
        op_replacements,
        scope,
        candidates,
        visiting,
    )
}

fn const_i64_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<i64> {
    use num_traits::ToPrimitive;

    let value = const_expr_to_value_with_replacements(expr, constants, op_replacements)?;
    match &value {
        Value::SmallInt(n) => Some(*n),
        Value::Int(n) => n.to_i64(),
        _ => None,
    }
}

fn integer_bound_domains(
    bounds: &[BoundVar],
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> BTreeMap<String, usize> {
    bounds
        .iter()
        .filter_map(|bound| {
            if bound.pattern.is_some() {
                return None;
            }
            let max = writer_const_int_range_max(
                &bound.domain.as_ref()?.node,
                constants,
                op_replacements,
            )?;
            Some((bound.name.node.clone(), max))
        })
        .collect()
}

fn writer_const_int_range_max(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<usize> {
    let Expr::Range(lo, hi) = expr else {
        return None;
    };
    let lo = const_i64_expr(&lo.node, constants, op_replacements)?;
    if lo != 1 {
        return None;
    }
    const_usize_expr(&hi.node, constants, op_replacements)
}

fn sequence_binding_from_writer_domain(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
    int_domains: &BTreeMap<String, usize>,
) -> Option<SequenceWriterBinding> {
    match expr {
        Expr::BigUnion(inner) => {
            let Expr::SetBuilder(body, bounds) = &inner.node else {
                return None;
            };
            let mut scoped_int_domains = int_domains.clone();
            for bound in bounds {
                if bound.pattern.is_none() {
                    if let Some(max) = writer_const_int_range_max(
                        &bound.domain.as_ref()?.node,
                        constants,
                        op_replacements,
                    ) {
                        scoped_int_domains.insert(bound.name.node.clone(), max);
                    }
                }
            }
            sequence_binding_from_writer_domain(
                &body.node,
                constants,
                op_replacements,
                &scoped_int_domains,
            )
        }
        Expr::FuncSet(domain, range) => {
            let capacity = writer_domain_capacity_upper(
                &domain.node,
                constants,
                op_replacements,
                int_domains,
            )?;
            let element_layout = writer_domain_element_layout(&range.node, op_replacements)?;
            Some(SequenceWriterBinding {
                capacity,
                element_layout,
            })
        }
        _ => None,
    }
}

fn writer_domain_capacity_upper(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
    int_domains: &BTreeMap<String, usize>,
) -> Option<usize> {
    let Expr::Range(lo, hi) = expr else {
        return None;
    };
    let lo = const_i64_expr(&lo.node, constants, op_replacements)?;
    if lo != 1 {
        return None;
    }
    match &hi.node {
        Expr::Ident(name, _) => int_domains
            .get(name)
            .copied()
            .or_else(|| const_usize_expr(&hi.node, constants, op_replacements)),
        _ => const_usize_expr(&hi.node, constants, op_replacements),
    }
}

fn writer_domain_element_layout(
    expr: &Expr,
    op_replacements: Option<&OpReplacements>,
) -> Option<FlatValueLayout> {
    match expr {
        Expr::Range(_, _) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Expr::Ident(_, _) => {
            // Constant aliases are handled later by normal value-expression
            // inference; only direct range domains are admitted here.
            let _ = op_replacements;
            None
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_numeric_upper(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    candidates: &BTreeMap<usize, usize>,
    visiting: &mut BTreeSet<String>,
) -> Option<usize> {
    match expr {
        Expr::Int(_) => const_usize_expr(expr, constants, op_replacements),
        Expr::Ident(name, _) => {
            if let Some(value) = const_usize_expr(expr, constants, op_replacements) {
                return Some(value);
            }
            if let Some(bound) = scope.get(name) {
                if matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                    return None;
                }
                return writer_numeric_upper(
                    bound,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    candidates,
                    visiting,
                );
            }
            writer_numeric_upper_from_op(
                name,
                &[],
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )
        }
        Expr::Add(left, right) => Some(
            writer_numeric_upper(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )?
            .checked_add(writer_numeric_upper(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )?)?,
        ),
        Expr::Sub(left, right) => Some(
            writer_numeric_upper(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )?
            .saturating_sub(writer_numeric_upper(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                candidates,
                visiting,
            )?),
        ),
        Expr::Apply(op, args) => {
            let name = operator_ident_name(&op.node)?;
            let resolved = resolve_layout_op_name(name, op_replacements)?;
            match (resolved, args.as_slice()) {
                ("Len", [seq]) => {
                    if let Some(var_idx) = sequence_identity_var_idx(&seq.node, registry, scope) {
                        if let Some(bound) = scope.len_upper_bound(var_idx) {
                            return Some(bound);
                        }
                    }
                    sequence_writer_expr_capacity(
                        &seq.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        candidates,
                        visiting,
                    )
                }
                _ => writer_numeric_upper_from_op(
                    name,
                    args,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    candidates,
                    visiting,
                ),
            }
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_numeric_upper_from_op(
    name: &str,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    candidates: &BTreeMap<usize, usize>,
    visiting: &mut BTreeSet<String>,
) -> Option<usize> {
    let (resolved_name, def) = writer_safe_op_def(name, op_defs, op_replacements)?;
    if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
        return None;
    }
    let mut child_scope = scope.clone();
    for (param, arg) in def.params.iter().zip(args) {
        child_scope.push(
            param.name.node.clone(),
            resolve_writer_arg_expr(&arg.node, scope),
        );
    }
    let result = writer_numeric_upper(
        &def.body.node,
        registry,
        constants,
        op_defs,
        op_replacements,
        &child_scope,
        candidates,
        visiting,
    );
    visiting.remove(resolved_name);
    result
}

#[allow(clippy::too_many_arguments)]
// Outer Option = analysis succeeded/failed; inner Option = element layout
// present/absent. The two layers carry distinct meaning, so keep nested Option.
#[allow(clippy::option_option)]
fn sequence_writer_expr_element_layout(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    seed_slot_types: &BTreeMap<usize, SlotType>,
    candidates: &BTreeMap<usize, FlatValueLayout>,
    visiting: &mut BTreeSet<String>,
) -> Option<Option<FlatValueLayout>> {
    match expr {
        Expr::Tuple(elems) if elems.is_empty() => Some(None),
        Expr::Tuple(elems) => {
            let mut layout = None;
            for elem in elems {
                let elem_layout = writer_value_expr_layout(
                    &elem.node,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                )?;
                layout = Some(if let Some(existing) = layout {
                    merge_flat_value_layouts(&existing, &elem_layout)?
                } else {
                    elem_layout
                });
            }
            Some(layout)
        }
        Expr::FuncDef(bounds, body) if bounds.len() == 1 && bounds[0].pattern.is_none() => {
            if !int_range_domain(
                &bounds[0].domain.as_ref()?.node,
                constants,
                op_defs,
                op_replacements,
            ) {
                return None;
            }
            let mut scoped = scope.clone();
            scoped.push_int(bounds[0].name.node.clone());
            let layout = writer_value_expr_layout(
                &body.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                &scoped,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            Some(Some(layout))
        }
        Expr::Except(base, specs) => {
            let base_layout = sequence_writer_expr_element_layout(
                &base.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            if let Some(layout) = base_layout.as_ref() {
                for spec in specs {
                    let mut scoped = scope.clone();
                    scoped.push_value_layout("@".to_owned(), layout.clone());
                    let value_layout = writer_value_expr_layout(
                        &spec.value.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        &scoped,
                        seed_slot_types,
                        candidates,
                        visiting,
                    )?;
                    scoped.pop_value_layout("@");
                    merge_flat_value_layouts(layout, &value_layout)?;
                }
            }
            Some(base_layout)
        }
        Expr::StateVar(_, idx, _) => Some(candidates.get(&(*idx as usize)).cloned()),
        Expr::Ident(name, _) => {
            if let Some(binding) = scope.sequence(name) {
                return Some(Some(binding.element_layout.clone()));
            }
            if let Some(bound) = scope.get(name) {
                if matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                    return None;
                }
                return sequence_writer_expr_element_layout(
                    bound,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                );
            }
            registry
                .get(name)
                .and_then(|idx| candidates.get(&idx.as_usize()).cloned())
                .map(Some)
        }
        Expr::If(_, then_expr, else_expr) => {
            let then_layout = sequence_writer_expr_element_layout(
                &then_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            let else_layout = sequence_writer_expr_element_layout(
                &else_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            merge_optional_layouts(then_layout, else_layout)
        }
        Expr::Apply(op, args) => {
            let name = operator_ident_name(&op.node)?;
            let resolved = resolve_layout_op_name(name, op_replacements)?;
            match (resolved, args.as_slice()) {
                ("Append", [seq, elem]) => {
                    let seq_layout = sequence_writer_expr_element_layout(
                        &seq.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        seed_slot_types,
                        candidates,
                        visiting,
                    )?;
                    let elem_layout = writer_value_expr_layout(
                        &elem.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        seed_slot_types,
                        candidates,
                        visiting,
                    )?;
                    merge_optional_layouts(seq_layout, Some(elem_layout))
                }
                ("Tail", [seq]) => sequence_writer_expr_element_layout(
                    &seq.node,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                ),
                // WP-15 (`TY_FLAT_WRITE_ADMIT=1`): every element of
                // `SubSeq(seq, m, n)` is an element of `seq` (a contiguous
                // sub-window introduces no new elements), so `seq`'s element
                // layout covers the result exactly. Mirrors the capacity arm
                // above; gated for the same default-surface reason.
                ("SubSeq", [seq, _, _]) if flat_write_admission_enabled() => {
                    sequence_writer_expr_element_layout(
                        &seq.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        seed_slot_types,
                        candidates,
                        visiting,
                    )
                }
                ("\\o" | "\\circ", [left, right]) => {
                    let left_layout = sequence_writer_expr_element_layout(
                        &left.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        seed_slot_types,
                        candidates,
                        visiting,
                    )?;
                    let right_layout = sequence_writer_expr_element_layout(
                        &right.node,
                        registry,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        seed_slot_types,
                        candidates,
                        visiting,
                    )?;
                    merge_optional_layouts(left_layout, right_layout)
                }
                _ => sequence_writer_expr_element_layout_from_op(
                    name,
                    args,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                ),
            }
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
// Outer Option = analysis succeeded/failed; inner Option = element layout
// present/absent. The two layers carry distinct meaning, so keep nested Option.
#[allow(clippy::option_option)]
fn sequence_writer_expr_element_layout_from_op(
    name: &str,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    seed_slot_types: &BTreeMap<usize, SlotType>,
    candidates: &BTreeMap<usize, FlatValueLayout>,
    visiting: &mut BTreeSet<String>,
) -> Option<Option<FlatValueLayout>> {
    let (resolved_name, def) = writer_safe_op_def(name, op_defs, op_replacements)?;
    if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
        return None;
    }
    let mut child_scope = scope.clone();
    for (param, arg) in def.params.iter().zip(args) {
        child_scope.push(
            param.name.node.clone(),
            resolve_writer_arg_expr(&arg.node, scope),
        );
    }
    let result = sequence_writer_expr_element_layout(
        &def.body.node,
        registry,
        constants,
        op_defs,
        op_replacements,
        &child_scope,
        seed_slot_types,
        candidates,
        visiting,
    );
    visiting.remove(resolved_name);
    result
}

// Outer Option = merge succeeded/failed; inner Option = merged layout
// present/absent. The two layers carry distinct meaning, so keep nested Option.
#[allow(clippy::option_option)]
fn merge_optional_layouts(
    left: Option<FlatValueLayout>,
    right: Option<FlatValueLayout>,
) -> Option<Option<FlatValueLayout>> {
    match (left, right) {
        (Some(left), Some(right)) => merge_flat_value_layouts(&left, &right).map(Some),
        (Some(layout), None) | (None, Some(layout)) => Some(Some(layout)),
        (None, None) => Some(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_value_expr_layout(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    seed_slot_types: &BTreeMap<usize, SlotType>,
    candidates: &BTreeMap<usize, FlatValueLayout>,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    match expr {
        Expr::Bool(_) => Some(FlatValueLayout::Scalar(SlotType::Bool)),
        Expr::Int(_) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Expr::String(_) => Some(FlatValueLayout::Scalar(SlotType::String)),
        Expr::FuncApply(base, _) => sequence_writer_expr_element_layout(
            &base.node,
            registry,
            constants,
            op_defs,
            op_replacements,
            scope,
            seed_slot_types,
            candidates,
            visiting,
        )?,
        Expr::StateVar(_, idx, _) => seed_slot_types
            .get(&(*idx as usize))
            .copied()
            .map(FlatValueLayout::Scalar),
        Expr::Ident(name, _) => {
            if let Some(layout) = scope.value_layout(name) {
                return Some(layout.clone());
            }
            if scope.is_int(name) {
                return Some(FlatValueLayout::Scalar(SlotType::Int));
            }
            if let Some(bound) = scope.get(name) {
                if matches!(bound, Expr::Ident(bound_name, _) if bound_name == name) {
                    return None;
                }
                return writer_value_expr_layout(
                    bound,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                );
            }
            registry
                .get(name)
                .and_then(|idx| seed_slot_types.get(&idx.as_usize()).copied())
                .map(FlatValueLayout::Scalar)
                .or_else(|| {
                    flat_layout_from_value_expr_scoped(
                        expr,
                        constants,
                        op_defs,
                        op_replacements,
                        &LayoutScope::new(),
                        visiting,
                    )
                })
        }
        Expr::Record(fields) => {
            let mut field_pairs = Vec::with_capacity(fields.len());
            for (name, value_expr) in fields {
                let field_name = Arc::from(name.node.as_str());
                let layout = writer_value_expr_layout(
                    &value_expr.node,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                )?;
                field_pairs.push((intern_name(&field_name), field_name, layout));
            }
            // Canonical record field order: field-name STRING, matching
            // RecordValue's storage order (NameId order is run-dependent).
            field_pairs.sort_by(|a, b| a.1.cmp(&b.1));
            Some(FlatValueLayout::Record {
                field_names: field_pairs
                    .iter()
                    .map(|(_, name, _)| Arc::clone(name))
                    .collect(),
                field_layouts: field_pairs
                    .into_iter()
                    .map(|(_, _, layout)| layout)
                    .collect(),
            })
        }
        Expr::RecordAccess(base, field) => {
            let layout = writer_value_expr_layout(
                &base.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            record_field_layout(&layout, field.name.node.as_str())
        }
        Expr::Apply(op, args) => {
            let name = operator_ident_name(&op.node)?;
            let resolved = resolve_layout_op_name(name, op_replacements)?;
            match (resolved, args.as_slice()) {
                ("Head", [seq]) => sequence_writer_expr_element_layout(
                    &seq.node,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                )?,
                _ => writer_value_expr_layout_from_op(
                    name,
                    args,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    seed_slot_types,
                    candidates,
                    visiting,
                ),
            }
        }
        Expr::If(_, then_expr, else_expr) => {
            let then_layout = writer_value_expr_layout(
                &then_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            let else_layout = writer_value_expr_layout(
                &else_expr.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            merge_flat_value_layouts(&then_layout, &else_layout)
        }
        // Boolean negation `~e`: total over `BOOLEAN`, so the result is always a
        // `Scalar(Bool)` exactly when the operand provably has `Bool` layout.
        // This proves elements written via e.g. `flag' = [flag EXCEPT ![self] =
        // ~flag[self]]` stay one-bit booleans for *every* reachable state,
        // because negation maps `{TRUE, FALSE}` onto itself and the operand's
        // Bool layout is itself proven (here from a `Scalar(Bool)` function-range
        // element). A non-Bool operand fails closed (returns `None`).
        Expr::Not(operand) => {
            let operand_layout = writer_value_expr_layout(
                &operand.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            matches!(operand_layout, FlatValueLayout::Scalar(SlotType::Bool))
                .then_some(FlatValueLayout::Scalar(SlotType::Bool))
        }
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_layout = writer_value_expr_layout(
                &left.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            let right_layout = writer_value_expr_layout(
                &right.node,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                seed_slot_types,
                candidates,
                visiting,
            )?;
            (matches!(left_layout, FlatValueLayout::Scalar(SlotType::Int))
                && matches!(right_layout, FlatValueLayout::Scalar(SlotType::Int)))
            .then_some(FlatValueLayout::Scalar(SlotType::Int))
        }
        _ => flat_layout_from_value_expr_scoped(
            expr,
            constants,
            op_defs,
            op_replacements,
            &LayoutScope::new(),
            visiting,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_value_expr_layout_from_op(
    name: &str,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &WriterExprScope,
    seed_slot_types: &BTreeMap<usize, SlotType>,
    candidates: &BTreeMap<usize, FlatValueLayout>,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    let (resolved_name, def) = writer_safe_op_def(name, op_defs, op_replacements)?;
    if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
        return None;
    }
    let mut child_scope = scope.clone();
    for (param, arg) in def.params.iter().zip(args) {
        child_scope.push(
            param.name.node.clone(),
            resolve_writer_arg_expr(&arg.node, scope),
        );
    }
    let result = writer_value_expr_layout(
        &def.body.node,
        registry,
        constants,
        op_defs,
        op_replacements,
        &child_scope,
        seed_slot_types,
        candidates,
        visiting,
    );
    visiting.remove(resolved_name);
    result
}

fn record_field_layout(layout: &FlatValueLayout, field: &str) -> Option<FlatValueLayout> {
    let FlatValueLayout::Record {
        field_names,
        field_layouts,
    } = layout
    else {
        return None;
    };
    field_names
        .iter()
        .position(|name| name.as_ref() == field)
        .and_then(|idx| field_layouts.get(idx).cloned())
}

fn resolve_writer_arg_expr(expr: &Expr, scope: &WriterExprScope) -> Expr {
    resolve_writer_expr_deep(expr, scope)
}

fn resolve_writer_expr_deep(expr: &Expr, scope: &WriterExprScope) -> Expr {
    resolve_writer_expr_deep_inner(expr, scope, &mut BTreeSet::new())
}

fn resolve_writer_expr_deep_inner(
    expr: &Expr,
    scope: &WriterExprScope,
    resolving: &mut BTreeSet<String>,
) -> Expr {
    fn span_expr(
        original: &tla_core::span::Spanned<Expr>,
        node: Expr,
    ) -> tla_core::span::Spanned<Expr> {
        tla_core::span::Spanned::new(node, original.span)
    }

    match expr {
        Expr::Ident(name, _) => scope
            .get(name)
            .filter(|_| resolving.insert(name.clone()))
            .map(|bound| {
                let resolved = resolve_writer_expr_deep_inner(bound, scope, resolving);
                resolving.remove(name);
                resolved
            })
            .unwrap_or_else(|| expr.clone()),
        Expr::Apply(op, args) => Expr::Apply(
            Box::new(span_expr(
                op,
                resolve_writer_expr_deep_inner(&op.node, scope, resolving),
            )),
            args.iter()
                .map(|arg| {
                    span_expr(
                        arg,
                        resolve_writer_expr_deep_inner(&arg.node, scope, resolving),
                    )
                })
                .collect(),
        ),
        Expr::Tuple(elems) => Expr::Tuple(
            elems
                .iter()
                .map(|elem| {
                    span_expr(
                        elem,
                        resolve_writer_expr_deep_inner(&elem.node, scope, resolving),
                    )
                })
                .collect(),
        ),
        Expr::If(cond, then_expr, else_expr) => Expr::If(
            Box::new(span_expr(
                cond,
                resolve_writer_expr_deep_inner(&cond.node, scope, resolving),
            )),
            Box::new(span_expr(
                then_expr,
                resolve_writer_expr_deep_inner(&then_expr.node, scope, resolving),
            )),
            Box::new(span_expr(
                else_expr,
                resolve_writer_expr_deep_inner(&else_expr.node, scope, resolving),
            )),
        ),
        Expr::And(left, right) => Expr::And(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Or(left, right) => Expr::Or(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Not(inner) => Expr::Not(Box::new(span_expr(
            inner,
            resolve_writer_expr_deep_inner(&inner.node, scope, resolving),
        ))),
        Expr::Eq(left, right) => Expr::Eq(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Neq(left, right) => Expr::Neq(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Lt(left, right) => Expr::Lt(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Leq(left, right) => Expr::Leq(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Gt(left, right) => Expr::Gt(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Geq(left, right) => Expr::Geq(
            Box::new(span_expr(
                left,
                resolve_writer_expr_deep_inner(&left.node, scope, resolving),
            )),
            Box::new(span_expr(
                right,
                resolve_writer_expr_deep_inner(&right.node, scope, resolving),
            )),
        ),
        Expr::Record(fields) => Expr::Record(
            fields
                .iter()
                .map(|(name, value)| {
                    (
                        name.clone(),
                        span_expr(
                            value,
                            resolve_writer_expr_deep_inner(&value.node, scope, resolving),
                        ),
                    )
                })
                .collect(),
        ),
        Expr::RecordAccess(base, field) => Expr::RecordAccess(
            Box::new(span_expr(
                base,
                resolve_writer_expr_deep_inner(&base.node, scope, resolving),
            )),
            field.clone(),
        ),
        _ => expr.clone(),
    }
}

pub(crate) fn collect_sequence_fixed_domain_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<SequenceFixedDomainTypeProof>,
) {
    collect_sequence_fixed_domain_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

pub(crate) fn collect_tagged_scalar_set_range_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedScalarSetRangeTypeProof>,
) {
    collect_tagged_scalar_set_range_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

pub(crate) fn collect_fixed_scalar_range_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<FixedScalarRangeTypeProof>,
) {
    collect_fixed_scalar_range_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

pub(crate) fn collect_set_bitmask_range_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<SetBitmaskRangeTypeProof>,
) {
    collect_set_bitmask_range_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

pub(crate) fn collect_set_bitmask_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<SetBitmaskTypeProof>,
) {
    collect_set_bitmask_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

/// Collect proofs that a *state variable itself* is constrained to a finite
/// homogeneous scalar (string / model-value) universe by `expr` (typically the
/// `TypeOK` invariant), e.g. `tmState \in {"init", "commit", "abort"}`.
///
/// These proofs authorize a bare scalar-string/model-value state variable to be
/// stored as a primary-flat [`VarLayoutKind::FixedScalar`] slot. The encoding is
/// unchanged (interned `NameId` in one i64 — total and bijective over all
/// strings), so the proof only certifies primary-flat admissibility; it never
/// risks aliasing.
pub(crate) fn collect_fixed_scalar_var_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<FixedScalarVarTypeProof>,
) {
    collect_fixed_scalar_var_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

/// Collect [`TaggedScalarUnionVarTypeProof`]s from a checked type invariant.
///
/// Walks the (already constant-preserving-lowered) `TypeOK` body for whole-var
/// membership conjuncts `var \in T` whose type `T` inference resolves to a
/// heterogeneous finite scalar union (a `\cup` of distinct scalar lanes, or a
/// mixed set literal like `{"get", "insert", NIL}`). Only fires when the
/// `TY_TAGGED_SCALAR_UNION` gate is on (that gate controls whether
/// `flat_layout_from_type_set_expr_with_ops` constructs the union at all), so it
/// is a no-op on the default surface.
pub(crate) fn collect_tagged_scalar_union_var_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedScalarUnionVarTypeProof>,
) {
    collect_tagged_scalar_union_var_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        op_defs,
        op_replacements,
        &ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_tagged_scalar_union_var_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarUnionVarTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_scalar_union_var_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_tagged_scalar_union_var_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            else {
                return;
            };
            // Only the *whole variable* (empty path) maps to a top-level scalar
            // `VarLayoutKind`; nested `v[k]`/`v.f` sub-paths describe function
            // ranges / record fields handled elsewhere and are not one-slot
            // scalar unions.
            if !path.is_empty() {
                return;
            }
            if let Some(FlatValueLayout::TaggedScalarUnion { proof }) =
                flat_layout_from_type_set_expr_with_ops(
                    &right.node,
                    constants,
                    Some(op_defs),
                    Some(op_replacements),
                )
            {
                if !out
                    .iter()
                    .any(|existing| existing.var_idx == var_idx && existing.proof == proof)
                {
                    out.push(TaggedScalarUnionVarTypeProof {
                        var_idx,
                        proof,
                        invariant: Arc::from(invariant),
                    });
                }
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_union_var_type_proofs_inner(
                &def.body.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

/// Apply collected [`TaggedScalarUnionVarTypeProof`]s as whole-variable layout
/// overrides on an inferred [`StateLayout`].
///
/// For each proven scalar-union variable whose observed layout is a fail-closed
/// one-slot scalar kind (`ScalarModelValue` / `ScalarString`) — or an already
/// homogeneous `FixedScalar` that the union strictly widens — promote it to
/// `Recursive { TaggedScalarUnion }` so the slot stores the injective universe
/// index instead of a raw `NameId` that can alias an Int slot. The override only
/// fires when the sampled value is inside the proven universe (the round-trip is
/// verified per whole state, so a non-fitting promotion would disable
/// flat-primary for the ENTIRE layout — fail closed by skipping). `Scalar` /
/// `ScalarBool` (Int/Bool) are never touched: they are already primary-flat and
/// changing their encoding is neither needed nor sound here.
pub(crate) fn apply_tagged_scalar_union_var_overrides(
    layout: &mut StateLayout,
    proofs: &[TaggedScalarUnionVarTypeProof],
    sample_rows: &[Vec<Value>],
) {
    for proof in proofs {
        let Some(var) = layout.var_layout(proof.var_idx) else {
            continue;
        };
        // Only promote a fail-closed one-slot interned-scalar kind. `Scalar`
        // (Int) / `ScalarBool` are already primary-flat and must keep their raw
        // encoding; `FixedScalar` is an already-admitted homogeneous universe
        // (a heterogeneous union never resolves to one); compound kinds and a
        // veto-demoted `Dynamic` are not single scalar lanes.
        if !matches!(
            var.kind,
            VarLayoutKind::ScalarModelValue | VarLayoutKind::ScalarString
        ) {
            continue;
        }
        // EVERY sampled value of this var must be inside the proven universe.
        // Promoting a var whose sample holds an out-of-universe value would make
        // that whole state fail flat serialization, disabling flat-primary for
        // the ENTIRE layout — so fail closed (skip the override) instead. A
        // non-scalar sample also fails `flat_scalar_from_value` and is skipped.
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx)
                    .and_then(flat_scalar_from_value)
                    .is_some_and(|flat| proof.proof.universe().contains(&flat))
            });
        if !all_fit {
            continue;
        }
        let new_kind = VarLayoutKind::Recursive {
            layout: FlatValueLayout::TaggedScalarUnion {
                proof: proof.proof.clone(),
            },
        };
        layout.replace_var_kind_same_slots(proof.var_idx, new_kind);
    }
}

/// Collect [`TaggedScalarUnionRangeTypeProof`]s from a checked type invariant
/// (WP-09/Part A).
///
/// Walks the (constant-preserving-lowered) `TypeOK` body for whole-var
/// membership conjuncts `var \in [D -> R]` where `D` enumerates to a concrete
/// finite domain (including the `S \X T` tuple product — btree
/// `[Nodes \X Keys -> ...]`) and `R` enumerates to a concrete finite scalar
/// universe (`scalar_domain_from_type_set_expr_scoped`, which handles `1..N`,
/// set literals, `\cup`, `\ `, and zero-arg aliases; anything else fails
/// closed). Both the heterogeneous (`Nodes \cup {NIL}`) and homogeneous
/// (`Vals \cup {NIL}`) union shapes are carried by the same injective
/// universe-index proof. No-op unless the `TY_TAGGED_SCALAR_UNION` gate is on,
/// so the default surface is unchanged.
pub(crate) fn collect_tagged_scalar_union_range_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedScalarUnionRangeTypeProof>,
) {
    if !tagged_scalar_union_native_flat_primary_enabled() {
        return;
    }
    collect_tagged_scalar_union_range_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_tagged_scalar_union_range_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarUnionRangeTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_scalar_union_range_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_tagged_scalar_union_range_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            else {
                return;
            };
            // Only the whole variable (empty path) is a tuple-keyed function
            // whose range encoding this proof can override; nested sub-paths
            // describe other shapes and stay fail-closed.
            if !path.is_empty() {
                return;
            }
            collect_tagged_scalar_union_range_proof_from_type_expr(
                &right.node,
                invariant,
                var_idx,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_union_range_type_proofs_inner(
                &def.body.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_tagged_scalar_union_range_proof_from_type_expr(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarUnionRangeTypeProof>,
) {
    match expr {
        Expr::FuncSet(domain, range) => {
            let Some(domain) = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            ) else {
                return;
            };
            let Some(universe) = scalar_domain_from_type_set_expr_scoped(
                &range.node,
                constants,
                op_defs,
                Some(op_replacements),
                &LayoutScope::new(),
                visiting,
            ) else {
                return;
            };
            let Ok(proof) = TaggedScalarUnionProof::new(universe, Arc::from(invariant)) else {
                return;
            };
            let candidate = TaggedScalarUnionRangeTypeProof {
                var_idx,
                domain,
                proof,
                invariant: Arc::from(invariant),
            };
            if !out.iter().any(|existing| {
                existing.var_idx == candidate.var_idx
                    && existing.domain == candidate.domain
                    && existing.proof == candidate.proof
            }) {
                out.push(candidate);
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_union_range_proof_from_type_expr(
                &def.body.node,
                invariant,
                var_idx,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

/// Apply collected [`TaggedScalarUnionRangeTypeProof`]s as range-encoding
/// overrides on an inferred [`StateLayout`] (WP-09/Part A).
///
/// For each proven variable whose observed kind is a fail-closed
/// `TupleKeyedArray { range_encoding: ScalarSlots }` with at least one non-i64
/// sampled slot, and whose canonical sorted tuple-key table EXACTLY equals the
/// proof's enumerated `FuncSet` domain, upgrade the range encoding to
/// `TaggedScalarUnion(proof)` so every range slot stores the injective
/// universe index instead of a raw payload that could alias across scalar
/// sorts. The override only fires when EVERY sampled state's range values all
/// fit the proven universe (a non-fitting promotion would fail flat
/// serialization for that whole state — fail closed by skipping). Plain-i64
/// ranges are never touched: they are already primary-flat with the raw
/// encoding, and changing their encoding is neither needed nor sound here.
/// Slot count is unchanged (one slot per key), so the in-place kind swap
/// preserves every other variable's offset.
pub(crate) fn apply_tuple_keyed_tagged_scalar_union_range_overrides(
    layout: &mut StateLayout,
    proofs: &[TaggedScalarUnionRangeTypeProof],
    sample_rows: &[Vec<Value>],
) {
    let debug = std::env::var_os("TY_LAYOUT_PROOF_DEBUG").is_some_and(|v| v == "1");
    for proof in proofs {
        let Some(var) = layout.var_layout(proof.var_idx) else {
            continue;
        };
        let VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding: TupleKeyedArrayRangeEncoding::ScalarSlots,
        } = &var.kind
        else {
            if debug {
                eprintln!(
                    "[layout-proof] union-range override var={} skipped: kind is not \
                     TupleKeyedArray/ScalarSlots",
                    proof.var_idx
                );
            }
            continue;
        };
        // A plain-i64 range is already structurally primary-flat with the raw
        // encoding; re-encoding it would change fingerprints for no admission
        // gain. Only promote a range that is currently fail-closed.
        if value_types
            .iter()
            .all(|ty| matches!(ty, SlotType::Int | SlotType::Bool))
        {
            continue;
        }
        // The proof's enumerated domain must be EXACTLY the observed canonical
        // tuple-key table (both are `Value::cmp`-sorted and deduplicated), so
        // slot `i` provably stores the range value at `domain_keys[i]`.
        if proof.domain.as_ref() != domain_keys.as_slice() {
            if debug {
                eprintln!(
                    "[layout-proof] union-range override var={} skipped: proof domain \
                     ({} keys) != layout domain ({} keys)",
                    proof.var_idx,
                    proof.domain.len(),
                    domain_keys.len()
                );
            }
            continue;
        }
        // EVERY sampled value must be a function over exactly this domain
        // whose range values all fit the proven universe; otherwise promoting
        // would make that state fail flat serialization (fail closed by
        // skipping the override).
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx).is_some_and(|value| {
                    let Value::Func(func) = value else {
                        return false;
                    };
                    func.domain_len() == domain_keys.len()
                        && domain_keys.iter().all(|key| {
                            func.apply(key)
                                .and_then(flat_scalar_from_value)
                                .is_some_and(|observed| proof.proof.universe().contains(&observed))
                        })
                })
            });
        if !all_fit {
            if debug {
                eprintln!(
                    "[layout-proof] union-range override var={} skipped: a sampled value \
                     does not fit the proven universe",
                    proof.var_idx
                );
            }
            continue;
        }
        let new_kind = VarLayoutKind::TupleKeyedArray {
            domain_keys: domain_keys.clone(),
            value_types: value_types.clone(),
            range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof.proof.clone()),
        };
        if layout.replace_var_kind_same_slots(proof.var_idx, new_kind) && debug {
            eprintln!(
                "[layout-proof] union-range override var={} APPLIED (|universe|={})",
                proof.var_idx,
                proof.proof.universe().len()
            );
        }
    }
}

/// One classified writer of a candidate scalar-or-tuple union variable.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ScalarTupleArmEvidence {
    /// `v' = <scalar constant>` — the scalar sentinel arm (btree `args = NIL`).
    Scalar(FlatScalarValue),
    /// `v' = <<e1, .., ek>>` — a fixed-arity tuple arm. Each entry is the finite
    /// value domain of that position (a quantifier domain such as `Keys`, or the
    /// singleton domain of a constant).
    Tuple(Vec<Vec<FlatScalarValue>>),
}

/// Total-writer-coverage collector for a scalar-or-tuple union variable.
///
/// Walks `Init`/`Next` and classifies every assignment to `target_var_idx`.
/// Returns `None` — abandoning the whole proof — as soon as ANY writer cannot be
/// classified, because a missed writer would mean the union universe is not
/// closed and a later successor could carry a shape the tag cannot name.
struct ScalarTupleUnionWriterCollector<'a> {
    target_var_idx: usize,
    registry: &'a VarRegistry,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &'a BTreeMap<String, Arc<[Value]>>,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
}

impl ScalarTupleUnionWriterCollector<'_> {
    /// Walk `expr`, appending one [`ScalarTupleArmEvidence`] per classified
    /// writer. Returns `false` to abandon the proof (fail closed).
    fn collect(
        &mut self,
        expr: &Expr,
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
        out: &mut Vec<ScalarTupleArmEvidence>,
    ) -> bool {
        match expr {
            Expr::And(left, right) | Expr::Or(left, right) => {
                // Deliberately NOT short-circuiting: every branch must be
                // walked so a writer in the right branch still contributes its
                // arm when the left branch is a non-writer.
                let left_ok = self.collect(&left.node, scope, visiting, out);
                let right_ok = self.collect(&right.node, scope, visiting, out);
                left_ok && right_ok
            }
            // `LET d == .. IN body` — btree's `UpdateReq`/`GetValue`/`AddToLeaf`
            // all wrap their conjunct list in one. The definitions are value
            // bindings, not writers, so only the body is walked; a definition
            // that somehow writes the target var is not classifiable and fails
            // closed.
            Expr::Let(defs, body) => {
                let defs_clean = defs.iter().all(|def| {
                    !expr_mentions_prime_var(&def.body.node, self.target_var_idx, self.registry)
                });
                let body_ok = self.collect(&body.node, scope, visiting, out);
                defs_clean && body_ok
            }
            Expr::If(cond, then_expr, else_expr) => {
                // A condition that itself writes the var is not a shape we can
                // classify positionally; only the branches are writers.
                let cond_clean =
                    !expr_mentions_prime_var(&cond.node, self.target_var_idx, self.registry);
                let then_ok = self.collect(&then_expr.node, scope, visiting, out);
                let else_ok = self.collect(&else_expr.node, scope, visiting, out);
                cond_clean && then_ok && else_ok
            }
            Expr::Exists(vars, body) | Expr::Forall(vars, body) => {
                let Some(added) = self.push_bound_vars(vars, scope) else {
                    // An unresolvable binder over a body that writes the var
                    // leaves the element domains unknown — fail closed.
                    return !expr_mentions_prime_var(expr, self.target_var_idx, self.registry);
                };
                let ok = self.collect(&body.node, scope, visiting, out);
                for name in added {
                    scope.pop(&name);
                }
                ok
            }
            Expr::Eq(left, right) => {
                // `Init` writes the unprimed var (`args = NIL`); `Next` writes
                // the primed one (`args' = <<key>>`). Both are writers.
                let target = match &left.node {
                    Expr::Prime(inner) => state_var_idx(&inner.node, self.registry),
                    other => state_var_idx(other, self.registry),
                };
                if target != Some(self.target_var_idx) {
                    return !expr_mentions_prime_var(expr, self.target_var_idx, self.registry);
                }
                match self.classify_writer(&right.node, scope) {
                    Some(arm) => {
                        out.push(arm);
                        true
                    }
                    None => false,
                }
            }
            // `UNCHANGED <<.., args, ..>>` preserves the current arm and so adds
            // no new shape to the universe.
            Expr::Unchanged(_) => true,
            Expr::Apply(op, args) => {
                let Some(name) = operator_ident_name(&op.node) else {
                    return !expr_mentions_prime_var(expr, self.target_var_idx, self.registry);
                };
                self.collect_operator(name, args, scope, visiting, out)
            }
            Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
                self.collect_operator(name, &[], scope, visiting, out)
            }
            _ => !expr_mentions_prime_var(expr, self.target_var_idx, self.registry),
        }
    }

    fn collect_operator(
        &mut self,
        name: &str,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
        out: &mut Vec<ScalarTupleArmEvidence>,
    ) -> bool {
        let Some((resolved_name, def)) =
            writer_safe_op_def(name, self.op_defs, Some(self.op_replacements))
        else {
            return true;
        };
        if def.params.len() != args.len() {
            return false;
        }
        // A recursive operator would re-enter with the same param domains; the
        // body is already being walked, so skipping is sound (no NEW shape).
        if !visiting.insert(resolved_name.to_owned()) {
            return true;
        }
        // Bind each parameter to the caller's argument domain so a body writer
        // `args' = <<key, val>>` resolves `key`/`val` to `Keys`/`Vals`.
        let mut added = Vec::with_capacity(def.params.len());
        for (param, arg) in def.params.iter().zip(args.iter()) {
            let domain = self.arg_domain(&arg.node, scope);
            let param_name = param.name.node.clone();
            scope.push(param_name.clone(), domain);
            added.push(param_name);
        }
        let ok = self.collect(&def.body.node, scope, visiting, out);
        for param_name in added {
            scope.pop(&param_name);
        }
        visiting.remove(resolved_name);
        ok
    }

    /// Classify one right-hand side into an arm, or `None` to fail closed.
    fn classify_writer(
        &self,
        rhs: &Expr,
        scope: &WriterProofScope,
    ) -> Option<ScalarTupleArmEvidence> {
        if let Expr::Tuple(elems) = rhs {
            // The empty tuple is a degenerate arm that would alias the zeroed
            // payload window of the scalar arm; refuse it.
            if elems.is_empty() {
                return None;
            }
            let mut positions = Vec::with_capacity(elems.len());
            for elem in elems {
                positions.push(self.scalar_domain_of(&elem.node, scope)?);
            }
            return Some(ScalarTupleArmEvidence::Tuple(positions));
        }
        let mut domain = self.scalar_domain_of(rhs, scope)?;
        // A scalar writer must name ONE value. A writer ranging over a whole
        // quantifier domain is still a sound scalar arm — every member joins the
        // arm universe — so keep them all.
        if domain.is_empty() {
            return None;
        }
        if domain.len() == 1 {
            return Some(ScalarTupleArmEvidence::Scalar(domain.remove(0)));
        }
        // Multi-valued scalar writer: fold into the scalar arm as several
        // singleton observations by returning the first and letting the caller
        // union the rest is NOT expressible here, so fail closed rather than
        // silently narrowing the universe.
        None
    }

    /// Resolve an expression to the finite set of scalar values it can take.
    fn scalar_domain_of(
        &self,
        expr: &Expr,
        scope: &WriterProofScope,
    ) -> Option<Vec<FlatScalarValue>> {
        // A quantifier-bound or parameter-bound name ranges over its domain.
        if let Expr::Ident(name, _) = expr {
            if let Some(domain) = scope.bound_domain(name) {
                return domain
                    .iter()
                    .map(flat_scalar_from_value)
                    .collect::<Option<Vec<_>>>()
                    .filter(|values| !values.is_empty());
            }
        }
        // Otherwise it must be a compile-time constant (a model value such as
        // `NIL`, a string, an int literal). Anything referring to state is not a
        // statically-known arm shape.
        let value =
            const_expr_to_value_with_replacements(expr, self.constants, Some(self.op_replacements))
                .or_else(|| const_expr_to_value(expr, self.constants))?;
        flat_scalar_from_value(&value).map(|scalar| vec![scalar])
    }

    fn arg_domain(&self, expr: &Expr, scope: &WriterProofScope) -> Option<Arc<[Value]>> {
        match expr {
            Expr::Ident(name, _) => scope.bound_domain(name),
            _ => None,
        }
    }

    fn push_bound_vars(
        &self,
        vars: &[BoundVar],
        scope: &mut WriterProofScope,
    ) -> Option<Vec<String>> {
        let mut added = Vec::with_capacity(vars.len());
        for var in vars {
            if !matches!(&var.pattern, None | Some(BoundPattern::Var(_))) {
                return None;
            }
            let domain = var.domain.as_ref().and_then(|domain| {
                type_domain_values_with_replacements(
                    &domain.node,
                    self.constants,
                    self.proof_domains,
                    Some(self.op_replacements),
                )
            });
            let name = match &var.pattern {
                Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
                _ => var.name.node.clone(),
            };
            scope.push(name.clone(), domain);
            added.push(name);
        }
        Some(added)
    }
}

/// WP-ARGS: collect scalar-or-tuple union proofs for every state variable whose
/// `Init`/`Next` writers are all classifiable into a scalar sentinel arm or a
/// fixed-arity tuple arm.
///
/// Empty unless the `TY_SCALAR_TUPLE_UNION` gate is on.
///
/// # Arm assembly
///
/// All tuple writers collapse into a SINGLE `Sequence` variant of capacity
/// `max_arity`, never one variant per arity. Two `Sequence` variants of
/// different capacity would both accept a short tuple (`<<k>>` fits both
/// `max_len: 1` and `max_len: 2`), making the tag ambiguous — `tagged_union_variant_for_value`
/// would then refuse to encode and the var would fail closed anyway. Folding
/// them keeps exactly two variants (scalar, sequence) with disjoint acceptance,
/// and the arity is recovered losslessly from the sequence's own length slot.
///
/// The sequence element layout is the `TaggedScalarUnion` over every element
/// value any position of any tuple writer can hold. Position precision is
/// deliberately dropped (`<<val, key>>` would also encode) — that costs
/// universe tightness, never soundness, because the encoding stays injective.
pub(crate) fn collect_scalar_tuple_union_var_writer_proofs(
    init_expr: &Expr,
    next_expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<ScalarTupleUnionVarWriterProof>,
) {
    if !super::state_layout::scalar_tuple_union_native_flat_primary_enabled() {
        return;
    }
    for var_idx in 0..registry.len() {
        let mut collector = ScalarTupleUnionWriterCollector {
            target_var_idx: var_idx,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
        };
        let mut arms = Vec::new();
        let mut ok = collector.collect(
            init_expr,
            &mut WriterProofScope::default(),
            &mut BTreeSet::new(),
            &mut arms,
        );
        ok = ok
            && collector.collect(
                next_expr,
                &mut WriterProofScope::default(),
                &mut BTreeSet::new(),
                &mut arms,
            );
        if std::env::var_os("TY_SCALAR_TUPLE_UNION_DEBUG").is_some() {
            eprintln!(
                "[scalar-tuple-union] var_idx={var_idx} writers_ok={ok} arms={arms:?} proof={:?}",
                scalar_tuple_union_proof_from_arms(&arms).is_some()
            );
        }
        if !ok {
            continue;
        }
        if let Some(proof) = scalar_tuple_union_proof_from_arms(&arms) {
            out.push(ScalarTupleUnionVarWriterProof { var_idx, proof });
        }
    }
}

/// Assemble a [`TaggedUnionProof`] from classified writer arms, or `None` when
/// the evidence does not describe a genuine scalar-plus-tuple union.
///
/// Thin wrapper over [`scalar_tuple_union_variants_from_arms`] so the arm
/// FOLDING is unit-testable without the process-global `TY_TAGGED_SCALAR_UNION`
/// gate that `TaggedUnionProof::new`'s per-variant finiteness check consults.
fn scalar_tuple_union_proof_from_arms(arms: &[ScalarTupleArmEvidence]) -> Option<TaggedUnionProof> {
    let variants = scalar_tuple_union_variants_from_arms(arms)?;
    TaggedUnionProof::new(variants, Arc::from("scalar-tuple-union:writer-coverage")).ok()
}

/// Fold classified writer arms into the union's variant layouts, in tag order:
/// `[scalar sentinel arm, arity-1 tuple arm, arity-2 tuple arm, ..]`.
///
/// Tuple arms are grouped by ARITY, and within an arity each POSITION keeps its
/// own universe. This is what makes every slot's encode statically known: btree's
/// `<<key, val>>` has position 0 drawn from `Keys` (`Int`) and position 1 from
/// `Vals` (`ModelValue`), and folding those two universes together would leave
/// the lane of a raw i64 element slot unrecoverable at runtime. Because the
/// arity is carried by the union TAG rather than by a length slot, distinct
/// arities are distinct variants and never contend for one width.
fn scalar_tuple_union_variants_from_arms(
    arms: &[ScalarTupleArmEvidence],
) -> Option<Vec<FlatValueLayout>> {
    let mut scalar_universe: Vec<FlatScalarValue> = Vec::new();
    // arity -> per-position universes, ascending by arity for canonical tags.
    let mut tuple_arities: BTreeMap<usize, Vec<Vec<FlatScalarValue>>> = BTreeMap::new();
    for arm in arms {
        match arm {
            ScalarTupleArmEvidence::Scalar(value) => {
                if !scalar_universe.contains(value) {
                    scalar_universe.push(value.clone());
                }
            }
            ScalarTupleArmEvidence::Tuple(positions) => {
                if positions.is_empty() {
                    // A zero-arity tuple carries no payload and would collide
                    // with any other empty payload window.
                    return None;
                }
                let merged = tuple_arities
                    .entry(positions.len())
                    .or_insert_with(|| vec![Vec::new(); positions.len()]);
                for (slot, position) in merged.iter_mut().zip(positions.iter()) {
                    for value in position {
                        if !slot.contains(value) {
                            slot.push(value.clone());
                        }
                    }
                }
            }
        }
    }
    // This carrier exists for the MIXED case. A var with only scalar writers is
    // an ordinary scalar (or a `TaggedScalarUnion`) and a var with only tuple
    // writers is an ordinary sequence; both are handled by existing inference
    // and must not be re-encoded here.
    if scalar_universe.is_empty() || tuple_arities.is_empty() {
        return None;
    }
    scalar_universe.sort();

    let mut variants = vec![scalar_position_layout(scalar_universe)?];
    for (_arity, mut positions) in tuple_arities {
        let element_layouts = positions
            .iter_mut()
            .map(|universe| {
                universe.sort();
                scalar_position_layout(std::mem::take(universe))
            })
            .collect::<Option<Vec<_>>>()?;
        variants.push(FlatValueLayout::HeterogeneousTuple { element_layouts });
    }
    Some(variants)
}

/// Narrowest sound layout for ONE position's finite value universe.
///
/// A universe whose members all share a slot type has no lane ambiguity, so the
/// position stores the RAW scalar and needs no index indirection — that is what
/// lets a native read of `args[2]` be a plain slot load. Only a genuinely mixed
/// universe (e.g. `{NIL} \cup Keys`, a `ModelValue` beside `Int`s) needs the
/// [`FlatValueLayout::TaggedScalarUnion`] index encoding, which is the same
/// encoding WP-05 already lowers.
fn scalar_position_layout(universe: Vec<FlatScalarValue>) -> Option<FlatValueLayout> {
    let slot_type = universe.first()?.slot_type();
    if universe.iter().all(|value| value.slot_type() == slot_type) {
        return Some(FlatValueLayout::Scalar(slot_type));
    }
    Some(FlatValueLayout::TaggedScalarUnion {
        proof: TaggedScalarUnionProof::new(
            universe,
            Arc::from("scalar-tuple-union:writer-coverage"),
        )
        .ok()?,
    })
}

/// Apply collected [`ScalarTupleUnionVarWriterProof`]s as whole-variable layout
/// overrides.
///
/// Only promotes a variable whose inferred layout is a fail-closed ONE-SLOT
/// scalar kind — the Init-sampling artifact this carrier exists to fix (btree
/// samples `args = NIL` and infers `Scalar(String)`, so every tuple write and
/// every `args[i]` read fails closed). A var that already inferred a compound
/// layout is left alone.
///
/// The slot count changes (1 → `1 + max_payload`), so this uses the resizing
/// override rather than `replace_var_kind_same_slots`.
pub(crate) fn apply_scalar_tuple_union_var_overrides(
    layout: &mut StateLayout,
    proofs: &[ScalarTupleUnionVarWriterProof],
    sample_rows: &[Vec<Value>],
) {
    for proof in proofs {
        let Some(var) = layout.var_layout(proof.var_idx) else {
            continue;
        };
        if !matches!(
            var.kind,
            VarLayoutKind::ScalarModelValue | VarLayoutKind::ScalarString
        ) {
            continue;
        }
        // Every sampled value must fit the union, otherwise that state would
        // fail flat serialization and disable flat-primary for the ENTIRE
        // layout. Fail closed by skipping the override instead.
        let union_layout = FlatValueLayout::TaggedUnion {
            proof: proof.proof.clone(),
        };
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx)
                    .is_some_and(|value| value_fits_flat_value_layout(value, &union_layout))
            });
        if !all_fit {
            continue;
        }
        layout.replace_var_kind_recompute(
            proof.var_idx,
            VarLayoutKind::Recursive {
                layout: union_layout,
            },
        );
    }
}

/// Source-level proof that a top-level state variable is a finite SUM TYPE over
/// DISTINCT SHAPES — a scalar model-value/string AND a bounded sequence of finite
/// scalars — e.g. btree `args \in {NIL} \cup {<<k>> : k \in Keys} \cup {<<k,v>> :
/// ...}`.
///
/// Unlike [`TaggedScalarUnionVarTypeProof`] (a scalar∪scalar union of one-slot
/// lanes), this is a shape union: the variants have different flat widths, so the
/// var is encoded as `1 tag slot + max_payload_slots` via
/// [`FlatValueLayout::TaggedUnion`]. The tag disambiguates variants of different
/// shape so `NIL` (tag 0) can never collide with a sequence (tag 1) regardless of
/// payload bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedUnionVarTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) proof: TaggedUnionProof,
    pub(crate) invariant: Arc<str>,
}

/// Build a [`TaggedUnionProof`] from a checked union-of-distinct-shapes type set
/// expression `T1 \cup T2 \cup ...` (e.g. `{NIL} \cup {<<k>>:...} \cup
/// {<<k,v>>:...}`).
///
/// The whole type set is const-evaluated to its finite value universe, then
/// partitioned by shape into distinct variant [`FlatValueLayout`]s:
///   * scalars (Bool/Int/String/ModelValue) -> distinct `Scalar` variants;
///   * tuples/sequences -> ONE `Sequence` variant whose `max_len` is the widest
///     observed length and whose element layout is the common element layout of
///     every sequence value (a checked whole-var invariant proves the bound, so
///     the sequence variant is finite/closed).
/// Merging all sequences into a single `Sequence(max_len)` variant keeps the
/// variants non-overlapping (a len-1 and a len-2 sequence both match the same
/// variant), which the encoder requires for an unambiguous tag. Fails closed
/// (`None`) for a non-constant type set, a non-scalar/non-sequence element, a
/// heterogeneous sequence element type, or a universe that does not form >=2
/// distinct finite variants (`TaggedUnionProof::new` enforces the rest).
fn tagged_union_variants_from_type_expr(
    expr: &Expr,
    invariant: &str,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<TaggedUnionProof> {
    let value = const_expr_to_value_with_replacements(expr, constants, Some(op_replacements))?;
    let set = value.to_sorted_set()?;
    let context = LayoutInferenceContext::default();
    let root = SequencePath::root(usize::MAX);

    let mut scalar_variants: Vec<FlatValueLayout> = Vec::new();
    let mut seq_element: Option<FlatValueLayout> = None;
    let mut seq_max_len: usize = 0;
    let mut saw_sequence = false;

    for value in set.iter() {
        match value {
            Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::String(_)
            | Value::ModelValue(_) => {
                let layout = infer_fixed_value_layout(value, &context, &root)?;
                if !scalar_variants.contains(&layout) {
                    scalar_variants.push(layout);
                }
            }
            Value::Tuple(elems) => {
                saw_sequence = true;
                seq_max_len = seq_max_len.max(elems.len());
                if !elems.is_empty() {
                    let elem_layout = infer_common_flat_layout(
                        elems.iter(),
                        &context,
                        &root.child(SequencePathStep::SequenceElement),
                    )?;
                    match &seq_element {
                        None => seq_element = Some(elem_layout),
                        Some(existing) if *existing == elem_layout => {}
                        // Positionally-heterogeneous sequence elements (e.g. btree
                        // `<<int-key, model-value-val>>`) are not a homogeneous
                        // `Sequence(element_layout)`. Fail closed — that needs a
                        // per-position/record variant, out of this increment.
                        Some(_) => return None,
                    }
                }
            }
            Value::Seq(seq) => {
                saw_sequence = true;
                seq_max_len = seq_max_len.max(seq.len());
                if !seq.is_empty() {
                    let elem_layout = infer_common_flat_layout(
                        seq.iter(),
                        &context,
                        &root.child(SequencePathStep::SequenceElement),
                    )?;
                    match &seq_element {
                        None => seq_element = Some(elem_layout),
                        Some(existing) if *existing == elem_layout => {}
                        Some(_) => return None,
                    }
                }
            }
            _ => return None,
        }
    }

    let mut variants = scalar_variants;
    if saw_sequence {
        let element_layout = seq_element.unwrap_or(FlatValueLayout::Scalar(SlotType::Int));
        // The checked whole-var invariant bounds the sequence length AND its
        // element type, so both the capacity and the element layout are proven.
        variants.push(FlatValueLayout::Sequence {
            bound: SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                invariant: Arc::from(invariant),
                element_invariant: Arc::from(invariant),
            },
            max_len: seq_max_len,
            element_layout: Box::new(element_layout),
        });
    }

    TaggedUnionProof::new(variants, Arc::from(invariant)).ok()
}

/// Collect [`TaggedUnionVarTypeProof`]s from a checked type invariant: whole-var
/// conjuncts `v \in T1 \cup T2 \cup ...` whose universe forms a finite sum type of
/// distinct shapes (scalar ∪ bounded-sequence). Mirrors
/// [`collect_tagged_scalar_union_var_type_proofs_with_ops`] but for the
/// shape-union (tag + payload) case.
pub(crate) fn collect_tagged_union_var_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedUnionVarTypeProof>,
) {
    collect_tagged_union_var_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        op_defs,
        op_replacements,
        &ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_tagged_union_var_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedUnionVarTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_union_var_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_tagged_union_var_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            else {
                return;
            };
            if !path.is_empty() {
                return;
            }
            if let Some(proof) = tagged_union_variants_from_type_expr(
                &right.node,
                invariant,
                constants,
                op_replacements,
            ) {
                if !out
                    .iter()
                    .any(|existing| existing.var_idx == var_idx && existing.proof == proof)
                {
                    out.push(TaggedUnionVarTypeProof {
                        var_idx,
                        proof,
                        invariant: Arc::from(invariant),
                    });
                }
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_union_var_type_proofs_inner(
                &def.body.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

/// Apply collected [`TaggedUnionVarTypeProof`]s: promote a polymorphic top-level
/// var whose value-derived layout collapsed to a fail-closed scalar/dynamic kind
/// to `Recursive { TaggedUnion }` (a `1 + max_payload`-slot tag+payload sum type).
/// The width changes, so this recomputes offsets. Fires only when the sampled
/// value fits a variant (belt-and-suspenders to the checked whole-var invariant).
///
/// NOTE: `FlatValueLayout::TaggedUnion` currently maps to `CompoundLayout::Dynamic`
/// on the native ABI and `supports_flat_primary()` returns `false`, so the var is
/// NOT flat-primary and every action touching it fails closed to the interpreter
/// (per-action). The promotion still gives the var a fixed, round-trip-correct
/// flat encoding (proven by `test_tagged_union_sum_type_roundtrips_and_fingerprints_distinctly`)
/// for fingerprinting; the native tag-dispatch lowering that would make it
/// flat-primary is future work (see the TaggedUnion ABI gap).
pub(crate) fn apply_tagged_union_var_overrides(
    layout: &mut StateLayout,
    proofs: &[TaggedUnionVarTypeProof],
    sample_rows: &[Vec<Value>],
) {
    for proof in proofs {
        let Some(var) = layout.var_layout(proof.var_idx) else {
            continue;
        };
        // Only promote a fail-closed observed kind for a polymorphic var: a
        // one-slot interned scalar (sampled as the scalar variant) or a
        // veto-demoted `Dynamic`. Never touch a proven primary-flat kind.
        if !matches!(
            var.kind,
            VarLayoutKind::ScalarModelValue | VarLayoutKind::ScalarString | VarLayoutKind::Dynamic
        ) {
            continue;
        }
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx).is_some_and(|value| {
                    super::flat_state::value_fits_flat_value_layout(
                        value,
                        &FlatValueLayout::TaggedUnion {
                            proof: proof.proof.clone(),
                        },
                    )
                })
            });
        if !all_fit {
            continue;
        }
        let new_kind = VarLayoutKind::Recursive {
            layout: FlatValueLayout::TaggedUnion {
                proof: proof.proof.clone(),
            },
        };
        layout.replace_var_kind_recompute(proof.var_idx, new_kind);
    }
}

/// A single recognized whole-variable write `v = rhs` (Init) or `v' = rhs`
/// (Next), captured with the binder/substitution scope active at the write site
/// so a tuple position that is a quantifier-bound variable can be typed from its
/// `\in` domain.
#[derive(Clone)]
struct TaggedUnionWrite {
    rhs: Expr,
    scope: DfScope,
}

/// WRITER-ANALYSIS inference for a polymorphic top-level SUM-TYPE var that has NO
/// `TypeOK` conjunct constraining it (e.g. btree `args`, whose reachable universe
/// is `{NIL} \cup {<<k>>:k\in Keys} \cup {<<k,v>>:k\in Keys,v\in Vals}` and is
/// visible ONLY through the Init/Next writes `args = NIL`, `args' = <<key>>`,
/// `args' = <<key, val>>`).
///
/// The variant set is derived from the classified write RHS shapes:
///   * a scalar constant RHS (`NIL`, a literal) -> a `Scalar` variant;
///   * a tuple literal `<<e0, e1, ...>>` -> a fixed-arity
///     [`FlatValueLayout::HeterogeneousTuple`] whose position `i` is typed from
///     `e_i` (a scalar constant, or a quantifier-bound variable typed from its
///     `\in` domain — `key \in Keys` is `Int`, `val \in Vals` a `ModelValue`).
///
/// # Soundness (fail-closed + retry backstop)
/// A var is promoted ONLY when EVERY one of its recognized writes classifies to a
/// scalar/tuple variant AND the distinct variants form a valid
/// [`TaggedUnionProof`] (>=2 non-overlapping injective shapes). If ANY write is
/// unclassifiable the var is skipped (never a partial promotion). Even a promoted
/// var is fail-safe: each variant encoder is injective and the variants are
/// non-overlapping, so a reachable value either encodes to a unique `(tag,
/// payload)` or — if it fits no variant (an unmodeled write shape slipped past the
/// census) — makes `try_write_flat_value_slots` raise
/// `TaggedUnionValueOutsideUniverse`, which the CLI catches and transparently
/// retries WITHOUT flat storage (correct count, never a silent undercount). So an
/// incomplete census can only cost performance, never soundness. Gated by the
/// caller on `TY_TAGGED_UNION`.
pub(crate) fn collect_tagged_union_var_writer_proofs_with_ops(
    init_expr: &Expr,
    next_expr: &Expr,
    proof_source: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedUnionVarTypeProof>,
) {
    let mut writes: BTreeMap<usize, Vec<TaggedUnionWrite>> = BTreeMap::new();
    let mut scope = DfScope::default();
    walk_tagged_union_writes(
        init_expr,
        &mut scope,
        false,
        registry,
        op_defs,
        op_replacements,
        &mut BTreeSet::new(),
        &mut writes,
    );
    let mut scope = DfScope::default();
    walk_tagged_union_writes(
        next_expr,
        &mut scope,
        true,
        registry,
        op_defs,
        op_replacements,
        &mut BTreeSet::new(),
        &mut writes,
    );

    for (var_idx, var_writes) in writes {
        // Never touch a var already covered by a TypeOK-derived proof.
        if out.iter().any(|existing| existing.var_idx == var_idx) {
            continue;
        }
        let mut variants: Vec<FlatValueLayout> = Vec::new();
        let mut all_classified = true;
        for write in &var_writes {
            match classify_tagged_union_write_variant(
                &write.rhs,
                &write.scope,
                constants,
                op_replacements,
            ) {
                Some(variant) => {
                    if !variants.contains(&variant) {
                        variants.push(variant);
                    }
                }
                None => {
                    // An unclassifiable write means the writer-derived universe is
                    // incomplete; do NOT promote (a partial variant set would only
                    // force the retry-without-flat backstop on the missed values).
                    all_classified = false;
                    break;
                }
            }
        }
        if !all_classified {
            continue;
        }
        // A shape-union (tag + payload) is only the right encoding when at least
        // one variant is a genuine NON-scalar shape (a tuple). A var whose writes
        // are ALL scalars (e.g. btree `op \in {"get","insert",NIL}`) is a
        // one-slot scalar UNION — leave it to the `TaggedScalarUnion` path (a
        // single injective universe-index slot) rather than spend a wider
        // tag+payload here. This keeps the two promotions from competing for the
        // same var, especially when `TY_TAGGED_SCALAR_UNION` is off.
        if !variants
            .iter()
            .any(|variant| !matches!(variant, FlatValueLayout::Scalar(_)))
        {
            continue;
        }
        // At least two DISTINCT shapes are required for a tagged union (a single
        // shape is not a sum type). `TaggedUnionProof::new` enforces the rest
        // (finiteness of every variant, no duplicate/overlapping layouts).
        if let Ok(proof) = TaggedUnionProof::new(variants, Arc::from(proof_source)) {
            out.push(TaggedUnionVarTypeProof {
                var_idx,
                proof,
                invariant: Arc::from(proof_source),
            });
        }
    }
}

/// Classify a single write RHS into a tagged-union variant layout, or `None`
/// (fail closed) when the RHS is not a scalar constant or a fixed-arity tuple of
/// scalar-typed positions.
fn classify_tagged_union_write_variant(
    rhs: &Expr,
    scope: &DfScope,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<FlatValueLayout> {
    let context = LayoutInferenceContext::default();
    let root = SequencePath::root(usize::MAX);
    // A whole-RHS scalar constant (e.g. `args = NIL`, `x' = "get"`).
    if let Some(value) =
        const_expr_to_value_with_replacements(rhs, constants, Some(op_replacements))
    {
        let layout = infer_fixed_value_layout(&value, &context, &root)?;
        return matches!(layout, FlatValueLayout::Scalar(_)).then_some(layout);
    }
    // A tuple literal `<<e0, e1, ...>>` with per-position scalar types.
    if let Expr::Tuple(elems) = rhs {
        if elems.is_empty() {
            return None;
        }
        let mut element_layouts = Vec::with_capacity(elems.len());
        for elem in elems {
            let child_path = root.child(SequencePathStep::SequenceElement);
            let elem_layout = classify_tagged_union_tuple_position(
                &elem.node,
                scope,
                constants,
                op_replacements,
                &context,
                &child_path,
            )?;
            element_layouts.push(elem_layout);
        }
        return Some(FlatValueLayout::HeterogeneousTuple { element_layouts });
    }
    None
}

/// Type a single tuple position: a scalar constant, or a quantifier-bound
/// variable typed from its `\in` domain (following operator-parameter
/// substitutions). Only a SCALAR position layout is accepted (fail closed
/// otherwise) so the tuple variant stays a flat, natively-lowerable shape.
fn classify_tagged_union_tuple_position(
    elem: &Expr,
    scope: &DfScope,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    context: &LayoutInferenceContext,
    path: &SequencePath,
) -> Option<FlatValueLayout> {
    // A constant-valued position (a literal or a config-overridden model value).
    if let Some(value) =
        const_expr_to_value_with_replacements(elem, constants, Some(op_replacements))
    {
        let layout = infer_fixed_value_layout(&value, context, path)?;
        return matches!(layout, FlatValueLayout::Scalar(_)).then_some(layout);
    }
    // A bound variable: resolve its binder domain (or an operator-argument
    // substitution chain) and type it from the domain's scalar universe.
    if let Expr::Ident(name, _) = elem {
        match scope.get(name).cloned() {
            Some(DfScopeEntry::Binder {
                domain: Some(domain),
                ..
            }) => {
                let domain_value = const_expr_to_value_with_replacements(
                    &domain,
                    constants,
                    Some(op_replacements),
                )?;
                let set = domain_value.to_sorted_set()?;
                if set.is_empty() {
                    return None;
                }
                let layout = infer_common_flat_layout(set.iter(), context, path)?;
                return matches!(layout, FlatValueLayout::Scalar(_)).then_some(layout);
            }
            Some(DfScopeEntry::Subst {
                expr,
                scope: sub_scope,
            }) => {
                return classify_tagged_union_tuple_position(
                    &expr,
                    &sub_scope,
                    constants,
                    op_replacements,
                    context,
                    path,
                );
            }
            _ => {}
        }
    }
    None
}

/// Structural walk collecting recognized whole-variable writes (`v = rhs` /
/// `v' = rhs`) with their active binder/substitution scope. Descends both arms of
/// `\/`, `/\`, and `IF`, tracks `\E`/`\A` binder domains and `LET`/operator
/// parameter substitutions, and inlines resolvable zero-recursion operators.
/// Unrecognized forms are simply not collected (an incomplete census only costs
/// the retry-without-flat backstop, never soundness — see
/// [`collect_tagged_union_var_writer_proofs_with_ops`]).
#[allow(clippy::too_many_arguments)]
fn walk_tagged_union_writes(
    expr: &Expr,
    scope: &mut DfScope,
    primed: bool,
    registry: &VarRegistry,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    writes: &mut BTreeMap<usize, Vec<TaggedUnionWrite>>,
) {
    match expr {
        Expr::And(left, right) | Expr::Or(left, right) => {
            walk_tagged_union_writes(
                &left.node,
                scope,
                primed,
                registry,
                op_defs,
                op_replacements,
                visiting,
                writes,
            );
            walk_tagged_union_writes(
                &right.node,
                scope,
                primed,
                registry,
                op_defs,
                op_replacements,
                visiting,
                writes,
            );
        }
        Expr::If(_, then_expr, else_expr) => {
            walk_tagged_union_writes(
                &then_expr.node,
                scope,
                primed,
                registry,
                op_defs,
                op_replacements,
                visiting,
                writes,
            );
            walk_tagged_union_writes(
                &else_expr.node,
                scope,
                primed,
                registry,
                op_defs,
                op_replacements,
                visiting,
                writes,
            );
        }
        Expr::Exists(bounds, body) | Expr::Forall(bounds, body) => {
            let mut pushed: Vec<String> = Vec::new();
            for bound in bounds {
                let names = tla_core::visit::single_bound_var_names(bound);
                let plain_domain = if bound.pattern.is_none() && names.len() == 1 {
                    bound.domain.as_ref().map(|domain| domain.node.clone())
                } else {
                    None
                };
                for name in names {
                    self_scope_push_binder(scope, name.clone(), plain_domain.clone());
                    pushed.push(name);
                }
            }
            walk_tagged_union_writes(
                &body.node,
                scope,
                primed,
                registry,
                op_defs,
                op_replacements,
                visiting,
                writes,
            );
            for name in pushed.into_iter().rev() {
                scope.pop(&name);
            }
        }
        Expr::Let(defs, body) => {
            let mut pushed: Vec<String> = Vec::new();
            for def in defs {
                let entry = if def.params.is_empty() {
                    DfScopeEntry::Subst {
                        expr: def.body.node.clone(),
                        scope: scope.clone(),
                    }
                } else {
                    // A parameterized LET shadows its name so a position never
                    // resolves it as a scalar domain; such a write fails closed.
                    DfScopeEntry::Binder {
                        domain: None,
                        domain_scope: DfScope::default(),
                    }
                };
                scope.push(def.name.node.clone(), entry);
                pushed.push(def.name.node.clone());
            }
            walk_tagged_union_writes(
                &body.node,
                scope,
                primed,
                registry,
                op_defs,
                op_replacements,
                visiting,
                writes,
            );
            for name in pushed.into_iter().rev() {
                scope.pop(&name);
            }
        }
        Expr::Eq(left, right) => {
            let target = if primed {
                match &left.node {
                    Expr::Prime(inner) => tagged_union_write_target(&inner.node, registry),
                    _ => None,
                }
            } else {
                tagged_union_write_target(&left.node, registry)
            };
            if let Some(var_idx) = target {
                writes.entry(var_idx).or_default().push(TaggedUnionWrite {
                    rhs: right.node.clone(),
                    scope: scope.clone(),
                });
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            if scope.get(name).is_some() {
                return;
            }
            if let Some((resolved, def)) =
                tagged_union_expandable_op(name, 0, op_defs, op_replacements)
            {
                if visiting.insert(resolved.clone()) {
                    let mut body_scope = DfScope::default();
                    walk_tagged_union_writes(
                        &def.body.node,
                        &mut body_scope,
                        primed,
                        registry,
                        op_defs,
                        op_replacements,
                        visiting,
                        writes,
                    );
                    visiting.remove(&resolved);
                }
            }
        }
        Expr::Apply(op, args) => {
            let Some(name) = operator_ident_name(&op.node) else {
                return;
            };
            if scope.get(name).is_some() {
                return;
            }
            if let Some((resolved, def)) =
                tagged_union_expandable_op(name, args.len(), op_defs, op_replacements)
            {
                if visiting.insert(resolved.clone()) {
                    let mut body_scope = DfScope::default();
                    for (param, arg) in def.params.iter().zip(args) {
                        body_scope.push(
                            param.name.node.clone(),
                            DfScopeEntry::Subst {
                                expr: arg.node.clone(),
                                scope: scope.clone(),
                            },
                        );
                    }
                    walk_tagged_union_writes(
                        &def.body.node,
                        &mut body_scope,
                        primed,
                        registry,
                        op_defs,
                        op_replacements,
                        visiting,
                        writes,
                    );
                    visiting.remove(&resolved);
                }
            }
        }
        _ => {}
    }
}

/// Resolve a write LHS to a root state-variable index (by NAME, mirroring
/// `df_registry_var_idx`). A primed LHS is always a state variable, so the
/// registry lookup is authoritative.
fn tagged_union_write_target(expr: &Expr, registry: &VarRegistry) -> Option<usize> {
    match expr {
        Expr::StateVar(name, _, _) | Expr::Ident(name, _) => {
            registry.get(name).map(|idx| idx.as_usize())
        }
        _ => None,
    }
}

/// A resolvable, non-recursive, all-zero-arity-parameter operator suitable for
/// structural inlining during the write census. Mirrors
/// `DfSeqAnalysis::expandable_op`.
fn tagged_union_expandable_op<'a>(
    name: &str,
    args_len: usize,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
) -> Option<(String, &'a OperatorDef)> {
    let resolved = resolve_layout_op_name(name, Some(op_replacements))?;
    let def = op_defs.get(resolved)?.as_ref();
    (def.params.len() == args_len
        && !def.is_recursive
        && def.params.iter().all(|param| param.arity == 0))
    .then(|| (resolved.to_owned(), def))
}

/// Source-level proof that a whole *function* variable's range is a heterogeneous
/// finite scalar union (e.g. btree `lastOf \in [Nodes -> Nodes \cup {NIL}]`).
///
/// Unlike [`TaggedScalarUnionVarTypeProof`] (a one-slot top-level scalar var),
/// this carries the FULL proven `[D -> union]` [`FlatValueLayout`] (an
/// `IntFunction`/`Function` whose `value_layout` is `TaggedScalarUnion`). The
/// override replaces the var's fail-closed observed function kind (an `IntArray`
/// / `StringKeyedArray` whose heterogeneous range sampled to `Dynamic`/mixed
/// element types, so the union index never reaches native code) with
/// `Recursive { <this layout> }`, so every range slot stores the injective
/// universe index and the native FuncApply/FuncExcept union lowering engages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarUnionFunctionVarTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) layout: FlatValueLayout,
    pub(crate) invariant: Arc<str>,
}

/// The `TaggedScalarUnionProof` of a `[D -> union]` function layout whose
/// `value_layout` is directly a `TaggedScalarUnion`, else `None`. Only this
/// shape (function of a single-slot scalar-union range) is promoted for the
/// first increment; any richer nesting fails closed (the observed layout is
/// kept, so those vars stay on the interpreter — never a wrong encoding).
fn function_range_union_proof(layout: &FlatValueLayout) -> Option<&TaggedScalarUnionProof> {
    let value_layout = match layout {
        FlatValueLayout::IntFunction { value_layout, .. }
        | FlatValueLayout::Function { value_layout, .. } => value_layout.as_ref(),
        _ => return None,
    };
    match value_layout {
        FlatValueLayout::TaggedScalarUnion { proof } => Some(proof),
        _ => None,
    }
}

/// Whether every range value of a sampled function `value` is a scalar inside
/// the union `universe`. Fail-closed: a non-function sample, or any out-of
/// -universe / non-scalar range value, returns `false` (skip the promotion).
fn function_range_values_fit_union(value: &Value, universe: &[FlatScalarValue]) -> bool {
    let mut all_fit = true;
    let mut any = false;
    let mut check = |v: &Value| {
        any = true;
        match flat_scalar_from_value(v) {
            Some(flat) if universe.contains(&flat) => {}
            _ => all_fit = false,
        }
    };
    match value {
        Value::IntFunc(f) => {
            for v in f.values() {
                check(v);
            }
        }
        Value::Func(f) => {
            for (_, v) in f.iter() {
                check(v);
            }
        }
        _ => return false,
    }
    // An empty function trivially fits (no range slot can be out-of-universe);
    // `any` guards nothing here but keeps the intent explicit.
    let _ = any;
    all_fit
}

/// Collect [`TaggedScalarUnionFunctionVarTypeProof`]s from a checked type
/// invariant: whole-var membership conjuncts `f \in [D -> union]` whose range
/// inference resolves to a heterogeneous finite scalar union. Mirrors
/// [`collect_tagged_scalar_union_var_type_proofs_with_ops`] but for the function
/// -range case. No-op on the default surface (`TY_TAGGED_SCALAR_UNION` gates the
/// union construction).
pub(crate) fn collect_tagged_scalar_union_function_var_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedScalarUnionFunctionVarTypeProof>,
) {
    collect_tagged_scalar_union_function_var_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        op_defs,
        op_replacements,
        &ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_tagged_scalar_union_function_var_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarUnionFunctionVarTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_scalar_union_function_var_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_tagged_scalar_union_function_var_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            else {
                return;
            };
            // Only the *whole variable* (empty path) is a function var here; a
            // nested `f[k] \in ...` sub-path is a different (range-element) fact.
            if !path.is_empty() {
                return;
            }
            let Some(layout) = flat_layout_from_type_set_expr_with_ops(
                &right.node,
                constants,
                Some(op_defs),
                Some(op_replacements),
            ) else {
                return;
            };
            // Only admit a `[D -> single-slot-scalar-union]` layout for the first
            // increment (btree `lastOf`/`childOf`/`valOf`); richer nesting is
            // skipped (fail closed).
            if function_range_union_proof(&layout).is_none() {
                return;
            }
            if !out
                .iter()
                .any(|existing| existing.var_idx == var_idx && existing.layout == layout)
            {
                out.push(TaggedScalarUnionFunctionVarTypeProof {
                    var_idx,
                    layout,
                    invariant: Arc::from(invariant),
                });
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_union_function_var_type_proofs_inner(
                &def.body.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

/// Apply collected [`TaggedScalarUnionFunctionVarTypeProof`]s as whole-variable
/// layout overrides: promote a function var whose observed kind is a fail-closed
/// function (its heterogeneous range never carried the union) to
/// `Recursive { [D -> union] }` so each range slot stores the injective universe
/// index. Fires only when (1) the promoted layout has the SAME slot count as the
/// observed var (`replace_var_kind_same_slots` enforces this — a bijective
/// re-encoding), and (2) EVERY range value in EVERY sampled row is inside the
/// proven universe. A non-fitting promotion would make that whole state fail
/// flat serialization (disabling flat-primary for the entire layout), so it is
/// skipped — fail closed.
pub(crate) fn apply_tagged_scalar_union_function_var_overrides(
    layout: &mut StateLayout,
    proofs: &[TaggedScalarUnionFunctionVarTypeProof],
    sample_rows: &[Vec<Value>],
) {
    for proof in proofs {
        let Some(union_proof) = function_range_union_proof(&proof.layout) else {
            continue;
        };
        // Do not touch a var already promoted to a compound recursive layout or
        // demoted to `Dynamic` by a writer veto: only an as-yet fail-closed
        // OBSERVED function kind (IntArray / StringKeyedArray) whose range never
        // carried the union is a promotion candidate. Being conservative here
        // keeps a var that some other (sound) pass already shaped untouched.
        let Some(var) = layout.var_layout(proof.var_idx) else {
            continue;
        };
        if !matches!(
            var.kind,
            VarLayoutKind::IntArray { .. } | VarLayoutKind::StringKeyedArray { .. }
        ) {
            continue;
        }
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx).is_some_and(|value| {
                    function_range_values_fit_union(value, union_proof.universe())
                })
            });
        if !all_fit {
            continue;
        }
        let new_kind = VarLayoutKind::Recursive {
            layout: proof.layout.clone(),
        };
        // `replace_var_kind_same_slots` fails (leaves the var untouched) unless
        // the promoted layout's slot count matches the observed one — the
        // fail-closed backstop against a domain/width mismatch.
        layout.replace_var_kind_same_slots(proof.var_idx, new_kind);
    }
}

/// Proof metadata for a tuple/cross-product-keyed function var whose RANGE is a
/// heterogeneous finite scalar union (`f \in [D1 \X D2 -> s1 \cup s2]`, e.g.
/// btree `childOf \in [Nodes \X Keys -> Nodes \cup {NIL}]`).
///
/// Unlike [`TaggedScalarUnionFunctionVarTypeProof`] — which carries a full
/// scalar-domain `[D -> union]` `FlatValueLayout` — a tuple/cross-product domain
/// is NOT representable as a scalar `FlatValueLayout::Function.domain`
/// (`FlatScalarValue` is scalar-only). So only the range union universe is
/// carried; it is applied to the observed `TupleKeyedArray` var's existing
/// row-major domain by switching its range encoding to `TaggedScalarUnion`,
/// leaving `domain_keys`/`value_types`/slot-count untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedScalarUnionTupleFunctionVarTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) proof: TaggedScalarUnionProof,
    pub(crate) invariant: Arc<str>,
}

/// Collect [`TaggedScalarUnionTupleFunctionVarTypeProof`]s from a checked type
/// invariant: whole-var conjuncts `f \in [<tuple/cross-product domain> -> union]`
/// whose RANGE resolves to a heterogeneous finite scalar union. The scalar-domain
/// case is handled by [`collect_tagged_scalar_union_function_var_type_proofs_with_ops`];
/// this covers exactly the tuple-domain complement it cannot represent. No-op on
/// the default surface (the union construction is gated by `TY_TAGGED_SCALAR_UNION`).
pub(crate) fn collect_tagged_scalar_union_tuple_function_var_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedScalarUnionTupleFunctionVarTypeProof>,
) {
    collect_tagged_scalar_union_tuple_function_var_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        op_defs,
        op_replacements,
        &ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_tagged_scalar_union_tuple_function_var_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarUnionTupleFunctionVarTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_scalar_union_tuple_function_var_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_tagged_scalar_union_tuple_function_var_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            else {
                return;
            };
            // Only the whole variable (empty path) is a function var here.
            if !path.is_empty() {
                return;
            }
            let Expr::FuncSet(_domain, range) = &right.node else {
                return;
            };
            // Defer to the scalar-domain function override when the WHOLE
            // `[D -> union]` resolves to a scalar-domain layout; only the
            // tuple/cross-product domain (which cannot be a scalar
            // `FlatValueLayout::Function.domain`, so this returns `None`) is ours.
            if flat_layout_from_type_set_expr_with_ops(
                &right.node,
                constants,
                Some(op_defs),
                Some(op_replacements),
            )
            .is_some()
            {
                return;
            }
            // Extract JUST the range union universe. The same canonical universe
            // order the scalar-domain path (and the interpreter) uses.
            let Some(FlatValueLayout::TaggedScalarUnion { proof }) =
                flat_layout_from_type_set_expr_with_ops(
                    &range.node,
                    constants,
                    Some(op_defs),
                    Some(op_replacements),
                )
            else {
                return;
            };
            if !out
                .iter()
                .any(|existing| existing.var_idx == var_idx && existing.proof == proof)
            {
                out.push(TaggedScalarUnionTupleFunctionVarTypeProof {
                    var_idx,
                    proof,
                    invariant: Arc::from(invariant),
                });
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_union_tuple_function_var_type_proofs_inner(
                &def.body.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

/// Apply collected [`TaggedScalarUnionTupleFunctionVarTypeProof`]s: promote an
/// observed scalar-slot `TupleKeyedArray` (whose heterogeneous all-`NIL`-sampled
/// range collapsed to a fail-closed flat-primary blocker) to the SAME tuple
/// layout with a `TaggedScalarUnion` range encoding, so each row-major range
/// slot stores the injective universe index and the native tuple FuncApply /
/// FuncExcept union lowering engages. Fires only when (1) the var is still a
/// scalar-slot `TupleKeyedArray` (untouched by any other pass) and (2) EVERY
/// range value in EVERY sampled row is inside the proven universe — belt-and
/// -suspenders to the checked whole-var invariant's range-closure guarantee.
/// `replace_var_kind_same_slots` is a no-op unless the slot count is unchanged,
/// which it always is here (same `domain_keys`).
pub(crate) fn apply_tagged_scalar_union_tuple_function_var_overrides(
    layout: &mut StateLayout,
    proofs: &[TaggedScalarUnionTupleFunctionVarTypeProof],
    sample_rows: &[Vec<Value>],
) {
    for proof in proofs {
        let (domain_keys, value_types) = {
            let Some(var) = layout.var_layout(proof.var_idx) else {
                continue;
            };
            let VarLayoutKind::TupleKeyedArray {
                domain_keys,
                value_types,
                range_encoding: TupleKeyedArrayRangeEncoding::ScalarSlots,
            } = &var.kind
            else {
                continue;
            };
            (domain_keys.clone(), value_types.clone())
        };
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx).is_some_and(|value| {
                    function_range_values_fit_union(value, proof.proof.universe())
                })
            });
        if !all_fit {
            continue;
        }
        let new_kind = VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof.proof.clone()),
        };
        layout.replace_var_kind_same_slots(proof.var_idx, new_kind);
    }
}

/// Proof metadata for a tuple/cross-product-keyed function var whose RANGE is a
/// HOMOGENEOUS finite model-value/string set (`f \in [D1 \X D2 -> {a, b, …}]`,
/// e.g. btree `valOf \in [Nodes \X Keys -> Vals \cup {NIL}]` where `Vals \cup
/// {NIL}` = `{x, y, z, nil}` is homogeneous model-value).
///
/// The tuple analogue of the `FixedScalar` arm of `StringKeyedArrayRangeEncoding`
/// / the 1-D [`FixedScalarRangeTypeProof`]. Unlike that 1-D proof it carries no
/// domain (the cross-product domain isn't a scalar `type_domain_values`); the
/// homogeneous scalar universe is applied to the observed `TupleKeyedArray`'s
/// row-major domain by switching its range encoding to `FixedScalar`. The raw
/// `NameId` slot encoding is unchanged — the proof only certifies flat-primary
/// safety (the universe is non-int, so the `NameId` slot can never alias an int).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FixedScalarRangeTupleFunctionVarTypeProof {
    pub(crate) var_idx: usize,
    pub(crate) proof: FixedScalarRangeProof,
    pub(crate) invariant: Arc<str>,
}

/// Enumerate a function-range type expression to its finite scalar universe,
/// tolerating the `\union` shapes (`Vals \cup {NIL}`) that the structural
/// [`scalar_domain_from_type_set_expr_scoped`] does not itself model, by
/// recursively resolving each union arm with that same primitive (which resolves
/// model-value/string idents and set literals — exactly the resolution the
/// `TaggedScalarUnion` path uses, so a model-value operator-override like btree's
/// `NIL == CHOOSE …` / `NIL = nil` resolves identically). Fails closed for
/// anything not finitely enumerable at inference time.
fn resolve_finite_scalar_set_universe(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<Vec<FlatScalarValue>> {
    if let Expr::Union(left, right) = expr {
        let mut left_universe = resolve_finite_scalar_set_universe(
            &left.node,
            constants,
            op_defs,
            op_replacements,
            scope,
            visiting,
        )?;
        let right_universe = resolve_finite_scalar_set_universe(
            &right.node,
            constants,
            op_defs,
            op_replacements,
            scope,
            visiting,
        )?;
        left_universe.extend(right_universe);
        return normalize_flat_scalar_domain(left_universe);
    }
    scalar_domain_from_type_set_expr_scoped(
        expr,
        constants,
        op_defs,
        op_replacements,
        scope,
        visiting,
    )
}

/// Evaluate a function-range type expression to its HOMOGENEOUS finite scalar
/// universe. Returns `None` (fail closed) for a non-enumerable or heterogeneous
/// (mixed-type) range — a heterogeneous `int \cup model-value` range is the
/// `TaggedScalarUnion` path's job, not this one. The universe ORDER does not
/// matter here (unlike the union index): a `FixedScalar` slot is the raw
/// interned `NameId`, so the universe is used only for membership checks.
fn tuple_function_range_homogeneous_universe(
    range: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    let mut visiting = BTreeSet::new();
    let universe = resolve_finite_scalar_set_universe(
        range,
        constants,
        op_defs,
        Some(op_replacements),
        &LayoutScope::new(),
        &mut visiting,
    )?;
    finite_homogeneous_scalar_domain_from_flat_values(universe)
}

/// Collect [`FixedScalarRangeTupleFunctionVarTypeProof`]s from a checked type
/// invariant: whole-var conjuncts `f \in [<tuple/cross-product domain> ->
/// <homogeneous finite model-value/string set>]`. The scalar-domain case is the
/// 1-D FixedScalar range path; the heterogeneous-range case is the
/// `TaggedScalarUnion` path — this covers exactly the homogeneous tuple-domain
/// complement. Only `String`/`ModelValue` ranges are collected (`Int`/`Bool`
/// homogeneous ranges are already plain-i64 flat-primary safe as `ScalarSlots`).
pub(crate) fn collect_fixed_scalar_range_tuple_function_var_type_proofs_with_ops(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<FixedScalarRangeTupleFunctionVarTypeProof>,
) {
    collect_fixed_scalar_range_tuple_function_var_type_proofs_inner(
        expr,
        invariant,
        registry,
        constants,
        op_defs,
        op_replacements,
        &ElementProofScope::default(),
        &mut BTreeSet::new(),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_fixed_scalar_range_tuple_function_var_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarRangeTupleFunctionVarTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_fixed_scalar_range_tuple_function_var_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_fixed_scalar_range_tuple_function_var_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            else {
                return;
            };
            if !path.is_empty() {
                return;
            }
            let Expr::FuncSet(_domain, range) = &right.node else {
                return;
            };
            // Defer to the scalar-domain 1-D FixedScalar path when the whole
            // `[D -> range]` resolves to a scalar-domain layout; only the
            // tuple/cross-product domain (which returns `None`) is ours.
            if flat_layout_from_type_set_expr_with_ops(
                &right.node,
                constants,
                Some(op_defs),
                Some(op_replacements),
            )
            .is_some()
            {
                return;
            }
            let Some((scalar_type, scalar_universe)) = tuple_function_range_homogeneous_universe(
                &range.node,
                constants,
                op_defs,
                op_replacements,
            ) else {
                return;
            };
            // Only string/model-value ranges need FixedScalar; a homogeneous
            // Int/Bool range is already plain-i64 flat-primary safe as ScalarSlots.
            if !matches!(scalar_type, SlotType::String | SlotType::ModelValue) {
                return;
            }
            let Ok(proof) =
                FixedScalarRangeProof::new(scalar_type, scalar_universe, Arc::from(invariant))
            else {
                return;
            };
            if !out
                .iter()
                .any(|existing| existing.var_idx == var_idx && existing.proof == proof)
            {
                out.push(FixedScalarRangeTupleFunctionVarTypeProof {
                    var_idx,
                    proof,
                    invariant: Arc::from(invariant),
                });
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_fixed_scalar_range_tuple_function_var_type_proofs_inner(
                &def.body.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

/// Apply collected [`FixedScalarRangeTupleFunctionVarTypeProof`]s: promote an
/// observed scalar-slot `TupleKeyedArray` whose model-value/string range never
/// earned flat-primary safety to the SAME tuple layout with a `FixedScalar`
/// range encoding. Fires only when (1) the var is still a scalar-slot
/// `TupleKeyedArray` whose sampled `value_types` are all the proof's scalar type
/// (so a plain-i64 array — already primary-safe — is never touched) and (2) every
/// sampled range value is inside the proven universe (belt-and-suspenders to the
/// checked whole-var invariant's range-closure guarantee). `replace_var_kind_same_slots`
/// keeps the slot count unchanged (same `domain_keys`).
pub(crate) fn apply_fixed_scalar_range_tuple_function_var_overrides(
    layout: &mut StateLayout,
    proofs: &[FixedScalarRangeTupleFunctionVarTypeProof],
    sample_rows: &[Vec<Value>],
) {
    for proof in proofs {
        let (domain_keys, value_types) = {
            let Some(var) = layout.var_layout(proof.var_idx) else {
                continue;
            };
            let VarLayoutKind::TupleKeyedArray {
                domain_keys,
                value_types,
                range_encoding: TupleKeyedArrayRangeEncoding::ScalarSlots,
            } = &var.kind
            else {
                continue;
            };
            // Only a homogeneous model-value/string array (the kind ScalarSlots
            // can't make primary-safe) is a candidate; a plain-i64 array is
            // already safe and must be left untouched.
            if !value_types
                .iter()
                .all(|ty| *ty == proof.proof.scalar_type())
            {
                continue;
            }
            (domain_keys.clone(), value_types.clone())
        };
        let all_fit = !sample_rows.is_empty()
            && sample_rows.iter().all(|row| {
                row.get(proof.var_idx).is_some_and(|value| {
                    function_range_values_fit_union(value, proof.proof.scalar_universe())
                })
            });
        if !all_fit {
            continue;
        }
        let new_kind = VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding: TupleKeyedArrayRangeEncoding::FixedScalar(proof.proof.clone()),
        };
        layout.replace_var_kind_same_slots(proof.var_idx, new_kind);
    }
}

/// G2: collect top-level finite-universe scalar (model-value) var proofs from an
/// `Init`/`Next` pair, for specs that constrain the variable only in `Init`
/// (e.g. DijkstraMutex `k \in Proc` with no `TypeOK` invariant).
///
/// Unlike [`collect_fixed_scalar_var_type_proofs_with_ops`], which trusts a
/// *checked* `TypeOK` invariant as a closure proof, an `Init` clause `v \in S`
/// only constrains the *initial* state. To stay sound we additionally require
/// the writer-coverage analysis [`preserved_model_value_scalar_domains`] to
/// prove that `v`'s model-value domain `S` is **closed under every `Next`
/// writer** (every assignment to `v` writes a value that provably stays in `S`,
/// reaching a fixpoint). Only when that closure proof succeeds is the var
/// admitted as a primary-flat `FixedScalar` slot.
///
/// Soundness: this is the exact closure obligation a checked `TypeOK(v \in S)`
/// invariant already discharges, established here directly from the transition
/// relation instead. A `ScalarModelValue` whose universe is proven total/closed
/// has all reachable values inside a known finite interned set, each fitting one
/// slot with no cross-type aliasing — identical basis to the existing
/// `FixedScalar` (invariant-sourced) proof. Vars whose domain is not proven
/// closed under `Next` are dropped, so an init-sampled scalar can never be
/// admitted (the flat-primary-for-unproven-sampled-scalar wall is preserved).
pub(crate) fn collect_fixed_scalar_var_writer_proofs_with_ops(
    init_expr: &Expr,
    next_expr: &Expr,
    proof_source: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<FixedScalarVarTypeProof>,
) {
    let preserved = preserved_model_value_scalar_domains(
        init_expr,
        next_expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
    );
    for (var_idx, domain) in preserved {
        // Re-derive the canonical flat scalar universe from the proven-closed
        // model-value domain. `model_value_flat_domain` fails closed for empty,
        // oversized (> 63), or non-homogeneous-model-value domains, so anything
        // that slips through here is a sound, non-empty model-value universe.
        let Some(scalar_universe) = model_value_flat_domain(domain.as_ref()) else {
            continue;
        };
        push_fixed_scalar_var_type_proof(
            out,
            FixedScalarVarTypeProof {
                var_idx,
                path: Vec::new(),
                scalar_type: SlotType::ModelValue,
                scalar_universe,
                invariant: Arc::from(proof_source),
            },
        );
    }
}

pub(crate) fn collect_tagged_scalar_set_range_writer_proofs_with_ops(
    init_expr: &Expr,
    next_expr: &Expr,
    proof_source: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<TaggedScalarSetRangeTypeProof>,
) {
    let init_candidates = collect_tagged_scalar_set_range_init_candidates(
        init_expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
    );
    if init_candidates.is_empty() {
        return;
    }

    let scalar_domains = preserved_model_value_scalar_domains(
        init_expr,
        next_expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
    );

    for candidate in init_candidates {
        let Some(set_universe) = model_value_flat_domain(candidate.domain.as_ref()) else {
            continue;
        };
        let mut checker = TaggedRangeWriterChecker {
            target_var_idx: candidate.var_idx,
            function_domain: Arc::clone(&candidate.domain),
            set_universe_values: Arc::clone(&candidate.domain),
            scalar_domains: &scalar_domains,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
        };
        let mut scope = WriterProofScope::default();
        let mut visiting = BTreeSet::new();
        let coverage = checker.prove(next_expr, &mut scope, &mut visiting);
        if !matches!(
            coverage,
            WriterCoverage::Covered {
                saw_set_writer: true
            }
        ) {
            continue;
        }
        push_tagged_scalar_set_range_type_proof(
            out,
            TaggedScalarSetRangeTypeProof {
                var_idx: candidate.var_idx,
                path: Vec::new(),
                domain: Arc::clone(&candidate.domain),
                scalar_type: SlotType::ModelValue,
                set_universe,
                invariant: Arc::from(proof_source),
            },
        );
    }
}

pub(crate) fn collect_fixed_scalar_range_writer_proofs_with_ops(
    init_expr: &Expr,
    next_expr: &Expr,
    proof_source: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    out: &mut Vec<FixedScalarRangeTypeProof>,
) {
    let init_candidates = collect_fixed_scalar_range_init_candidates(
        init_expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
    );
    if init_candidates.is_empty() {
        return;
    }

    for mut candidate in init_candidates {
        collect_fixed_scalar_range_next_values(
            next_expr,
            &mut candidate,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
        );
        let Some(scalar_universe) = normalize_flat_scalar_domain(candidate.scalar_universe) else {
            continue;
        };
        let mut checker = FixedScalarRangeWriterChecker {
            target_var_idx: candidate.var_idx,
            function_domain: Arc::clone(&candidate.domain),
            scalar_type: candidate.scalar_type,
            scalar_universe: scalar_universe.clone(),
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
        };
        let mut scope = WriterProofScope::default();
        let mut visiting = BTreeSet::new();
        if !matches!(
            checker.prove(next_expr, &mut scope, &mut visiting),
            WriterCoverage::Covered { .. }
        ) {
            continue;
        }
        push_fixed_scalar_range_type_proof(
            out,
            FixedScalarRangeTypeProof {
                var_idx: candidate.var_idx,
                path: Vec::new(),
                domain: Arc::clone(&candidate.domain),
                scalar_type: candidate.scalar_type,
                scalar_universe,
                invariant: Arc::from(proof_source),
            },
        );
    }
}

#[derive(Clone, PartialEq, Eq)]
struct TaggedRangeInitCandidate {
    var_idx: usize,
    domain: Arc<[Value]>,
}

#[derive(Clone, PartialEq, Eq)]
struct FixedScalarRangeInitCandidate {
    var_idx: usize,
    domain: Arc<[Value]>,
    scalar_type: SlotType,
    scalar_universe: Vec<FlatScalarValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriterCoverage {
    NotCovered,
    Covered { saw_set_writer: bool },
    Unsupported,
}

impl WriterCoverage {
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unsupported, _) | (_, Self::Unsupported) => Self::Unsupported,
            (Self::Covered { saw_set_writer: a }, Self::Covered { saw_set_writer: b }) => {
                Self::Covered {
                    saw_set_writer: a || b,
                }
            }
            (Self::Covered { saw_set_writer }, Self::NotCovered)
            | (Self::NotCovered, Self::Covered { saw_set_writer }) => {
                Self::Covered { saw_set_writer }
            }
            (Self::NotCovered, Self::NotCovered) => Self::NotCovered,
        }
    }

    fn branch(self, other: Self) -> Self {
        match (self, other) {
            (Self::Covered { saw_set_writer: a }, Self::Covered { saw_set_writer: b }) => {
                Self::Covered {
                    saw_set_writer: a || b,
                }
            }
            (Self::Unsupported, _) | (_, Self::Unsupported) => Self::Unsupported,
            (Self::NotCovered, Self::NotCovered) => Self::NotCovered,
            _ => Self::Unsupported,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaggedRangeExprKind {
    Scalar,
    Set,
}

#[derive(Default)]
struct WriterProofScope {
    bindings: BTreeMap<String, Vec<Option<Arc<[Value]>>>>,
}

impl WriterProofScope {
    fn push(&mut self, name: String, domain: Option<Arc<[Value]>>) {
        self.bindings.entry(name).or_default().push(domain);
    }

    fn pop(&mut self, name: &str) {
        if let Some(stack) = self.bindings.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.bindings.remove(name);
            }
        }
    }

    fn is_bound(&self, name: &str) -> bool {
        self.bindings
            .get(name)
            .is_some_and(|stack| !stack.is_empty())
    }

    fn bound_domain(&self, name: &str) -> Option<Arc<[Value]>> {
        self.bindings
            .get(name)
            .and_then(|stack| stack.last())
            .and_then(|domain| domain.as_ref().map(Arc::clone))
    }
}

struct TaggedRangeWriterChecker<'a> {
    target_var_idx: usize,
    function_domain: Arc<[Value]>,
    set_universe_values: Arc<[Value]>,
    scalar_domains: &'a BTreeMap<usize, Arc<[Value]>>,
    registry: &'a VarRegistry,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &'a BTreeMap<String, Arc<[Value]>>,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
}

impl TaggedRangeWriterChecker<'_> {
    fn prove(
        &mut self,
        expr: &Expr,
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        match expr {
            Expr::And(left, right) => {
                let left = self.prove(&left.node, scope, visiting);
                let right = self.prove(&right.node, scope, visiting);
                left.or(right)
            }
            Expr::Or(left, right) => {
                let left = self.prove(&left.node, scope, visiting);
                let right = self.prove(&right.node, scope, visiting);
                left.branch(right)
            }
            Expr::If(_, then_expr, else_expr) => {
                let then_coverage = self.prove(&then_expr.node, scope, visiting);
                let else_coverage = self.prove(&else_expr.node, scope, visiting);
                then_coverage.branch(else_coverage)
            }
            Expr::Exists(vars, body) | Expr::Forall(vars, body) => {
                let Some(added) = self.push_bound_vars(vars, scope) else {
                    return if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                        WriterCoverage::Unsupported
                    } else {
                        WriterCoverage::NotCovered
                    };
                };
                let coverage = self.prove(&body.node, scope, visiting);
                for name in added {
                    scope.pop(&name);
                }
                coverage
            }
            Expr::Eq(left, right) => self
                .prove_assignment(&left.node, &right.node, scope)
                .or_else(|| self.prove_assignment(&right.node, &left.node, scope))
                .unwrap_or_else(|| {
                    if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                        WriterCoverage::Unsupported
                    } else {
                        WriterCoverage::NotCovered
                    }
                }),
            Expr::Unchanged(vars) => {
                if unchanged_mentions_var(&vars.node, self.target_var_idx, self.registry) {
                    WriterCoverage::Covered {
                        saw_set_writer: false,
                    }
                } else {
                    WriterCoverage::NotCovered
                }
            }
            Expr::Apply(op, args) => self.prove_operator_call(&op.node, args, scope, visiting),
            Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
                self.prove_operator_name(name, &[], scope, visiting)
            }
            _ => {
                if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                    WriterCoverage::Unsupported
                } else {
                    WriterCoverage::NotCovered
                }
            }
        }
    }

    fn prove_assignment(
        &self,
        left: &Expr,
        right: &Expr,
        scope: &WriterProofScope,
    ) -> Option<WriterCoverage> {
        let Expr::Prime(inner) = left else {
            return None;
        };
        if state_var_idx(&inner.node, self.registry)? != self.target_var_idx {
            return None;
        }
        self.replacement_kind(right, scope)
            .map(|kind| WriterCoverage::Covered {
                saw_set_writer: matches!(kind, TaggedRangeExprKind::Set),
            })
            .or(Some(WriterCoverage::Unsupported))
    }

    fn replacement_kind(
        &self,
        expr: &Expr,
        scope: &WriterProofScope,
    ) -> Option<TaggedRangeExprKind> {
        if same_state_var(expr, self.target_var_idx, self.registry) {
            return Some(TaggedRangeExprKind::Scalar);
        }

        if self.scalar_expr_in_domain(expr, &self.set_universe_values, scope)
            || model_value_constant_expr(expr, self.constants, Some(self.op_replacements)).is_some()
        {
            return Some(TaggedRangeExprKind::Scalar);
        }

        let Expr::Except(base, specs) = expr else {
            return None;
        };
        if !same_state_var(&base.node, self.target_var_idx, self.registry) || specs.len() != 1 {
            return None;
        }
        let spec = &specs[0];
        if spec.path.len() != 1 {
            return None;
        }
        let ExceptPathElement::Index(index) = &spec.path[0] else {
            return None;
        };
        if !self.scalar_expr_in_domain(&index.node, &self.function_domain, scope) {
            return None;
        }
        self.range_value_kind(&spec.value.node, scope)
    }

    fn range_value_kind(
        &self,
        expr: &Expr,
        scope: &WriterProofScope,
    ) -> Option<TaggedRangeExprKind> {
        if self.scalar_expr_in_domain(expr, &self.set_universe_values, scope)
            || model_value_constant_expr(expr, self.constants, Some(self.op_replacements)).is_some()
        {
            return Some(TaggedRangeExprKind::Scalar);
        }
        if self.set_expr_in_universe(expr, scope) {
            return Some(TaggedRangeExprKind::Set);
        }
        None
    }

    fn set_expr_in_universe(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        match expr {
            Expr::SetEnum(elems) if elems.is_empty() => true,
            Expr::Ident(_, _)
                if domain_values_equal(
                    precomputed_constant_set_values(
                        expr,
                        self.constants,
                        Some(self.op_replacements),
                    )
                    .as_deref(),
                    self.set_universe_values.as_ref(),
                ) =>
            {
                true
            }
            Expr::SetMinus(left, right) => {
                self.set_source_in_universe(&left.node, scope)
                    && self.set_removal_in_universe(&right.node, scope)
            }
            _ => false,
        }
    }

    fn set_source_in_universe(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        if self.set_expr_in_universe(expr, scope) {
            return true;
        }
        if domain_values_equal(
            precomputed_constant_set_values(expr, self.constants, Some(self.op_replacements))
                .as_deref(),
            self.set_universe_values.as_ref(),
        ) {
            return true;
        }
        let Expr::FuncApply(func, arg) = expr else {
            return false;
        };
        same_state_var(&func.node, self.target_var_idx, self.registry)
            && self.scalar_expr_in_domain(&arg.node, &self.function_domain, scope)
    }

    fn set_removal_in_universe(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        match expr {
            Expr::SetEnum(elems) => elems.iter().all(|elem| {
                self.scalar_expr_in_domain(&elem.node, &self.set_universe_values, scope)
            }),
            _ => self.set_expr_in_universe(expr, scope),
        }
    }

    fn scalar_expr_in_domain(
        &self,
        expr: &Expr,
        domain: &[Value],
        scope: &WriterProofScope,
    ) -> bool {
        match expr {
            Expr::Ident(name, _) => {
                scope
                    .bound_domain(name)
                    .is_some_and(|bound_domain| bound_domain.as_ref() == domain)
                    || state_var_idx(expr, self.registry).is_some_and(|idx| {
                        self.scalar_domains
                            .get(&idx)
                            .is_some_and(|candidate| candidate.as_ref() == domain)
                    })
                    || const_expr_to_value_with_replacements(
                        expr,
                        self.constants,
                        Some(self.op_replacements),
                    )
                    .is_some_and(|value| domain.contains(&value))
            }
            Expr::StateVar(_, idx, _) => self
                .scalar_domains
                .get(&(*idx as usize))
                .is_some_and(|candidate| candidate.as_ref() == domain),
            _ => const_expr_to_value_with_replacements(
                expr,
                self.constants,
                Some(self.op_replacements),
            )
            .is_some_and(|value| domain.contains(&value)),
        }
    }

    fn push_bound_vars(
        &self,
        vars: &[BoundVar],
        scope: &mut WriterProofScope,
    ) -> Option<Vec<String>> {
        let mut added = Vec::with_capacity(vars.len());
        for var in vars {
            if !matches!(&var.pattern, None | Some(BoundPattern::Var(_))) {
                return None;
            }
            let domain = self.bound_var_domain(var, scope);
            let name = match &var.pattern {
                Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
                _ => var.name.node.clone(),
            };
            scope.push(name.clone(), domain);
            added.push(name);
        }
        Some(added)
    }

    fn bound_var_domain(&self, var: &BoundVar, scope: &WriterProofScope) -> Option<Arc<[Value]>> {
        let domain = var.domain.as_ref()?;
        type_domain_values_with_replacements(
            &domain.node,
            self.constants,
            self.proof_domains,
            Some(self.op_replacements),
        )
        .or_else(|| {
            self.bound_domain_from_target_func_apply(&domain.node, scope)
                .map(Arc::clone)
        })
    }

    fn bound_domain_from_target_func_apply(
        &self,
        expr: &Expr,
        scope: &WriterProofScope,
    ) -> Option<&Arc<[Value]>> {
        let Expr::FuncApply(func, arg) = expr else {
            return None;
        };
        (same_state_var(&func.node, self.target_var_idx, self.registry)
            && self.scalar_expr_in_domain(&arg.node, &self.function_domain, scope))
        .then_some(&self.set_universe_values)
    }

    fn prove_operator_call(
        &mut self,
        op: &Expr,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        let Some(name) = operator_ident_name(op) else {
            return WriterCoverage::Unsupported;
        };
        self.prove_operator_name(name, args, scope, visiting)
    }

    fn prove_operator_name(
        &mut self,
        name: &str,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        let Some((resolved_name, def)) =
            writer_safe_op_def(name, self.op_defs, Some(self.op_replacements))
        else {
            return WriterCoverage::Unsupported;
        };
        if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
            return WriterCoverage::Unsupported;
        }
        let mut added = Vec::with_capacity(def.params.len());
        for (param, arg) in def.params.iter().zip(args.iter()) {
            let domain = self.scalar_domain_for_arg(&arg.node, scope);
            let name = param.name.node.clone();
            scope.push(name.clone(), domain);
            added.push(name);
        }
        let coverage = self.prove(&def.body.node, scope, visiting);
        for name in added {
            scope.pop(&name);
        }
        visiting.remove(resolved_name);
        coverage
    }

    fn scalar_domain_for_arg(&self, expr: &Expr, scope: &WriterProofScope) -> Option<Arc<[Value]>> {
        match expr {
            Expr::Ident(name, _) => scope.bound_domain(name).or_else(|| {
                state_var_idx(expr, self.registry)
                    .and_then(|idx| self.scalar_domains.get(&idx).map(Arc::clone))
            }),
            Expr::StateVar(_, idx, _) => self.scalar_domains.get(&(*idx as usize)).map(Arc::clone),
            _ => None,
        }
    }
}

struct FixedScalarRangeWriterChecker<'a> {
    target_var_idx: usize,
    function_domain: Arc<[Value]>,
    scalar_type: SlotType,
    scalar_universe: Vec<FlatScalarValue>,
    registry: &'a VarRegistry,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &'a BTreeMap<String, Arc<[Value]>>,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
}

impl FixedScalarRangeWriterChecker<'_> {
    fn prove(
        &mut self,
        expr: &Expr,
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        match expr {
            Expr::And(left, right) => {
                let left = self.prove(&left.node, scope, visiting);
                let right = self.prove(&right.node, scope, visiting);
                left.or(right)
            }
            Expr::Or(left, right) => {
                let left = self.prove(&left.node, scope, visiting);
                let right = self.prove(&right.node, scope, visiting);
                left.branch(right)
            }
            Expr::If(_, then_expr, else_expr) => {
                let then_coverage = self.prove(&then_expr.node, scope, visiting);
                let else_coverage = self.prove(&else_expr.node, scope, visiting);
                then_coverage.branch(else_coverage)
            }
            Expr::Exists(vars, body) | Expr::Forall(vars, body) => {
                let Some(added) = self.push_bound_vars(vars, scope) else {
                    return if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                        WriterCoverage::Unsupported
                    } else {
                        WriterCoverage::NotCovered
                    };
                };
                let coverage = self.prove(&body.node, scope, visiting);
                for name in added {
                    scope.pop(&name);
                }
                coverage
            }
            Expr::Eq(left, right) => self
                .prove_assignment(&left.node, &right.node, scope)
                .or_else(|| self.prove_assignment(&right.node, &left.node, scope))
                .unwrap_or_else(|| {
                    if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                        WriterCoverage::Unsupported
                    } else {
                        WriterCoverage::NotCovered
                    }
                }),
            Expr::Unchanged(vars) => {
                if unchanged_mentions_var(&vars.node, self.target_var_idx, self.registry) {
                    WriterCoverage::Covered {
                        saw_set_writer: false,
                    }
                } else {
                    WriterCoverage::NotCovered
                }
            }
            Expr::Apply(op, args) => self.prove_operator_call(&op.node, args, scope, visiting),
            Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
                self.prove_operator_name(name, &[], scope, visiting)
            }
            _ => {
                if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                    WriterCoverage::Unsupported
                } else {
                    WriterCoverage::NotCovered
                }
            }
        }
    }

    fn prove_assignment(
        &self,
        left: &Expr,
        right: &Expr,
        scope: &WriterProofScope,
    ) -> Option<WriterCoverage> {
        let Expr::Prime(inner) = left else {
            return None;
        };
        if state_var_idx(&inner.node, self.registry)? != self.target_var_idx {
            return None;
        }
        self.replacement_is_fixed_scalar_range(right, scope)
            .then_some(WriterCoverage::Covered {
                saw_set_writer: false,
            })
            .or(Some(WriterCoverage::Unsupported))
    }

    fn replacement_is_fixed_scalar_range(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        if same_state_var(expr, self.target_var_idx, self.registry) {
            return true;
        }

        let Expr::Except(base, specs) = expr else {
            return false;
        };
        if !same_state_var(&base.node, self.target_var_idx, self.registry) || specs.len() != 1 {
            return false;
        }
        let spec = &specs[0];
        if spec.path.len() != 1 {
            return false;
        }
        let ExceptPathElement::Index(index) = &spec.path[0] else {
            return false;
        };
        self.scalar_expr_in_domain(&index.node, &self.function_domain, scope)
            && self
                .flat_scalar_expr_value(&spec.value.node)
                .is_some_and(|value| self.scalar_universe.contains(&value))
    }

    fn flat_scalar_expr_value(&self, expr: &Expr) -> Option<FlatScalarValue> {
        let value = const_expr_to_value_with_replacements(
            expr,
            self.constants,
            Some(self.op_replacements),
        )?;
        let flat = flat_scalar_from_value(&value)?;
        (flat.slot_type() == self.scalar_type).then_some(flat)
    }

    fn scalar_expr_in_domain(
        &self,
        expr: &Expr,
        domain: &[Value],
        scope: &WriterProofScope,
    ) -> bool {
        match expr {
            Expr::Ident(name, _) => {
                scope
                    .bound_domain(name)
                    .is_some_and(|bound_domain| bound_domain.as_ref() == domain)
                    || const_expr_to_value_with_replacements(
                        expr,
                        self.constants,
                        Some(self.op_replacements),
                    )
                    .is_some_and(|value| domain.contains(&value))
            }
            _ => const_expr_to_value_with_replacements(
                expr,
                self.constants,
                Some(self.op_replacements),
            )
            .is_some_and(|value| domain.contains(&value)),
        }
    }

    fn push_bound_vars(
        &self,
        vars: &[BoundVar],
        scope: &mut WriterProofScope,
    ) -> Option<Vec<String>> {
        let mut added = Vec::with_capacity(vars.len());
        for var in vars {
            if !matches!(&var.pattern, None | Some(BoundPattern::Var(_))) {
                return None;
            }
            let domain = var.domain.as_ref().and_then(|domain| {
                type_domain_values_with_replacements(
                    &domain.node,
                    self.constants,
                    self.proof_domains,
                    Some(self.op_replacements),
                )
            });
            let name = match &var.pattern {
                Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
                _ => var.name.node.clone(),
            };
            scope.push(name.clone(), domain);
            added.push(name);
        }
        Some(added)
    }

    fn prove_operator_call(
        &mut self,
        op: &Expr,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        let Some(name) = operator_ident_name(op) else {
            return WriterCoverage::Unsupported;
        };
        self.prove_operator_name(name, args, scope, visiting)
    }

    fn prove_operator_name(
        &mut self,
        name: &str,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        let Some((resolved_name, def)) =
            writer_safe_op_def(name, self.op_defs, Some(self.op_replacements))
        else {
            return WriterCoverage::Unsupported;
        };
        if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
            return WriterCoverage::Unsupported;
        }
        let mut added = Vec::with_capacity(def.params.len());
        for (param, arg) in def.params.iter().zip(args.iter()) {
            let domain = match &arg.node {
                Expr::Ident(name, _) => scope.bound_domain(name),
                _ => None,
            };
            let name = param.name.node.clone();
            scope.push(name.clone(), domain);
            added.push(name);
        }
        let coverage = self.prove(&def.body.node, scope, visiting);
        for name in added {
            scope.pop(&name);
        }
        visiting.remove(resolved_name);
        coverage
    }
}

struct ScalarVarWriterChecker<'a> {
    target_var_idx: usize,
    domain: Arc<[Value]>,
    scalar_domains: &'a BTreeMap<usize, Arc<[Value]>>,
    registry: &'a VarRegistry,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &'a BTreeMap<String, Arc<[Value]>>,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
}

impl ScalarVarWriterChecker<'_> {
    fn prove(
        &mut self,
        expr: &Expr,
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        match expr {
            Expr::And(left, right) => {
                let left = self.prove(&left.node, scope, visiting);
                let right = self.prove(&right.node, scope, visiting);
                left.or(right)
            }
            Expr::Or(left, right) => {
                let left = self.prove(&left.node, scope, visiting);
                let right = self.prove(&right.node, scope, visiting);
                left.branch(right)
            }
            Expr::If(_, then_expr, else_expr) => {
                let then_coverage = self.prove(&then_expr.node, scope, visiting);
                let else_coverage = self.prove(&else_expr.node, scope, visiting);
                then_coverage.branch(else_coverage)
            }
            Expr::Exists(vars, body) | Expr::Forall(vars, body) => {
                let Some(added) = self.push_bound_vars(vars, scope) else {
                    return if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                        WriterCoverage::Unsupported
                    } else {
                        WriterCoverage::NotCovered
                    };
                };
                let coverage = self.prove(&body.node, scope, visiting);
                for name in added {
                    scope.pop(&name);
                }
                coverage
            }
            Expr::Eq(left, right) => self
                .prove_assignment(&left.node, &right.node, scope)
                .or_else(|| self.prove_assignment(&right.node, &left.node, scope))
                .unwrap_or_else(|| {
                    if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                        WriterCoverage::Unsupported
                    } else {
                        WriterCoverage::NotCovered
                    }
                }),
            Expr::Unchanged(vars) => {
                if unchanged_mentions_var(&vars.node, self.target_var_idx, self.registry) {
                    WriterCoverage::Covered {
                        saw_set_writer: false,
                    }
                } else {
                    WriterCoverage::NotCovered
                }
            }
            Expr::Apply(op, args) => self.prove_operator_call(&op.node, args, scope, visiting),
            Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
                self.prove_operator_name(name, &[], scope, visiting)
            }
            _ => {
                if expr_mentions_prime_var(expr, self.target_var_idx, self.registry) {
                    WriterCoverage::Unsupported
                } else {
                    WriterCoverage::NotCovered
                }
            }
        }
    }

    fn prove_assignment(
        &self,
        left: &Expr,
        right: &Expr,
        scope: &WriterProofScope,
    ) -> Option<WriterCoverage> {
        let Expr::Prime(inner) = left else {
            return None;
        };
        if state_var_idx(&inner.node, self.registry)? != self.target_var_idx {
            return None;
        }
        (self.scalar_expr_in_domain(right, scope)).then_some(WriterCoverage::Covered {
            saw_set_writer: false,
        })
    }

    fn scalar_expr_in_domain(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        match expr {
            Expr::Ident(name, _) => {
                scope
                    .bound_domain(name)
                    .is_some_and(|bound_domain| bound_domain == self.domain)
                    || state_var_idx(expr, self.registry).is_some_and(|idx| {
                        self.scalar_domains
                            .get(&idx)
                            .is_some_and(|candidate| candidate == &self.domain)
                    })
                    || const_expr_to_value_with_replacements(
                        expr,
                        self.constants,
                        Some(self.op_replacements),
                    )
                    .is_some_and(|value| self.domain.contains(&value))
            }
            Expr::StateVar(_, idx, _) => self
                .scalar_domains
                .get(&(*idx as usize))
                .is_some_and(|candidate| candidate == &self.domain),
            _ => const_expr_to_value_with_replacements(
                expr,
                self.constants,
                Some(self.op_replacements),
            )
            .is_some_and(|value| self.domain.contains(&value)),
        }
    }

    fn push_bound_vars(
        &self,
        vars: &[BoundVar],
        scope: &mut WriterProofScope,
    ) -> Option<Vec<String>> {
        let mut added = Vec::with_capacity(vars.len());
        for var in vars {
            if !matches!(&var.pattern, None | Some(BoundPattern::Var(_))) {
                return None;
            }
            let domain = var.domain.as_ref().and_then(|domain| {
                type_domain_values_with_replacements(
                    &domain.node,
                    self.constants,
                    self.proof_domains,
                    Some(self.op_replacements),
                )
            });
            let name = match &var.pattern {
                Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
                _ => var.name.node.clone(),
            };
            scope.push(name.clone(), domain);
            added.push(name);
        }
        Some(added)
    }

    fn prove_operator_call(
        &mut self,
        op: &Expr,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        let Some(name) = operator_ident_name(op) else {
            return WriterCoverage::Unsupported;
        };
        self.prove_operator_name(name, args, scope, visiting)
    }

    fn prove_operator_name(
        &mut self,
        name: &str,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) -> WriterCoverage {
        let Some((resolved_name, def)) =
            writer_safe_op_def(name, self.op_defs, Some(self.op_replacements))
        else {
            return WriterCoverage::Unsupported;
        };
        if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
            return WriterCoverage::Unsupported;
        }
        let mut added = Vec::with_capacity(def.params.len());
        for (param, arg) in def.params.iter().zip(args.iter()) {
            let domain = match &arg.node {
                Expr::Ident(name, _) => scope.bound_domain(name).or_else(|| {
                    state_var_idx(&arg.node, self.registry)
                        .and_then(|idx| self.scalar_domains.get(&idx).map(Arc::clone))
                }),
                Expr::StateVar(_, idx, _) => {
                    self.scalar_domains.get(&(*idx as usize)).map(Arc::clone)
                }
                _ => None,
            };
            let name = param.name.node.clone();
            scope.push(name.clone(), domain);
            added.push(name);
        }
        let coverage = self.prove(&def.body.node, scope, visiting);
        for name in added {
            scope.pop(&name);
        }
        visiting.remove(resolved_name);
        coverage
    }
}

fn collect_tagged_scalar_set_range_init_candidates(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Vec<TaggedRangeInitCandidate> {
    let mut out = Vec::new();
    collect_tagged_scalar_set_range_init_candidates_inner(
        expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut BTreeSet::new(),
        &mut out,
    );
    dedup_exact(out)
}

fn collect_tagged_scalar_set_range_init_candidates_inner(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedRangeInitCandidate>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_scalar_set_range_init_candidates_inner(
                &left.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            collect_tagged_scalar_set_range_init_candidates_inner(
                &right.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::Eq(left, right) => {
            if let Some(candidate) = tagged_init_candidate_from_assignment(
                &left.node,
                &right.node,
                registry,
                constants,
                proof_domains,
                op_replacements,
            )
            .or_else(|| {
                tagged_init_candidate_from_assignment(
                    &right.node,
                    &left.node,
                    registry,
                    constants,
                    proof_domains,
                    op_replacements,
                )
            }) {
                out.push(candidate);
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_set_range_init_candidates_inner(
                &def.body.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn tagged_init_candidate_from_assignment(
    left: &Expr,
    right: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<TaggedRangeInitCandidate> {
    let var_idx = state_var_idx(left, registry)?;
    let Expr::FuncDef(vars, body) = right else {
        return None;
    };
    if vars.len() != 1 || !matches!(&vars[0].pattern, None | Some(BoundPattern::Var(_))) {
        return None;
    }
    let domain_expr = vars[0].domain.as_ref()?;
    let domain = type_domain_values_with_replacements(
        &domain_expr.node,
        constants,
        proof_domains,
        Some(op_replacements),
    )?;
    model_value_flat_domain(domain.as_ref())?;

    let mut scope = WriterProofScope::default();
    let name = match &vars[0].pattern {
        Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
        _ => vars[0].name.node.clone(),
    };
    scope.push(name.clone(), Some(Arc::clone(&domain)));
    let scalar_ok = model_value_constant_expr(&body.node, constants, Some(op_replacements))
        .is_some()
        || scope
            .bound_domain(expr_ident_name(&body.node)?)
            .is_some_and(|bound_domain| bound_domain == domain);
    scope.pop(&name);
    scalar_ok.then_some(TaggedRangeInitCandidate { var_idx, domain })
}

fn collect_fixed_scalar_range_init_candidates(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Vec<FixedScalarRangeInitCandidate> {
    let mut out = Vec::new();
    collect_fixed_scalar_range_init_candidates_inner(
        expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut BTreeSet::new(),
        &mut out,
    );
    dedup_exact(out)
}

fn collect_fixed_scalar_range_init_candidates_inner(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarRangeInitCandidate>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_fixed_scalar_range_init_candidates_inner(
                &left.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            collect_fixed_scalar_range_init_candidates_inner(
                &right.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::Eq(left, right) => {
            if let Some(candidate) = fixed_scalar_init_candidate_from_assignment(
                &left.node,
                &right.node,
                registry,
                constants,
                proof_domains,
                op_replacements,
            )
            .or_else(|| {
                fixed_scalar_init_candidate_from_assignment(
                    &right.node,
                    &left.node,
                    registry,
                    constants,
                    proof_domains,
                    op_replacements,
                )
            }) {
                out.push(candidate);
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_fixed_scalar_range_init_candidates_inner(
                &def.body.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn fixed_scalar_init_candidate_from_assignment(
    left: &Expr,
    right: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<FixedScalarRangeInitCandidate> {
    let var_idx = state_var_idx(left, registry)?;
    let Expr::FuncDef(vars, body) = right else {
        return None;
    };
    if vars.len() != 1 || !matches!(&vars[0].pattern, None | Some(BoundPattern::Var(_))) {
        return None;
    }
    let domain_expr = vars[0].domain.as_ref()?;
    let domain = type_domain_values_with_replacements(
        &domain_expr.node,
        constants,
        proof_domains,
        Some(op_replacements),
    )?;
    model_value_flat_domain(domain.as_ref())?;

    let value =
        const_expr_to_value_with_replacements(&body.node, constants, Some(op_replacements))?;
    let flat = flat_scalar_from_value(&value)?;
    Some(FixedScalarRangeInitCandidate {
        var_idx,
        domain,
        scalar_type: flat.slot_type(),
        scalar_universe: vec![flat],
    })
}

fn collect_fixed_scalar_range_next_values(
    expr: &Expr,
    candidate: &mut FixedScalarRangeInitCandidate,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) {
    let mut scope = WriterProofScope::default();
    let mut visiting = BTreeSet::new();
    collect_fixed_scalar_range_next_values_inner(
        expr,
        candidate,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut scope,
        &mut visiting,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_fixed_scalar_range_next_values_inner(
    expr: &Expr,
    candidate: &mut FixedScalarRangeInitCandidate,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut WriterProofScope,
    visiting: &mut BTreeSet<String>,
) {
    match expr {
        Expr::And(left, right) | Expr::Or(left, right) => {
            collect_fixed_scalar_range_next_values_inner(
                &left.node,
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
            collect_fixed_scalar_range_next_values_inner(
                &right.node,
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
        }
        Expr::If(_, then_expr, else_expr) => {
            collect_fixed_scalar_range_next_values_inner(
                &then_expr.node,
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
            collect_fixed_scalar_range_next_values_inner(
                &else_expr.node,
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
        }
        Expr::Exists(vars, body) | Expr::Forall(vars, body) => {
            let Some(added) = push_fixed_scalar_writer_bound_vars(
                vars,
                scope,
                constants,
                proof_domains,
                op_replacements,
            ) else {
                return;
            };
            collect_fixed_scalar_range_next_values_inner(
                &body.node,
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
            for name in added {
                scope.pop(&name);
            }
        }
        Expr::Eq(left, right) => {
            collect_fixed_scalar_range_values_from_assignment(
                &left.node,
                &right.node,
                candidate,
                registry,
                constants,
                op_replacements,
            );
            collect_fixed_scalar_range_values_from_assignment(
                &right.node,
                &left.node,
                candidate,
                registry,
                constants,
                op_replacements,
            );
        }
        Expr::Apply(op, args) => {
            let Some(name) = operator_ident_name(&op.node) else {
                return;
            };
            collect_fixed_scalar_range_next_values_from_operator(
                name,
                args,
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
        }
        Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
            collect_fixed_scalar_range_next_values_from_operator(
                name,
                &[],
                candidate,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_fixed_scalar_range_next_values_from_operator(
    name: &str,
    args: &[tla_core::span::Spanned<Expr>],
    candidate: &mut FixedScalarRangeInitCandidate,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut WriterProofScope,
    visiting: &mut BTreeSet<String>,
) {
    let Some((resolved_name, def)) = writer_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    let mut added = Vec::with_capacity(def.params.len());
    for (param, arg) in def.params.iter().zip(args.iter()) {
        let domain = match &arg.node {
            Expr::Ident(name, _) => scope.bound_domain(name),
            _ => None,
        };
        let name = param.name.node.clone();
        scope.push(name.clone(), domain);
        added.push(name);
    }
    collect_fixed_scalar_range_next_values_inner(
        &def.body.node,
        candidate,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
    );
    for name in added {
        scope.pop(&name);
    }
    visiting.remove(resolved_name);
}

fn push_fixed_scalar_writer_bound_vars(
    vars: &[BoundVar],
    scope: &mut WriterProofScope,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<Vec<String>> {
    let mut added = Vec::with_capacity(vars.len());
    for var in vars {
        if !matches!(&var.pattern, None | Some(BoundPattern::Var(_))) {
            return None;
        }
        let domain = var.domain.as_ref().and_then(|domain| {
            type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            )
        });
        let name = match &var.pattern {
            Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
            _ => var.name.node.clone(),
        };
        scope.push(name.clone(), domain);
        added.push(name);
    }
    Some(added)
}

fn collect_fixed_scalar_range_values_from_assignment(
    left: &Expr,
    right: &Expr,
    candidate: &mut FixedScalarRangeInitCandidate,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) {
    let Expr::Prime(inner) = left else {
        return;
    };
    if state_var_idx(&inner.node, registry) != Some(candidate.var_idx) {
        return;
    }
    let Expr::Except(base, specs) = right else {
        return;
    };
    if !same_state_var(&base.node, candidate.var_idx, registry) || specs.len() != 1 {
        return;
    }
    let Some(value) = fixed_scalar_flat_const_expr_value(
        &specs[0].value.node,
        candidate.scalar_type,
        constants,
        op_replacements,
    ) else {
        return;
    };
    candidate.scalar_universe.push(value);
}

fn fixed_scalar_flat_const_expr_value(
    expr: &Expr,
    scalar_type: SlotType,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<FlatScalarValue> {
    let value = const_expr_to_value_with_replacements(expr, constants, Some(op_replacements))?;
    let flat = flat_scalar_from_value(&value)?;
    (flat.slot_type() == scalar_type).then_some(flat)
}

fn preserved_model_value_scalar_domains(
    init_expr: &Expr,
    next_expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> BTreeMap<usize, Arc<[Value]>> {
    let mut candidates = BTreeMap::new();
    collect_model_value_scalar_domain_init_candidates(
        init_expr,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        &mut BTreeSet::new(),
        &mut candidates,
    );
    candidates.retain(|_, domain| !domain.is_empty());

    loop {
        let before = candidates.len();
        let keys: Vec<usize> = candidates.keys().copied().collect();
        for var_idx in keys {
            let Some(domain) = candidates.get(&var_idx).cloned() else {
                continue;
            };
            let mut checker = ScalarVarWriterChecker {
                target_var_idx: var_idx,
                domain,
                scalar_domains: &candidates,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
            };
            let mut scope = WriterProofScope::default();
            let mut visiting = BTreeSet::new();
            if !matches!(
                checker.prove(next_expr, &mut scope, &mut visiting),
                WriterCoverage::Covered { .. }
            ) {
                candidates.remove(&var_idx);
            }
        }
        if candidates.len() == before {
            break;
        }
    }

    candidates
}

fn collect_model_value_scalar_domain_init_candidates(
    expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut BTreeMap<usize, Arc<[Value]>>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_model_value_scalar_domain_init_candidates(
                &left.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            collect_model_value_scalar_domain_init_candidates(
                &right.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::In(left, right) => {
            if let Some(var_idx) = state_var_idx(&left.node, registry) {
                if let Some(domain) = type_domain_values_with_replacements(
                    &right.node,
                    constants,
                    proof_domains,
                    Some(op_replacements),
                ) {
                    if model_value_flat_domain(domain.as_ref()).is_some() {
                        match out.get(&var_idx) {
                            None => {
                                out.insert(var_idx, domain);
                            }
                            Some(existing) if existing.is_empty() => {}
                            Some(existing) if existing.as_ref() == domain.as_ref() => {}
                            Some(_) => {
                                out.insert(
                                    var_idx,
                                    Arc::from(Vec::<Value>::new().into_boxed_slice()),
                                );
                            }
                        }
                    }
                }
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_model_value_scalar_domain_init_candidates(
                &def.body.node,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn model_value_flat_domain(values: &[Value]) -> Option<Vec<FlatScalarValue>> {
    if values.is_empty() || values.len() > 63 {
        return None;
    }
    let flat: Option<Vec<_>> = values
        .iter()
        .map(|value| match value {
            Value::ModelValue(name) => Some(FlatScalarValue::ModelValue(name.clone().into())),
            _ => None,
        })
        .collect();
    normalize_flat_scalar_domain(flat?)
}

fn model_value_constant_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<Value> {
    let value = const_expr_to_value_with_replacements(expr, constants, op_replacements)?;
    matches!(value, Value::ModelValue(_)).then_some(value)
}

fn precomputed_constant_set_values(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<Vec<Value>> {
    const_expr_to_value_with_replacements(expr, constants, op_replacements)?
        .to_sorted_set()
        .map(|set| set.iter().cloned().collect())
}

fn domain_values_equal(left: Option<&[Value]>, right: &[Value]) -> bool {
    left.is_some_and(|left| left == right)
}

fn state_var_idx(expr: &Expr, registry: &VarRegistry) -> Option<usize> {
    match expr {
        Expr::StateVar(_, idx, _) => Some(*idx as usize),
        Expr::Ident(name, _) => registry.get(name).map(|idx| idx.0 as usize),
        _ => None,
    }
}

fn same_state_var(expr: &Expr, var_idx: usize, registry: &VarRegistry) -> bool {
    state_var_idx(expr, registry) == Some(var_idx)
}

fn expr_ident_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Ident(name, _) => Some(name.as_str()),
        _ => None,
    }
}

fn unchanged_mentions_var(expr: &Expr, var_idx: usize, registry: &VarRegistry) -> bool {
    match expr {
        Expr::Ident(_, _) | Expr::StateVar(_, _, _) => same_state_var(expr, var_idx, registry),
        Expr::Tuple(elems) => elems
            .iter()
            .any(|elem| unchanged_mentions_var(&elem.node, var_idx, registry)),
        _ => false,
    }
}

fn expr_mentions_state_var(expr: &Expr, var_idx: usize, registry: &VarRegistry) -> bool {
    let name = Arc::from(registry.name(crate::var_index::VarIndex::new(var_idx)));
    tla_core::expr_references_state_vars_v(expr, &[name])
}

fn expr_mentions_prime_var(expr: &Expr, var_idx: usize, registry: &VarRegistry) -> bool {
    match expr {
        Expr::Prime(inner) => expr_mentions_state_var(&inner.node, var_idx, registry),
        Expr::Apply(op, args) => {
            expr_mentions_prime_var(&op.node, var_idx, registry)
                || args
                    .iter()
                    .any(|arg| expr_mentions_prime_var(&arg.node, var_idx, registry))
        }
        Expr::And(left, right)
        | Expr::Or(left, right)
        | Expr::Implies(left, right)
        | Expr::Equiv(left, right)
        | Expr::In(left, right)
        | Expr::NotIn(left, right)
        | Expr::Subseteq(left, right)
        | Expr::Union(left, right)
        | Expr::Intersect(left, right)
        | Expr::SetMinus(left, right)
        | Expr::FuncApply(left, right)
        | Expr::FuncSet(left, right)
        | Expr::Eq(left, right)
        | Expr::Neq(left, right)
        | Expr::Lt(left, right)
        | Expr::Leq(left, right)
        | Expr::Gt(left, right)
        | Expr::Geq(left, right)
        | Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::IntDiv(left, right)
        | Expr::Mod(left, right)
        | Expr::Pow(left, right)
        | Expr::Range(left, right)
        | Expr::LeadsTo(left, right) => {
            expr_mentions_prime_var(&left.node, var_idx, registry)
                || expr_mentions_prime_var(&right.node, var_idx, registry)
        }
        Expr::Not(inner)
        | Expr::Powerset(inner)
        | Expr::BigUnion(inner)
        | Expr::Domain(inner)
        | Expr::Enabled(inner)
        | Expr::Unchanged(inner)
        | Expr::Always(inner)
        | Expr::Eventually(inner)
        | Expr::Neg(inner) => expr_mentions_prime_var(&inner.node, var_idx, registry),
        Expr::Forall(vars, body) | Expr::Exists(vars, body) => {
            vars.iter().any(|var| {
                var.domain
                    .as_ref()
                    .is_some_and(|domain| expr_mentions_prime_var(&domain.node, var_idx, registry))
            }) || expr_mentions_prime_var(&body.node, var_idx, registry)
        }
        Expr::Choose(var, body) | Expr::SetFilter(var, body) => {
            var.domain
                .as_ref()
                .is_some_and(|domain| expr_mentions_prime_var(&domain.node, var_idx, registry))
                || expr_mentions_prime_var(&body.node, var_idx, registry)
        }
        Expr::SetBuilder(body, vars) => {
            expr_mentions_prime_var(&body.node, var_idx, registry)
                || vars.iter().any(|var| {
                    var.domain.as_ref().is_some_and(|domain| {
                        expr_mentions_prime_var(&domain.node, var_idx, registry)
                    })
                })
        }
        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => elems
            .iter()
            .any(|elem| expr_mentions_prime_var(&elem.node, var_idx, registry)),
        Expr::FuncDef(vars, body) => {
            vars.iter().any(|var| {
                var.domain
                    .as_ref()
                    .is_some_and(|domain| expr_mentions_prime_var(&domain.node, var_idx, registry))
            }) || expr_mentions_prime_var(&body.node, var_idx, registry)
        }
        Expr::Except(base, specs) => {
            expr_mentions_prime_var(&base.node, var_idx, registry)
                || specs.iter().any(|spec| {
                    spec.path.iter().any(|elem| match elem {
                        ExceptPathElement::Index(index) => {
                            expr_mentions_prime_var(&index.node, var_idx, registry)
                        }
                        ExceptPathElement::Field(_) => false,
                    }) || expr_mentions_prime_var(&spec.value.node, var_idx, registry)
                })
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_, value)| expr_mentions_prime_var(&value.node, var_idx, registry)),
        Expr::RecordAccess(base, _) => expr_mentions_prime_var(&base.node, var_idx, registry),
        Expr::If(cond, then_expr, else_expr) => {
            expr_mentions_prime_var(&cond.node, var_idx, registry)
                || expr_mentions_prime_var(&then_expr.node, var_idx, registry)
                || expr_mentions_prime_var(&else_expr.node, var_idx, registry)
        }
        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_mentions_prime_var(&arm.guard.node, var_idx, registry)
                    || expr_mentions_prime_var(&arm.body.node, var_idx, registry)
            }) || other
                .as_ref()
                .is_some_and(|other| expr_mentions_prime_var(&other.node, var_idx, registry))
        }
        Expr::Let(defs, body) => {
            defs.iter()
                .any(|def| expr_mentions_prime_var(&def.body.node, var_idx, registry))
                || expr_mentions_prime_var(&body.node, var_idx, registry)
        }
        Expr::SubstIn(_, body) => expr_mentions_prime_var(&body.node, var_idx, registry),
        Expr::Lambda(_, body) => expr_mentions_prime_var(&body.node, var_idx, registry),
        Expr::Label(label) => expr_mentions_prime_var(&label.body.node, var_idx, registry),
        Expr::WeakFair(action, vars) | Expr::StrongFair(action, vars) => {
            expr_mentions_prime_var(&action.node, var_idx, registry)
                || expr_mentions_prime_var(&vars.node, var_idx, registry)
        }
        Expr::ModuleRef(_, _, _) | Expr::InstanceExpr(_, _) => false,
        Expr::Bool(_)
        | Expr::Int(_)
        | Expr::String(_)
        | Expr::Ident(_, _)
        | Expr::StateVar(_, _, _)
        | Expr::OpRef(_) => false,
    }
}

fn writer_safe_op_def<'a>(
    name: &'a str,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: Option<&'a OpReplacements>,
) -> Option<(&'a str, &'a OperatorDef)> {
    let resolved = resolve_layout_op_name(name, op_replacements)?;
    let def = op_defs.get(resolved)?.as_ref();
    (!def.has_primed_param && !def.is_recursive && def.params.iter().all(|param| param.arity == 0))
        .then_some((resolved, def))
}

/// Soundness veto (#43): scan `init`/`next` for any write to a state variable
/// (whole-var `v' = E` or per-element `v' = [v EXCEPT ![i] = E]`) whose assigned
/// value `E` is **not provably a flat scalar** (string / model-value / int / bool).
///
/// Returns the set of state-var indices whose writers can assign a SET, record,
/// sequence, function, or any other non-scalar value — i.e. the variables that
/// MUST NOT be encoded as a flat-primary scalar slot. This is the fail-closed
/// gate for the `TypeOK`-derived `FixedScalar` range / var proofs, which on their
/// own only assert a *type*-level membership and do not verify that every writer
/// keeps the variable scalar. A permissive or incidentally-true `TypeOK` (e.g.
/// `temp \in [Proc -> Proc]`) must never override the writer reality that some
/// branch stores `Proc \ {self}` (a set) into the same slot, which would alias
/// distinct states in the flat fingerprint and silently undercount the BFS.
///
/// Conservative by construction: a variable is vetoed unless EVERY syntactic
/// write to it is provably scalar. Writes the scanner cannot structurally
/// classify (opaque operator results, non-scalar-shaped RHS, EXCEPT specs with a
/// nested/record path, etc.) veto the variable. Over-rejection only costs the
/// flat-primary fast path (the variable stays on the sound interpreter successor
/// path); under-rejection is the soundness hole this closes.
fn vars_with_nonscalar_writers(
    init: &Expr,
    next: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> BTreeSet<usize> {
    scan_writer_veto(
        init,
        next,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        WriterVetoMode::NotProvablyScalar,
    )
}

fn scan_writer_veto(
    init: &Expr,
    next: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    mode: WriterVetoMode,
) -> BTreeSet<usize> {
    let mut vetoed = BTreeSet::new();
    let mut scanner = NonScalarWriterScanner {
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        vetoed: &mut vetoed,
        mode,
    };
    let mut scope = WriterProofScope::default();
    let mut visiting = BTreeSet::new();
    scanner.scan(init, &mut scope, &mut visiting);
    let mut visiting = BTreeSet::new();
    scanner.scan(next, &mut scope, &mut visiting);
    vetoed
}

/// Public entry point for the init-sampled-scalar-slot soundness veto set.
///
/// Returns the set of state-var indices whose Init/Next writers can *provably
/// produce a SET* (or another collision-prone non-scalar aggregate). The #43 fix
/// dropped *TypeOK*-derived `FixedScalar` type-proofs using the broader
/// "not provably scalar" classifier. This entry point closes the parallel hole
/// where a slot was admitted to flat-primary by *init-sampling alone* (no
/// type-proof): e.g. `x = 0` in Init makes `x` a plain `Scalar`, yet a successor
/// `x' = {1, 2}` (or `x' = M` for a set constant `M`, or `x' = S \ {e}`) aliases
/// the set into the same i64 slot and silently undercounts the BFS — a missed
/// violation.
///
/// Consumed only by [`StateLayout::veto_flat_primary_scalar_slot_vars`]. Because
/// the scalar layout here carries no type-proof, the veto must be *precise*: it
/// flags only writers that are structurally set/aggregate-producing (so a
/// genuinely-scalar var whose RHS is a record-field read, `Head`, function
/// application, cross-var copy, or arithmetic is NOT over-rejected and stays
/// flat-primary), while still failing closed on any actual set-valued writer.
pub(crate) fn nonscalar_writer_vetoed_vars(
    init: &Expr,
    next: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> BTreeSet<usize> {
    scan_writer_veto(
        init,
        next,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        WriterVetoMode::ProvablySetValued,
    )
}

/// Which side of the scalar/non-scalar boundary the writer scanner errs toward.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WriterVetoMode {
    /// #43 (TypeOK-proof path): veto a variable unless EVERY writer is *provably
    /// scalar*. Conservative toward dropping a proof; over-rejection here only
    /// costs a TypeOK-derived flat-primary upgrade, so unrecognized writers
    /// (record-field reads, `Head`, function applications) safely veto.
    NotProvablyScalar,
    /// Init-sampled-layout path: veto a variable only when a writer can
    /// *provably produce a SET* (or another collision-prone non-scalar
    /// aggregate). Here the scalar layout was admitted with no type-proof, so a
    /// broad "not provably scalar" veto would over-reject genuinely-scalar vars
    /// whose RHS the scanner cannot structurally prove scalar (e.g.
    /// `cursorLine' = entry.cursorLine`, a record-field read of an `Int` field;
    /// or `x' = Head(stack)`). Those reads are NOT set constructors, so they are
    /// not flagged — only an actual set-valued expression (`{..}`, `\cup`, `\`,
    /// `SUBSET`, `UNION`, `a..b`, a set-typed constant, or an inlined op body
    /// that produces one) aliases a scalar slot and is vetoed.
    ProvablySetValued,
}

struct NonScalarWriterScanner<'a> {
    registry: &'a VarRegistry,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &'a BTreeMap<String, Arc<[Value]>>,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
    vetoed: &'a mut BTreeSet<usize>,
    mode: WriterVetoMode,
}

impl NonScalarWriterScanner<'_> {
    fn scan(&mut self, expr: &Expr, scope: &mut WriterProofScope, visiting: &mut BTreeSet<String>) {
        match expr {
            Expr::And(left, right) | Expr::Or(left, right) => {
                self.scan(&left.node, scope, visiting);
                self.scan(&right.node, scope, visiting);
            }
            Expr::If(cond, then_expr, else_expr) => {
                self.scan(&cond.node, scope, visiting);
                self.scan(&then_expr.node, scope, visiting);
                self.scan(&else_expr.node, scope, visiting);
            }
            Expr::Case(arms, other) => {
                for arm in arms {
                    self.scan(&arm.body.node, scope, visiting);
                }
                if let Some(other) = other {
                    self.scan(&other.node, scope, visiting);
                }
            }
            Expr::Exists(vars, body) | Expr::Forall(vars, body) => {
                let added = self.push_bound_vars(vars, scope);
                self.scan(&body.node, scope, visiting);
                for name in added {
                    scope.pop(&name);
                }
            }
            Expr::Let(defs, body) => {
                self.scan(&body.node, scope, visiting);
                for def in defs {
                    self.scan(&def.body.node, scope, visiting);
                }
            }
            Expr::Label(label) => self.scan(&label.body.node, scope, visiting),
            Expr::Eq(left, right) => {
                self.scan_assignment(&left.node, &right.node, scope);
                self.scan_assignment(&right.node, &left.node, scope);
            }
            Expr::Apply(op, args) => {
                if let Some(name) = operator_ident_name(&op.node) {
                    self.scan_operator(name, args, scope, visiting);
                }
            }
            Expr::Ident(name, _) | Expr::OpRef(name) if !scope.is_bound(name) => {
                self.scan_operator(name, &[], scope, visiting);
            }
            _ => {}
        }
    }

    /// Inspect a `lhs = rhs` conjunct. If `lhs` is a primed state var, classify
    /// `rhs` and veto the variable per the active [`WriterVetoMode`].
    fn scan_assignment(&mut self, left: &Expr, right: &Expr, scope: &mut WriterProofScope) {
        let Expr::Prime(inner) = left else {
            return;
        };
        let Some(var_idx) = state_var_idx(&inner.node, self.registry) else {
            return;
        };
        if self.vetoed.contains(&var_idx) {
            return;
        }
        // A whole-function rebuild `var' = [x \in D |-> ..]` whose DOMAIN is not
        // provably preserved would overflow (or under-fill) the variable's FIXED
        // flat element-slot count — the flat encoding then silently truncates the
        // extra keys and drops reachable successor states (the `PrimedDomain.tla`
        // missed DEADLOCK, where `f' = [x \in DOMAIN f \cup {k} |-> ..]` grows the
        // domain {1} -> {1,2} -> {1,2,3}). This hazard is independent of the
        // scalar/set question both veto modes test, so flag it in every mode; the
        // demotion path (`veto_flat_primary_scalar_slot_vars`) already collapses a
        // vetoed `IntArray`/scalar layout to `Dynamic`.
        if self.funcdef_write_may_change_domain(var_idx, right) {
            self.vetoed.insert(var_idx);
            return;
        }
        let vetoes = match self.mode {
            // #43 TypeOK-proof path: veto unless provably scalar.
            WriterVetoMode::NotProvablyScalar => {
                !self.replacement_is_scalar_safe(var_idx, right, scope)
            }
            // Init-sampled-layout path: veto only when the writer can provably
            // produce a set/aggregate that would alias the scalar slot.
            WriterVetoMode::ProvablySetValued => {
                self.replacement_can_produce_aliasing_nonscalar(var_idx, right, scope)
            }
        };
        if vetoes {
            self.vetoed.insert(var_idx);
        }
    }

    /// True when the primed assignment `var' = expr` can *provably produce a SET*
    /// (or another collision-prone non-scalar aggregate) that, encoded into a
    /// single i64 scalar slot, would alias a distinct scalar state. Used only by
    /// [`WriterVetoMode::ProvablySetValued`].
    ///
    /// This is the dual of [`Self::replacement_is_scalar_safe`]: instead of
    /// "veto unless provably scalar" (which over-rejects scalar reads the scanner
    /// cannot prove, like `entry.cursorLine` or `Head(stack)`), it fires only on
    /// structurally set/aggregate-producing RHS. An identity/cross-var copy is
    /// never an aliasing producer here (a set-valued source would already have
    /// been init-sampled as a set layout, not a scalar slot).
    fn replacement_can_produce_aliasing_nonscalar(
        &self,
        var_idx: usize,
        expr: &Expr,
        scope: &mut WriterProofScope,
    ) -> bool {
        // An identity copy (`var' = var`) preserves the var's own scalar shape.
        if same_state_var(expr, var_idx, self.registry) {
            return false;
        }
        match expr {
            // `[var EXCEPT ![i] = v]` — a per-element function update. The var is
            // a function/array, not a top-level scalar slot, UNLESS its layout is
            // an `IntArray` (scalar-range function). The aliasing hazard is a SET
            // written into an element slot, so recurse into each updated value.
            Expr::Except(base, specs) if same_state_var(&base.node, var_idx, self.registry) => {
                specs
                    .iter()
                    .any(|spec| self.expr_can_produce_aliasing_nonscalar(&spec.value.node, scope))
            }
            // `[x \in D |-> body]` — whole-function rebuild; the per-key value is
            // `body`. Aliasing iff a key's value can be a set.
            Expr::FuncDef(vars, body) => {
                let added = self.push_bound_vars(vars, scope);
                let hazard = self.expr_can_produce_aliasing_nonscalar(&body.node, scope);
                for name in added {
                    scope.pop(&name);
                }
                hazard
            }
            // Any other whole-var RHS: aliasing iff the value can be a set.
            _ => self.expr_can_produce_aliasing_nonscalar(expr, scope),
        }
    }

    /// True when the primed assignment `var' = expr` is a whole-function rebuild
    /// `[x \in D |-> ..]` whose domain `D` is NOT provably equal to `var`'s
    /// current domain — i.e. the write may change (typically grow) the function's
    /// domain. A flat function/array slot reserves a fixed element-slot count, so
    /// a domain-changing rebuild silently truncates successors; fail closed.
    ///
    /// Only `FuncDef` rebuilds are a hazard here: an `EXCEPT` update keeps the
    /// domain by construction, and a cross-var copy/opaque RHS that genuinely
    /// changes the shape is already covered by the scalar/set veto modes. Returns
    /// `false` (no veto) for every non-`FuncDef` RHS so it never over-rejects.
    fn funcdef_write_may_change_domain(&self, var_idx: usize, expr: &Expr) -> bool {
        let Expr::FuncDef(vars, _) = expr else {
            return false;
        };
        // Veto unless EVERY binder domain is provably layout-stable.
        !vars
            .iter()
            .all(|bv| self.binder_domain_is_layout_stable(var_idx, bv))
    }

    /// Whether a single `x \in D` binder of a function rebuild preserves the
    /// written variable's flat-layout domain. Recognized stable forms:
    /// * `DOMAIN v` for the same variable `v` being written — domain carried over
    ///   verbatim, unchanged by construction.
    /// * a provably-constant finite set — identical in every reachable state.
    ///
    /// `DOMAIN v \cup {..}`, a domain built from another state variable, or any
    /// other non-constant set expression are rejected (fail closed).
    fn binder_domain_is_layout_stable(&self, var_idx: usize, bv: &BoundVar) -> bool {
        let Some(domain) = bv.domain.as_deref() else {
            return false;
        };
        if let Expr::Domain(inner) = &domain.node {
            if same_state_var(&inner.node, var_idx, self.registry) {
                return true;
            }
        }
        // A domain that statically resolves to a concrete finite set (a set
        // literal, range, set-builder, or constant-set ident) is the same in
        // every reachable state, hence layout-stable. `DOMAIN v \cup {..}` and
        // other state-dependent shapes resolve to `None` and are rejected.
        type_domain_values_with_replacements(
            &domain.node,
            self.constants,
            self.proof_domains,
            Some(self.op_replacements),
        )
        .is_some()
    }

    /// Whether `expr` can *provably evaluate to a SET* (or another non-scalar
    /// aggregate that does not fit a scalar slot) in some reachable state.
    /// Conservative toward `true` ONLY for structurally set/aggregate-producing
    /// shapes; returns `false` for reads whose shape it cannot determine
    /// (record-field access, function application, `Head`/`Tail`, bare state/var
    /// references) — those are not set constructors, so they never alias a slot
    /// that init-sampled as scalar.
    fn expr_can_produce_aliasing_nonscalar(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        match expr {
            // === Structural set constructors / set algebra — always sets. ===
            // `{...}` (including the empty set `{}`), `{e : x \in S}`,
            // `{x \in S : P}`, `\cup` / `\cap` / `\` set algebra, `SUBSET S`,
            // `UNION S`, and `a..b` are all set-valued regardless of operands —
            // encoding any of them into a single i64 scalar slot aliases the
            // slot, so a var whose scalar layout was init-sampled must be vetoed.
            Expr::SetEnum(_)
            | Expr::SetBuilder(_, _)
            | Expr::SetFilter(_, _)
            | Expr::Union(_, _)
            | Expr::Intersect(_, _)
            | Expr::SetMinus(_, _)
            | Expr::Powerset(_)
            | Expr::BigUnion(_)
            | Expr::Range(_, _) => true,
            // A `CHOOSE` picks an element of a set; it is scalar iff the set's
            // elements are scalar, which we cannot cheaply prove — but `CHOOSE`
            // is an *element selector*, never itself a set constructor, so it
            // does not alias a scalar slot. Treat as non-hazard (matches the
            // record-field/`Head` element-read rationale).
            Expr::Choose(_, _) => false,
            // === Control flow: a hazard iff ANY branch can produce a set. ===
            Expr::If(_, then_expr, else_expr) => {
                self.expr_can_produce_aliasing_nonscalar(&then_expr.node, scope)
                    || self.expr_can_produce_aliasing_nonscalar(&else_expr.node, scope)
            }
            Expr::Case(arms, other) => {
                arms.iter()
                    .any(|arm| self.expr_can_produce_aliasing_nonscalar(&arm.body.node, scope))
                    || other.as_ref().is_some_and(|other| {
                        self.expr_can_produce_aliasing_nonscalar(&other.node, scope)
                    })
            }
            Expr::Label(label) => self.expr_can_produce_aliasing_nonscalar(&label.body.node, scope),
            Expr::Let(defs, body) => {
                self.expr_can_produce_aliasing_nonscalar(&body.node, scope)
                    || defs
                        .iter()
                        .any(|def| self.expr_can_produce_aliasing_nonscalar(&def.body.node, scope))
            }
            // A bound quantifier variable ranges over a domain's *elements*; if
            // the domain is a set of sets the element could be a set. Flag when
            // the bound domain is known to contain a non-scalar element.
            Expr::Ident(name, _) if scope.is_bound(name) => scope
                .bound_domain(name)
                .is_some_and(|domain| domain.iter().any(|v| !is_scalar_value(v))),
            // A constant / zero-arg operator that resolves to a non-scalar value
            // (e.g. a model-value set constant `M = {a, b}` used as `x' = M`).
            Expr::Ident(_, _) | Expr::OpRef(_) => const_expr_to_value_with_replacements(
                expr,
                self.constants,
                Some(self.op_replacements),
            )
            .as_ref()
            .is_some_and(|v| !is_scalar_value(v)),
            // A zero-arg, non-primed, non-recursive operator whose body can
            // produce a set (inline it — mirrors `expr_is_provably_scalar`).
            Expr::Apply(op, args) if args.is_empty() => operator_ident_name(&op.node)
                .and_then(|name| writer_safe_op_def(name, self.op_defs, Some(self.op_replacements)))
                .is_some_and(|(_, def)| {
                    self.expr_can_produce_aliasing_nonscalar(&def.body.node, scope)
                }),
            // Everything else (record-field access, function application,
            // `Head`/`Tail`, arithmetic, opaque parameterized operators, bare
            // state-var references, literals): NOT a set constructor. A read that
            // *happens* to return a set would have made the var a set layout at
            // init-sampling time, so the scalar-slot veto does not apply.
            _ => false,
        }
    }

    /// True only when the primed assignment `var' = expr` provably keeps `var`
    /// scalar (or a function with a scalar range) in every reachable state.
    /// Recognised safe shapes:
    /// * `var' = var`               (identity copy — preserves whatever `var`
    ///   already is, which is scalar by induction)
    /// * `var' = <provably-scalar>` (whole-var scalar replacement)
    /// * `var' = [var EXCEPT ![i] = <provably-scalar>]` (single index update
    ///   whose new range element is scalar)
    /// * `var' = [x \in D |-> <provably-scalar>]` (whole-function rebuild whose
    ///   range element is scalar — does NOT change a
    ///   scalar-range function into a set-range one)
    ///
    /// Anything else (set/record/seq-valued RHS, EXCEPT with a record/nested path,
    /// a function whose range element is non-scalar, opaque operator, etc.) is
    /// treated as potentially non-scalar and vetoes the variable (fail closed).
    fn replacement_is_scalar_safe(
        &self,
        var_idx: usize,
        expr: &Expr,
        scope: &mut WriterProofScope,
    ) -> bool {
        if same_state_var(expr, var_idx, self.registry) {
            return true;
        }
        if self.expr_is_provably_scalar(expr, scope) {
            return true;
        }
        match expr {
            // `[x \in D |-> body]` — a whole-function rebuild. The new range value
            // at each key is `body`; it is range-scalar-safe iff `body` is
            // provably scalar (with the index variable(s) bound to their domains).
            Expr::FuncDef(vars, body) => {
                let added = self.push_bound_vars(vars, scope);
                let ok = self.expr_is_provably_scalar(&body.node, scope);
                for name in added {
                    scope.pop(&name);
                }
                ok
            }
            // `[var EXCEPT ![i] = v]` — single-index range update; safe iff every
            // updated range element `v` is provably scalar and the path is a plain
            // index (not a nested/record path).
            Expr::Except(base, specs) => {
                same_state_var(&base.node, var_idx, self.registry)
                    && specs.iter().all(|spec| {
                        spec.path.len() == 1
                            && matches!(spec.path[0], ExceptPathElement::Index(_))
                            && self.expr_is_provably_scalar(&spec.value.node, scope)
                    })
            }
            // `[f1 |-> v1, f2 |-> v2, ...]` — a whole-record write. The per-field
            // `FixedScalar` record-field proofs store each scalar field in its own
            // slot, so this is field-scalar-safe iff every field value is provably
            // scalar. A non-scalar field (e.g. a set) still vetoes the variable.
            //
            // A field value may additionally be a *value-identity self-copy* of
            // the SAME field being written (`var.f`); that re-stores the field's
            // own prior value and therefore introduces no new value/type into the
            // slot (see `record_field_value_is_scalar_preserving`).
            Expr::Record(fields) => fields.iter().all(|(name, value)| {
                self.record_field_value_is_scalar_preserving(
                    var_idx,
                    name.node.as_str(),
                    &value.node,
                    scope,
                )
            }),
            _ => false,
        }
    }

    /// Whether a record field's write value keeps that field a flat scalar, for
    /// the whole-record-write scalar-safety veto.
    ///
    /// Identical to [`Self::expr_is_provably_scalar`] except it additionally
    /// treats a *value-identity self-copy of the same field* — `var.f`, where
    /// `var` is the record variable being assigned and `f` is exactly the field
    /// currently being written — as scalar-preserving. Such a self-copy only
    /// re-stores the field's own prior value, so it can never introduce a NEW
    /// value or type into the slot: whatever type-safety the field's OTHER
    /// writers establish is preserved by induction (every other writer is
    /// scanned independently and would veto the variable on its own if it could
    /// store a non-scalar). The base case is the field's initial value, and the
    /// fixpoint over all writers still vetoes iff any writer introduces a
    /// non-scalar. Reading a DIFFERENT field (`var.g`, `g != f`) or another
    /// variable's field is NOT recognized, because that could change the slot's
    /// scalar type (e.g. copy an `Int` field into a `String` slot and alias the
    /// interned-`NameId` space). Control-flow wrappers are transparent: a branch
    /// is scalar-preserving iff every reachable sub-branch is.
    fn record_field_value_is_scalar_preserving(
        &self,
        var_idx: usize,
        field_name: &str,
        expr: &Expr,
        scope: &WriterProofScope,
    ) -> bool {
        match expr {
            Expr::RecordAccess(base, f)
                if same_state_var(&base.node, var_idx, self.registry)
                    && f.name.node.as_str() == field_name =>
            {
                true
            }
            Expr::If(_, then_expr, else_expr) => {
                self.record_field_value_is_scalar_preserving(
                    var_idx,
                    field_name,
                    &then_expr.node,
                    scope,
                ) && self.record_field_value_is_scalar_preserving(
                    var_idx,
                    field_name,
                    &else_expr.node,
                    scope,
                )
            }
            Expr::Case(arms, other) => {
                arms.iter().all(|arm| {
                    self.record_field_value_is_scalar_preserving(
                        var_idx,
                        field_name,
                        &arm.body.node,
                        scope,
                    )
                }) && other.as_ref().is_none_or(|other| {
                    self.record_field_value_is_scalar_preserving(
                        var_idx,
                        field_name,
                        &other.node,
                        scope,
                    )
                })
            }
            Expr::Label(label) => self.record_field_value_is_scalar_preserving(
                var_idx,
                field_name,
                &label.body.node,
                scope,
            ),
            _ => self.expr_is_provably_scalar(expr, scope),
        }
    }

    /// Whether `expr` provably evaluates to a flat scalar (string / model-value /
    /// int / bool) in every reachable state. Conservative: returns `false` for
    /// anything it cannot prove scalar.
    fn expr_is_provably_scalar(&self, expr: &Expr, scope: &WriterProofScope) -> bool {
        match expr {
            // Literal scalars.
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
            // A bound quantifier variable ranges over the elements of its domain;
            // if that domain is a flat scalar universe, the element is scalar.
            // Otherwise (unknown domain, or a domain of sets) it is not provable.
            Expr::Ident(name, _) if scope.is_bound(name) => scope
                .bound_domain(name)
                .is_some_and(|domain| domain.iter().all(is_scalar_value)),
            // A constant or constant-level operator that resolves to a scalar.
            Expr::Ident(_, _) | Expr::OpRef(_) => const_expr_to_value_with_replacements(
                expr,
                self.constants,
                Some(self.op_replacements),
            )
            .as_ref()
            .is_some_and(is_scalar_value),
            // Arithmetic operators are only defined on integers in TLA+ and
            // ALWAYS produce an integer scalar — the result type is scalar by
            // definition, independent of the operand shapes (a non-integer
            // operand is a TLA+ type error, never a set/record written into the
            // slot). So, like the boolean connectives below, these are
            // unconditionally scalar-producing. This is what lets a plain scalar
            // counter `x' = x + 1` (whose operand `x` is a bare state-var
            // reference the scanner cannot otherwise prove scalar) stay a
            // flat-primary scalar slot instead of being false-vetoed.
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::IntDiv(_, _)
            | Expr::Mod(_, _)
            | Expr::Pow(_, _)
            | Expr::Neg(_) => true,
            // Boolean connectives produce a bool.
            Expr::Not(_)
            | Expr::And(_, _)
            | Expr::Or(_, _)
            | Expr::Implies(_, _)
            | Expr::Equiv(_, _)
            | Expr::Eq(_, _)
            | Expr::Neq(_, _)
            | Expr::Lt(_, _)
            | Expr::Leq(_, _)
            | Expr::Gt(_, _)
            | Expr::Geq(_, _)
            | Expr::In(_, _)
            | Expr::NotIn(_, _)
            | Expr::Subseteq(_, _) => true,
            // IF/CASE are scalar iff every branch is scalar.
            Expr::If(_, then_expr, else_expr) => {
                self.expr_is_provably_scalar(&then_expr.node, scope)
                    && self.expr_is_provably_scalar(&else_expr.node, scope)
            }
            Expr::Case(arms, other) => {
                arms.iter()
                    .all(|arm| self.expr_is_provably_scalar(&arm.body.node, scope))
                    && other
                        .as_ref()
                        .is_none_or(|other| self.expr_is_provably_scalar(&other.node, scope))
            }
            Expr::Label(label) => self.expr_is_provably_scalar(&label.body.node, scope),
            // A zero-arg, non-primed, non-recursive operator whose body is scalar.
            Expr::Apply(op, args) if args.is_empty() => operator_ident_name(&op.node)
                .and_then(|name| writer_safe_op_def(name, self.op_defs, Some(self.op_replacements)))
                .is_some_and(|(_, def)| self.expr_is_provably_scalar(&def.body.node, scope)),
            // Anything else: const-fold as a last resort, else not provable.
            _ => const_expr_to_value_with_replacements(
                expr,
                self.constants,
                Some(self.op_replacements),
            )
            .as_ref()
            .is_some_and(is_scalar_value),
        }
    }

    fn push_bound_vars(&self, vars: &[BoundVar], scope: &mut WriterProofScope) -> Vec<String> {
        let mut added = Vec::with_capacity(vars.len());
        for var in vars {
            let domain = var.domain.as_ref().and_then(|domain| {
                type_domain_values_with_replacements(
                    &domain.node,
                    self.constants,
                    self.proof_domains,
                    Some(self.op_replacements),
                )
            });
            let name = match &var.pattern {
                Some(BoundPattern::Var(var_name)) => var_name.node.clone(),
                _ => var.name.node.clone(),
            };
            scope.push(name.clone(), domain);
            added.push(name);
        }
        added
    }

    fn scan_operator(
        &mut self,
        name: &str,
        args: &[tla_core::span::Spanned<Expr>],
        scope: &mut WriterProofScope,
        visiting: &mut BTreeSet<String>,
    ) {
        let Some((resolved_name, def)) =
            writer_safe_op_def(name, self.op_defs, Some(self.op_replacements))
        else {
            return;
        };
        if def.params.len() != args.len() || !visiting.insert(resolved_name.to_owned()) {
            return;
        }
        let mut added = Vec::with_capacity(def.params.len());
        for (param, arg) in def.params.iter().zip(args.iter()) {
            let domain = match &arg.node {
                Expr::Ident(name, _) => scope.bound_domain(name),
                _ => None,
            };
            let name = param.name.node.clone();
            scope.push(name.clone(), domain);
            added.push(name);
        }
        self.scan(&def.body.node, scope, visiting);
        for name in added {
            scope.pop(&name);
        }
        visiting.remove(resolved_name);
    }
}

/// Drop any `FixedScalar` range / var type-proof whose variable has a writer that
/// can assign a non-scalar value (#43 fail-closed gate). Writer-derived proofs
/// (already verified scalar-only) and proofs for vars with no detected
/// non-scalar writer are retained.
pub(crate) fn retain_writer_corroborated_fixed_scalar_range_proofs(
    proofs: &mut Vec<FixedScalarRangeTypeProof>,
    init: &Expr,
    next: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) {
    if proofs.is_empty() {
        return;
    }
    let vetoed = vars_with_nonscalar_writers(
        init,
        next,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
    );
    if vetoed.is_empty() {
        return;
    }
    proofs.retain(|proof| !vetoed.contains(&proof.var_idx));
}

/// Same fail-closed gate as
/// [`retain_writer_corroborated_fixed_scalar_range_proofs`], for the top-level
/// scalar `FixedScalarVarTypeProof`s.
pub(crate) fn retain_writer_corroborated_fixed_scalar_var_proofs(
    proofs: &mut Vec<FixedScalarVarTypeProof>,
    init: &Expr,
    next: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) {
    if proofs.is_empty() {
        return;
    }
    let vetoed = vars_with_nonscalar_writers(
        init,
        next,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
    );
    if vetoed.is_empty() {
        return;
    }
    proofs.retain(|proof| !vetoed.contains(&proof.var_idx));
}

fn collect_tagged_scalar_set_range_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarSetRangeTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_tagged_scalar_set_range_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_tagged_scalar_set_range_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_tagged_scalar_set_range_type_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                collect_tagged_scalar_set_range_type_proofs_from_type_expr(
                    &right.node,
                    invariant,
                    var_idx,
                    path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                    out,
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_tagged_scalar_set_range_type_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_tagged_scalar_set_range_type_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

fn collect_tagged_scalar_set_range_type_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarSetRangeTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_tagged_scalar_set_range_type_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_tagged_scalar_set_range_type_proofs_from_type_expr(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    path: Vec<SequenceCapacityPathStep>,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<TaggedScalarSetRangeTypeProof>,
) {
    match expr {
        Expr::FuncSet(domain, range) => {
            let Some(domain) = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            ) else {
                return;
            };

            if let Some((scalar_type, set_universe)) = tagged_scalar_set_range_from_type_set_expr(
                &range.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            ) {
                push_tagged_scalar_set_range_type_proof(
                    out,
                    TaggedScalarSetRangeTypeProof {
                        var_idx,
                        path: path.clone(),
                        domain: Arc::clone(&domain),
                        scalar_type,
                        set_universe,
                        invariant: Arc::from(invariant),
                    },
                );
            }

            let mut child_path = path;
            child_path.push(SequenceCapacityPathStep::HomogeneousRange { domain });
            collect_tagged_scalar_set_range_type_proofs_from_type_expr(
                &range.node,
                invariant,
                var_idx,
                child_path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_tagged_scalar_set_range_type_proofs_from_type_expr(
                &def.body.node,
                invariant,
                var_idx,
                path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn collect_fixed_scalar_range_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarRangeTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_fixed_scalar_range_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_fixed_scalar_range_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_fixed_scalar_range_type_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                collect_fixed_scalar_range_type_proofs_from_type_expr(
                    &right.node,
                    invariant,
                    var_idx,
                    path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                    out,
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_fixed_scalar_range_type_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_fixed_scalar_range_type_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

fn collect_fixed_scalar_range_type_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarRangeTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_fixed_scalar_range_type_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_fixed_scalar_range_type_proofs_from_type_expr(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    path: Vec<SequenceCapacityPathStep>,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarRangeTypeProof>,
) {
    match expr {
        Expr::FuncSet(domain, range) => {
            let Some(domain) = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            ) else {
                return;
            };

            if let Some((scalar_type, scalar_universe)) =
                finite_homogeneous_scalar_domain_from_type_expr(
                    &range.node,
                    constants,
                    proof_domains,
                    op_defs,
                    Some(op_replacements),
                    visiting,
                )
            {
                push_fixed_scalar_range_type_proof(
                    out,
                    FixedScalarRangeTypeProof {
                        var_idx,
                        path: path.clone(),
                        domain: Arc::clone(&domain),
                        scalar_type,
                        scalar_universe,
                        invariant: Arc::from(invariant),
                    },
                );
            }

            let mut child_path = path;
            child_path.push(SequenceCapacityPathStep::HomogeneousRange { domain });
            collect_fixed_scalar_range_type_proofs_from_type_expr(
                &range.node,
                invariant,
                var_idx,
                child_path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_fixed_scalar_range_type_proofs_from_type_expr(
                &def.body.node,
                invariant,
                var_idx,
                path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn collect_set_bitmask_range_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SetBitmaskRangeTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_set_bitmask_range_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_set_bitmask_range_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_set_bitmask_range_type_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                collect_set_bitmask_range_type_proofs_from_type_expr(
                    &right.node,
                    invariant,
                    var_idx,
                    path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                    out,
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_set_bitmask_range_type_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_set_bitmask_range_type_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

fn collect_set_bitmask_range_type_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SetBitmaskRangeTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_set_bitmask_range_type_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_set_bitmask_range_type_proofs_from_type_expr(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    path: Vec<SequenceCapacityPathStep>,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SetBitmaskRangeTypeProof>,
) {
    match expr {
        Expr::FuncSet(domain, range) => {
            let Some(domain) = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            ) else {
                return;
            };

            if let Some(set_universe) = set_bitmask_range_universe_from_type_set_expr(
                &range.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            ) {
                push_set_bitmask_range_type_proof(
                    out,
                    SetBitmaskRangeTypeProof {
                        var_idx,
                        path: path.clone(),
                        domain: Arc::clone(&domain),
                        set_universe,
                        invariant: Arc::from(invariant),
                    },
                );
            }

            let mut child_path = path;
            child_path.push(SequenceCapacityPathStep::HomogeneousRange { domain });
            collect_set_bitmask_range_type_proofs_from_type_expr(
                &range.node,
                invariant,
                var_idx,
                child_path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        // Record-set type `[f1: T1, f2: T2, ...]`: a record whose field types are
        // themselves type-set expressions (e.g. RingBuffer's
        // `ringbuffer \in [writers: UNION { [0..LastIndex -> SUBSET(Writers)] }, ...]`).
        // Descend into each field's type at the record-field sub-path so a
        // `[D -> SUBSET(universe)]` range nested inside a record field still
        // yields a proven-closed SetBitmask-range proof. The universe is proven
        // exactly when the field type proves it (`SUBSET(const)`); fields that do
        // not prove a universe simply contribute no proof (fail-closed).
        Expr::RecordSet(fields) => {
            for (field_name, field_type) in fields {
                let mut child_path = path.clone();
                child_path.push(SequenceCapacityPathStep::RecordField(Arc::from(
                    field_name.node.as_str(),
                )));
                collect_set_bitmask_range_type_proofs_from_type_expr(
                    &field_type.node,
                    invariant,
                    var_idx,
                    child_path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                    out,
                );
            }
        }
        // `UNION { X }` over a singleton set literal is exactly `X`. RingBuffer's
        // TypeOk wraps each field's function type in `UNION { ... }`. Unwrap the
        // singleton so the inner `[D -> SUBSET(universe)]` is reachable. A
        // non-singleton `UNION` is not a fixed type-set shape we can prove, so it
        // is left to the catch-all (no proof, fail-closed).
        Expr::BigUnion(inner) => {
            if let Expr::SetEnum(elems) = &inner.node {
                if let [single] = elems.as_slice() {
                    collect_set_bitmask_range_type_proofs_from_type_expr(
                        &single.node,
                        invariant,
                        var_idx,
                        path,
                        constants,
                        proof_domains,
                        op_defs,
                        op_replacements,
                        visiting,
                        out,
                    );
                }
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_set_bitmask_range_type_proofs_from_type_expr(
                &def.body.node,
                invariant,
                var_idx,
                path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn collect_set_bitmask_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SetBitmaskTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_set_bitmask_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_set_bitmask_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_set_bitmask_type_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                if let Some(set_universe) = set_bitmask_type_universe_from_membership_expr(
                    &right.node,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                ) {
                    push_set_bitmask_type_proof(
                        out,
                        SetBitmaskTypeProof {
                            var_idx,
                            path,
                            set_universe,
                            invariant: Arc::from(invariant),
                        },
                    );
                }
            }
        }
        Expr::Subseteq(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                if let Some(set_universe) = set_bitmask_type_universe_from_subseteq_rhs(
                    &right.node,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                ) {
                    push_set_bitmask_type_proof(
                        out,
                        SetBitmaskTypeProof {
                            var_idx,
                            path,
                            set_universe,
                            invariant: Arc::from(invariant),
                        },
                    );
                }
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_set_bitmask_type_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_set_bitmask_type_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

fn collect_set_bitmask_type_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SetBitmaskTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_set_bitmask_type_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_fixed_scalar_var_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarVarTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_fixed_scalar_var_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_fixed_scalar_var_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_fixed_scalar_var_type_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                // Only the *whole variable* (empty path) maps to a `VarLayoutKind`;
                // sub-paths (`v[k]`, `v.f`) describe nested function ranges / record
                // fields handled by the other collectors, not a scalar var layout.
                if !path.is_empty() {
                    return;
                }
                if let Some((scalar_type, scalar_universe)) =
                    finite_homogeneous_scalar_domain_from_type_expr(
                        &right.node,
                        constants,
                        proof_domains,
                        op_defs,
                        Some(op_replacements),
                        visiting,
                    )
                {
                    // Only string / model-value scalars get the `FixedScalar`
                    // treatment. Integers and booleans already have dedicated
                    // primary-flat layouts (`Scalar` / `ScalarBool`), so there is
                    // nothing to gain by compacting them, and restricting to
                    // string/model-value keeps the encoding contract simple.
                    if matches!(scalar_type, SlotType::String | SlotType::ModelValue) {
                        push_fixed_scalar_var_type_proof(
                            out,
                            FixedScalarVarTypeProof {
                                var_idx,
                                path,
                                scalar_type,
                                scalar_universe,
                                invariant: Arc::from(invariant),
                            },
                        );
                    }
                    return;
                }
                // G2-extension: a record-set type (`v \in [f1: T1, f2: T2, ...]`)
                // does not collapse to one scalar universe, but each
                // string/model-value field whose own type is a finite homogeneous
                // scalar set yields a per-field proof at the record-field sub-path
                // `v.f`. This lets a record with mixed `i64`/string-enum fields
                // (e.g. EWD998 `token \in [pos: Node, q: Int, color: Color]`) be
                // admitted as default flat-BFS storage.
                collect_fixed_scalar_var_record_field_proofs(
                    &right.node,
                    invariant,
                    var_idx,
                    &path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                    out,
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_fixed_scalar_var_type_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_fixed_scalar_var_type_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

/// G2-extension: collect per-field `FixedScalarVarTypeProof`s for a record-set
/// type expression `[f1: T1, f2: T2, ...]`. Each string/model-value field whose
/// own type resolves to a finite homogeneous scalar set produces a proof at the
/// record-field sub-path `base_path ++ [RecordField(f)]`. Idents/oprefs naming a
/// zero-arg record-set operator (e.g. `token \in Token` where
/// `Token == [pos: Node, q: Int, color: Color]`) are resolved recursively.
fn collect_fixed_scalar_var_record_field_proofs(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    base_path: &[SequenceCapacityPathStep],
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarVarTypeProof>,
) {
    match expr {
        Expr::RecordSet(fields) => {
            for (name, field_set) in fields {
                let Some((scalar_type, scalar_universe)) =
                    finite_homogeneous_scalar_domain_from_type_expr(
                        &field_set.node,
                        constants,
                        proof_domains,
                        op_defs,
                        Some(op_replacements),
                        visiting,
                    )
                else {
                    continue;
                };
                if !matches!(scalar_type, SlotType::String | SlotType::ModelValue) {
                    continue;
                }
                let mut path = base_path.to_vec();
                path.push(SequenceCapacityPathStep::RecordField(Arc::from(
                    name.node.as_str(),
                )));
                push_fixed_scalar_var_type_proof(
                    out,
                    FixedScalarVarTypeProof {
                        var_idx,
                        path,
                        scalar_type,
                        scalar_universe,
                        invariant: Arc::from(invariant),
                    },
                );
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let Some((resolved_name, def)) =
                layout_safe_op_def(name, op_defs, Some(op_replacements))
            else {
                return;
            };
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return;
            }
            collect_fixed_scalar_var_record_field_proofs(
                &def.body.node,
                invariant,
                var_idx,
                base_path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
            visiting.remove(resolved_name);
        }
        _ => {}
    }
}

fn collect_fixed_scalar_var_type_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<FixedScalarVarTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_fixed_scalar_var_type_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_sequence_fixed_domain_type_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SequenceFixedDomainTypeProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_sequence_fixed_domain_type_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_sequence_fixed_domain_type_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_sequence_fixed_domain_type_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                collect_sequence_fixed_domain_type_proofs_from_type_expr(
                    &right.node,
                    invariant,
                    var_idx,
                    path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    visiting,
                    out,
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_sequence_fixed_domain_type_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_sequence_fixed_domain_type_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

fn collect_sequence_fixed_domain_type_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SequenceFixedDomainTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !def.params.is_empty() {
        return;
    }
    if !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_sequence_fixed_domain_type_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_sequence_element_layout_proofs_inner(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: Option<&tla_core::OpEnv>,
    op_replacements: Option<&tla_core::kani_types::HashMap<String, String>>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SequenceElementLayoutProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_sequence_element_layout_proofs_inner(
                &left.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
            collect_sequence_element_layout_proofs_inner(
                &right.node,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_element_bounded_quantifier_names(vars, proof_domains, scope) {
                collect_sequence_element_layout_proofs_inner(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::In(left, right) => {
            let mut used_bindings = BTreeSet::new();
            if let Some((var_idx, path)) =
                extract_type_state_path(&left.node, registry, scope, &mut used_bindings)
            {
                collect_sequence_element_layout_proofs_from_type_expr(
                    &right.node,
                    invariant,
                    var_idx,
                    path,
                    constants,
                    proof_domains,
                    op_defs,
                    op_replacements,
                    out,
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_sequence_element_layout_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => collect_sequence_element_layout_proofs_from_zero_arg_op(
            name,
            invariant,
            registry,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            scope,
            visiting,
            out,
        ),
        _ => {}
    }
}

fn collect_sequence_element_layout_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: Option<&tla_core::OpEnv>,
    op_replacements: Option<&tla_core::kani_types::HashMap<String, String>>,
    scope: &mut ElementProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SequenceElementLayoutProof>,
) {
    let Some(op_defs) = op_defs else {
        return;
    };
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, op_replacements) else {
        return;
    };
    if !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_sequence_element_layout_proofs_inner(
        &def.body.node,
        invariant,
        registry,
        constants,
        proof_domains,
        Some(op_defs),
        op_replacements,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn collect_sequence_element_layout_proofs_from_type_expr(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    path: Vec<SequenceCapacityPathStep>,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: Option<&tla_core::OpEnv>,
    op_replacements: Option<&tla_core::kani_types::HashMap<String, String>>,
    out: &mut Vec<SequenceElementLayoutProof>,
) {
    match expr {
        Expr::FuncSet(domain, range) => {
            let Some(domain) = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                op_replacements,
            ) else {
                return;
            };
            let mut child_path = path;
            child_path.push(SequenceCapacityPathStep::HomogeneousRange { domain });
            collect_sequence_element_layout_proofs_from_type_expr(
                &range.node,
                invariant,
                var_idx,
                child_path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                out,
            );
        }
        Expr::Apply(op, args) if args.len() == 1 && is_seq_operator(&op.node, op_replacements) => {
            if let Some(element_layout) = flat_layout_from_type_set_expr_with_ops(
                &args[0].node,
                constants,
                op_defs,
                op_replacements,
            ) {
                push_sequence_element_layout_proof(
                    out,
                    SequenceElementLayoutProof {
                        var_idx,
                        path,
                        element_layout,
                        invariant: Arc::from(invariant),
                    },
                );
            }
        }
        _ => {}
    }
}

fn push_sequence_element_layout_proof(
    out: &mut Vec<SequenceElementLayoutProof>,
    proof: SequenceElementLayoutProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

fn collect_sequence_fixed_domain_type_proofs_from_type_expr(
    expr: &Expr,
    invariant: &str,
    var_idx: usize,
    path: Vec<SequenceCapacityPathStep>,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SequenceFixedDomainTypeProof>,
) {
    match expr {
        Expr::FuncSet(domain, range) => {
            let Some(domain) = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            ) else {
                return;
            };
            let element_layout = sequence_type_layout_proof_from_type_set_expr(
                &range.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            );
            if domain_is_one_based_int_interval(&domain, 1, domain.len()) {
                if let Some(element_layout) = element_layout.clone() {
                    push_sequence_fixed_domain_type_proof(
                        out,
                        SequenceFixedDomainTypeProof {
                            var_idx,
                            path: path.clone(),
                            domain: Arc::clone(&domain),
                            element_layout,
                            invariant: Arc::from(invariant),
                        },
                    );
                }
            }
            let mut child_path = path;
            child_path.push(SequenceCapacityPathStep::HomogeneousRange { domain });
            collect_sequence_fixed_domain_type_proofs_from_type_expr(
                &range.node,
                invariant,
                var_idx,
                child_path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            collect_sequence_fixed_domain_type_proofs_from_type_alias(
                name,
                invariant,
                var_idx,
                path,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
                out,
            );
        }
        _ => {}
    }
}

fn collect_sequence_fixed_domain_type_proofs_from_type_alias(
    name: &str,
    invariant: &str,
    var_idx: usize,
    path: Vec<SequenceCapacityPathStep>,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<SequenceFixedDomainTypeProof>,
) {
    let Some((resolved_name, def)) = layout_safe_op_def(name, op_defs, Some(op_replacements))
    else {
        return;
    };
    if !visiting.insert(resolved_name.to_owned()) {
        return;
    }
    collect_sequence_fixed_domain_type_proofs_from_type_expr(
        &def.body.node,
        invariant,
        var_idx,
        path,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        visiting,
        out,
    );
    visiting.remove(resolved_name);
}

fn tagged_scalar_set_range_from_type_set_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    match expr {
        Expr::Union(left, right) => tagged_scalar_set_range_from_union_arms(
            &left.node,
            &right.node,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            visiting,
        )
        .or_else(|| {
            tagged_scalar_set_range_from_union_arms(
                &right.node,
                &left.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            )
        }),
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let (resolved_name, def) = layout_safe_op_def(name, op_defs, Some(op_replacements))?;
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return None;
            }
            let result = tagged_scalar_set_range_from_type_set_expr(
                &def.body.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            );
            visiting.remove(resolved_name);
            result
        }
        _ => None,
    }
}

fn tagged_scalar_set_range_from_union_arms(
    scalar_arm: &Expr,
    set_arm: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    let (scalar_type, _) = finite_homogeneous_scalar_domain_from_type_expr(
        scalar_arm,
        constants,
        proof_domains,
        op_defs,
        Some(op_replacements),
        visiting,
    )?;
    let (set_universe_type, set_universe) = finite_homogeneous_scalar_universe_from_powerset_expr(
        set_arm,
        constants,
        proof_domains,
        op_defs,
        Some(op_replacements),
        visiting,
    )?;

    (scalar_type == set_universe_type).then_some((scalar_type, set_universe))
}

fn finite_homogeneous_scalar_universe_from_powerset_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    visiting: &mut BTreeSet<String>,
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    match expr {
        Expr::Powerset(base) => finite_homogeneous_scalar_domain_from_type_expr(
            &base.node,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            visiting,
        ),
        // `SUBSET S \ T`: subtracting whole subsets from a powerset can only
        // *remove* members, never enlarge one, so every remaining member is
        // still a subset of `S`. The per-element scalar universe is therefore
        // exactly `elements(S)`, identical to the bare `SUBSET S` case, and it
        // stays closed under any successor write the type invariant admits. The
        // subtrahend `T` is irrelevant to the universe and is ignored. We only
        // peel when the left side itself reduces to a powerset universe (the
        // recursive call fails closed otherwise), keeping the proof tight.
        Expr::SetMinus(left, _right) => finite_homogeneous_scalar_universe_from_powerset_expr(
            &left.node,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            visiting,
        ),
        Expr::Ident(name, _) | Expr::OpRef(name) => {
            let (resolved_name, def) = layout_safe_op_def(name, op_defs, op_replacements)?;
            if !def.params.is_empty() || !visiting.insert(resolved_name.to_owned()) {
                return None;
            }
            let result = finite_homogeneous_scalar_universe_from_powerset_expr(
                &def.body.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            );
            visiting.remove(resolved_name);
            result
        }
        _ => None,
    }
}

fn set_bitmask_range_universe_from_type_set_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<Vec<FlatScalarValue>> {
    let (_, set_universe) = finite_homogeneous_scalar_universe_from_powerset_expr(
        expr,
        constants,
        proof_domains,
        op_defs,
        Some(op_replacements),
        visiting,
    )?;
    Some(set_universe)
}

fn set_bitmask_type_universe_from_membership_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<Vec<FlatScalarValue>> {
    let (_, set_universe) = finite_homogeneous_scalar_universe_from_powerset_expr(
        expr,
        constants,
        proof_domains,
        op_defs,
        Some(op_replacements),
        visiting,
    )?;
    Some(set_universe)
}

fn set_bitmask_type_universe_from_subseteq_rhs(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<Vec<FlatScalarValue>> {
    let (_, set_universe) = finite_homogeneous_scalar_domain_from_type_expr(
        expr,
        constants,
        proof_domains,
        op_defs,
        Some(op_replacements),
        visiting,
    )?;
    Some(set_universe)
}

fn finite_homogeneous_scalar_domain_from_type_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    visiting: &mut BTreeSet<String>,
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    if let Some(values) =
        type_domain_values_with_replacements(expr, constants, proof_domains, op_replacements)
    {
        return finite_homogeneous_scalar_domain_from_values(values.as_ref());
    }

    let domain = scalar_domain_from_type_set_expr_scoped(
        expr,
        constants,
        op_defs,
        op_replacements,
        &LayoutScope::new(),
        visiting,
    )?;
    finite_homogeneous_scalar_domain_from_flat_values(domain)
}

fn finite_homogeneous_scalar_domain_from_values(
    values: &[Value],
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    let domain: Option<Vec<FlatScalarValue>> = values.iter().map(flat_scalar_from_value).collect();
    finite_homogeneous_scalar_domain_from_flat_values(domain?)
}

fn finite_homogeneous_scalar_domain_from_flat_values(
    values: Vec<FlatScalarValue>,
) -> Option<(SlotType, Vec<FlatScalarValue>)> {
    let scalar_type = values.first()?.slot_type();
    if values.iter().any(|value| value.slot_type() != scalar_type) {
        return None;
    }
    Some((scalar_type, normalize_flat_scalar_domain(values)?))
}

fn push_tagged_scalar_set_range_type_proof(
    out: &mut Vec<TaggedScalarSetRangeTypeProof>,
    proof: TaggedScalarSetRangeTypeProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

fn push_fixed_scalar_range_type_proof(
    out: &mut Vec<FixedScalarRangeTypeProof>,
    proof: FixedScalarRangeTypeProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

fn push_fixed_scalar_var_type_proof(
    out: &mut Vec<FixedScalarVarTypeProof>,
    proof: FixedScalarVarTypeProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

fn push_set_bitmask_range_type_proof(
    out: &mut Vec<SetBitmaskRangeTypeProof>,
    proof: SetBitmaskRangeTypeProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

fn push_set_bitmask_type_proof(out: &mut Vec<SetBitmaskTypeProof>, proof: SetBitmaskTypeProof) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

fn sequence_type_layout_proof_from_type_set_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<SequenceTypeLayoutProof> {
    match expr {
        Expr::Apply(op, args)
            if args.len() == 1 && is_seq_operator(&op.node, Some(op_replacements)) =>
        {
            let element_layout = sequence_type_layout_proof_from_type_set_expr(
                &args[0].node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            )?;
            Some(SequenceTypeLayoutProof::Sequence {
                element_layout: Box::new(element_layout),
            })
        }
        Expr::FuncSet(domain, range) => {
            let domain = type_domain_values_with_replacements(
                &domain.node,
                constants,
                proof_domains,
                Some(op_replacements),
            )?;
            let element_layout = sequence_type_layout_proof_from_type_set_expr(
                &range.node,
                constants,
                proof_domains,
                op_defs,
                op_replacements,
                visiting,
            )?;
            if domain_is_one_based_int_interval(&domain, 1, domain.len()) {
                Some(SequenceTypeLayoutProof::FixedDomainSequence {
                    max_len: domain.len(),
                    element_layout: Box::new(element_layout),
                })
            } else {
                flat_layout_from_type_set_expr_with_ops(
                    expr,
                    constants,
                    Some(op_defs),
                    Some(op_replacements),
                )
                .map(SequenceTypeLayoutProof::Flat)
            }
        }
        Expr::Ident(name, _) | Expr::OpRef(name) => sequence_type_layout_proof_from_type_alias(
            name,
            constants,
            proof_domains,
            op_defs,
            op_replacements,
            visiting,
        )
        .or_else(|| {
            flat_layout_from_type_set_expr_with_ops(
                expr,
                constants,
                Some(op_defs),
                Some(op_replacements),
            )
            .map(SequenceTypeLayoutProof::Flat)
        }),
        _ => flat_layout_from_type_set_expr_with_ops(
            expr,
            constants,
            Some(op_defs),
            Some(op_replacements),
        )
        .map(SequenceTypeLayoutProof::Flat),
    }
}

fn sequence_type_layout_proof_from_type_alias(
    name: &str,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<SequenceTypeLayoutProof> {
    let (resolved_name, def) = layout_safe_op_def(name, op_defs, Some(op_replacements))?;
    if !def.params.is_empty() {
        return None;
    }
    if !visiting.insert(resolved_name.to_owned()) {
        return None;
    }
    let result = sequence_type_layout_proof_from_type_set_expr(
        &def.body.node,
        constants,
        proof_domains,
        op_defs,
        op_replacements,
        visiting,
    );
    visiting.remove(resolved_name);
    result
}

fn sequence_type_layout_proof_apply_flat_layout(
    proof: &SequenceTypeLayoutProof,
    observed: &FlatValueLayout,
) -> Option<FlatValueLayout> {
    match proof {
        SequenceTypeLayoutProof::Flat(proven) => {
            flat_layout_proof_apply_flat_layout(proven, observed)
        }
        SequenceTypeLayoutProof::Sequence { element_layout } => {
            let FlatValueLayout::Sequence {
                bound,
                max_len,
                element_layout: observed_element,
            } = observed
            else {
                return None;
            };
            let element_layout =
                sequence_type_layout_proof_apply_flat_layout(element_layout, observed_element)?;
            Some(FlatValueLayout::Sequence {
                bound: bound.clone(),
                max_len: *max_len,
                element_layout: Box::new(element_layout),
            })
        }
        SequenceTypeLayoutProof::FixedDomainSequence {
            max_len,
            element_layout,
        } => {
            let FlatValueLayout::Sequence {
                bound,
                max_len: observed_max_len,
                element_layout: observed_element,
            } = observed
            else {
                return None;
            };
            if max_len != observed_max_len {
                return None;
            }
            let element_layout =
                sequence_type_layout_proof_apply_flat_layout(element_layout, observed_element)?;
            Some(FlatValueLayout::Sequence {
                bound: bound.clone(),
                max_len: *observed_max_len,
                element_layout: Box::new(element_layout),
            })
        }
    }
}

fn flat_function_layout(
    domain: Vec<FlatScalarValue>,
    value_layout: FlatValueLayout,
) -> FlatValueLayout {
    if let Some((lo, len)) = ordered_dense_int_domain(&domain) {
        FlatValueLayout::IntFunction {
            lo,
            len,
            value_layout: Box::new(value_layout),
        }
    } else {
        FlatValueLayout::Function {
            domain,
            value_layout: Box::new(value_layout),
        }
    }
}

fn flat_layout_proof_apply_flat_layout(
    proven: &FlatValueLayout,
    observed: &FlatValueLayout,
) -> Option<FlatValueLayout> {
    match (proven, observed) {
        (FlatValueLayout::Scalar(proven), FlatValueLayout::Scalar(observed))
            if proven == observed =>
        {
            Some(FlatValueLayout::Scalar(*proven))
        }
        (
            FlatValueLayout::TaggedScalarUnion { proof: proven },
            FlatValueLayout::TaggedScalarUnion { proof: observed },
        ) if proven == observed => Some(FlatValueLayout::TaggedScalarUnion {
            proof: proven.clone(),
        }),
        // Upgrade a homogeneous `Scalar` slot that was only *sampled* (e.g. a
        // `[Nodes -> Nodes \cup {NIL}]` range observed as all-`NIL` in `Init`, so
        // inference produced `Scalar(ModelValue)`) into the heterogeneous
        // `TaggedScalarUnion` proven by the `TypeOK` type expression. This is the
        // sole path that lets the union shape assembled at the `Expr::Union` node
        // reach a real variable's stored layout: the observed side never
        // independently infers a union (it only sees one scalar lane per sampled
        // state), so without this arm the proof can never apply and the shape
        // stays dead. Sound because (1) the union index encoding is injective
        // over its universe, so distinct in-universe values map to distinct
        // slots; (2) the proven universe is derived from the `TypeOK` clause that
        // constrains every reachable value, so a successor that stays type-valid
        // is always in-universe; and (3) an out-of-universe write can only happen
        // on a `TypeOK` violation, which the flat serializer rejects loudly
        // (never a silent miscount). Requiring the observed scalar's lane to be
        // present in the universe rejects a type-mismatched proof. GATED behind
        // `TY_TAGGED_SCALAR_UNION`: with the flag off the union proof is never
        // constructed, so this arm is also unreachable — the extra guard is
        // defense-in-depth so the default surface is provably byte-identical.
        (
            FlatValueLayout::TaggedScalarUnion { proof: proven },
            FlatValueLayout::Scalar(observed),
        ) if tagged_scalar_union_native_flat_primary_enabled()
            && proven
                .universe()
                .iter()
                .any(|value| value.slot_type() == *observed) =>
        {
            Some(FlatValueLayout::TaggedScalarUnion {
                proof: proven.clone(),
            })
        }
        (
            FlatValueLayout::SetBitmask {
                universe: proven_universe,
                universe_closure: proven_closure,
            },
            FlatValueLayout::SetBitmask {
                universe: observed_universe,
                ..
            },
        ) if observed_universe
            .iter()
            .all(|value| proven_universe.contains(value)) =>
        {
            Some(FlatValueLayout::SetBitmask {
                universe: proven_universe.clone(),
                universe_closure: proven_closure.clone(),
            })
        }
        (
            FlatValueLayout::Record {
                field_names: proven_names,
                field_layouts: proven_fields,
            },
            FlatValueLayout::Record {
                field_names: observed_names,
                field_layouts: observed_fields,
            },
        ) if proven_names == observed_names && proven_fields.len() == observed_fields.len() => {
            let mut field_layouts = Vec::with_capacity(proven_fields.len());
            for (proven_field, observed_field) in proven_fields.iter().zip(observed_fields.iter()) {
                field_layouts.push(flat_layout_proof_apply_flat_layout(
                    proven_field,
                    observed_field,
                )?);
            }
            Some(FlatValueLayout::Record {
                field_names: proven_names.clone(),
                field_layouts,
            })
        }
        (
            FlatValueLayout::IntFunction {
                lo: proven_lo,
                len: proven_len,
                value_layout: proven_value,
            },
            FlatValueLayout::IntFunction {
                lo: observed_lo,
                len: observed_len,
                value_layout: observed_value,
            },
        ) if proven_lo == observed_lo && proven_len == observed_len => {
            let value_layout = flat_layout_proof_apply_flat_layout(proven_value, observed_value)?;
            Some(FlatValueLayout::IntFunction {
                lo: *proven_lo,
                len: *proven_len,
                value_layout: Box::new(value_layout),
            })
        }
        (
            FlatValueLayout::IntFunction {
                lo: proven_lo,
                len: proven_len,
                value_layout: proven_value,
            },
            FlatValueLayout::Function {
                domain: observed_domain,
                value_layout: observed_value,
            },
        ) if ordered_dense_int_domain(observed_domain) == Some((*proven_lo, *proven_len)) => {
            let value_layout = flat_layout_proof_apply_flat_layout(proven_value, observed_value)?;
            Some(FlatValueLayout::IntFunction {
                lo: *proven_lo,
                len: *proven_len,
                value_layout: Box::new(value_layout),
            })
        }
        (
            FlatValueLayout::Function {
                domain: proven_domain,
                value_layout: proven_value,
            },
            FlatValueLayout::IntFunction {
                lo: observed_lo,
                len: observed_len,
                value_layout: observed_value,
            },
        ) if ordered_dense_int_domain(proven_domain) == Some((*observed_lo, *observed_len)) => {
            let value_layout = flat_layout_proof_apply_flat_layout(proven_value, observed_value)?;
            Some(FlatValueLayout::IntFunction {
                lo: *observed_lo,
                len: *observed_len,
                value_layout: Box::new(value_layout),
            })
        }
        (
            FlatValueLayout::IntFunction {
                lo,
                len,
                value_layout: proven_value,
            },
            FlatValueLayout::Sequence {
                bound,
                max_len,
                element_layout: observed_element,
            },
        ) if *lo == 1 && len == max_len => {
            let element_layout =
                flat_layout_proof_apply_flat_layout(proven_value, observed_element)?;
            Some(FlatValueLayout::Sequence {
                bound: bound.clone(),
                max_len: *max_len,
                element_layout: Box::new(element_layout),
            })
        }
        (
            FlatValueLayout::Function {
                domain: proven_domain,
                value_layout: proven_value,
            },
            FlatValueLayout::Function {
                domain: observed_domain,
                value_layout: observed_value,
            },
        ) if proven_domain == observed_domain => {
            let value_layout = flat_layout_proof_apply_flat_layout(proven_value, observed_value)?;
            Some(flat_function_layout(proven_domain.clone(), value_layout))
        }
        (
            FlatValueLayout::Sequence {
                bound,
                max_len: proven_max_len,
                element_layout: proven_element,
            },
            FlatValueLayout::Sequence {
                max_len: observed_max_len,
                element_layout: observed_element,
                ..
            },
        ) if observed_max_len <= proven_max_len => {
            let element_layout =
                flat_layout_proof_apply_flat_layout(proven_element, observed_element)?;
            Some(FlatValueLayout::Sequence {
                bound: bound.clone(),
                max_len: *proven_max_len,
                element_layout: Box::new(element_layout),
            })
        }
        _ => None,
    }
}

fn push_sequence_fixed_domain_type_proof(
    out: &mut Vec<SequenceFixedDomainTypeProof>,
    proof: SequenceFixedDomainTypeProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

#[derive(Default)]
struct ElementProofScope {
    bindings: BTreeMap<String, Vec<Option<Arc<[Value]>>>>,
}

impl ElementProofScope {
    fn push(&mut self, name: String, homogeneous_domain: Option<Arc<[Value]>>) {
        self.bindings
            .entry(name)
            .or_default()
            .push(homogeneous_domain);
    }

    fn pop(&mut self, name: &str) {
        if let Some(stack) = self.bindings.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.bindings.remove(name);
            }
        }
    }

    fn is_bound(&self, name: &str) -> bool {
        self.bindings
            .get(name)
            .is_some_and(|stack| !stack.is_empty())
    }

    fn homogeneous_bound_domain(&self, name: &str) -> Option<Arc<[Value]>> {
        self.bindings
            .get(name)
            .and_then(|stack| stack.last())
            .and_then(|domain| domain.as_ref().map(Arc::clone))
    }
}

fn push_element_bounded_quantifier_names(
    vars: &[BoundVar],
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    scope: &mut ElementProofScope,
) -> Option<Vec<String>> {
    let mut added = Vec::new();
    for var in vars {
        let homogeneous_domain = element_bound_var_domain(var, proof_domains, scope);
        homogeneous_domain.as_ref()?;
        match &var.pattern {
            None | Some(BoundPattern::Var(_)) => {
                let name = var.name.node.clone();
                scope.push(name.clone(), homogeneous_domain);
                added.push(name);
            }
            Some(BoundPattern::Tuple(_)) => return None,
        }
    }
    Some(added)
}

fn element_bound_var_domain(
    var: &BoundVar,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    scope: &ElementProofScope,
) -> Option<Arc<[Value]>> {
    var.domain.as_ref().and_then(|domain| {
        element_full_homogeneous_domain_values(&domain.node, proof_domains, scope)
    })
}

fn element_full_homogeneous_domain_values(
    expr: &Expr,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    scope: &ElementProofScope,
) -> Option<Arc<[Value]>> {
    match expr {
        Expr::Ident(name, _) if !scope.is_bound(name) => proof_domains.get(name).cloned(),
        _ => None,
    }
}

fn extract_type_state_path(
    expr: &Expr,
    registry: &VarRegistry,
    scope: &ElementProofScope,
    used_bindings: &mut BTreeSet<String>,
) -> Option<(usize, Vec<SequenceCapacityPathStep>)> {
    match expr {
        Expr::StateVar(_, idx, _) => Some((*idx as usize, Vec::new())),
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            registry.get(name).map(|idx| (idx.0 as usize, Vec::new()))
        }
        Expr::FuncApply(func, arg) => {
            let (binding, domain) = element_bound_subscript_arg(&arg.node, scope)?;
            if !used_bindings.insert(binding) {
                return None;
            }
            let (var_idx, mut path) =
                extract_type_state_path(&func.node, registry, scope, used_bindings)?;
            path.push(SequenceCapacityPathStep::HomogeneousRange { domain });
            Some((var_idx, path))
        }
        Expr::RecordAccess(base, field) => {
            let (var_idx, mut path) =
                extract_type_state_path(&base.node, registry, scope, used_bindings)?;
            path.push(SequenceCapacityPathStep::RecordField(Arc::from(
                field.name.node.as_str(),
            )));
            Some((var_idx, path))
        }
        _ => None,
    }
}

fn element_bound_subscript_arg(
    expr: &Expr,
    scope: &ElementProofScope,
) -> Option<(String, Arc<[Value]>)> {
    match expr {
        Expr::Ident(name, _) => Some((name.clone(), scope.homogeneous_bound_domain(name)?)),
        _ => None,
    }
}

fn type_domain_values_with_replacements(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_replacements: Option<&OpReplacements>,
) -> Option<Arc<[Value]>> {
    match expr {
        Expr::Ident(name, _) => {
            let resolved = resolve_layout_op_name(name, op_replacements)?;
            proof_domains
                .get(resolved)
                .cloned()
                .or_else(|| {
                    (!is_replaced_layout_name(name, op_replacements))
                        .then(|| proof_domains.get(name).cloned())
                        .flatten()
                })
                .or_else(|| {
                    const_expr_to_value_with_replacements(expr, constants, op_replacements)
                        .and_then(|value| {
                            value
                                .to_sorted_set()
                                .map(|set| Arc::from(set.iter().cloned().collect::<Vec<_>>()))
                        })
                })
        }
        Expr::Range(left, right) => {
            let lo = const_int_value_with_replacements(&left.node, constants, op_replacements)?;
            let hi = const_int_value_with_replacements(&right.node, constants, op_replacements)?;
            (lo <= hi).then(|| Arc::from((lo..=hi).map(Value::SmallInt).collect::<Vec<_>>()))
        }
        Expr::SetEnum(elems) => {
            let mut values = Vec::with_capacity(elems.len());
            for elem in elems {
                values.push(const_expr_to_value_with_replacements(
                    &elem.node,
                    constants,
                    op_replacements,
                )?);
            }
            values.sort();
            values.dedup();
            Some(Arc::from(values))
        }
        Expr::SetBuilder(body, bounds) => {
            let mut values = const_function_image_set_values(
                &body.node,
                bounds,
                constants,
                proof_domains,
                op_replacements,
            )?;
            values.sort();
            values.dedup();
            Some(Arc::from(values))
        }
        // Cartesian product `S \X T [\X ...]` — the tuple-keyed function domain
        // (btree `[Nodes \X Keys -> ...]`). Each factor must itself resolve to
        // a concrete constant domain; the product is enumerated as
        // `Value::Tuple`s and canonically sorted so the resulting domain agrees
        // with the flat layout's `Value::cmp` slot order. Empty factors yield
        // an empty domain, which every consumer rejects (fail closed).
        Expr::Times(factors) => {
            if factors.len() < 2 {
                return None;
            }
            let factor_domains: Vec<Arc<[Value]>> = factors
                .iter()
                .map(|factor| {
                    type_domain_values_with_replacements(
                        &factor.node,
                        constants,
                        proof_domains,
                        op_replacements,
                    )
                })
                .collect::<Option<Vec<_>>>()?;
            let mut tuples: Vec<Vec<Value>> = vec![Vec::new()];
            for domain in &factor_domains {
                // Cap the enumeration so a huge constant product cannot blow up
                // proof collection; oversized domains simply yield no proof.
                if tuples.len().checked_mul(domain.len())? > 4096 {
                    return None;
                }
                tuples = tuples
                    .into_iter()
                    .flat_map(|prefix| {
                        domain.iter().map(move |value| {
                            let mut tuple = prefix.clone();
                            tuple.push(value.clone());
                            tuple
                        })
                    })
                    .collect();
            }
            let mut values: Vec<Value> = tuples.into_iter().map(Value::tuple).collect();
            values.sort();
            values.dedup();
            Some(Arc::from(values))
        }
        _ => None,
    }
}

/// Single binding name for a comprehension/quantifier `BoundVar`, or `None` for
/// a tuple-destructuring pattern (which this constant evaluator does not model).
fn bound_single_var_name(bound: &BoundVar) -> Option<&str> {
    match &bound.pattern {
        Some(BoundPattern::Tuple(_)) => None,
        Some(BoundPattern::Var(var)) => Some(var.node.as_str()),
        None => Some(bound.name.node.as_str()),
    }
}

/// Apply a concrete (precomputed-constant) function value at `key`. Dispatches
/// across the enumerable function representations, mirroring the evaluator's
/// eager function-application (`Func`/`IntFunc` keyed lookup; `Seq`/`Tuple`
/// 1-based integer indexing). A TLA+ function whose domain is `1..n` — e.g.
/// `Id == [i \in 1..N |-> i]` — is materialized by the constant-precompute pass
/// as a `Tuple`/`Seq`, so those representations must be handled here. Lazy /
/// non-enumerable functions, out-of-domain keys, and non-function values fail
/// closed.
fn const_func_value_apply<'a>(func: &'a Value, key: &Value) -> Option<&'a Value> {
    match func {
        Value::Func(f) => f.mapping_get(key),
        Value::IntFunc(f) => f.apply(key),
        Value::Seq(s) => {
            let idx = key.as_i64()?;
            (idx >= 1 && (idx as usize) <= s.len()).then(|| &s[(idx - 1) as usize])
        }
        Value::Tuple(t) => {
            let idx = key.as_i64()?;
            (idx >= 1 && (idx as usize) <= t.len()).then(|| &t[(idx - 1) as usize])
        }
        _ => None,
    }
}

/// Evaluate a function-image set comprehension `{ F[x] : x \in D }` to its exact
/// element set, where `F` resolves to a precomputed-constant function value and
/// `D` resolves to a concrete constant domain.
///
/// Correctness is by construction: `F` and `D` are the concrete `Value`s the
/// constant-precompute pass already produced with the real evaluator, and the
/// image is built with the canonical function-application primitive — there is
/// no hand-rolled expression evaluation, so the computed universe can never be a
/// wrong (too-small) under-approximation. Every other shape — a non-constant
/// function, a tuple/multi-variable binding, a body that is not exactly `F[x]`
/// applied to the bound variable, or an unresolved domain — fails closed
/// (returns `None`), so a SetBitmask universe is only ever derived from a
/// genuine compile-time constant set.
fn const_function_image_set_values(
    body: &Expr,
    bounds: &[BoundVar],
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    proof_domains: &BTreeMap<String, Arc<[Value]>>,
    op_replacements: Option<&OpReplacements>,
) -> Option<Vec<Value>> {
    let [bound] = bounds else {
        return None;
    };
    let var_name = bound_single_var_name(bound)?;
    let domain_expr = bound.domain.as_ref()?;

    let Expr::FuncApply(func_expr, arg_expr) = body else {
        return None;
    };
    if expr_ident_name(&arg_expr.node) != Some(var_name) {
        return None;
    }
    let func_value =
        precomputed_constant_value_with_replacements(&func_expr.node, constants, op_replacements)?;
    let domain_values = type_domain_values_with_replacements(
        &domain_expr.node,
        constants,
        proof_domains,
        op_replacements,
    )?;

    let mut image = Vec::with_capacity(domain_values.len());
    for key in domain_values.iter() {
        image.push(const_func_value_apply(func_value, key)?.clone());
    }
    Some(image)
}

fn is_seq_operator(expr: &Expr, op_replacements: Option<&OpReplacements>) -> bool {
    matches!(
        expr,
        Expr::Ident(name, _) | Expr::OpRef(name)
            if matches!(resolve_layout_op_name(name, op_replacements), Some("Seq"))
    )
}

fn flat_layout_from_type_set_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
) -> Option<FlatValueLayout> {
    if let Some(value) = precomputed_constant_value(expr, constants) {
        if let Some(layout) = flat_layout_from_type_value(value) {
            return Some(layout);
        }
    }

    match expr {
        Expr::Ident(name, _) if name == "Nat" || name == "Int" => {
            Some(FlatValueLayout::Scalar(SlotType::Int))
        }
        Expr::Ident(name, _) if name == "BOOLEAN" => Some(FlatValueLayout::Scalar(SlotType::Bool)),
        Expr::Ident(name, _) if name == "STRING" => Some(FlatValueLayout::Scalar(SlotType::String)),
        Expr::Range(_, _) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Expr::SetEnum(elems) => {
            let values: Option<Vec<Value>> = elems
                .iter()
                .map(|elem| const_expr_to_value(&elem.node, constants))
                .collect();
            flat_layout_from_type_values(values?.iter())
        }
        Expr::Powerset(base) => {
            let universe = scalar_domain_from_type_set_expr(&base.node, constants)?;
            // Universe is static here, but this helper has no invariant source
            // in scope; closure is established at proof-collection time. Default
            // to `Sampled` (sound/conservative).
            Some(FlatValueLayout::SetBitmask {
                universe,
                universe_closure: SetBitmaskUniverseClosure::Sampled,
            })
        }
        Expr::RecordSet(fields) => {
            let mut field_pairs: Vec<(NameId, Arc<str>, FlatValueLayout)> =
                Vec::with_capacity(fields.len());
            for (name, field_set) in fields {
                let field_name = Arc::from(name.node.as_str());
                let layout = flat_layout_from_type_set_expr(&field_set.node, constants)?;
                field_pairs.push((intern_name(&field_name), field_name, layout));
            }
            // Canonical record field order: field-name STRING, matching
            // RecordValue's storage order (NameId order is run-dependent).
            field_pairs.sort_by(|a, b| a.1.cmp(&b.1));
            Some(FlatValueLayout::Record {
                field_names: field_pairs
                    .iter()
                    .map(|(_, name, _)| Arc::clone(name))
                    .collect(),
                field_layouts: field_pairs
                    .into_iter()
                    .map(|(_, _, layout)| layout)
                    .collect(),
            })
        }
        Expr::FuncSet(domain, range) => {
            let domain = scalar_domain_from_type_set_expr(&domain.node, constants)?;
            let value_layout = flat_layout_from_type_set_expr(&range.node, constants)?;
            contiguous_int_flat_domain(&domain)
                .map(|(lo, len)| FlatValueLayout::IntFunction {
                    lo,
                    len,
                    value_layout: Box::new(value_layout.clone()),
                })
                .or_else(|| {
                    Some(FlatValueLayout::Function {
                        domain,
                        value_layout: Box::new(value_layout),
                    })
                })
        }
        _ => None,
    }
}

type LayoutScope = BTreeMap<String, FlatValueLayout>;
type OpReplacements = tla_core::kani_types::HashMap<String, String>;

fn flat_layout_from_type_set_expr_with_ops(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: Option<&tla_core::OpEnv>,
    op_replacements: Option<&OpReplacements>,
) -> Option<FlatValueLayout> {
    let Some(op_defs) = op_defs else {
        return flat_layout_from_type_set_expr(expr, constants);
    };
    flat_layout_from_type_set_expr_scoped(
        expr,
        constants,
        op_defs,
        op_replacements,
        &LayoutScope::new(),
        &mut BTreeSet::new(),
    )
}

fn flat_layout_from_type_set_expr_scoped(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    if let Some(value) =
        precomputed_constant_value_with_replacements(expr, constants, op_replacements)
    {
        if let Some(layout) = flat_layout_from_type_value(value) {
            return Some(layout);
        }
    }

    match expr {
        Expr::Ident(name, _)
            if !is_replaced_layout_name(name, op_replacements)
                && (name == "Nat" || name == "Int") =>
        {
            Some(FlatValueLayout::Scalar(SlotType::Int))
        }
        Expr::Ident(name, _)
            if !is_replaced_layout_name(name, op_replacements) && name == "BOOLEAN" =>
        {
            Some(FlatValueLayout::Scalar(SlotType::Bool))
        }
        Expr::Ident(name, _)
            if !is_replaced_layout_name(name, op_replacements) && name == "STRING" =>
        {
            Some(FlatValueLayout::Scalar(SlotType::String))
        }
        Expr::Ident(name, _) => {
            infer_zero_arg_type_layout(name, constants, op_defs, op_replacements, scope, visiting)
        }
        Expr::Range(_, _) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Expr::Union(left, right) => {
            // First try the existing homogeneous merge path. When both arms
            // infer to the SAME (or mergeable) flat layout — `Int \cup Int`
            // (`Scalar(Int)`), `SUBSET S \cup SUBSET T` (`SetBitmask`),
            // `ModelValue \cup ModelValue` (`Scalar(ModelValue)`), etc. — this
            // succeeds and is the byte-identical historical behavior.
            let merged = (|| {
                let left_layout = flat_layout_from_type_set_expr_scoped(
                    &left.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )?;
                let right_layout = flat_layout_from_type_set_expr_scoped(
                    &right.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )?;
                merge_flat_value_layouts(&left_layout, &right_layout)
            })();
            if merged.is_some() {
                return merged;
            }

            // The merge failed — the two arms are *heterogeneous* finite scalar
            // sets (`Int \cup {NIL}`, `Vals \cup {"ok", ...}`, ...). `merge_*`
            // cannot handle this because it only sees two `SlotType`s, not the
            // concrete element values, and two differing scalar lanes collapse
            // to overlapping compact `i64` payloads. Assemble the concrete
            // deduplicated universe from BOTH arms here (where both are still
            // enumerable) and encode the var via the injective typed universe
            // index. GATED behind `TY_TAGGED_SCALAR_UNION` (default OFF): when
            // off this returns `None`, exactly matching today's behavior (the
            // union shape is never constructed).
            if tagged_scalar_union_native_flat_primary_enabled() {
                if let Some(layout) = tagged_scalar_union_layout_from_union_arms(
                    &left.node,
                    &right.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                ) {
                    return Some(layout);
                }
            }
            None
        }
        Expr::SetMinus(left, _) => flat_layout_from_type_set_expr_scoped(
            &left.node,
            constants,
            op_defs,
            op_replacements,
            scope,
            visiting,
        ),
        Expr::SetEnum(elems) => {
            let values: Option<Vec<Value>> = elems
                .iter()
                .map(|elem| {
                    const_expr_to_value_with_replacements(&elem.node, constants, op_replacements)
                })
                .collect();
            if let Some(values) = values {
                if let Some(layout) = flat_layout_from_type_values(values.iter()) {
                    return Some(layout);
                }
                // A *heterogeneous* finite scalar set literal (e.g. btree
                // `op \in {"get", "insert", "update", NIL}` — string ∪ model
                // value) does not collapse to one homogeneous `Scalar` layout,
                // so `flat_layout_from_type_values` returns `None`. Assemble the
                // injective typed union universe from the concrete element
                // values. GATED behind `TY_TAGGED_SCALAR_UNION` (default OFF):
                // when off this returns `None`, exactly matching today.
                if tagged_scalar_union_native_flat_primary_enabled() {
                    if let Some(layout) = tagged_scalar_union_layout_from_scalar_values(&values) {
                        return Some(layout);
                    }
                }
                return None;
            }

            let mut iter = elems.iter();
            let first = iter.next()?;
            let mut layout = flat_layout_from_value_expr_scoped(
                &first.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            for elem in iter {
                let next = flat_layout_from_value_expr_scoped(
                    &elem.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )?;
                layout = merge_flat_value_layouts(&layout, &next)?;
            }
            Some(layout)
        }
        Expr::SetBuilder(body, bounds) => {
            let mut child_scope = scope.clone();
            for bound in bounds {
                let domain = bound.domain.as_ref()?;
                let layout = flat_layout_from_type_set_expr_scoped(
                    &domain.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )?;
                bind_bound_layout(bound, layout, &mut child_scope)?;
            }
            flat_layout_from_value_expr_scoped(
                &body.node,
                constants,
                op_defs,
                op_replacements,
                &child_scope,
                visiting,
            )
        }
        Expr::Powerset(base) => {
            let universe = scalar_domain_from_type_set_expr_scoped(
                &base.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            Some(FlatValueLayout::SetBitmask {
                universe,
                universe_closure: SetBitmaskUniverseClosure::Sampled,
            })
        }
        Expr::RecordSet(fields) => {
            let mut field_pairs: Vec<(NameId, Arc<str>, FlatValueLayout)> =
                Vec::with_capacity(fields.len());
            for (name, field_set) in fields {
                let field_name = Arc::from(name.node.as_str());
                let layout = flat_layout_from_type_set_expr_scoped(
                    &field_set.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )?;
                field_pairs.push((intern_name(&field_name), field_name, layout));
            }
            // Canonical record field order: field-name STRING, matching
            // RecordValue's storage order (NameId order is run-dependent).
            field_pairs.sort_by(|a, b| a.1.cmp(&b.1));
            Some(FlatValueLayout::Record {
                field_names: field_pairs
                    .iter()
                    .map(|(_, name, _)| Arc::clone(name))
                    .collect(),
                field_layouts: field_pairs
                    .into_iter()
                    .map(|(_, _, layout)| layout)
                    .collect(),
            })
        }
        Expr::FuncSet(domain, range) => {
            let domain = scalar_domain_from_type_set_expr_scoped(
                &domain.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            let value_layout = flat_layout_from_type_set_expr_scoped(
                &range.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            contiguous_int_flat_domain(&domain)
                .map(|(lo, len)| FlatValueLayout::IntFunction {
                    lo,
                    len,
                    value_layout: Box::new(value_layout.clone()),
                })
                .or_else(|| {
                    Some(FlatValueLayout::Function {
                        domain,
                        value_layout: Box::new(value_layout),
                    })
                })
        }
        _ => None,
    }
}

fn flat_layout_from_value_expr_scoped(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    if let Some(value) =
        precomputed_constant_value_with_replacements(expr, constants, op_replacements)
    {
        if let Some(layout) = infer_fixed_value_layout(
            value,
            &LayoutInferenceContext::default(),
            &SequencePath::root(usize::MAX),
        ) {
            return Some(layout);
        }
    }

    match expr {
        Expr::Bool(_) => Some(FlatValueLayout::Scalar(SlotType::Bool)),
        Expr::Int(_) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Expr::String(_) => Some(FlatValueLayout::Scalar(SlotType::String)),
        Expr::Ident(name, _) => scope.get(name).cloned().or_else(|| {
            infer_zero_arg_value_layout(name, constants, op_defs, op_replacements, scope, visiting)
        }),
        Expr::Record(fields) => {
            let mut field_pairs: Vec<(NameId, Arc<str>, FlatValueLayout)> =
                Vec::with_capacity(fields.len());
            for (name, value_expr) in fields {
                let field_name = Arc::from(name.node.as_str());
                let layout = flat_layout_from_value_expr_scoped(
                    &value_expr.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )?;
                field_pairs.push((intern_name(&field_name), field_name, layout));
            }
            // Canonical record field order: field-name STRING, matching
            // RecordValue's storage order (NameId order is run-dependent).
            field_pairs.sort_by(|a, b| a.1.cmp(&b.1));
            Some(FlatValueLayout::Record {
                field_names: field_pairs
                    .iter()
                    .map(|(_, name, _)| Arc::clone(name))
                    .collect(),
                field_layouts: field_pairs
                    .into_iter()
                    .map(|(_, _, layout)| layout)
                    .collect(),
            })
        }
        Expr::Apply(op, args) => {
            let name = operator_ident_name(&op.node)?;
            let (resolved_name, def) = layout_safe_op_def(name, op_defs, op_replacements)?;
            if def.params.len() != args.len() || visiting.contains(resolved_name) {
                return None;
            }
            let mut child_scope = scope.clone();
            for (param, arg) in def.params.iter().zip(args.iter()) {
                if param.arity != 0 {
                    return None;
                }
                let layout = flat_layout_from_value_expr_scoped(
                    &arg.node,
                    constants,
                    op_defs,
                    op_replacements,
                    scope,
                    visiting,
                )
                .or_else(|| {
                    flat_layout_from_type_set_expr_scoped(
                        &arg.node,
                        constants,
                        op_defs,
                        op_replacements,
                        scope,
                        visiting,
                    )
                })?;
                child_scope.insert(param.name.node.clone(), layout);
            }
            visiting.insert(resolved_name.to_owned());
            let result = flat_layout_from_value_expr_scoped(
                &def.body.node,
                constants,
                op_defs,
                op_replacements,
                &child_scope,
                visiting,
            );
            visiting.remove(resolved_name);
            result
        }
        Expr::If(_, then_expr, else_expr) => {
            let then_layout = flat_layout_from_value_expr_scoped(
                &then_expr.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            let else_layout = flat_layout_from_value_expr_scoped(
                &else_expr.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            merge_flat_value_layouts(&then_layout, &else_layout)
        }
        Expr::SetEnum(_) | Expr::SetBuilder(_, _) | Expr::Powerset(_) => {
            flat_layout_from_type_set_expr_scoped(
                expr,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )
        }
        _ => None,
    }
}

fn infer_zero_arg_type_layout(
    name: &str,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    let (resolved_name, def) = layout_safe_op_def(name, op_defs, op_replacements)?;
    if !def.params.is_empty() || visiting.contains(resolved_name) {
        return None;
    }
    visiting.insert(resolved_name.to_owned());
    let result = flat_layout_from_type_set_expr_scoped(
        &def.body.node,
        constants,
        op_defs,
        op_replacements,
        scope,
        visiting,
    );
    visiting.remove(resolved_name);
    result
}

fn infer_zero_arg_value_layout(
    name: &str,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    let (resolved_name, def) = layout_safe_op_def(name, op_defs, op_replacements)?;
    if !def.params.is_empty() || visiting.contains(resolved_name) {
        return None;
    }
    visiting.insert(resolved_name.to_owned());
    let result = flat_layout_from_value_expr_scoped(
        &def.body.node,
        constants,
        op_defs,
        op_replacements,
        scope,
        visiting,
    )
    .or_else(|| {
        flat_layout_from_type_set_expr_scoped(
            &def.body.node,
            constants,
            op_defs,
            op_replacements,
            scope,
            visiting,
        )
    });
    visiting.remove(resolved_name);
    result
}

#[allow(clippy::only_used_in_recursion)]
fn scalar_domain_from_type_set_expr_scoped(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<Vec<FlatScalarValue>> {
    if let Some(value) =
        precomputed_constant_value_with_replacements(expr, constants, op_replacements)
    {
        if let Some(domain) = scalar_domain_from_type_value(value) {
            return Some(domain);
        }
    }

    match expr {
        Expr::Ident(name, _) => {
            let (resolved_name, def) = layout_safe_op_def(name, op_defs, op_replacements)?;
            if !def.params.is_empty() || visiting.contains(resolved_name) {
                return None;
            }
            visiting.insert(resolved_name.to_owned());
            let result = scalar_domain_from_type_set_expr_scoped(
                &def.body.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            );
            visiting.remove(resolved_name);
            result
        }
        Expr::SetEnum(elems) => {
            let mut values = Vec::with_capacity(elems.len());
            for elem in elems {
                values.push(flat_scalar_from_value(
                    &const_expr_to_value_with_replacements(&elem.node, constants, op_replacements)?,
                )?);
            }
            normalize_flat_scalar_domain(values)
        }
        Expr::Range(left, right) => {
            let lo = const_expr_to_value_with_replacements(&left.node, constants, op_replacements)?;
            let hi =
                const_expr_to_value_with_replacements(&right.node, constants, op_replacements)?;
            let (Some(FlatScalarValue::Int(lo)), Some(FlatScalarValue::Int(hi))) =
                (flat_scalar_from_value(&lo), flat_scalar_from_value(&hi))
            else {
                return None;
            };
            if hi < lo || hi - lo >= 63 {
                return None;
            }
            normalize_flat_scalar_domain((lo..=hi).map(FlatScalarValue::Int).collect())
        }
        Expr::SetMinus(left, right) => {
            let mut domain = scalar_domain_from_type_set_expr_scoped(
                &left.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            let remove = scalar_domain_from_type_set_expr_scoped(
                &right.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            domain.retain(|value| !remove.contains(value));
            normalize_flat_scalar_domain(domain)
        }
        // `S \cup T` with both arms finitely enumerable (btree
        // `Vals \cup {NIL}` — the valOf range). The concatenation is
        // normalized (sorted, deduped); consumers that require a HOMOGENEOUS
        // universe (`finite_homogeneous_scalar_domain_from_flat_values`) still
        // fail closed on heterogeneous unions, which stay the TaggedScalarUnion
        // collector's territory.
        Expr::Union(left, right) => {
            let mut domain = scalar_domain_from_type_set_expr_scoped(
                &left.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            let extend = scalar_domain_from_type_set_expr_scoped(
                &right.node,
                constants,
                op_defs,
                op_replacements,
                scope,
                visiting,
            )?;
            domain.extend(extend);
            normalize_flat_scalar_domain(domain)
        }
        _ => None,
    }
}

/// Assemble the concrete deduplicated universe for a heterogeneous finite
/// scalar-union `FlatValueLayout::TaggedScalarUnion` from the two arms of a
/// `left \cup right` type-set expression.
///
/// Both arms are enumerated to their concrete finite scalar universes via
/// [`scalar_domain_from_type_set_expr_scoped`] (which handles `1..N`,
/// `{a, b, ...}`, `S \ T`, and zero-arg type aliases resolving to those, and
/// fails closed — returns `None` — for anything not finitely enumerable at
/// inference time, e.g. `Nat`/`Int`/`SUBSET S`). The two universes are then
/// concatenated, sorted, and deduplicated into a single canonical
/// `Vec<FlatScalarValue>`, and validated by [`TaggedScalarUnionProof::new`]
/// (non-empty, no duplicates). The result covers ALL four wishlist arm-type
/// combinations, because `FlatScalarValue`'s total order keeps distinct scalar
/// lanes distinct in the universe:
///   * `Int \cup ModelValue` (e.g. `Nodes \cup {NIL}` — btree `focus`/`lastOf`),
///   * `Int \cup String`,
///   * `ModelValue \cup String` (e.g. `Vals \cup {"ok", "error"}` — btree `ret`),
///   * `Int \cup Int` with a distinct sentinel (e.g. `1..8 \cup {-1}`).
///
/// Returns `None` (fail closed to the interpreter / historical behavior) if
/// either arm is not finitely enumerable or the combined universe fails
/// validation.
fn tagged_scalar_union_layout_from_union_arms(
    left: &Expr,
    right: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: Option<&OpReplacements>,
    scope: &LayoutScope,
    visiting: &mut BTreeSet<String>,
) -> Option<FlatValueLayout> {
    let left_universe = scalar_domain_from_type_set_expr_scoped(
        left,
        constants,
        op_defs,
        op_replacements,
        scope,
        visiting,
    )?;
    let right_universe = scalar_domain_from_type_set_expr_scoped(
        right,
        constants,
        op_defs,
        op_replacements,
        scope,
        visiting,
    )?;
    let universe = assemble_tagged_scalar_union_universe(left_universe, right_universe)?;
    tagged_scalar_union_layout_from_universe(universe)
}

/// Build a `TaggedScalarUnion` layout from a heterogeneous finite scalar SET
/// LITERAL's concrete element values (e.g. `{"get", "insert", NIL}`). Every
/// element must be a scalar (`flat_scalar_from_value` succeeds); the universe is
/// sorted + deduplicated into the injective typed index space. Fails closed if
/// any element is not a scalar or the universe fails validation.
fn tagged_scalar_union_layout_from_scalar_values(values: &[Value]) -> Option<FlatValueLayout> {
    let universe: Option<Vec<FlatScalarValue>> =
        values.iter().map(flat_scalar_from_value).collect();
    let mut universe = universe?;
    universe.sort();
    universe.dedup();
    (!universe.is_empty())
        .then_some(())
        .and_then(|()| tagged_scalar_union_layout_from_universe(universe))
}

/// Wrap a canonical (sorted, deduplicated, non-empty) scalar universe in a
/// validated `FlatValueLayout::TaggedScalarUnion`. Single source string so
/// equal universes compare equal regardless of whether they came from a
/// `\cup` node or a heterogeneous set literal.
fn tagged_scalar_union_layout_from_universe(
    universe: Vec<FlatScalarValue>,
) -> Option<FlatValueLayout> {
    let proof =
        TaggedScalarUnionProof::new(universe, Arc::from("tagged-scalar-union:type-set-expr"))
            .ok()?;
    Some(FlatValueLayout::TaggedScalarUnion { proof })
}

/// Merge two enumerated finite scalar universes into a single canonical
/// (sorted, deduplicated) universe for a `TaggedScalarUnion` slot.
///
/// Kept as a separate, dependency-free helper so the universe assembly is
/// unit-testable across all arm-type combinations without threading an
/// `OpEnv`/scope. Returns `None` when the combined universe is empty (a union of
/// two empty arms cannot happen for a non-empty type, but fail closed anyway).
fn assemble_tagged_scalar_union_universe(
    mut left_universe: Vec<FlatScalarValue>,
    right_universe: Vec<FlatScalarValue>,
) -> Option<Vec<FlatScalarValue>> {
    left_universe.extend(right_universe);
    left_universe.sort();
    left_universe.dedup();
    (!left_universe.is_empty()).then_some(left_universe)
}

fn bind_bound_layout(
    bound: &tla_core::ast::BoundVar,
    layout: FlatValueLayout,
    scope: &mut LayoutScope,
) -> Option<()> {
    match &bound.pattern {
        Some(BoundPattern::Tuple(_)) => None,
        Some(BoundPattern::Var(var)) => {
            scope.insert(var.node.clone(), layout);
            Some(())
        }
        None => {
            scope.insert(bound.name.node.clone(), layout);
            Some(())
        }
    }
}

fn operator_ident_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Ident(name, _) | Expr::OpRef(name) => Some(name.as_str()),
        _ => None,
    }
}

fn resolve_layout_op_name<'a>(
    name: &'a str,
    op_replacements: Option<&'a OpReplacements>,
) -> Option<&'a str> {
    let Some(op_replacements) = op_replacements else {
        return Some(name);
    };
    let mut current = name;
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(current) {
            return None;
        }
        let Some(next) = op_replacements.get(current) else {
            return Some(current);
        };
        current = next.as_str();
    }
}

fn is_replaced_layout_name(name: &str, op_replacements: Option<&OpReplacements>) -> bool {
    op_replacements.is_some_and(|op_replacements| op_replacements.contains_key(name))
}

fn layout_safe_op_def<'a>(
    name: &'a str,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: Option<&'a OpReplacements>,
) -> Option<(&'a str, &'a OperatorDef)> {
    let resolved = resolve_layout_op_name(name, op_replacements)?;
    let def = op_defs.get(resolved)?.as_ref();
    (!def.contains_prime
        && !def.has_primed_param
        && !def.is_recursive
        && def.params.iter().all(|param| param.arity == 0))
    .then_some((resolved, def))
}

fn precomputed_constant_value<'a>(
    expr: &Expr,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
) -> Option<&'a Value> {
    let Expr::Ident(name, name_id) = expr else {
        return None;
    };
    let id = if *name_id == NameId::INVALID {
        intern_name(name)
    } else {
        *name_id
    };
    constants.get(&id)
}

fn precomputed_constant_value_with_replacements<'a>(
    expr: &Expr,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<&'a Value> {
    let Expr::Ident(name, name_id) = expr else {
        return None;
    };
    let resolved = resolve_layout_op_name(name, op_replacements)?;
    if resolved != name {
        if let Some(value) = constants.get(&intern_name(resolved)) {
            return Some(value);
        }
        return None;
    }
    let id = if *name_id == NameId::INVALID {
        intern_name(name)
    } else {
        *name_id
    };
    constants.get(&id)
}

fn const_expr_to_value(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
) -> Option<Value> {
    match expr {
        Expr::Bool(value) => Some(Value::Bool(*value)),
        Expr::Int(value) => {
            use num_traits::ToPrimitive;

            value
                .to_i64()
                .map(Value::SmallInt)
                .or_else(|| Some(Value::Int(Rp::new(value.clone()))))
        }
        Expr::String(value) => Some(Value::String(Rp::from(value.as_str()))),
        Expr::Ident(_, _) => precomputed_constant_value(expr, constants).cloned(),
        Expr::Record(fields) => {
            let mut entries = Vec::with_capacity(fields.len());
            for (name, value_expr) in fields {
                entries.push((
                    Arc::from(name.node.as_str()),
                    const_expr_to_value(&value_expr.node, constants)?,
                ));
            }
            // Source-declaration order is arbitrary; collect() sorts into the
            // canonical record field order (field-name string).
            Some(Value::Record(
                entries
                    .into_iter()
                    .collect::<tla_value::value::RecordValue>(),
            ))
        }
        _ => None,
    }
}

fn const_expr_to_value_with_replacements(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<Value> {
    match expr {
        Expr::Ident(name, name_id) => {
            let resolved = resolve_layout_op_name(name, op_replacements)?;
            let resolved_id = intern_name(resolved);
            if resolved != name {
                return constants.get(&resolved_id).cloned();
            }
            let id = if *name_id == NameId::INVALID {
                intern_name(name)
            } else {
                *name_id
            };
            constants.get(&id).cloned()
        }
        Expr::Record(fields) => {
            let mut entries = Vec::with_capacity(fields.len());
            for (name, value_expr) in fields {
                entries.push((
                    Arc::from(name.node.as_str()),
                    const_expr_to_value_with_replacements(
                        &value_expr.node,
                        constants,
                        op_replacements,
                    )?,
                ));
            }
            // Source-declaration order is arbitrary; collect() sorts into the
            // canonical record field order (field-name string).
            Some(Value::Record(
                entries
                    .into_iter()
                    .collect::<tla_value::value::RecordValue>(),
            ))
        }
        _ => const_expr_to_value(expr, constants),
    }
}

/// Fold a constant integer expression, resolving simple arithmetic over model
/// constants (e.g. `N - 1`, `2 * N`, `-1`). Used to evaluate range-domain
/// bounds like `0 .. N-1` after operator inlining leaves arithmetic in the
/// type expression. Exact integer ops over already-resolved constants, so the
/// fold is sound; overflowing/non-integer forms return `None`.
fn const_int_value_with_replacements(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_replacements: Option<&OpReplacements>,
) -> Option<i64> {
    match expr {
        Expr::Add(left, right) => {
            let a = const_int_value_with_replacements(&left.node, constants, op_replacements)?;
            let b = const_int_value_with_replacements(&right.node, constants, op_replacements)?;
            a.checked_add(b)
        }
        Expr::Sub(left, right) => {
            let a = const_int_value_with_replacements(&left.node, constants, op_replacements)?;
            let b = const_int_value_with_replacements(&right.node, constants, op_replacements)?;
            a.checked_sub(b)
        }
        Expr::Mul(left, right) => {
            let a = const_int_value_with_replacements(&left.node, constants, op_replacements)?;
            let b = const_int_value_with_replacements(&right.node, constants, op_replacements)?;
            a.checked_mul(b)
        }
        Expr::Neg(inner) => {
            const_int_value_with_replacements(&inner.node, constants, op_replacements)?
                .checked_neg()
        }
        _ => match flat_scalar_from_value(&const_expr_to_value_with_replacements(
            expr,
            constants,
            op_replacements,
        )?) {
            Some(FlatScalarValue::Int(value)) => Some(value),
            _ => None,
        },
    }
}

fn flat_layout_from_type_value(value: &Value) -> Option<FlatValueLayout> {
    match value {
        Value::StringSet => Some(FlatValueLayout::Scalar(SlotType::String)),
        Value::Interval(_) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Value::Set(set) => flat_layout_from_type_values(set.iter()),
        Value::Subset(subset) => {
            let universe = scalar_domain_from_type_value(subset.base())?;
            Some(FlatValueLayout::SetBitmask {
                universe,
                universe_closure: SetBitmaskUniverseClosure::Sampled,
            })
        }
        Value::FuncSet(func_set) => {
            let domain = scalar_domain_from_type_value(func_set.domain())?;
            let value_layout = flat_layout_from_type_value(func_set.codomain())?;
            contiguous_int_flat_domain(&domain)
                .map(|(lo, len)| FlatValueLayout::IntFunction {
                    lo,
                    len,
                    value_layout: Box::new(value_layout.clone()),
                })
                .or_else(|| {
                    Some(FlatValueLayout::Function {
                        domain,
                        value_layout: Box::new(value_layout),
                    })
                })
        }
        _ => value
            .to_sorted_set()
            .and_then(|set| flat_layout_from_type_values(set.iter())),
    }
}

fn flat_layout_from_type_values<'a, I>(values: I) -> Option<FlatValueLayout>
where
    I: IntoIterator<Item = &'a Value>,
{
    infer_common_flat_layout(
        values,
        &LayoutInferenceContext::default(),
        &SequencePath::root(usize::MAX),
    )
}

fn scalar_domain_from_type_set_expr(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
) -> Option<Vec<FlatScalarValue>> {
    if let Some(value) = precomputed_constant_value(expr, constants) {
        return scalar_domain_from_type_value(value);
    }

    match expr {
        Expr::SetEnum(elems) => {
            let mut values = Vec::with_capacity(elems.len());
            for elem in elems {
                values.push(flat_scalar_from_value(&const_expr_to_value(
                    &elem.node, constants,
                )?)?);
            }
            normalize_flat_scalar_domain(values)
        }
        Expr::Range(left, right) => {
            let lo = const_expr_to_value(&left.node, constants)?;
            let hi = const_expr_to_value(&right.node, constants)?;
            let (Some(FlatScalarValue::Int(lo)), Some(FlatScalarValue::Int(hi))) =
                (flat_scalar_from_value(&lo), flat_scalar_from_value(&hi))
            else {
                return None;
            };
            if hi < lo || hi - lo >= 63 {
                return None;
            }
            normalize_flat_scalar_domain((lo..=hi).map(FlatScalarValue::Int).collect())
        }
        _ => None,
    }
}

fn scalar_domain_from_type_value(value: &Value) -> Option<Vec<FlatScalarValue>> {
    use num_traits::ToPrimitive;

    if value.set_len()?.to_usize()? > 63 {
        return None;
    }
    let set = value.to_sorted_set()?;
    let values: Option<Vec<FlatScalarValue>> = set.iter().map(flat_scalar_from_value).collect();
    normalize_flat_scalar_domain(values?)
}

fn normalize_flat_scalar_domain(mut values: Vec<FlatScalarValue>) -> Option<Vec<FlatScalarValue>> {
    values.sort();
    values.dedup();
    (!values.is_empty() && values.len() <= 63).then_some(values)
}

fn contiguous_int_flat_domain(domain: &[FlatScalarValue]) -> Option<(i64, usize)> {
    ordered_dense_int_domain(domain)
}

fn collect_scalar_domain_candidates(value: &Value, out: &mut Vec<Vec<FlatScalarValue>>) {
    match value {
        Value::IntFunc(func) => {
            if !func.is_empty() && func.len() <= 63 {
                let domain: Vec<FlatScalarValue> = (0..func.len())
                    .map(|i| FlatScalarValue::Int(func.as_ref().min() + i as i64))
                    .collect();
                push_unique_domain_candidate(out, domain);
            }
            for value in func.values() {
                collect_scalar_domain_candidates(value, out);
            }
        }
        Value::Func(func) => {
            if !func.domain_is_empty() && func.domain_len() <= 63 {
                let mut domain = Vec::with_capacity(func.domain_len());
                let mut all_scalar = true;
                for key in func.domain_iter() {
                    if let Some(key) = flat_scalar_from_value(key) {
                        domain.push(key);
                    } else {
                        all_scalar = false;
                        break;
                    }
                }
                if all_scalar {
                    push_unique_domain_candidate(out, domain);
                }
            }
            for (_, value) in func.iter() {
                collect_scalar_domain_candidates(value, out);
            }
        }
        Value::Record(record) => {
            for (_, value) in record.iter() {
                collect_scalar_domain_candidates(value, out);
            }
        }
        Value::Set(set) => {
            for value in set.iter() {
                collect_scalar_domain_candidates(value, out);
            }
        }
        Value::Seq(seq) => {
            if !seq.is_empty() && seq.len() <= 63 {
                let domain: Vec<FlatScalarValue> = (1..=seq.len())
                    .map(|i| FlatScalarValue::Int(i as i64))
                    .collect();
                push_unique_domain_candidate(out, domain);
            }
            for value in seq.iter() {
                collect_scalar_domain_candidates(value, out);
            }
        }
        Value::Tuple(elems) => {
            if !elems.is_empty() && elems.len() <= 63 {
                let domain: Vec<FlatScalarValue> = (1..=elems.len())
                    .map(|i| FlatScalarValue::Int(i as i64))
                    .collect();
                push_unique_domain_candidate(out, domain);
            }
            for value in elems.iter() {
                collect_scalar_domain_candidates(value, out);
            }
        }
        _ => {}
    }
}

fn push_unique_domain_candidate(
    out: &mut Vec<Vec<FlatScalarValue>>,
    mut domain: Vec<FlatScalarValue>,
) {
    domain.sort();
    domain.dedup();
    if !domain.is_empty() && !out.contains(&domain) {
        out.push(domain);
    }
}

fn collect_sequence_hints(
    value: &Value,
    context: &LayoutInferenceContext,
    out: &mut Vec<SequenceHint>,
    path: &SequencePath,
) {
    match value {
        Value::Seq(seq) => {
            if !seq.is_empty() {
                let element_path = path.child(SequencePathStep::SequenceElement);
                if let Some(element_layout) =
                    infer_common_flat_layout(seq.iter(), context, &element_path)
                {
                    push_sequence_hint(
                        out,
                        SequenceHint {
                            path: path.clone(),
                            max_len: seq.len(),
                            element_layout,
                        },
                    );
                }
            }
            let element_path = path.child(SequencePathStep::SequenceElement);
            for value in seq.iter() {
                collect_sequence_hints(value, context, out, &element_path);
            }
        }
        Value::Tuple(elems) => {
            if !elems.is_empty() {
                let element_path = path.child(SequencePathStep::SequenceElement);
                if let Some(element_layout) =
                    infer_common_flat_layout(elems.iter(), context, &element_path)
                {
                    push_sequence_hint(
                        out,
                        SequenceHint {
                            path: path.clone(),
                            max_len: elems.len(),
                            element_layout,
                        },
                    );
                }
            }
            let element_path = path.child(SequencePathStep::SequenceElement);
            for value in elems.iter() {
                collect_sequence_hints(value, context, out, &element_path);
            }
        }
        Value::IntFunc(func) => {
            if func.is_empty() {
                return;
            }
            let value_path = path.child(int_func_domain_path_step(func));
            for value in func.values() {
                collect_sequence_hints(value, context, out, &value_path);
            }
        }
        Value::Func(func) => {
            if let Some(value_path) = func_domain_path_step(func).map(|step| path.child(step)) {
                for (_, value) in func.iter() {
                    collect_sequence_hints(value, context, out, &value_path);
                }
            }
        }
        Value::Record(record) => {
            for (name_id, value) in record.iter() {
                let field_path = path.child(SequencePathStep::RecordField(
                    tla_core::resolve_name_id(name_id),
                ));
                collect_sequence_hints(value, context, out, &field_path);
            }
        }
        Value::Set(set) => {
            let element_path = path.child(SequencePathStep::SetElement);
            for value in set.iter() {
                collect_sequence_hints(value, context, out, &element_path);
            }
        }
        _ => {}
    }
}

fn push_sequence_hint(out: &mut Vec<SequenceHint>, hint: SequenceHint) {
    for existing in out.iter_mut() {
        if existing.path != hint.path {
            continue;
        }
        if let Some(element_layout) =
            merge_flat_value_layouts(&existing.element_layout, &hint.element_layout)
        {
            existing.max_len = existing.max_len.max(hint.max_len);
            existing.element_layout = element_layout;
            return;
        }
    }
    out.push(hint);
}

/// Infer layout kind from a full Value using finite-domain context gathered
/// across the sampled states.
fn infer_kind_from_value_with_context(
    value: &Value,
    context: &LayoutInferenceContext,
    path: &SequencePath,
) -> VarLayoutKind {
    match value {
        Value::Bool(_) => VarLayoutKind::ScalarBool,
        Value::SmallInt(_) | Value::Int(_) => VarLayoutKind::Scalar,
        Value::String(s) => {
            // Upgrade to a primary-flat `FixedScalar` slot when a unique proof
            // certifies that this state variable's value-set is a finite,
            // homogeneous, total universe of strings (e.g. `v \in {"a", "b"}` in
            // `TypeOK`). The encoding is identical to `ScalarString` (interned
            // `NameId` in one i64), so the upgrade never changes how the value is
            // stored — it only authorizes the variable to be primary-flat.
            let observed = FlatScalarValue::String(s.clone().into());
            if let Some(hint) =
                context.unique_fixed_scalar_var_proof(path, SlotType::String, &observed)
            {
                VarLayoutKind::FixedScalar {
                    base: SlotType::String,
                    proof: hint.proof.clone(),
                }
            } else {
                VarLayoutKind::ScalarString
            }
        }
        Value::ModelValue(s) => {
            let observed = FlatScalarValue::ModelValue(s.clone().into());
            if let Some(hint) =
                context.unique_fixed_scalar_var_proof(path, SlotType::ModelValue, &observed)
            {
                VarLayoutKind::FixedScalar {
                    base: SlotType::ModelValue,
                    proof: hint.proof.clone(),
                }
            } else {
                VarLayoutKind::ScalarModelValue
            }
        }

        // Integer-indexed function: flatten to IntArray if all values are scalar.
        Value::IntFunc(func) => {
            let f = func.as_ref();
            let all_scalar = f.values().iter().all(is_scalar_value);
            if all_scalar && !f.is_empty() {
                let elements_are_bool = f.values().iter().all(|v| matches!(v, Value::Bool(_)));
                let has_string = f
                    .values()
                    .iter()
                    .any(|v| matches!(v, Value::String(_) | Value::ModelValue(_)));
                let element_types: Option<Vec<SlotType>> = if has_string || !elements_are_bool {
                    // Track per-element types when there are mixed types.
                    Some(f.values().iter().map(slot_type_from_value).collect())
                } else {
                    None
                };
                // G2-extension: a homogeneous string/model-value range proved by a
                // `TypeOK` `[Dom -> ScalarEnum]` clause upgrades the array to
                // default flat-BFS storage. The element range proof keys off the
                // function's integer domain `lo .. lo+len`.
                let element_range_proof = if has_string {
                    let lo = f.min();
                    let domain_values: Vec<Value> =
                        (lo..lo + f.len() as i64).map(Value::SmallInt).collect();
                    let value_types: Vec<SlotType> =
                        f.values().iter().map(slot_type_from_value).collect();
                    context
                        .unique_fixed_scalar_range_proof(path, &domain_values, &value_types)
                        .map(|hint| hint.proof.clone())
                } else {
                    None
                };
                VarLayoutKind::IntArray {
                    lo: f.min(),
                    len: f.len(),
                    elements_are_bool,
                    element_types,
                    element_range_proof,
                }
            } else if let Some(layout) = infer_fixed_value_layout(value, context, path) {
                VarLayoutKind::Recursive { layout }
            } else if f.is_empty() {
                // Empty function: treat as scalar (zero slots would be odd).
                VarLayoutKind::Dynamic
            } else {
                VarLayoutKind::Dynamic
            }
        }

        // General function: flatten to IntArray if domain is integer interval
        // and all range values are scalar, or StringKeyedArray if domain is
        // strings/model-values.
        Value::Func(func) => {
            if func.domain_is_empty() {
                return VarLayoutKind::Dynamic;
            }

            // Check if domain is contiguous integers.
            let mut is_int_domain = true;
            let mut is_string_domain = true;
            // A tuple/cross-product domain (e.g. GameOfLife's `grid` keyed by
            // `<<x, y>>`): every key must be a `Value::Tuple` whose elements are
            // all scalars. Tracked alongside the int/string checks; the
            // dedicated branch below emits a `TupleKeyedArray`.
            let mut is_tuple_domain = true;
            let mut min_key = i64::MAX;
            let mut max_key = i64::MIN;
            for key in func.domain_iter() {
                match key {
                    Value::SmallInt(n) => {
                        is_string_domain = false;
                        is_tuple_domain = false;
                        min_key = min_key.min(*n);
                        max_key = max_key.max(*n);
                    }
                    Value::String(_) | Value::ModelValue(_) => {
                        is_int_domain = false;
                        is_tuple_domain = false;
                    }
                    Value::Tuple(elems) if elems.iter().all(is_scalar_value) => {
                        is_int_domain = false;
                        is_string_domain = false;
                    }
                    _ => {
                        is_int_domain = false;
                        is_string_domain = false;
                        is_tuple_domain = false;
                        break;
                    }
                }
            }

            if is_int_domain {
                let expected_len = (max_key - min_key + 1) as usize;
                if expected_len == func.domain_len() {
                    // Contiguous integer domain. Check range values.
                    if let Some(int_array) =
                        try_int_array_from_func(func, min_key, expected_len, path, context)
                    {
                        return int_array;
                    }
                }
            }

            // String-keyed function: flatten to StringKeyedArray.
            // Part of #3908: compound type flat state roundtrip.
            if is_string_domain {
                let mut domain_values: Vec<Value> = Vec::with_capacity(func.domain_len());
                let mut domain_keys: Vec<Arc<str>> = Vec::with_capacity(func.domain_len());
                let mut domain_types: Vec<SlotType> = Vec::with_capacity(func.domain_len());
                let mut range_values: Vec<&Value> = Vec::with_capacity(func.domain_len());
                for (key, val) in func.iter() {
                    let (key_str, key_ty) = match key {
                        Value::String(s) => (s.clone().into(), SlotType::String),
                        Value::ModelValue(s) => (s.clone().into(), SlotType::ModelValue),
                        _ => unreachable!("checked is_string_domain above"),
                    };
                    domain_values.push(key.clone());
                    domain_keys.push(key_str);
                    domain_types.push(key_ty);
                    range_values.push(val);
                }

                if let Some(hint) = context.unique_tagged_scalar_set_range_proof_for_values(
                    path,
                    &domain_values,
                    &range_values,
                ) {
                    let value_types = vec![hint.proof.scalar_type(); func.domain_len()];
                    return VarLayoutKind::StringKeyedArray {
                        domain_keys,
                        domain_types,
                        value_types,
                        range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(
                            hint.proof.clone(),
                        ),
                    };
                }

                let all_range_scalar = range_values.iter().all(|value| is_scalar_value(value));
                if all_range_scalar {
                    let value_types: Vec<SlotType> = range_values
                        .iter()
                        .map(|value| slot_type_from_value(value))
                        .collect();
                    let range_encoding = context
                        .unique_tagged_scalar_set_range_proof(path, &domain_values, &value_types)
                        .map(|hint| {
                            StringKeyedArrayRangeEncoding::TaggedScalarOrSet(hint.proof.clone())
                        })
                        .or_else(|| {
                            context
                                .unique_fixed_scalar_range_proof(path, &domain_values, &value_types)
                                .map(|hint| {
                                    StringKeyedArrayRangeEncoding::FixedScalar(hint.proof.clone())
                                })
                        })
                        .unwrap_or(StringKeyedArrayRangeEncoding::ScalarSlots);
                    return VarLayoutKind::StringKeyedArray {
                        domain_keys,
                        domain_types,
                        value_types,
                        range_encoding,
                    };
                }
            }

            // Tuple/cross-product-keyed function: flatten to TupleKeyedArray.
            //
            // The domain is a fully-enumerated, static finite set of scalar
            // tuples (a concrete `Value::Func` always has its complete domain
            // materialized, so it is statically enumerable here). Sort the keys
            // canonically (`Value` lexicographic order) and assign one
            // contiguous i64 slot per key, exactly like StringKeyedArray. Only
            // admit a plain-scalar range so the flat slots are plain i64; a
            // non-scalar range falls through to the recursive/Dynamic path.
            if is_tuple_domain {
                let all_range_scalar = func.iter().all(|(_, value)| is_scalar_value(value));
                if all_range_scalar {
                    // Collect (key, value) pairs and sort by canonical key order
                    // so the slot assignment is stable and reconstruction yields
                    // a sorted function.
                    let mut entries: Vec<(Value, &Value)> =
                        func.iter().map(|(key, val)| (key.clone(), val)).collect();
                    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
                    let domain_keys: Vec<Value> =
                        entries.iter().map(|(key, _)| key.clone()).collect();
                    let value_types: Vec<SlotType> = entries
                        .iter()
                        .map(|(_, value)| slot_type_from_value(value))
                        .collect();
                    // A homogeneous string/model-value range can be upgraded by
                    // a validated finite-universe `TypeOK` proof (the tuple
                    // analogue of the `StringKeyedArray` `FixedScalar` route;
                    // btree `valOf \in [Nodes \X Keys -> Vals \cup {NIL}]`).
                    // Absent or mismatched proofs stay `ScalarSlots` — plain-i64
                    // ranges are already structurally safe, anything else fails
                    // closed at the admission gates.
                    let range_encoding = context
                        .unique_fixed_scalar_range_proof(path, &domain_keys, &value_types)
                        .map(|hint| TupleKeyedArrayRangeEncoding::FixedScalar(hint.proof.clone()))
                        .unwrap_or(TupleKeyedArrayRangeEncoding::ScalarSlots);
                    return VarLayoutKind::TupleKeyedArray {
                        domain_keys,
                        value_types,
                        range_encoding,
                    };
                }
            }

            if let Some(layout) = infer_fixed_value_layout(value, context, path) {
                VarLayoutKind::Recursive { layout }
            } else {
                VarLayoutKind::Dynamic
            }
        }

        // Record: flatten if all fields are scalar.
        Value::Record(rec) => {
            let mut all_scalar = true;
            let mut field_names = Vec::with_capacity(rec.len());
            let mut field_is_bool = Vec::with_capacity(rec.len());
            let mut field_types = Vec::with_capacity(rec.len());
            // G2-extension: per-field finite-universe proofs (parallel to
            // `field_types`). A `Some` entry certifies a string/model-value field
            // is drawn from a finite homogeneous scalar universe (e.g. EWD998
            // `token.color \in Color`), making the record default-flat-admissible.
            let mut field_range_proofs: Vec<Option<FixedScalarRangeProof>> =
                Vec::with_capacity(rec.len());
            let mut any_field_proof = false;

            for (nid, val) in rec.iter() {
                let field_name = tla_core::resolve_name_id(nid);
                let field_proof = match val {
                    Value::String(s) => {
                        let field_path =
                            path.child(SequencePathStep::RecordField(Arc::clone(&field_name)));
                        let observed = FlatScalarValue::String(s.clone().into());
                        context
                            .unique_fixed_scalar_var_proof(&field_path, SlotType::String, &observed)
                            .map(|hint| hint.proof.clone())
                    }
                    Value::ModelValue(s) => {
                        let field_path =
                            path.child(SequencePathStep::RecordField(Arc::clone(&field_name)));
                        let observed = FlatScalarValue::ModelValue(s.clone().into());
                        context
                            .unique_fixed_scalar_var_proof(
                                &field_path,
                                SlotType::ModelValue,
                                &observed,
                            )
                            .map(|hint| hint.proof.clone())
                    }
                    _ => None,
                };
                any_field_proof |= field_proof.is_some();
                field_range_proofs.push(field_proof);
                field_names.push(field_name);
                field_is_bool.push(matches!(val, Value::Bool(_)));
                field_types.push(slot_type_from_value(val));
                if !is_scalar_value(val) {
                    all_scalar = false;
                    break;
                }
            }

            if all_scalar && !field_names.is_empty() {
                VarLayoutKind::Record {
                    field_names,
                    field_is_bool,
                    field_types,
                    field_range_proofs: any_field_proof.then_some(field_range_proofs),
                }
            } else if let Some(layout) = infer_fixed_value_layout(value, context, path) {
                VarLayoutKind::Recursive { layout }
            } else {
                VarLayoutKind::Dynamic
            }
        }

        Value::Set(_) | Value::Seq(_) | Value::Tuple(_) => {
            if let Some(layout) = infer_fixed_value_layout(value, context, path) {
                VarLayoutKind::Recursive { layout }
            } else {
                VarLayoutKind::Dynamic
            }
        }

        // Everything else: Dynamic fallback.
        _ => VarLayoutKind::Dynamic,
    }
}

fn infer_fixed_value_layout(
    value: &Value,
    context: &LayoutInferenceContext,
    path: &SequencePath,
) -> Option<FlatValueLayout> {
    match value {
        Value::Bool(_) => Some(FlatValueLayout::Scalar(SlotType::Bool)),
        Value::SmallInt(_) | Value::Int(_) => Some(FlatValueLayout::Scalar(SlotType::Int)),
        Value::String(_) => Some(FlatValueLayout::Scalar(SlotType::String)),
        Value::ModelValue(_) => Some(FlatValueLayout::Scalar(SlotType::ModelValue)),

        Value::IntFunc(func) => {
            if func.is_empty() {
                return None;
            }
            let value_path = path.child(int_func_domain_path_step(func));
            let value_layout =
                infer_common_flat_layout(func.values().iter(), context, &value_path)?;
            Some(FlatValueLayout::IntFunction {
                lo: func.as_ref().min(),
                len: func.len(),
                value_layout: Box::new(value_layout),
            })
        }

        Value::Func(func) => {
            if func.domain_is_empty() {
                return None;
            }
            let value_path = path.child(func_domain_path_step(func)?);
            let value_layout =
                infer_common_flat_layout(func.mapping_values(), context, &value_path)?;
            if let Some((lo, len)) = contiguous_int_domain(func) {
                return Some(FlatValueLayout::IntFunction {
                    lo,
                    len,
                    value_layout: Box::new(value_layout),
                });
            }

            let mut domain = Vec::with_capacity(func.domain_len());
            for key in func.domain_iter() {
                domain.push(flat_scalar_from_value(key)?);
            }
            Some(FlatValueLayout::Function {
                domain,
                value_layout: Box::new(value_layout),
            })
        }

        Value::Record(record) => {
            if record.is_empty() {
                return None;
            }
            let mut field_names = Vec::with_capacity(record.len());
            let mut field_layouts = Vec::with_capacity(record.len());
            for (name_id, field_value) in record.iter() {
                let field_name = tla_core::resolve_name_id(name_id);
                let field_path = path.child(SequencePathStep::RecordField(Arc::clone(&field_name)));
                field_names.push(field_name);
                field_layouts.push(infer_fixed_value_layout(field_value, context, &field_path)?);
            }
            Some(FlatValueLayout::Record {
                field_names,
                field_layouts,
            })
        }

        Value::Set(set) => infer_set_bitmask_layout(set, context, path),

        Value::Seq(seq) => {
            if seq.is_empty() {
                let proven_element = context.unique_sequence_element_proof(path);
                let (observed_len, element_layout) = if let Some(proof) = proven_element {
                    (0, proof.element_layout.clone())
                } else if let Some(hint) = context.unique_sequence_hint(path) {
                    (hint.max_len, hint.element_layout.clone())
                } else {
                    (0, FlatValueLayout::Scalar(SlotType::Int))
                };
                let (bound, max_len, element_layout) = fixed_domain_sequence_layout_for_path(
                    context,
                    path,
                    observed_len,
                    &element_layout,
                )
                .unwrap_or_else(|| {
                    let (bound, max_len) = sequence_bound_evidence_for_path(
                        context,
                        path,
                        observed_len,
                        proven_element.map(|proof| &proof.invariant),
                    );
                    (bound, max_len, element_layout)
                });
                return Some(FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout: Box::new(element_layout),
                });
            }
            let element_path = path.child(SequencePathStep::SequenceElement);
            let observed_element_layout =
                infer_common_flat_layout(seq.iter(), context, &element_path)?;
            let proven_element = context.unique_sequence_element_proof(path).filter(|proof| {
                seq.iter()
                    .all(|value| value_fits_flat_value_layout(value, &proof.element_layout))
            });
            let element_layout = proven_element
                .map(|proof| proof.element_layout.clone())
                .unwrap_or(observed_element_layout);
            let (bound, max_len, element_layout) =
                fixed_domain_sequence_layout_for_path(context, path, seq.len(), &element_layout)
                    .unwrap_or_else(|| {
                        let (bound, max_len) = sequence_bound_evidence_for_path(
                            context,
                            path,
                            seq.len(),
                            proven_element.map(|proof| &proof.invariant),
                        );
                        (bound, max_len, element_layout)
                    });
            Some(FlatValueLayout::Sequence {
                bound,
                max_len,
                element_layout: Box::new(element_layout),
            })
        }

        Value::Tuple(elems) => {
            if elems.is_empty() {
                let proven_element = context.unique_sequence_element_proof(path);
                let (observed_len, element_layout) = if let Some(proof) = proven_element {
                    (0, proof.element_layout.clone())
                } else if let Some(hint) = context.unique_sequence_hint(path) {
                    (hint.max_len, hint.element_layout.clone())
                } else {
                    (0, FlatValueLayout::Scalar(SlotType::Int))
                };
                let (bound, max_len, element_layout) = fixed_domain_sequence_layout_for_path(
                    context,
                    path,
                    observed_len,
                    &element_layout,
                )
                .unwrap_or_else(|| {
                    let (bound, max_len) = sequence_bound_evidence_for_path(
                        context,
                        path,
                        observed_len,
                        proven_element.map(|proof| &proof.invariant),
                    );
                    (bound, max_len, element_layout)
                });
                return Some(FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout: Box::new(element_layout),
                });
            }
            let element_path = path.child(SequencePathStep::SequenceElement);
            let observed_element_layout =
                infer_common_flat_layout(elems.iter(), context, &element_path)?;
            let proven_element = context.unique_sequence_element_proof(path).filter(|proof| {
                elems
                    .iter()
                    .all(|value| value_fits_flat_value_layout(value, &proof.element_layout))
            });
            let element_layout = proven_element
                .map(|proof| proof.element_layout.clone())
                .unwrap_or(observed_element_layout);
            let (bound, max_len, element_layout) =
                fixed_domain_sequence_layout_for_path(context, path, elems.len(), &element_layout)
                    .unwrap_or_else(|| {
                        let (bound, max_len) = sequence_bound_evidence_for_path(
                            context,
                            path,
                            elems.len(),
                            proven_element.map(|proof| &proof.invariant),
                        );
                        (bound, max_len, element_layout)
                    });
            Some(FlatValueLayout::Sequence {
                bound,
                max_len,
                element_layout: Box::new(element_layout),
            })
        }

        _ => None,
    }
}

fn infer_common_flat_layout<'a, I>(
    values: I,
    context: &LayoutInferenceContext,
    path: &SequencePath,
) -> Option<FlatValueLayout>
where
    I: IntoIterator<Item = &'a Value>,
{
    let mut iter = values.into_iter();
    let first = iter.next()?;
    let mut layout = infer_fixed_value_layout(first, context, path)?;
    for value in iter {
        let next = infer_fixed_value_layout(value, context, path)?;
        layout = merge_flat_value_layouts(&layout, &next)?;
    }
    Some(layout)
}

fn infer_set_bitmask_layout(
    set: &tla_value::value::SortedSet,
    context: &LayoutInferenceContext,
    path: &SequencePath,
) -> Option<FlatValueLayout> {
    // Nested-set (set-of-sets) bitmask: a value whose every element is itself a
    // set is a candidate for the two-level `NestedSetBitmask` layout. This
    // single-VALUE inference arm stays INERT (always returns `None`) even after
    // A5 promotion — see `infer_nested_set_bitmask_layout`. A5 promotes via the
    // FROZEN multi-board monitor (`freeze_nested_set_monitors_from_seeds`) at the
    // dedup-fingerprint hook, NOT via this arm (which would derive an incomplete
    // single-board universe). So the var's inferred layout stays `Dynamic`; the
    // interpreter generates successors and the monitor gates the dedup.
    if !set.is_empty() && set.iter().all(|elem| matches!(elem, Value::Set(_))) {
        if let Some(layout) = infer_nested_set_bitmask_layout(set) {
            return Some(layout);
        }
        // Fall through: a set-of-sets that did not yield a nested-set layout is
        // not a scalar/record bitmask either, so the scalar path below will
        // also decline (no element is scalar-convertible) → Dynamic.
    }
    // Record-set bitmask: a `v \in SUBSET RecSet` proof over a finite, statically
    // enumerable record universe. This takes priority over the scalar SetBitmask
    // path because record elements are not scalar-convertible. The proof's
    // universe is canonical (sorted, deduped, all records) and proven closed by
    // the source-level type invariant, so every sampled element must already lie
    // inside it; an out-of-universe sample fails closed (returns None →
    // Dynamic).
    if let Some(proof) = context.unique_record_set_bitmask_type_proof(path) {
        let all_in_universe = set.iter().all(|value| {
            matches!(value, Value::Record(_)) && proof.record_universe.iter().any(|u| u == value)
        });
        if all_in_universe
            && proof.record_universe.len() <= super::flat_state::MAX_RECORD_SET_BITMASK_UNIVERSE
        {
            return Some(FlatValueLayout::RecordSetBitmask {
                universe: proof.record_universe.clone(),
                universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                    invariant: Arc::clone(&proof.invariant),
                },
            });
        }
        return None;
    }

    let mut elements = Vec::with_capacity(set.len());
    for value in set.iter() {
        elements.push(flat_scalar_from_value(value)?);
    }
    if let Some(proof) = context.unique_set_bitmask_type_proof(path) {
        return elements
            .iter()
            .all(|elem| proof.set_universe.contains(elem))
            .then(|| FlatValueLayout::SetBitmask {
                universe: proof.set_universe.clone(),
                universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                    invariant: Arc::clone(&proof.invariant),
                },
            });
    }
    if let Some(proof) = context.unique_set_bitmask_range_proof(path) {
        return elements
            .iter()
            .all(|elem| proof.set_universe.contains(elem))
            .then(|| FlatValueLayout::SetBitmask {
                universe: proof.set_universe.clone(),
                universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                    invariant: Arc::clone(&proof.invariant),
                },
            });
    }
    if elements.is_empty() {
        if path_has_range_like_ancestor(path) {
            return None;
        }
        return None;
    }
    // Sampled universe: no closure proof, so the bitmask is only sound for
    // flat-state roundtrip / top-level slots, never as a function-range
    // flat-primary slot.
    let universe = context.unique_scalar_domain_covering(&elements)?;
    Some(FlatValueLayout::SetBitmask {
        universe,
        universe_closure: SetBitmaskUniverseClosure::Sampled,
    })
}

/// Infer a sampled two-level [`FlatValueLayout::NestedSetBitmask`] from a single
/// set-of-sets value (single-VALUE inference construction).
///
/// Mirrors the discovery sampler's derivation but over one observed value: the
/// distinct inner elements (sorted + deduped) define the `inner_universe` scalar
/// id space, each piece folds into a `u64` inner-mask, and the distinct masks
/// (sorted + deduped) form the `outer_universe`. The closure is
/// `DynamicallyDiscovered { monitor_enforced: false }` — a sampled universe,
/// never proven closed.
///
/// INERT GATE: returns `None` unconditionally (its own
/// `NESTED_SET_BITMASK_LAYOUT_INFERENCE = false`), independent of the A5
/// promotion gate. A single-board universe is incomplete (the next sliding
/// successor escapes), so it is UNSOUND as a frozen flat-state layout. The sound
/// A5 universe is the FROZEN multi-board one discovered by the successor-aware
/// sampler and installed as a per-successor monitor at the dedup hook — not as
/// this var's `VarLayoutKind`. This arm becomes the Step-B native-layout seam.
fn infer_nested_set_bitmask_layout(set: &tla_value::value::SortedSet) -> Option<FlatValueLayout> {
    // A5 deliberately keeps this single-VALUE inference arm INERT.
    //
    // It would derive a universe from ONE observed board — an incomplete
    // universe that the very next sliding successor escapes — which is UNSOUND
    // as a frozen flat-state layout. The sound A5 universe is the multi-board
    // FROZEN universe discovered by the successor-aware sampler
    // (`freeze_nested_set_monitors_from_seeds`), installed as a per-successor
    // `NestedSetVarMonitor` at the dedup-fingerprint hook — NOT as the var's
    // inferred `VarLayoutKind` (which would route to flat-primary/native; that
    // is Step B). So the variable's inferred layout stays `Dynamic` (the
    // interpreter generates successors), and this arm returns `None`
    // unconditionally. `NESTED_SET_BITMASK_LAYOUT_INFERENCE` is the Step-B seam.
    const NESTED_SET_BITMASK_LAYOUT_INFERENCE: bool = false;
    if !NESTED_SET_BITMASK_LAYOUT_INFERENCE {
        return None;
    }

    if set.is_empty() {
        return None;
    }
    // (1) Distinct inner elements (canonical `Value` order) → scalar id space.
    let mut inner_set: std::collections::BTreeSet<Value> = std::collections::BTreeSet::new();
    for piece in set.iter() {
        let Value::Set(inner) = piece else {
            return None;
        };
        for elem in inner.iter() {
            inner_set.insert(elem.clone());
        }
    }
    if inner_set.len() > super::flat_state::MAX_NESTED_SET_INNER_UNIVERSE {
        return None;
    }
    let inner_index: std::collections::BTreeMap<Value, usize> = inner_set
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();
    let inner_universe: Vec<FlatScalarValue> = (0..inner_set.len())
        .map(|i| FlatScalarValue::Int(i as i64))
        .collect();

    // (2) Fold pieces → distinct u64 inner-masks → outer universe.
    let mut outer_set: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for piece in set.iter() {
        let Value::Set(inner) = piece else {
            return None;
        };
        let mut mask = 0u64;
        for elem in inner.iter() {
            let &id = inner_index.get(elem)?;
            mask |= 1u64 << id;
        }
        outer_set.insert(mask);
    }
    let outer_universe: Vec<u64> = outer_set.into_iter().collect();
    if outer_universe.is_empty()
        || outer_universe.len() > super::flat_state::MAX_RECORD_SET_BITMASK_UNIVERSE
    {
        return None;
    }

    Some(FlatValueLayout::NestedSetBitmask {
        outer_universe,
        inner_universe,
        outer_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
            monitor_enforced: false,
        },
        inner_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
            monitor_enforced: false,
        },
    })
}

fn path_has_range_like_ancestor(path: &SequencePath) -> bool {
    path.0.iter().any(|step| {
        matches!(
            step,
            SequencePathStep::HomogeneousRange(_) | SequencePathStep::SequenceElement
        )
    })
}

fn int_func_domain_path_step(func: &tla_value::value::IntIntervalFunc) -> SequencePathStep {
    let domain: Vec<Value> = (func.min()..=func.max()).map(Value::SmallInt).collect();
    SequencePathStep::HomogeneousRange(Arc::from(domain.into_boxed_slice()))
}

fn func_domain_path_step(func: &tla_value::value::FuncValue) -> Option<SequencePathStep> {
    let domain: Vec<Value> = func.domain_iter().cloned().collect();
    normalized_scalar_domain_values(domain)
        .map(|domain| SequencePathStep::HomogeneousRange(Arc::from(domain.into_boxed_slice())))
}

fn normalized_scalar_domain_values(mut values: Vec<Value>) -> Option<Vec<Value>> {
    if values.is_empty() {
        return None;
    }
    if !values.iter().all(is_scalar_value) {
        return None;
    }
    values.sort();
    values.dedup();
    Some(values)
}

fn contiguous_int_domain(func: &tla_value::value::FuncValue) -> Option<(i64, usize)> {
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    for key in func.domain_iter() {
        match key {
            Value::SmallInt(n) => {
                min_key = min_key.min(*n);
                max_key = max_key.max(*n);
            }
            _ => return None,
        }
    }
    let len = (max_key - min_key + 1) as usize;
    (len == func.domain_len()).then_some((min_key, len))
}

fn merge_flat_value_layouts(a: &FlatValueLayout, b: &FlatValueLayout) -> Option<FlatValueLayout> {
    match (a, b) {
        (FlatValueLayout::Scalar(a), FlatValueLayout::Scalar(b)) if a == b => {
            Some(FlatValueLayout::Scalar(*a))
        }
        (
            FlatValueLayout::TaggedScalarUnion { proof: a },
            FlatValueLayout::TaggedScalarUnion { proof: b },
        ) if a == b => Some(FlatValueLayout::TaggedScalarUnion { proof: a.clone() }),
        (FlatValueLayout::TaggedUnion { proof: a }, FlatValueLayout::TaggedUnion { proof: b })
            if a == b =>
        {
            Some(FlatValueLayout::TaggedUnion { proof: a.clone() })
        }
        (
            FlatValueLayout::IntFunction {
                lo: lo_a,
                len: len_a,
                value_layout: value_a,
            },
            FlatValueLayout::IntFunction {
                lo: lo_b,
                len: len_b,
                value_layout: value_b,
            },
        ) if lo_a == lo_b && len_a == len_b => {
            let value_layout = merge_flat_value_layouts(value_a, value_b)?;
            Some(FlatValueLayout::IntFunction {
                lo: *lo_a,
                len: *len_a,
                value_layout: Box::new(value_layout),
            })
        }
        (
            FlatValueLayout::IntFunction {
                lo: lo_a,
                len: len_a,
                value_layout: value_a,
            },
            FlatValueLayout::Function {
                domain: domain_b,
                value_layout: value_b,
            },
        ) if ordered_dense_int_domain(domain_b) == Some((*lo_a, *len_a)) => {
            let value_layout = merge_flat_value_layouts(value_a, value_b)?;
            Some(FlatValueLayout::IntFunction {
                lo: *lo_a,
                len: *len_a,
                value_layout: Box::new(value_layout),
            })
        }
        (
            FlatValueLayout::Function {
                domain: domain_a,
                value_layout: value_a,
            },
            FlatValueLayout::IntFunction {
                lo: lo_b,
                len: len_b,
                value_layout: value_b,
            },
        ) if ordered_dense_int_domain(domain_a) == Some((*lo_b, *len_b)) => {
            let value_layout = merge_flat_value_layouts(value_a, value_b)?;
            Some(FlatValueLayout::IntFunction {
                lo: *lo_b,
                len: *len_b,
                value_layout: Box::new(value_layout),
            })
        }
        (
            FlatValueLayout::Function {
                domain: domain_a,
                value_layout: value_a,
            },
            FlatValueLayout::Function {
                domain: domain_b,
                value_layout: value_b,
            },
        ) if domain_a == domain_b => {
            let value_layout = merge_flat_value_layouts(value_a, value_b)?;
            Some(flat_function_layout(domain_a.clone(), value_layout))
        }
        (
            FlatValueLayout::Record {
                field_names: names_a,
                field_layouts: fields_a,
            },
            FlatValueLayout::Record {
                field_names: names_b,
                field_layouts: fields_b,
            },
        ) if names_a == names_b && fields_a.len() == fields_b.len() => {
            let mut field_layouts = Vec::with_capacity(fields_a.len());
            for (field_a, field_b) in fields_a.iter().zip(fields_b.iter()) {
                field_layouts.push(merge_flat_value_layouts(field_a, field_b)?);
            }
            Some(FlatValueLayout::Record {
                field_names: names_a.clone(),
                field_layouts,
            })
        }
        (
            FlatValueLayout::SetBitmask {
                universe: universe_a,
                universe_closure: closure_a,
            },
            FlatValueLayout::SetBitmask {
                universe: universe_b,
                universe_closure: closure_b,
            },
        ) => {
            let mut universe = universe_a.clone();
            universe.extend(universe_b.iter().cloned());
            universe.sort();
            universe.dedup();
            // Closure only survives when both sides were proven closed by the
            // same invariant. A grown union of two sampled universes (or a
            // proven/sampled mix) is no longer provably closed.
            let universe_closure = if universe == *universe_a && universe == *universe_b {
                closure_a.merge(closure_b)
            } else {
                SetBitmaskUniverseClosure::Sampled
            };
            (universe.len() <= 63).then_some(FlatValueLayout::SetBitmask {
                universe,
                universe_closure,
            })
        }
        (
            FlatValueLayout::RecordSetBitmask {
                universe: universe_a,
                universe_closure: closure_a,
            },
            FlatValueLayout::RecordSetBitmask {
                universe: universe_b,
                universe_closure: closure_b,
            },
        ) => {
            let mut universe = universe_a.clone();
            universe.extend(universe_b.iter().cloned());
            universe.sort();
            universe.dedup();
            // Closure only survives when both sides were proven closed by the
            // same invariant and neither side grew the universe; otherwise the
            // grown union is no longer provably closed (fail-closed).
            let universe_closure = if universe == *universe_a && universe == *universe_b {
                closure_a.merge(closure_b)
            } else {
                SetBitmaskUniverseClosure::Sampled
            };
            (universe.len() <= super::flat_state::MAX_RECORD_SET_BITMASK_UNIVERSE).then_some(
                FlatValueLayout::RecordSetBitmask {
                    universe,
                    universe_closure,
                },
            )
        }
        // Nested-set (set-of-sets) two-level merge (nested-set discovery A4).
        //
        // The two sides may carry DIFFERENT inner-universe index spaces (each
        // side's outer masks are bit-indexed against its own `inner_universe`),
        // so a naive outer-mask union would be unsound. Merge in two tiers:
        //
        //   (1) union the inner scalar universes (sorted + deduped) → the merged
        //       inner index space; build, per side, a remap old-bit → new-bit;
        //   (2) RE-BASE both sides' outer masks onto the merged inner index
        //       space via that remap, then union + dedup the re-based masks.
        //
        // The closure degrades to `DynamicallyDiscovered { monitor_enforced:
        // false }` (the sampled provenance) whenever either universe grew — a
        // grown union is not proven closed; this never out-grows the sampled
        // fail-closed posture. Caps mirror the codec (`inner ≤ 64`, `outer ≤`
        // the multi-slot cap) so a merged layout is always encodable.
        (
            FlatValueLayout::NestedSetBitmask {
                outer_universe: outer_a,
                inner_universe: inner_a,
                ..
            },
            FlatValueLayout::NestedSetBitmask {
                outer_universe: outer_b,
                inner_universe: inner_b,
                ..
            },
        ) => {
            // (1) Merge inner universes (union scalars, canonical order).
            let mut inner_universe: Vec<FlatScalarValue> = inner_a.clone();
            inner_universe.extend(inner_b.iter().cloned());
            inner_universe.sort();
            inner_universe.dedup();
            if inner_universe.len() > super::flat_state::MAX_NESTED_SET_INNER_UNIVERSE {
                return None;
            }
            // Per-side remap: old inner bit index → merged inner bit index.
            let remap_for = |inner: &[FlatScalarValue]| -> Option<Vec<usize>> {
                inner
                    .iter()
                    .map(|elem| inner_universe.iter().position(|m| m == elem))
                    .collect()
            };
            let remap_a = remap_for(inner_a)?;
            let remap_b = remap_for(inner_b)?;
            let rebase = |mask: u64, remap: &[usize]| -> u64 {
                let mut out = 0u64;
                for (old_bit, &new_bit) in remap.iter().enumerate() {
                    if (mask & (1u64 << old_bit)) != 0 {
                        out |= 1u64 << new_bit;
                    }
                }
                out
            };
            // (2) Re-base + union + dedup outer masks onto the merged space.
            let mut outer_set: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
            for &m in outer_a {
                outer_set.insert(rebase(m, &remap_a));
            }
            for &m in outer_b {
                outer_set.insert(rebase(m, &remap_b));
            }
            let outer_universe: Vec<u64> = outer_set.into_iter().collect();
            if outer_universe.is_empty()
                || outer_universe.len() > super::flat_state::MAX_RECORD_SET_BITMASK_UNIVERSE
            {
                return None;
            }
            // Closure: a discovered (sampled) nested-set universe is NEVER
            // proven closed — neither input could be (only the const-gated A5
            // promotion path could carry a proof), and a grown union is even
            // less so. So the merged closure is unconditionally the sampled
            // `DynamicallyDiscovered { monitor_enforced: false }`, matching the
            // discovery sampler. This stays fail-closed for flat-primary exactly
            // like `Sampled` (see `SetBitmaskUniverseClosure::is_proven_closed`).
            Some(FlatValueLayout::NestedSetBitmask {
                outer_universe,
                inner_universe,
                outer_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
                    monitor_enforced: false,
                },
                inner_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
                    monitor_enforced: false,
                },
            })
        }
        (
            FlatValueLayout::Sequence {
                bound: bound_a,
                max_len: max_a,
                element_layout: elem_a,
            },
            FlatValueLayout::Sequence {
                bound: bound_b,
                max_len: max_b,
                element_layout: elem_b,
            },
        ) => {
            if *max_a == 0 {
                return Some(FlatValueLayout::Sequence {
                    bound: bound_b.clone(),
                    max_len: *max_b,
                    element_layout: elem_b.clone(),
                });
            }
            if *max_b == 0 {
                return Some(FlatValueLayout::Sequence {
                    bound: bound_a.clone(),
                    max_len: *max_a,
                    element_layout: elem_a.clone(),
                });
            }
            let element_layout = merge_flat_value_layouts(elem_a, elem_b)?;
            Some(FlatValueLayout::Sequence {
                bound: merge_sequence_bound_evidence(bound_a, bound_b),
                max_len: (*max_a).max(*max_b),
                element_layout: Box::new(element_layout),
            })
        }
        _ => None,
    }
}

fn merge_sequence_bound_evidence(
    a: &SequenceBoundEvidence,
    b: &SequenceBoundEvidence,
) -> SequenceBoundEvidence {
    // A proven bound survives a merge only when both sides agree. A HEURISTIC
    // universe-capacity bound also survives IDENTICAL merges: it is derived from
    // the same checked `v \in Seq(U)` invariant for every state (not from the
    // sampled value), so two wavefront layouts for the same var carry byte-equal
    // heuristic bounds — demoting them to `Observed` would needlessly drop the
    // flat-primary promotion. Soundness is unaffected either way (the overflow
    // backstop guards the heuristic bound; `Observed` merely fails closed). A
    // DISAGREEING merge still collapses to `Observed`.
    if a == b
        && (a.is_proven() || matches!(a, SequenceBoundEvidence::HeuristicUniverseCapacity { .. }))
    {
        a.clone()
    } else {
        SequenceBoundEvidence::Observed
    }
}

fn value_fits_flat_value_layout(value: &Value, layout: &FlatValueLayout) -> bool {
    match layout {
        FlatValueLayout::Scalar(slot_type) => value_fits_slot_type(value, *slot_type),
        FlatValueLayout::IntFunction {
            lo,
            len,
            value_layout,
        } => value_fits_recursive_int_function(value, *lo, *len, value_layout),
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => value_fits_recursive_function(value, domain, value_layout),
        FlatValueLayout::Record {
            field_names,
            field_layouts,
        } => {
            let Value::Record(record) = value else {
                return false;
            };
            record.len() == field_names.len()
                && field_names
                    .iter()
                    .zip(field_layouts.iter())
                    .all(|(field_name, field_layout)| {
                        record
                            .get(field_name)
                            .is_some_and(|child| value_fits_flat_value_layout(child, field_layout))
                    })
        }
        FlatValueLayout::SetBitmask { universe, .. } => {
            let Value::Set(set) = value else {
                return false;
            };
            universe.len() <= 63
                && set
                    .iter()
                    .all(|elem| universe.iter().any(|u| flat_scalar_to_value(u) == *elem))
        }
        FlatValueLayout::RecordSetBitmask { universe, .. } => {
            let Value::Set(set) = value else {
                return false;
            };
            universe.len() <= super::flat_state::MAX_RECORD_SET_BITMASK_UNIVERSE
                && set.iter().all(|elem| universe.iter().any(|u| u == elem))
        }
        // Nested-set (set-of-sets) fit (A3): every element is itself a set whose
        // inner elements are in `inner_universe` and whose folded inner-mask is
        // an admitted member of `outer_universe`. Shares
        // `inner_set_value_to_mask` with the serializer so the fit predicate is
        // exactly the codec's tier-1 fold. INERT: no inference site constructs
        // this variant yet (A4), so this arm is unreachable in production.
        FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe,
            ..
        } => {
            let Value::Set(set) = value else {
                return false;
            };
            if inner_universe.len() > super::flat_state::MAX_NESTED_SET_INNER_UNIVERSE
                || outer_universe.len() > super::flat_state::MAX_RECORD_SET_BITMASK_UNIVERSE
            {
                return false;
            }
            set.iter().all(|piece| {
                super::flat_state::inner_set_value_to_mask(piece, inner_universe)
                    .is_ok_and(|mask| outer_universe.contains(&mask))
            })
        }
        FlatValueLayout::TaggedScalarUnion { proof } => flat_scalar_from_value(value)
            .is_some_and(|flat| proof.universe().iter().any(|candidate| candidate == &flat)),
        // A value fits a tagged union when exactly one variant accepts it. If
        // more than one variant matches the tag would be ambiguous, so the value
        // does not fit (fail closed).
        FlatValueLayout::TaggedUnion { proof } => {
            let mut matched = false;
            for variant in proof.variants() {
                if value_fits_flat_value_layout(value, variant) {
                    if matched {
                        return false;
                    }
                    matched = true;
                }
            }
            matched
        }
        // Fixed-arity heterogeneous tuple: exactly the layout's arity, each
        // position fitting its own layout.
        FlatValueLayout::HeterogeneousTuple { element_layouts } => match value {
            Value::Tuple(elems) => {
                elems.len() == element_layouts.len()
                    && elems
                        .iter()
                        .zip(element_layouts.iter())
                        .all(|(child, layout)| value_fits_flat_value_layout(child, layout))
            }
            Value::Seq(seq) => {
                seq.len() == element_layouts.len()
                    && element_layouts.iter().enumerate().all(|(index, layout)| {
                        seq.get(index)
                            .is_some_and(|child| value_fits_flat_value_layout(child, layout))
                    })
            }
            _ => false,
        },
        FlatValueLayout::Sequence {
            max_len,
            element_layout,
            ..
        } => match value {
            Value::Seq(seq) => {
                seq.len() <= *max_len
                    && seq
                        .iter()
                        .all(|child| value_fits_flat_value_layout(child, element_layout))
            }
            Value::Tuple(elems) => {
                elems.len() <= *max_len
                    && elems
                        .iter()
                        .all(|child| value_fits_flat_value_layout(child, element_layout))
            }
            _ => false,
        },
    }
}

fn value_fits_slot_type(value: &Value, slot_type: SlotType) -> bool {
    matches!(
        (value, slot_type),
        (Value::SmallInt(_) | Value::Int(_), SlotType::Int)
            | (Value::Bool(_), SlotType::Bool)
            | (Value::String(_), SlotType::String)
            | (Value::ModelValue(_), SlotType::ModelValue)
    )
}

fn value_fits_recursive_int_function(
    value: &Value,
    lo: i64,
    len: usize,
    value_layout: &FlatValueLayout,
) -> bool {
    match value {
        Value::IntFunc(func) => {
            let expected_hi = if len == 0 {
                lo.checked_sub(1)
            } else {
                lo.checked_add(len as i64 - 1)
            };
            expected_hi.is_some_and(|hi| func.as_ref().min() == lo && func.as_ref().max() == hi)
                && func
                    .values()
                    .iter()
                    .all(|child| value_fits_flat_value_layout(child, value_layout))
        }
        Value::Func(func) => {
            if func.domain_len() != len {
                return false;
            }
            (0..len).all(|index| {
                let key = Value::SmallInt(lo + index as i64);
                func.mapping_get(&key)
                    .is_some_and(|child| value_fits_flat_value_layout(child, value_layout))
            })
        }
        _ => false,
    }
}

fn value_fits_recursive_function(
    value: &Value,
    domain: &[FlatScalarValue],
    value_layout: &FlatValueLayout,
) -> bool {
    let Value::Func(func) = value else {
        return false;
    };
    func.domain_len() == domain.len()
        && domain.iter().all(|key| {
            let key_value = flat_scalar_to_value(key);
            func.mapping_get(&key_value)
                .is_some_and(|child| value_fits_flat_value_layout(child, value_layout))
        })
}

fn flat_scalar_to_value(value: &FlatScalarValue) -> Value {
    match value {
        FlatScalarValue::Int(n) => Value::SmallInt(*n),
        FlatScalarValue::Bool(b) => Value::Bool(*b),
        FlatScalarValue::String(s) => Value::String(s.clone().into()),
        FlatScalarValue::ModelValue(s) => Value::ModelValue(s.clone().into()),
    }
}

fn flat_scalar_from_value(value: &Value) -> Option<FlatScalarValue> {
    match value {
        Value::SmallInt(n) => Some(FlatScalarValue::Int(*n)),
        Value::Int(n) => {
            use num_traits::ToPrimitive;
            n.to_i64().map(FlatScalarValue::Int)
        }
        Value::Bool(b) => Some(FlatScalarValue::Bool(*b)),
        Value::String(s) => Some(FlatScalarValue::String(s.clone().into())),
        Value::ModelValue(s) => Some(FlatScalarValue::ModelValue(s.clone().into())),
        _ => None,
    }
}

/// Try to create an IntArray layout from a Func with contiguous integer domain.
fn try_int_array_from_func(
    func: &tla_value::value::FuncValue,
    min_key: i64,
    expected_len: usize,
    path: &SequencePath,
    context: &LayoutInferenceContext,
) -> Option<VarLayoutKind> {
    let mut all_scalar = true;
    let mut all_bool = true;
    let mut has_string = false;
    for (_key, val) in func.iter() {
        if !is_scalar_value(val) {
            all_scalar = false;
            break;
        }
        if !matches!(val, Value::Bool(_)) {
            all_bool = false;
        }
        if matches!(val, Value::String(_) | Value::ModelValue(_)) {
            has_string = true;
        }
    }
    if all_scalar {
        let element_types: Option<Vec<SlotType>> = if has_string || !all_bool {
            // Collect per-element types for reconstruction.
            let mut types = Vec::with_capacity(expected_len);
            for i in 0..expected_len {
                let key = Value::SmallInt(min_key + i as i64);
                if let Some(val) = func.apply(&key) {
                    types.push(slot_type_from_value(val));
                } else {
                    types.push(SlotType::Int);
                }
            }
            Some(types)
        } else {
            None
        };
        // G2-extension: upgrade a homogeneous string/model-value range to default
        // flat-BFS storage when a `TypeOK` `[Dom -> ScalarEnum]` proof certifies
        // the finite element universe. Keyed off the contiguous integer domain
        // `min_key .. min_key+expected_len`.
        let element_range_proof = if has_string {
            let domain_values: Vec<Value> = (min_key..min_key + expected_len as i64)
                .map(Value::SmallInt)
                .collect();
            let value_types: Vec<SlotType> = element_types
                .clone()
                .unwrap_or_else(|| vec![SlotType::Int; expected_len]);
            context
                .unique_fixed_scalar_range_proof(path, &domain_values, &value_types)
                .map(|hint| hint.proof.clone())
        } else {
            None
        };
        Some(VarLayoutKind::IntArray {
            lo: min_key,
            len: expected_len,
            elements_are_bool: all_bool,
            element_types,
            element_range_proof,
        })
    } else {
        None
    }
}

/// Check if a Value is a scalar that fits in a single i64.
fn is_scalar_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::String(_)
            | Value::ModelValue(_)
    )
}

/// Determine the SlotType for a scalar Value.
/// Part of #3908.
fn slot_type_from_value(value: &Value) -> SlotType {
    match value {
        Value::Bool(_) => SlotType::Bool,
        Value::String(_) => SlotType::String,
        Value::ModelValue(_) => SlotType::ModelValue,
        _ => SlotType::Int,
    }
}

// ============================================================================
// Duplicate-free bounded-universe sequence capacity proofs
// ============================================================================
//
// Proves `Len(v) <= |U|` for a sequence state variable `v` whose CHECKED type
// invariant contains `v \in Seq(D)` with `D` a finite constant set (`U`), by
// establishing the inductive invariant
//
//     DF_U(v):  v is a duplicate-free sequence with Range(v) ⊆ U
//
// over the Init/Next writer relation:
//
//   * base case — every Init write to `v` is the empty sequence `<< >>`;
//   * step case — every Next write to `v` is one of the DF_U-preserving forms:
//       - identity (`v' = v`, `UNCHANGED v`),
//       - the empty sequence,
//       - `Tail(v)` / `SubSeq(v, a, b)` (contiguous sub-windows of a
//         duplicate-free sequence stay duplicate-free; an out-of-range window
//         is a runtime evaluation error that aborts checking before any state
//         is stored),
//       - the element-removal idiom
//         `SubSeq(v, 1, i-1) \o SubSeq(v, i+1, Len(v))` (the two index windows
//         `[1, i-1]` and `[i+1, Len]` are disjoint for every `i`, so the
//         concatenation is a sub-multiset of the duplicate-free `v`),
//       - the disjoint append `v \o q` where `q` is existentially bound to a
//         domain `F({x \in B : ... /\ x \notin Range(v)})` whose base set `B`
//         is a constant subset of `U`, certified by EXHAUSTIVE EVALUATION:
//         for EVERY subset `X ⊆ B`, every member of `F(X)` is a duplicate-free
//         sequence over `X` (evaluated with the real evaluator at constant
//         level, so any state dependence outside the distinguished set-builder
//         fails the certificate).
//
// DF_U(v) implies `Len(v) = |Range(v)| <= |U|` in every reachable state, so
// `max_len = |U|` is a sound flat-slot capacity keyed to the checked
// `v \in Seq(D)` invariant (if the invariant is violated the checker reports
// it, and the `SequenceLengthExceedsCapacity` encode backstop fails closed on
// any capacity miss).
//
// Completeness of the write census (the soundness linchpin): a separate
// account-every-prime pass records the address of EVERY `Expr::Prime` node
// reachable from Init/Next — descending into every resolvable operator body
// and flattening `ModuleRef` instance references — and the proof is only
// emitted for `v` when every recorded `v'` occurrence is exactly one of the
// writes the structural walker classified. Any prime this analysis cannot
// attribute (or any residual/unresolvable `ModuleRef`) poisons the analysis.

/// Source-level proof that every element of the sequence stored at a state
/// variable is drawn from a finite constant universe: a checked invariant
/// conjunct `v \in Seq(D)` where `D` evaluates to a finite constant set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceUniverseProof {
    pub(crate) var_idx: usize,
    /// Sorted, deduplicated, non-empty element universe.
    pub(crate) universe: Vec<Value>,
    /// Name of the checked invariant the universe was extracted from.
    pub(crate) invariant: Arc<str>,
}

/// Evaluator/flattener callbacks for the duplicate-free sequence analysis.
///
/// The hooks keep the structural analysis testable without a full evaluator;
/// the real driver (run_prepare) wires them to `flatten_property_module_refs`
/// and `try_eval_const_level` (whose dependency tracking rejects any
/// current/next-state read, so a certificate can never silently evaluate
/// against a concrete state).
pub(crate) struct DuplicateFreeSeqProofHooks<'a> {
    /// Resolve an `Expr::ModuleRef` subtree to its fully flattened
    /// (substitution-applied, recursively inlined) body, or `None` when the
    /// flattened result still contains any `ModuleRef` (fail closed). MUST be
    /// memoized per input node address AND keep the returned `Rc`s alive for
    /// the duration of the analysis: both passes rely on getting the SAME
    /// allocation back so `Prime`-node accounting addresses line up.
    #[allow(clippy::type_complexity)]
    pub(crate) flatten_module_ref:
        &'a dyn Fn(&Expr) -> Option<std::rc::Rc<tla_core::span::Spanned<Expr>>>,
    /// Evaluate a state-free expression to a finite set, returning its sorted,
    /// deduplicated elements. `None` on evaluation error, state dependence,
    /// or a non-(finite-)set result.
    pub(crate) eval_const_set: &'a dyn Fn(&Expr) -> Option<Vec<Value>>,
    /// Evaluate `domain_expr` with `name` bound to `arg` (a finite set value)
    /// at constant level — any current/next-state read MUST yield `None` —
    /// returning the elements of the resulting finite set.
    pub(crate) eval_domain_with_set_arg: &'a dyn Fn(&Expr, &str, &Value) -> Option<Vec<Value>>,
}

/// Fresh identifier used for the certificate's set-builder replacement. The
/// name is not legal TLA+ source, so it can never collide with a spec binder.
const DF_CERT_ARG: &str = "__ty_df_cert_arg";

/// `TY_DF_SEQ_DEBUG=1`: trace why the duplicate-free sequence capacity
/// analysis accepts/rejects each candidate (diagnostic only).
fn df_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| matches!(std::env::var("TY_DF_SEQ_DEBUG").as_deref(), Ok("1")))
}

macro_rules! df_debug {
    ($($arg:tt)*) => {
        if df_debug_enabled() {
            eprintln!("[df-seq] {}", format!($($arg)*));
        }
    };
}
/// Largest `|U|` accepted as a proven capacity (slot-width sanity bound).
const DF_MAX_UNIVERSE: usize = 32;
/// Largest base-set size for the exhaustive `F(X)` certificate (2^n evals).
const DF_MAX_BASE_SET: usize = 8;
/// Total-budget cap on certificate sequence enumeration across all subsets.
const DF_MAX_CERT_ELEMENTS: usize = 65_536;
/// Recursion cap for scope resolution / operator expansion.
const DF_MAX_DEPTH: usize = 64;

/// True when the expression (syntactically) contains any `ModuleRef` /
/// `InstanceExpr` node.
pub(crate) fn expr_contains_module_ref(expr: &Expr) -> bool {
    struct Scan;
    impl tla_core::visit::ExprVisitor for Scan {
        type Output = bool;
        fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
            matches!(expr, Expr::ModuleRef(..) | Expr::InstanceExpr(..)).then_some(true)
        }
    }
    tla_core::visit::ExprVisitor::walk_expr(&mut Scan, expr)
}

/// Extract `v \in Seq(D)` universe proofs from a checked invariant body
/// (descending top-level conjunctions only — exactly the forms enforced on
/// every stored state).
pub(crate) fn collect_sequence_universe_proofs(
    expr: &Expr,
    invariant: &str,
    registry: &VarRegistry,
    op_replacements: &OpReplacements,
    eval_const_set: &dyn Fn(&Expr) -> Option<Vec<Value>>,
    out: &mut Vec<SequenceUniverseProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_sequence_universe_proofs(
                &left.node,
                invariant,
                registry,
                op_replacements,
                eval_const_set,
                out,
            );
            collect_sequence_universe_proofs(
                &right.node,
                invariant,
                registry,
                op_replacements,
                eval_const_set,
                out,
            );
        }
        Expr::In(lhs, rhs) => {
            let Some(var_idx) = df_registry_var_idx(&lhs.node, registry) else {
                return;
            };
            let Expr::Apply(op, args) = &rhs.node else {
                return;
            };
            if args.len() != 1 || !is_seq_operator(&op.node, Some(op_replacements)) {
                return;
            }
            let Some(mut universe) = eval_const_set(&args[0].node) else {
                return;
            };
            universe.sort();
            universe.dedup();
            if universe.is_empty() || out.iter().any(|proof| proof.var_idx == var_idx) {
                return;
            }
            out.push(SequenceUniverseProof {
                var_idx,
                universe,
                invariant: Arc::from(invariant),
            });
        }
        _ => {}
    }
}

/// Resolve a bare state-variable reference by NAME against the root registry.
///
/// `StateVar` nodes are also resolved by name (never by their embedded index):
/// flattened INSTANCE bodies can carry nodes minted against another module's
/// variable order, and the substitution machinery guarantees only the NAME is
/// meaningful in the instantiating module.
fn df_registry_var_idx(expr: &Expr, registry: &VarRegistry) -> Option<usize> {
    match expr {
        Expr::StateVar(name, _, _) | Expr::Ident(name, _) => {
            registry.get(name).map(|idx| idx.as_usize())
        }
        _ => None,
    }
}

#[derive(Clone)]
enum DfScopeEntry {
    /// Call-by-name operator parameter / zero-arg LET definition: the argument
    /// expression together with the scope it was captured in.
    Subst { expr: Expr, scope: DfScope },
    /// Quantifier-bound name; `domain` is only present for a plain
    /// single-name binder with an explicit `\in` domain.
    Binder {
        domain: Option<Expr>,
        domain_scope: DfScope,
    },
}

#[derive(Default, Clone)]
struct DfScope {
    entries: BTreeMap<String, Vec<DfScopeEntry>>,
}

impl DfScope {
    fn push(&mut self, name: String, entry: DfScopeEntry) {
        self.entries.entry(name).or_default().push(entry);
    }

    fn pop(&mut self, name: &str) {
        if let Some(stack) = self.entries.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.entries.remove(name);
            }
        }
    }

    fn get(&self, name: &str) -> Option<&DfScopeEntry> {
        self.entries.get(name).and_then(|stack| stack.last())
    }
}

#[derive(Clone)]
struct DfWrite {
    rhs: Expr,
    scope: DfScope,
}

struct DfSeqAnalysis<'a> {
    registry: &'a VarRegistry,
    constants: &'a tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a OpReplacements,
    hooks: &'a DuplicateFreeSeqProofHooks<'a>,
    poison_all: bool,
    init_writes: BTreeMap<usize, Vec<DfWrite>>,
    next_writes: BTreeMap<usize, Vec<DfWrite>>,
    /// Addresses of `Expr::Prime` nodes the structural walker recognized as
    /// the left-hand side of a whole-variable write.
    classified_prime_addrs: BTreeSet<usize>,
    /// Addresses of ALL `Expr::Prime` nodes per target var (accounting pass).
    prime_addrs: BTreeMap<usize, BTreeSet<usize>>,
    /// Operator bodies already scanned by the accounting pass (context-free,
    /// so scanning once per operator is complete).
    scanned_ops: BTreeSet<String>,
}

impl<'a> DfSeqAnalysis<'a> {
    fn expandable_op(&self, name: &str, args_len: usize) -> Option<(String, &'a OperatorDef)> {
        let resolved = resolve_layout_op_name(name, Some(self.op_replacements))?;
        let def = self.op_defs.get(resolved)?.as_ref();
        (def.params.len() == args_len
            && !def.is_recursive
            && def.params.iter().all(|param| param.arity == 0))
        .then(|| (resolved.to_owned(), def))
    }

    fn resolve_var(&self, expr: &Expr, scope: &DfScope, depth: usize) -> Option<usize> {
        if depth == 0 {
            return None;
        }
        match expr {
            Expr::StateVar(name, _, _) => self.registry.get(name).map(|idx| idx.as_usize()),
            Expr::Ident(name, _) => match scope.get(name) {
                Some(DfScopeEntry::Subst { expr, scope }) => {
                    self.resolve_var(expr, scope, depth - 1)
                }
                Some(DfScopeEntry::Binder { .. }) => None,
                None => self.registry.get(name).map(|idx| idx.as_usize()),
            },
            _ => None,
        }
    }

    fn is_identity(&self, expr: &Expr, scope: &DfScope, var_idx: usize) -> bool {
        self.resolve_var(expr, scope, DF_MAX_DEPTH) == Some(var_idx)
    }

    // ------------------------------------------------------------------
    // Pass A: structural write collection
    // ------------------------------------------------------------------

    fn walk_structure(
        &mut self,
        expr: &Expr,
        scope: &mut DfScope,
        primed: bool,
        visiting: &mut BTreeSet<String>,
    ) {
        match expr {
            Expr::And(left, right) | Expr::Or(left, right) => {
                self.walk_structure(&left.node, scope, primed, visiting);
                self.walk_structure(&right.node, scope, primed, visiting);
            }
            Expr::If(_, then_expr, else_expr) => {
                self.walk_structure(&then_expr.node, scope, primed, visiting);
                self.walk_structure(&else_expr.node, scope, primed, visiting);
            }
            Expr::Exists(bounds, body) | Expr::Forall(bounds, body) => {
                let mut pushed: Vec<String> = Vec::new();
                for bound in bounds {
                    let names = tla_core::visit::single_bound_var_names(bound);
                    let plain_domain = if bound.pattern.is_none() && names.len() == 1 {
                        bound.domain.as_ref().map(|domain| domain.node.clone())
                    } else {
                        None
                    };
                    for name in names {
                        self_scope_push_binder(scope, name.clone(), plain_domain.clone());
                        pushed.push(name);
                    }
                }
                self.walk_structure(&body.node, scope, primed, visiting);
                for name in pushed.into_iter().rev() {
                    scope.pop(&name);
                }
            }
            Expr::Let(defs, body) => {
                let mut pushed: Vec<String> = Vec::new();
                for def in defs {
                    let entry = if def.params.is_empty() {
                        DfScopeEntry::Subst {
                            expr: def.body.node.clone(),
                            scope: scope.clone(),
                        }
                    } else {
                        // Parameterized LET definitions shadow their name so
                        // the classifier never resolves it against a module
                        // operator; writes hidden behind them are caught by
                        // the accounting pass.
                        DfScopeEntry::Binder {
                            domain: None,
                            domain_scope: DfScope::default(),
                        }
                    };
                    scope.push(def.name.node.clone(), entry);
                    pushed.push(def.name.node.clone());
                }
                self.walk_structure(&body.node, scope, primed, visiting);
                for name in pushed.into_iter().rev() {
                    scope.pop(&name);
                }
            }
            Expr::Eq(left, right) => {
                if primed {
                    if let Expr::Prime(inner) = &left.node {
                        if let Some(var_idx) = self.resolve_var(&inner.node, scope, DF_MAX_DEPTH) {
                            self.classified_prime_addrs
                                .insert(std::ptr::from_ref(&left.node) as usize);
                            self.next_writes.entry(var_idx).or_default().push(DfWrite {
                                rhs: right.node.clone(),
                                scope: scope.clone(),
                            });
                        }
                    }
                } else if let Some(var_idx) = self.resolve_var(&left.node, scope, DF_MAX_DEPTH) {
                    self.init_writes.entry(var_idx).or_default().push(DfWrite {
                        rhs: right.node.clone(),
                        scope: scope.clone(),
                    });
                }
            }
            Expr::Unchanged(target) => {
                self.mark_unchanged(&target.node, scope, 0);
            }
            Expr::Ident(name, _) | Expr::OpRef(name) => {
                if scope.get(name).is_some() {
                    return;
                }
                if let Some((resolved, def)) = self.expandable_op(name, 0) {
                    if visiting.insert(resolved.clone()) {
                        let mut body_scope = DfScope::default();
                        self.walk_structure(&def.body.node, &mut body_scope, primed, visiting);
                        visiting.remove(&resolved);
                    }
                }
            }
            Expr::Apply(op, args) => {
                let Some(name) = operator_ident_name(&op.node) else {
                    return;
                };
                if scope.get(name).is_some() {
                    return;
                }
                if let Some((resolved, def)) = self.expandable_op(name, args.len()) {
                    if visiting.insert(resolved.clone()) {
                        let mut body_scope = DfScope::default();
                        for (param, arg) in def.params.iter().zip(args) {
                            body_scope.push(
                                param.name.node.clone(),
                                DfScopeEntry::Subst {
                                    expr: arg.node.clone(),
                                    scope: scope.clone(),
                                },
                            );
                        }
                        self.walk_structure(&def.body.node, &mut body_scope, primed, visiting);
                        visiting.remove(&resolved);
                    }
                }
            }
            Expr::ModuleRef(..) => match (self.hooks.flatten_module_ref)(expr) {
                Some(flattened) => {
                    self.walk_structure(&flattened.node, scope, primed, visiting);
                }
                None => {
                    self.poison_all = true;
                }
            },
            // Any other connective cannot introduce a *recognized* write; any
            // prime hidden inside is caught by the accounting pass.
            _ => {}
        }
    }

    fn mark_unchanged(&mut self, expr: &Expr, scope: &DfScope, depth: usize) {
        if depth > DF_MAX_DEPTH {
            self.poison_all = true;
            return;
        }
        match expr {
            Expr::Tuple(elems) => {
                for elem in elems {
                    self.mark_unchanged(&elem.node, scope, depth + 1);
                }
            }
            _ => {
                if self.resolve_var(expr, scope, DF_MAX_DEPTH).is_some() {
                    // UNCHANGED v — identity, trivially DF_U-preserving.
                    return;
                }
                // `UNCHANGED vars` where `vars` is a tuple-of-variables
                // operator: expand and recurse.
                if let Expr::Ident(name, _) = expr {
                    if scope.get(name).is_none() {
                        if let Some((_, def)) = self.expandable_op(name, 0) {
                            self.mark_unchanged(&def.body.node, &DfScope::default(), depth + 1);
                            return;
                        }
                    }
                }
                // Unknown UNCHANGED target: could stutter a sequence variable
                // this analysis then wrongly treats as fully accounted — the
                // target itself is fine (identity), but an unresolvable form
                // means we cannot tell WHICH variables it covers. Fail closed.
                self.poison_all = true;
            }
        }
    }

    // ------------------------------------------------------------------
    // Pass B: account-every-prime scan
    // ------------------------------------------------------------------

    fn account_primes(&mut self, expr: &Expr) {
        let mut scan = DfPrimeScan { analysis: self };
        tla_core::visit::ExprVisitor::walk_expr(&mut scan, expr);
    }

    fn scan_op_body(&mut self, name: &str) {
        let Some(resolved) = resolve_layout_op_name(name, Some(self.op_replacements)) else {
            // Unresolvable replacement chain: cannot know what it hides.
            self.poison_all = true;
            return;
        };
        let Some(def) = self.op_defs.get(resolved) else {
            // Built-in operator / constant / bound name: never contains a
            // prime of a root state variable.
            return;
        };
        if self.scanned_ops.insert(resolved.to_owned()) {
            let def = Arc::clone(def);
            self.account_primes(&def.body.node);
        }
    }

    // ------------------------------------------------------------------
    // Write classification (DF_U preservation)
    // ------------------------------------------------------------------

    fn classify_init_write(&self, rhs: &Expr, scope: &DfScope, depth: usize) -> bool {
        if depth == 0 {
            return false;
        }
        match rhs {
            Expr::Tuple(elems) if elems.is_empty() => true,
            Expr::If(_, then_expr, else_expr) => {
                self.classify_init_write(&then_expr.node, scope, depth - 1)
                    && self.classify_init_write(&else_expr.node, scope, depth - 1)
            }
            Expr::Ident(name, _) => match scope.get(name) {
                Some(DfScopeEntry::Subst { expr, scope }) => {
                    let scope = scope.clone();
                    self.classify_init_write(&expr.clone(), &scope, depth - 1)
                }
                Some(DfScopeEntry::Binder { .. }) => false,
                None => match self.expandable_op(name, 0) {
                    Some((_, def)) => {
                        self.classify_init_write(&def.body.node, &DfScope::default(), depth - 1)
                    }
                    None => false,
                },
            },
            _ => false,
        }
    }

    fn classify_next_write(
        &self,
        rhs: &Expr,
        scope: &DfScope,
        var_idx: usize,
        universe: &[Value],
        depth: usize,
    ) -> bool {
        if depth == 0 {
            return false;
        }
        if self.is_identity(rhs, scope, var_idx) {
            return true;
        }
        match rhs {
            Expr::Tuple(elems) if elems.is_empty() => true,
            Expr::If(_, then_expr, else_expr) => {
                self.classify_next_write(&then_expr.node, scope, var_idx, universe, depth - 1)
                    && self.classify_next_write(
                        &else_expr.node,
                        scope,
                        var_idx,
                        universe,
                        depth - 1,
                    )
            }
            Expr::Ident(name, _) => match scope.get(name) {
                Some(DfScopeEntry::Subst { expr, scope }) => {
                    let expr = expr.clone();
                    let scope = scope.clone();
                    self.classify_next_write(&expr, &scope, var_idx, universe, depth - 1)
                }
                Some(DfScopeEntry::Binder { .. }) => false,
                None => match self.expandable_op(name, 0) {
                    Some((_, def)) => self.classify_next_write(
                        &def.body.node,
                        &DfScope::default(),
                        var_idx,
                        universe,
                        depth - 1,
                    ),
                    None => false,
                },
            },
            Expr::Apply(op, args) => {
                let Some(name) = operator_ident_name(&op.node) else {
                    return false;
                };
                if scope.get(name).is_some() {
                    return false;
                }
                let Some(resolved) = resolve_layout_op_name(name, Some(self.op_replacements))
                else {
                    return false;
                };
                match (resolved, args.as_slice()) {
                    ("Tail", [base]) => self.is_identity(&base.node, scope, var_idx),
                    ("SubSeq", [base, _, _]) => self.is_identity(&base.node, scope, var_idx),
                    ("\\o" | "\\circ", [left, right]) => {
                        self.classify_concat(&left.node, &right.node, scope, var_idx, universe)
                    }
                    _ => match self.expandable_op(name, args.len()) {
                        Some((_, def)) => {
                            let mut body_scope = DfScope::default();
                            for (param, arg) in def.params.iter().zip(args) {
                                body_scope.push(
                                    param.name.node.clone(),
                                    DfScopeEntry::Subst {
                                        expr: arg.node.clone(),
                                        scope: scope.clone(),
                                    },
                                );
                            }
                            self.classify_next_write(
                                &def.body.node,
                                &body_scope,
                                var_idx,
                                universe,
                                depth - 1,
                            )
                        }
                        None => false,
                    },
                }
            }
            _ => false,
        }
    }

    fn classify_concat(
        &self,
        left: &Expr,
        right: &Expr,
        scope: &DfScope,
        var_idx: usize,
        universe: &[Value],
    ) -> bool {
        if self.is_drop_idiom(left, right, scope, var_idx) {
            return true;
        }
        // Disjoint append: `v \o q` with `q` existentially bound over a
        // certified disjoint-range domain.
        if !self.is_identity(left, scope, var_idx) {
            return false;
        }
        let Expr::Ident(qname, _) = right else {
            return false;
        };
        let Some(DfScopeEntry::Binder {
            domain: Some(domain),
            domain_scope,
        }) = scope.get(qname)
        else {
            return false;
        };
        self.check_disjoint_append_domain(domain, domain_scope, var_idx, universe)
    }

    /// `SubSeq(v, 1, i-1) \o SubSeq(v, i+1, Len(v))` — the element-removal
    /// idiom. Both windows are index-disjoint for every value of `i`, so the
    /// result is a sub-multiset of the (duplicate-free) `v`.
    fn is_drop_idiom(&self, left: &Expr, right: &Expr, scope: &DfScope, var_idx: usize) -> bool {
        let Some((lbase, llo, lhi)) = self.subseq_parts(left, scope) else {
            return false;
        };
        let Some((rbase, rlo, rhi)) = self.subseq_parts(right, scope) else {
            return false;
        };
        if !self.is_identity(lbase, scope, var_idx) || !self.is_identity(rbase, scope, var_idx) {
            return false;
        }
        // Left window starts at literal 1.
        if const_usize_expr(llo, self.constants, Some(self.op_replacements)) != Some(1) {
            return false;
        }
        // Left end `i - 1`, right start `i + 1`, with the SAME index expr `i`
        // (compared as identical simple forms: same bound name or same
        // constant).
        let Expr::Sub(l_idx, l_one) = lhi else {
            return false;
        };
        let Expr::Add(r_idx, r_one) = rlo else {
            return false;
        };
        if const_usize_expr(&l_one.node, self.constants, Some(self.op_replacements)) != Some(1)
            || const_usize_expr(&r_one.node, self.constants, Some(self.op_replacements)) != Some(1)
        {
            return false;
        }
        if !df_simple_exprs_match(&l_idx.node, &r_idx.node) {
            return false;
        }
        // Right end `Len(v)`.
        let Expr::Apply(len_op, len_args) = rhi else {
            return false;
        };
        if len_args.len() != 1 || !is_seq_len_operator(&len_op.node, Some(self.op_replacements)) {
            return false;
        }
        self.is_identity(&len_args[0].node, scope, var_idx)
    }

    fn subseq_parts<'e>(
        &self,
        expr: &'e Expr,
        scope: &DfScope,
    ) -> Option<(&'e Expr, &'e Expr, &'e Expr)> {
        let Expr::Apply(op, args) = expr else {
            return None;
        };
        let name = operator_ident_name(&op.node)?;
        if scope.get(name).is_some() {
            return None;
        }
        if resolve_layout_op_name(name, Some(self.op_replacements)) != Some("SubSeq") {
            return None;
        }
        let [base, lo, hi] = args.as_slice() else {
            return None;
        };
        Some((&base.node, &lo.node, &hi.node))
    }

    /// Certify the existential domain of a disjoint append (see module docs).
    fn check_disjoint_append_domain(
        &self,
        domain: &Expr,
        domain_scope: &DfScope,
        var_idx: usize,
        universe: &[Value],
    ) -> bool {
        // 1. Find the distinguished set-builder `{x \in B : … /\ x \notin
        //    Range(v)}` and evaluate its constant base `B ⊆ U`.
        let mut finder = DfSetFilterFinder {
            analysis: self,
            scope: domain_scope,
            var_idx,
            universe,
            found: None,
        };
        tla_core::visit::ExprVisitor::walk_expr(&mut finder, domain);
        let Some((set_filter, base_elems)) = finder.found else {
            df_debug!("append domain: no disjoint-range set-builder found");
            return false;
        };
        if base_elems.len() > DF_MAX_BASE_SET {
            df_debug!("append domain: base set too large ({})", base_elems.len());
            return false;
        }
        // 2. Replace every occurrence of the set-builder with a fresh
        //    identifier, producing `F(fresh)`.
        let mut replacer = DfSetFilterReplacer {
            target: &set_filter,
            replaced: 0,
        };
        let replaced_domain = tla_core::ExprFold::fold_expr(
            &mut replacer,
            tla_core::span::Spanned::dummy(domain.clone()),
        );
        if replacer.replaced == 0 {
            return false;
        }
        // 3. Capture check on the REPLACED domain (the set-builder's own
        //    binders are gone): no remaining binder may shadow a name the
        //    set-builder reads — otherwise a replaced occurrence would have
        //    denoted a different set than the outer one. Free names are the
        //    EXACT (binder-aware) free variables plus every state-variable
        //    name mentioned (belt-and-braces for `StateVar` nodes).
        let filter_names = df_filter_free_names(&set_filter);
        let domain_binders = df_collect_binder_names(&replaced_domain.node);
        if !filter_names.is_disjoint(&domain_binders) {
            df_debug!(
                "append domain: capture risk — filter reads {:?}, domain binds {:?}",
                filter_names,
                domain_binders
            );
            return false;
        }
        // 3b. Static state-freedom: after the replacement, the remaining
        //     function `F` must not read any state variable (directly, primed,
        //     or through any reachable operator body). Belt-and-braces with the
        //     evaluation hook, which runs against a state-cleared context so a
        //     missed read errors out instead of silently sampling a state.
        if self.expr_reads_state(&replaced_domain.node, &mut BTreeMap::new()) {
            df_debug!("append domain: replaced domain still reads state");
            return false;
        }
        // 4. Exhaustive certificate: for EVERY subset X ⊆ B, every member of
        //    F(X) must be a duplicate-free sequence over X. The evaluation
        //    hook runs at constant level, so any residual state dependence in
        //    F fails the certificate here.
        let n = base_elems.len();
        let mut budget = DF_MAX_CERT_ELEMENTS;
        for mask in 0u32..(1u32 << n) {
            let subset: Vec<Value> = base_elems
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, value)| value.clone())
                .collect();
            let subset_value = Value::Set(Rp::new(tla_value::value::SortedSet::from_sorted_vec(
                subset.clone(),
            )));
            let Some(members) = (self.hooks.eval_domain_with_set_arg)(
                &replaced_domain.node,
                DF_CERT_ARG,
                &subset_value,
            ) else {
                df_debug!("append domain: certificate evaluation failed for subset {subset_value}");
                return false;
            };
            if members.len() > budget {
                df_debug!("append domain: certificate budget exhausted");
                return false;
            }
            budget -= members.len();
            for member in &members {
                if !df_value_is_duplicate_free_seq_over(member, &subset) {
                    df_debug!(
                        "append domain: member {member} not a duplicate-free sequence over {subset_value}"
                    );
                    return false;
                }
            }
        }
        true
    }

    /// Transitive, conservative "reads any state variable" scan: `StateVar` /
    /// `Prime` / residual instance nodes anywhere, a free identifier naming a
    /// registry variable, or any reachable operator body doing the same.
    /// Over-approximates (binder shadowing of a variable name still counts).
    fn expr_reads_state(&self, expr: &Expr, memo: &mut BTreeMap<String, bool>) -> bool {
        struct Scan<'x, 'a> {
            analysis: &'x DfSeqAnalysis<'a>,
            memo: &'x mut BTreeMap<String, bool>,
        }
        impl Scan<'_, '_> {
            fn op_reads_state(&mut self, name: &str) -> bool {
                let Some(resolved) =
                    resolve_layout_op_name(name, Some(self.analysis.op_replacements))
                else {
                    return true;
                };
                if self.analysis.registry.get(resolved).is_some() {
                    return true;
                }
                let Some(def) = self.analysis.op_defs.get(resolved) else {
                    // Built-in / constant / bound name: no state access.
                    return false;
                };
                if let Some(&cached) = self.memo.get(resolved) {
                    return cached;
                }
                // Cycle-conservative: while computing, assume it reads state.
                self.memo.insert(resolved.to_owned(), true);
                let def = Arc::clone(def);
                let result = tla_core::visit::ExprVisitor::walk_expr(self, &def.body.node);
                self.memo.insert(resolved.to_owned(), result);
                result
            }
        }
        impl tla_core::visit::ExprVisitor for Scan<'_, '_> {
            type Output = bool;
            fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
                match expr {
                    Expr::StateVar(..)
                    | Expr::Prime(_)
                    | Expr::ModuleRef(..)
                    | Expr::InstanceExpr(..) => Some(true),
                    Expr::Ident(name, _) | Expr::OpRef(name) => {
                        let name = name.clone();
                        Some(self.op_reads_state(&name))
                    }
                    _ => None,
                }
            }
            fn visit_apply(
                &mut self,
                op_expr: &tla_core::span::Spanned<Expr>,
                args: &[tla_core::span::Spanned<Expr>],
            ) -> Option<bool> {
                let mut result = match operator_ident_name(&op_expr.node) {
                    Some(name) => {
                        let name = name.to_owned();
                        self.op_reads_state(&name)
                    }
                    None => self.walk_expr(&op_expr.node),
                };
                for arg in args {
                    if result {
                        break;
                    }
                    result = result || self.walk_expr(&arg.node);
                }
                Some(result)
            }
        }
        let mut scan = Scan {
            analysis: self,
            memo,
        };
        tla_core::visit::ExprVisitor::walk_expr(&mut scan, expr)
    }

    /// Match `{x \in B : … /\ x \notin {v[y] : y \in DOMAIN v} /\ …}` and
    /// return the evaluated, sorted elements of `B` when `B ⊆ U`.
    fn match_disjoint_range_filter(
        &self,
        bound: &BoundVar,
        pred: &Expr,
        scope: &DfScope,
        var_idx: usize,
        universe: &[Value],
    ) -> Option<Vec<Value>> {
        if bound.pattern.is_some() {
            return None;
        }
        let bound_name = bound.name.node.as_str();
        let base = bound.domain.as_ref()?;
        let base_elems = (self.hooks.eval_const_set)(&base.node)?;
        if !base_elems
            .iter()
            .all(|elem| universe.binary_search(elem).is_ok())
        {
            return None;
        }
        let mut conjuncts: Vec<&Expr> = Vec::new();
        df_flatten_conjuncts(pred, &mut conjuncts, DF_MAX_DEPTH);
        conjuncts
            .iter()
            .any(|conjunct| {
                let Expr::NotIn(lhs, rhs) = conjunct else {
                    return false;
                };
                matches!(&lhs.node, Expr::Ident(name, _) if name == bound_name)
                    && self.is_range_of_var(&rhs.node, scope, var_idx)
            })
            .then_some(base_elems)
    }

    /// `{v[y] : y \in DOMAIN v}` — the inline Range-of-sequence form.
    fn is_range_of_var(&self, expr: &Expr, scope: &DfScope, var_idx: usize) -> bool {
        let Expr::SetBuilder(body, bounds) = expr else {
            return false;
        };
        let [bound] = bounds.as_slice() else {
            return false;
        };
        if bound.pattern.is_some() {
            return false;
        }
        let Some(domain) = bound.domain.as_ref() else {
            return false;
        };
        let Expr::Domain(domain_base) = &domain.node else {
            return false;
        };
        if !self.is_identity(&domain_base.node, scope, var_idx) {
            return false;
        }
        let Expr::FuncApply(func, arg) = &body.node else {
            return false;
        };
        self.is_identity(&func.node, scope, var_idx)
            && matches!(&arg.node, Expr::Ident(name, _) if name == &bound.name.node)
    }
}

fn self_scope_push_binder(scope: &mut DfScope, name: String, domain: Option<Expr>) {
    let domain_scope = scope.clone();
    scope.push(
        name,
        DfScopeEntry::Binder {
            domain,
            domain_scope,
        },
    );
}

/// Limited structural identity for the drop-idiom index expression: the same
/// bound identifier or the same integer literal (span-insensitive).
fn df_simple_exprs_match(left: &Expr, right: &Expr) -> bool {
    match (left, right) {
        (Expr::Ident(a, _), Expr::Ident(b, _)) => a == b,
        (Expr::Int(a), Expr::Int(b)) => a == b,
        _ => false,
    }
}

fn df_flatten_conjuncts<'e>(expr: &'e Expr, out: &mut Vec<&'e Expr>, depth: usize) {
    if depth == 0 {
        return;
    }
    if let Expr::And(left, right) = expr {
        df_flatten_conjuncts(&left.node, out, depth - 1);
        df_flatten_conjuncts(&right.node, out, depth - 1);
    } else {
        out.push(expr);
    }
}

/// A duplicate-free sequence whose elements all belong to `allowed` (sorted).
fn df_value_is_duplicate_free_seq_over(value: &Value, allowed: &[Value]) -> bool {
    let items: Vec<&Value> = match value {
        Value::Seq(seq) => seq.iter().collect(),
        Value::Tuple(elems) => elems.iter().collect(),
        _ => return false,
    };
    let mut seen: BTreeSet<&Value> = BTreeSet::new();
    for item in items {
        if allowed.binary_search(item).is_err() || !seen.insert(item) {
            return false;
        }
    }
    true
}

/// Names a set-builder occurrence READS from its enclosing scope: the exact
/// (binder-aware) free variables, plus every state-variable name mentioned
/// anywhere in it (a `StateVar` node resolves by variable identity, but the
/// belt-and-braces union keeps the capture check conservative).
fn df_filter_free_names(expr: &Expr) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = tla_core::free_vars(expr).into_iter().collect();
    struct StateVarScan {
        names: BTreeSet<String>,
    }
    impl tla_core::visit::ExprVisitor for StateVarScan {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            if let Expr::StateVar(name, _, _) = expr {
                self.names.insert(name.clone());
            }
            None
        }
    }
    let mut scan = StateVarScan {
        names: BTreeSet::new(),
    };
    tla_core::visit::ExprVisitor::walk_expr(&mut scan, expr);
    names.extend(scan.names);
    names
}

/// Every name bound by any binder construct anywhere in the expression
/// (quantifiers, CHOOSE, set builders/filters, function defs, LET, lambdas).
fn df_collect_binder_names(expr: &Expr) -> BTreeSet<String> {
    struct Scan {
        names: BTreeSet<String>,
    }
    impl Scan {
        fn add_bounds(&mut self, bounds: &[BoundVar]) {
            for bound in bounds {
                self.names
                    .extend(tla_core::visit::single_bound_var_names(bound));
            }
        }
    }
    impl tla_core::visit::ExprVisitor for Scan {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            match expr {
                Expr::Forall(bounds, _) | Expr::Exists(bounds, _) => self.add_bounds(bounds),
                Expr::SetBuilder(_, bounds) | Expr::FuncDef(bounds, _) => self.add_bounds(bounds),
                Expr::SetFilter(bound, _) | Expr::Choose(bound, _) => {
                    self.add_bounds(std::slice::from_ref(bound));
                }
                Expr::Lambda(params, _) => {
                    self.names.extend(params.iter().map(|p| p.node.clone()));
                }
                Expr::Let(defs, _) => {
                    for def in defs {
                        self.names.insert(def.name.node.clone());
                        self.names
                            .extend(def.params.iter().map(|p| p.name.node.clone()));
                    }
                }
                _ => {}
            }
            None
        }
    }
    let mut scan = Scan {
        names: BTreeSet::new(),
    };
    tla_core::visit::ExprVisitor::walk_expr(&mut scan, expr);
    scan.names
}

struct DfSetFilterFinder<'x, 'a> {
    analysis: &'x DfSeqAnalysis<'a>,
    scope: &'x DfScope,
    var_idx: usize,
    universe: &'x [Value],
    found: Option<(Expr, Vec<Value>)>,
}

impl tla_core::visit::ExprVisitor for DfSetFilterFinder<'_, '_> {
    type Output = bool;
    fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
        if self.found.is_some() {
            return Some(true);
        }
        if let Expr::SetFilter(bound, pred) = expr {
            if let Some(base_elems) = self.analysis.match_disjoint_range_filter(
                bound,
                &pred.node,
                self.scope,
                self.var_idx,
                self.universe,
            ) {
                self.found = Some((expr.clone(), base_elems));
                return Some(true);
            }
        }
        None
    }
}

struct DfSetFilterReplacer<'x> {
    target: &'x Expr,
    replaced: usize,
}

impl tla_core::ExprFold for DfSetFilterReplacer<'_> {
    fn fold_expr(&mut self, expr: tla_core::span::Spanned<Expr>) -> tla_core::span::Spanned<Expr> {
        if &expr.node == self.target {
            self.replaced += 1;
            return tla_core::span::Spanned::dummy(Expr::Ident(
                DF_CERT_ARG.to_string(),
                intern_name(DF_CERT_ARG),
            ));
        }
        let span = expr.span;
        let node = self.fold_expr_inner(expr.node);
        tla_core::span::Spanned { node, span }
    }
}

struct DfPrimeScan<'x, 'a> {
    analysis: &'x mut DfSeqAnalysis<'a>,
}

impl tla_core::visit::ExprVisitor for DfPrimeScan<'_, '_> {
    type Output = ();

    fn visit_node(&mut self, expr: &Expr) -> Option<()> {
        match expr {
            Expr::Prime(inner) => {
                match df_registry_var_idx(&inner.node, self.analysis.registry) {
                    Some(var_idx) => {
                        self.analysis
                            .prime_addrs
                            .entry(var_idx)
                            .or_default()
                            .insert(std::ptr::from_ref(expr) as usize);
                    }
                    // A prime whose target this analysis cannot attribute to
                    // a root variable: fail closed for everything.
                    None => self.analysis.poison_all = true,
                }
                Some(())
            }
            Expr::ModuleRef(..) => {
                match (self.analysis.hooks.flatten_module_ref)(expr) {
                    Some(flattened) => {
                        self.walk_expr(&flattened.node);
                    }
                    None => self.analysis.poison_all = true,
                }
                Some(())
            }
            Expr::InstanceExpr(..) => {
                self.analysis.poison_all = true;
                Some(())
            }
            Expr::Ident(name, _) | Expr::OpRef(name) => {
                let name = name.clone();
                self.analysis.scan_op_body(&name);
                Some(())
            }
            _ => None,
        }
    }

    fn visit_apply(
        &mut self,
        op_expr: &tla_core::span::Spanned<Expr>,
        args: &[tla_core::span::Spanned<Expr>],
    ) -> Option<()> {
        match operator_ident_name(&op_expr.node) {
            Some(name) => {
                let name = name.to_owned();
                self.analysis.scan_op_body(&name);
            }
            None => {
                self.walk_expr(&op_expr.node);
            }
        }
        for arg in args {
            self.walk_expr(&arg.node);
        }
        Some(())
    }
}

/// Entry point: emit duplicate-free bounded-universe sequence capacity proofs
/// (see the section docs above for the soundness argument).
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_duplicate_free_sequence_capacity_proofs(
    init_expr: &Expr,
    next_expr: &Expr,
    registry: &VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &OpReplacements,
    universe_proofs: &[SequenceUniverseProof],
    hooks: &DuplicateFreeSeqProofHooks,
    out: &mut Vec<SequenceCapacityProof>,
) {
    if universe_proofs.is_empty() {
        return;
    }
    df_debug!(
        "analyzing {} universe proof(s): {:?}",
        universe_proofs.len(),
        universe_proofs
            .iter()
            .map(|p| (p.var_idx, p.universe.len(), p.invariant.as_ref()))
            .collect::<Vec<_>>()
    );
    let mut analysis = DfSeqAnalysis {
        registry,
        constants,
        op_defs,
        op_replacements,
        hooks,
        poison_all: false,
        init_writes: BTreeMap::new(),
        next_writes: BTreeMap::new(),
        classified_prime_addrs: BTreeSet::new(),
        prime_addrs: BTreeMap::new(),
        scanned_ops: BTreeSet::new(),
    };
    analysis.walk_structure(
        init_expr,
        &mut DfScope::default(),
        false,
        &mut BTreeSet::new(),
    );
    analysis.walk_structure(
        next_expr,
        &mut DfScope::default(),
        true,
        &mut BTreeSet::new(),
    );
    analysis.account_primes(init_expr);
    analysis.account_primes(next_expr);
    if analysis.poison_all {
        df_debug!("poisoned: unattributable prime or unresolvable ModuleRef/InstanceExpr");
        return;
    }

    for universe_proof in universe_proofs {
        let var_idx = universe_proof.var_idx;
        let max_len = universe_proof.universe.len();
        if max_len == 0 || max_len > DF_MAX_UNIVERSE {
            df_debug!("var {var_idx}: universe size {max_len} out of range");
            continue;
        }
        // Accounting: every primed occurrence of `v` must be a classified
        // whole-variable write.
        if analysis
            .prime_addrs
            .get(&var_idx)
            .is_some_and(|addrs| !addrs.is_subset(&analysis.classified_prime_addrs))
        {
            df_debug!(
                "var {var_idx}: unaccounted prime occurrence(s): {} primes, {} classified total",
                analysis.prime_addrs.get(&var_idx).map_or(0, BTreeSet::len),
                analysis.classified_prime_addrs.len(),
            );
            continue;
        }
        // Base case: at least one recognized Init write, all empty.
        let Some(init_writes) = analysis.init_writes.get(&var_idx) else {
            df_debug!("var {var_idx}: no recognized Init write");
            continue;
        };
        if init_writes.is_empty()
            || !init_writes
                .iter()
                .all(|write| analysis.classify_init_write(&write.rhs, &write.scope, DF_MAX_DEPTH))
        {
            df_debug!(
                "var {var_idx}: Init write not provably empty ({} writes)",
                init_writes.len()
            );
            continue;
        }
        // Step case: every recognized Next write preserves DF_U.
        let empty_writes: Vec<DfWrite> = Vec::new();
        let next_writes = analysis.next_writes.get(&var_idx).unwrap_or(&empty_writes);
        if !next_writes.iter().all(|write| {
            let ok = analysis.classify_next_write(
                &write.rhs,
                &write.scope,
                var_idx,
                &universe_proof.universe,
                DF_MAX_DEPTH,
            );
            if !ok {
                df_debug!("var {var_idx}: unclassified Next write: {:?}", write.rhs);
            }
            ok
        }) {
            continue;
        }
        df_debug!(
            "var {var_idx}: PROVEN duplicate-free over |U|={max_len} (invariant {})",
            universe_proof.invariant
        );
        // A degenerate zero-capacity claim for the same slot (emitted when the
        // plain writer fixpoint saw no growth writes) is strictly weaker than
        // this proof and would otherwise trip the fail-closed uniqueness check
        // in `unique_sequence_proof`; drop it. Defer to any OTHER pre-existing
        // (non-degenerate) proof.
        out.retain(|proof| {
            !(proof.var_idx == var_idx && proof.path.is_empty() && proof.max_len == 0)
        });
        if out
            .iter()
            .any(|proof| proof.var_idx == var_idx && proof.path.is_empty())
        {
            continue;
        }
        push_sequence_capacity_proof(
            out,
            SequenceCapacityProof {
                var_idx,
                path: Vec::new(),
                max_len,
                invariant: Arc::clone(&universe_proof.invariant),
                heuristic: false,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::var_index::VarRegistry;
    use std::sync::Arc;
    use tla_value::value::{FuncValue, IntIntervalFunc, RecordValue, SortedSet};

    /// WP-15 test env guard: set/unset `TY_FLAT_WRITE_ADMIT` while holding the
    /// process-wide env lock (`crate::process_env_lock`), restoring the
    /// previous value on drop so no other test observes the mutation.
    pub(crate) struct FlatWriteAdmitEnvGuard {
        previous: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl FlatWriteAdmitEnvGuard {
        pub(crate) fn set(value: Option<&str>) -> Self {
            let lock = crate::process_env_lock();
            let previous = std::env::var_os("TY_FLAT_WRITE_ADMIT");
            match value {
                Some(value) => crate::env_guard::set_var("TY_FLAT_WRITE_ADMIT", value),
                None => crate::env_guard::remove_var("TY_FLAT_WRITE_ADMIT"),
            }
            Self {
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for FlatWriteAdmitEnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => crate::env_guard::set_var("TY_FLAT_WRITE_ADMIT", value),
                None => crate::env_guard::remove_var("TY_FLAT_WRITE_ADMIT"),
            }
        }
    }

    fn wp15_seq(values: Vec<Value>) -> Value {
        Value::Seq(Rp::new(tla_value::value::SeqValue::from_vec(values)))
    }

    fn wp15_int_seq_state() -> (VarRegistry, ArrayState) {
        let registry = VarRegistry::from_names(["nxt"]);
        let state =
            ArrayState::from_values(vec![wp15_seq(vec![Value::SmallInt(1), Value::SmallInt(1)])]);
        (registry, state)
    }

    fn wp15_fixed_domain_proof(invariant: &str) -> SequenceFixedDomainTypeProof {
        SequenceFixedDomainTypeProof {
            var_idx: 0,
            path: Vec::new(),
            domain: Arc::from(vec![Value::SmallInt(1), Value::SmallInt(2)].into_boxed_slice()),
            element_layout: SequenceTypeLayoutProof::Flat(FlatValueLayout::Scalar(SlotType::Int)),
            invariant: Arc::from(invariant),
        }
    }

    fn wp15_infer_with_proofs(
        registry: &VarRegistry,
        state: &ArrayState,
        capacity_proofs: &[SequenceCapacityProof],
        fixed_domain_proofs: &[SequenceFixedDomainTypeProof],
    ) -> StateLayout {
        infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
            state,
            registry,
            capacity_proofs,
            &[],
            fixed_domain_proofs,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
    }

    /// WP-15 (MCBakery `nxt`): a spec checking BOTH `TypeOK` and
    /// `Inv == TypeOK /\ IInv` collects every `v \in [1..N -> T]` clause twice,
    /// once per invariant. The two fixed-domain proofs are identical except for
    /// the proving-invariant label; historically the label-sensitive uniqueness
    /// check treated them as ambiguous and fell back to weaker evidence, so the
    /// var was never flat-primary admissible. Under `TY_FLAT_WRITE_ADMIT=1` the
    /// duplicates are judged structurally and the var admits; with the gate OFF
    /// the historical fail-closed veto is preserved exactly.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_wp15_duplicate_fixed_domain_proofs_fail_closed_default_admit_under_gate() {
        let (registry, state) = wp15_int_seq_state();
        let proofs = [
            wp15_fixed_domain_proof("TypeOK"),
            wp15_fixed_domain_proof("Inv"),
        ];

        {
            let _guard = FlatWriteAdmitEnvGuard::set(None);
            let layout = wp15_infer_with_proofs(&registry, &state, &[], &proofs);
            match &layout.var_layout(0).unwrap().kind {
                VarLayoutKind::Recursive {
                    layout: FlatValueLayout::Sequence { bound, max_len, .. },
                } => {
                    assert_eq!(*max_len, 2);
                    assert!(
                        matches!(bound, SequenceBoundEvidence::Observed),
                        "gate OFF: label-only duplicate fixed-domain proofs must stay \
                         ambiguous (historical veto), got {bound:?}"
                    );
                }
                other => panic!("expected sequence layout, got {other:?}"),
            }
            assert!(
                !layout.supports_flat_primary(),
                "gate OFF: the var must NOT be flat-primary admissible"
            );
        }

        {
            let _guard = FlatWriteAdmitEnvGuard::set(Some("1"));
            let layout = wp15_infer_with_proofs(&registry, &state, &[], &proofs);
            match &layout.var_layout(0).unwrap().kind {
                VarLayoutKind::Recursive {
                    layout:
                        FlatValueLayout::Sequence {
                            bound,
                            max_len,
                            element_layout,
                        },
                } => {
                    assert_eq!(*max_len, 2);
                    assert!(
                        matches!(bound, SequenceBoundEvidence::FixedDomainTypeLayout { .. }),
                        "gate ON: structurally-identical duplicates must resolve to the \
                         fixed-domain type layout, got {bound:?}"
                    );
                    assert_eq!(
                        element_layout.as_ref(),
                        &FlatValueLayout::Scalar(SlotType::Int)
                    );
                }
                other => panic!("expected sequence layout, got {other:?}"),
            }
            assert!(
                layout.supports_flat_primary(),
                "gate ON: the proven fixed-domain sequence must be flat-primary admissible"
            );
        }
    }

    /// WP-15 fail-closed pin: fixed-domain proofs that GENUINELY disagree on
    /// the element layout must stay ambiguous even under the opt-in gate — the
    /// structural rule only tolerates label-only duplicates, never a real
    /// encoding conflict.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_wp15_conflicting_fixed_domain_proofs_stay_fail_closed_under_gate() {
        let (registry, state) = wp15_int_seq_state();
        let mut conflicting = wp15_fixed_domain_proof("Inv");
        conflicting.element_layout =
            SequenceTypeLayoutProof::Flat(FlatValueLayout::Scalar(SlotType::Bool));
        let proofs = [wp15_fixed_domain_proof("TypeOK"), conflicting];

        let _guard = FlatWriteAdmitEnvGuard::set(Some("1"));
        let layout = wp15_infer_with_proofs(&registry, &state, &[], &proofs);
        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence { bound, .. },
            } => {
                assert!(
                    matches!(bound, SequenceBoundEvidence::Observed),
                    "gate ON: structurally-conflicting proofs must fail closed, got {bound:?}"
                );
            }
            other => panic!("expected sequence layout, got {other:?}"),
        }
        assert!(!layout.supports_flat_primary());
    }

    /// WP-15 capacity analogue: duplicate capacity proofs agreeing on
    /// `max_len` resolve under the gate (`ProvenInvariant`, the wider proven
    /// capacity); duplicates disagreeing on `max_len` stay fail-closed
    /// (`Observed`, the sampled length). Gate OFF preserves the historical
    /// label-sensitive veto for both.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_wp15_duplicate_capacity_proofs_structural_uniqueness_under_gate() {
        let (registry, state) = wp15_int_seq_state();
        let capacity = |max_len: usize, invariant: &str| SequenceCapacityProof {
            var_idx: 0,
            path: Vec::new(),
            max_len,
            invariant: Arc::from(invariant),
            heuristic: false,
        };

        let agreeing = [capacity(5, "TypeOK"), capacity(5, "Inv")];
        let disagreeing = [capacity(5, "TypeOK"), capacity(6, "Inv")];

        let bound_of = |layout: &StateLayout| match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence { bound, max_len, .. },
            } => (bound.clone(), *max_len),
            other => panic!("expected sequence layout, got {other:?}"),
        };

        {
            let _guard = FlatWriteAdmitEnvGuard::set(None);
            let (bound, max_len) =
                bound_of(&wp15_infer_with_proofs(&registry, &state, &agreeing, &[]));
            assert!(
                matches!(bound, SequenceBoundEvidence::Observed) && max_len == 2,
                "gate OFF: label-only duplicate capacity proofs stay ambiguous, got \
                 {bound:?} max_len={max_len}"
            );
        }
        {
            let _guard = FlatWriteAdmitEnvGuard::set(Some("1"));
            let (bound, max_len) =
                bound_of(&wp15_infer_with_proofs(&registry, &state, &agreeing, &[]));
            assert!(
                matches!(bound, SequenceBoundEvidence::ProvenInvariant { .. }) && max_len == 5,
                "gate ON: agreeing duplicate capacity proofs must resolve, got \
                 {bound:?} max_len={max_len}"
            );

            let (bound, max_len) = bound_of(&wp15_infer_with_proofs(
                &registry,
                &state,
                &disagreeing,
                &[],
            ));
            assert!(
                matches!(bound, SequenceBoundEvidence::Observed) && max_len == 2,
                "gate ON: capacity proofs disagreeing on max_len must fail closed, got \
                 {bound:?} max_len={max_len}"
            );
        }
    }

    /// WP-15 (`TY_FLAT_WRITE_ADMIT=1`, SubSeqNativeFold `s` / AllocateTest
    /// `sched` class): a sequence whose only Next write is a `SubSeq`
    /// sub-window of itself earns a writer capacity proof (a sub-window can
    /// never exceed the base capacity) and a writer element proof (a
    /// sub-window introduces no new elements). Gate OFF preserves the
    /// historical fail-closed behavior: `SubSeq` writes stay unclassified and
    /// no proof is emitted.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_wp15_subseq_writer_capacity_and_element_proofs_under_gate() {
        let registry = VarRegistry::from_names(["s"]);
        let seed_values = vec![wp15_seq(vec![
            Value::SmallInt(10),
            Value::SmallInt(20),
            Value::SmallInt(30),
            Value::SmallInt(40),
        ])];
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();

        // Init: s = <<10, 20, 30, 40>>; Next: s' = SubSeq(s, 2, Len(s)).
        let init = eq(
            ident("s"),
            Expr::Tuple(vec![
                expr(int_lit(10)),
                expr(int_lit(20)),
                expr(int_lit(30)),
                expr(int_lit(40)),
            ]),
        );
        let next = eq(
            prime(ident("s")),
            Expr::Apply(
                boxed(ident("SubSeq")),
                vec![
                    expr(ident("s")),
                    expr(int_lit(2)),
                    expr(Expr::Apply(boxed(ident("Len")), vec![expr(ident("s"))])),
                ],
            ),
        );

        let collect_capacity = || {
            let mut out = Vec::new();
            collect_sequence_capacity_writer_proofs_with_ops(
                &init,
                &next,
                "Init/Next sequence writer proof",
                &registry,
                &seed_values,
                &constants,
                &op_defs,
                &op_replacements,
                &mut out,
            );
            out
        };
        let collect_elements = || {
            let mut out = Vec::new();
            collect_sequence_element_layout_writer_proofs_with_ops(
                &init,
                &next,
                "Init/Next sequence writer proof",
                &registry,
                &seed_values,
                &constants,
                &op_defs,
                &op_replacements,
                &[],
                &mut out,
            );
            out
        };

        {
            let _guard = FlatWriteAdmitEnvGuard::set(None);
            assert!(
                collect_capacity().is_empty(),
                "gate OFF: a SubSeq write must stay unclassified (no capacity proof)"
            );
            assert!(
                collect_elements().is_empty(),
                "gate OFF: a SubSeq write must stay unclassified (no element proof)"
            );
        }

        let _guard = FlatWriteAdmitEnvGuard::set(Some("1"));
        let capacity = collect_capacity();
        assert_eq!(
            capacity.len(),
            1,
            "gate ON: expected one capacity proof, got {capacity:?}"
        );
        assert_eq!(capacity[0].var_idx, 0);
        assert!(capacity[0].path.is_empty());
        assert_eq!(
            capacity[0].max_len, 4,
            "the SubSeq window is bounded by the base capacity (init length 4)"
        );

        let elements = collect_elements();
        assert_eq!(
            elements.len(),
            1,
            "gate ON: expected one element proof, got {elements:?}"
        );
        assert_eq!(elements[0].var_idx, 0);
        assert_eq!(
            elements[0].element_layout,
            FlatValueLayout::Scalar(SlotType::Int),
            "a SubSeq window's elements are covered by the base element layout"
        );
    }

    /// WP-15 companion pin (AllocateTest `Drop` idiom, inline form): the
    /// element-removal write `s' = SubSeq(s, 1, 0) \o SubSeq(s, 2, Len(s))`
    /// must classify under the gate via the DISJOINT-WINDOW rule
    /// (`concat_of_disjoint_subseq_windows`): the concatenation of two
    /// index-disjoint sub-windows of `s` is bounded by `capacity(s)` itself.
    /// The general `\o` SUM bound (`capacity + capacity`) would also be sound
    /// but is not a fixed point of the candidates iteration (2 -> 4 -> 8 ...),
    /// so the collector would fail to converge and emit nothing — this pin
    /// guards exactly that regression.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_wp15_subseq_concat_writer_capacity_under_gate() {
        let registry = VarRegistry::from_names(["s"]);
        let seed_values = vec![wp15_seq(vec![Value::SmallInt(10), Value::SmallInt(20)])];
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();

        let init = eq(
            ident("s"),
            Expr::Tuple(vec![expr(int_lit(10)), expr(int_lit(20))]),
        );
        let subseq = |lo: Expr, hi: Expr| {
            Expr::Apply(
                boxed(ident("SubSeq")),
                vec![expr(ident("s")), expr(lo), expr(hi)],
            )
        };
        let next = eq(
            prime(ident("s")),
            Expr::Apply(
                boxed(ident("\\o")),
                vec![
                    expr(subseq(int_lit(1), int_lit(0))),
                    expr(subseq(
                        int_lit(2),
                        Expr::Apply(boxed(ident("Len")), vec![expr(ident("s"))]),
                    )),
                ],
            ),
        );

        let _guard = FlatWriteAdmitEnvGuard::set(Some("1"));
        let mut out = Vec::new();
        collect_sequence_capacity_writer_proofs_with_ops(
            &init,
            &next,
            "Init/Next sequence writer proof",
            &registry,
            &seed_values,
            &constants,
            &op_defs,
            &op_replacements,
            &mut out,
        );
        assert_eq!(out.len(), 1, "expected one capacity proof, got {out:?}");
        assert_eq!(
            out[0].max_len, 2,
            "disjoint SubSeq windows of the same base are bounded by the base \
             capacity itself (fixpoint-stable), not the divergent sum"
        );
    }

    /// WP-ARGS: btree's `args` writer evidence — `NIL` from `Init`, `<<key>>`
    /// from `GetReq`, `<<key, val>>` from `InsertReq`/`UpdateReq` — must
    /// assemble into ONE VARIANT PER ARITY, each carrying PER-POSITION layouts.
    ///
    /// This is the shape the folded design could not express: position 0 is an
    /// `Int` (from `Keys`) and position 1 a `ModelValue` (from `Vals`), so a
    /// single shared element layout would lose the lane of each raw slot.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_scalar_tuple_union_arms_assemble_btree_args_shape() {
        let keys: Vec<FlatScalarValue> = (1..=4).map(FlatScalarValue::Int).collect();
        let vals: Vec<FlatScalarValue> = ["x", "y", "z"]
            .into_iter()
            .map(|v| FlatScalarValue::ModelValue(Arc::from(v)))
            .collect();
        let arms = vec![
            ScalarTupleArmEvidence::Scalar(FlatScalarValue::ModelValue(Arc::from("nil"))),
            ScalarTupleArmEvidence::Tuple(vec![keys.clone()]),
            ScalarTupleArmEvidence::Tuple(vec![keys.clone(), vals.clone()]),
        ];
        // Assert the gate-independent FOLDING; `TaggedUnionProof::new`'s
        // per-variant finiteness check consults a process-global env gate, so
        // testing through it would race other tests in this binary.
        let variants = scalar_tuple_union_variants_from_arms(&arms)
            .expect("btree args arms must fold into union variants");
        // Scalar sentinel, then one variant per ARITY in ascending order. The
        // arity lives in the tag, which is what dissolves the ambiguity that
        // forced the two tuple arities to be folded together.
        assert_eq!(variants.len(), 3);
        assert_eq!(variants[0], FlatValueLayout::Scalar(SlotType::ModelValue));
        assert_eq!(
            variants[1],
            FlatValueLayout::HeterogeneousTuple {
                element_layouts: vec![FlatValueLayout::Scalar(SlotType::Int)],
            }
        );
        // The mixed-kind case: each position keeps its OWN lane.
        assert_eq!(
            variants[2],
            FlatValueLayout::HeterogeneousTuple {
                element_layouts: vec![
                    FlatValueLayout::Scalar(SlotType::Int),
                    FlatValueLayout::Scalar(SlotType::ModelValue),
                ],
            }
        );
        // No length slot: the arity-2 arm is exactly two payload slots.
        let widest = variants
            .iter()
            .map(FlatValueLayout::slot_count)
            .max()
            .expect("three variants");
        assert_eq!(widest, 2, "payload window is the widest variant");
    }

    /// A position whose universe MIXES scalar lanes cannot store a raw value
    /// (the lane would be unrecoverable), so it falls back to the
    /// `TaggedScalarUnion` universe-index encoding — for that position only,
    /// never for the whole tuple.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_scalar_tuple_union_mixed_lane_position_uses_index_encoding() {
        let mixed = vec![
            FlatScalarValue::Int(1),
            FlatScalarValue::ModelValue(Arc::from("nil")),
        ];
        let arms = vec![
            ScalarTupleArmEvidence::Scalar(FlatScalarValue::ModelValue(Arc::from("sentinel"))),
            ScalarTupleArmEvidence::Tuple(vec![vec![FlatScalarValue::Int(7)], mixed]),
        ];
        let variants = scalar_tuple_union_variants_from_arms(&arms).expect("union must assemble");
        let FlatValueLayout::HeterogeneousTuple { element_layouts } = &variants[1] else {
            panic!("second variant must be the arity-2 tuple arm");
        };
        assert_eq!(element_layouts[0], FlatValueLayout::Scalar(SlotType::Int));
        let FlatValueLayout::TaggedScalarUnion { proof } = &element_layouts[1] else {
            panic!("a mixed-lane position must use the universe-index encoding");
        };
        assert_eq!(proof.universe().len(), 2);
    }

    /// Two writers of the SAME arity merge per position; they must not become
    /// two variants (the tag would be ambiguous) nor fold their positions
    /// together (position 0 would absorb position 1's universe).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_scalar_tuple_union_same_arity_writers_merge_per_position() {
        let arms = vec![
            ScalarTupleArmEvidence::Scalar(FlatScalarValue::ModelValue(Arc::from("nil"))),
            ScalarTupleArmEvidence::Tuple(vec![
                vec![FlatScalarValue::Int(1)],
                vec![FlatScalarValue::ModelValue(Arc::from("x"))],
            ]),
            ScalarTupleArmEvidence::Tuple(vec![
                vec![FlatScalarValue::Int(2)],
                vec![FlatScalarValue::ModelValue(Arc::from("y"))],
            ]),
        ];
        let variants = scalar_tuple_union_variants_from_arms(&arms).expect("union must assemble");
        assert_eq!(variants.len(), 2, "one arity means one tuple variant");
        assert_eq!(
            variants[1],
            FlatValueLayout::HeterogeneousTuple {
                element_layouts: vec![
                    FlatValueLayout::Scalar(SlotType::Int),
                    FlatValueLayout::Scalar(SlotType::ModelValue),
                ],
            },
            "positions keep their own lanes after merging both writers"
        );
    }

    /// A variable with only scalar writers, or only tuple writers, is NOT a
    /// scalar-or-tuple union — existing inference already carries those shapes,
    /// and re-encoding them here would change their storage for no reason.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_scalar_tuple_union_requires_both_a_scalar_and_a_tuple_arm() {
        let scalar_only = vec![ScalarTupleArmEvidence::Scalar(FlatScalarValue::Int(0))];
        assert!(scalar_tuple_union_variants_from_arms(&scalar_only).is_none());

        let tuple_only = vec![ScalarTupleArmEvidence::Tuple(vec![vec![
            FlatScalarValue::Int(1),
        ]])];
        assert!(scalar_tuple_union_variants_from_arms(&tuple_only).is_none());

        assert!(
            scalar_tuple_union_variants_from_arms(&[]).is_none(),
            "no writer evidence must not synthesize a union"
        );
    }

    /// Promoting a 1-slot scalar var to the 4-slot union carrier must repack
    /// every FOLLOWING variable's offset. A stale offset would alias two
    /// variables onto the same slots and silently corrupt the state buffer.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_resized_var_kind_override_repacks_following_offsets() {
        let registry = VarRegistry::from_names(["args", "after"]);
        let mut layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue, VarLayoutKind::Scalar],
        );
        assert_eq!(layout.var_layout(1).unwrap().offset, 1);
        assert_eq!(layout.total_slots(), 2);

        // Built from unconditionally-flat-primary variants so the assertion does
        // not depend on the process-global union gates.
        let proof = TaggedUnionProof::new(
            vec![
                FlatValueLayout::Scalar(SlotType::ModelValue),
                FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                        invariant: Arc::from("writer-coverage"),
                        element_invariant: Arc::from("writer-coverage"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::ModelValue)),
                },
            ],
            Arc::from("writer-coverage"),
        )
        .expect("union must build");
        let slots = FlatValueLayout::TaggedUnion {
            proof: proof.clone(),
        }
        .slot_count();
        assert!(layout.replace_var_kind_recompute(
            0,
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::TaggedUnion { proof },
            },
        ));
        assert_eq!(layout.var_layout(0).unwrap().slot_count, slots);
        assert_eq!(
            layout.var_layout(1).unwrap().offset,
            slots,
            "the following variable must be pushed past the widened union"
        );
        assert_eq!(layout.total_slots(), slots + 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_all_scalar() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let state = ArrayState::from_values(vec![
            Value::SmallInt(1),
            Value::Bool(true),
            Value::SmallInt(-5),
        ]);

        let layout = infer_layout(&state, &registry);
        assert_eq!(layout.var_count(), 3);
        assert_eq!(layout.total_slots(), 3);
        assert!(layout.is_all_scalar());

        // Bool variable gets ScalarBool, not Scalar.
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Scalar
        ));
        assert!(matches!(
            layout.var_layout(1).unwrap().kind,
            VarLayoutKind::ScalarBool
        ));
        assert!(matches!(
            layout.var_layout(2).unwrap().kind,
            VarLayoutKind::Scalar
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_int_func() {
        let registry = VarRegistry::from_names(["active"]);
        // active = [0 |-> FALSE, 1 |-> TRUE, 2 |-> FALSE]
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![Value::Bool(false), Value::Bool(true), Value::Bool(false)],
        );
        let state = ArrayState::from_values(vec![Value::IntFunc(Rp::new(func))]);

        let layout = infer_layout(&state, &registry);
        assert_eq!(layout.var_count(), 1);
        assert_eq!(layout.total_slots(), 3);

        let vl = layout.var_layout(0).unwrap();
        match &vl.kind {
            VarLayoutKind::IntArray {
                lo,
                len,
                elements_are_bool,
                ..
            } => {
                assert_eq!(*lo, 0);
                assert_eq!(*len, 3);
                assert!(
                    *elements_are_bool,
                    "Bool-valued IntFunc should have elements_are_bool=true"
                );
            }
            other => panic!("expected IntArray, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_record_all_scalar() {
        let registry = VarRegistry::from_names(["msg"]);
        let rec = RecordValue::from_sorted_str_entries(vec![
            (Arc::from("src"), Value::SmallInt(1)),
            (Arc::from("type"), Value::SmallInt(0)),
        ]);
        let state = ArrayState::from_values(vec![Value::Record(rec)]);

        let layout = infer_layout(&state, &registry);
        assert_eq!(layout.var_count(), 1);

        let vl = layout.var_layout(0).unwrap();
        match &vl.kind {
            VarLayoutKind::Record { field_names, .. } => {
                assert_eq!(field_names.len(), 2);
            }
            other => panic!("expected Record, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_small_set_as_dynamic() {
        // Sets are always Dynamic until bitmask encoding is implemented (Phase 6).
        // See #4007.
        let registry = VarRegistry::from_names(["nodes"]);
        let set = SortedSet::from_sorted_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
        ]);
        let state = ArrayState::from_values(vec![Value::Set(Rp::new(set))]);

        let layout = infer_layout(&state, &registry);
        let vl = layout.var_layout(0).unwrap();
        assert!(
            matches!(&vl.kind, VarLayoutKind::Dynamic),
            "expected Dynamic for set, got {:?}",
            vl.kind
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_recursive_mcl_req_shape() {
        let registry = VarRegistry::from_names(["req"]);
        let inner = || {
            Value::IntFunc(Rp::new(IntIntervalFunc::new(
                1,
                3,
                vec![Value::SmallInt(0), Value::SmallInt(1), Value::SmallInt(2)],
            )))
        };
        let req = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            3,
            vec![inner(), inner(), inner()],
        )));
        let state = ArrayState::from_values(vec![req]);

        let layout = infer_layout(&state, &registry);

        assert!(layout.is_fully_flat());
        assert_eq!(layout.total_slots(), 9);
        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout:
                    FlatValueLayout::IntFunction {
                        lo,
                        len,
                        value_layout,
                    },
            } => {
                assert_eq!((*lo, *len), (1, 3));
                match value_layout.as_ref() {
                    FlatValueLayout::IntFunction {
                        lo,
                        len,
                        value_layout,
                    } => {
                        assert_eq!((*lo, *len), (1, 3));
                        assert_eq!(**value_layout, FlatValueLayout::Scalar(SlotType::Int));
                    }
                    other => panic!("expected nested IntFunction, got {other:?}"),
                }
            }
            other => panic!("expected recursive req layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_recursive_mcl_subset_proc_shapes() {
        let registry = VarRegistry::from_names(["clock", "ack", "crit"]);
        let clock = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            3,
            vec![Value::SmallInt(1), Value::SmallInt(1), Value::SmallInt(1)],
        )));
        let empty_proc_set = || Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![])));
        let ack = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            3,
            vec![empty_proc_set(), empty_proc_set(), empty_proc_set()],
        )));
        let crit = empty_proc_set();
        let state = ArrayState::from_values(vec![clock, ack, crit]);

        let layout = infer_layout(&state, &registry);

        assert!(
            !layout.is_fully_flat(),
            "unproven empty top-level set must stay dynamic even when other variables expose a scalar domain"
        );
        assert_eq!(layout.total_slots(), 5);
        match &layout.var_layout(1).unwrap().kind {
            VarLayoutKind::Dynamic => {}
            other => panic!("expected dynamic ack layout, got {other:?}"),
        }
        match &layout.var_layout(2).unwrap().kind {
            VarLayoutKind::Dynamic => {}
            other => panic!("expected dynamic crit layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_recursive_mcl_seq_subset_proc_shapes() {
        use tla_value::value::SeqValue;

        let registry = VarRegistry::from_names(["clock", "ack", "crit"]);
        let clock = Value::Seq(Rp::new(SeqValue::from_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(1),
            Value::SmallInt(1),
        ])));
        let empty_proc_set = || Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![])));
        let ack = Value::Seq(Rp::new(SeqValue::from_vec(vec![
            empty_proc_set(),
            empty_proc_set(),
            empty_proc_set(),
        ])));
        let crit = empty_proc_set();
        let state = ArrayState::from_values(vec![clock, ack, crit]);

        let layout = infer_layout(&state, &registry);

        assert!(
            !layout.is_fully_flat(),
            "unproven empty top-level set must stay dynamic even when a sequence exposes a scalar domain"
        );
        assert_eq!(layout.total_slots(), 6);
        match &layout.var_layout(1).unwrap().kind {
            VarLayoutKind::Dynamic => {}
            other => panic!("expected dynamic ack sequence layout, got {other:?}"),
        }
        match &layout.var_layout(2).unwrap().kind {
            VarLayoutKind::Dynamic => {}
            other => panic!("expected dynamic crit layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_recursive_mcl_init_tuple_channel_shapes() {
        use tla_value::value::SeqValue;

        fn seq(values: Vec<Value>) -> Value {
            Value::Seq(Rp::new(SeqValue::from_vec(values)))
        }

        let registry = VarRegistry::from_names(["clock", "req", "ack", "network", "crit"]);
        let clock = seq(vec![
            Value::SmallInt(1),
            Value::SmallInt(1),
            Value::SmallInt(1),
        ]);
        let req_row = || {
            seq(vec![
                Value::SmallInt(0),
                Value::SmallInt(0),
                Value::SmallInt(0),
            ])
        };
        let req = seq(vec![req_row(), req_row(), req_row()]);
        let empty_proc_set = || Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![])));
        let ack = seq(vec![empty_proc_set(), empty_proc_set(), empty_proc_set()]);
        let empty_channel = || Value::tuple(Vec::<Value>::new());
        let network_row = || seq(vec![empty_channel(), empty_channel(), empty_channel()]);
        let network = seq(vec![network_row(), network_row(), network_row()]);
        let crit = empty_proc_set();
        let state = ArrayState::from_values(vec![clock, req, ack, network, crit]);

        let layout = infer_layout(&state, &registry);

        assert!(
            !layout.is_fully_flat(),
            "unproven empty process sets must stay dynamic while tuple channel shapes stay flat"
        );
        assert_eq!(layout.total_slots(), 32);
        assert_eq!(layout.var_layout(0).unwrap().slot_count, 4);
        assert_eq!(layout.var_layout(1).unwrap().slot_count, 13);
        assert_eq!(layout.var_layout(2).unwrap().slot_count, 1);
        assert_eq!(layout.var_layout(3).unwrap().slot_count, 13);
        assert_eq!(layout.var_layout(4).unwrap().slot_count, 1);
        match &layout.var_layout(3).unwrap().kind {
            VarLayoutKind::Recursive { .. } => {}
            other => panic!("expected recursive network layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_recursive_mcl_network_shape_with_observed_message() {
        use tla_value::value::SeqValue;

        let registry = VarRegistry::from_names(["network"]);
        let msg = Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("clock"), Value::SmallInt(1)),
            (Arc::from("type"), Value::String(Rp::from("req"))),
        ]));
        let nonempty = Value::Seq(Rp::new(SeqValue::from_vec(vec![msg])));
        let empty = || Value::Seq(Rp::new(SeqValue::from_vec(vec![])));
        let row1 = Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![empty(), nonempty])));
        let row2 = Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![empty(), empty()])));
        let network = Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![row1, row2])));
        let state = ArrayState::from_values(vec![network]);

        let layout = infer_layout(&state, &registry);

        assert!(layout.is_fully_flat());
        assert_eq!(layout.total_slots(), 12);
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Recursive { .. }
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_wavefront_recursive_network_empty_then_observed_message() {
        use tla_value::value::SeqValue;

        let registry = VarRegistry::from_names(["network"]);
        let empty = || Value::Seq(Rp::new(SeqValue::from_vec(vec![])));
        let empty_row =
            || Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![empty(), empty()])));
        let empty_network = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            2,
            vec![empty_row(), empty_row()],
        )));

        let msg = Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("clock"), Value::SmallInt(1)),
            (Arc::from("type"), Value::String(Rp::from("req"))),
        ]));
        let observed_row = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            2,
            vec![empty(), Value::Seq(Rp::new(SeqValue::from_vec(vec![msg])))],
        )));
        let observed_network = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            2,
            vec![observed_row, empty_row()],
        )));
        let states = vec![
            ArrayState::from_values(vec![empty_network]),
            ArrayState::from_values(vec![observed_network]),
        ];

        let layout = infer_layout_from_wavefront(&states, &registry);

        assert!(layout.is_fully_flat());
        assert_eq!(layout.total_slots(), 12);
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Recursive { .. }
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_wavefront_empty_sequence_uses_path_scoped_element_hint() {
        use tla_value::value::SeqValue;

        let registry = VarRegistry::from_names(["network", "log"]);
        let empty = || Value::Seq(Rp::new(SeqValue::from_vec(vec![])));
        let empty_row =
            || Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![empty(), empty()])));
        let empty_network = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            2,
            vec![empty_row(), empty_row()],
        )));

        let msg = Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("clock"), Value::SmallInt(1)),
            (Arc::from("type"), Value::String(Rp::from("req"))),
        ]));
        let observed_row = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            2,
            vec![Value::Seq(Rp::new(SeqValue::from_vec(vec![msg]))), empty()],
        )));
        let observed_network = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            1,
            2,
            vec![observed_row, empty_row()],
        )));

        let states = vec![
            ArrayState::from_values(vec![
                empty_network,
                Value::Seq(Rp::new(SeqValue::from_vec(vec![Value::SmallInt(1)]))),
            ]),
            ArrayState::from_values(vec![
                observed_network,
                Value::Seq(Rp::new(SeqValue::from_vec(vec![
                    Value::SmallInt(2),
                    Value::SmallInt(3),
                ]))),
            ]),
        ];

        let layout = infer_layout_from_wavefront(&states, &registry);

        assert!(
            layout.is_fully_flat(),
            "path-scoped sequence hints should avoid falling back to Dynamic: {:?}",
            layout.var_layout(0).unwrap().kind
        );
        assert_eq!(layout.total_slots(), 15);
        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction { value_layout, .. },
            } => match value_layout.as_ref() {
                FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                    FlatValueLayout::Sequence {
                        max_len,
                        element_layout,
                        ..
                    } => {
                        assert_eq!(*max_len, 1);
                        assert!(matches!(
                            element_layout.as_ref(),
                            FlatValueLayout::Record { .. }
                        ));
                    }
                    other => panic!("expected network channel sequence layout, got {other:?}"),
                },
                other => panic!("expected nested network function layout, got {other:?}"),
            },
            other => panic!("expected recursive network layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_mixed() {
        let registry = VarRegistry::from_names(["pc", "network", "msgs"]);

        // pc = 0 (scalar)
        // network = [0 |-> 0, 1 |-> 0, 2 |-> 0] (IntArray)
        // msgs = <<1, 2>> (fixed sequence)
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let seq =
            tla_value::value::SeqValue::from_vec(vec![Value::SmallInt(1), Value::SmallInt(2)]);
        let state = ArrayState::from_values(vec![
            Value::SmallInt(0),
            Value::IntFunc(Rp::new(func)),
            Value::Seq(Rp::new(seq)),
        ]);

        let layout = infer_layout(&state, &registry);
        assert_eq!(layout.var_count(), 3);
        // pc: 1 slot + network: 3 slots + msgs: 3 slots (len + 2 elems) = 7
        assert_eq!(layout.total_slots(), 7);
        assert!(!layout.is_all_scalar());
        assert!(!layout.is_trivial());

        // Verify kinds
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Scalar
        ));
        assert!(matches!(
            layout.var_layout(1).unwrap().kind,
            VarLayoutKind::IntArray { lo: 0, len: 3, .. }
        ));
        assert!(matches!(
            layout.var_layout(2).unwrap().kind,
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence { max_len: 2, .. }
            }
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_range_proof_selects_tagged_function_layout() {
        let registry = VarRegistry::from_names(["owner"]);
        let proc_domain: Arc<[Value]> = Arc::from(
            vec![
                Value::ModelValue(Rp::from("p1")),
                Value::ModelValue(Rp::from("p2")),
            ]
            .into_boxed_slice(),
        );
        let state = ArrayState::from_values(vec![Value::Func(Rp::new(
            FuncValue::from_sorted_entries(vec![
                (
                    Value::ModelValue(Rp::from("p1")),
                    Value::ModelValue(Rp::from("none")),
                ),
                (
                    Value::ModelValue(Rp::from("p2")),
                    Value::ModelValue(Rp::from("none")),
                ),
            ]),
        ))]);

        let unproven = infer_layout(&state, &registry);
        match &unproven.var_layout(0).unwrap().kind {
            VarLayoutKind::StringKeyedArray { range_encoding, .. } => {
                assert_eq!(range_encoding, &StringKeyedArrayRangeEncoding::ScalarSlots);
            }
            other => panic!("expected scalar string-keyed function layout, got {other:?}"),
        }
        assert!(!unproven.supports_flat_bfs_auto_admission());

        let proof = TaggedScalarSetRangeTypeProof {
            var_idx: 0,
            path: Vec::new(),
            domain: Arc::clone(&proc_domain),
            scalar_type: SlotType::ModelValue,
            set_universe: vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            ],
            invariant: Arc::from("TypeOK"),
        };
        let tagged = infer_layout_with_sequence_layout_and_tagged_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[proof],
        );

        match &tagged.var_layout(0).unwrap().kind {
            VarLayoutKind::StringKeyedArray { range_encoding, .. } => match range_encoding {
                StringKeyedArrayRangeEncoding::TaggedScalarOrSet(range_proof) => {
                    assert_eq!(range_proof.scalar_type(), SlotType::ModelValue);
                    assert_eq!(range_proof.set_universe().len(), 2);
                    assert_eq!(range_proof.source().as_ref(), "TypeOK");
                }
                other => panic!("expected tagged scalar/set range encoding, got {other:?}"),
            },
            other => panic!("expected tagged string-keyed function layout, got {other:?}"),
        }
        assert!(tagged.has_model_value_keyed_tagged_scalar_set_range());
        assert!(tagged.supports_flat_bfs_auto_admission());
        assert!(tagged.supports_flat_primary());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_range_proof_accepts_compatible_duplicate_sources() {
        let registry = VarRegistry::from_names(["owner"]);
        let proc_domain: Arc<[Value]> = Arc::from(
            vec![
                Value::ModelValue(Rp::from("p1")),
                Value::ModelValue(Rp::from("p2")),
            ]
            .into_boxed_slice(),
        );
        let state = ArrayState::from_values(vec![Value::Func(Rp::new(
            FuncValue::from_sorted_entries(vec![
                (
                    Value::ModelValue(Rp::from("p1")),
                    Value::ModelValue(Rp::from("none")),
                ),
                (
                    Value::ModelValue(Rp::from("p2")),
                    Value::ModelValue(Rp::from("none")),
                ),
            ]),
        ))]);
        let set_universe = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
        ];
        let writer_proof = TaggedScalarSetRangeTypeProof {
            var_idx: 0,
            path: Vec::new(),
            domain: Arc::clone(&proc_domain),
            scalar_type: SlotType::ModelValue,
            set_universe: set_universe.clone(),
            invariant: Arc::from("Init/Next writer proof"),
        };
        let action_producer_proof = TaggedScalarSetRangeTypeProof {
            invariant: Arc::from("action-producer:var0"),
            ..writer_proof.clone()
        };

        let tagged = infer_layout_with_sequence_layout_and_tagged_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[writer_proof, action_producer_proof],
        );

        match &tagged.var_layout(0).unwrap().kind {
            VarLayoutKind::StringKeyedArray { range_encoding, .. } => {
                assert!(matches!(
                    range_encoding,
                    StringKeyedArrayRangeEncoding::TaggedScalarOrSet(_)
                ));
            }
            other => panic!("expected tagged string-keyed function layout, got {other:?}"),
        }
        assert!(tagged.has_model_value_keyed_tagged_scalar_set_range());
    }

    fn expr(expr: Expr) -> tla_core::span::Spanned<Expr> {
        tla_core::span::Spanned::dummy(expr)
    }

    fn boxed(expr: Expr) -> Box<tla_core::span::Spanned<Expr>> {
        Box::new(self::expr(expr))
    }

    fn ident(name: &str) -> Expr {
        Expr::Ident(name.to_string(), NameId::INVALID)
    }

    fn bound_var(name: &str, domain: Expr) -> BoundVar {
        BoundVar {
            name: tla_core::span::Spanned::dummy(name.to_string()),
            domain: Some(boxed(domain)),
            pattern: None,
        }
    }

    fn eq(left: Expr, right: Expr) -> Expr {
        Expr::Eq(boxed(left), boxed(right))
    }

    fn neq(left: Expr, right: Expr) -> Expr {
        Expr::Neq(boxed(left), boxed(right))
    }

    fn and(left: Expr, right: Expr) -> Expr {
        Expr::And(boxed(left), boxed(right))
    }

    fn or(left: Expr, right: Expr) -> Expr {
        Expr::Or(boxed(left), boxed(right))
    }

    fn in_(left: Expr, right: Expr) -> Expr {
        Expr::In(boxed(left), boxed(right))
    }

    fn subseteq(left: Expr, right: Expr) -> Expr {
        Expr::Subseteq(boxed(left), boxed(right))
    }

    fn prime(inner: Expr) -> Expr {
        Expr::Prime(boxed(inner))
    }

    fn func_apply(func: Expr, arg: Expr) -> Expr {
        Expr::FuncApply(boxed(func), boxed(arg))
    }

    fn set_enum(elems: Vec<Expr>) -> Expr {
        Expr::SetEnum(elems.into_iter().map(self::expr).collect())
    }

    fn record_set(fields: Vec<(&str, Expr)>) -> Expr {
        Expr::RecordSet(
            fields
                .into_iter()
                .map(|(name, expr)| {
                    (
                        tla_core::span::Spanned::dummy(name.to_string()),
                        self::expr(expr),
                    )
                })
                .collect(),
        )
    }

    fn set_union(left: Expr, right: Expr) -> Expr {
        Expr::Union(boxed(left), boxed(right))
    }

    fn set_minus(left: Expr, right: Expr) -> Expr {
        Expr::SetMinus(boxed(left), boxed(right))
    }

    fn powerset(base: Expr) -> Expr {
        Expr::Powerset(boxed(base))
    }

    fn func_set(domain: Expr, range: Expr) -> Expr {
        Expr::FuncSet(boxed(domain), boxed(range))
    }

    fn bool_lit(value: bool) -> Expr {
        Expr::Bool(value)
    }

    fn string_lit(value: &str) -> Expr {
        Expr::String(value.to_string())
    }

    fn except_update(base: Expr, index: Expr, value: Expr) -> Expr {
        Expr::Except(
            boxed(base),
            vec![tla_core::ast::ExceptSpec {
                path: vec![ExceptPathElement::Index(self::expr(index))],
                value: self::expr(value),
            }],
        )
    }

    fn model_value(name: &str) -> Value {
        Value::ModelValue(Rp::from(name))
    }

    fn model_set(values: Vec<Value>) -> Value {
        Value::Set(Rp::new(SortedSet::from_sorted_vec(values)))
    }

    fn dijkstra_temp_init(proc_name: &str, default_name: &str) -> Expr {
        and(
            in_(ident("k"), ident(proc_name)),
            eq(
                ident("temp"),
                Expr::FuncDef(
                    vec![bound_var("self", ident(proc_name))],
                    boxed(ident(default_name)),
                ),
            ),
        )
    }

    fn dijkstra_temp_positive_next(proc_name: &str) -> Expr {
        let write_k_self = || eq(prime(ident("k")), ident("self"));
        let preserve_k = || eq(prime(ident("k")), ident("k"));
        let write_temp = |value| {
            eq(
                prime(ident("temp")),
                except_update(ident("temp"), ident("self"), value),
            )
        };
        let temp_self = || func_apply(ident("temp"), ident("self"));

        let li3a = and(write_temp(ident("k")), write_k_self());
        let li4a = and(
            write_temp(set_minus(ident(proc_name), set_enum(vec![ident("self")]))),
            preserve_k(),
        );
        let li4b_then = Expr::Exists(
            vec![bound_var("j", temp_self())],
            boxed(and(
                write_temp(set_minus(temp_self(), set_enum(vec![ident("j")]))),
                preserve_k(),
            )),
        );
        let li4b_else = and(eq(prime(ident("temp")), ident("temp")), preserve_k());
        let li4b = Expr::If(
            boxed(neq(temp_self(), set_enum(vec![]))),
            boxed(li4b_then),
            boxed(li4b_else),
        );

        Expr::Exists(
            vec![bound_var("self", ident(proc_name))],
            boxed(or(or(li3a, li4a), li4b)),
        )
    }

    fn collect_writer_proofs(
        init_expr: &Expr,
        next_expr: &Expr,
        constants: tla_core::kani_types::HashMap<NameId, Value>,
        registry: &VarRegistry,
    ) -> Vec<TaggedScalarSetRangeTypeProof> {
        let proof_domains = BTreeMap::new();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();
        let mut proofs = Vec::new();
        collect_tagged_scalar_set_range_writer_proofs_with_ops(
            init_expr,
            next_expr,
            "Init/Next writer proof",
            registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        proofs
    }

    fn dijkstra_constants(
        proc_name: &str,
        proc_values: Vec<Value>,
    ) -> tla_core::kani_types::HashMap<NameId, Value> {
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(intern_name(proc_name), model_set(proc_values));
        constants.insert(
            intern_name("defaultInitValue"),
            model_value("defaultInitValue"),
        );
        constants
    }

    fn collect_type_proofs(
        expr: &Expr,
        constants: tla_core::kani_types::HashMap<NameId, Value>,
        registry: &VarRegistry,
    ) -> Vec<TaggedScalarSetRangeTypeProof> {
        let proof_domains = BTreeMap::new();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();
        let mut proofs = Vec::new();
        collect_tagged_scalar_set_range_type_proofs_with_ops(
            expr,
            "TypeOK",
            registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        proofs
    }

    fn collect_set_range_proofs(
        expr: &Expr,
        constants: tla_core::kani_types::HashMap<NameId, Value>,
        registry: &VarRegistry,
    ) -> Vec<SetBitmaskRangeTypeProof> {
        let proof_domains = BTreeMap::new();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();
        let mut proofs = Vec::new();
        collect_set_bitmask_range_type_proofs_with_ops(
            expr,
            "TypeOK",
            registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        proofs
    }

    fn collect_set_type_proofs(
        expr: &Expr,
        constants: tla_core::kani_types::HashMap<NameId, Value>,
        registry: &VarRegistry,
    ) -> Vec<SetBitmaskTypeProof> {
        let proof_domains = BTreeMap::new();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();
        let mut proofs = Vec::new();
        collect_set_bitmask_type_proofs_with_ops(
            expr,
            "TypeOK",
            registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        proofs
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_tagged_scalar_set_range_type_proof_for_domain_union_subset_domain() {
        let registry = VarRegistry::from_names(["temp"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let type_ok = in_(
            ident("temp"),
            func_set(
                ident("Proc"),
                set_union(ident("Proc"), powerset(ident("Proc"))),
            ),
        );

        let proofs = collect_type_proofs(&type_ok, constants, &registry);

        assert_eq!(proofs.len(), 1);
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 0);
        assert_eq!(proof.path, Vec::<SequenceCapacityPathStep>::new());
        assert_eq!(
            proof.domain.as_ref(),
            &[model_value("p1"), model_value("p2")]
        );
        assert_eq!(proof.scalar_type, SlotType::ModelValue);
        assert_eq!(
            proof.set_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            ]
        );

        let state = ArrayState::from_values(vec![Value::Func(Rp::new(
            FuncValue::from_sorted_entries(vec![
                (model_value("p1"), model_value("p1")),
                (
                    model_value("p2"),
                    Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![model_value("p1")]))),
                ),
            ]),
        ))]);
        let layout = infer_layout_with_sequence_layout_and_tagged_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &proofs,
        );

        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::StringKeyedArray { range_encoding, .. } => {
                assert!(matches!(
                    range_encoding,
                    StringKeyedArrayRangeEncoding::TaggedScalarOrSet(_)
                ));
            }
            other => panic!("expected tagged string-keyed function layout, got {other:?}"),
        }
        assert!(layout.has_model_value_keyed_tagged_scalar_set_range());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_set_bitmask_range_type_proof_for_function_subset_range() {
        let registry = VarRegistry::from_names(["requests"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Clients"),
            model_set(vec![model_value("c1"), model_value("c2")]),
        );
        constants.insert(
            intern_name("Resources"),
            model_set(vec![model_value("r1"), model_value("r2")]),
        );
        let type_ok = in_(
            ident("requests"),
            func_set(ident("Clients"), powerset(ident("Resources"))),
        );

        let proofs = collect_set_range_proofs(&type_ok, constants, &registry);

        assert_eq!(proofs.len(), 1);
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 0);
        assert_eq!(proof.path, Vec::<SequenceCapacityPathStep>::new());
        assert_eq!(
            proof.domain.as_ref(),
            &[model_value("c1"), model_value("c2")]
        );
        assert_eq!(
            proof.set_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("r1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("r2")),
            ]
        );
    }

    /// Exact ChangRoberts shape: `msgs \in [Node -> SUBSET {Id[n] : n \in Node}]`
    /// with an INT function domain `Node = 1..3` and `Id = [1|->1,2|->2,3|->3]`.
    /// The int domain takes the Sequence/IntArray function-layout path (distinct
    /// from the model-value StringKeyedArray path above), so this proves the
    /// range proof is collected for the int-keyed case too.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_set_bitmask_range_type_proof_for_int_domain_function_image_range() {
        let registry = VarRegistry::from_names(["msgs"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Node"),
            model_set(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ]),
        );
        constants.insert(
            intern_name("Id"),
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (Value::SmallInt(1), Value::SmallInt(1)),
                (Value::SmallInt(2), Value::SmallInt(2)),
                (Value::SmallInt(3), Value::SmallInt(3)),
            ]))),
        );
        let id_image = Expr::SetBuilder(
            boxed(func_apply(ident("Id"), ident("n"))),
            vec![bound_var("n", ident("Node"))],
        );
        let type_ok = in_(ident("msgs"), func_set(ident("Node"), powerset(id_image)));

        let proofs = collect_set_range_proofs(&type_ok, constants, &registry);

        assert_eq!(
            proofs.len(),
            1,
            "int-domain function-image range proof should be collected"
        );
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 0);
        assert_eq!(
            proof.set_universe,
            vec![
                FlatScalarValue::Int(1),
                FlatScalarValue::Int(2),
                FlatScalarValue::Int(3),
            ]
        );
    }

    /// Core of the SetBitmask comprehension-universe extractor: `{ Id[n] : n \in
    /// Node }` resolves to the exact image of `Id` over `Node`. A non-identity
    /// `Id` proves the image is computed by applying the function, not by echoing
    /// the domain.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_type_domain_values_for_function_image_set_builder() {
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Node"),
            model_set(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ]),
        );
        constants.insert(
            intern_name("Id"),
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (Value::SmallInt(1), Value::SmallInt(10)),
                (Value::SmallInt(2), Value::SmallInt(20)),
                (Value::SmallInt(3), Value::SmallInt(30)),
            ]))),
        );
        let set_builder = Expr::SetBuilder(
            boxed(func_apply(ident("Id"), ident("n"))),
            vec![bound_var("n", ident("Node"))],
        );
        let proof_domains = BTreeMap::new();

        let values =
            type_domain_values_with_replacements(&set_builder, &constants, &proof_domains, None)
                .expect("function-image comprehension should resolve to its exact image");
        assert_eq!(
            &*values,
            &[
                Value::SmallInt(10),
                Value::SmallInt(20),
                Value::SmallInt(30),
            ]
        );
    }

    /// Fail-closed boundary: an identity comprehension `{ n : n \in Node }` is
    /// not of the supported `F[x]` shape, so the extractor declines (returns
    /// `None`) rather than guess a universe.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_type_domain_values_set_builder_non_func_apply_body_fails_closed() {
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Node"),
            model_set(vec![Value::SmallInt(1), Value::SmallInt(2)]),
        );
        let set_builder = Expr::SetBuilder(boxed(ident("n")), vec![bound_var("n", ident("Node"))]);
        let proof_domains = BTreeMap::new();

        assert!(type_domain_values_with_replacements(
            &set_builder,
            &constants,
            &proof_domains,
            None
        )
        .is_none());
    }

    /// End-to-end: a function-range type invariant whose SUBSET universe is a
    /// function-image comprehension `requests \in [Clients -> SUBSET {Id[n] : n
    /// \in Node}]` yields a SetBitmask range proof with the comprehension's exact
    /// image as its universe. This is the ChangRoberts `msgs \in [Node -> SUBSET
    /// {Id[n] : n \in Node}]` blocker shape.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_set_bitmask_range_type_proof_for_function_image_subset_range() {
        let registry = VarRegistry::from_names(["requests"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Clients"),
            model_set(vec![model_value("c1"), model_value("c2")]),
        );
        constants.insert(
            intern_name("Node"),
            model_set(vec![model_value("n1"), model_value("n2")]),
        );
        constants.insert(
            intern_name("Id"),
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (model_value("n1"), model_value("r1")),
                (model_value("n2"), model_value("r2")),
            ]))),
        );
        let id_image = Expr::SetBuilder(
            boxed(func_apply(ident("Id"), ident("n"))),
            vec![bound_var("n", ident("Node"))],
        );
        let type_ok = in_(
            ident("requests"),
            func_set(ident("Clients"), powerset(id_image)),
        );

        let proofs = collect_set_range_proofs(&type_ok, constants, &registry);

        assert_eq!(proofs.len(), 1);
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 0);
        assert_eq!(
            proof.domain.as_ref(),
            &[model_value("c1"), model_value("c2")]
        );
        assert_eq!(
            proof.set_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("r1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("r2")),
            ]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_set_bitmask_type_proof_for_top_level_subseteq() {
        let registry = VarRegistry::from_names(["tx"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("TxId"),
            model_set(vec![model_value("t1"), model_value("t2")]),
        );
        let type_ok = subseteq(ident("tx"), ident("TxId"));

        let proofs = collect_set_type_proofs(&type_ok, constants, &registry);

        assert_eq!(proofs.len(), 1);
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 0);
        assert_eq!(proof.path, Vec::<SequenceCapacityPathStep>::new());
        assert_eq!(
            proof.set_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("t1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("t2")),
            ]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_set_bitmask_type_proof_for_top_level_in_subset() {
        let registry = VarRegistry::from_names(["tx"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("TxId"),
            model_set(vec![model_value("t1"), model_value("t2")]),
        );
        let type_ok = in_(ident("tx"), powerset(ident("TxId")));

        let proofs = collect_set_type_proofs(&type_ok, constants, &registry);

        assert_eq!(proofs.len(), 1);
        assert_eq!(
            proofs[0].set_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("t1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("t2")),
            ]
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_record_valued_subset_universe_fails_closed() {
        let registry = VarRegistry::from_names(["msgs"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(intern_name("RM"), model_set(vec![model_value("rm1")]));
        constants.insert(
            intern_name("Acceptor"),
            model_set(vec![model_value("a1"), model_value("a2")]),
        );

        let message = set_union(
            record_set(vec![
                ("type", set_enum(vec![string_lit("phase1a")])),
                ("ins", ident("RM")),
            ]),
            record_set(vec![
                ("type", set_enum(vec![string_lit("phase1b")])),
                ("ins", ident("RM")),
                ("acc", ident("Acceptor")),
            ]),
        );
        let type_ok = in_(ident("msgs"), powerset(message));

        let proofs = collect_set_type_proofs(&type_ok, constants, &registry);
        assert!(
            proofs.is_empty(),
            "record-valued SUBSET universes must not be compacted by scalar SetBitmask proofs"
        );

        let state = ArrayState::from_values(vec![Value::Set(Rp::new(SortedSet::from_sorted_vec(
            vec![],
        )))]);
        let layout = infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[],
            &[],
            &proofs,
            &[],
            &[],
            &[],
        );
        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "a record-valued SUBSET with no record-set bitmask proof must stay Dynamic \
             (scalar SetBitmask proofs must never compact records): {:?}",
            layout.var_layout(0).unwrap().kind
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_record_set_bitmask_proof_infers_record_set_layout() {
        // A finite record universe + a RecordSetBitmaskTypeProof compacts a
        // top-level set-of-records state var to a ProvenClosed RecordSetBitmask.
        let registry = VarRegistry::from_names(["msgs"]);

        let rec = |ty: &str, ins: i64| {
            Value::Record(RecordValue::from_sorted_str_entries(vec![
                (Arc::from("ins"), Value::SmallInt(ins)),
                (Arc::from("type"), Value::String(Rp::from(ty))),
            ]))
        };
        // Canonical sorted+deduped universe.
        let mut universe = vec![rec("phase1a", 1), rec("phase1a", 2), rec("phase2a", 1)];
        universe.sort();
        universe.dedup();

        let proof = RecordSetBitmaskTypeProof {
            var_idx: 0,
            path: Vec::new(),
            record_universe: universe.clone(),
            invariant: Arc::from("TypeOK"),
        };

        // Sampled init value is a non-empty subset of the universe.
        let init_set = Value::Set(Rp::new(SortedSet::from_iter(
            [rec("phase1a", 1), rec("phase2a", 1)].into_iter(),
        )));
        let state = ArrayState::from_values(vec![init_set.clone()]);

        let layout = infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[proof],
        );

        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout:
                    FlatValueLayout::RecordSetBitmask {
                        universe: u,
                        universe_closure,
                    },
            } => {
                assert_eq!(u, &universe, "universe must be canonical sorted+deduped");
                assert!(
                    universe_closure.is_proven_closed(),
                    "a TypeOK-backed record-set universe must be proven-closed"
                );
            }
            other => panic!("expected RecordSetBitmask layout, got {other:?}"),
        }

        // Round-trip: serialize the sampled set through the flat layout and back.
        let arc = std::sync::Arc::new(layout);
        let flat = crate::state::FlatState::from_array_state(&state, std::sync::Arc::clone(&arc));
        let restored = flat.to_array_state(&registry);
        let restored_values: Vec<Value> = restored.values().iter().map(Value::from).collect();
        assert_eq!(
            restored_values[0], init_set,
            "record-set bitmask round-trip must be value-identical"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_record_set_bitmask_multi_slot_roundtrip() {
        // A 150-record universe spans 3 i64 slots (ceil(150/64)). The multi-slot
        // bitmask must value-roundtrip a subset whose members land in EVERY slot
        // (low, middle, and high), and the slot count must be ceil(|universe|/64).
        let registry = VarRegistry::from_names(["msgs"]);

        let rec = |ins: i64| {
            Value::Record(RecordValue::from_sorted_str_entries(vec![(
                Arc::from("ins"),
                Value::SmallInt(ins),
            )]))
        };
        let mut universe: Vec<Value> = (0..150).map(rec).collect();
        universe.sort();
        universe.dedup();
        assert_eq!(universe.len(), 150);

        let proof = RecordSetBitmaskTypeProof {
            var_idx: 0,
            path: Vec::new(),
            record_universe: universe.clone(),
            invariant: Arc::from("TypeOK"),
        };

        // Pick members from slot 0 (idx 0,63), slot 1 (idx 64,127), slot 2 (idx
        // 149) — exercising bit 63 (the i64 sign bit) of full slots too.
        let members: Vec<Value> = [0i64, 63, 64, 127, 149].iter().map(|&i| rec(i)).collect();
        let init_set = Value::Set(Rp::new(SortedSet::from_iter(members.iter().cloned())));
        let state = ArrayState::from_values(vec![init_set.clone()]);

        let layout = infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[proof],
        );

        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout: rsb @ FlatValueLayout::RecordSetBitmask { universe: u, .. },
            } => {
                assert_eq!(u.len(), 150);
                assert_eq!(rsb.slot_count(), 3, "150 records must span 3 i64 slots");
            }
            other => panic!("expected RecordSetBitmask layout, got {other:?}"),
        }

        let arc = std::sync::Arc::new(layout);
        let flat = crate::state::FlatState::from_array_state(&state, std::sync::Arc::clone(&arc));
        let restored = flat.to_array_state(&registry);
        let restored_values: Vec<Value> = restored.values().iter().map(Value::from).collect();
        assert_eq!(
            restored_values[0], init_set,
            "multi-slot record-set bitmask round-trip must be value-identical"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_empty_top_level_set_fails_closed_until_subset_type_proof() {
        let registry = VarRegistry::from_names(["tx"]);
        let state = ArrayState::from_values(vec![Value::Set(Rp::new(SortedSet::from_sorted_vec(
            vec![],
        )))]);

        let unproven = infer_layout(&state, &registry);
        assert!(
            matches!(unproven.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "unproven empty top-level set must not infer a fixed bitmask universe: {:?}",
            unproven.var_layout(0).unwrap().kind
        );

        let proof = SetBitmaskTypeProof {
            var_idx: 0,
            path: Vec::new(),
            set_universe: vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("t1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("t2")),
            ],
            invariant: Arc::from("TypeOK"),
        };
        let proven = infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[proof],
            &[],
            &[],
            &[],
        );

        match &proven.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout:
                    FlatValueLayout::SetBitmask {
                        universe,
                        universe_closure,
                    },
            } => {
                assert_eq!(
                    universe,
                    &vec![
                        FlatScalarValue::ModelValue(std::sync::Arc::from("t1")),
                        FlatScalarValue::ModelValue(std::sync::Arc::from("t2")),
                    ]
                );
                assert!(
                    universe_closure.is_proven_closed(),
                    "an invariant-backed top-level bitmask type proof must mark the universe proven-closed"
                );
            }
            other => panic!("expected recursive top-level bitmask layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_empty_function_range_sets_fail_closed_until_subset_range_proof() {
        let registry = VarRegistry::from_names(["requests"]);
        let clients: Arc<[Value]> =
            Arc::from(vec![model_value("c1"), model_value("c2")].into_boxed_slice());
        let empty_set = || Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![])));
        let state = ArrayState::from_values(vec![Value::Func(Rp::new(
            FuncValue::from_sorted_entries(vec![
                (model_value("c1"), empty_set()),
                (model_value("c2"), empty_set()),
            ]),
        ))]);

        let unproven = infer_layout(&state, &registry);
        assert!(
            matches!(unproven.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "unproven empty range sets must not infer a fixed bitmask universe: {:?}",
            unproven.var_layout(0).unwrap().kind
        );

        let proof = SetBitmaskRangeTypeProof {
            var_idx: 0,
            path: Vec::new(),
            domain: clients,
            set_universe: vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("r1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("r2")),
            ],
            invariant: Arc::from("TypeOK"),
        };
        let proven = infer_layout_with_sequence_layout_tagged_and_set_range_proofs(
            &state,
            &registry,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[proof],
        );

        match &proven.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function { value_layout, .. },
            } => match value_layout.as_ref() {
                FlatValueLayout::SetBitmask {
                    universe,
                    universe_closure,
                } => {
                    assert_eq!(
                        universe,
                        &vec![
                            FlatScalarValue::ModelValue(std::sync::Arc::from("r1")),
                            FlatScalarValue::ModelValue(std::sync::Arc::from("r2")),
                        ]
                    );
                    assert!(
                        universe_closure.is_proven_closed(),
                        "an invariant-backed function-range bitmask proof must mark the universe proven-closed"
                    );
                }
                other => panic!("expected proven range set bitmask layout, got {other:?}"),
            },
            other => panic!("expected recursive function layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_tagged_scalar_set_range_type_proof_rejects_mixed_scalar_types() {
        let registry = VarRegistry::from_names(["temp"]);

        let mixed_scalar_arm = in_(
            ident("temp"),
            func_set(
                set_enum(vec![bool_lit(false), bool_lit(true)]),
                set_union(
                    set_enum(vec![bool_lit(true), string_lit("sentinel")]),
                    powerset(set_enum(vec![bool_lit(false), bool_lit(true)])),
                ),
            ),
        );
        assert!(
            collect_type_proofs(
                &mixed_scalar_arm,
                tla_core::kani_types::HashMap::default(),
                &registry,
            )
            .is_empty(),
            "mixed scalar-arm types must not infer tagged scalar/set range proof"
        );

        let mismatched_set_universe = in_(
            ident("temp"),
            func_set(
                set_enum(vec![string_lit("p1"), string_lit("p2")]),
                set_union(
                    set_enum(vec![string_lit("sentinel")]),
                    powerset(set_enum(vec![bool_lit(false), bool_lit(true)])),
                ),
            ),
        );
        assert!(
            collect_type_proofs(
                &mismatched_set_universe,
                tla_core::kani_types::HashMap::default(),
                &registry,
            )
            .is_empty(),
            "scalar arm and SUBSET universe types must agree"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_tagged_scalar_set_range_writer_proof_for_dijkstra_temp() {
        let registry = VarRegistry::from_names(["k", "temp"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let init_expr = dijkstra_temp_init("Proc", "defaultInitValue");
        let next_expr = dijkstra_temp_positive_next("Proc");

        let proofs = collect_writer_proofs(&init_expr, &next_expr, constants, &registry);

        assert_eq!(proofs.len(), 1);
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 1);
        assert_eq!(proof.path, Vec::<SequenceCapacityPathStep>::new());
        assert_eq!(
            proof.domain.as_ref(),
            &[model_value("p1"), model_value("p2")]
        );
        assert_eq!(proof.scalar_type, SlotType::ModelValue);
        assert_eq!(
            proof.set_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2"))
            ]
        );
    }

    // ====================================================================
    // #43 fail-closed gate: a `TypeOK`-derived FixedScalar range / var proof
    // must NOT promote a variable whose writers can assign a SET.
    // ====================================================================

    fn empty_op_env() -> (
        BTreeMap<String, Arc<[Value]>>,
        tla_core::OpEnv,
        tla_core::kani_types::HashMap<String, String>,
    ) {
        (
            BTreeMap::new(),
            tla_core::OpEnv::default(),
            tla_core::kani_types::HashMap::default(),
        )
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_nonscalar_writer_veto_flags_set_bearing_dijkstra_temp() {
        // `temp` (var 1) is written `temp' = [temp EXCEPT ![self] = Proc \ {self}]`
        // (a SET) in one branch — it MUST be vetoed. `k` (var 0) is only ever
        // assigned `self`/`k` (model-value scalars) — it must NOT be vetoed.
        let registry = VarRegistry::from_names(["k", "temp"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let init_expr = dijkstra_temp_init("Proc", "defaultInitValue");
        let next_expr = dijkstra_temp_positive_next("Proc");
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let vetoed = vars_with_nonscalar_writers(
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );

        assert!(vetoed.contains(&1), "temp (set-bearing) must be vetoed");
        assert!(
            !vetoed.contains(&0),
            "k (scalar-only model value) must not be vetoed"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_init_sampled_scalar_slot_demoted_for_set_bearing_writer() {
        // MissViol2 shape: `x` is INIT-sampled as `x = 0` (a plain `Scalar` flat
        // slot, NO type-proof), but a Next disjunct writes `x' = {1, 2}` (a SET).
        // `phase` is only ever assigned scalars. Without the #43-extension gate,
        // `x` would be admitted to flat-primary as `Scalar`; the set value would
        // be encoded into the same i64 slot and alias a distinct scalar state in
        // the flat fingerprint, silently undercounting the BFS (missed violation).
        //
        // The veto must flag `x` (set-bearing) but NOT `phase` (scalar-only), and
        // `veto_flat_primary_scalar_slot_vars` must demote `x`'s init-sampled
        // `Scalar` layout to `Dynamic` (no longer flat-primary) while leaving
        // `phase` a flat-primary `Scalar`.
        let registry = VarRegistry::from_names(["x", "phase"]);
        let constants = tla_core::kani_types::HashMap::default();
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        // Init == x = 0 /\ phase = 0
        let init_expr = and(
            eq(ident("x"), Expr::Int(0.into())),
            eq(ident("phase"), Expr::Int(0.into())),
        );
        // Next ==
        //   \/ phase = 0 /\ phase' = 1 /\ x' = {1, 2}   (SET into x)
        //   \/ phase = 0 /\ phase' = 1 /\ x' = 9         (scalar into x)
        //   \/ phase = 1 /\ x = 9 /\ phase' = 2 /\ x' = 7
        let next_expr = or(
            or(
                and(
                    and(
                        eq(ident("phase"), Expr::Int(0.into())),
                        eq(prime(ident("phase")), Expr::Int(1.into())),
                    ),
                    eq(
                        prime(ident("x")),
                        set_enum(vec![Expr::Int(1.into()), Expr::Int(2.into())]),
                    ),
                ),
                and(
                    and(
                        eq(ident("phase"), Expr::Int(0.into())),
                        eq(prime(ident("phase")), Expr::Int(1.into())),
                    ),
                    eq(prime(ident("x")), Expr::Int(9.into())),
                ),
            ),
            and(
                and(
                    and(
                        eq(ident("phase"), Expr::Int(1.into())),
                        eq(ident("x"), Expr::Int(9.into())),
                    ),
                    eq(prime(ident("phase")), Expr::Int(2.into())),
                ),
                eq(prime(ident("x")), Expr::Int(7.into())),
            ),
        );

        let vetoed = nonscalar_writer_vetoed_vars(
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            vetoed.contains(&0),
            "x (set-bearing via `x' = {{1,2}}`) must be vetoed"
        );
        assert!(
            !vetoed.contains(&1),
            "phase (scalar-only) must not be vetoed"
        );

        // The init-sampled state `x = 0 /\ phase = 0` infers two plain Scalars.
        let init_state = ArrayState::from_values(vec![Value::SmallInt(0), Value::SmallInt(0)]);
        let mut layout = infer_layout(&init_state, &registry);
        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Scalar),
            "x is init-sampled as a plain Scalar before the gate"
        );
        assert!(
            layout.supports_flat_primary(),
            "all-scalar init layout would be flat-primary without the gate (this is the bug)"
        );

        let demoted = layout.veto_flat_primary_scalar_slot_vars(&vetoed);
        assert_eq!(demoted, vec![0], "only x must be demoted");
        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "fail closed: x's init-sampled scalar slot must be demoted to Dynamic"
        );
        assert!(
            matches!(layout.var_layout(1).unwrap().kind, VarLayoutKind::Scalar),
            "phase (scalar-only) must stay a flat Scalar (no over-rejection)"
        );
        assert!(
            !layout.supports_flat_primary(),
            "layout with a demoted set-bearing var must NOT be flat-primary"
        );
        assert!(
            !layout.is_fully_flat(),
            "a Dynamic var makes the layout no longer fully-flat"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_function_domain_growth_writer_demotes_flat_primary() {
        // PrimedDomain shape: `f` is INIT-sampled as `[x \in {1} |-> x]` (a
        // scalar-range IntArray with a FIXED single element slot), but Next
        // rebuilds it over a GROWING domain `DOMAIN f \cup {2}`. Encoding the
        // grown function into f's fixed-width flat slot silently truncates the
        // extra key and drops reachable states (the observed missed DEADLOCK).
        // The writer veto must flag `f`, and `veto_flat_primary_scalar_slot_vars`
        // must demote its IntArray to Dynamic.
        let registry = VarRegistry::from_names(["f", "result"]);
        let constants = tla_core::kani_types::HashMap::default();
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let funcdef =
            |domain: Expr, body: Expr| Expr::FuncDef(vec![bound_var("x", domain)], boxed(body));
        let domain_of = |v: &str| Expr::Domain(boxed(ident(v)));

        // Init == f = [x \in {1} |-> x] /\ result = 0
        let init_expr = and(
            eq(
                ident("f"),
                funcdef(set_enum(vec![Expr::Int(1.into())]), ident("x")),
            ),
            eq(ident("result"), Expr::Int(0.into())),
        );
        // Next == f' = [x \in DOMAIN f \cup {2} |-> x] /\ result' = 1
        let next_expr = and(
            eq(
                prime(ident("f")),
                funcdef(
                    set_union(domain_of("f"), set_enum(vec![Expr::Int(2.into())])),
                    ident("x"),
                ),
            ),
            eq(prime(ident("result")), Expr::Int(1.into())),
        );

        let vetoed = nonscalar_writer_vetoed_vars(
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            vetoed.contains(&0),
            "f (domain grows via `DOMAIN f \\cup {{2}}`) must be vetoed"
        );
        assert!(
            !vetoed.contains(&1),
            "result (scalar-only) must not be vetoed (no over-rejection)"
        );

        // CONTROL 1: a domain-PRESERVING rebuild `f' = [x \in DOMAIN f |-> x]`
        // must NOT veto — the common in-place map idiom stays flat.
        let next_preserving = eq(prime(ident("f")), funcdef(domain_of("f"), ident("x")));
        let vetoed_preserving = nonscalar_writer_vetoed_vars(
            &init_expr,
            &next_preserving,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            !vetoed_preserving.contains(&0),
            "`f' = [x \\in DOMAIN f |-> x]` preserves the domain — must NOT veto"
        );

        // CONTROL 2: a provably-CONSTANT domain `f' = [x \in {1,2,3} |-> x]`
        // is layout-stable across states — must NOT veto.
        let next_const = eq(
            prime(ident("f")),
            funcdef(
                set_enum(vec![
                    Expr::Int(1.into()),
                    Expr::Int(2.into()),
                    Expr::Int(3.into()),
                ]),
                ident("x"),
            ),
        );
        let vetoed_const = nonscalar_writer_vetoed_vars(
            &init_expr,
            &next_const,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            !vetoed_const.contains(&0),
            "a provably-constant domain is layout-stable — must NOT veto"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_init_sampled_scalar_slot_kept_for_scalar_only_writer() {
        // A purely-scalar counter `x` (`x' = x + 1`) must STAY flat-primary: the
        // veto set is empty, so `veto_flat_primary_scalar_slot_vars` is a no-op
        // and the init-sampled `Scalar` layout is preserved (no over-rejection).
        let registry = VarRegistry::from_names(["x"]);
        let constants = tla_core::kani_types::HashMap::default();
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let init_expr = eq(ident("x"), Expr::Int(0.into()));
        // Next == x' = x + 1
        let next_expr = eq(
            prime(ident("x")),
            Expr::Add(boxed(ident("x")), boxed(Expr::Int(1.into()))),
        );

        let vetoed = nonscalar_writer_vetoed_vars(
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            vetoed.is_empty(),
            "a scalar-only counter must not be vetoed"
        );

        let init_state = ArrayState::from_values(vec![Value::SmallInt(0)]);
        let mut layout = infer_layout(&init_state, &registry);
        let demoted = layout.veto_flat_primary_scalar_slot_vars(&vetoed);
        assert!(demoted.is_empty(), "no var should be demoted");
        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Scalar),
            "scalar-only var stays a flat Scalar"
        );
        assert!(
            layout.supports_flat_primary(),
            "scalar-only var stays flat-primary (native-fused path preserved)"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_range_type_proof_dropped_for_set_bearing_writer() {
        // A permissive `TypeOK` claims `temp \in [Proc -> Proc]`, which yields a
        // FixedScalar(ModelValue) range type-proof. But the Init/Next writers
        // store a SET into the range, so the gate must DROP the proof (fail
        // closed) — otherwise `temp` would become a flat-primary scalar slot and
        // alias distinct set-valued states.
        let registry = VarRegistry::from_names(["k", "temp"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let init_expr = dijkstra_temp_init("Proc", "defaultInitValue");
        let next_expr = dijkstra_temp_positive_next("Proc");
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        // `TypeOK == temp \in [Proc -> Proc]`.
        let type_ok = in_(ident("temp"), func_set(ident("Proc"), ident("Proc")));
        let mut proofs = Vec::new();
        collect_fixed_scalar_range_type_proofs_with_ops(
            &type_ok,
            "TypeOK",
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 1),
            "TypeOK should produce a FixedScalar range proof for temp before the gate"
        );

        retain_writer_corroborated_fixed_scalar_range_proofs(
            &mut proofs,
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            !proofs.iter().any(|p| p.var_idx == 1),
            "fail closed: a set-bearing var's FixedScalar range proof must be dropped"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_range_type_proof_over_cross_product_domain() {
        // btree `valOf`-shaped clause: `valOf \in [Nodes \X Keys -> Vals]` with
        // model-value `Vals`. The `\X` domain must enumerate to the canonical
        // Value::cmp-sorted tuple set — the exact `TupleKeyedArray.domain_keys`
        // order — so the hint matches the inferred tuple-keyed layout.
        let registry = VarRegistry::from_names(["valOf"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Nodes"),
            Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
            ]))),
        );
        constants.insert(
            intern_name("Keys"),
            Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
            ]))),
        );
        constants.insert(
            intern_name("Vals"),
            model_set(vec![model_value("x"), model_value("y")]),
        );
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let type_ok = in_(
            ident("valOf"),
            func_set(
                Expr::Times(vec![expr(ident("Nodes")), expr(ident("Keys"))]),
                ident("Vals"),
            ),
        );
        let mut proofs = Vec::new();
        collect_fixed_scalar_range_type_proofs_with_ops(
            &type_ok,
            "TypeOK",
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );

        let proof = proofs
            .iter()
            .find(|p| p.var_idx == 0)
            .expect("a cross-product FuncSet domain must yield a FixedScalar range proof");
        assert_eq!(proof.scalar_type, SlotType::ModelValue);
        let expected_domain: Vec<Value> = vec![
            Value::tuple([Value::SmallInt(1), Value::SmallInt(1)]),
            Value::tuple([Value::SmallInt(1), Value::SmallInt(2)]),
            Value::tuple([Value::SmallInt(2), Value::SmallInt(1)]),
            Value::tuple([Value::SmallInt(2), Value::SmallInt(2)]),
        ];
        assert_eq!(
            proof.domain.as_ref(),
            expected_domain.as_slice(),
            "the \\X domain must enumerate in canonical Value::cmp tuple order"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_range_type_proof_over_homogeneous_union_range() {
        // The exact btree `valOf` clause shape:
        // `valOf \in [Nodes \X Keys -> Vals \cup {NIL}]` where `Vals` and `NIL`
        // are all model values — a HOMOGENEOUS union range must enumerate to a
        // FixedScalar(ModelValue) universe {nil, x, y}. (A heterogeneous union
        // such as `Nodes \cup {NIL}` still yields no FixedScalar proof — the
        // TaggedScalarUnion collector's territory.)
        let registry = VarRegistry::from_names(["valOf", "childOf"]);
        let mut constants = tla_core::kani_types::HashMap::default();
        constants.insert(
            intern_name("Nodes"),
            Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
            ]))),
        );
        constants.insert(
            intern_name("Keys"),
            Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
            ]))),
        );
        constants.insert(
            intern_name("Vals"),
            model_set(vec![model_value("x"), model_value("y")]),
        );
        constants.insert(intern_name("NIL"), model_value("nil"));
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let times_domain = || Expr::Times(vec![expr(ident("Nodes")), expr(ident("Keys"))]);
        let type_ok = and(
            in_(
                ident("valOf"),
                func_set(
                    times_domain(),
                    set_union(ident("Vals"), set_enum(vec![ident("NIL")])),
                ),
            ),
            // childOf: Int ∪ ModelValue range — heterogeneous, must NOT yield
            // a FixedScalar proof (fail closed, WP-05 union-carrier seam).
            in_(
                ident("childOf"),
                func_set(
                    times_domain(),
                    set_union(ident("Nodes"), set_enum(vec![ident("NIL")])),
                ),
            ),
        );
        let mut proofs = Vec::new();
        collect_fixed_scalar_range_type_proofs_with_ops(
            &type_ok,
            "TypeOk",
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );

        let proof = proofs
            .iter()
            .find(|p| p.var_idx == 0)
            .expect("homogeneous model-value union range must yield a FixedScalar proof");
        assert_eq!(proof.scalar_type, SlotType::ModelValue);
        let mut universe = proof.scalar_universe.clone();
        universe.sort();
        assert_eq!(
            universe,
            vec![
                FlatScalarValue::ModelValue(Arc::from("nil")),
                FlatScalarValue::ModelValue(Arc::from("x")),
                FlatScalarValue::ModelValue(Arc::from("y")),
            ]
        );
        assert!(
            !proofs.iter().any(|p| p.var_idx == 1),
            "a heterogeneous Int ∪ ModelValue union range must stay fail-closed"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_range_type_proof_kept_for_scalar_only_writer() {
        // `g` is a function whose range is only ever assigned model-value scalars
        // (`g' = [g EXCEPT ![self] = self]`). Its TypeOK FixedScalar range proof
        // must SURVIVE the gate (no over-rejection).
        let registry = VarRegistry::from_names(["g"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let init_expr = eq(
            ident("g"),
            Expr::FuncDef(vec![bound_var("self", ident("Proc"))], boxed(ident("self"))),
        );
        let next_expr = Expr::Exists(
            vec![bound_var("self", ident("Proc"))],
            boxed(eq(
                prime(ident("g")),
                except_update(ident("g"), ident("self"), ident("self")),
            )),
        );

        let type_ok = in_(ident("g"), func_set(ident("Proc"), ident("Proc")));
        let mut proofs = Vec::new();
        collect_fixed_scalar_range_type_proofs_with_ops(
            &type_ok,
            "TypeOK",
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 0),
            "TypeOK should produce a FixedScalar range proof for g"
        );

        retain_writer_corroborated_fixed_scalar_range_proofs(
            &mut proofs,
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 0),
            "no over-rejection: a scalar-only var's FixedScalar range proof must survive"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_range_type_proof_kept_for_whole_function_scalar_rebuild() {
        // `g' = [x \in Proc |-> x]` is a whole-function rebuild whose range
        // element is a model-value scalar. The gate must NOT veto `g` (the
        // relaxation that keeps legitimate scalar-range function rewrites native).
        let registry = VarRegistry::from_names(["g"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let func_def = || Expr::FuncDef(vec![bound_var("x", ident("Proc"))], boxed(ident("x")));
        let init_expr = eq(ident("g"), func_def());
        let next_expr = eq(prime(ident("g")), func_def());

        let vetoed = vars_with_nonscalar_writers(
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            !vetoed.contains(&0),
            "a whole-function scalar rebuild must not veto the variable"
        );

        let type_ok = in_(ident("g"), func_set(ident("Proc"), ident("Proc")));
        let mut proofs = Vec::new();
        collect_fixed_scalar_range_type_proofs_with_ops(
            &type_ok,
            "TypeOK",
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        retain_writer_corroborated_fixed_scalar_range_proofs(
            &mut proofs,
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 0),
            "no over-rejection: whole-function scalar rebuild keeps its range proof"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_range_type_proof_dropped_for_whole_function_set_rebuild() {
        // `g' = [x \in Proc |-> Proc \ {x}]` rebuilds the whole function with a
        // SET range element — the gate MUST veto `g` (fail closed).
        let registry = VarRegistry::from_names(["g"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let init_expr = eq(
            ident("g"),
            Expr::FuncDef(vec![bound_var("x", ident("Proc"))], boxed(ident("x"))),
        );
        let next_expr = eq(
            prime(ident("g")),
            Expr::FuncDef(
                vec![bound_var("x", ident("Proc"))],
                boxed(set_minus(ident("Proc"), set_enum(vec![ident("x")]))),
            ),
        );

        let vetoed = vars_with_nonscalar_writers(
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            vetoed.contains(&0),
            "fail closed: a whole-function SET rebuild must veto the variable"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fixed_scalar_var_type_proof_dropped_for_set_bearing_writer() {
        // Top-level scalar var `v` whose TypeOK claims `v \in Proc` (model value),
        // but a writer assigns `v' = Proc \ {p1}` (a SET). The gate must drop the
        // var proof. A second var `w` is scalar-only and must survive.
        let registry = VarRegistry::from_names(["v", "w"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let (proof_domains, op_defs, op_replacements) = empty_op_env();

        let init_expr = and(
            in_(ident("v"), ident("Proc")),
            in_(ident("w"), ident("Proc")),
        );
        let next_expr = and(
            eq(
                prime(ident("v")),
                set_minus(ident("Proc"), set_enum(vec![ident("p1")])),
            ),
            eq(prime(ident("w")), ident("w")),
        );

        let type_ok = and(
            in_(ident("v"), ident("Proc")),
            in_(ident("w"), ident("Proc")),
        );
        let mut proofs = Vec::new();
        collect_fixed_scalar_var_type_proofs_with_ops(
            &type_ok,
            "TypeOK",
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 0),
            "TypeOK should produce a FixedScalar var proof for v before the gate"
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 1),
            "TypeOK should produce a FixedScalar var proof for w"
        );

        retain_writer_corroborated_fixed_scalar_var_proofs(
            &mut proofs,
            &init_expr,
            &next_expr,
            &registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
        );
        assert!(
            !proofs.iter().any(|p| p.var_idx == 0),
            "fail closed: a set-bearing top-level var's FixedScalar var proof must be dropped"
        );
        assert!(
            proofs.iter().any(|p| p.var_idx == 1),
            "no over-rejection: a scalar-only top-level var's proof must survive"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_tagged_scalar_set_range_writer_proof_rejects_unsafe_shapes() {
        let registry = VarRegistry::from_names(["k", "temp", "other"]);
        let init_expr = dijkstra_temp_init("Proc", "defaultInitValue");

        let mut other_proc_constants =
            dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        other_proc_constants.insert(
            intern_name("OtherProc"),
            model_set(vec![model_value("q1"), model_value("q2")]),
        );

        let write_temp = |value| {
            eq(
                prime(ident("temp")),
                except_update(ident("temp"), ident("self"), value),
            )
        };
        let preserve_k = || eq(prime(ident("k")), ident("k"));

        let unknown_writer_branch = Expr::Exists(
            vec![bound_var("self", ident("Proc"))],
            boxed(or(
                and(
                    write_temp(set_minus(ident("Proc"), set_enum(vec![ident("self")]))),
                    preserve_k(),
                ),
                preserve_k(),
            )),
        );
        assert!(
            collect_writer_proofs(
                &init_expr,
                &unknown_writer_branch,
                dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]),
                &registry,
            )
            .is_empty(),
            "a disjunct that does not prove a temp writer must not infer tagged slots"
        );

        let mismatched_universe = Expr::Exists(
            vec![bound_var("self", ident("Proc"))],
            boxed(and(
                write_temp(set_minus(ident("OtherProc"), set_enum(vec![ident("self")]))),
                preserve_k(),
            )),
        );
        assert!(
            collect_writer_proofs(
                &init_expr,
                &mismatched_universe,
                other_proc_constants,
                &registry,
            )
            .is_empty(),
            "set writers over a different universe must not infer tagged slots"
        );

        let store_var = Expr::Exists(
            vec![bound_var("self", ident("Proc"))],
            boxed(and(eq(prime(ident("temp")), ident("other")), preserve_k())),
        );
        assert!(
            collect_writer_proofs(
                &init_expr,
                &store_var,
                dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]),
                &registry,
            )
            .is_empty(),
            "unproven StoreVar-shaped writes must not infer tagged slots"
        );

        let int_domain_constants = dijkstra_constants(
            "Proc",
            vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)],
        );
        assert!(
            collect_writer_proofs(
                &init_expr,
                &dijkstra_temp_positive_next("Proc"),
                int_domain_constants,
                &registry,
            )
            .is_empty(),
            "integer domains without bounded nonnegative proof must remain untagged"
        );

        let broad_proc: Vec<Value> = (0..64)
            .map(|idx| model_value(&format!("p{idx:02}")))
            .collect();
        assert!(
            collect_writer_proofs(
                &init_expr,
                &dijkstra_temp_positive_next("Proc"),
                dijkstra_constants("Proc", broad_proc),
                &registry,
            )
            .is_empty(),
            "universes wider than the tagged 63-bit mask must remain untagged"
        );
    }

    fn collect_var_writer_proofs(
        init_expr: &Expr,
        next_expr: &Expr,
        constants: tla_core::kani_types::HashMap<NameId, Value>,
        registry: &VarRegistry,
    ) -> Vec<FixedScalarVarTypeProof> {
        let proof_domains = BTreeMap::new();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();
        let mut proofs = Vec::new();
        collect_fixed_scalar_var_writer_proofs_with_ops(
            init_expr,
            next_expr,
            "Init/Next writer proof",
            registry,
            &constants,
            &proof_domains,
            &op_defs,
            &op_replacements,
            &mut proofs,
        );
        proofs
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_g2_collect_fixed_scalar_var_writer_proof_for_dijkstra_k() {
        // DijkstraMutex `k \in Proc` is constrained ONLY in Init (no TypeOK
        // invariant). The single writer to `k` is `k' = self` where
        // `self \in Proc`, so `k`'s model-value universe is closed under Next.
        // The G2 writer collector must emit a top-level FixedScalar proof.
        let registry = VarRegistry::from_names(["k", "temp"]);
        let constants = dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]);
        let init_expr = dijkstra_temp_init("Proc", "defaultInitValue");
        let next_expr = dijkstra_temp_positive_next("Proc");

        let proofs = collect_var_writer_proofs(&init_expr, &next_expr, constants, &registry);

        assert_eq!(proofs.len(), 1, "exactly one proof, for `k`");
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 0, "var 0 is `k`");
        assert_eq!(proof.path, Vec::<SequenceCapacityPathStep>::new());
        assert_eq!(proof.scalar_type, SlotType::ModelValue);
        assert_eq!(
            proof.scalar_universe,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            ],
            "universe is the proven-closed Proc model-value set"
        );

        // End-to-end: a sampled `k = p1` state must infer a primary-flat
        // `FixedScalar` slot given the collected proof, and that layout must be
        // flat-primary safe. Use a `k`-only registry/state so the whole-layout
        // gate reflects `k` alone (`temp` in the real spec is a function, not a
        // scalar; modelling it here would only add an unrelated blocker).
        let k_only_registry = VarRegistry::from_names(["k"]);
        let state = ArrayState::from_values(vec![model_value("p1")]);
        let layout = infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
            &state,
            &k_only_registry,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            std::slice::from_ref(proof),
            &[],
        );
        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::FixedScalar { base, proof: p } => {
                assert_eq!(*base, SlotType::ModelValue);
                assert_eq!(p.scalar_type(), SlotType::ModelValue);
            }
            other => panic!("expected FixedScalar layout for `k`, got {other:?}"),
        }
        assert!(
            layout.supports_flat_primary(),
            "a proven-finite-universe ScalarModelValue var is flat-primary safe (G2)"
        );
        assert!(layout.supports_flat_bfs_auto_admission());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_g2_writer_proof_fails_closed_when_universe_not_closed_under_next() {
        // If `k` can be assigned a value outside its Init domain, the universe
        // is NOT proven closed and NO proof may be emitted — the unproven
        // sampled-scalar wall must stay closed.
        let registry = VarRegistry::from_names(["k", "temp"]);
        let init_expr = dijkstra_temp_init("Proc", "defaultInitValue");

        // Next writes `k' = other` where `other` is a free state var with no
        // proven model-value domain — escapes the universe.
        let escaping_next = Expr::Exists(
            vec![bound_var("self", ident("Proc"))],
            boxed(and(
                eq(prime(ident("k")), ident("other")),
                eq(prime(ident("temp")), ident("temp")),
            )),
        );
        assert!(
            collect_var_writer_proofs(
                &init_expr,
                &escaping_next,
                dijkstra_constants("Proc", vec![model_value("p1"), model_value("p2")]),
                &registry,
            )
            .is_empty(),
            "a writer that may escape the Init domain must not earn a closure proof"
        );

        // An Init constraint over an INTEGER domain is not a model-value
        // universe, so the model-value collector never fires (Int/Bool already
        // have dedicated primary-flat layouts).
        let int_init = and(
            in_(ident("k"), ident("Proc")),
            eq(ident("temp"), ident("temp")),
        );
        let int_next = Expr::Exists(
            vec![bound_var("self", ident("Proc"))],
            boxed(and(
                eq(prime(ident("k")), ident("self")),
                eq(prime(ident("temp")), ident("temp")),
            )),
        );
        assert!(
            collect_var_writer_proofs(
                &int_init,
                &int_next,
                dijkstra_constants(
                    "Proc",
                    vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)],
                ),
                &registry,
            )
            .is_empty(),
            "integer Init domains do not yield a model-value FixedScalar proof"
        );
    }

    // ====================================================================
    // Wavefront inference tests (Part of #3986)
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_single_state() {
        // Wavefront with one state should match single-state inference.
        let registry = VarRegistry::from_names(["x", "y"]);
        let state = ArrayState::from_values(vec![Value::SmallInt(1), Value::Bool(true)]);

        let single = infer_layout(&state, &registry);
        let wavefront = infer_layout_from_wavefront(&[state], &registry);

        assert_eq!(single.var_count(), wavefront.var_count());
        assert_eq!(single.total_slots(), wavefront.total_slots());
        for i in 0..single.var_count() {
            assert_eq!(
                single.var_layout(i).unwrap().kind,
                wavefront.var_layout(i).unwrap().kind,
                "var {i} layout kind mismatch between single and wavefront"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_consistent_states() {
        // Multiple states with the same structure should produce the same layout.
        let registry = VarRegistry::from_names(["x", "y"]);
        let states = vec![
            ArrayState::from_values(vec![Value::SmallInt(1), Value::Bool(true)]),
            ArrayState::from_values(vec![Value::SmallInt(2), Value::Bool(false)]),
            ArrayState::from_values(vec![Value::SmallInt(3), Value::Bool(true)]),
        ];

        let layout = infer_layout_from_wavefront(&states, &registry);

        assert_eq!(layout.var_count(), 2);
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Scalar
        ));
        assert!(matches!(
            layout.var_layout(1).unwrap().kind,
            VarLayoutKind::ScalarBool
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_int_array_consistent() {
        // Multiple states with compatible IntFunc should keep IntArray layout.
        let registry = VarRegistry::from_names(["arr"]);
        let mk_state = |a: i64, b: i64, c: i64| {
            let func = IntIntervalFunc::new(
                0,
                2,
                vec![Value::SmallInt(a), Value::SmallInt(b), Value::SmallInt(c)],
            );
            ArrayState::from_values(vec![Value::IntFunc(Rp::new(func))])
        };

        let states = vec![mk_state(1, 2, 3), mk_state(4, 5, 6), mk_state(7, 8, 9)];
        let layout = infer_layout_from_wavefront(&states, &registry);

        assert_eq!(layout.total_slots(), 3);
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::IntArray {
                lo: 0,
                len: 3,
                elements_are_bool: false,
                ..
            }
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_incompatible_downgrades() {
        // If one state has a Scalar but another has a different shape,
        // the wavefront should downgrade to Dynamic.
        let registry = VarRegistry::from_names(["x"]);
        let state_int = ArrayState::from_values(vec![Value::SmallInt(42)]);
        let state_bool = ArrayState::from_values(vec![Value::Bool(true)]);

        // SmallInt -> Scalar, Bool -> ScalarBool: these are incompatible.
        let layout = infer_layout_from_wavefront(&[state_int, state_bool], &registry);

        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "incompatible Scalar vs ScalarBool should downgrade to Dynamic, got {:?}",
            layout.var_layout(0).unwrap().kind
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_int_array_length_mismatch() {
        // IntArray with different lengths should downgrade to Dynamic.
        let registry = VarRegistry::from_names(["arr"]);
        let func3 = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)],
        );
        let func4 = IntIntervalFunc::new(
            0,
            3,
            vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
                Value::SmallInt(4),
            ],
        );

        let states = vec![
            ArrayState::from_values(vec![Value::IntFunc(Rp::new(func3))]),
            ArrayState::from_values(vec![Value::IntFunc(Rp::new(func4))]),
        ];
        let layout = infer_layout_from_wavefront(&states, &registry);

        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "IntArray with different lengths should downgrade to Dynamic, got {:?}",
            layout.var_layout(0).unwrap().kind
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_mixed_keeps_stable() {
        // A wavefront where some vars are stable and others are not.
        let registry = VarRegistry::from_names(["stable_int", "unstable"]);
        let states = vec![
            ArrayState::from_values(vec![Value::SmallInt(1), Value::SmallInt(10)]),
            ArrayState::from_values(vec![Value::SmallInt(2), Value::Bool(true)]),
        ];

        let layout = infer_layout_from_wavefront(&states, &registry);

        // stable_int: Scalar in both states -> stays Scalar
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Scalar
        ));
        // unstable: Scalar in first, ScalarBool in second -> Dynamic
        assert!(matches!(
            layout.var_layout(1).unwrap().kind,
            VarLayoutKind::Dynamic
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_infer_layout_from_wavefront_record_field_type_mismatch_downgrades() {
        let registry = VarRegistry::from_names(["msg"]);
        let string_state = ArrayState::from_values(vec![Value::Record(
            RecordValue::from_sorted_str_entries(vec![
                (Arc::from("kind"), Value::String(Rp::from("ready"))),
                (Arc::from("round"), Value::SmallInt(1)),
            ]),
        )]);
        let model_value_state = ArrayState::from_values(vec![Value::Record(
            RecordValue::from_sorted_str_entries(vec![
                (Arc::from("kind"), Value::ModelValue(Rp::from("ready"))),
                (Arc::from("round"), Value::SmallInt(2)),
            ]),
        )]);

        let layout = infer_layout_from_wavefront(&[string_state, model_value_state], &registry);

        assert!(
            matches!(layout.var_layout(0).unwrap().kind, VarLayoutKind::Dynamic),
            "record slot type mismatches must downgrade to Dynamic, got {:?}",
            layout.var_layout(0).unwrap().kind
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_dense_ordered_function_layout_normalizes_to_int_function() {
        let dense_function = FlatValueLayout::Function {
            domain: vec![
                FlatScalarValue::Int(2),
                FlatScalarValue::Int(3),
                FlatScalarValue::Int(4),
            ],
            value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
        };
        let observed = FlatValueLayout::IntFunction {
            lo: 2,
            len: 3,
            value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
        };

        let proven_applied =
            flat_layout_proof_apply_flat_layout(&dense_function, &observed).unwrap();
        assert_eq!(proven_applied, observed);

        let merged = merge_flat_value_layouts(&dense_function, &dense_function).unwrap();
        assert_eq!(merged, observed);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_dense_function_layout_normalization_requires_domain_order() {
        let wrong_order = FlatValueLayout::Function {
            domain: vec![
                FlatScalarValue::Int(2),
                FlatScalarValue::Int(4),
                FlatScalarValue::Int(3),
            ],
            value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
        };
        let ordered_int_function = FlatValueLayout::IntFunction {
            lo: 2,
            len: 3,
            value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
        };

        assert!(
            flat_layout_proof_apply_flat_layout(&wrong_order, &ordered_int_function).is_none(),
            "wrong-order generic function domains must not prove IntFunction layout"
        );

        let merged = merge_flat_value_layouts(&wrong_order, &wrong_order).unwrap();
        assert_eq!(merged, wrong_order);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layout_kinds_compatible_same() {
        assert!(layout_kinds_compatible(
            &VarLayoutKind::Scalar,
            &VarLayoutKind::Scalar
        ));
        assert!(layout_kinds_compatible(
            &VarLayoutKind::ScalarBool,
            &VarLayoutKind::ScalarBool
        ));
        assert!(layout_kinds_compatible(
            &VarLayoutKind::Dynamic,
            &VarLayoutKind::Dynamic
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layout_kinds_compatible_different_variant() {
        assert!(!layout_kinds_compatible(
            &VarLayoutKind::Scalar,
            &VarLayoutKind::ScalarBool
        ));
        assert!(!layout_kinds_compatible(
            &VarLayoutKind::Scalar,
            &VarLayoutKind::Dynamic
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layout_kinds_compatible_rejects_slot_type_mismatches() {
        assert!(!layout_kinds_compatible(
            &VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 0,
                len: 2,
                elements_are_bool: false,
                element_types: Some(vec![SlotType::String, SlotType::Int]),
            },
            &VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 0,
                len: 2,
                elements_are_bool: false,
                element_types: Some(vec![SlotType::ModelValue, SlotType::Int]),
            }
        ));

        assert!(!layout_kinds_compatible(
            &VarLayoutKind::Record {
                field_range_proofs: None,
                field_names: vec![Arc::from("kind")],
                field_is_bool: vec![false],
                field_types: vec![SlotType::String],
            },
            &VarLayoutKind::Record {
                field_range_proofs: None,
                field_names: vec![Arc::from("kind")],
                field_is_bool: vec![false],
                field_types: vec![SlotType::ModelValue],
            }
        ));

        assert!(!layout_kinds_compatible(
            &VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("kind")],
                domain_types: vec![SlotType::String],
                value_types: vec![SlotType::String],
                range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
            },
            &VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("kind")],
                domain_types: vec![SlotType::String],
                value_types: vec![SlotType::ModelValue],
                range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
            }
        ));
    }

    #[test]
    fn fixed_domain_sequence_layout_rejects_empty_proof_domain() {
        use tla_value::value::SeqValue;

        let registry = VarRegistry::from_names(["clock"]);
        let state =
            ArrayState::from_values(vec![Value::Seq(Rp::new(SeqValue::from_vec(Vec::new())))]);
        let proofs = vec![SequenceFixedDomainTypeProof {
            var_idx: 0,
            path: vec![],
            domain: Arc::from(Vec::<Value>::new().into_boxed_slice()),
            element_layout: SequenceTypeLayoutProof::Flat(FlatValueLayout::Scalar(SlotType::Int)),
            invariant: Arc::from("TypeOK"),
        }];

        let layout = infer_layout_with_sequence_layout_proofs(&state, &registry, &[], &[], &proofs);

        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout:
                    FlatValueLayout::Sequence {
                        bound,
                        max_len,
                        element_layout,
                    },
            } => {
                assert_eq!(*bound, SequenceBoundEvidence::Observed);
                assert_eq!(*max_len, 0);
                assert_eq!(**element_layout, FlatValueLayout::Scalar(SlotType::Int));
            }
            other => panic!("expected observed empty sequence layout, got {other:?}"),
        }
        assert!(!layout.supports_flat_primary());
    }

    // ---- Flat-primary sequence-element proof arms (MCBakery family) ----

    fn int_lit(value: i64) -> Expr {
        Expr::Int(value.into())
    }

    fn range(lo: Expr, hi: Expr) -> Expr {
        Expr::Range(boxed(lo), boxed(hi))
    }

    fn not(inner: Expr) -> Expr {
        Expr::Not(boxed(inner))
    }

    fn set_filter(name: &str, domain: Expr, pred: Expr) -> Expr {
        Expr::SetFilter(bound_var(name, domain), boxed(pred))
    }

    fn zero_arg_op(name: &str, body: Expr) -> std::sync::Arc<tla_core::ast::OperatorDef> {
        std::sync::Arc::new(tla_core::ast::OperatorDef {
            name: tla_core::span::Spanned::dummy(name.to_string()),
            params: Vec::new(),
            body: self::expr(body),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        })
    }

    fn int_constants(pairs: &[(&str, i64)]) -> tla_core::kani_types::HashMap<NameId, Value> {
        let mut constants = tla_core::kani_types::HashMap::default();
        for (name, value) in pairs {
            constants.insert(intern_name(name), Value::SmallInt(*value));
        }
        constants
    }

    // Arm 1: `int_range_domain` must resolve a binder domain through an
    // op-replacement (`Nat <- NatOverride`) and a zero-arity operator body
    // (`NatOverride == 0..MaxNat`) and recognize it as a finite int range.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn int_range_domain_resolves_op_replacement_chain_to_int_range() {
        let constants = int_constants(&[("MaxNat", 2)]);
        let mut op_defs = tla_core::OpEnv::default();
        op_defs.insert(
            "NatOverride".to_string(),
            zero_arg_op("NatOverride", range(int_lit(0), ident("MaxNat"))),
        );
        let mut op_replacements = tla_core::kani_types::HashMap::default();
        op_replacements.insert("Nat".to_string(), "NatOverride".to_string());

        // `\E k \in Nat` — Nat resolves to NatOverride == 0..MaxNat.
        assert!(int_range_domain(
            &ident("Nat"),
            &constants,
            &op_defs,
            Some(&op_replacements)
        ));
        // A literal `1..MaxNat` range is also recognized directly.
        assert!(int_range_domain(
            &range(int_lit(1), ident("MaxNat")),
            &constants,
            &op_defs,
            Some(&op_replacements)
        ));
    }

    // Arm 1: `int_range_domain` must recognize a `SetFilter` whose base domain is
    // a finite int range (`{j \in Nat : j > max}`), because a filtered subset of
    // an integer domain is still integer-typed.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn int_range_domain_recognizes_set_filter_over_int_base() {
        let constants = int_constants(&[("MaxNat", 2)]);
        let mut op_defs = tla_core::OpEnv::default();
        op_defs.insert(
            "NatOverride".to_string(),
            zero_arg_op("NatOverride", range(int_lit(0), ident("MaxNat"))),
        );
        let mut op_replacements = tla_core::kani_types::HashMap::default();
        op_replacements.insert("Nat".to_string(), "NatOverride".to_string());

        // `{j \in Nat : j > max}` — base resolves to a finite int range.
        let filter = set_filter(
            "j",
            ident("Nat"),
            Expr::Gt(boxed(ident("j")), boxed(ident("max"))),
        );
        assert!(int_range_domain(
            &filter,
            &constants,
            &op_defs,
            Some(&op_replacements)
        ));

        // A SetFilter over a NON-integer base (a model-value set) is NOT an int
        // range — fail closed.
        let mut model_constants = tla_core::kani_types::HashMap::default();
        model_constants.insert(
            intern_name("Procs"),
            model_set(vec![model_value("p1"), model_value("p2")]),
        );
        let non_int_filter = set_filter("j", ident("Procs"), bool_lit(true));
        assert!(!int_range_domain(
            &non_int_filter,
            &model_constants,
            &tla_core::OpEnv::default(),
            None
        ));
    }

    // Arm 1: an unresolvable / non-finite domain must fail closed.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn int_range_domain_fails_closed_for_unknown_domain() {
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        // A bare unresolved identifier is not a provable int range.
        assert!(!int_range_domain(
            &ident("UnknownSet"),
            &constants,
            &op_defs,
            None
        ));
    }

    // Arm 1 (regression): a `lo..hi` range with *non-constant*, state-dependent
    // bounds (e.g. `1..(Len(s)+1)`) is still an integer binder domain — a range
    // is always a set of integers regardless of whether the bounds fold to
    // constants. (Tightening this to require constant bounds previously broke the
    // growable writer-bounded sequence element-layout proof.)
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn int_range_domain_recognizes_non_constant_range() {
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        // `1 .. (Len(lineLengths) + 1)` — upper bound is state-dependent.
        let non_const_range = range(
            int_lit(1),
            Expr::Add(
                boxed(Expr::Apply(
                    boxed(ident("Len")),
                    vec![self::expr(ident("lineLengths"))],
                )),
                boxed(int_lit(1)),
            ),
        );
        assert!(int_range_domain(
            &non_const_range,
            &constants,
            &op_defs,
            None
        ));
    }

    // Arm 2: `writer_value_expr_layout` proves `~e` is `Scalar(Bool)` exactly
    // when the operand provably has Bool layout (e.g. `~flag[self]` where the
    // `flag` function range is Bool).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn writer_value_expr_layout_handles_boolean_negation() {
        let registry = VarRegistry::from_names(["flag"]);
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        let scope = WriterExprScope::default();
        let seed_slot_types = BTreeMap::new();
        // candidates[flag] = Scalar(Bool): the function range is boolean.
        let mut candidates = BTreeMap::new();
        candidates.insert(0usize, FlatValueLayout::Scalar(SlotType::Bool));

        // `~flag[self]` → Scalar(Bool).
        let neg_flag = not(func_apply(ident("flag"), ident("self")));
        let layout = writer_value_expr_layout(
            &neg_flag,
            &registry,
            &constants,
            &op_defs,
            None,
            &scope,
            &seed_slot_types,
            &candidates,
            &mut BTreeSet::new(),
        );
        assert_eq!(layout, Some(FlatValueLayout::Scalar(SlotType::Bool)));

        // `~TRUE` (literal bool) → Scalar(Bool).
        let neg_true = not(bool_lit(true));
        let layout_true = writer_value_expr_layout(
            &neg_true,
            &registry,
            &constants,
            &op_defs,
            None,
            &scope,
            &seed_slot_types,
            &candidates,
            &mut BTreeSet::new(),
        );
        assert_eq!(layout_true, Some(FlatValueLayout::Scalar(SlotType::Bool)));
    }

    // Arm 2: negation of a non-bool operand must fail closed (returns None),
    // never silently coerce a non-bool into a Bool slot.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn writer_value_expr_layout_negation_rejects_non_bool() {
        let registry = VarRegistry::from_names(["num"]);
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        let scope = WriterExprScope::default();
        let seed_slot_types = BTreeMap::new();
        let mut candidates = BTreeMap::new();
        candidates.insert(0usize, FlatValueLayout::Scalar(SlotType::Int));

        // `~num[self]` where num is an int function → not provably Bool → None.
        let neg_num = not(func_apply(ident("num"), ident("self")));
        let layout = writer_value_expr_layout(
            &neg_num,
            &registry,
            &constants,
            &op_defs,
            None,
            &scope,
            &seed_slot_types,
            &candidates,
            &mut BTreeSet::new(),
        );
        assert_eq!(layout, None);
    }

    fn set_range_proof(
        var_idx: usize,
        domain: Vec<Value>,
        universe: Vec<FlatScalarValue>,
        invariant: &str,
    ) -> SetBitmaskRangeTypeProof {
        SetBitmaskRangeTypeProof {
            var_idx,
            path: Vec::new(),
            domain: Arc::from(domain.into_boxed_slice()),
            set_universe: universe,
            invariant: Arc::from(invariant),
        }
    }

    fn scalar_range_proof(
        var_idx: usize,
        domain: Vec<Value>,
        scalar_type: SlotType,
        invariant: &str,
    ) -> FixedScalarRangeTypeProof {
        FixedScalarRangeTypeProof {
            var_idx,
            path: Vec::new(),
            domain: Arc::from(domain.into_boxed_slice()),
            scalar_type,
            scalar_universe: Vec::new(),
            invariant: Arc::from(invariant),
        }
    }

    // Arm 3: a `[1..N -> SUBSET universe]` range proof yields a proven-closed
    // SetBitmask sequence-element proof (covers empty-at-INIT: {} subset of any
    // universe).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn derive_set_valued_element_proof_from_subset_range() {
        let domain = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let universe = vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)];
        let proofs = vec![set_range_proof(5, domain, universe.clone(), "TypeOK")];
        let mut out = Vec::new();
        derive_set_valued_sequence_element_proofs(&proofs, &[], &mut out);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].var_idx, 5);
        assert!(out[0].path.is_empty());
        match &out[0].element_layout {
            FlatValueLayout::SetBitmask {
                universe: u,
                universe_closure,
            } => {
                assert_eq!(*u, universe);
                assert!(universe_closure.is_proven_closed());
            }
            other => panic!("expected SetBitmask element layout, got {other:?}"),
        }
    }

    // Arm 3 (nxt): a `[1..N -> 1..N]` scalar-int range proof yields a
    // Scalar(Int) sequence-element proof.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn derive_scalar_int_element_proof_from_scalar_range() {
        let domain = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let proofs = vec![scalar_range_proof(3, domain, SlotType::Int, "TypeOK")];
        let mut out = Vec::new();
        derive_set_valued_sequence_element_proofs(&[], &proofs, &mut out);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].var_idx, 3);
        assert_eq!(
            out[0].element_layout,
            FlatValueLayout::Scalar(SlotType::Int)
        );
    }

    // Arm 3: the SAME shape proven by two invariants (TypeOK and Inv) must dedup
    // to exactly one structural element proof — otherwise the downstream
    // uniqueness check would reject both as ambiguous.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn derive_set_valued_element_proof_dedups_multi_invariant() {
        let domain = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let universe = vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)];
        let proofs = vec![
            set_range_proof(5, domain.clone(), universe.clone(), "TypeOK"),
            set_range_proof(5, domain, universe, "Inv"),
        ];
        let mut out = Vec::new();
        derive_set_valued_sequence_element_proofs(&proofs, &[], &mut out);
        assert_eq!(out.len(), 1, "multi-invariant proofs must collapse to one");
    }

    // Arm 3: two STRUCTURALLY-conflicting proofs at the same location (different
    // universes) must fail closed — emit nothing rather than pick arbitrarily.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn derive_set_valued_element_proof_conflict_fails_closed() {
        let domain = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let proofs = vec![
            set_range_proof(5, domain.clone(), vec![FlatScalarValue::Int(1)], "TypeOK"),
            set_range_proof(
                5,
                domain,
                vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                "Inv",
            ),
        ];
        let mut out = Vec::new();
        derive_set_valued_sequence_element_proofs(&proofs, &[], &mut out);
        assert!(out.is_empty(), "conflicting universes must fail closed");
    }

    // Arm 3: a non-sequence-shaped domain (not a 1..N interval) fails closed —
    // the function is not stored as a sequence, so no sequence-element proof.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn derive_set_valued_element_proof_rejects_non_sequence_domain() {
        // Domain {0,1} is zero-based, not a one-based sequence interval.
        let domain = vec![Value::SmallInt(0), Value::SmallInt(1)];
        let universe = vec![FlatScalarValue::Int(1)];
        let proofs = vec![set_range_proof(5, domain, universe, "TypeOK")];
        let mut out = Vec::new();
        derive_set_valued_sequence_element_proofs(&proofs, &[], &mut out);
        assert!(out.is_empty(), "non-one-based domain must fail closed");
    }

    // Arm 3: a scalar STRING range stays fail-closed (one-word scalar slot would
    // overlap ordinary integer slots — only Int/Bool are admitted).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn derive_scalar_element_proof_rejects_string_range() {
        let domain = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let proofs = vec![scalar_range_proof(3, domain, SlotType::String, "TypeOK")];
        let mut out = Vec::new();
        derive_set_valued_sequence_element_proofs(&[], &proofs, &mut out);
        assert!(out.is_empty(), "string scalar range must fail closed");
    }

    // -----------------------------------------------------------------------
    // Nested-set (set-of-sets) two-level merge arm (nested-set discovery A4).
    // -----------------------------------------------------------------------

    fn nested_layout(
        inner_universe: Vec<FlatScalarValue>,
        outer_universe: Vec<u64>,
    ) -> FlatValueLayout {
        FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe,
            outer_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
                monitor_enforced: false,
            },
            inner_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
                monitor_enforced: false,
            },
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn nested_set_merge_unions_inner_and_rebases_outer_masks() {
        // Side A: inner universe {Int(0), Int(2)} (bit0=0, bit1=2); one piece
        // = {0, 2} → mask 0b11.
        let a = nested_layout(
            vec![FlatScalarValue::Int(0), FlatScalarValue::Int(2)],
            vec![0b11],
        );
        // Side B: inner universe {Int(1), Int(2)} (bit0=1, bit1=2); one piece
        // = {2} → mask 0b10.
        let b = nested_layout(
            vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
            vec![0b10],
        );

        let merged = merge_flat_value_layouts(&a, &b).expect("nested-set merge must succeed");
        let FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe,
            outer_closure,
            inner_closure,
        } = merged
        else {
            panic!("merge must yield a NestedSetBitmask");
        };
        // Merged inner universe = {0, 1, 2} (canonical sort+dedup), bit indices
        // 0→Int(0), 1→Int(1), 2→Int(2).
        assert_eq!(
            inner_universe,
            vec![
                FlatScalarValue::Int(0),
                FlatScalarValue::Int(1),
                FlatScalarValue::Int(2)
            ]
        );
        // A's piece {0,2} re-bases: old bit0(Int0)→new bit0, old bit1(Int2)→new
        // bit2 ⇒ 0b101. B's piece {2} re-bases: old bit1(Int2)→new bit2 ⇒
        // 0b100. Union+dedup, sorted = [0b100, 0b101].
        assert_eq!(outer_universe, vec![0b100, 0b101]);
        // Grown union ⇒ sampled (DynamicallyDiscovered) provenance, never proven.
        assert!(!outer_closure.is_proven_closed());
        assert!(!inner_closure.is_proven_closed());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn nested_set_single_value_inference_arm_stays_inert() {
        // A5/A6 promotes via the FROZEN multi-board monitor at the dedup hook,
        // NOT via the single-VALUE inference arm. That arm derives a universe
        // from ONE board (incomplete → next successor escapes), so it stays INERT
        // (returns None) regardless of the promotion gate (which is DEFAULT-OFF
        // in A6): the var's inferred layout remains Dynamic and the interpreter
        // generates successors. The arm's own `NESTED_SET_BITMASK_LAYOUT_INFERENCE`
        // const is the independent (still-false) Step-B seam.
        let inner = Value::Set(Rp::new(SortedSet::from_iter([
            Value::SmallInt(0),
            Value::SmallInt(1),
        ])));
        let board = SortedSet::from_iter([inner]);
        assert!(
            infer_nested_set_bitmask_layout(&board).is_none(),
            "single-value inference arm must stay inert (sound universe comes from the freeze path)"
        );
    }

    // ====================================================================
    // Duplicate-free bounded-universe sequence capacity proofs
    // ====================================================================

    fn df_clients() -> Vec<Value> {
        let mut clients = vec![model_value("c1"), model_value("c2"), model_value("c3")];
        clients.sort();
        clients
    }

    /// All permutation sequences of `elems` (as `Value::Tuple`s).
    fn df_permutations(elems: &[Value]) -> Vec<Value> {
        if elems.is_empty() {
            return vec![Value::Tuple(Rp::from(Vec::<Value>::new()))];
        }
        let mut out = Vec::new();
        for (i, head) in elems.iter().enumerate() {
            let mut rest = elems.to_vec();
            rest.remove(i);
            for tail in df_permutations(&rest) {
                let Value::Tuple(ref tail_elems) = tail else {
                    unreachable!()
                };
                let mut seq = vec![head.clone()];
                seq.extend(tail_elems.iter().cloned());
                out.push(Value::Tuple(Rp::from(seq)));
            }
        }
        out
    }

    fn df_set_elems(value: &Value) -> Vec<Value> {
        match value {
            Value::Set(set) => set.iter().cloned().collect(),
            _ => panic!("expected set"),
        }
    }

    /// `{c \in Clients : unsat[c] # {} /\ c \notin {sched[y] : y \in DOMAIN sched}}`
    fn df_to_schedule_filter(with_not_in_range: bool) -> Expr {
        let range_of_sched = Expr::SetBuilder(
            boxed(func_apply(ident("sched"), ident("y"))),
            vec![bound_var("y", Expr::Domain(boxed(ident("sched"))))],
        );
        let not_in_range = Expr::NotIn(boxed(ident("c")), boxed(range_of_sched));
        let unsat_nonempty = neq(func_apply(ident("unsat"), ident("c")), set_enum(vec![]));
        let pred = if with_not_in_range {
            and(unsat_nonempty, not_in_range)
        } else {
            unsat_nonempty
        };
        Expr::SetFilter(bound_var("c", ident("Clients")), boxed(pred))
    }

    /// `\E sq \in PermSeqs(<filter>) : sched' = sched \o sq` — the disjoint
    /// append. `PermSeqs` stays an opaque application; the certificate hook
    /// evaluates it.
    fn df_schedule_action(with_not_in_range: bool) -> Expr {
        let domain = Expr::Apply(
            boxed(ident("PermSeqs")),
            vec![expr(df_to_schedule_filter(with_not_in_range))],
        );
        Expr::Exists(
            vec![bound_var("sq", domain)],
            boxed(eq(
                prime(ident("sched")),
                Expr::Apply(
                    boxed(ident("\\o")),
                    vec![expr(ident("sched")), expr(ident("sq"))],
                ),
            )),
        )
    }

    /// `\E i \in DOMAIN sched : sched' = IF g THEN SubSeq(sched,1,i-1) \o
    /// SubSeq(sched,i+1,Len(sched)) ELSE sched` — the drop idiom.
    fn df_allocate_action() -> Expr {
        let drop = Expr::Apply(
            boxed(ident("\\o")),
            vec![
                expr(Expr::Apply(
                    boxed(ident("SubSeq")),
                    vec![
                        expr(ident("sched")),
                        expr(int_lit(1)),
                        expr(Expr::Sub(boxed(ident("i")), boxed(int_lit(1)))),
                    ],
                )),
                expr(Expr::Apply(
                    boxed(ident("SubSeq")),
                    vec![
                        expr(ident("sched")),
                        expr(Expr::Add(boxed(ident("i")), boxed(int_lit(1)))),
                        expr(Expr::Apply(boxed(ident("Len")), vec![expr(ident("sched"))])),
                    ],
                )),
            ],
        );
        let write = eq(
            prime(ident("sched")),
            Expr::If(
                boxed(neq(ident("S"), set_enum(vec![]))),
                boxed(drop),
                boxed(ident("sched")),
            ),
        );
        Expr::Exists(
            vec![bound_var("i", Expr::Domain(boxed(ident("sched"))))],
            boxed(write),
        )
    }

    fn df_hooks<'h>(
        eval_const_set: &'h dyn Fn(&Expr) -> Option<Vec<Value>>,
        eval_domain_with_set_arg: &'h dyn Fn(&Expr, &str, &Value) -> Option<Vec<Value>>,
        flatten_module_ref: &'h dyn Fn(&Expr) -> Option<std::rc::Rc<tla_core::span::Spanned<Expr>>>,
    ) -> DuplicateFreeSeqProofHooks<'h> {
        DuplicateFreeSeqProofHooks {
            flatten_module_ref,
            eval_const_set,
            eval_domain_with_set_arg,
        }
    }

    fn df_std_eval_const_set(expr: &Expr) -> Option<Vec<Value>> {
        matches!(expr, Expr::Ident(name, _) if name == "Clients").then(df_clients)
    }

    /// Certificate evaluator stub: requires the replaced domain to be
    /// `PermSeqs(<fresh>)` and returns the permutations of the argument set.
    fn df_std_eval_domain(expr: &Expr, name: &str, arg: &Value) -> Option<Vec<Value>> {
        let Expr::Apply(op, args) = expr else {
            return None;
        };
        if operator_ident_name(&op.node) != Some("PermSeqs") || args.len() != 1 {
            return None;
        }
        let Expr::Ident(arg_name, _) = &args[0].node else {
            return None;
        };
        if arg_name != name {
            return None;
        }
        Some(df_permutations(&df_set_elems(arg)))
    }

    fn df_no_flatten(_: &Expr) -> Option<std::rc::Rc<tla_core::span::Spanned<Expr>>> {
        None
    }

    fn df_collect(
        init: &Expr,
        next: &Expr,
        universe_proofs: &[SequenceUniverseProof],
        hooks: &DuplicateFreeSeqProofHooks,
        seed: Vec<SequenceCapacityProof>,
    ) -> Vec<SequenceCapacityProof> {
        let registry = VarRegistry::from_names(["unsat", "alloc", "sched"]);
        let constants = tla_core::kani_types::HashMap::default();
        let op_defs = tla_core::OpEnv::default();
        let op_replacements = tla_core::kani_types::HashMap::default();
        let mut out = seed;
        collect_duplicate_free_sequence_capacity_proofs(
            init,
            next,
            &registry,
            &constants,
            &op_defs,
            &op_replacements,
            universe_proofs,
            hooks,
            &mut out,
        );
        out
    }

    fn df_universe_proofs() -> Vec<SequenceUniverseProof> {
        vec![SequenceUniverseProof {
            var_idx: 2,
            universe: df_clients(),
            invariant: Arc::from("TypeInvariant"),
        }]
    }

    fn df_allocator_init() -> Expr {
        and(
            eq(ident("unsat"), ident("whatever")),
            eq(ident("sched"), Expr::Tuple(vec![])),
        )
    }

    fn df_allocator_next(with_not_in_range: bool) -> Expr {
        or(
            or(df_allocate_action(), df_schedule_action(with_not_in_range)),
            Expr::Unchanged(boxed(Expr::Tuple(vec![
                expr(ident("unsat")),
                expr(ident("alloc")),
                expr(ident("sched")),
            ]))),
        )
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_sequence_capacity_proof_allocator_shape() {
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let proofs = df_collect(
            &df_allocator_init(),
            &df_allocator_next(true),
            &df_universe_proofs(),
            &hooks,
            Vec::new(),
        );
        assert_eq!(proofs.len(), 1, "expected exactly one proof: {proofs:?}");
        let proof = &proofs[0];
        assert_eq!(proof.var_idx, 2);
        assert!(proof.path.is_empty());
        assert_eq!(proof.max_len, 3, "max_len must equal |Clients|");
        assert_eq!(proof.invariant.as_ref(), "TypeInvariant");
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_sequence_capacity_proof_supersedes_degenerate_zero_claim() {
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let degenerate = SequenceCapacityProof {
            var_idx: 2,
            path: Vec::new(),
            max_len: 0,
            invariant: Arc::from("Init/Next sequence writer proof"),
            heuristic: false,
        };
        let proofs = df_collect(
            &df_allocator_init(),
            &df_allocator_next(true),
            &df_universe_proofs(),
            &hooks,
            vec![degenerate],
        );
        assert_eq!(
            proofs.len(),
            1,
            "degenerate max_len=0 claim must be dropped: {proofs:?}"
        );
        assert_eq!(proofs[0].max_len, 3);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_sequence_capacity_proof_defers_to_existing_nonzero_proof() {
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let existing = SequenceCapacityProof {
            var_idx: 2,
            path: Vec::new(),
            max_len: 5,
            invariant: Arc::from("LenInvariant"),
            heuristic: false,
        };
        let proofs = df_collect(
            &df_allocator_init(),
            &df_allocator_next(true),
            &df_universe_proofs(),
            &hooks,
            vec![existing.clone()],
        );
        assert_eq!(
            proofs,
            vec![existing],
            "a pre-existing non-degenerate proof must win (uniqueness stays intact)"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_proof_fails_closed_without_not_in_range_conjunct() {
        // Same spec shape but the set-builder lacks `c \notin Range(sched)`:
        // the append is NOT provably disjoint — no proof may be emitted.
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let proofs = df_collect(
            &df_allocator_init(),
            &df_allocator_next(false),
            &df_universe_proofs(),
            &hooks,
            Vec::new(),
        );
        assert!(
            proofs.is_empty(),
            "missing distinctness conjunct must fail closed: {proofs:?}"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_proof_fails_closed_without_universe_invariant() {
        // No checked `sched \in Seq(Clients)` invariant: no universe, no proof.
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let proofs = df_collect(
            &df_allocator_init(),
            &df_allocator_next(true),
            &[],
            &hooks,
            Vec::new(),
        );
        assert!(proofs.is_empty());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_proof_fails_closed_when_certificate_sees_duplicates() {
        // Certificate evaluator returns a member sequence WITH a duplicate:
        // the domain is not permutation-shaped — must fail closed.
        let dup_domain = |expr: &Expr, name: &str, arg: &Value| -> Option<Vec<Value>> {
            df_std_eval_domain(expr, name, arg)?;
            let elems = df_set_elems(arg);
            if elems.len() >= 2 {
                Some(vec![Value::Tuple(Rp::from(vec![
                    elems[0].clone(),
                    elems[0].clone(),
                ]))])
            } else {
                Some(df_permutations(&elems))
            }
        };
        let hooks = df_hooks(&df_std_eval_const_set, &dup_domain, &df_no_flatten);
        let proofs = df_collect(
            &df_allocator_init(),
            &df_allocator_next(true),
            &df_universe_proofs(),
            &hooks,
            Vec::new(),
        );
        assert!(
            proofs.is_empty(),
            "duplicate-bearing domain members must fail the certificate: {proofs:?}"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_proof_fails_closed_on_unaccounted_prime() {
        // An extra `sched'` occurrence the structural walker cannot classify
        // (hidden under a negation) must poison the variable.
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let next = and(
            df_allocator_next(true),
            Expr::Not(boxed(eq(prime(ident("sched")), Expr::Tuple(vec![])))),
        );
        let proofs = df_collect(
            &df_allocator_init(),
            &next,
            &df_universe_proofs(),
            &hooks,
            Vec::new(),
        );
        assert!(
            proofs.is_empty(),
            "unaccounted prime occurrence must fail closed: {proofs:?}"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_proof_fails_closed_without_empty_init() {
        // Init writes a non-empty literal: the base case is not established.
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let init = eq(ident("sched"), Expr::Tuple(vec![expr(ident("c1"))]));
        let proofs = df_collect(
            &init,
            &df_allocator_next(true),
            &df_universe_proofs(),
            &hooks,
            Vec::new(),
        );
        assert!(proofs.is_empty());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_duplicate_free_proof_fails_closed_on_unresolvable_module_ref() {
        // A residual ModuleRef anywhere in Next poisons the analysis.
        let hooks = df_hooks(&df_std_eval_const_set, &df_std_eval_domain, &df_no_flatten);
        let next = or(
            df_allocator_next(true),
            Expr::ModuleRef(
                tla_core::ast::ModuleTarget::Named("Sched".to_string()),
                "Schedule".to_string(),
                Vec::new(),
            ),
        );
        let proofs = df_collect(
            &df_allocator_init(),
            &next,
            &df_universe_proofs(),
            &hooks,
            Vec::new(),
        );
        assert!(
            proofs.is_empty(),
            "unresolvable ModuleRef must poison the analysis: {proofs:?}"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_collect_sequence_universe_proofs_extracts_seq_membership() {
        let registry = VarRegistry::from_names(["unsat", "alloc", "sched"]);
        let op_replacements = tla_core::kani_types::HashMap::default();
        let invariant_body = and(
            in_(
                ident("sched"),
                Expr::Apply(boxed(ident("Seq")), vec![expr(ident("Clients"))]),
            ),
            in_(ident("unsat"), ident("whatever")),
        );
        let mut proofs = Vec::new();
        collect_sequence_universe_proofs(
            &invariant_body,
            "TypeInvariant",
            &registry,
            &op_replacements,
            &df_std_eval_const_set,
            &mut proofs,
        );
        assert_eq!(
            proofs,
            vec![SequenceUniverseProof {
                var_idx: 2,
                universe: df_clients(),
                invariant: Arc::from("TypeInvariant"),
            }]
        );
    }

    // TaggedScalarUnion universe assembly (Int ∪ model-value, Int ∪ string,
    // model-value ∪ string, Int ∪ Int-with-a-distinct-sentinel), plus dedup and
    // the empty-arms fail-closed guard. Pure helper — no env, no OpEnv.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn tagged_scalar_union_universe_assembly_covers_all_arm_type_combinations() {
        // Int ∪ model-value (btree `focus`/`lastOf` shape).
        let int_model = assemble_tagged_scalar_union_universe(
            vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
            vec![FlatScalarValue::ModelValue(std::sync::Arc::from("nil"))],
        )
        .expect("Int ∪ ModelValue universe");
        assert_eq!(int_model.len(), 3);
        assert!(int_model.contains(&FlatScalarValue::Int(1)));
        assert!(int_model.contains(&FlatScalarValue::ModelValue(std::sync::Arc::from("nil"))));

        // Int ∪ string.
        let int_string = assemble_tagged_scalar_union_universe(
            vec![FlatScalarValue::Int(0)],
            vec![
                FlatScalarValue::String(std::sync::Arc::from("ok")),
                FlatScalarValue::String(std::sync::Arc::from("error")),
            ],
        )
        .expect("Int ∪ String universe");
        assert_eq!(int_string.len(), 3);

        // model-value ∪ string (btree `ret`/`op` shape).
        let model_string = assemble_tagged_scalar_union_universe(
            vec![FlatScalarValue::ModelValue(std::sync::Arc::from("nil"))],
            vec![FlatScalarValue::String(std::sync::Arc::from("ok"))],
        )
        .expect("ModelValue ∪ String universe");
        assert_eq!(model_string.len(), 2);

        // Int ∪ Int with a distinct sentinel (`1..8 ∪ {-1}`) — a single scalar
        // lane, still a valid injective universe.
        let int_sentinel = assemble_tagged_scalar_union_universe(
            vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
            vec![FlatScalarValue::Int(-1)],
        )
        .expect("Int ∪ Int-sentinel universe");
        assert_eq!(int_sentinel.len(), 3);
        // Canonical order: sorted, so the sentinel -1 sorts first.
        assert_eq!(int_sentinel[0], FlatScalarValue::Int(-1));

        // Overlapping arms are deduplicated.
        let dedup = assemble_tagged_scalar_union_universe(
            vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
            vec![FlatScalarValue::Int(2), FlatScalarValue::Int(3)],
        )
        .expect("deduplicated universe");
        assert_eq!(dedup.len(), 3);

        // Two empty arms fail closed.
        assert!(assemble_tagged_scalar_union_universe(vec![], vec![]).is_none());
    }

    // End-to-end assembly from concrete type-set-expression arms: `1..3 ∪ {NIL}`
    // (with `NIL` a model-value constant) constructs a `TaggedScalarUnion` whose
    // universe covers both arms; an infinite arm (`Int`) fails closed.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn tagged_scalar_union_layout_from_union_arms_builds_shape_and_fails_closed() {
        let mut constants: tla_core::kani_types::HashMap<NameId, Value> =
            tla_core::kani_types::HashMap::default();
        constants.insert(intern_name("NIL"), Value::ModelValue(Rp::from("nil")));
        let op_defs = tla_core::OpEnv::default();
        let scope = LayoutScope::new();

        // `1..3 ∪ {NIL}` → TaggedScalarUnion over {Int(1), Int(2), Int(3), nil}.
        let mut visiting = BTreeSet::new();
        let layout = tagged_scalar_union_layout_from_union_arms(
            &range(int_lit(1), int_lit(3)),
            &set_enum(vec![ident("NIL")]),
            &constants,
            &op_defs,
            None,
            &scope,
            &mut visiting,
        )
        .expect("heterogeneous union constructs a TaggedScalarUnion");
        match layout {
            FlatValueLayout::TaggedScalarUnion { proof } => {
                assert_eq!(proof.universe().len(), 4);
                assert!(proof
                    .universe()
                    .contains(&FlatScalarValue::ModelValue(std::sync::Arc::from("nil"))));
                assert!(proof.universe().contains(&FlatScalarValue::Int(2)));
            }
            other => panic!("expected TaggedScalarUnion, got {other:?}"),
        }

        // An infinite / non-enumerable arm (`Int`) fails closed to None.
        let mut visiting = BTreeSet::new();
        assert!(tagged_scalar_union_layout_from_union_arms(
            &ident("Int"),
            &set_enum(vec![ident("NIL")]),
            &constants,
            &op_defs,
            None,
            &scope,
            &mut visiting,
        )
        .is_none());
    }

    // A heterogeneous finite scalar SET LITERAL (`{"get", "put", NIL}` — string ∪
    // model value, btree `op`) builds a `TaggedScalarUnion`; a homogeneous set
    // and a non-scalar element fail closed.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn tagged_scalar_union_layout_from_scalar_values_covers_set_literal() {
        let values = vec![
            Value::String(Rp::from("get")),
            Value::String(Rp::from("put")),
            Value::ModelValue(Rp::from("nil")),
        ];
        let layout = tagged_scalar_union_layout_from_scalar_values(&values)
            .expect("heterogeneous set literal builds a TaggedScalarUnion");
        match layout {
            FlatValueLayout::TaggedScalarUnion { proof } => {
                assert_eq!(proof.universe().len(), 3);
                assert!(proof
                    .universe()
                    .contains(&FlatScalarValue::String(std::sync::Arc::from("get"))));
                assert!(proof
                    .universe()
                    .contains(&FlatScalarValue::ModelValue(std::sync::Arc::from("nil"))));
            }
            other => panic!("expected TaggedScalarUnion, got {other:?}"),
        }

        // A non-scalar element (a set) fails closed.
        let with_set = vec![
            Value::String(Rp::from("get")),
            Value::Set(Rp::new(tla_value::value::SortedSet::from_iter([
                Value::SmallInt(1),
            ]))),
        ];
        assert!(tagged_scalar_union_layout_from_scalar_values(&with_set).is_none());
    }

    // The whole-variable override promotes an observed `ScalarModelValue` var to
    // `Recursive { TaggedScalarUnion }` when the sample fits the proven universe,
    // preserves the one-slot span, and fails closed when the sample is
    // out-of-universe.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn apply_tagged_scalar_union_var_overrides_promotes_fitting_scalar_var() {
        let registry = VarRegistry::from_names(["focus", "root"]);
        let universe = vec![
            FlatScalarValue::Int(1),
            FlatScalarValue::Int(2),
            FlatScalarValue::ModelValue(std::sync::Arc::from("nil")),
        ];
        let proof = TaggedScalarUnionProof::new(universe, Arc::from("TypeOk")).unwrap();
        let proofs = vec![TaggedScalarUnionVarTypeProof {
            var_idx: 0,
            proof: proof.clone(),
            invariant: Arc::from("TypeOk"),
        }];

        // focus sampled as `nil` (in universe) → promoted; root stays Scalar.
        let mut layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue, VarLayoutKind::Scalar],
        );
        let fits = vec![vec![Value::ModelValue(Rp::from("nil")), Value::SmallInt(1)]];
        apply_tagged_scalar_union_var_overrides(&mut layout, &proofs, &fits);
        assert!(matches!(
            layout.var_layout(0).unwrap().kind,
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::TaggedScalarUnion { .. }
            }
        ));
        assert_eq!(layout.var_layout(0).unwrap().slot_count, 1);
        assert_eq!(layout.var_layout(0).unwrap().offset, 0);
        assert_eq!(layout.var_layout(1).unwrap().offset, 1);
        assert!(matches!(
            layout.var_layout(1).unwrap().kind,
            VarLayoutKind::Scalar
        ));

        // A sample OUTSIDE the universe (`ModelValue("other")`) fails closed —
        // the var keeps its observed scalar kind.
        let mut layout2 = StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue, VarLayoutKind::Scalar],
        );
        let unfit = vec![vec![
            Value::ModelValue(Rp::from("other")),
            Value::SmallInt(1),
        ]];
        apply_tagged_scalar_union_var_overrides(&mut layout2, &proofs, &unfit);
        assert!(matches!(
            layout2.var_layout(0).unwrap().kind,
            VarLayoutKind::ScalarModelValue
        ));
    }
}
