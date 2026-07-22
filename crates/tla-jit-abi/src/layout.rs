// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compound state variable layout descriptors for the JIT/AOT ABI.
//!
//! The JIT state array is a flat `[i64]` where each slot holds one scalar
//! state variable. This module defines the **pure-data** layout descriptors
//! that describe how compound state variables (records, sequences, sets,
//! functions, tuples) map onto the flat buffer:
//!
//! 1. **`CompoundLayout`** — describes how a compound value maps onto a
//!    contiguous region of the flat state array.
//! 2. **`VarLayout`** — per-variable descriptor (scalar vs. compound).
//! 3. **`StateLayout`** — full description of all state variables.
//! 4. **`TAG_*` constants** — type tag words used in the self-describing
//!    serialized format.
//!
//! Only the pure-data structures live here. The serialization functions
//! (`serialize_value`, `deserialize_value`, `infer_layout`, `infer_var_layout`)
//! live in `compound_runtime` because they require `tla-value::Value`, which
//! transitively pulls in runtime machinery the leaf layout module does not
//! want.
//!
//! Part of #4267 (Wave 7d, epic #4251 Stage 2d): consolidated duplicate
//! compound-layout definitions into a single canonical ABI definition.
//!
//! # Wire format summary
//!
//! ```text
//! Record [a |-> 1, b |-> TRUE]:
//!   slot[0] = TAG_RECORD (1)
//!   slot[1] = 2 (field count)
//!   slot[2] = name_id("a") as i64
//!   slot[3] = TAG_INT (5)
//!   slot[4] = 1
//!   slot[5] = name_id("b") as i64
//!   slot[6] = TAG_BOOL (6)
//!   slot[7] = 1 (TRUE)
//!
//! Sequence <<3, 7>>:
//!   slot[0] = TAG_SEQ (2)
//!   slot[1] = 2 (length)
//!   slot[2] = TAG_INT (5)
//!   slot[3] = 3
//!   slot[4] = TAG_INT (5)
//!   slot[5] = 7
//! ```

use tla_core::NameId;

// ============================================================================
// Value type tags for the flat i64 representation
// ============================================================================

/// Type tag for a record value in the flat i64 state array.
pub const TAG_RECORD: i64 = 1;
/// Type tag for a sequence value.
pub const TAG_SEQ: i64 = 2;
/// Type tag for a set value (finite, enumerated).
pub const TAG_SET: i64 = 3;
/// Type tag for a function value.
pub const TAG_FUNC: i64 = 4;
/// Type tag for an integer scalar.
pub const TAG_INT: i64 = 5;
/// Type tag for a boolean scalar.
pub const TAG_BOOL: i64 = 6;
/// Type tag for a string value (stored as interned NameId).
pub const TAG_STRING: i64 = 7;
/// Type tag for a tuple value.
pub const TAG_TUPLE: i64 = 8;

// ============================================================================
// Compound layout descriptors
// ============================================================================

/// One element in a compact set-bitmask universe.
///
/// The order of this vector is ABI-significant: bit `i` in the compact
/// `i64` mask represents `universe[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SetBitmaskElement {
    /// Integer element.
    Int(i64),
    /// Boolean element.
    Bool(bool),
    /// String element, represented by its interned name id.
    String(NameId),
    /// Model value element, represented by its interned name id.
    ModelValue(NameId),
}

/// Scalar payload kind for compact one-slot range encodings.
///
/// This is separate from [`CompoundLayout::String`] because the legacy compact
/// representation stores strings and model values as interned `NameId`s, while
/// the tagged scalar-or-set proof identity must preserve which semantic lane is
/// allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarSlotKind {
    /// Integer scalar payload.
    Int,
    /// Boolean scalar payload.
    Bool,
    /// String scalar payload.
    String,
    /// Model value scalar payload.
    ModelValue,
}

/// Describes the layout of a single state variable in the JIT state array.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VarLayout {
    /// Scalar integer — occupies 1 i64 slot.
    ScalarInt,
    /// Scalar boolean — occupies 1 i64 slot (0 = false, 1 = true).
    ScalarBool,
    /// Compound value — occupies a variable number of i64 slots determined
    /// by the value's serialized form. The `CompoundLayout` descriptor
    /// provides the structure, but the actual slot count depends on the
    /// runtime value (e.g., sequence length, record field count).
    Compound(CompoundLayout),
}

/// Describes the expected structure of a compound state variable.
///
/// Used by the JIT to understand the memory layout of compound values
/// and to validate serialized data during deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompoundLayout {
    /// Record with known field names (sorted by NameId).
    /// Each field has its own layout descriptor.
    Record {
        /// (field_name_id, field_layout) pairs, sorted by NameId.
        fields: Vec<(NameId, CompoundLayout)>,
    },

    /// Function (domain -> range mapping).
    /// Stored as interleaved key-value pairs.
    Function {
        /// Layout of domain keys.
        key_layout: Box<CompoundLayout>,
        /// Layout of range values.
        value_layout: Box<CompoundLayout>,
        /// Number of key-value pairs when inferred from a concrete value.
        /// `None` when the cardinality is unknown (e.g., parsed from metadata).
        /// When `Some(n)`, `fixed_serialized_slots()` can compute the total
        /// size as `2 + n * (key_size + value_size)`.
        pair_count: Option<usize>,
        /// For integer-domain functions `[lo..hi -> T]`, the inclusive lower
        /// bound of the domain interval. When `Some(lo)` and `pair_count` is
        /// `Some(n)`, the function maps `lo..lo+n-1` to contiguous values.
        /// This enables O(1) direct-index lookup: `value_at(k) = slots[base + 2 + (k - lo) * pair_slots + key_slots]`.
        ///
        /// `None` for non-integer domains or non-contiguous keys.
        ///
        /// Part of #3985: Phase 2 compound layout wiring.
        domain_lo: Option<i64>,
    },

    /// Scalar key layout plus the exact ordered finite domain.
    ///
    /// This descriptor is intended for [`CompoundLayout::Function::key_layout`]
    /// when a compact function stores only range slots and keeps non-dense
    /// domain keys as metadata. The wrapped `key_layout` describes each key's
    /// scalar representation; `keys` records the canonical compact-slot order.
    ExplicitScalarDomain {
        /// Layout of each scalar key.
        key_layout: Box<CompoundLayout>,
        /// Exact finite domain in compact slot order.
        keys: Vec<SetBitmaskElement>,
    },

    /// Tuple key layout plus the exact ordered finite tuple domain.
    ///
    /// The tuple analogue of [`CompoundLayout::ExplicitScalarDomain`]: intended
    /// for [`CompoundLayout::Function::key_layout`] when a compact function is
    /// keyed by a fully-enumerated, static finite set of scalar tuples (e.g.
    /// btree's `[Nodes \X Keys -> T]`) and stores only range slots. The wrapped
    /// `key_layout` describes each key tuple's per-position scalar
    /// representation (a [`CompoundLayout::Tuple`]); `keys` records the exact
    /// domain in canonical compact-slot order.
    ///
    /// ABI-significant ordering: `keys[i]` is the domain key whose range value
    /// lives at compact slot `i`. The producer (the model checker's flat layout
    /// bridge) emits keys in the check-side canonical order — ascending
    /// `Value::cmp` over the tuple keys — and consumers must NOT re-sort:
    /// per-position sort orders (String lexicographic order vs interned-NameId
    /// order) are not recoverable from the carried elements alone (the H5
    /// String/ModelValue sort discipline).
    ExplicitTupleDomain {
        /// Per-position scalar layout of each key tuple.
        key_layout: Box<CompoundLayout>,
        /// Exact finite tuple domain in compact slot order; `keys[i][j]` is
        /// position `j` (0-indexed) of the `i`-th canonical key.
        keys: Vec<Vec<SetBitmaskElement>>,
    },

    /// Sequence of homogeneous or heterogeneous elements.
    Sequence {
        /// Layout of each element (all elements share this layout).
        element_layout: Box<CompoundLayout>,
        /// Number of elements when inferred from a concrete value.
        element_count: Option<usize>,
        /// True when `element_count` is a *proven upper bound* on the length of
        /// every reachable value of this sequence (backed by a checked
        /// source-level invariant / fixed-domain type proof), as opposed to a
        /// length merely observed from sampled states.
        ///
        /// Capacity-driven domain enumeration (e.g. expanding
        /// `\E i \in 1..Len(seq)` into compile-time guarded candidates `1..C`)
        /// is only sound when this is `true`: an observed bound could be
        /// exceeded by a later reachable state, which would silently drop
        /// successors. Defaults to `false` (fail-closed) everywhere except the
        /// authoritative flat-layout bridge that carries the proof evidence.
        capacity_proven: bool,
    },

    /// Finite enumerated set.
    Set {
        /// Layout of each element.
        element_layout: Box<CompoundLayout>,
        /// Number of elements when inferred from a concrete value.
        element_count: Option<usize>,
    },

    /// Compact finite scalar set encoded as one raw `i64` bitmask slot.
    ///
    /// This is distinct from [`CompoundLayout::Set`], which describes the
    /// materialized self-describing set ABI (`TAG_SET`, count, elements).
    SetBitmask {
        /// Exact finite universe in canonical bit-index order.
        universe: Vec<SetBitmaskElement>,
        /// True when the source-level type invariant proved this universe is
        /// closed under every successor write (e.g. a TypeOK `SUBSET {0,1}`
        /// range). The model-checker side carries this as
        /// `SetBitmaskUniverseClosure::ProvenClosed`; it is collapsed to a bool
        /// here because the ABI does not retain the `invariant` source string.
        ///
        /// ABI-significant for *function-range* slots: only a proven-closed
        /// universe may be stored canonically into the native flat-primary i64
        /// buffer, because a successor write outside a merely sampled universe
        /// would silently corrupt the fixed-width bitmask slot. A round trip
        /// through the ABI must preserve this bit so the model checker can
        /// re-admit (or fail-close) the function-range flat-primary path.
        is_proven_closed: bool,
    },

    /// Compact finite *record* set encoded as a fixed-width multi-slot `i64`
    /// bitmask over a finite, provably/monitored-closed record universe.
    ///
    /// This is the record analogue of [`CompoundLayout::SetBitmask`]: a set
    /// variable (or function range) whose elements are records drawn from a
    /// finite universe, packed so bit `i` (across `slot_count` i64 slots, low
    /// slot first) means "universe record `i` is present". The model-checker
    /// side is `FlatValueLayout::RecordSetBitmask`; this carrier transports the
    /// universe to native code so the byte-exact `set_ops` RecordSetBitmask
    /// lowering (membership / union / diff) becomes reachable for a state var —
    /// without it the bridge mapped the slot to `Dynamic`, the lowering was
    /// never fed an `AggregateShape::RecordSetBitmask`, and a compiled action
    /// would `IntToPtr`-deref the packed mask (rc=139).
    ///
    /// Each universe record is a list of `(field_name, scalar_value)` pairs in
    /// any order; the consumer canonicalizes to the same field-name-string order
    /// the interpreter's record universe uses, so bit indices agree exactly.
    RecordSetBitmask {
        /// Universe records in canonical (sorted, deduped) bit-index order. Bit
        /// `i` maps to `universe[i]`. Each record is its scalar field tuple.
        universe: Vec<Vec<(NameId, SetBitmaskElement)>>,
        /// `ceil(universe.len() / 64)` — the fixed i64 slot width.
        slot_count: usize,
        /// True when the source-level type invariant (or the monitored
        /// write-barrier) proved the universe closed under every successor
        /// write. Only a closed universe is sound as a flat-primary slot.
        is_proven_closed: bool,
    },

    /// Compact one-slot `scalar | finite-set` value encoded with an explicit
    /// sign tag.
    ///
    /// This descriptor is intended for [`CompoundLayout::Function::value_layout`]
    /// when the function's range can be either a scalar sentinel or a compact
    /// set bitmask over `set_universe`. The containing function's
    /// [`CompoundLayout::ExplicitScalarDomain`] carries the key order; this
    /// variant carries the range universe order and proof source identity.
    TaggedScalarOrSet {
        /// Semantic scalar lane accepted by non-negative tagged slots.
        scalar_kind: ScalarSlotKind,
        /// Exact finite set universe in canonical bit-index order.
        set_universe: Vec<SetBitmaskElement>,
        /// Interned proof/source identity that justified this range encoding.
        proof_source: NameId,
    },

    /// Compact one-slot finite *scalar union* (`scalar | scalar`, e.g.
    /// `Nodes \cup {NIL}`) encoded as a single typed universe-INDEX slot.
    ///
    /// This is the scalar-union sibling of [`Self::TaggedScalarOrSet`]. The
    /// range/var value is one of a finite, deduplicated, ordered `universe`, and
    /// the compact slot stores the value's universe INDEX
    /// (`universe.position(value)`), never the raw scalar payload. Domain-index
    /// encoding keeps `Int(1)`, `String(name_id=1)` and `ModelValue(name_id=1)`
    /// in DISTINCT slots even though the three legacy scalar lanes would
    /// otherwise collapse to the same compact `i64`.
    ///
    /// ABI-significant ordering: `universe[i]` is the value whose slot index is
    /// `i`. The producer (the model checker's flat layout bridge) emits the
    /// universe in ty's sorted `FlatScalarValue` assembly order. Because
    /// `FlatScalarValue` derives `Ord` with the `Int` variant first, every `Int`
    /// member forms a contiguous ascending prefix at index base 0 when the ints
    /// are consecutive — the property the native `(v - lo) + base` range
    /// encoding relies on. Consumers must NOT re-sort.
    TaggedScalarUnion {
        /// Exact finite union universe in canonical index order (`universe[i]`
        /// has slot index `i`). Ordered, deduplicated, collision-free.
        universe: Vec<SetBitmaskElement>,
        /// Interned proof/source identity that justified this union encoding.
        proof_source: NameId,
    },

    /// Finite tagged union of *heterogeneous* shapes (a TLA+ sum type) encoded
    /// as `1 + max_payload` slots.
    ///
    /// This is the sum-type sibling of [`Self::TaggedScalarUnion`]: where that
    /// carrier folds a union of scalar LANES into one index slot, this one
    /// carries a union of whole SHAPES — btree's `args`, which is the model
    /// value `NIL` or a fixed-arity tuple `<<k>>` / `<<k,v>>`.
    ///
    /// Layout:
    ///   * slot 0 — the **tag**: the index of the active variant in `variants`.
    ///   * slots `1..=max_payload` — the active variant's own compact encoding,
    ///     occupying its leading `variants[tag].compact_slot_count()` slots.
    ///     Every trailing payload slot the active variant does not use is
    ///     canonically **zero**.
    ///
    /// ABI-significant ordering: `variants[i]` is the shape selected by tag `i`.
    /// The producer (the model checker's flat layout bridge) emits the variants
    /// in the proof's canonical tag order; consumers must NOT re-order. A
    /// consumer that cannot prove which variant is live for a given access MUST
    /// fail closed rather than guess a payload offset — the payload slots carry
    /// no self-describing type tag.
    TaggedUnion {
        /// Variant shapes in canonical tag order (`variants[i]` ⇔ tag `i`).
        /// At least two, pairwise distinct.
        variants: Vec<CompoundLayout>,
        /// Payload slot count: the widest variant's compact slot count. The
        /// total carrier width is `1 + max_payload_slots`.
        max_payload_slots: usize,
        /// Interned proof/source identity that justified this union encoding.
        proof_source: NameId,
    },

    /// A topological adjacency list representing the graph structure of a compound
    /// state variable. Used to derive and track structural symmetries and canonicalize
    /// states within verified orbits.
    TopologicalAdjacencyList {
        /// Layout of the nodes in the adjacency graph.
        node_layout: Box<CompoundLayout>,
        /// Adjacency mapping (e.g., node -> list of edges).
        edges: Vec<(NameId, Vec<NameId>)>,
    },

    /// Tuple with known arity and per-position layouts.
    Tuple {
        /// Layout of each position (1-indexed in TLA+, stored 0-indexed).
        element_layouts: Vec<CompoundLayout>,
    },

    /// Scalar integer leaf — no compound structure.
    Int,

    /// Scalar boolean leaf — no compound structure.
    Bool,

    /// String leaf — serialized as its interned NameId (u32 as i64).
    String,

    /// Dynamic (type-tagged) — the actual type is encoded inline via
    /// a tag word. Used for heterogeneous collections where the element
    /// type is not statically known.
    Dynamic,
}

impl CompoundLayout {
    /// Compute the compact no-tag slot count for tla-check flat buffers.
    ///
    /// This is distinct from [`Self::fixed_serialized_slots`], which counts the
    /// self-describing tagged wire format. Compact flat buffers store only the
    /// mutable scalar payload slots: function domain keys, record field names,
    /// and aggregate type/count tags are layout metadata, not buffer contents.
    ///
    /// Layouts without a fixed compact representation occupy one placeholder
    /// slot, matching tla-check's `Dynamic` compact layout.
    #[must_use]
    pub fn compact_slot_count(&self) -> usize {
        match self {
            CompoundLayout::Int | CompoundLayout::Bool | CompoundLayout::String => 1,
            CompoundLayout::ExplicitScalarDomain { key_layout, .. }
            | CompoundLayout::ExplicitTupleDomain { key_layout, .. } => {
                key_layout.compact_slot_count()
            }
            CompoundLayout::Function {
                pair_count: Some(n),
                value_layout,
                ..
            } => *n * value_layout.compact_slot_count(),
            CompoundLayout::Record { fields } => fields
                .iter()
                .map(|(_, field_layout)| field_layout.compact_slot_count())
                .sum(),
            CompoundLayout::Tuple { element_layouts } => element_layouts
                .iter()
                .map(CompoundLayout::compact_slot_count)
                .sum(),
            CompoundLayout::Sequence {
                element_layout,
                element_count: Some(n),
                ..
            } => 1 + *n * element_layout.compact_slot_count(),
            CompoundLayout::SetBitmask { .. }
            | CompoundLayout::TaggedScalarOrSet { .. }
            | CompoundLayout::TaggedScalarUnion { .. } => 1,
            CompoundLayout::RecordSetBitmask { slot_count, .. } => *slot_count,
            // Tag slot + the widest variant's payload. Mirrors the check-side
            // `TaggedUnionProof::slot_count`.
            CompoundLayout::TaggedUnion {
                max_payload_slots, ..
            } => 1 + *max_payload_slots,
            CompoundLayout::TopologicalAdjacencyList { .. } => 1,
            CompoundLayout::Set { .. }
            | CompoundLayout::Function {
                pair_count: None, ..
            }
            | CompoundLayout::Sequence {
                element_count: None,
                ..
            }
            | CompoundLayout::Dynamic => 1,
        }
    }

    /// Compute the fixed serialized size in i64 slots, if statically known.
    ///
    /// Returns `Some(n)` when the entire compound value has a fixed, predictable
    /// serialized size. Returns `None` for dynamic or variable-length layouts.
    ///
    /// Scalar leaves: TAG + value = 2 slots.
    /// Records: TAG + field_count + sum(name_id + field_serialized_size per field).
    /// Tuples: TAG + elem_count + sum(elem_serialized_size per element).
    /// Functions: TAG + pair_count + sum(key_size + value_size per pair).
    ///   When `pair_count` is `Some(n)` (inferred from a concrete value),
    ///   the total is `2 + n * (key_slots + value_slots)`.
    /// Sequences/Sets: TAG + count + n * element_slots when count is known.
    #[must_use]
    pub fn fixed_serialized_slots(&self) -> Option<usize> {
        match self {
            CompoundLayout::Int | CompoundLayout::Bool | CompoundLayout::String => Some(2),
            CompoundLayout::ExplicitScalarDomain { key_layout, .. }
            | CompoundLayout::ExplicitTupleDomain { key_layout, .. } => {
                key_layout.fixed_serialized_slots()
            }
            CompoundLayout::Record { fields } => {
                let mut total = 2; // TAG_RECORD + field_count
                for (_, field_layout) in fields {
                    total += 1; // name_id slot
                    total += field_layout.fixed_serialized_slots()?;
                }
                Some(total)
            }
            CompoundLayout::Tuple { element_layouts } => {
                let mut total = 2; // TAG_TUPLE + elem_count
                for elem_layout in element_layouts {
                    total += elem_layout.fixed_serialized_slots()?;
                }
                Some(total)
            }
            CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                ..
            } => {
                let n = (*pair_count)?;
                if n == 0 {
                    return Some(2); // TAG + count header only
                }
                let key_slots = key_layout.fixed_serialized_slots()?;
                let value_slots = value_layout.fixed_serialized_slots()?;
                Some(2 + n * (key_slots + value_slots))
            }
            CompoundLayout::Sequence {
                element_layout,
                element_count,
                ..
            } => {
                let n = (*element_count)?;
                if n == 0 {
                    return Some(2); // TAG + count header only
                }
                let elem_slots = element_layout.fixed_serialized_slots()?;
                Some(2 + n * elem_slots)
            }
            CompoundLayout::Set {
                element_layout,
                element_count,
            } => {
                let n = (*element_count)?;
                if n == 0 {
                    return Some(2); // TAG + count header only
                }
                let elem_slots = element_layout.fixed_serialized_slots()?;
                Some(2 + n * elem_slots)
            }
            CompoundLayout::SetBitmask { .. }
            | CompoundLayout::TaggedScalarOrSet { .. }
            | CompoundLayout::TaggedScalarUnion { .. } => Some(1),
            // A raw fixed-width multi-slot bitmask: no self-describing tag, the
            // serialized width is the compact width (`slot_count`).
            CompoundLayout::RecordSetBitmask { slot_count, .. } => Some(*slot_count),
            // A tag slot plus a fixed-width payload window: no self-describing
            // tag, so the serialized width is the compact width. The window is
            // sized for the WIDEST variant and zero-filled beyond the active
            // variant's own slots, which is what makes the width fixed even
            // though the live variant varies per state.
            CompoundLayout::TaggedUnion {
                max_payload_slots, ..
            } => Some(1 + *max_payload_slots),
            CompoundLayout::TopologicalAdjacencyList { .. } => None,
            CompoundLayout::Dynamic => None,
        }
    }

    /// Check if this is a scalar leaf type (Int, Bool, or String).
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            CompoundLayout::Int | CompoundLayout::Bool | CompoundLayout::String
        ) || matches!(
            self,
            CompoundLayout::ExplicitScalarDomain { key_layout, .. } if key_layout.is_scalar()
        )
    }

    /// Check if this function has a contiguous integer domain enabling O(1)
    /// direct-index lookup instead of O(n) linear scan.
    ///
    /// Returns `Some((lo, len))` when the function maps `[lo..lo+len-1] -> T`
    /// with scalar keys and known pair count.
    ///
    /// Part of #3985: Phase 2 compound layout wiring.
    #[must_use]
    pub fn int_array_bounds(&self) -> Option<(i64, usize)> {
        match self {
            CompoundLayout::Function {
                key_layout,
                pair_count: Some(n),
                domain_lo: Some(lo),
                ..
            } if key_layout.is_scalar() && *n > 0 => Some((*lo, *n)),
            _ => None,
        }
    }
}

impl VarLayout {
    /// Compute the compact no-tag slot count for this variable.
    ///
    /// Scalar variables occupy one raw `i64` slot. Compound variables use
    /// [`CompoundLayout::compact_slot_count`].
    #[must_use]
    pub fn compact_slot_count(&self) -> usize {
        match self {
            VarLayout::ScalarInt | VarLayout::ScalarBool => 1,
            VarLayout::Compound(layout) => layout.compact_slot_count(),
        }
    }
}

/// Describes the layout of the full state vector (all state variables).
///
/// Maps each state variable index to its layout descriptor and its
/// starting offset in the flat i64 array.
#[derive(Clone)]
pub struct StateLayout {
    /// Per-variable layout descriptors, in VarIdx order.
    vars: Vec<VarLayout>,
    /// Hybrid flat-view compile mode (wishlist item 4 M0).
    ///
    /// When `true`, this layout describes ty's **hybrid** flat view: every
    /// [`CompoundLayout::Dynamic`] variable is an inert 1-slot placeholder
    /// for a compound value that lives ONLY in the checker's compound parent
    /// state — the slot content is meaningless and must never be read or
    /// written by native code. Lowering enforces this by declining any
    /// `LoadVar`/`LoadPrime`/`StoreVar` of a `Dynamic`-mapped variable.
    ///
    /// When `false` (all pre-existing layouts), `Dynamic` keeps its historical
    /// whole-state-buffer meaning (type-tagged inline / serialized-tail
    /// encoding reachable through the handle bridge).
    hybrid_flat_view: bool,
}

/// Manual `Debug`: byte-identical to the historical derived output for
/// non-hybrid layouts (artifact/warm-cache identities hash this string, so
/// whole-state cache identity must not drift), while hybrid layouts carry an
/// explicit marker — which also namespaces every layout-derived cache identity
/// so hybrid and whole-state artifacts can never cross.
impl std::fmt::Debug for StateLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.hybrid_flat_view {
            f.debug_struct("StateLayout")
                .field("vars", &self.vars)
                .field("hybrid_flat_view", &self.hybrid_flat_view)
                .finish()
        } else {
            f.debug_struct("StateLayout")
                .field("vars", &self.vars)
                .finish()
        }
    }
}

impl StateLayout {
    /// Create a new state layout from variable descriptors.
    pub fn new(vars: Vec<VarLayout>) -> Self {
        StateLayout {
            vars,
            hybrid_flat_view: false,
        }
    }

    /// Mark this layout as ty's hybrid flat view (see the field docs).
    #[must_use]
    pub fn with_hybrid_flat_view(mut self) -> Self {
        self.hybrid_flat_view = true;
        self
    }

    /// Whether this layout is a hybrid flat view whose `Dynamic` variables are
    /// inert placeholders that native code must never access.
    #[must_use]
    pub fn is_hybrid_flat_view(&self) -> bool {
        self.hybrid_flat_view
    }

    /// Get the number of state variables.
    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    /// Get the layout for a specific variable.
    pub fn var_layout(&self, idx: usize) -> Option<&VarLayout> {
        self.vars.get(idx)
    }

    /// Get a mutable reference to the layout for a specific variable.
    pub fn var_layout_mut(&mut self, idx: usize) -> Option<&mut VarLayout> {
        self.vars.get_mut(idx)
    }

    /// Check if all variables are scalar (legacy flat i64 layout).
    pub fn is_all_scalar(&self) -> bool {
        self.vars
            .iter()
            .all(|v| matches!(v, VarLayout::ScalarInt | VarLayout::ScalarBool))
    }

    /// Iterate over all variable layouts.
    pub fn iter(&self) -> impl Iterator<Item = &VarLayout> {
        self.vars.iter()
    }

    /// Compute the starting slot offset for each variable in the tagged
    /// serialized i64 array.
    ///
    /// Do not use this for active compact state buffers. Compact paths must
    /// use compact layout metadata because compound values omit tags, counts,
    /// and record field-name slots there.
    ///
    /// Returns a vector where `offsets[i]` is `Some(offset)` for variables
    /// whose starting position can be determined at compile time, or `None`
    /// for variables that come after a dynamic-size compound variable.
    ///
    /// Scalar variables occupy 1 slot. Compound variables with fixed
    /// serialized size occupy their `fixed_serialized_slots()` count.
    /// Once a variable with dynamic size is encountered, all subsequent
    /// variables get `None` (their offsets cannot be computed statically).
    #[must_use]
    pub fn compute_var_offsets(&self) -> Vec<Option<usize>> {
        let mut offsets = Vec::with_capacity(self.vars.len());
        let mut current: Option<usize> = Some(0);
        for var in &self.vars {
            offsets.push(current);
            if let Some(cur) = current {
                match var {
                    VarLayout::ScalarInt | VarLayout::ScalarBool => {
                        current = Some(cur + 1);
                    }
                    VarLayout::Compound(layout) => {
                        current = layout.fixed_serialized_slots().map(|s| cur + s);
                    }
                }
            }
        }
        offsets
    }

    /// Compute starting slot offsets for each variable in the compact no-tag
    /// flat buffer.
    ///
    /// Unlike [`Self::compute_var_offsets`], this always returns concrete
    /// offsets because dynamic or unsupported compact layouts occupy one
    /// placeholder slot.
    #[must_use]
    pub fn compute_compact_var_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.vars.len());
        let mut current = 0;
        for var in &self.vars {
            offsets.push(current);
            current += var.compact_slot_count();
        }
        offsets
    }

    /// Return the compact no-tag slot range occupied by one state variable.
    ///
    /// This is the shared contract native backends should use when translating
    /// source-level variable metadata, such as [`crate::ActionDescriptor`]
    /// `write_vars`, into flat compact-buffer slots.
    #[must_use]
    pub fn compact_var_slot_range(&self, idx: usize) -> Option<std::ops::Range<usize>> {
        let layout = self.var_layout(idx)?;
        let offset = self
            .vars
            .iter()
            .take(idx)
            .map(VarLayout::compact_slot_count)
            .sum::<usize>();
        let slot_count = layout.compact_slot_count();
        Some(offset..offset + slot_count)
    }

    /// Return one TLA+ state-variable slot descriptor for compact native kernels.
    ///
    /// Returns `None` when `idx` is out of range or the compact offset/count
    /// cannot be represented in the stable `u32` ABI descriptor.
    #[must_use]
    pub fn compact_var_kernel_slot(&self, idx: usize) -> Option<crate::KernelStateSlot> {
        let range = self.compact_var_slot_range(idx)?;
        let ordinal = u32::try_from(idx).ok()?;
        let offset = u32::try_from(range.start).ok()?;
        let slot_count = u32::try_from(range.end.checked_sub(range.start)?).ok()?;
        Some(crate::KernelStateSlot::tla_state_var(
            ordinal, offset, slot_count,
        ))
    }

    /// Canonicalize TLA+ variable write metadata into compact kernel slot ranges.
    ///
    /// Input variable indexes are sorted and deduplicated before mapping so
    /// downstream code receives stable slot descriptors regardless of bytecode
    /// instruction order. Returns `None` if any variable index is invalid.
    #[must_use]
    pub fn compact_write_kernel_slots(
        &self,
        write_vars: &[u16],
    ) -> Option<Vec<crate::KernelStateSlot>> {
        let mut vars = write_vars.to_vec();
        vars.sort_unstable();
        vars.dedup();
        vars.into_iter()
            .map(|idx| self.compact_var_kernel_slot(usize::from(idx)))
            .collect()
    }

    /// Total compact no-tag slot count for this state layout.
    #[must_use]
    pub fn compact_slot_count(&self) -> usize {
        self.vars.iter().map(VarLayout::compact_slot_count).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_values_stable() {
        // The tag constants are part of the wire format — do not change these
        // values without coordinating with every serializer/deserializer in
        // trust-codegen and the model-checker fallback.
        assert_eq!(TAG_RECORD, 1);
        assert_eq!(TAG_SEQ, 2);
        assert_eq!(TAG_SET, 3);
        assert_eq!(TAG_FUNC, 4);
        assert_eq!(TAG_INT, 5);
        assert_eq!(TAG_BOOL, 6);
        assert_eq!(TAG_STRING, 7);
        assert_eq!(TAG_TUPLE, 8);
    }

    #[test]
    fn test_compound_layout_scalar_slot_count() {
        assert_eq!(CompoundLayout::Int.fixed_serialized_slots(), Some(2));
        assert_eq!(CompoundLayout::Bool.fixed_serialized_slots(), Some(2));
        assert_eq!(CompoundLayout::String.fixed_serialized_slots(), Some(2));
        assert!(CompoundLayout::Int.is_scalar());
        assert!(!CompoundLayout::Dynamic.is_scalar());
    }

    #[test]
    fn test_compound_layout_dynamic_has_no_fixed_size() {
        assert_eq!(CompoundLayout::Dynamic.fixed_serialized_slots(), None);
    }

    #[test]
    fn test_hybrid_flat_view_marker_debug_and_clone() {
        let vars = vec![
            VarLayout::ScalarInt,
            VarLayout::Compound(CompoundLayout::Dynamic),
        ];
        let plain = StateLayout::new(vars.clone());
        assert!(!plain.is_hybrid_flat_view());
        // Non-hybrid Debug output must be byte-identical to the historical
        // derived format: layout-derived artifact/warm-cache identities hash
        // this string, so whole-state cache identity must not drift.
        assert_eq!(
            format!("{plain:?}"),
            format!("StateLayout {{ vars: {:?} }}", vars),
        );

        let hybrid = StateLayout::new(vars.clone()).with_hybrid_flat_view();
        assert!(hybrid.is_hybrid_flat_view());
        // The hybrid marker must be visible in Debug so every layout-derived
        // cache identity is namespaced away from the whole-state layout.
        assert_ne!(format!("{plain:?}"), format!("{hybrid:?}"));
        assert!(format!("{hybrid:?}").contains("hybrid_flat_view"));
        // Clone preserves the marker and the compact geometry is unchanged.
        let cloned = hybrid.clone();
        assert!(cloned.is_hybrid_flat_view());
        assert_eq!(cloned.compact_slot_count(), plain.compact_slot_count());
        assert_eq!(
            cloned.compute_compact_var_offsets(),
            plain.compute_compact_var_offsets(),
        );
    }

    #[test]
    fn test_compound_layout_compact_slot_counts() {
        let rec_layout = CompoundLayout::Record {
            fields: vec![
                (tla_core::intern_name("a"), CompoundLayout::Int),
                (tla_core::intern_name("b"), CompoundLayout::Bool),
            ],
        };
        assert_eq!(CompoundLayout::Int.compact_slot_count(), 1);
        assert_eq!(CompoundLayout::String.compact_slot_count(), 1);
        assert_eq!(CompoundLayout::Dynamic.compact_slot_count(), 1);
        assert_eq!(rec_layout.compact_slot_count(), 2);
        assert_eq!(
            CompoundLayout::Tuple {
                element_layouts: vec![CompoundLayout::Int, CompoundLayout::Bool],
            }
            .compact_slot_count(),
            2
        );
    }

    #[test]
    fn test_compound_layout_compact_slot_count_nested_set_bitmask() {
        let layout = CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::Int),
            value_layout: Box::new(CompoundLayout::SetBitmask {
                universe: vec![
                    SetBitmaskElement::Int(1),
                    SetBitmaskElement::Int(2),
                    SetBitmaskElement::Int(3),
                ],
                is_proven_closed: false,
            }),
            pair_count: Some(3),
            domain_lo: Some(1),
        };

        assert_eq!(layout.compact_slot_count(), 3);
        assert_eq!(layout.fixed_serialized_slots(), Some(11));
    }

    #[test]
    fn test_compound_layout_tagged_scalar_or_set_is_one_compact_slot() {
        let tagged_range = CompoundLayout::TaggedScalarOrSet {
            scalar_kind: ScalarSlotKind::ModelValue,
            set_universe: vec![
                SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
            ],
            proof_source: tla_core::intern_name("DijkstraTempTypeOK"),
        };
        assert_eq!(tagged_range.compact_slot_count(), 1);
        assert_eq!(tagged_range.fixed_serialized_slots(), Some(1));
        assert!(!tagged_range.is_scalar());

        let layout = CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::ExplicitScalarDomain {
                key_layout: Box::new(CompoundLayout::String),
                keys: vec![
                    SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                    SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                    SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
                ],
            }),
            value_layout: Box::new(tagged_range),
            pair_count: Some(3),
            domain_lo: None,
        };

        assert_eq!(layout.compact_slot_count(), 3);
        assert_eq!(layout.fixed_serialized_slots(), Some(11));
    }

    #[test]
    fn test_compound_layout_tagged_scalar_union_is_one_compact_slot() {
        // `Nodes \cup {NIL}` — Int arm {1,2,3} as a contiguous prefix (Ord puts
        // Int first) followed by the model-value NIL. One index slot regardless
        // of universe size.
        let union = CompoundLayout::TaggedScalarUnion {
            universe: vec![
                SetBitmaskElement::Int(1),
                SetBitmaskElement::Int(2),
                SetBitmaskElement::Int(3),
                SetBitmaskElement::ModelValue(tla_core::intern_name("NIL")),
            ],
            proof_source: tla_core::intern_name("ChildOfTypeOK"),
        };
        assert_eq!(union.compact_slot_count(), 1);
        assert_eq!(union.fixed_serialized_slots(), Some(1));
        // A union carrier is not a scalar leaf — it decodes through the universe.
        assert!(!union.is_scalar());

        // As a function range: the pair count multiplies the 1-slot range width.
        let layout = CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::ExplicitScalarDomain {
                key_layout: Box::new(CompoundLayout::Int),
                keys: vec![SetBitmaskElement::Int(1), SetBitmaskElement::Int(2)],
            }),
            value_layout: Box::new(union),
            pair_count: Some(2),
            domain_lo: None,
        };
        assert_eq!(layout.compact_slot_count(), 2);
    }

    #[test]
    fn test_compound_layout_explicit_tuple_domain_slot_counts() {
        // A tuple-keyed compact function stores only its range slots: the
        // explicit tuple-key table is layout metadata, exactly like its scalar
        // sibling. `[{1,2} \X {n1}] -> Bool` with 2 keys => 2 compact slots.
        let key_layout = CompoundLayout::ExplicitTupleDomain {
            key_layout: Box::new(CompoundLayout::Tuple {
                element_layouts: vec![CompoundLayout::Int, CompoundLayout::String],
            }),
            keys: vec![
                vec![
                    SetBitmaskElement::Int(1),
                    SetBitmaskElement::ModelValue(tla_core::intern_name("n1")),
                ],
                vec![
                    SetBitmaskElement::Int(2),
                    SetBitmaskElement::ModelValue(tla_core::intern_name("n1")),
                ],
            ],
        };
        // The wrapper delegates to the wrapped Tuple layout, mirroring
        // ExplicitScalarDomain's delegation to its scalar key layout.
        assert_eq!(key_layout.compact_slot_count(), 2);
        assert!(!key_layout.is_scalar());

        let layout = CompoundLayout::Function {
            key_layout: Box::new(key_layout),
            value_layout: Box::new(CompoundLayout::Bool),
            pair_count: Some(2),
            domain_lo: None,
        };
        assert_eq!(layout.compact_slot_count(), 2);
        // Serialized: TAG + count + 2 * (key(tuple: TAG+count+2*2=6) + value(2)).
        assert_eq!(layout.fixed_serialized_slots(), Some(2 + 2 * (6 + 2)));
        assert_eq!(
            layout.int_array_bounds(),
            None,
            "a tuple-keyed function is never a contiguous int array"
        );
    }

    #[test]
    fn test_compound_layout_function_int_array_bounds() {
        let layout = CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::Int),
            value_layout: Box::new(CompoundLayout::Bool),
            pair_count: Some(4),
            domain_lo: Some(0),
        };
        assert_eq!(layout.int_array_bounds(), Some((0, 4)));
    }

    #[test]
    fn test_compound_layout_function_without_domain_lo() {
        let layout = CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::Int),
            value_layout: Box::new(CompoundLayout::Bool),
            pair_count: Some(4),
            domain_lo: None,
        };
        assert_eq!(layout.int_array_bounds(), None);
    }

    #[test]
    fn test_state_layout_all_scalar() {
        let layout = StateLayout::new(vec![VarLayout::ScalarInt, VarLayout::ScalarBool]);
        assert_eq!(layout.var_count(), 2);
        assert!(layout.is_all_scalar());
    }

    #[test]
    fn test_state_layout_mixed_not_all_scalar() {
        let layout = StateLayout::new(vec![
            VarLayout::ScalarInt,
            VarLayout::Compound(CompoundLayout::Sequence {
                element_layout: Box::new(CompoundLayout::Int),
                element_count: None,
                capacity_proven: false,
            }),
        ]);
        assert_eq!(layout.var_count(), 2);
        assert!(!layout.is_all_scalar());
    }

    #[test]
    fn test_compute_var_offsets_scalar_run() {
        let layout = StateLayout::new(vec![
            VarLayout::ScalarInt,
            VarLayout::ScalarBool,
            VarLayout::ScalarInt,
        ]);
        assert_eq!(
            layout.compute_var_offsets(),
            vec![Some(0), Some(1), Some(2)]
        );
    }

    #[test]
    fn test_compute_var_offsets_truncates_after_dynamic() {
        let layout = StateLayout::new(vec![
            VarLayout::ScalarInt,
            VarLayout::Compound(CompoundLayout::Dynamic),
            VarLayout::ScalarInt,
        ]);
        assert_eq!(layout.compute_var_offsets(), vec![Some(0), Some(1), None]);
    }

    #[test]
    fn test_compute_var_offsets_uses_serialized_record_width_not_compact_width() {
        let layout = StateLayout::new(vec![
            VarLayout::Compound(CompoundLayout::Record {
                fields: vec![
                    (tla_core::intern_name("a"), CompoundLayout::Int),
                    (tla_core::intern_name("b"), CompoundLayout::Int),
                ],
            }),
            VarLayout::ScalarInt,
        ]);

        assert_eq!(
            layout.compute_var_offsets(),
            vec![Some(0), Some(8)],
            "compute_var_offsets is for tagged serialized buffers; a compact two-int record would occupy 2 slots"
        );
    }

    #[test]
    fn test_compute_compact_var_offsets_uses_compact_record_width() {
        let layout = StateLayout::new(vec![
            VarLayout::Compound(CompoundLayout::Record {
                fields: vec![
                    (tla_core::intern_name("a"), CompoundLayout::Int),
                    (tla_core::intern_name("b"), CompoundLayout::Int),
                ],
            }),
            VarLayout::ScalarInt,
        ]);

        assert_eq!(layout.compute_var_offsets(), vec![Some(0), Some(8)]);
        assert_eq!(layout.compute_compact_var_offsets(), vec![0, 2]);
        assert_eq!(layout.compact_slot_count(), 3);
    }

    #[test]
    fn test_compact_var_slot_range_uses_compact_widths() {
        let layout = StateLayout::new(vec![
            VarLayout::ScalarInt,
            VarLayout::Compound(CompoundLayout::Record {
                fields: vec![
                    (tla_core::intern_name("a"), CompoundLayout::Int),
                    (tla_core::intern_name("b"), CompoundLayout::Bool),
                ],
            }),
            VarLayout::ScalarBool,
        ]);

        assert_eq!(layout.compact_var_slot_range(0), Some(0..1));
        assert_eq!(layout.compact_var_slot_range(1), Some(1..3));
        assert_eq!(layout.compact_var_slot_range(2), Some(3..4));
        assert_eq!(layout.compact_var_slot_range(3), None);
    }

    #[test]
    fn test_compact_write_kernel_slots_are_sorted_and_deduped() {
        let layout = StateLayout::new(vec![
            VarLayout::ScalarInt,
            VarLayout::Compound(CompoundLayout::Record {
                fields: vec![
                    (tla_core::intern_name("left"), CompoundLayout::Int),
                    (tla_core::intern_name("right"), CompoundLayout::Int),
                ],
            }),
            VarLayout::ScalarBool,
        ]);

        let slots = layout
            .compact_write_kernel_slots(&[2, 1, 1, 0])
            .expect("all write vars should map to compact slots");

        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0], crate::KernelStateSlot::tla_state_var(0, 0, 1));
        assert_eq!(slots[1], crate::KernelStateSlot::tla_state_var(1, 1, 2));
        assert_eq!(slots[2], crate::KernelStateSlot::tla_state_var(2, 3, 1));
        assert!(layout.compact_write_kernel_slots(&[3]).is_none());
    }

    #[test]
    fn test_compute_compact_var_offsets_diverge_for_recursive_set_bitmask() {
        let layout = StateLayout::new(vec![
            VarLayout::Compound(CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(CompoundLayout::SetBitmask {
                    universe: vec![
                        SetBitmaskElement::Int(1),
                        SetBitmaskElement::Int(2),
                        SetBitmaskElement::Int(3),
                    ],
                    is_proven_closed: false,
                }),
                pair_count: Some(3),
                domain_lo: Some(1),
            }),
            VarLayout::ScalarBool,
        ]);

        assert_eq!(layout.compute_var_offsets(), vec![Some(0), Some(11)]);
        assert_eq!(layout.compute_compact_var_offsets(), vec![0, 3]);
        assert_eq!(layout.compact_slot_count(), 4);
    }
}
