// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bridge between tla-check's `StateLayout` and `tla-jit-abi`'s `StateLayout`.
//!
//! The model checker (`tla-check`) and the native ABI layer each have their own
//! `StateLayout` type with different trade-offs:
//!
//! | Property            | tla-check layout            | native ABI layout          |
//! |--------------------|-----------------------------|----------------------------|
//! | Purpose            | BFS flat state storage      | Native codegen metadata    |
//! | Var descriptor     | `VarLayoutKind` (enum)      | `VarLayout` (enum)         |
//! | Compound support   | IntArray, Record, Bitmask   | CompoundLayout (recursive) |
//! | Offset tracking    | Built-in (contiguous pack)  | Compact/serialized APIs             |
//! | Buffer format      | Compact (no type tags)      | Self-describing (type tags)|
//!
//! This module provides conversion functions so that:
//!
//! 1. The JIT compiled BFS step can understand the model checker's layout.
//! 2. The model checker can convert native-produced buffers back to ArrayState.
//! 3. Layout compatibility is verified at BFS initialization time.
//!
//! # Buffer format conversion
//!
//! The two buffer formats are **not** directly compatible:
//!
//! - **tla-check format** (compact): `[raw_value, raw_value, ...]`
//!   - Scalar: 1 slot, raw i64
//!   - IntArray{lo, len}: `len` contiguous raw i64 values
//!   - Record{fields}: `len(fields)` contiguous raw i64 values
//!   - Dynamic: 1 slot, zero placeholder
//!
//! - **native ABI format** (tagged): `[TAG, value, TAG, count, TAG, val, ...]`
//!   - Every value prefixed with a type tag word
//!   - Records include field count + per-field name_id
//!   - Self-describing for deserialization without layout
//!
//! Compiled BFS steps operate directly on the **compact** format
//! (tla-check's). Native backends must generate code aware of the slot offsets,
//! not the self-describing format. This module provides the layout metadata
//! conversion that enables that code generation.
//!
//! Part of #3986: Phase 3 flat state buffer layout bridge.

use tla_value::Rp;
use super::state_layout::{
    ordered_dense_int_domain, FlatScalarValue, FlatValueLayout, SlotType, StateLayout,
    StringKeyedArrayRangeEncoding, TaggedScalarSetRangeProof, TupleKeyedArrayRangeEncoding,
    VarLayoutKind,
};

/// Convert a tla-check `StateLayout` into the equivalent native ABI `StateLayout`.
///
/// This enables native code generation to understand the model checker's flat
/// buffer format. Backends can use the resulting layout to read/write the
/// compact (no type tags) buffer directly.
///
/// # Mapping
///
/// | tla-check `VarLayoutKind`     | native ABI `VarLayout`                        |
/// |------------------------------|-----------------------------------------------|
/// | `Scalar`                     | `ScalarInt`                                   |
/// | `ScalarBool`                 | `ScalarBool`                                  |
/// | `ScalarString`               | `Compound(String)` (NameId as i64)            |
/// | `ScalarModelValue`           | `Compound(String)` (NameId as i64)            |
/// | `IntArray { lo, len, bool }` | `Compound(Function { Int->Int/Bool, n })`     |
/// | `Record { fields, bools }`   | `Compound(Record { fields })`                 |
/// | `StringKeyedArray { keys }`  | `Compound(Function { ExplicitDomain->T, n })` |
/// | `TaggedScalarOrSet` range    | `Compound(TaggedScalarOrSet)` as range layout |
/// | `Recursive { layout }`       | Recursive `CompoundLayout`                    |
/// | `Recursive SetBitmask`       | `Compound(SetBitmask)` in one compact slot    |
/// | `Bitmask { size }`           | `ScalarInt` (bitmask is a single i64)         |
/// | `Dynamic`                    | `Compound(Dynamic)`                           |
///
/// Note: The returned native ABI layout describes the **compact** buffer format
/// (offsets match tla-check's slot packing), not the self-describing
/// tagged format used by `serialize_value()`.
#[must_use]
pub(crate) fn check_layout_to_jit_layout(check_layout: &StateLayout) -> tla_jit_abi::StateLayout {
    let jit_vars: Vec<tla_jit_abi::VarLayout> = check_layout
        .iter()
        .map(|var| check_var_to_jit_var(&var.kind))
        .collect();
    tla_jit_abi::StateLayout::new(jit_vars)
}

/// Overlay proven function-range / top-level `SetBitmask` universes from the
/// flat check layout onto a value-inferred JIT layout, in place.
///
/// When a spec is not fully flat its action JIT layout is inferred from a
/// sampled init state (`tla_jit_abi::infer_var_layout`). A function range whose
/// init value is an empty set inflates to a universe-less
/// `CompoundLayout::Set { element_count: Some(0) }`, dropping the element
/// universe that a `\in [Dom -> SUBSET <const>]` type invariant proves. The
/// trust-cg next-state buffer (`prepare_trust_cg_next_state`) is nonetheless
/// always encoded by the flat `FlatBfsBridge`, which writes those slots as a
/// single bitmask ordered by the proven universe. The JIT layout handed to
/// trust-ir lowering must therefore carry the *same* universe so the read/write
/// bit positions agree with the buffer the action receives.
///
/// Only `FlatValueLayout::SetBitmask` universes with a `ProvenClosed` closure
/// are projected (the GAP B soundness fact: no successor can introduce an
/// out-of-universe element, and the native EXCEPT-set lowering writes the
/// canonical bitmask). `Sampled` universes stay fail-closed, and every other
/// var keeps its value-inferred layout, so the blast radius is exactly the
/// proven function-range/top-level set slots.
pub(crate) fn overlay_proven_set_bitmask_universes_from_flat(
    inferred: &mut tla_jit_abi::StateLayout,
    flat_layout: &StateLayout,
) {
    for (var_idx, flat_var) in flat_layout.iter().enumerate() {
        let Some(overlay) = proven_set_bitmask_var_layout_from_flat(&flat_var.kind) else {
            continue;
        };
        let Some(slot) = inferred.var_layout_mut(var_idx) else {
            continue;
        };
        // The overlay must occupy the identical compact slot width as the
        // value-inferred layout; otherwise the slot offsets the buffer encoder
        // produced would no longer line up. A proven function-range bitmask and
        // the inferred universe-less `Set` both encode one compact slot per
        // range value, so the widths match — but verify rather than assume.
        if slot.compact_slot_count() == overlay.compact_slot_count() {
            *slot = overlay;
        }
    }
}

/// Build the JIT var layout for a flat var whose layout proves a closed
/// `SetBitmask` universe (a top-level `SUBSET <const>` slot or a
/// `[Dom -> SUBSET <const>]` function range). Returns `None` for every other
/// shape so the caller leaves the value-inferred layout untouched.
fn proven_set_bitmask_var_layout_from_flat(kind: &VarLayoutKind) -> Option<tla_jit_abi::VarLayout> {
    let VarLayoutKind::Recursive { layout } = kind else {
        return None;
    };
    match layout {
        FlatValueLayout::SetBitmask {
            universe_closure, ..
        } if universe_closure.is_proven_closed() => Some(check_var_to_jit_var(kind)),
        FlatValueLayout::Function { value_layout, .. }
        | FlatValueLayout::IntFunction { value_layout, .. }
            if matches!(
                value_layout.as_ref(),
                FlatValueLayout::SetBitmask { universe_closure, .. }
                    if universe_closure.is_proven_closed()
            ) =>
        {
            Some(check_var_to_jit_var(kind))
        }
        _ => None,
    }
}

/// Convert a single tla-check `VarLayoutKind` to a native ABI `VarLayout`.
fn check_var_to_jit_var(kind: &VarLayoutKind) -> tla_jit_abi::VarLayout {
    match kind {
        VarLayoutKind::Scalar => tla_jit_abi::VarLayout::ScalarInt,
        VarLayoutKind::ScalarBool => tla_jit_abi::VarLayout::ScalarBool,
        VarLayoutKind::IntArray {
            lo,
            len,
            elements_are_bool,
            element_types,
            ..
        } => {
            let value_layout = int_array_value_layout(*elements_are_bool, element_types.as_deref());
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                value_layout: Box::new(value_layout),
                pair_count: Some(*len),
                domain_lo: Some(*lo),
            })
        }
        VarLayoutKind::Record {
            field_names,
            field_types,
            ..
        } => {
            let fields: Vec<(tla_core::NameId, tla_jit_abi::CompoundLayout)> = field_names
                .iter()
                .zip(field_types.iter())
                .map(|(name, ty)| {
                    let nid = tla_core::intern_name(name);
                    let layout = match ty {
                        super::state_layout::SlotType::Bool => tla_jit_abi::CompoundLayout::Bool,
                        super::state_layout::SlotType::String
                        | super::state_layout::SlotType::ModelValue => {
                            tla_jit_abi::CompoundLayout::String
                        }
                        super::state_layout::SlotType::Int => tla_jit_abi::CompoundLayout::Int,
                    };
                    (nid, layout)
                })
                .collect();
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Record { fields })
        }
        VarLayoutKind::ScalarString | VarLayoutKind::ScalarModelValue => {
            // String/ModelValue scalars are interned NameIds stored as i64.
            // Preserve their scalar lane so trust-ir can distinguish string
            // equality from integer equality after flat-primary promotion.
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::String)
        }
        VarLayoutKind::FixedScalar { .. } => {
            // A finite-universe scalar string/model-value enum (G2) uses the
            // identical one-slot interned-NameId encoding as a plain string
            // scalar, so it maps to the same single-compact-slot JIT layout.
            // This is what makes the check layout and JIT layout agree (the
            // layout-mismatch warning disappears and the run stays fully flat).
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::String)
        }
        VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            range_encoding,
        } => {
            // String-keyed function: `domain_keys.len()` contiguous i64 slots
            // for the range values. Domain keys are metadata (not in buffer).
            // Map as Function { explicit scalar domain -> value_type, n,
            // lo=None } so compact FuncApply can use the same proof tla-check
            // used to allocate the fixed slots.
            // Legacy ScalarSlots keep the historical common-element layout.
            // TaggedScalarOrSet carries a distinct proof-bearing range layout
            // so it cannot be treated as the old untagged scalar ABI.
            let value_layout =
                string_keyed_array_range_to_jit_compound(value_types, range_encoding);
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(explicit_string_domain_to_jit_compound(
                    domain_keys,
                    domain_types,
                )),
                value_layout: Box::new(value_layout),
                pair_count: Some(domain_keys.len()),
                domain_lo: None,
            })
        }
        VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding,
        } => {
            // Tuple/cross-product-keyed function: `domain_keys.len()` contiguous
            // i64 slots for the range values. The tuple domain keys are layout
            // metadata (not in the buffer). Map as Function { Tuple-key ->
            // value_type, n, lo=None } so the native ABI agrees with the JIT's
            // own value-derived inference (which produces an identical
            // `Function { key_layout: Tuple{..}, .. }` for a scalar-tuple
            // domain). This keeps the flat-buffer slot counts in sync.
            //
            // When every tuple key converts to native bitmask elements, wrap
            // the key layout in `ExplicitTupleDomain` so the canonical ordered
            // key table travels across the ABI (the tuple analogue of the
            // `StringKeyedArray` -> `ExplicitScalarDomain` carry above). The
            // stored `domain_keys` are already in ascending `Value::cmp` order
            // — the exact compact-slot order `write_tuple_keyed_array_slots`
            // assigns — so the carrier is emitted as-is, never re-sorted. An
            // unconvertible key (nested/non-scalar position, out-of-range int)
            // falls back to the bare `Tuple` key layout, which downstream
            // lowering treats exactly as before (const-pool recovery or fail
            // closed) — never a partial table.
            let value_layout = tuple_keyed_array_range_to_jit_compound(value_types, range_encoding);
            let tuple_key_layout = tuple_domain_key_layout_to_jit_compound(domain_keys);
            let key_layout = match tuple_domain_keys_to_jit_elements(domain_keys) {
                Some(keys)
                    if matches!(tuple_key_layout, tla_jit_abi::CompoundLayout::Tuple { .. }) =>
                {
                    tla_jit_abi::CompoundLayout::ExplicitTupleDomain {
                        key_layout: Box::new(tuple_key_layout),
                        keys,
                    }
                }
                _ => tuple_key_layout,
            };
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(key_layout),
                value_layout: Box::new(value_layout),
                pair_count: Some(domain_keys.len()),
                domain_lo: None,
            })
        }
        VarLayoutKind::Recursive { layout } => {
            tla_jit_abi::VarLayout::Compound(flat_value_layout_to_jit_compound(layout))
        }
        VarLayoutKind::Bitmask { .. } => {
            // Bitmask is a single i64 slot — treat as scalar for JIT purposes.
            tla_jit_abi::VarLayout::ScalarInt
        }
        VarLayoutKind::Dynamic => {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Dynamic)
        }
    }
}

/// Build the JIT `CompoundLayout::Tuple` key layout for a tuple-keyed function
/// from its canonical domain keys.
///
/// Mirrors the JIT's own value-derived inference: each tuple key has the same
/// arity and per-position scalar shape, so the first key's element types fix the
/// `element_layouts`. Falls back to `Dynamic` if the domain is empty or the
/// first key is not a scalar tuple (which the inference path should never
/// admit).
fn tuple_domain_key_layout_to_jit_compound(
    domain_keys: &[tla_value::Value],
) -> tla_jit_abi::CompoundLayout {
    use tla_value::Value;
    let Some(Value::Tuple(elems)) = domain_keys.first() else {
        return tla_jit_abi::CompoundLayout::Dynamic;
    };
    let element_layouts: Vec<tla_jit_abi::CompoundLayout> = elems
        .iter()
        .map(|elem| match elem {
            Value::Bool(_) => tla_jit_abi::CompoundLayout::Bool,
            Value::String(_) | Value::ModelValue(_) => tla_jit_abi::CompoundLayout::String,
            _ => tla_jit_abi::CompoundLayout::Int,
        })
        .collect();

    // Opt-in tuple-domain-key carrier (`TY_TUPLE_KEY_CARRIER`). When enabled and
    // the whole domain is a homogeneous scalar (Int/Bool) tuple table, carry the
    // exact ordered domain keys so the native tuple-keyed FuncApply/FuncExcept
    // can recover the flat slot for a tuple key WITHOUT re-deriving the
    // cross-product from the constant pool (which fails when the domain is not a
    // materialized constant). The carried order is `domain_keys` verbatim — the
    // same sequence `flat_state::write_tuple_keyed_array_slots` serializes slot
    // `i` <- `domain_keys[i]` — so it must never be re-sorted here. Falls back to
    // the plain `Tuple` key layout (byte-identical to the default path) when the
    // flag is off or any key is not a homogeneous Int/Bool tuple.
    if tuple_key_carrier_enabled() {
        if let Some(keys) = tuple_domain_carrier_keys(domain_keys, &element_layouts) {
            return tla_jit_abi::CompoundLayout::ExplicitTupleDomain {
                key_layout: Box::new(tla_jit_abi::CompoundLayout::Tuple { element_layouts }),
                keys,
            };
        }
    }

    tla_jit_abi::CompoundLayout::Tuple { element_layouts }
}

/// Whether the opt-in tuple-domain-key carrier is enabled.
///
/// Default OFF so the produced JIT layout — and therefore all downstream native
/// admission and flat-buffer serialization — is byte-identical to the historical
/// plain-`Tuple` key layout unless `TY_TUPLE_KEY_CARRIER` is explicitly set.
fn tuple_key_carrier_enabled() -> bool {
    std::env::var_os("TY_TUPLE_KEY_CARRIER").is_some()
}

/// Build the ordered per-position scalar encoding of a homogeneous tuple domain,
/// preserving the exact `domain_keys` order (which is the flat-buffer slot
/// order). Returns `None` (fail closed to the plain `Tuple` layout) when any key
/// is not a fixed-arity tuple of plain `Int`/`Bool` scalars matching
/// `element_layouts`.
fn tuple_domain_carrier_keys(
    domain_keys: &[tla_value::Value],
    element_layouts: &[tla_jit_abi::CompoundLayout],
) -> Option<Vec<Vec<tla_jit_abi::SetBitmaskElement>>> {
    use tla_value::Value;
    // Only homogeneous plain-scalar tuple domains are in scope. A String/model
    // position has no native tuple-key equality lowering yet, so fall back.
    if element_layouts.is_empty()
        || !element_layouts.iter().all(|layout| {
            matches!(
                layout,
                tla_jit_abi::CompoundLayout::Int | tla_jit_abi::CompoundLayout::Bool
            )
        })
    {
        return None;
    }
    let mut keys = Vec::with_capacity(domain_keys.len());
    for key in domain_keys {
        let Value::Tuple(elems) = key else {
            return None;
        };
        if elems.len() != element_layouts.len() {
            return None;
        }
        let mut row = Vec::with_capacity(elems.len());
        for (elem, layout) in elems.iter().zip(element_layouts.iter()) {
            let encoded = match (layout, elem) {
                (tla_jit_abi::CompoundLayout::Int, Value::SmallInt(n)) => {
                    tla_jit_abi::SetBitmaskElement::Int(*n)
                }
                (tla_jit_abi::CompoundLayout::Bool, Value::Bool(b)) => {
                    tla_jit_abi::SetBitmaskElement::Bool(*b)
                }
                _ => return None,
            };
            row.push(encoded);
        }
        keys.push(row);
    }
    Some(keys)
}

/// Convert the canonical tuple domain keys into the native ABI's per-position
/// bitmask elements, preserving the stored (already `Value::cmp`-sorted) order.
///
/// Returns `None` (fail closed — the caller keeps the bare `Tuple` key layout)
/// when any key is not a scalar tuple, positions are of inconsistent arity, or
/// any position holds a non-scalar / non-`i64`-representable value. A partial
/// table is never emitted.
fn tuple_domain_keys_to_jit_elements(
    domain_keys: &[tla_value::Value],
) -> Option<Vec<Vec<tla_jit_abi::SetBitmaskElement>>> {
    use tla_value::Value;
    let Some(Value::Tuple(first)) = domain_keys.first() else {
        return None;
    };
    let arity = first.len();
    if arity == 0 {
        return None;
    }
    domain_keys
        .iter()
        .map(|key| {
            let Value::Tuple(elems) = key else {
                return None;
            };
            if elems.len() != arity {
                return None;
            }
            elems
                .iter()
                .map(|elem| match elem {
                    Value::Bool(b) => Some(tla_jit_abi::SetBitmaskElement::Bool(*b)),
                    Value::SmallInt(n) => Some(tla_jit_abi::SetBitmaskElement::Int(*n)),
                    Value::String(s) => Some(tla_jit_abi::SetBitmaskElement::String(
                        tla_core::intern_name(s),
                    )),
                    Value::ModelValue(s) => Some(tla_jit_abi::SetBitmaskElement::ModelValue(
                        tla_core::intern_name(s),
                    )),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()
}

fn tuple_keyed_array_range_to_jit_compound(
    value_types: &[SlotType],
    range_encoding: &super::state_layout::TupleKeyedArrayRangeEncoding,
) -> tla_jit_abi::CompoundLayout {
    use super::state_layout::TupleKeyedArrayRangeEncoding;
    match range_encoding {
        // Both encodings store one plain scalar payload per slot; the proof
        // only authorizes flat-primary admission check-side. The JIT layout is
        // derived from the (homogeneous) slot types either way, mirroring the
        // `StringKeyedArray` `ScalarSlots`/`FixedScalar` mapping.
        TupleKeyedArrayRangeEncoding::ScalarSlots
        | TupleKeyedArrayRangeEncoding::FixedScalar(_) => {
            uniform_slot_types_to_jit_compound(value_types)
        }
        // WP-09/Part A: a heterogeneous finite scalar-union range slot crosses
        // the ABI as the WP-05 `TaggedScalarUnion` carrier — the EXACT ordered
        // universe plus the proof-source identity, so the native side stores
        // the same injective universe index the check-side encode uses. Every
        // `FlatScalarValue` variant converts to a `SetBitmaskElement`, so this
        // never partially converts (mirrors the `FlatValueLayout::
        // TaggedScalarUnion` arm of `flat_value_layout_to_jit_compound`).
        TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof) => {
            tla_jit_abi::CompoundLayout::TaggedScalarUnion {
                universe: proof
                    .universe()
                    .iter()
                    .map(flat_scalar_to_jit_bitmask_element)
                    .collect(),
                proof_source: tla_core::intern_name(proof.source().as_ref()),
            }
        }
    }
}

fn int_array_value_layout(
    elements_are_bool: bool,
    element_types: Option<&[super::state_layout::SlotType]>,
) -> tla_jit_abi::CompoundLayout {
    match element_types {
        Some(types) => uniform_slot_types_to_jit_compound(types),
        None if elements_are_bool => tla_jit_abi::CompoundLayout::Bool,
        None => tla_jit_abi::CompoundLayout::Int,
    }
}

fn uniform_slot_types_to_jit_compound(
    slot_types: &[super::state_layout::SlotType],
) -> tla_jit_abi::CompoundLayout {
    use super::state_layout::SlotType;

    let Some(first) = slot_types.first() else {
        return tla_jit_abi::CompoundLayout::Dynamic;
    };
    if !slot_types.iter().all(|slot_type| slot_type == first) {
        return tla_jit_abi::CompoundLayout::Dynamic;
    }
    match first {
        SlotType::Bool => tla_jit_abi::CompoundLayout::Bool,
        SlotType::String | SlotType::ModelValue => tla_jit_abi::CompoundLayout::String,
        SlotType::Int => tla_jit_abi::CompoundLayout::Int,
    }
}

fn string_keyed_array_range_to_jit_compound(
    value_types: &[SlotType],
    range_encoding: &StringKeyedArrayRangeEncoding,
) -> tla_jit_abi::CompoundLayout {
    match range_encoding {
        StringKeyedArrayRangeEncoding::ScalarSlots
        | StringKeyedArrayRangeEncoding::FixedScalar(_) => {
            uniform_slot_types_to_jit_compound(value_types)
        }
        StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof) => {
            tagged_scalar_set_range_to_jit_compound(proof)
        }
    }
}

fn tagged_scalar_set_range_to_jit_compound(
    proof: &TaggedScalarSetRangeProof,
) -> tla_jit_abi::CompoundLayout {
    tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
        scalar_kind: slot_type_to_jit_scalar_kind(proof.scalar_type()),
        set_universe: proof
            .set_universe()
            .iter()
            .map(flat_scalar_to_jit_bitmask_element)
            .collect(),
        proof_source: tla_core::intern_name(proof.source().as_ref()),
    }
}

/// Transport a heterogeneous finite scalar-union proof to the native ABI. The
/// universe is carried in the SAME canonical index order the interpreter uses
/// (`TaggedScalarUnionProof::universe()` is the sorted-deduped
/// `Vec<FlatScalarValue>`), so element position IS the encoded slot index; the
/// tla-ir lowering reads it back to convert a scalar write to its universe
/// index.
fn tagged_scalar_union_to_jit_compound(
    proof: &super::state_layout::TaggedScalarUnionProof,
) -> tla_jit_abi::CompoundLayout {
    tla_jit_abi::CompoundLayout::TaggedScalarUnion {
        universe: proof
            .universe()
            .iter()
            .map(flat_scalar_to_jit_bitmask_element)
            .collect(),
        proof_source: tla_core::intern_name(proof.source().as_ref()),
    }
}

/// Map a `FlatValueLayout` to its native `CompoundLayout` carrier ONLY when the
/// carrier's compact slot width equals the flat layout's own slot count — i.e.
/// the native buffer offsets will agree with the model-checker's flat layout
/// byte-for-byte.
///
/// Returns `None` (the caller then fails closed to `Dynamic`) when the mapping is
/// lossy (`Dynamic`) or the widths disagree, so a tagged-union / heterogeneous-
/// tuple carrier can never silently misalign a downstream variable's slot offset.
fn faithful_flat_value_layout_to_jit_compound(
    layout: &super::state_layout::FlatValueLayout,
) -> Option<tla_jit_abi::CompoundLayout> {
    let jit = flat_value_layout_to_jit_compound(layout);
    if matches!(jit, tla_jit_abi::CompoundLayout::Dynamic) {
        return None;
    }
    if jit.compact_slot_count() != layout.slot_count() {
        return None;
    }
    Some(jit)
}

fn flat_value_layout_to_jit_compound(
    layout: &super::state_layout::FlatValueLayout,
) -> tla_jit_abi::CompoundLayout {
    match layout {
        super::state_layout::FlatValueLayout::Scalar(slot_type) => {
            slot_type_to_jit_compound(*slot_type)
        }
        super::state_layout::FlatValueLayout::IntFunction {
            lo,
            len,
            value_layout,
        } => tla_jit_abi::CompoundLayout::Function {
            key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
            value_layout: Box::new(flat_value_layout_to_jit_compound(value_layout)),
            pair_count: Some(*len),
            domain_lo: Some(*lo),
        },
        super::state_layout::FlatValueLayout::Function {
            domain,
            value_layout,
        } => {
            let value_layout = Box::new(flat_value_layout_to_jit_compound(value_layout));
            if let Some((lo, len)) = ordered_dense_int_domain(domain) {
                tla_jit_abi::CompoundLayout::Function {
                    key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                    value_layout,
                    pair_count: Some(len),
                    domain_lo: Some(lo),
                }
            } else {
                tla_jit_abi::CompoundLayout::Function {
                    key_layout: Box::new(explicit_flat_domain_to_jit_compound(domain)),
                    value_layout,
                    pair_count: Some(domain.len()),
                    domain_lo: None,
                }
            }
        }
        super::state_layout::FlatValueLayout::Record {
            field_names,
            field_layouts,
        } => {
            let fields = field_names
                .iter()
                .zip(field_layouts.iter())
                .map(|(name, field_layout)| {
                    (
                        tla_core::intern_name(name),
                        flat_value_layout_to_jit_compound(field_layout),
                    )
                })
                .collect();
            tla_jit_abi::CompoundLayout::Record { fields }
        }
        super::state_layout::FlatValueLayout::SetBitmask {
            universe,
            universe_closure,
        } => tla_jit_abi::CompoundLayout::SetBitmask {
            universe: universe
                .iter()
                .map(flat_scalar_to_jit_bitmask_element)
                .collect(),
            // Carry the closure proof across the ABI so a round-tripped
            // function-range slot can be re-admitted (or kept fail-closed) for
            // native flat-primary dispatch. A `Sampled` universe stays
            // fail-closed on the far side.
            is_proven_closed: universe_closure.is_proven_closed(),
        },
        // A record-set bitmask: a set whose elements are records drawn from a
        // finite, provably/monitored-closed universe, packed as a fixed-width
        // multi-slot i64 bitmask (bit `i` = universe record `i` present). The
        // native compound ABI carries this as
        // `CompoundLayout::RecordSetBitmask`, transporting the universe so the
        // byte-exact `set_ops` RecordSetBitmask lowering (membership / union /
        // diff) becomes reachable for the var. If ANY universe record cannot be
        // flattened to a scalar field tuple (e.g. a nested/non-scalar field),
        // fail closed to `Dynamic` rather than emit a partial carrier — a
        // partial carrier would let the lowering set a wrong bit / IntToPtr the
        // mask in an uncovered context (the rc=139 trap).
        super::state_layout::FlatValueLayout::RecordSetBitmask {
            universe,
            universe_closure,
        } => match record_set_bitmask_universe_to_jit(universe) {
            Some(jit_universe) => {
                let slot_count = jit_universe.len().div_ceil(64);
                tla_jit_abi::CompoundLayout::RecordSetBitmask {
                    universe: jit_universe,
                    slot_count,
                    // Carry the closure proof across the ABI so a round-tripped
                    // slot can be re-admitted (or kept fail-closed). A `Sampled`
                    // universe stays fail-closed on the far side; the lowering
                    // arms gate on the var being a real flat-primary slot, which
                    // a sampled universe is not.
                    is_proven_closed: universe_closure.is_proven_closed(),
                }
            }
            None => tla_jit_abi::CompoundLayout::Dynamic,
        },
        // The native compound ABI has no nested-set (set-of-sets) shape, so a
        // nested-set bitmask slot is opaque to native access: map it to
        // `Dynamic` (fail-closed), exactly like `RecordSetBitmask`. INERT: never
        // constructed yet (A2 scaffolding).
        super::state_layout::FlatValueLayout::NestedSetBitmask { .. } => {
            tla_jit_abi::CompoundLayout::Dynamic
        }
        // WP-05 item 2: a heterogeneous finite scalar union (`Nodes \cup {NIL}`
        // — Int arm plus a model value) crosses the ABI as its ordered,
        // deduplicated universe plus the proof-source identity. The native side
        // stores the injective universe INDEX in one slot (mirroring the
        // check-side `encode_tagged_scalar_union_value`), so `Int(k)` and
        // `ModelValue(nil)` occupy distinct slots. The universe order is
        // ABI-significant and preserved verbatim from the proof (ty's sorted
        // `FlatScalarValue` assembly order — Int members form a contiguous
        // ascending prefix), so the native arm-aware `(v - lo) + base` encoding
        // is derivable. Every `FlatScalarValue` variant converts to a
        // `SetBitmaskElement`, so this never partially converts; if that ever
        // changed, `flat_scalar_to_jit_bitmask_element` would be the choke point
        // to fail closed to `Dynamic`.
        super::state_layout::FlatValueLayout::TaggedScalarUnion { proof } => {
            tla_jit_abi::CompoundLayout::TaggedScalarUnion {
                universe: proof
                    .universe()
                    .iter()
                    .map(flat_scalar_to_jit_bitmask_element)
                    .collect(),
                proof_source: tla_core::intern_name(proof.source().as_ref()),
            }
        }
        // WP-ARGS: a finite scalar-or-tuple union (btree's `args`: the model
        // value `NIL`, or `<<k>>` / `<<k,v>>`) crosses the ABI as its ordered
        // variant list plus the payload width, so the native side can lower a
        // tag store + payload stores and a tag-guarded payload read.
        //
        // ALL-OR-NOTHING: if ANY variant maps to `Dynamic` the whole carrier
        // degrades to `Dynamic`. A partial carrier would be worse than none —
        // the lowering would trust the tag and compute a payload offset into a
        // variant whose width it cannot actually predict, reading a neighbouring
        // variable's slot. `max_payload_slots` is recomputed here from the
        // CONVERTED variants (not copied from the proof) so the ABI width can
        // never disagree with the ABI variant shapes.
        super::state_layout::FlatValueLayout::TaggedUnion { proof } => {
            let variants: Vec<tla_jit_abi::CompoundLayout> = proof
                .variants()
                .iter()
                .map(flat_value_layout_to_jit_compound)
                .collect();
            if variants
                .iter()
                .any(|variant| matches!(variant, tla_jit_abi::CompoundLayout::Dynamic))
            {
                tla_jit_abi::CompoundLayout::Dynamic
            } else {
                let max_payload_slots = variants
                    .iter()
                    .map(tla_jit_abi::CompoundLayout::compact_slot_count)
                    .max()
                    .unwrap_or(0);
                // The check-side proof and the ABI carrier must agree on the
                // payload window, otherwise a native store would zero-fill a
                // different trailing range than the interpreter and the two
                // representations would fingerprint differently.
                if max_payload_slots == proof.max_payload_slots() {
                    tla_jit_abi::CompoundLayout::TaggedUnion {
                        variants,
                        max_payload_slots,
                        proof_source: tla_core::intern_name(proof.source().as_ref()),
                    }
                } else {
                    tla_jit_abi::CompoundLayout::Dynamic
                }
            }
        }
        // WP-ARGS: a fixed-arity product crosses the ABI as `CompoundLayout::Tuple`,
        // whose `compact_slot_count` is likewise the sum of the per-position
        // widths (no length slot). ALL-OR-NOTHING for the same reason as the
        // union above: one `Dynamic` position would make every LATER position's
        // offset unpredictable, so the whole tuple degrades.
        super::state_layout::FlatValueLayout::HeterogeneousTuple { element_layouts } => {
            let element_layouts: Vec<tla_jit_abi::CompoundLayout> = element_layouts
                .iter()
                .map(flat_value_layout_to_jit_compound)
                .collect();
            if element_layouts
                .iter()
                .any(|element| matches!(element, tla_jit_abi::CompoundLayout::Dynamic))
            {
                tla_jit_abi::CompoundLayout::Dynamic
            } else {
                tla_jit_abi::CompoundLayout::Tuple { element_layouts }
            }
        }
        super::state_layout::FlatValueLayout::Sequence {
            max_len,
            element_layout,
            bound,
        } => tla_jit_abi::CompoundLayout::Sequence {
            element_layout: Box::new(flat_value_layout_to_jit_compound(element_layout)),
            element_count: Some(*max_len),
            // Only a checked source-level capacity proof makes `max_len` a sound
            // upper bound on `Len(seq)` across all reachable states; an observed
            // bound must stay fail-closed. See `SequenceBoundEvidence::is_proven`.
            capacity_proven: bound.is_proven(),
        },
    }
}

fn slot_type_to_jit_compound(
    slot_type: super::state_layout::SlotType,
) -> tla_jit_abi::CompoundLayout {
    match slot_type {
        super::state_layout::SlotType::Bool => tla_jit_abi::CompoundLayout::Bool,
        super::state_layout::SlotType::String | super::state_layout::SlotType::ModelValue => {
            tla_jit_abi::CompoundLayout::String
        }
        super::state_layout::SlotType::Int => tla_jit_abi::CompoundLayout::Int,
    }
}

fn slot_type_to_jit_scalar_kind(slot_type: SlotType) -> tla_jit_abi::ScalarSlotKind {
    match slot_type {
        SlotType::Bool => tla_jit_abi::ScalarSlotKind::Bool,
        SlotType::String => tla_jit_abi::ScalarSlotKind::String,
        SlotType::ModelValue => tla_jit_abi::ScalarSlotKind::ModelValue,
        SlotType::Int => tla_jit_abi::ScalarSlotKind::Int,
    }
}

fn jit_scalar_kind_to_slot_type(scalar_kind: tla_jit_abi::ScalarSlotKind) -> SlotType {
    match scalar_kind {
        tla_jit_abi::ScalarSlotKind::Bool => SlotType::Bool,
        tla_jit_abi::ScalarSlotKind::String => SlotType::String,
        tla_jit_abi::ScalarSlotKind::ModelValue => SlotType::ModelValue,
        tla_jit_abi::ScalarSlotKind::Int => SlotType::Int,
    }
}

fn flat_domain_to_jit_compound(
    domain: &[super::state_layout::FlatScalarValue],
) -> tla_jit_abi::CompoundLayout {
    let Some(first) = domain.first() else {
        return tla_jit_abi::CompoundLayout::Dynamic;
    };
    let first_type = first.slot_type();
    if domain.iter().all(|key| key.slot_type() == first_type) {
        slot_type_to_jit_compound(first_type)
    } else {
        tla_jit_abi::CompoundLayout::Dynamic
    }
}

fn explicit_flat_domain_to_jit_compound(
    domain: &[super::state_layout::FlatScalarValue],
) -> tla_jit_abi::CompoundLayout {
    let key_layout = flat_domain_to_jit_compound(domain);
    if !key_layout.is_scalar() && !matches!(key_layout, tla_jit_abi::CompoundLayout::Dynamic) {
        return key_layout;
    }
    tla_jit_abi::CompoundLayout::ExplicitScalarDomain {
        key_layout: Box::new(key_layout),
        keys: domain
            .iter()
            .map(flat_scalar_to_jit_bitmask_element)
            .collect(),
    }
}

fn explicit_string_domain_to_jit_compound(
    domain_keys: &[std::sync::Arc<str>],
    domain_types: &[super::state_layout::SlotType],
) -> tla_jit_abi::CompoundLayout {
    if domain_keys.len() != domain_types.len() {
        return tla_jit_abi::CompoundLayout::Dynamic;
    }
    let keys = domain_keys
        .iter()
        .zip(domain_types.iter())
        .map(|(key, slot_type)| {
            let name = tla_core::intern_name(key);
            match slot_type {
                super::state_layout::SlotType::String => {
                    Some(tla_jit_abi::SetBitmaskElement::String(name))
                }
                super::state_layout::SlotType::ModelValue => {
                    Some(tla_jit_abi::SetBitmaskElement::ModelValue(name))
                }
                _ => None,
            }
        })
        .collect::<Option<Vec<_>>>();
    keys.map_or(tla_jit_abi::CompoundLayout::Dynamic, |keys| {
        tla_jit_abi::CompoundLayout::ExplicitScalarDomain {
            key_layout: Box::new(tla_jit_abi::CompoundLayout::String),
            keys,
        }
    })
}

/// Flatten one `Value::Record` from a record-set-bitmask universe into the
/// native ABI carrier's `(field_name, scalar)` tuple, in the universe's
/// canonical record field order (field-name string), preserving it exactly.
///
/// Returns `None` (fail-closed) when the value is not a record, or when any
/// field value is not one of the four scalar leaves the native bitmask element
/// supports. `Value::Int` (arbitrary-precision BigInt) is deliberately rejected:
/// the native `SetBitmaskElement::Int` is `i64`-only, so a BigInt field cannot
/// be represented and the whole universe must fail closed.
/// Fold a flat set of scalar leaves into a single deterministic `i64` carrier
/// value. Order-independent (XOR of per-element mixes) so it is invariant to set
/// iteration order, and total over `SmallInt` / `Bool` / `String` / `ModelValue`
/// elements. Returns `None` (fail-closed) if any element is not a scalar leaf,
/// exactly mirroring `record_set_bitmask_field_native_representable`'s set arm so
/// this bridge never disagrees with the flat-primary admission gate.
///
/// The returned value is intentionally NOT a faithful set-membership encoding:
/// it is never used by a firing native op (they reject a set-shaped runtime
/// element field first) nor by flat-state storage (which uses the interpreter's
/// full-`Value` bit assignment). It only distinguishes distinct sets so the
/// carrier universe stays deterministic across runs.
fn fold_scalar_set_field(set: &tla_value::value::SortedSet) -> Option<i64> {
    const MIX: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut acc: u64 = 0;
    for elem in set.iter() {
        let leaf: u64 = match elem {
            crate::Value::SmallInt(n) => (*n as u64).wrapping_mul(MIX),
            crate::Value::Bool(b) => (u64::from(*b)).wrapping_add(1).wrapping_mul(MIX),
            crate::Value::String(s) => (tla_core::intern_name(s).0 as u64).wrapping_mul(MIX ^ 0x1),
            crate::Value::ModelValue(m) => {
                (tla_core::intern_name(m).0 as u64).wrapping_mul(MIX ^ 0x2)
            }
            _ => return None,
        };
        acc ^= leaf.rotate_left(17);
    }
    Some(acc as i64)
}

fn record_to_jit_bitmask_fields(
    value: &crate::Value,
) -> Option<Vec<(tla_core::NameId, tla_jit_abi::SetBitmaskElement)>> {
    let crate::Value::Record(record) = value else {
        return None;
    };
    // `RecordValue::iter` yields fields in the canonical field-name-string
    // order; the JIT carrier records bit-index order over the universe, and the
    // consumer re-canonicalizes each record's field order through
    // `RecordBitKey::from_fields`, so any order here is sound — but emitting the
    // canonical order keeps the carrier deterministic.
    record
        .iter()
        .map(|(name, field_value)| {
            let element = match field_value {
                crate::Value::SmallInt(n) => tla_jit_abi::SetBitmaskElement::Int(*n),
                crate::Value::Bool(b) => tla_jit_abi::SetBitmaskElement::Bool(*b),
                crate::Value::String(s) => {
                    tla_jit_abi::SetBitmaskElement::String(tla_core::intern_name(s))
                }
                crate::Value::ModelValue(m) => {
                    tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(m))
                }
                // A set-valued field (e.g. `rsrc : SUBSET Resources`) is folded
                // to a single deterministic scalar `Int` carrier slot. This fold
                // is NOT load-bearing for soundness: every native record-set op
                // (enum-fold union/diff, membership) fails closed on a set-shaped
                // runtime element field BEFORE this constant is compared, so the
                // action falls back to the interpreter, and flat-state STORAGE bit
                // assignment goes through the interpreter's full-`Value`
                // `record_set_bitmask_value_to_slots`, never this fold. It only
                // needs to be total and deterministic so the carrier is a faithful
                // `RecordSetBitmask` (never `Dynamic`), keeping the flat-primary
                // gate and this bridge in agreement. Fold is order-independent
                // (XOR) so it does not depend on set iteration order.
                crate::Value::Set(set) => {
                    tla_jit_abi::SetBitmaskElement::Int(fold_scalar_set_field(set)?)
                }
                _ => return None,
            };
            Some((name, element))
        })
        .collect()
}

/// Convert an entire record-set-bitmask universe (`Vec<Value::Record>` in
/// canonical bit-index order) into the native ABI carrier's universe.
///
/// Returns `None` (fail-closed) if ANY universe record cannot be flattened to a
/// scalar field tuple; a partial universe must never reach the carrier (the
/// lowering would then assign a bit index to a record the native side cannot
/// represent — an unsound encoding). The universe ORDER is preserved exactly
/// (bit `i` stays mapped to `universe[i]`), matching the interpreter's
/// `record_set_bitmask_value_to_slots`.
fn record_set_bitmask_universe_to_jit(
    universe: &[crate::Value],
) -> Option<Vec<Vec<(tla_core::NameId, tla_jit_abi::SetBitmaskElement)>>> {
    universe.iter().map(record_to_jit_bitmask_fields).collect()
}

fn flat_scalar_to_jit_bitmask_element(
    value: &super::state_layout::FlatScalarValue,
) -> tla_jit_abi::SetBitmaskElement {
    match value {
        super::state_layout::FlatScalarValue::Int(n) => tla_jit_abi::SetBitmaskElement::Int(*n),
        super::state_layout::FlatScalarValue::Bool(b) => tla_jit_abi::SetBitmaskElement::Bool(*b),
        super::state_layout::FlatScalarValue::String(s) => {
            tla_jit_abi::SetBitmaskElement::String(tla_core::intern_name(s))
        }
        super::state_layout::FlatScalarValue::ModelValue(s) => {
            tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(s))
        }
    }
}

// Test-only reverse-direction (JIT ABI -> tla-check) layout helpers. The forward
// direction is what production JIT compilation uses; the reverse only backs
// roundtrip tests.
#[cfg_attr(not(test), allow(dead_code))]
fn jit_bitmask_element_to_flat_scalar(
    value: tla_jit_abi::SetBitmaskElement,
) -> super::state_layout::FlatScalarValue {
    match value {
        tla_jit_abi::SetBitmaskElement::Int(n) => super::state_layout::FlatScalarValue::Int(n),
        tla_jit_abi::SetBitmaskElement::Bool(b) => super::state_layout::FlatScalarValue::Bool(b),
        tla_jit_abi::SetBitmaskElement::String(name) => {
            super::state_layout::FlatScalarValue::String(tla_core::resolve_name_id(name))
        }
        tla_jit_abi::SetBitmaskElement::ModelValue(name) => {
            super::state_layout::FlatScalarValue::ModelValue(tla_core::resolve_name_id(name))
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn explicit_string_domain_from_jit(
    key_layout: &tla_jit_abi::CompoundLayout,
) -> Option<(Vec<std::sync::Arc<str>>, Vec<SlotType>)> {
    let tla_jit_abi::CompoundLayout::ExplicitScalarDomain {
        key_layout,
        keys: jit_keys,
    } = key_layout
    else {
        return None;
    };
    if !matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::String) {
        return None;
    }

    jit_keys
        .iter()
        .map(|key| match key {
            tla_jit_abi::SetBitmaskElement::String(name) => {
                Some((tla_core::resolve_name_id(*name), SlotType::String))
            }
            tla_jit_abi::SetBitmaskElement::ModelValue(name) => {
                Some((tla_core::resolve_name_id(*name), SlotType::ModelValue))
            }
            _ => None,
        })
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
fn tagged_scalar_set_range_from_jit(
    value_layout: &tla_jit_abi::CompoundLayout,
) -> Option<TaggedScalarSetRangeProof> {
    let tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
        scalar_kind,
        set_universe,
        proof_source,
    } = value_layout
    else {
        return None;
    };
    TaggedScalarSetRangeProof::new(
        jit_scalar_kind_to_slot_type(*scalar_kind),
        set_universe
            .iter()
            .copied()
            .map(jit_bitmask_element_to_flat_scalar)
            .collect(),
        tla_core::resolve_name_id(*proof_source),
    )
    .ok()
}

/// Convert a native ABI `StateLayout` into the equivalent tla-check `StateLayout`.
///
/// This is the inverse of `check_layout_to_jit_layout`. Used when a native
/// backend produces a layout (e.g., from `infer_var_layout`) and the model
/// checker needs to create a compatible flat buffer.
///
/// Requires a `VarRegistry` to populate variable names in the tla-check layout.
///
/// # Mapping
///
/// | native ABI `VarLayout`        | tla-check `VarLayoutKind`                     |
/// |------------------------------|-----------------------------------------------|
/// | `ScalarInt`                  | `Scalar`                                      |
/// | `ScalarBool`                 | `ScalarBool`                                  |
/// | `Compound(Function{Int,*,n,lo})` | `IntArray { lo, len }` (if int-array-like) |
/// | `Compound(Record{fields})`   | `Record { field_names }` (if all scalar)      |
/// | `Compound(Dynamic)`          | `Dynamic`                                     |
/// | Other `Compound`             | `Dynamic` (fallback)                          |
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn jit_layout_to_check_layout(
    jit_layout: &tla_jit_abi::StateLayout,
    registry: &crate::var_index::VarRegistry,
) -> StateLayout {
    let kinds: Vec<VarLayoutKind> = (0..jit_layout.var_count())
        .map(|i| {
            let jit_var = jit_layout
                .var_layout(i)
                .expect("native ABI layout var_count mismatch");
            jit_var_to_check_var(jit_var)
        })
        .collect();
    StateLayout::new(registry, kinds)
}

/// Convert a single native ABI `VarLayout` to a tla-check `VarLayoutKind`.
#[cfg_attr(not(test), allow(dead_code))]
fn jit_var_to_check_var(jit_var: &tla_jit_abi::VarLayout) -> VarLayoutKind {
    match jit_var {
        tla_jit_abi::VarLayout::ScalarInt => VarLayoutKind::Scalar,
        tla_jit_abi::VarLayout::ScalarBool => VarLayoutKind::ScalarBool,
        tla_jit_abi::VarLayout::Compound(compound) => compound_to_check_var(compound),
        // non_exhaustive: future VarLayout variants fall back to Dynamic.
        _ => VarLayoutKind::Dynamic,
    }
}

/// Convert a native ABI `CompoundLayout` to a tla-check `VarLayoutKind`.
#[cfg_attr(not(test), allow(dead_code))]
fn compound_to_check_var(compound: &tla_jit_abi::CompoundLayout) -> VarLayoutKind {
    match compound {
        tla_jit_abi::CompoundLayout::Int => VarLayoutKind::Scalar,
        tla_jit_abi::CompoundLayout::Bool => VarLayoutKind::ScalarBool,
        tla_jit_abi::CompoundLayout::String => VarLayoutKind::ScalarString,

        // Integer-array function: [lo..hi -> Int/Bool]
        tla_jit_abi::CompoundLayout::Function {
            key_layout,
            value_layout,
            pair_count: Some(len),
            domain_lo: Some(lo),
        } if key_layout.is_scalar() && value_layout.is_scalar() => {
            let elements_are_bool = matches!(**value_layout, tla_jit_abi::CompoundLayout::Bool);
            VarLayoutKind::IntArray {
                lo: *lo,
                len: *len,
                elements_are_bool,
                element_types: None,
                // The JIT ABI does not carry the finite-universe element proof;
                // a roundtripped layout fails closed (never auto-admitted).
                element_range_proof: None,
            }
        }

        // Proof-bearing string/model-value-keyed function:
        // [{p1,p2,...} -> scalar | SetBitmask(P)].
        tla_jit_abi::CompoundLayout::Function {
            key_layout,
            value_layout,
            pair_count: Some(len),
            domain_lo: None,
        } if matches!(
            value_layout.as_ref(),
            tla_jit_abi::CompoundLayout::TaggedScalarOrSet { .. }
        ) =>
        {
            let Some((domain_keys, domain_types)) =
                explicit_string_domain_from_jit(key_layout.as_ref())
            else {
                return VarLayoutKind::Dynamic;
            };
            let Some(proof) = tagged_scalar_set_range_from_jit(value_layout.as_ref()) else {
                return VarLayoutKind::Dynamic;
            };
            if domain_keys.len() != *len {
                return VarLayoutKind::Dynamic;
            }
            VarLayoutKind::StringKeyedArray {
                domain_keys,
                domain_types,
                value_types: vec![proof.scalar_type(); *len],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
            }
        }

        // String-keyed function: [{"a","b"} -> Int/Bool/String]
        // Reverse of StringKeyedArray -> Function { String -> T, n, None }.
        // Part of #3908.
        tla_jit_abi::CompoundLayout::Function {
            key_layout,
            value_layout,
            pair_count: Some(len),
            domain_lo: None,
        } if matches!(**key_layout, tla_jit_abi::CompoundLayout::String)
            && value_layout.is_scalar() =>
        {
            // We cannot recover the domain key strings from the native ABI layout
            // alone (NameIds are not stored). Return Dynamic as a safe fallback
            // for the reverse direction. The forward direction (check -> jit) is
            // what matters for JIT compilation; the reverse is only used for
            // testing roundtrips which typically go through the check layout.
            // To make this fully invertible we'd need to store domain_keys in
            // the JIT CompoundLayout, which is future work.
            let _ = len;
            VarLayoutKind::Dynamic
        }

        tla_jit_abi::CompoundLayout::Function {
            key_layout,
            value_layout,
            pair_count: Some(len),
            domain_lo: None,
        } if matches!(
            key_layout.as_ref(),
            tla_jit_abi::CompoundLayout::ExplicitScalarDomain {
                key_layout,
                ..
            } if matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::String)
        ) && value_layout.is_scalar() =>
        {
            let _ = len;
            VarLayoutKind::Dynamic
        }

        // Record with all-scalar fields
        tla_jit_abi::CompoundLayout::Record { fields } => {
            let all_scalar = fields.iter().all(|(_, layout)| layout.is_scalar());
            if all_scalar && !fields.is_empty() {
                let field_names: Vec<std::sync::Arc<str>> = fields
                    .iter()
                    .map(|(nid, _)| tla_core::resolve_name_id(*nid))
                    .collect();
                let field_is_bool: Vec<bool> = fields
                    .iter()
                    .map(|(_, layout)| matches!(layout, tla_jit_abi::CompoundLayout::Bool))
                    .collect();
                let field_types: Vec<super::state_layout::SlotType> = fields
                    .iter()
                    .map(|(_, layout)| match layout {
                        tla_jit_abi::CompoundLayout::Bool => super::state_layout::SlotType::Bool,
                        tla_jit_abi::CompoundLayout::String => {
                            super::state_layout::SlotType::String
                        }
                        _ => super::state_layout::SlotType::Int,
                    })
                    .collect();
                VarLayoutKind::Record {
                    field_names,
                    field_is_bool,
                    field_types,
                    // The JIT ABI does not carry per-field finite-universe proofs;
                    // a roundtripped record fails closed (never auto-admitted).
                    field_range_proofs: None,
                }
            } else {
                VarLayoutKind::Dynamic
            }
        }

        // Explicit dynamic
        tla_jit_abi::CompoundLayout::Dynamic => VarLayoutKind::Dynamic,

        tla_jit_abi::CompoundLayout::SetBitmask {
            universe,
            is_proven_closed,
        } => VarLayoutKind::Recursive {
            layout: super::state_layout::FlatValueLayout::SetBitmask {
                universe: universe
                    .iter()
                    .copied()
                    .map(jit_bitmask_element_to_flat_scalar)
                    .collect(),
                // The JIT ABI descriptor carries only a proven-closed bit, not
                // the originating invariant source string. Reconstruct
                // `ProvenClosed` with an ABI-round-trip marker when the bit is
                // set so the function-range flat-primary path can re-admit;
                // otherwise stay fail-closed (`Sampled`).
                universe_closure: if *is_proven_closed {
                    super::state_layout::SetBitmaskUniverseClosure::ProvenClosed {
                        invariant: std::sync::Arc::from("jit-abi-roundtrip-proven-closed"),
                    }
                } else {
                    super::state_layout::SetBitmaskUniverseClosure::Sampled
                },
            },
        },

        // All other compound types: fallback to Dynamic
        _ => VarLayoutKind::Dynamic,
    }
}

/// Verify that two layouts are structurally compatible for buffer sharing.
///
/// Two layouts are compatible when they produce the same total slot count,
/// each variable maps to the same compact offset/width, and compound shapes
/// agree recursively. This means a flat buffer created with one layout can be
/// read by code using the other without changing bit meaning.
///
/// Does NOT require the layouts to be identical — scalar slot-compatible
/// shapes may still share a buffer.
/// For example, tla-check's `IntArray{lo=0, len=3}` is compatible with
/// the native ABI's `Compound(Function{Int->Int, n=3, lo=0})` because both
/// produce 3 contiguous i64 slots.
#[must_use]
pub(crate) fn layouts_compatible(
    check_layout: &StateLayout,
    jit_layout: &tla_jit_abi::StateLayout,
) -> bool {
    if check_layout.var_count() != jit_layout.var_count() {
        return false;
    }

    let jit_offsets = jit_layout.compute_compact_var_offsets();
    for (i, &jit_offset) in jit_offsets.iter().enumerate() {
        let check_var = check_layout
            .var_layout(i)
            .expect("check var_count mismatch");
        let jit_var = jit_layout.var_layout(i).expect("jit var_count mismatch");

        let check_slots = check_var.kind.slot_count();
        let jit_slots = jit_var.compact_slot_count();

        if check_var.offset != jit_offset || check_slots != jit_slots {
            return false;
        }

        if !compact_var_layouts_compatible(&check_var.kind, jit_var) {
            return false;
        }
    }

    true
}

fn compact_var_layouts_compatible(
    check_kind: &VarLayoutKind,
    jit_var: &tla_jit_abi::VarLayout,
) -> bool {
    match check_kind {
        VarLayoutKind::Scalar => scalar_var_compatible(SlotType::Int, jit_var),
        VarLayoutKind::ScalarBool => scalar_var_compatible(SlotType::Bool, jit_var),
        VarLayoutKind::ScalarString | VarLayoutKind::ScalarModelValue => {
            scalar_var_compatible(SlotType::String, jit_var)
        }
        VarLayoutKind::FixedScalar { .. } => scalar_var_compatible(SlotType::String, jit_var),
        VarLayoutKind::IntArray {
            lo,
            len,
            elements_are_bool,
            element_types,
            ..
        } => match jit_var {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count: Some(pair_count),
                domain_lo: Some(domain_lo),
            }) => {
                matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::Int)
                    && lo == domain_lo
                    && len == pair_count
                    && compound_layout_matches_slot_types(
                        value_layout,
                        *elements_are_bool,
                        element_types.as_deref(),
                    )
            }
            _ => false,
        },
        VarLayoutKind::Record {
            field_names,
            field_types,
            ..
        } => match jit_var {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Record { fields }) => {
                fields.len() == field_names.len()
                    && field_names
                        .iter()
                        .zip(field_types.iter())
                        .zip(fields.iter())
                        .all(|((check_name, check_type), (jit_name, jit_layout))| {
                            tla_core::intern_name(check_name) == *jit_name
                                && compound_layout_matches_slot_type(jit_layout, *check_type)
                        })
            }
            _ => false,
        },
        VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            range_encoding,
        } => match jit_var {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count: Some(pair_count),
                domain_lo: None,
            }) => {
                function_key_layout_matches_string_domain(key_layout, domain_keys, domain_types)
                    && *pair_count == domain_keys.len()
                    && string_keyed_array_range_compatible(
                        value_layout,
                        value_types,
                        range_encoding,
                    )
            }
            _ => false,
        },
        VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding,
        } => match jit_var {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count: Some(pair_count),
                domain_lo: None,
            }) => {
                function_key_layout_matches_tuple_domain(key_layout, domain_keys)
                    && *pair_count == domain_keys.len()
                    && tuple_keyed_array_range_compatible(value_layout, value_types, range_encoding)
            }
            _ => false,
        },
        VarLayoutKind::Recursive { layout } => match jit_var {
            tla_jit_abi::VarLayout::Compound(jit_layout) => {
                flat_layout_compact_compatible(layout, jit_layout)
            }
            _ => false,
        },
        VarLayoutKind::Bitmask { .. } => matches!(jit_var, tla_jit_abi::VarLayout::ScalarInt),
        VarLayoutKind::Dynamic => matches!(
            jit_var,
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Dynamic)
        ),
    }
}

fn scalar_var_compatible(slot_type: SlotType, jit_var: &tla_jit_abi::VarLayout) -> bool {
    match (slot_type, jit_var) {
        (SlotType::Int, tla_jit_abi::VarLayout::ScalarInt)
        | (SlotType::Bool, tla_jit_abi::VarLayout::ScalarBool) => true,
        (_, tla_jit_abi::VarLayout::Compound(layout)) => {
            compound_layout_matches_slot_type(layout, slot_type)
        }
        _ => false,
    }
}

fn compound_layout_matches_slot_types(
    jit: &tla_jit_abi::CompoundLayout,
    elements_are_bool: bool,
    slot_types: Option<&[SlotType]>,
) -> bool {
    match slot_types {
        Some(types) => compound_layout_matches_uniform_slot_types(jit, types),
        None if elements_are_bool => compound_layout_matches_slot_type(jit, SlotType::Bool),
        None => compound_layout_matches_slot_type(jit, SlotType::Int),
    }
}

fn compound_layout_matches_uniform_slot_types(
    jit: &tla_jit_abi::CompoundLayout,
    slot_types: &[SlotType],
) -> bool {
    let Some(first) = slot_types.first() else {
        return matches!(jit, tla_jit_abi::CompoundLayout::Dynamic);
    };
    if !slot_types.iter().all(|slot_type| slot_type == first) {
        return matches!(jit, tla_jit_abi::CompoundLayout::Dynamic);
    }
    compound_layout_matches_slot_type(jit, *first)
}

fn string_keyed_array_range_compatible(
    jit: &tla_jit_abi::CompoundLayout,
    value_types: &[SlotType],
    range_encoding: &StringKeyedArrayRangeEncoding,
) -> bool {
    match range_encoding {
        StringKeyedArrayRangeEncoding::ScalarSlots
        | StringKeyedArrayRangeEncoding::FixedScalar(_) => {
            compound_layout_matches_uniform_slot_types(jit, value_types)
        }
        StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof) => {
            value_types
                .iter()
                .all(|slot_type| *slot_type == proof.scalar_type())
                && tagged_scalar_set_range_compatible(proof, jit)
        }
    }
}

/// Verify a tuple-keyed function's JIT value layout matches its check-side
/// range encoding (mirror of [`string_keyed_array_range_compatible`]). Under
/// `TaggedScalarUnion` the JIT value must be a `TaggedScalarUnion` carrier whose
/// universe matches the proof member-for-member in canonical index order, so
/// index `i` decodes to the identical value on both sides.
fn tuple_keyed_array_range_compatible(
    jit: &tla_jit_abi::CompoundLayout,
    value_types: &[SlotType],
    range_encoding: &TupleKeyedArrayRangeEncoding,
) -> bool {
    match range_encoding {
        // `FixedScalar` shares the raw-`NameId` value layout with `ScalarSlots`;
        // additionally require every sampled slot type to equal the proof's
        // scalar type (mirror of `string_keyed_array_range_compatible`).
        TupleKeyedArrayRangeEncoding::ScalarSlots => {
            compound_layout_matches_uniform_slot_types(jit, value_types)
        }
        TupleKeyedArrayRangeEncoding::FixedScalar(proof) => {
            value_types
                .iter()
                .all(|slot_type| *slot_type == proof.scalar_type())
                && compound_layout_matches_uniform_slot_types(jit, value_types)
        }
        TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof) => match jit {
            tla_jit_abi::CompoundLayout::TaggedScalarUnion {
                universe,
                proof_source,
            } => {
                set_bitmask_universe_compatible(proof.universe(), universe)
                    && *proof_source == tla_core::intern_name(proof.source().as_ref())
            }
            _ => false,
        },
    }
}

fn tagged_scalar_set_range_compatible(
    proof: &TaggedScalarSetRangeProof,
    jit: &tla_jit_abi::CompoundLayout,
) -> bool {
    match jit {
        tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
            scalar_kind,
            set_universe,
            proof_source,
        } => {
            jit_scalar_kind_to_slot_type(*scalar_kind) == proof.scalar_type()
                && set_bitmask_universe_compatible(proof.set_universe(), set_universe)
                && *proof_source == tla_core::intern_name(proof.source().as_ref())
        }
        _ => false,
    }
}

fn compound_layout_matches_slot_type(
    jit: &tla_jit_abi::CompoundLayout,
    slot_type: SlotType,
) -> bool {
    matches!(
        (slot_type, jit),
        (SlotType::Int, tla_jit_abi::CompoundLayout::Int)
            | (SlotType::Bool, tla_jit_abi::CompoundLayout::Bool)
            | (
                SlotType::String | SlotType::ModelValue,
                tla_jit_abi::CompoundLayout::String
            )
    )
}

fn flat_layout_compact_compatible(
    check: &FlatValueLayout,
    jit: &tla_jit_abi::CompoundLayout,
) -> bool {
    match (check, jit) {
        (FlatValueLayout::Scalar(slot_type), _) => {
            compound_layout_matches_slot_type(jit, *slot_type)
        }
        (
            FlatValueLayout::SetBitmask {
                universe: check, ..
            },
            tla_jit_abi::CompoundLayout::SetBitmask { universe: jit, .. },
        ) => set_bitmask_universe_compatible(check, jit),
        // A tagged scalar-union slot is compact-compatible only with a native
        // `TaggedScalarUnion` carrier over the EXACT same ordered universe: the
        // slot stores a universe index, so any divergence in universe membership
        // or order would remap indices to different values. Every other pairing
        // fails closed.
        (
            FlatValueLayout::TaggedScalarUnion { proof },
            tla_jit_abi::CompoundLayout::TaggedScalarUnion { universe: jit, .. },
        ) => set_bitmask_universe_compatible(proof.universe(), jit),
        (FlatValueLayout::TaggedScalarUnion { .. }, _) => false,
        (
            FlatValueLayout::IntFunction {
                lo,
                len,
                value_layout,
            },
            tla_jit_abi::CompoundLayout::Function {
                key_layout,
                pair_count: Some(pair_count),
                domain_lo: Some(domain_lo),
                value_layout: jit_value,
            },
        ) => {
            matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::Int)
                && lo == domain_lo
                && len == pair_count
                && flat_layout_compact_compatible(value_layout, jit_value)
        }
        (
            FlatValueLayout::Function {
                domain,
                value_layout,
            },
            tla_jit_abi::CompoundLayout::Function {
                key_layout,
                pair_count: Some(pair_count),
                domain_lo: Some(domain_lo),
                value_layout: jit_value,
            },
        ) => {
            matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::Int)
                && ordered_dense_int_domain(domain) == Some((*domain_lo, *pair_count))
                && flat_layout_compact_compatible(value_layout, jit_value)
        }
        (
            FlatValueLayout::Function {
                domain,
                value_layout,
            },
            tla_jit_abi::CompoundLayout::Function {
                key_layout,
                pair_count: Some(pair_count),
                domain_lo: None,
                value_layout: jit_value,
            },
        ) => {
            domain.len() == *pair_count
                && ordered_dense_int_domain(domain).is_none()
                && function_key_layout_matches_flat_domain(domain, key_layout)
                && flat_layout_compact_compatible(value_layout, jit_value)
        }
        (
            FlatValueLayout::Record {
                field_names,
                field_layouts,
            },
            tla_jit_abi::CompoundLayout::Record { fields },
        ) if field_names.len() == fields.len() => field_names
            .iter()
            .zip(field_layouts.iter())
            .zip(fields.iter())
            .all(|((check_name, check_layout), (jit_name, jit_layout))| {
                tla_core::intern_name(check_name) == *jit_name
                    && flat_layout_compact_compatible(check_layout, jit_layout)
            }),
        (
            FlatValueLayout::Sequence {
                max_len,
                element_layout,
                ..
            },
            tla_jit_abi::CompoundLayout::Sequence {
                element_count: Some(element_count),
                element_layout: jit_element,
                ..
            },
        ) => {
            max_len == element_count && flat_layout_compact_compatible(element_layout, jit_element)
        }
        // A record-set bitmask roundtrips iff the JIT carrier's universe matches
        // the check-side universe RECORD-FOR-RECORD in the same bit-index order
        // (so bit `i` maps to the identical record on both sides). The bridge
        // builds the carrier from this same universe, so they agree by
        // construction; this is the fail-closed verification that a malformed or
        // partial carrier (e.g. a universe the bridge had to truncate) is
        // rejected rather than admitted into native dispatch.
        (
            FlatValueLayout::RecordSetBitmask {
                universe: check_universe,
                ..
            },
            tla_jit_abi::CompoundLayout::RecordSetBitmask {
                universe: jit_universe,
                ..
            },
        ) => {
            check_universe.len() == jit_universe.len()
                && check_universe.iter().zip(jit_universe.iter()).all(
                    |(check_record, jit_fields)| {
                        record_to_jit_bitmask_fields(check_record).as_deref()
                            == Some(jit_fields.as_slice())
                    },
                )
        }
        // A tagged union roundtrips iff the JIT carrier has the same variant
        // count and each variant is recursively compatible in canonical tag
        // order (so tag `i` decodes the identical variant on both sides). The
        // bridge builds the carrier from these same variants, so they agree by
        // construction; this is the fail-closed verification that a malformed or
        // partial carrier is rejected rather than admitted into native dispatch.
        (
            FlatValueLayout::TaggedUnion { proof },
            tla_jit_abi::CompoundLayout::TaggedUnion { variants, .. },
        ) => {
            proof.variants().len() == variants.len()
                && proof
                    .variants()
                    .iter()
                    .zip(variants.iter())
                    .all(|(check_variant, jit_variant)| {
                        flat_layout_compact_compatible(check_variant, jit_variant)
                    })
        }
        (FlatValueLayout::TaggedUnion { .. }, _) => false,
        // A fixed-arity heterogeneous tuple roundtrips iff the JIT `Tuple`
        // carrier has the same arity and each position is recursively
        // compatible.
        (
            FlatValueLayout::HeterogeneousTuple { element_layouts },
            tla_jit_abi::CompoundLayout::Tuple {
                element_layouts: jit_layouts,
            },
        ) => {
            element_layouts.len() == jit_layouts.len()
                && element_layouts
                    .iter()
                    .zip(jit_layouts.iter())
                    .all(|(check_layout, jit_layout)| {
                        flat_layout_compact_compatible(check_layout, jit_layout)
                    })
        }
        (FlatValueLayout::HeterogeneousTuple { .. }, _) => false,
        _ => false,
    }
}

fn flat_domain_compact_compatible(
    domain: &[FlatScalarValue],
    jit: &tla_jit_abi::CompoundLayout,
) -> bool {
    let Some(first) = domain.first() else {
        return matches!(jit, tla_jit_abi::CompoundLayout::Dynamic);
    };
    let first_type = first.slot_type();
    domain.iter().all(|key| key.slot_type() == first_type)
        && compound_layout_matches_slot_type(jit, first_type)
}

fn function_key_layout_matches_flat_domain(
    domain: &[FlatScalarValue],
    jit: &tla_jit_abi::CompoundLayout,
) -> bool {
    match jit {
        tla_jit_abi::CompoundLayout::ExplicitScalarDomain { key_layout, keys } => {
            keys.len() == domain.len()
                && (matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::Dynamic)
                    || flat_domain_compact_compatible(domain, key_layout))
                && domain
                    .iter()
                    .zip(keys.iter())
                    .all(|(check, jit)| flat_scalar_to_jit_bitmask_element(check) == *jit)
        }
        _ => flat_domain_compact_compatible(domain, jit),
    }
}

fn function_key_layout_matches_string_domain(
    key_layout: &tla_jit_abi::CompoundLayout,
    domain_keys: &[std::sync::Arc<str>],
    domain_types: &[SlotType],
) -> bool {
    match key_layout {
        tla_jit_abi::CompoundLayout::ExplicitScalarDomain {
            key_layout,
            keys: jit_keys,
        } => {
            matches!(key_layout.as_ref(), tla_jit_abi::CompoundLayout::String)
                && jit_keys.len() == domain_keys.len()
                && domain_keys.len() == domain_types.len()
                && domain_keys
                    .iter()
                    .zip(domain_types.iter())
                    .zip(jit_keys)
                    .all(|((key, slot_type), jit_key)| {
                        let name = tla_core::intern_name(key);
                        match slot_type {
                            SlotType::String => {
                                *jit_key == tla_jit_abi::SetBitmaskElement::String(name)
                            }
                            SlotType::ModelValue => {
                                *jit_key == tla_jit_abi::SetBitmaskElement::ModelValue(name)
                            }
                            _ => false,
                        }
                    })
        }
        _ => matches!(key_layout, tla_jit_abi::CompoundLayout::String),
    }
}

/// True when the JIT key layout is a `Tuple` whose arity and per-position scalar
/// shape agree with the check-side tuple domain keys.
///
/// The JIT infers `Function { key_layout: Tuple{..} }` directly from a
/// scalar-tuple-keyed function value, so a compatible bridge produces the same
/// tuple key layout. All canonical keys share the same arity/shape, so the first
/// key fixes the expected element layouts.
fn function_key_layout_matches_tuple_domain(
    key_layout: &tla_jit_abi::CompoundLayout,
    domain_keys: &[tla_value::Value],
) -> bool {
    use tla_value::Value;
    // An explicit tuple-domain carrier is compatible exactly when its wrapped
    // tuple layout matches AND its carried key table is the identical ordered
    // table this check layout would emit (element-exact, sort-aware).
    if let tla_jit_abi::CompoundLayout::ExplicitTupleDomain {
        key_layout: inner,
        keys,
    } = key_layout
    {
        return function_key_layout_matches_tuple_domain(inner, domain_keys)
            && tuple_domain_keys_to_jit_elements(domain_keys)
                .is_some_and(|expected| expected == *keys);
    }
    let tla_jit_abi::CompoundLayout::Tuple { element_layouts } = key_layout else {
        return false;
    };
    let Some(Value::Tuple(first)) = domain_keys.first() else {
        return false;
    };
    if first.len() != element_layouts.len() {
        return false;
    }
    first
        .iter()
        .zip(element_layouts.iter())
        .all(|(elem, jit_layout)| match elem {
            Value::Bool(_) => matches!(jit_layout, tla_jit_abi::CompoundLayout::Bool),
            Value::String(_) | Value::ModelValue(_) => {
                matches!(jit_layout, tla_jit_abi::CompoundLayout::String)
            }
            Value::SmallInt(_) | Value::Int(_) => {
                matches!(jit_layout, tla_jit_abi::CompoundLayout::Int)
            }
            _ => false,
        })
}

fn set_bitmask_universe_compatible(
    check: &[FlatScalarValue],
    jit: &[tla_jit_abi::SetBitmaskElement],
) -> bool {
    check.len() == jit.len()
        && check
            .iter()
            .zip(jit.iter())
            .all(|(check, jit)| flat_scalar_to_jit_bitmask_element(check) == *jit)
}

/// Compute the compact (no-tag) slot count for an entire native ABI layout.
///
/// This matches the check-side flat buffer width, not the JIT tagged
/// serialization width.
#[must_use]
pub(crate) fn jit_layout_compact_slot_count(jit_layout: &tla_jit_abi::StateLayout) -> usize {
    jit_layout.compact_slot_count()
}

/// Item 4 M0-G5: byte-exact geometry parity between a check-side layout (the
/// encoding `FlatState`/`HybridFlatView::project` produce) and its converted
/// jit-abi layout (the geometry native code compiles against).
///
/// Returns a description of the FIRST divergence — var count, total slot
/// width, or any per-variable offset/slot-count — or `None` when the two
/// geometries agree exactly. The hybrid native cache build fails closed on
/// `Some` (offset-mismatched native code must never dispatch).
#[must_use]
pub(crate) fn layout_geometry_mismatch(
    check_layout: &StateLayout,
    jit_layout: &tla_jit_abi::StateLayout,
) -> Option<String> {
    if check_layout.var_count() != jit_layout.var_count() {
        return Some(format!(
            "var count {} (check) != {} (jit)",
            check_layout.var_count(),
            jit_layout.var_count()
        ));
    }
    if check_layout.total_slots() != jit_layout.compact_slot_count() {
        return Some(format!(
            "total slots {} (check) != {} (jit compact)",
            check_layout.total_slots(),
            jit_layout.compact_slot_count()
        ));
    }
    let jit_offsets = jit_layout.compute_compact_var_offsets();
    for (idx, check_var) in check_layout.iter().enumerate() {
        let jit_var = jit_layout.var_layout(idx)?;
        let jit_offset = *jit_offsets.get(idx)?;
        if check_var.offset != jit_offset {
            return Some(format!(
                "var {idx} offset {} (check) != {} (jit compact)",
                check_var.offset, jit_offset
            ));
        }
        if check_var.slot_count != jit_var.compact_slot_count() {
            return Some(format!(
                "var {idx} slot count {} (check) != {} (jit compact)",
                check_var.slot_count,
                jit_var.compact_slot_count()
            ));
        }
    }
    None
}

/// Return the first variable whose check and JIT compact slot widths disagree.
#[must_use]
pub(crate) fn first_layout_slot_mismatch(
    check_layout: &StateLayout,
    jit_layout: &tla_jit_abi::StateLayout,
) -> Option<(usize, usize, usize)> {
    let count = check_layout.var_count().min(jit_layout.var_count());
    for i in 0..count {
        let check_slots = check_layout.var_layout(i)?.kind.slot_count();
        let jit_slots = jit_layout.var_layout(i)?.compact_slot_count();
        if check_slots != jit_slots {
            return Some((i, check_slots, jit_slots));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::layout_inference::infer_layout;
    use crate::state::state_layout::{StringKeyedArrayRangeEncoding, TaggedScalarSetRangeProof};
    use crate::state::ArrayState;
    use crate::var_index::VarRegistry;
    use crate::Value;
    use std::sync::Arc;
    use tla_value::value::IntIntervalFunc;

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_check_to_jit_all_scalar() {
        let registry = VarRegistry::from_names(["x", "y", "z"]);
        let state = ArrayState::from_values(vec![
            Value::SmallInt(42),
            Value::Bool(true),
            Value::SmallInt(-7),
        ]);
        let check_layout = infer_layout(&state, &registry);
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(jit_layout.var_count(), 3);
        assert!(jit_layout.is_all_scalar());

        // Verify individual var layouts
        assert_eq!(
            jit_layout.var_layout(0),
            Some(&tla_jit_abi::VarLayout::ScalarInt)
        );
        assert_eq!(
            jit_layout.var_layout(1),
            Some(&tla_jit_abi::VarLayout::ScalarBool)
        );
        assert_eq!(
            jit_layout.var_layout(2),
            Some(&tla_jit_abi::VarLayout::ScalarInt)
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_check_to_jit_int_array() {
        let registry = VarRegistry::from_names(["active"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![Value::Bool(false), Value::Bool(true), Value::Bool(false)],
        );
        let state = ArrayState::from_values(vec![Value::IntFunc(Rp::new(func))]);
        let check_layout = infer_layout(&state, &registry);
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(jit_layout.var_count(), 1);
        let var = jit_layout.var_layout(0).unwrap();
        match var {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                pair_count,
                domain_lo,
                ..
            }) => {
                assert_eq!(*pair_count, Some(3));
                assert_eq!(*domain_lo, Some(0));
            }
            other => panic!("expected Compound(Function), got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_string_keyed_array_preserves_explicit_model_value_domain() {
        let registry = VarRegistry::from_names(["color"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2"), Arc::from("p3")],
                domain_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                value_types: vec![SlotType::Int, SlotType::Int, SlotType::Int],
                range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 3);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 3);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            }) => {
                assert_eq!(*pair_count, Some(3));
                assert_eq!(*domain_lo, None);
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::Int
                ));
                match key_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::ExplicitScalarDomain { key_layout, keys } => {
                        assert!(matches!(
                            key_layout.as_ref(),
                            tla_jit_abi::CompoundLayout::String
                        ));
                        assert_eq!(
                            keys.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p1"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p2"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p3"
                                )),
                            ]
                        );
                    }
                    other => panic!("expected explicit scalar domain, got {other:?}"),
                }
            }
            other => panic!("expected explicit-domain function layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tuple_keyed_array_carries_explicit_tuple_domain_table() {
        use crate::state::state_layout::{FixedScalarRangeProof, TupleKeyedArrayRangeEncoding};

        // btree `valOf`-shaped var: [ {1,2} \X {1,2} -> {x, y, nil} ] with a
        // homogeneous model-value range proof. The bridge must carry the exact
        // Value::cmp-sorted tuple-key table (never re-sorted) so the native
        // lowering fires deterministically without const-pool mining.
        let registry = VarRegistry::from_names(["valOf"]);
        let domain_keys: Vec<Value> = vec![
            Value::tuple([Value::SmallInt(1), Value::SmallInt(1)]),
            Value::tuple([Value::SmallInt(1), Value::SmallInt(2)]),
            Value::tuple([Value::SmallInt(2), Value::SmallInt(1)]),
            Value::tuple([Value::SmallInt(2), Value::SmallInt(2)]),
        ];
        let proof = FixedScalarRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(Arc::from("x")),
                FlatScalarValue::ModelValue(Arc::from("y")),
                FlatScalarValue::ModelValue(Arc::from("nil")),
            ],
            Arc::from("TypeOk"),
        )
        .expect("valid proof");
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::TupleKeyedArray {
                domain_keys,
                value_types: vec![SlotType::ModelValue; 4],
                range_encoding: TupleKeyedArrayRangeEncoding::FixedScalar(proof),
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 4);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 4);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        // The var is flat-primary admissible via the range proof (the tuple
        // analogue of the StringKeyedArray FixedScalar route).
        assert!(check_layout
            .iter()
            .all(|var| var.kind.supports_flat_primary()));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            }) => {
                assert_eq!(*pair_count, Some(4));
                assert_eq!(*domain_lo, None);
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::String
                ));
                match key_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::ExplicitTupleDomain { key_layout, keys } => {
                        assert!(matches!(
                            key_layout.as_ref(),
                            tla_jit_abi::CompoundLayout::Tuple { element_layouts }
                                if element_layouts.len() == 2
                        ));
                        let int_row = |a: i64, b: i64| {
                            vec![
                                tla_jit_abi::SetBitmaskElement::Int(a),
                                tla_jit_abi::SetBitmaskElement::Int(b),
                            ]
                        };
                        assert_eq!(
                            keys.as_slice(),
                            &[int_row(1, 1), int_row(1, 2), int_row(2, 1), int_row(2, 2)],
                            "the carried table must be the check-side canonical slot order"
                        );
                    }
                    other => panic!("expected explicit tuple domain, got {other:?}"),
                }
            }
            other => panic!("expected tuple-keyed function layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tuple_keyed_array_mixed_i64_range_stays_fail_closed() {
        use crate::state::state_layout::TupleKeyedArrayRangeEncoding;

        // btree `childOf`-shaped var: Int ∪ model-value range (sampled all-NIL
        // => homogeneous ModelValue value_types, but NO homogeneous proof can
        // exist for the union). Without a FixedScalar proof the var must stay
        // non-flat-primary (the WP-05 TaggedScalarUnion carrier seam), while
        // the bridge still carries the tuple table for read-side lowering.
        let registry = VarRegistry::from_names(["childOf"]);
        let domain_keys: Vec<Value> = vec![
            Value::tuple([Value::SmallInt(1), Value::SmallInt(1)]),
            Value::tuple([Value::SmallInt(1), Value::SmallInt(2)]),
        ];
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::TupleKeyedArray {
                domain_keys,
                value_types: vec![SlotType::ModelValue; 2],
                range_encoding: TupleKeyedArrayRangeEncoding::ScalarSlots,
            }],
        );

        assert!(
            !check_layout
                .iter()
                .any(|var| var.kind.supports_flat_primary()),
            "an unproven model-value tuple-keyed range must stay fail-closed"
        );
        let jit_layout = check_layout_to_jit_layout(&check_layout);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                ..
            }) => {
                assert!(matches!(
                    key_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::ExplicitTupleDomain { .. }
                ));
            }
            other => panic!("expected tuple-keyed function layout, got {other:?}"),
        }
    }

    /// WP-09/Part A: a tuple-keyed function var whose range encoding is the
    /// proven `TaggedScalarUnion` carrier bridges to
    /// `CompoundLayout::TaggedScalarUnion` over the EXACT ordered universe
    /// (plus the explicit tuple-key table), and the check/JIT layouts agree
    /// slot-for-slot. A jit value layout with a divergent universe is
    /// incompatible (fail closed) — the slot stores a universe index, so any
    /// divergence would remap indices.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tuple_keyed_union_range_bridges_to_carrier_preserving_order() {
        use crate::state::state_layout::{TaggedScalarUnionProof, TupleKeyedArrayRangeEncoding};

        let registry = VarRegistry::from_names(["childOf"]);
        let domain_keys: Vec<Value> = vec![
            Value::tuple([Value::SmallInt(1), Value::SmallInt(1)]),
            Value::tuple([Value::SmallInt(1), Value::SmallInt(2)]),
        ];
        let universe = vec![
            FlatScalarValue::Int(1),
            FlatScalarValue::Int(2),
            FlatScalarValue::ModelValue(Arc::from("nil")),
        ];
        let proof =
            TaggedScalarUnionProof::new(universe, Arc::from("TypeOk")).expect("valid proof");
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::TupleKeyedArray {
                domain_keys,
                value_types: vec![SlotType::ModelValue; 2],
                range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof),
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);
        assert_eq!(check_layout.total_slots(), 2);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 2);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            }) => {
                assert_eq!(*pair_count, Some(2));
                assert_eq!(*domain_lo, None);
                assert!(matches!(
                    key_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::ExplicitTupleDomain { .. }
                ));
                match value_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::TaggedScalarUnion { universe, .. } => {
                        assert_eq!(
                            universe.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::Int(1),
                                tla_jit_abi::SetBitmaskElement::Int(2),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "nil"
                                )),
                            ],
                            "the ABI universe must preserve the proof's exact order"
                        );
                    }
                    other => panic!("expected TaggedScalarUnion range carrier, got {other:?}"),
                }
            }
            other => panic!("expected tuple-keyed function layout, got {other:?}"),
        }

        // Divergent-universe carrier: incompatible, fail closed.
        let mismatched_jit = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(tla_jit_abi::CompoundLayout::Tuple {
                    element_layouts: vec![
                        tla_jit_abi::CompoundLayout::Int,
                        tla_jit_abi::CompoundLayout::Int,
                    ],
                }),
                value_layout: Box::new(tla_jit_abi::CompoundLayout::TaggedScalarUnion {
                    universe: vec![
                        tla_jit_abi::SetBitmaskElement::Int(1),
                        tla_jit_abi::SetBitmaskElement::Int(2),
                    ],
                    proof_source: tla_core::intern_name("TypeOk"),
                }),
                pair_count: Some(2),
                domain_lo: None,
            },
        )]);
        assert!(!layouts_compatible(&check_layout, &mismatched_jit));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_tagged_scalar_set_string_keyed_array_exports_abi_proof_identity() {
        let registry = VarRegistry::from_names(["temp"]);
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p3")),
            ],
            Arc::from("DijkstraTempTypeOK"),
        )
        .unwrap();
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2"), Arc::from("p3")],
                domain_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                value_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof.clone()),
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 3);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 3);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            }) => {
                assert_eq!(*pair_count, Some(3));
                assert_eq!(*domain_lo, None);
                match key_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::ExplicitScalarDomain { key_layout, keys } => {
                        assert!(matches!(
                            key_layout.as_ref(),
                            tla_jit_abi::CompoundLayout::String
                        ));
                        assert_eq!(
                            keys.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p1"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p2"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p3"
                                )),
                            ]
                        );
                    }
                    other => panic!("expected explicit model-value domain, got {other:?}"),
                }
                match value_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
                        scalar_kind,
                        set_universe,
                        proof_source,
                    } => {
                        assert_eq!(*scalar_kind, tla_jit_abi::ScalarSlotKind::ModelValue);
                        assert_eq!(
                            set_universe.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p1"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p2"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p3"
                                )),
                            ]
                        );
                        assert_eq!(*proof_source, tla_core::intern_name("DijkstraTempTypeOK"));
                    }
                    other => panic!("expected tagged scalar-or-set range, got {other:?}"),
                }
            }
            other => panic!("expected proof-bearing function layout, got {other:?}"),
        }

        let roundtrip_layout = jit_layout_to_check_layout(&jit_layout, &registry);
        match &roundtrip_layout.var_layout(0).unwrap().kind {
            VarLayoutKind::StringKeyedArray {
                domain_keys,
                domain_types,
                value_types,
                range_encoding,
            } => {
                assert_eq!(
                    domain_keys.as_slice(),
                    &[Arc::from("p1"), Arc::from("p2"), Arc::from("p3")]
                );
                assert_eq!(
                    domain_types.as_slice(),
                    &[
                        SlotType::ModelValue,
                        SlotType::ModelValue,
                        SlotType::ModelValue
                    ]
                );
                assert_eq!(
                    value_types.as_slice(),
                    &[
                        SlotType::ModelValue,
                        SlotType::ModelValue,
                        SlotType::ModelValue
                    ]
                );
                assert_eq!(
                    range_encoding,
                    &StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof)
                );
            }
            other => panic!("expected tagged string-keyed array roundtrip, got {other:?}"),
        }
        assert!(
            roundtrip_layout.supports_flat_bfs_auto_admission(),
            "round-tripped ABI metadata must preserve tagged scalar/set admission"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_tagged_scalar_set_range_identity_mismatch() {
        let registry = VarRegistry::from_names(["temp"]);
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p3")),
            ],
            Arc::from("DijkstraTempTypeOK"),
        )
        .unwrap();
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2"), Arc::from("p3")],
                domain_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                value_types: vec![
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                    SlotType::ModelValue,
                ],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
            }],
        );
        let key_layout = Box::new(tla_jit_abi::CompoundLayout::ExplicitScalarDomain {
            key_layout: Box::new(tla_jit_abi::CompoundLayout::String),
            keys: vec![
                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
            ],
        });

        let scalar_only = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout: key_layout.clone(),
                value_layout: Box::new(tla_jit_abi::CompoundLayout::String),
                pair_count: Some(3),
                domain_lo: None,
            },
        )]);
        assert_eq!(
            first_layout_slot_mismatch(&check_layout, &scalar_only),
            None
        );
        assert!(!layouts_compatible(&check_layout, &scalar_only));

        let wrong_source = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout: key_layout.clone(),
                value_layout: Box::new(tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
                    scalar_kind: tla_jit_abi::ScalarSlotKind::ModelValue,
                    set_universe: vec![
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
                    ],
                    proof_source: tla_core::intern_name("OtherProof"),
                }),
                pair_count: Some(3),
                domain_lo: None,
            },
        )]);
        assert_eq!(
            first_layout_slot_mismatch(&check_layout, &wrong_source),
            None
        );
        assert!(!layouts_compatible(&check_layout, &wrong_source));

        let wrong_universe = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout: Box::new(tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
                    scalar_kind: tla_jit_abi::ScalarSlotKind::ModelValue,
                    set_universe: vec![
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
                    ],
                    proof_source: tla_core::intern_name("DijkstraTempTypeOK"),
                }),
                pair_count: Some(3),
                domain_lo: None,
            },
        )]);
        assert_eq!(
            first_layout_slot_mismatch(&check_layout, &wrong_universe),
            None
        );
        assert!(!layouts_compatible(&check_layout, &wrong_universe));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_recursive_set_bitmask_keeps_set_shape_and_one_compact_slot() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["ack"]);
        let proc_domain = vec![
            FlatScalarValue::Int(1),
            FlatScalarValue::Int(2),
            FlatScalarValue::Int(3),
        ];
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 1,
                    len: 3,
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: proc_domain,
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 3);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 3);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                value_layout,
                ..
            }) => match value_layout.as_ref() {
                tla_jit_abi::CompoundLayout::SetBitmask { universe, .. } => {
                    assert_eq!(
                        universe.as_slice(),
                        &[
                            tla_jit_abi::SetBitmaskElement::Int(1),
                            tla_jit_abi::SetBitmaskElement::Int(2),
                            tla_jit_abi::SetBitmaskElement::Int(3),
                        ]
                    );
                }
                other => panic!("expected set-bitmask range layout, got {other:?}"),
            },
            other => panic!("expected recursive function layout, got {other:?}"),
        }
    }

    /// Item 4 M0-G5: property-style geometry parity over representative hybrid
    /// layouts.
    ///
    /// For every rotation × length of a pool of representative var kinds
    /// (scalars, IntArray, record, capacity-proven recursive sequence, plus
    /// non-admissible kinds that the hybrid view demotes to `Dynamic`
    /// placeholders), the check-side hybrid layout's per-var offsets, per-var
    /// slot counts, and total width must equal the converted jit-abi layout's
    /// compact geometry exactly — the invariant that makes hybrid-compiled
    /// native slot accesses line up with `HybridFlatView::project`'s buffer.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_hybrid_layout_geometry_parity_over_representative_layouts() {
        use super::super::hybrid_flat_view::HybridFlatView;
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence};

        let kind_pool: Vec<VarLayoutKind> = vec![
            VarLayoutKind::Scalar,
            VarLayoutKind::ScalarBool,
            VarLayoutKind::IntArray {
                lo: 0,
                len: 3,
                elements_are_bool: false,
                element_types: None,
                element_range_proof: None,
            },
            VarLayoutKind::Record {
                field_names: vec![Arc::from("a"), Arc::from("b")],
                field_is_bool: vec![false, true],
                field_types: vec![SlotType::Int, SlotType::Bool],
                field_range_proofs: None,
            },
            // Capacity-proven recursive sequence: flat-admissible, 1 length
            // slot + max_len element slots.
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                        invariant: Arc::from("TypeOK"),
                        element_invariant: Arc::from("TypeOK"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            },
            // Non-admissible kinds — the hybrid view demotes them to 1-slot
            // Dynamic placeholders.
            VarLayoutKind::Dynamic,
            VarLayoutKind::ScalarString,
        ];

        let mut covered = 0usize;
        for rotation in 0..kind_pool.len() {
            for len in 1..=kind_pool.len() {
                let kinds: Vec<VarLayoutKind> = (0..len)
                    .map(|i| kind_pool[(rotation + i) % kind_pool.len()].clone())
                    .collect();
                let names: Vec<String> = (0..len).map(|i| format!("v{i}")).collect();
                let registry = VarRegistry::from_names(names.iter().map(String::as_str));
                let check_layout = StateLayout::new(&registry, kinds);

                let Some(view) = HybridFlatView::from_layout(&check_layout, &registry) else {
                    // No flat-admissible var in this combination — the hybrid
                    // path stays inert (also a valid outcome).
                    continue;
                };
                let hybrid_check = view.hybrid_layout();
                let hybrid_jit = check_layout_to_jit_layout(hybrid_check).with_hybrid_flat_view();
                assert!(hybrid_jit.is_hybrid_flat_view());
                assert_eq!(
                    super::layout_geometry_mismatch(hybrid_check, &hybrid_jit),
                    None,
                    "geometry mismatch for rotation={rotation} len={len} kinds={:?}",
                    hybrid_check.iter().map(|v| &v.kind).collect::<Vec<_>>(),
                );
                // Every demoted var occupies exactly one placeholder slot on
                // BOTH sides.
                for (idx, var) in hybrid_check.iter().enumerate() {
                    if matches!(var.kind, VarLayoutKind::Dynamic) {
                        assert_eq!(var.slot_count, 1);
                        assert_eq!(
                            hybrid_jit.var_layout(idx),
                            Some(&tla_jit_abi::VarLayout::Compound(
                                tla_jit_abi::CompoundLayout::Dynamic
                            )),
                        );
                    }
                }
                covered += 1;
            }
        }
        assert!(
            covered >= 30,
            "expected the sweep to cover a meaningful number of layouts, got {covered}"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_recursive_function_preserves_explicit_non_dense_domain() {
        use super::super::state_layout::{FlatScalarValue, FlatValueLayout};

        let registry = VarRegistry::from_names(["pc"]);
        let proc_domain = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("p4")),
        ];
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: proc_domain,
                    value_layout: Box::new(FlatValueLayout::Scalar(SlotType::String)),
                },
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 3);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 3);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            }) => {
                assert_eq!(*pair_count, Some(3));
                assert_eq!(*domain_lo, None);
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::String
                ));
                match key_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::ExplicitScalarDomain { key_layout, keys } => {
                        assert!(matches!(
                            key_layout.as_ref(),
                            tla_jit_abi::CompoundLayout::String
                        ));
                        assert_eq!(
                            keys.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p1"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p2"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "p4"
                                )),
                            ]
                        );
                    }
                    other => panic!("expected explicit scalar domain, got {other:?}"),
                }
            }
            other => panic!("expected recursive function layout, got {other:?}"),
        }
    }

    #[test]
    fn test_tagged_scalar_union_bridges_to_carrier_preserving_order() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, TaggedScalarUnionProof,
        };

        // `Nodes \cup {NIL}` — Int arm {1,2,3} (contiguous ascending prefix by
        // Ord) then the model-value NIL, exactly ty's sorted assembly order.
        let universe = vec![
            FlatScalarValue::Int(1),
            FlatScalarValue::Int(2),
            FlatScalarValue::Int(3),
            FlatScalarValue::ModelValue(Arc::from("NIL")),
        ];
        let proof =
            TaggedScalarUnionProof::new(universe.clone(), Arc::from("ChildOfTypeOK")).unwrap();
        let jit = flat_value_layout_to_jit_compound(&FlatValueLayout::TaggedScalarUnion {
            proof: proof.clone(),
        });

        match &jit {
            tla_jit_abi::CompoundLayout::TaggedScalarUnion {
                universe: jit_universe,
                proof_source,
            } => {
                assert_eq!(
                    jit_universe.as_slice(),
                    &[
                        tla_jit_abi::SetBitmaskElement::Int(1),
                        tla_jit_abi::SetBitmaskElement::Int(2),
                        tla_jit_abi::SetBitmaskElement::Int(3),
                        tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("NIL")),
                    ]
                );
                assert_eq!(*proof_source, tla_core::intern_name("ChildOfTypeOK"));
            }
            other => panic!("expected TaggedScalarUnion carrier, got {other:?}"),
        }

        // Exact universe match is compact-compatible; a divergent universe is
        // not (indices would remap to different values).
        assert!(flat_layout_compact_compatible(
            &FlatValueLayout::TaggedScalarUnion {
                proof: proof.clone()
            },
            &jit,
        ));
        let mismatched = tla_jit_abi::CompoundLayout::TaggedScalarUnion {
            universe: vec![
                tla_jit_abi::SetBitmaskElement::Int(1),
                tla_jit_abi::SetBitmaskElement::Int(2),
                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name("NIL")),
            ],
            proof_source: tla_core::intern_name("ChildOfTypeOK"),
        };
        assert!(!flat_layout_compact_compatible(
            &FlatValueLayout::TaggedScalarUnion { proof },
            &mismatched,
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_recursive_function_preserves_explicit_mixed_scalar_domain() {
        use super::super::state_layout::{FlatScalarValue, FlatValueLayout};

        let registry = VarRegistry::from_names(["pc"]);
        let proc_domain = vec![
            FlatScalarValue::ModelValue(std::sync::Arc::from("rm1")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("rm2")),
            FlatScalarValue::ModelValue(std::sync::Arc::from("rm3")),
            FlatScalarValue::Int(0),
            FlatScalarValue::Int(10),
        ];
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: proc_domain,
                    value_layout: Box::new(FlatValueLayout::Scalar(SlotType::String)),
                },
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 5);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 5);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            }) => {
                assert_eq!(*pair_count, Some(5));
                assert_eq!(*domain_lo, None);
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::String
                ));
                match key_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::ExplicitScalarDomain { key_layout, keys } => {
                        assert!(matches!(
                            key_layout.as_ref(),
                            tla_jit_abi::CompoundLayout::Dynamic
                        ));
                        assert_eq!(
                            keys.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "rm1"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "rm2"
                                )),
                                tla_jit_abi::SetBitmaskElement::ModelValue(tla_core::intern_name(
                                    "rm3"
                                )),
                                tla_jit_abi::SetBitmaskElement::Int(0),
                                tla_jit_abi::SetBitmaskElement::Int(10),
                            ]
                        );
                    }
                    other => panic!("expected explicit dynamic scalar domain, got {other:?}"),
                }
            }
            other => panic!("expected recursive function layout, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_recursive_set_bitmask_plus_tail_uses_abi_compact_offsets() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["ack", "tail"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::Recursive {
                    layout: FlatValueLayout::IntFunction {
                        lo: 1,
                        len: 3,
                        value_layout: Box::new(FlatValueLayout::SetBitmask {
                            universe: vec![
                                FlatScalarValue::Int(1),
                                FlatScalarValue::Int(2),
                                FlatScalarValue::Int(3),
                            ],
                            universe_closure: SetBitmaskUniverseClosure::Sampled,
                        }),
                    },
                },
                VarLayoutKind::ScalarBool,
            ],
        );
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 4);
        assert_eq!(check_layout.var_layout(1).unwrap().offset, 3);
        assert_eq!(jit_layout.compact_slot_count(), 4);
        assert_eq!(jit_layout.compute_compact_var_offsets(), vec![0, 3]);
        assert_eq!(jit_layout.compute_var_offsets(), vec![Some(0), Some(11)]);
        assert_ne!(
            jit_layout.compute_compact_var_offsets(),
            jit_layout
                .compute_var_offsets()
                .into_iter()
                .map(|offset| offset.unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_recursive_set_bitmask_universe_value_mismatch() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["ack"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 1,
                    len: 3,
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![
                            FlatScalarValue::Int(1),
                            FlatScalarValue::Int(2),
                            FlatScalarValue::Int(3),
                        ],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                value_layout: Box::new(tla_jit_abi::CompoundLayout::SetBitmask {
                    universe: vec![
                        tla_jit_abi::SetBitmaskElement::Int(1),
                        tla_jit_abi::SetBitmaskElement::Int(2),
                        tla_jit_abi::SetBitmaskElement::Int(4),
                    ],
                    is_proven_closed: false,
                }),
                pair_count: Some(3),
                domain_lo: Some(1),
            },
        )]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_recursive_set_bitmask_universe_order_mismatch() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["ack"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 1,
                    len: 3,
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![
                            FlatScalarValue::Int(1),
                            FlatScalarValue::Int(2),
                            FlatScalarValue::Int(3),
                        ],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                value_layout: Box::new(tla_jit_abi::CompoundLayout::SetBitmask {
                    universe: vec![
                        tla_jit_abi::SetBitmaskElement::Int(2),
                        tla_jit_abi::SetBitmaskElement::Int(1),
                        tla_jit_abi::SetBitmaskElement::Int(3),
                    ],
                    is_proven_closed: false,
                }),
                pair_count: Some(3),
                domain_lo: Some(1),
            },
        )]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_recursive_set_bitmask_replaced_by_scalar() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["enabled"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::SetBitmask {
                    universe: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                    universe_closure: SetBitmaskUniverseClosure::Sampled,
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::ScalarInt]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_sequence_set_bitmask_universe_mismatch() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SequenceBoundEvidence, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["history"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedHistory"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![
                            FlatScalarValue::Int(1),
                            FlatScalarValue::Int(2),
                            FlatScalarValue::Int(3),
                        ],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Sequence {
                element_layout: Box::new(tla_jit_abi::CompoundLayout::SetBitmask {
                    universe: vec![
                        tla_jit_abi::SetBitmaskElement::Int(1),
                        tla_jit_abi::SetBitmaskElement::Int(2),
                        tla_jit_abi::SetBitmaskElement::Int(4),
                    ],
                    is_proven_closed: false,
                }),
                element_count: Some(2),
                capacity_proven: false,
            },
        )]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_recursive_sequence_of_set_bitmask_uses_compact_set_slots() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SequenceBoundEvidence, SetBitmaskUniverseClosure,
        };

        let registry = VarRegistry::from_names(["history"]);
        let proc_domain = vec![
            FlatScalarValue::Int(1),
            FlatScalarValue::Int(2),
            FlatScalarValue::Int(3),
        ];
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedHistory"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: proc_domain,
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 3);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 3);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Sequence {
                element_layout,
                element_count,
                capacity_proven,
            }) => {
                assert_eq!(*element_count, Some(2));
                // The source layout carried a `ProvenInvariant` bound, so the
                // bridge must propagate it as a proven capacity upper bound.
                assert!(
                    *capacity_proven,
                    "ProvenInvariant sequence bound must bridge to capacity_proven=true"
                );
                assert_eq!(element_layout.compact_slot_count(), 1);
                match element_layout.as_ref() {
                    tla_jit_abi::CompoundLayout::SetBitmask { universe, .. } => {
                        assert_eq!(
                            universe.as_slice(),
                            &[
                                tla_jit_abi::SetBitmaskElement::Int(1),
                                tla_jit_abi::SetBitmaskElement::Int(2),
                                tla_jit_abi::SetBitmaskElement::Int(3),
                            ]
                        );
                    }
                    other => panic!("expected compact set-bitmask sequence element, got {other:?}"),
                }
            }
            other => panic!("expected recursive sequence layout, got {other:?}"),
        }
    }

    /// Soundness guard for the sequence-capacity bridge seam
    /// (`FlatValueLayout::Sequence.bound` -> `CompoundLayout::Sequence.capacity_proven`).
    ///
    /// A `max_len` that is only OBSERVED from sampled states is NOT a proven
    /// upper bound on `Len(seq)` across all reachable states. If the bridge
    /// leaked `capacity_proven = true` for such a bound, the native path could
    /// flat-slot-overflow a longer runtime sequence (corrupt dedup / wrong
    /// counts). This locks the fail-closed direction: only a genuinely proven
    /// bound (the f1d0571b duplicate-free proof, a checked invariant, or a fixed
    /// domain-type proof) may bridge to `capacity_proven = true`; every other
    /// bound (`Observed`) must bridge to `false` while still carrying the storage
    /// width (`element_count`).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_sequence_capacity_proven_bridge_is_fail_closed() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = VarRegistry::from_names(["sched"]);
        let bridge_capacity_proven = |bound: SequenceBoundEvidence| -> (Option<usize>, bool) {
            let check_layout = StateLayout::new(
                &registry,
                vec![VarLayoutKind::Recursive {
                    layout: FlatValueLayout::Sequence {
                        bound,
                        max_len: 3,
                        element_layout: Box::new(FlatValueLayout::Scalar(SlotType::ModelValue)),
                    },
                }],
            );
            let jit_layout = check_layout_to_jit_layout(&check_layout);
            match jit_layout.var_layout(0).unwrap() {
                tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Sequence {
                    element_count,
                    capacity_proven,
                    ..
                }) => (*element_count, *capacity_proven),
                other => panic!("expected sequence layout, got {other:?}"),
            }
        };

        // PROVEN capacity (a checked source-level bound, e.g. the f1d0571b
        // duplicate-free `sched \in Seq(Clients)` proof) => capacity_proven=true.
        assert_eq!(
            bridge_capacity_proven(SequenceBoundEvidence::ProvenInvariant {
                invariant: Arc::from("TypeInvariant"),
            }),
            (Some(3), true),
            "a checked-invariant capacity proof must bridge to capacity_proven=true",
        );
        // A proof that also fixes the element layout is still proven.
        assert_eq!(
            bridge_capacity_proven(SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                invariant: Arc::from("TypeInvariant"),
                element_invariant: Arc::from("TypeInvariant"),
            }),
            (Some(3), true),
        );
        // A fixed finite function-domain type proof is proven.
        assert_eq!(
            bridge_capacity_proven(SequenceBoundEvidence::FixedDomainTypeLayout {
                invariant: Arc::from("TypeInvariant"),
            }),
            (Some(3), true),
        );
        // OBSERVED-only bound (sampled from wavefront states) => fail closed.
        // The storage width still bridges, but capacity_proven MUST be false so
        // the native capacity-driven paths never fire on an unproven bound.
        assert_eq!(
            bridge_capacity_proven(SequenceBoundEvidence::Observed),
            (Some(3), false),
            "an observed-only sequence bound must bridge to capacity_proven=false (fail closed)",
        );
        // HEURISTIC element-universe capacity: also NOT a certified bound, so it
        // MUST bridge to capacity_proven=false — the native lowering treats it
        // exactly like Observed (fail closed). Only the flat STORAGE path (with
        // its own SequenceLengthExceedsCapacity overflow backstop) relies on the
        // width; the native side must never enumerate/write against it.
        assert_eq!(
            bridge_capacity_proven(SequenceBoundEvidence::HeuristicUniverseCapacity {
                universe_invariant: Arc::from("TypeOk"),
            }),
            (Some(3), false),
            "a heuristic universe-capacity bound must bridge to capacity_proven=false (fail closed)",
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_roundtrip_check_to_jit_to_check() {
        let registry = VarRegistry::from_names(["pc", "counter", "flag"]);
        let func = IntIntervalFunc::new(
            1,
            3,
            vec![
                Value::SmallInt(10),
                Value::SmallInt(20),
                Value::SmallInt(30),
            ],
        );
        let state = ArrayState::from_values(vec![
            Value::SmallInt(0),
            Value::IntFunc(Rp::new(func)),
            Value::Bool(true),
        ]);
        let check_layout = infer_layout(&state, &registry);
        let jit_layout = check_layout_to_jit_layout(&check_layout);
        let roundtrip_layout = jit_layout_to_check_layout(&jit_layout, &registry);

        // Verify structural equivalence
        assert_eq!(check_layout.var_count(), roundtrip_layout.var_count());
        assert_eq!(check_layout.total_slots(), roundtrip_layout.total_slots());
        assert_eq!(
            check_layout.is_all_scalar(),
            roundtrip_layout.is_all_scalar()
        );
        assert_eq!(
            check_layout.is_fully_flat(),
            roundtrip_layout.is_fully_flat()
        );

        // Verify per-variable slot counts match
        for i in 0..check_layout.var_count() {
            let orig = check_layout.var_layout(i).unwrap();
            let rt = roundtrip_layout.var_layout(i).unwrap();
            assert_eq!(
                orig.slot_count, rt.slot_count,
                "var {i} slot_count mismatch: orig={}, roundtrip={}",
                orig.slot_count, rt.slot_count
            );
            assert_eq!(
                orig.offset, rt.offset,
                "var {i} offset mismatch: orig={}, roundtrip={}",
                orig.offset, rt.offset
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_compatible_all_scalar() {
        let registry = VarRegistry::from_names(["x", "y"]);
        let state = ArrayState::from_values(vec![Value::SmallInt(1), Value::Bool(false)]);
        let check_layout = infer_layout(&state, &registry);
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert!(layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_compatible_int_array() {
        let registry = VarRegistry::from_names(["pc", "arr"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let state =
            ArrayState::from_values(vec![Value::SmallInt(0), Value::IntFunc(Rp::new(func))]);
        let check_layout = infer_layout(&state, &registry);
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert!(layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_compatible_dynamic_var() {
        let registry = VarRegistry::from_names(["count", "data"]);
        let set = tla_value::value::SortedSet::from_sorted_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
        ]);
        let state = ArrayState::from_values(vec![Value::SmallInt(99), Value::Set(Rp::new(set))]);
        let check_layout = infer_layout(&state, &registry);
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert!(layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_var_count_mismatch() {
        let registry2 = VarRegistry::from_names(["x", "y"]);
        let state2 = ArrayState::from_values(vec![Value::SmallInt(1), Value::SmallInt(2)]);
        let check_layout = infer_layout(&state2, &registry2);

        // Create a native ABI layout with 3 vars
        let jit_layout = tla_jit_abi::StateLayout::new(vec![
            tla_jit_abi::VarLayout::ScalarInt,
            tla_jit_abi::VarLayout::ScalarInt,
            tla_jit_abi::VarLayout::ScalarInt,
        ]);

        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_ewd998_layout_bridge() {
        // EWD998 N=3: 7 variables including IntFunc arrays
        let registry = VarRegistry::from_names([
            "active",
            "color",
            "counter",
            "pending",
            "token_pos",
            "token_q",
            "token_color",
        ]);
        let active = IntIntervalFunc::new(
            0,
            2,
            vec![Value::Bool(true), Value::Bool(false), Value::Bool(false)],
        );
        let color = IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::String(Rp::from("white")),
                Value::String(Rp::from("white")),
                Value::String(Rp::from("white")),
            ],
        );
        let counter = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let pending = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );

        let state = ArrayState::from_values(vec![
            Value::IntFunc(Rp::new(active)),
            Value::IntFunc(Rp::new(color)),
            Value::IntFunc(Rp::new(counter)),
            Value::IntFunc(Rp::new(pending)),
            Value::SmallInt(0),
            Value::SmallInt(0),
            Value::String(Rp::from("black")),
        ]);

        let check_layout = infer_layout(&state, &registry);
        assert_eq!(check_layout.total_slots(), 15);
        assert!(check_layout.is_fully_flat());

        let jit_layout = check_layout_to_jit_layout(&check_layout);
        assert_eq!(jit_layout.var_count(), 7);
        match jit_layout.var_layout(1).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                value_layout,
                domain_lo,
                pair_count,
                ..
            }) => {
                assert_eq!(*domain_lo, Some(0));
                assert_eq!(*pair_count, Some(3));
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::String
                ));
            }
            other => panic!("expected string-valued color function layout, got {other:?}"),
        }
        assert!(matches!(
            jit_layout.var_layout(6).unwrap(),
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::String)
        ));

        // Verify compatibility
        assert!(layouts_compatible(&check_layout, &jit_layout));

        // Roundtrip
        let roundtrip = jit_layout_to_check_layout(&jit_layout, &registry);
        assert_eq!(roundtrip.total_slots(), 15);
        assert!(roundtrip.is_fully_flat());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_recursive_layout_bridge_uses_compound_layout() {
        use super::super::state_layout::{FlatValueLayout, SlotType};

        let registry = VarRegistry::from_names(["network"]);
        let message_layout = FlatValueLayout::Record {
            field_names: vec![Arc::from("clock"), Arc::from("type")],
            field_layouts: vec![
                FlatValueLayout::Scalar(SlotType::Int),
                FlatValueLayout::Scalar(SlotType::String),
            ],
        };
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 1,
                    len: 2,
                    value_layout: Box::new(FlatValueLayout::IntFunction {
                        lo: 1,
                        len: 2,
                        value_layout: Box::new(FlatValueLayout::Sequence {
                            bound:
                                super::super::state_layout::SequenceBoundEvidence::ProvenInvariant {
                                    invariant: Arc::from("BoundedNetwork"),
                                },
                            max_len: 3,
                            element_layout: Box::new(message_layout),
                        }),
                    }),
                },
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 28);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 28);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                pair_count,
                value_layout,
                domain_lo,
                ..
            }) => {
                assert_eq!(*pair_count, Some(2));
                assert_eq!(*domain_lo, Some(1));
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::Function {
                        pair_count: Some(2),
                        ..
                    }
                ));
            }
            other => panic!("expected recursive Compound(Function), got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_dense_ordered_generic_function_bridges_with_domain_lo() {
        use super::super::state_layout::{FlatScalarValue, FlatValueLayout, SlotType};

        let registry = VarRegistry::from_names(["clock"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: vec![
                        FlatScalarValue::Int(2),
                        FlatScalarValue::Int(3),
                        FlatScalarValue::Int(4),
                    ],
                    value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );

        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
        match jit_layout.var_layout(0).unwrap() {
            tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
                key_layout,
                pair_count,
                domain_lo,
                value_layout,
            }) => {
                assert!(matches!(
                    key_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::Int
                ));
                assert_eq!(*pair_count, Some(3));
                assert_eq!(*domain_lo, Some(2));
                assert!(matches!(
                    value_layout.as_ref(),
                    tla_jit_abi::CompoundLayout::Int
                ));
            }
            other => panic!("expected dense generic function bridge, got {other:?}"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_dense_generic_function_incompatible_with_domain_lo_none() {
        use super::super::state_layout::{FlatScalarValue, FlatValueLayout, SlotType};

        let registry = VarRegistry::from_names(["clock"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: vec![
                        FlatScalarValue::Int(2),
                        FlatScalarValue::Int(3),
                        FlatScalarValue::Int(4),
                    ],
                    value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Function {
                key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                value_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                pair_count: Some(3),
                domain_lo: None,
            },
        )]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_domain_lo_some_requires_dense_ordered_generic_function_domain() {
        use super::super::state_layout::{FlatScalarValue, FlatValueLayout, SlotType};

        let registry = VarRegistry::from_names(["clock"]);
        for domain in [
            vec![
                FlatScalarValue::Int(2),
                FlatScalarValue::Int(4),
                FlatScalarValue::Int(5),
            ],
            vec![
                FlatScalarValue::Int(2),
                FlatScalarValue::Int(4),
                FlatScalarValue::Int(3),
            ],
        ] {
            let check_layout = StateLayout::new(
                &registry,
                vec![VarLayoutKind::Recursive {
                    layout: FlatValueLayout::Function {
                        domain,
                        value_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                    },
                }],
            );
            let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
                tla_jit_abi::CompoundLayout::Function {
                    key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                    value_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
                    pair_count: Some(3),
                    domain_lo: Some(2),
                },
            )]);

            assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
            assert!(
                !layouts_compatible(&check_layout, &jit_layout),
                "domain_lo Some must reject non-dense or wrong-order generic function domains"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_compatible_recursive_record_exact_field_order() {
        use super::super::state_layout::{FlatValueLayout, SlotType};

        let registry = VarRegistry::from_names(["token"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Record {
                    field_names: vec![Arc::from("pos"), Arc::from("color"), Arc::from("q")],
                    field_layouts: vec![
                        FlatValueLayout::Scalar(SlotType::Int),
                        FlatValueLayout::Scalar(SlotType::String),
                        FlatValueLayout::Scalar(SlotType::Int),
                    ],
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Record {
                fields: vec![
                    (
                        tla_core::intern_name("pos"),
                        tla_jit_abi::CompoundLayout::Int,
                    ),
                    (
                        tla_core::intern_name("color"),
                        tla_jit_abi::CompoundLayout::String,
                    ),
                    (tla_core::intern_name("q"), tla_jit_abi::CompoundLayout::Int),
                ],
            },
        )]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_layouts_incompatible_recursive_record_same_width_field_order_mismatch() {
        use super::super::state_layout::{FlatValueLayout, SlotType};

        let registry = VarRegistry::from_names(["token"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Record {
                    field_names: vec![Arc::from("pos"), Arc::from("color"), Arc::from("q")],
                    field_layouts: vec![
                        FlatValueLayout::Scalar(SlotType::Int),
                        FlatValueLayout::Scalar(SlotType::String),
                        FlatValueLayout::Scalar(SlotType::Int),
                    ],
                },
            }],
        );
        let jit_layout = tla_jit_abi::StateLayout::new(vec![tla_jit_abi::VarLayout::Compound(
            tla_jit_abi::CompoundLayout::Record {
                fields: vec![
                    (
                        tla_core::intern_name("color"),
                        tla_jit_abi::CompoundLayout::String,
                    ),
                    (
                        tla_core::intern_name("pos"),
                        tla_jit_abi::CompoundLayout::Int,
                    ),
                    (tla_core::intern_name("q"), tla_jit_abi::CompoundLayout::Int),
                ],
            },
        )]);

        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(!layouts_compatible(&check_layout, &jit_layout));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_compact_slot_count_scalars() {
        assert_eq!(tla_jit_abi::CompoundLayout::Int.compact_slot_count(), 1);
        assert_eq!(tla_jit_abi::CompoundLayout::Bool.compact_slot_count(), 1);
        assert_eq!(tla_jit_abi::CompoundLayout::String.compact_slot_count(), 1);
        assert_eq!(tla_jit_abi::CompoundLayout::Dynamic.compact_slot_count(), 1);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_compact_slot_count_int_array() {
        let func_layout = tla_jit_abi::CompoundLayout::Function {
            key_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
            value_layout: Box::new(tla_jit_abi::CompoundLayout::Int),
            pair_count: Some(5),
            domain_lo: Some(0),
        };
        assert_eq!(func_layout.compact_slot_count(), 5);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_compact_slot_count_record() {
        let nid_a = tla_core::intern_name("a");
        let nid_b = tla_core::intern_name("b");
        let rec_layout = tla_jit_abi::CompoundLayout::Record {
            fields: vec![
                (nid_a, tla_jit_abi::CompoundLayout::Int),
                (nid_b, tla_jit_abi::CompoundLayout::Bool),
            ],
        };
        assert_eq!(rec_layout.compact_slot_count(), 2);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_record_plus_scalar_compact_offsets_do_not_use_jit_serialized_offsets() {
        use super::super::state_layout::SlotType;

        let registry = VarRegistry::from_names(["rec", "tail"]);
        let check_layout = StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::Record {
                    field_range_proofs: None,
                    field_names: vec![Arc::from("a"), Arc::from("b")],
                    field_is_bool: vec![false, false],
                    field_types: vec![SlotType::Int, SlotType::Int],
                },
                VarLayoutKind::Scalar,
            ],
        );
        let jit_layout = check_layout_to_jit_layout(&check_layout);

        assert_eq!(check_layout.total_slots(), 3);
        assert_eq!(check_layout.var_layout(0).unwrap().offset, 0);
        assert_eq!(check_layout.var_layout(1).unwrap().offset, 2);
        assert_eq!(jit_layout_compact_slot_count(&jit_layout), 3);
        assert_eq!(first_layout_slot_mismatch(&check_layout, &jit_layout), None);
        assert!(layouts_compatible(&check_layout, &jit_layout));

        let serialized_offsets = jit_layout.compute_var_offsets();
        assert_eq!(serialized_offsets, vec![Some(0), Some(8)]);
        assert_ne!(
            serialized_offsets[1],
            Some(check_layout.var_layout(1).unwrap().offset),
            "active compact paths must use check-layout offsets/compact slot counts, not tla-jit-abi::StateLayout::compute_var_offsets"
        );
    }
}
