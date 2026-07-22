// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Constant and frame condition lowering: LoadConst, FuncSet, Unchanged.

use crate::TrustIrError;
use num_traits::ToPrimitive;
use tla_value::Value;
use trust_ir::inst::*;
use trust_ir::ty::Ty;
use trust_ir::value::ValueId;

use super::{Ctx, LoweringMode};

/// Maximum number of elements in a `Value::Interval` constant that can be
/// statically materialized into a trust-ir aggregate during lowering.
///
/// Intervals with more elements than this fall through to the "compound
/// constant not representable" error and are handled by interpreter fallback.
/// Chosen to accommodate small constant ranges (typical `Node == 0..N` for
/// `N <= ~60` while keeping stack alloca size bounded — the aggregate layout
/// is 1 length header + N elements, so 64 elements = 65 slots = 520 bytes).
const MAX_INTERVAL_MATERIALIZE: i64 = 64;

/// Maximum aggregate slots a single compound `LoadConst` may materialize.
///
/// This keeps finite compound constants on the trust_cg-compatible static-allocation
/// path without allowing arbitrary constant pools to expand into unbounded stack
/// storage during lowering.
const MAX_COMPOUND_CONST_SLOTS: usize = 256;

struct MaterializedConst {
    value: ValueId,
    scalar: Option<i64>,
    shape: Option<super::AggregateShape>,
    /// True iff `value` is a `PtrToInt` of a freshly allocated flat aggregate
    /// (the `[len, elems..]` / `[len, k, v, ..]` layout produced by
    /// `alloc_aggregate`), rather than a bare scalar immediate. Compound
    /// constants always materialize a real aggregate here, so the loaded
    /// register genuinely holds a materialized aggregate base pointer and must
    /// be recorded with `aggregate_pointer_regs` provenance — otherwise it
    /// degrades to an `untracked_fixed_compound` register the pointer wall
    /// vetoes even though it provably points at real allocated storage.
    is_aggregate_pointer: bool,
}

/// Derive a `ScalarIntDomain` element shape for a constant sequence/tuple whose
/// elements are all exact integers. This lets `FuncApply` results over the
/// constant (and singleton set-builders `{f[x]}` over them) carry the compact
/// integer universe of the constant's value range, so they can fuse into a
/// matching `SetBitmask` operand. The universe is canonicalized with the same
/// `from_elements` reduction used for state-variable universes, so the exact
/// representations line up for `compatible_set_bitmask_universe`. Returns `None`
/// when the values do not form a usable universe, leaving the caller's
/// uniform-shape fallback in place.
fn const_int_domain_element_shape(
    all_exact_int: bool,
    values: &[i64],
) -> Option<super::AggregateShape> {
    if !all_exact_int || values.is_empty() {
        return None;
    }
    let mut distinct = values.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let universe_len = u32::try_from(distinct.len()).ok()?;
    let elements: Vec<super::SetBitmaskElement> = distinct
        .iter()
        .map(|&v| super::SetBitmaskElement::Int(v))
        .collect();
    Some(super::AggregateShape::ScalarIntDomain {
        universe_len,
        universe: super::SetBitmaskUniverse::from_elements(&elements),
    })
}

impl<'cp> Ctx<'cp> {
    // =====================================================================
    // LoadConst: load a compile-time constant from the constant pool
    // =====================================================================

    fn load_const_scalar_imm(&mut self, value: &Value) -> Result<Option<i64>, TrustIrError> {
        match value {
            Value::SmallInt(n) => Ok(Some(*n)),
            Value::Int(n) => Ok(n.to_i64()),
            Value::Bool(b) => Ok(Some(i64::from(*b))),
            Value::String(s) => Ok(Some(i64::from(tla_core::intern_name(s).0))),
            Value::ModelValue(m) => Ok(Some(i64::from(tla_core::intern_name(m.as_ref()).0))),
            _ => Ok(None),
        }
    }

    fn materialize_scalar_const(
        &mut self,
        block_idx: usize,
        value: &Value,
    ) -> Result<Option<MaterializedConst>, TrustIrError> {
        let Some(imm) = self.load_const_scalar_imm(value)? else {
            return Ok(None);
        };
        let shape = match value {
            Value::SmallInt(_) | Value::Int(_) => {
                Some(super::AggregateShape::Scalar(super::ScalarShape::Int))
            }
            Value::Bool(_) => Some(super::AggregateShape::Scalar(super::ScalarShape::Bool)),
            Value::String(_) => Some(super::AggregateShape::Scalar(super::ScalarShape::String)),
            Value::ModelValue(name) => {
                if let Some(domain) = super::SymbolicDomain::from_model_value(name) {
                    Some(super::AggregateShape::SymbolicDomain(domain))
                } else {
                    Some(super::AggregateShape::Scalar(
                        super::ScalarShape::ModelValue,
                    ))
                }
            }
            _ => None,
        };
        Ok(Some(MaterializedConst {
            value: self.emit_i64_const(block_idx, imm),
            scalar: Some(imm),
            shape,
            is_aggregate_pointer: false,
        }))
    }

    fn const_len_to_u32(kind: &str, len: usize) -> Result<u32, TrustIrError> {
        u32::try_from(len).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "LoadConst: {kind} has too many elements to materialize: {len}"
            ))
        })
    }

    fn consume_const_slots(
        value: &Value,
        slots: u32,
        remaining_slots: &mut usize,
    ) -> Result<(), TrustIrError> {
        let slots = usize::try_from(slots).expect("u32 slot count must fit in usize");
        if slots > *remaining_slots {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "LoadConst: compound constant exceeds trust-ir materialization limit of {MAX_COMPOUND_CONST_SLOTS} aggregate slots: {value:?}"
            )));
        }
        *remaining_slots -= slots;
        Ok(())
    }

    fn uniform_nested_shape(
        shapes: &[Option<super::AggregateShape>],
    ) -> Option<Box<super::AggregateShape>> {
        let first = shapes.first()?.as_ref()?;
        if shapes
            .iter()
            .all(|shape| shape.as_ref().is_some_and(|shape| shape == first))
        {
            Some(Box::new(first.clone()))
        } else {
            None
        }
    }

    /// Element shape for a materialized constant SET whose members may be
    /// fixed-arity tuples that share a tuple arity but differ in their per-slot
    /// scalar refinements (e.g. `{<<x, y>> : x, y \in 1..N}`, whose member
    /// shapes are all `Sequence { Exact(2), .. }` but carry per-tuple integer
    /// component domains that are not all structurally identical).
    ///
    /// First tries exact structural uniformity ([`Self::uniform_nested_shape`]).
    /// If that fails, falls back to an *arity-only* collapse: when **every**
    /// member shape is a `Sequence` with the **same** `Exact(arity)` extent, it
    /// returns `Sequence { Exact(arity), element }`, where the inner `element`
    /// is kept only when all members agree on it and is otherwise dropped to
    /// `None`. This is sound: it asserts strictly less than the per-member
    /// shapes already proved (each member *is* an `arity`-tuple), so any
    /// consumer relying on `Exact(arity)` (e.g. tuple-keyed `FuncApply` argument
    /// arity matching) gets a fact that holds for every concrete member. The
    /// extent must be `Exact` (a concretely known element count) for this to
    /// fire; `Capacity` and non-`Sequence` shapes fall through to `None`.
    fn uniform_set_element_shape(
        shapes: &[Option<super::AggregateShape>],
    ) -> Option<Box<super::AggregateShape>> {
        if let Some(shape) = Self::uniform_nested_shape(shapes) {
            return Some(shape);
        }
        let first = shapes.first()?.as_ref()?;
        let super::AggregateShape::Sequence {
            extent: super::SequenceExtent::Exact(arity),
            ..
        } = first
        else {
            return None;
        };
        let arity = *arity;
        // Every member must be an `Exact(arity)` tuple; collect the per-member
        // inner element shape so we can attempt a canonical single-slot collapse
        // when the members carry different *domain metadata* (e.g. per-tuple
        // `ScalarIntDomain` universes for `{<<-1,-1>>, <<-1,0>>, ...}`) but share
        // the same 1-slot scalar storage class.
        let mut inner_elements: Vec<Option<super::AggregateShape>> =
            Vec::with_capacity(shapes.len());
        for shape in shapes {
            let shape = shape.as_ref()?;
            let super::AggregateShape::Sequence {
                extent: super::SequenceExtent::Exact(member_arity),
                element,
            } = shape
            else {
                return None;
            };
            if *member_arity != arity {
                return None;
            }
            inner_elements.push(element.as_deref().cloned());
        }
        // `uniform_tuple_element_shape` first tries exact structural uniformity
        // (preserving richer nested layouts), then collapses onto a shared
        // single-slot scalar class iff every member's inner element is a 1-slot
        // flat scalar of a compatible compact-slot class. If the members disagree
        // on storage class — or any lacks a tracked element — it returns `None`
        // and we keep `element: None` (the prior fail-closed behavior), still
        // asserting the proven `Exact(arity)`. Soundness: the collapse asserts
        // strictly less than each member already proved (each element *is* a
        // 1-slot scalar of that class), so any consumer relying on the collapsed
        // element (e.g. tuple-component `FuncApply` slot reads) gets a fact that
        // holds for every concrete member.
        let element = super::uniform_tuple_element_shape(&inner_elements);
        Some(Box::new(super::AggregateShape::Sequence {
            extent: super::SequenceExtent::Exact(arity),
            element,
        }))
    }

    fn materialize_const_value(
        &mut self,
        block_idx: usize,
        value: &Value,
        remaining_slots: &mut usize,
    ) -> Result<MaterializedConst, TrustIrError> {
        if let Some(materialized) = self.materialize_scalar_const(block_idx, value)? {
            return Ok(materialized);
        }

        match value {
            Value::Interval(iv) => {
                let (Some(lo), Some(hi)) = (iv.low().to_i64(), iv.high().to_i64()) else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: interval bounds do not fit in i64: {iv:?}"
                    )));
                };
                let count_i64 = if hi >= lo { hi - lo + 1 } else { 0 };
                if count_i64 > MAX_INTERVAL_MATERIALIZE {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: interval {lo}..{hi} has {count_i64} elements, exceeds trust-ir materialization limit of {MAX_INTERVAL_MATERIALIZE}"
                    )));
                }

                let count = count_i64 as u32;
                let total_slots = count.checked_add(1).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: interval {lo}..{hi} has too many slots to materialize"
                    ))
                })?;
                Self::consume_const_slots(value, total_slots, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_val = self.emit_i64_const(block_idx, count_i64);
                self.store_at_offset(block_idx, agg_ptr, 0, len_val);
                for i in 0..count {
                    let elem_val = self.emit_i64_const(block_idx, lo + i64::from(i));
                    self.store_at_offset(block_idx, agg_ptr, i + 1, elem_val);
                }

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::Interval { lo, hi }),
                })
            }
            Value::RecordSet(record_set) => {
                let field_count = Self::const_len_to_u32("record set", record_set.fields_len())?;
                Self::consume_const_slots(value, field_count, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, field_count);
                let mut fields = Vec::with_capacity(record_set.fields_len());
                for (slot, (field_name, field_set)) in
                    record_set.fields_check_order_iter().enumerate()
                {
                    let field =
                        self.materialize_record_set_domain(block_idx, field_set, remaining_slots)?;
                    self.store_at_offset(
                        block_idx,
                        agg_ptr,
                        u32::try_from(slot).expect("record set slot index must fit in u32"),
                        field.value,
                    );
                    let Some(shape) = field.shape else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "LoadConst: record set field {field_name:?} is not a materializable set domain: {field_set:?}"
                        )));
                    };
                    fields.push((tla_core::intern_name(field_name), shape));
                }

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::RecordSet { fields }),
                })
            }
            Value::Set(set) => {
                let len = Self::const_len_to_u32("set", set.len())?;
                let exact_scalar_set = super::exact_scalar_set_values_from_const_set(set.iter());
                let total_slots = len.checked_add(1).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: set has too many slots to materialize: {}",
                        set.len()
                    ))
                })?;
                Self::consume_const_slots(value, total_slots, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_val = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, agg_ptr, 0, len_val);
                let mut element_shapes = Vec::with_capacity(set.len());
                let mut exact_int_values = Vec::with_capacity(set.len());
                let mut all_exact_int_values = true;
                for (idx, elem_value) in set.iter().enumerate() {
                    match elem_value {
                        Value::SmallInt(n) => exact_int_values.push(*n),
                        Value::Int(n) => {
                            if let Some(n) = n.to_i64() {
                                exact_int_values.push(n);
                            } else {
                                all_exact_int_values = false;
                            }
                        }
                        _ => all_exact_int_values = false,
                    }
                    let elem =
                        self.materialize_const_value(block_idx, elem_value, remaining_slots)?;
                    element_shapes.push(elem.shape);
                    self.store_at_offset(
                        block_idx,
                        agg_ptr,
                        u32::try_from(idx + 1).expect("set slot index must fit in u32"),
                        elem.value,
                    );
                }

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(if all_exact_int_values {
                        super::AggregateShape::ExactIntSet {
                            values: exact_int_values,
                        }
                    } else if let Some((scalar, values)) = exact_scalar_set {
                        super::AggregateShape::ExactScalarSet { scalar, values }
                    } else if let Some(shape) =
                        super::lazy_union_shape_from_const_set_elements(set.iter())
                    {
                        // A constant-folded `(SUBSET U) \cup S` value: hand
                        // the register the same static lazy-union shape the
                        // unfolded opcodes would produce, so TypeOK-style
                        // membership keeps its exact native lowering (lever
                        // L1). The materialized payload stays in the register
                        // but no lazy-shape consumer ever dereferences it.
                        shape
                    } else if let Some(shape) = super::subset_powerset_shape_enabled()
                        .then(|| super::nonempty_powerset_shape_from_const_set_elements(set.iter()))
                        .flatten()
                    {
                        // A constant-folded `(SUBSET U) \ {{}}` value (opt-in,
                        // TY_SUBSET_POWERSET_SHAPE): reconstruct the lazy
                        // `NonEmptyPowerset { base }` shape the unfolded
                        // opcodes would produce so the native compact-mask
                        // membership fires (SimpleRegular's TypeOK). Same
                        // static-payload contract as the lazy-union
                        // reconstruction above — the materialized payload stays
                        // in the register but the non-`SetBitmask`-base
                        // membership arm reads only the base's compile-time
                        // universe, never dereferencing that payload.
                        shape
                    } else {
                        super::AggregateShape::Set {
                            len,
                            element: Self::uniform_set_element_shape(&element_shapes),
                        }
                    }),
                })
            }
            Value::Seq(seq) => {
                let len = Self::const_len_to_u32("sequence", seq.len())?;
                let total_slots = len.checked_add(1).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: sequence has too many slots to materialize: {}",
                        seq.len()
                    ))
                })?;
                Self::consume_const_slots(value, total_slots, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_val = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, agg_ptr, 0, len_val);
                let mut element_shapes = Vec::with_capacity(seq.len());
                let mut exact_int_values = Vec::with_capacity(seq.len());
                let mut all_exact_int_values = true;
                for (idx, elem_value) in seq.iter().enumerate() {
                    match elem_value {
                        Value::SmallInt(n) => exact_int_values.push(*n),
                        Value::Int(n) => match n.to_i64() {
                            Some(n) => exact_int_values.push(n),
                            None => all_exact_int_values = false,
                        },
                        _ => all_exact_int_values = false,
                    }
                    let elem =
                        self.materialize_const_value(block_idx, elem_value, remaining_slots)?;
                    element_shapes.push(elem.shape);
                    self.store_at_offset(
                        block_idx,
                        agg_ptr,
                        u32::try_from(idx + 1).expect("sequence slot index must fit in u32"),
                        elem.value,
                    );
                }

                let element =
                    const_int_domain_element_shape(all_exact_int_values, &exact_int_values)
                        .map(Box::new)
                        .or_else(|| Self::uniform_nested_shape(&element_shapes));

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::Sequence {
                        extent: super::SequenceExtent::Exact(len),
                        element,
                    }),
                })
            }
            Value::Tuple(tuple) => {
                let len = Self::const_len_to_u32("tuple", tuple.len())?;
                let total_slots = len.checked_add(1).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: tuple has too many slots to materialize: {}",
                        tuple.len()
                    ))
                })?;
                Self::consume_const_slots(value, total_slots, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_val = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, agg_ptr, 0, len_val);
                let mut element_shapes = Vec::with_capacity(tuple.len());
                let mut exact_int_values = Vec::with_capacity(tuple.len());
                let mut all_exact_int_values = true;
                for (idx, elem_value) in tuple.iter().enumerate() {
                    match elem_value {
                        Value::SmallInt(n) => exact_int_values.push(*n),
                        Value::Int(n) => match n.to_i64() {
                            Some(n) => exact_int_values.push(n),
                            None => all_exact_int_values = false,
                        },
                        _ => all_exact_int_values = false,
                    }
                    let elem =
                        self.materialize_const_value(block_idx, elem_value, remaining_slots)?;
                    element_shapes.push(elem.shape);
                    self.store_at_offset(
                        block_idx,
                        agg_ptr,
                        u32::try_from(idx + 1).expect("tuple slot index must fit in u32"),
                        elem.value,
                    );
                }

                let element =
                    const_int_domain_element_shape(all_exact_int_values, &exact_int_values)
                        .map(Box::new)
                        .or_else(|| Self::uniform_nested_shape(&element_shapes));

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::Sequence {
                        extent: super::SequenceExtent::Exact(len),
                        element,
                    }),
                })
            }
            Value::SeqSet(seq_set) => {
                let base =
                    self.materialize_const_value(block_idx, seq_set.base(), remaining_slots)?;
                let Some(base_shape) = base.shape else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: SeqSet base is not a materializable set/domain: {:?}",
                        seq_set.base()
                    )));
                };
                base_shape.validate_seq_base("LoadConst: SeqSet")?;

                Ok(MaterializedConst {
                    value: base.value,
                    scalar: None,
                    shape: Some(super::AggregateShape::SeqSet {
                        base: Box::new(base_shape),
                    }),
                    is_aggregate_pointer: base.is_aggregate_pointer,
                })
            }
            // A constant-folded `SUBSET S` (lazy powerset value). Mirrors
            // `lower_powerset`: the runtime payload is the BASE set's payload
            // (existing powerset consumers dereference it as the base), and the
            // shape marks the register lazy so `x \in SUBSET S` lowers as
            // `x \subseteq S` without enumeration.
            Value::Subset(subset) => {
                let base = self.materialize_const_value(block_idx, subset.base(), remaining_slots)?;
                let Some(base_shape) = base.shape else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: SUBSET base is not a materializable set/domain: {:?}",
                        subset.base()
                    )));
                };
                base_shape.validate_powerset_base("LoadConst: SUBSET")?;

                Ok(MaterializedConst {
                    value: base.value,
                    scalar: None,
                    shape: Some(super::AggregateShape::Powerset {
                        base: Box::new(base_shape),
                    }),
                    is_aggregate_pointer: base.is_aggregate_pointer,
                })
            }
            // A constant-folded `[S -> T]` (lazy function-set value, e.g. the
            // folded `[Proc -> BOOLEAN]` in a TypeOK conjunct). Mirrors
            // `lower_func_set`: a 2-slot aggregate `[domain_payload,
            // codomain_payload]` with a `FunctionSet` shape, so `f \in [S -> T]`
            // takes the existing function-set membership lowerings.
            Value::FuncSet(func_set) => {
                let domain =
                    self.materialize_const_value(block_idx, func_set.domain(), remaining_slots)?;
                let Some(domain_shape) = domain.shape else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: function-set domain is not a materializable set/domain: {:?}",
                        func_set.domain()
                    )));
                };
                domain_shape.validate_powerset_base("LoadConst: FuncSet domain")?;
                domain_shape.tracked_len().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: function-set domain cardinality is not statically known: {domain_shape:?}"
                    ))
                })?;
                let range =
                    self.materialize_const_value(block_idx, func_set.codomain(), remaining_slots)?;
                let Some(range_shape) = range.shape else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "LoadConst: function-set range is not a materializable set/domain: {:?}",
                        func_set.codomain()
                    )));
                };
                range_shape.validate_function_set_range("LoadConst: FuncSet range")?;

                Self::consume_const_slots(value, 2, remaining_slots)?;
                let agg_ptr = self.alloc_aggregate(block_idx, 2);
                self.store_at_offset(block_idx, agg_ptr, 0, domain.value);
                self.store_at_offset(block_idx, agg_ptr, 1, range.value);

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::FunctionSet {
                        domain: Box::new(domain_shape),
                        range: Box::new(range_shape),
                    }),
                })
            }
            Value::Record(rec) => {
                let field_count = Self::const_len_to_u32("record", rec.len())?;
                Self::consume_const_slots(value, field_count, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, field_count);
                let mut fields = Vec::with_capacity(rec.len());
                for (slot, (field_name, field_value)) in rec.iter().enumerate() {
                    let field =
                        self.materialize_const_value(block_idx, field_value, remaining_slots)?;
                    self.store_at_offset(
                        block_idx,
                        agg_ptr,
                        u32::try_from(slot).expect("record slot index must fit in u32"),
                        field.value,
                    );
                    fields.push((field_name, field.shape.map(Box::new)));
                }

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::Record { fields }),
                })
            }
            Value::Func(func) => {
                let entries: Vec<_> = func.iter().collect();
                let len = Self::const_len_to_u32("function", entries.len())?;
                let domain_lo =
                    super::dense_ordered_int_values_lo(entries.iter().map(|(key, _)| *key))
                        .and_then(|(lo, domain_len)| (domain_len == len).then_some(lo));
                let total_slots = len
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_add(1))
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "LoadConst: function has too many slots to materialize: {}",
                            entries.len()
                        ))
                    })?;
                Self::consume_const_slots(value, total_slots, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_val = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, agg_ptr, 0, len_val);

                let mut value_shapes = Vec::with_capacity(entries.len());
                for (idx, (key, mapped_value)) in entries.into_iter().enumerate() {
                    let key = self.materialize_const_value(block_idx, key, remaining_slots)?;
                    let mapped =
                        self.materialize_const_value(block_idx, mapped_value, remaining_slots)?;
                    let idx = u32::try_from(idx).expect("function entry index must fit in u32");
                    self.store_at_offset(block_idx, agg_ptr, 1 + (2 * idx), key.value);
                    self.store_at_offset(block_idx, agg_ptr, 2 + (2 * idx), mapped.value);
                    value_shapes.push(mapped.shape);
                }

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::Function {
                        len,
                        domain_lo,
                        domain: None,
                        value: Self::uniform_nested_shape(&value_shapes),
                    }),
                })
            }
            Value::IntFunc(func) => {
                let len = Self::const_len_to_u32("int-interval function", func.len())?;
                let total_slots = len
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_add(1))
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "LoadConst: int-interval function has too many slots to materialize: {}",
                            func.len()
                        ))
                    })?;
                Self::consume_const_slots(value, total_slots, remaining_slots)?;

                let agg_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_val = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, agg_ptr, 0, len_val);

                let mut value_shapes = Vec::with_capacity(func.len());
                for (idx, mapped_value) in func.values().iter().enumerate() {
                    let offset = i64::try_from(idx).map_err(|_| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "LoadConst: int-interval function index too large: {idx}"
                        ))
                    })?;
                    let key_imm = func.as_ref().min().checked_add(offset).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "LoadConst: int-interval function key overflows i64 at index {idx}"
                        ))
                    })?;
                    let key = self.emit_i64_const(block_idx, key_imm);
                    let mapped =
                        self.materialize_const_value(block_idx, mapped_value, remaining_slots)?;
                    let idx = u32::try_from(idx).expect("function entry index must fit in u32");
                    self.store_at_offset(block_idx, agg_ptr, 1 + (2 * idx), key);
                    self.store_at_offset(block_idx, agg_ptr, 2 + (2 * idx), mapped.value);
                    value_shapes.push(mapped.shape);
                }

                Ok(MaterializedConst {
                    value: self.ptr_to_i64(block_idx, agg_ptr),
                    scalar: None,
                    is_aggregate_pointer: true,
                    shape: Some(super::AggregateShape::Function {
                        len,
                        domain_lo: Some(func.as_ref().min()),
                        domain: None,
                        value: Self::uniform_nested_shape(&value_shapes),
                    }),
                })
            }
            other => Err(TrustIrError::UnsupportedOpcode(format!(
                "LoadConst: compound constant pool value not representable as fixed trust-ir aggregate: {other:?}"
            ))),
        }
    }

    fn materialize_record_set_domain(
        &mut self,
        block_idx: usize,
        value: &Value,
        remaining_slots: &mut usize,
    ) -> Result<MaterializedConst, TrustIrError> {
        if let Value::Interval(iv) = value {
            let (Some(lo), Some(hi)) = (iv.low().to_i64(), iv.high().to_i64()) else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "LoadConst: interval bounds do not fit in i64: {iv:?}"
                )));
            };
            Self::consume_const_slots(value, 3, remaining_slots)?;

            let agg_ptr = self.alloc_aggregate(block_idx, 3);
            let count = if hi >= lo { hi - lo + 1 } else { 0 };
            let len_val = self.emit_i64_const(block_idx, count);
            let lo_val = self.emit_i64_const(block_idx, lo);
            let hi_val = self.emit_i64_const(block_idx, hi);
            self.store_at_offset(block_idx, agg_ptr, 0, len_val);
            self.store_at_offset(block_idx, agg_ptr, 1, lo_val);
            self.store_at_offset(block_idx, agg_ptr, 2, hi_val);

            return Ok(MaterializedConst {
                value: self.ptr_to_i64(block_idx, agg_ptr),
                scalar: None,
                shape: Some(super::AggregateShape::Interval { lo, hi }),
                is_aggregate_pointer: true,
            });
        }

        self.materialize_const_value(block_idx, value, remaining_slots)
    }

    /// Lower `LoadConst { rd, idx }`.
    ///
    /// Resolves the constant pool entry at compile time and emits the
    /// appropriate immediate load. Supports:
    /// - `SmallInt` → `LoadImm i64`
    /// - `Bool` → `LoadImm 0/1`
    /// - `String` / `ModelValue` → `LoadImm (NameId.0 as i64)` via parse-time
    ///   interning. This matches trust_cg's serialized-state representation for
    ///   scalar names. Part of #4280.
    ///
    /// Finite compound constants are materialized into the same fixed aggregate
    /// layouts used by the corresponding constructor opcodes.
    pub(super) fn lower_load_const(
        &mut self,
        block_idx: usize,
        rd: u8,
        idx: u16,
    ) -> Result<Option<usize>, TrustIrError> {
        let pool = self.require_const_pool()?;
        let value = pool.get_value(idx);

        let mut remaining_slots = MAX_COMPOUND_CONST_SLOTS;
        let materialized = self.materialize_const_value(block_idx, value, &mut remaining_slots)?;
        self.store_reg_value(block_idx, rd, materialized.value)?;

        if let Some(shape) = materialized.shape {
            let is_function_shape = matches!(shape, super::AggregateShape::Function { .. });
            // The pointer wall vetoes a register tracked with a fixed-compound
            // Record / Sequence shape that carries no aggregate-pointer
            // provenance (`untracked_fixed_compound`). A compound constant
            // materialized above is a freshly `alloc_aggregate`d flat buffer, so
            // its register genuinely holds `PtrToInt(agg_ptr)` and must record
            // that provenance. (Function constants are handled by
            // `mark_flat_funcdef_pair_list`, which sets `Flat` itself; set-like
            // shapes are not vetoed and keep their existing bare-register
            // contract, so we scope this to exactly the vetoed compound shapes.)
            let is_vetoed_fixed_compound = matches!(
                shape,
                super::AggregateShape::Record { .. } | super::AggregateShape::Sequence { .. }
            );
            self.compact_state_slots.remove(&rd);
            if let Some(len) = shape.tracked_len() {
                self.record_set_size(rd, len);
            } else {
                self.const_set_sizes.remove(&rd);
            }
            self.aggregate_shapes.insert(rd, shape);
            if is_function_shape {
                self.mark_flat_funcdef_pair_list(rd);
            } else {
                self.clear_flat_funcdef_pair_list(rd);
                if materialized.is_aggregate_pointer && is_vetoed_fixed_compound {
                    self.aggregate_pointer_regs
                        .insert(rd, super::AggregatePointerKind::Flat);
                } else {
                    self.aggregate_pointer_regs.remove(&rd);
                }
            }
            if let Some(imm) = materialized.scalar {
                self.record_scalar(rd, imm);
            } else {
                self.const_scalar_values.remove(&rd);
            }
        } else if let Some(imm) = materialized.scalar {
            self.aggregate_shapes.remove(&rd);
            self.const_set_sizes.remove(&rd);
            self.record_scalar(rd, imm);
        } else {
            self.invalidate_reg_tracking(rd);
        }

        Ok(Some(block_idx))
    }

    // =====================================================================
    // FuncSet: [S -> T] — the set of all functions from S to T
    // =====================================================================

    /// Lower `FuncSet { rd, domain, range }`.
    ///
    /// A function set `[S -> T]` is represented as a 2-slot aggregate containing
    /// the domain set pointer and the codomain set pointer:
    ///
    ///   `[domain_ptr_as_i64, codomain_ptr_as_i64]`
    ///
    /// This is a lazy representation — the function set is not materialized into
    /// all |T|^|S| functions. The aggregate stores enough information for
    /// downstream operations (membership testing, iteration) to reconstruct the
    /// semantics.
    pub(super) fn lower_func_set(
        &mut self,
        block_idx: usize,
        rd: u8,
        domain_reg: u8,
        range_reg: u8,
    ) -> Result<(), TrustIrError> {
        let domain_shape = self
            .aggregate_shapes
            .get(&domain_reg)
            .cloned()
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "FuncSet: domain must have a tracked finite shape".to_owned(),
                )
            })?;
        domain_shape.validate_powerset_base("FuncSet domain")?;
        domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "FuncSet: domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;

        let range_shape = self
            .aggregate_shapes
            .get(&range_reg)
            .cloned()
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "FuncSet: range must have a tracked shape".to_owned(),
                )
            })?;
        range_shape.validate_function_set_range("FuncSet range")?;

        // Load the domain and range set pointers from their registers.
        let domain_val = self.load_reg(block_idx, domain_reg)?;
        let range_val = self.load_reg(block_idx, range_reg)?;

        // Allocate a 2-slot aggregate: [domain_ptr, codomain_ptr].
        let agg_ptr = self.alloc_aggregate(block_idx, 2);

        // Store domain set pointer at slot 0.
        self.store_at_offset(block_idx, agg_ptr, 0, domain_val);

        // Store codomain set pointer at slot 1.
        self.store_at_offset(block_idx, agg_ptr, 1, range_val);

        // Store the aggregate pointer into rd.
        self.store_reg_ptr(block_idx, rd, agg_ptr)?;
        self.compact_state_slots.remove(&rd);
        self.aggregate_shapes.insert(
            rd,
            super::AggregateShape::FunctionSet {
                domain: Box::new(domain_shape),
                range: Box::new(range_shape),
            },
        );
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }

    // =====================================================================
    // Unchanged: frame condition (next' = current for listed vars)
    // =====================================================================

    /// Lower `Unchanged { rd, start, count }`.
    ///
    /// Iterates over `count` entries in the constant pool starting at `start`.
    /// Each entry must be `SmallInt(var_idx)`. For each, loads
    /// `state_in[var_idx]` and `state_out[var_idx]`, compares equality, and
    /// ANDs the results. Stores `1` (TRUE) or `0` (FALSE) into `rd`.
    ///
    /// Requires `NextState` mode (needs both `state_in` and `state_out`).
    pub(super) fn lower_unchanged(
        &mut self,
        block_idx: usize,
        rd: u8,
        start: u16,
        count: u8,
    ) -> Result<Option<usize>, TrustIrError> {
        if self.config.mode != LoweringMode::NextState {
            return Err(TrustIrError::NotEligible {
                reason: "Unchanged requires next-state lowering".to_owned(),
            });
        }
        let state_out = self.state_out_ptr.ok_or_else(|| {
            TrustIrError::Emission(
                "missing state_out parameter for Unchanged in next-state lowering".to_owned(),
            )
        })?;
        let pool = self.require_const_pool()?;

        // Resolve all var indices at compile time.
        let mut var_indices = Vec::with_capacity(count as usize);
        for i in 0..(count as u16) {
            let val = pool.get_value(start + i);
            match val {
                Value::SmallInt(idx) => var_indices.push(*idx as u16),
                other => {
                    return Err(TrustIrError::Emission(format!(
                        "Unchanged: constant pool entry at {} is not SmallInt: {other:?}",
                        start + i
                    )));
                }
            }
        }

        if var_indices.is_empty() {
            // UNCHANGED <<>> is vacuously true.
            self.store_reg_imm(block_idx, rd, 1)?;
            self.compact_state_slots.remove(&rd);
            self.aggregate_shapes
                .insert(rd, super::AggregateShape::Scalar(super::ScalarShape::Bool));
            self.const_set_sizes.remove(&rd);
            self.const_scalar_values.remove(&rd);
            return Ok(Some(block_idx));
        }

        let state_in = self.state_in_ptr;

        // For a single-slot variable, compare directly. For compact
        // multi-slot variables, compare every flat state slot. For multiple
        // variables, AND all slot comparisons together.
        //
        // Emit: for each var_idx and compact slot offset:
        //   cur = load state_in[base_slot + offset]
        //   nxt = load state_out[base_slot + offset]
        //   eq_i = icmp eq cur, nxt
        //   eq_i_i64 = zext eq_i to i64
        // Then AND all eq_i_i64 values together.

        let mut result_val: Option<ValueId> = None;

        for &var_idx in &var_indices {
            let base_slot = self.compact_state_slot_offset_or_legacy(
                var_idx,
                "Unchanged compact state slot offset",
            )?;
            let slot_count = self.compact_state_slot_count_or_legacy(
                var_idx,
                "Unchanged compact state slot count",
            )?;

            for offset in 0..slot_count {
                let slot = base_slot.checked_add(offset).ok_or_else(|| {
                    TrustIrError::Emission(format!(
                        "Unchanged: compact slot offset overflow for var_idx {var_idx}"
                    ))
                })?;

                // Load current state slot value.
                let cur_ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_in, slot);
                let cur = self.emit_with_result(
                    block_idx,
                    Inst::Load {
                        ty: Ty::I64,
                        ptr: cur_ptr,
                        align: None,
                        volatile: false,
                    },
                );

                // Load next state slot value.
                let nxt_ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, slot);
                let nxt = self.emit_with_result(
                    block_idx,
                    Inst::Load {
                        ty: Ty::I64,
                        ptr: nxt_ptr,
                        align: None,
                        volatile: false,
                    },
                );

                // Compare.
                let eq = self.emit_with_result(
                    block_idx,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: cur,
                        rhs: nxt,
                    },
                );
                let eq_i64 = self.emit_with_result(
                    block_idx,
                    Inst::Cast {
                        op: CastOp::ZExt,
                        src_ty: Ty::Bool,
                        dst_ty: Ty::I64,
                        operand: eq,
                    },
                );

                result_val = Some(match result_val {
                    None => eq_i64,
                    Some(prev) => self.emit_with_result(
                        block_idx,
                        Inst::BinOp {
                            op: BinOp::And,
                            ty: Ty::I64,
                            lhs: prev,
                            rhs: eq_i64,
                        },
                    ),
                });
            }
        }

        // Store the final result into rd.
        let result_val = match result_val {
            Some(result_val) => result_val,
            None => self.emit_i64_const(block_idx, 1),
        };
        self.store_reg_value(block_idx, rd, result_val)?;
        self.compact_state_slots.remove(&rd);
        self.aggregate_shapes
            .insert(rd, super::AggregateShape::Scalar(super::ScalarShape::Bool));
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);

        Ok(Some(block_idx))
    }
}
